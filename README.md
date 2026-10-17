<img src="assets/icon.svg" width="64" height="64" alt="">

# lust-lang

Lust is an experiment. It asks one question: how fast can `check` and `build` get if a Rust-like language gives up the features that make Rust slow to compile?

To test this, lust compiles a small subset of Rust. Every valid lust program is also a valid Rust program, and it must print the same output as the `rustc` build. That rule makes `rustc` a free test oracle, and it gives lust programs a release path: compile the same file with `rustc -O`.

**Status: prototype.** About 11k lines of Rust and C. It has a real type checker and a real borrow checker, and it builds runnable programs end to end. It is not a usable Rust replacement. It has no generics, no traits, and no closures. Normal idiomatic Rust will not compile. This will fail, and so will most real crates:

```rust
let total: i64 = xs.iter().map(|x| x * 2).sum(); // lust: error[L9001], no closures yet
```

Read [DESIGN.md](DESIGN.md) as a plan, not as a description of the code. The design describes later phases that do not exist yet. This README describes only what is in the tree today.

## Numbers

All numbers come from one machine: 24-core AMD EPYC 7B13, x86_64 Linux, `rustc 1.99.0`. We re-ran both benchmarks on 2026-10-17 and got results close to the runs recorded before. Expect run-to-run variance. Read the test shape before you read the speedup. The shape is most of the result.

Short version: cold compiles of generated single-file code are about **20x to 45x** faster than `rustc`. Small edits through the warm daemon are about **130x** faster, on a test built to match the daemon's fast path. Programs that lust builds run **slower** than programs that `rustc` builds.

### Warm edit loop (the best case)

`bench/realish/agent_loop.py` imitates a coding agent. It makes 12 small edits to a 31,405-line file, then runs `check` and `build` after each edit. Two edits are mistakes, one type error and one borrow error. Both compilers must accept or reject together, and both programs must print the same output.

- `rustc` runs with `-Z threads=8`, `-C incremental`, and `-C debuginfo=0`. This is its fastest setup on this machine for this test.
- `lust` runs through a warm `lust daemon`.
- Per edit: `rustc` takes 0.78 to 1.53 s to check and 1.14 to 2.84 s to build. Lust takes 5.6 to 17.9 ms to check and 14.1 to 19.5 ms to build.

| 12 edits, total | rustc | lust | Speedup |
|---|---|---|---|
| check | 14.06 s | 0.111 s | 127x |
| build (10 edits that pass) | 22.30 s | 0.157 s | 142x |
| check + build | 36.36 s | 0.268 s | 136x |
| running the program (10 runs) | 132 ms | 252 ms | 0.5x |

Count the program runs as part of each iteration and the loop is about 70x faster. Count the cold runs that fill the caches first and it is about 60x faster.

What makes this test favorable to lust:

- The 31k-line file is one 780-line warehouse service cloned 40 times with renamed items. The code is real in style, but it has none of the variety of a real project.
- The file is one crate in one file. That is a bad case for `rustc`. Real projects split code into crates, and Cargo builds crates in parallel. Lust has no Cargo support, so we cannot run that comparison yet.
- Almost every edit changes one function body. That is exactly the case the daemon optimizes. An edit to a type or an `impl` block makes lust do a full rebuild.
- The code uses none of the Rust features lust lacks. It cannot. Lust would reject it.

Reproduce with `python3 bench/realish/agent_loop.py`. The script times each edit once.

### Cold compiles

`bench/compare.sh plain 640 3200` generates one file and gives it to both compilers. The "plain" shape is structs, enums, `match`, `Vec`, `String`, and `format!`, cloned into many similar units. `rustc` gets the fastest of three configurations for each number. Lust gets the median of three runs.

| Lines | Operation | rustc | lust | Speedup |
|---|---|---|---|---|
| 9,602 | check | 1.36 s | 55 ms | 25x |
| 9,602 | debug build | 2.83 s | 134 ms | 21x |
| 48,002 | check | 6.18 s | 139 ms | 44x |
| 48,002 | debug build | 12.81 s | 379 ms | 34x |

There is no "idiomatic Rust" row. Lust cannot compile iterator chains, closures, `Result` with `?`, or generic functions, so a speedup for that shape does not exist. [DESIGN.md](DESIGN.md) has the `rustc` baseline for that shape. It is the biggest open risk for the speed claim: generics and closures produce much more code to check and compile.

### Run speed

Lust compiles fast. Its programs do not run fast. The backend emits C and compiles it with `tcc`, which does not optimize. In daemon mode, every call also goes through a function table. In the test above, the lust program takes about 25 ms per run and the `rustc` debug program takes about 13 ms. For a fast binary, compile the same source with `rustc -O`. Lust has no release mode.

## Why it is fast

Four reasons:

1. **Less language to check.** No user macros and no proc macros. No traits, so no trait search. Signatures are explicit. The checker does less work because the language asks for less.
2. **No LLVM in dev builds.** Lust emits C and compiles it with `tcc`. `tcc` is much faster than LLVM and produces worse code.
3. **Parallel work per function.** Parsing, type checking, borrow checking, and C code generation run per function, on a flat IR, across cores.
4. **The daemon redoes only what an edit touches.** See the next section.

None of this proves that the speed survives generics, traits, and closures. That test comes later.

## How the daemon works

`lust daemon` keeps one warm session per file for the current directory. The `lust` command is a thin client. If no daemon answers, the client runs `lust-core` cold in a fresh process.

On each edit, the daemon does this:

1. It compares the new source with the last source and lexes again only the items the edit touched. Item boundaries come from content hashes, so an inserted function does not move the others.
2. It hashes each function body and everything outside bodies, with no positions. If only bodies changed, it parses, checks, and borrow-checks only those bodies. If top-level functions changed signature, it re-checks them and every body that names them. Any other change, such as a type or an `impl`, is a full rebuild.
3. It compiles only the changed functions and patches them in without a full relink. Every call goes through a function table, and each function owns one stable slot. The first build links all units into `libbase.so`. Later builds link a small executable with the changed units as overlays against `libbase.so`. At startup, the overlays replace their table entries. After 32 overlays, the daemon rebuilds the base.

Daemon builds are not standalone. The executable loads `libbase.so` from the daemon cache directory. Do not ship it. A cold `lust build` links a standalone executable.

## Build and use

You need a Rust toolchain and a C build environment. The first step clones `tinycc` from GitHub, so it needs network access once. We develop and test on x86_64 Linux. The daemon uses Unix sockets. Other platforms are not verified.

```sh
scripts/build-tcc.sh      # builds the pinned tinycc into third_party/tcc
cargo build --release     # builds `lust` (client) and `lust-core` (compiler)
```

```sh
lust daemon                                        # optional: warm compiles for this directory
lust check main.rs [--json] [--timings]
lust build main.rs [-o out] [-g] [--emit-c prefix] [--timings]
lust run main.rs [-- args]
```

Each command takes one `.rs` file. There is no Cargo support, no multi-file crate, and no external crates.

`--json` prints one diagnostic per line, each with `code`, `message`, `at`, and `fix`. The `fix` text tells the user what to write instead. This format exists for coding agents.

This is a complete lust program. It is also a complete Rust program:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point { x: i64, y: i64 }

impl Point {
    fn manhattan(&self, other: &Point) -> i64 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

fn main() {
    let a = Point { x: 1, y: 2 };
    let b = Point { x: 4, y: 6 };
    println!("{} {:?}", a.manhattan(&b), a == b);
}
```

## Supported subset

Supported today:

- Items: `fn`, `struct` (named fields or unit), `enum` (unit, tuple, and struct variants), inherent `impl` blocks, `use`.
- Derives: `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`. `Hash` is accepted, but nothing uses it yet.
- Types: integers (no 128-bit), `bool`, `char`, `String`, `&str`, `Vec<T>`, `Option<T>`, `&T`, `&mut T`.
- Control flow: `if`, `if let`, `match` (guards, `|`, ranges, nested patterns), `loop`, `while`, `while let`, `for` over ranges, `&vec`, `&mut vec`, and `s.chars()`, `break` with a value, early `return`.
- Macros: `println!`, `print!`, `eprintln!`, `eprint!`, `format!`, `vec!`, `panic!`, `assert!`, `assert_eq!`, `assert_ne!`, `todo!`, `unreachable!`. These are built in.
- Std: common methods on `String`, `str`, `Vec`, `Option`, `char`, and integers, plus `std::env::args().count()`. This is a small hand-written surface, not the Rust standard library.
- Borrow checking: moves, initialization, mutability, conflicting borrows with NLL-style lifetimes and two-phase borrows, dangling references, temporaries dropped while borrowed, and returned references by elision only.

Not supported yet. Each of these gives error `L9001`:

- Generics, traits, and closures.
- Iterators beyond `chars()` and the `for` loops above.
- `Result` and `?`, `HashMap`, floats, tuples, arrays, slices, modules, `const`, `let`-else.
- A move of one field out of a struct that keeps the rest of the struct alive.

Rejected on purpose, by design:

- Lifetime syntax (`L0203`). Elision is the only option.
- References in struct fields (`L0201`). Store an owned value, or pass the reference to each method.
- User macros (`L0401`). Write a function.

## Tests

```sh
scripts/difftest.sh             # compare with rustc
SANITIZE=1 scripts/difftest.sh  # also run the generated C under ASan and UBSan
```

- `tests/pass`: lust and `rustc` must give the same stdout and exit status.
- `tests/fail`: `rustc` and lust must both reject the program, and lust must give the expected error code.
- `tests/reject`: valid Rust that lust rejects on purpose.

The corpus is small: 37 programs today (11 pass, 23 fail, 3 reject). It checks that the two compilers agree on these programs. It does not prove that lust matches `rustc` in general. The agent-loop benchmark adds one larger check: it compares both compilers after every edit.

## Code map

| File | Role |
|---|---|
| `src/client.rs`, `src/protocol.rs` | The thin `lust` client and the daemon wire format. |
| `src/main.rs`, `src/daemon.rs`, `src/driver.rs` | `lust-core`: cold compiles, the daemon, and the incremental pipeline. |
| `src/lexer.rs`, `src/scan.rs`, `src/parser.rs`, `src/ast.rs` | Syntax in flat arenas, plus the token scan that finds what changed. |
| `src/collect.rs`, `src/check.rs` | Item tables and signatures, then type inference, method resolution, and match exhaustiveness. |
| `src/lower.rs`, `src/ir.rs`, `src/borrowck.rs` | Lowering to a MIR-like CFG, then init, move, and loan dataflow. |
| `src/cgen.rs`, `src/runtime.c`, `src/tcc.rs` | C backend, runtime, and `tcc` integration. Checked math runs in 64-bit unsigned, so the C code never has signed overflow. |
