#!/usr/bin/env bash
# Tests for the agent kit commands and the versioned JSON diagnostics.
# The `lust test` parity checks against rustc live in scripts/testtest.sh, and the
# explain examples are checked by scripts/check-explain.sh.
# Usage: scripts/agentkit-test.sh
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
lust=$root/target/release/lust
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
pass=0
fail=0
report() {
  if [ "$1" = ok ]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $2: $3"; fi
}
check() { # name, condition detail...
  if [ "$2" = 0 ]; then report ok "$1"; else report bad "$1" "$3"; fi
}

# --- 1. JSON diagnostics: schema, spans, labels, source hash ---
"$lust" check --json "$root/tests/fail/two_mut_borrows.rs" > "$work/diag.json" 2>/dev/null
[ $? -eq 1 ]; check diag-exit $? "exit code must be 1"
python3 - "$work/diag.json" "$root/tests/fail/two_mut_borrows.rs" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
assert len(recs) == 1, recs
d = recs[0]
src = open(sys.argv[2], "rb").read()
def fnv(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{h:016x}"
assert d["schema"] == 1 and d["kind"] == "diagnostic"
assert d["code"] == "L0304" and d["severity"] == "error"
assert d["source_hash"] == fnv(src), (d["source_hash"], fnv(src))
span = d["span"]
assert src[span["start"]:span["end"]] == b"&mut s", src[span["start"]:span["end"]]
line_start = src.rfind(b"\n", 0, span["start"]) + 1
assert span["line"] == src[:span["start"]].count(b"\n") + 1
assert span["col"] == span["start"] - line_start + 1
texts = [l["label"] for l in d["labels"]]
assert "mutably borrowed here" in texts, texts
assert "borrow later used here" in texts, texts
borrow = [l for l in d["labels"] if l["label"] == "mutably borrowed here"][0]
assert src[borrow["start"]:borrow["end"]] == b"&mut s"
assert d["edits"] == [] and d["fix"]
PYEOF
check diag-schema $? "see the python assert above"

# Human output carries the same labels.
"$lust" check "$root/tests/fail/two_mut_borrows.rs" >"$work/diag.txt" 2>&1
grep -q "mutably borrowed here" "$work/diag.txt"; check diag-human-label $? "human output must show the label"

# --- 1b. Machine-applicable edits: integer mismatches carry a cast edit ---
# L0110 (mismatched integer types) and L0113 (an operator on two integer types)
# carry a `(expr) as T` edit; `auto` only for a widening cast. Applying every
# edit gives a program that lust and rustc both accept.
cat > "$work/cast.rs" <<'RSEOF'
fn widen(x: u32) -> u64 {
    let y: u64 = x << 2;
    let mut z: u64 = y + x;
    z += x;
    z
}
fn narrow(a: u64) -> u8 {
    let b: u8 = a;
    b
}
fn main() {
    println!("{} {}", widen(3), narrow(7));
}
RSEOF
"$lust" check --json "$work/cast.rs" > "$work/cast.json" 2>/dev/null
python3 - "$work/cast.json" "$work/cast.rs" "$work/cast_fixed.rs" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
errs = [d for d in recs if d.get("severity") == "error"]
assert {d["code"] for d in errs} == {"L0110", "L0113"}, [(d["code"], d["message"]) for d in errs]
assert all(len(d["edits"]) == 2 for d in errs), [d["edits"] for d in errs]
auto = {d["message"]: d["edits"][0]["auto"] for d in errs}
# u32 -> u64 widens (auto); u64 -> u8 narrows (offered, not auto).
assert any(a for m, a in auto.items() if "found `u32`" in m or "and `u32`" in m), auto
assert not auto["mismatched types: expected `u8`, found `u64`"], auto
src = open(sys.argv[2], "rb").read()
edits = sorted({(e["start"], e["end"], e["replacement"]) for d in errs for e in d["edits"]}, reverse=True)
for s, e, r in edits:
    src = src[:s] + r.encode() + src[e:]
open(sys.argv[3], "wb").write(src)
PYEOF
check diag-cast-edits $? "see the python assert above"
"$lust" check "$work/cast_fixed.rs" >/dev/null 2>&1; check diag-cast-edits-apply $? "the program with every edit applied must check"
if command -v rustc >/dev/null 2>&1; then
  rustc --edition 2024 --emit=metadata -o "$work/cast_fixed.rmeta" "$work/cast_fixed.rs" >/dev/null 2>&1
  check diag-cast-edits-rustc $? "rustc must accept the program with every edit applied"
fi

# --- 2. Bounded output: default cap, truncation record, --max-errors 0 ---
python3 -c '
lines = ["fn main() {"]
for i in range(25):
    lines.append(f"    let v{i}: i32 = \"x\";")
lines.append("}")
open("'"$work"'/many.rs", "w").write("\n".join(lines) + "\n")'
"$lust" check --json "$work/many.rs" > "$work/many.json" 2>/dev/null
[ $? -eq 1 ]; check trunc-exit $? "truncated output must still exit 1"
python3 - "$work/many.json" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
diags = [r for r in recs if r["kind"] == "diagnostic"]
trunc = [r for r in recs if r["kind"] == "truncated"]
assert len(diags) == 20, len(diags)
assert len(trunc) == 1 and trunc[0]["omitted"] == 5 and trunc[0]["total"] == 25, trunc
PYEOF
check trunc-default $? "default cap must show 20 and one truncation record"
"$lust" check --json --max-errors 0 "$work/many.rs" 2>/dev/null | grep -c '"kind":"diagnostic"' > "$work/n"
[ "$(cat "$work/n")" = 25 ]; check trunc-zero $? "--max-errors 0 must show all 25"
"$lust" check --json --max-errors=3 "$work/many.rs" 2>/dev/null | grep -c '"kind":"diagnostic"' > "$work/n"
[ "$(cat "$work/n")" = 3 ]; check trunc-three $? "--max-errors=3 must show 3"
"$lust" check "$work/many.rs" >"$work/many.txt" 2>&1
grep -q "more error(s) not shown" "$work/many.txt"; check trunc-human $? "human output must say errors were not shown"

# A changed file gives a different source hash.
h1=$("$lust" check --json "$work/many.rs" 2>/dev/null | head -1 | grep -o 'fnv1a64:[0-9a-f]*')
printf '\n' >> "$work/many.rs"
h2=$("$lust" check --json "$work/many.rs" 2>/dev/null | head -1 | grep -o 'fnv1a64:[0-9a-f]*')
[ -n "$h1" ] && [ "$h1" != "$h2" ]; check hash-change $? "source hash must change with the file ($h1 vs $h2)"

# The daemon wire format carries --max-errors too.
mkdir -p "$work/daemon"
(cd "$work/daemon" && LUST_DAEMON_SOCKET="$work/daemon/sock" exec "$lust" daemon) >"$work/daemon/log" 2>&1 &
daemon_pid=$!
for i in 1 2 3 4 5 6 7 8 9 10; do [ -S "$work/daemon/sock" ] && break; sleep 0.2; done
out=$(cd "$work/daemon" && LUST_DAEMON_SOCKET="$work/daemon/sock" timeout 60 "$lust" check --json --max-errors 2 "$work/many.rs" 2>/dev/null)
# Bounded stop: a daemon that ignores TERM gets KILL after 5 s, so the wait cannot hang.
kill "$daemon_pid" 2>/dev/null
for i in $(seq 50); do kill -0 "$daemon_pid" 2>/dev/null || break; sleep 0.1; done
kill -9 "$daemon_pid" 2>/dev/null
wait "$daemon_pid" 2>/dev/null
grep -q '"kind":"truncated"' <<<"$out"; check daemon-max-errors $? "daemon must honor --max-errors: $out"

# --- 3. explain ---
"$lust" explain L0301 > "$work/explain.txt" 2>&1; [ $? -eq 0 ]; check explain-exit $? "known code must exit 0"
diff -q "$work/explain.txt" "$root/docs/errors/L0301.md" >/dev/null; check explain-text $? "output must be the doc file"
"$lust" explain l0301 >/dev/null 2>&1; [ $? -eq 0 ]; check explain-lower $? "lowercase code must work"
"$lust" explain L9999 >/dev/null 2>&1; [ $? -eq 2 ]; check explain-unknown $? "unknown code must exit 2"
"$lust" explain > "$work/explain-list.txt" 2>/dev/null; grep -q "L0301: use of moved value" "$work/explain-list.txt"; check explain-list $? "no argument must list codes"

# --- 4. capabilities ---
"$lust" capabilities --json > "$work/cap.json" 2>/dev/null; [ $? -eq 0 ]; check cap-exit $? "must exit 0"
python3 - "$work/cap.json" "$root" <<'PYEOF'
import json, re, sys, pathlib
cap = json.load(open(sys.argv[1]))
root = pathlib.Path(sys.argv[2])
assert cap["schema"] == 1 and cap["kind"] == "capabilities"
assert "push" in cap["methods"]["Vec<T>"] and "unwrap" in cap["methods"]["Option<T>"]
assert cap["associated_functions"]["String"] == ["from", "from_utf16_lossy", "from_utf8", "from_utf8_lossy", "new", "with_capacity"]
# A pinned list of every std fn collided when two landings each added one
# (like the codes check below). Check the shape and a stable core instead.
fns = cap["functions"]
assert all(isinstance(v, list) and v == sorted(set(v)) for v in fns.values()), fns
core = {"std::env": {"args"}, "std::mem": {"drop", "forget"}, "core::mem": {"replace", "swap", "take"}, "std::thread": {"spawn", "scope", "sleep"}, "std::process": {"exit"}}
assert all(m in fns and names <= set(fns[m]) for m, names in core.items()), fns
assert len(cap["integer_types"]) == 12
# Every literal at an unsupported(...) call site is named in the rejected list.
features = [r["feature"] for r in cap["rejected"]]
src = "\n".join(p.read_text() for p in (root / "src").rglob("*.rs") if p.name != "capabilities.rs")
missing = []
for lit in re.findall(r'unsupported\([^,]+,\s*(?:&format!\()?"((?:[^"\\]|\\.)*)"', src):
    lit = lit.replace("{{", "{").replace("}}", "}")
    prefix = lit.split("{")[0]
    if len(prefix) >= 4 and not any(prefix in f for f in features):
        missing.append(prefix)
assert not missing, f"rejected list misses: {missing}"
# Named std types rejected in check/collect.rs appear by name.
m = re.search(r'\[("BTreeMap"[^]]*)\]', (root / "src/check/collect.rs").read_text())
names = re.findall(r'"([A-Za-z0-9]+)"', m.group(1))
assert all(any(n in f for f in features) for n in names), names
# The Builtin enum and ALL_BUILTINS have the same number of variants.
enum_body = re.search(r"pub enum Builtin \{(.*?)\n\}", src, re.S).group(1)
enum_count = len(re.findall(r"^\s+[A-Z]\w+(?:\([\w:]+\))?,", enum_body, re.M))
cap_src = (root / "src/capabilities.rs").read_text()
list_count = cap_src.count("Builtin::", 0, cap_src.index("];", cap_src.index("ALL_BUILTINS")))
assert enum_count == list_count, (enum_count, list_count)
# MacroKind variant count plus the token-level builtin list equals the
# macro list length.
macro_body = re.search(r"pub enum MacroKind \{(.*?)\n\}", src, re.S).group(1)
token_body = re.search(r"const TOKEN_MACROS: &\[&str\] = &\[(.*?)\];", cap_src, re.S).group(1)
token_count = len(re.findall(r'"[a-z_]+",', token_body))
assert len(re.findall(r"^\s+[A-Z]\w+,", macro_body, re.M)) + token_count == len(cap["macros"])
# Rejected codes are all documented codes.
codes = {c["code"] for c in cap["diagnostics"]["codes"]}
assert all(r["code"] in codes for r in cap["rejected"])
# Every code lust reports has an explain page, and every page is a code lust reports.
# (A pinned count collided when two landings each added a code.)
doc_codes = {p.stem for p in (root / "docs/errors").glob("L*.md")}
assert codes == doc_codes, (sorted(codes - doc_codes), sorted(doc_codes - codes))
PYEOF
check cap-content $? "see the python assert above"
"$lust" capabilities > "$work/cap.txt" 2>/dev/null; grep -q "methods on Vec<T>" "$work/cap.txt"; check cap-human $? "human output must list methods"
# Every supported and rejected line has a probe in tests/capabilities/ that shows it.
python3 "$root/scripts/test-capabilities.py" "$lust" > "$work/cap-probes.txt" 2>&1; check cap-probes $? "a capability line is stale (scripts/test-capabilities.py): $(grep FAIL "$work/cap-probes.txt" | head -5)"

# --- 5. test command edges (parity with rustc is in testtest.sh) ---
"$lust" test "$root/tests/testrun/basic.rs" --list > "$work/test-list.txt" 2>/dev/null; grep -q "test_add: test" "$work/test-list.txt"; check test-list $? "--list must name the tests"
printf '#[test]\nfn t(x: i32) {}\n' > "$work/badsig.rs"
"$lust" test "$work/badsig.rs" >/dev/null 2>&1; [ $? -eq 101 ]; check test-badsig $? "bad test signature must exit 101"
printf 'fn main() { let x: i32 = "s"; }\n#[test]\nfn t() {}\n' > "$work/badbuild.rs"
out=$("$lust" test "$work/badbuild.rs" --json 2>/dev/null)
grep -q '"build":"failed"' <<<"$out"; check test-buildfail-json $? "json summary must say the build failed"
printf 'fn main() {}\n' > "$work/notests.rs"
"$lust" test "$work/notests.rs" >/dev/null 2>&1; [ $? -eq 0 ]; check test-none $? "a file with no tests must exit 0"
"$lust" test "$root/tests/testrun/basic.rs" no-such-test >/dev/null 2>&1; [ $? -eq 0 ]; check test-filter-none $? "a filter that matches nothing must exit 0"
"$lust" test >/dev/null 2>&1; [ $? -eq 2 ]; check test-usage $? "missing file must exit 2"

echo "agentkit-test: $pass passed, $fail failed"
[ $fail -eq 0 ]
