// Cargo graph loading (docs/ARCH.md section 1.1).
//
// Lust does not write a Cargo resolver. Cargo owns version solving: this
// file runs `cargo metadata --format-version 1` once per graph key and
// reads packages, targets, dependencies (kind, rename, target cfg), and the
// resolve nodes. Per-unit features come from the resolve nodes when the
// graph shape makes them exact (one workspace member, no dependencies),
// and otherwise from one cached `cargo check --message-format=json` run
// for the same request, whose compiler-artifact messages carry the exact
// feature set of each selected unit.
//
// The graph key is the full resolver input: every ancestor `Cargo.toml`
// plus `Cargo.lock`, the requested package and target, the requested
// features with the default-feature policy, the target triple, the
// `.cargo/config.toml` chain that applies, and the env vars that steer
// Cargo. Two feature selections or two targets therefore key two
// sessions in one daemon (section 1.8). A cached graph also carries a
// stamp of what the key cannot see before metadata runs: the manifest
// and target discovery inputs of every path package (`watch_stamp`).
// A daemon checks the stamp on each request, so an added `src/bin/x.rs`
// or a sibling member's manifest edit reloads the graph.
use crate::diag::{Diag, Span};
use crate::intern::FxHasher;
use crate::resolve::cfg::CfgEnv;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

// ---------- Minimal JSON reader ----------
//
// `cargo metadata` output only. Objects keep insertion order in a Vec; the
// reader is total (any malformed byte is an Err, never a panic).

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kvs) => kvs.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }
    fn as_bool(&self) -> bool {
        matches!(self, Json::Bool(true))
    }
    fn strings(&self) -> Vec<String> {
        self.as_arr().iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
    }
}

fn parse_json(text: &str) -> Result<Json, String> {
    let b = text.as_bytes();
    let mut p = 0usize;
    let v = json_value(b, &mut p)?;
    Ok(v)
}

fn json_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && b[*p].is_ascii_whitespace() {
        *p += 1;
    }
}

fn json_value(b: &[u8], p: &mut usize) -> Result<Json, String> {
    json_ws(b, p);
    match b.get(*p) {
        Some(b'{') => {
            *p += 1;
            let mut kvs = Vec::new();
            loop {
                json_ws(b, p);
                if b.get(*p) == Some(&b'}') {
                    *p += 1;
                    return Ok(Json::Obj(kvs));
                }
                let Json::Str(k) = json_value(b, p)? else { return Err("object key is not a string".into()) };
                json_ws(b, p);
                if b.get(*p) != Some(&b':') {
                    return Err("expected `:` in object".into());
                }
                *p += 1;
                let v = json_value(b, p)?;
                kvs.push((k, v));
                json_ws(b, p);
                if b.get(*p) == Some(&b',') {
                    *p += 1;
                }
            }
        }
        Some(b'[') => {
            *p += 1;
            let mut vs = Vec::new();
            loop {
                json_ws(b, p);
                if b.get(*p) == Some(&b']') {
                    *p += 1;
                    return Ok(Json::Arr(vs));
                }
                vs.push(json_value(b, p)?);
                json_ws(b, p);
                if b.get(*p) == Some(&b',') {
                    *p += 1;
                }
            }
        }
        Some(b'"') => {
            *p += 1;
            let mut s = String::new();
            loop {
                match b.get(*p) {
                    None => return Err("unterminated string".into()),
                    Some(b'"') => {
                        *p += 1;
                        return Ok(Json::Str(s));
                    }
                    Some(b'\\') => {
                        *p += 1;
                        match b.get(*p) {
                            Some(b'n') => s.push('\n'),
                            Some(b't') => s.push('\t'),
                            Some(b'r') => s.push('\r'),
                            Some(b'u') => {
                                let hex = std::str::from_utf8(b.get(*p + 1..*p + 5).unwrap_or(&[])).unwrap_or("");
                                let mut cp = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u escape".to_string())?;
                                *p += 4;
                                // Surrogate pair for astral characters.
                                if (0xD800..0xDC00).contains(&cp) && b.get(*p + 1) == Some(&b'\\') && b.get(*p + 2) == Some(&b'u') {
                                    let hex2 = std::str::from_utf8(b.get(*p + 3..*p + 7).unwrap_or(&[])).unwrap_or("");
                                    if let Ok(lo) = u32::from_str_radix(hex2, 16) {
                                        cp = 0x10000 + ((cp - 0xD800) << 10) + lo.saturating_sub(0xDC00);
                                        *p += 6;
                                    }
                                }
                                s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            Some(&c) => s.push(c as char),
                            None => return Err("unterminated escape".into()),
                        }
                        *p += 1;
                    }
                    Some(_) => {
                        // The run up to the next quote or backslash at once: both
                        // are ASCII, so the run ends on a char boundary.
                        let start = *p;
                        while *p < b.len() && b[*p] != b'"' && b[*p] != b'\\' {
                            *p += 1;
                        }
                        s.push_str(std::str::from_utf8(&b[start..*p]).map_err(|_| "invalid utf-8 in string".to_string())?);
                    }
                }
            }
        }
        Some(b't') if b[*p..].starts_with(b"true") => {
            *p += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*p..].starts_with(b"false") => {
            *p += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*p..].starts_with(b"null") => {
            *p += 4;
            Ok(Json::Null)
        }
        Some(_) => {
            let start = *p;
            while *p < b.len() && matches!(b[*p], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                *p += 1;
            }
            let text = std::str::from_utf8(&b[start..*p]).unwrap_or("");
            text.parse::<f64>().map(Json::Num).map_err(|_| format!("bad number `{text}`"))
        }
        None => Err("unexpected end of JSON".into()),
    }
}

// ---------- Graph types ----------

/// One target of one package (a lib, bin, test, example, or bench crate).
#[derive(Clone, Debug)]
pub struct TargetInfo {
    pub kind: Vec<String>,
    pub crate_types: Vec<String>,
    pub name: String,
    pub src_path: PathBuf,
    pub edition: String,
    /// The `test = false`-style target flags; `test_flag` is Cargo's
    /// `test` key, which baselines runners turn off (section 1.1).
    pub test_flag: bool,
}

/// One resolved dependency of a package, before kind/target filtering.
#[derive(Clone, Debug)]
pub struct DepInfo {
    pub name: String,
    pub rename: Option<String>,
    pub kind: String,
    /// The `cfg(...)` target string, when the dependency is target-gated.
    pub target: Option<String>,
    /// The resolved package id repr, when the resolve graph names it.
    pub pkg: Option<String>,
}

/// One Cargo package in the graph.
#[derive(Clone, Debug)]
pub struct PackageInfo {
    /// Cargo's package id repr, for example
    /// `path+file:///x#name@0.1.0` or `registry+...#serde@1.0.0`.
    pub id_repr: String,
    pub name: String,
    pub version: String,
    pub edition: String,
    pub manifest_path: PathBuf,
    /// Manifest fields Cargo exposes to rustc as `CARGO_PKG_*` env vars.
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub homepage: Option<String>,
    pub license: Option<String>,
    pub license_file: Option<String>,
    pub repository: Option<String>,
    pub readme: Option<String>,
    pub rust_version: Option<String>,
    pub source: Option<String>,
    pub targets: Vec<TargetInfo>,
    pub dependencies: Vec<DepInfo>,
    pub features: Vec<String>,
    /// Features the resolve node activated for this package id.
    pub resolved_features: Vec<String>,
}

/// One selected compilation unit: a package target with its exact
/// per-unit feature set and its dependency tree (lib crates only).
#[derive(Clone, Debug)]
pub struct UnitInfo {
    pub package: PackageInfo,
    pub target: TargetInfo,
    /// The identity string the stable crate hash covers (section 1.7
    /// step 1): the package id for registry crates; for a path package,
    /// name, version, and the workspace-relative manifest path.
    pub package_id: String,
    pub features: Vec<String>,
    /// Env values Cargo sets for rustc for this unit, before build
    /// script outputs (`package_cargo_env`).
    pub cargo_env: Vec<(String, String)>,
    /// Direct dependency lib units, each with its own dependencies.
    pub deps: Vec<DepUnit>,
}

/// One dependency lib unit of a selected crate, recursively: everything
/// the resolver needs to classify the dependency (ARCH 1.1 table) and,
/// for a lust-source dependency, to compile it in the same workspace
/// session (ARCH 1.8).
#[derive(Clone, Debug)]
pub struct DepUnit {
    /// The name paths in the dependent crate use (rename or the package
    /// name, with `-` read as `_`).
    pub extern_name: String,
    /// The stable package identity string (`package_id_string`).
    pub package_id: String,
    /// The Cargo package name as written (may contain `-`).
    pub package_name: String,
    pub version: String,
    /// The package source: None for a path package, else the registry
    /// or git source string from metadata.
    pub source: Option<String>,
    /// The lib target's root file.
    pub lib_src: PathBuf,
    /// The lib target's name (the crate name, with `-` as written).
    pub lib_name: String,
    pub edition: String,
    /// The package manifest directory (`CARGO_MANIFEST_DIR`).
    pub manifest_dir: PathBuf,
    /// Env values Cargo sets for rustc for this unit. The driver adds
    /// build script outputs (`OUT_DIR`, `rustc-env`) for path packages.
    pub cargo_env: Vec<(String, String)>,
    /// Build script `rustc-cfg` values for this unit (driver-filled).
    pub build_cfgs: Vec<String>,
    /// Build script `rustc-check-cfg` values for this unit (driver-filled).
    pub build_check_cfgs: Vec<String>,
    /// The exact feature set of this unit, as for the selected unit.
    pub features: Vec<String>,
    pub deps: Vec<DepUnit>,
}

/// The full graph for one request.
#[derive(Clone, Debug)]
pub struct CrateGraph {
    /// The graph key mixed with the watch stamp: it names this exact
    /// graph (the features probe caches by it).
    pub key: u64,
    /// The path package dirs whose manifests and target discovery
    /// inputs the graph read (`watch_stamp`), and their stamp before
    /// metadata ran.
    pub watch: Vec<PathBuf>,
    pub stamp: u64,
    pub workspace_root: PathBuf,
    pub packages: Vec<PackageInfo>,
    pub workspace_members: Vec<String>,
}

/// The user's resolver request, mirroring the SessionKey fields (1.8).
#[derive(Clone, Debug, Default)]
pub struct GraphRequest {
    pub manifest_path: PathBuf,
    pub package: Option<String>,
    pub features: Vec<String>,
    pub default_features: bool,
    pub all_features: bool,
    pub target_triple: String,
    pub cargo_net: crate::protocol::CargoNet,
    /// The env vars that steer Cargo (`steers_cargo`) as the client sees
    /// them, sorted. None: this process's env (a cold run, unit tests).
    /// A daemon passes its client's: the daemon's own env is the env of
    /// the client that started it.
    pub env: Option<Vec<(String, String)>>,
}

/// True for an env var that changes what Cargo reads or runs: its
/// config and registry (`CARGO*`, `HOME`), the toolchain (`RUSTUP_*`,
/// `RUSTC`, `PATH`), and flags (`RUSTFLAGS`).
fn steers_cargo(name: &str) -> bool {
    name.starts_with("CARGO") || name.starts_with("RUSTUP_") || matches!(name, "RUSTC" | "RUSTFLAGS" | "HOME" | "PATH")
}

/// The Cargo-steering vars of a client env, sorted (`GraphRequest::env`).
pub fn cargo_env_of(client: &[(String, String)]) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = client.iter().filter(|(k, _)| steers_cargo(k)).cloned().collect();
    env.sort();
    env
}

/// Gives `cmd` a client's Cargo env (`GraphRequest::env`; None keeps
/// this process's): this process's steering vars are replaced by the
/// client's. Vars a command sets itself, such as the probe's
/// `RUSTC_WRAPPER`, do not steer and stay.
pub fn apply_cargo_env(cmd: &mut Command, env: Option<&[(String, String)]>) {
    let Some(env) = env else { return };
    for (k, _) in std::env::vars_os() {
        if k.to_str().is_some_and(steers_cargo) {
            cmd.env_remove(k);
        }
    }
    cmd.envs(env.iter().map(|(k, v)| (k, v)));
}

/// A `cargo` command for a client's Cargo env (`apply_cargo_env`), with
/// `cargo` found on the PATH that env gives. A command that sets PATH and
/// names a bare program makes std fork instead of posix_spawn, and a fork
/// copies the page table of every resident page of this process: several
/// GB after a front image restore (src/image.rs), hundreds of ms a spawn.
pub fn cargo_command(env: Option<&[(String, String)]>) -> Command {
    use std::os::unix::fs::PermissionsExt;
    let path = match env {
        Some(env) => env.iter().find(|(k, _)| k == "PATH").map(|(_, v)| v.clone()),
        None => std::env::var("PATH").ok(),
    };
    let found = path.iter().flat_map(std::env::split_paths).map(|d| d.join("cargo")).find(|c| c.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0));
    let mut cmd = Command::new(found.unwrap_or_else(|| PathBuf::from("cargo")));
    apply_cargo_env(&mut cmd, env);
    cmd
}

impl GraphRequest {
    /// One env var as Cargo sees it for this request.
    fn var(&self, name: &str) -> Option<String> {
        match &self.env {
            Some(env) => env.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()),
            None => std::env::var(name).ok(),
        }
    }
}

fn hash_bytes(h: &mut FxHasher, path: &Path) {
    path.to_string_lossy().hash(h);
    if let Ok(bytes) = std::fs::read(path) {
        bytes.hash(h);
    }
}

/// The graph key of section 1.1: manifest texts plus the lockfile, the
/// request, and the cargo config chain. Computed before metadata runs so
/// the same key can serve as the cache key for the metadata result: the
/// workspace member manifests join through the root manifest's own text
/// plus a directory walk of sibling member dirs is not possible without
/// metadata, so the key covers the named manifest, every ancestor
/// manifest, and the lock next to each. After metadata, `refined_key`
/// folds the member manifests in; the driver keys sessions by the
/// SessionKey, and this key keys the in-process graph cache.
pub fn graph_key(req: &GraphRequest) -> u64 {
    let mut h = FxHasher::default();
    "lust-graph-v1".hash(&mut h);
    let mut dir = req.manifest_path.parent();
    while let Some(d) = dir {
        let m = d.join("Cargo.toml");
        if m.exists() {
            hash_bytes(&mut h, &m);
            hash_bytes(&mut h, &d.join("Cargo.lock"));
        }
        // The config chain: `.cargo/config.toml` (and the legacy
        // `.cargo/config`) in every ancestor.
        hash_bytes(&mut h, &d.join(".cargo/config.toml"));
        hash_bytes(&mut h, &d.join(".cargo/config"));
        dir = d.parent();
    }
    let home = req.var("CARGO_HOME").map(PathBuf::from).or_else(|| req.var("HOME").map(|h| Path::new(&h).join(".cargo")));
    if let Some(home) = home {
        hash_bytes(&mut h, &home.join("config.toml"));
        hash_bytes(&mut h, &home.join("config"));
    }
    match &req.env {
        Some(env) => env.hash(&mut h),
        None => cargo_env_of(&std::env::vars().collect::<Vec<_>>()).hash(&mut h),
    }
    req.package.hash(&mut h);
    let mut feats = req.features.clone();
    feats.sort();
    feats.dedup();
    feats.hash(&mut h);
    req.default_features.hash(&mut h);
    req.all_features.hash(&mut h);
    req.target_triple.hash(&mut h);
    req.cargo_net.hash(&mut h);
    h.finish()
}

/// Hashes the inputs of Cargo's view of one path package dir that the
/// graph key does not cover: the manifest text, whether `build.rs`,
/// `src/main.rs`, and `src/lib.rs` exist (target discovery reads their
/// presence, not their text), and the entries of `src/bin`, `examples`,
/// `tests`, and `benches` (a file or a dir with a `main.rs` is a target).
fn hash_package_dir(h: &mut FxHasher, dir: &Path) {
    hash_bytes(h, &dir.join("Cargo.toml"));
    for f in ["build.rs", "src/main.rs", "src/lib.rs"] {
        dir.join(f).exists().hash(h);
    }
    for sub in ["src/bin", "examples", "tests", "benches"] {
        let mut names: Vec<(std::ffi::OsString, bool)> = std::fs::read_dir(dir.join(sub))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| (e.file_name(), e.path().join("main.rs").exists()))
            .collect();
        names.sort();
        names.hash(h);
    }
}

/// The watch stamp of a set of path package dirs (`hash_package_dir`).
fn watch_stamp(dirs: &[PathBuf]) -> u64 {
    let mut h = FxHasher::default();
    "lust-watch-v1".hash(&mut h);
    for d in dirs {
        d.hash(&mut h);
        hash_package_dir(&mut h, d);
    }
    h.finish()
}

/// The dirs a graph's watch stamp covers: the named manifest's dir and
/// every path package (workspace members and path dependencies).
fn watch_dirs(req: &GraphRequest, packages: &[PackageInfo]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = req.manifest_path.parent().map(Path::to_path_buf).into_iter().collect();
    for p in packages.iter().filter(|p| p.source.is_none()) {
        if let Some(d) = p.manifest_path.parent() {
            dirs.push(d.to_path_buf());
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

/// In-process cache of loaded graphs, by graph key. A daemon serves many
/// requests per graph; metadata runs once per key while inputs hold
/// (the key, and the graph's watch stamp).
fn graph_cache() -> &'static Mutex<std::collections::HashMap<u64, std::sync::Arc<CrateGraph>>> {
    static CACHE: OnceLock<Mutex<std::collections::HashMap<u64, std::sync::Arc<CrateGraph>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// The loaded graphs by graph key, as `graph_cache` holds them.
pub type GraphCache = std::collections::HashMap<u64, std::sync::Arc<CrateGraph>>;

/// Takes the in-process graph cache (it reads empty after), for a front
/// image to carry to the next build (src/image.rs `Roots`): an image build
/// is a new process, which would otherwise run `cargo metadata` every time.
pub fn take_graph_cache() -> GraphCache {
    std::mem::take(&mut *graph_cache().lock().unwrap())
}

/// Puts back a graph cache `take_graph_cache` gave an earlier build; each
/// entry stands while its key and watch stamp do (`load_graph`). An entry
/// that names a package manifest which is gone (a registry source removed
/// since) is dropped: `cargo metadata` must run again to fetch it.
pub fn put_graph_cache(mut cache: GraphCache) {
    cache.retain(|_, g| g.packages.iter().all(|p| p.manifest_path.exists()));
    *graph_cache().lock().unwrap() = cache;
}

/// Loads (or reuses) the Cargo graph for one request. Errors are message
/// strings: the driver turns them into code-2 responses, since a graph
/// Cargo itself rejects has no crate to attach a span diagnostic to.
pub fn load_graph(req: &GraphRequest) -> Result<std::sync::Arc<CrateGraph>, String> {
    let key = graph_key(req);
    let cached = graph_cache().lock().unwrap().get(&key).cloned();
    if let Some(g) = &cached
        && watch_stamp(&g.watch) == g.stamp
    {
        return Ok(g.clone());
    }
    // The stamp is taken before metadata runs, over the dirs the last
    // graph watched (else the named manifest's dir), so an edit made
    // while Cargo runs moves the stamp. When metadata names a path
    // package outside that set, the run repeats with it in the set.
    let mut watch = cached.map_or_else(|| watch_dirs(req, &[]), |g| g.watch.clone());
    for _ in 0..4 {
        let stamp = watch_stamp(&watch);
        let mut graph = run_metadata(req, key)?;
        let dirs = watch_dirs(req, &graph.packages);
        if dirs.iter().all(|d| watch.contains(d)) {
            let mut h = FxHasher::default();
            (key, stamp).hash(&mut h);
            (graph.key, graph.watch, graph.stamp) = (h.finish(), watch, stamp);
            let graph = std::sync::Arc::new(graph);
            graph_cache().lock().unwrap().insert(key, graph.clone());
            return Ok(graph);
        }
        watch.extend(dirs);
        watch.sort();
        watch.dedup();
    }
    Err("the Cargo package set kept changing while `cargo metadata` ran".into())
}

/// One `cargo metadata` run for `req`, read into a graph (key, watch,
/// and stamp are filled by `load_graph`).
fn run_metadata(req: &GraphRequest, key: u64) -> Result<CrateGraph, String> {
    let mut cmd = cargo_command(req.env.as_deref());
    // Run in the manifest's directory: Cargo discovers `.cargo/config.toml`
    // from the working directory, and the project's config (source
    // replacement, aliases) must apply as it does for the user there.
    if let Some(dir) = req.manifest_path.parent() {
        cmd.current_dir(dir);
    }
    cmd.args(["metadata", "--format-version", "1", "--manifest-path"]).arg(&req.manifest_path);
    if !req.features.is_empty() {
        cmd.args(["--features", &req.features.join(",")]);
    }
    if !req.default_features {
        cmd.arg("--no-default-features");
    }
    if req.all_features {
        cmd.arg("--all-features");
    }
    // Empty means the host (SessionKey): filter to it, so metadata
    // never downloads platform crates lust cannot target anyway
    // (a windows-only dependency of rustix broke offline resolves).
    let filter = if req.target_triple.is_empty() { crate::protocol::host_triple() } else { &req.target_triple };
    if !filter.is_empty() {
        cmd.args(["--filter-platform", filter]);
    }
    cmd.args(req.cargo_net.args());
    let out = cmd.output().map_err(|e| format!("cannot run `cargo metadata`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&out.stderr).lines().take(3).collect::<Vec<_>>().join(" ")
        ));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let root = parse_json(&text).map_err(|e| format!("cannot parse `cargo metadata` output: {e}"))?;
    let workspace_root = PathBuf::from(root.get("workspace_root").and_then(|v| v.as_str()).unwrap_or(""));
    let members: Vec<String> = root.get("workspace_members").map(|v| v.strings()).unwrap_or_default();
    // Resolve nodes: features per package id, and the exact dep edges.
    let mut node_features: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    // (dependency name as in the manifest, resolved package id, kind, target)
    let mut node_deps: std::collections::HashMap<String, Vec<(String, String, String, Option<String>)>> = std::collections::HashMap::new();
    if let Some(nodes) = root.get("resolve").and_then(|r| r.get("nodes")) {
        for n in nodes.as_arr() {
            let Some(id) = n.get("id").and_then(|v| v.as_str()) else { continue };
            node_features.insert(id.to_string(), n.get("features").map(|v| v.strings()).unwrap_or_default());
            let mut deps = Vec::new();
            for d in n.get("deps").map(|v| v.as_arr()).unwrap_or(&[]) {
                let name = d.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let pkg = d.get("pkg").and_then(|v| v.as_str()).unwrap_or("").to_string();
                for k in d.get("dep_kinds").map(|v| v.as_arr()).unwrap_or(&[]) {
                    let kind = match k.get("kind") {
                        Some(Json::Str(s)) => s.clone(),
                        _ => "normal".to_string(),
                    };
                    let target = k.get("target").and_then(|v| v.as_str()).map(str::to_string);
                    deps.push((name.clone(), pkg.clone(), kind, target));
                }
                if d.get("dep_kinds").is_none() {
                    deps.push((name, pkg, "normal".into(), None));
                }
            }
            node_deps.insert(id.to_string(), deps);
        }
    }
    let mut packages = Vec::new();
    for p in root.get("packages").map(|v| v.as_arr()).unwrap_or(&[]) {
        let id_repr = p.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let mut dependencies: Vec<DepInfo> = p
            .get("dependencies")
            .map(|v| v.as_arr())
            .unwrap_or(&[])
            .iter()
            .map(|d| DepInfo {
                name: d.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                rename: d.get("rename").and_then(|v| v.as_str()).map(str::to_string),
                kind: match d.get("kind") {
                    Some(Json::Str(s)) => s.clone(),
                    _ => "normal".to_string(),
                },
                target: d.get("target").and_then(|v| v.as_str()).map(str::to_string),
                pkg: None,
            })
            .collect();
        // Fill each dependency's resolved package by matching the resolve
        // edges of this package (dependency name plus kind plus target;
        // the resolve node's name is the dependency lib name, with `-`
        // read as `_` on both sides).
        if let Some(edges) = node_deps.get(&id_repr) {
            for dep in &mut dependencies {
                let want = dep.name.replace('-', "_");
                dep.pkg = edges
                    .iter()
                    .find(|(name, _, kind, target)| name.replace('-', "_") == want && *kind == dep.kind && *target == dep.target)
                    .or_else(|| edges.iter().find(|(name, _, kind, _)| name.replace('-', "_") == want && *kind == dep.kind))
                    .map(|(_, pkg, _, _)| pkg.clone());
            }
        }
        packages.push(PackageInfo {
            resolved_features: node_features.get(&id_repr).cloned().unwrap_or_default(),
            id_repr: id_repr.clone(),
            name: p.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            version: p.get("version").and_then(|v| v.as_str()).unwrap_or("0.0.0").to_string(),
            edition: p.get("edition").and_then(|v| v.as_str()).unwrap_or("2015").to_string(),
            manifest_path: PathBuf::from(p.get("manifest_path").and_then(|v| v.as_str()).unwrap_or("")),
            authors: p.get("authors").map(|v| v.strings()).unwrap_or_default(),
            description: p.get("description").and_then(|v| v.as_str()).map(str::to_string),
            homepage: p.get("homepage").and_then(|v| v.as_str()).map(str::to_string),
            license: p.get("license").and_then(|v| v.as_str()).map(str::to_string),
            license_file: p.get("license_file").and_then(|v| v.as_str()).map(str::to_string),
            repository: p.get("repository").and_then(|v| v.as_str()).map(str::to_string),
            readme: p.get("readme").and_then(|v| v.as_str()).map(str::to_string),
            rust_version: p.get("rust_version").and_then(|v| v.as_str()).map(str::to_string),
            source: p.get("source").and_then(|v| v.as_str()).map(str::to_string),
            targets: p
                .get("targets")
                .map(|v| v.as_arr())
                .unwrap_or(&[])
                .iter()
                .map(|t| TargetInfo {
                    kind: t.get("kind").map(|v| v.strings()).unwrap_or_default(),
                    crate_types: t.get("crate_types").map(|v| v.strings()).unwrap_or_default(),
                    name: t.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    src_path: PathBuf::from(t.get("src_path").and_then(|v| v.as_str()).unwrap_or("")),
                    edition: t.get("edition").and_then(|v| v.as_str()).unwrap_or("2015").to_string(),
                    test_flag: t.get("test").map(|v| v.as_bool()).unwrap_or(true),
                })
                .collect(),
            dependencies,
            features: p
                .get("features")
                .map(|v| match v {
                    Json::Obj(kvs) => kvs.iter().map(|(k, _)| k.clone()).collect(),
                    _ => Vec::new(),
                })
                .unwrap_or_default(),
        });
    }
    Ok(CrateGraph { key, watch: Vec::new(), stamp: 0, workspace_root, packages, workspace_members: members })
}

/// The directory a path package id names, decoded: Cargo writes a path
/// source as a `file://` URL, so `path+file:///w/sp%20ace%23%C3%A9#a@0.1.0`
/// (and the older `a 0.1.0 (path+file:///w/sp%20ace)`) is `/w/sp ace#é`.
/// None for any other source. Compare package ids with filesystem paths
/// only through this: a space, `#`, `%` or a non-ASCII char in the path
/// is percent-encoded in the id.
pub fn path_package_dir(id: &str) -> Option<PathBuf> {
    const PREFIX: &str = "path+file://";
    let rest = &id[id.find(PREFIX)? + PREFIX.len()..];
    let enc = &rest[..rest.find(['#', ')']).unwrap_or(rest.len())];
    let bytes = enc.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%'
            && let (Some(h), Some(l)) = (bytes.get(i + 1).and_then(|&b| hex(b)), bytes.get(i + 2).and_then(|&b| hex(b)))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    use std::os::unix::ffi::OsStringExt;
    Some(PathBuf::from(std::ffi::OsString::from_vec(out)))
}

/// The stable package identity of section 1.7 step 1.
pub fn package_id_string(graph: &CrateGraph, pkg: &PackageInfo) -> String {
    match &pkg.source {
        Some(source) => format!("{} {} ({})", pkg.name, pkg.version, source),
        None => {
            let rel = pkg.manifest_path.strip_prefix(&graph.workspace_root).unwrap_or(&pkg.manifest_path);
            format!("{} {} (path+{})", pkg.name, pkg.version, rel.display())
        }
    }
}

/// The env values Cargo sets for rustc when it builds one target of
/// one package (Cargo reference, "environment variables Cargo sets for
/// crates"), from the metadata of the package. Every `CARGO_PKG_*`
/// var is always set by Cargo, empty when the manifest field is
/// absent, so each one appears here. Not included: `CARGO` (the
/// cargo binary path; see REPORT gap), `OUT_DIR` and build-script
/// `rustc-env` pairs (the driver adds those from the recorded build
/// script output), and `DEP_*` links vars of dependencies.
pub fn package_cargo_env(pkg: &PackageInfo, target: &TargetInfo) -> Vec<(String, String)> {
    let manifest_dir = pkg.manifest_path.parent().map_or_else(String::new, |p| p.to_string_lossy().into_owned());
    // Version parts as Cargo splits them: major.minor.patch[-pre].
    let (core, pre) = match pkg.version.split_once('-') {
        Some((c, p)) => (c, p),
        None => (pkg.version.as_str(), ""),
    };
    let mut nums = core.split('.');
    let (major, minor, patch) = (nums.next().unwrap_or(""), nums.next().unwrap_or(""), nums.next().unwrap_or(""));
    let empty = String::new();
    let mut env: Vec<(String, String)> = vec![
        ("CARGO_MANIFEST_DIR".into(), manifest_dir),
        ("CARGO_MANIFEST_PATH".into(), pkg.manifest_path.to_string_lossy().into_owned()),
        ("CARGO_PKG_NAME".into(), pkg.name.clone()),
        ("CARGO_PKG_VERSION".into(), pkg.version.clone()),
        ("CARGO_PKG_VERSION_MAJOR".into(), major.to_string()),
        ("CARGO_PKG_VERSION_MINOR".into(), minor.to_string()),
        ("CARGO_PKG_VERSION_PATCH".into(), patch.to_string()),
        ("CARGO_PKG_VERSION_PRE".into(), pre.to_string()),
        ("CARGO_PKG_AUTHORS".into(), pkg.authors.join(":")),
        ("CARGO_PKG_DESCRIPTION".into(), pkg.description.clone().unwrap_or_default()),
        ("CARGO_PKG_HOMEPAGE".into(), pkg.homepage.clone().unwrap_or_default()),
        ("CARGO_PKG_LICENSE".into(), pkg.license.clone().unwrap_or_default()),
        ("CARGO_PKG_LICENSE_FILE".into(), pkg.license_file.clone().unwrap_or_default()),
        ("CARGO_PKG_REPOSITORY".into(), pkg.repository.clone().unwrap_or_default()),
        ("CARGO_PKG_README".into(), pkg.readme.clone().unwrap_or(empty)),
        ("CARGO_PKG_RUST_VERSION".into(), pkg.rust_version.clone().unwrap_or_default()),
        ("CARGO_CRATE_NAME".into(), target.name.replace('-', "_")),
    ];
    if target.kind.iter().any(|k| k == "bin") {
        env.push(("CARGO_BIN_NAME".into(), target.name.clone()));
    }
    env
}

/// Cargo's build dir for `manifest` (`build_directory` of `cargo metadata`,
/// else `target_directory` for a Cargo without build-dir): the probe of
/// `resolve::probe` needs its deps dir. None when the call fails.
pub fn build_dir(manifest: &Path, env: Option<&[(String, String)]>) -> Option<PathBuf> {
    let mut cmd = cargo_command(env);
    if let Some(dir) = manifest.parent() {
        cmd.current_dir(dir);
    }
    let out = cmd.args(["metadata", "--format-version", "1", "--no-deps", "--manifest-path"]).arg(manifest).output().ok()?;
    let root = parse_json(&String::from_utf8_lossy(&out.stdout)).ok()?;
    let dir = root.get("build_directory").or_else(|| root.get("target_directory"))?.as_str()?;
    (out.status.success() && !dir.is_empty()).then(|| PathBuf::from(dir))
}

/// The per-unit feature sets from one `cargo check --message-format=json`
/// run for the same request (section 1.1), keyed by (package id repr,
/// target name). The run is the cheap probe of `resolve::probe`: rustc
/// runs only for build scripts and proc macros. Cached per `graph.key`
/// (graph key plus watch stamp) in this process, and on disk (see `features_disk_path`): its answer does
/// not depend on the sources. Returns None when the run fails; the caller then falls back
/// to resolve-node features and the session still works for single-package
/// graphs.
fn unit_features_via_check(req: &GraphRequest, graph: &CrateGraph) -> Option<std::collections::HashMap<(String, String), Vec<String>>> {
    let key = graph.key;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<u64, std::collections::HashMap<(String, String), Vec<String>>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return Some(v.clone());
    }
    let disk = features_disk_path(graph);
    if let Some(map) = disk.as_deref().and_then(read_features_file) {
        cache.lock().unwrap().insert(key, map.clone());
        return Some(map);
    }
    let mut cmd = crate::resolve::probe::check_command(&req.manifest_path, &req.target_triple, req.env.as_deref());
    if let Some(pkg) = &req.package {
        cmd.args(["-p", pkg]);
    }
    if !req.features.is_empty() {
        cmd.args(["--features", &req.features.join(",")]);
    }
    if !req.default_features {
        cmd.arg("--no-default-features");
    }
    if req.all_features {
        cmd.arg("--all-features");
    }
    cmd.args(req.cargo_net.args());
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let Ok(v) = parse_json(line) else { continue };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let pkg = v.get("package_id").and_then(|p| p.as_str()).unwrap_or("").to_string();
        let target = v.get("target").and_then(|t| t.get("name")).and_then(|n| n.as_str()).unwrap_or("").to_string();
        let feats = v.get("features").map(|f| f.strings()).unwrap_or_default();
        map.insert((pkg, target), feats);
    }
    if map.is_empty() {
        return None;
    }
    if let Some(path) = disk {
        write_features_file(&path, &map);
    }
    cache.lock().unwrap().insert(key, map.clone());
    Some(map)
}

/// The disk cache file of `unit_features_via_check`:
/// `$LUST_CACHE_DIR` (else `<workspace root>/.lust-cache`)`/features-<key>.txt`.
/// The key is `graph.key`: the graph key plus the watch stamp, which
/// covers every path package manifest (a sibling member is not an
/// ancestor of the named manifest, so the graph key alone misses it).
/// None under `LUST_NO_CACHE`.
fn features_disk_path(graph: &CrateGraph) -> Option<PathBuf> {
    if matches!(std::env::var("LUST_NO_CACHE"), Ok(v) if !v.is_empty() && v != "0") {
        return None;
    }
    let mut h = FxHasher::default();
    "lust-features-v2".hash(&mut h);
    graph.key.hash(&mut h);
    let dir = match std::env::var_os("LUST_CACHE_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => graph.workspace_root.join(".lust-cache"),
    };
    Some(dir.join(format!("features-{:016x}.txt", h.finish())))
}

/// One line per unit: package id, tab, target name, tab, comma-joined features.
fn write_features_file(path: &Path, map: &std::collections::HashMap<(String, String), Vec<String>>) {
    let mut text: String = map.iter().map(|((p, t), f)| format!("{p}\t{t}\t{}\n", f.join(","))).collect();
    text.insert_str(0, "lust-features-v1\n");
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::create_dir_all(path.parent().unwrap_or(Path::new("."))).is_ok() && std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

fn read_features_file(path: &Path) -> Option<std::collections::HashMap<(String, String), Vec<String>>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    if lines.next()? != "lust-features-v1" {
        return None;
    }
    let mut map = std::collections::HashMap::new();
    for line in lines {
        let mut parts = line.splitn(3, '\t');
        let (p, t, f) = (parts.next()?, parts.next()?, parts.next()?);
        let feats = if f.is_empty() { Vec::new() } else { f.split(',').map(str::to_string).collect() };
        map.insert((p.to_string(), t.to_string()), feats);
    }
    (!map.is_empty()).then_some(map)
}

/// Evaluates a Cargo dependency target string (`cfg(windows)`,
/// `cfg(target_arch = "wasm32")`) against the session cfg environment.
/// The string is re-lexed and read through the same predicate evaluator
/// as source `#[cfg]`, so the two never disagree.
pub fn dep_target_matches(env: &CfgEnv, target: Option<&str>, interner: &crate::intern::Interner) -> bool {
    let Some(target) = target else { return true };
    let src = format!("#[{target}]\nmod m;");
    let Ok(lexed) = crate::lexer::lex(&src, crate::diag::FileId(0), 0, interner) else { return false };
    let o = crate::scan::outline(&lexed, interner);
    match o.mods.first() {
        Some(m) => crate::resolve::cfg::outline_enabled(env, &m.cfg, &lexed.strings, interner, Span::default()).unwrap_or(false),
        None => false,
    }
}

/// Picks one unit from the graph: the package (by name, or the one whose
/// manifest the request names) and the target (kind plus name). The
/// unit's features prefer the `cargo check` dry run; for a graph with a
/// single workspace member and no dependencies the resolve-node features
/// are exact and no check runs.
pub fn select_unit(
    graph: &CrateGraph,
    req: &GraphRequest,
    target_kind: &str,
    target_name: &str,
    env: &CfgEnv,
    interner: &crate::intern::Interner,
    test_crate: bool,
) -> Result<UnitInfo, String> {
    let pkg = graph
        .packages
        .iter()
        .find(|p| Some(&p.name) == req.package.as_ref())
        .or_else(|| graph.packages.iter().find(|p| p.manifest_path == req.manifest_path))
        .or_else(|| graph.packages.iter().find(|p| graph.workspace_members.contains(&p.id_repr)))
        .ok_or_else(|| format!("no package in the Cargo graph matches {}", req.manifest_path.display()))?;
    // `--lib` selects the package's lib target by kind, not by name: a
    // package has at most one, and Cargo names it after `[lib] name`, which
    // defaults to the package name with `-` as `_`. Matching the package
    // name failed for `foo-bar` packages and picked a bin named after the
    // package when the lib was renamed (issue #4). Other kinds, and a lib
    // whose kinds are neither `lib` nor `rlib`, keep the name lookup.
    let lib_by_kind =
        if target_kind == "lib" { pkg.targets.iter().find(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib")) } else { None };
    let target = lib_by_kind
        .or_else(|| pkg.targets.iter().find(|t| t.name == target_name && t.kind.iter().any(|k| k == target_kind)))
        .or_else(|| pkg.targets.iter().find(|t| t.name == target_name))
        .ok_or_else(|| format!("package `{}` has no {target_kind} target named `{target_name}`", pkg.name))?;
    let simple_graph = graph.workspace_members.len() <= 1 && pkg.dependencies.is_empty();
    let check_map = if simple_graph { None } else { unit_features_via_check(req, graph) };
    let features = check_map
        .as_ref()
        .and_then(|m| m.get(&(pkg.id_repr.clone(), target.name.clone())).cloned())
        .unwrap_or_else(|| pkg.resolved_features.clone());
    // Dependency tree: direct dependencies whose kind and target select
    // them for this unit, recursively. Dev-dependencies join only the
    // selected test crate itself, never its dependencies.
    let mut stack = vec![pkg.id_repr.clone()];
    let mut deps = build_deps(graph, pkg, env, interner, test_crate, &check_map, &mut stack);
    // A bin, test, example, or bench target of a package with a lib
    // target depends on that lib implicitly (Cargo adds it as an
    // extern named by the lib crate name; it is not in the manifest
    // dependency list). The same holds in ts-rust (the `tsgo` bin
    // uses the `ts_goport` lib).
    let target_is_lib = target.kind.iter().any(|k| k == "lib" || k == "rlib");
    if !target_is_lib {
        if let Some(lib) = pkg.targets.iter().find(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib")) {
            let lib_features = check_map
                .as_ref()
                .and_then(|m| m.get(&(pkg.id_repr.clone(), lib.name.clone())).cloned())
                .unwrap_or_else(|| features.clone());
            deps.insert(
                0,
                DepUnit {
                    extern_name: lib.name.replace('-', "_"),
                    package_id: package_id_string(graph, pkg),
                    package_name: pkg.name.clone(),
                    version: pkg.version.clone(),
                    source: pkg.source.clone(),
                    lib_src: lib.src_path.clone(),
                    lib_name: lib.name.clone(),
                    edition: lib.edition.clone(),
                    manifest_dir: pkg.manifest_path.parent().map_or_else(PathBuf::new, Path::to_path_buf),
                    cargo_env: package_cargo_env(pkg, lib),
                    build_cfgs: Vec::new(),
                    build_check_cfgs: Vec::new(),
                    features: lib_features,
                    deps: deps.clone(),
                },
            );
        }
    }
    Ok(UnitInfo {
        package_id: package_id_string(graph, pkg),
        package: pkg.clone(),
        target: target.clone(),
        features,
        cargo_env: package_cargo_env(pkg, target),
        deps,
    })
}

/// Builds the lib dependency units of one package, recursively. A
/// package already on the stack (a dev-dependency cycle, the only cycle
/// Cargo allows) is not entered again; its unit still appears at the
/// level where it was first reached.
fn build_deps(
    graph: &CrateGraph,
    pkg: &PackageInfo,
    env: &CfgEnv,
    interner: &crate::intern::Interner,
    test_crate: bool,
    check_map: &Option<std::collections::HashMap<(String, String), Vec<String>>>,
    stack: &mut Vec<String>,
) -> Vec<DepUnit> {
    let mut out = Vec::new();
    for dep in &pkg.dependencies {
        let kind_ok = dep.kind == "normal" || (test_crate && dep.kind == "dev");
        if !kind_ok || !dep_target_matches(env, dep.target.as_deref(), interner) {
            continue;
        }
        let Some(dep_pkg_id) = &dep.pkg else { continue };
        let Some(dep_pkg) = graph.packages.iter().find(|p| &p.id_repr == dep_pkg_id) else { continue };
        let Some(lib) = dep_pkg.targets.iter().find(|t| t.kind.iter().any(|k| k == "lib" || k == "rlib")) else { continue };
        let extern_name = dep.rename.clone().unwrap_or_else(|| dep.name.replace('-', "_"));
        let features = check_map
            .as_ref()
            .and_then(|m| m.get(&(dep_pkg.id_repr.clone(), lib.name.clone())).cloned())
            .unwrap_or_else(|| dep_pkg.resolved_features.clone());
        let sub = if stack.contains(&dep_pkg.id_repr) {
            Vec::new()
        } else {
            stack.push(dep_pkg.id_repr.clone());
            let s = build_deps(graph, dep_pkg, env, interner, false, check_map, stack);
            stack.pop();
            s
        };
        out.push(DepUnit {
            extern_name,
            package_id: package_id_string(graph, dep_pkg),
            package_name: dep_pkg.name.clone(),
            version: dep_pkg.version.clone(),
            source: dep_pkg.source.clone(),
            lib_src: lib.src_path.clone(),
            lib_name: lib.name.clone(),
            edition: lib.edition.clone(),
            manifest_dir: dep_pkg.manifest_path.parent().map_or_else(PathBuf::new, Path::to_path_buf),
            cargo_env: package_cargo_env(dep_pkg, lib),
            build_cfgs: Vec::new(),
            build_check_cfgs: Vec::new(),
            features,
            deps: sub,
        });
    }
    out
}

/// The edition gate for a selected unit (section 1.1): lust source crates
/// must be edition 2024. Applied only to selected units.
pub fn check_edition(unit: &UnitInfo) -> Result<(), Diag> {
    if unit.target.edition == "2024" {
        return Ok(());
    }
    Err(Diag::new(
        "L0530",
        Span::default(),
        format!("crate `{}` is edition {}, but lust source crates must be edition 2024", unit.package.name, unit.target.edition),
        "Set `edition = \"2024\"` in the package manifest.",
    ))
}

/// Reads `[lints]` and `[workspace.lints]` from the manifest texts of a
/// selected unit (cargo metadata does not report lints). `unsafe_code` at
/// forbid/deny is enforced by the parser's blanket `unsafe` reject; any
/// other rust lint at deny/forbid rejects the session with L0531 naming
/// the lint, because rustc would reject code that triggers it and lust
/// does not check it (section 1.1).
pub fn check_lints(graph: &CrateGraph, unit: &UnitInfo) -> Result<(), Diag> {
    let mut texts: Vec<String> = Vec::new();
    if let Ok(t) = std::fs::read_to_string(&unit.package.manifest_path) {
        texts.push(t);
    }
    let root_manifest = graph.workspace_root.join("Cargo.toml");
    if root_manifest != unit.package.manifest_path
        && let Ok(t) = std::fs::read_to_string(&root_manifest)
    {
        texts.push(t);
    }
    for text in &texts {
        if let Some(lint) = first_deny_lint(text) {
            return Err(Diag::new(
                "L0531",
                Span::default(),
                format!("lint `{lint}` is denied in the manifest, but lust does not check it"),
                "Lower the lint to warn for lust dev builds, or build this crate with Cargo.",
            ));
        }
    }
    Ok(())
}

/// The first non-`unsafe_code` rust lint at deny or forbid level in one
/// manifest text, from `[lints]` / `[workspace.lints]` (inline-table form)
/// or `[lints.rust]` / `[workspace.lints.rust]` (table form).
fn first_deny_lint(text: &str) -> Option<String> {
    let mut section = String::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            section = line.trim_matches(|c| c == '[' || c == ']').trim().to_string();
            continue;
        }
        let in_lints = matches!(section.as_str(), "lints" | "workspace.lints" | "lints.rust" | "workspace.lints.rust");
        if !in_lints || !line.contains('=') {
            continue;
        }
        // Two shapes: `rust = { unsafe_code = "forbid", dead_code = "deny" }`
        // and one `name = "deny"` per line under `[lints.rust]`.
        let is_table_form = matches!(section.as_str(), "lints.rust" | "workspace.lints.rust");
        if is_table_form {
            let (name, value) = line.split_once('=')?;
            let name = name.trim();
            let level = value.trim().trim_matches('"');
            if (level == "deny" || level == "forbid") && name != "unsafe_code" && !name.contains("::") {
                return Some(name.to_string());
            }
        } else if let Some((_, inner)) = line.split_once('{') {
            for part in inner.trim_end_matches('}').split(',') {
                let Some((name, value)) = part.split_once('=') else { continue };
                let name = name.trim();
                let level = value.trim().trim_matches('"');
                if (level == "deny" || level == "forbid") && name != "unsafe_code" && !name.contains("::") {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #[test]
    fn path_package_dir_decodes() {
        let d = |id: &str| super::path_package_dir(id).map(|p| p.to_string_lossy().into_owned());
        assert_eq!(d("path+file:///w/sp%20ace%23%C3%A9#a@0.1.0").as_deref(), Some("/w/sp ace#é"));
        assert_eq!(d("path+file:///w/plain#0.4.0").as_deref(), Some("/w/plain"));
        // `%` itself is `%25`: decoded once, `%41` stays literal.
        assert_eq!(d("path+file:///w/pc%2541#x@1.0.0").as_deref(), Some("/w/pc%41"));
        assert_eq!(d("a 0.1.0 (path+file:///w/sp%20ace)").as_deref(), Some("/w/sp ace"));
        assert_eq!(d("registry+https://github.com/rust-lang/crates.io-index#serde@1.0.0"), None);
    }

    use super::*;

    #[test]
    fn json_reader_handles_metadata_shapes() {
        let v = parse_json(r#"{"a":[1,"x",true,null],"b":{"c":"d\né"},"n":-1.5e3}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().len(), 4);
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str(), Some("d\né"));
        assert_eq!(v.get("n"), Some(&Json::Num(-1500.0)));
        assert!(parse_json("{").is_err());
    }

    #[test]
    fn features_file_round_trip() {
        let dir = std::env::temp_dir().join(format!("lust-features-test-{}", std::process::id()));
        let path = dir.join("features-0.txt");
        let mut map = std::collections::HashMap::new();
        map.insert(("path+file:///w#a@0.1.0".to_string(), "a".to_string()), vec!["default".to_string(), "std".to_string()]);
        map.insert(("registry+x#b@1.0.0".to_string(), "b".to_string()), vec![]);
        write_features_file(&path, &map);
        assert_eq!(read_features_file(&path), Some(map));
        std::fs::write(&path, "other-format\n").unwrap();
        assert_eq!(read_features_file(&path), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deny_lint_scan() {
        let inline = "[workspace.lints.rust]\nunsafe_code = \"forbid\"\n";
        assert_eq!(first_deny_lint(inline), None);
        let table = "[lints.rust]\ndead_code = \"deny\"\n";
        assert_eq!(first_deny_lint(table).as_deref(), Some("dead_code"));
        let inline_table = "[lints]\nrust = { missing_docs = \"deny\" }\n";
        assert_eq!(first_deny_lint(inline_table).as_deref(), Some("missing_docs"));
        let warn = "[lints.rust]\ndead_code = \"warn\"\n";
        assert_eq!(first_deny_lint(warn), None);
    }

    #[test]
    fn graph_of_a_fixture_package() {
        // The checked-in hello fixture: one package, one bin, edition 2024.
        // Copied to a temp dir first: `cargo metadata` writes a Cargo.lock
        // next to the manifest, and the repo fixture must stay clean.
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/crates/hello");
        let dir = std::env::temp_dir().join(format!("lust-cargo-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::copy(src_dir.join("Cargo.toml"), dir.join("Cargo.toml")).unwrap();
        std::fs::copy(src_dir.join("src/main.rs"), dir.join("src/main.rs")).unwrap();
        let req = GraphRequest {
            manifest_path: dir.join("Cargo.toml"),
            package: None,
            features: vec![],
            default_features: true,
            all_features: false,
            target_triple: String::new(),
            cargo_net: Default::default(),
            env: None,
        };
        let graph = load_graph(&req).expect("cargo metadata on the hello fixture");
        let interner = crate::intern::Interner::default();
        let env = CfgEnv::new("", &[], false, crate::protocol::ProfileFlags::DEV);
        let unit = select_unit(&graph, &req, "bin", "hello", &env, &interner, false).expect("hello unit");
        assert_eq!(unit.package.name, "hello");
        assert_eq!(unit.target.edition, "2024");
        assert!(unit.target.src_path.ends_with("src/main.rs"));
        assert!(unit.package_id.contains("hello 0.1.0"), "{}", unit.package_id);
        check_edition(&unit).unwrap();
        check_lints(&graph, &unit).unwrap();
    }

    #[test]
    fn lib_target_is_selected_by_kind_not_package_name() {
        // Issue #4: Cargo names the lib target after `[lib] name` (default:
        // the package name with `-` as `_`), so `--lib` must not look the
        // target up by the package name.
        let dir = std::env::temp_dir().join(format!("lust-cargo-libname-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let interner = crate::intern::Interner::default();
        let env = CfgEnv::new("", &[], false, crate::protocol::ProfileFlags::DEV);
        let graph_of = |name: &str, manifest: &str, files: &[(&str, &str)]| {
            let pkg_dir = dir.join(name);
            std::fs::create_dir_all(pkg_dir.join("src")).unwrap();
            std::fs::write(pkg_dir.join("Cargo.toml"), manifest).unwrap();
            for (file, text) in files {
                std::fs::write(pkg_dir.join(file), text).unwrap();
            }
            let req = GraphRequest {
                manifest_path: pkg_dir.join("Cargo.toml"),
                package: None,
                features: vec![],
                default_features: true,
                all_features: false,
                target_triple: String::new(),
                cargo_net: Default::default(),
                env: None,
            };
            let graph = load_graph(&req).expect("cargo metadata on a temp package");
            (graph, req)
        };
        // A hyphenated package: the lib target is `hyphen_lib`.
        let (graph, req) = graph_of(
            "hyphen-lib",
            "[package]\nname = \"hyphen-lib\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[workspace]\n",
            &[("src/lib.rs", "pub fn add(a: u32, b: u32) -> u32 { a + b }\n")],
        );
        let unit = select_unit(&graph, &req, "lib", "hyphen-lib", &env, &interner, false).expect("lib unit of a hyphenated package");
        assert!(unit.target.kind.iter().any(|k| k == "lib"), "{:?}", unit.target.kind);
        assert!(unit.target.src_path.ends_with("src/lib.rs"));
        // A renamed lib beside a bin named after the package: `--lib` is the lib.
        let (graph, req) = graph_of(
            "app",
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lib]\nname = \"app_lib\"\n\n[workspace]\n",
            &[("src/lib.rs", "pub fn answer() -> u32 { 42 }\n"), ("src/main.rs", "fn main() { println!(\"{}\", app_lib::answer()); }\n")],
        );
        let lib = select_unit(&graph, &req, "lib", "app", &env, &interner, false).expect("renamed lib unit");
        assert_eq!(lib.target.name, "app_lib");
        assert!(lib.target.src_path.ends_with("src/lib.rs"));
        let bin = select_unit(&graph, &req, "bin", "app", &env, &interner, false).expect("bin unit");
        assert!(bin.target.src_path.ends_with("src/main.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn graph_reloads_when_discovery_or_a_path_manifest_moves() {
        // A daemon keeps graphs; an added bin or a path dependency's
        // manifest edit moves none of the graph key's files.
        let dir = std::env::temp_dir().join(format!("lust-cargo-watch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src/bin")).unwrap();
        std::fs::create_dir_all(dir.join("util/src")).unwrap();
        let pkg = "[package]\nname = \"w\"\nversion = \"0.1.0\"\nedition = \"2024\"\n";
        std::fs::write(dir.join("Cargo.toml"), format!("{pkg}\n[dependencies]\nutil = {{ path = \"util\" }}\n\n[workspace]\n")).unwrap();
        std::fs::write(dir.join("util/Cargo.toml"), "[package]\nname = \"util\"\nversion = \"0.1.0\"\nedition = \"2024\"\n").unwrap();
        std::fs::write(dir.join("util/src/lib.rs"), "").unwrap();
        std::fs::write(dir.join("src/bin/a.rs"), "fn main() {}").unwrap();
        let req = GraphRequest { manifest_path: dir.join("Cargo.toml"), default_features: true, ..Default::default() };
        let bins = |g: &CrateGraph| g.packages.iter().find(|p| p.name == "w").unwrap().targets.iter().map(|t| t.name.clone()).collect::<Vec<_>>();
        // The first run writes Cargo.lock, which the graph key covers.
        load_graph(&req).expect("metadata");
        let first = load_graph(&req).unwrap();
        assert_eq!(bins(&first), ["a"]);
        assert!(std::sync::Arc::ptr_eq(&first, &load_graph(&req).unwrap()), "inputs hold: the cached graph");
        std::fs::write(dir.join("src/bin/b.rs"), "fn main() {}").unwrap();
        let second = load_graph(&req).unwrap();
        assert_eq!(bins(&second), ["a", "b"]);
        std::fs::write(dir.join("util/Cargo.toml"), "[package]\nname = \"util\"\nversion = \"0.1.0\"\nedition = \"2021\"\n").unwrap();
        let third = load_graph(&req).unwrap();
        assert_eq!(third.packages.iter().find(|p| p.name == "util").unwrap().edition, "2021");
        assert_ne!(second.key, third.key, "the features probe caches by the graph key");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
