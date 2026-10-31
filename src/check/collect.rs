// Collection: builds the item tables (types, functions, methods) and resolves all signatures.
// This is the only stage that runs in sequence. Function bodies are checked after it, in parallel.
use crate::ast::{self, Ast, ItemKind, SelfKind, TyExpr, VariantFields};
use crate::check::generics::{self, GenTables};
pub use crate::def::{CrateId, DefId, DefPathHash, LocalDefId, StableCrateHash};
use crate::def::{CrateIds, DiskDefId};
use crate::diag::{CtxtId, Diag, Span, not_yet};
use crate::intern::FxMap as HashMap;
use crate::intern::{Interner, Known, Sym};
use std::sync::Arc;
use crate::ty::defs::{DefKind as DefsKind, Vis};
use crate::ty::lang::ShippedCrate;
use crate::ty::*;

/// The derive list of one Adt. Derived impls are synthetic impl
/// records (generics.rs `add_synthetic_impls`); their bodies are
/// codegen glue over the fields (cgen.rs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Derives {
    pub debug: bool,
    pub clone: bool,
    pub copy: bool,
    pub partial_eq: bool,
    pub eq: bool,
    pub hash: bool,
    pub partial_ord: bool,
    pub ord: bool,
    pub default: bool,
    /// The `#[default]` variant of an enum that derives `Default`.
    pub default_variant: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VariantShape {
    Unit,
    Tuple,
    Named,
}

pub struct Field {
    pub name: Sym,
    pub ty: Ty,
    /// The span of the field's written type (the Adt's own span for a
    /// synthetic field). Written-type diagnostics (WF bounds) point
    /// here, at the field that owes the bound, not at the Adt name.
    pub span: Span,
    /// The parts of the written type that rustc's derive bounds besides
    /// the type params (`check_derive_bounds`): each path that starts at
    /// a type param of the Adt (`T::Item` in `Vec<T::Item>`; not
    /// `<T as Tr>::Item`). Empty for synthetic fields and instances.
    pub derive_bounds: Vec<Ty>,
}

pub struct Variant {
    pub name: Sym,
    pub shape: VariantShape,
    pub fields: Vec<Field>,
    /// The discriminant value (explicit, or the previous variant's plus one).
    pub disc: i128,
    /// `#[non_exhaustive]` on this variant: outside the defining
    /// crate it cannot be built with a struct literal, its tuple
    /// constructor is private, and a named-field pattern needs `..`.
    pub non_exhaustive: bool,
}

pub struct AdtDef {
    pub name: Sym,
    pub is_enum: bool,
    /// A struct has exactly one variant, named after the struct.
    pub variants: Vec<Variant>,
    /// Private: readers go through `Program::derives_of`, which logs the
    /// read for the derive edit path (check/derive_delta.rs).
    derives: Derives,
    pub span: Span,
    /// The `repr` of the type, in layout terms.
    pub repr: crate::ty::layout::Repr,
    /// `#[non_exhaustive]` on the type: outside the defining crate a
    /// struct cannot be built with a struct literal, its tuple and
    /// unit constructors are private, a named-field pattern needs
    /// `..`, and an enum match needs a wildcard arm. Inside the
    /// defining crate it changes nothing.
    pub non_exhaustive: bool,
}

impl AdtDef {
    /// An Adt with its derive list (the field is private, see `derives`).
    pub fn new(
        name: Sym,
        is_enum: bool,
        variants: Vec<Variant>,
        derives: Derives,
        span: Span,
        repr: crate::ty::layout::Repr,
        non_exhaustive: bool,
    ) -> AdtDef {
        AdtDef { name, is_enum, variants, derives, span, repr, non_exhaustive }
    }

    /// The derive list with no read noted: for fingerprints, which
    /// hash it rather than answer from it (driver/graph.rs).
    pub(crate) fn derives_unlogged(&self) -> Derives {
        self.derives
    }

    /// Sets the derive list (a derive delta on a kept program,
    /// check/derive_delta.rs).
    pub(crate) fn set_derives(&mut self, d: Derives) {
        self.derives = d;
    }
}

/// One evaluated `const` item.
pub struct ConstItem {
    pub name: Sym,
    pub ty: Ty,
    pub value: crate::ty::consteval::ConstValue,
    /// rustc's promotion qualifs of the value (`ty::constqual`): `&C`
    /// is promoted when neither holds.
    pub qualif: crate::ty::constqual::Qualif,
    /// The value's fingerprint, made once on first use
    /// (driver/graph.rs `value_fp`).
    pub value_fp: std::sync::OnceLock<u64>,
}

/// What one item inside a function body resolved to, by `Ast::stmt_items`
/// index. Nested statics live in the main statics table; nested consts get
/// their own values table because their names are block-scoped.
#[derive(Clone, Copy, Debug)]
pub enum BodyItem {
    Fn(LocalDefId),
    /// Index into `Program::body_consts`.
    Const(u32),
    /// Index into `Program::statics`.
    Static(u32),
}

/// What one type item inside a function body resolved to, by
/// `Ast::stmt_items` index (R18). Body checking turns the items of
/// its enclosing blocks into the resolve scope's local tables.
#[derive(Clone, Copy, Debug)]
pub enum BodyTypeItem {
    Adt(AdtId),
    /// A non-generic alias, fully resolved.
    Alias(Ty),
    Trait(DefId),
}

/// The value name an item inside a function body binds, if any.
pub(crate) fn stmt_item_name(ast: &Ast, idx: u32) -> Option<Sym> {
    match &ast.stmt_items[idx as usize].kind {
        ItemKind::Fn(d) => Some(d.name),
        ItemKind::Const(c) => c.name,
        ItemKind::Static(s) => Some(s.name),
        _ => None,
    }
}

/// The innermost body item named `name` visible at the use site
/// `at`: the item of that name in the smallest block whose span
/// holds the use. Blocks nest by span, as for fn siblings; a use
/// in macro-expanded code (a non-root context) may also see a
/// root block it expanded in, so only a root use requires a root
/// block. None when no enclosing block holds such an item: the
/// name then names a module item, or nothing.
pub(crate) fn stmt_item_in_blocks(ast: &Ast, blocks: &[(Span, Vec<u32>)], name: Sym, at: Span) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for (span, idxs) in blocks {
        if span.file != at.file || span.lo > at.lo || at.hi > span.hi {
            continue;
        }
        if at.ctxt == CtxtId::ROOT && span.ctxt != CtxtId::ROOT {
            continue;
        }
        for &idx in idxs {
            if stmt_item_name(ast, idx) == Some(name) {
                let len = span.hi - span.lo;
                if best.is_none_or(|(l, _)| len < l) {
                    best = Some((len, idx));
                }
            }
        }
    }
    best.map(|(_, idx)| idx)
}

/// `stmt_item_in_blocks` over every item block of the crate, for
/// callers that run before `Program::item_blocks` exists (the
/// scalar interpreter). The name check first keeps the common
/// no-body-item case off the block scan.
pub(crate) fn stmt_item_at(ast: &Ast, name: Sym, at: Span) -> Option<u32> {
    if !(0..ast.stmt_items.len() as u32).any(|i| stmt_item_name(ast, i) == Some(name)) {
        return None;
    }
    let blocks: Vec<(Span, Vec<u32>)> = ast
        .blocks
        .iter()
        .filter(|b| b.stmts.iter().any(|s| matches!(s, ast::Stmt::Item(_))))
        .map(|b| (b.span, b.stmts.iter().filter_map(|s| if let ast::Stmt::Item(i) = s { Some(*i) } else { None }).collect()))
        .collect();
    stmt_item_in_blocks(ast, &blocks, name, at)
}

/// One evaluated `static` item: its value becomes static data (ARCH 2.13).
pub struct StaticItem {
    pub name: Sym,
    pub ty: Ty,
    pub mutable: bool,
    pub value: crate::ty::consteval::ConstValue,
    /// Set for a `thread_local!` static: `ty` is `LocalKey<T>` and
    /// `value` the key's slot word, from 1 (check/tls.rs).
    pub tls: Option<super::tls::TlsKey>,
    /// As `ConstItem::value_fp`.
    pub value_fp: std::sync::OnceLock<u64>,
}

/// Where a fn's `FnDef` sits in the Ast. `Program` names defs by
/// location, not by reference, so it can outlive one request: the
/// daemon keeps it while the body patch edits a def in place (W-INCR3).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FnLoc {
    /// `ast.items[i]`, a fn item.
    Item(u32),
    /// Method `k` of the impl at `ast.items[i]`.
    Impl(u32, u32),
    /// Method `k` of the trait at `ast.items[i]`.
    Trait(u32, u32),
    /// `ast.stmt_items[i]`, a fn nested in a body.
    Nested(u32),
    /// The initializer body (`ast::ConstDef::init_fn`) of the const or
    /// static at `ast.items[i]`. This and the three below live only
    /// while check/const_body.rs checks the initializers.
    ConstItem(u32),
    /// Of associated const `k` of the impl at `ast.items[i]`.
    ImplConst(u32, u32),
    /// Of associated const `k` of the trait at `ast.items[i]`.
    TraitConst(u32, u32),
    /// Of the const or static at `ast.stmt_items[i]`.
    NestedConst(u32),
    /// Of the anonymous const `ast.anon_consts[i]` (an array length, a
    /// const generic argument or default, a discriminant).
    AnonConst(u32),
}

impl FnLoc {
    /// The def at this location.
    pub fn def(self, ast: &Ast) -> &ast::FnDef {
        let def = match self {
            FnLoc::Item(i) => match &ast.items[i as usize].kind {
                ItemKind::Fn(d) => Some(d),
                _ => None,
            },
            FnLoc::Impl(i, k) => match &ast.items[i as usize].kind {
                ItemKind::Impl(imp) => imp.fns.get(k as usize),
                _ => None,
            },
            FnLoc::Trait(i, k) => match &ast.items[i as usize].kind {
                ItemKind::Trait(t) => t.fns.get(k as usize),
                _ => None,
            },
            FnLoc::Nested(i) => match &ast.stmt_items[i as usize].kind {
                ItemKind::Fn(d) => Some(d),
                _ => None,
            },
            FnLoc::ConstItem(i) => item_init_fn(&ast.items[i as usize]),
            FnLoc::ImplConst(i, k) => match &ast.items[i as usize].kind {
                ItemKind::Impl(imp) => imp.consts.get(k as usize).and_then(|c| c.init_fn.as_deref()),
                _ => None,
            },
            FnLoc::TraitConst(i, k) => match &ast.items[i as usize].kind {
                ItemKind::Trait(t) => t.consts.get(k as usize).and_then(|c| c.init_fn.as_deref()),
                _ => None,
            },
            FnLoc::NestedConst(i) => item_init_fn(&ast.stmt_items[i as usize]),
            FnLoc::AnonConst(i) => ast.anon_consts.get(i as usize).map(|a| &a.def),
        };
        def.expect("a fn location names a fn of its Ast")
    }

    /// True for the initializer body of a const or static item.
    pub fn is_const_init(self) -> bool {
        matches!(self, FnLoc::ConstItem(_) | FnLoc::ImplConst(..) | FnLoc::TraitConst(..) | FnLoc::NestedConst(_) | FnLoc::AnonConst(_))
    }
}

/// The initializer body of a const or static item.
fn item_init_fn(item: &ast::Item) -> Option<&ast::FnDef> {
    match &item.kind {
        ItemKind::Const(c) => c.init_fn.as_deref(),
        ItemKind::Static(s) => s.init_fn.as_deref(),
        _ => None,
    }
}

#[derive(Debug)]
pub struct FnInfo {
    pub name: Sym,
    /// Where the def sits: `def` reads it.
    pub loc: FnLoc,
    pub owner: Option<AdtId>,
    pub self_kind: Option<SelfKind>,
    /// Parameter types, including `self` first when present.
    pub params: Vec<Ty>,
    pub ret: Ty,
    /// For a function that returns a reference: the index of the parameter it borrows from.
    pub ret_borrows_from: Option<usize>,
    /// True for a `const fn`, callable from const contexts (ARCH 2.13).
    pub is_const: bool,
}

impl FnInfo {
    /// The def this fn's location names in `ast` (the program's Ast).
    pub fn def<'x>(&self, ast: &'x Ast) -> &'x ast::FnDef {
        self.loc.def(ast)
    }
}

/// The old name for the session-local function index. New code uses
/// `LocalDefId` (src/def.rs); the alias stays while S3b edits this file in
/// parallel, then goes.
#[allow(dead_code)]
pub type FnId = LocalDefId;

/// The collected program. It owns its inputs (shared with the driver
/// session), so the daemon can keep it across requests (W-INCR3).
pub struct Program {
    pub ast: Arc<Ast>,
    pub interner: Arc<Interner>,
    pub known: Arc<Known>,
    pub tys: GlobalTys,
    /// The definition table (ARCH 2.2): kinds, parents, generics, and param envs by
    /// `DefId`, plus the bridge to the session-local `AdtId` compat tables below.
    pub defs: Defs,
    pub adts: Vec<AdtDef>,
    pub fns: Vec<FnInfo>,
    /// Evaluated const items, and name to index (ARCH 2.13).
    pub consts: Vec<ConstItem>,
    pub const_names: HashMap<Sym, u32>,
    /// Each const item's stable identity, in `consts` order.
    pub const_defs: Vec<DefId>,
    /// Each static item's stable identity, in `statics` order.
    pub static_defs: Vec<DefId>,
    /// Associated consts of inherent impls: (Adt, name) to the
    /// indexes into `consts` of every impl that defines the name
    /// (disjoint Self patterns may share one). They are const
    /// items like free consts, but not in `const_names`: a plain
    /// name must never find them.
    pub assoc_const_names: HashMap<(AdtId, Sym), Vec<u32>>,
    /// Associated const values of trait impls: (impl def, assoc
    /// def) to an index into `consts`. Filled with the other const
    /// tables in `eval_const_items`, before template checks run.
    pub impl_assoc_consts: HashMap<(DefId, DefId), u32>,
    /// Trait associated const default values: assoc def to an
    /// index into `consts`.
    pub trait_assoc_const_values: HashMap<DefId, u32>,
    /// Evaluated static items, and name to index. Codegen emits these as
    /// static data in the glue unit.
    pub statics: Vec<StaticItem>,
    pub static_names: HashMap<Sym, u32>,
    /// Items inside function bodies, by `Ast::stmt_items` index.
    pub body_items: HashMap<u32, BodyItem>,
    /// Type items inside function bodies, by `Ast::stmt_items` index.
    pub body_type_items: HashMap<u32, BodyTypeItem>,
    /// Every block that holds items: its span and its stmt item
    /// indexes. A name use inside a block's span sees that block's
    /// items (`stmt_item_in_blocks`); const contexts resolve their
    /// plain names through this, as fn siblings do.
    pub item_blocks: Vec<(Span, Vec<u32>)>,
    /// Evaluated consts declared inside function bodies.
    pub body_consts: Vec<ConstItem>,
    /// For a nested fn, the stmt item indexes of its containing block:
    /// the sibling items its body may name.
    pub fn_siblings: HashMap<LocalDefId, Vec<u32>>,
    /// The reads of each const and static initializer body's last check
    /// (check/const_body.rs), by its location: a kept program checks
    /// again only the bodies whose reads moved (`check_dirty`).
    pub const_body_reads: Vec<(FnLoc, crate::driver::graph::Reads)>,
    /// The AST counts at the last walk over every initializer site
    /// (`const_body::Shape`): `check_dirty` may skip the walk while they
    /// hold and no edited header has a signature anon const.
    pub const_sites_shape: super::const_body::Shape,
    pub type_names: crate::driver::read_cover::NameMap<AdtId>,
    /// Adts with a written `Drop` impl, by written name. Const items
    /// evaluate before the impl index and `gx.drop_methods` exist;
    /// the const evaluator's drop check (`needs_drop_const`) reads
    /// this set instead. Filled in `collect` before const items run.
    pub drop_adts: std::collections::HashSet<AdtId>,
    /// Non-generic type aliases by (flat) name, fully resolved.
    pub aliases: crate::driver::read_cover::NameMap<Ty>,
    pub fn_names: HashMap<Sym, LocalDefId>,
    /// Inherent methods by (template Adt, name), in registration
    /// order. Several inherent impls of one Adt may define the
    /// same name under disjoint Self patterns (`impl W<u8>` and
    /// `impl W<u16>`); the probe unifies the receiver against
    /// each candidate, and the overlap check in `finish_collect`
    /// rejects patterns that can both apply.
    pub methods: crate::check::spec::Logged<(AdtId, Sym), Vec<LocalDefId>, { crate::check::spec::tab::METHODS }>,
    pub main: Option<LocalDefId>,
    /// The crate this program was collected as: session-local id and stable hash.
    pub krate: CrateId,
    pub stable_crate: StableCrateHash,
    /// Each function's stable identity, in `fns` order (the `Vec<DefId>` side
    /// table of docs/ARCH.md section 1.7). User Adts carry their `DefId` in
    /// `defs` and in the type store, where `TyKind::Adt(DefId, ArgsId)` reads it.
    pub fn_defs: Vec<DefId>,
    /// The `LocalDefId` of a function by its stable hash (section 1.7).
    pub def_by_hash: crate::check::spec::Logged<DefPathHash, LocalDefId, { crate::check::spec::tab::DEF_BY_HASH }>,
    /// Source std fns routed into this compile by the std loader (S5):
    /// full Rust path segments to the local fn that carries the body.
    /// Empty unless the driver spliced std source in. Filled by the
    /// driver after collection, from the loader's decl table.
    pub std_fns: Vec<(Vec<Sym>, LocalDefId)>,
    /// Byte ranges `(lo, hi)` of the root file that belong to a shipped
    /// library crate, with the crate. Spliced std source sits after the
    /// user file, so user spans never fall in a range: this is the
    /// provenance intrinsic privilege keys on (section 3.1 item 1).
    /// Empty when no std source was spliced. Filled by the driver.
    pub shipped_ranges: Vec<(u32, u32, ShippedCrate)>,
    /// The module binding tables of section 1.3, when the crate went
    /// through the resolver. Attached by the driver at collection
    /// (`collect_with_order_resolved`), before any body is checked:
    /// template and instance checks read their defining module and
    /// the visibility tables from it. The S7 `Binding` query fns in
    /// driver/graph.rs read through it.
    pub resolved: Option<std::sync::Arc<crate::resolve::imports::ResolvedTables>>,
    /// The generic tables (traits, templates, instances) built by
    /// `check::generics` during collection and finished by mono.
    pub gx: GenTables,
    /// False when no later request replays body read sets (a one-shot
    /// process): body checks then use a sink that records nothing. Set by
    /// the driver after collection.
    pub record_reads: bool,
    /// The panic strategy is unwind (`-C panic=unwind`, or a Cargo
    /// profile that unwinds): lowering records cleanup entries for
    /// statements and codegen propagates panics (docs/ARCH.md 7.2).
    /// Set by the driver after collection from the session's profile.
    pub unwind: bool,
    /// Integer overflow panics (`-C overflow-checks`, or the Cargo profile's
    /// `overflow-checks`): `+ - *`, negation, shifts, `abs` and `pow` check as
    /// in a debug build. Off, they wrap and shift amounts are masked, as in a
    /// release build. Set by the driver after collection from the session's profile.
    pub overflow_checks: bool,
    /// Debug assertions (`-C debug-assertions`, or the Cargo profile's `debug-assertions`):
    /// on, release C keeps the abort of an exhaustive match's otherwise edge
    /// (`cgen::assume_unreachable`). Set by the driver after collection from the session's profile.
    pub debug_assertions: bool,
    /// panic=unwind with unwind tables (`lust_unwind_mode` 2, runtime/panic.c): codegen gives
    /// each frame with drops one cleanup variable and stores its ladder entry before calls
    /// that can unwind, instead of a flag check after each call. Needs a C compiler with
    /// -fexceptions (clang, gcc, zig cc) and its C++ driver for the throw and the catch
    /// (runtime/unwind.cc), never tcc. Set by the driver.
    pub unwind_tables: bool,
    /// Region signatures for the borrow checker (W-BORROW, ARCH 5): the
    /// signatures with their lifetimes kept, beside the erased ones above.
    pub regions: crate::borrowck::RegionTables,
    /// Every const and static value `eval_const_items` evaluated, in its
    /// pending order (None: failed), and the body item scopes it read: a
    /// kept program checks its values again with them (`const_values_hold`).
    pub const_values: Vec<Option<crate::ty::consteval::ConstValue>>,
    pub scope_of_stmt: HashMap<u32, Vec<u32>>,
    /// Deep Adt fingerprints, kept while the driver holds the program
    /// unchanged (driver/graph.rs `FpMemo`). Off after collect.
    pub fp_memo: crate::driver::graph::FpMemo,
    /// The trait goals of bounded const fn calls in const and static
    /// initializers (`prove_const_goals`), kept for a derive delta.
    pub const_goals: Vec<(TraitRef, Span)>,
    /// Each impl's generics and scope as its methods' signatures saw
    /// them, by `ast.items` index (`fn_record` rebuilds a method with it).
    pub(crate) impl_parents: HashMap<u32, ParentGen>,
    /// Every stored template check and discovery output up to this
    /// `gen_reuse` stamp agrees with this program (`gen_reuse::stamp_now`
    /// when the driver keeps it, `patch::keep`); 0 before.
    pub(crate) reuse_stamp: u64,
    /// The driver's generation of this program (`compile`'s generic
    /// block): every passing slot read set marked with it holds against
    /// this program. 0 when no slot is (a fresh collect, or a kept
    /// program a delta moved before its read sets were walked).
    pub(crate) verified: u64,
    /// The fn bodies of `ast` by file, for `outer_param_names` (built on
    /// first use, with the address of the Ast it indexes: a new Ast, as a
    /// patch sets, builds it again).
    pub(crate) outer_params: std::sync::RwLock<Option<(usize, std::sync::Arc<OuterParams>)>>,
}

impl Program {
    pub fn name(&self, s: Sym) -> &str {
        self.interner.get(s)
    }
    /// The source (short) name of a def name, for program output such
    /// as derived `Debug`: strips the resolver's flat-name module
    /// prefix (`m0$Stats` reads `Stats`, as rustc prints). Display
    /// only; lookups keep using `name`.
    pub fn display_name(&self, s: Sym) -> &str {
        crate::intern::source_name(self.interner.get(s))
    }
    /// The source path of a def name, for diagnostics (`m0::Stats`,
    /// `base::inner::Deep` for a source dependency). Reads the
    /// resolver's display table when it is attached; before that (in
    /// collect itself) it falls back to the path carried in the flat
    /// name, which lacks only a dependency's crate name.
    pub fn display_path(&self, s: Sym) -> String {
        if let Some(tables) = &self.resolved
            && let Some(&d) = tables.display_names.get(&s)
        {
            return self.interner.get(d).to_string();
        }
        crate::intern::source_path(self.interner.get(s))
    }
    pub fn adt(&self, id: AdtId) -> &AdtDef {
        &self.adts[id.0 as usize]
    }
    /// The derive list of Adt `id` (an instance's is its template's).
    /// Every reader of a derive list comes here: inside a body check it
    /// notes a `Derives` read of the Adt (check/derive_delta.rs), so a
    /// derive edit rechecks the bodies whose answers it can move.
    pub fn derives_of(&self, id: AdtId) -> Derives {
        crate::check::derive_delta::note(self, id, crate::check::derive_delta::bit::ALL);
        self.adts[id.0 as usize].derives
    }
    /// Whether Adt `id` derives the trait of `bit` (check/derive_delta.rs
    /// `bit`): as `derives_of`, but a body check notes that trait only, so a
    /// toggle of another trait leaves the body alone.
    pub fn derive_has(&self, id: AdtId, bit: u16) -> bool {
        crate::check::derive_delta::note(self, id, bit);
        crate::check::derive_delta::masked(&self.adts[id.0 as usize].derives, bit) != 0
    }
    pub fn fn_info(&self, id: LocalDefId) -> &FnInfo {
        &self.fns[id.0 as usize]
    }
    /// The Ast def of fn `id` (an instance's or closure's is its source fn's).
    pub fn fn_def(&self, id: LocalDefId) -> &ast::FnDef {
        self.fns[id.0 as usize].loc.def(&self.ast)
    }
    /// The return type a caller (or the fn's own body) checks against:
    /// `FnInfo::ret`, except that a plain fn whose `impl Trait` return
    /// was revealed for codegen keeps its opaque type here.
    pub fn ret_view(&self, id: LocalDefId) -> Ty {
        self.gx.opaque_ret_of.get(&id).copied().unwrap_or(self.fns[id.0 as usize].ret)
    }
    /// A function's stable identity in this crate.
    #[allow(dead_code)] // First production readers are S3a defs and mono instance keys; tests use it now.
    pub fn def_id(&self, id: LocalDefId) -> DefId {
        self.fn_defs[id.0 as usize]
    }
    /// A function's disk identity: the only form that may enter disk or slot
    /// keys (docs/ARCH.md sections 1.7 and 8.4).
    #[allow(dead_code)] // See `def_id`.
    pub fn disk_id(&self, id: LocalDefId) -> DiskDefId {
        DiskDefId { krate: self.stable_crate, hash: self.def_id(id).hash }
    }
    /// The session-local id of the function with this stable hash, if this
    /// crate defines it.
    #[allow(dead_code)] // See `def_id`.
    pub fn local_def(&self, hash: DefPathHash) -> Option<LocalDefId> {
        self.def_by_hash.get(&hash).copied()
    }
    /// A user Adt's stable identity, as carried by `TyKind::Adt(DefId, ArgsId)`.
    #[allow(dead_code)] // First production readers are generic collect and mono; the store form is used now.
    pub fn def_of_adt(&self, id: AdtId) -> DefId {
        self.defs.def_of_adt(id).expect("ADT defs are recorded during collection")
    }
    /// The session-local index of the user Adt with this def, if this crate defines it.
    #[allow(dead_code)] // See `def_of_adt`.
    pub fn adt_by_def(&self, def: DefId) -> Option<AdtId> {
        self.defs.local_adt(def)
    }
    /// True when an `FnDef` over `def` names a fn with a body (a fn
    /// item). The other `FnDef`s are constructor items over an Adt
    /// def: a user tuple struct or variant (`Ctx::ctor_value_ty`), or
    /// a builtin item over a builtin Adt def (`Some`, `Ok`, `Err`,
    /// `Box::new`, `Rc::new`, `String::from`; `Ctx::builtin_item_value`).
    pub fn is_fn_item_def(&self, def: DefId) -> bool {
        self.tys.builtins().kind_of(def).is_none() && self.adt_by_def(def).is_none()
    }
    /// The index of the const item with this def, if this crate defines it.
    pub fn const_by_def(&self, def: DefId) -> Option<u32> {
        self.const_defs.iter().position(|&d| d == def).map(|i| i as u32)
    }
    /// The indexes into `consts` of the inherent associated
    /// consts with this name: one per impl that defines it.
    /// Materialized Adt instances answer through their template:
    /// the value is evaluated once per impl (an initializer that
    /// names the impl's own parameters fails evaluation and never
    /// reaches the table, so one value serves every instance).
    /// The caller picks by the written Self type.
    pub fn inherent_assoc_consts(&self, adt: AdtId, name: Sym) -> &[u32] {
        if let Some(v) = self.assoc_const_names.get(&(adt, name)) {
            return v;
        }
        let Some(def) = self.defs.def_of_adt(adt) else { return &[] };
        let Some(template) = self.defs.local_adt(def) else { return &[] };
        self.assoc_const_names.get(&(template, name)).map_or(&[], |v| v.as_slice())
    }
    /// The index of the static item with this def, if this crate defines it.
    pub fn static_by_def(&self, def: DefId) -> Option<u32> {
        self.static_defs.iter().position(|&d| d == def).map(|i| i as u32)
    }

    /// False for a hole of the statics table: the place of a static that
    /// went (`stable_ranks`), an inert placeholder row with no C.
    pub fn static_live(&self, i: usize) -> bool {
        self.static_defs.get(i).is_some_and(|d| d.hash.0 != 0)
    }
    /// The shipped crate a span's source belongs to, if any. Only spans
    /// in spliced shipped source, or in a file of a shipped adapter
    /// crate (ARCH 3.6), answer; every user span gets None.
    pub fn shipped_crate_at(&self, span: Span) -> Option<ShippedCrate> {
        if let Some(a) = self.adapter_at(span) {
            return ShippedCrate::ALL.iter().copied().find(|k| k.name() == a.name);
        }
        if span.file != crate::diag::FileId(0) {
            return None;
        }
        let ranges = match &self.resolved {
            Some(t) if self.shipped_ranges.is_empty() => &t.std_ranges,
            _ => &self.shipped_ranges,
        };
        ranges.iter().find(|(lo, hi, _)| span.lo >= *lo && span.lo < *hi).map(|(_, _, k)| *k)
    }
    /// True when `span` is in the std splice of the selected crate's
    /// root file: shipped std source, not user source (W-STD-C).
    pub fn std_source_span(&self, span: Span) -> bool {
        span.file == crate::diag::FileId(0) && self.resolved.as_ref().and_then(|t| t.std_splice_lo).is_some_and(|lo| span.lo >= lo)
    }
    /// True when Adt `a` is defined in spliced std source (a std source
    /// module type such as `Duration`): a std type, foreign to the user
    /// crate for coherence and with no user-visible layout.
    pub fn is_std_adt(&self, a: AdtId) -> bool {
        self.adts.get(a.0 as usize).is_some_and(|adt| self.std_source_span(adt.span))
    }
    /// True for BTree `range`/`range_mut` iterators: std gives them the
    /// default `size_hint` of `(0, None)`, while their inner slice
    /// iterator is exact. Paths that would hint them as the inner
    /// iterator leave them out.
    pub fn is_btree_range_iter(&self, tys: &mut impl TyIntern, t: Ty) -> bool {
        tys.local_adt(t).is_some_and(|a| {
            self.is_std_adt(a)
                && matches!(
                    self.display_path(self.adt(a).name).as_str(),
                    "alloc::collections::btree_map::Range" | "alloc::collections::btree_map::RangeMut" | "alloc::collections::btree_set::Range"
                )
        })
    }
    /// The inner iterator type when `t` is one of the std source
    /// collection iterator structs (`alloc::collections::{btree_map,
    /// btree_set, vec_deque}` iterators, W-STD-C): each is a nominal
    /// struct whose one field holds a builtin iterator, so check
    /// types its adapter and consumer methods as the inner type's
    /// and lowering unwraps field 0 (check/iter.rs, lower/iter.rs).
    /// The types stay distinct from their inner types, as in Rust.
    pub fn collection_iter_inner(&self, tys: &mut impl TyIntern, t: Ty) -> Option<Ty> {
        let a = tys.local_adt(t)?;
        if !self.is_std_adt(a) {
            return None;
        }
        let path = self.display_path(self.adt(a).name);
        const WRAPPERS: &[&str] = &[
            "alloc::collections::btree_map::Iter",
            "alloc::collections::btree_map::IterMut",
            "alloc::collections::btree_map::Keys",
            "alloc::collections::btree_map::Values",
            "alloc::collections::btree_map::ValuesMut",
            "alloc::collections::btree_map::IntoIter",
            "alloc::collections::btree_map::Range",
            "alloc::collections::btree_map::RangeMut",
            "alloc::collections::btree_map::IntoKeys",
            "alloc::collections::btree_map::IntoValues",
            "alloc::collections::btree_set::Iter",
            "alloc::collections::btree_set::IntoIter",
            "alloc::collections::btree_set::Range",
            "alloc::collections::vec_deque::Iter",
            "alloc::collections::vec_deque::IterMut",
            "alloc::collections::vec_deque::IntoIter",
        ];
        if !WRAPPERS.contains(&path.as_str()) {
            return None;
        }
        let field_ty = self.adt(a).variants.first()?.fields.first()?.ty;
        // A materialized generic instance already carries substituted
        // field types; a template application still names the struct's
        // parameters, so instantiate the field at the wrapper type's
        // own value arguments (as a struct literal does for a field).
        let is_instance = self.gx.adt_instance_of.get(a.0 as usize).is_some_and(|o| o.is_some());
        if is_instance {
            return Some(field_ty);
        }
        let (_, args) = tys.adt_parts(t)?;
        let value_args: Vec<crate::ty::GenericArg> =
            tys.args(args).iter().copied().filter(|g| !matches!(g, crate::ty::GenericArg::Lifetime(_))).collect();
        Some(crate::ty::subst(tys, field_ty, &value_args))
    }
    /// True when Adt `a` is a known global allocator type: std's
    /// `System` or the `tikv-jemallocator` adapter's `Jemalloc`.
    /// A `#[global_allocator]` static of one of these types is
    /// accepted and the runtime keeps its allocator; any other
    /// type would need a user `GlobalAlloc` impl, which takes
    /// `unsafe` code lust does not support.
    pub fn is_known_allocator_adt(&self, a: AdtId) -> bool {
        let Some(adt) = self.adts.get(a.0 as usize) else { return false };
        let path = self.display_path(adt.name);
        (self.is_std_adt(a) && path == "std::alloc::System")
            || (path.rsplit("::").next() == Some("Jemalloc") && self.adapter_at(adt.span).is_some_and(|x| x.name == "tikv_jemallocator"))
    }
    /// True when Adt `a` is std's `core::marker::PhantomData` (std
    /// source, W-STD-C): zero-sized, yet it acts as if it held its `T`
    /// for unused-parameter checks and auto traits.
    pub fn is_phantom_adt(&self, a: AdtId) -> bool {
        self.is_std_adt(a) && self.display_path(self.adt(a).name) == "core::marker::PhantomData"
    }
    /// True when Adt `a` (a template or an instance) is std's
    /// `core::mem::ManuallyDrop` (std source, R2): it has no drop glue,
    /// so the value inside is never dropped (`needs_drop_sub`,
    /// cgen's `needs_drop_c`), and layout queries see it.
    pub fn is_manually_drop_adt(&self, a: AdtId) -> bool {
        self.is_std_adt(a) && self.display_path(self.adt(a).name) == "core::mem::ManuallyDrop"
    }
    /// True when Adt `a` is std's `core::cmp::Reverse` (std source,
    /// library/core/src/cmp.rs): the compare glue orders its `T` with
    /// the operands swapped, as each of its written methods does.
    pub fn is_reverse_adt(&self, a: AdtId) -> bool {
        self.is_std_adt(a) && self.display_path(self.adt(a).name) == "core::cmp::Reverse"
    }
    /// True when Adt `a` is std's `core::ops::ControlFlow` (std source,
    /// W-STD-C): `?` on it returns its `Break` from the enclosing fn
    /// (`try_expr` in check, `try_into` in lower).
    pub fn is_control_flow_adt(&self, a: AdtId) -> bool {
        self.is_std_adt(a) && self.display_path(self.adt(a).name) == "core::ops::ControlFlow"
    }
    /// True when Adt `a` is std's `VecDeque` (std source,
    /// W-STD-C): `collect` into it goes through the `Vec` collect
    /// machinery, whose capacity rules are std's for a deque too
    /// (check/iter.rs, lower/iter.rs).
    pub fn is_vec_deque_adt(&self, a: AdtId) -> bool {
        self.is_std_adt(a) && self.display_path(self.adt(a).name).rsplit("::").next() == Some("VecDeque")
    }
    /// The adapter crate whose source file holds `span`, if any. The
    /// resolver records each adapter crate's files (W-DEPS).
    pub fn adapter_at(&self, span: Span) -> Option<&crate::resolve::imports::AdapterCrate> {
        self.resolved.as_ref()?.adapters.iter().find(|a| a.files.contains(&span.file))
    }
    /// The shipped crate that defines function `f`, if it is a shipped
    /// (spliced std) fn. Codegen runs its intrinsic privilege check
    /// against this, as the second line behind check (section 3.2).
    pub fn fn_shipped_crate(&self, f: LocalDefId) -> Option<ShippedCrate> {
        self.shipped_crate_at(self.fn_def(f).name_span)
    }
    /// True when `lust check` neither checks nor walks fn `f`: a fn in
    /// shipped core, alloc, or std source (the std splice and the std
    /// platform crate; their instances and closures included). Those
    /// bodies see no user name, so their answer is the same in every
    /// program, and every `lust build` still checks them. `lust check`
    /// already walks no instances (ARCH 4.1); this extends that to std's
    /// own plain bodies, its templates, and the discovery pass over them.
    pub fn check_skips(&self, f: LocalDefId) -> bool {
        !self.gx.walk_instances
            && matches!(
                self.shipped_crate_at(self.fn_def(f).name_span),
                Some(ShippedCrate::Core | ShippedCrate::Alloc | ShippedCrate::Std)
            )
    }
    /// True for a span in the std methods of the lang enums
    /// (`Ast::lang_methods`): std source with its own scope, where no
    /// user name is visible.
    pub fn in_lang_methods(&self, span: Span) -> bool {
        let (lo, hi) = self.ast.lang_methods;
        span.file == crate::diag::FileId(0) && span.lo >= lo && span.lo < hi
    }
    /// True for a std method of a lang enum (`Option::map`), its
    /// instances included; closure bodies inside one are not.
    pub fn is_lang_method(&self, f: LocalDefId) -> bool {
        !self.gx.closure_fns.contains_key(&f) && self.in_lang_methods(self.fn_def(f).name_span)
    }
    /// Marks the program generic, from an item at `span`: code other
    /// than the lang enum methods also sets `user_generics` (see
    /// `GenTables::user_generics`).
    pub fn mark_generic(&mut self, span: Span) {
        self.gx.has_generics = true;
        if !self.in_lang_methods(span) {
            self.gx.user_generics = true;
        }
    }
    /// The local fn a shipped std path (for example
    /// `std::process::exit`) was routed to in this compile, if any.
    pub fn std_fn_for_path(&self, path: &[Sym]) -> Option<LocalDefId> {
        self.std_fns.iter().find(|(p, _)| p.as_slice() == path).map(|(_, f)| *f)
    }
}

/// Human-readable type name, for error messages. Builtin Adts keep their std names
/// through the compat helpers; kinds the subset cannot form yet print their Rust
/// shape so later waves inherit a total printer.
pub fn ty_name(p: &Program, tys: &impl TyIntern, t: Ty) -> String {
    match tys.kind(t) {
        TyKind::Error => "{error}".into(),
        TyKind::Unit => "()".into(),
        TyKind::Bool => "bool".into(),
        TyKind::Char => "char".into(),
        TyKind::Str => "str".into(),
        TyKind::Never => "!".into(),
        TyKind::Int(i) => i.name().into(),
        TyKind::Float(f) => f.name().into(),
        TyKind::Ref(r, m, e) => {
            let inner = ty_name(p, tys, e);
            // `&(dyn A + Send)`: the parens are part of the Rust syntax.
            let inner = if matches!(tys.kind(e), TyKind::Dyn(..)) && inner.contains(" + ") { format!("({inner})") } else { inner };
            let lt = if r == Region::Static { "'static " } else { "" };
            format!("&{lt}{}{inner}", if m.is_mut() { "mut " } else { "" })
        }
        TyKind::RawPtr(m, e) => format!("*{}{}", if m.is_mut() { "mut " } else { "const " }, ty_name(p, tys, e)),
        TyKind::Slice(e) => format!("[{}]", ty_name(p, tys, e)),
        TyKind::Array(e, _) => format!("[{}; _]", ty_name(p, tys, e)),
        TyKind::Tuple(ts) => {
            let parts: Vec<String> = tys.tys_of(ts).iter().map(|&e| ty_name(p, tys, e)).collect();
            format!("({})", parts.join(", "))
        }
        TyKind::Adt(..) => match tys.builtin_of(t) {
            Some(BuiltinTy::String) => "String".into(),
            Some(BuiltinTy::Vec) => format!("Vec<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::Option) => {
                format!("Option<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e)))
            }
            Some(BuiltinTy::Result) => {
                let (a, b) = tys.result_args(t).unwrap_or((ERROR, ERROR));
                format!("Result<{}, {}>", ty_name(p, tys, a), ty_name(p, tys, b))
            }
            Some(BuiltinTy::Boxed) => format!("Box<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::Rc) => format!("Rc<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::Weak) => format!("std::rc::Weak<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::RefCell) => format!("RefCell<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::Cell) => format!("Cell<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::RefGuard) => format!("Ref<'_, {}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::RefMutGuard) => format!("RefMut<'_, {}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::Chars) => "std::str::Chars".into(),
            Some(BuiltinTy::Fmt(k)) => format!("std::fmt::{}", k.name()),
            Some(BuiltinTy::ExitCode) => "ExitCode".into(),
            Some(BuiltinTy::Sync(k)) => match tys.adt_first_ty_arg(t) {
                Some(e) => format!("{}<{}>", k.display(), ty_name(p, tys, e)),
                None => k.display(),
            },
            Some(BuiltinTy::LocalKey) => format!("LocalKey<{}>", tys.adt_first_ty_arg(t).map_or("_".into(), |e| ty_name(p, tys, e))),
            Some(BuiltinTy::AccessError) => "AccessError".into(),
            Some(BuiltinTy::Hash(k)) => {
                let a = tys.adt_parts(t).map(|(_, a)| tys.args(a).to_vec()).unwrap_or_default();
                let shown: Vec<String> = a.iter().map(|g| g.as_ty().map_or("_".into(), |e| ty_name(p, tys, e))).collect();
                if shown.is_empty() { k.path().1.to_string() } else { format!("{}<{}>", k.path().1, shown.join(", ")) }
            }
            // The `next` wrapper of another iterator prints as that iterator.
            Some(BuiltinTy::Iter(crate::ty::iter::IterKind::ByNext)) => {
                let a = tys.adt_parts(t).map(|(_, a)| tys.args(a).to_vec()).unwrap_or_default();
                a.first().and_then(|g| g.as_ty()).map_or("_".into(), |e| ty_name(p, tys, e))
            }
            // A str split prints as its std struct (W-STD-B).
            Some(BuiltinTy::Iter(crate::ty::iter::IterKind::StrSplit)) => {
                use crate::ty::iter::SplitMode as M;
                let a: Vec<Ty> = tys.adt_parts(t).map(|(_, a)| tys.args(a).iter().filter_map(|g| g.as_ty()).collect()).unwrap_or_default();
                let mode = a.get(1).and_then(|&m| crate::ty::iter::split_mode(tys, m));
                let name = match mode {
                    Some(M::RSplit) => "RSplit",
                    Some(M::SplitTerminator) => "SplitTerminator",
                    Some(M::RSplitTerminator) => "RSplitTerminator",
                    Some(M::SplitN) => "SplitN",
                    Some(M::RSplitN) => "RSplitN",
                    Some(M::SplitInclusive) => "SplitInclusive",
                    Some(M::SplitWhitespace) => return "std::str::SplitWhitespace".into(),
                    Some(M::SplitAsciiWhitespace) => return "std::str::SplitAsciiWhitespace".into(),
                    Some(M::Lines) => return "std::str::Lines".into(),
                    _ => "Split",
                };
                let p_name = a.first().map_or("_".into(), |&e| ty_name(p, tys, e));
                format!("std::str::{name}<{p_name}>")
            }
            Some(BuiltinTy::Iter(k)) => {
                let a = tys.adt_parts(t).map(|(_, a)| tys.args(a).to_vec()).unwrap_or_default();
                // `array::IntoIter<T, N>` (fx-array-iter) prints its
                // length, which lust stores in the array type arg.
                if k == crate::ty::iter::IterKind::ArrayIntoIter
                    && let [t0, arr] = a.as_slice()
                    && let Some(arr) = arr.as_ty()
                    && let TyKind::Array(_, c) = tys.kind(arr)
                    && let ConstKind::Int(n) = tys.const_kind(c)
                {
                    let e = t0.as_ty().map_or("_".into(), |e| ty_name(p, tys, e));
                    return format!("{}<{e}, {n}>", k.display());
                }
                let shown: Vec<String> = a.iter().take(k.shown_args()).map(|g| g.as_ty().map_or("_".into(), |e| ty_name(p, tys, e))).collect();
                if shown.is_empty() { k.display().to_string() } else { format!("{}<{}>", k.display(), shown.join(", ")) }
            }
            None => match tys.local_adt(t) {
                Some(a) => p.display_path(p.adt(a).name),
                None => "{unknown}".into(),
            },
        },
        TyKind::FnPtr(_) | TyKind::FnDef(..) => "fn(..)".into(),
        TyKind::Closure(..) => "{closure}".into(),
        TyKind::Dyn(b, r) => {
            let b = tys.dyn_bounds(b);
            let mut parts: Vec<String> = Vec::new();
            if let Some(pr) = b.principal {
                let name = p.gx.index.trait_record(pr.value.def).map_or("?", |rec| p.name(rec.name));
                let args: Vec<String> = tys.args(pr.value.args)[1..].iter().filter_map(|g| g.as_ty()).map(|e| ty_name(p, tys, e)).collect();
                // The Fn family prints in its sugar: `Fn(A, B) -> R`.
                let fn_sugar = p.gx.lang.is_fn_family(pr.value.def).then(|| {
                    let ins = tys.args(pr.value.args).get(1).and_then(|g| g.as_ty()).map_or_else(Vec::new, |t| match tys.kind(t) {
                        TyKind::Tuple(ts) => tys.tys_of(ts).iter().map(|&e| ty_name(p, tys, e)).collect(),
                        _ => Vec::new(),
                    });
                    let out = tys.bindings(pr.value.bindings).first().and_then(|(_, g)| g.as_ty()).filter(|&t| t != UNIT);
                    let out = out.map_or_else(String::new, |t| format!(" -> {}", ty_name(p, tys, t)));
                    format!("{name}({}){out}", ins.join(", "))
                });
                parts.push(match fn_sugar {
                    Some(s) => s,
                    None if args.is_empty() => name.to_string(),
                    None => format!("{name}<{}>", args.join(", ")),
                });
            }
            for (on, name) in [(b.auto.send, "Send"), (b.auto.sync, "Sync"), (b.auto.unpin, "Unpin")] {
                if on {
                    parts.push(name.to_string());
                }
            }
            if r == Region::Static {
                parts.push("'static".to_string());
            }
            format!("dyn {}", parts.join(" + "))
        }
        TyKind::Opaque(..) => "impl ..".into(),
        TyKind::Param(param) => format!("T{}", param.0),
        TyKind::Projection(..) => "{projection}".into(),
        TyKind::Infer(v) if v.is_int() => "{integer}".into(),
        TyKind::Infer(v) if v.is_float() => "{float}".into(),
        TyKind::Infer(_) => "_".into(),
    }
}

/// The impl's regions (W-BORROW): its regioned Self type and associated
/// types, for the region signatures of its methods. An elided lifetime in
/// the header (`impl Tr for &str`, `impl S<'_>`) is an anonymous lifetime
/// param of the impl; a lifetime hidden in a path (`impl S` for
/// `struct S<'a>`) is rustc E0726. A named impl lifetime that only an
/// associated type uses is unconstrained (rustc E0207).
fn impl_regions(p: &mut Program, imp: &ast::ImplDef, impl_def: DefId, scope: &mut generics::Scope, errs: &mut Vec<Diag>) {
    let (ast, interner, known) = (&p.ast, &p.interner, &p.known);
    // Std source names no user type (`resolve_cx_in`), so the hidden
    // lifetime test (which reads the user type table) does not apply.
    let std = p.in_lang_methods(imp.span);
    // The impl's own type params shadow user types of the same name
    // (`impl<R: Read> Read for &mut R` with a `struct R<'a>` around).
    let params: Vec<Sym> = imp.generics.params.iter().filter_map(|g| if let ast::GenericParam::Type { name, .. } = g { Some(*name) } else { None }).collect();
    if !std && p.shipped_crate_at(imp.span).is_none() && let Some(span) = hidden_lifetime(p, imp.self_ty, &params) {
        errs.push(Diag::new(
            "L0202",
            span,
            "implicit elided lifetime not allowed here: this type has lifetime parameters",
            "Write them, for example `impl S<'_>` or `impl<'a> S<'a>`.",
        ));
        return;
    }
    let named = scope.lifetimes.len() as u32;
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(named));
    let mut ks = scope.clone();
    ks.regions = generics::RegionMode::Keep { next: Some(counter.clone()), elided: None };
    let self_r = {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        match generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, imp.self_ty, &ks) {
            Ok(t) => t,
            Err(_) => return,
        }
    };
    // The trait ref's lifetime args, for a trait with lifetime params
    // (W-BORROW): calls through the impl and trait selection replay them.
    let mut trait_lts: Vec<Region> = Vec::new();
    if let Some(tref) = &imp.trait_ref
        && let Some(seg) = tref.path.last()
        && let ast::GenericArgs::Angle(args) = &seg.args
    {
        for g in args {
            if let ast::GenericArg::Lifetime(lt) = g {
                match generics::scope_region(interner, &ks, Some(*lt), lt.span) {
                    Ok(r) => trait_lts.push(r),
                    Err(_) => return,
                }
            }
        }
    }
    if !trait_lts.is_empty() {
        let where_lts = {
            let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
            generics::trait_lts_of(ast, interner, known, &cx, &ks, &imp.generics, &imp.where_clause)
        };
        p.regions.impl_heads.insert(impl_def, crate::borrowck::ImplHead { self_ty: self_r, trait_lifetimes: trait_lts.clone(), where_lts });
    }
    // The header's anonymous lifetimes join the impl's params, so the
    // methods' own lifetimes number after them.
    let total = counter.load(std::sync::atomic::Ordering::Relaxed);
    for k in named..total {
        scope.lifetimes.push((interner.intern(&format!("'_impl{}", k - named)), k));
    }
    scope.self_ty_regioned = Some(self_r);
    ks.lifetimes = scope.lifetimes.clone();
    ks.regions = generics::RegionMode::Keep { next: None, elided: None };
    ks.self_ty_regioned = Some(self_r);
    let mut in_header: Vec<Region> = trait_lts;
    crate::borrowck::regions_of(&p.tys, self_r, &mut in_header);
    scope.header_regioned = vec![self_r];
    if let Some(tref) = &imp.trait_ref {
        for seg in &tref.path {
            if let ast::GenericArgs::Angle(args) = &seg.args {
                for g in args {
                    if let ast::GenericArg::Type(t) = g {
                        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                        if let Ok(r) = generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, *t, &ks) {
                            crate::borrowck::regions_of(&p.tys, r, &mut in_header);
                            scope.header_regioned.push(r);
                        }
                    }
                }
            }
        }
    }
    for at in &imp.types {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        let Some(aty) = at.default else { continue };
        let Ok(t) = generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, aty, &ks) else { continue };
        let mut used: Vec<Region> = Vec::new();
        crate::borrowck::regions_of(&p.tys, t, &mut used);
        // A dyn's object lifetime is not in the region form: read it as written.
        let mut used_dyn: Vec<Sym> = Vec::new();
        dyn_lifetimes(ast, aty, &mut used_dyn);
        for &(n, i) in scope.lifetimes.iter().take(named as usize) {
            let in_dyn_header = || {
                let mut hs: Vec<Sym> = Vec::new();
                dyn_lifetimes(ast, imp.self_ty, &mut hs);
                for seg in imp.trait_ref.iter().flat_map(|t| &t.path) {
                    if let ast::GenericArgs::Angle(args) = &seg.args {
                        for g in args {
                            if let ast::GenericArg::Type(t) = g {
                                dyn_lifetimes(ast, *t, &mut hs);
                            }
                        }
                    }
                }
                hs.contains(&n)
            };
            let uses = used.contains(&Region::Param(i)) || used_dyn.contains(&n);
            if uses && !in_header.contains(&Region::Param(i)) && !in_dyn_header() {
                errs.push(Diag::new(
                    "L0110",
                    ast.ty_span(aty),
                    format!("the lifetime parameter `'{}` is not constrained by the impl trait, self type, or predicates", interner.get(n)),
                    "Use the lifetime in the Self type or the trait's arguments, or remove it.",
                ));
            }
        }
        scope.assoc_regioned.push((at.name, t));
    }
}

/// The lifetime names written as dyn object bounds anywhere in `t`
/// (`dyn K + 'a`, `Box<dyn K + 'a>`): the region form erases them.
fn dyn_lifetimes(ast: &Ast, t: ast::TyId, out: &mut Vec<Sym>) {
    let args = |segs: &[ast::PathSeg], out: &mut Vec<Sym>| {
        for seg in segs {
            match &seg.args {
                ast::GenericArgs::Angle(xs) => {
                    for g in xs {
                        match g {
                            ast::GenericArg::Type(t) => dyn_lifetimes(ast, *t, out),
                            ast::GenericArg::Binding { kind: ast::BindingKind::EqTy(t), .. } => dyn_lifetimes(ast, *t, out),
                            _ => {}
                        }
                    }
                }
                ast::GenericArgs::Parenthesized { inputs, output } => {
                    for &t in inputs.iter().chain(output) {
                        dyn_lifetimes(ast, t, out);
                    }
                }
                ast::GenericArgs::None => {}
            }
        }
    };
    match ast.ty(t) {
        TyExpr::Dyn { bounds, lifetime, .. } => {
            out.extend(lifetime.map(|l| l.name));
            for b in bounds {
                match b {
                    ast::TraitBound::Lifetime(l) => out.push(l.name),
                    ast::TraitBound::Trait { trait_ref, .. } => args(&trait_ref.path, out),
                }
            }
        }
        TyExpr::Path(segs) => args(segs, out),
        TyExpr::Ref { inner, .. } | TyExpr::Paren(inner) | TyExpr::Slice(inner) | TyExpr::Ptr { inner, .. } | TyExpr::Array { inner, .. } => {
            dyn_lifetimes(ast, *inner, out)
        }
        TyExpr::Tuple(ts) => ts.iter().for_each(|&e| dyn_lifetimes(ast, e, out)),
        TyExpr::QPath { self_ty, trait_path, segs } => {
            dyn_lifetimes(ast, *self_ty, out);
            args(trait_path, out);
            args(segs, out);
        }
        _ => {}
    }
}

/// The target of a generic type alias, over its own params (`Param(i)`
/// in declared order, lifetimes left out).
fn resolve_generic_alias(p: &mut Program, a: &ast::TypeAliasDef) -> Result<Ty, Diag> {
    let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
    let mut scope = generics::Scope::default();
    let mut idx = 0u32;
    for g in &a.generics.params {
        match g {
            ast::GenericParam::Type { name, .. } => {
                scope.type_params.push((*name, idx));
                idx += 1;
            }
            ast::GenericParam::Const { name, ty, span, .. } => {
                // The declared type, for the const typing of its uses.
                let t = generics::resolve_scoped(&p.ast, &p.interner, &p.known, &cx, &mut p.tys, *ty, &generics::Scope::default())?;
                if generics::ConstScalar::of(&p.tys, t).is_none() {
                    return Err(Diag::new(
                        "L0110",
                        *span,
                        "this type is forbidden as the type of a const generic parameter",
                        "Use an integer type, `bool`, or `char`.",
                    ));
                }
                scope.const_params.push((*name, idx, t));
                idx += 1;
            }
            ast::GenericParam::Lifetime { lt, .. } => scope.lifetimes.push((lt.name, 0)),
        }
    }
    generics::resolve_scoped(&p.ast, &p.interner, &p.known, &cx, &mut p.tys, a.ty, &scope)
}

/// The rules on generic aliases once their targets resolve: the param
/// default rules (`check::defaults`), and every type param must appear
/// in the target (rustc E0091). Their resolved defaults join
/// `param_defaults` under the alias def.
fn check_generic_aliases(p: &mut Program, errs: &mut Vec<Diag>) {
    let (ast, interner, known) = (&p.ast, &p.interner, &p.known);
    for item in &ast.items {
        let ItemKind::Type(a) = &item.kind else { continue };
        if !a.generics.params.iter().any(|g| !matches!(g, ast::GenericParam::Lifetime { .. })) {
            continue;
        }
        crate::check::defaults::check_written(ast, known, &a.generics, crate::check::defaults::Owner::Type, errs);
        let Some(&target) = p.aliases.get(&a.name) else { continue };
        let used = crate::check::variance::params_in(p, target);
        let mut idx = 0u32;
        for g in &a.generics.params {
            match g {
                ast::GenericParam::Type { name, span, .. } => {
                    if !used.contains(&idx) {
                        errs.push(Diag::new(
                            "L0110",
                            *span,
                            format!("type parameter `{}` is never used", interner.get(*name)),
                            "Use the parameter in the alias's type, or remove it.",
                        ));
                    }
                    idx += 1;
                }
                ast::GenericParam::Const { .. } => idx += 1,
                ast::GenericParam::Lifetime { .. } => {}
            }
        }
        let defaults = {
            let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
            crate::check::defaults::resolve_all(ast, interner, known, &cx, &mut p.tys, &a.generics, false, errs)
        };
        if defaults.iter().any(Option::is_some) {
            let hash = crate::def::item_hash(p.stable_crate, &[], crate::def::DefKind::TypeAlias, interner.get(a.name), 0);
            p.gx.param_defaults.insert(DefId { krate: p.krate, hash }, defaults);
        }
    }
}

/// A const, static, or associated const item's written type: the plain
/// resolver, else the scoped one (a generic alias with its args and
/// defaults, a generic Adt applied to args).
fn resolve_item_ty(p: &mut Program, id: ast::TyId, self_ty: Option<Ty>) -> Result<Ty, Diag> {
    let (ast, interner, known) = (&p.ast, &p.interner, &p.known);
    // The plain resolver does not know lazy aliases (W-BORROW's generic
    // aliases): their errors (arity, defaults) come from the scoped one.
    let lazy_alias = p.gx.generic_aliases.keys().any(|&n| ty_mentions(ast, id, n));
    resolve_ty(ast, interner, known, &p.type_names, &p.aliases, &mut p.tys, id, self_ty).or_else(|d| {
        let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
        let scope = generics::Scope { self_ty, ..generics::Scope::default() };
        let generic_alias = lazy_alias || d.message.contains("needs the scoped resolver");
        generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, id, &scope).map_err(|e| if generic_alias { e } else { d })
    })
}

/// `resolve_item_ty` for the type of a const or static inside a fn
/// body: with block type items in scope (R18), it resolves in their
/// overlay (an alias `type G = Box<dyn Send>;` or a block struct), and
/// the enclosing fns' generic params are the E0401 error there.
fn resolve_stmt_item_ty(p: &mut Program, id: ast::TyId, at: Span) -> Result<Ty, Diag> {
    let mut scope = generics::Scope::default();
    apply_stmt_overlay(p, &mut scope, at);
    if scope.local_types.is_empty() && scope.local_aliases.is_empty() && scope.local_traits.is_empty() {
        return resolve_item_ty(p, id, None);
    }
    let (ast, interner, known) = (&p.ast, &p.interner, &p.known);
    let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
    generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, id, &scope)
}

/// The span of a path to a user Adt with lifetime params that writes none
/// (a hidden lifetime), anywhere in a written type.
fn hidden_lifetime(p: &Program, id: ast::TyId, params: &[Sym]) -> Option<Span> {
    match p.ast.ty(id) {
        TyExpr::Ref { inner, .. } | TyExpr::Paren(inner) | TyExpr::Slice(inner) => hidden_lifetime(p, *inner, params),
        TyExpr::Array { inner, .. } | TyExpr::Ptr { inner, .. } => hidden_lifetime(p, *inner, params),
        TyExpr::Tuple(elems) => elems.iter().find_map(|&e| hidden_lifetime(p, e, params)),
        TyExpr::Path(segs) => {
            let last = segs.last()?;
            if segs.len() == 1 && params.contains(&last.name) {
                return None;
            }
            let lts = p
                .type_names
                .get(&last.name)
                .and_then(|&a| p.gx.adt_lifetimes.get(&p.gx.adt_info[a.0 as usize].0))
                .map(|v| v.len())
                .or_else(|| p.gx.generic_aliases.get(&last.name).map(|a| a.lifetimes.len()))
                .unwrap_or(0);
            let written = match &last.args {
                ast::GenericArgs::Angle(a) => a.iter().any(|g| matches!(g, ast::GenericArg::Lifetime(_))),
                _ => false,
            };
            if lts > 0 && !written {
                return Some(p.ast.ty_span(id));
            }
            segs.iter().find_map(|s| match &s.args {
                ast::GenericArgs::Angle(a) => a.iter().find_map(|g| if let ast::GenericArg::Type(t) = g { hidden_lifetime(p, *t, params) } else { None }),
                _ => None,
            })
        }
        _ => None,
    }
}

/// Lifetime errors in a type alias body (W-BORROW): a reference must name
/// `'static` or one of the alias's lifetime params `lts` (an elided one is
/// rustc E0106, L0202; any other name is undeclared, E0261, L0203).
/// Elided lifetimes inside fn pointer types are the pointer's own; the
/// lifetime args of paths resolve with the body (`GenericAlias`).
fn alias_lifetimes(ast: &Ast, interner: &Interner, id: ast::TyId, lts: &[Sym], errs: &mut Vec<Diag>) {
    match ast.ty(id) {
        TyExpr::Ref { lifetime, inner, .. } => {
            match lifetime.map(|lt| (interner.get(lt.name), lt.span, lt.name)) {
                None | Some(("_", _, _)) => errs.push(Diag::new(
                    "L0202",
                    ast.ty_span(id),
                    "missing lifetime specifier in a type alias",
                    "Write `&'static`, name a lifetime parameter of the alias, or use an owned type.",
                )),
                Some(("static", _, _)) => {}
                Some((_, _, n)) if lts.contains(&n) => {}
                Some((_, span, _)) => errs.push(crate::check::generics::lifetime_diag(span)),
            }
            alias_lifetimes(ast, interner, *inner, lts, errs);
        }
        TyExpr::Tuple(elems) => elems.iter().for_each(|&e| alias_lifetimes(ast, interner, e, lts, errs)),
        TyExpr::Paren(i) | TyExpr::Slice(i) => alias_lifetimes(ast, interner, *i, lts, errs),
        TyExpr::Array { inner, .. } | TyExpr::Ptr { inner, .. } => alias_lifetimes(ast, interner, *inner, lts, errs),
        TyExpr::Path(segs) => {
            for seg in segs {
                if let ast::GenericArgs::Angle(args) = &seg.args {
                    for g in args {
                        if let ast::GenericArg::Type(t) = g {
                            alias_lifetimes(ast, interner, *t, lts, errs);
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// True when a written type names `name` as a path's first segment
/// anywhere (an alias's type param is used, rustc E0091).
fn ty_mentions(ast: &Ast, id: ast::TyId, name: Sym) -> bool {
    let seg_args = |seg: &ast::PathSeg| -> bool {
        match &seg.args {
            ast::GenericArgs::Angle(args) => args.iter().any(|g| match g {
                ast::GenericArg::Type(t) => ty_mentions(ast, *t, name),
                ast::GenericArg::Binding { kind: ast::BindingKind::EqTy(t), .. } => ty_mentions(ast, *t, name),
                _ => false,
            }),
            ast::GenericArgs::Parenthesized { inputs, output } => {
                inputs.iter().any(|&t| ty_mentions(ast, t, name)) || output.is_some_and(|t| ty_mentions(ast, t, name))
            }
            ast::GenericArgs::None => false,
        }
    };
    let bounds = |bs: &[ast::TraitBound]| bs.iter().any(|b| matches!(b, ast::TraitBound::Trait { trait_ref, .. } if trait_ref.path.iter().any(seg_args)));
    match ast.ty(id) {
        TyExpr::Path(segs) => segs.first().is_some_and(|s| s.name == name) || segs.iter().any(seg_args),
        TyExpr::Ref { inner, .. } | TyExpr::Paren(inner) | TyExpr::Slice(inner) => ty_mentions(ast, *inner, name),
        TyExpr::Array { inner, .. } | TyExpr::Ptr { inner, .. } => ty_mentions(ast, *inner, name),
        TyExpr::Tuple(elems) => elems.iter().any(|&e| ty_mentions(ast, e, name)),
        TyExpr::FnPtr { inputs, ret, .. } => inputs.iter().any(|i| ty_mentions(ast, i.ty, name)) || ret.is_some_and(|r| ty_mentions(ast, r, name)),
        TyExpr::Dyn { bounds: bs, .. } | TyExpr::ImplTrait { bounds: bs, .. } => bounds(bs),
        TyExpr::QPath { self_ty, trait_path, segs } => ty_mentions(ast, *self_ty, name) || trait_path.iter().any(seg_args) || segs.iter().any(seg_args),
        _ => false,
    }
}

/// Reports each path that starts with a non-prelude std type lust
/// knows by name (`ty::lang::STD_IMPORTED_TYPES`), unless the file
/// defines an item of that name. Only for files with no `use`.
fn std_names_without_use(ast: &Ast, interner: &Interner, errs: &mut Vec<Diag>) {
    // The driver sends a file with any `use` to the resolver; a bare
    // collect (unit tests) can still see one here.
    if ast.items.iter().chain(&ast.stmt_items).any(|it| matches!(it.kind, ItemKind::Use(_))) {
        return;
    }
    let defined = |name: Sym| {
        ast.items.iter().any(|it| match &it.kind {
            ItemKind::Struct(s) => s.name == name,
            ItemKind::Enum(e) => e.name == name,
            ItemKind::Type(a) => a.name == name,
            ItemKind::Trait(t) => t.name == name,
            _ => false,
        })
    };
    // Paths repeat their first names (`i64`, `Vec`), so each name is
    // looked up in the std table once.
    let mut known: HashMap<Sym, bool> = HashMap::default();
    let mut report = |first: Sym, span: Span| {
        let n = interner.get(first);
        if *known.entry(first).or_insert_with(|| crate::ty::lang::needs_std_use(n)) && !defined(first) {
            let hint = crate::ty::lang::std_use_hint(n);
            errs.push(Diag::new("L0102", span, format!("cannot find type `{n}` in this scope"), format!("Import it: `use {hint};`.")));
        }
    };
    for (i, t) in ast.tys.iter().enumerate() {
        if let ast::TyExpr::Path(segs) = t
            && let Some(s) = segs.first()
        {
            report(s.name, ast.ty_spans[i]);
        }
    }
    for (i, e) in ast.exprs.iter().enumerate() {
        if let ast::Expr::Path(segs) = e
            && segs.len() > 1
            && let Some(s) = segs.first()
        {
            report(s.name, ast.expr_spans[i]);
        }
    }
    // Variant patterns (`Ordering::Less =>`).
    for (i, pat) in ast.pats.iter().enumerate() {
        if let ast::Pat::Path(segs) | ast::Pat::TupleStruct { path: segs, .. } | ast::Pat::Struct { path: segs, .. } = pat
            && segs.len() > 1
            && let Some(s) = segs.first()
        {
            report(s.name, ast.pat_spans[i]);
        }
    }
}

/// The reject for a canonical std type path that lust cannot mean
/// here: its last segment names no builtin, and a user type of that
/// name must not stand in for std's.
pub(crate) fn std_path_shadowed(interner: &Interner, name: Sym, span: Span) -> Diag {
    let n = interner.get(name);
    if ["HashMap", "HashSet", "BTreeMap", "Arc"].contains(&n) {
        return not_yet("L0641", span, &format!("the `{n}` type"));
    }
    not_yet("L0505", span, &format!("the std path to `{n}` where a user type has the same name"))
}

/// Resolves a written type. `self_ty` is the type that `Self` means, inside an impl.
pub fn resolve_ty(
    ast: &Ast,
    interner: &Interner,
    known: &Known,
    type_names: &crate::driver::read_cover::NameMap<AdtId>,
    aliases: &crate::driver::read_cover::NameMap<Ty>,
    tys: &mut impl TyIntern,
    id: ast::TyId,
    self_ty: Option<Ty>,
) -> Result<Ty, Diag> {
    let span = ast.ty_span(id);
    match ast.ty(id) {
        TyExpr::Unit => Ok(UNIT),
        TyExpr::Never => Ok(NEVER),
        TyExpr::Ref { mutable, inner, .. } => {
            let inner = resolve_ty(ast, interner, known, type_names, aliases, tys, *inner, self_ty)?;
            Ok(tys.mk_ref(*mutable, inner))
        }
        TyExpr::Path(segs) => {
            // Accept a canonical std type path by its last segment
            // (`std::vec::Vec`); a wrong-module path
            // (`std::option::Vec`) never falls through.
            let std_prefixed = segs.len() > 1 && (segs[0].name == known.std || matches!(interner.get(segs[0].name), "core" | "alloc"));
            let seg = if std_prefixed {
                let path: Vec<&str> = segs.iter().map(|s| interner.get(s.name)).collect();
                if crate::ty::lang::std_path_kind(&path) != Some(crate::ty::lang::StdPathKind::Type)
                    && !crate::ty::lang::std_is_primitive_path(&path)
                {
                    let n = interner.get(segs.last().unwrap().name);
                    return Err(Diag::new("L0102", span, format!("unknown type `{n}`"), "Check the spelling, or define the type."));
                }
                // `std::cmp::Ordering` is the core enum (and the UTF-8
                // error paths the std structs) even where a user type of
                // that name owns the flat name.
                let last = segs.last().unwrap();
                if let Some(def) = generics::std_adt_of_path(&path)
                    && !last.has_args()
                {
                    return Ok(tys.intern(TyKind::Adt(def, EMPTY_ARGS)));
                }
                // `std::ops::Bound<T>` (W-STD-C phase 2).
                if path.len() == 3 && path[1] == "ops" && path[2] == "Bound" {
                    let seg = segs.last().unwrap();
                    let targs = seg.type_args();
                    let [b] = targs.as_slice() else {
                        return Err(Diag::new("L0110", span, "`Bound` takes one type argument", "Write `Bound<&T>`, for example."));
                    };
                    let b = resolve_ty(ast, interner, known, type_names, aliases, tys, *b, self_ty)?;
                    return Ok(tys.mk_adt(generics::bound_def(), &[GenericArg::Ty(b)]));
                }
                // `std::thread::Result<T>`: std's alias for
                // `Result<T, Box<dyn Any + Send>>` (R26).
                if path.len() == 3 && path[0] == "std" && path[1] == "thread" && path[2] == "Result" {
                    let seg = segs.last().unwrap();
                    let targs = seg.type_args();
                    let [t] = targs.as_slice() else {
                        return Err(Diag::new("L0110", span, "`thread::Result` takes one type argument", "Write `thread::Result<T>`."));
                    };
                    let t = resolve_ty(ast, interner, known, type_names, aliases, tys, *t, self_ty)?;
                    let any = crate::check::unwind::dyn_any_send(tys, generics::any_trait_def());
                    let err = tys.mk_boxed(any);
                    return Ok(tys.mk_result(t, err));
                }
                // `std::borrow::Cow<'a, B>` (W-STD-B), regions erased.
                if path.len() == 3 && path[1] == "borrow" && path[2] == "Cow" {
                    let seg = segs.last().unwrap();
                    let targs = seg.type_args();
                    let [b] = targs.as_slice() else {
                        return Err(Diag::new("L0110", span, "`Cow` takes one type argument", "Write `Cow<'a, str>`, for example."));
                    };
                    let b = resolve_ty(ast, interner, known, type_names, aliases, tys, *b, self_ty)?;
                    return Ok(tys.mk_adt(generics::cow_def(), &[GenericArg::Ty(b)]));
                }
                // The owning std iterators (W-STD-B, ty/iter.rs).
                if let Some(k) = crate::ty::iter::owning_by_path(&path) {
                    let seg = segs.last().unwrap();
                    let mut args = Vec::new();
                    for &a in seg.type_args().iter() {
                        args.push(resolve_ty(ast, interner, known, type_names, aliases, tys, a, self_ty)?);
                    }
                    if args.len() != k.n_args() {
                        return Err(Diag::new(
                            "L0110",
                            span,
                            format!("`{}` takes {} type argument(s)", k.display(), k.n_args()),
                            "Write the type arguments.",
                        ));
                    }
                    return Ok(crate::ty::iter::mk_iter(tys, k, &args));
                }
                // The hashed collections and hashers (W-STD-B): written
                // args, with the default `S = RandomState` filled.
                if let Some(k) = crate::ty::hash::HashTy::by_name(path[path.len() - 1])
                    && matches!(path[1], "collections" | "hash")
                {
                    use crate::ty::hash::{HashTy, mk_hash};
                    let seg = segs.last().unwrap();
                    let mut args = Vec::new();
                    for &a in seg.type_args().iter() {
                        args.push(resolve_ty(ast, interner, known, type_names, aliases, tys, a, self_ty)?);
                    }
                    let n = k.n_args();
                    let fill = matches!(k, HashTy::HashMap | HashTy::HashSet) && args.len() + 1 == n;
                    if fill {
                        args.push(mk_hash(tys, HashTy::RandomState, &[]));
                    }
                    if args.len() != n {
                        return Err(Diag::new(
                            "L0110",
                            span,
                            format!("`{}` takes {n} type argument(s) here", k.path().1),
                            "Write the type arguments.",
                        ));
                    }
                    return Ok(mk_hash(tys, k, &args));
                }
                segs.last().unwrap()
            } else if segs.len() == 1 {
                &segs[0]
            } else {
                let names: Vec<&str> = segs.iter().map(|s| interner.get(s.name)).collect();
                return Err(crate::diag::module_path_not_yet(&names, span));
            };
            let name = seg.name;
            let no_generics = |tys: Ty| -> Result<Ty, Diag> {
                if !seg.has_args() {
                    Ok(tys)
                } else {
                    Err(Diag::new("L0110", span, "this type takes no generic arguments", "Remove the `<...>` part."))
                }
            };
            if let Some(t) = known.int_ty(name) {
                return no_generics(int(t));
            }
            if name == known.bool {
                return no_generics(BOOL);
            }
            if name == known.char {
                return no_generics(CHAR);
            }
            if name == known.str {
                return no_generics(STR);
            }
            if name == known.String {
                return no_generics(STRING);
            }
            if name == known.self_type {
                return self_ty.ok_or_else(|| Diag::new("L0102", span, "`Self` is only valid inside an impl", "Write the type name."));
            }
            if name == known.Vec {
                let args = seg.type_args();
                if args.len() != 1 {
                    return Err(Diag::new("L0110", span, "Vec takes one type argument", "Write `Vec<T>`."));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                return Ok(tys.mk_vec(e));
            }
            if name == known.Option {
                let args = seg.type_args();
                if args.len() != 1 {
                    return Err(Diag::new("L0110", span, "Option takes one type argument", "Write `Option<T>`."));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                return Ok(tys.mk_option(e));
            }
            if name == known.Box {
                let args = seg.type_args();
                if args.len() != 1 {
                    return Err(Diag::new("L0110", span, "Box takes one type argument", "Write `Box<T>`."));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                // `Box<[T]>` (W-STD-A) and `Box<str>` (check/own.rs);
                // `Box<dyn Tr>` takes the scoped resolver.
                if matches!(tys.kind(e), TyKind::Dyn(..)) {
                    return Err(not_yet("L0641", span, "a Box of an unsized type"));
                }
                return Ok(tys.mk_boxed(e));
            }
            // A user type named `Rc` shadows std's (which needs a `use`),
            // but never on a std path.
            if name == known.Rc && (std_prefixed || !type_names.contains_key(&name)) {
                let args = seg.type_args();
                if args.len() != 1 {
                    return Err(Diag::new("L0110", span, "Rc takes one type argument", "Write `Rc<T>`."));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                // `Rc<str>` and `Rc<[T]>` (check/own.rs); `Rc<dyn Tr>`
                // takes the scoped resolver.
                if matches!(tys.kind(e), TyKind::Dyn(..)) {
                    return Err(not_yet("L0641", span, "an Rc of an unsized type"));
                }
                return Ok(tys.mk_rc(e));
            }
            // The sync and thread types (W-THREADS): by their std path,
            // or by a bare name a module glob of their std module imports.
            if let Some(k) = crate::check::sync::sync_type_named(interner, segs) {
                if k.arity() == 0 && k.regions() == 0 {
                    return no_generics(crate::ty::sync::sync_ty(tys, k, &[]));
                }
                let args = seg.type_args();
                if args.len() != k.arity() {
                    return Err(Diag::new("L0110", span, format!("{} takes one type argument", k.name()), format!("Write `{}<T>`.", k.name())));
                }
                let lts = match &seg.args {
                    crate::ast::GenericArgs::Angle(a) => a.iter().filter(|g| matches!(g, crate::ast::GenericArg::Lifetime(_))).count(),
                    _ => 0,
                };
                if lts != 0 && lts != k.regions() as usize {
                    return Err(Diag::new("L0110", span, format!("`{}` takes {} lifetime argument(s)", k.name(), k.regions()), "Match the number of lifetime parameters."));
                }
                if k.arity() == 0 {
                    return Ok(crate::ty::sync::sync_ty(tys, k, &[]));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                crate::check::sync::check_sync_arg(tys, k, e, span)?;
                return Ok(crate::ty::sync::sync_ty(tys, k, &[e]));
            }
            // `Weak<T>`, `RefCell<T>`, `Cell<T>`, and the `RefCell`
            // guards `Ref<'_, T>` / `RefMut<'_, T>` (W-STD-A, R26; the
            // import itself is checked where paths resolve). A guard's
            // element may be unsized (`Ref<str>`), as in std.
            if let n @ ("Weak" | "RefCell" | "Cell" | "Ref" | "RefMut") = interner.get(name)
                && (std_prefixed || !type_names.contains_key(&name))
            {
                let args = seg.type_args();
                if args.len() != 1 {
                    return Err(Diag::new("L0110", span, format!("{n} takes one type argument"), format!("Write `{n}<T>`.")));
                }
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                // `rc::Weak<dyn Tr>`, `Weak<str>` and `Weak<[T]>` (cgen/own.rs).
                let weak_dyn = n == "Weak" && matches!(tys.kind(e), TyKind::Dyn(..) | TyKind::Str | TyKind::Slice(_));
                let guard = matches!(n, "Ref" | "RefMut");
                if !weak_dyn && !guard && matches!(tys.kind(e), TyKind::Slice(_) | TyKind::Str | TyKind::Dyn(..)) {
                    return Err(not_yet("L0641", span, &format!("a {n} of an unsized type")));
                }
                let which: fn(&crate::ty::BuiltinDefs) -> crate::def::DefId = match n {
                    "Weak" => |d| d.weak,
                    "RefCell" => |d| d.refcell,
                    "Ref" => |d| d.ref_guard,
                    "RefMut" => |d| d.ref_mut_guard,
                    _ => |d| d.cell,
                };
                return Ok(tys.mk_builtin1(which, e));
            }
            if interner.get(name) == "ExitCode" && (std_prefixed || !type_names.contains_key(&name)) {
                return no_generics(tys.exit_code_ty());
            }
            if name == known.Result {
                let args = seg.type_args();
                if args.len() != 2 {
                    return Err(Diag::new("L0110", span, "Result takes two type arguments", "Write `Result<T, E>`."));
                }
                let t = resolve_ty(ast, interner, known, type_names, aliases, tys, args[0], self_ty)?;
                let e = resolve_ty(ast, interner, known, type_names, aliases, tys, args[1], self_ty)?;
                return Ok(tys.mk_result(t, e));
            }
            // A std path names std's type, never a user type or alias
            // of the same name (std's `Ordering` is the one std Adt).
            if let Some(&adt) = type_names.get(&name)
                && (!std_prefixed || generics::is_std_adt(tys, adt))
            {
                return no_generics(tys.adt_ty(adt));
            }
            if !std_prefixed && let Some(&t) = aliases.get(&name) {
                // A generic alias needs the scoped resolver (its args and
                // defaults): callers fall back to it on this error.
                if tys.flags(t).contains_param() {
                    return Err(Diag::new("L0110", span, "this generic alias needs the scoped resolver", "Write the alias's arguments."));
                }
                return no_generics(t);
            }
            let n = interner.get(name);
            if name == known.f32 {
                return no_generics(tys.float_ty(FloatTy::F32));
            }
            if name == known.f64 {
                return no_generics(tys.float_ty(FloatTy::F64));
            }
            if std_prefixed {
                return Err(std_path_shadowed(interner, name, span));
            }
            if ["BTreeMap", "Arc"].contains(&n) {
                return Err(not_yet("L0641", span, &format!("the `{n}` type")));
            }
            if let Some(k) = crate::ty::hash::HashTy::by_name(n) {
                return Err(crate::ty::hash::not_imported(k, span));
            }
            Err(Diag::new("L0102", span, format!("unknown type `{n}`"), "Check the spelling, or define the type."))
        }
        TyExpr::Tuple(elems) => {
            let mut out = Vec::with_capacity(elems.len());
            for &e in elems {
                out.push(resolve_ty(ast, interner, known, type_names, aliases, tys, e, self_ty)?);
            }
            let ts = tys.intern_tys(&out);
            Ok(tys.intern(TyKind::Tuple(ts)))
        }
        TyExpr::Slice(inner) => {
            let e = resolve_ty(ast, interner, known, type_names, aliases, tys, *inner, self_ty)?;
            Ok(tys.intern(TyKind::Slice(e)))
        }
        TyExpr::Array { inner, len } => {
            let e = resolve_ty(ast, interner, known, type_names, aliases, tys, *inner, self_ty)?;
            let n = eval_array_len(ast, interner, known, *len)
                .or_else(|| eval_len_layout_query(ast, interner, known, type_names, aliases, tys, *len))
                .ok_or_else(|| {
                    Diag::new("L0110", span, "array length must be a constant integer", "Write a literal length such as `[u8; 4]`.")
                })?;
            let c = tys.intern_const(ConstKind::Int(n));
            Ok(tys.intern(TyKind::Array(e, c)))
        }
        TyExpr::Paren(inner) => resolve_ty(ast, interner, known, type_names, aliases, tys, *inner, self_ty),
        TyExpr::FnPtr { qual, inputs, variadic, ret, .. } => {
            if *variadic || qual.is_unsafe || qual.is_async || qual.is_const || qual.abi.is_some() {
                return Err(not_yet("L0602", span, "qualified or variadic function pointer types"));
            }
            // W-BORROW3: this resolver erases every region, but a fn pointer
            // type keeps a written `'static` and its free lifetimes
            // (`generics::resolve_scoped`). Callers fall back to that one.
            if generics::fn_ptr_names_lifetime(ast, interner, id) {
                return Err(not_yet("L0602", span, "a fn pointer type with lifetimes here (it needs the scoped resolver)"));
            }
            let mut ins = Vec::with_capacity(inputs.len());
            for p in inputs {
                ins.push(resolve_ty(ast, interner, known, type_names, aliases, tys, p.ty, self_ty)?);
            }
            let out = match ret {
                Some(r) => resolve_ty(ast, interner, known, type_names, aliases, tys, *r, self_ty)?,
                None => UNIT,
            };
            // A pointer whose input borrows may need `FnSig::own_out`, which
            // only the scoped resolver can tell (an alias in its output may be
            // `'static`).
            if ins.iter().any(|&t| crate::check::closure::may_hold_region(tys, t)) {
                return Err(not_yet("L0602", span, "a fn pointer type with a borrowed input here (it needs the scoped resolver)"));
            }
            let inputs = tys.intern_tys(&ins);
            let sig = tys.intern_sig(crate::ty::Binder::plain(crate::ty::FnSig::plain(inputs, out)));
            Ok(tys.intern(TyKind::FnPtr(sig)))
        }
        _ => Err(not_yet("L0602", span, "this type form")),
    }
}

/// `t` with each reference `'static`, through tuples, arrays, slices,
/// and type args (a no-param fn's return type, `fn_item`).
fn static_refs_in(tys: &mut impl TyIntern, t: Ty) -> Ty {
    match tys.kind(t) {
        TyKind::Ref(_, m, inner) => {
            let i = static_refs_in(tys, inner);
            tys.intern(TyKind::Ref(crate::ty::Region::Static, m, i))
        }
        TyKind::Tuple(ts) => {
            let v: Vec<Ty> = tys.tys_of(ts).to_vec().into_iter().map(|e| static_refs_in(tys, e)).collect();
            let id = tys.intern_tys(&v);
            tys.intern(TyKind::Tuple(id))
        }
        TyKind::Array(e, n) => {
            let e = static_refs_in(tys, e);
            tys.intern(TyKind::Array(e, n))
        }
        TyKind::Slice(e) => {
            let e = static_refs_in(tys, e);
            tys.intern(TyKind::Slice(e))
        }
        TyKind::Adt(d, a) => {
            let args: Vec<crate::ty::GenericArg> = tys
                .args(a)
                .to_vec()
                .into_iter()
                .map(|g| match g {
                    crate::ty::GenericArg::Ty(e) => crate::ty::GenericArg::Ty(static_refs_in(tys, e)),
                    g => g,
                })
                .collect();
            tys.mk_adt(d, &args)
        }
        _ => t,
    }
}

/// Evaluates an array length expression at collection time. Only literal
/// integer arithmetic is in the bound today; named consts join with the
/// const evaluator (ARCH 2.13) in `ty::consteval`.
pub fn eval_array_len(ast: &Ast, interner: &Interner, known: &Known, e: ast::ExprId) -> Option<u128> {
    eval_array_len_in(ast, interner, known, e, None)
}

/// `eval_array_len` where `Self::X` names an associated const of the
/// type with flat name `self_name` (a non-generic `Self`, `Scope::self_name`).
pub fn eval_array_len_in(ast: &Ast, interner: &Interner, known: &Known, e: ast::ExprId, self_name: Option<Sym>) -> Option<u128> {
    let mut ev = ScalarEval::new(ast, interner, known);
    ev.self_name = self_name;
    u128::try_from(ev.int(e, None)?).ok()
}

/// The value expression of a named const found by scanning the
/// crate's items: a free const item for one name, an inherent
/// associated const for two (`Type::CONST`, by flat type name).
/// Signature-form const resolution uses this before the const
/// tables exist; the scan is by written (flat) name, as the
/// scalar interpreter's own lookups are. An associated const
/// answers only from one defining impl without type or const
/// params: same-name consts in disjoint impls (`impl G<u8>`,
/// `impl G<i32>`) and impl params the path must fix need the
/// typed pick, which knows the written type.
fn named_const_expr(ast: &Ast, names: &[Sym]) -> Option<ast::ExprId> {
    match names {
        [name] => {
            for item in &ast.items {
                if let ItemKind::Const(c) = &item.kind
                    && c.name == Some(*name)
                    && let Some(v) = c.value
                {
                    return Some(v);
                }
            }
            None
        }
        [ty_name, name] => {
            let mut found = None;
            for item in &ast.items {
                let ItemKind::Impl(imp) = &item.kind else { continue };
                // A body impl's self name may be a block type (R18).
                if imp.trait_ref.is_some() || imp.in_body {
                    continue;
                }
                let TyExpr::Path(segs) = ast.ty(imp.self_ty) else { continue };
                if segs.first().map(|s| s.name) != Some(*ty_name) {
                    continue;
                }
                for c in &imp.consts {
                    if c.name == *name
                        && let Some(v) = c.default
                    {
                        let generic = imp.generics.params.iter().any(|g| !matches!(g, ast::GenericParam::Lifetime { .. }));
                        if found.is_some() || generic {
                            return None;
                        }
                        found = Some(v);
                    }
                }
            }
            found
        }
        _ => None,
    }
}

/// The declared type of the named const `named_const_expr` finds,
/// with the same ambiguity rules (None when the pick is ambiguous).
fn named_const_decl(ast: &Ast, names: &[Sym]) -> Option<ast::TyId> {
    match names {
        [name] => {
            for item in &ast.items {
                if let ItemKind::Const(c) = &item.kind
                    && c.name == Some(*name)
                    && c.value.is_some()
                {
                    return Some(c.ty);
                }
            }
            None
        }
        [ty_name, name] => {
            let mut found = None;
            for item in &ast.items {
                let ItemKind::Impl(imp) = &item.kind else { continue };
                // A body impl's self name may be a block type (R18).
                if imp.trait_ref.is_some() || imp.in_body {
                    continue;
                }
                let TyExpr::Path(segs) = ast.ty(imp.self_ty) else { continue };
                if segs.first().map(|s| s.name) != Some(*ty_name) {
                    continue;
                }
                for c in &imp.consts {
                    if c.name == *name && c.default.is_some() {
                        let generic = imp.generics.params.iter().any(|g| !matches!(g, ast::GenericParam::Lifetime { .. }));
                        if found.is_some() || generic {
                            return None;
                        }
                        found = Some(c.ty);
                    }
                }
            }
            found
        }
        _ => None,
    }
}

/// The integer value of a named const (see `named_const_expr`),
/// for const generic arguments written as bare paths.
pub fn named_const_int(ast: &Ast, interner: &Interner, known: &Known, names: &[Sym]) -> Option<u128> {
    let e = named_const_expr(ast, names)?;
    u128::try_from(eval_const_int(ast, interner, known, e)?).ok()
}

/// The const-generic value of a free const item by name, in the
/// kind of its declared type: an integer const for the integer
/// types, `Bool` for `bool`, `Char` for `char`. A body const in
/// scope at the use site `at` wins over a module const of the
/// name, and another body item shadows it.
pub fn named_const_kind(ast: &Ast, interner: &Interner, known: &Known, name: Sym, at: Span) -> Option<ConstKind> {
    let kind_of = |c: &ast::ConstDef| -> Option<ConstKind> {
        let v = c.value?;
        let scalar = eval_const_int(ast, interner, known, v)?;
        match ast.ty(c.ty) {
            TyExpr::Path(segs) if segs.len() == 1 && segs[0].name == known.bool => Some(ConstKind::Bool(scalar != 0)),
            TyExpr::Path(segs) if segs.len() == 1 && segs[0].name == known.char => char::from_u32(scalar as u32).map(ConstKind::Char),
            _ => u128::try_from(scalar).ok().map(ConstKind::Int),
        }
    };
    if let Some(si) = stmt_item_at(ast, name, at) {
        let ItemKind::Const(c) = &ast.stmt_items[si as usize].kind else { return None };
        return kind_of(c);
    }
    for item in &ast.items {
        let ItemKind::Const(c) = &item.kind else { continue };
        if c.name != Some(name) {
            continue;
        }
        return kind_of(c);
    }
    None
}

/// A `size_of::<T>()` or `align_of::<T>()` array length, for types whose
/// layout needs no Adt fields (scalars, tuples, arrays). Adt lengths wait
/// for the full evaluator at the end of collection.
fn eval_len_layout_query(
    ast: &Ast,
    interner: &Interner,
    known: &Known,
    type_names: &crate::driver::read_cover::NameMap<AdtId>,
    aliases: &crate::driver::read_cover::NameMap<Ty>,
    tys: &mut impl TyIntern,
    e: ast::ExprId,
) -> Option<u128> {
    let ast::Expr::Call { callee, args } = ast.expr(e) else { return None };
    if !args.is_empty() {
        return None;
    }
    let ast::Expr::Path(segs) = ast.expr(*callee) else { return None };
    let last = segs.last()?;
    let name = interner.get(last.name);
    if !matches!(name, "size_of" | "align_of") {
        return None;
    }
    let targs = last.type_args();
    if targs.len() != 1 {
        return None;
    }
    let t = resolve_ty(ast, interner, known, type_names, aliases, tys, targs[0], None).ok()?;
    struct NoShapes;
    impl crate::ty::layout::AdtShapes for NoShapes {
        fn adt_shape(&self, _id: AdtId) -> Option<crate::ty::layout::AdtShape> {
            None
        }
    }
    let l = crate::ty::layout::layout_of_ty(tys, &NoShapes, t).ok()?;
    Some(if name == "align_of" { l.align as u128 } else { l.size as u128 })
}

pub fn parse_float_literal(text: &str) -> Option<(u64, Option<FloatTy>)> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    if let Some(t) = clean.strip_suffix("f32") {
        return t.parse::<f32>().ok().map(|v| (v.to_bits() as u64, Some(FloatTy::F32)));
    }
    if let Some(t) = clean.strip_suffix("f64") {
        return t.parse::<f64>().ok().map(|v| (v.to_bits(), Some(FloatTy::F64)));
    }
    clean.parse::<f64>().ok().map(|v| (v.to_bits(), None))
}

/// Parses a float literal straight to the target type, with one rounding.
/// The runtime parser in src/runtime/float.c gives the same bits: both are
/// correctly rounded (ties to even) for the target precision. A literal
/// that rounds to infinity is out of range; callers report it through
/// `float_literal_overflows`.
pub fn parse_float_literal_as(text: &str, f: FloatTy) -> Option<u64> {
    let clean: String = text.chars().filter(|&c| c != '_').collect();
    let t = clean.strip_suffix("f32").or_else(|| clean.strip_suffix("f64")).unwrap_or(&clean);
    match f {
        FloatTy::F32 => t.parse::<f32>().ok().map(|v| v.to_bits() as u64),
        FloatTy::F64 => t.parse::<f64>().ok().map(|v| v.to_bits()),
    }
}

/// True when a float literal is out of range for the target type: its
/// correctly rounded value is infinite. Rust literals cannot write inf,
/// so an infinite parse always means overflow (the deny-level
/// `overflowing_literals` lint in rustc).
pub fn float_literal_overflows(text: &str, f: FloatTy) -> bool {
    match parse_float_literal_as(text, f) {
        Some(bits) => match f {
            FloatTy::F32 => f32::from_bits(bits as u32).is_infinite(),
            FloatTy::F64 => f64::from_bits(bits).is_infinite(),
        },
        None => false,
    }
}

/// Evaluates a constant integer expression at collection time: literals,
/// negation and bit-not, integer arithmetic and shifts, and `as` casts
/// between scalar types. Returns None for anything else, and for every
/// const failure (typed overflow, division by zero, a shift at or past
/// the bit width, an out-of-range literal). The full bounded evaluator
/// lives in `ty::consteval` (ARCH 2.13); this is its scalar subset for
/// discriminants and array lengths, which collect needs before bodies
/// run, sharing the same typed arithmetic core.
pub fn eval_const_int(ast: &Ast, interner: &Interner, known: &Known, e: ast::ExprId) -> Option<i128> {
    eval_const_int_exp(ast, interner, known, e, None)
}

/// `eval_const_int` with an expected integer type pinning the
/// expression's unpinned literals (a discriminant's repr type, a
/// const parameter's declared type): a literal or operation out
/// of that type's range is None, as rustc rejects it.
pub fn eval_const_int_exp(ast: &Ast, interner: &Interner, known: &Known, e: ast::ExprId, exp: Option<IntTy>) -> Option<i128> {
    ScalarEval::new(ast, interner, known).int(e, exp)
}

/// The written integer type of a discriminant expression, in the
/// subset `eval_scalar` can value. `AnyInt` is an unsuffixed literal
/// or arithmetic over only `AnyInt` leaves: rustc infers the
/// discriminant type for those. Every other leaf must name the
/// discriminant type exactly (rustc E0308): a suffixed literal, a
/// const item, an associated const, `T::MAX` and kin, a const fn
/// call, or a cast. `Bad` covers non-integer leaves and forms this
/// walk does not type; the caller rejects them, so a form it cannot
/// prove is a false reject, never a false accept.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DTy {
    Exact(IntTy),
    AnyInt,
    Bad,
}

fn dty_combine(a: DTy, b: DTy) -> DTy {
    match (a, b) {
        (DTy::Exact(x), DTy::Exact(y)) if x == y => DTy::Exact(x),
        (DTy::Exact(x), DTy::AnyInt) | (DTy::AnyInt, DTy::Exact(x)) => DTy::Exact(x),
        (DTy::AnyInt, DTy::AnyInt) => DTy::AnyInt,
        _ => DTy::Bad,
    }
}

/// True when a discriminant leaf of type `got` may stand where the
/// discriminant type is `want`.
fn dty_accepts(got: DTy, want: IntTy) -> bool {
    matches!(got, DTy::AnyInt) || matches!(got, DTy::Exact(it) if it == want)
}

/// The integer type a one-segment written type denotes, as a `DTy`.
fn dty_of_ty(ast: &Ast, known: &Known, ty: ast::TyId) -> DTy {
    match ast.ty(ty) {
        TyExpr::Path(segs) if segs.len() == 1 => match known.int_ty(segs[0].name) {
            Some(it) => DTy::Exact(it),
            None => DTy::Bad,
        },
        _ => DTy::Bad,
    }
}

/// The discriminant of `E::V` (as the scalar evaluator values `E::V as
/// int`), typed by E's repr integer. E must be the one item of that
/// type name in the crate (resolved paths carry unique flat names; a
/// struct, alias, trait, block item, or `use` of the same name makes it
/// None) and a unit-only enum without generics. None when any
/// discriminant does not evaluate or overflows (rustc E0370).
fn unit_variant_discr(ast: &Ast, interner: &Interner, known: &Known, segs: &[ast::PathSeg]) -> Option<SVal> {
    let [e, v] = segs else { return None };
    if e.has_args() || v.has_args() {
        return None;
    }
    fn binds_use(t: &ast::UseTree, name: Sym) -> bool {
        match &t.kind {
            ast::UseKind::Simple { rename } => rename.or(t.prefix.last().map(|s| s.name)) == Some(name),
            ast::UseKind::Glob => false,
            ast::UseKind::Group(ts) => ts.iter().any(|c| binds_use(c, name)),
        }
    }
    fn scan<'a>(items: &'a [ast::Item], name: Sym, found: &mut Vec<&'a ast::Item>, other: &mut bool) {
        for item in items {
            match &item.kind {
                ItemKind::Enum(d) if d.name == name => found.push(item),
                ItemKind::Struct(d) if d.name == name => *other = true,
                ItemKind::Union(d) if d.name == name => *other = true,
                ItemKind::Trait(d) if d.name == name => *other = true,
                ItemKind::Type(d) if d.name == name => *other = true,
                ItemKind::Use(u) if binds_use(&u.tree, name) => *other = true,
                ItemKind::Mod(m) => scan(m.body.as_deref().unwrap_or(&[]), name, found, other),
                _ => {}
            }
        }
    }
    let (mut found, mut other) = (Vec::new(), false);
    scan(&ast.items, e.name, &mut found, &mut other);
    let mut in_blocks = Vec::new();
    scan(&ast.stmt_items, e.name, &mut in_blocks, &mut other);
    let ([item], false, true) = (found.as_slice(), other, in_blocks.is_empty()) else { return None };
    let ItemKind::Enum(d) = &item.kind else { return None };
    if !d.generics.params.is_empty() || d.variants.iter().any(|x| !matches!(x.fields, ast::VariantFields::Unit)) {
        return None;
    }
    let it = disc_want_ty(interner, &item.attrs);
    let mut next: i128 = 0;
    for x in &d.variants {
        let disc = match x.discriminant {
            Some(de) => eval_const_int_exp(ast, interner, known, de, Some(it))?,
            None if next >= 0 && next as u128 > it.max128() => return None,
            None => next,
        };
        if x.name == v.name {
            return Some(SVal::Int(disc, Some(it)));
        }
        next = disc.checked_add(1)?;
    }
    None
}

/// The discriminant type of an enum: the repr integer when written,
/// else `isize`. Reads the first integer hint in the repr lists;
/// `parse_repr` reports conflicting hints separately.
pub(crate) fn disc_want_ty(interner: &Interner, attrs: &[ast::Attr]) -> IntTy {
    use crate::lexer::{Tok, TokenTree};
    for attr in attrs {
        if attr.path.len() != 1 || interner.get(attr.path[0].name) != "repr" {
            continue;
        }
        let ast::AttrInput::List(_, trees) = &attr.input else { continue };
        for t in trees {
            let TokenTree::Token(t) = t else { continue };
            let Tok::Ident(name) = t.tok else { continue };
            let it = match interner.get(name) {
                "u8" => IntTy::U8,
                "u16" => IntTy::U16,
                "u32" => IntTy::U32,
                "u64" => IntTy::U64,
                "u128" => IntTy::U128,
                "usize" => IntTy::Usize,
                "i8" => IntTy::I8,
                "i16" => IntTy::I16,
                "i32" => IntTy::I32,
                "i64" => IntTy::I64,
                "i128" => IntTy::I128,
                "isize" => IntTy::Isize,
                _ => continue,
            };
            return it;
        }
    }
    IntTy::Isize
}

/// The written integer type of discriminant expression `e`. Mirrors
/// the name resolution of `eval_scalar` (scan of the crate's items),
/// so every leaf that evaluator can value is typed here.
fn disc_ty(ast: &Ast, interner: &Interner, known: &Known, e: ast::ExprId, env: &mut Vec<(Sym, DTy)>, depth: u32) -> DTy {
    if depth > 32 {
        return DTy::Bad;
    }
    use crate::ast::{BinOp, Expr, Pat, Stmt};
    match ast.expr(e) {
        Expr::Int(_, suffix) => match suffix {
            Some(s) => DTy::Exact(IntTy::from_suffix(*s)),
            None => DTy::AnyInt,
        },
        Expr::Unary { expr, .. } => disc_ty(ast, interner, known, *expr, env, depth + 1),
        Expr::Binary { op, lhs, rhs } => match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
                dty_combine(disc_ty(ast, interner, known, *lhs, env, depth + 1), disc_ty(ast, interner, known, *rhs, env, depth + 1))
            }
            BinOp::Shl | BinOp::Shr => {
                let l = disc_ty(ast, interner, known, *lhs, env, depth + 1);
                if disc_ty(ast, interner, known, *rhs, env, depth + 1) == DTy::Bad { DTy::Bad } else { l }
            }
            _ => DTy::Bad,
        },
        Expr::Cast { ty, .. } => dty_of_ty(ast, known, *ty),
        Expr::Path(segs) if segs.len() == 1 => {
            let name = segs[0].name;
            if let Some((_, t)) = env.iter().rev().find(|(n, _)| *n == name) {
                return *t;
            }
            for item in &ast.items {
                if let ItemKind::Const(c) = &item.kind
                    && c.name == Some(name)
                {
                    return dty_of_ty(ast, known, c.ty);
                }
            }
            DTy::Bad
        }
        Expr::Path(segs) if segs.len() == 2 && !segs[0].has_args() => {
            if let Some(it) = known.int_ty(segs[0].name) {
                return match interner.get(segs[1].name) {
                    "MAX" | "MIN" => DTy::Exact(it),
                    "BITS" => DTy::Exact(IntTy::U32),
                    _ => DTy::Bad,
                };
            }
            let mut found = DTy::Bad;
            let mut seen = false;
            for item in &ast.items {
                let ItemKind::Impl(imp) = &item.kind else { continue };
                if imp.trait_ref.is_some() || imp.in_body {
                    continue;
                }
                let TyExpr::Path(tsegs) = ast.ty(imp.self_ty) else { continue };
                if tsegs.first().map(|s| s.name) != Some(segs[0].name) {
                    continue;
                }
                for c in &imp.consts {
                    if c.name == segs[1].name && c.default.is_some() {
                        let generic = imp.generics.params.iter().any(|g| !matches!(g, ast::GenericParam::Lifetime { .. }));
                        if seen || generic {
                            return DTy::Bad;
                        }
                        seen = true;
                        found = dty_of_ty(ast, known, c.ty);
                    }
                }
            }
            found
        }
        Expr::Call { callee, args } => {
            let Expr::Path(segs) = ast.expr(*callee) else { return DTy::Bad };
            if segs.len() != 1 {
                return DTy::Bad;
            }
            let name = segs[0].name;
            for item in &ast.items {
                let ItemKind::Fn(def) = &item.kind else { continue };
                if def.name != name || !def.qual.is_const || def.self_param.is_some() || def.params.len() != args.len() {
                    continue;
                }
                for (param, &arg) in def.params.iter().zip(args) {
                    if param.pat.is_some() {
                        return DTy::Bad;
                    }
                    if let DTy::Exact(pt) = dty_of_ty(ast, known, param.ty)
                        && !dty_accepts(disc_ty(ast, interner, known, arg, env, depth + 1), pt)
                    {
                        return DTy::Bad;
                    }
                }
                return match def.ret {
                    Some(ty) => dty_of_ty(ast, known, ty),
                    None => DTy::Bad,
                };
            }
            DTy::Bad
        }
        Expr::Block(b) | Expr::ConstBlock(b) => disc_ty_block(ast, interner, known, *b, env, depth + 1),
        Expr::If { then, els: Some(els), .. } => {
            let t = disc_ty_block(ast, interner, known, *then, env, depth + 1);
            dty_combine(t, disc_ty(ast, interner, known, *els, env, depth + 1))
        }
        _ => DTy::Bad,
    }
}

/// Block half of `disc_ty`: plain binding lets (a written type on a
/// let pins the binding, and the init must fit it), then the tail.
fn disc_ty_block(ast: &Ast, interner: &Interner, known: &Known, b: ast::BlockId, env: &mut Vec<(Sym, DTy)>, depth: u32) -> DTy {
    use crate::ast::{Pat, Stmt};
    let saved = env.len();
    let block = ast.block(b);
    let mut out = DTy::Bad;
    let mut ok = true;
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { pat, ty, init: Some(init), els: None, .. } => {
                let Pat::Binding { name, .. } = ast.pat(*pat) else {
                    ok = false;
                    break;
                };
                let name = *name;
                let it = disc_ty(ast, interner, known, *init, env, depth + 1);
                let bt = match ty {
                    Some(t) => {
                        let wt = dty_of_ty(ast, known, *t);
                        match wt {
                            DTy::Exact(x) if dty_accepts(it, x) => wt,
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    None => it,
                };
                if bt == DTy::Bad {
                    ok = false;
                    break;
                }
                env.push((name, bt));
            }
            Stmt::Expr { .. } => {}
            _ => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        out = match block.tail {
            Some(tail) => disc_ty(ast, interner, known, tail, env, depth + 1),
            None => DTy::Bad,
        };
    }
    env.truncate(saved);
    out
}

/// A scalar value in the interpreter below. Integers carry their
/// type when anything pins it (a suffix, a written type, a cast, a
/// const item's declared type) and use the stored form of
/// `ty::consteval` (sign- or zero-extended, raw bits at 128), so
/// arithmetic through `ty::consteval::int_binop` matches the main
/// evaluator exactly. `None` is an unpinned `{integer}` value kept
/// wide in i128.
#[derive(Clone, Copy, PartialEq)]
enum SVal {
    Int(i128, Option<IntTy>),
    Float(u64, FloatTy),
    Bool(bool),
    Char(char),
    Unit,
    /// A reference to a local's slot (`&x`, `&mut x`, a `&mut` param).
    Ref(usize),
}

/// The scalar type a context expects, for pinning literals.
#[derive(Clone, Copy, PartialEq)]
enum SExp {
    Int(IntTy),
    Float(FloatTy),
    Bool,
    Char,
}

/// Control flow of the scalar interpreter below. `V(None)` is
/// "outside the subset, or a const failure" (overflow, division
/// by zero, a bad shift, an out-of-range literal, a panic):
/// callers reject both, never a wrong value.
enum SFlow {
    V(Option<SVal>),
    Break(Option<Sym>, Option<SVal>),
    Continue(Option<Sym>),
    Return(Option<SVal>),
}

/// Locals by name, pointing into the interpreter's slot arena.
/// Slots outlive their name scope, so a `&mut` param can write
/// through to a caller's local.
type SEnv = Vec<(Sym, usize)>;

/// The scalar type a written one-segment type names, if scalar.
fn sexp_of_written(ast: &Ast, interner: &Interner, known: &Known, ty: ast::TyId) -> Option<SExp> {
    let TyExpr::Path(segs) = ast.ty(ty) else { return None };
    if segs.len() != 1 {
        return None;
    }
    let name = segs[0].name;
    if let Some(it) = known.int_ty(name) {
        return Some(SExp::Int(it));
    }
    match interner.get(name) {
        "f32" => Some(SExp::Float(FloatTy::F32)),
        "f64" => Some(SExp::Float(FloatTy::F64)),
        "bool" => Some(SExp::Bool),
        "char" => Some(SExp::Char),
        _ => None,
    }
}

/// True when the plain (wide) integer `v` is a value of `it`.
fn s_fits(v: i128, it: IntTy) -> bool {
    let bits = it.bits();
    if bits >= 128 {
        return true;
    }
    if it.signed() { v >= -(1i128 << (bits - 1)) && v < (1i128 << (bits - 1)) } else { v >= 0 && v < (1i128 << bits) }
}

/// Pins a scalar value to an expected integer type: a flex value
/// must fit, a typed value must already have the type.
fn s_coerce_int(v: SVal, it: IntTy) -> Option<SVal> {
    match v {
        SVal::Int(_, Some(t)) if t == it => Some(v),
        SVal::Int(x, None) if s_fits(x, it) => Some(SVal::Int(x, Some(it))),
        _ => None,
    }
}

/// The expected scalar type of a stored value, for assignments.
fn s_exp_of(v: SVal) -> Option<SExp> {
    match v {
        SVal::Int(_, Some(it)) => Some(SExp::Int(it)),
        SVal::Float(_, f) => Some(SExp::Float(f)),
        SVal::Bool(_) => Some(SExp::Bool),
        SVal::Char(_) => Some(SExp::Char),
        _ => None,
    }
}

/// A shift amount from a scalar value of any integer type.
fn s_shift_amt(v: SVal) -> Option<u32> {
    match v {
        SVal::Int(x, Some(it)) if !it.signed() => u32::try_from(x as u128).ok(),
        SVal::Int(x, _) if x >= 0 => u32::try_from(x).ok(),
        _ => None,
    }
}

/// One arithmetic, bitwise, shift, or comparison operation over
/// two scalar values whose types the caller has reconciled (flex
/// operands re-pinned or coerced). Typed integer arithmetic goes
/// through `int_binop`, the main evaluator's own core.
fn s_combine(op: crate::ast::BinOp, a: SVal, b: SVal) -> Option<SVal> {
    use crate::ast::BinOp;
    use crate::ty::consteval::{int_binop, int_cmp};
    if op.is_cmp() {
        let ord = match (a, b) {
            (SVal::Int(x, Some(it)), SVal::Int(y, Some(it2))) if it == it2 => Some(int_cmp(it, x, y)),
            (SVal::Int(x, _), SVal::Int(y, _)) => Some(x.cmp(&y)),
            (SVal::Char(x), SVal::Char(y)) => Some(x.cmp(&y)),
            (SVal::Bool(x), SVal::Bool(y)) => Some(x.cmp(&y)),
            (SVal::Float(x, FloatTy::F64), SVal::Float(y, FloatTy::F64)) => f64::from_bits(x).partial_cmp(&f64::from_bits(y)),
            (SVal::Float(x, FloatTy::F32), SVal::Float(y, FloatTy::F32)) => f32::from_bits(x as u32).partial_cmp(&f32::from_bits(y as u32)),
            _ => return None,
        };
        let eq = ord == Some(std::cmp::Ordering::Equal);
        let r = match op {
            BinOp::Eq => eq,
            BinOp::Ne => !eq,
            BinOp::Lt => ord == Some(std::cmp::Ordering::Less),
            BinOp::Le => matches!(ord, Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)),
            BinOp::Gt => ord == Some(std::cmp::Ordering::Greater),
            BinOp::Ge => matches!(ord, Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)),
            _ => return None,
        };
        return Some(SVal::Bool(r));
    }
    match (a, b) {
        // A flex left side with a typed right side (a compound
        // assignment through a `&mut` param into an unpinned
        // local): the operation runs in the typed side's type.
        (SVal::Int(x, None), SVal::Int(_, Some(it))) if !matches!(op, BinOp::Shl | BinOp::Shr) => {
            let x = match s_coerce_int(a, it) {
                Some(SVal::Int(x, _)) => x,
                _ => return None,
            };
            let SVal::Int(y, _) = b else { return None };
            int_binop(op, it, x, y).map(|r| SVal::Int(r, Some(it)))
        }
        (SVal::Int(x, Some(it)), SVal::Int(y, t2)) => {
            let y = if matches!(op, BinOp::Shl | BinOp::Shr) {
                s_shift_amt(b)? as i128
            } else {
                if t2 != Some(it) {
                    return None;
                }
                y
            };
            int_binop(op, it, x, y).map(|r| SVal::Int(r, Some(it)))
        }
        (SVal::Int(x, None), SVal::Int(y, None)) => {
            let r = match op {
                BinOp::Add => x.checked_add(y),
                BinOp::Sub => x.checked_sub(y),
                BinOp::Mul => x.checked_mul(y),
                BinOp::Div => x.checked_div(y),
                BinOp::Rem => x.checked_rem(y),
                BinOp::BitAnd => Some(x & y),
                BinOp::BitOr => Some(x | y),
                BinOp::BitXor => Some(x ^ y),
                BinOp::Shl => s_shift_amt(b).and_then(|s| x.checked_shl(s)),
                BinOp::Shr => s_shift_amt(b).and_then(|s| x.checked_shr(s)),
                _ => return None,
            }?;
            Some(SVal::Int(r, None))
        }
        (SVal::Float(x, f), SVal::Float(y, g)) if f == g => {
            let bits = match f {
                FloatTy::F64 => {
                    let (x, y) = (f64::from_bits(x), f64::from_bits(y));
                    let r = match op {
                        BinOp::Add => x + y,
                        BinOp::Sub => x - y,
                        BinOp::Mul => x * y,
                        BinOp::Div => x / y,
                        BinOp::Rem => x % y,
                        _ => return None,
                    };
                    r.to_bits()
                }
                FloatTy::F32 => {
                    let (x, y) = (f32::from_bits(x as u32), f32::from_bits(y as u32));
                    let r = match op {
                        BinOp::Add => x + y,
                        BinOp::Sub => x - y,
                        BinOp::Mul => x * y,
                        BinOp::Div => x / y,
                        BinOp::Rem => x % y,
                        _ => return None,
                    };
                    r.to_bits() as u64
                }
            };
            Some(SVal::Float(bits, f))
        }
        (SVal::Bool(x), SVal::Bool(y)) => match op {
            BinOp::BitAnd => Some(SVal::Bool(x & y)),
            BinOp::BitOr => Some(SVal::Bool(x | y)),
            BinOp::BitXor => Some(SVal::Bool(x ^ y)),
            _ => None,
        },
        _ => None,
    }
}

/// The scalar interpreter behind `eval_const_int`: typed scalar
/// arithmetic over literals, env-bound names, const items found by
/// scanning the crate's items, and calls to const fns whose bodies
/// stay in this scalar subset (lets, assignments through locals
/// and `&mut` params, if, while, loop, labeled break/continue).
/// It answers at collection time, before function signatures
/// exist, so it cannot use the full evaluator; it runs the same
/// typed arithmetic core (`ty::consteval::int_binop` and friends),
/// so the two cannot disagree on a value. Anything outside the
/// subset, and every const failure, is None.
struct ScalarEval<'x> {
    ast: &'x Ast,
    interner: &'x Interner,
    known: &'x Known,
    slots: Vec<SVal>,
    steps: u64,
    /// How many const fn bodies the evaluation is inside. A const fn
    /// body is borrow-checked as a fn; the const expression itself is
    /// not (it has no body of its own), so at depth 0 this subset
    /// takes no borrow and no assignment to a binding without `mut`
    /// (rustc E0499, E0502, E0506, E0384 and kin could not be seen).
    fn_depth: u32,
    /// The slots of the bindings without `mut` at depth 0.
    fixed: Vec<usize>,
    /// The flat type name `Self` names in the expression being
    /// evaluated: the caller's non-generic `Self` at the top, the
    /// owning type inside an associated const's value, None inside a
    /// free const or fn. `Self::X` with None is outside the subset.
    self_name: Option<Sym>,
}

impl<'x> ScalarEval<'x> {
    fn new(ast: &'x Ast, interner: &'x Interner, known: &'x Known) -> Self {
        ScalarEval { ast, interner, known, slots: Vec::new(), steps: 0, fn_depth: 0, fixed: Vec::new(), self_name: None }
    }

    /// The integer value of `e` (a bool or char counts as its code),
    /// with `exp` pinning its unpinned literals as `eval_const_int_exp`.
    fn int(&mut self, e: ast::ExprId, exp: Option<IntTy>) -> Option<i128> {
        let mut env: SEnv = Vec::new();
        let v = match self.expr(&mut env, e, exp.map(SExp::Int), 0) {
            SFlow::V(Some(v)) => v,
            _ => return None,
        };
        match v {
            SVal::Int(x, _) => match exp {
                Some(it) => s_coerce_int(v, it).map(|c| match c {
                    SVal::Int(x, _) => x,
                    _ => unreachable!(),
                }),
                None => Some(x),
            },
            SVal::Bool(b) => Some(b as i128),
            SVal::Char(c) => Some(c as i128),
            _ => None,
        }
    }

    fn new_slot(&mut self, v: SVal) -> usize {
        self.slots.push(v);
        self.slots.len() - 1
    }

    /// Stores `v` into a slot, pinning a flex integer to the
    /// slot's type. False on a kind or type mismatch.
    fn store(&mut self, slot: usize, v: SVal) -> bool {
        let cur = self.slots[slot];
        let v = match (cur, v) {
            (SVal::Int(_, Some(it)), SVal::Int(..)) => match s_coerce_int(v, it) {
                Some(v) => v,
                None => return false,
            },
            (SVal::Int(..), SVal::Int(..)) => v,
            (SVal::Float(_, f), SVal::Float(_, g)) if f == g => v,
            (SVal::Bool(_), SVal::Bool(_)) | (SVal::Char(_), SVal::Char(_)) | (SVal::Ref(_), SVal::Ref(_)) => v,
            _ => return false,
        };
        self.slots[slot] = v;
        true
    }

    /// The plain value of an expression, or None outside the subset.
    fn val(&mut self, env: &mut SEnv, e: ast::ExprId, exp: Option<SExp>, depth: u32) -> Option<SVal> {
        match self.expr(env, e, exp, depth) {
            SFlow::V(v) => v,
            _ => None,
        }
    }

    fn bool_val(&mut self, env: &mut SEnv, e: ast::ExprId, depth: u32) -> Option<bool> {
        match self.val(env, e, Some(SExp::Bool), depth)? {
            SVal::Bool(b) => Some(b),
            SVal::Int(x, _) => Some(x != 0),
            _ => None,
        }
    }

    /// The slot an assignable place names: a local, or a deref of
    /// a reference to one.
    fn place_slot(&mut self, env: &mut SEnv, e: ast::ExprId, depth: u32) -> Option<usize> {
        use crate::ast::Expr;
        match self.ast.expr(e) {
            Expr::Path(segs) if segs.len() == 1 => env.iter().rev().find(|(n, _)| *n == segs[0].name).map(|(_, s)| *s),
            Expr::Deref(inner) => match self.val(env, *inner, None, depth + 1)? {
                SVal::Ref(s) => Some(s),
                _ => None,
            },
            _ => None,
        }
    }

    /// One block: plain binding lets and expression statements,
    /// then the tail value. A statement outside the subset aborts
    /// the whole evaluation (it may have effects this interpreter
    /// cannot see, so skipping it could give a wrong value).
    fn block(&mut self, env: &mut SEnv, b: ast::BlockId, exp: Option<SExp>, depth: u32) -> SFlow {
        use crate::ast::{Pat, Stmt};
        self.steps += 1;
        if self.steps > 100_000 {
            return SFlow::V(None);
        }
        let ast = self.ast;
        let saved = env.len();
        let block = ast.block(b);
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { pat, ty, init, els, .. } => {
                    let (Some(init), None, Pat::Binding { name, mutable, .. }) = (init, els, ast.pat(*pat)) else {
                        env.truncate(saved);
                        return SFlow::V(None);
                    };
                    let (name, mutable) = (*name, *mutable);
                    let lexp = ty.and_then(|t| sexp_of_written(self.ast, self.interner, self.known, t));
                    let Some(mut v) = self.val(env, *init, lexp, depth + 1) else {
                        env.truncate(saved);
                        return SFlow::V(None);
                    };
                    if let Some(SExp::Int(it)) = lexp {
                        match s_coerce_int(v, it) {
                            Some(c) => v = c,
                            None => {
                                env.truncate(saved);
                                return SFlow::V(None);
                            }
                        }
                    }
                    let slot = self.new_slot(v);
                    if !mutable && self.fn_depth == 0 {
                        self.fixed.push(slot);
                    }
                    env.push((name, slot));
                }
                Stmt::Expr { expr, .. } => match self.expr(env, *expr, None, depth + 1) {
                    SFlow::V(Some(_)) => {}
                    other => {
                        env.truncate(saved);
                        return other;
                    }
                },
                _ => {
                    env.truncate(saved);
                    return SFlow::V(None);
                }
            }
        }
        let out = match block.tail {
            Some(tail) => self.expr(env, tail, exp, depth + 1),
            None => SFlow::V(Some(SVal::Unit)),
        };
        env.truncate(saved);
        out
    }

    /// A binary operation: operand types reconcile as in the main
    /// evaluator (an unpinned operand takes the other side's type,
    /// two unpinned comparison operands fall back to i32, a shift's
    /// amount never pins the shifted value).
    fn binary(&mut self, env: &mut SEnv, op: crate::ast::BinOp, lhs: ast::ExprId, rhs: ast::ExprId, exp: Option<SExp>, depth: u32) -> Option<SVal> {
        use crate::ast::BinOp;
        use crate::ty::consteval::flex_int_expr;
        let ast = self.ast;
        let shift = matches!(op, BinOp::Shl | BinOp::Shr);
        let lhs_exp = if op.is_cmp() { None } else { exp };
        let mut a = self.val(env, lhs, lhs_exp, depth + 1)?;
        let rhs_exp = if shift { None } else { s_exp_of(a).or(lhs_exp) };
        let mut b = self.val(env, rhs, rhs_exp, depth + 1)?;
        if shift {
            if let SVal::Int(_, None) = a
                && flex_int_expr(ast, lhs)
            {
                let pin = match lhs_exp {
                    Some(SExp::Int(it)) => it,
                    _ => IntTy::I32,
                };
                a = self.val(env, lhs, Some(SExp::Int(pin)), depth + 1)?;
            }
        } else {
            match (a, b) {
                (SVal::Int(_, Some(it)), SVal::Int(_, None)) => {
                    b = if flex_int_expr(ast, rhs) {
                        self.val(env, rhs, Some(SExp::Int(it)), depth + 1)?
                    } else {
                        s_coerce_int(b, it)?
                    };
                }
                (SVal::Int(_, None), SVal::Int(_, Some(it))) => {
                    a = if flex_int_expr(ast, lhs) {
                        self.val(env, lhs, Some(SExp::Int(it)), depth + 1)?
                    } else {
                        s_coerce_int(a, it)?
                    };
                }
                (SVal::Int(_, None), SVal::Int(_, None)) if op.is_cmp() && flex_int_expr(ast, lhs) && flex_int_expr(ast, rhs) => {
                    a = self.val(env, lhs, Some(SExp::Int(IntTy::I32)), depth + 1)?;
                    b = self.val(env, rhs, Some(SExp::Int(IntTy::I32)), depth + 1)?;
                }
                _ => {}
            }
        }
        s_combine(op, a, b)
    }

    /// An `as` cast between scalar types, with the main
    /// evaluator's literal rules and saturating float-to-int casts.
    fn cast(&mut self, env: &mut SEnv, expr: ast::ExprId, ty: ast::TyId, depth: u32) -> Option<SVal> {
        use crate::ty::consteval::{flex_int_expr, float_to_int, is_unsuffixed_lit, truncate_to_int as trunc};
        let ast = self.ast;
        let target = sexp_of_written(ast, self.interner, self.known, ty)?;
        let src_exp = if is_unsuffixed_lit(ast, expr) {
            Some(target)
        } else if flex_int_expr(ast, expr) {
            Some(SExp::Int(IntTy::I32))
        } else {
            None
        };
        let v = match (ast.expr(expr), target) {
            (ast::Expr::Path(segs), SExp::Int(_)) if segs.len() >= 2 => match unit_variant_discr(ast, self.interner, self.known, segs) {
                Some(v) => v,
                None => self.val(env, expr, src_exp, depth + 1)?,
            },
            _ => self.val(env, expr, src_exp, depth + 1)?,
        };
        match (v, target) {
            (SVal::Int(x, _), SExp::Int(it)) => Some(SVal::Int(trunc(x, it), Some(it))),
            (SVal::Bool(b), SExp::Int(it)) => Some(SVal::Int(b as i128, Some(it))),
            (SVal::Char(c), SExp::Int(it)) => Some(SVal::Int(trunc(c as i128, it), Some(it))),
            (SVal::Float(bits, f), SExp::Int(it)) => Some(SVal::Int(float_to_int(bits, f, it), Some(it))),
            (SVal::Int(x, t), SExp::Float(g)) => {
                let unsigned = matches!(t, Some(it) if !it.signed());
                let bits = match (g, unsigned) {
                    (FloatTy::F64, true) => ((x as u128) as f64).to_bits(),
                    (FloatTy::F32, true) => ((x as u128) as f32).to_bits() as u64,
                    (FloatTy::F64, false) => (x as f64).to_bits(),
                    (FloatTy::F32, false) => (x as f32).to_bits() as u64,
                };
                Some(SVal::Float(bits, g))
            }
            (SVal::Float(bits, f), SExp::Float(g)) => {
                let r = match (f, g) {
                    (FloatTy::F64, FloatTy::F32) => (f64::from_bits(bits) as f32).to_bits() as u64,
                    (FloatTy::F32, FloatTy::F64) => (f32::from_bits(bits as u32) as f64).to_bits(),
                    _ => bits,
                };
                Some(SVal::Float(r, g))
            }
            (SVal::Int(x, _), SExp::Char) => u32::try_from(x).ok().and_then(char::from_u32).map(SVal::Char),
            _ => None,
        }
    }

    /// A call to a const fn found by scanning the crate's items.
    /// Argument types come from the written parameter types, so
    /// the body's arithmetic runs in its real types; a `&mut`
    /// param aliases the caller's slot, so writes through it land.
    fn call(&mut self, env: &mut SEnv, callee: ast::ExprId, args: &[ast::ExprId], depth: u32) -> Option<SVal> {
        use crate::ast::Expr;
        let ast = self.ast;
        let Expr::Path(segs) = ast.expr(callee) else { return None };
        if segs.len() != 1 {
            return None;
        }
        let name = segs[0].name;
        crate::check::scalar_reads::note(name);
        // A body item of this name in scope at the call wins: a
        // nested const fn is the only candidate, and another body
        // item shadows every module fn.
        let mut defs: Vec<&ast::FnDef> = Vec::new();
        match stmt_item_at(ast, name, ast.expr_span(callee)) {
            Some(si) => match &ast.stmt_items[si as usize].kind {
                ItemKind::Fn(d) => defs.push(d),
                _ => return None,
            },
            None => {
                for item in &ast.items {
                    if let ItemKind::Fn(def) = &item.kind {
                        defs.push(def);
                    }
                }
            }
        }
        for def in defs {
            if def.name != name || !def.qual.is_const || def.self_param.is_some() || def.params.len() != args.len() {
                continue;
            }
            let body = def.body?;
            let mut fenv: SEnv = Vec::new();
            for (param, &arg) in def.params.iter().zip(args) {
                if param.pat.is_some() {
                    return None;
                }
                let is_ref = matches!(ast.ty(param.ty), TyExpr::Ref { .. });
                let pexp = if is_ref { None } else { sexp_of_written(ast, self.interner, self.known, param.ty) };
                let mut v = self.val(env, arg, pexp, depth + 1)?;
                if is_ref && !matches!(v, SVal::Ref(_)) {
                    return None;
                }
                if let Some(SExp::Int(it)) = pexp {
                    v = s_coerce_int(v, it)?;
                }
                if let (Some(SExp::Float(f)), SVal::Float(_, g)) = (pexp, v)
                    && f != g
                {
                    return None;
                }
                let slot = self.new_slot(v);
                fenv.push((param.name, slot));
            }
            let rexp = def.ret.and_then(|t| sexp_of_written(ast, self.interner, self.known, t));
            self.fn_depth += 1;
            let outer_self = self.self_name.take();
            let flow = self.block(&mut fenv, body, rexp, depth + 1);
            self.self_name = outer_self;
            self.fn_depth -= 1;
            let out = match flow {
                SFlow::V(v) | SFlow::Return(v) => v?,
                _ => return None,
            };
            return match rexp {
                Some(SExp::Int(it)) => s_coerce_int(out, it),
                _ => Some(out),
            };
        }
        None
    }

    /// The value of a one- or two-segment path: a local, a const
    /// item (free or inherent associated), or an integer type's
    /// `MAX`/`MIN`/`BITS`. A const item's value takes its declared
    /// type, never the use site's expected type.
    fn path_value(&mut self, env: &mut SEnv, segs: &[ast::PathSeg], e: ast::ExprId, exp: Option<SExp>, depth: u32) -> Option<SVal> {
        let ast = self.ast;
        if segs.len() == 1 {
            let name = segs[0].name;
            if let Some((_, slot)) = env.iter().rev().find(|(n, _)| *n == name) {
                let v = self.slots[*slot];
                return match (v, exp) {
                    (SVal::Int(_, None), Some(SExp::Int(it))) => s_coerce_int(v, it),
                    _ => Some(v),
                };
            }
            let (target, decl) = match stmt_item_at(ast, name, ast.expr_span(e)) {
                Some(si) => match &ast.stmt_items[si as usize].kind {
                    ItemKind::Const(c) => (c.value, Some(c.ty)),
                    _ => (None, None),
                },
                None => (named_const_expr(ast, &[name]), named_const_decl(ast, &[name])),
            };
            let v = self.const_body(target?, decl, None, depth)?;
            return Some(v);
        }
        if segs.len() == 2 {
            // `Self::X` reads the const of the type `Self` names here.
            let ty_name = if segs[0].name == self.known.self_type { self.self_name? } else { segs[0].name };
            if !segs[0].has_args()
                && let Some(v) = named_const_expr(ast, &[ty_name, segs[1].name])
            {
                let decl = named_const_decl(ast, &[ty_name, segs[1].name]);
                return self.const_body(v, decl, Some(ty_name), depth);
            }
            let it = self.known.int_ty(segs[0].name)?;
            let bits = it.bits();
            return Some(match self.interner.get(segs[1].name) {
                "MAX" => SVal::Int(crate::ty::consteval::int_stored(it.max128(), it), Some(it)),
                "MIN" => SVal::Int(if it.signed() { if bits >= 128 { i128::MIN } else { -(1i128 << (bits - 1)) } } else { 0 }, Some(it)),
                "BITS" => SVal::Int(bits as i128, Some(IntTy::U32)),
                _ => return None,
            });
        }
        None
    }

    /// Evaluates one expression.
    fn expr(&mut self, env: &mut SEnv, e: ast::ExprId, exp: Option<SExp>, depth: u32) -> SFlow {
        use crate::ast::{BinOp, Expr, MacroKind, UnOp};
        use crate::ty::consteval::{int_neg, int_not, int_stored};
        self.steps += 1;
        if depth > 32 || self.steps > 100_000 {
            return SFlow::V(None);
        }
        let ast = self.ast;
        match ast.expr(e) {
            Expr::Int(mag, suffix) => {
                let mag = *mag;
                let v = match (suffix, exp) {
                    (Some(s), _) => {
                        let it = IntTy::from_suffix(*s);
                        if mag > it.max128() { None } else { Some(SVal::Int(int_stored(mag, it), Some(it))) }
                    }
                    (None, Some(SExp::Int(it))) => {
                        if mag > it.max128() { None } else { Some(SVal::Int(int_stored(mag, it), Some(it))) }
                    }
                    (None, _) => {
                        if mag > i128::MAX as u128 {
                            Some(SVal::Int(mag as i128, Some(IntTy::U128)))
                        } else {
                            Some(SVal::Int(mag as i128, None))
                        }
                    }
                };
                SFlow::V(v)
            }
            Expr::Float(i) => {
                let text = ast.strings[*i as usize].as_str();
                let v = parse_float_literal(text).and_then(|(_, suffix)| {
                    let f = suffix.unwrap_or(match exp {
                        Some(SExp::Float(f)) => f,
                        _ => FloatTy::F64,
                    });
                    if float_literal_overflows(text, f) {
                        return None;
                    }
                    parse_float_literal_as(text, f).map(|bits| SVal::Float(bits, f))
                });
                SFlow::V(v)
            }
            Expr::Char(c) => SFlow::V(Some(SVal::Char(*c))),
            Expr::Bool(b) => SFlow::V(Some(SVal::Bool(*b))),
            Expr::Unit => SFlow::V(Some(SVal::Unit)),
            Expr::Unary { op, expr } => {
                // A negated literal names its magnitude first: the
                // magnitude may exceed the type's max by one (its min).
                if *op == UnOp::Neg
                    && let Expr::Int(mag, suffix) = ast.expr(*expr)
                {
                    let (mag, suffix) = (*mag, *suffix);
                    let it = suffix.map(IntTy::from_suffix).or(match exp {
                        Some(SExp::Int(it)) => Some(it),
                        _ => None,
                    });
                    let v = match it {
                        Some(it) => {
                            if !it.signed() || mag > it.max128() + 1 {
                                None
                            } else {
                                Some(SVal::Int((mag as i128).wrapping_neg(), Some(it)))
                            }
                        }
                        None => {
                            if mag <= (1u128 << 127) { Some(SVal::Int((mag as i128).wrapping_neg(), None)) } else { None }
                        }
                    };
                    return SFlow::V(v);
                }
                let Some(v) = self.val(env, *expr, exp, depth + 1) else { return SFlow::V(None) };
                SFlow::V(match (op, v) {
                    (UnOp::Neg, SVal::Int(x, Some(it))) => int_neg(it, x).map(|r| SVal::Int(r, Some(it))),
                    (UnOp::Neg, SVal::Int(x, None)) => x.checked_neg().map(|r| SVal::Int(r, None)),
                    (UnOp::Neg, SVal::Float(bits, f)) => Some(SVal::Float(
                        match f {
                            FloatTy::F64 => (-f64::from_bits(bits)).to_bits(),
                            FloatTy::F32 => (-f32::from_bits(bits as u32)).to_bits() as u64,
                        },
                        f,
                    )),
                    (UnOp::Not, SVal::Int(x, Some(it))) => Some(SVal::Int(int_not(it, x), Some(it))),
                    (UnOp::Not, SVal::Int(x, None)) => Some(SVal::Int(!x, None)),
                    (UnOp::Not, SVal::Bool(b)) => Some(SVal::Bool(!b)),
                    _ => None,
                })
            }
            Expr::Binary { op, lhs, rhs } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let Some(a) = self.bool_val(env, *lhs, depth + 1) else { return SFlow::V(None) };
                    if (*op == BinOp::And && !a) || (*op == BinOp::Or && a) {
                        return SFlow::V(Some(SVal::Bool(a)));
                    }
                    return SFlow::V(self.bool_val(env, *rhs, depth + 1).map(SVal::Bool));
                }
                SFlow::V(self.binary(env, *op, *lhs, *rhs, exp, depth))
            }
            Expr::Cast { expr, ty } => SFlow::V(self.cast(env, *expr, *ty, depth)),
            Expr::Block(b) | Expr::ConstBlock(b) => self.block(env, *b, exp, depth + 1),
            Expr::LabeledBlock { label, body } => match self.block(env, *body, exp, depth + 1) {
                SFlow::Break(l, v) if l.is_none() || l == Some(label.name) => SFlow::V(v),
                other => other,
            },
            Expr::If { cond, then, els } => {
                let Some(c) = self.bool_val(env, *cond, depth + 1) else { return SFlow::V(None) };
                if c {
                    self.block(env, *then, exp, depth + 1)
                } else {
                    match els {
                        Some(els) => self.expr(env, *els, exp, depth + 1),
                        None => SFlow::V(Some(SVal::Unit)),
                    }
                }
            }
            Expr::While { label, cond, body } => loop {
                let Some(c) = self.bool_val(env, *cond, depth + 1) else { return SFlow::V(None) };
                if !c {
                    return SFlow::V(Some(SVal::Unit));
                }
                match self.block(env, *body, None, depth + 1) {
                    SFlow::V(Some(_)) => {}
                    SFlow::Continue(l) if s_label_matches(l, label) => {}
                    SFlow::Break(l, _) if s_label_matches(l, label) => return SFlow::V(Some(SVal::Unit)),
                    other => return other,
                }
            },
            Expr::Loop { label, body } => loop {
                match self.block(env, *body, None, depth + 1) {
                    SFlow::V(Some(_)) => {}
                    SFlow::Continue(l) if s_label_matches(l, label) => {}
                    SFlow::Break(l, v) if s_label_matches(l, label) => return SFlow::V(v),
                    other => return other,
                }
            },
            Expr::Break { label, value } => {
                let v = match value {
                    Some(v) => self.val(env, *v, None, depth + 1),
                    None => Some(SVal::Unit),
                };
                SFlow::Break(label.map(|l| l.name), v)
            }
            Expr::Continue(label) => SFlow::Continue(label.map(|l| l.name)),
            Expr::Return(v) => {
                let v = match v {
                    Some(v) => self.val(env, *v, exp, depth + 1),
                    None => Some(SVal::Unit),
                };
                SFlow::Return(v)
            }
            Expr::Assign { lhs, rhs } => {
                let Some(slot) = self.place_slot(env, *lhs, depth + 1).filter(|s| !self.fixed.contains(s)) else { return SFlow::V(None) };
                let exp = s_exp_of(self.slots[slot]);
                let Some(v) = self.val(env, *rhs, exp, depth + 1) else { return SFlow::V(None) };
                SFlow::V(if self.store(slot, v) { Some(SVal::Unit) } else { None })
            }
            Expr::AssignOp { op, lhs, rhs } => {
                let Some(slot) = self.place_slot(env, *lhs, depth + 1).filter(|s| !self.fixed.contains(s)) else { return SFlow::V(None) };
                let cur = self.slots[slot];
                let shift = matches!(op, BinOp::Shl | BinOp::Shr);
                let rexp = if shift { None } else { s_exp_of(cur) };
                let Some(mut b) = self.val(env, *rhs, rexp, depth + 1) else { return SFlow::V(None) };
                if !shift
                    && let (SVal::Int(_, Some(it)), SVal::Int(_, None)) = (cur, b)
                {
                    let Some(c) = s_coerce_int(b, it) else { return SFlow::V(None) };
                    b = c;
                }
                let Some(r) = s_combine(*op, cur, b) else { return SFlow::V(None) };
                SFlow::V(if self.store(slot, r) { Some(SVal::Unit) } else { None })
            }
            Expr::Ref { .. } if self.fn_depth == 0 => SFlow::V(None),
            Expr::Ref { expr, .. } => SFlow::V(self.place_slot(env, *expr, depth + 1).map(SVal::Ref)),
            Expr::Deref(inner) => SFlow::V(match self.val(env, *inner, None, depth + 1) {
                Some(SVal::Ref(s)) => Some(self.slots[s]),
                _ => None,
            }),
            Expr::Call { callee, args } => SFlow::V(self.call(env, *callee, args, depth)),
            // `"lit".len()` (`str::len` is a const fn): the literal's
            // byte length, a `usize`.
            Expr::MethodCall { recv, name, generic_args, args, .. }
                if args.is_empty() && generic_args.is_empty() && self.interner.get(*name) == "len" =>
            {
                SFlow::V(match self.ast.expr(*recv) {
                    Expr::Str(i) => self.ast.strings.get(*i as usize).map(|s| SVal::Int(s.len() as i128, Some(IntTy::Usize))),
                    _ => None,
                })
            }
            Expr::Path(segs) => SFlow::V(self.path_value(env, segs, e, exp, depth)),
            Expr::Macro { kind, args, .. } => SFlow::V(match kind {
                // A panic in const evaluation is a const failure,
                // never a skipped statement.
                MacroKind::Assert => match args.first().and_then(|a| self.bool_val(env, *a, depth + 1)) {
                    Some(true) => Some(SVal::Unit),
                    _ => None,
                },
                _ => None,
            }),
            _ => SFlow::V(None),
        }
    }

    /// Evaluates a const item's initializer in a fresh env and
    /// pins the result to its declared type when that is scalar.
    fn const_body(&mut self, body: ast::ExprId, decl: Option<ast::TyId>, owner: Option<Sym>, depth: u32) -> Option<SVal> {
        let dexp = decl.and_then(|t| sexp_of_written(self.ast, self.interner, self.known, t));
        let mut empty: SEnv = Vec::new();
        // A const item's initializer is borrow-checked as its own body
        // (check/const_body.rs), like a const fn's. `Self` in it is its
        // owning type (`owner`), or nothing for a free const.
        self.fn_depth += 1;
        let outer_self = std::mem::replace(&mut self.self_name, owner);
        let v = self.val(&mut empty, body, dexp, depth + 1);
        self.self_name = outer_self;
        self.fn_depth -= 1;
        let v = v?;
        match dexp {
            Some(SExp::Int(it)) => s_coerce_int(v, it),
            Some(SExp::Bool) => match v {
                SVal::Bool(_) => Some(v),
                SVal::Int(x, _) => Some(SVal::Bool(x != 0)),
                _ => None,
            },
            Some(SExp::Char) => match v {
                SVal::Char(_) => Some(v),
                SVal::Int(x, _) => u32::try_from(x).ok().and_then(char::from_u32).map(SVal::Char),
                _ => None,
            },
            _ => Some(v),
        }
    }
}

/// True when a break/continue with `flow` label targets a loop
/// with `loop_label` (an unlabeled jump targets the innermost loop).
fn s_label_matches(flow: Option<Sym>, loop_label: &Option<crate::ast::Label>) -> bool {
    match (flow, loop_label) {
        (None, _) => true,
        (Some(f), Some(l)) => f == l.name,
        (Some(_), None) => false,
    }
}

/// Wraps `v` to the range of an integer type, as an `as` cast does.
fn truncate_to_int(v: i128, it: IntTy) -> i128 {
    let bits = it.bits();
    if bits >= 128 {
        return v;
    }
    let mask: i128 = (1i128 << bits) - 1;
    let u = v & mask;
    if it.signed() && (u >> (bits - 1)) & 1 == 1 { u - (1i128 << bits) } else { u }
}

/// Parses the `repr` attributes of an item into layout terms. Unknown or
/// unsupported forms are errors: this is where the 4.4 grammar cut lands
/// for `repr(simd)`. Conflicting hints are rustc's errors, not a cut:
/// more than one integer hint, `Rust` with `C` or an integer, a second
/// `packed` hint, and `transparent` with any other hint all reject.
/// `packed(N)` combines with `C` or `Rust` in either order and caps
/// field alignments at N (a power of two up to 2^29, as in rustc);
/// several `align` hints take the largest.
fn parse_repr(interner: &Interner, attrs: &[ast::Attr]) -> Result<crate::ty::layout::Repr, Diag> {
    use crate::ast::AttrInput;
    use crate::lexer::{Tok, TokenTree};
    use crate::ty::layout::{Repr, ReprKind};
    let mut repr = Repr::default();
    let mut first_span = None;
    let (mut seen_c, mut seen_rust, mut seen_transparent, mut seen_packed, mut seen_int) = (false, false, false, false, false);
    let mut hints = 0u32;
    for attr in attrs {
        if attr.path.len() != 1 || interner.get(attr.path[0].name) != "repr" {
            continue;
        }
        if first_span.is_none() {
            first_span = Some(attr.span);
        }
        let AttrInput::List(_, trees) = &attr.input else {
            return Err(Diag::new("L0110", attr.span, "`repr` needs a form such as `repr(C)`", "Write the repr in parentheses."));
        };
        let bad = |what: &str| {
            Diag::new(
                "L0652",
                attr.span,
                format!("unsupported layout form: `repr({what})`"),
                "Use a supported repr: C, an integer, transparent, packed, or align.",
            )
        };
        let conflict = |what: &str| Diag::new("L0110", attr.span, what.to_string(), "Remove one of the conflicting hints.");
        let mut i = 0;
        while i < trees.len() {
            match &trees[i] {
                TokenTree::Token(t) if t.tok == Tok::Comma => i += 1,
                TokenTree::Token(t) => {
                    let Tok::Ident(name) = t.tok else {
                        return Err(bad(".."));
                    };
                    let text = interner.get(name);
                    hints += 1;
                    match text {
                        "C" => seen_c = true,
                        "Rust" => seen_rust = true,
                        "transparent" => seen_transparent = true,
                        "packed" => {
                            if seen_packed {
                                return Err(conflict("type has conflicting packed representation hints"));
                            }
                            seen_packed = true;
                            repr.pack = Some(1);
                            if let Some(TokenTree::Delim(_, inner, _)) = trees.get(i + 1) {
                                let invalid = Diag::new(
                                    "L0110",
                                    attr.span,
                                    "invalid alignment value in `repr(packed(N))`",
                                    "Use a power of two up to 2^29, such as `packed(2)` or `packed(4)`.",
                                );
                                match inner.as_slice() {
                                    [TokenTree::Token(t2)] => match t2.tok {
                                        Tok::Int(n, None, _) if n > 0 && n <= (1 << 29) && n.is_power_of_two() => {
                                            repr.pack = Some(n as u64);
                                            i += 1;
                                        }
                                        _ => return Err(invalid),
                                    },
                                    _ => return Err(invalid),
                                }
                            }
                        }
                        "align" => match trees.get(i + 1) {
                            Some(TokenTree::Delim(_, inner, _)) => match inner.as_slice() {
                                [TokenTree::Token(t2)] => match t2.tok {
                                    Tok::Int(n, None, _) if n > 0 && n <= (1 << 29) && n.is_power_of_two() => {
                                        repr.align = Some(repr.align.unwrap_or(0).max(n as u64));
                                        i += 1;
                                    }
                                    _ => return Err(bad("align(..)")),
                                },
                                _ => return Err(bad("align(..)")),
                            },
                            _ => return Err(bad("align")),
                        },
                        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => {
                            if seen_int {
                                return Err(conflict("conflicting representation hints"));
                            }
                            seen_int = true;
                            let signed = text.starts_with('i');
                            let bits: u32 = match text {
                                "u8" | "i8" => 8,
                                "u16" | "i16" => 16,
                                "u32" | "i32" => 32,
                                "u128" | "i128" => 128,
                                _ => 64,
                            };
                            repr.int = Some((bits, signed));
                            repr.word = matches!(text, "isize" | "usize");
                        }
                        _ => return Err(bad(text)),
                    }
                    i += 1;
                }
                TokenTree::Delim(..) => return Err(bad("..")),
            }
        }
    }
    if seen_transparent && hints > 1 {
        return Err(Diag::new(
            "L0110",
            first_span.unwrap_or_default(),
            "a transparent type cannot have other repr hints",
            "Remove every other hint, or remove `transparent`.",
        ));
    }
    if seen_rust && (seen_c || seen_int) {
        return Err(Diag::new(
            "L0110",
            first_span.unwrap_or_default(),
            "conflicting representation hints",
            "Remove one of the conflicting hints.",
        ));
    }
    repr.kind = if seen_transparent {
        ReprKind::Transparent
    } else if seen_c {
        ReprKind::C
    } else {
        ReprKind::Default
    };
    Ok(repr)
}

/// Converts written variant or struct fields to (shape, (name, type, span) list).
/// Tuple fields get their index text as their name, so `T { 0: 5 }` field
/// matching and field diagnostics work like named fields.
fn variant_fields_src(interner: &Interner, vf: &VariantFields) -> (VariantShape, Vec<(Sym, ast::TyId, Span)>) {
    match vf {
        VariantFields::Unit => (VariantShape::Unit, Vec::new()),
        VariantFields::Tuple(fs) => {
            (VariantShape::Tuple, fs.iter().enumerate().map(|(i, f)| (interner.intern(&i.to_string()), f.ty, f.span)).collect())
        }
        VariantFields::Struct(fs) => (VariantShape::Named, fs.iter().map(|f| (f.name, f.ty, f.span)).collect()),
    }
}

/// Session-stable numbering state for Adts and statics: every def
/// hash ever numbered in this session, in first-seen order (hashes
/// first seen in the same compile append in hash order, so a fresh
/// session numbers by `DefPathHash` alone, independent of file and
/// item order). A def's session-local index is its rank among the
/// defs present in this compile: adding a def appends it after
/// every def seen before, so present defs keep their indices.
/// Cached output carries those indices (`CTy::Adt(AdtId)` in glue
/// and type sets, static indices in IR), so index stability is
/// what lets an edit that adds a file reuse the rest. The driver
/// session owns one `DefOrder` across requests.
#[derive(Default)]
pub struct DefOrder {
    pub adts: Vec<crate::def::DefPathHash>,
    pub statics: Vec<crate::def::DefPathHash>,
    /// Adt instance ids (see `InstOrder`).
    pub instances: InstOrder,
}

/// Session-stable Adt instance ids (W-PERF): an instance Adt's id is
/// `base` plus the place of its instance key (`Program::instance_key_hash`
/// of template def and args) in first-seen order. Places are never
/// reused, so a new instance never moves the id of another: C names Adts
/// `A<id>`, and cached C, type sets, and glue keep their meaning. Rows
/// from the template Adts up to `base`, and the places of instances a
/// program does not make, are inert placeholder rows. `base` sits past
/// the template Adts with room for more; it moves (and every instance id
/// with it) only when templates outgrow it.
#[derive(Default, Clone)]
pub struct InstOrder {
    pub base: u32,
    pub keys: Vec<u64>,
    pub place: HashMap<u64, u32>,
}

impl InstOrder {
    /// Fits `base` past `templates` Adt rows (before a collect's first instance).
    pub fn fit(&mut self, templates: usize) {
        if (self.base as usize) < templates {
            self.base = (templates + 32 + templates / 8) as u32;
        }
    }

    /// The place of `key`, appended when it is new.
    pub fn place_of(&mut self, key: u64) -> u32 {
        let keys = &mut self.keys;
        *self.place.entry(key).or_insert_with(|| {
            keys.push(key);
            keys.len() as u32 - 1
        })
    }
}

/// The rank of each hash among `hashes`: its place in `order` after
/// appending unseen hashes in hash order. Places are never reused: a def
/// that goes leaves a hole (an inert placeholder row of the Adt or static
/// table) and moves no other def's index, so the caches that name defs by
/// index stay valid. The driver restarts an order whose holes pile up
/// (compile.rs `def_order_sparse`). `hashes` must hold each def once.
pub(crate) fn stable_ranks(order: &mut Vec<crate::def::DefPathHash>, hashes: &[crate::def::DefPathHash]) -> Vec<u32> {
    let pos: std::collections::HashMap<crate::def::DefPathHash, usize> = order.iter().enumerate().map(|(i, h)| (*h, i)).collect();
    let mut unseen: Vec<crate::def::DefPathHash> = hashes.iter().copied().filter(|h| !pos.contains_key(h)).collect();
    unseen.sort_by_key(|h| h.0);
    unseen.dedup();
    order.extend(unseen);
    stable_ranks_preview(order, hashes)
}

/// The ranks `stable_ranks` would assign, without mutating `order`
/// (the driver predicts this compile's indices before collect runs
/// to decide whether cached output still applies).
pub(crate) fn stable_ranks_preview(order: &[crate::def::DefPathHash], hashes: &[crate::def::DefPathHash]) -> Vec<u32> {
    let mut pos: std::collections::HashMap<crate::def::DefPathHash, usize> = order.iter().enumerate().map(|(i, h)| (*h, i)).collect();
    let mut unseen: Vec<crate::def::DefPathHash> = hashes.iter().copied().filter(|h| !pos.contains_key(h)).collect();
    unseen.sort_by_key(|h| h.0);
    unseen.dedup();
    for (k, h) in unseen.into_iter().enumerate() {
        pos.insert(h, order.len() + k);
    }
    hashes.iter().map(|h| pos[h] as u32).collect()
}

/// Builds the program tables. `crate_ids` names the crate for stable def
/// identity (docs/ARCH.md section 1.7). Returns all collection errors.
pub fn collect(ast: Arc<Ast>, interner: Arc<Interner>, known: Arc<Known>, crate_ids: CrateIds) -> Result<Program, Vec<Diag>> {
    collect_mode(ast, interner, known, crate_ids, true)
}

/// `collect` for one command: a build walks every reachable generic
/// instance (mono); `lust check` passes `walk_instances` false and
/// stops at the template checks, like rustc check (ARCH 4.1).
pub fn collect_mode(
    ast: Arc<Ast>,
    interner: Arc<Interner>,
    known: Arc<Known>,
    crate_ids: CrateIds,
    walk_instances: bool,
) -> Result<Program, Vec<Diag>> {
    collect_with_order(ast, interner, known, crate_ids, walk_instances, &mut DefOrder::default())
}

/// `collect_mode` with a caller-kept `DefOrder` (the driver session
/// holds one across requests, so Adt and static indices survive
/// edits that add or reorder defs).
pub fn collect_with_order(
    ast: Arc<Ast>,
    interner: Arc<Interner>,
    known: Arc<Known>,
    crate_ids: CrateIds,
    walk_instances: bool,
    def_order: &mut DefOrder,
) -> Result<Program, Vec<Diag>> {
    collect_with_order_resolved(ast, interner, known, crate_ids, walk_instances, def_order, None)
}

/// `collect_with_order` with the resolver's module tables (section
/// 1.3) attached from the start. The tables must be in place before
/// collection's body checks run (mono's template checks in
/// `finish_collect`): every checked body reads its defining module
/// and the visibility tables from `Program::resolved`.
#[allow(clippy::too_many_arguments)]
pub fn collect_with_order_resolved(
    ast: Arc<Ast>,
    interner: Arc<Interner>,
    known: Arc<Known>,
    crate_ids: CrateIds,
    walk_instances: bool,
    def_order: &mut DefOrder,
    resolved: Option<std::sync::Arc<crate::resolve::imports::ResolvedTables>>,
) -> Result<Program, Vec<Diag>> {
    collect_reuse(ast, interner, known, crate_ids, walk_instances, def_order, resolved, &mut Default::default(), &[])
}

/// `collect_with_order_resolved` with the driver session's generic
/// reuse cache (driver/gen_reuse.rs, W-PERF): the cache rides in
/// `GenTables::reuse` for this collect, so template checks and instance
/// discovery skip bodies whose stored results still hold, and it goes
/// back to the caller on every return, with what this collect stored.
#[allow(clippy::too_many_arguments)]
pub fn collect_reuse(
    ast: Arc<Ast>,
    interner: Arc<Interner>,
    known: Arc<Known>,
    crate_ids: CrateIds,
    walk_instances: bool,
    def_order: &mut DefOrder,
    resolved: Option<std::sync::Arc<crate::resolve::imports::ResolvedTables>>,
    reuse: &mut crate::driver::gen_reuse::GenReuse,
    std_routes: &[&'static crate::ty::lang::StdFnDef],
) -> Result<Program, Vec<Diag>> {
    let mut p = Program {
        ast: ast.clone(),
        interner: interner.clone(),
        known: known.clone(),
        tys: reuse.type_pool.take().and_then(|pool| pool.take(crate_ids.local, crate_ids.stable, &interner)).unwrap_or_else(GlobalTys::new),
        defs: Defs::new(),
        adts: Vec::new(),
        fns: Vec::new(),
        consts: Vec::new(),
        assoc_const_names: HashMap::default(),
        impl_assoc_consts: HashMap::default(),
        trait_assoc_const_values: HashMap::default(),
        const_names: HashMap::default(),
        const_defs: Vec::new(),
        static_defs: Vec::new(),
        statics: Vec::new(),
        static_names: HashMap::default(),
        body_items: HashMap::default(),
        body_type_items: HashMap::default(),
        item_blocks: Vec::new(),
        body_consts: Vec::new(),
        const_values: Vec::new(),
        scope_of_stmt: HashMap::default(),
        fn_siblings: HashMap::default(),
        const_body_reads: Vec::new(),
        const_sites_shape: Default::default(),
        type_names: crate::driver::read_cover::NameMap::new("type"),
        drop_adts: Default::default(),
        aliases: crate::driver::read_cover::NameMap::new("alias"),
        fn_names: HashMap::default(),
        methods: Default::default(),
        main: None,
        krate: crate_ids.local,
        stable_crate: crate_ids.stable,
        fn_defs: Vec::new(),
        def_by_hash: Default::default(),
        std_fns: Vec::new(),
        shipped_ranges: ast.shipped.clone(),
        resolved,
        gx: GenTables::default(),
        record_reads: true,
        unwind: false,
        overflow_checks: true,
        debug_assertions: true,
        unwind_tables: false,
        regions: Default::default(),
        fp_memo: Default::default(),
        const_goals: Vec::new(),
        impl_parents: HashMap::default(),
        reuse_stamp: 0,
        verified: 0,
        outer_params: Default::default(),
    };
    let (ast, interner, known) = (&*ast, &*interner, &*known);
    p.gx.walk_instances = walk_instances;
    p.gx.reuse = std::mem::take(reuse);
    p.record_reads = !p.gx.reuse.one_shot;
    p.gx.inst_order = std::mem::take(&mut def_order.instances);
    p.gx.reuse.begin();
    // The builtin Adt defs join the table with their one type param where they take
    // one, so `Generics` queries answer for builtins and user types alike. Their
    // code paths stay builtin until source std lands (ARCH 2.1, 9.3).
    {
        let b = *p.tys.builtins();
        let dummy = Span::default();
        let one_ty_param = || Generics {
            parent: None,
            params: vec![ParamDef { name: interner.intern("T"), kind: ParamKind::Type, span: dummy }],
            predicates: Vec::new(),
        };
        let insert_builtin = |p: &mut Program, def: DefId, kind: DefsKind, name: Sym, generics: Generics| {
            let mut data = DefData::skeleton(kind, None, name, Vis::Pub, dummy);
            data.generics = generics;
            p.defs.insert(def, data);
        };
        insert_builtin(&mut p, b.string, DefsKind::Struct, known.String, Generics::default());
        insert_builtin(&mut p, b.vec, DefsKind::Struct, known.Vec, one_ty_param());
        insert_builtin(&mut p, b.boxed, DefsKind::Struct, known.Box, one_ty_param());
        insert_builtin(&mut p, b.rc, DefsKind::Struct, known.Rc, one_ty_param());
        insert_builtin(&mut p, b.option, DefsKind::Enum, known.Option, one_ty_param());
        let two_ty_params = Generics {
            parent: None,
            params: vec![
                ParamDef { name: interner.intern("T"), kind: ParamKind::Type, span: dummy },
                ParamDef { name: interner.intern("E"), kind: ParamKind::Type, span: dummy },
            ],
            predicates: Vec::new(),
        };
        insert_builtin(&mut p, b.result, DefsKind::Enum, known.Result, two_ty_params);
        insert_builtin(&mut p, b.chars, DefsKind::Struct, interner.intern("Chars"), Generics::default());
        for k in crate::ty::fmt::FmtTy::ALL {
            insert_builtin(&mut p, b.fmt_tys[k as usize], DefsKind::Struct, interner.intern(k.name()), Generics::default());
        }
        insert_builtin(&mut p, b.exit_code, DefsKind::Struct, interner.intern("ExitCode"), Generics::default());
        for k in crate::ty::sync::SyncTy::ALL {
            let params = (0..k.arity()).map(|i| ParamDef { name: interner.intern(["T", "U"][i]), kind: ParamKind::Type, span: dummy }).collect();
            let g = Generics { parent: None, params, predicates: Vec::new() };
            insert_builtin(&mut p, b.sync_tys[k as usize], DefsKind::Struct, interner.intern(k.name()), g);
        }
        for (def, n) in [(b.weak, "Weak"), (b.refcell, "RefCell"), (b.cell, "Cell"), (b.ref_guard, "Ref"), (b.ref_mut_guard, "RefMut")] {
            insert_builtin(&mut p, def, DefsKind::Struct, interner.intern(n), one_ty_param());
        }
        insert_builtin(&mut p, b.local_key, DefsKind::Struct, interner.intern("LocalKey"), one_ty_param());
        insert_builtin(&mut p, b.access_error, DefsKind::Struct, interner.intern("AccessError"), Generics::default());
        for k in crate::ty::hash::HashTy::ALL {
            let params = (0..k.n_args())
                .map(|i| ParamDef { name: interner.intern(["K", "V", "S"][i.min(2)]), kind: ParamKind::Type, span: dummy })
                .collect();
            let g = Generics { parent: None, params, predicates: Vec::new() };
            insert_builtin(&mut p, b.hash_tys[k as usize], DefsKind::Struct, interner.intern(k.path().1), g);
        }
        for k in crate::ty::iter::IterKind::ALL {
            let params = (0..k.n_args())
                .map(|i| ParamDef { name: interner.intern(["I", "F", "B"][i]), kind: ParamKind::Type, span: dummy })
                .collect();
            let g = Generics { parent: None, params, predicates: Vec::new() };
            insert_builtin(&mut p, b.iters[k as usize], DefsKind::Struct, interner.intern(k.path().1), g);
        }
    }
    let mut errs = Vec::new();
    let dup = |span: Span, n: &str| Diag::new("L0106", span, format!("`{n}` is defined more than once"), "Rename one of the definitions.");
    // Without resolver tables the file has no `use` at all (any `use`
    // routes to the resolver), so a non-prelude std type lust knows by
    // name cannot be in scope (W-STD-A; the resolver checks its paths).
    if p.resolved.is_none() {
        std_names_without_use(ast, interner, &mut errs);
    }

    // Generic pass 0: lang traits and user trait defs, so bounds in
    // every later signature can name them.
    p.gx.item_index = crate::check::defaults::item_index(ast);
    generics::init_lang(&mut p);
    generics::collect_trait_defs(&mut p, &mut errs);
    crate::driver::laps::mark("collect/setup+traits");

    // Type items inside function bodies (R18): their identities are
    // the enclosing fn's hash folded with their name (`def::
    // nested_type_ids`, as nested fns fold), computed now because
    // pass 1 ranks every Adt by hash. `p.item_blocks` also lands
    // here; the nested-fn section below recomputes the same table.
    p.item_blocks = ast
        .blocks
        .iter()
        .filter(|b| b.stmts.iter().any(|s| matches!(s, ast::Stmt::Item(_))))
        .map(|b| (b.span, b.stmts.iter().filter_map(|s| if let ast::Stmt::Item(i) = s { Some(*i) } else { None }).collect()))
        .collect();
    // Without a type item in any body both tables are all `None`; the
    // fn hashes they fold from are skipped (a whole-program pass).
    let (stmt_parent_hashes, stmt_type_hashes): (Vec<Option<crate::def::DefPathHash>>, Vec<Option<crate::def::DefPathHash>>) =
        if crate::def::has_body_type_items(ast) {
            let early_top_hashes = top_fn_hashes(ast, interner, p.stable_crate);
            let nested_fn_hashes: Vec<Option<crate::def::DefPathHash>> = crate::def::nested_fn_ids(
                ast,
                interner,
                |b| early_top_hashes.get(&b).copied(),
                |parent, name, disamb| crate::def::child_hash(*parent, crate::def::DefKind::Fn, name, disamb),
                |seg, name| crate::def::free_fn_hash(p.stable_crate, &[seg], name),
            );
            let parent_hash_of = |parent: crate::def::NestParent| match parent {
                crate::def::NestParent::Top(b) => early_top_hashes.get(&b).copied(),
                crate::def::NestParent::Nested(k) => nested_fn_hashes.get(k as usize).copied().flatten(),
                crate::def::NestParent::Unknown => None,
            };
            let stmt_type_parents = crate::def::nested_type_parents(ast);
            let parents = stmt_type_parents.iter().map(|&(par, _)| parent_hash_of(par)).collect();
            let types = crate::def::nested_type_ids(
                ast,
                interner,
                |par| parent_hash_of(par),
                |parent, kind, name, disamb| crate::def::child_hash(*parent, kind, name, disamb),
                |seg, kind, name| crate::def::item_hash(p.stable_crate, &[seg], kind, name, 0),
            );
            (parents, types)
        } else {
            (vec![None; ast.stmt_items.len()], vec![None; ast.stmt_items.len()])
        };
    let mut adt_id_of_stmt_item: Vec<Option<AdtId>> = vec![None; ast.stmt_items.len()];

    // Pass 1: type names, so fields can refer to types defined later.
    // AdtIds come from the session DefOrder (rank by DefPathHash
    // history), not from item order: compute every Adt's hash and
    // rank up front, then fill the tables at those indices.
    let adt_base = p.adts.len();
    // The `adt_srcs` index tag of a type item inside a function
    // body: its `Ast::stmt_items` index under this base (synthetic
    // std Adts sit at the top of the range instead).
    const STMT_ADT_BASE: usize = 1 << 62;
    let mut adt_srcs: Vec<(usize, Sym, Span, bool, crate::def::DefPathHash)> = Vec::new();
    {
        let mut counts: HashMap<Sym, u32> = HashMap::default();
        for (idx, item) in ast.items.iter().enumerate() {
            let (name, span, is_enum) = match &item.kind {
                ItemKind::Struct(s) => (s.name, s.name_span, false),
                ItemKind::Enum(e) => (e.name, e.name_span, true),
                _ => continue,
            };
            let disamb = *counts.entry(name).and_modify(|c| *c += 1).or_insert(0);
            let hash = crate::def::item_hash(crate_ids.stable, &[], crate::def::DefKind::Adt, interner.get(name), disamb);
            adt_srcs.push((idx, name, span, is_enum, hash));
        }
        // Structs and enums inside function bodies rank with every
        // other Adt by their folded hash (R18).
        for (idx, item) in ast.stmt_items.iter().enumerate() {
            let (name, span, is_enum) = match &item.kind {
                ItemKind::Struct(s) => (s.name, s.name_span, false),
                ItemKind::Enum(e) => (e.name, e.name_span, true),
                _ => continue,
            };
            let hash = stmt_type_hashes[idx].expect("every type item in a body has an identity");
            adt_srcs.push((STMT_ADT_BASE + idx, name, span, is_enum, hash));
        }
    }
    // The synthetic std Adts (`generics::std_adts`): `core::cmp::Ordering`
    // is an ordinary fieldless enum with a builtin def (lang signatures
    // name it; its variants and derives are the plain enum machinery),
    // `fmt::Error` the unit struct of `fmt::Result` (W-STD-B), and
    // `str::Utf8Error` and `string::FromUtf8Error` plain structs
    // (W-STD-A), and `any::TypeId` (W-CHECK-G3, never in the flat name
    // table either). They take part in the same stable ranking as user
    // Adts (the driver predicts the same set: see `predicted_def_ids`). A
    // user type, alias, or trait of the same name shadows one in the
    // flat name table, as with any name here; the Adt itself always
    // exists, so `cmp`, `from_utf8`, and the full std paths still name
    // the std type. `fmt::Error` never enters the flat name table (only
    // `fmt::Error` paths reach it; see `generics::resolve_fmt_path`).
    // Index k marks its source as `usize::MAX - k`.
    let std_taken: Vec<Sym> = generics::std_adts()
        .iter()
        .map(|s| interner.intern(s.1))
        .filter(|&n| {
            adt_srcs.iter().any(|s| s.1 == n)
                || p.gx.trait_names.contains_key(&n)
                || ast.items.iter().any(|it| matches!(&it.kind, ItemKind::Type(a) if a.name == n))
        })
        .collect();
    for (k, name, def, is_enum) in generics::std_adts() {
        adt_srcs.push((usize::MAX - k, interner.intern(name), Span::default(), is_enum, def.hash));
    }
    // A program that uses `Cow` (W-STD-B) instantiates it: mono runs.
    if uses_cow(&mut p, ast, interner) {
        p.gx.has_generics = true;
        p.gx.user_generics = true;
    }
    let adt_ranks = {
        let hashes: Vec<crate::def::DefPathHash> = adt_srcs.iter().map(|s| s.4).collect();
        stable_ranks(&mut def_order.adts, &hashes)
    };
    let mut adt_id_of_item: Vec<Option<AdtId>> = vec![None; ast.items.len()];
    {
        let blank_name = interner.intern("");
        // A row per place up to the last one used: the places of Adts that
        // went are inert placeholder rows (`stable_ranks`).
        let rows = adt_ranks.iter().map(|&r| r as usize + 1).max().unwrap_or(0);
        for _ in 0..rows {
            p.adts.push(AdtDef {
                name: blank_name,
                is_enum: false,
                variants: Vec::new(),
                derives: Derives::default(),
                span: Span::default(),
                repr: crate::ty::layout::Repr::default(),
                non_exhaustive: false,
            });
            p.gx.adt_info.push((DefId::PLACEHOLDER, 0));
            p.gx.adt_generics.push(Generics::default());
            p.gx.adt_envs.push(Vec::new());
            p.gx.adt_scopes.push(generics::Scope::default());
            p.gx.adt_instance_of.push(None);
        }
    }
    // Whether the program uses a generic synthetic std Adt: one Ast
    // pass, read in the loop below.
    let uses_generic_std = generics::ast_uses_generic_std(ast, interner);
    for (src, &rank) in adt_srcs.iter().zip(&adt_ranks) {
        let (item_idx, name, span, is_enum, hash) = (src.0, src.1, src.2, src.3, src.4);
        // `Cow` is entry 9 of `std_adts`.
        if item_idx == usize::MAX - 9 {
            // `enum Cow<'a, B: ?Sized + ToOwned> { Borrowed(&'a B),
            // Owned(<B as ToOwned>::Owned) }`, as std declares it. The
            // instance at a concrete B normalizes the Owned field
            // (`Cow<str>` holds a String). Its traits are lang facts with
            // glue in cgen (generics.rs, cgen/fmt.rs).
            let id = AdtId(adt_base as u32 + rank);
            let def = generics::cow_def();
            let b = p.tys.intern(TyKind::Param(ParamId(0)));
            let rb = p.tys.mk_ref(false, b);
            let owned = {
                let a = p.tys.intern_args(&[GenericArg::Ty(b)]);
                p.tys.intern(TyKind::Projection(AssocId(p.gx.lang.assoc_owned), a))
            };
            let zero = interner.intern("0");
            let variant = |n: &str, ty: Ty, disc: i128| Variant {
                name: interner.intern(n),
                shape: VariantShape::Tuple,
                fields: vec![Field { name: zero, ty, span, derive_bounds: Vec::new() }],
                disc,
                non_exhaustive: false,
            };
            p.adts[id.0 as usize] = AdtDef {
                name,
                is_enum: true,
                variants: vec![variant("Borrowed", rb, 0), variant("Owned", owned, 1)],
                derives: Derives::default(),
                span,
                repr: crate::ty::layout::Repr::default(),
                non_exhaustive: false,
            };
            p.gx.adt_info[id.0 as usize] = (def, 1);
            p.gx.adt_lifetimes.insert(def, vec![interner.intern("a")]);
            let to_owned = {
                let a = p.tys.intern_args(&[GenericArg::Ty(b)]);
                Predicate::Trait(Binder::plain(TraitRef { def: p.gx.lang.to_owned, args: a, bindings: EMPTY_BINDINGS }))
            };
            p.gx.adt_generics[id.0 as usize] = Generics {
                parent: None,
                params: vec![ParamDef { name: interner.intern("B"), kind: ParamKind::Type, span }],
                predicates: vec![to_owned.clone()],
            };
            p.gx.adt_envs[id.0 as usize] = vec![to_owned];
            let regioned = p.tys.intern(TyKind::Ref(Region::Param(0), Mut::Shared, b));
            p.regions.adt_fields.insert(def, vec![vec![regioned], vec![owned]]);
            p.tys.register_adt(def, id);
            let mut data = DefData::skeleton(DefsKind::Enum, None, name, Vis::Pub, span);
            data.generics = p.gx.adt_generics[id.0 as usize].clone();
            p.defs.insert_adt(def, id, data);
            continue;
        }
        if item_idx > usize::MAX - generics::std_adts().len() {
            // A synthetic std Adt (see above): variants, fields, and
            // derives are std's, written here directly (no AST item).
            let id = AdtId(adt_base as u32 + rank);
            let unit = |n: &str, disc: i128| Variant { name: interner.intern(n), shape: VariantShape::Unit, fields: Vec::new(), disc, non_exhaustive: false };
            let field = |n: &str, ty: Ty| Field { name: interner.intern(n), ty, span, derive_bounds: Vec::new() };
            let named = |fields: Vec<Field>| vec![Variant { name, shape: VariantShape::Named, fields, disc: 0, non_exhaustive: false }];
            let (def, variants, derives) = match usize::MAX - item_idx {
                0 => (
                    generics::ordering_def(),
                    vec![unit("Less", -1), unit("Equal", 0), unit("Greater", 1)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, hash: true, partial_ord: true, ord: true, ..Derives::default() },
                ),
                // core: `#[derive(Copy, Clone, Debug, Default, Eq, Hash,
                // Ord, PartialEq, PartialOrd)]`; its Display is the runtime's.
                1 => (
                    generics::fmt_error_def(),
                    vec![Variant { name, shape: VariantShape::Unit, fields: Vec::new(), disc: 0, non_exhaustive: false }],
                    Derives {
                        debug: true,
                        clone: true,
                        copy: true,
                        partial_eq: true,
                        eq: true,
                        hash: true,
                        partial_ord: true,
                        ord: true,
                        default: true,
                        default_variant: None,
                    },
                ),
                2 => {
                    let error_len = p.tys.mk_option(int(IntTy::U8));
                    (
                        generics::utf8_error_def(),
                        named(vec![field("valid_up_to", int(IntTy::Usize)), field("error_len", error_len)]),
                        Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                // core: `#[derive(Copy, Clone, Debug, Eq, PartialEq,
                // Hash)] #[non_exhaustive] pub enum Ordering` (W-THREADS).
                4 => (
                    generics::atomic_ordering_def(),
                    vec![unit("Relaxed", 0), unit("Release", 1), unit("Acquire", 2), unit("AcqRel", 3), unit("SeqCst", 4)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, hash: true, ..Derives::default() },
                ),
                // std: `#[derive(PartialEq, Eq, Clone, Copy, Debug)] pub
                // struct RecvError;` and `pub enum TryRecvError { Empty,
                // Disconnected }` with the same derives (W-THREADS).
                5 => (
                    generics::recv_error_def(),
                    vec![Variant { name, shape: VariantShape::Unit, fields: Vec::new(), disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                6 => (
                    generics::try_recv_error_def(),
                    vec![unit("Empty", 0), unit("Disconnected", 1)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                // std: `#[derive(PartialEq, Eq, Clone, Copy, Debug)] pub
                // enum RecvTimeoutError { Timeout, Disconnected }` (tf-p6).
                29 => (
                    generics::recv_timeout_error_def(),
                    vec![unit("Timeout", 0), unit("Disconnected", 1)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                3 => {
                    let bytes = p.tys.mk_vec(int(IntTy::U8));
                    let error = p.tys.intern(TyKind::Adt(generics::utf8_error_def(), EMPTY_ARGS));
                    (
                        generics::from_utf8_error_def(),
                        named(vec![field("bytes", bytes), field("error", error)]),
                        Derives { debug: true, clone: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                // fx-tryfrom: core's conversion errors. `TryFromIntError`
                // is a tuple struct over `IntErrorKind`; the two char
                // errors are tuple structs over `()`; `Infallible` is
                // the empty enum. Derives are core's.
                13 => {
                    let kind = p.tys.intern(TyKind::Adt(generics::int_error_kind_def(), EMPTY_ARGS));
                    (
                        generics::try_from_int_error_def(),
                        vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", kind)], disc: 0, non_exhaustive: false }],
                        Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                14 => (
                    generics::int_error_kind_def(),
                    vec![unit("Empty", 0), unit("InvalidDigit", 1), unit("PosOverflow", 2), unit("NegOverflow", 3), unit("Zero", 4)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                15 => (
                    generics::char_try_from_error_def(),
                    vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", UNIT)], disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                16 => (
                    generics::try_from_char_error_def(),
                    vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", UNIT)], disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                // tf-str: core's parse errors. The three structs hold
                // a `kind` field (Debug prints it); `ParseBoolError`
                // is a unit struct. Derives are core's (no Copy, no
                // Hash).
                18 => {
                    let kind = p.tys.intern(TyKind::Adt(generics::int_error_kind_def(), EMPTY_ARGS));
                    (
                        generics::parse_int_error_def(),
                        named(vec![field("kind", kind)]),
                        Derives { debug: true, clone: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                19 => {
                    let kind = p.tys.intern(TyKind::Adt(generics::float_error_kind_def(), EMPTY_ARGS));
                    (
                        generics::parse_float_error_def(),
                        named(vec![field("kind", kind)]),
                        Derives { debug: true, clone: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                20 => (
                    generics::float_error_kind_def(),
                    vec![unit("Empty", 0), unit("Invalid", 1)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                21 => (
                    generics::parse_bool_error_def(),
                    vec![Variant { name, shape: VariantShape::Unit, fields: Vec::new(), disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                22 => {
                    let kind = p.tys.intern(TyKind::Adt(generics::char_error_kind_def(), EMPTY_ARGS));
                    (
                        generics::parse_char_error_def(),
                        named(vec![field("kind", kind)]),
                        Derives { debug: true, clone: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                23 => (
                    generics::char_error_kind_def(),
                    vec![unit("EmptyString", 0), unit("TooManyChars", 1)],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                // core: `#[derive(Debug, Copy, Clone)] pub struct
                // TryFromSliceError(())` (tf-slice; no PartialEq).
                24 => (
                    generics::try_from_slice_error_def(),
                    vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", UNIT)], disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, copy: true, ..Derives::default() },
                ),
                17 => (
                    generics::infallible_def(),
                    Vec::new(),
                    Derives {
                        debug: true,
                        clone: true,
                        copy: true,
                        partial_eq: true,
                        eq: true,
                        hash: true,
                        partial_ord: true,
                        ord: true,
                        ..Derives::default()
                    },
                ),
                // `core::any::TypeId` (check/any.rs): two `u64` words of the
                // type hash. Only the derives lust can match (no Debug,
                // Hash, or Ord: their output depends on rustc's hash values).
                7 => (
                    generics::type_id_def(),
                    named(vec![field("lo", int(IntTy::U64)), field("hi", int(IntTy::U64))]),
                    Derives { clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                ),
                // `core::cell::BorrowError` and `BorrowMutError` (W-STD-B):
                // unit structs whose Debug prints the name, as std's does.
                10 => (
                    generics::borrow_error_def(),
                    vec![Variant { name, shape: VariantShape::Unit, fields: Vec::new(), disc: 0, non_exhaustive: false }],
                    Derives { debug: true, ..Derives::default() },
                ),
                11 => (
                    generics::borrow_mut_error_def(),
                    vec![Variant { name, shape: VariantShape::Unit, fields: Vec::new(), disc: 0, non_exhaustive: false }],
                    Derives { debug: true, ..Derives::default() },
                ),
                // `core::panic::Location` (check/caller.rs): its Debug
                // prints these field names. Ord compares file, then
                // line, then column, as std's does. No Hash: a hash
                // value printed by a program would have to match std's
                // field hash exactly, which is not checked here.
                8 => {
                    let file = p.tys.mk_ref(false, STR);
                    (
                        generics::location_def(),
                        named(vec![field("file", file), field("line", int(IntTy::U32)), field("column", int(IntTy::U32))]),
                        Derives {
                            debug: true,
                            clone: true,
                            copy: true,
                            partial_eq: true,
                            eq: true,
                            partial_ord: true,
                            ord: true,
                            ..Derives::default()
                        },
                    )
                }

                // std: `enum TrySendError<T> { Full(T),
                // Disconnected(T) }` (TRIAGE T5). Its Debug hides
                // the value (`TrySendError::Full(..)`) and its
                // Display is std's text: synthetic impls in
                // generics.rs, glue in cgen/fmt.rs.
                26 => {
                    let t = p.tys.intern(TyKind::Param(ParamId(0)));
                    let tuple = |n: &str, disc: i128| Variant { name: interner.intern(n), shape: VariantShape::Tuple, fields: vec![field("0", t)], disc, non_exhaustive: false };
                    (
                        generics::try_send_error_def(),
                        vec![tuple("Full", 0), tuple("Disconnected", 1)],
                        Derives { clone: true, copy: true, partial_eq: true, eq: true, ..Derives::default() },
                    )
                }
                // std: `enum TryLockError<T> { Poisoned(PoisonError<T>),
                // WouldBlock }`, no derives: its Debug and Display are
                // std's texts (synthetic impls in generics.rs, glue in
                // cgen/fmt.rs).
                30 => {
                    let t = p.tys.intern(TyKind::Param(ParamId(0)));
                    let pe = crate::ty::sync::sync_ty(&mut p.tys, crate::ty::sync::SyncTy::PoisonError, &[t]);
                    let poisoned = Variant { name: interner.intern("Poisoned"), shape: VariantShape::Tuple, fields: vec![field("0", pe)], disc: 0, non_exhaustive: false };
                    (generics::try_lock_error_def(), vec![poisoned, unit("WouldBlock", 1)], Derives::default())
                }
                // std: `struct WaitTimeoutResult(bool)` with
                // derived Debug (TRIAGE T5).
                27 => (
                    generics::wait_timeout_result_def(),
                    vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", BOOL)], disc: 0, non_exhaustive: false }],
                    Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, hash: true, ..Derives::default() },
                ),
                // core: `union MaybeUninit<T>` (TRIAGE T5) as a
                // one-field struct: layout drops the niche and
                // drop is suppressed (layout.rs, needs_drop).
                // Clone is unconditional in core, so it is a
                // synthetic impl (generics.rs), not a derive.
                28 => {
                    let t = p.tys.intern(TyKind::Param(ParamId(0)));
                    (
                        generics::maybe_uninit_def(),
                        vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", t)], disc: 0, non_exhaustive: false }],
                        Derives { clone: true, copy: true, ..Derives::default() },
                    )
                }
                // core: `struct NonZero<T>(T)` (TRIAGE T5). Debug
                // and Display forward to the integer (synthetic
                // impls in generics.rs, glue in cgen/fmt.rs), so
                // only the value traits derive here.
                25 => {
                    let t = p.tys.intern(TyKind::Param(ParamId(0)));
                    (
                        generics::nonzero_def(),
                        vec![Variant { name, shape: VariantShape::Tuple, fields: vec![field("0", t)], disc: 0, non_exhaustive: false }],
                        Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, partial_ord: true, ord: true, hash: true, ..Derives::default() },
                    )
                }
                // core: `#[derive(Clone, Copy, Debug, Hash, PartialEq,
                // Eq)] enum Bound<T>`, in this order.
                _ => {
                    let t = p.tys.intern(TyKind::Param(ParamId(0)));
                    let tuple = |n: &str, disc: i128| Variant { name: interner.intern(n), shape: VariantShape::Tuple, fields: vec![field("0", t)], disc, non_exhaustive: false };
                    (
                        generics::bound_def(),
                        vec![tuple("Included", 0), tuple("Excluded", 1), unit("Unbounded", 2)],
                        Derives { debug: true, clone: true, copy: true, partial_eq: true, eq: true, hash: true, ..Derives::default() },
                    )
                }
            };
            // `core::sync::atomic::Ordering` (index 4) and RefCell's
            // `BorrowError` and `BorrowMutError` (10, 11) are std's
            // synthetic `#[non_exhaustive]` types here.
            let synthetic_ne = matches!(usize::MAX - item_idx, 4 | 10 | 11);
            // `cmp::Ordering` (index 0) is `#[repr(i8)]` in core: its
            // derived `Hash` writes an `i8` discriminant.
            let repr = crate::ty::layout::Repr { int: (item_idx == usize::MAX).then_some((8, true)), ..Default::default() };
            p.adts[id.0 as usize] = AdtDef { name, is_enum, variants, derives, span, repr, non_exhaustive: synthetic_ne };
            p.gx.adt_info[id.0 as usize] = (def, 0);
            if matches!(usize::MAX - item_idx, 12 | 25 | 26 | 28 | 30) {
                // `Bound<T>`, `NonZero<T>`, `TrySendError<T>`, `MaybeUninit<T>`, and `TryLockError<T>`:
                // the generic synthetic std Adts. Give each its param and field table, as
                // the field pass does for a user generic type.
                p.gx.adt_info[id.0 as usize] = (def, 1);
                // Their instances need the generic machinery, but only
                // in programs that use them (plain programs keep the
                // fast path; the perf gate measures those). Both
                // flags, like `Cow`: mono's lang-enum fast path
                // (user_generics false) makes no Adt instances.
                if uses_generic_std {
                    p.gx.has_generics = true;
                    p.gx.user_generics = true;
                }
                let t0 = p.tys.intern(TyKind::Param(ParamId(0)));
                let sized_args = p.tys.intern_args(&[GenericArg::Ty(t0)]);
                let sized = Predicate::Trait(Binder::plain(TraitRef { def: p.gx.lang.sized, args: sized_args, bindings: EMPTY_BINDINGS }));
                p.gx.adt_generics[id.0 as usize] = Generics {
                    parent: None,
                    params: vec![ParamDef { name: interner.intern("T"), kind: ParamKind::Type, span }],
                    predicates: vec![sized],
                };
                let env = generics::elaborate_env(&mut p.tys, &p.gx.index, &p.gx.adt_generics[id.0 as usize].predicates.clone());
                p.gx.adt_envs[id.0 as usize] = env;
                let t = p.tys.intern(TyKind::Param(ParamId(0)));
                if usize::MAX - item_idx == 12 {
                    p.regions.adt_fields.insert(def, vec![vec![t], vec![t], Vec::new()]);
                } else if usize::MAX - item_idx == 26 {
                    p.regions.adt_fields.insert(def, vec![vec![t], vec![t]]);
                } else if usize::MAX - item_idx == 30 {
                    let pe = crate::ty::sync::sync_ty(&mut p.tys, crate::ty::sync::SyncTy::PoisonError, &[t]);
                    p.regions.adt_fields.insert(def, vec![vec![pe], Vec::new()]);
                } else {
                    p.regions.adt_fields.insert(def, vec![vec![t]]);
                }
            }
            if generics::std_adt_has_flat_name(usize::MAX - item_idx) && !std_taken.contains(&name) {
                p.type_names.insert(name, id);
            }
            p.tys.register_adt(def, id);
            let kind = if is_enum { DefsKind::Enum } else { DefsKind::Struct };
            p.defs.insert_adt(def, id, DefData::skeleton(kind, None, name, Vis::Pub, span));
            p.tys.intern(TyKind::Adt(def, EMPTY_ARGS));
            continue;
        }
        // A type item inside a function body takes the same record,
        // but its name is block-scoped: it never enters the flat
        // type table, and its def's parent is the enclosing fn so
        // paths print through it (`crate::f::Local`).
        let stmt_idx = item_idx.checked_sub(STMT_ADT_BASE);
        let item = match stmt_idx {
            Some(si) => &ast.stmt_items[si],
            None => &ast.items[item_idx],
        };
        let id = AdtId(adt_base as u32 + rank);
        match stmt_idx {
            Some(si) => {
                adt_id_of_stmt_item[si] = Some(id);
                p.body_type_items.insert(si as u32, BodyTypeItem::Adt(id));
            }
            None => {
                adt_id_of_item[item_idx] = Some(id);
                // The first type of a name owns it; a duplicate keeps its own Adt
                // record (below) but never replaces the name entry.
                if p.type_names.contains_key(&name) || p.gx.trait_names.contains_key(&name) {
                    errs.push(dup(span, interner.get(name)));
                } else {
                    p.type_names.insert(name, id);
                }
            }
        }
        let n_params = match &item.kind {
            ItemKind::Struct(st) => generics::ast_param_count(&st.generics),
            ItemKind::Enum(e) => generics::ast_param_count(&e.generics),
            _ => 0,
        };
        if n_params > 0 {
            p.mark_generic(item.span);
        }
        // The Adt's stable identity (ARCH 1.7): same name earlier in the file is a
        // duplicate definition (the error above), and the disambiguator keeps the
        // two records apart in the def table, as the old dense ids did.
        let def = DefId { krate: p.krate, hash };
        p.adts[id.0 as usize] =
            AdtDef {
                name,
                is_enum,
                variants: Vec::new(),
                derives: Derives::default(),
                span,
                repr: crate::ty::layout::Repr::default(),
                non_exhaustive: false,
            };
        p.gx.adt_info[id.0 as usize] = (def, n_params);
        // Lifetime params (W-BORROW): names only; the field pass gives
        // them meaning.
        let lifetimes: Vec<Sym> = match &item.kind {
            ItemKind::Struct(st) => &st.generics.params,
            ItemKind::Enum(e) => &e.generics.params,
            _ => &[][..],
        }
        .iter()
        .filter_map(|g| if let ast::GenericParam::Lifetime { lt, .. } = g { Some(lt.name) } else { None })
        .collect();
        if !lifetimes.is_empty() {
            p.gx.adt_lifetimes.insert(def, lifetimes);
        }
        p.tys.register_adt(def, id);
        let kind = if is_enum { DefsKind::Enum } else { DefsKind::Struct };
        let parent = stmt_idx.and_then(|si| stmt_parent_hashes[si]).map(|hash| DefId { krate: p.krate, hash });
        p.defs.insert_adt(def, id, DefData::skeleton(kind, parent, name, Vis::Priv, span));
        p.tys.intern(TyKind::Adt(def, EMPTY_ARGS));
    }

    // Pass 1.5: type aliases. Resolved by a small fixpoint: an alias
    // whose target still names an unresolved alias defers to the next
    // sweep; a sweep with no progress means a cycle (L0601) or a genuine
    // unknown type (the first error is kept). A generic alias's target is
    // over its own params (`Param(i)`); each use substitutes its args
    // (`generics::resolve_path_ty`).
    {
        let mut pending: Vec<&ast::TypeAliasDef> = Vec::new();
        // `aliases` fills only in the fixpoint below, so registration needs
        // its own seen set to catch a second alias of one name.
        let mut seen_aliases: Vec<Sym> = Vec::new();
        for item in &ast.items {
            if let ItemKind::Type(a) = &item.kind {
                if p.type_names.contains_key(&a.name) || p.gx.trait_names.contains_key(&a.name) || seen_aliases.contains(&a.name) {
                    errs.push(dup(a.name_span, interner.get(a.name)));
                    continue;
                }
                seen_aliases.push(a.name);
                // W-BORROW: an alias whose generics are lifetimes only
                // resolves at each use (`generics::resolve_generic_alias`);
                // its body is checked once below, after Adts and traits.
                let lts: Vec<Sym> = a
                    .generics
                    .params
                    .iter()
                    .filter_map(|g| if let ast::GenericParam::Lifetime { lt, .. } = g { Some(lt.name) } else { None })
                    .collect();
                // Lifetime and type params resolve at each use; const
                // params stay with the eager path (unsupported there).
                let type_params: Vec<(Sym, Option<ast::TyId>)> = a
                    .generics
                    .params
                    .iter()
                    .filter_map(|g| if let ast::GenericParam::Type { name, default, .. } = g { Some((*name, *default)) } else { None })
                    .collect();
                let generic = !a.generics.params.is_empty() && lts.len() + type_params.len() == a.generics.params.len();
                let n_errs = errs.len();
                alias_lifetimes(ast, interner, a.ty, &lts, &mut errs);
                // `alias_lifetimes` reads references only; a dyn bound or
                // a path argument of `'_` is rustc E0106 too.
                if errs.len() == n_errs
                    && let Some(d) = generics::item_ty_lifetime_err(ast, interner, a.ty, &lts, false)
                {
                    errs.push(d);
                }
                if generic {
                    // rustc E0091: every type param of an alias is used.
                    for (n, _) in &type_params {
                        if !ty_mentions(ast, a.ty, *n) {
                            errs.push(Diag::new(
                                "L0110",
                                a.name_span,
                                format!("type parameter `{}` is never used in this alias", interner.get(*n)),
                                "Use the parameter in the aliased type, or remove it.",
                            ));
                        }
                    }
                    let names: Vec<Sym> = lts.iter().copied().chain(type_params.iter().map(|&(n, _)| n)).collect();
                    for (i, n) in names.iter().enumerate() {
                        if names[..i].contains(n) {
                            errs.push(Diag::new(
                                "L0110",
                                a.name_span,
                                format!("the parameter `{}` is declared twice on this alias", interner.get(*n)),
                                "Give each parameter its own name.",
                            ));
                        }
                    }
                    let body_hash = crate::driver::ast_hash::item_hash(ast, item);
                    p.gx.generic_aliases.insert(a.name, generics::GenericAlias { lifetimes: lts, type_params, ty: a.ty, erased: ERROR, body_hash });
                } else {
                    pending.push(a);
                }
                let hash = crate::def::item_hash(p.stable_crate, &[], crate::def::DefKind::TypeAlias, interner.get(a.name), 0);
                let def = DefId { krate: p.krate, hash };
                p.defs.insert(def, DefData::skeleton(DefsKind::TypeAlias, None, a.name, Vis::Priv, a.name_span));
            }
        }
        let total = pending.len();
        for _ in 0..=total {
            if pending.is_empty() {
                break;
            }
            let mut progress = false;
            let mut first_err: Option<Diag> = None;
            pending.retain(|a| {
                let generic = a.generics.params.iter().any(|g| !matches!(g, ast::GenericParam::Lifetime { .. }));
                let resolved = if generic {
                    resolve_generic_alias(&mut p, a)
                } else {
                    // The scoped resolver covers what the plain one cannot
                    // (a generic Adt applied to arguments, `W<i32>`).
                    resolve_ty(ast, interner, known, &p.type_names, &p.aliases, &mut p.tys, a.ty, None).or_else(|d| {
                        let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                        generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, a.ty, &generics::Scope::default()).map_err(|_| d)
                    })
                };
                match resolved {
                    Ok(t) => {
                        p.aliases.insert(a.name, t);
                        progress = true;
                        false
                    }
                    Err(d) => {
                        if first_err.is_none() {
                            first_err = Some(d);
                        }
                        true
                    }
                }
            });
            if !progress && !pending.is_empty() {
                // A cycle if every leftover target names another alias.
                let names: Vec<&str> = pending.iter().map(|a| interner.get(a.name)).collect();
                let cyclic = pending.iter().all(|a| match ast.ty(a.ty) {
                    TyExpr::Path(segs) => segs.len() == 1 && names.contains(&interner.get(segs[0].name)),
                    _ => false,
                });
                if cyclic {
                    errs.push(Diag::new(
                        "L0601",
                        pending[0].name_span,
                        format!("cycle detected in type aliases: {}", names.join(", ")),
                        "Break the cycle: an alias must not name itself, directly or through other aliases.",
                    ));
                } else {
                    // W-BORROW: an alias this early pass cannot resolve (a
                    // generic Adt target, an alias of a lazy alias) resolves
                    // at each use like a generic alias, and its body is
                    // checked with theirs below. Const params stay errors.
                    let mut reported = false;
                    for a in &pending {
                        if a.generics.params.is_empty()
                            && let Some(item) = ast.items.iter().find(|it| matches!(&it.kind, ItemKind::Type(x) if x.name == a.name))
                        {
                            let body_hash = crate::driver::ast_hash::item_hash(ast, item);
                            p.gx.generic_aliases.insert(
                                a.name,
                                generics::GenericAlias { lifetimes: Vec::new(), type_params: Vec::new(), ty: a.ty, erased: ERROR, body_hash },
                            );
                        } else if !reported && let Some(d) = first_err.take() {
                            errs.push(d);
                            reported = true;
                        }
                    }
                }
                break;
            }
        }
        check_generic_aliases(&mut p, &mut errs);
    }

    // Pass 1.6: type aliases inside function bodies (R18), by the same
    // fixpoint: each resolves in the overlay of the blocks around it
    // (the Adts pass 1 numbered and the aliases resolved so far), so a
    // later sweep picks up an alias that names a later alias; the
    // enclosing fns' generic params are the E0401 error there. A
    // generic alias in a body is not supported yet.
    {
        let mut pending: Vec<u32> = Vec::new();
        for (si, item) in ast.stmt_items.iter().enumerate() {
            let ItemKind::Type(a) = &item.kind else { continue };
            if a.generics.params.is_empty() {
                pending.push(si as u32);
            } else {
                errs.push(not_yet("L0820", item.span, "generic type aliases inside function bodies"));
            }
        }
        let total = pending.len();
        for _ in 0..=total {
            if pending.is_empty() {
                break;
            }
            let mut progress = false;
            let mut first_err: Option<Diag> = None;
            let waiting = pending.clone();
            pending.retain(|&si| {
                let item = &ast.stmt_items[si as usize];
                let ItemKind::Type(a) = &item.kind else { return false };
                // A name of a still-pending alias visible here (its own
                // name included) must wait for it: resolving now would
                // read an outer item of that name instead (rustc reads the
                // target in the alias's own block: `type u8 = u8;` there
                // is a cycle, E0391).
                let visible = visible_stmt_items(&p.item_blocks, item.span);
                let blocked = waiting.iter().any(|&w| {
                    visible.contains(&w)
                        && matches!(&ast.stmt_items[w as usize].kind, ItemKind::Type(b) if ty_mentions(ast, a.ty, b.name))
                });
                if blocked {
                    if first_err.is_none() {
                        first_err = Some(Diag::new(
                            "L0601",
                            a.name_span,
                            format!("cycle detected in type aliases: {}", interner.get(a.name)),
                            "Break the cycle: an alias must not name itself, directly or through other aliases.",
                        ));
                    }
                    return true;
                }
                let mut scope = generics::Scope::default();
                apply_stmt_overlay(&p, &mut scope, item.span);
                let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                match generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, a.ty, &scope) {
                    Ok(t) => {
                        p.body_type_items.insert(si, BodyTypeItem::Alias(t));
                        progress = true;
                        false
                    }
                    Err(d) => {
                        first_err.get_or_insert(d);
                        true
                    }
                }
            });
            if !progress {
                errs.extend(first_err);
                break;
            }
        }
    }

    // Generic pass 1.5: trait generics, method templates. Fields
    // and fn signatures after this can name traits in bounds.
    crate::driver::laps::mark("collect/adts");
    generics::build_trait_tables(&mut p, &mut errs);
    crate::driver::laps::mark("collect/trait tables");

    // Pass 2: fields, variants, derives. Type items inside
    // function bodies take the same pass (R18): the work list pairs
    // every Adt item with the AdtId pass 1 numbered.
    let mut adt_work: Vec<(&ast::Item, AdtId, Option<usize>)> = Vec::new();
    for (item_idx, item) in ast.items.iter().enumerate() {
        if let Some(id) = adt_id_of_item[item_idx] {
            adt_work.push((item, id, None));
        }
    }
    for (si, item) in ast.stmt_items.iter().enumerate() {
        if let Some(id) = adt_id_of_stmt_item[si] {
            adt_work.push((item, id, Some(si)));
        }
    }
    for (item, id, stmt_idx) in adt_work {
        type FieldSrc = (Sym, ast::TyId, Span);
        type VariantSrc = (Sym, Span, VariantShape, Vec<FieldSrc>, Option<ast::ExprId>, bool);
        let (variants_src, derives_src): (Vec<VariantSrc>, &Vec<(Sym, Span)>) = match &item.kind {
            ItemKind::Struct(s) => {
                let (shape, fields) = variant_fields_src(interner, &s.fields);
                (vec![(s.name, s.name_span, shape, fields, None, false)], &s.derives)
            }
            ItemKind::Enum(e) => {
                let vs = e
                    .variants
                    .iter()
                    .map(|v| {
                        let (shape, fs) = variant_fields_src(interner, &v.fields);
                        (v.name, v.span, shape, fs, v.discriminant, has_non_exhaustive(interner, &v.attrs))
                    })
                    .collect();
                (vs, &e.derives)
            }
            _ => continue,
        };
        // `id` comes from the work list (pass 1 numbered every Adt item).
        // The Adt's own generics scope its field types.
        {
            let (g_ast, w_ast) = match &item.kind {
                ItemKind::Struct(st) => (st.generics.clone(), st.where_clause.clone()),
                ItemKind::Enum(e) => (e.generics.clone(), e.where_clause.clone()),
                _ => unreachable!(),
            };
            let (adt_generics, adt_scope) = {
                let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                generics::build_adt_generics(ast, interner, known, &cx, &mut p.tys, &g_ast, &w_ast, &mut errs)
            };
            let defaults = {
                let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                crate::check::defaults::resolve_all(ast, interner, known, &cx, &mut p.tys, &g_ast, false, &mut errs)
            };
            if defaults.iter().any(Option::is_some) {
                let def = p.defs.def_of_adt(id).expect("ADT defs are recorded in pass 1");
                p.gx.param_defaults.insert(def, defaults);
            }
            let bounds: Vec<Predicate> = {
                let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                generics::regioned_bound_preds(ast, interner, known, &cx, &mut p.tys, &adt_scope, &g_ast, &w_ast, &[]).1
            };
            if bounds.iter().any(|pr| matches!(pr, Predicate::Trait(_))) {
                let def = p.defs.def_of_adt(id).expect("ADT defs are recorded in pass 1");
                p.regions.adt_bounds.insert(def, bounds);
            }
            let env = generics::elaborate_env(&mut p.tys, &p.gx.index, &adt_generics.predicates);
            p.gx.adt_generics[id.0 as usize] = adt_generics;
            p.gx.adt_envs[id.0 as usize] = env;
            p.gx.adt_scopes[id.0 as usize] = adt_scope;
            // A type item in a function body resolves its field
            // types against the type items of its enclosing blocks,
            // and may not use the enclosing functions' parameters.
            if stmt_idx.is_some() {
                let at = p.adts[id.0 as usize].span;
                let mut sc = p.gx.adt_scopes[id.0 as usize].clone();
                apply_stmt_overlay(&p, &mut sc, at);
                p.gx.adt_scopes[id.0 as usize] = sc;
            }
            // Mirror the generics into the def table record.
            let def = p.defs.def_of_adt(id).expect("ADT defs are recorded in pass 1");
            let name = p.adts[id.0 as usize].name;
            let is_enum_now = p.adts[id.0 as usize].is_enum;
            let mut data = DefData::skeleton(
                if is_enum_now { DefsKind::Enum } else { DefsKind::Struct },
                // Pass 1 recorded the parent (the enclosing fn of a
                // type item inside a function body); keep it.
                p.defs.get(def).and_then(|d| d.parent),
                name,
                Vis::Priv,
                p.adts[id.0 as usize].span,
            );
            data.generics = p.gx.adt_generics[id.0 as usize].clone();
            p.defs.insert(def, data);
        }
        let is_enum = p.adts[id.0 as usize].is_enum;
        let disc_want = disc_want_ty(interner, &item.attrs);
        let mut variants = Vec::new();
        // (variant, field, regioned type) for the borrow checker.
        let mut region_fields: Vec<(usize, usize, Ty)> = Vec::new();
        let mut disc_explicit: Vec<bool> = Vec::new();
        // The next implicit discriminant, or None when the previous
        // variant already holds the discriminant type's largest
        // value (the implicit increment then overflows, E0370).
        let mut next_disc: Option<i128> = Some(0);
        let disc_max = crate::ty::consteval::int_stored(disc_want.max128(), disc_want);
        // `Self` in a field type names the Adt at its own params
        // (`next: Option<Box<Self>>`), as in rustc. Only for Adts whose
        // params are all types: a lifetime or const param keeps the
        // L0102 reject. The scope is the fields' own copy.
        let self_args: Option<Vec<GenericArg>> = {
            let g = &p.gx.adt_generics[id.0 as usize];
            let lifetimes = match &item.kind {
                ItemKind::Struct(st) => st.generics.params.iter().any(|g| matches!(g, ast::GenericParam::Lifetime { .. })),
                ItemKind::Enum(e) => e.generics.params.iter().any(|g| matches!(g, ast::GenericParam::Lifetime { .. })),
                _ => true,
            };
            (!lifetimes && g.params.iter().all(|q| matches!(q.kind, ParamKind::Type)))
                .then(|| (0..g.params.len() as u32).map(|i| GenericArg::Ty(p.tys.intern(TyKind::Param(ParamId(i))))).collect())
        };
        if let Some(args) = self_args {
            let def = p.def_of_adt(id);
            let self_ty = p.tys.mk_adt(def, &args);
            p.gx.adt_scopes[id.0 as usize].self_ty.get_or_insert(self_ty);
            // A field's const (`[u8; Self::N]`) may name `Self` only when
            // the Adt has no parameters (rustc rejects a generic `Self`).
            if args.is_empty() {
                p.gx.adt_scopes[id.0 as usize].self_name = match &item.kind {
                    ItemKind::Struct(st) => Some(st.name),
                    ItemKind::Enum(e) => Some(e.name),
                    _ => None,
                };
            }
        }
        // The Adt's scope with regions kept, for the fields' regioned types.
        let mut keep_scope = p.gx.adt_scopes[id.0 as usize].clone();
        keep_scope.regions = generics::RegionMode::Keep { next: None, elided: None };
        for (vname, vspan, shape, field_srcs, disc_src, variant_ne) in variants_src {
            let mut fields = Vec::new();
            for (fname, fty, fspan) in field_srcs {
                if shape == VariantShape::Named && fields.iter().any(|f: &Field| f.name == fname) {
                    errs.push(dup(fspan, interner.get(fname)));
                }
                let resolved = {
                    let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                    generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, fty, &p.gx.adt_scopes[id.0 as usize])
                };
                match resolved {
                    Ok(t) => {
                        // The field's regions (W-BORROW): every reference
                        // needs a lifetime the Adt declares, or `'static`
                        // (an elided one is rustc E0106).
                        let regioned = {
                            let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                            generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, fty, &keep_scope)
                        };
                        match regioned {
                            Ok(r) => region_fields.push((variants.len(), fields.len(), r)),
                            Err(d) if d.code == "L0202" => errs.push(Diag::new(
                                "L0201",
                                ast.ty_span(fty),
                                "a reference in a struct or enum field needs a lifetime",
                                "Declare a lifetime on the type (`struct S<'a> { x: &'a str }`), use `'static`, or store an owned value.",
                            )),
                            Err(_) => {}
                        }
                        let derive_bounds = derive_bound_tys(&mut p, ast, interner, known, id, fty);
                        fields.push(Field { name: fname, ty: t, span: ast.ty_span(fty), derive_bounds });
                    }
                    Err(d) => {
                        errs.push(d);
                        fields.push(Field { name: fname, ty: ERROR, span: ast.ty_span(fty), derive_bounds: Vec::new() });
                    }
                }
            }
            if is_enum && variants.iter().any(|v: &Variant| v.name == vname) {
                errs.push(dup(vspan, interner.get(vname)));
            }
            disc_explicit.push(disc_src.is_some());
            let disc = match disc_src {
                Some(de) => match eval_const_int_exp(ast, interner, known, de, Some(disc_want)) {
                    Some(v) => {
                        // The written type must be the discriminant
                        // type (rustc E0308); only unsuffixed literal
                        // forms take it by inference.
                        let got = disc_ty(ast, interner, known, de, &mut Vec::new(), 0);
                        if !dty_accepts(got, disc_want) {
                            let found = match got {
                                DTy::Exact(it) => format!("`{}`", it.name()),
                                _ => "a non-integer or untyped form".to_string(),
                            };
                            errs.push(Diag::new(
                                "L0110",
                                vspan,
                                format!("mismatched types: expected `{}`, found {found}", disc_want.name()),
                                "Write the discriminant as a value of the repr integer type (or `isize` with no repr).",
                            ));
                        }
                        v
                    }
                    None => {
                        errs.push(Diag::new(
                            "L0110",
                            vspan,
                            "enum discriminant must be a constant integer",
                            "Write a literal value such as `A = 3`.",
                        ));
                        next_disc.unwrap_or(0)
                    }
                },
                None => match next_disc {
                    Some(d) => d,
                    None => {
                        errs.push(Diag::new(
                            "L0110",
                            vspan,
                            "enum discriminant overflowed",
                            "Give this variant an explicit discriminant, or use a wider repr integer type.",
                        ));
                        0
                    }
                },
            };
            next_disc = if disc == disc_max { None } else { Some(disc.wrapping_add(1)) };
            variants.push(Variant { name: vname, shape, fields, disc, non_exhaustive: variant_ne });
        }
        let mut d = Derives::default();
        let mut seen_derives: Vec<Sym> = Vec::new();
        for &(name, span) in derives_src {
            // A trait derived twice (in one list, across `#[derive]`
            // attributes, or through `cfg_attr`) is two impls (rustc E0119).
            if seen_derives.contains(&name) {
                let tname = interner.get(name);
                errs.push(Diag::new(
                    "L0615",
                    span,
                    format!("conflicting implementations of trait `{tname}`: it is derived twice"),
                    "Derive each trait once.",
                ));
                continue;
            }
            seen_derives.push(name);
            match interner.get(name) {
                "Debug" => d.debug = true,
                "Clone" => d.clone = true,
                "Copy" => d.copy = true,
                "PartialEq" => d.partial_eq = true,
                "Eq" => d.eq = true,
                "Hash" => d.hash = true,
                "PartialOrd" => d.partial_ord = true,
                "Ord" => d.ord = true,
                "Default" => d.default = true,
                n => errs.push(Diag::new(
                    "L0403",
                    span,
                    format!("cannot derive `{n}`"),
                    "Lust can derive Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, and Default.",
                )),
            }
        }
        // `Copy` needs a `Clone` impl, derived or written: checked in
        // finish_collect (`check_derive_bounds`), once impls exist.
        // `#[default]` variants (rustc's rules): only with a `Default`
        // derive, exactly one per enum, and only on a unit variant.
        if let ItemKind::Enum(e) = &item.kind {
            let mut dup_default: Option<Span> = None;
            let marked: Vec<(u32, Span)> = e
                .variants
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    let mut it = v.attrs.iter().filter(|a| a.path.len() == 1 && interner.get(a.path[0].name) == "default");
                    let first = it.next()?;
                    if let Some(second) = it.next()
                        && dup_default.is_none()
                    {
                        dup_default = Some(second.span);
                    }
                    Some((i as u32, first.span))
                })
                .collect();
            if let Some(sp) = dup_default {
                errs.push(Diag::new("L0403", sp, "multiple `#[default]` attributes", "Keep one `#[default]` on the variant."));
            }
            if !d.default {
                if let Some(&(_, sp)) = marked.first() {
                    errs.push(Diag::new(
                        "L0402",
                        sp,
                        "cannot find attribute `default` in this scope",
                        "Add `Default` to the derive list, or remove `#[default]`.",
                    ));
                }
            } else if marked.is_empty() {
                errs.push(Diag::new(
                    "L0403",
                    p.adts[id.0 as usize].span,
                    "`#[derive(Default)]` on enum with no `#[default]`",
                    "Mark one unit variant with `#[default]`.",
                ));
            } else if marked.len() > 1 {
                errs.push(Diag::new("L0403", marked[1].1, "multiple declared defaults", "Keep `#[default]` on one variant only."));
            } else if variants.get(marked[0].0 as usize).is_some_and(|v: &Variant| v.shape != VariantShape::Unit) {
                errs.push(Diag::new(
                    "L0403",
                    marked[0].1,
                    "the `#[default]` attribute may only be used on unit enum variants",
                    "Move `#[default]` to a variant with no fields.",
                ));
            } else {
                d.default_variant = Some(marked[0].0);
            }
        }
        // Regioned field types (W-BORROW). Unused type and lifetime
        // params (rustc E0392) are checked after every Adt has its
        // fields (`variance::check_unused_params`).
        {
            let def = p.gx.adt_info[id.0 as usize].0;
            let mut by_variant: Vec<Vec<Ty>> = variants.iter().map(|v| v.fields.iter().map(|f| f.ty).collect()).collect();
            for &(vi, fi, r) in &region_fields {
                if let Some(slot) = by_variant.get_mut(vi).and_then(|v| v.get_mut(fi)) {
                    *slot = r;
                }
            }
            let fields: Vec<Ty> = by_variant.iter().flatten().copied().collect();
            p.regions.adt_fields.insert(def, by_variant);
            let facts = p.gx.adt_scopes[id.0 as usize].region_facts.clone();
            if !facts.is_empty() {
                p.regions.adt_facts.insert(def, facts.clone());
            }
            // The types its bounds name may assume what its fields need
            // (rustc's inferred outlives).
            let (g_ast, w_ast) = match &item.kind {
                ItemKind::Struct(st) => (&st.generics, &st.where_clause),
                ItemKind::Enum(e) => (&e.generics, &e.where_clause),
                _ => unreachable!(),
            };
            let required = {
                let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
                let scope = p.gx.adt_scopes[id.0 as usize].clone();
                generics::regioned_bound_tys(ast, interner, known, &cx, &mut p.tys, &scope, g_ast, w_ast, &[])
            };
            let preds = p.gx.adt_generics[id.0 as usize].predicates.clone();
            let item = crate::check::outlives::Item { implied: fields, required, facts, preds, span: p.adts[id.0 as usize].span };
            if crate::check::outlives::wanted(&p, &item) {
                p.regions.outlives_items.push(item);
            }
        }
        let adt = &mut p.adts[id.0 as usize];
        adt.variants = variants;
        adt.derives = d;
        adt.non_exhaustive = has_non_exhaustive(interner, &item.attrs);
        match parse_repr(interner, &item.attrs) {
            Ok(r) => adt.repr = r,
            Err(dg) => errs.push(dg),
        }
        // Repr and discriminant rules Rust also enforces (ARCH 4.4).
        {
            use crate::ty::layout::ReprKind;
            let adt = &p.adts[id.0 as usize];
            if !adt.is_enum && adt.repr.int.is_some() {
                errs.push(Diag::new(
                    "L0110",
                    adt.span,
                    "`#[repr(...)]` with an integer type can only be used on an enum",
                    "Remove the integer from the repr, or make the type an enum.",
                ));
            }
            if adt.is_enum && adt.repr.kind == ReprKind::C && adt.repr.int.is_some() && adt.variants.iter().all(|v| v.fields.is_empty()) {
                errs.push(Diag::new(
                    "L0110",
                    adt.span,
                    "conflicting representation hints: `repr(C)` and an integer repr on a fieldless enum",
                    "Use `repr(C)` or the integer repr alone; fieldless enums have no payload for `repr(C)` to order.",
                ));
            }
            if adt.repr.pack.is_some() && adt.repr.align.is_some() {
                errs.push(Diag::new(
                    "L0110",
                    adt.span,
                    "`#[repr(packed)]` and `#[repr(align(...))]` conflict",
                    "Remove one of them: packed caps alignment, align raises it.",
                ));
            }
            if adt.is_enum && adt.repr.pack.is_some() {
                errs.push(Diag::new(
                    "L0110",
                    adt.span,
                    "the `repr(packed)` attribute cannot be used on enums",
                    "Remove `packed`, or make the type a struct.",
                ));
            }
            // `repr(transparent)` field counts are checked after
            // every Adt has its fields (`check_transparent`, below):
            // whether a field is zero-sized can need another Adt's
            // layout.
            if let Some((bits, signed)) = adt.repr.int {
                // A discriminant is stored in the repr type's bit
                // form, so a u128 value at or above 2^127 is a
                // negative i128 here and still in range.
                for v in &adt.variants {
                    let in_range = if signed {
                        bits >= 128 || (v.disc >= -(1i128 << (bits - 1)) && v.disc < (1i128 << (bits - 1)))
                    } else if bits >= 128 {
                        true
                    } else {
                        v.disc >= 0 && v.disc < (1i128 << bits)
                    };
                    if !in_range {
                        errs.push(Diag::new(
                            "L0110",
                            adt.span,
                            format!("discriminant value {} overflows the repr range", v.disc),
                            "Change the discriminant or the repr integer type.",
                        ));
                    }
                }
            }
            if adt.is_enum {
                // With no int repr the discriminant type is `isize`; an
                // implicit discriminant after `isize::MAX` lands outside it.
                if adt.repr.int.is_none() {
                    for v in &adt.variants {
                        if v.disc < i64::MIN as i128 || v.disc > i64::MAX as i128 {
                            errs.push(Diag::new(
                                "L0110",
                                adt.span,
                                format!("discriminant value {} overflows `isize`", v.disc),
                                "Change the discriminant, or add an int repr such as `#[repr(i64)]`.",
                            ));
                        }
                    }
                }
                // Duplicate discriminants, explicit or implicit (rustc E0081).
                for i in 0..adt.variants.len() {
                    if adt.variants[..i].iter().any(|v| v.disc == adt.variants[i].disc) {
                        errs.push(Diag::new(
                            "L0110",
                            adt.span,
                            format!("discriminant value {} is used more than once", adt.variants[i].disc),
                            "Give each variant a different discriminant value.",
                        ));
                    }
                }
                // Explicit discriminants on an enum with data variants need
                // an int repr (rustc E0732).
                let any_explicit = disc_explicit.iter().any(|&x| x);
                let any_data = adt.variants.iter().any(|v| !v.fields.is_empty());
                if any_explicit && any_data && adt.repr.int.is_none() {
                    errs.push(Diag::new(
                        "L0110",
                        adt.span,
                        "enums with explicit discriminants and data variants need an int repr",
                        "Add a repr such as `#[repr(u8)]`, or remove the discriminants.",
                    ));
                }
            }
        }
    }

    // `repr(transparent)` shape, now that every Adt has its fields:
    // a struct needs at most one field of nonzero size, an enum
    // exactly one variant with at most one such field (rustc
    // E0690, E0731). A field whose layout lust cannot compute here
    // (a type parameter, an unsized type) counts as nonzero, as
    // rustc counts it.
    {
        use crate::ty::layout::ReprKind;
        for adt in &p.adts {
            if adt.repr.kind != ReprKind::Transparent {
                continue;
            }
            let nontrivial = |v: &Variant| {
                v.fields
                    .iter()
                    .filter(|f| {
                        f.ty != ERROR
                            && crate::ty::layout::layout_of_ty(&p.tys, &UserLayout(&p), f.ty).map_or(true, |l| l.size > 0)
                    })
                    .count()
            };
            let bad = if adt.is_enum {
                adt.variants.len() != 1 || nontrivial(&adt.variants[0]) > 1
            } else {
                adt.variants.first().map_or(0, nontrivial) > 1
            };
            if bad {
                errs.push(Diag::new(
                    "L0110",
                    adt.span,
                    if adt.is_enum {
                        "`repr(transparent)` needs an enum with exactly one variant and at most one non-zero-sized field"
                    } else {
                        "`repr(transparent)` needs a struct with at most one non-zero-sized field"
                    },
                    "Remove the repr, or give the type one field plus zero-sized fields.",
                ));
            }
        }
    }

    // Unused type and lifetime params, now that every Adt has its
    // fields (a use through another Adt reads that Adt's own uses).
    {
        let lt_spans = |item: &ast::Item| -> Vec<Span> {
            let params = match &item.kind {
                ItemKind::Struct(st) => &st.generics.params,
                ItemKind::Enum(e) => &e.generics.params,
                _ => return Vec::new(),
            };
            params.iter().filter_map(|g| if let ast::GenericParam::Lifetime { lt, .. } = g { Some(lt.span) } else { None }).collect()
        };
        let mut templates: Vec<(AdtId, Vec<Span>)> = adt_id_of_item
            .iter()
            .enumerate()
            .filter_map(|(i, a)| a.map(|a| (a, lt_spans(&ast.items[i]))))
            .collect();
        for (si, item) in ast.stmt_items.iter().enumerate() {
            if let Some(id) = adt_id_of_stmt_item[si] {
                templates.push((id, lt_spans(item)));
            }
        }
        crate::check::variance::check_unused_params(&p, &templates, &mut errs);
    }

    // Derived traits need every field type to have the trait too.
    p.gx.derive_marks[0] = p.gx.adt_created.len();
    let deferred_debug = check_derive_fields(&p, &|_| true, &mut errs);
    p.gx.deferred_debug_fields.extend(deferred_debug);

    // Unsized newtypes (W-SHIM): a struct whose only field is `str`,
    // `[T]`, or (non-generic only) another unsized newtype is itself
    // unsized and laid out as that field (`std::path::Path`,
    // `std::ffi::OsStr`, indexmap's `map::Slice<K, V>` over
    // `[Bucket<K, V>]`). Nested ones settle in a fixpoint, whatever the
    // declaration order.
    loop {
        let mut grew = false;
        for (i, adt) in p.adts.iter().enumerate() {
            let [v] = adt.variants.as_slice() else { continue };
            let [f] = v.fields.as_slice() else { continue };
            let Some(def) = p.tys.def_of_local_adt(AdtId(i as u32)) else { continue };
            let generic = p.gx.adt_generics.get(i).is_some_and(|g| !g.params.is_empty());
            if adt.is_enum || p.tys.unsized_field(def).is_some() || (generic && !matches!(p.tys.kind(f.ty), TyKind::Str | TyKind::Slice(_))) {
                continue;
            }
            if matches!(p.tys.kind(f.ty), TyKind::Str | TyKind::Slice(_)) || !is_sized_ty(&p.tys, f.ty) {
                let (def, fty) = (def, f.ty);
                p.tys.register_unsized_adt(def, fty);
                grew = true;
                break;
            }
        }
        if !grew {
            break;
        }
    }
    // Fields must be sized: a bare slice or str cannot sit in a struct
    // (the one field of an unsized newtype excepted).
    for (i, adt) in p.adts.iter().enumerate() {
        if p.tys.def_of_local_adt(AdtId(i as u32)).is_some_and(|d| p.tys.unsized_field(d).is_some()) {
            continue;
        }
        for v in &adt.variants {
            for f in &v.fields {
                if !is_sized_ty(&p.tys, f.ty) {
                    errs.push(Diag::new(
                        "L0110",
                        adt.span,
                        format!("the size for values of type `{}` cannot be known", ty_name(&p, &p.tys, f.ty)),
                        "Put the value behind a reference, such as `&[T]` or `&str`.",
                    ));
                }
            }
        }
    }

    // A type that contains itself by value has infinite size.
    for i in 0..p.adts.len() {
        let mut seen = vec![false; p.adts.len()];
        if contains_adt_by_value(&p, AdtId(i as u32), AdtId(i as u32), &mut seen) {
            errs.push(Diag::new(
                "L0117",
                p.adts[i].span,
                format!("recursive type `{}` has infinite size", p.display_path(p.adts[i].name)),
                "Put the recursive field behind a `Box`, in a `Vec`, or behind a reference.",
            ));
        }
    }

    // Functions and methods. Each gets the stable path hash its caller computed
    // (src/def.rs). Hash collisions are checked by the driver when it assigns
    // slots from the same hashes, before `collect` runs on any new def set;
    // here the maps only record the hashes. A duplicate definition keeps the
    // first local id in `def_by_hash`; the dup errors below report it.
    // Functions and methods. Each gets the stable path hash its caller computed
    // (src/def.rs). Hash collisions are checked by the driver when it assigns
    // slots from the same hashes, before `collect` runs on any new def set;
    // here the maps only record the hashes. A duplicate definition keeps the
    // first local id in `def_by_hash`; the dup errors below report it.
    //
    // A fn with generic parameters (its own, or its impl's) is a template:
    // its FnInfo holds the signature over `Param` tys, and mono appends a
    // concrete FnInfo per instance after this pass. Plain fns are their
    // own only instance, exactly as before generics.
    let plain_parent = ParentGen::default;
    let add_fn = |p: &mut Program, site: FnSite, parent_gen: &ParentGen, errs: &mut Vec<Diag>| -> LocalDefId {
        let rec = fn_record(p, ast, interner, known, site, parent_gen, errs);
        install_fn(p, rec, Install::Push)
    };
    // W-BORROW: each lifetime alias's body, checked once like rustc does
    // (an unused alias with an unknown type is still an error), and its
    // erased form kept for the driver's fingerprints.
    crate::driver::laps::mark("collect/fields+impls");
    let mut aliases: Vec<(Sym, generics::GenericAlias)> = p.gx.generic_aliases.iter().map(|(&n, a)| (n, a.clone())).collect();
    aliases.sort_by_key(|(_, a)| a.ty.0);
    for (name, la) in aliases {
        let mut scope = generics::Scope { alias_lifetimes: la.lifetimes.iter().map(|&n| (n, Region::Erased)).collect(), alias_depth: 1, ..generics::Scope::default() };
        for (i, &(n, _)) in la.type_params.iter().enumerate() {
            let t = p.tys.intern(TyKind::Param(ParamId(i as u32)));
            scope.param_tys.push((n, t));
        }
        let res = {
            let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
            let body = generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, la.ty, &scope);
            // Defaults are types of their own: check them too.
            let mut defaults = Ok(());
            for &(_, d) in &la.type_params {
                if let Some(d) = d
                    && let Err(e) = generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, d, &scope)
                {
                    defaults = Err(e);
                    break;
                }
            }
            defaults.and(body)
        };
        match res {
            Ok(t) => {
                if let Some(e) = p.gx.generic_aliases.get_mut(&name) {
                    e.erased = t;
                }
            }
            Err(d) => errs.push(d),
        }
    }
    // Each impl's path and its methods' (`def::impl_ids`, as the driver's `fn_keys`).
    let impl_ids = crate::def::impl_ids(p.stable_crate, ast, interner);
    // Top-level fn and method hashes by body block: the parents of fns
    // nested in bodies (`def::nested_fn_ids`).
    let mut top_hashes: HashMap<ast::BlockId, crate::def::DefPathHash> = HashMap::default();
    // The trait of each trait impl, by item index, for the trait-system
    // key (`gen_reuse::trait_key`).
    let mut impl_traits: HashMap<usize, (DefId, bool, DefId)> = HashMap::default();
    for (item_idx, item) in ast.items.iter().enumerate() {
        match &item.kind {
            ItemKind::Fn(def) => {
                let hash = crate::def::free_fn_hash(p.stable_crate, &[], interner.get(def.name));
                if let Some(b) = def.body {
                    top_hashes.insert(b, hash);
                }
                let site = FnSite { loc: FnLoc::Item(item_idx as u32), owner: None, self_ty: None, parent: None, hash };
                let id = add_fn(&mut p, site, &plain_parent(), &mut errs);
                // The first value of a name owns it. A unit or tuple struct
                // constructor is a value of the struct's name too, so a fn
                // of that name is a duplicate; a braced struct is not.
                let ctor_taken = p.type_names.get(&def.name).is_some_and(|&a| {
                    !p.adts[a.0 as usize].is_enum && p.adts[a.0 as usize].variants.first().is_some_and(|v| v.shape != VariantShape::Named)
                });
                if p.fn_names.contains_key(&def.name) || ctor_taken {
                    errs.push(dup(def.name_span, interner.get(def.name)));
                } else {
                    p.fn_names.insert(def.name, id);
                }
                if def.name == known.main && p.main.is_none() {
                    p.main = Some(id);
                }
            }
            ItemKind::Impl(imp) => {
                if imp.is_negative {
                    errs.push(not_yet("L0614", imp.span, "negative impls"));
                    continue;
                }
                let ids = impl_ids[item_idx].as_ref().expect("an impl item has impl ids");
                let parent = ids.path.0;
                for (k, def) in imp.fns.iter().enumerate() {
                    if let Some(b) = def.body {
                        top_hashes.insert(b, ids.methods[k].0);
                    }
                }
                let impl_def = DefId { krate: p.krate, hash: parent };
                let std = p.in_lang_methods(imp.span);
                p.defs.insert(impl_def, DefData::skeleton(DefsKind::Impl, None, known.self_type, Vis::Priv, imp.span));

                let (impl_generics, mut impl_scope) = {
                    // A body impl's bounds see the block type items (R18).
                    let mut seed = generics::Scope::default();
                    if imp.in_body {
                        apply_stmt_overlay(&p, &mut seed, imp.span);
                    }
                    let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                    generics::build_generics_shifted(ast, interner, known, &cx, &mut p.tys, &imp.generics, &imp.where_clause, 0, &seed, &mut errs)
                };
                if !impl_generics.params.is_empty() {
                    p.mark_generic(imp.span);
                }
                // An impl the resolver hoisted out of a fn body sees the
                // block type items around it, not the enclosing fns'
                // params (R18).
                if imp.in_body {
                    apply_stmt_overlay(&p, &mut impl_scope, imp.span);
                }
                let self_pat = {
                    let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                    match generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, imp.self_ty, &impl_scope) {
                        Ok(t) => t,
                        Err(d) => {
                            errs.push(d);
                            continue;
                        }
                    }
                };
                // An impl for a projection (`impl Tr for
                // <S as X>::A`) stands for the normalized type
                // when the projection resolves through impls
                // already collected.
                // A derive delta reads the written type (`written_projections`).
                if p.tys.flags(self_pat).contains_projection() {
                    p.gx.written_projections.entry(impl_def).or_default().push(self_pat);
                }
                let self_pat = generics::normalize_global(&mut p, self_pat);
                impl_scope.self_ty = Some(self_pat);
                // `Self::X` in a const of an impl with no parameters
                // names the self type's consts (a generic `Self` is
                // rustc's anonymous const error).
                if imp.generics.params.is_empty()
                    && let TyExpr::Path(segs) = ast.ty(imp.self_ty)
                    && let [seg, ..] = segs.as_slice()
                {
                    impl_scope.self_name = Some(seg.name);
                }
                impl_regions(&mut p, imp, impl_def, &mut impl_scope, &mut errs);
                {
                    let mut required = {
                        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                        generics::regioned_bound_tys(ast, interner, known, &cx, &mut p.tys, &impl_scope, &imp.generics, &imp.where_clause, &[])
                    };
                    if imp.trait_ref.is_some() {
                        let preds = {
                            let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                            generics::regioned_bound_preds(ast, interner, known, &cx, &mut p.tys, &impl_scope, &imp.generics, &imp.where_clause, &[]).1
                        };
                        let head = crate::borrowck::ImplRegions {
                            header: impl_scope.header_regioned.clone(),
                            facts: impl_scope.region_facts.clone(),
                            preds,
                            assoc: impl_scope.assoc_regioned.clone(),
                        };
                        if head.wanted(&p.tys) {
                            p.regions.impl_regions.insert(impl_def, head);
                        }
                    }
                    required.extend(impl_scope.assoc_regioned.iter().map(|&(_, t)| t));
                    let item = crate::check::outlives::Item {
                        implied: impl_scope.header_regioned.clone(),
                        required,
                        facts: impl_scope.region_facts.clone(),
                        preds: impl_generics.predicates.clone(),
                        span: imp.span,
                    };
                    if crate::check::outlives::wanted(&p, &item) {
                p.regions.outlives_items.push(item);
            }
                }
                // The owner template Adt, when the Self pattern is a
                // local Adt application (inherent and trait impls).
                let owner: Option<AdtId> = match p.tys.kind(self_pat) {
                    TyKind::Adt(def, _) => p.defs.local_adt(def),
                    _ => None,
                };
                let trait_ref = match &imp.trait_ref {
                    None => None,
                    Some(tref) => {
                        p.mark_generic(imp.span);
                        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
                        match generics::resolve_trait_ref(ast, interner, known, &cx, &mut p.tys, &impl_scope, self_pat, tref, imp.span) {
                            Ok(tr) => {
                                impl_traits.insert(item_idx, (tr.def, p.lang_keyed_impl(&tr), impl_def));
                                Some(tr)
                            }
                            Err(d) => {
                                errs.push(d);
                                continue;
                            }
                        }
                    }
                };
                // An inherent impl parameter must occur in the
                // Self type (rustc E0207's inherent half).
                if trait_ref.is_none() {
                    for (i, param) in impl_generics.params.iter().enumerate() {
                        let used = match param.kind {
                            ParamKind::Type => generics::ty_uses_param(&p.tys, self_pat, i as u32),
                            ParamKind::Const(_) => generics::ty_uses_const_param(&p.tys, self_pat, i as u32),
                            ParamKind::Lifetime => continue,
                        };
                        if !used {
                            errs.push(Diag::new(
                                "L0110",
                                param.span,
                                format!("the parameter `{}` is not used in the impl header", interner.get(param.name)),
                                "Use it in the Self type, or remove it.",
                            ));
                        }
                    }
                }
                // `impl dyn Trait { .. }`: inherent methods on a dyn
                // type, found by the dyn receiver probe (check/method.rs).
                let dyn_self = trait_ref.is_none() && matches!(p.tys.kind(self_pat), TyKind::Dyn(..));
                if dyn_self {
                    p.mark_generic(imp.span);
                    p.gx.dyn_impl_sites.push((self_pat, ast.ty_span(imp.self_ty)));
                }
                // A header dyn whose object lifetime is `'static` (a dyn
                // Self outside a reference, a dyn trait argument, or a
                // written `dyn Tr + 'static` anywhere in the header).
                // A written trait arg `dyn Tr + 'a` takes any lifetime, and
                // so does a default arg that is Self (`Rhs = Self`).
                let written: Vec<ast::TyId> = match imp.trait_ref.as_ref().and_then(|t| t.path.last()).map(|s| &s.args) {
                    Some(ast::GenericArgs::Angle(xs)) => xs.iter().filter_map(|g| if let ast::GenericArg::Type(t) = g { Some(*t) } else { None }).collect(),
                    _ => Vec::new(),
                };
                let static_dyn = crate::check::object::header_dyn_is_static(ast, interner, imp.self_ty)
                    || p.tys.flags(self_pat).contains_static_region()
                    || trait_ref.is_some_and(|tr| {
                        p.tys.args(tr.args).iter().skip(1).enumerate().any(|(i, g)| {
                            g.as_ty().is_some_and(|t| {
                                let dyn_static = written.get(i).map_or(t != self_pat, |&a| crate::check::object::header_dyn_is_static(ast, interner, a));
                                (matches!(p.tys.kind(t), TyKind::Dyn(..)) && dyn_static) || p.tys.flags(t).contains_static_region()
                            })
                        })
                    });
                if static_dyn {
                    p.gx.static_dyn_impls.push(impl_def);
                }
                // A `&'static` reference to a dyn value in a header: the
                // DynStatic proof covers object lifetimes only, and lust
                // tracks no other reference lifetime.
                let static_ref = crate::check::object::has_static_ref(&p.tys, self_pat)
                    || trait_ref.is_some_and(|tr| {
                        p.tys.args(tr.args).iter().any(|g| g.as_ty().is_some_and(|t| crate::check::object::has_static_ref(&p.tys, t)))
                    });
                if static_ref {
                    errs.push(Diag::new(
                        "L0705",
                        ast.ty_span(imp.self_ty),
                        "lust cannot check a `&'static` reference to a `dyn` value in an impl header",
                        "Implement the trait for `dyn Tr` itself, or for a type without `&'static dyn`.",
                    ));
                }
                // Rustc E0116: no inherent impl on a std type (W-STD-C).
                if trait_ref.is_none()
                    && let Some(a) = owner
                    && p.is_std_adt(a)
                    && !p.std_source_span(imp.span)
                {
                    errs.push(Diag::new(
                        "L0616",
                        ast.ty_span(imp.self_ty),
                        "cannot define an inherent impl for a type outside of the crate where the type is defined",
                        "Define a trait with the methods, and implement it for the type.",
                    ));
                    continue;
                }
                // Std source's inherent impls of the builtin lang enums
                // (`impl<T> Option<T>` in library/core): their methods are
                // found by the builtin receiver probe in check's method call.
                let lang_self = match p.tys.kind(self_pat) {
                    TyKind::Adt(def, _)
                        if trait_ref.is_none()
                            && matches!(p.tys.builtins().kind_of(def), Some(BuiltinTy::Option | BuiltinTy::Result))
                            && p.in_lang_methods(imp.span) =>
                    {
                        Some(def)
                    }
                    _ => None,
                };
                // Std source's inherent impls on a primitive (`impl
                // char` in `core::char`, TRIAGE T8): only the splice
                // may name one, and its methods register apart from
                // the Adt tables (`prim_inherent_methods`).
                let prim_self = if trait_ref.is_none()
                    && p.std_source_span(imp.span)
                    && matches!(p.tys.kind(self_pat), TyKind::Char | TyKind::Bool | TyKind::Int(_) | TyKind::Float(_))
                {
                    Some(self_pat)
                } else {
                    None
                };
                if trait_ref.is_none() && owner.is_none() && !dyn_self && lang_self.is_none() && prim_self.is_none() {
                    errs.push(Diag::new(
                        "L0102",
                        ast.ty_span(imp.self_ty),
                        "inherent impls must name a struct or enum defined in this file",
                        "Write `impl MyType { ... }`.",
                    ));
                    continue;
                }
                // An inherent impl of a shipped crate's type (an adapter,
                // or the std platform crate of W-SHIM) from another crate
                // (rustc E0116).
                if trait_ref.is_none()
                    && let Some(a) = owner
                    && p.shipped_crate_at(p.adts[a.0 as usize].span) != p.shipped_crate_at(imp.span)
                {
                    errs.push(Diag::new(
                        "L0102",
                        ast.ty_span(imp.self_ty),
                        "inherent impls must name a struct or enum defined in this crate",
                        "Define a trait for the methods and implement it for the type.",
                    ));
                    continue;
                }
                if trait_ref.is_none() {
                    p.gx.wf_headers.push((self_pat, impl_generics.clone(), imp.span));
                    p.gx.inherent_impl_self.insert(impl_def, (self_pat, impl_generics.clone()));
                }
                // The impl's trait is an assoc source on the Self
                // pattern inside its method signatures.
                if let Some(tr) = &trait_ref
                    && let Some(rec) = p.gx.index.trait_record(tr.def).cloned()
                {
                    impl_scope.assoc_srcs.push(generics::AssocSrc {
                        self_ty: self_pat,
                        trait_def: tr.def,
                        args: tr.args,
                        assocs: rec.assoc_items.iter().map(|&(d, n, _)| (n, d)).collect(),
                    });
                    // Its supertraits' too: `Self::Item` in a
                    // `DoubleEndedIterator` impl is `Iterator`'s (W-STD-B).
                    let targs: Vec<GenericArg> = p.tys.args(tr.args).to_vec();
                    let p0 = p.tys.intern(TyKind::Param(ParamId(0)));
                    for pred in &rec.generics.predicates {
                        let Predicate::Trait(b) = pred else { continue };
                        if b.count != 0 || p.tys.args(b.value.args).first().and_then(|g| g.as_ty()) != Some(p0) {
                            continue;
                        }
                        let Some(srec) = p.gx.index.trait_record(b.value.def).cloned() else { continue };
                        if srec.assoc_items.is_empty() {
                            continue;
                        }
                        let sargs: Vec<GenericArg> = p
                            .tys
                            .args(b.value.args)
                            .to_vec()
                            .into_iter()
                            .map(|g| match g {
                                GenericArg::Ty(t) => GenericArg::Ty(crate::ty::subst(&mut p.tys, t, &targs)),
                                other => other,
                            })
                            .collect();
                        let sargs = p.tys.intern_args(&sargs);
                        impl_scope.assoc_srcs.push(generics::AssocSrc {
                            self_ty: self_pat,
                            trait_def: b.value.def,
                            args: sargs,
                            assocs: srec.assoc_items.iter().map(|&(d, n, _)| (n, d)).collect(),
                        });
                    }
                }
                let mut impl_scope = impl_scope;
                impl_scope.type_outlives = trait_ref.is_none();
                let parent_gen = ParentGen { generics: impl_generics.clone(), scope: impl_scope, impl_item: Some(item_idx as u32) };
                let mut method_ids: Vec<LocalDefId> = Vec::new();
                for (k, def) in imp.fns.iter().enumerate() {
                    // Functions in trait impls cannot be declared const
                    // (rustc E0379); inherent impl methods can.
                    if trait_ref.is_some() && def.qual.is_const {
                        errs.push(Diag::new(
                            "L0632",
                            def.name_span,
                            "functions in trait impls cannot be declared const",
                            "Remove the `const`, or move the method to an inherent impl.",
                        ));
                    }
                    let hash = ids.methods[k].0;
                    let site = FnSite { loc: FnLoc::Impl(item_idx as u32, k as u32), owner, self_ty: Some(self_pat), parent: Some(impl_def), hash };
                    let id = add_fn(&mut p, site, &parent_gen, &mut errs);
                    method_ids.push(id);
                    // A body impl's methods see the block items around it.
                    if imp.in_body {
                        p.fn_siblings.insert(id, visible_stmt_items(&p.item_blocks, imp.span));
                    }
                    if static_dyn {
                        p.gx.static_dyn_fns.push(id);
                    }
                    if dyn_self {
                        let key = (crate::check::object::erase_dyn_region(&mut p.tys, self_pat), def.name);
                        if p.gx.dyn_inherent.insert(key, id).is_some() {
                            errs.push(dup(def.name_span, interner.get(def.name)));
                        }
                    }
                    if trait_ref.is_none() {
                        // Inherent methods join the compat method
                        // table under the *template* Adt; instances
                        // get their own entries at materialization.
                        if let Some(owner) = owner {
                            p.methods.entry((owner, def.name)).or_default().push(id);
                            let adt_def = p.gx.adt_info[owner.0 as usize].0;
                            p.gx.inherent_methods.entry((adt_def, def.name)).or_default().push(id);
                            p.gx.inherent_self_pats.insert(id, (self_pat, parent_gen.generics.clone()));
                        } else if let Some(adt_def) = lang_self {
                            p.gx.inherent_methods.entry((adt_def, def.name)).or_default().push(id);
                            p.gx.inherent_self_pats.insert(id, (self_pat, parent_gen.generics.clone()));
                        } else if let Some(prim_ty) = prim_self {
                            p.gx.prim_inherent_methods.entry((prim_ty, def.name)).or_default().push(id);
                            p.gx.inherent_self_pats.insert(id, (self_pat, parent_gen.generics.clone()));
                        }
                    }
                }
                if let Some(tr) = trait_ref {
                    generics::finish_trait_impl(
                        &mut p,
                        imp,
                        impl_def,
                        impl_generics,
                        self_pat,
                        tr,
                        &parent_gen.scope,
                        &method_ids,
                        &mut errs,
                    );
                }
                p.impl_parents.insert(item_idx as u32, parent_gen);
                // Associated consts are evaluated with the const
                // items below (W-RESOLVE phase 2).
            }
            _ => {}
        }
    }
    // Trait default methods parent the items nested in their bodies too.
    for (b, hash, _) in crate::def::trait_body_paths(p.stable_crate, ast, interner) {
        top_hashes.insert(b, hash);
    }
    // Items inside function bodies: nested fns join the fn table (their
    // names are block-scoped, so they stay out of `fn_names`); nested
    // consts and statics evaluate with the top-level pass below. Type
    // items in bodies took passes 1 and 2 above. A nested fn's identity
    // is its parent fn's plus its name (section 1.7), as the driver's
    // slot keys compute it.
    let krate = p.stable_crate;
    let nested_hashes = crate::def::nested_fn_ids(
        ast,
        interner,
        |b| top_hashes.get(&b).copied(),
        |&parent, name, disamb| crate::def::child_hash(parent, crate::def::DefKind::Fn, name, disamb),
        |seg, name| crate::def::free_fn_hash(krate, &[seg], name),
    );
    // Each nested fn's enclosing fn, made once for the loop below.
    let nest_parents = crate::def::nested_fn_parents(ast);
    for (idx, item) in ast.stmt_items.iter().enumerate() {
        match &item.kind {
            ItemKind::Fn(_) => {
                let hash = nested_hashes[idx].expect("every nested fn has an identity");
                // The nested fn's signature resolves against the
                // type items of its enclosing blocks, and may not
                // use the outer functions' parameters (R18).
                let mut parent = plain_parent();
                apply_stmt_overlay(&p, &mut parent.scope, item.span);
                let site = FnSite { loc: FnLoc::Nested(idx as u32), owner: None, self_ty: None, parent: None, hash };
                let id = add_fn(&mut p, site, &parent, &mut errs);
                p.body_items.insert(idx as u32, BodyItem::Fn(id));
                // The def parent chain names the enclosing fn, so
                // paths print through it (`crate::f::inner::Local`);
                // the generics parent stays empty (a nested fn may
                // not use the outer fn's parameters).
                {
                    let ph = match nest_parents[idx].0 {
                        crate::def::NestParent::Top(b) => top_hashes.get(&b).copied(),
                        crate::def::NestParent::Nested(k) => nested_hashes[k as usize],
                        crate::def::NestParent::Unknown => None,
                    };
                    if let Some(ph) = ph
                        && let Some(data) = p.defs.get_mut(DefId { krate: p.krate, hash })
                    {
                        data.parent = Some(DefId { krate: p.krate, hash: ph });
                    }
                }
            }
            ItemKind::Use(_) => {}
            ItemKind::Const(_) | ItemKind::Static(_) => {}
            ItemKind::Struct(_) | ItemKind::Enum(_) | ItemKind::Type(_) => {}
            _ => {
                errs.push(not_yet("L0820", item.span, "type items inside function bodies"));
            }
        }
    }
    block_item_dups(ast, interner, &mut errs);
    block_item_builtin_names(ast, interner, &mut errs);
    // Sibling visibility for nested fns: the items of the block that
    // directly contains each nested fn, and of the blocks around it.
    // The same scope (by stmt_items index) also governs what a nested
    // const or static initializer may name in `eval_const_items`.
    let mut scope_of_stmt: HashMap<u32, Vec<u32>> = HashMap::default();
    let item_blocks: Vec<&ast::Block> = ast.blocks.iter().filter(|b| b.stmts.iter().any(|s| matches!(s, ast::Stmt::Item(_)))).collect();
    p.item_blocks = item_blocks
        .iter()
        .map(|b| (b.span, b.stmts.iter().filter_map(|s| if let ast::Stmt::Item(i) = s { Some(*i) } else { None }).collect()))
        .collect();
    for block in &item_blocks {
        let idxs: Vec<u32> = block
            .stmts
            .iter()
            .filter_map(|s| match s {
                ast::Stmt::Item(i) => Some(*i),
                _ => None,
            })
            .collect();
        if idxs.is_empty() {
            continue;
        }
        // Items in one block share one value namespace, like items in a
        // module: two of one name in the same block are a duplicate.
        let mut seen: Vec<Sym> = Vec::new();
        for &i in &idxs {
            let (name, span) = match &ast.stmt_items[i as usize].kind {
                ItemKind::Fn(def) => (Some(def.name), def.name_span),
                ItemKind::Const(c) => (c.name, c.name_span),
                ItemKind::Static(s) => (Some(s.name), s.name_span),
                _ => (None, ast.stmt_items[i as usize].span),
            };
            if let Some(name) = name {
                if seen.contains(&name) {
                    errs.push(dup(span, interner.get(name)));
                } else {
                    seen.push(name);
                }
            }
        }
        // A nested fn also sees the items of every block around its own
        // (item scopes are not cut by fn bodies): outer blocks first, so
        // the innermost item of a name wins. Blocks nest by span; a block
        // from a macro expansion keeps its own items only.
        let mut sibs: Vec<u32> = Vec::new();
        if block.span.ctxt == crate::diag::CtxtId::ROOT {
            let mut outer: Vec<&ast::Block> = item_blocks
                .iter()
                .copied()
                .filter(|o| {
                    let (s, b) = (o.span, block.span);
                    !std::ptr::eq(*o, *block) && s.ctxt == crate::diag::CtxtId::ROOT && s.file == b.file && s.lo <= b.lo && b.hi <= s.hi
                })
                .collect();
            outer.sort_by_key(|o| o.span.lo);
            for o in outer {
                sibs.extend(o.stmts.iter().filter_map(|s| if let ast::Stmt::Item(i) = s { Some(*i) } else { None }));
            }
        }
        sibs.extend(idxs.iter().copied());
        for &i in &idxs {
            scope_of_stmt.insert(i, sibs.clone());
            if let Some(BodyItem::Fn(fid)) = p.body_items.get(&i).copied() {
                p.fn_siblings.insert(fid, sibs.clone());
            }
        }
    }
    // Adts with a written `Drop` impl, for the const evaluator's
    // drop check: it runs next, before the impl index exists, so
    // the scan reads the written impl heads by name (a same-named
    // Adt in another module is counted too; the check only ever
    // over-rejects a const drop, never miscompiles).
    for item in &ast.items {
        let ItemKind::Impl(imp) = &item.kind else { continue };
        if imp.is_negative {
            continue;
        }
        let Some(tr) = &imp.trait_ref else { continue };
        if tr.path.last().is_none_or(|s| interner.get(s.name) != "Drop") {
            continue;
        }
        let TyExpr::Path(segs) = ast.ty(imp.self_ty) else { continue };
        let Some(name) = segs.last().map(|s| s.name) else { continue };
        if let Some(&a) = p.type_names.get(&name) {
            p.drop_adts.insert(a);
        } else {
            let text = interner.get(name);
            for (i, adt) in p.adts.iter().enumerate() {
                if crate::intern::source_name(interner.get(adt.name)) == text {
                    p.drop_adts.insert(AdtId(i as u32));
                }
            }
        }
    }
    // Const and static items: register names first, then evaluate with the
    // bounded evaluator (ARCH 2.13). A failed item keeps a placeholder value
    // so one bad item reports once; collection fails anyway on the error.
    // This runs before `finish_collect`: its template checks (and, in a
    // build, the instance checks in mono) read consts, statics, and assoc
    // consts from these tables, in generic and plain bodies alike.
    crate::driver::laps::mark("collect/fns+items");
    let const_goals = eval_const_items(&mut p, &mut errs, &mut def_order.statics, &scope_of_stmt);
    crate::driver::laps::mark("collect/const eval");
    p.scope_of_stmt = scope_of_stmt;
    // The generic reuse cache holds while the trait system does (W-PERF);
    // the key needs the resolved trait of each impl, and the cache serves
    // the template checks and discovery that `finish_collect` runs.
    let key = crate::driver::gen_reuse::trait_key(&p, &impl_traits);
    p.gx.reuse.gate(key);
    // Every value is evaluated by now: each body const's joins its owner
    // fn's source, before any stored entry is used.
    fold_body_const_values(&mut p);
    // The spliced source std fns route before mono: its body checks
    // resolve `std::process::exit` and the like as the driver's do. A
    // missing fn is the driver's error (it routes again after collect).
    for def in std_routes {
        if let Some(&f) = p.fn_names.get(&interner.intern(def.mangled)) {
            p.std_fns.push((def.path.iter().map(|s| interner.intern(s)).collect(), f));
        }
    }
    crate::driver::laps::mark("collect/trait key");
    generics::finish_collect(&mut p, &mut errs);
    p.const_goals = const_goals.clone();
    prove_const_goals(&mut p, const_goals, &mut errs);
    crate::driver::laps::mark("collect/const goals");
    if errs.is_empty() {
        crate::check::delta_audit::self_test_if_asked(&mut p);
        crate::check::derive_delta::census_if_asked(&p);
    }
    check_main_termination(&p, &mut errs);

    crate::check::delta_audit::dump_if_asked(&p, errs.len());
    *reuse = std::mem::take(&mut p.gx.reuse);
    def_order.instances = std::mem::take(&mut p.gx.inst_order);
    if errs.is_empty() { Ok(p) } else { Err(errs) }
}

/// Rust's Termination for main: (), ExitCode, or Result<T, E> with T one
/// of those and E: Debug (collect's last check; a derive delta runs it
/// again).
pub(crate) fn check_main_termination(p: &Program, errs: &mut Vec<Diag>) {
    let Some(main) = p.main else { return };
    let f = p.fn_info(main);
    let term = |t: Ty| t == UNIT || p.tys.is_exit_code(t);
    let ret_ok = term(f.ret) || matches!(p.tys.result_args(f.ret), Some((ok, e)) if term(ok) && has_debug(p, &p.tys, e));
    if !f.params.is_empty() || !ret_ok {
        errs.push(Diag::new(
            "L0110",
            f.def(&p.ast).name_span,
            "`main` must take no parameters and return (), ExitCode, or Result<T, E> with T one of those and E: Debug",
            "Write `fn main() { ... }` or `fn main() -> Result<(), E> { ... }`.",
        ));
    }
}

/// The parts of field type `fty` of Adt `id` that rustc's derive bounds
/// on top of `T: Trait` per type param: every path type whose first
/// segment names a type param, with more segments (`T::Item`, also inside
/// generic args), resolved in the Adt's scope. rustc decides this on the
/// written syntax, so `<T as Iterator>::Item` adds no bound.
fn derive_bound_tys(p: &mut Program, ast: &Ast, interner: &Interner, known: &Known, id: AdtId, fty: ast::TyId) -> Vec<Ty> {
    let params: Vec<Sym> =
        p.gx.adt_generics[id.0 as usize].params.iter().filter(|q| q.kind == ParamKind::Type).map(|q| q.name).collect();
    let mut out = Vec::new();
    if params.is_empty() {
        return out;
    }
    let mut stack = vec![fty];
    while let Some(t) = stack.pop() {
        let te = ast.ty(t);
        if let TyExpr::Path(segs) = te
            && segs.len() > 1
            && params.contains(&segs[0].name)
        {
            let cx = p.gx.resolve_cx(&p.type_names, &p.aliases);
            if let Ok(r) = generics::resolve_scoped(ast, interner, known, &cx, &mut p.tys, t, &p.gx.adt_scopes[id.0 as usize])
                && !out.contains(&r)
            {
                out.push(r);
            }
        }
        stack.extend(crate::driver::sig_nodes::ty_children(te));
    }
    out
}

/// Collect's field checks of the derived `Copy` and `Debug` impls, for
/// the Adt rows `rows` takes (collect: all of them; a derive delta: the
/// edited Adt and the rows whose fields reach it). Errors go to `errs`;
/// returns the `Debug` fields that wait for the impl index
/// (`deferred_debug_fields`).
pub(crate) fn check_derive_fields(p: &Program, rows: &dyn Fn(usize) -> bool, errs: &mut Vec<Diag>) -> Vec<(Ty, Span)> {
    let mut deferred_debug: Vec<(Ty, Span)> = Vec::new();
    for i in (0..p.adts.len()).filter(|&i| rows(i)) {
        let adt = &p.adts[i];
        let d = p.derives_of(AdtId(i as u32));
        for v in &adt.variants {
            for f in &v.fields {
                if p.tys.flags(f.ty).contains_param() {
                    // A generic field: the synthetic derive impl
                    // carries the bound (section 2.2), checked per
                    // instance, not against the abstract field type.
                    continue;
                }
                // Each query runs only for a derive the type asks for: the queries
                // are pure, so skipping the others changes no result.
                let mut checks: [(bool, &str); 4] = [(true, ""); 4];
                let mut n_checks = 0;
                // `Clone` fields are checked in finish_collect
                // (`check_derive_bounds`): a field type with a written
                // `Clone` impl needs the impl index first.
                // A field of a local type may be `Copy` through a written
                // impl, which needs the impl index: `check_derive_bounds`
                // proves it.
                if d.copy && !crate::check::fmt::mentions_local_adt(&p.tys, f.ty) {
                    checks[n_checks] = (is_copy(p, &p.tys, f.ty), "Copy");
                    n_checks += 1;
                }
                // `PartialEq` fields are checked in finish_collect too.
                if d.debug {
                    // A field of a type with a written `impl Debug` waits
                    // for the impl index: finish_collect proves it (W-STD-B).
                    let ok = has_debug(p, &p.tys, f.ty);
                    if !ok && crate::check::fmt::mentions_local_adt(&p.tys, f.ty) {
                        deferred_debug.push((f.ty, adt.span));
                    } else {
                        checks[n_checks] = (ok, "Debug");
                        n_checks += 1;
                    }
                }
                for &(ok, name) in &checks[..n_checks] {
                    if !ok {
                        errs.push(Diag::new(
                            "L0112",
                            adt.span,
                            format!("cannot derive {name}: field type `{}` does not implement {name}", ty_name(p, &p.tys, f.ty)),
                            format!("Derive {name} on the field type too, or remove {name} from the derive list."),
                        ));
                    }
                }
            }
        }
    }
    deferred_debug
}

/// The enclosing item's part of a fn signature: the impl's generics and
/// scope (empty for a free fn), and the impl item whose written `F: Fn*`
/// bounds join each method's region signature (W-BORROW). Collect keeps
/// each impl's in `Program::impl_parents`, so a signature delta rebuilds
/// one method the way collect built it.
#[derive(Clone, Default)]
pub(crate) struct ParentGen {
    pub(crate) generics: Generics,
    pub(crate) scope: generics::Scope,
    /// The impl's `ast.items` index, for its written generics.
    pub(crate) impl_item: Option<u32>,
}

/// Where one fn sits and what owns it: `fn_record`'s identity input.
#[derive(Clone, Copy)]
pub(crate) struct FnSite {
    pub(crate) loc: FnLoc,
    pub(crate) owner: Option<AdtId>,
    pub(crate) self_ty: Option<Ty>,
    pub(crate) parent: Option<DefId>,
    pub(crate) hash: DefPathHash,
}

/// What collect learns from one fn's signature, before any table but
/// the type store changes. `install_fn` writes it into the program.
pub(crate) struct FnRecord {
    site: FnSite,
    name: Sym,
    name_span: Span,
    self_kind: Option<SelfKind>,
    is_const: bool,
    params: Vec<Ty>,
    ret: Ty,
    ret_borrows_from: Option<usize>,
    /// The combined generics (the impl's params, then the fn's own).
    generics: Generics,
    env: Vec<Predicate>,
    body_scope: generics::Scope,
    region_sig: Option<crate::borrowck::RegionSig>,
    opaque: Option<(DefId, generics::OpaqueInfo)>,
    outlives: Option<crate::check::outlives::Item>,
}

impl FnRecord {
    /// The fn has generic parameters (its own, or its impl's).
    pub(crate) fn is_template(&self) -> bool {
        !self.generics.params.is_empty()
    }
    /// The combined generics (the impl's params, then the fn's own).
    pub(crate) fn generics(&self) -> &Generics {
        &self.generics
    }
    /// The signature returns `impl Trait`.
    pub(crate) fn has_opaque(&self) -> bool {
        self.opaque.is_some() || self.region_sig.as_ref().is_some_and(|r| r.opaque_caps.is_some())
    }
}

/// Resolves the signature of the fn at `site.loc` under its parent's
/// generics. A fn with generic parameters (its own, or its impl's) is a
/// template: its record holds the signature over `Param` tys, and mono
/// appends a concrete FnInfo per instance after collect. Plain fns are
/// their own only instance. Only `p.tys` changes.
pub(crate) fn fn_record(
    p: &mut Program,
    ast: &Ast,
    interner: &Interner,
    known: &Known,
    site: FnSite,
    parent_gen: &ParentGen,
    errs: &mut Vec<Diag>,
) -> FnRecord {
    let FnSite { loc, self_ty, hash, .. } = site;
    let def = loc.def(ast);
    let shift = parent_gen.generics.params.len() as u32;
    let std = p.in_lang_methods(def.name_span);
    // `T: 'a` bounds: free fns and inherent methods (`ParentGen::scope`
    // of an inherent impl allows them; see `Scope::type_outlives`).
    let mut seed = parent_gen.scope.clone();
    seed.type_outlives |= matches!(loc, FnLoc::Item(_) | FnLoc::Nested(_));
    let (own_generics, own_scope) = {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        generics::build_generics_shifted(ast, interner, known, &cx, &mut p.tys, &def.generics, &def.where_clause, shift, &seed, errs)
    };
    let mut generics = parent_gen.generics.clone();
    generics.params.extend(own_generics.params.iter().cloned());
    generics.predicates.extend(own_generics.predicates.iter().copied());
    generics.parent = site.parent;
    // A parameter may not reuse a name from the enclosing impl or
    // trait (rustc E0403). Duplicates inside one list are reported by
    // `build_generics_shifted`.
    for param in &own_generics.params {
        if parent_gen.generics.params.iter().any(|q| q.name == param.name) {
            errs.push(Diag::new(
                "L0106",
                param.span,
                format!("the parameter `{}` is already used for a parameter of an enclosing item", interner.get(param.name)),
                "Rename one of the parameters.",
            ));
        }
    }
    // `seed` is the parent's scope with this fn's `type_outlives`
    // (`impl Tr + 'a` arguments take `T: 'a` like written params do).
    let mut scope = seed;
    scope.lifetimes.extend(own_scope.lifetimes.iter().copied());
    scope.region_facts.extend(own_scope.region_facts.iter().copied());
    scope.type_params.extend(own_scope.type_params.iter().copied());
    scope.const_params.extend(own_scope.const_params.iter().copied());
    scope.param_tys.extend(own_scope.param_tys.iter().copied());
    scope.assoc_srcs.extend(own_scope.assoc_srcs.iter().cloned());
    // Associated sources from every bound in the combined env,
    // so `T::Assoc` resolves in this signature.
    {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        let mut srcs = generics::assoc_srcs_of(&cx, &p.tys, &generics.predicates);
        scope.assoc_srcs.append(&mut srcs);
    }
    let (params, ret, ret_borrows_from, generics, scope, region_sig, opaque) = {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        let owner = DefId { krate: p.krate, hash };
        generics::resolve_fn_sig(ast, interner, known, &mut p.tys, &cx, def, generics, &scope, self_ty, Some(owner), errs)
    };
    let mut outlives = None;
    {
        let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
        let skip = usize::from(def.self_param.is_some());
        let extra = generics::apit_bounds(ast, def, &params[skip.min(params.len())..], &scope);
        let mut required = generics::regioned_bound_tys(ast, interner, known, &cx, &mut p.tys, &scope, &def.generics, &def.where_clause, &extra);
        let copied_proj = params.iter().chain([&ret]).any(|&t| crate::check::outlives::has_copied_projection(&p.tys, t));
        if !required.is_empty() || copied_proj {
            let implied = generics::implied_sig_tys(ast, interner, known, &cx, &mut p.tys, &scope, def, &params);
            required.extend(crate::check::outlives::copied_projection_selves(&p.tys, &implied));
            let item = crate::check::outlives::Item {
                implied,
                required,
                facts: scope.region_facts.clone(),
                preds: generics.predicates.clone(),
                span: def.name_span,
            };
            if crate::check::outlives::wanted(p, &item) {
                outlives = Some(item);
            }
        }
    }
    if def.self_param.is_some() && self_ty.is_none() {
        errs.push(Diag::new("L0001", def.name_span, "`self` parameter outside an impl", "Move this function into an impl block."));
    }
    let env = generics::body_env(&mut p.tys, &p.gx.index, &generics.predicates);
    let region_sig = region_sig.map(|mut rs| {
        // The impl's lifetimes are params of the impl: early-bound for
        // its methods (a method does not coerce to a fn pointer that
        // is generic over them).
        for e in rs.early.iter_mut().take(parent_gen.scope.lifetimes.len()) {
            *e = true;
        }
        if let Some(i) = parent_gen.impl_item
            && let ItemKind::Impl(imp) = &ast.items[i as usize].kind
        {
            let (g, w) = (&imp.generics, &imp.where_clause);
            let cx = p.gx.resolve_cx_in(&p.type_names, &p.aliases, std);
            rs.fn_bounds.extend(generics::fn_bounds_of(ast, interner, known, &mut p.tys, &cx, &scope, g, w));
            rs.trait_bounds.extend(generics::trait_lts_of(ast, interner, known, &cx, &scope, g, w));
            rs.item_bounds.extend(generics::item_bounds_of(ast, interner, known, &mut p.tys, &cx, &scope, g, w));
            rs.bounds.extend(generics::regioned_bound_preds(ast, interner, known, &cx, &mut p.tys, &scope, g, w, &[]).1);
        }
        rs
    });
    // The body scope of a template keeps its parameters (the
    // abstract-instance check substitutes them); of a plain
    // fn it is empty but for Self and assoc sources.
    let mut body_scope = scope;
    if generics.params.is_empty() {
        body_scope.type_params.clear();
        body_scope.const_params.clear();
    }
    FnRecord {
        site,
        name: def.name,
        name_span: def.name_span,
        self_kind: def.self_param.as_ref().map(|s| s.kind),
        is_const: def.qual.is_const,
        params,
        ret,
        ret_borrows_from,
        generics,
        env,
        body_scope,
        region_sig,
        opaque,
        outlives,
    }
}

/// Where `install_fn` writes a record.
#[derive(Clone, Copy)]
pub(crate) enum Install {
    /// As the next `LocalDefId` (collect).
    Push,
    /// Over this fn's entry (a signature delta, `delta_audit::redo_sig`):
    /// the fn keeps its identity, location, and slots in every table.
    Replace(LocalDefId),
}

/// Writes one fn's record into the program's fn tables and returns its
/// id. Replace leaves the def record's other fields (parent, vis) as they
/// are, and queues the outlives item like collect does: the caller checks
/// `regions.outlives_items` (`outlives::check_all` took collect's).
pub(crate) fn install_fn(p: &mut Program, rec: FnRecord, mode: Install) -> LocalDefId {
    let FnRecord {
        site,
        name,
        name_span,
        self_kind,
        is_const,
        params,
        ret,
        ret_borrows_from,
        generics,
        env,
        body_scope,
        region_sig,
        opaque,
        outlives,
    } = rec;
    let has_opaque = opaque.is_some();
    if let Some((od, info)) = opaque {
        p.gx.opaques.insert(od, info);
    }
    if let Some(item) = outlives {
        p.regions.outlives_items.push(item);
    }
    let is_template = !generics.params.is_empty();
    if is_template {
        p.mark_generic(name_span);
    }
    let id = match mode {
        Install::Push => LocalDefId(p.fns.len() as u32),
        Install::Replace(f) => {
            p.regions.opaque_sigs.remove(&f);
            p.regions.fns.remove(&f);
            f
        }
    };
    if let Some(rs) = region_sig {
        if rs.opaque_caps.is_some() || has_opaque {
            p.regions.opaque_sigs.insert(id, rs);
        } else {
            p.regions.fns.insert(id, rs);
        }
    }
    let def_id = DefId { krate: p.krate, hash: site.hash };
    let template = if is_template { Some(generics::FnTemplate { generics: generics.clone(), env: env.clone() }) } else { None };
    let class = if is_template { generics::FnClass::Template } else { generics::FnClass::Plain };
    let scope = generics::FnScope { env, scope: body_scope };
    if let Install::Replace(f) = mode {
        let i = f.0 as usize;
        debug_assert_eq!(p.fn_defs[i], def_id);
        if let Some(data) = p.defs.get_mut(def_id) {
            data.generics = generics;
            data.sig = SigData::Fn { params: params.clone(), ret };
        }
        let info = &mut p.fns[i];
        (info.self_kind, info.params, info.ret, info.ret_borrows_from, info.is_const) = (self_kind, params, ret, ret_borrows_from, is_const);
        (p.gx.fn_templates[i], p.gx.fn_scopes[i], p.gx.fn_class[i]) = (template, scope, class);
        return f;
    }
    p.fn_defs.push(def_id);
    p.def_by_hash.entry(site.hash).or_insert(id);
    // The def table record mirrors `FnInfo` for identity, kind, and generics;
    // body checking still reads `FnInfo` through the compat path.
    let kind = if site.owner.is_some() { DefsKind::Method } else { DefsKind::Fn };
    let mut data = DefData::skeleton(kind, site.parent, name, Vis::Priv, name_span);
    data.generics = generics;
    data.sig = SigData::Fn { params: params.clone(), ret };
    p.defs.insert(def_id, data);
    p.fns.push(FnInfo { name, loc: site.loc, owner: site.owner, self_kind, params, ret, ret_borrows_from, is_const });
    p.gx.fn_templates.push(template);
    p.gx.fn_scopes.push(scope);
    p.gx.fn_class.push(class);
    id
}

/// Where an associated const's evaluated value lands.
enum PendingAssoc {
    /// An inherent impl const: `assoc_const_names` under its
    /// Adt, with the impl's Self pattern for use-site picks.
    Inherent { owner: AdtId, impl_def: DefId, self_pat: Ty, generics: Generics },
    /// A trait impl const: `impl_assoc_consts` under its impl.
    TraitImpl { impl_def: DefId, assoc: DefId },
    /// A trait default: `trait_assoc_const_values`.
    TraitDefault { trait_def: DefId, assoc: DefId },
}

/// One const or static item waiting for evaluation.
struct PendingConst {
    name: Option<Sym>,
    /// Set for an item inside a function body: its stmt_items index.
    stmt_idx: Option<u32>,
    ty: Ty,
    expr: ast::ExprId,
    span: Span,
    /// Some(mutable) for a static, None for a const.
    is_static: Option<bool>,
    /// The index in `Program::statics` for a static item.
    static_idx: u32,
    /// A static's identity (ARCH 1.7; set with `static_idx`): a top-level
    /// static's item hash, a nested one's `nested_static_hashes` entry.
    def_hash: crate::def::DefPathHash,
    /// Set for an associated const (never a plain-name lookup).
    assoc: Option<PendingAssoc>,
    /// Set for a `thread_local!` static: whether it is the `const`
    /// form. Its value is the key's slot word, never evaluated.
    tls: Option<bool>,
    /// For an item in a function body: the stmt_items indexes its
    /// initializer may name (its block's items and every enclosing
    /// block's, outermost first). Empty for module and associated items.
    scope: Vec<u32>,
}

enum PendingState {
    Todo,
    Doing,
    Done(crate::ty::consteval::ConstValue),
    Failed,
}

/// Registers and evaluates every const and static item in the crate.
/// Static indices come from the session `DefOrder` (see `collect`):
/// top-level statics rank by def hash history, nested statics keep
/// stmt order after them.
/// True when expression `e` (or a type it writes) names one of
/// `names` as a path head: a const param read, a type param's
/// associated item (`T::V`), or `Self::X`. A missed shape only
/// keeps the eager evaluation (and its reject), never a wrong
/// value.
fn expr_names_param(ast: &Ast, e: ast::ExprId, names: &[Sym]) -> bool {
    use ast::Expr;
    let go = |x: ast::ExprId| expr_names_param(ast, x, names);
    match ast.expr(e) {
        Expr::Path(segs) => path_names_param(ast, segs, names),
        Expr::QPath { self_ty, .. } => ty_names_param(ast, *self_ty, names),
        Expr::Binary { lhs, rhs, .. } | Expr::Assign { lhs, rhs } | Expr::AssignOp { lhs, rhs, .. } | Expr::Index { base: lhs, index: rhs } => {
            go(*lhs) || go(*rhs)
        }
        Expr::Unary { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Field { base: expr, .. } | Expr::TupleField { base: expr, .. } => {
            go(*expr)
        }
        Expr::Cast { expr, ty } => go(*expr) || ty_names_param(ast, *ty, names),
        Expr::Call { callee, args } => go(*callee) || args.iter().any(|&a| go(a)),
        Expr::MethodCall { recv, args, generic_args, .. } => {
            go(*recv) || args.iter().any(|&a| go(a)) || generic_args.iter().any(|g| arg_names_param(ast, g, names))
        }
        Expr::Tuple(es) | Expr::Array(es) => es.iter().any(|&x| go(x)),
        Expr::ArrayRepeat { value, len } => go(*value) || go(*len),
        Expr::Block(b) | Expr::UnsafeBlock(b) | Expr::ConstBlock(b) | Expr::Loop { body: b, .. } => block_names_param(ast, *b, names),
        Expr::While { cond, body, .. } => go(*cond) || block_names_param(ast, *body, names),
        Expr::If { cond, then, els } => go(*cond) || block_names_param(ast, *then, names) || els.is_some_and(go),
        Expr::Match { scrut, arms } => go(*scrut) || arms.iter().any(|a| a.guard.is_some_and(go) || go(a.body)),
        Expr::StructLit { path, fields, update } => {
            path_names_param(ast, path, names) || fields.iter().any(|&(_, _, x)| go(x)) || update.is_some_and(go)
        }
        Expr::Range { lo, hi, .. } => lo.is_some_and(go) || hi.is_some_and(go),
        Expr::Return(x) | Expr::Break { value: x, .. } => x.is_some_and(go),
        _ => false,
    }
}

/// `expr_names_param` for a block: its `let` types and values, its
/// statement expressions, and its tail.
fn block_names_param(ast: &Ast, b: ast::BlockId, names: &[Sym]) -> bool {
    let b = ast.block(b);
    b.stmts.iter().any(|s| match s {
        ast::Stmt::Let { ty, init, .. } => ty.is_some_and(|t| ty_names_param(ast, t, names)) || init.is_some_and(|i| expr_names_param(ast, i, names)),
        ast::Stmt::Expr { expr: x, .. } => expr_names_param(ast, *x, names),
        _ => false,
    }) || b.tail.is_some_and(|t| expr_names_param(ast, t, names))
}

/// True when a path's first segment is one of `names`, or a generic
/// argument of any segment names one (`size_of::<V>()`).
fn path_names_param(ast: &Ast, segs: &[ast::PathSeg], names: &[Sym]) -> bool {
    segs.first().is_some_and(|s| names.contains(&s.name))
        || segs.iter().any(|s| match &s.args {
            ast::GenericArgs::Angle(args) => args.iter().any(|g| arg_names_param(ast, g, names)),
            ast::GenericArgs::Parenthesized { inputs, output } => {
                inputs.iter().any(|&t| ty_names_param(ast, t, names)) || output.is_some_and(|t| ty_names_param(ast, t, names))
            }
            ast::GenericArgs::None => false,
        })
}

fn arg_names_param(ast: &Ast, g: &ast::GenericArg, names: &[Sym]) -> bool {
    match g {
        ast::GenericArg::Type(t) => ty_names_param(ast, *t, names),
        ast::GenericArg::Const(x) => expr_names_param(ast, *x, names),
        _ => false,
    }
}

/// `expr_names_param` for a written type.
fn ty_names_param(ast: &Ast, t: ast::TyId, names: &[Sym]) -> bool {
    use ast::TyExpr;
    match ast.ty(t) {
        TyExpr::Path(segs) => path_names_param(ast, segs, names),
        TyExpr::Ref { inner, .. } | TyExpr::Slice(inner) | TyExpr::Ptr { inner, .. } | TyExpr::Paren(inner) => ty_names_param(ast, *inner, names),
        TyExpr::Array { inner, len } => ty_names_param(ast, *inner, names) || expr_names_param(ast, *len, names),
        TyExpr::Tuple(ts) => ts.iter().any(|&x| ty_names_param(ast, x, names)),
        TyExpr::FnPtr { inputs, ret, .. } => inputs.iter().any(|p| ty_names_param(ast, p.ty, names)) || ret.is_some_and(|r| ty_names_param(ast, r, names)),
        _ => false,
    }
}

/// The zero placeholder value of a generic associated const
/// template in `Program::consts` (its real value is per instance).
fn zero_const_value(tys: &crate::ty::GlobalTys, ty: Ty) -> crate::ty::consteval::ConstValue {
    use crate::ty::consteval::ConstValue as CV;
    match tys.kind(ty) {
        TyKind::Bool => CV::Bool(false),
        TyKind::Char => CV::Char('\0'),
        TyKind::Float(f) => CV::Float(0, f),
        TyKind::Unit => CV::Unit,
        _ => CV::Int(0),
    }
}

/// Returns the trait goals of bounded const fn calls, for
/// `prove_const_goals` after `finish_collect`.
fn eval_const_items(
    p: &mut Program,
    errs: &mut Vec<Diag>,
    static_order: &mut Vec<crate::def::DefPathHash>,
    scope_of_stmt: &HashMap<u32, Vec<u32>>,
) -> Vec<(TraitRef, Span)> {
    let Some(ev) = eval_pending_items(p, errs, static_order, scope_of_stmt) else { return Vec::new() };
    p.const_values = pending_values(&ev.pending);
    install_const_items(p, errs, ev.pending, ev.static_count, &ev.tmpls, &ev.quals);
    ev.goals
}

/// The values of evaluated pending items, in order (None: failed).
fn pending_values(pending: &[(PendingConst, PendingState)]) -> Vec<Option<crate::ty::consteval::ConstValue>> {
    pending.iter().map(|(_, st)| if let PendingState::Done(v) = st { Some(v.clone()) } else { None }).collect()
}

/// True when evaluating every const and static item again on `p` (a
/// kept program whose Ast took a const fn body edit) gives the values
/// it holds, with no error: collect would build the same tables. The
/// value tables are set aside meanwhile, as collect has them empty when
/// it evaluates. `static_order` is the session's (`DefOrder::statics`).
pub(crate) fn const_values_hold(p: &mut Program, static_order: &[crate::def::DefPathHash]) -> bool {
    let tables = (
        std::mem::take(&mut p.consts),
        std::mem::take(&mut p.const_names),
        std::mem::take(&mut p.statics),
        std::mem::take(&mut p.static_names),
        std::mem::take(&mut p.assoc_const_names),
    );
    let scope = std::mem::take(&mut p.scope_of_stmt);
    let (mut errs, mut order) = (Vec::new(), static_order.to_vec());
    let (values, goals) = match eval_pending_items(p, &mut errs, &mut order, &scope) {
        Some(ev) => (pending_values(&ev.pending), ev.goals),
        None => (Vec::new(), Vec::new()),
    };
    p.scope_of_stmt = scope;
    (p.consts, p.const_names, p.statics, p.static_names, p.assoc_const_names) = tables;
    errs.is_empty() && values == p.const_values && prove_const_goals(p, goals, &mut errs)
}

/// Registers and evaluates every const and static item (the first half
/// of `eval_const_items`). None when there is no item.
fn eval_pending_items(
    p: &mut Program,
    errs: &mut Vec<Diag>,
    static_order: &mut Vec<crate::def::DefPathHash>,
    scope_of_stmt: &HashMap<u32, Vec<u32>>,
) -> Option<EvaluatedItems> {
    let ast_arc = p.ast.clone();
    let ast = &*ast_arc;
    let mut pending: Vec<(PendingConst, PendingState)> = Vec::new();
    let mut by_name: HashMap<Sym, usize> = HashMap::default();
    let mut static_count = 0u32;
    let mut global_allocs = 0u32;
    for (item, stmt_idx) in ast.items.iter().map(|i| (i, None)).chain(ast.stmt_items.iter().enumerate().map(|(i, it)| (it, Some(i as u32))))
    {
        let (name, ty_id, value, span, is_static) = match &item.kind {
            ItemKind::Const(c) => (c.name, c.ty, c.value, c.span, None),
            ItemKind::Static(s) => (Some(s.name), s.ty, s.value, s.span, Some(s.mutable)),
            _ => continue,
        };
        let Some(value) = value else { continue };
        // Only `'static` and `'_` name a lifetime in a const or static
        // type; the plain resolver below drops lifetimes unchecked.
        if let Some(d) = generics::item_ty_lifetime_err(ast, &p.interner, ty_id, &[], true) {
            errs.push(d);
            continue;
        }
        let tls = if is_static.is_some() { super::tls::key_form(&item.attrs, &p.interner) } else { None };
        let resolved = match stmt_idx {
            Some(_) => resolve_stmt_item_ty(p, ty_id, item.span),
            None => resolve_item_ty(p, ty_id, None),
        };
        let ty = match resolved {
            Ok(t) => {
                // A derive delta reads the written type (`written_projections`).
                if p.tys.flags(t).contains_projection() {
                    p.gx.written_projections.entry(generics::item_types_key(p)).or_default().push(t);
                }
                generics::normalize_item_ty(p, t)
            }
            Err(d) => {
                errs.push(d);
                continue;
            }
        };
        // Dyn compatibility needs `dyn_infos`: checked with the
        // signatures (`object::check_signatures`).
        if crate::check::object::mentions_dyn(&p.tys, ty) {
            p.gx.dyn_impl_sites.push((ty, ast.ty_span(ty_id)));
        }
        if !is_sized_ty(&p.tys, ty) {
            errs.push(Diag::new(
                "L0110",
                span,
                format!("the size for values of type `{}` cannot be known", ty_name(p, &p.tys, ty)),
                "Put the value behind a reference, such as `&[T]` or `&str`.",
            ));
            continue;
        }
        // `#[global_allocator]`: rustc takes one static whose type
        // implements `GlobalAlloc`. Lust takes the known allocator
        // types (std `System`, the jemallocator adapter's
        // `Jemalloc`) and keeps its runtime allocator; a user
        // `GlobalAlloc` impl needs `unsafe`, which lust rejects.
        if item.attrs.iter().any(|a| a.path.len() == 1 && p.interner.get(a.path[0].name) == "global_allocator") {
            if is_static.is_none() {
                errs.push(Diag::new(
                    "L0524",
                    span,
                    "the `global_allocator` attribute cannot be used on constants",
                    "Move the `#[global_allocator]` attribute to a static.",
                ));
                continue;
            }
            global_allocs += 1;
            if global_allocs > 1 {
                errs.push(Diag::new(
                    "L0524",
                    span,
                    "cannot define multiple global allocators",
                    "Keep one `#[global_allocator]` static and remove the others.",
                ));
                continue;
            }
            let known = match crate::ty::Tys::new(&p.tys).local_adt(ty) {
                Some(a) => p.is_known_allocator_adt(a),
                None => false,
            };
            if !known {
                errs.push(Diag::new(
                    "L0112",
                    span,
                    format!("`{}` does not implement `GlobalAlloc`", ty_name(p, &p.tys, ty)),
                    "Use `std::alloc::System` or an allocator adapter crate such as `tikv_jemallocator::Jemalloc`.",
                ));
                continue;
            }
        }
        // A `thread_local!` static's own type is `LocalKey<T>`. Every
        // region in a static's type is `'static`; the key's value type
        // says so, so closure params and results carry it.
        let ty = if tls.is_some() {
            let t = generics::staticize(&mut p.tys, ty);
            p.tys.mk_local_key(t)
        } else {
            ty
        };
        // A static is shared by every thread: its type must be `Sync`
        // (rustc E0277), proven by the auto trait walk (ARCH 2.7).
        if is_static.is_some() {
            let sync = p.gx.lang.sync;
            let mut tys = crate::ty::Tys::new(&p.tys);
            if crate::check::collect::auto_structural(p, &mut tys, &[], sync, ty, &mut AutoSeen::default()) != Some(true) {
                let tn = ty_name(p, &p.tys, ty);
                errs.push(Diag::new(
                    "L0612",
                    span,
                    format!("`{tn}` cannot be shared between threads safely: a static's type must be `Sync`"),
                    "Use a `Sync` type, such as an atomic or a `Mutex`, or a `thread_local!` for per-thread state.",
                ));
                continue;
            }
        }
        // Static indices are assigned after registration below,
        // once every surviving static is known (stable ranks).
        let static_idx = 0;
        if let Some(n) = name {
            let dup = if stmt_idx.is_some() {
                // A nested name is block-scoped: it never enters the
                // by-name map, so only an initializer whose scope holds
                // this item can read it (`PendingConst::scope`).
                false
            } else {
                // The first value of a name owns it, and a unit or tuple
                // struct constructor is a value of the struct's name.
                let ctor_taken = p.type_names.get(&n).is_some_and(|&a| {
                    !p.adts[a.0 as usize].is_enum && p.adts[a.0 as usize].variants.first().is_some_and(|v| v.shape != VariantShape::Named)
                });
                let taken = by_name.contains_key(&n) || p.fn_names.contains_key(&n) || ctor_taken;
                by_name.entry(n).or_insert(pending.len());
                taken
            };
            if dup {
                errs.push(Diag::new(
                    "L0106",
                    span,
                    format!("`{}` is defined more than once", p.display_name(n)),
                    "Rename one of the definitions.",
                ));
            }
        }
        let scope = stmt_idx.and_then(|si| scope_of_stmt.get(&si).cloned()).unwrap_or_default();
        pending.push((
            PendingConst { name, stmt_idx, ty, expr: value, span, is_static, static_idx, def_hash: crate::def::DefPathHash(0), assoc: None, tls, scope },
            PendingState::Todo,
        ));
    }
    // Associated consts evaluate in the same pending list, so a
    // free const can read an inherent associated const (and back)
    // with the same cycle detection as free items.
    let mut assoc_by_key: HashMap<(Sym, Sym), Vec<usize>> = HashMap::default();
    register_assoc_consts(p, errs, &mut pending, &mut assoc_by_key);
    if pending.is_empty() {
        return None;
    }
    // Assign static indices now that registration is complete: every
    // static, nested ones included, by the session rank of its identity
    // (`DefOrder::statics`, append-only), so an added static moves no
    // other static's index. Evaluation reads these indices, so this runs
    // before any pending item evaluates.
    {
        let nested = if pending.iter().any(|(it, _)| it.is_static.is_some() && it.stmt_idx.is_some()) {
            let fns: Vec<(Option<ast::BlockId>, crate::def::DefPathHash)> =
                p.fns.iter().zip(&p.fn_defs).map(|(f, d)| (f.def(&p.ast).body, d.hash)).collect();
            nested_static_hashes(&p.ast, &p.interner, p.stable_crate, &fns)
        } else {
            Vec::new()
        };
        for (it, _) in pending.iter_mut().filter(|(it, _)| it.is_static.is_some()) {
            it.def_hash = match it.stmt_idx {
                None => crate::def::item_hash(p.stable_crate, &[], crate::def::DefKind::Static, p.interner.get(it.name.unwrap()), 0),
                Some(si) => nested[si as usize].expect("a body static has an identity"),
            };
        }
        let hashes: Vec<crate::def::DefPathHash> = pending.iter().filter(|(it, _)| it.is_static.is_some()).map(|(it, _)| it.def_hash).collect();
        let ranks = stable_ranks(static_order, &hashes);
        for ((it, _), &r) in pending.iter_mut().filter(|(it, _)| it.is_static.is_some()).zip(&ranks) {
            it.static_idx = r;
        }
        static_count = ranks.iter().copied().max().map_or(0, |m| m + 1);
    }
    // Generic associated const templates: an initializer that
    // names a parameter of its impl (or trait) has one value per
    // instance, not one value here.
    let tmpls: Vec<Option<Generics>> = pending
        .iter()
        .map(|(it, _)| {
            let generics = match &it.assoc {
                Some(PendingAssoc::Inherent { generics, .. }) => generics.clone(),
                Some(PendingAssoc::TraitImpl { impl_def, .. }) => p.gx.index.by_def(*impl_def).map(|(_, r)| r.generics.clone())?,
                Some(PendingAssoc::TraitDefault { trait_def, .. }) => p.gx.index.trait_record(*trait_def)?.generics.clone(),
                None => return None,
            };
            if generics.params.is_empty() {
                return None;
            }
            let mut names: Vec<Sym> = generics.params.iter().map(|g| g.name).collect();
            names.push(p.known.self_type);
            expr_names_param(ast, it.expr, &names).then_some(generics)
        })
        .collect();
    let n = pending.len();
    let by_stmt: HashMap<u32, usize> = pending.iter().enumerate().filter_map(|(i, (it, _))| it.stmt_idx.map(|si| (si, i))).collect();
    let mut cx = CollectCx {
        p,
        ast,
        pending,
        by_name,
        by_stmt,
        assoc_by_key,
        self_name: None,
        self_pat: None,
        self_stack: Vec::new(),
        cur_scope: Vec::new(),
        scope_stack: Vec::new(),
        next_scope: None,
        module: None,
        errs,
        goals: Vec::new(),
        tmpls,
        quals: HashMap::default(),
        body_modules: HashMap::default(),
    };
    cx.body_modules = body_item_modules(cx.p, &cx.pending);
    for i in 0..n {
        cx.eval_pending(i);
    }
    Some(EvaluatedItems { pending: cx.pending, static_count, tmpls: cx.tmpls, goals: cx.goals, quals: cx.quals })
}

/// The module of each pending body item's initializer (`CollectCx::
/// pending_module`): the module of the smallest fn expression range that
/// holds it, the first such fn in fn order on a tie, among fns whose
/// module is known. One pass over the fns answers every item (the fns do
/// not change while consts evaluate), in place of a pass per item.
fn body_item_modules(p: &Program, pending: &[(PendingConst, PendingState)]) -> HashMap<u32, Option<crate::ty::defs::ModuleId>> {
    let mut out: HashMap<u32, Option<crate::ty::defs::ModuleId>> = HashMap::default();
    let Some(tables) = p.resolved.as_deref() else { return out };
    let mut exprs: Vec<u32> =
        pending.iter().map(|(it, _)| it).filter(|it| it.stmt_idx.is_some() && !tables.const_init_module.contains_key(&it.span)).map(|it| it.expr.0).collect();
    exprs.sort_unstable();
    exprs.dedup();
    if exprs.is_empty() {
        return out;
    }
    let mut best: Vec<Option<(u32, crate::ty::defs::ModuleId)>> = vec![None; exprs.len()];
    for (i, f) in p.fns.iter().enumerate() {
        let (lo, hi) = f.def(&p.ast).exprs;
        let size = hi.wrapping_sub(lo);
        let (a, b) = (exprs.partition_point(|&e| e < lo), exprs.partition_point(|&e| e < hi));
        let mut module: Option<Option<crate::ty::defs::ModuleId>> = None;
        for k in a..b {
            if best[k].is_none_or(|(s, _)| size < s) {
                let m = *module.get_or_insert_with(|| def_module_of_fn(p, i).or_else(|| tables.fn_module.get(i).copied()));
                if let Some(m) = m {
                    best[k] = Some((size, m));
                }
            }
        }
    }
    for (k, e) in exprs.into_iter().enumerate() {
        out.insert(e, best[k].map(|(_, m)| m));
    }
    out
}

/// What `eval_pending_items` gives: each item with its state, the
/// static count, the generic assoc const templates, and the trait
/// goals of bounded const fn calls, proved once the impls are known
/// (`prove_const_goals`).
struct EvaluatedItems {
    pending: Vec<(PendingConst, PendingState)>,
    static_count: u32,
    tmpls: Vec<Option<Generics>>,
    goals: Vec<(TraitRef, Span)>,
    /// The promotion qualifs of evaluated items, by pending index.
    quals: HashMap<usize, crate::ty::constqual::Qualif>,
}

/// Moves evaluated pending items into the program tables, in item order
/// (the second half of `eval_const_items`).
fn install_const_items(
    p: &mut Program,
    errs: &mut Vec<Diag>,
    pending: Vec<(PendingConst, PendingState)>,
    static_count: u32,
    tmpls: &[Option<Generics>],
    quals: &HashMap<usize, crate::ty::constqual::Qualif>,
) {
    let mut statics: Vec<Option<StaticItem>> = (0..static_count).map(|_| None).collect();
    for (pi, (item, state)) in pending.into_iter().enumerate() {
        let PendingState::Done(value) = state else { continue };
        // Not computed (a generic template): qualifies by its type.
        let qualif = quals.get(&pi).copied().unwrap_or(crate::ty::constqual::Qualif::ALL);
        if let Some(assoc) = item.assoc {
            // An associated const is a const item with a child def,
            // found only through its assoc table.
            let name = item.name.unwrap();
            let idx = p.consts.len() as u32;
            let (parent, parent_hash) = match &assoc {
                PendingAssoc::Inherent { impl_def, .. } | PendingAssoc::TraitImpl { impl_def, .. } => (*impl_def, impl_def.hash),
                PendingAssoc::TraitDefault { trait_def, .. } => (*trait_def, trait_def.hash),
            };
            let hash = crate::def::child_hash(parent_hash, crate::def::DefKind::Const, p.interner.get(name), 0);
            let def = DefId { krate: p.krate, hash };
            p.defs.insert(def, DefData::skeleton(DefsKind::Const, Some(parent), name, Vis::Priv, item.span));
            p.const_defs.push(def);
            p.consts.push(ConstItem { name, ty: item.ty, value, qualif, value_fp: Default::default() });
            if let Some(generics) = &tmpls[pi] {
                p.gx.assoc_const_tmpls.insert(idx, (item.expr, generics.clone()));
            }
            match assoc {
                PendingAssoc::Inherent { owner, self_pat, generics, .. } => {
                    p.assoc_const_names.entry((owner, name)).or_default().push(idx);
                    p.gx.inherent_const_self.insert(idx, (self_pat, generics, item.span));
                }
                PendingAssoc::TraitImpl { impl_def, assoc } => {
                    p.impl_assoc_consts.insert((impl_def, assoc), idx);
                }
                PendingAssoc::TraitDefault { assoc, .. } => {
                    p.trait_assoc_const_values.insert(assoc, idx);
                }
            }
            continue;
        }
        match item.is_static {
            None if item.stmt_idx.is_some() => {
                if let Some(name) = item.name {
                    let idx = p.body_consts.len() as u32;
                    p.body_consts.push(ConstItem { name, ty: item.ty, value, qualif, value_fp: Default::default() });
                    p.body_items.insert(item.stmt_idx.unwrap(), BodyItem::Const(idx));
                }
            }
            None => {
                if let Some(name) = item.name {
                    let idx = p.consts.len() as u32;
                    p.const_names.insert(name, idx);
                    let disamb = 0;
                    let hash = crate::def::item_hash(p.stable_crate, &[], crate::def::DefKind::Const, p.interner.get(name), disamb);
                    let def = DefId { krate: p.krate, hash };
                    p.defs.insert(def, DefData::skeleton(DefsKind::Const, None, name, Vis::Priv, item.span));
                    p.const_defs.push(def);
                    p.consts.push(ConstItem { name, ty: item.ty, value, qualif, value_fp: Default::default() });
                }
            }
            Some(mutable) => {
                let name = item.name.unwrap();
                // A shared static must be `Sync` (rustc E0277); a type
                // whose answer needs inference stays accepted here.
                if !mutable {
                    let mut tys = crate::ty::Tys::new(&p.tys);
                    if auto_structural(p, &mut tys, &[], p.gx.lang.sync, item.ty, &mut AutoSeen::default()) == Some(false) {
                        errs.push(Diag::new(
                            "L0612",
                            item.span,
                            format!("`{}` cannot be shared between threads safely", ty_name(p, &p.tys, item.ty)),
                            "A `static` needs a `Sync` type; use a `const`, a `thread_local!`, or a `Sync` wrapper.",
                        ));
                    }
                }
                if let Some(sidx) = item.stmt_idx {
                    p.body_items.insert(sidx, BodyItem::Static(item.static_idx));
                }
                // Top-level statics join the crate name tables; nested ones
                // resolve through the checker's block item scopes instead.
                if item.stmt_idx.is_none() {
                    let hash = crate::def::item_hash(p.stable_crate, &[], crate::def::DefKind::Static, p.interner.get(name), 0);
                    let def = DefId { krate: p.krate, hash };
                    p.defs.insert(def, DefData::skeleton(DefsKind::Static, None, name, Vis::Priv, item.span));
                    if p.static_defs.len() <= item.static_idx as usize {
                        p.static_defs.resize(item.static_idx as usize + 1, DefId { krate: p.krate, hash: crate::def::DefPathHash(0) });
                    }
                    p.static_defs[item.static_idx as usize] = def;
                    p.static_names.insert(name, item.static_idx);
                } else {
                    // A nested static's identity (ARCH 1.7): its C symbol
                    // and cached IR name it by this hash, so two nested
                    // statics must never share one.
                    let def = DefId { krate: p.krate, hash: item.def_hash };
                    if p.static_defs.len() <= item.static_idx as usize {
                        p.static_defs.resize(item.static_idx as usize + 1, DefId { krate: p.krate, hash: crate::def::DefPathHash(0) });
                    }
                    p.static_defs[item.static_idx as usize] = def;
                }
                let tls = match item.tls {
                    Some(eager) => match super::tls::init_fn(p, item.expr, item.stmt_idx) {
                        Some(init) => Some(super::tls::TlsKey { init, eager }),
                        None => {
                            errs.push(Diag::new("L0806", item.span, "thread_local! static without its init fn", "Declare the static inside thread_local!."));
                            None
                        }
                    },
                    None => None,
                };
                statics[item.static_idx as usize] = Some(StaticItem { name, ty: item.ty, mutable, value, tls, value_fp: Default::default() });
            }
        }
    }
    p.statics = statics
        .into_iter()
        .map(|s| s.unwrap_or(StaticItem { name: Sym(0), ty: ERROR, mutable: false, value: crate::ty::consteval::ConstValue::Unit, tls: None , value_fp: Default::default() }))
        .collect();
    // Each `thread_local!` key's slot word: 1, 2, .. in static order
    // (runtime/thread.c finds the thread's cell by it).
    let mut slot = 0i128;
    for st in p.statics.iter_mut().filter(|s| s.tls.is_some()) {
        slot += 1;
        st.value = crate::ty::consteval::ConstValue::Int(slot);
        st.value_fp = Default::default();
    }
}

/// The def hash of the static at `stmt_items[sidx]` (ARCH 1.7: a static
/// in a body is its parent fn's hash, its name, and its ordinal among the
/// same-name statics of that fn). The parent is the innermost fn whose
/// body holds the item; edits elsewhere in the file do not move it.
/// Top-level fn and method hashes by body block, computed exactly
/// as the fn collect loop below computes them (free fns by name;
/// methods by impl header ordinal and name). Pass 1 needs them
/// before that loop runs, as parents of type items in bodies.
pub(crate) fn top_fn_hashes(ast: &Ast, interner: &Interner, stable: crate::def::StableCrateHash) -> HashMap<ast::BlockId, crate::def::DefPathHash> {
    let mut out: HashMap<ast::BlockId, crate::def::DefPathHash> = HashMap::default();
    let impl_ids = crate::def::impl_ids(stable, ast, interner);
    for (item_idx, item) in ast.items.iter().enumerate() {
        match &item.kind {
            ItemKind::Fn(def) => {
                if let Some(b) = def.body {
                    out.insert(b, crate::def::free_fn_hash(stable, &[], interner.get(def.name)));
                }
            }
            ItemKind::Impl(imp) => {
                if imp.is_negative {
                    continue;
                }
                let ids = impl_ids[item_idx].as_ref().expect("an impl item has impl ids");
                for (k, def) in imp.fns.iter().enumerate() {
                    if let Some(b) = def.body {
                        out.insert(b, ids.methods[k].0);
                    }
                }
            }
            _ => {}
        }
    }
    for (b, hash, _) in crate::def::trait_body_paths(stable, ast, interner) {
        out.insert(b, hash);
    }
    out
}

/// The stmt item indexes visible at `at`: the items of every block
/// whose span holds `at`, outermost block first (so a later entry
/// shadows an earlier one of the same name).
/// Two items of one name in one namespace in one block are rustc
/// E0428 (L0106): a type item binds the type namespace, a fn, const,
/// or static the value namespace, and a unit or tuple struct both.
/// The resolver reports these too, but a single-file program that
/// skips it (`needs_resolve`) is checked only here. One report per
/// duplicate item, at its name.
fn block_item_dups(ast: &Ast, interner: &Interner, errs: &mut Vec<Diag>) {
    for block in &ast.blocks {
        let mut seen: Vec<(bool, Sym)> = Vec::new();
        for s in &block.stmts {
            let ast::Stmt::Item(i) = s else { continue };
            let item = &ast.stmt_items[*i as usize];
            let (name, span, ty, val) = match &item.kind {
                ItemKind::Struct(st) => (st.name, st.name_span, true, !matches!(st.fields, ast::VariantFields::Struct(_))),
                ItemKind::Enum(e) => (e.name, e.name_span, true, false),
                ItemKind::Union(u) => (u.name, u.name_span, true, false),
                ItemKind::Trait(t) => (t.name, t.name_span, true, false),
                ItemKind::Type(a) => (a.name, a.name_span, true, false),
                ItemKind::Fn(f) => (f.name, f.name_span, false, true),
                ItemKind::Const(c) => match c.name {
                    Some(n) => (n, c.name_span, false, true),
                    None => continue,
                },
                ItemKind::Static(st) => (st.name, st.name_span, false, true),
                _ => continue,
            };
            let keys: Vec<(bool, Sym)> = [(true, ty), (false, val)].iter().filter(|k| k.1).map(|k| (k.0, name)).collect();
            if keys.iter().any(|k| seen.contains(k)) {
                errs.push(Diag::new(
                    "L0106",
                    span,
                    format!("`{}` is defined more than once", interner.get(name)),
                    "Rename one of the definitions.",
                ));
            }
            seen.extend(keys);
        }
    }
}

/// A block type item named like a primitive type, a std prelude type
/// or trait, or a std type lust knows by name is not supported yet
/// (L0820): check reads those names by their text in many value paths
/// (`u8::default()`, `String::new()`, `Rc::new(..)`) before any block
/// overlay, so the block item would lose to the builtin (a wrong
/// output: rustc calls the block item).
fn block_item_builtin_names(ast: &Ast, interner: &Interner, errs: &mut Vec<Diag>) {
    const PRIMS: &[&str] =
        &["u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32", "f64", "bool", "char", "str"];
    for item in &ast.stmt_items {
        let (name, span) = match &item.kind {
            ItemKind::Struct(s) => (s.name, s.name_span),
            ItemKind::Enum(e) => (e.name, e.name_span),
            ItemKind::Union(u) => (u.name, u.name_span),
            ItemKind::Trait(t) => (t.name, t.name_span),
            ItemKind::Type(a) => (a.name, a.name_span),
            _ => continue,
        };
        let n = interner.get(name);
        if PRIMS.contains(&n)
            || crate::resolve::imports::in_std_prelude(n, crate::driver::graph::Namespace::Type)
            || crate::ty::lang::needs_std_use(n)
        {
            errs.push(not_yet("L0820", span, "a block type item named like a builtin or std type"));
        }
    }
    // A block item named like a generic parameter of an enclosing fn
    // shadows it in rustc, but a mono instance reads the name as the
    // parameter's argument (`struct T` in `fn f<T>` printed the
    // argument's `T::default()`, a wrong output). Only names some
    // generic list declares take the per-item scope walk.
    let mut param_names: Vec<Sym> = Vec::new();
    let mut add = |g: &ast::Generics| {
        for q in &g.params {
            if let ast::GenericParam::Type { name, .. } | ast::GenericParam::Const { name, .. } = q {
                param_names.push(*name);
            }
        }
    };
    for item in ast.items.iter().chain(&ast.stmt_items) {
        match &item.kind {
            ItemKind::Fn(f) => add(&f.generics),
            ItemKind::Impl(imp) => {
                add(&imp.generics);
                imp.fns.iter().for_each(|f| add(&f.generics));
            }
            ItemKind::Trait(t) => {
                add(&t.generics);
                t.fns.iter().for_each(|f| add(&f.generics));
            }
            _ => {}
        }
    }
    if param_names.is_empty() {
        return;
    }
    for item in &ast.stmt_items {
        let (name, span) = match &item.kind {
            ItemKind::Struct(s) => (s.name, s.name_span),
            ItemKind::Enum(e) => (e.name, e.name_span),
            ItemKind::Union(u) => (u.name, u.name_span),
            ItemKind::Trait(t) => (t.name, t.name_span),
            ItemKind::Type(a) => (a.name, a.name_span),
            ItemKind::Const(c) => match c.name {
                Some(n) => (n, c.name_span),
                None => continue,
            },
            ItemKind::Static(st) => (st.name, st.name_span),
            ItemKind::Fn(f) => (f.name, f.name_span),
            _ => continue,
        };
        if param_names.contains(&name) && outer_param_names(ast, item.span).contains(&name) {
            errs.push(not_yet("L0820", span, "a block item named like a generic parameter of the enclosing fn"));
        }
    }
}

fn visible_stmt_items(item_blocks: &[(Span, Vec<u32>)], at: Span) -> Vec<u32> {
    let mut blocks: Vec<&(Span, Vec<u32>)> = item_blocks
        .iter()
        .filter(|(s, _)| s.file == at.file && s.ctxt == at.ctxt && s.lo <= at.lo && at.hi <= s.hi)
        .collect();
    blocks.sort_by_key(|(s, _)| std::cmp::Reverse(s.hi - s.lo));
    blocks.iter().flat_map(|(_, idxs)| idxs.iter().copied()).collect()
}

/// The resolve-scope overlay of the type items visible at `at`
/// (R18): every type item of every block holding `at`, outermost
/// first, split by kind for `generics::Scope`'s local tables.
fn stmt_type_overlay(
    p: &Program,
    at: Span,
) -> (Vec<(Sym, AdtId, Option<ast::Generics>)>, Vec<(Sym, Ty)>, Vec<(Sym, DefId)>) {
    overlay_of_items(p, visible_stmt_items(&p.item_blocks, at))
}

/// One block layer's type tables (`generics::Scope::local_types`,
/// `local_aliases`, `local_traits`).
pub(crate) type TypeOverlay = (Vec<(Sym, AdtId, Option<ast::Generics>)>, Vec<(Sym, Ty)>, Vec<(Sym, DefId)>);

/// Appends an inner block's overlay to `acc`. The three tables share
/// the type namespace, so an inner entry first drops every outer entry
/// of its name from all three: each name keeps only its innermost
/// item, whatever kind the outer one was.
pub(crate) fn overlay_extend(acc: &mut TypeOverlay, inner: TypeOverlay) {
    let names: Vec<Sym> = inner.0.iter().map(|e| e.0).chain(inner.1.iter().map(|e| e.0)).chain(inner.2.iter().map(|e| e.0)).collect();
    if !names.is_empty() {
        acc.0.retain(|e| !names.contains(&e.0));
        acc.1.retain(|e| !names.contains(&e.0));
        acc.2.retain(|e| !names.contains(&e.0));
    }
    acc.0.extend(inner.0);
    acc.1.extend(inner.1);
    acc.2.extend(inner.2);
}

/// The overlay entries of the given stmt items, in order, an inner
/// block's entries replacing outer ones of the same name
/// (`overlay_extend`: `visible_stmt_items` lists the outermost block
/// first, and one block holds one item per name, rustc E0428).
pub(crate) fn overlay_of_items(p: &Program, idxs: Vec<u32>) -> TypeOverlay {
    let mut acc: TypeOverlay = Default::default();
    for si in idxs {
        let mut one: TypeOverlay = Default::default();
        overlay_item(p, si, &mut one);
        overlay_extend(&mut acc, one);
    }
    acc
}

/// One stmt item's overlay entry, if it is a collected type item.
fn overlay_item(p: &Program, si: u32, (types, aliases, traits): &mut TypeOverlay) {
    let Some(bti) = p.body_type_items.get(&si) else { return };
    let item = &p.ast.stmt_items[si as usize];
    match (*bti, &item.kind) {
        (BodyTypeItem::Adt(id), ItemKind::Struct(s)) => types.push((s.name, id, Some(s.generics.clone()))),
        (BodyTypeItem::Adt(id), ItemKind::Enum(e)) => types.push((e.name, id, Some(e.generics.clone()))),
        (BodyTypeItem::Alias(t), ItemKind::Type(a)) => aliases.push((a.name, t)),
        (BodyTypeItem::Trait(d), ItemKind::Trait(t)) => traits.push((t.name, d)),
        _ => {}
    }
}

/// Sets the body-local overlay of `scope` for an item or signature
/// at `at`, plus the generic parameter names of every enclosing
/// function (which an item in a body may not use, E0401).
pub(crate) fn apply_stmt_overlay(p: &Program, scope: &mut generics::Scope, at: Span) {
    let (types, aliases, traits) = stmt_type_overlay(p, at);
    scope.local_types = types;
    scope.local_aliases = aliases;
    scope.local_traits = traits;
    scope.outer_params = outer_param_names_in(p, at);
}

/// The fn bodies of one Ast, by file, in item order: each body's span and
/// the generic param names `outer_param_names` takes for it.
pub(crate) struct OuterParams {
    by_file: HashMap<crate::diag::FileId, Vec<(Span, Vec<Sym>)>>,
}

impl OuterParams {
    fn new(ast: &Ast) -> OuterParams {
        let mut by_file: HashMap<crate::diag::FileId, Vec<(Span, Vec<Sym>)>> = HashMap::default();
        let names = |gs: &[&ast::Generics]| -> Vec<Sym> {
            gs.iter()
                .flat_map(|g| g.params.iter())
                .filter_map(|p| match p {
                    ast::GenericParam::Type { name, .. } | ast::GenericParam::Const { name, .. } => Some(*name),
                    ast::GenericParam::Lifetime { .. } => None,
                })
                .collect()
        };
        let mut add = |b: Option<ast::BlockId>, gs: &[&ast::Generics]| {
            if let Some(b) = b {
                let s = ast.block(b).span;
                by_file.entry(s.file).or_default().push((s, names(gs)));
            }
        };
        for item in &ast.items {
            match &item.kind {
                ItemKind::Fn(f) => add(f.body, &[&f.generics]),
                ItemKind::Impl(imp) => imp.fns.iter().for_each(|f| add(f.body, &[&imp.generics, &f.generics])),
                ItemKind::Trait(t) => t.fns.iter().for_each(|f| add(f.body, &[&t.generics, &f.generics])),
                _ => {}
            }
        }
        for item in &ast.stmt_items {
            if let ItemKind::Fn(f) = &item.kind {
                add(f.body, &[&f.generics]);
            }
        }
        OuterParams { by_file }
    }
}

/// `outer_param_names` through the program's index of fn bodies
/// (`Program::outer_params`): the same names in the same order, from the
/// bodies of `at`'s file only.
fn outer_param_names_in(p: &Program, at: Span) -> Vec<Sym> {
    let key = Arc::as_ptr(&p.ast) as usize;
    let held = p.outer_params.read().unwrap_or_else(|e| e.into_inner()).as_ref().filter(|(k, _)| *k == key).map(|(_, ix)| ix.clone());
    let ix = held.unwrap_or_else(|| {
        let ix = Arc::new(OuterParams::new(&p.ast));
        *p.outer_params.write().unwrap_or_else(|e| e.into_inner()) = Some((key, ix.clone()));
        ix
    });
    let mut out: Vec<Sym> = Vec::new();
    for (s, names) in ix.by_file.get(&at.file).map_or(&[][..], |v| &v[..]) {
        if s.ctxt == at.ctxt && s.lo <= at.lo && at.hi <= s.hi && (s.lo != at.lo || s.hi != at.hi || s.lo != s.hi) {
            out.extend_from_slice(names);
        }
    }
    out
}

/// The type and const generic parameter names of every function
/// whose body span holds `at` (impl generics included for methods).
fn outer_param_names(ast: &Ast, at: Span) -> Vec<Sym> {
    let mut out: Vec<Sym> = Vec::new();
    let mut take = |g: &ast::Generics, out: &mut Vec<Sym>| {
        for p in &g.params {
            match p {
                ast::GenericParam::Type { name, .. } => out.push(*name),
                ast::GenericParam::Const { name, .. } => out.push(*name),
                ast::GenericParam::Lifetime { .. } => {}
            }
        }
    };
    let holds = |b: Option<ast::BlockId>| {
        b.is_some_and(|b| {
            let s = ast.block(b).span;
            s.file == at.file && s.ctxt == at.ctxt && s.lo <= at.lo && at.hi <= s.hi && (s.lo != at.lo || s.hi != at.hi || s.lo != s.hi)
        })
    };
    for item in &ast.items {
        match &item.kind {
            ItemKind::Fn(f) if holds(f.body) => take(&f.generics, &mut out),
            ItemKind::Impl(imp) => {
                for f in &imp.fns {
                    if holds(f.body) {
                        take(&imp.generics, &mut out);
                        take(&f.generics, &mut out);
                    }
                }
            }
            ItemKind::Trait(t) => {
                for f in &t.fns {
                    if holds(f.body) {
                        take(&t.generics, &mut out);
                        take(&f.generics, &mut out);
                    }
                }
            }
            _ => {}
        }
    }
    for item in &ast.stmt_items {
        if let ItemKind::Fn(f) = &item.kind
            && holds(f.body)
        {
            take(&f.generics, &mut out);
        }
    }
    out
}

/// The identity of each static inside a fn body (ARCH 1.7), by
/// `Ast::stmt_items` index (None for any other item): the innermost fn
/// whose body holds it by span, folded with its name and its ordinal
/// among that fn's same-name statics; statics no fn body holds fold from
/// one `#body` root. `fns` is every fn in LocalDefId order (body, def
/// hash): collect passes its own tables (instances after the AST fns),
/// and the driver its fn keys, so both get the same identities
/// (`predicted_def_ids`) and a static keeps its identity, and so its
/// session rank, when other items come or go.
pub(crate) fn nested_static_hashes(
    ast: &Ast,
    interner: &Interner,
    stable: crate::def::StableCrateHash,
    fns: &[(Option<ast::BlockId>, crate::def::DefPathHash)],
) -> Vec<Option<crate::def::DefPathHash>> {
    let mut out = vec![None; ast.stmt_items.len()];
    let statics: Vec<usize> = (0..ast.stmt_items.len()).filter(|&j| matches!(ast.stmt_items[j].kind, ItemKind::Static(_))).collect();
    if statics.is_empty() {
        return out;
    }
    let parent = stmt_item_parents(ast, fns, &statics);
    let root = crate::def::item_hash(stable, &["#body"], crate::def::DefKind::Static, "", 0);
    let mut seen: HashMap<(Option<usize>, Sym), u32> = HashMap::default();
    for &j in &statics {
        let ItemKind::Static(st) = &ast.stmt_items[j].kind else { continue };
        let ordinal = seen.entry((parent[j], st.name)).or_insert(0);
        let parent_hash = parent[j].map_or(root, |i| fns[i].1);
        out[j] = Some(crate::def::child_hash(parent_hash, crate::def::DefKind::Static, interner.get(st.name), *ordinal));
        *ordinal += 1;
    }
    out
}

/// The fn whose body holds each of the body items `items` (by
/// `Ast::stmt_items` index; None for the others): the innermost by span.
/// fn bodies sorted by (file, start, longest first) form a nesting; a sweep
/// keeps the bodies open at each item. An instance shares its template's
/// body: the first fn with a body (the template, in LocalDefId order) is
/// the parent. `fns` is every fn in LocalDefId order (body, def hash).
pub(crate) fn stmt_item_parents(ast: &Ast, fns: &[(Option<ast::BlockId>, crate::def::DefPathHash)], items: &[usize]) -> Vec<Option<usize>> {
    let mut events: Vec<(u32, u32, std::cmp::Reverse<u32>, Option<usize>, usize)> = Vec::new();
    let mut bodies: std::collections::HashSet<ast::BlockId> = Default::default();
    for (i, (body, _)) in fns.iter().enumerate() {
        if let Some(b) = body
            && bodies.insert(*b)
        {
            let s = ast.block(*b).span;
            events.push((s.file.0, s.lo, std::cmp::Reverse(s.hi), Some(i), 0));
        }
    }
    for &j in items {
        let s = ast.stmt_items[j].span;
        events.push((s.file.0, s.lo, std::cmp::Reverse(s.hi), None, j));
    }
    events.sort_unstable();
    let mut parent: Vec<Option<usize>> = vec![None; ast.stmt_items.len()];
    let mut open: Vec<(u32, u32, usize)> = Vec::new();
    for (file, lo, std::cmp::Reverse(hi), f, j) in events {
        while open.last().is_some_and(|&(of, ohi, _)| of != file || ohi <= lo) {
            open.pop();
        }
        match f {
            Some(i) => open.push((file, hi, i)),
            None => parent[j] = open.iter().rev().find(|&&(_, ohi, _)| hi <= ohi).map(|&(_, _, i)| i),
        }
    }
    parent
}

/// Folds the value of each body const into the source fingerprint of the
/// fn whose body holds it (`GenReuse::ast_src`), right after collect
/// evaluates it: a body const has no def a read could name, so a const fn
/// edit that moves its value rechecks its owner, and what replays the
/// owner's check (its instances and closures), through their sources.
fn fold_body_const_values(p: &mut Program) {
    if p.body_consts.is_empty() || p.gx.reuse.ast_src.is_empty() {
        return;
    }
    let mut consts: Vec<(u32, u32)> = p.body_items.iter().filter_map(|(&j, it)| if let BodyItem::Const(ci) = *it { Some((j, ci)) } else { None }).collect();
    consts.sort_unstable();
    let fns: Vec<(Option<ast::BlockId>, crate::def::DefPathHash)> = p.fns.iter().zip(&p.fn_defs).map(|(f, d)| (f.def(&p.ast).body, d.hash)).collect();
    let items: Vec<usize> = consts.iter().map(|&(j, _)| j as usize).collect();
    let parent = stmt_item_parents(&p.ast, &fns, &items);
    let owned: Vec<(usize, u32)> = consts.iter().filter_map(|&(j, ci)| Some((parent[j as usize]?, ci))).filter(|&(_, ci)| (ci as usize) < p.body_consts.len()).collect();
    // The value fingerprints are independent (each made once, `value_fp`):
    // in parallel, then folded in order.
    let fps = {
        let p = &*p;
        crate::pool::par_map(&owned, |&(_, ci)| {
            let c = &p.body_consts[ci as usize];
            crate::driver::graph::value_fp(p, &c.value, &c.value_fp)
        })
    };
    for (&(owner, _), fp) in owned.iter().zip(fps) {
        if let Some(src) = p.gx.reuse.ast_src.get_mut(owner) {
            let mut h = crate::intern::FxHasher::default();
            std::hash::Hash::hash(&(*src, fp), &mut h);
            *src = std::hash::Hasher::finish(&h);
        }
    }
}

/// The owner Adt of an impl hoisted out of a fn body (`ImplDef::in_body`),
/// read from the Self type the fn pass resolved in the block overlay: its
/// written name may be a block type that shadows a module type of that
/// name, so `type_names` cannot answer.
fn body_impl_owner(p: &Program, impl_def: DefId) -> Option<AdtId> {
    let self_ty = match p.gx.inherent_impl_self.get(&impl_def) {
        Some(&(t, _)) => t,
        None => p.gx.index.by_def(impl_def)?.1.self_ty(&p.tys),
    };
    match p.tys.kind(self_ty) {
        TyKind::Adt(def, _) => p.defs.local_adt(def),
        _ => None,
    }
}

/// Registers the associated consts of inherent impls, trait
/// impls, and trait defaults as pending items of
/// `eval_const_items` (W-RESOLVE phase 2, on the merged
/// evaluator). They become const items with child defs under
/// their impl or trait, found through their assoc tables only:
/// initializers read free consts by plain name and inherent
/// associated consts by `Type::CONST` (the `assoc_by_key`
/// entries, keyed by flat type name), never other associated
/// consts by plain name (Rust requires a qualified path there).
/// Impl ordinals must match the fn pass exactly (same
/// iteration, same owner test), so child hashes name the same
/// impl.
fn register_assoc_consts(
    p: &mut Program,
    errs: &mut Vec<Diag>,
    pending: &mut Vec<(PendingConst, PendingState)>,
    assoc_by_key: &mut HashMap<(Sym, Sym), Vec<usize>>,
) {
    let ast_arc = p.ast.clone();
    let ast = &*ast_arc;
    // The impl paths the fn pass gave each impl (`def::impl_ids`).
    let impl_ids = crate::def::impl_ids(p.stable_crate, ast, &p.interner);
    let mut inherent_seen: Vec<(DefId, Sym)> = Vec::new();
    for (item_idx, item) in ast.items.iter().enumerate() {
        let ItemKind::Impl(imp) = &item.kind else { continue };
        if imp.is_negative || imp.consts.is_empty() {
            continue;
        }
        let parent_hash = impl_ids[item_idx].as_ref().expect("an impl item has impl ids").path.0;
        let impl_def = DefId { krate: p.krate, hash: parent_hash };
        let owner = match ast.ty(imp.self_ty) {
            _ if imp.in_body => body_impl_owner(p, impl_def),
            TyExpr::Path(segs) if segs.len() == 1 => p.type_names.get(&segs[0].name).copied(),
            _ => None,
        };
        if imp.trait_ref.is_some() {
            // A trait impl const: its trait and assoc def come
            // from the index record the fn pass inserted.
            let Some((_, rec)) = p.gx.index.by_def(impl_def) else { continue };
            let trait_def = rec.trait_def;
            let Some(trec) = p.gx.index.trait_record(trait_def).cloned() else { continue };
            for c in &imp.consts {
                // Unknown and duplicate names are reported by
                // `finish_trait_impl`; only value items register.
                let Some(assoc) = trec.assoc_by_name(c.name).filter(|a| p.gx.trait_assoc_consts.contains_key(a)) else {
                    continue;
                };
                if pending
                    .iter()
                    .any(|(it, _)| matches!(it.assoc, Some(PendingAssoc::TraitImpl { impl_def: d, assoc: a }) if d == impl_def && a == assoc))
                {
                    continue;
                }
                let self_ty = owner.map(|o| p.tys.adt_ty(o));
                let ty = match resolve_item_ty(p, c.ty, self_ty) {
                    Ok(t) => t,
                    Err(d) => match generics::impl_const_decl_ty(p, impl_def, assoc) {
                        // The scoped check in `finish_trait_impl`
                        // already validated the written type; a
                        // plain resolve failure here only means it
                        // names the impl's parameters or `Self::A`,
                        // so the trait's type (with this impl's
                        // bindings) stands in.
                        Some(t) => t,
                        None => {
                            errs.push(d);
                            continue;
                        }
                    },
                };
                let Some(value) = c.default else {
                    errs.push(not_yet("L0632", c.name_span, "associated consts without a value"));
                    continue;
                };
                pending.push((
                    PendingConst {
                        name: Some(c.name),
                        ty,
                        expr: value,
                        span: c.span,
                        is_static: None,
                        static_idx: 0,
                        def_hash: crate::def::DefPathHash(0),
                        stmt_idx: None,
                        assoc: Some(PendingAssoc::TraitImpl { impl_def, assoc }),
                        tls: None,
                        scope: Vec::new(),
                    },
                    PendingState::Todo,
                ));
            }
            continue;
        }
        let Some(owner) = owner else { continue };
        let self_ty = p.tys.adt_ty(owner);
        // Disjoint impls may define the same name (the overlap
        // check in `finish_collect` rejects the ones that can
        // both apply, methods included); two in one impl block
        // cannot.
        let (self_pat, impl_generics) = p.gx.inherent_impl_self.get(&impl_def).cloned().unwrap_or((self_ty, Generics::default()));
        for c in &imp.consts {
            if inherent_seen.contains(&(impl_def, c.name)) {
                errs.push(Diag::new(
                    "L0106",
                    c.name_span,
                    format!("`{}` is defined more than once", p.interner.get(c.name)),
                    "Rename one of the definitions.",
                ));
                continue;
            }
            let impl_lts: Vec<Sym> = imp
                .generics
                .params
                .iter()
                .filter_map(|g| if let ast::GenericParam::Lifetime { lt, .. } = g { Some(lt.name) } else { None })
                .collect();
            if let Some(d) = generics::item_ty_lifetime_err(ast, &p.interner, c.ty, &impl_lts, true) {
                errs.push(d);
                continue;
            }
            let ty = match resolve_item_ty(p, c.ty, Some(self_ty)) {
                Ok(t) => t,
                Err(d) => {
                    errs.push(d);
                    continue;
                }
            };
            let Some(value) = c.default else {
                errs.push(not_yet("L0632", c.name_span, "associated consts without a value"));
                continue;
            };
            inherent_seen.push((impl_def, c.name));
            assoc_by_key.entry((p.adts[owner.0 as usize].name, c.name)).or_default().push(pending.len());
            pending.push((
                PendingConst {
                    name: Some(c.name),
                    ty,
                    expr: value,
                    span: c.span,
                    is_static: None,
                    static_idx: 0,
                    def_hash: crate::def::DefPathHash(0),
                    stmt_idx: None,
                    assoc: Some(PendingAssoc::Inherent { owner, impl_def, self_pat, generics: impl_generics.clone() }),
                    tls: None, scope: Vec::new(),
                },
                PendingState::Todo,
            ));
        }
    }
    // Trait defaults. A default whose type names the trait's
    // parameters has no single value to evaluate here; it stays
    // unregistered and use sites reject, as before.
    for item in &ast.items {
        let ItemKind::Trait(tdef) = &item.kind else { continue };
        let Some(&trait_def) = p.gx.trait_names.get(&tdef.name) else { continue };
        let Some(rec) = p.gx.index.trait_record(trait_def).cloned() else { continue };
        for c in &tdef.consts {
            let Some(value) = c.default else { continue };
            let Some(assoc) = rec.assoc_by_name(c.name) else { continue };
            let Some(&(ty, _)) = p.gx.trait_assoc_consts.get(&assoc) else { continue };
            if p.tys.flags(ty).contains_param() {
                continue;
            }
            if pending.iter().any(|(it, _)| matches!(it.assoc, Some(PendingAssoc::TraitDefault { assoc: a, .. }) if a == assoc)) {
                continue;
            }
            pending.push((
                PendingConst {
                    name: Some(c.name),
                    ty,
                    expr: value,
                    span: c.span,
                    is_static: None,
                    static_idx: 0,
                    def_hash: crate::def::DefPathHash(0),
                    stmt_idx: None,
                    assoc: Some(PendingAssoc::TraitDefault { trait_def, assoc }),
                    tls: None, scope: Vec::new(),
                },
                PendingState::Todo,
            ));
        }
    }
}

/// True when an impl Self pattern matches a written instance,
/// binding the impl's parameters to the parts they name. One
/// direction only (the pattern's parameters are variables, the
/// written type is ground), for const evaluation picks before
/// the checker exists.
pub fn self_pat_matches(tys: &impl crate::ty::TyIntern, pat: Ty, target: Ty, map: &mut Vec<(u32, Ty)>, cmap: &mut Vec<(u32, crate::ty::ConstId)>) -> bool {
    use crate::ty::TyIntern;
    if let TyKind::Param(p) = tys.kind(pat) {
        return match map.iter().find(|&&(i, _)| i == p.0) {
            Some(&(_, t)) => t == target,
            None => {
                map.push((p.0, target));
                true
            }
        };
    }
    let mut args_match = |a: crate::ty::ArgsId, b: crate::ty::ArgsId, map: &mut Vec<(u32, Ty)>, cmap: &mut Vec<(u32, crate::ty::ConstId)>| {
        let (xs, ys) = (tys.args(a), tys.args(b));
        xs.len() == ys.len()
            && std::iter::zip(xs.iter(), ys.iter()).all(|(x, y)| match (x, y) {
                (crate::ty::GenericArg::Ty(x), crate::ty::GenericArg::Ty(y)) => self_pat_matches(tys, *x, *y, map, cmap),
                (crate::ty::GenericArg::Const(x), crate::ty::GenericArg::Const(y)) => match tys.const_kind(*x) {
                    crate::ty::ConstKind::Param(i) => match cmap.iter().find(|&&(j, _)| j == i) {
                        Some(&(_, c)) => c == *y,
                        None => {
                            cmap.push((i, *y));
                            true
                        }
                    },
                    _ => x == y,
                },
                _ => x == y,
            })
    };
    match (tys.kind(pat), tys.kind(target)) {
        (TyKind::Adt(d, a), TyKind::Adt(e, b)) => d == e && args_match(a, b, map, cmap),
        (TyKind::Ref(_, m, x), TyKind::Ref(_, n, y)) => m == n && self_pat_matches(tys, x, y, map, cmap),
        (TyKind::RawPtr(m, x), TyKind::RawPtr(n, y)) => m == n && self_pat_matches(tys, x, y, map, cmap),
        (TyKind::Slice(x), TyKind::Slice(y)) => self_pat_matches(tys, x, y, map, cmap),
        (TyKind::Array(x, n), TyKind::Array(y, m)) => {
            (match tys.const_kind(n) {
                crate::ty::ConstKind::Param(i) => match cmap.iter().find(|&&(j, _)| j == i) {
                    Some(&(_, c)) => c == m,
                    None => {
                        cmap.push((i, m));
                        true
                    }
                },
                _ => n == m,
            }) && self_pat_matches(tys, x, y, map, cmap)
        }
        (TyKind::Tuple(x), TyKind::Tuple(y)) => {
            let (xs, ys) = (tys.tys_of(x), tys.tys_of(y));
            xs.len() == ys.len() && std::iter::zip(xs.iter(), ys.iter()).all(|(a, b)| self_pat_matches(tys, *a, *b, map, cmap))
        }
        _ => pat == target,
    }
}

/// True when `attrs` carries `#[non_exhaustive]`.
pub(crate) fn has_non_exhaustive(interner: &Interner, attrs: &[ast::Attr]) -> bool {
    attrs.iter().any(|a| a.path.len() == 1 && interner.get(a.path[0].name) == "non_exhaustive")
}

/// True when `adt` is defined in a different crate than module
/// `from`, for the `#[non_exhaustive]` rules: the attribute has no
/// effect inside its defining crate. A synthetic std Adt is foreign
/// to every body lust checks. With no resolver tables the program is
/// one crate, so nothing is foreign.
pub(crate) fn adt_is_foreign(p: &Program, adt: AdtId, from: crate::ty::defs::ModuleId) -> bool {
    if generics::is_std_adt(&p.tys, adt) {
        return true;
    }
    let Some(tables) = &p.resolved else { return false };
    let Some(def) = p.defs.def_of_adt(adt) else { return false };
    let Some(info) = tables.def_info.get(&def) else { return false };
    tables.root_of(info.module) != tables.root_of(from)
}

/// What a struct literal of variant `variant` of `adt` may not
/// build from module `from` under `#[non_exhaustive]` (rustc
/// E0639): "struct" for a marked struct, "variant" for a marked
/// variant, only across a crate border. An enum's own mark does not
/// block its variants' literals.
pub(crate) fn ne_lit_what(p: &Program, adt: AdtId, variant: u32, from: crate::ty::defs::ModuleId) -> Option<&'static str> {
    let a = p.adt(adt);
    let what = if a.is_enum {
        a.variants.get(variant as usize).is_some_and(|x| x.non_exhaustive).then_some("variant")?
    } else {
        a.non_exhaustive.then_some("struct")?
    };
    adt_is_foreign(p, adt, from).then_some(what)
}

/// What a tuple or unit constructor use (call, value, or pattern)
/// may not name from module `from` under `#[non_exhaustive]`
/// (rustc E0603): a marked struct's constructors and a marked
/// variant's tuple constructor are private across a crate border.
/// A unit variant stays usable.
pub(crate) fn ne_ctor_what(p: &Program, adt: AdtId, variant: u32, from: crate::ty::defs::ModuleId) -> Option<&'static str> {
    let a = p.adt(adt);
    let v = a.variants.get(variant as usize)?;
    let what = if a.is_enum {
        (v.non_exhaustive && v.shape == VariantShape::Tuple).then_some("tuple variant")?
    } else if v.shape == VariantShape::Unit {
        a.non_exhaustive.then_some("unit struct")?
    } else {
        a.non_exhaustive.then_some("tuple struct constructor")?
    };
    adt_is_foreign(p, adt, from).then_some(what)
}

/// What a named-field pattern without `..` may not match from
/// module `from` under `#[non_exhaustive]` (rustc E0638).
pub(crate) fn ne_pat_what(p: &Program, adt: AdtId, variant: u32, rest: bool, from: crate::ty::defs::ModuleId) -> Option<&'static str> {
    if rest {
        return None;
    }
    let a = p.adt(adt);
    let what = if a.is_enum {
        a.variants.get(variant as usize).is_some_and(|x| x.non_exhaustive).then_some("variant")?
    } else {
        a.non_exhaustive.then_some("struct")?
    };
    adt_is_foreign(p, adt, from).then_some(what)
}

/// True when a named field of `adt` is not visible from module
/// `from`, by the resolver's field table. The const evaluator's
/// privacy hooks in collect and in body checking share this test;
/// it mirrors `check_field_vis` without its read recording.
pub(crate) fn named_field_is_private(p: &Program, adt: AdtId, field: Sym, from: crate::ty::defs::ModuleId) -> bool {
    let Some(tables) = &p.resolved else { return false };
    let Some(def) = p.defs.def_of_adt(adt) else { return false };
    match tables.field_vis.get(&def) {
        Some((def_mod, fields)) => fields.get(&field).is_some_and(|vis| !tables.visible(vis, *def_mod, from, &p.interner)),
        None => false,
    }
}

/// True when tuple field `index` of `adt` is not visible from `from`.
/// The defining module of AST fn `i` from the resolver's def table (an
/// associated fn climbs to its impl or trait, a body fn to its outer
/// fn), or None when the table has no entry. Unlike `fn_module[i]` it
/// cannot shift when collect drops an item the resolver counted (the
/// fns of an impl whose trait or type failed to resolve).
pub(crate) fn def_module_of_fn(p: &Program, i: usize) -> Option<crate::ty::defs::ModuleId> {
    let tables = p.resolved.as_deref()?;
    let mut def = *p.fn_defs.get(i)?;
    for _ in 0..4 {
        if let Some(info) = tables.def_info.get(&def) {
            return Some(info.module);
        }
        def = p.defs.parent_of(def).or_else(|| p.gx.index.trait_of_method(def)).or_else(|| p.gx.index.trait_of_assoc(def))?;
    }
    None
}

pub(crate) fn tuple_field_is_private(p: &Program, adt: AdtId, index: u32, from: crate::ty::defs::ModuleId) -> bool {
    let Some(tables) = &p.resolved else { return false };
    let Some(def) = p.defs.def_of_adt(adt) else { return false };
    match tables.tuple_field_vis.get(&def) {
        Some((def_mod, fields)) => fields.get(index as usize).is_some_and(|vis| !tables.visible(vis, *def_mod, from, &p.interner)),
        None => false,
    }
}

/// What a name in a nested item's scope denotes (`CollectCx::scope_lookup`).
enum ScopeItem {
    /// Index into `CollectCx::pending` (a nested const or static).
    Pending(usize),
    Fn(LocalDefId),
    /// An item that is neither (it still shadows module items).
    Other,
}

/// The const evaluator's context over the collected tables.
struct CollectCx<'x> {
    p: &'x mut Program,
    /// `pending_module`'s answer for each body item's initializer, by
    /// expression (`body_item_modules`).
    body_modules: HashMap<u32, Option<crate::ty::defs::ModuleId>>,
    /// The program's Ast, borrowed apart from `p` (const evaluation reads
    /// it while it writes the program's type store).
    ast: &'x Ast,
    pending: Vec<(PendingConst, PendingState)>,
    /// Module-level pending consts and statics by (flat) name.
    /// Nested items never enter it: only an initializer whose
    /// scope holds a nested item can read it.
    by_name: HashMap<Sym, usize>,
    /// Pending nested consts and statics by stmt_items index, for
    /// scope-checked reads from another nested item's initializer.
    by_stmt: HashMap<u32, usize>,
    /// The stmt_items indexes the initializer now evaluating may name
    /// (outermost first), set by `eval_pending` and by `enter_const_fn`.
    cur_scope: Vec<u32>,
    /// Saved `cur_scope` values for const fn bodies the evaluator entered.
    scope_stack: Vec<Vec<u32>>,
    /// The scope `enter_const_fn` installs next: the sibling items of
    /// the const fn `call_target` just resolved (empty for a module fn).
    next_scope: Option<Vec<u32>>,
    /// The promotion qualifs of evaluated items (`ty::constqual`), by
    /// pending index; an item with none qualifies by its type.
    quals: HashMap<usize, crate::ty::constqual::Qualif>,
    /// Pending inherent associated consts by (flat type name,
    /// const name), for `Type::CONST` reads in initializers.
    assoc_by_key: HashMap<(Sym, Sym), Vec<usize>>,
    /// The flat type name `Self` means while an inherent
    /// associated const evaluates (its owning Adt).
    self_name: Option<Sym>,
    /// The Self pattern of that same inherent impl, so a
    /// `Self::CONST` read picks this impl's const among
    /// same-name consts of disjoint impls.
    self_pat: Option<Ty>,
    /// The (self_name, self_pat) of each const fn body the evaluator
    /// entered (`enter_const_fn`), restored on exit.
    self_stack: Vec<(Option<Sym>, Option<Ty>)>,
    /// The defining module of the item now evaluating, for field
    /// privacy in its initializer. Nested item evaluation (an item
    /// that reads another const) restores the outer module, like
    /// `self_name`. None when the module is unknown (no resolver
    /// tables, or an item the tables do not name): no check then.
    module: Option<crate::ty::defs::ModuleId>,
    errs: &'x mut Vec<Diag>,
    /// Trait goals of bounded const fn calls (`const_fn_bounds_hold`).
    goals: Vec<(TraitRef, Span)>,
    /// Per pending index: the impl or trait generics when this
    /// associated const's initializer names one of its parameters
    /// (a const param, a type param's associated const, `Self::X`
    /// of a generic impl). Such a const has no single value: it
    /// evaluates per instance at use sites
    /// (`check::assoc_eval`), and its `consts` entry holds a zero
    /// placeholder for template reads.
    tmpls: Vec<Option<Generics>>,
}

impl CollectCx<'_> {
    /// The type instance a written `Type<...>` path head names,
    /// for picking among same-name inherent associated consts.
    /// `None` when the arguments are not all written (the pick
    /// is then ambiguous in a const context too).
    fn written_instance(&mut self, ty_name: Sym, seg: &ast::PathSeg) -> Option<Ty> {
        let adt = self.p.type_names.get(&ty_name).copied()?;
        let generics = self.p.gx.adt_generics.get(adt.0 as usize).cloned().unwrap_or_default();
        let n_args = generics.params.iter().filter(|p| p.kind != ParamKind::Lifetime).count();
        if n_args == 0 {
            return Some(self.p.tys.adt_ty(adt));
        }
        let ast::GenericArgs::Angle(written) = &seg.args else { return None };
        let mut args: Vec<crate::ty::GenericArg> = Vec::new();
        for a in written {
            match a {
                ast::GenericArg::Type(t) => {
                    let cx = self.p.gx.resolve_cx(&self.p.type_names, &self.p.aliases);
                    let ty = crate::check::generics::resolve_scoped(
                        &self.p.ast,
                        &self.p.interner,
                        &self.p.known,
                        &cx,
                        &mut self.p.tys,
                        *t,
                        &crate::check::generics::Scope::default(),
                    )
                    .ok()?;
                    args.push(crate::ty::GenericArg::Ty(ty));
                }
                ast::GenericArg::Const(e) => {
                    let v = eval_array_len(&self.p.ast, &self.p.interner, &self.p.known, *e)?;
                    args.push(crate::ty::GenericArg::Const(self.p.tys.intern_const(crate::ty::ConstKind::Int(v))));
                }
                _ => {}
            }
        }
        if args.len() != n_args {
            return None;
        }
        Some(self.p.tys.mk_adt(self.p.def_of_adt(adt), &args))
    }

    /// The defining module of one pending item: the resolver's
    /// span table for module items (anonymous and associated
    /// consts included), else the enclosing function's module for
    /// an item in a body (the smallest fn expression range that
    /// holds the initializer; nested fn items share their outer
    /// function's module, so any holding range gives it), read as
    /// `def_module_of_fn`.
    fn pending_module(&self, it: &PendingConst) -> Option<crate::ty::defs::ModuleId> {
        let tables = self.p.resolved.as_deref()?;
        if let Some(m) = tables.const_init_module.get(&it.span) {
            return Some(*m);
        }
        if it.stmt_idx.is_none() {
            return None;
        }
        self.body_modules.get(&it.expr.0).copied().flatten()
    }

    /// The body item `name` denotes in the scope now evaluating,
    /// innermost first: a pending const or static, a nested fn, or
    /// another item (which still shadows any module item of the name).
    fn scope_lookup(&self, name: Sym) -> Option<ScopeItem> {
        for &si in self.cur_scope.iter().rev() {
            let hit = match &self.p.ast.stmt_items[si as usize].kind {
                ItemKind::Fn(d) => (d.name == name).then(|| match self.p.body_items.get(&si) {
                    Some(BodyItem::Fn(f)) => ScopeItem::Fn(*f),
                    _ => ScopeItem::Other,
                }),
                ItemKind::Const(c) => (c.name == Some(name)).then(|| match self.by_stmt.get(&si) {
                    Some(&i) => ScopeItem::Pending(i),
                    None => ScopeItem::Other,
                }),
                ItemKind::Static(s) => (s.name == name).then(|| match self.by_stmt.get(&si) {
                    Some(&i) => ScopeItem::Pending(i),
                    None => ScopeItem::Other,
                }),
                _ => None,
            };
            if hit.is_some() {
                return hit;
            }
        }
        None
    }

    /// The body item `name` denotes at the use site `at`, by block
    /// span (`stmt_item_in_blocks`). This covers blocks inside the
    /// initializer itself, whose items are in no pending scope.
    fn span_lookup(&self, name: Sym, at: Span) -> Option<ScopeItem> {
        let si = stmt_item_in_blocks(&self.p.ast, &self.p.item_blocks, name, at)?;
        Some(match self.p.body_items.get(&si) {
            Some(BodyItem::Fn(f)) => ScopeItem::Fn(*f),
            _ => match self.by_stmt.get(&si) {
                Some(&i) => ScopeItem::Pending(i),
                None => ScopeItem::Other,
            },
        })
    }

    /// Evaluates one pending item, on demand from other items.
    /// A trait associated const read at a concrete Self type in a
    /// const initializer (`S::C`, `<S as Tr>::C`): the one non-generic
    /// impl for `self_ty` (of `trait_def` when written) that defines
    /// `name`, else that impl's trait default; a bare `S::C` takes
    /// only traits in scope. None when no impl or
    /// several apply (rustc E0599, E0034) or a generic impl would.
    fn trait_assoc_const(&mut self, self_ty: Ty, trait_def: Option<DefId>, name: Sym) -> Option<crate::ty::consteval::TypedValue> {
        let p = &*self.p;
        let mut hits: Vec<(usize, Option<DefId>)> = Vec::new();
        for (_, rec) in p.gx.index.all_impls() {
            if trait_def.is_some_and(|t| t != rec.trait_def) {
                continue;
            }
            // `S::C` names no trait: it must be in scope (rustc E0599).
            if trait_def.is_none()
                && let Some(m) = self.module
                && !p.trait_in_module_answer(m, rec.trait_def)
            {
                continue;
            }
            let Some(trec) = p.gx.index.trait_record(rec.trait_def) else { continue };
            let Some(assoc) = trec.assoc_by_name(name).filter(|a| p.gx.trait_assoc_consts.contains_key(a)) else { continue };
            if rec.self_ty(&p.tys) != self_ty {
                // A generic impl that may apply leaves the read to the checker.
                if !rec.generics.params.is_empty() && rec.blanket.is_some() {
                    return None;
                }
                continue;
            }
            let own = self.pending.iter().position(|(it, _)| matches!(it.assoc, Some(PendingAssoc::TraitImpl { impl_def, assoc: a }) if impl_def == rec.def && a == assoc));
            let dflt = self.pending.iter().position(|(it, _)| matches!(it.assoc, Some(PendingAssoc::TraitDefault { assoc: a, .. }) if a == assoc));
            match (own, dflt) {
                (Some(i), _) => hits.push((i, None)),
                (None, Some(d)) => hits.push((d, Some(rec.def))),
                (None, None) => return None,
            }
        }
        let [(mut i, impl_of_default)] = hits[..] else { return None };
        // A default that reads `Self` is a per-instance template (its
        // value here is a placeholder): this impl evaluates its own copy.
        // A generic impl's own template has no value here.
        if self.tmpls[i].is_some() {
            match impl_of_default {
                Some(impl_def) => i = self.default_copy(i, impl_def),
                None => return None,
            }
        }
        let ty = self.pending[i].0.ty;
        self.eval_pending(i).map(|v| (v, ty))
    }

    /// The pending index of trait default `d` evaluated as impl
    /// `impl_def`'s own const, made once (a copy is no template).
    fn default_copy(&mut self, d: usize, impl_def: DefId) -> usize {
        let Some(PendingAssoc::TraitDefault { assoc, .. }) = self.pending[d].0.assoc else { return d };
        if let Some(i) = self.pending.iter().position(|(it, _)| matches!(it.assoc, Some(PendingAssoc::TraitImpl { impl_def: x, assoc: a }) if x == impl_def && a == assoc)) {
            return i;
        }
        let it = &self.pending[d].0;
        let item = PendingConst {
            name: it.name,
            ty: it.ty,
            expr: it.expr,
            span: it.span,
            is_static: None,
            static_idx: 0,
            def_hash: crate::def::DefPathHash(0),
            stmt_idx: None,
            assoc: Some(PendingAssoc::TraitImpl { impl_def, assoc }),
            tls: None,
            scope: Vec::new(),
        };
        self.pending.push((item, PendingState::Todo));
        self.tmpls.push(None);
        self.pending.len() - 1
    }

    fn eval_pending(&mut self, idx: usize) -> Option<crate::ty::consteval::ConstValue> {
        match &self.pending[idx].1 {
            PendingState::Done(v) => return Some(v.clone()),
            PendingState::Failed => return None,
            PendingState::Doing => {
                let span = self.pending[idx].0.span;
                self.errs.push(Diag::new(
                    "L0110",
                    span,
                    "const items form a cycle",
                    "Break the cycle: a const item cannot use its own value.",
                ));
                self.pending[idx].1 = PendingState::Failed;
                return None;
            }
            PendingState::Todo => {}
        }
        // A `thread_local!` key's value is its slot word (numbered once
        // every static is placed, `eval_const_items`); its initializer is
        // the init fn, checked as a body (check/tls.rs).
        if self.pending[idx].0.tls.is_some() {
            let v = crate::ty::consteval::ConstValue::Int(0);
            self.pending[idx].1 = PendingState::Done(v.clone());
            return Some(v);
        }
        // A generic associated const template: no single value
        // exists here. The zero placeholder stands in `consts`;
        // const contexts that read it by name get None from
        // `const_item`/`assoc_const_item` (outside the evaluator
        // for now), and body reads evaluate per instance.
        if self.tmpls[idx].is_some() {
            let v = zero_const_value(&self.p.tys, self.pending[idx].0.ty);
            self.pending[idx].1 = PendingState::Done(v.clone());
            return Some(v);
        }
        self.pending[idx].1 = PendingState::Doing;
        let (expr, ty, span) = {
            let it = &self.pending[idx].0;
            (it.expr, it.ty, it.span)
        };
        // `Self::CONST` in an inherent associated const's
        // initializer names its owning Adt; every other item
        // evaluates with no `Self`.
        let prev_self = self.self_name;
        let prev_pat = self.self_pat;
        (self.self_name, self.self_pat) = match self.pending[idx].0.assoc {
            Some(PendingAssoc::Inherent { owner, self_pat, .. }) => (Some(self.p.adts[owner.0 as usize].name), Some(self_pat)),
            // A trait impl const over a local Adt: `Self` is that Adt.
            Some(PendingAssoc::TraitImpl { impl_def, .. }) => {
                let self_ty =
                    self.p.gx.index.by_def(impl_def).and_then(|(_, r)| self.p.tys.args(r.trait_ref.args).first().and_then(|g| g.as_ty()));
                match self_ty.and_then(|t| self.p.tys.local_adt(t)) {
                    Some(a) => (Some(self.p.adts[a.0 as usize].name), self_ty),
                    None => (None, self_ty),
                }
            }
            _ => (None, None),
        };
        let prev_module = self.module;
        self.module = self.pending_module(&self.pending[idx].0);
        // The initializer names body items through its own scope, not
        // the scope of the item that asked for this value.
        let prev_scope = std::mem::replace(&mut self.cur_scope, self.pending[idx].0.scope.clone());
        let evaluated = crate::ty::consteval::eval(self, expr, Some(ty));
        if evaluated.is_ok() && self.pending[idx].0.is_static.is_none() {
            let q = crate::ty::constqual::of_const(self, expr, ty);
            self.next_scope = None;
            self.quals.insert(idx, q);
        }
        self.cur_scope = prev_scope;
        self.module = prev_module;
        self.self_name = prev_self;
        self.self_pat = prev_pat;
        match evaluated {
            Ok((v, vt)) => {
                if self.pending[idx].0.is_static.is_some() && !static_form(self.p, &v, ty) {
                    self.errs.push(Diag::new(
                        crate::ty::consteval::C_OUTSIDE,
                        span,
                        "this static value form is outside the bounded const evaluator",
                        "Use a tagged enum form the static data emitter supports, or a const item.",
                    ));
                    self.pending[idx].1 = PendingState::Failed;
                    return None;
                }
                let v = match coerce_const_value(self, v, vt, ty, span) {
                    Ok(v) => v,
                    Err(d) => {
                        self.errs.push(d);
                        self.pending[idx].1 = PendingState::Failed;
                        return None;
                    }
                };
                self.pending[idx].1 = PendingState::Done(v.clone());
                Some(v)
            }
            Err(d) => {
                // A name in the message helps when the span is deep inside.
                self.errs.push(d);
                self.pending[idx].1 = PendingState::Failed;
                None
            }
        }
    }
}

/// True when static data emission can write this value as a C
/// initializer. Every variant form qualifies today, niche-filled
/// enums included: the emitter sets the niche scalar through the
/// variant struct's `_s` member in the same designator (W-CGEN).
fn static_form(p: &Program, v: &crate::ty::consteval::ConstValue, ty: Ty) -> bool {
    use crate::ty::consteval::ConstValue as CV;
    match v {
        CV::Variant(variant, vals) => {
            let elem_tys: Vec<Ty> = match p.tys.kind(ty) {
                _ if p.tys.option_elem(ty).is_some() => vec![p.tys.option_elem(ty).unwrap()],
                _ if p.tys.result_args(ty).is_some() => {
                    let (a, b) = p.tys.result_args(ty).unwrap();
                    vec![if *variant == 0 { a } else { b }]
                }
                _ => match p.tys.local_adt(ty) {
                    Some(a) => p.adt(a).variants[*variant as usize].fields.iter().map(|f| f.ty).collect(),
                    None => vec![ERROR; vals.len()],
                },
            };
            vals.iter().enumerate().all(|(i, x)| static_form(p, x, elem_tys.get(i).copied().unwrap_or(ERROR)))
        }
        CV::Aggregate(vals) => {
            let elem_tys: Vec<Ty> = match p.tys.kind(ty) {
                TyKind::Tuple(ts) => p.tys.tys_of(ts).to_vec(),
                TyKind::Array(e, _) => vec![e; vals.len()],
                _ => match p.tys.local_adt(ty) {
                    Some(a) => p.adt(a).variants[0].fields.iter().map(|f| f.ty).collect(),
                    None => vec![ERROR; vals.len()],
                },
            };
            vals.iter().enumerate().all(|(i, x)| static_form(p, x, elem_tys.get(i).copied().unwrap_or(ERROR)))
        }
        CV::Repeat(x, _) => match p.tys.kind(ty) {
            TyKind::Array(e, _) => static_form(p, x, e),
            _ => true,
        },
        CV::Ref(inner) => match p.tys.kind(ty) {
            TyKind::Ref(_, _, it) => static_form(p, inner, it),
            _ => true,
        },
        _ => true,
    }
}

/// Accepts an evaluated value for a declared item type only when the
/// value's type is the declared type, up to the unsized coercions in
/// `const_ty_compatible`. Tuple arity and array length are in the type the
/// evaluator builds, so they are checked here too. One more form passes: a
/// `None` variant, whose type the evaluator cannot name before the declared
/// `Option` type is known.
fn coerce_const_value(
    cx: &mut CollectCx<'_>,
    v: crate::ty::consteval::ConstValue,
    vt: Ty,
    ty: Ty,
    span: Span,
) -> Result<crate::ty::consteval::ConstValue, Diag> {
    use crate::ty::consteval::ConstValue;
    if crate::ty::consteval::const_ty_compatible(&cx.p.tys, vt, ty) {
        return Ok(v);
    }
    if matches!(&v, ConstValue::Variant(0, vals) if vals.is_empty()) && cx.p.tys.option_elem(ty).is_some() {
        return Ok(v);
    }
    // The evaluator builds no vtables: a value that becomes a dyn value
    // here (`static D: &(dyn Tr + Sync) = &S;`) is valid Rust lust
    // rejects. A dyn type in a runtime-built value (`LazyLock`, a
    // `thread_local!` init, `None`) needs no const vtable and passes.
    if crate::check::object::mentions_dyn(&cx.p.tys, ty) && !crate::check::object::mentions_dyn(&cx.p.tys, vt) {
        return Err(crate::check::generics::dyn_not_yet(span, "a `dyn` coercion in a const or static initializer"));
    }
    Err(Diag::new(
        "L0110",
        span,
        format!("mismatched types: expected `{}`, found `{}`", ty_name(cx.p, &cx.p.tys, ty), ty_name(cx.p, &cx.p.tys, vt)),
        "Change the value or the declared type.",
    ))
}

/// The primitive a non-generic type alias names, for `Alias::MAX` in
/// const position (`ConstCx::alias_primitive`).
pub fn alias_primitive_of(p: &Program, name: Sym) -> Option<&'static str> {
    match p.tys.kind(*p.aliases.get(&name)?) {
        TyKind::Int(it) => Some(it.name()),
        TyKind::Char => Some("char"),
        TyKind::Float(crate::ty::FloatTy::F32) => Some("f32"),
        TyKind::Float(crate::ty::FloatTy::F64) => Some("f64"),
        _ => None,
    }
}

/// The hashed collection a plain or generic type alias names
/// (`ConstCx::alias_hash`).
pub fn alias_hash_of(p: &Program, name: Sym) -> Option<crate::ty::hash::HashTy> {
    let t = match p.aliases.get(&name) {
        Some(&t) => t,
        None => p.gx.generic_aliases.get(&name)?.erased,
    };
    crate::ty::hash::hash_parts(&p.tys, t).map(|(k, _)| k)
}

/// How a const fn's generics meet a call in const position
/// (`ConstFnBody::generic`): from its template's predicates, with std's
/// `LazyLock::new` and `LazyCell::new` named by their std paths.
pub fn const_generic_kind(p: &Program, f: LocalDefId) -> crate::ty::consteval::ConstGeneric {
    use crate::ty::consteval::ConstGeneric;
    let Some(Some(t)) = p.gx.fn_templates.get(f.0 as usize) else { return ConstGeneric::No };
    // Implied `Sized` bounds need nothing: every value here is sized.
    let sized = p.gx.lang.sized;
    if t.generics.predicates.iter().all(|pr| matches!(pr, Predicate::Trait(b) if b.value.def == sized)) {
        return ConstGeneric::Free;
    }
    let info = p.fn_info(f);
    let lazy = p.std_source_span(info.def(&p.ast).name_span)
        && p.interner.get(info.name) == "new"
        && info.owner.is_some_and(|a| matches!(p.display_path(p.adt(a).name).as_str(), "std::sync::LazyLock" | "core::cell::LazyCell"));
    if lazy { ConstGeneric::LazyCtor } else { ConstGeneric::Bounded }
}

/// What a call of fn `f` in const position runs (`ConstCx::call_target`
/// and `method_target` of every context): its body with a method's
/// receiver as the first parameter, `NonConst` for a fn that is not
/// `const`, and `Unknown` (outside the bound) for a `&mut self` or
/// typed receiver.
pub fn const_fn_target(p: &Program, f: LocalDefId) -> crate::ty::consteval::CallTarget {
    use crate::ty::consteval::{CallTarget, ConstFnBody};
    let info = p.fn_info(f);
    let def = info.def(&p.ast);
    if !info.is_const {
        return CallTarget::NonConst;
    }
    let Some(body) = def.body else { return CallTarget::NonConst };
    let mut params: Vec<(Sym, bool, Option<ast::PatId>)> = Vec::new();
    if let Some(sp) = &def.self_param {
        if !matches!(sp.kind, ast::SelfKind::Value | ast::SelfKind::Ref) {
            return CallTarget::Unknown;
        }
        params.push((p.known.self_value, sp.mutable, None));
    }
    params.extend(def.params.iter().map(|pp| (pp.name, pp.mutable, pp.pat)));
    CallTarget::ConstFn(ConstFnBody {
        params,
        param_tys: info.params.clone(),
        ret: info.ret,
        body,
        owner: info.owner.map(|a| p.adt(a).name),
        generic: const_generic_kind(p, f),
        def: f.0,
        has_self: def.self_param.is_some(),
    })
}

/// The one inherent method `name` of Adt type `recv` with a `self`
/// parameter, for a method call in const position. Several inherent
/// impls under disjoint Self patterns are outside the bound.
pub fn const_method_of(p: &Program, tys: &impl TyIntern, recv: Ty, name: Sym) -> Option<LocalDefId> {
    let a = tys.local_adt(recv)?;
    if a.0 >= u32::MAX - 1 {
        return None;
    }
    let fs = p.methods.get(&(a, name))?;
    let [f] = fs.as_slice() else { return None };
    p.fn_info(*f).self_kind.is_some().then_some(*f)
}

/// The def and args of generic Adt `adt` at `binds`
/// (`ConstCx::adt_instance`): every param a type param, each one bound.
pub fn const_adt_args(p: &Program, adt: AdtId, binds: &[Option<Ty>]) -> Option<(DefId, Vec<GenericArg>)> {
    let g = p.gx.adt_generics.get(adt.0 as usize)?;
    if g.params.iter().any(|q| q.kind != ParamKind::Type) {
        return None;
    }
    let args: Option<Vec<GenericArg>> = (0..g.params.len()).map(|i| binds.get(i).copied().flatten().map(GenericArg::Ty)).collect();
    Some((p.def_of_adt(adt), args?))
}

/// True when inherent item `name` of `adt` is not visible from `from`.
pub(crate) fn assoc_item_is_private(p: &Program, adt: AdtId, name: Sym, from: crate::ty::defs::ModuleId) -> bool {
    let Some(tables) = &p.resolved else { return false };
    let def = p.def_of_adt(adt);
    tables.assoc_vis.get(&def).and_then(|items| items.get(&name)).is_some_and(|(vis, def_mod)| !tables.visible(vis, *def_mod, from, &p.interner))
}

/// Whether the where-clauses of const fn `f` hold with its type params
/// bound to `binds` (`ConstCx::bounds_hold`), in an empty param env:
/// const and static items have no params in scope. A trait bound with
/// closed args goes to `goals`, proved after `finish_collect` (std and
/// derived impls are not indexed while const items evaluate;
/// `prove_const_goals`). `T: 'static` (and `T: 'a` for a lifetime
/// param, which the call picks) holds for a type with no reference,
/// pointer, `dyn`, or lifetime in it. Any other predicate, a
/// higher-ranked bound, or a param the call does not bind is None.
pub fn const_fn_bounds_hold(p: &mut Program, f: u32, binds: &[Option<Ty>], span: Span, goals: &mut Vec<(TraitRef, Span)>) -> Option<bool> {
    let Some(Some(t)) = p.gx.fn_templates.get(f as usize) else { return None };
    let preds = t.generics.predicates.clone();
    let args: Vec<GenericArg> = binds
        .iter()
        .enumerate()
        .map(|(i, b)| GenericArg::Ty(b.unwrap_or_else(|| p.tys.intern(TyKind::Param(crate::ty::ParamId(i as u32))))))
        .collect();
    let closed = |p: &Program, t: Ty| !p.tys.flags(t).contains_param() && !p.tys.flags(t).contains_infer();
    let mut found = Vec::new();
    for pr in preds {
        match pr {
            Predicate::Trait(b) if b.count == 0 => {
                let targs: Vec<GenericArg> = p.tys.args(b.value.args).to_vec();
                let mut sub = Vec::with_capacity(targs.len());
                for g in targs {
                    sub.push(match g {
                        GenericArg::Ty(t) => {
                            let t = crate::ty::subst(&mut p.tys, t, &args);
                            if !closed(p, t) {
                                return None;
                            }
                            GenericArg::Ty(t)
                        }
                        GenericArg::Lifetime(_) => g,
                        GenericArg::Const(_) => return None,
                    });
                }
                if b.value.bindings != crate::ty::EMPTY_BINDINGS {
                    return None;
                }
                let sub = p.tys.intern_args(&sub);
                if b.value.def != p.gx.lang.sized {
                    found.push((TraitRef { def: b.value.def, args: sub, bindings: b.value.bindings }, span));
                }
            }
            Predicate::TypeOutlives(t, r) if matches!(r, crate::ty::Region::Static | crate::ty::Region::Param(_)) => {
                let t = crate::ty::subst(&mut p.tys, t, &args);
                if !closed(p, t) || !static_only(&p.tys, t) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    goals.extend(found);
    Some(true)
}

/// Proves the trait goals bounded const fn calls left
/// (`const_fn_bounds_hold`), on a program past `finish_collect`; each
/// failed goal is rustc's E0277 at its call. True when all hold.
pub(crate) fn prove_const_goals(p: &mut Program, goals: Vec<(TraitRef, Span)>, errs: &mut Vec<Diag>) -> bool {
    let mut ok = true;
    for (tr, span) in goals {
        if !generics::prove_closed_goal(p, tr) {
            ok = false;
            let args: Vec<GenericArg> = p.tys.args(tr.args).to_vec();
            let self_t = args.first().and_then(|g| g.as_ty()).map(|t| ty_name(p, &p.tys, t)).unwrap_or_default();
            let tn = p.gx.index.trait_record(tr.def).map(|r| p.display_name(r.name).to_string()).unwrap_or_else(|| "the trait".to_string());
            errs.push(Diag::new(
                "L0612",
                span,
                format!("the trait bound `{self_t}: {tn}` of this const fn call is not satisfied (or lust cannot prove it)"),
                "Call it with types that meet the fn's bounds.",
            ));
        }
    }
    ok
}

/// True when every region in a type is a written `'static` (a type
/// with no reference, `dyn`, or lifetime argument is too): the type
/// outlives `'static`. An erased region (an inferred borrow) does not
/// count. A fn pointer type with no free region does: its own regions
/// are late-bound or `'static` (`const_fn_bounds_hold`).
fn static_only(tys: &GlobalTys, t: Ty) -> bool {
    use crate::ty::Region;
    match tys.kind(t) {
        TyKind::Ref(r, _, e) => r == Region::Static && static_only(tys, e),
        TyKind::RawPtr(_, e) | TyKind::Array(e, _) | TyKind::Slice(e) => static_only(tys, e),
        TyKind::FnPtr(s) => tys.sig(s).value.free_slots == 0,
        TyKind::Dyn(..) => false,
        TyKind::Tuple(ts) => tys.tys_of(ts).iter().all(|&e| static_only(tys, e)),
        TyKind::Adt(_, a) => tys.args(a).iter().all(|g| match g {
            GenericArg::Ty(e) => static_only(tys, *e),
            GenericArg::Lifetime(r) => *r == Region::Static,
            GenericArg::Const(_) => true,
        }),
        _ => true,
    }
}

/// True when a type holds no reference, raw pointer, fn pointer, trait
/// object, or lifetime argument (`CollectCx::fn_item`).
fn region_free(tys: &GlobalTys, t: Ty) -> bool {
    match tys.kind(t) {
        TyKind::Ref(..) | TyKind::RawPtr(..) | TyKind::FnPtr(_) | TyKind::Dyn(..) => false,
        TyKind::Tuple(ts) => tys.tys_of(ts).iter().all(|&e| region_free(tys, e)),
        TyKind::Array(e, _) | TyKind::Slice(e) => region_free(tys, e),
        TyKind::Adt(_, a) => tys.args(a).iter().all(|g| match g {
            GenericArg::Ty(e) => region_free(tys, *e),
            GenericArg::Lifetime(_) => false,
            GenericArg::Const(_) => true,
        }),
        _ => true,
    }
}

impl<'a> crate::ty::consteval::ConstCx<'a> for CollectCx<'a> {
    type Tys = GlobalTys;

    fn tys(&mut self) -> &mut GlobalTys {
        &mut self.p.tys
    }
    fn tys_ref(&self) -> &GlobalTys {
        &self.p.tys
    }
    fn ast(&self) -> &'a Ast {
        self.ast
    }
    fn sym_text(&self, s: Sym) -> &str {
        self.p.interner.get(s)
    }
    fn patterns_resolved(&self) -> bool {
        self.p.resolved.is_some()
    }
    fn block_const_item(&mut self, name: Sym, at: Span) -> Option<crate::ty::consteval::TypedValue> {
        match self.span_lookup(name, at).or_else(|| self.scope_lookup(name))? {
            ScopeItem::Pending(i) => {
                let ty = self.pending[i].0.ty;
                self.eval_pending(i).map(|v| (v, ty))
            }
            ScopeItem::Fn(_) | ScopeItem::Other => None,
        }
    }
    fn const_item(&mut self, name: Sym, at: Span) -> Option<crate::ty::consteval::TypedValue> {
        // A body item of this name in scope wins over every module
        // item, and a name no scope item has never finds a nested
        // item: nested items are not in `by_name`.
        if self.span_lookup(name, at).or_else(|| self.scope_lookup(name)).is_some() {
            return self.block_const_item(name, at);
        }
        if let Some(&i) = self.by_name.get(&name) {
            let ty = self.pending[i].0.ty;
            return self.eval_pending(i).map(|v| (v, ty));
        }
        self.p.const_names.get(&name).map(|&i| {
            let c = &self.p.consts[i as usize];
            (c.value.clone(), c.ty)
        })
    }
    fn const_item_qualif(&mut self, name: Sym, at: Span) -> Option<crate::ty::constqual::Qualif> {
        // The same lookup as `const_item`.
        let i = match self.span_lookup(name, at).or_else(|| self.scope_lookup(name)) {
            Some(ScopeItem::Pending(i)) => i,
            Some(_) => return None,
            None => match self.by_name.get(&name) {
                Some(&i) => i,
                None => return self.p.const_names.get(&name).map(|&i| self.p.consts[i as usize].qualif),
            },
        };
        // A static is no const operand; an item still evaluating is a
        // cycle that `eval` reports where it reads the value.
        if self.pending[i].0.is_static.is_some() || matches!(self.pending[i].1, PendingState::Doing) {
            return None;
        }
        self.eval_pending(i)?;
        Some(self.quals.get(&i).copied().unwrap_or(crate::ty::constqual::Qualif::ALL))
    }
    fn adt_has_drop(&self, adt: AdtId) -> bool {
        self.p.drop_adts.contains(&adt)
    }
    fn const_item_ref(&self, name: Sym, at: Span) -> Option<(&crate::ty::consteval::ConstValue, Ty)> {
        // The same lookup as `const_item`.
        let i = match self.span_lookup(name, at).or_else(|| self.scope_lookup(name)) {
            Some(ScopeItem::Pending(i)) => i,
            Some(ScopeItem::Fn(_) | ScopeItem::Other) => return None,
            None => match self.by_name.get(&name) {
                Some(&i) => i,
                None => {
                    let c = &self.p.consts[*self.p.const_names.get(&name)? as usize];
                    return Some((&c.value, c.ty));
                }
            },
        };
        match &self.pending[i] {
            (it, PendingState::Done(v)) => Some((v, it.ty)),
            _ => None,
        }
    }
    fn assoc_const_item(&mut self, segs: &[ast::PathSeg]) -> Option<crate::ty::consteval::TypedValue> {
        // The last two segments name the type and the const;
        // earlier segments are module anchors the resolver has
        // already folded into the flat type name where it can.
        let [.., ty_seg, item_seg] = segs else { return None };
        let self_form = ty_seg.name == self.p.known.self_type;
        if self_form && self.self_name.is_none() {
            // `Self` of a trait impl for a builtin type, or no Self at all
            // (a trait default, a per-instance template).
            let t = self.self_pat?;
            return self.trait_assoc_const(t, None, item_seg.name);
        }
        let ty_name = if self_form { self.self_name? } else { ty_seg.name };
        if let Some(cands) = self.assoc_by_key.get(&(ty_name, item_seg.name)).cloned() {
            // Several disjoint impls may define the name: pick
            // by the written instance (`Self` is the evaluating
            // impl's own pattern), exactly one must match.
            let pick = if cands.len() == 1 {
                Some(cands[0])
            } else {
                let written = if self_form { self.self_pat } else { self.written_instance(ty_name, ty_seg) };
                let written = written?;
                let hits: Vec<usize> = cands
                    .iter()
                    .copied()
                    .filter(|&i| match self.pending[i].0.assoc {
                        Some(PendingAssoc::Inherent { self_pat, .. }) => {
                            self_pat_matches(&self.p.tys, self_pat, written, &mut Vec::new(), &mut Vec::new())
                        }
                        _ => false,
                    })
                    .collect();
                if hits.len() == 1 { Some(hits[0]) } else { None }
            };
            let i = pick?;
            if self.tmpls[i].is_some() {
                return None;
            }
            let ty = self.pending[i].0.ty;
            return self.eval_pending(i).map(|v| (v, ty));
        }
        // A primitive Self (`u8::N` of a user trait impl for `u8`).
        if !self_form
            && let Some(it) = self.p.known.int_ty(ty_name)
        {
            return self.trait_assoc_const(int(it), None, item_seg.name);
        }
        let adt = self.p.type_names.get(&ty_name).copied()?;
        if let &[idx] = self.p.inherent_assoc_consts(adt, item_seg.name) {
            let c = &self.p.consts[idx as usize];
            return Some((c.value.clone(), c.ty));
        }
        // A trait's const for a non-generic Adt (`S::C`).
        if self.p.gx.adt_generics.get(adt.0 as usize).is_some_and(|g| !g.params.is_empty()) {
            return None;
        }
        let self_ty = self.p.tys.adt_ty(adt);
        self.trait_assoc_const(self_ty, None, item_seg.name)
    }
    fn qpath_const(&mut self, self_ty: ast::TyId, trait_path: &[ast::PathSeg], name: Sym) -> Option<crate::ty::consteval::TypedValue> {
        let t = resolve_item_ty(self.p, self_ty, None).ok()?;
        let [.., tseg] = trait_path else { return None };
        if !matches!(tseg.args, ast::GenericArgs::None) {
            return None;
        }
        let trait_def = self.p.gx.trait_names.get(&tseg.name).copied()?;
        self.trait_assoc_const(t, Some(trait_def), name)
    }
    fn static_item(&mut self, name: Sym, at: Span) -> Option<(u32, Ty)> {
        if let Some(hit) = self.span_lookup(name, at).or_else(|| self.scope_lookup(name)) {
            return match hit {
                ScopeItem::Pending(i) if self.pending[i].0.is_static.is_some() => {
                    let it = &self.pending[i].0;
                    Some((it.static_idx, it.ty))
                }
                _ => None,
            };
        }
        if let Some(&i) = self.by_name.get(&name)
            && self.pending[i].0.is_static.is_some()
        {
            let it = &self.pending[i].0;
            return Some((it.static_idx, it.ty));
        }
        self.p.static_names.get(&name).map(|&i| (i, self.p.statics[i as usize].ty))
    }
    fn static_value(&mut self, idx: u32) -> Option<&crate::ty::consteval::ConstValue> {
        let Some(i) = self.pending.iter().position(|(it, _)| it.is_static.is_some() && it.static_idx == idx) else {
            return self.p.statics.get(idx as usize).map(|s| &s.value);
        };
        if matches!(self.pending[i].1, PendingState::Todo) {
            self.eval_pending(i)?;
        }
        match &self.pending[i].1 {
            PendingState::Done(v) => Some(v),
            _ => None,
        }
    }
    fn fn_item(&mut self, segs: &[ast::PathSeg], at: Span) -> Option<crate::ty::consteval::TypedValue> {
        // A plain safe Rust-ABI free fn with no generics. With params,
        // its return type may not hold a reference or lifetime: the
        // pointer type here keeps no link between an elided output
        // lifetime and the input it comes from, so a region mismatch
        // rustc reports could not show. Params may hold elided
        // references only: each one is late-bound, as the written
        // pointer type's are. With no params every lifetime in the
        // return type is `'static` (W-STD-C).
        let [seg] = segs else { return None };
        if !seg.type_args().is_empty() {
            return None;
        }
        let f = match self.span_lookup(seg.name, at).or_else(|| self.scope_lookup(seg.name)) {
            Some(ScopeItem::Fn(f)) => f,
            Some(_) => return None,
            None => *self.p.fn_names.get(&seg.name)?,
        };
        let info = self.p.fn_info(f);
        let q = &info.def(&self.p.ast).qual;
        if info.owner.is_some()
            || info.self_kind.is_some()
            || q.is_async
            || q.is_unsafe
            || q.abi.is_some()
            || info.def(&self.p.ast).variadic
        {
            return None;
        }
        if !info.def(&self.p.ast).generics.params.is_empty() || !info.def(&self.p.ast).where_clause.preds.is_empty() {
            return None;
        }
        let (params, ret) = (info.params.clone(), info.ret);
        if !params.is_empty() && !region_free(&self.p.tys, ret) {
            return None;
        }
        // The fn's param types erase a written `'static` (and any
        // other written lifetime), so only fully elided ones compare
        // right; an `impl Trait` param is a hidden type param.
        if info.def(&self.p.ast).params.iter().any(|pp| generics::names_lifetime(&self.p.ast, &self.p.interner, pp.ty))
            || params.iter().any(|&t| self.p.tys.flags(t).contains_param())
        {
            return None;
        }
        let inputs = self.p.tys.intern_tys(&params);
        // With no params, every reference the fn returns is `'static`
        // (elision): the pointer type keeps that (W-BORROW3).
        let ret = if params.is_empty() { static_refs_in(&mut self.p.tys, ret) } else { ret };
        let sig = self.p.tys.intern_sig(crate::ty::Binder::plain(crate::ty::FnSig::plain(inputs, ret)));
        let t = self.p.tys.intern(TyKind::FnPtr(sig));
        Some((crate::ty::consteval::ConstValue::FnPtr(f.0), t))
    }
    fn call_target(&mut self, callee: ast::ExprId) -> crate::ty::consteval::CallTarget {
        use crate::ty::consteval::CallTarget;
        let ast::Expr::Path(segs) = self.p.ast.expr(callee) else { return CallTarget::Unknown };
        let last = segs[segs.len() - 1].name;
        let ltext = self.p.interner.get(last).to_string();
        if ltext == "size_of" || ltext == "align_of" {
            let ok_path = segs.len() == 1
                || (segs.len() == 3
                    && matches!(self.p.interner.get(segs[0].name), "std" | "core")
                    && self.p.interner.get(segs[1].name) == "mem");
            if ok_path {
                let args = segs[segs.len() - 1].type_args();
                if args.len() == 1 {
                    // Scoped resolution, not the plain one: the queried
                    // type can apply a generic Adt (`size_of::<W<u32>>()`),
                    // whose arguments the plain resolver rejects. A const
                    // context has no params in scope here.
                    let cx = self.p.gx.resolve_cx(&self.p.type_names, &self.p.aliases);
                    let t = crate::check::generics::resolve_scoped(
                        &self.p.ast,
                        &self.p.interner,
                        &self.p.known,
                        &cx,
                        &mut self.p.tys,
                        args[0],
                        &crate::check::generics::Scope::default(),
                    )
                    .unwrap_or(ERROR);
                    return if ltext == "size_of" { CallTarget::SizeOf(t) } else { CallTarget::AlignOf(t) };
                }
            }
            return CallTarget::Unknown;
        }
        if segs.len() == 1 && last == self.p.known.Some {
            return CallTarget::VariantCtor { adt: AdtId(u32::MAX), variant: 1, ty: ERROR };
        }
        // The Result shim sits one below the Option shim (u32::MAX):
        // Ok is variant 0, Err is variant 1.
        if segs.len() == 1 && last == self.p.known.Ok {
            return CallTarget::VariantCtor { adt: AdtId(u32::MAX - 1), variant: 0, ty: ERROR };
        }
        if segs.len() == 1 && last == self.p.known.Err {
            return CallTarget::VariantCtor { adt: AdtId(u32::MAX - 1), variant: 1, ty: ERROR };
        }
        let fid = if segs.len() == 1 {
            match self.span_lookup(last, self.p.ast.expr_span(callee)).or_else(|| self.scope_lookup(last)) {
                Some(ScopeItem::Fn(f)) => Some(f),
                // A body const or static of this name shadows any
                // module fn: the call names no fn here.
                Some(_) => return CallTarget::Unknown,
                None => self.p.fn_names.get(&last).copied(),
            }
        } else if segs.len() == 2 {
            // `Self::f(..)` names the impl type `Self` means here.
            let head = if segs[0].name == self.p.known.self_type { self.self_name.unwrap_or(segs[0].name) } else { segs[0].name };
            match self.p.type_names.get(&head) {
                Some(&a) => self.p.methods.get(&(a, last)).and_then(|fs| fs.first()).copied(),
                None => None,
            }
        } else {
            None
        };
        if let Some(f) = fid {
            let target = const_fn_target(self.p, f);
            if let CallTarget::ConstFn(b) = &target {
                // Written type args on a generic call are not matched yet.
                if b.generic != crate::ty::consteval::ConstGeneric::No && segs.iter().any(|s| !s.type_args().is_empty()) {
                    return CallTarget::Unknown;
                }
                // The body names its own sibling items, not the caller's.
                self.next_scope = Some(self.p.fn_siblings.get(&f).cloned().unwrap_or_default());
            }
            return target;
        }
        if let Some((adt, variant, ty)) = self.variant_ctor(segs) {
            return CallTarget::VariantCtor { adt, variant, ty };
        }
        // Known std constructors that are not const fn here report NonConst,
        // so `static S: String = String::from(..)` gets the right error.
        if segs.len() == 2 && matches!(self.p.interner.get(segs[0].name), "String" | "Vec") {
            return CallTarget::NonConst;
        }
        CallTarget::Unknown
    }
    fn method_target(&mut self, recv: Ty, name: Sym) -> crate::ty::consteval::CallTarget {
        let Some(f) = const_method_of(self.p, &self.p.tys, recv, name) else { return crate::ty::consteval::CallTarget::Unknown };
        let target = const_fn_target(self.p, f);
        if matches!(target, crate::ty::consteval::CallTarget::ConstFn(_)) {
            self.next_scope = Some(self.p.fn_siblings.get(&f).cloned().unwrap_or_default());
        }
        target
    }
    fn method_private(&mut self, recv: Ty, name: Sym) -> bool {
        match (self.module, self.p.tys.local_adt(recv)) {
            (Some(m), Some(a)) => assoc_item_is_private(self.p, a, name, m),
            _ => false,
        }
    }
    fn bounds_hold(&mut self, def: u32, binds: &[Option<Ty>], span: Span) -> Option<bool> {
        const_fn_bounds_hold(self.p, def, binds, span, &mut self.goals)
    }
    fn adt_instance(&mut self, adt: AdtId, binds: &[Option<Ty>]) -> Option<Ty> {
        let (def, args) = const_adt_args(self.p, adt, binds)?;
        Some(self.p.tys.mk_adt(def, &args))
    }
    fn alias_primitive(&self, name: Sym) -> Option<&'static str> {
        alias_primitive_of(self.p, name)
    }
    fn alias_hash(&self, name: Sym) -> Option<crate::ty::hash::HashTy> {
        alias_hash_of(self.p, name)
    }
    fn names_type(&mut self, name: Sym) -> bool {
        let p = &*self.p;
        name == p.known.self_type
            || p.type_names.contains_key(&name)
            || p.aliases.contains_key(&name)
            || p.gx.generic_aliases.contains_key(&name)
            || p.gx.trait_names.contains_key(&name)
            || matches!(
                p.interner.get(name),
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "f32" | "f64" | "bool" | "char" | "str"
            )
    }
    fn enter_const_fn(&mut self, owner: Option<Sym>) {
        self.self_stack.push((self.self_name, self.self_pat));
        self.self_name = owner;
        self.self_pat = None;
        let scope = self.next_scope.take().unwrap_or_default();
        self.scope_stack.push(std::mem::replace(&mut self.cur_scope, scope));
    }
    fn exit_const_fn(&mut self) {
        if let Some((name, pat)) = self.self_stack.pop() {
            self.self_name = name;
            self.self_pat = pat;
        }
        if let Some(scope) = self.scope_stack.pop() {
            self.cur_scope = scope;
        }
    }
    fn unit_variant(&mut self, segs: &[ast::PathSeg], expected: Option<Ty>) -> Option<(AdtId, u32, Ty)> {
        if segs.len() == 1 && segs[0].name == self.p.known.None {
            let elem = expected.and_then(|t| self.p.tys.option_elem(t)).unwrap_or(ERROR);
            let ty = self.p.tys.mk_option(elem);
            return Some((AdtId(u32::MAX), 0, ty));
        }
        let (adt, variant) = self.variant_of(segs)?;
        if self.p.adt(adt).variants[variant as usize].shape != VariantShape::Unit {
            return None;
        }
        // A generic unit value (`PhantomData`) takes the expected
        // instance of its own Adt.
        let def = self.p.def_of_adt(adt);
        let ty = match expected {
            Some(t) if segs.iter().all(|s| s.type_args().is_empty()) && self.p.tys.adt_parts(t).is_some_and(|(d, _)| d == def) => t,
            _ => self.p.tys.adt_ty(adt),
        };
        Some((adt, variant, ty))
    }
    fn struct_target(&mut self, path: &[ast::PathSeg]) -> Option<(AdtId, u32, Ty)> {
        let (adt, variant) = self.variant_of(path)?;
        Some((adt, variant, self.p.tys.adt_ty(adt)))
    }
    fn pattern_variant(&mut self, segs: &[ast::PathSeg]) -> Option<(AdtId, u32)> {
        self.variant_of(segs)
    }
    fn variant_fields(&self, adt: AdtId, variant: u32) -> Vec<(Sym, Ty)> {
        if adt.0 >= u32::MAX - 1 {
            return Vec::new();
        }
        self.p.adt(adt).variants[variant as usize].fields.iter().map(|f| (f.name, f.ty)).collect()
    }
    fn variant_count(&self, adt: AdtId) -> u32 {
        if adt.0 >= u32::MAX - 1 {
            return 0;
        }
        self.p.adt(adt).variants.len() as u32
    }
    fn resolve_ty(&mut self, t: ast::TyId) -> Ty {
        // Scoped resolution first: a written type can apply a generic
        // Adt (`let k: K<u8> = ..`), whose args the plain resolver
        // rejects. A const context has no params in scope here.
        let cx = self.p.gx.resolve_cx(&self.p.type_names, &self.p.aliases);
        let scoped = crate::check::generics::resolve_scoped(
            &self.p.ast,
            &self.p.interner,
            &self.p.known,
            &cx,
            &mut self.p.tys,
            t,
            &crate::check::generics::Scope::default(),
        );
        match scoped {
            Ok(r) => r,
            Err(_) => resolve_ty(&self.p.ast, &self.p.interner, &self.p.known, &self.p.type_names, &self.p.aliases, &mut self.p.tys, t, None)
                .unwrap_or(ERROR),
        }
    }
    fn layout(&mut self, t: Ty) -> Option<crate::ty::layout::Layout> {
        let p: &Program = self.p;
        crate::ty::layout::layout_of_ty(&p.tys, &UserLayout(p), t).ok()
    }
    fn adt_repr(&self, adt: AdtId) -> crate::ty::layout::Repr {
        self.p.adt(adt).repr
    }
    fn discriminant(&self, adt: AdtId, variant: u32) -> i128 {
        self.p.adt(adt).variants[variant as usize].disc
    }
    fn is_enum(&self, adt: AdtId) -> bool {
        adt.0 < u32::MAX - 1 && self.p.adt(adt).is_enum
    }
    fn needs_drop(&self, t: Ty) -> bool {
        needs_drop_const(self.p, &self.p.tys, t)
    }
    fn interior_mut(&self, t: Ty) -> bool {
        interior_mut(self.p, &self.p.tys, t)
    }
    fn ty_text(&self, t: Ty) -> String {
        ty_name(self.p, &self.p.tys, t)
    }
    fn is_copy(&self, t: Ty) -> bool {
        is_copy(self.p, &self.p.tys, t)
    }
    fn named_field_private(&mut self, adt: AdtId, field: Sym) -> bool {
        match self.module {
            Some(m) => named_field_is_private(self.p, adt, field, m),
            None => false,
        }
    }
    fn tuple_field_private(&mut self, adt: AdtId, index: u32) -> bool {
        match self.module {
            Some(m) => tuple_field_is_private(self.p, adt, index, m),
            None => false,
        }
    }
    fn ne_lit_what(&mut self, adt: AdtId, variant: u32) -> Option<&'static str> {
        self.module.and_then(|m| ne_lit_what(self.p, adt, variant, m))
    }
    fn ne_ctor_what(&mut self, adt: AdtId, variant: u32) -> Option<&'static str> {
        self.module.and_then(|m| ne_ctor_what(self.p, adt, variant, m))
    }
    fn ne_pat_what(&mut self, adt: AdtId, variant: u32, rest: bool) -> Option<&'static str> {
        self.module.and_then(|m| ne_pat_what(self.p, adt, variant, rest, m))
    }
}

impl CollectCx<'_> {
    /// A tuple variant constructor path, for calls in const position.
    fn variant_ctor(&self, segs: &[ast::PathSeg]) -> Option<(AdtId, u32, Ty)> {
        let (adt, variant) = self.variant_of(segs)?;
        if self.p.adt(adt).variants[variant as usize].shape != VariantShape::Tuple {
            return None;
        }
        Some((adt, variant, self.p.tys.get_adt_ty(adt).unwrap_or(ERROR)))
    }

    /// The (Adt, variant) a two-segment path or a struct name denotes.
    fn variant_of(&self, segs: &[ast::PathSeg]) -> Option<(AdtId, u32)> {
        // `Self` in an inherent associated const's initializer names the
        // owning Adt (`Self(1)`, `Self::A`, `Self { .. }`).
        let head = |n: Sym| if n == self.p.known.self_type { self.self_name.unwrap_or(n) } else { n };
        // `std::borrow::Cow::Borrowed` and `Owned` (W-STD-B).
        if segs.len() == 4
            && matches!(self.p.interner.get(segs[0].name), "std" | "alloc")
            && self.p.interner.get(segs[1].name) == "borrow"
            && self.p.interner.get(segs[2].name) == "Cow"
            && let Some(a) = self.p.tys.local_adt_of(generics::cow_def())
        {
            let v = self.p.adt(a).variants.iter().position(|v| v.name == segs[3].name)?;
            return Some((a, v as u32));
        }
        // `std::cmp::Ordering::Less`: the same std variant paths
        // `resolve_assoc` answers, for const contexts.
        if segs.len() >= 4 {
            let path: Vec<&str> = segs.iter().map(|s| self.p.interner.get(s.name)).collect();
            return generics::std_variant_of(self.p, &path);
        }
        // A type item of a block in scope shadows the module's (R18): a
        // block `const K = E::X as usize` next to its own `enum E` reads
        // that enum. A block alias or trait of the name is no Adt here.
        let adt_named = |n: Sym| -> Option<AdtId> {
            for &si in self.cur_scope.iter().rev() {
                let item = &self.p.ast.stmt_items[si as usize];
                let named = match &item.kind {
                    ItemKind::Struct(d) => d.name == n,
                    ItemKind::Enum(d) => d.name == n,
                    ItemKind::Union(d) => d.name == n,
                    ItemKind::Type(d) => d.name == n,
                    ItemKind::Trait(d) => d.name == n,
                    _ => false,
                };
                if named {
                    return match self.p.body_type_items.get(&si) {
                        Some(BodyTypeItem::Adt(a)) => Some(*a),
                        _ => None,
                    };
                }
            }
            self.p.type_names.get(&n).copied()
        };
        if segs.len() == 1 {
            let a = adt_named(head(segs[0].name))?;
            if self.p.adt(a).is_enum {
                return None;
            }
            return Some((a, 0));
        }
        if segs.len() == 2 {
            let a = adt_named(head(segs[0].name))?;
            let v = self.p.adt(a).variants.iter().position(|v| v.name == segs[1].name)?;
            return Some((a, v as u32));
        }
        None
    }
}

/// The capture types of a closure type (its args, ARCH 2.9): the
/// closure value is a tuple of these.
pub fn closure_fields(tys: &impl TyIntern, a: ArgsId) -> Vec<Ty> {
    tys.args(a).iter().filter_map(|g| g.as_ty()).collect()
}

pub fn contains_ref(tys: &impl TyIntern, t: Ty) -> bool {
    match tys.kind(t) {
        TyKind::Ref(..) => true,
        TyKind::Tuple(ts) => tys.tys_of(ts).iter().any(|&e| contains_ref(tys, e)),
        TyKind::Closure(_, a) => closure_fields(tys, a).iter().any(|&e| contains_ref(tys, e)),
        TyKind::Array(e, _) | TyKind::Slice(e) => contains_ref(tys, e),
        _ if tys.is_chars_ty(t) => true,
        // A `RefCell` guard holds its borrow of the cell.
        _ if tys.guard_elem(t).is_some() => true,
        _ => match tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)).or_else(|| tys.weak_elem(t)).or_else(|| tys.arc_weak_elem(t)).or_else(|| tys.refcell_elem(t)).or_else(|| tys.cell_elem(t)) {
            Some(e) => contains_ref(tys, e),
            None => match tys.result_args(t) {
                Some((a, b)) => contains_ref(tys, a) || contains_ref(tys, b),
                None => match crate::ty::iter::iter_parts(tys, t) {
                    Some((k, a)) => crate::ty::iter::iter_holds_ref(k, &a, &|e| contains_ref(tys, e)),
                    None => false,
                },
            },
        },
    }
}

fn contains_adt_by_value(p: &Program, root: AdtId, cur: AdtId, seen: &mut Vec<bool>) -> bool {
    if seen[cur.0 as usize] {
        return false;
    }
    seen[cur.0 as usize] = true;
    for v in &p.adt(cur).variants {
        for f in &v.fields {
            if type_holds_adt_by_value(p, root, f.ty, seen) {
                return true;
            }
        }
    }
    false
}

/// True when a type has a known size and may sit in a field, parameter, or
/// return slot. Bare slices, `str`, and `dyn` are unsized; they may only
/// appear behind a reference (ARCH 2.7 `Sized` basics).
pub fn is_sized_ty(tys: &impl TyIntern, t: Ty) -> bool {
    match tys.kind(t) {
        TyKind::Slice(_) | TyKind::Str | TyKind::Dyn(..) => false,
        TyKind::Adt(d, _) if tys.unsized_field(d).is_some() => false,
        TyKind::Adt(..) if crate::check::own::dyn_wrapped_elem(tys, t).is_some() => false,
        TyKind::Tuple(ts) => tys.tys_of(ts).iter().all(|&e| is_sized_ty(tys, e)),
        TyKind::Array(e, _) => is_sized_ty(tys, e),
        TyKind::Adt(..) => match tys.adt_first_ty_arg(t) {
            Some(e) if tys.vec_elem(t).is_some() || tys.option_elem(t).is_some() => is_sized_ty(tys, e),
            _ => match tys.result_args(t) {
                Some((a, b)) => is_sized_ty(tys, a) && is_sized_ty(tys, b),
                None => true,
            },
        },
        _ => true,
    }
}

/// True when a field type holds the root Adt by value, through Option, tuple,
/// and array wrappers (all of which store their elements inline).
fn type_holds_adt_by_value(p: &Program, root: AdtId, t: Ty, seen: &mut Vec<bool>) -> bool {
    match p.tys.kind(t) {
        TyKind::Tuple(ts) => p.tys.tys_of(ts).iter().any(|&e| type_holds_adt_by_value(p, root, e, seen)),
        TyKind::Array(e, _) => type_holds_adt_by_value(p, root, e, seen),
        _ => {
            if let Some(inner) = p.tys.option_elem(t) {
                return type_holds_adt_by_value(p, root, inner, seen);
            }
            if let Some((a, b)) = p.tys.result_args(t) {
                return type_holds_adt_by_value(p, root, a, seen) || type_holds_adt_by_value(p, root, b, seen);
            }
            match p.tys.local_adt(t) {
                Some(a) => a == root || contains_adt_by_value(p, root, a, seen),
                None => false,
            }
        }
    }
}

// Trait queries. Lust has no user traits yet, so these follow fixed rules plus derives.

// The queries below follow the old fixed rules, now read through the v2 compat
// helpers: builtin Adts answer through `BuiltinTy`, user Adts through their derives.

/// True when the param env of the body `tys` belongs to (see
/// `Tys::env`) has the bound `t: trait_def`, with `rhs` as the
/// second trait argument when given. The answer for a `Param` or
/// projection type: it has exactly the traits its bounds give it.
fn env_proves(tys: &impl TyIntern, t: Ty, trait_def: DefId, rhs: Option<Ty>) -> bool {
    tys.param_env().iter().any(|pred| match pred {
        Predicate::Trait(b) if b.count == 0 && b.value.def == trait_def => {
            let a = tys.args(b.value.args);
            a.first().and_then(|g| g.as_ty()) == Some(t) && rhs.is_none_or(|r| a.get(1).and_then(|g| g.as_ty()) == Some(r))
        }
        _ => false,
    })
}

/// True when a raw pointer to `e` has glue for an operation (tf-fatptr):
/// a sized pointee (a generic one by its `Sized` bound), or `str` and
/// `[T]` when `slice_ok` (they compare by address and length). A dyn
/// pointer has none: rustc compares vtable addresses, which it does not
/// fix.
fn raw_glue_ok(p: &Program, tys: &impl TyIntern, e: Ty, slice_ok: bool) -> bool {
    match tys.kind(e) {
        TyKind::Dyn(..) => false,
        TyKind::Str | TyKind::Slice(_) => slice_ok,
        TyKind::Adt(d, _) => tys.unsized_field(d).is_none(),
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, e, p.gx.lang.sized, None),
        _ => true,
    }
}

pub fn is_copy(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.copy, None),
        TyKind::Opaque(..) => crate::check::opaque::opaque_bound_holds(p, tys, t, p.gx.lang.copy),
        TyKind::Int(_) | TyKind::Float(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Never | TyKind::Error => true,
        // Fn items, fn pointers, and (non-capturing) closures are Copy.
        TyKind::FnDef(..) | TyKind::FnPtr(_) => true,
        // Raw pointers are Copy (w-cinterop): the value is the address.
        TyKind::RawPtr(..) => true,
        TyKind::Infer(v) => v.is_int() || v.is_float(),
        TyKind::Ref(_, m, _) => !m.is_mut(),
        TyKind::Tuple(ts) => tys.tys_of(ts).iter().all(|&e| is_copy(p, tys, e)),
        TyKind::Closure(_, a) => closure_fields(tys, a).iter().all(|&e| is_copy(p, tys, e)),
        TyKind::Array(e, _) => is_copy(p, tys, e),
        _ => {
            // `MaybeUninit<T>` is Copy for every T (TRIAGE T5), and so
            // is `NonZero<T>` (T is always a primitive integer): a field
            // of either is Copy before any instance of it exists.
            if tys.adt_parts(t).is_some_and(|(d, _)| d == generics::maybe_uninit_def() || d == generics::nonzero_def()) {
                return true;
            }
            if tys.is_exit_code(t)
                || tys.builtin_of(t) == Some(BuiltinTy::AccessError)
                // `fmt::Arguments` is Copy in core (R9); its lust value
                // (`lust_Arguments`: a fn and borrowed pointers) has no
                // drop call, so a copy is a plain struct copy.
                || crate::ty::fmt::fmt_kind(tys, t) == Some(crate::ty::fmt::FmtTy::Arguments)
            {
                return true;
            }
            if let Some(Ok(b)) = crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::Copy) {
                return b;
            }
            if let Some(e) = tys.option_elem(t) {
                return is_copy(p, tys, e);
            }
            if let Some((a, b)) = tys.result_args(t) {
                return is_copy(p, tys, a) && is_copy(p, tys, b);
            }
            // core's `RangeTo`, `RangeToInclusive`, and `RangeFull` are Copy
            // (`Range`, `RangeFrom`, `RangeInclusive` are not).
            if let Some((crate::ty::iter::IterKind::RangeTo | crate::ty::iter::IterKind::RangeToInclusive | crate::ty::iter::IterKind::RangeFull, a)) =
                crate::ty::iter::iter_parts(tys, t)
            {
                return a.iter().all(|&x| is_copy(p, tys, x));
            }
            // A written `where K<T>: Copy` bound, and a written `impl Copy`.
            if (tys.flags(t).contains_param() && env_proves(tys, t, p.gx.lang.copy, None)) || written_copy(p, tys, t) {
                return true;
            }
            match tys.local_adt(t) {
                // A generic Adt whose instance does not exist yet (a
                // closure capture checked before collect made `W<u8>`):
                // the template's fields name its params, so ask the
                // derive's own bounds (`impl<T: Copy> Copy for W<T>`) on
                // the type's args, as the instance answer would.
                Some(a)
                    if p.gx.adt_generics.get(a.0 as usize).is_some_and(|g| !g.params.is_empty())
                        && p.gx.adt_instance_of.get(a.0 as usize).is_none_or(|o| o.is_none())
                        && tys.adt_parts(t).is_some_and(|(_, args)| args != crate::ty::EMPTY_ARGS) =>
                {
                    let (_, args) = tys.adt_parts(t).expect("checked above");
                    p.derives_of(a).copy
                        && (p.is_phantom_adt(a) || tys.args(args).iter().filter_map(|g| g.as_ty()).all(|x| is_copy(p, tys, x)))
                }
                Some(a) => {
                    p.derive_has(a, crate::check::derive_delta::bit::COPY)
                        && (p.gx.adt_generics.get(a.0 as usize).is_none_or(|g| g.params.is_empty())
                            && p.gx.adt_instance_of.get(a.0 as usize).is_none_or(|o| o.is_none())
                            || {
                                // The derive's bounds on the args first; a field
                                // whose type is a proven arg is not asked again
                                // (`G<G<..>>` asked each level twice: 2^n).
                                let mut proven: Vec<Ty> = Vec::new();
                                derive_args_all(p, a, &mut GlueSeen::default(), &mut |x, _| {
                                    let ok = is_copy(p, tys, x);
                                    proven.push(x);
                                    ok
                                }) && p.adt(a).variants.iter().all(|v| v.fields.iter().all(|f| proven.contains(&f.ty) || is_copy(p, tys, f.ty)))
                            })
                }
                None => false,
            }
        }
    }
}

/// True when a written (not derived) `impl Copy` applies to user Adt
/// type `t`: its header matches and its bounds hold, as rustc selects
/// it (`impl<T> Copy for Key<T> {}` has no `T: Copy` bound, unlike the
/// derive). `check_copy_impls` makes every field `Copy` (E0204), so a
/// copy is a plain value copy with no drop.
fn written_copy(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    use crate::traits::index::{SelfHead, Synthetic};
    let Some((def, _)) = tys.adt_parts(t) else { return false };
    if tys.local_adt(t).is_none() {
        return false;
    }
    p.gx.index.bucket(p.gx.lang.copy, SelfHead::Adt(def)).iter().any(|&id| {
        let rec = p.gx.index.get(id);
        if rec.synthetic != Synthetic::None {
            return false;
        }
        let mut assign: Vec<Option<Ty>> = Vec::new();
        if !match_impl_pat(p, tys, rec.self_ty(&p.tys), t, &mut assign) {
            return false;
        }
        let mut seen = GlueSeen::default();
        rec.generics.predicates.iter().all(|pred| glue_impl_pred_holds(p, tys, pred, &assign, &mut seen))
    })
}

/// Matches one impl pattern type (in the global store) against a
/// caller type, recording parameter assignments (`assign[i]` is the
/// caller type for `Param(i)`). Only shapes a glue impl header can
/// take in practice match: parameters, Adts, references and raw
/// pointers, tuples, arrays, slices, and scalars; lifetimes are not
/// compared, const arguments must be equal values. Anything else
/// does not match, which can only keep a reject.
fn match_impl_pat(p: &Program, tys: &impl TyIntern, pat: Ty, t: Ty, assign: &mut Vec<Option<Ty>>) -> bool {
    match p.tys.kind(pat) {
        TyKind::Param(id) => {
            let i = id.0 as usize;
            if assign.len() <= i {
                assign.resize(i + 1, None);
            }
            match assign[i] {
                Some(prev) => prev == t,
                None => {
                    assign[i] = Some(t);
                    true
                }
            }
        }
        TyKind::Adt(def, args) => {
            let TyKind::Adt(def2, args2) = tys.kind(t) else { return false };
            if def != def2 {
                return false;
            }
            let (pa, ta): (Vec<GenericArg>, Vec<GenericArg>) = (p.tys.args(args).to_vec(), tys.args(args2).to_vec());
            pa.len() == ta.len()
                && pa.iter().zip(&ta).all(|(g, h)| match (g.as_ty(), h.as_ty(), g, h) {
                    (Some(x), Some(y), _, _) => match_impl_pat(p, tys, x, y, assign),
                    (_, _, GenericArg::Const(c), GenericArg::Const(d)) => p.tys.const_kind(*c) == tys.const_kind(*d),
                    (_, _, GenericArg::Lifetime(_), GenericArg::Lifetime(_)) => true,
                    _ => false,
                })
        }
        TyKind::Ref(_, m, e) => matches!(tys.kind(t), TyKind::Ref(_, m2, e2) if m == m2 && match_impl_pat(p, tys, e, e2, assign)),
        TyKind::RawPtr(m, e) => matches!(tys.kind(t), TyKind::RawPtr(m2, e2) if m == m2 && match_impl_pat(p, tys, e, e2, assign)),
        TyKind::Slice(e) => matches!(tys.kind(t), TyKind::Slice(e2) if match_impl_pat(p, tys, e, e2, assign)),
        TyKind::Array(e, n) => {
            matches!(tys.kind(t), TyKind::Array(e2, n2) if p.tys.const_kind(n) == tys.const_kind(n2) && match_impl_pat(p, tys, e, e2, assign))
        }
        TyKind::Tuple(ts) => match tys.kind(t) {
            TyKind::Tuple(ts2) => {
                let (xs, ys): (Vec<Ty>, Vec<Ty>) = (p.tys.tys_of(ts).to_vec(), tys.tys_of(ts2).to_vec());
                xs.len() == ys.len() && xs.iter().zip(&ys).all(|(&x, &y)| match_impl_pat(p, tys, x, y, assign))
            }
            _ => false,
        },
        TyKind::Int(i) => tys.kind(t) == TyKind::Int(i),
        TyKind::Float(f) => tys.kind(t) == TyKind::Float(f),
        TyKind::Bool => tys.kind(t) == TyKind::Bool,
        TyKind::Char => tys.kind(t) == TyKind::Char,
        TyKind::Unit => tys.kind(t) == TyKind::Unit,
        TyKind::Str => tys.kind(t) == TyKind::Str,
        TyKind::Never => tys.kind(t) == TyKind::Never,
        _ => false,
    }
}

/// Proves one predicate of a written glue impl after matching:
/// trait predicates whose Self (and `Rhs`) are single matched
/// parameters by the glue walk of their trait, and other trait
/// predicates (and `Sized`) over a concrete instance by the full
/// solver (`inst_args`). Anything else is not proven here (the impl
/// then does not count for glue, which can only keep a reject),
/// except outlives predicates, which do not change what the glue
/// calls (as in `prove_trait_ref`).
fn glue_impl_pred_holds(p: &Program, tys: &impl TyIntern, pred: &Predicate, assign: &[Option<Ty>], seen: &mut GlueSeen) -> bool {
    let arg_ty = |g: &GenericArg| -> Option<Ty> {
        match g.as_ty().map(|t| p.tys.kind(t)) {
            Some(TyKind::Param(id)) => assign.get(id.0 as usize).copied().flatten(),
            _ => None,
        }
    };
    match pred {
        Predicate::Trait(b) if b.count == 0 => {
            let args = p.tys.args(b.value.args);
            let self_ty = args.first().and_then(arg_ty);
            let inst = inst_args(p, tys, assign);
            // `Sized` (implicit on every impl param) by the solver's
            // structural answer when the instance is concrete; a body's
            // generic or local type by structure here.
            if b.value.def == p.gx.lang.sized {
                return match (&inst, self_ty) {
                    (Some(a), _) => crate::check::generics::prove_inst_trait_ref(p, b.value, a),
                    (None, Some(t)) => match tys.kind(t) {
                        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.sized, None),
                        _ => is_sized_ty(tys, t),
                    },
                    (None, None) => false,
                };
            }
            let by_glue = self_ty.is_some_and(|t| glue_family_holds(p, tys, b.value.def, t, args.get(1).and_then(arg_ty), seen));
            by_glue || inst.is_some_and(|a| crate::check::generics::prove_inst_trait_ref(p, b.value, &a))
        }
        Predicate::Trait(_) => false,
        Predicate::Outlives(..) | Predicate::TypeOutlives(..) => true,
        Predicate::AssocEq(_) => false,
    }
}

/// A written glue impl's bound `self_ty: d` (with `Rhs = rhs` for
/// `PartialEq` and `PartialOrd`), for a trait the glue walks know.
fn glue_family_holds(p: &Program, tys: &impl TyIntern, d: DefId, self_ty: Ty, rhs: Option<Ty>, seen: &mut GlueSeen) -> bool {
    let lang = &p.gx.lang;
    if d == lang.clone {
        glue_clone(p, tys, self_ty, seen)
    } else if d == lang.partial_eq {
        rhs == Some(self_ty) && glue_eq(p, tys, self_ty, seen)
    } else if d == lang.eq {
        glue_total_eq(p, tys, self_ty, seen)
    } else if d == lang.partial_ord {
        rhs == Some(self_ty) && glue_cmp(p, tys, self_ty, false, seen)
    } else if d == lang.ord {
        glue_cmp(p, tys, self_ty, true, seen)
    } else if d == lang.hash {
        glue_hash(p, tys, self_ty, seen)
    } else if d == lang.default {
        glue_default(p, tys, self_ty, seen)
    } else if d == lang.debug {
        glue_debug(p, tys, self_ty, seen)
    } else if d == lang.display {
        has_display(p, tys, self_ty)
    } else if d == lang.copy {
        is_copy(p, tys, self_ty)
    } else {
        false
    }
}

/// A written glue impl's matched params as solver args, for bounds
/// the glue walks do not prove (`S: BuildHasher`, `Vec<T>: Tr`):
/// None unless every assigned type is concrete and global (a
/// body-local or generic one stays with the glue walks, which can
/// only keep a reject).
fn inst_args(p: &Program, tys: &impl TyIntern, assign: &[Option<Ty>]) -> Option<Vec<GenericArg>> {
    let mut args: Vec<GenericArg> = Vec::with_capacity(assign.len());
    for a in assign {
        match *a {
            Some(t) if tys.is_global_ty(t) && !p.tys.flags(t).contains_param() && !p.tys.flags(t).contains_infer() => args.push(GenericArg::Ty(t)),
            Some(_) => return None,
            // An unassigned parameter stays as written (subst keeps a
            // param with no type argument), so the solver's open check
            // rejects a bound that names it.
            None => match p.tys.get(TyKind::Param(ParamId(args.len() as u32))) {
                Some(q) => args.push(GenericArg::Ty(q)),
                None => args.push(GenericArg::Lifetime(Region::Static)),
            },
        }
    }
    Some(args)
}

/// True when a written (not derived) impl of `trait_def` applies to
/// Adt type `t` and the glue can call its `method` (no method for a
/// marker trait like `Eq`, asked only as an impl bound): the impl
/// header matches `t` (and `rhs` for `PartialEq`/`PartialOrd`,
/// which glue only ever asks with `Rhs = Self`), its bounds hold by
/// glue walks, and it carries none of `bad_items` (a written `ne`
/// or `lt` changes what the glue's one call must mean; see
/// `compute_glue_methods`). Generic impls count: mono instantiates
/// the method for each concrete Adt (`ensure_glue_methods`), which
/// inserts exactly the impls this answer accepts. For `Clone`, a
/// `Copy` type's written `clone` does not count, as there.
fn written_glue_impl(
    p: &Program,
    tys: &impl TyIntern,
    t: Ty,
    trait_def: DefId,
    method: Option<&str>,
    bad_items: &[&str],
    seen: &mut GlueSeen,
) -> bool {
    use crate::traits::index::{SelfHead, Synthetic};
    if seen.on_walk(t) {
        return true;
    }
    let Some((def, _)) = tys.adt_parts(t) else { return false };
    let lang = &p.gx.lang;
    // A Copy type's written `clone` does not count. Asked of this instance
    // (`is_copy` checks the Copy impl's bounds): a header match alone made
    // `Reverse<String>` count as Copy through `impl<T: Copy> Copy for Reverse<T>`,
    // and so `Vec<Reverse<String>>` was not Clone.
    if trait_def == lang.clone
        && p.gx.index.impls_of_trait(lang.copy).iter().any(|&id| {
            p.gx.index.peek(id).head == SelfHead::Adt(def) && match_impl_pat(p, tys, p.gx.index.get(id).self_ty(&p.tys), t, &mut Vec::new())
        })
        && is_copy(p, tys, t)
    {
        return false;
    }
    let has_item = |rec: &crate::traits::index::ImplRecord, n: &str| {
        rec.items.iter().any(|&m| p.local_def(m.hash).is_some_and(|l| p.name(p.fn_info(l).name) == n))
    };
    // Written impls only (`peek`: the derived records it skips are no read).
    p.gx.index.impls_of_trait(trait_def).iter().any(|&id| {
        let rec = p.gx.index.peek(id);
        if rec.synthetic != Synthetic::None || rec.head != SelfHead::Adt(def) {
            return false;
        }
        if method.is_some_and(|m| !has_item(rec, m)) || bad_items.iter().any(|&n| has_item(rec, n)) {
            return false;
        }
        let mut assign: Vec<Option<Ty>> = Vec::new();
        if !match_impl_pat(p, tys, rec.self_ty(&p.tys), t, &mut assign) {
            return false;
        }
        if (trait_def == lang.partial_eq || trait_def == lang.partial_ord)
            && !p.tys.args(rec.trait_ref.args).get(1).and_then(|g| g.as_ty()).is_some_and(|r| match_impl_pat(p, tys, r, t, &mut assign))
        {
            return false;
        }
        seen.push(t);
        let ok = rec.generics.predicates.iter().all(|pred| glue_impl_pred_holds(p, tys, pred, &assign, seen));
        seen.pop();
        ok
    })
}

pub fn has_clone(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_clone(p, tys, t, &mut GlueSeen::default())
}

/// True when a `clone_from` of `t` by structure would run a hash
/// table's `clone_from`: `t` holds a `HashMap` or `HashSet` by
/// value, through fields and owning containers (not through a
/// reference or a shared `Rc`/`Arc`, whose `clone_from` only
/// moves the count). The check side rejects those calls (L0640):
/// a table's own `clone_from` reuses its bucket array in ways the
/// `clonefrom` glue does not model.
pub fn mentions_hash_table(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    fn walk(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
        use crate::ty::hash::HashTy;
        if let Some((k, _)) = crate::ty::hash::hash_parts(tys, t) {
            return matches!(k, HashTy::HashMap | HashTy::HashSet);
        }
        match tys.kind(t) {
            TyKind::Tuple(ts) => return tys.tys_of(ts).iter().any(|&e| walk(p, tys, e, seen)),
            TyKind::Array(e, _) => return walk(p, tys, e, seen),
            _ => {}
        }
        // Shared pointers never clone the table behind them.
        if tys.rc_elem(t).is_some() || tys.weak_elem(t).is_some() || tys.arc_weak_elem(t).is_some() {
            return false;
        }
        if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.refcell_elem(t)).or_else(|| tys.boxed_slice_elem(t)) {
            return walk(p, tys, e, seen);
        }
        if let Some((a, b)) = tys.result_args(t) {
            return walk(p, tys, a, seen) || walk(p, tys, b, seen);
        }
        // A `Box<T>` (not the shared forms above) clones into its pointee.
        if let Some(e) = tys.smart_elem(t) {
            return walk(p, tys, e, seen);
        }
        match tys.local_adt(t) {
            Some(a) => !adt_instance_fields_all(p, tys, a, t, GlueKey::HashTable, seen, &mut |f, seen| !walk(p, tys, f, seen)),
            None => false,
        }
    }
    walk(p, tys, t, &mut GlueSeen::default())
}

/// The walk behind `has_clone`; `seen` stops recursive generic types.
fn glue_clone(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    // `Cow<B>` clones its Owned value (`B: ToOwned` holds by its WF).
    if cow_arg(tys, t).is_some() {
        return true;
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => return env_proves(tys, t, p.gx.lang.clone, None),
        TyKind::Tuple(ts) => return tys.tys_of(ts).iter().all(|&e| glue_clone(p, tys, e, seen)),
        // A closure is Clone when its captures are (rustc's clone shim
        // clones each capture in order; cgen's tuple glue). A `&mut`
        // capture is not Clone.
        TyKind::Closure(_, a) => return closure_fields(tys, a).iter().all(|&e| glue_clone(p, tys, e, seen)),
        TyKind::Array(e, _) => return glue_clone(p, tys, e, seen),
        _ => {}
    }
    if tys.is_string_ty(t) || tys.is_exit_code(t) || tys.builtin_of(t) == Some(BuiltinTy::AccessError) {
        return true;
    }
    if tys.rc_elem(t).is_some() || tys.weak_elem(t).is_some() || tys.arc_weak_elem(t).is_some() {
        return true;
    }
    if let Some(Ok(b)) = crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::Clone) {
        return b;
    }
    // `Ref`/`RefMut` are not Clone (`Ref::clone` is an associated fn);
    // `Cell<T>` is Clone for `T: Copy`, `RefCell<T>` for `T: Clone`.
    if tys.guard_elem(t).is_some() {
        return false;
    }
    if let Some(e) = tys.cell_elem(t) {
        return is_copy(p, tys, e);
    }
    // `Box<str>` clones its bytes (check/own.rs).
    if crate::check::own::own_parts(tys, t).is_some_and(|(k, e)| k == crate::check::own::Owner::Box && e == STR) {
        return true;
    }
    // `Box<[T]>` clones its elements. The same walk (not a fresh
    // `has_clone`): a recursive `Tree<T> { Node(Box<[Tree<T>]>) }`
    // must meet itself on the stack, or the walk never ends.
    if let Some(e) = tys.boxed_slice_elem(t) {
        return glue_clone(p, tys, e, seen);
    }
    // Hash tables clone their args; the std hashers are Clone (W-STD-B).
    if let Some((k, a)) = crate::ty::hash::hash_parts(tys, t) {
        use crate::ty::hash::HashTy;
        return match k {
            HashTy::HashMap | HashTy::HashSet => a.iter().all(|&e| glue_clone(p, tys, e, seen)),
            HashTy::Entry | HashTy::OccupiedEntry | HashTy::VacantEntry => false,
            _ => true,
        };
    }
    if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)).or_else(|| tys.refcell_elem(t)) {
        return glue_clone(p, tys, e, seen);
    }
    if let Some((a, b)) = tys.result_args(t) {
        return glue_clone(p, tys, a, seen) && glue_clone(p, tys, b, seen);
    }
    if let Some(e) = owning_iter_elem(tys, t) {
        return glue_clone(p, tys, e, seen);
    }
    // A builtin iterator std makes Clone, when a bitwise copy is its
    // clone (`iter_clone_is_copy`).
    if tys.is_chars_ty(t) || crate::ty::iter::iter_parts(tys, t).is_some() {
        return iter_clone_is_copy(p, tys, t);
    }
    // std's `Reverse<T>` when it is Copy: its written `clone` is
    // `Reverse(self.0.clone())` (library/core/src/cmp.rs), so the glue's
    // bitwise copy is that clone exactly when `T`'s glue clone is a copy,
    // which `glue_clone` of a Copy `T` answers (a Copy `T` with a written
    // `Clone` stays rejected there). Without this, `Vec<Reverse<u32>>`,
    // `Option<Reverse<_>>`, and `VecDeque<Reverse<_>>` were not Clone: the
    // written impl of a Copy type does not count for glue (`written_glue_impl`).
    // A non-Copy `Reverse<T>` takes the written-impl path below.
    if let Some(a) = tys.local_adt(t)
        && p.is_reverse_adt(a)
        && is_copy(p, tys, t)
        && let Some((_, args)) = tys.adt_parts(t)
        && let Some(inner) = tys.args(args).first().and_then(|g| g.as_ty())
    {
        return glue_clone(p, tys, inner, seen);
    }
    match tys.local_adt(t) {
        Some(a) => {
            // A written non-generic impl: the glue calls its method.
            if p.gx.glue_methods.contains_key(&(a, p.gx.lang.clone)) {
                return true;
            }
            (p.derive_has(a, crate::check::derive_delta::bit::CLONE) && adt_instance_fields_all(p, tys, a, t, GlueKey::Clone, seen, &mut |f, seen| glue_clone(p, tys, f, seen)))
                || written_glue_impl(p, tys, t, p.gx.lang.clone, Some("clone"), &[], seen)
        }
        None => is_copy(p, tys, t) || written_glue_impl(p, tys, t, p.gx.lang.clone, Some("clone"), &[], seen),
    }
}

/// The element of a `vec::IntoIter<T>` (W-STD-A), whose Debug, Clone,
/// and Default std writes over what it still owns (cgen `CTy::VecIter`
/// glue): `IntoIter([..])`, a fresh buffer of clones, an empty one.
pub(crate) fn vec_into_iter_elem(tys: &impl TyIntern, t: Ty) -> Option<Ty> {
    match crate::ty::iter::iter_parts(tys, t)? {
        (crate::ty::iter::IterKind::VecIntoIter, a) => a.first().copied(),
        _ => None,
    }
}

/// core's Debug text of a `PhantomData<T>` type `t`
/// (`PhantomData<{type_name::<T>()}>`), or None when `t` is not one or
/// lust cannot print `T`'s type name exactly. It decides Debug of
/// `PhantomData` everywhere (glue walks, body derive queries, the
/// solver's record-less step), so cgen's glue only meets printable ones.
pub(crate) fn phantom_debug_text(p: &Program, tys: &impl TyIntern, t: Ty) -> Option<String> {
    let a = tys.local_adt(t)?;
    if !p.is_phantom_adt(a) {
        return None;
    }
    // An instance's args are global; a template application read in
    // collect is too. A body-local arg type (none in practice) gets None.
    let arg = match p.gx.adt_instance_of.get(a.0 as usize) {
        Some(Some((_, args))) => p.tys.args(*args).first()?.as_ty()?,
        _ => tys.adt_first_ty_arg(t)?,
    };
    if p.tys.get(tys.kind(arg)) != Some(arg) {
        return None;
    }
    let mut scratch = crate::ty::Tys::new(&p.tys);
    Some(format!("PhantomData<{}>", crate::check::any::type_name_of(p, &mut scratch, arg, false)?))
}

/// The element of a std owning iterator whose Debug and Clone hold
/// exactly when its element's do and that cgen writes as core does:
/// `vec::IntoIter<T>` (`CTy::VecIter`), `option::IntoIter<T>` and
/// `result::IntoIter<T>` (`CTy::OptIter`).
pub(crate) fn owning_iter_elem(tys: &impl TyIntern, t: Ty) -> Option<Ty> {
    use crate::ty::iter::IterKind as K;
    match crate::ty::iter::iter_parts(tys, t)? {
        (K::VecIntoIter | K::OptionIntoIter | K::ResultIntoIter, a) => a.first().copied(),
        _ => None,
    }
}

/// True when `t` is a builtin iterator (or `Chars`) whose std type is
/// Clone and whose stored state is plain data, so a bitwise copy is its
/// clone (cgen's `Clone` of a C tuple of Copy fields): every stored
/// part is a shared slice or `&str`, an index or flag, a Copy value (a
/// closure with Copy captures, a Copy pattern, a Copy item in a
/// `Peekable` slot), or such an iterator. Owning and mutable iterators
/// (`vec::IntoIter`, `IterMut`, the drains, the owning and `mut` hash
/// iterators, `Args`) and core's slice `SplitN`/`RSplitN` (not Clone)
/// stay false. The shared hash iterators (`Iter`, `Keys`, `Values`, set
/// `Iter`) hold only the raw scan state (`lust_hash_iter`), as
/// hashbrown's Clone `RawIter` does.
pub fn iter_clone_is_copy(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    use crate::ty::iter::{Field, IterKind as K, SliceSplitMode};
    if tys.is_chars_ty(t) {
        return true;
    }
    let Some((k, a)) = crate::ty::iter::iter_parts(tys, t) else { return false };
    let part = |t: Ty| if crate::ty::iter::is_iter(tys, t) { iter_clone_is_copy(p, tys, t) } else { is_copy(p, tys, t) };
    let ok_kind = match k {
        K::SliceIter | K::Range | K::RangeInclusive | K::Map | K::Filter | K::Enumerate | K::Rev | K::Take | K::Skip | K::Chain | K::Zip => true,
        K::RangeFrom | K::RangeTo | K::RangeToInclusive | K::RangeFull => true,
        K::Copied | K::Cloned | K::FilterMap | K::MapWhile | K::TakeWhile | K::SkipWhile | K::StepBy | K::Peekable | K::Inspect | K::RepeatN | K::ByNext => true,
        K::Windows | K::CharIndices | K::Flatten | K::OptionIntoIter | K::OptionIter | K::ResultIter | K::ResultIntoIter => true,
        K::StrSplit | K::StrMatches | K::EncodeUtf16 | K::Chunks | K::ChunksExact | K::Utf8Chunks => true,
        K::MapIter | K::MapKeys | K::MapValues | K::SetIter => true,
        K::SliceSplit => {
            a.get(2).and_then(|&m| crate::ty::iter::slice_split_mode(tys, m)).is_some_and(|m| !matches!(m, SliceSplitMode::SplitN | SliceSplitMode::RSplitN))
        }
        _ => false,
    };
    ok_kind
        && k.fields().iter().all(|f| match *f {
            Field::Usize | Field::Bool | Field::Chars | Field::Str | Field::OptRef(_, false) | Field::SliceRef(_, false) => true,
            Field::Arg(i) | Field::OptArg(i) => a.get(i as usize).is_some_and(|&t| part(t)),
            Field::PeekSlot(i) => a.get(i as usize).is_some_and(|&t| iter_item_is_copy(p, tys, t)),
            _ => false,
        })
}

/// True when the item type of the builtin iterator `t` is Copy, by its
/// kind (no interning; `ty::iter::item_ty` builds the type itself).
fn iter_item_is_copy(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    use crate::ty::iter::IterKind as K;
    if tys.is_chars_ty(t) {
        return true;
    }
    let Some((k, a)) = crate::ty::iter::iter_parts(tys, t) else { return false };
    let arg = |i: usize| a.get(i).copied().unwrap_or(ERROR);
    match k {
        K::SliceIter | K::Windows | K::Chunks | K::ChunksExact | K::SliceSplit | K::CharIndices | K::StrSplit | K::StrMatches | K::EncodeUtf16 => true,
        K::OptionIter | K::ResultIter | K::Range | K::RangeInclusive => true,
        K::MapIter | K::MapKeys | K::MapValues | K::SetIter => true,
        K::Map | K::FilterMap | K::MapWhile => is_copy(p, tys, arg(2)),
        K::Copied | K::Cloned => is_copy(p, tys, arg(1)),
        K::OptionIntoIter | K::ResultIntoIter | K::RepeatN => is_copy(p, tys, arg(0)),
        K::ByNext => is_copy(p, tys, arg(1)),
        K::Filter | K::Rev | K::Take | K::Skip | K::TakeWhile | K::SkipWhile | K::StepBy | K::Peekable | K::Inspect | K::Enumerate | K::Chain => {
            iter_item_is_copy(p, tys, arg(0))
        }
        K::Zip => iter_item_is_copy(p, tys, arg(0)) && iter_item_is_copy(p, tys, arg(1)),
        _ => false,
    }
}

/// True when the ordering operators and `partial_cmp` apply by
/// structure: scalars, strings, tuples, arrays, slices, `Vec`,
/// `Option`, `Result`, `Box`, `Rc`, references, and Adts that
/// derive `PartialOrd` with fields of such types (codegen compares
/// them with the `cmp` glue; tuples of scalars expand in lowering).
/// A field type whose `PartialOrd` impl is written by hand needs
/// that impl's method, which the glue cannot call: false.
pub fn has_ord(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_cmp(p, tys, t, false, &mut GlueSeen::default())
}

/// `has_ord` for `Ord`: the same shapes without floats, and Adts
/// that derive `Ord`.
pub fn has_total_ord(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_cmp(p, tys, t, true, &mut GlueSeen::default())
}

/// True when `PartialOrd::lt` of `t` is exactly the `PCmp` glue:
/// `has_ord` shapes, plus an Adt whose written `partial_cmp` the
/// glue can call even when an `Ord` impl also exists (then the
/// `Cmp` glue means `Ord::cmp`, which may disagree with `lt`).
/// Std's `sort` family compares with `lt`, so it needs both this
/// and `has_total_ord`.
pub fn has_partial_cmp(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_cmp_mode(p, tys, t, false, &mut GlueSeen::default(), true)
}

/// The structural walk behind `has_ord` and `has_total_ord`. `seen`
/// holds the Adt types on the walk (a recursive type through `Box`
/// or `Vec` compares by the same glue: the cycle adds no field type).
fn glue_cmp(p: &Program, tys: &impl TyIntern, t: Ty, total: bool, seen: &mut GlueSeen) -> bool {
    glue_cmp_mode(p, tys, t, total, seen, false)
}

/// `glue_cmp` with the `PCmp` rule (`pcmp`): a written `partial_cmp`
/// in `pcmp_methods` also answers, and a parameter bound by `Ord`
/// answers for its `PartialOrd` supertrait. Field walks that the
/// derive rule sends to `Ord` (`fields_total`) use the plain walk.
fn glue_cmp_mode(p: &Program, tys: &impl TyIntern, t: Ty, total: bool, seen: &mut GlueSeen, pcmp: bool) -> bool {
    // `Cow<B>` compares as the `B` it derefs to (W-STD-B).
    if let Some(b) = cow_arg(tys, t) {
        return glue_cmp_mode(p, tys, b, total, seen, pcmp);
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => {
            if total {
                env_proves(tys, t, p.gx.lang.ord, None)
            } else {
                env_proves(tys, t, p.gx.lang.partial_ord, Some(t)) || (pcmp && env_proves(tys, t, p.gx.lang.ord, None))
            }
        }
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Str | TyKind::Error => true,
        // Raw pointers order by address (w-cinterop); fat ones are not
        // built (tf-fatptr).
        TyKind::RawPtr(_, e) => raw_glue_ok(p, tys, e, false),
        TyKind::Float(_) => !total,
        TyKind::Infer(v) => v.is_int() || (v.is_float() && !total),
        TyKind::Ref(_, _, e) | TyKind::Array(e, _) | TyKind::Slice(e) => glue_cmp_mode(p, tys, e, total, seen, pcmp),
        // core's tuple impls of the derive family (`PartialOrd`
        // here, and `Eq`, `Hash`, `Default`, `Debug` below) stop
        // at 12 elements; `Clone` and `Copy` hold at every arity.
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_cmp_mode(p, tys, e, total, seen, pcmp)),
        _ => {
            if tys.is_string_ty(t) {
                return true;
            }
            // A guard (`Ref`, `MutexGuard`) has no `PartialOrd` or `Hash`.
            if tys.guard_elem(t).is_some() {
                return false;
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)) {
                return glue_cmp_mode(p, tys, e, total, seen, pcmp);
            }
            if let Some((a, b)) = tys.result_args(t) {
                return glue_cmp_mode(p, tys, a, total, seen, pcmp) && glue_cmp_mode(p, tys, b, total, seen, pcmp);
            }
            // The `Cmp` glue's written method needs no `Ord` impl on
            // the type (its `partial_cmp` is then every ordering), as
            // in `compute_glue_methods`.
            let no_ord = !total
                && !pcmp
                && tys.adt_parts(t).is_some_and(|(def, _)| {
                    !p.gx.index.impls_of_trait(p.gx.lang.ord).iter().any(|&id| {
                        p.gx.index.peek(id).head == crate::traits::index::SelfHead::Adt(def)
                            && match_impl_pat(p, tys, p.gx.index.get(id).self_ty(&p.tys), t, &mut Vec::new())
                    })
                });
            // std's `Reverse<T>`: `T`'s order, swapped (`is_reverse_adt`).
            if let Some(a) = tys.local_adt(t)
                && p.is_reverse_adt(a)
                && let Some((_, args)) = tys.adt_parts(t)
                && let Some(inner) = tys.args(args).first().and_then(|g| g.as_ty())
            {
                return glue_cmp_mode(p, tys, inner, total, seen, pcmp);
            }
            match tys.local_adt(t) {
                // A written non-generic `PartialOrd` (and no `Ord`): the
                // glue calls its `partial_cmp`.
                Some(a) if !total && p.gx.glue_methods.contains_key(&(a, p.gx.lang.partial_ord)) => true,
                // The `PCmp` glue calls a written `partial_cmp` even
                // when an `Ord` impl also exists.
                Some(a) if pcmp && !total && p.gx.pcmp_methods.contains_key(&(a, p.gx.lang.partial_ord)) => true,
                // A written generic `PartialOrd`: as the two arms
                // above, through `written_glue_impl` (mono makes the
                // instance `ensure_glue_methods` records).
                Some(_) if pcmp && !total && written_glue_impl(p, tys, t, p.gx.lang.partial_ord, Some("partial_cmp"), &["lt"], seen) => true,
                Some(a)
                    if no_ord
                        && !p.derive_has(a, crate::check::derive_delta::bit::ORD)
                        && written_glue_impl(p, tys, t, p.gx.lang.partial_ord, Some("partial_cmp"), &["lt", "le", "gt", "ge"], seen) =>
                {
                    true
                }
                Some(a) => {
                    if !p.derive_has(a, if total { crate::check::derive_delta::bit::ORD } else { crate::check::derive_delta::bit::PARTIAL_ORD }) {
                        return false;
                    }
                    // rustc's derived `partial_cmp` on a non-generic
                    // type that also derives `Ord` is `Some(self.cmp(
                    // other))`: its fields must then order by `Ord`.
                    let generic = p.gx.adt_instance_of.get(a.0 as usize).is_some_and(|o| o.is_some())
                        || p.gx.adt_generics.get(a.0 as usize).is_some_and(|g| !g.params.is_empty());
                    let fields_total = total || (!generic && p.derive_has(a, crate::check::derive_delta::bit::ORD));
                    adt_fields_all(p, a, t, GlueKey::Cmp(fields_total, pcmp && !fields_total), seen, &mut |f, seen| {
                        glue_cmp_mode(p, tys, f, fields_total, seen, pcmp && !fields_total)
                    })
                }
                None => false,
            }
        }
    }
}

/// True when `ok` holds for every field type of a materialized
/// generic Adt instance `a` (type `t`); a non-generic Adt or a
/// template answers true (collect checked the derive's fields, and
/// a template's instances answer for themselves). `seen` stops
/// recursion through recursive generic types.
fn adt_instance_fields_all(
    p: &Program,
    tys: &impl TyIntern,
    a: AdtId,
    t: Ty,
    key: GlueKey,
    seen: &mut GlueSeen,
    ok: &mut dyn FnMut(Ty, &mut GlueSeen) -> bool,
) -> bool {
    let instance = p.gx.adt_instance_of.get(a.0 as usize).is_some_and(|o| o.is_some());
    if !instance {
        // A generic Adt whose instance does not exist yet (collect's
        // derive field checks run before materialize; `Cow<Table>` in a
        // non-generic Adt): the template's fields are checked at its own
        // definition, so the derive's bounds on this type's args decide
        // (`impl<T: Debug> Debug for Cow<T>`), as `is_copy` asks them.
        // An arg that still holds an inference var keeps the old answer.
        let generic = p.gx.adt_generics.get(a.0 as usize).is_some_and(|g| !g.params.is_empty());
        if key == GlueKey::HashTable || !generic || p.is_phantom_adt(a) {
            return true;
        }
        let Some((_, args)) = tys.adt_parts(t) else { return true };
        return tys.args(args).iter().filter_map(|g| g.as_ty()).all(|x| tys.flags(x).contains_infer() || ok(x, seen));
    }
    seen.fields_all(p, a, t, key, ok)
}

/// Rustc's derive bounds: a derived impl on a generic Adt bounds every
/// type parameter by the trait (`#[derive(Clone)] struct W<T>` is
/// `impl<T: Clone> Clone for W<T>`), even one only a `fn() -> T` or
/// `PhantomData<T>` field uses. True when `ok` holds for every type
/// argument of instance `a` (always for std's `PhantomData`, whose
/// impls have no bounds, and for non-instances).
fn derive_args_all(p: &Program, a: AdtId, seen: &mut GlueSeen, ok: &mut dyn FnMut(Ty, &mut GlueSeen) -> bool) -> bool {
    let Some(Some((tdef, args))) = p.gx.adt_instance_of.get(a.0 as usize) else { return true };
    if p.defs.local_adt(*tdef).is_some_and(|tmpl| p.is_phantom_adt(tmpl)) {
        return true;
    }
    p.tys.args(*args).iter().filter_map(|g| g.as_ty()).all(|x| ok(x, seen))
}

/// True when `ok` holds for every field type of Adt `a` (the Adt of
/// type `t`), or when the fields are a generic template's (a template
/// body: its instances answer with their own substituted fields).
/// `seen` stops recursion through recursive types; `key` names `ok`
/// for its memo (`GlueSeen`).
fn adt_fields_all(p: &Program, a: AdtId, t: Ty, key: GlueKey, seen: &mut GlueSeen, ok: &mut dyn FnMut(Ty, &mut GlueSeen) -> bool) -> bool {
    let template = p.gx.adt_instance_of.get(a.0 as usize).is_none_or(|o| o.is_none())
        && p.gx.adt_generics.get(a.0 as usize).is_some_and(|g| !g.params.is_empty());
    if template {
        return true;
    }
    seen.fields_all(p, a, t, key, ok)
}

/// Which field walk a `GlueSeen` memo entry answers: the glue trait,
/// and for the ordering walk its `total` and `pcmp` flags.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum GlueKey {
    Hash,
    Default,
    Cmp(bool, bool),
    Clone,
    TotalEq,
    Eq,
    Debug,
    HashTable,
}

/// The state of one glue walk (`has_clone`, `has_default`, ...): the
/// Adt types on the walk (`stack`; one on it again counts as holding,
/// which stops recursive types), and the answers of finished Adt field
/// walks. A struct DAG (fields that share sub-types) walked without
/// the memo costs one visit per path, which grows exponentially with
/// its depth. An answer is kept only when its walk met no type on the
/// stack (`hits` did not move): it then holds in every walk.
#[derive(Default)]
struct GlueSeen {
    stack: Vec<Ty>,
    memo: HashMap<(GlueKey, Ty), bool>,
    hits: u32,
}

impl GlueSeen {
    /// True when `t` is on the walk already (a cycle, which counts as
    /// holding).
    fn on_walk(&mut self, t: Ty) -> bool {
        let on = self.stack.contains(&t);
        self.hits += u32::from(on);
        on
    }

    fn push(&mut self, t: Ty) {
        self.stack.push(t);
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    /// `ok` over every field type and derive arg of Adt `a` (type `t`).
    fn fields_all(&mut self, p: &Program, a: AdtId, t: Ty, key: GlueKey, ok: &mut dyn FnMut(Ty, &mut GlueSeen) -> bool) -> bool {
        if let Some(&r) = self.memo.get(&(key, t)) {
            return r;
        }
        if self.on_walk(t) {
            return true;
        }
        let hits = self.hits;
        self.push(t);
        let all = p.adt(a).variants.iter().all(|v| v.fields.iter().all(|f| ok(f.ty, self))) && derive_args_all(p, a, self, ok);
        self.pop();
        if self.hits == hits {
            self.memo.insert((key, t), all);
        }
        all
    }
}

/// The (trait, type) pairs an auto trait walk has entered (`auto_structural`;
/// a pair is entered once): a list for `pop`, and a set for `contains`, since
/// a walk over a large type enters thousands.
#[derive(Default)]
pub struct AutoSeen {
    list: Vec<(DefId, Ty)>,
    set: crate::intern::FxMap<(DefId, Ty), ()>,
}

impl AutoSeen {
    pub fn contains(&self, k: &(DefId, Ty)) -> bool {
        self.set.contains_key(k)
    }
    pub fn push(&mut self, k: (DefId, Ty)) {
        self.list.push(k);
        self.set.insert(k, ());
    }
    pub fn pop(&mut self) {
        if let Some(k) = self.list.pop() {
            self.set.remove(&k);
        }
    }
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

/// An auto trait (`Send`, `Sync`, `Unpin`) of `t` by structure (ARCH
/// 2.7), shared by body checks (`Ctx::auto_holds`, on resolved
/// types) and collect-time proofs: a parameter answers from `env`,
/// references and raw pointers by the fixed table, compound types
/// and Adts through their parts (a generic Adt's fields substituted
/// with this application's args), closures through their captures,
/// `dyn` through its bounds, and the builtin std types by std's own
/// impls (`Rc` and `env::Args` are neither Send nor Sync; `Vec`,
/// `Box`, and `Rc` are Unpin for every element type). None when an
/// inference variable, a projection, or another form leaves it open.
/// `seen` holds the (trait, type) pairs on the walk: a cycle counts as
/// holding. The trait is part of the key: `&T: Send` asks `T: Sync`.
pub fn auto_structural(p: &Program, tys: &mut impl TyIntern, env: &[Predicate], trait_def: DefId, t: Ty, seen: &mut AutoSeen) -> Option<bool> {
    let lang = &p.gx.lang;
    if seen.contains(&(trait_def, t)) {
        return Some(true);
    }
    let all = |tys: &mut _, parts: Vec<Ty>, seen: &mut AutoSeen| -> Option<bool> {
        let mut unknown = false;
        for e in parts {
            match auto_structural(p, tys, env, trait_def, e, seen) {
                Some(false) => return Some(false),
                None => unknown = true,
                _ => {}
            }
        }
        if unknown { None } else { Some(true) }
    };
    match tys.kind(t) {
        TyKind::Error => Some(true),
        // Every integer and float type has every auto trait, so a
        // literal not yet fixed does too (`vec![1, 2]` moved into a
        // `Send` closure).
        TyKind::Infer(v) if v.is_int() || v.is_float() => Some(true),
        TyKind::Infer(_) | TyKind::Projection(..) => None,
        // An `impl Trait` type leaks its hidden type's auto traits, as in
        // rustc (check/opaque.rs); a bound that names the trait holds too.
        TyKind::Opaque(od, a) => {
            if env.iter().any(|pred| matches!(pred, Predicate::Trait(b) if b.value.def == trait_def && tys.args(b.value.args)[0].as_ty() == Some(t))) {
                return Some(true);
            }
            let h = *p.gx.opaque_hidden.get(&od)?;
            let args: Vec<GenericArg> = tys.args(a).to_vec();
            let h = subst(tys, h, &args);
            seen.push((trait_def, t));
            let out = auto_structural(p, tys, env, trait_def, h, seen);
            seen.pop();
            out
        }
        TyKind::Param(_) => Some(env.iter().any(|pred| {
            matches!(pred, Predicate::Trait(b) if b.value.def == trait_def && tys.args(b.value.args)[0].as_ty() == Some(t))
        })),
        TyKind::RawPtr(..) => Some(trait_def == lang.unpin),
        TyKind::Dyn(..) => {
            let a = crate::check::object::dyn_auto(p, tys, t);
            Some(if trait_def == lang.send {
                a.send
            } else if trait_def == lang.sync {
                a.sync
            } else {
                a.unpin
            })
        }
        TyKind::Ref(_, m, e) => {
            if trait_def == lang.unpin {
                return Some(true);
            }
            seen.push((trait_def, t));
            // `&T: Send` needs `T: Sync`; every other pair keeps the trait.
            let need = if trait_def == lang.send && !m.is_mut() { lang.sync } else { trait_def };
            auto_structural(p, tys, env, need, e, seen)
        }
        TyKind::Slice(e) | TyKind::Array(e, _) => {
            seen.push((trait_def, t));
            auto_structural(p, tys, env, trait_def, e, seen)
        }
        TyKind::Tuple(ts) => {
            seen.push((trait_def, t));
            let elems = tys.tys_of(ts).to_vec();
            all(tys, elems, seen)
        }
        TyKind::Closure(_, a) => {
            seen.push((trait_def, t));
            let caps = closure_fields(tys, a);
            all(tys, caps, seen)
        }
        TyKind::Int(_)
        | TyKind::Float(_)
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Unit
        | TyKind::Str
        | TyKind::Never
        | TyKind::FnDef(..)
        | TyKind::FnPtr(_) => Some(true),
        _ => {
            // The std types' own facts (check/sync.rs keeps the table).
            if let Some(r) = crate::check::sync::builtin_auto(p, tys, env, trait_def, t, seen) {
                return r;
            }
            // `Box<T>` is `Unpin` for every `T` (std's explicit impl).
            if trait_def == lang.unpin && tys.boxed_elem(t).is_some() {
                return Some(true);
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.boxed_elem(t)) {
                seen.push((trait_def, t));
                return auto_structural(p, tys, env, trait_def, e, seen);
            }
            // `Cow<B>` has an auto trait when `&B` and its Owned type
            // do (std's auto impls for Cow): str and slices by their
            // Owned shapes, a sized `B` by itself, and any other `B`
            // by the concrete Owned field of its instance.
            if let Some(b) = cow_arg(tys, t) {
                seen.push((trait_def, t));
                let rb = tys.mk_ref(false, b);
                let owned = match tys.kind(b) {
                    TyKind::Str => Some(STRING),
                    TyKind::Slice(e) => Some(tys.mk_vec(e)),
                    _ if is_sized_ty(tys, b) && has_clone(p, tys, b) => Some(b),
                    _ => match tys.local_adt(t) {
                        Some(a) => match p.adt(a).variants.get(1).and_then(|v| v.fields.first()) {
                            Some(f) if !matches!(tys.kind(f.ty), TyKind::Projection(..)) => Some(f.ty),
                            _ => None,
                        },
                        None => None,
                    },
                };
                return match owned {
                    Some(o) => all(tys, vec![rb, o], seen),
                    None => None,
                };
            }
            if let Some((a, b)) = tys.result_args(t) {
                seen.push((trait_def, t));
                return all(tys, vec![a, b], seen);
            }
            if let Some(a) = tys.local_adt(t) {
                // Written `unsafe impl Send for S {}` replaces the
                // structural rule (W-THREADS).
                if let Some(r) = crate::check::sync::explicit_auto_impl(p, tys, t, trait_def) {
                    return Some(r);
                }
                seen.push((trait_def, t));
                let args: Vec<GenericArg> = tys.adt_parts(t).map(|(_, a)| tys.args(a).to_vec()).unwrap_or_default();
                // std's `unsafe impl<T: Sync + Send, F: Send> Sync for
                // LazyLock<T, F>`: only one thread ever runs `F`.
                let name = p.adt(a).name;
                if trait_def == lang.sync
                    && crate::intern::source_name(p.interner.get(name)) == "LazyLock"
                    && p.display_path(name) == "std::sync::LazyLock"
                {
                    let (Some(v), Some(f)) = (args.first().and_then(|g| g.as_ty()), args.get(1).and_then(|g| g.as_ty())) else { return None };
                    let mut unknown = false;
                    for (tr, e) in [(lang.send, v), (lang.sync, v), (lang.send, f)] {
                        match auto_structural(p, tys, env, tr, e, seen) {
                            Some(false) => return Some(false),
                            None => unknown = true,
                            Some(true) => {}
                        }
                    }
                    return if unknown { None } else { Some(true) };
                }
                // `PhantomData<T>` has T's auto traits (std source, W-STD-C).
                if p.is_phantom_adt(a) {
                    let held: Vec<Ty> = args.iter().filter_map(|g| g.as_ty()).collect();
                    return all(tys, held, seen);
                }
                let raw: Vec<Ty> = p.adt(a).variants.iter().flat_map(|v| v.fields.iter().map(|f| f.ty)).collect();
                let fields: Vec<Ty> =
                    raw.into_iter().map(|f| if args.is_empty() || !tys.flags(f).contains_param() { f } else { subst(tys, f, &args) }).collect();
                return all(tys, fields, seen);
            }
            None
        }
    }
}

/// True when `t` implements `Hash` by structure: scalars other than
/// floats, strings, references, tuples, arrays, slices, the builtin
/// containers of such types, and Adts that derive `Hash` with
/// fields of such types. (Lust has no `Hasher` yet: this answers
/// bounds, no hashing code exists.)
/// For the six `core::ops` range types (builtin iterator kinds over an
/// integer, ty/iter.rs): `Some(None)` for `RangeFull`, else
/// `Some(Some(elem))`. Std derives `PartialEq`, `Eq`, and `Hash` on each
/// over its fields (`start`, `end`, and `RangeInclusive`'s `exhausted`
/// flag); their C form is the tuple of those fields, so the tuple glue
/// gives std's answers. `RangeFull`'s C form is one dummy flag: equal
/// always, but it hashes no glue (std writes nothing).
fn range_elem(tys: &impl TyIntern, t: Ty) -> Option<Option<Ty>> {
    let (k, a) = crate::ty::iter::iter_parts(tys, t)?;
    if !k.is_range() {
        return None;
    }
    Some(if k == crate::ty::iter::IterKind::RangeFull { None } else { a.first().copied() })
}

pub fn has_hash(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_hash(p, tys, t, &mut GlueSeen::default())
}

fn glue_hash(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    if let Some(b) = cow_arg(tys, t) {
        return glue_hash(p, tys, b, seen);
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.hash, None),
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Str | TyKind::Error => true,
        // A raw pointer or fn pointer hashes its address as a usize
        // (w-cinterop; core's `impl<F: FnPtr> Hash for F`). Fat raw
        // pointers have no Hash glue (tf-fatptr).
        TyKind::RawPtr(_, e) => raw_glue_ok(p, tys, e, false),
        TyKind::FnPtr(_) => true,
        TyKind::Infer(v) => v.is_int(),
        TyKind::Ref(_, _, e) | TyKind::Array(e, _) | TyKind::Slice(e) => glue_hash(p, tys, e, seen),
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_hash(p, tys, e, seen)),
        _ => {
            if tys.is_string_ty(t) {
                return true;
            }
            // `RangeFull`'s C form holds a dummy flag: no Hash glue.
            if let Some(e) = range_elem(tys, t) {
                return e.is_some_and(|e| glue_hash(p, tys, e, seen));
            }
            if tys.guard_elem(t).is_some() {
                return false;
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)) {
                return glue_hash(p, tys, e, seen);
            }
            if let Some((a, b)) = tys.result_args(t) {
                return glue_hash(p, tys, a, seen) && glue_hash(p, tys, b, seen);
            }
            match tys.local_adt(t) {
                // A written impl the glue calls (`gx.written_hash`).
                Some(a) if p.gx.written_hash.contains_key(&a) => true,
                Some(a) if p.derive_has(a, crate::check::derive_delta::bit::HASH) => adt_fields_all(p, a, t, GlueKey::Hash, seen, &mut |f, seen| glue_hash(p, tys, f, seen)),
                // A written generic impl (`impl<T: Hash> Hash for
                // VecDeque<T>`): the glue calls its instance
                // (`gx.generic_hash`) when its bounds hold by glue.
                _ => written_glue_impl(p, tys, t, p.gx.lang.hash, Some("hash"), &["hash_slice"], seen),
            }
        }
    }
}

/// True when `Default::default()` builds a value of `t` by
/// structure (the `default` glue): scalars, `String`, `Vec`,
/// `Option`, tuples, arrays of at most 32 elements, `Box` and `Rc`
/// of such types, `&str` and `&[T]`, and Adts that derive `Default`
/// with fields of such types.
/// A reference type's `Default`: core's `&str` and `&[T]`, or a
/// written `impl Default for &'static A` (`gx.ref_default_methods`) at
/// a shared reference of any lifetime (`&'static A` coerces to `&'a A`).
pub fn ref_default(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    match tys.kind(t) {
        TyKind::Ref(_, _, e) if matches!(tys.kind(e), TyKind::Str | TyKind::Slice(_)) => true,
        TyKind::Ref(_, Mut::Shared, e) => tys.local_adt(e).is_some_and(|a| p.gx.ref_default_methods.contains_key(&a)),
        _ => false,
    }
}

pub fn has_default(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_default(p, tys, t, &mut GlueSeen::default())
}

fn glue_default(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    // `Cow<B>` defaults to `Owned(<B as ToOwned>::Owned::default())`
    // (std's impl): str and slices always qualify (String and Vec),
    // a sized `B` through itself, and any other `B` through the
    // concrete Owned field of its instance when one is registered.
    if let Some(b) = cow_arg(tys, t) {
        return match tys.kind(b) {
            TyKind::Str | TyKind::Slice(_) => true,
            _ if is_sized_ty(tys, b) && has_clone(p, tys, b) => glue_default(p, tys, b, seen),
            _ => match tys.local_adt(t) {
                Some(a) => match p.adt(a).variants.get(1).and_then(|v| v.fields.first()) {
                    Some(f) if !matches!(tys.kind(f.ty), TyKind::Projection(..)) => glue_default(p, tys, f.ty, seen),
                    _ => false,
                },
                None => false,
            },
        };
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.default, None),
        TyKind::Int(_) | TyKind::Float(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Error => true,
        TyKind::Infer(v) => v.is_int() || v.is_float(),
        TyKind::Ref(..) => ref_default(p, tys, t),
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_default(p, tys, e, seen)),
        TyKind::Array(e, n) => match tys.const_kind(n) {
            ConstKind::Int(0) => true,
            ConstKind::Int(k) if k <= 32 => glue_default(p, tys, e, seen),
            _ => false,
        },
        _ => {
            if tys.is_string_ty(t) || tys.vec_elem(t).is_some() || tys.option_elem(t).is_some() || vec_into_iter_elem(tys, t).is_some() {
                return true;
            }
            // std's empty `Box<str>`, `Rc<str>`, `Arc<[T]>`, kin (check/own.rs).
            if let Some(d) = crate::check::own::own_default(tys, t) {
                return d;
            }
            match crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::Default) {
                Some(Ok(b)) => return b,
                Some(Err(e)) => return glue_default(p, tys, e, seen),
                None => {}
            }
            // core's `impl<T: Default> Default` for RefCell and Cell.
            if let Some(e) = tys.refcell_elem(t).or_else(|| tys.cell_elem(t)) {
                return glue_default(p, tys, e, seen);
            }
            // std's `impl<K, V, S: Default> Default for HashMap<K, V, S>`
            // (and HashSet), and the hashers' own impls (W-STD-B, cgen/hash.rs
            // `default_body`). Entry and the iterators have none here.
            if let Some((k, args)) = crate::ty::hash::hash_parts(tys, t) {
                use crate::ty::hash::HashTy as H;
                return match k {
                    H::HashMap | H::HashSet => args.last().is_some_and(|&s| glue_default(p, tys, s, seen)),
                    H::RandomState | H::DefaultHasher | H::BuildHasherDefault => true,
                    _ => false,
                };
            }
            if tys.guard_elem(t).is_some() {
                return false;
            }
            // alloc's `impl<T> Default for Weak<T>` (rc and sync): `Weak::new()`
            // for every sized `T` (cgen's Default glue of both).
            if let Some(e) = tys.weak_elem(t).or_else(|| tys.arc_weak_elem(t)) {
                return is_sized_ty(tys, e);
            }
            // alloc's `impl<T> Default for Box<[T]>` is the empty
            // boxed slice: no bound on `T` (the `smart_elem` step
            // below would ask `[T]: Default`, which never holds).
            if tys.boxed_slice_elem(t).is_some() {
                return true;
            }
            if let Some(e) = tys.smart_elem(t) {
                return glue_default(p, tys, e, seen);
            }
            match tys.local_adt(t) {
                Some(a) => {
                    // A written non-generic impl: the glue calls its method.
                    if p.gx.glue_methods.contains_key(&(a, p.gx.lang.default)) {
                        return true;
                    }
                    if written_glue_impl(p, tys, t, p.gx.lang.default, Some("default"), &[], seen) {
                        return true;
                    }
                    let adt = p.adt(a);
                    if !p.derive_has(a, crate::check::derive_delta::bit::DEFAULT) {
                        return false;
                    }
                    // An enum's default is its unit `#[default]` variant.
                    adt.is_enum || adt_fields_all(p, a, t, GlueKey::Default, seen, &mut |f, seen| glue_default(p, tys, f, seen))
                }
                None => written_glue_impl(p, tys, t, p.gx.lang.default, Some("default"), &[], seen),
            }
        }
    }
}

/// True when `t` implements `Eq` by structure: the `glue_eq`
/// shapes with no floats anywhere (a float's `==` is not total),
/// Adts that derive `Eq` over such fields, and written generic
/// `Eq` impls whose bounds hold. Asked by `glue_impl_pred_holds`
/// (`impl<K: Eq + Hash, ..> PartialEq` on a map type) and by
/// `derive_query_holds` for an `Eq` bound on a tuple, array,
/// slice, or reference (the solver has no impl records for those
/// heads); a derived `Eq`'s own field check stays with the solver
/// in `check_derive_bounds`.
pub fn has_total_eq(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_total_eq(p, tys, t, &mut GlueSeen::default())
}

/// The walk behind `has_total_eq`; `seen` stops recursive generic types.
fn glue_total_eq(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    if let Some(b) = cow_arg(tys, t) {
        return glue_total_eq(p, tys, b, seen);
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.eq, None),
        TyKind::FnPtr(_) => true,
        TyKind::RawPtr(_, e) => raw_glue_ok(p, tys, e, true),
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Str | TyKind::Error => true,
        TyKind::Infer(v) => v.is_int(),
        TyKind::Ref(_, _, e) => glue_total_eq(p, tys, e, seen),
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_total_eq(p, tys, e, seen)),
        TyKind::Array(e, _) | TyKind::Slice(e) => glue_total_eq(p, tys, e, seen),
        _ => {
            // std's `ExitCode` is `PartialEq` but not `Eq`.
            if tys.is_string_ty(t) || tys.builtin_of(t) == Some(BuiltinTy::AccessError) {
                return true;
            }
            // The sync types `sync_glue` calls `PartialEq` are
            // `Duration` and `ThreadId`, and both are `Eq` in std.
            if let Some(Ok(b)) = crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::PartialEq) {
                return b;
            }
            if let Some(e) = range_elem(tys, t) {
                return e.is_none_or(|e| glue_total_eq(p, tys, e, seen));
            }
            if tys.guard_elem(t).is_some() {
                return false;
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)) {
                return glue_total_eq(p, tys, e, seen);
            }
            if let Some((a, b)) = tys.result_args(t) {
                return glue_total_eq(p, tys, a, seen) && glue_total_eq(p, tys, b, seen);
            }
            // std's `Eq` for the hash collections also needs the
            // key to hash (`K: Eq + Hash`), as in `glue_eq`.
            if let Some((k, args)) = crate::ty::hash::hash_parts(tys, t) {
                use crate::ty::hash::HashTy;
                return match k {
                    HashTy::HashMap => {
                        has_hash(p, tys, args[0]) && glue_total_eq(p, tys, args[0], seen) && glue_total_eq(p, tys, args[1], seen)
                    }
                    HashTy::HashSet => has_hash(p, tys, args[0]) && glue_total_eq(p, tys, args[0], seen),
                    _ => false,
                };
            }
            match tys.local_adt(t) {
                Some(a) => {
                    (p.derive_has(a, crate::check::derive_delta::bit::EQ) && adt_instance_fields_all(p, tys, a, t, GlueKey::TotalEq, seen, &mut |f, seen| glue_total_eq(p, tys, f, seen)))
                        || written_glue_impl(p, tys, t, p.gx.lang.eq, None, &[], seen)
                }
                None => written_glue_impl(p, tys, t, p.gx.lang.eq, None, &[], seen),
            }
        }
    }
}

pub fn has_eq(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_eq(p, tys, t, &mut GlueSeen::default())
}

/// The walk behind `has_eq`; `seen` stops recursive generic types.
fn glue_eq(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    if let Some(b) = cow_arg(tys, t) {
        return glue_eq(p, tys, b, seen);
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.partial_eq, Some(t)),
        // Fn pointers compare by address in Rust; fn items do not.
        TyKind::FnPtr(_) => true,
        // Raw pointers compare by address (w-cinterop), a `str` or `[T]`
        // pointer by address and length; a dyn pointer's vtable address
        // is not fixed by rustc, so it has no `==` (tf-fatptr).
        TyKind::RawPtr(_, e) => raw_glue_ok(p, tys, e, true),
        TyKind::Float(_) => true,
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Str | TyKind::Error => true,
        TyKind::Infer(v) => v.is_int() || v.is_float(),
        TyKind::Ref(_, _, e) => glue_eq(p, tys, e, seen),
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_eq(p, tys, e, seen)),
        TyKind::Array(e, _) | TyKind::Slice(e) => glue_eq(p, tys, e, seen),
        _ => {
            if tys.is_string_ty(t) || tys.is_exit_code(t) || tys.builtin_of(t) == Some(BuiltinTy::AccessError) {
                return true;
            }
            if let Some(Ok(b)) = crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::PartialEq) {
                return b;
            }
            if let Some(e) = range_elem(tys, t) {
                return e.is_none_or(|e| glue_eq(p, tys, e, seen));
            }
            if tys.guard_elem(t).is_some() {
                return false;
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)) {
                return glue_eq(p, tys, e, seen);
            }
            if let Some((a, b)) = tys.result_args(t) {
                return glue_eq(p, tys, a, seen) && glue_eq(p, tys, b, seen);
            }
            // HashMap and HashSet (W-STD-B): keys the glue hashes and
            // compares, a map's values compared.
            if let Some((k, args)) = crate::ty::hash::hash_parts(tys, t) {
                use crate::ty::hash::HashTy;
                return match k {
                    HashTy::HashMap => has_hash(p, tys, args[0]) && glue_eq(p, tys, args[0], seen) && glue_eq(p, tys, args[1], seen),
                    HashTy::HashSet => has_hash(p, tys, args[0]) && glue_eq(p, tys, args[0], seen),
                    _ => false,
                };
            }
            match tys.local_adt(t) {
                // A written non-generic impl: the glue calls its `eq`.
                Some(a) if p.gx.glue_methods.contains_key(&(a, p.gx.lang.partial_eq)) => true,
                Some(a) => {
                    (p.derive_has(a, crate::check::derive_delta::bit::PARTIAL_EQ) && adt_instance_fields_all(p, tys, a, t, GlueKey::Eq, seen, &mut |f, seen| glue_eq(p, tys, f, seen)))
                        || written_glue_impl(p, tys, t, p.gx.lang.partial_eq, Some("eq"), &["ne"], seen)
                }
                None => written_glue_impl(p, tys, t, p.gx.lang.partial_eq, Some("eq"), &["ne"], seen),
            }
        }
    }
}

pub fn has_debug(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    glue_debug(p, tys, t, &mut GlueSeen::default())
}

/// The walk behind `has_debug`; `seen` stops recursive generic types.
fn glue_debug(p: &Program, tys: &impl TyIntern, t: Ty, seen: &mut GlueSeen) -> bool {
    if let Some(b) = cow_arg(tys, t) {
        return glue_debug(p, tys, b, seen);
    }
    // A `RefCell<dyn Tr>` or `Mutex<dyn Tr>` has no Debug glue yet
    // (check/own.rs): a reject, even for `dyn Any`.
    if crate::check::own::dyn_wrapped_elem(tys, t).is_some() {
        return false;
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.debug, None),
        TyKind::Opaque(..) => crate::check::opaque::opaque_bound_holds(p, tys, t, p.gx.lang.debug),
        TyKind::Float(_) => true,
        // A raw pointer's Debug is its address (w-cinterop); not built
        // for fat ones (tf-fatptr).
        TyKind::RawPtr(_, e) => raw_glue_ok(p, tys, e, false),
        // Core implements Debug for every fn pointer signature,
        // higher-ranked ones included (the address, like a raw pointer).
        TyKind::FnPtr(_) => true,
        // `impl Debug for !` (a `catch_unwind` of a closure that always
        // panics gives `Result<!, _>`).
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Unit | TyKind::Str | TyKind::Error | TyKind::Never => true,
        TyKind::Infer(v) => v.is_int() || v.is_float(),
        TyKind::Ref(_, _, e) => glue_debug(p, tys, e, seen),
        TyKind::Tuple(ts) => tys.tys_of(ts).len() <= 12 && tys.tys_of(ts).iter().all(|&e| glue_debug(p, tys, e, seen)),
        TyKind::Array(e, _) | TyKind::Slice(e) => glue_debug(p, tys, e, seen),
        // std's `impl Debug for dyn Any (+ Send (+ Sync))`: `Any { .. }`.
        // Any other dyn type is Debug when its principal trait has a
        // `Debug` supertrait (the vtable's fmt shim, check/object.rs).
        TyKind::Dyn(b, _) => {
            let b = tys.dyn_bounds(b);
            let Some(pr) = b.principal.map(|pr| pr.value.def) else { return false };
            (pr == p.gx.lang.any && !b.auto.unpin && (b.auto.send || !b.auto.sync))
                || crate::check::object::principal_has_supertrait(p, pr, p.gx.lang.debug)
        }
        _ => {
            // `env::Args` prints `Args { inner: [..] }` in Rust; lust has
            // no such rendering, so it is not Debug here (a reject).
            if tys.is_string_ty(t) || tys.is_exit_code(t) || tys.weak_elem(t).is_some() || tys.arc_weak_elem(t).is_some()
                || tys.builtin_of(t) == Some(BuiltinTy::AccessError)
                // `fmt::Arguments` Debug is its Display (R9).
                || crate::ty::fmt::fmt_kind(tys, t) == Some(crate::ty::fmt::FmtTy::Arguments)
            {
                return true;
            }
            match crate::check::sync::sync_glue(tys, t, crate::check::sync::SyncGlue::Debug) {
                Some(Ok(b)) => return b,
                Some(Err(e)) => return glue_debug(p, tys, e, seen),
                None => {}
            }
            if let Some(e) = tys.cell_elem(t) {
                return is_copy(p, tys, e) && glue_debug(p, tys, e, seen);
            }
            if let Some(e) = tys.vec_elem(t).or_else(|| tys.option_elem(t)).or_else(|| tys.smart_elem(t)).or_else(|| tys.refcell_elem(t)).or_else(|| owning_iter_elem(tys, t)) {
                return glue_debug(p, tys, e, seen);
            }
            // HashMap and HashSet print their keys and values (W-STD-B).
            if let Some((k, a)) = crate::ty::hash::hash_parts(tys, t) {
                use crate::ty::hash::HashTy;
                return match k {
                    HashTy::HashMap => glue_debug(p, tys, a[0], seen) && glue_debug(p, tys, a[1], seen),
                    HashTy::HashSet => glue_debug(p, tys, a[0], seen),
                    HashTy::RandomState => true,
                    _ => false,
                };
            }
            if let Some((a, b)) = tys.result_args(t) {
                return glue_debug(p, tys, a, seen) && glue_debug(p, tys, b, seen);
            }
            // `TrySendError`'s and `TryLockError`'s Debug hide the
            // value, so they need no Debug on T (TRIAGE T5).
            if tys.adt_parts(t).is_some_and(|(d, _)| d == generics::try_send_error_def() || d == generics::try_lock_error_def()) {
                return true;
            }
            if tys.local_adt(t).is_some_and(|a| p.is_phantom_adt(a)) {
                return phantom_debug_text(p, tys, t).is_some();
            }
            match tys.local_adt(t) {
                Some(a) => {
                    (p.derive_has(a, crate::check::derive_delta::bit::DEBUG) && adt_instance_fields_all(p, tys, a, t, GlueKey::Debug, seen, &mut |f, seen| glue_debug(p, tys, f, seen)))
                        || crate::check::fmt::user_impl_holds(p, tys, t, p.gx.lang.debug)
                        // A conditional written impl (`impl<K: Debug>
                        // Debug for W<K>`) whose bounds the glue walks
                        // prove for this instance: `ensure_fmt_methods`
                        // selects the same impl for cgen. Without it a
                        // compound type over the Adt (`&W<u8>`, the
                        // `entries` item of a generic Debug body at its
                        // instance) read as not Debug (fx-bench4-gaps).
                        || written_glue_impl(p, tys, t, p.gx.lang.debug, Some("fmt"), &[], seen)
                }
                None => false,
            }
        }
    }
}

pub fn has_display(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    if let Some(b) = cow_arg(tys, t) {
        return has_display(p, tys, b);
    }
    match tys.kind(t) {
        TyKind::Param(_) | TyKind::Projection(..) => env_proves(tys, t, p.gx.lang.display, None),
        TyKind::Opaque(..) => crate::check::opaque::opaque_bound_holds(p, tys, t, p.gx.lang.display),
        TyKind::Float(_) => true,
        TyKind::Int(_) | TyKind::Bool | TyKind::Char | TyKind::Str | TyKind::Error => true,
        TyKind::Infer(v) => v.is_int() || v.is_float(),
        TyKind::Ref(_, _, e) => has_display(p, tys, e),
        // A dyn type is Display when its principal trait has a
        // `Display` supertrait (check/object.rs, as for Debug).
        TyKind::Dyn(b, _) => {
            tys.dyn_bounds(b).principal.is_some_and(|pr| crate::check::object::principal_has_supertrait(p, pr.value.def, p.gx.lang.display))
        }
        _ => {
            tys.is_string_ty(t)
                || tys.smart_elem(t).is_some_and(|e| has_display(p, tys, e))
                || tys.poison_elem(t).is_some()
                || matches!(crate::ty::sync::sync_kind(tys, t), Some(crate::ty::sync::SyncTy::SendError))
                || tys.adt_parts(t).is_some_and(|(d, _)| {
                    d == crate::check::generics::fmt_error_def()
                        || d == generics::utf8_error_def()
                        || d == generics::from_utf8_error_def()
                        || d == generics::recv_error_def()
                        || d == generics::try_recv_error_def()
                        || d == generics::recv_timeout_error_def()
                        || d == generics::try_lock_error_def()
                        || d == generics::location_def()
                        || d == generics::borrow_error_def()
                        || d == generics::borrow_mut_error_def()
                        || d == generics::try_from_int_error_def()
                        || d == generics::try_from_slice_error_def()
                        || d == generics::char_try_from_error_def()
                        || d == generics::try_from_char_error_def()
                        || d == generics::parse_int_error_def()
                        || d == generics::parse_float_error_def()
                        || d == generics::parse_bool_error_def()
                        || d == generics::parse_char_error_def()
                        // `Infallible`'s Display is `match *self {}`:
                        // it exists, and no value reaches it.
                        || d == generics::infallible_def()
                        // `NonZero<T>` displays its integer (TRIAGE T5).
                        || d == generics::nonzero_def()
                        || d == generics::try_send_error_def()
                })
                || tys.builtin_of(t) == Some(BuiltinTy::AccessError)
                // `fmt::Arguments` Display formats its parts (R9).
                || crate::ty::fmt::fmt_kind(tys, t) == Some(crate::ty::fmt::FmtTy::Arguments)
                || (tys.local_adt(t).is_some() && crate::check::fmt::user_impl_holds(p, tys, t, p.gx.lang.display))
        }
    }
}

/// True if values of this type own heap memory and need a drop call.
/// A `Param` or projection type may (rustc's drop check assumes so).
pub fn needs_drop(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    needs_drop_sub(p, tys, t, &[], false)
}

/// `needs_drop` for the const evaluator: exact where the type is
/// concrete (a generic Adt's fields read through its written args),
/// and false for a type the evaluator cannot resolve (an unbound
/// `Param`, a projection, an inference var). A generic `const fn`
/// body is judged by the static const-fn check under template
/// rules; the evaluator only ever sees concrete calls, and saying
/// "may drop" there would reject valid const items.
pub fn needs_drop_const(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    needs_drop_sub(p, tys, t, &[], true)
}

/// Resolves a written generic argument through the enclosing Adt's
/// args, for `needs_drop_sub` and `interior_mut_sub`: a direct
/// `Param(i)` becomes `args[i]`; anything else stays as written
/// (a `Param` nested deeper keeps the caller's frame index, which
/// the next Adt's args then resolve the same way).
fn resolve_arg(tys: &impl TyIntern, t: Ty, args: &[GenericArg]) -> Ty {
    match tys.kind(t) {
        TyKind::Param(i) => args.get(i.0 as usize).and_then(|g| g.as_ty()).unwrap_or(t),
        _ => t,
    }
}

/// The `needs_drop` walk. `args` are the written args of the Adt
/// whose fields are being read (empty at the top). With `exact`
/// false this is `needs_drop` exactly: an Adt's fields keep their
/// template types and an unresolved `Param` may drop. With `exact`
/// true, fields read through the Adt's args and an unresolved
/// `Param` or projection does not drop (see `needs_drop_const`).
fn needs_drop_sub(p: &Program, tys: &impl TyIntern, t: Ty, args: &[GenericArg], exact: bool) -> bool {
    match tys.kind(t) {
        TyKind::Param(i) => {
            return if exact {
                match args.get(i.0 as usize).and_then(|g| g.as_ty()) {
                    Some(u) => needs_drop_sub(p, tys, u, &[], exact),
                    None => false,
                }
            } else {
                true
            };
        }
        TyKind::Projection(..) => return !exact,
        TyKind::Tuple(ts) => return tys.tys_of(ts).iter().any(|&e| needs_drop_sub(p, tys, e, args, exact)),
        TyKind::Closure(_, a) => return closure_fields(tys, a).iter().any(|&e| needs_drop_sub(p, tys, e, args, exact)),
        // An empty array holds nothing to drop, whatever its element
        // (`[String; 0]`, rustc `needs_drop` is false).
        TyKind::Array(_, n) if tys.const_kind(n) == ConstKind::Int(0) => return false,
        TyKind::Array(e, _) => return needs_drop_sub(p, tys, e, args, exact),
        _ => {}
    }
    // `MaybeUninit<T>` never drops its bytes (TRIAGE T5).
    if tys.adt_parts(t).is_some_and(|(d, _)| d == generics::maybe_uninit_def()) {
        return false;
    }
    if tys.is_string_ty(t) || tys.vec_elem(t).is_some() || tys.smart_elem(t).is_some() || tys.weak_elem(t).is_some() || tys.arc_weak_elem(t).is_some() {
        return true;
    }
    // A join handle detaches (or frees) its thread; a `Thread` is counted.
    if matches!(
        crate::ty::sync::sync_kind(tys, t),
        Some(
            crate::ty::sync::SyncTy::JoinHandle
                | crate::ty::sync::SyncTy::ScopedJoinHandle
                | crate::ty::sync::SyncTy::Thread
                | crate::ty::sync::SyncTy::Sender
                | crate::ty::sync::SyncTy::SyncSender
                | crate::ty::sync::SyncTy::Receiver
        )
    ) {
        return true;
    }
    if let Some(e) = tys.refcell_elem(t).or_else(|| tys.cell_elem(t)).or_else(|| tys.lock_elem(t).map(|l| l.1)).or_else(|| tys.poison_elem(t)).or_else(|| {
        matches!(crate::ty::sync::sync_kind(tys, t), Some(crate::ty::sync::SyncTy::SendError | crate::ty::sync::SyncTy::OnceLock))
            .then(|| tys.adt_first_ty_arg(t))
            .flatten()
    }) {
        return needs_drop_sub(p, tys, e, args, exact);
    }
    if let Some(e) = tys.option_elem(t) {
        return needs_drop_sub(p, tys, e, args, exact);
    }
    if let Some((a, b)) = tys.result_args(t) {
        return needs_drop_sub(p, tys, a, args, exact) || needs_drop_sub(p, tys, b, args, exact);
    }
    // The hash tables own their buckets; an Entry its key (W-STD-B).
    if let Some((k, a)) = crate::ty::hash::hash_parts(tys, t) {
        use crate::ty::hash::HashTy;
        return match k {
            HashTy::HashMap | HashTy::HashSet => true,
            HashTy::Entry | HashTy::VacantEntry => a.first().is_some_and(|&e| needs_drop_sub(p, tys, e, args, exact)),
            _ => false,
        };
    }
    if let Some((k, a)) = crate::ty::iter::iter_parts(tys, t) {
        // `vec::IntoIter` owns its buffer. Other kinds drop what their
        // args hold (an over-count for the `Map` item arg is harmless:
        // the drop glue only drops stored fields).
        // A hash iterator owns its table only when made by value; a
        // drain drops what it has not yielded and gives the table back.
        if k.is_hash() {
            return k.is_hash_owning() || k.is_hash_drain();
        }
        return matches!(
            k,
            crate::ty::iter::IterKind::VecIntoIter
                | crate::ty::iter::IterKind::RecvIntoIter
                | crate::ty::iter::IterKind::VecDrain
                | crate::ty::iter::IterKind::StringDrain
        ) || a.iter().any(|&e| needs_drop_sub(p, tys, e, args, exact));
    }
    match tys.local_adt(t) {
        Some(a) => {
            if p.is_manually_drop_adt(a) {
                return false;
            }
            if p.gx.drop_methods.contains_key(&a) || (exact && p.drop_adts.contains(&a)) {
                return true;
            }
            if exact {
                // Read the fields through this instance's written
                // args, resolved through the enclosing frame first.
                let own: Vec<GenericArg> = match tys.adt_parts(t) {
                    Some((_, aargs)) => tys
                        .args(aargs)
                        .iter()
                        .map(|g| match g {
                            GenericArg::Ty(u) => GenericArg::Ty(resolve_arg(tys, *u, args)),
                            other => *other,
                        })
                        .collect(),
                    None => Vec::new(),
                };
                p.adt(a).variants.iter().any(|v| v.fields.iter().any(|f| needs_drop_sub(p, &p.tys, f.ty, &own, exact)))
            } else {
                p.adt(a).variants.iter().any(|v| v.fields.iter().any(|f| needs_drop_sub(p, &p.tys, f.ty, &[], exact)))
            }
        }
        None => false,
    }
}

/// True if values of this type can be mutated through a shared
/// reference (they contain an `UnsafeCell`, not behind a pointer):
/// the E0492 question for a borrow of a temporary that a `const`
/// or `static` item would promote to `'static`. Heap pointer types
/// (`Box`, `Vec`, `Rc`) answer by their own cell, never by their
/// element: the element lives on the heap, not in the promoted
/// memory. An unresolved `Param` answers false, as in
/// `needs_drop_const`.
pub fn interior_mut(p: &Program, tys: &impl TyIntern, t: Ty) -> bool {
    interior_mut_sub(p, tys, t, &[])
}

fn interior_mut_sub(p: &Program, tys: &impl TyIntern, t: Ty, args: &[GenericArg]) -> bool {
    match tys.kind(t) {
        TyKind::Param(i) => {
            return match args.get(i.0 as usize).and_then(|g| g.as_ty()) {
                Some(u) => interior_mut_sub(p, tys, u, &[]),
                None => false,
            };
        }
        TyKind::Tuple(ts) => return tys.tys_of(ts).iter().any(|&e| interior_mut_sub(p, tys, e, args)),
        TyKind::Array(e, _) | TyKind::Slice(e) => return interior_mut_sub(p, tys, e, args),
        TyKind::Closure(_, a) => return closure_fields(tys, a).iter().any(|&e| interior_mut_sub(p, tys, e, args)),
        // Behind a reference or pointer the cell is not in this value.
        TyKind::Ref(..) | TyKind::RawPtr(..) => return false,
        _ => {}
    }
    if tys.cell_elem(t).is_some()
        || tys.refcell_elem(t).is_some()
        || tys.lock_elem(t).is_some()
        || crate::ty::sync::sync_kind(tys, t) == Some(crate::ty::sync::SyncTy::OnceLock)
    {
        return true;
    }
    if crate::ty::sync::atomic_of(tys, t).is_some() {
        return true;
    }
    if let Some(e) = tys.option_elem(t) {
        return interior_mut_sub(p, tys, e, args);
    }
    if let Some((a, b)) = tys.result_args(t) {
        return interior_mut_sub(p, tys, a, args) || interior_mut_sub(p, tys, b, args);
    }
    match tys.local_adt(t) {
        Some(a) => {
            // `UnsafeCell` itself is a shipped-core Adt whose one
            // field is its element; the name is the marker. The
            // std once/lazy types are the same: their lust models
            // hold the value without an `UnsafeCell` field to find.
            if matches!(
                crate::intern::source_name(p.interner.get(p.adt(a).name)),
                "UnsafeCell" | "Once" | "OnceLock" | "OnceCell" | "LazyLock" | "LazyCell"
            ) {
                return true;
            }
            let own: Vec<GenericArg> = match tys.adt_parts(t) {
                Some((_, aargs)) => tys
                    .args(aargs)
                    .iter()
                    .map(|g| match g {
                        GenericArg::Ty(u) => GenericArg::Ty(resolve_arg(tys, *u, args)),
                        other => *other,
                    })
                    .collect(),
                None => Vec::new(),
            };
            p.adt(a).variants.iter().any(|v| v.fields.iter().any(|f| interior_mut_sub(p, &p.tys, f.ty, &own)))
        }
        None => false,
    }
}

/// Layout shapes for the layout answers user code can observe
/// (`size_of`, `align_of`, `size_of_val`, `offset_of!`, const
/// evaluation): `Program`'s shapes, except that an adapter type whose
/// manifest does not list it as exact-layout (ARCH 3.6) has none, so
/// the query rejects with L0652 instead of answering with a size the
/// real crate does not have. Codegen keeps using `Program` itself.
pub struct UserLayout<'p>(pub &'p Program);

impl crate::ty::layout::AdtShapes for UserLayout<'_> {
    fn adt_shape(&self, id: AdtId) -> Option<crate::ty::layout::AdtShape> {
        let p = self.0;
        let adt = p.adts.get(id.0 as usize)?;
        // Std source module types (W-STD-C) keep std's niches out of
        // their source form, so no layout query may see them; the
        // field-less `PhantomData` (size 0, align 1) is exact, and
        // so is the unit-struct `System` allocator (size 0, align 1).
        // `OnceCell<T>` is std's `UnsafeCell<Option<T>>`: its source
        // fields (the `Option<T>` and a `PhantomData`) with the niche
        // hidden.
        // `ManuallyDrop<T>` is core's repr(transparent) wrapper over its
        // `MaybeDangling<T>`, itself repr(transparent) over `T`: the
        // source form is exact, and `T`'s niche stays visible.
        if p.std_source_span(adt.span) && !p.is_phantom_adt(id) && !p.is_known_allocator_adt(id) {
            match p.display_path(adt.name).as_str() {
                "core::cell::OnceCell" => return p.adt_shape(id).map(|s| crate::ty::layout::AdtShape { hide_niche: true, ..s }),
                "core::mem::ManuallyDrop" | "core::mem::MaybeDangling" => return p.adt_shape(id),
                _ => return None,
            }
        }
        if let Some(a) = p.adapter_at(adt.span) {
            let path = p.display_path(adt.name);
            if !a.exact_layout.iter().any(|t| *t == path) {
                return None;
            }
        }
        p.adt_shape(id)
    }
}

/// Layout shapes for the engine in `ty::layout`, read from the collected tables.
impl crate::ty::layout::AdtShapes for Program {
    fn adt_shape(&self, id: AdtId) -> Option<crate::ty::layout::AdtShape> {
        let adt = self.adts.get(id.0 as usize)?;
        Some(crate::ty::layout::AdtShape {
            repr: adt.repr,
            is_enum: adt.is_enum,
            variants: adt.variants.iter().map(|v| v.fields.iter().map(|f| f.ty).collect()).collect(),
            discs: adt.variants.iter().map(|v| v.disc).collect(),
            hide_niche: false,
        })
    }
}

/// True when the program uses `Cow` (W-STD-B: the synthetic generic std
/// enum must be instantiated): user source names it in a type, value, or
/// pattern path, or calls a fn that hands one out without the caller
/// naming it. Those are the lossy decoders (std routes, not spliced
/// source) and every spliced shipped fn whose signature names `Cow`. A
/// mention inside spliced shipped source alone does not count: an
/// `impl Option<Cow<_>>` in core's option.rs would otherwise turn the
/// generic machinery on for every compile that splices it (tf-std2,
/// realish20 collect 19 to 74 ms).
fn uses_cow(p: &mut Program, ast: &Ast, interner: &crate::intern::Interner) -> bool {
    let cow = interner.intern(generics::COW);
    let shipped = |sp: Span| p.shipped_crate_at(sp).is_some();
    let hit = |segs: &[ast::PathSeg]| segs.iter().any(|s| s.name == cow);
    let mut shipped_cow_tys: Vec<Span> = Vec::new();
    for (i, t) in ast.tys.iter().enumerate() {
        if matches!(t, TyExpr::Path(segs) if hit(segs)) {
            let sp = ast.ty_span(ast::TyId(i as u32));
            if !shipped(sp) {
                return true;
            }
            shipped_cow_tys.push(sp);
        }
    }
    let user_value = ast.exprs.iter().enumerate().any(|(i, e)| matches!(e, ast::Expr::Path(segs) if hit(segs)) && !shipped(ast.expr_span(ast::ExprId(i as u32))));
    let user_pat = ast.pats.iter().enumerate().any(|(i, pat)| {
        matches!(pat, ast::Pat::Path(segs) | ast::Pat::TupleStruct { path: segs, .. } if hit(segs)) && !shipped(ast.pat_span(ast::PatId(i as u32)))
    });
    if user_value || user_pat {
        return true;
    }
    // Names of fns that hand out a `Cow`: the routed lossy decoders, and
    // shipped fns whose parameter or return types hold a shipped `Cow` type.
    let mut names = vec![interner.intern("from_utf8_lossy"), interner.intern("to_string_lossy")];
    if !shipped_cow_tys.is_empty() {
        let within = |outer: Span| shipped_cow_tys.iter().any(|s| s.file == outer.file && s.lo >= outer.lo && s.hi <= outer.hi);
        let mut sig = |def: &ast::FnDef| {
            if shipped(def.name_span)
                && (def.ret.is_some_and(|r| within(ast.ty_span(r))) || def.params.iter().any(|q| within(ast.ty_span(q.ty))))
            {
                names.push(def.name);
            }
        };
        for it in &ast.items {
            match &it.kind {
                ItemKind::Fn(def) => sig(def),
                ItemKind::Impl(imp) => imp.fns.iter().for_each(&mut sig),
                ItemKind::Trait(tr) => tr.fns.iter().for_each(&mut sig),
                _ => {}
            }
        }
    }
    let used = ast.exprs.iter().enumerate().any(|(i, e)| {
        let named = match e {
            ast::Expr::MethodCall { name, .. } => names.contains(name),
            ast::Expr::Path(segs) => segs.last().is_some_and(|s| names.contains(&s.name)),
            _ => false,
        };
        named && !shipped(ast.expr_span(ast::ExprId(i as u32)))
    });
    p.gx.cow_fns = names;
    used
}

/// The `B` of a `Cow<'a, B>` type (W-STD-B): its Debug, Display,
/// equality, order, and hash are those of `B` (std derefs to it).
pub fn cow_arg(tys: &impl TyIntern, t: Ty) -> Option<Ty> {
    let (d, a) = tys.adt_parts(t)?;
    if d != generics::cow_def() {
        return None;
    }
    tys.args(a).iter().find_map(|g| g.as_ty())
}
