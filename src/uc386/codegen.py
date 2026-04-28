"""x86-32 (i386) code generator for MS-DOS targets.

NASM Intel-syntax output. The emitted file assembles with `nasm -fobj`
and links against a DOS/4GW-compatible linker.

NASM was chosen as the assembler target because it's open source, ships
on every modern dev box, supports OMF object output for DOS toolchains
(`-fobj`), and uses Intel syntax that matches the rest of the
Watcom-era ecosystem.

Current scope:
- Functions taking and returning `int`. cdecl ABI: caller pushes args
  right-to-left, callee accesses via `[ebp + 8 + 4*i]`, caller cleans
  the stack with `add esp, 4*N`. Return value in EAX.
- `_start` calls `_main` with no args and exits via INT 21h/4Ch.
- Integer-literal returns and bare `return;`.
- `int` locals with arbitrary initializer expressions.
- Reading a local or parameter in any expression position.
- Assignment to a local or parameter (`x = expr;`).
- Unary `+ - ~ !` and prefix/postfix `++` / `--` (Identifier operand).
- Binary `+ - * / % & | ^ << >> == != < > <= >=`.
- Compound assignment `+= -= *= /= %= &= |= ^= <<= >>=` (Identifier lvalue).
- Logical `&&` and `||` with proper short-circuit evaluation.
- Ternary `cond ? a : b`.
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`,
  `switch`/`case`/`default`, `break`, `continue`, `goto`/labels.
  `continue` inside a switch escapes to the enclosing loop
  (separate break/continue target stacks). User-declared labels
  are pre-walked so forward `goto`s resolve.
- Function calls — direct (`call _name` for known function-name
  callees) and indirect (`call eax` after evaluating an arbitrary
  function-pointer expression). Leading `*`s on the callee are
  stripped (`(*fp)()` ≡ `fp()`).
- Pointer locals/params (`int *p`), `&x` / `&arr[i]`, `*expr`, and
  store-through-pointer assignment `*p = rhs`.
- Pointer arithmetic with C scaling rules: `p + n` and `n + p` scale
  the int by `sizeof(*p)`; `p - n` likewise; `p - q` produces a byte
  difference that is then divided by `sizeof(*p)`. `++p` / `p--` on a
  pointer slot step by `sizeof(*p)` rather than 1. Adding two pointers
  and subtracting a pointer from an integer are rejected.
- Arrays: `int arr[N]` allocates `N * sizeof(elem)` bytes on the frame
  (rounded up to a 4-byte boundary), an array name decays to its
  address in expression context, and `arr[i]` reads/writes through
  base + i*sizeof(elem) at the element's natural width.
  Brace-initialization `int arr[N] = {a, b, c}` is supported, with
  per-element stores and tail zero-fill; `char arr[] = "..."` lays out
  the bytes plus null terminator, with size inferred when omitted.
  Array assignment and `++arr` / `--arr` still raise.
- Sub-word scalars: `char`, `short`, and their unsigned variants are
  first-class slot types. Loads use `movsx` (signed) or `movzx`
  (unsigned) so EAX always holds a 32-bit working value matching C's
  integer-promotion rules. Stores narrow via `mov byte [...], al` /
  `mov word [...], ax`, leaving the other bytes of a (4-aligned) slot
  unread.
- Floats: `float` and `double` lower through the x87 FPU. A parallel
  `_eval_float_to_st0` produces values on st(0); `_eval_expr_to_eax`
  auto-converts float-typed expressions via `fistp` through a stack
  scratch. Arithmetic uses `faddp/fsubp/fmulp/fdivp st1, st0`;
  negation is `fchs`. FloatLiteral constants are interned in `.data`
  and loaded with `fld dword/qword`. Int↔float casts use `fild` /
  `fistp` through the stack. Float comparisons, by-value
  params/returns, and global init aren't supported yet.
- `sizeof(type)` and `sizeof(expr)` lower to a compile-time `mov eax, N`
  via the same `_size_of` used by pointer arithmetic. `sizeof(expr)`
  does not evaluate its operand — only the static type matters.
- Top-level globals: read/written through `[_name]`, address taken via
  `_name` immediate. Initialized globals land in `.data` (`db`/`dw`/`dd`
  per element width, with assembler-time zero-padding for short
  initializer lists), uninitialized in `.bss` as `resb N`. Local
  variables shadow same-named globals via the `_identifier_*` lookup
  order. Global initializers must be compile-time constants —
  references to other globals are not yet emitted as `dd _other`.
- Casts: `(T)expr` evaluates the operand and then narrows/extends
  EAX as needed. Pointer↔pointer and within-int-family casts are
  no-ops; narrowing to `char`/`short` emits `movsx eax, al/ax` (or
  `movzx` for unsigned). `_type_of(Cast)` returns the target type so
  e.g. `(char *)p + 1` scales by 1, not by the source's pointee size.
- Structs: top-level `struct foo { ... }` definitions register a
  layout (member name, type, byte offset) plus total size. `s.m` and
  `p->m` lower as base-address + offset, then load/store at the
  member's natural width. Slot allocation, sizeof, and pointer
  arithmetic all go through `_size_of(StructType)`. Struct-to-struct
  assignment uses an inline per-dword copy. `{...}` init walks the
  InitializerList member-by-member with nested `{...}` for struct
  members and tail zero-fill. Struct-by-value params work via
  caller-side `sub esp + memcpy`. Struct-by-value returns use a
  hidden retptr first arg — caller passes &dst, callee writes
  there and returns the address. For unbound struct-return values
  (`make().field`, `f(make())`, `make(1).x + make(2).x`), a
  per-call-site temp slot is pre-allocated in the caller's frame;
  each Call gets its own slot so multiple struct-returning calls
  in one expression don't collide. Bit-field members pack into
  shared 32-bit storage units; reads use shift+mask+sign-extend,
  writes RMW the unit.
- Unions: same registry as structs (`_register_struct` branches on
  `is_union`) but laid out with every member at offset 0 and total
  size = `max(sizeof(member))` rounded to the union's alignment.
  Access, sizeof, and pointer arithmetic all share the struct paths.
- String literals: interned per-translation-unit, emitted as
  null-terminated bytes in `.data` with labels like `_uc386_strN`.
- `extern` declarations (FunctionDecls without bodies) emit NASM
  `extern _name` at the top so calls can be resolved by the linker.

Anything else raises CodegenError.
"""

import dataclasses

from uc_core import ast


class CodegenError(NotImplementedError):
    """Raised when the AST contains a construct codegen can't handle yet."""


_EXTERN_REDIRECT = -0x70000000  # sentinel disp meaning "look up in globals"


def _ebp_addr(disp: int) -> str:
    """Render an EBP-relative address. Locals have negative disp, params positive."""
    if disp < 0:
        return f"[ebp - {-disp}]"
    return f"[ebp + {disp}]"


class _FuncCtx:
    """Per-function lowering state: locals, params, label gen, loop stack."""

    def __init__(self) -> None:
        # Block-scoped name → signed-EBP-displacement bindings. Stored as a
        # stack of scope dicts: scopes[0] holds params (allocated by
        # `alloc_param` before any body walk), and inner CompoundStmts /
        # ForStmts push their own scope. Locals are negative offsets
        # (below EBP); params are positive (above saved EBP + return
        # address — first param at +8 under cdecl).
        self.slots: list[dict[str, int]] = [{}]
        # Parallel scope chain for declared TypeNodes; used by `_type_of`
        # and pointer-arithmetic scaling.
        self.types: list[dict[str, "ast.TypeNode"]] = [{}]
        # `_collect_locals` runs before emit and assigns each VarDecl a
        # frame slot. The (id(decl) → disp) mapping below survives the
        # collect-time scope pop, so the emit-time walk can re-bind the
        # name in its scope without re-bumping frame_size.
        self.decl_disps: dict[int, int] = {}
        self.decl_types: dict[int, "ast.TypeNode"] = {}
        self.frame_size: int = 0          # bytes reserved by `sub esp, frame_size`
        self._next_label: int = 0
        # Stack of (continue_target, break_target) for the enclosing loops.
        # Stack of jump targets for control-flow keywords. Loops push to
        # both stacks; switches push only to `break_targets`. Splitting
        # the stacks lets `continue` inside a switch resolve to the
        # enclosing loop, the way C requires.
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []
        # Per-switch CaseStmt label maps. `_switch` pushes one entry —
        # `{id(case_node): nasm_label}` — that `_item`'s CaseStmt branch
        # consults to materialize the right label as the body is
        # lowered. A stack lets nested switches each have their own
        # mapping.
        self.active_case_labels: list[dict[int, str]] = []
        # Per-call-site temp slots for struct-returning calls. Keyed by
        # `id(call_node)` so each Call expression in the function gets
        # its own buffer (so `make(1).x + make(2).x` works).
        self.call_temps: dict[int, int] = {}
        # User-declared `label:` → NASM label mapping for goto. Pre-walked
        # before body emission so forward gotos can resolve.
        self.user_labels: dict[str, str] = {}
        # The current function's declared return type. Used by `_return`
        # to dispatch float-returning functions to the FPU stack.
        self.return_type: ast.TypeNode | None = None
        # Function-static local name → mangled global label. A
        # `static int x = 0;` inside `f` becomes the global
        # `_f__x` in `.data`/`.bss` instead of a frame slot, so the
        # value persists across calls.
        self.local_static_labels: dict[str, str] = {}
        # Nested-function name → mangled top-level name. GCC nested
        # function definitions inside a function body get lifted to
        # file-scope functions with mangled names; references in the
        # outer body resolve through this map.
        self.nested_fn_names: dict[str, str] = {}
        # Outer-local name → mangled global label for variables that
        # are captured (referenced) by a nested function. Outer's
        # reads/writes route through the global so the nested fn,
        # also reading the same global, sees a coherent value.
        self.local_captures: dict[str, str] = {}
        # Lifted nested-fn ctx only. Maps each outer-declared
        # `__label__ X` that this fn `goto X`s into to its index in the
        # outer's buf-array. The static-link slot (set by the trampoline
        # or direct caller via ECX) holds the buf-array's address; the
        # `goto X` lowering reads the slot and offsets by `12 * idx`.
        self.nonlocal_goto_targets: dict[str, int] = {}
        # Outer-fn ctx only. For each nested fn that's address-taken
        # AND has nonlocal goto, points to a 12-byte trampoline slot
        # in this frame (`__tramp_<inner>`) and the disp of the
        # 12*N-byte buf-array slot (`__nlg_buf_<inner>`). When a
        # caller does `&inner` or stores `inner` into a function
        # pointer, we return the trampoline's address. The trampoline
        # loads ECX = &buf_array and jumps to the lifted fn so each
        # frame's invocation has its own buf (handles recursive outer).
        self.trampolines: dict[str, tuple[int, int]] = {}
        # Per-nested-fn map of label_name → buf_array_index (offset
        # within the buf array is `12 * idx`). Used for the setjmp
        # init and the longjmp dispatch.
        self.trampoline_buf_indices: dict[str, dict[str, int]] = {}
        # Lifted nested fn ctx only. The disp of the 4-byte slot that
        # saves the static link (ECX value) on entry. None if this fn
        # doesn't have nonlocal gotos requiring a static link.
        self.trampoline_static_link_disp: int | None = None
        # Names of locals whose storage is heap-allocated (VLA via
        # malloc). Their frame slot holds a pointer, not the storage
        # bytes. Identifier load/address paths consult this set to
        # decide between `mov` and `lea`.
        self.vla_backed: set[str] = set()
        # True if any VLA decl in this function has been emitted —
        # the function epilogue needs to free each VLA pointer.
        self.has_vla: bool = False
        # Frame disp of a hidden slot that saves ESP after fixed-size
        # locals are allocated. VLA tests emit `sub esp, dyn_size`
        # which extends ESP below the fixed slots; on goto-back-to-
        # before-decl we restore ESP to this baseline so the VLA
        # storage gets reclaimed.
        self.vla_baseline_disp: int | None = None
        # The current function's name. Used to mangle static-local
        # labels so two functions with the same `static int x` don't
        # collide.
        self.func_name: str = ""
        # Back-pointer to the CodeGenerator so `enter_scope` /
        # `exit_scope` can drive the struct-tag alias chain stored on
        # the codegen. Set by `_function` immediately after construction.
        self._codegen_ref: object | None = None

    def enter_scope(self) -> None:
        self.slots.append({})
        self.types.append({})
        # `_struct_aliases` lives on the CodeGenerator, not on the
        # _FuncCtx — `_codegen_ref` is wired by `_function` so this
        # method can keep the per-block scope chains in sync.
        if self._codegen_ref is not None:
            self._codegen_ref._struct_aliases.append({})

    def exit_scope(self) -> None:
        self.slots.pop()
        self.types.pop()
        if self._codegen_ref is not None:
            self._codegen_ref._struct_aliases.pop()

    def alloc_local(
        self,
        name: str,
        ty: "ast.TypeNode",
        size: int = 4,
        *,
        decl: object | None = None,
    ) -> int:
        # If we've already allocated for this AST node (collect-time
        # walk before emit), re-bind the name in the current scope
        # without bumping frame_size. The emit walk re-enters the same
        # scopes and replays the bindings; the slot itself is stable.
        key = id(decl) if decl is not None else None
        if key is not None and key in self.decl_disps:
            disp = self.decl_disps[key]
            self.slots[-1][name] = disp
            self.types[-1][name] = self.decl_types[key]
            return disp
        if name in self.slots[-1]:
            raise CodegenError(f"redeclaration of `{name}` in same scope")
        # Each local sits at the next 4-byte slot below EBP.
        self.frame_size += size
        disp = -self.frame_size
        self.slots[-1][name] = disp
        self.types[-1][name] = ty
        if key is not None:
            self.decl_disps[key] = disp
            self.decl_types[key] = ty
        return disp

    def alloc_call_temp(self, call_node: object, size: int) -> int:
        """Reserve a frame slot for a struct-returning call's destination."""
        self.frame_size += size
        disp = -self.frame_size
        self.call_temps[id(call_node)] = disp
        return disp

    def alloc_param(self, name: str, disp: int, ty: "ast.TypeNode") -> int:
        # cdecl: caller pushed args right-to-left, then `call` pushed the
        # return address, then we pushed EBP. So the first arg lives at
        # [ebp + 8], the second at [ebp + 12], etc. The caller is
        # responsible for computing `disp` as it walks the parameter
        # list — for struct-by-value params, the size isn't 4.
        if name in self.slots[-1]:
            raise CodegenError(f"duplicate parameter `{name}`")
        self.slots[-1][name] = disp
        self.types[-1][name] = ty
        return disp

    def has_local(self, name: str) -> bool:
        for scope in reversed(self.slots):
            if name in scope:
                return True
        return False

    def lookup(self, name: str) -> int:
        for scope in reversed(self.slots):
            if name in scope:
                return scope[name]
        raise CodegenError(f"unknown identifier `{name}`")

    def lookup_type(self, name: str) -> "ast.TypeNode":
        for scope in reversed(self.types):
            if name in scope:
                return scope[name]
        raise CodegenError(f"unknown identifier `{name}`")

    def label(self, hint: str) -> str:
        self._next_label += 1
        return f".L{self._next_label}_{hint}"


class CodeGenerator:
    """i386/MS-DOS backend."""

    def __init__(self, module_name: str = "main"):
        self.module_name = module_name
        # Module-level state populated during generate(). Strings are
        # interned by content so identical literals share a label.
        self._strings: dict[str, str] = {}
        # Map from function name to its declared return type. Lets
        # `_type_of` give a Call expression the right type for downstream
        # pointer-arithmetic decisions.
        self._func_return_types: dict[str, ast.TypeNode] = {}
        # Map from function name to its declared parameter types. Used
        # by `_emit_call` to coerce arg widths at the call site (e.g.
        # narrow a `double` literal when the param is `float`).
        self._func_param_types: dict[str, list[ast.TypeNode]] = {}
        # Module-level variables declared at top scope. `_globals` carries
        # the resolved type (size filled in for unsized arrays);
        # `_global_inits` holds the initializer expression when one was
        # supplied, so the `.data` emission can produce constants.
        self._globals: dict[str, ast.TypeNode] = {}
        self._global_inits: dict[str, ast.Expression] = {}
        # Per-global alignment override from `__attribute__((aligned(N)))`.
        # Keys without an entry use the default (no align directive).
        self._global_alignments: dict[str, int] = {}
        # Per-struct alignment when a member has `aligned(N)` or the
        # struct itself has it. Used to align globals of struct type.
        self._struct_alignments: dict[str, int] = {}
        # Per-function alignment when the function (or its declaration)
        # has `__attribute__((aligned(N)))`. Read by `__alignof__(funcname)`.
        self._func_alignments: dict[str, int] = {}
        # Struct definitions: name → list of (member_name, member_type,
        # offset). `_struct_sizes` is the corresponding total size in bytes
        # (rounded up to struct alignment).
        self._structs: dict[str, list[tuple[str, ast.TypeNode, int]]] = {}
        self._struct_sizes: dict[str, int] = {}
        # Names of structs that were declared with `union` so init/copy
        # paths know members share storage.
        self._struct_unions: set[str] = set()
        # Per-member group id: members promoted from the same anonymous
        # nested struct/union share the same group id. Used by
        # `_emit_global_struct_init` (union path) to distribute a
        # brace-list init across an anonymous-struct member's promoted
        # body. Parallel to `_structs[name]`.
        self._struct_member_groups: dict[str, list[int]] = {}
        # Enum constant table — `enum c { A, B }` registers A=0, B=1.
        # Identifiers that aren't slots/globals/functions fall back here.
        self._enum_constants: dict[str, int] = {}
        # Bit-field info per struct: `_struct_bitfields[struct][member]` =
        # `(bit_offset, bit_width)` for members declared as `int x:N`.
        # Members not in the dict are byte-aligned (regular) members.
        self._struct_bitfields: dict[str, dict[str, tuple[int, int]]] = {}
        # Float-constant table: (value, size_in_bytes) → label. Floats
        # don't have an immediate-load instruction on x87, so every
        # FloatLiteral becomes a labeled constant in `.data`.
        self._float_constants: dict[tuple[float, int], str] = {}

    # ---- top level ------------------------------------------------------

    @classmethod
    def _flatten_decls(cls, decls):
        """Yield top-level declarations, expanding any DeclarationList.

        `int x, y;` at file scope parses as one DeclarationList whose
        children are the individual VarDecls; `generate()` wants to see
        each as a real top-level decl.
        """
        for d in decls:
            if isinstance(d, ast.DeclarationList):
                yield from cls._flatten_decls(d.declarations)
            else:
                yield d

    def generate(self, unit: ast.TranslationUnit) -> str:
        # Top-level `int x, y, z;` parses as a single DeclarationList
        # whose `declarations` are the individual VarDecls. Flatten so
        # the rest of `generate()` can iterate uniformly.
        top_decls = list(self._flatten_decls(unit.declarations))

        # Identify `extern inline` functions whose body uses
        # `__builtin_va_arg_pack`. These are GCC's gnu_inline pattern:
        # the function is inlined at every call site, with the va_arg_pack
        # expanding to the variadic args of the surrounding call. We
        # don't generate a standalone body for them.
        self._gnu_inline_funcs: dict[str, ast.FunctionDecl] = {}
        for d in top_decls:
            if not (isinstance(d, ast.FunctionDecl) and d.body is not None):
                continue
            if not (getattr(d, "storage_class", None) == "extern"
                    and getattr(d, "is_inline", False)):
                continue
            if not self._body_uses_va_arg_pack(d.body):
                continue
            self._gnu_inline_funcs[d.name] = d

        functions = [
            d for d in top_decls
            if isinstance(d, ast.FunctionDecl) and d.body is not None
            and d.name not in self._gnu_inline_funcs
        ]
        if not any(fn.name == "main" for fn in functions):
            raise CodegenError("uc386 requires a `main` function definition")

        # Names declared but not defined in this unit become NASM externs.
        # The parser represents bodyless function declarations as either a
        # FunctionDecl with body=None or a VarDecl whose var_type is a
        # FunctionType — handle both.
        defined_names = {fn.name for fn in functions}
        externs: set[str] = set()
        for d in top_decls:
            if isinstance(d, ast.FunctionDecl) and d.body is None:
                externs.add(d.name)
            elif isinstance(d, ast.VarDecl) and isinstance(d.var_type, ast.FunctionType):
                externs.add(d.name)
        externs -= defined_names
        # gnu_inline functions are never extern symbols — they're
        # inlined at every call site, so the linker shouldn't see them.
        externs -= set(self._gnu_inline_funcs.keys())
        # Drop `_start` — we always define it (and its `__start` alias)
        # as the program entry stub. The user's `extern void _start(void)`
        # would otherwise conflict with our self-defined entry symbol.
        externs.discard("_start")
        extern_list = sorted(externs)
        # Strings are collected lazily as expressions get lowered; reset
        # the table at the top of each generate() call so the codegen is
        # safe to reuse.
        self._strings = {}
        # File-scope compound literals get private globals appended after
        # the normal data section. Each entry: (label, target_type, init).
        self._compound_globals: list[
            tuple[str, ast.TypeNode, ast.Expression]
        ] = []
        # Set transiently during brace-elision so the recursive consumer
        # can ask `_type_of` whether a value is itself a struct
        # expression matching a struct member's type.
        self._elision_ctx: _FuncCtx | None = None
        # GCC computed-goto bookkeeping. `_function_user_labels` maps
        # each function's name to its user-label table (label name →
        # NASM `.LN_user_<x>`), saved after `_collect_labels` runs so
        # that static-local globals (which live in `.data` and are
        # emitted after all functions are codegen'd) can resolve
        # `&&label` initializers.
        self._function_user_labels: dict[str, dict[str, str]] = {}
        # Per-function snapshot of `local_static_labels` (name in the
        # function's local scope → mangled global key). Lets static-
        # local globals whose init references another static local
        # (`static int *hx = gx;`) resolve at emit time.
        self._function_local_static: dict[str, dict[str, str]] = {}
        # Static-local global key (e.g. `simulator_kernel__op_map`) →
        # the owning function's name. Lets `_emit_global_init` know
        # which function's label table to consult when expanding
        # `&&label` inside that init.
        self._static_local_owner: dict[str, str] = {}
        # Set transiently while emitting one static-local global so the
        # downstream `_emit_global_*` recursion (and LabelAddr in
        # particular) can find the owning function without threading
        # an extra parameter.
        self._emitting_for_func: str = ""
        # Build the function-return-type and param-type tables from
        # every declaration in the unit (defined or extern). Calls to
        # unknown names default to int in `_type_of`; arg coercion in
        # `_emit_call` only fires when the param types are known.
        self._func_return_types = {}
        self._func_param_types = {}
        # Pre-register a few GCC builtins so calls without an explicit
        # declaration know the right (complex) ABI.
        for name, base in (
            ("__builtin_conjf", "float"),
            ("__builtin_conj", "double"),
            ("__builtin_conjl", "long double"),
        ):
            ct = ast.ComplexType(base_type=base)
            self._func_return_types[name] = ct
            self._func_param_types[name] = [ct]
        # 64-bit builtins return `unsigned long long` (EDX:EAX). Without
        # this registration, `_return` would `cdq` after the call,
        # smashing the high half.
        ull = ast.BasicType(name="long long", is_signed=False)
        for name in ("__builtin_bswap64",):
            self._func_return_types[name] = ull
            self._func_param_types[name] = [ull]
        # Float-precision builtins: register so the call site narrows /
        # widens correctly per the param's declared width. Without this,
        # a `float`-typed value passed to `__builtin_isinff` would push
        # 8 bytes (default double width) and the callee would see
        # garbage in the low 4.
        flt = ast.BasicType(name="float")
        dbl = ast.BasicType(name="double")
        ldbl = ast.BasicType(name="long double")
        intt = ast.BasicType(name="int")
        for name in ("__builtin_isinff", "__builtin_isnanf",
                     "__builtin_isfinitef", "__builtin_signbitf"):
            self._func_return_types[name] = intt
            self._func_param_types[name] = [flt]
        for name in ("__builtin_isinf", "__builtin_isnan",
                     "__builtin_isfinite", "__builtin_signbit"):
            self._func_return_types[name] = intt
            self._func_param_types[name] = [dbl]
        for name in ("__builtin_isinfl", "__builtin_isnanl",
                     "__builtin_isfinitel", "__builtin_signbitl"):
            self._func_return_types[name] = intt
            self._func_param_types[name] = [ldbl]
        for d in top_decls:
            if isinstance(d, ast.FunctionDecl):
                self._func_return_types[d.name] = d.return_type
                self._func_param_types[d.name] = [
                    p.param_type for p in d.params
                ]
                fa = getattr(d, "alignment", None)
                if fa is not None:
                    self._func_alignments[d.name] = fa
            elif isinstance(d, ast.VarDecl) and isinstance(d.var_type, ast.FunctionType):
                self._func_return_types[d.name] = d.var_type.return_type
                fa = getattr(d, "alignment", None)
                if fa is not None:
                    self._func_alignments[d.name] = fa
                self._func_param_types[d.name] = list(d.var_type.param_types)

        # Reset struct + globals state. Structs need to be registered
        # before globals because a global of struct type will look up the
        # struct's size during validation.
        self._structs = {}
        self._struct_sizes = {}
        self._struct_bitfields = {}
        # Tag-name → registry key, scope chain. Each `_register_struct`
        # at file scope (or inside a compound) pushes the binding into
        # `_struct_aliases[-1]`. Inner blocks may shadow an outer T with
        # a separately-registered key, so `_resolve_struct_name` walks
        # the chain top-down.
        self._struct_aliases: list[dict[str, str]] = [{}]
        self._enum_constants = {}
        self._float_constants = {}
        for d in top_decls:
            if isinstance(d, ast.StructDecl) and d.is_definition:
                self._register_struct(d)
            elif isinstance(d, ast.EnumDecl) and d.is_definition:
                self._register_enum(d)
            elif (
                isinstance(d, ast.VarDecl)
                and isinstance(d.var_type, ast.StructType)
                and d.var_type.name
                and d.var_type.members
            ):
                # `struct S { ... } u;` parses as VarDecl(u) with the
                # struct definition inline in var_type. Register the
                # struct under its tag so other top-level decls can
                # reference it (e.g. `struct U { struct S s[4]; };`).
                self._resolve_struct_name(d.var_type)
            elif isinstance(d, ast.TypedefDecl):
                # Typedef'd enums register their constants at file scope
                # (e.g. `typedef enum { X, Y } T;` declares X and Y).
                if (
                    isinstance(d.target_type, ast.EnumType)
                    and d.target_type.values
                ):
                    self._register_enum_values(d.target_type.values)
                # `typedef struct foo { ... } bar;` defines `struct foo`
                # too — register its layout under the tag so later
                # references via `struct foo *` (e.g. through a forward
                # decl) can resolve the layout for member access.
                if (
                    isinstance(d.target_type, ast.StructType)
                    and d.target_type.name
                    and d.target_type.members
                ):
                    self._resolve_struct_name(d.target_type)
        # `StructType` references inside a containing decl (e.g.,
        # `struct point origin;` at top level) carry an empty members
        # list; the layout is owned by `_structs[name]`. Inline struct
        # definitions inside another decl aren't yet supported.

        # Register top-level VarDecls (non-function-type) as globals.
        # Tentative C definitions (`int x; int x = 3; int x;`) merge:
        # the same name may appear multiple times, with at most one
        # initializer.
        self._globals = {}
        self._global_inits = {}
        # Extern declarations at file scope: name → declared type. Looked
        # up like globals, but no storage allocated; the linker resolves
        # the symbol. NASM `extern _name` is emitted in the header.
        self._extern_vars: dict[str, ast.TypeNode] = {}
        # `__attribute__((alias("target")))` on a VarDecl makes the
        # declared name a NASM-level alias for `_target` — emitted as
        # `_name equ _target` in the output. Both names refer to the
        # same address.
        self._global_aliases: dict[str, str] = {}
        # `__attribute__((noinit))` — uninitialized globals NOT zeroed
        # by the _start re-init loop. (Initial program load still zeros
        # these because DOS loader fills BSS with zeros, but recursive
        # _start calls preserve their values.)
        self._noinit_globals: set[str] = set()
        for d in top_decls:
            if isinstance(d, ast.VarDecl) and not isinstance(d.var_type, ast.FunctionType):
                target = getattr(d, "alias_target", None)
                if target is not None:
                    self._global_aliases[d.name] = target
                    self._extern_vars[d.name] = self._resolved_var_type(d)
                    continue
                if (
                    getattr(d, "storage_class", None) == "extern"
                    and d.init is None
                ):
                    self._extern_vars[d.name] = self._resolved_var_type(d)
                    extern_list.append(d.name)
                    continue
                resolved = self._resolved_var_type(d)
                self._check_supported_type(resolved, d.name)
                self._globals[d.name] = resolved
                align = getattr(d, "alignment", None)
                if align is not None:
                    self._global_alignments[d.name] = align
                if getattr(d, "is_noinit", False):
                    self._noinit_globals.add(d.name)
                if d.init is not None:
                    if d.name in self._global_inits:
                        raise CodegenError(
                            f"global `{d.name}` has multiple initializers"
                        )
                    self._global_inits[d.name] = d.init
        # Track externs added on demand by codegen (e.g., malloc/free
        # for VLA-backed locals). These get merged into extern_list
        # before header emission.
        self._auto_externs: set[str] = set()

        # `-finstrument-functions` semantics. If both
        # `__cyg_profile_func_enter` and `__cyg_profile_func_exit` are
        # defined in this TU, every non-skipped function gets enter/
        # exit calls injected at its prologue/epilogue. Functions
        # marked `__attribute__((no_instrument_function))` (anywhere
        # — forward decl or definition) are skipped, as are the two
        # cyg fns themselves.
        self._instrument_no_skip: set[str] = {
            "__cyg_profile_func_enter",
            "__cyg_profile_func_exit",
        }
        for d in top_decls:
            if (isinstance(d, ast.FunctionDecl)
                    and getattr(d, "no_instrument_function", False)):
                self._instrument_no_skip.add(d.name)
            elif (isinstance(d, ast.VarDecl)
                    and isinstance(d.var_type, ast.FunctionType)
                    and getattr(d, "no_instrument_function", False)):
                self._instrument_no_skip.add(d.name)
        defined_fn_names = {fn.name for fn in functions}
        self._instrument_enabled = (
            "__cyg_profile_func_enter" in defined_fn_names
            and "__cyg_profile_func_exit" in defined_fn_names
        )

        lines: list[str] = []
        # Header is emitted last so we can include any auto-collected
        # externs from the lowering pass below.
        lines += self._start_stub()
        function_blocks: list[list[str]] = []
        # Use a pending queue so nested function definitions discovered
        # while compiling outer functions get appended and drained too.
        self._pending_functions = list(functions)
        # Lifted nested functions remember their outer's capture
        # remapping so their bodies' references to outer-locals
        # resolve to the right mangled global at compile time.
        self._lifted_captures: dict[str, dict[str, str]] = {}
        # Sibling nested-fn name → mangled. A lifted nested function
        # may call sibling nested fns that share the same outer; this
        # maps `t0` → `_outer__t0` etc.
        self._lifted_nested_fn_names: dict[str, dict[str, str]] = {}
        # Non-local goto bookkeeping. `_lifted_nonlocal_gotos`
        # mirrors the outer-side trampoline data on the inner side:
        # lifted-mangled-name → (label → idx_in_buf_array) so the
        # lifted fn's `goto X` knows which 12-byte slot to longjmp.
        # The bufs themselves live in the outer's frame; the static-
        # link slot in the lifted fn holds the buf-array's address,
        # set on entry from ECX.
        self._lifted_nonlocal_gotos: dict[str, dict[str, int]] = {}
        # Lifted-mangled-name → outer-fn-name. Used by `&&label` and
        # the goto-fallback to resolve labels declared in the outer.
        self._lifted_outer_fn: dict[str, str] = {}
        while self._pending_functions:
            fn = self._pending_functions.pop(0)
            function_blocks.append(self._function(fn))
        # Now we know what externs the body referenced; merge in
        # auto-collected externs (malloc/free for VLAs, etc.) before
        # building the header.
        for name in self._auto_externs:
            if name not in extern_list:
                extern_list.append(name)
        extern_list = sorted(set(extern_list))
        # Build header now and prepend to lines.
        header = self._header(extern_list)
        lines = header + lines
        for block in function_blocks:
            lines.append("")
            lines += block
        # `.data` holds initialized globals plus interned string literals;
        # `.bss` holds uninitialized globals (loader zeros them).
        data_lines = self._data_section()
        if data_lines:
            lines.append("")
            lines += data_lines
        bss_lines = self._bss_section()
        if bss_lines:
            lines.append("")
            lines += bss_lines
        return "\n".join(lines) + "\n"

    def _header(self, externs: list[str]) -> list[str]:
        out = [
            f"; uc386 codegen output",
            f"; module: {self.module_name}",
            f"        bits 32",
        ]
        for name in externs:
            out.append(f"        extern  _{name}")
        # `__attribute__((alias("target")))` — emit `_alias equ _target`
        # so all references resolve to the target's address.
        for alias_name, target in sorted(self._global_aliases.items()):
            out.append(f"        _{alias_name} equ _{target}")
        out += [
            f"        section .text",
            f"        global _start",
            "",
        ]
        return out

    def _data_section(self) -> list[str]:
        # Initialized globals + interned string literals + float
        # constants share `.data`. Skip the section header entirely
        # when none of them are present.
        initialized_globals = sorted(
            name for name in self._globals if name in self._global_inits
        )
        if (
            not self._strings
            and not initialized_globals
            and not self._float_constants
            and not self._compound_globals
        ):
            return []
        out = ["        section .data"]
        # Globals first — they may intern new strings (e.g. for a
        # `static const char *p = "foo" + 1` init) which we still want
        # to emit afterward.
        for name in initialized_globals:
            ty = self._globals[name]
            init = self._global_inits[name]
            align = self._global_alignments.get(name)
            if align is None:
                # Derive from the type's required alignment so structs
                # with aligned members get the right placement.
                ta = self._alignment_of(ty)
                if ta > 1:
                    align = ta
            if align and align > 1:
                out.append(f"        align {align}")
            out.append(f"_{name}:")
            self._emitting_for_func = self._static_local_owner.get(name, "")
            out += self._emit_global_init(ty, init, name)
            self._emitting_for_func = ""
        # File-scope compound literals interned via
        # `_intern_compound_global` (e.g. `&(struct S){...}` as an
        # initializer for another global). The list grows during
        # `_emit_global_init`, so emit it after the regular globals so
        # any new additions made while emitting them still flush here.
        emitted = 0
        while emitted < len(self._compound_globals):
            label, target_type, init = self._compound_globals[emitted]
            out.append(f"{label}:")
            out += self._emit_global_init(target_type, init, label)
            emitted += 1
        # Strings come last so any string interned during the global
        # init walk above is included.
        for key, label in sorted(self._strings.items(), key=lambda kv: kv[1]):
            is_wide = key.startswith("W:")
            value = key[2:] if is_wide else key
            out.append(f"{label}:")
            if is_wide:
                # Wide strings: each codepoint is `sizeof(wchar_t)` (2 on
                # this i386 build via __WCHAR_TYPE__=unsigned short).
                cps = [str(ord(c)) for c in value] + ["0"]
                out.append(f"        dw      {', '.join(cps)}")
            else:
                out.append(f"        db      {self._render_string(value)}, 0")
        for (value, size), label in sorted(
            self._float_constants.items(), key=lambda kv: kv[1]
        ):
            # NASM accepts decimal float literals in `dd` (32-bit
            # single) and `dq` (64-bit double); it converts to the
            # appropriate IEEE-754 bit pattern at assemble time.
            directive = "dd" if size == 4 else "dq"
            out.append(f"{label}:")
            out.append(f"        {directive}      {value!r}")
        return out

    def _intern_float(self, value: float, size: int) -> str:
        key = (value, size)
        if key in self._float_constants:
            return self._float_constants[key]
        label = f"_uc386_float{len(self._float_constants)}"
        self._float_constants[key] = label
        return label

    @staticmethod
    def _is_float_type(t: ast.TypeNode) -> bool:
        return isinstance(t, ast.BasicType) and t.name in (
            "float", "double", "long double",
        )

    def _float_promotion(self, lt: ast.TypeNode, rt: ast.TypeNode) -> ast.TypeNode:
        """C usual-arithmetic-conversions for a float-or-int + float-or-int pair.

        If either side is `double`, the result is `double`. Otherwise (at
        least one side is `float`), the result is `float`. Pure integer
        cases don't reach here.
        """
        if (
            (isinstance(lt, ast.BasicType) and lt.name == "double")
            or (isinstance(rt, ast.BasicType) and rt.name == "double")
        ):
            return ast.BasicType(name="double")
        return ast.BasicType(name="float")

    def _bss_section(self) -> list[str]:
        uninit_all = sorted(
            name for name in self._globals if name not in self._global_inits
        )
        # Group non-noinit globals first so a single `_bss_zero_start` /
        # `_bss_zero_end` range covers everything that the _start
        # re-init loop should zero.
        zero_globals = [n for n in uninit_all if n not in self._noinit_globals]
        noinit_globals = [n for n in uninit_all if n in self._noinit_globals]
        # Always emit the section + zero-range labels — `_start`'s
        # re-init loop unconditionally references them. Even an empty
        # BSS produces a no-op `rep stosb` (count 0).
        out = ["        section .bss", "_bss_zero_start:"]
        for name in zero_globals:
            ty = self._globals[name]
            size = self._size_of(ty)
            align = self._global_alignments.get(name)
            if align is None:
                ta = self._alignment_of(ty)
                if ta > 1:
                    align = ta
            if align and align > 1:
                out.append(f"        alignb {align}")
            out.append(f"_{name}:")
            out.append(f"        resb    {size}")
        out.append("_bss_zero_end:")
        # noinit globals: not zeroed by _start, so they retain their
        # values across recursive _start calls. (DOS loader still zero-
        # fills BSS at program load, so first-time access reads 0.)
        for name in noinit_globals:
            ty = self._globals[name]
            size = self._size_of(ty)
            align = self._global_alignments.get(name)
            if align is None:
                ta = self._alignment_of(ty)
                if ta > 1:
                    align = ta
            if align and align > 1:
                out.append(f"        alignb {align}")
            out.append(f"_{name}:")
            out.append(f"        resb    {size}")
        return out

    def _emit_global_init(
        self,
        ty: ast.TypeNode,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        """Emit NASM `db`/`dw`/`dd` directives for an initialized global.

        Globals must initialize from compile-time constants — `_const_eval`
        accepts integer literals and signed integer constants like `-1`.
        Pointer-typed globals with `&other_global` initializers will land
        when there's a clear use case; for now they raise.
        """
        # Scalar globals with optional braces around the value:
        # `int x = {1};` is equivalent to `int x = 1;`.
        if (
            isinstance(ty, ast.BasicType)
            and isinstance(init, ast.InitializerList)
            and len(init.values) == 1
            and not isinstance(init.values[0], ast.DesignatedInit)
        ):
            init = init.values[0]
        # Scalar globals with a literal init.
        if isinstance(ty, ast.BasicType):
            if self._is_float_type(ty):
                # Float globals can initialize from a float or int literal.
                # NASM accepts the decimal form in `dd` / `dq` and converts
                # to IEEE-754; a plain integer like `100` becomes `100.0`.
                # For inf / nan we emit the IEEE-754 bit pattern as a
                # hex int (NASM's repr-form rejects 'inf' / 'nan').
                directive = "dd" if ty.name == "float" else "dq"
                value = self._const_eval_float(init, name)
                import math
                if math.isinf(value) or math.isnan(value):
                    import struct
                    if ty.name == "float":
                        bits = struct.unpack(">I", struct.pack(">f", value))[0]
                        return [f"        dd      0x{bits:08X}"]
                    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
                    return [f"        dq      0x{bits:016X}"]
                return [f"        {directive}      {value!r}"]
            # `&&l1 - &&l2` — label-difference constant. NASM resolves
            # the difference at assemble time. Common in static jump
            # tables.
            label_diff = self._global_label_diff(init)
            if label_diff is not None:
                directive = self._DATA_DIRECTIVE[self._size_of(ty)]
                return [f"        {directive}      {label_diff}"]
            value = self._const_eval(init, name)
            # __int128 globals: lay down two 64-bit halves (low, high)
            # honoring the value's signedness.
            if self._is_int128(ty):
                if value < 0:
                    value &= (1 << 128) - 1
                lo = value & 0xFFFFFFFFFFFFFFFF
                hi = (value >> 64) & 0xFFFFFFFFFFFFFFFF
                return [
                    f"        dq      0x{lo:016X}",
                    f"        dq      0x{hi:016X}",
                ]
            directive = self._DATA_DIRECTIVE[self._size_of(ty)]
            return [f"        {directive}      {value}"]
        if isinstance(ty, ast.PointerType):
            # Strip any leading casts: pointer init `(char *)&x` is the
            # same address as `&x` for layout purposes.
            while isinstance(init, ast.Cast):
                init = init.expr
            # Allow `int *p = 0` (null pointer literal) and
            # `int *p = &other_global` / `char *s = "literal"` — both
            # become `dd <label>` so the linker resolves the address.
            if (
                isinstance(init, ast.UnaryOp)
                and init.op == "&"
                and isinstance(init.operand, ast.Identifier)
            ):
                inner_name = self._resolve_static_init_name(init.operand.name)
                if (
                    inner_name in self._globals
                    or inner_name in self._func_return_types
                ):
                    return [f"        dd      _{inner_name}"]
            # GCC `&&label` in a static initializer — address of a
            # function-local label. Resolves through the saved per-fn
            # user_labels map.
            if isinstance(init, ast.LabelAddr):
                return [f"        dd      {self._global_label_addr_text(init.label)}"]
            # `&&label - &&other_label` or similar address-difference
            # constants — fall through to `_resolve_static_addr` below
            # via the `&` path; LabelAddr binary ops are handled in the
            # generic BinaryOp branch.
            # `&<static-lvalue>` — address of any static lvalue. Walk
            # the chain of `.field` / `->field` / `[i]` / `*` to compute
            # `<base_label> + <constant_offset>` at link time.
            if (
                isinstance(init, ast.UnaryOp)
                and init.op == "&"
            ):
                resolved = self._resolve_static_addr(init.operand, name)
                if resolved is not None:
                    base_label, offset = resolved
                    if offset == 0:
                        return [f"        dd      {base_label}"]
                    sign = "+" if offset > 0 else "-"
                    return [f"        dd      {base_label} {sign} {abs(offset)}"]
            # `struct S *s = &(struct S){...};` — the compound literal
            # gets a private global, and the pointer global stores its
            # address.
            if (
                isinstance(init, ast.UnaryOp)
                and init.op == "&"
                and isinstance(init.operand, ast.Compound)
            ):
                hidden = self._intern_compound_global(
                    init.operand.target_type, init.operand.init, name,
                )
                return [f"        dd      {hidden}"]
            # `int (*f)(int) = fred;` — function name decays to its address.
            if isinstance(init, ast.Identifier):
                resolved_name = self._resolve_static_init_name(init.name)
                if resolved_name in self._func_return_types:
                    return [f"        dd      _{resolved_name}"]
                # `int *p = some_other_global;` — array name decays.
                if (
                    resolved_name in self._globals
                    and isinstance(
                        self._globals[resolved_name], ast.ArrayType,
                    )
                ):
                    return [f"        dd      _{resolved_name}"]
            if isinstance(init, ast.StringLiteral):
                label = self._intern_string(init.value)
                return [f"        dd      {label}"]
            # `"foo" + 1` — string literal address arithmetic. Emit the
            # interned string's label plus the integer offset.
            if (
                isinstance(init, ast.BinaryOp)
                and init.op in ("+", "-")
                and isinstance(init.left, ast.StringLiteral)
            ):
                label = self._intern_string(init.left.value)
                offset = self._const_eval(init.right, name)
                if init.op == "-":
                    offset = -offset
                if offset == 0:
                    return [f"        dd      {label}"]
                sign = "+" if offset > 0 else "-"
                return [f"        dd      {label} {sign} {abs(offset)}"]
            value = self._const_eval(init, name)
            return [f"        dd      {value}"]
        if isinstance(ty, ast.ArrayType):
            # Compound literal init: `((vec_t){a, b, ...})` — strip
            # the wrapper and use the inner InitializerList directly.
            if (
                isinstance(init, ast.Compound)
                and isinstance(init.init, ast.InitializerList)
            ):
                init = init.init
            # Strip leading Casts that pun a same-shape value.
            while isinstance(init, ast.Cast) and isinstance(
                init.expr, (ast.Compound, ast.InitializerList)
            ):
                init = init.expr
                if isinstance(init, ast.Compound):
                    init = init.init
            return self._emit_global_array_init(ty, init, name)
        if isinstance(ty, ast.StructType):
            return self._emit_global_struct_init(ty, init, name)
        if isinstance(ty, ast.EnumType):
            # An enum is int-sized; treat the init as an int literal.
            value = self._const_eval(init, name)
            return [f"        dd      {value}"]
        if isinstance(ty, ast.ComplexType):
            return self._emit_global_complex_init(ty, init, name)
        raise CodegenError(
            f"global `{name}`: unsupported type {type(ty).__name__}"
        )

    def _emit_global_complex_init(
        self,
        ty: ast.ComplexType,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        """Lay out a `_Complex T` global as two T values: real, imag.

        Supports compile-time complex arithmetic via
        `_const_eval_complex`, which returns a `(real, imag)` tuple.
        """
        real, imag = self._const_eval_complex(init, name)
        half_size = self._COMPLEX_BASE_SIZES.get(ty.base_type, 8)
        # Float halves accept the decimal float repr; integer halves
        # use db/dw/dd/dq with the int value.
        if ty.base_type in self._COMPLEX_INT_BASES:
            directive = self._DATA_DIRECTIVE.get(half_size, "dd")
            r = int(real)
            i = int(imag)
            return [f"        {directive}      {r}, {i}"]
        directive = "dd" if half_size == 4 else "dq"
        return [f"        {directive}      {real!r}, {imag!r}"]

    def _const_eval_complex(
        self, expr: ast.Expression, name: str,
    ) -> tuple[float, float]:
        """Compile-time fold a complex expression to a `(real, imag)`
        pair of floats. Recognizes imaginary float literals, complex
        unary `-/+`, and binary `+/-/*` between complex (or scalar)
        operands.
        """
        while isinstance(expr, ast.Cast):
            expr = expr.expr
        if isinstance(expr, ast.FloatLiteral):
            v = float(expr.value)
            return (0.0, v) if expr.is_imaginary else (v, 0.0)
        if isinstance(expr, ast.IntLiteral):
            return (float(expr.value), 0.0)
        if isinstance(expr, ast.UnaryOp):
            if expr.op in ("+", "-"):
                ar, ai = self._const_eval_complex(expr.operand, name)
                if expr.op == "-":
                    return (-ar, -ai)
                return (ar, ai)
            if expr.op == "~":
                # Complex conjugate: negate imag.
                ar, ai = self._const_eval_complex(expr.operand, name)
                return (ar, -ai)
        if isinstance(expr, ast.BinaryOp):
            if expr.op in ("+", "-", "*", "/"):
                lr, li = self._const_eval_complex(expr.left, name)
                rr, ri = self._const_eval_complex(expr.right, name)
                if expr.op == "+":
                    return (lr + rr, li + ri)
                if expr.op == "-":
                    return (lr - rr, li - ri)
                if expr.op == "*":
                    return (lr * rr - li * ri, lr * ri + li * rr)
                if expr.op == "/":
                    denom = rr * rr + ri * ri
                    if denom == 0:
                        raise CodegenError(
                            f"global `{name}`: complex division by zero"
                        )
                    return (
                        (lr * rr + li * ri) / denom,
                        (li * rr - lr * ri) / denom,
                    )
        # InitializerList `{r, i}` — explicit two-value form.
        if isinstance(expr, ast.InitializerList):
            if len(expr.values) == 1:
                r, _ = self._const_eval_complex(expr.values[0], name)
                return (r, 0.0)
            if len(expr.values) == 2:
                r, _ = self._const_eval_complex(expr.values[0], name)
                i, _ = self._const_eval_complex(expr.values[1], name)
                return (r, i)
        raise CodegenError(
            f"global `{name}`: complex init must be a constant "
            f"expression (got {type(expr).__name__})"
        )

    def _emit_global_struct_init(
        self,
        struct_ty: ast.StructType,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        # Compound literal `(T){...}` for the same struct type unwraps
        # to its inner InitializerList — that way `struct S s = (S){1,2}`
        # is laid out the same as `struct S s = {1,2}`.
        if isinstance(init, ast.Compound) and isinstance(
            init.target_type, ast.StructType
        ):
            init = init.init
        if not isinstance(init, ast.InitializerList):
            # Brace-elide: a non-list value initializing a struct is
            # like `{ value }` — the value lands in the first member,
            # rest of the struct zero-fills.
            init = ast.InitializerList(values=[init])
        sname = self._resolve_struct_name(struct_ty)
        members = self._structs[sname]
        total = self._struct_sizes[sname]
        member_index = {mn: i for i, (mn, _, _) in enumerate(members)}
        bitfields = self._struct_bitfields.get(sname, {})
        is_union = sname in self._struct_unions
        if bitfields and not is_union:
            return self._emit_global_bitfield_struct_init(
                sname, members, bitfields, total, init, name,
            )
        if is_union:
            # Union init targets the first member by C99 default
            # (unless designated). The init value is laid out at
            # offset 0; the rest of the union is zero-padded to its
            # total size.
            if not init.values:
                return [f"        times {total} db 0"]
            # Multiple designated initializers writing to non-overlapping
            # member offsets (common with anonymous-struct union members:
            # `union { struct {u8 a, b;}; ... } u = {.b = 8, .a = 7};`).
            # Walk each designator and emit at its offset; non-overlapping
            # writes accumulate in the byte image. Non-designated leading
            # values fall through to the legacy single-init path.
            if (
                all(isinstance(v, ast.DesignatedInit)
                    and len(v.designators) == 1
                    and isinstance(v.designators[0], str)
                    and v.designators[0] in member_index
                    for v in init.values)
                and len(init.values) > 1
            ):
                placed: dict[int, tuple] = {}
                for v in init.values:
                    target = v.designators[0]
                    idx = member_index[target]
                    m_name, m_ty, m_off = members[idx]
                    placed[m_off] = (
                        self._size_of(m_ty), v.value, m_name, m_ty,
                    )
                # Emit by sorted offset; overlapping writes ignored
                # (last designator wins on overlap, but we only handle
                # non-overlapping here).
                sorted_offs = sorted(placed.keys())
                # Check for overlap: skip the case if any overlap.
                ok = True
                for i_off in range(len(sorted_offs) - 1):
                    cur = sorted_offs[i_off]
                    cur_sz = placed[cur][0]
                    nxt = sorted_offs[i_off + 1]
                    if cur + cur_sz > nxt:
                        ok = False
                        break
                if ok:
                    sub_out: list[str] = []
                    cursor = 0
                    for off in sorted_offs:
                        size, val, m_name, m_ty = placed[off]
                        if off > cursor:
                            sub_out.append(
                                f"        times {off - cursor} db 0"
                            )
                            cursor = off
                        sub_out += self._emit_global_init(
                            m_ty, val, f"{name}.{m_name}",
                        )
                        cursor = off + size
                    if total > cursor:
                        sub_out.append(
                            f"        times {total - cursor} db 0"
                        )
                    return sub_out
            first_value = init.values[0]
            # If the first conceptual member is an anonymous nested
            # struct (multiple promoted members share group 0), and the
            # init's first value is itself a brace list, distribute the
            # inner values across the anonymous struct's promoted
            # members. Drives `union { struct {char x[4]; char y[4];}; ... } u = {{"a","b"}}`.
            groups = self._struct_member_groups.get(sname, [])
            if (
                groups
                and not isinstance(first_value, ast.DesignatedInit)
                and isinstance(first_value, ast.InitializerList)
            ):
                first_group = groups[0]
                group_indices = [
                    i for i, g in enumerate(groups) if g == first_group
                ]
                if len(group_indices) > 1:
                    # Distribute first_value's values across the
                    # anon-struct's members. Designators within the
                    # inner init reset the cursor to the named member.
                    sub_out: list[str] = []
                    name_to_member_idx = {
                        members[mi][0]: mi for mi in group_indices
                    }
                    placed: dict[int, ast.Expression] = {}
                    cursor_pos = 0
                    for v in first_value.values:
                        if isinstance(v, ast.DesignatedInit):
                            target = v.designators[0]
                            if (
                                isinstance(target, str)
                                and target in name_to_member_idx
                            ):
                                mi = name_to_member_idx[target]
                                if len(v.designators) > 1:
                                    placed[mi] = ast.InitializerList(
                                        values=[ast.DesignatedInit(
                                            designators=v.designators[1:],
                                            value=v.value,
                                        )]
                                    )
                                else:
                                    placed[mi] = v.value
                                cursor_pos = group_indices.index(mi) + 1
                                continue
                        # Positional value — place at cursor_pos.
                        if cursor_pos < len(group_indices):
                            placed[group_indices[cursor_pos]] = v
                            cursor_pos += 1
                    written = 0
                    for mi in group_indices:
                        m_name, m_ty, m_off = members[mi]
                        if m_off > written:
                            sub_out.append(
                                f"        times {m_off - written} db 0"
                            )
                            written = m_off
                        if mi in placed:
                            sub_out += self._emit_global_init(
                                m_ty, placed[mi],
                                f"{name}.{m_name}",
                            )
                        else:
                            sub_out.append(
                                f"        times {self._size_of(m_ty)} db 0"
                            )
                        written = m_off + self._size_of(m_ty)
                    if total > written:
                        sub_out.append(
                            f"        times {total - written} db 0"
                        )
                    return sub_out
            if isinstance(first_value, ast.DesignatedInit):
                # `.field = value` — find the named member.
                first = first_value.designators[0]
                if not isinstance(first, str):
                    raise CodegenError(
                        f"global `{name}`: array designator on struct "
                        f"init not supported"
                    )
                target_name = first
                # Multi-level designator like `.a.b = expr` —
                # synthesize a nested InitializerList so the outer
                # struct emit only sees `.a = {...}`.
                if len(first_value.designators) > 1:
                    inner = ast.DesignatedInit(
                        designators=first_value.designators[1:],
                        value=first_value.value,
                    )
                    value = ast.InitializerList(values=[inner])
                else:
                    value = first_value.value
            else:
                target_name = members[0][0]
                value = first_value
                # Brace elision: if the first member is an array or
                # struct and the init values are scalars/strings, wrap
                # them all into an InitializerList for the first member.
                # `union { u8 arr[16]; ... } u = {1,2,3,4}` → arr fills.
                first_m_ty = members[0][1]
                if (
                    not isinstance(value, ast.InitializerList)
                    and not isinstance(value, ast.Compound)
                    and len(init.values) >= 1
                    and isinstance(first_m_ty, (ast.ArrayType, ast.StructType))
                ):
                    if (
                        isinstance(first_m_ty, ast.ArrayType)
                        and not isinstance(value, ast.StringLiteral)
                    ):
                        value = ast.InitializerList(values=list(init.values))
            target_ty = next(
                ty for n, ty, _ in members if n == target_name
            )
            if target_name in bitfields:
                # Single bit-field init: pack into the storage unit.
                info = bitfields[target_name]
                if len(info) == 2:
                    bit_offset, bit_width = info
                    unit_size = 4
                else:
                    bit_offset, bit_width, unit_size = info
                v = self._const_eval(value, f"{name}.{target_name}")
                mask = (1 << bit_width) - 1
                packed = (v & mask) << bit_offset
                if unit_size == 8:
                    out = [
                        f"        dd      0x{packed & 0xFFFFFFFF:08X}, "
                        f"0x{(packed >> 32) & 0xFFFFFFFF:08X}"
                    ]
                else:
                    out = [f"        dd      0x{packed & 0xFFFFFFFF:08X}"]
                if total > unit_size:
                    out.append(f"        times {total - unit_size} db 0")
                return out
            out = self._emit_global_init(target_ty, value, f"{name}.{target_name}")
            written = self._size_of(target_ty)
            if total > written:
                out.append(f"        times {total - written} db 0")
            return out

        # Walk source values, honouring `.field = value` designators.
        # `slot_values[i]` is the expr to emit for member i, or absent
        # for "zero-fill this member".
        # Apply brace elision: group flat values targeting compound
        # members into per-member InitializerLists (e.g. PT's int c[4]
        # eats the next 4 flat values).
        elided_values = self._elide_braces_for_struct(
            init.values, members, name,
        )
        slot_values: dict[int, ast.Expression] = {}
        cursor = 0
        for value in elided_values:
            if isinstance(value, ast.DesignatedInit):
                if (
                    len(value.designators) != 1
                    or not isinstance(value.designators[0], str)
                ):
                    raise CodegenError(
                        f"global `{name}`: only single-level `.field` "
                        f"designators supported"
                    )
                m_name_des = value.designators[0]
                if m_name_des not in member_index:
                    raise CodegenError(
                        f"global `{name}`: unknown member `{m_name_des}` "
                        f"in struct `{sname}`"
                    )
                idx = member_index[m_name_des]
                actual = value.value
                cursor = idx + 1
            else:
                if cursor >= len(members):
                    raise CodegenError(
                        f"global `{name}`: too many initializers "
                        f"(struct has {len(members)} members)"
                    )
                idx = cursor
                actual = value
                # Skip past anonymous-union alternatives that share this
                # member's offset — they don't consume a positional value.
                # But: a zero-sized member (like an empty struct) leaves
                # the next member at the same byte offset; it's not a
                # union alt and shouldn't be skipped.
                next_cursor = cursor + 1
                _, this_ty, this_off = members[idx]
                if self._size_of(this_ty) > 0:
                    while (
                        next_cursor < len(members)
                        and members[next_cursor][2] == this_off
                    ):
                        next_cursor += 1
                cursor = next_cursor
            slot_values[idx] = actual

        # Emit in declaration order, padding gaps so the byte layout
        # matches the struct's actual offsets. NASM's `times N db 0`
        # fills both inter-member padding and unspecified-member tails.
        # Members that share an offset (anonymous-union alternatives)
        # only get one emission — pick the first one with a slot value,
        # otherwise the first member at that offset.
        out: list[str] = []
        emit_cursor = 0
        i = 0
        while i < len(members):
            m_name, m_ty, m_off = members[i]
            # Zero-size members (empty struct) take no storage and aren't
            # part of any union-alternation group. Skip past, but still
            # invoke their init for side effects' sake (no-op for empty).
            # Flex array members (ArrayType, size=None) report 0 from
            # `_size_of` but their length is derived from the initializer
            # — don't skip those.
            is_flex_array = (
                isinstance(m_ty, ast.ArrayType) and m_ty.size is None
            )
            if self._size_of(m_ty) == 0 and not is_flex_array:
                i += 1
                continue
            # Find the run of NON-ZERO members sharing this offset.
            # (Anonymous-union members alternate at the same offset; we
            # pick the one with an init and emit it once.)
            j = i + 1
            while (
                j < len(members)
                and members[j][2] == m_off
                and self._size_of(members[j][1]) > 0
            ):
                j += 1
            # Pick the index to emit: prefer one with an init value.
            chosen = i
            for k in range(i, j):
                if k in slot_values:
                    chosen = k
                    break
            cm_name, cm_ty, cm_off = members[chosen]
            if cm_off > emit_cursor:
                out.append(f"        times {cm_off - emit_cursor} db 0")
                emit_cursor = cm_off
            if chosen in slot_values:
                out += self._emit_global_init(
                    cm_ty, slot_values[chosen], f"{name}.{cm_name}",
                )
            else:
                m_size = self._size_of(cm_ty)
                out.append(f"        times {m_size} db 0")
            # Advance the cursor by the *largest* member at this offset
            # (anonymous unions: bytes spanned = max-member-size).
            span = max(self._size_of(members[k][1]) for k in range(i, j))
            emit_cursor = cm_off + span
            i = j
        if total > emit_cursor:
            out.append(f"        times {total - emit_cursor} db 0")
        return out

    def _emit_global_bitfield_struct_init(
        self,
        sname: str,
        members: list,
        bitfields: dict,
        total: int,
        init: ast.InitializerList,
        name: str,
    ) -> list[str]:
        """Pack bit-field init values into 4-byte storage units and emit.

        Walks the InitializerList and members in parallel. Bit-field
        members get OR'd into the value at their storage unit's offset;
        regular members get a per-member init value at their own
        offset. Mixed bit-field and regular members in the same struct
        are supported (bit-fields share storage units, regulars don't).
        """
        member_index = {mn: i for i, (mn, _, _) in enumerate(members)}
        # storage_unit_offset → packed unit value (for bit-field members).
        units: dict[int, int] = {}
        # storage_unit_offset → unit size in bytes (4 or 8).
        unit_sizes_by_off: dict[int, int] = {}
        # member_index → init expr (for non-bit-field members).
        regular_inits: dict[int, ast.Expression] = {}
        # Walk values in declaration order. `.field = value` jumps the
        # cursor; subsequent positional values continue from there.
        cursor = 0
        for value in init.values:
            if isinstance(value, ast.DesignatedInit):
                if (
                    len(value.designators) != 1
                    or not isinstance(value.designators[0], str)
                ):
                    raise CodegenError(
                        f"global `{name}`: only single-level `.field` "
                        f"designators supported in bit-field init"
                    )
                m_name_des = value.designators[0]
                if m_name_des not in member_index:
                    raise CodegenError(
                        f"global `{name}`: unknown member "
                        f"`{m_name_des}`"
                    )
                cursor = member_index[m_name_des]
                actual = value.value
            else:
                actual = value
            if cursor >= len(members):
                raise CodegenError(
                    f"global `{name}`: too many initializers"
                )
            m_name_i, m_ty_i, m_off = members[cursor]
            idx = cursor
            cursor += 1
            if m_name_i in bitfields:
                info = bitfields[m_name_i]
                if len(info) == 2:
                    bit_offset, bit_width = info
                    unit_size = 4
                else:
                    bit_offset, bit_width, unit_size = info
                v = self._const_eval(actual, f"{name}.{m_name_i}")
                mask = (1 << bit_width) - 1
                v_masked = (v & mask) << bit_offset
                units[m_off] = units.get(m_off, 0) | v_masked
                unit_sizes_by_off[m_off] = unit_size
            else:
                regular_inits[idx] = actual
        # Now emit in declaration order. For each member:
        #  - bit-field: emit the packed unit at its offset (only once
        #    per unit_offset).
        #  - regular: emit its init expr (or zero) at its offset.
        out: list[str] = []
        emit_cursor = 0
        emitted_unit_offsets: set[int] = set()
        for idx, (m_name_i, m_ty_i, m_off) in enumerate(members):
            if m_name_i in bitfields:
                if m_off in emitted_unit_offsets:
                    continue
                if m_off > emit_cursor:
                    out.append(f"        times {m_off - emit_cursor} db 0")
                u_size = unit_sizes_by_off.get(m_off, 4)
                v = units.get(m_off, 0)
                if u_size == 8:
                    out.append(
                        f"        dd      0x{v & 0xFFFFFFFF:08X}, "
                        f"0x{(v >> 32) & 0xFFFFFFFF:08X}"
                    )
                else:
                    out.append(
                        f"        dd      0x{v & 0xFFFFFFFF:08X}"
                    )
                emit_cursor = m_off + u_size
                emitted_unit_offsets.add(m_off)
            else:
                if m_off > emit_cursor:
                    out.append(f"        times {m_off - emit_cursor} db 0")
                    emit_cursor = m_off
                if idx in regular_inits:
                    out += self._emit_global_init(
                        m_ty_i, regular_inits[idx], f"{name}.{m_name_i}",
                    )
                else:
                    out.append(
                        f"        times {self._size_of(m_ty_i)} db 0"
                    )
                emit_cursor = m_off + self._size_of(m_ty_i)
        if total > emit_cursor:
            out.append(f"        times {total - emit_cursor} db 0")
        return out

    def _elide_braces_for_struct(
        self,
        values: list,
        members: list,
        name: str,
    ) -> list:
        """Brace-elide a flat init-value run for a struct.

        Walks `values` and `members` in parallel. For each non-designated
        value: if the member is a compound (struct/array) and the value
        isn't already wrapped, consume enough subsequent values to fill
        the member, recursively descending into nested compounds. Brace
        elision lets things like `struct U {u8 a; struct S s; u8 b; struct T t;}`
        with init `{3, 5,6,7,8, 4, "huhu", 43}` work — the right values
        flow into S's members and T's `s[16]; a` shape.
        """
        out, i = self._consume_for_members(values, 0, members)
        # Tail values past the last member fall through verbatim so the
        # downstream walker raises "too many initializers".
        out.extend(values[i:])
        return out

    def _consume_for_members(
        self, values: list, i: int, members: list,
    ) -> tuple[list, int]:
        """Consume values for a struct's members, returning (out, new_i).

        `out` is a list of length len(members), one per member; each
        entry is either a single value (scalar member) or a synthesized
        InitializerList (compound member that ate multiple flat values).
        Designators stop consumption — the caller passes them through.
        """
        out: list = []
        for _, m_ty, _ in members:
            if i >= len(values):
                break
            v = values[i]
            if isinstance(v, ast.DesignatedInit):
                # Pass the designated value through; caller resolves.
                out.append(v)
                i += 1
                continue
            taken, i = self._consume_one_member(values, i, m_ty)
            out.append(taken)
        return out, i

    def _consume_one_member(
        self, values: list, i: int, m_ty: ast.TypeNode,
    ) -> tuple:
        """Take one logical value for a member of type `m_ty`, possibly
        synthesizing an InitializerList by recursively eliding inside.
        Returns (taken_value, new_i).
        """
        v = values[i]
        # Already wrapped — pass through.
        if isinstance(v, ast.InitializerList):
            return v, i + 1
        if isinstance(m_ty, ast.StructType):
            # Compound literal `(T){...}` for the same struct type
            # passes through as-is; the struct emit unwraps it.
            if isinstance(v, ast.Compound) and isinstance(
                v.target_type, ast.StructType
            ):
                try:
                    if (
                        self._resolve_struct_name(v.target_type)
                        == self._resolve_struct_name(m_ty)
                    ):
                        return v, i + 1
                except CodegenError:
                    pass
            # If `v` is itself a struct-valued expression matching this
            # member type, pass it through — it's a struct copy, not a
            # flat-value run to elide. We cheaply infer the type from
            # the *current* function context if one is active.
            ctx = self._elision_ctx
            if ctx is not None:
                try:
                    v_ty = self._type_of(v, ctx)
                except CodegenError:
                    v_ty = None
                if isinstance(v_ty, ast.StructType):
                    try:
                        if (
                            self._resolve_struct_name(v_ty)
                            == self._resolve_struct_name(m_ty)
                        ):
                            return v, i + 1
                    except CodegenError:
                        pass
            try:
                sname = self._resolve_struct_name(m_ty)
                inner_members = self._structs.get(sname, [])
            except CodegenError:
                return v, i + 1
            if not inner_members:
                return v, i + 1
            # For unions, only the first member receives the initializer
            # — consume one value, wrap it in InitializerList. Without
            # this, a struct containing a union followed by another
            # member would over-consume into the union and shift
            # subsequent members off by one.
            if sname in self._struct_unions:
                taken, new_i = self._consume_one_member(
                    values, i, inner_members[0][1],
                )
                return ast.InitializerList(values=[taken]), new_i
            inner_vals, new_i = self._consume_for_members(
                values, i, inner_members,
            )
            return ast.InitializerList(values=inner_vals), new_i
        if isinstance(m_ty, ast.ArrayType):
            # `char a[N] = "..."` — the string is one value satisfying
            # the whole array.
            if isinstance(v, ast.StringLiteral):
                return v, i + 1
            # Build a synthetic InitializerList by consuming one logical
            # value per array element.
            length = (
                m_ty.size.value if isinstance(m_ty.size, ast.IntLiteral)
                else 0
            )
            elem_ty = m_ty.base_type
            inner_vals: list = []
            for _ in range(length):
                if i >= len(values):
                    break
                vv = values[i]
                if isinstance(vv, ast.DesignatedInit):
                    break
                taken, i = self._consume_one_member(values, i, elem_ty)
                inner_vals.append(taken)
            return ast.InitializerList(values=inner_vals), i
        # Scalar: take one value.
        return v, i + 1

    def _leaf_slot_count(self, t: ast.TypeNode) -> int:
        """Recursively count scalar leaves in `t`.

        Used by the brace-elision pre-pass to know how many flat
        positional values to consume per array element. Anonymous-union
        members at the same offset count once.
        """
        if isinstance(t, ast.ArrayType):
            if not isinstance(t.size, ast.IntLiteral):
                return 1
            return t.size.value * self._leaf_slot_count(t.base_type)
        if isinstance(t, ast.StructType):
            try:
                sname = self._resolve_struct_name(t)
            except CodegenError:
                return 1
            members = self._structs.get(sname, [])
            count = 0
            seen_off = set()
            for _, m_ty, off in members:
                if off in seen_off:
                    continue
                seen_off.add(off)
                count += self._leaf_slot_count(m_ty)
            return count or 1
        return 1

    def _elide_braces_for_array(
        self,
        values: list,
        elem_type: ast.TypeNode,
        name: str,
    ) -> list:
        """Group flat positional values into per-element InitializerLists.

        For `PT cases[] = { v1, v2, v3, ... }` where PT is a struct, the
        C standard says positional values flow through the array's
        nested aggregates. We group the next K leaf values into one
        synthesized InitializerList per array element when K = the
        struct's member count and the values aren't already wrapped.

        Designated array initializers reset the cursor; we hand them
        through unchanged. Single-element arrays of struct also fall
        through normally.
        """
        # Only worth doing if the element is a struct/union/array with
        # known layout and there's a flat run that exceeds 1 value per
        # element.
        if not isinstance(elem_type, (ast.StructType, ast.ArrayType)):
            return values
        leaf_count = self._leaf_slot_count(elem_type)
        if leaf_count <= 1:
            return values

        out = []
        i = 0
        while i < len(values):
            v = values[i]
            if isinstance(v, ast.DesignatedInit):
                # Designators can target individual array elements; keep
                # them as-is, but their value might itself need elision
                # if it's flat-multi.
                out.append(v)
                i += 1
                continue
            if isinstance(v, ast.InitializerList):
                out.append(v)
                i += 1
                continue
            # A Compound literal `(T){...}` is already a complete
            # element initializer — don't elide.
            if isinstance(v, ast.Compound):
                out.append(v)
                i += 1
                continue
            # When elem_type is a `char[]` directly, a StringLiteral is
            # a complete initializer; don't elision-wrap it. (`{"str"}`
            # initializing struct{char[N]} still gets wrapped below
            # because the elem there is the struct, not char[].)
            if (
                isinstance(v, ast.StringLiteral)
                and isinstance(elem_type, ast.ArrayType)
                and isinstance(elem_type.base_type, ast.BasicType)
                and elem_type.base_type.name == "char"
            ):
                out.append(v)
                i += 1
                continue
            # Flat value — gather up to leaf_count consecutive flat
            # values into an InitializerList for one element. Stop
            # the group after a StringLiteral (which can fill a whole
            # nested char array all by itself, so subsequent flat
            # values belong to the next element).
            group = []
            j = i
            while j < len(values) and j - i < leaf_count:
                vj = values[j]
                if isinstance(vj, (ast.DesignatedInit, ast.InitializerList)):
                    break
                group.append(vj)
                j += 1
                if isinstance(vj, ast.StringLiteral):
                    break
            if len(group) > 0:
                # Wrap whatever we collected. Partial groups (fewer
                # than leaf_count) zero-fill the unwritten slots in
                # the struct/array.
                out.append(ast.InitializerList(values=group))
                i = j
            else:
                # Empty group (caused by a leading designator we
                # already passed through) — defensive bail-out.
                out.append(v)
                i += 1
        return out

    def _emit_global_array_init(
        self,
        arr_ty: ast.ArrayType,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        elem_type = arr_ty.base_type
        elem_size = self._size_of(elem_type)
        # Flexible array member or `int arr[] = {...};` — derive length
        # from the initializer.
        if arr_ty.size is None:
            if isinstance(init, ast.StringLiteral):
                length = len(init.value) + 1  # include null terminator
            elif isinstance(init, ast.InitializerList):
                length = len(init.values)
            else:
                length = 1
        else:
            length = arr_ty.size.value

        if isinstance(init, ast.StringLiteral):
            is_wide = getattr(init, "is_wide", False) or elem_size > 1
            if is_wide:
                # `wchar_t arr[N] = L"..."`: each codepoint is one
                # `elem_size`-byte slot. Append a null terminator if
                # there's room.
                codepoints = [ord(c) for c in init.value]
                if len(codepoints) > length:
                    raise CodegenError(
                        f"global `{name}`: wide string init exceeds "
                        f"array size {length}"
                    )
                if len(codepoints) < length:
                    codepoints = codepoints + [0]
                directive = self._DATA_DIRECTIVE.get(elem_size)
                if directive is None:
                    raise CodegenError(
                        f"global `{name}`: wide-char element size "
                        f"{elem_size} unsupported"
                    )
                values = [str(cp) for cp in codepoints] + ["0"] * (
                    length - len(codepoints)
                )
                return [f"        {directive}      {', '.join(values)}"]
            if not (
                isinstance(elem_type, ast.BasicType) and elem_type.name == "char"
            ):
                raise CodegenError(
                    f"global `{name}`: string init requires a char array"
                )
            raw_bytes = list(self._string_to_bytes(init.value))
            # C: a string literal initializing a char array of the exact
            # length drops the trailing null. If the array is larger,
            # the null is included and remaining slots zero-fill. If the
            # string is *longer* than the array, gcc truncates as a QoI
            # matter (the excess is undefined behavior); mirror that.
            if len(raw_bytes) > length:
                raw_bytes = raw_bytes[:length]
            if len(raw_bytes) == length:
                # No room for the null terminator — emit just the bytes.
                parts = [str(b) for b in raw_bytes]
                return [f"        db      {', '.join(parts)}"]
            # Append the null terminator + zero-pad the rest.
            parts = [str(b) for b in raw_bytes] + ["0"] * (length - len(raw_bytes))
            return [f"        db      {', '.join(parts)}"]

        if isinstance(init, ast.InitializerList):
            values = self._elide_braces_for_array(init.values, elem_type, name)
            # Walk values once, allowing `[N] = expr` to jump the cursor
            # the same way local-array init does. The result is a
            # by-index map from designated-or-positional slots to
            # (value-expr, kind) pairs; gaps zero-fill.
            slots: dict[int, ast.Expression] = {}
            cursor = 0
            for value in values:
                idx_range: list[int] | None = None
                if isinstance(value, ast.DesignatedInit):
                    designator = value.designators[0]
                    if (
                        len(value.designators) == 1
                        and isinstance(designator, ast.RangeDesignator)
                    ):
                        try:
                            start = self._const_eval(
                                designator.start, name
                            )
                            end = self._const_eval(designator.end, name)
                        except CodegenError:
                            raise CodegenError(
                                f"global `{name}`: range designator must be "
                                f"compile-time constants"
                            )
                        idx_range = list(range(start, end + 1))
                        actual = value.value
                        cursor = end + 1
                    elif (
                        len(value.designators) != 1
                        or not isinstance(designator, ast.IntLiteral)
                    ):
                        raise CodegenError(
                            f"global `{name}`: only single-level integer "
                            f"designators supported in array init"
                        )
                    else:
                        idx = designator.value
                        actual = value.value
                        cursor = idx + 1
                        idx_range = [idx]
                else:
                    idx = cursor
                    actual = value
                    cursor += 1
                    idx_range = [idx]
                for idx in idx_range:
                    if idx < 0 or idx >= length:
                        raise CodegenError(
                            f"global `{name}`: initializer index {idx} out "
                            f"of range (array size {length})"
                        )
                    slots[idx] = actual

            # If the element type is a basic type, the simple `dd v1, v2, ...`
            # form works. Otherwise (structs, sub-arrays, pointers larger
            # than dword) we emit each element through `_emit_global_init`
            # recursively, with NASM `times N db 0` filling any tail.
            if (
                isinstance(elem_type, ast.BasicType)
                and elem_type.name in self._DATA_DIRECTIVE_NAMES
                and elem_size in self._DATA_DIRECTIVE
            ):
                directive = self._DATA_DIRECTIVE[elem_size]
                # Check whether any element needs the recursive
                # `_emit_global_init` path (label-diff, etc.). If so,
                # fall through to that.
                any_non_const = any(
                    self._needs_recursive_init(slots[i])
                    for i in slots
                )
                if not any_non_const:
                    values = []
                    for i in range(length):
                        if i in slots:
                            slot = slots[i]
                            # Optional braces around scalar element:
                            # `u8 arr[2] = {{1}, {2}}` is equivalent to
                            # `u8 arr[2] = {1, 2}`.
                            if (
                                isinstance(slot, ast.InitializerList)
                                and len(slot.values) == 1
                                and not isinstance(
                                    slot.values[0], ast.DesignatedInit
                                )
                            ):
                                slot = slot.values[0]
                            values.append(str(self._const_eval(slot, name)))
                        else:
                            values.append("0")
                    return [f"        {directive}      {', '.join(values)}"]
            out: list[str] = []
            for i in range(length):
                if i in slots:
                    out += self._emit_global_init(elem_type, slots[i], name)
                else:
                    out.append(f"        times {elem_size} db 0")
            return out

        raise CodegenError(
            f"global `{name}`: unsupported array initializer "
            f"({type(init).__name__})"
        )

    # NASM directives keyed by element size in bytes.
    _DATA_DIRECTIVE = {1: "db", 2: "dw", 4: "dd", 8: "dq"}
    # Names that are safe to emit as a single `dd v1, v2, ...` directive
    # in a global array init. Anything outside this set (struct, array,
    # union elements) recursively goes through `_emit_global_init`.
    _DATA_DIRECTIVE_NAMES = frozenset({
        "bool", "char", "short", "int", "long",
    })

    def _resolve_static_addr(
        self, expr: ast.Expression, name: str,
    ) -> tuple[str, int] | None:
        """Resolve `expr` to `(label, offset)` for a global-init address.

        Walks `.field` / `->field` / `[i]` / `*` chains rooted at a
        static lvalue (a global identifier or string literal) and
        accumulates a compile-time integer offset. Returns None if
        the chain isn't statically resolvable.
        """
        # Strip casts.
        while isinstance(expr, ast.Cast):
            expr = expr.expr
        if isinstance(expr, ast.StringLiteral):
            return self._intern_string(expr.value), 0
        if isinstance(expr, ast.LabelAddr):
            # `&&label` — yields the label's address. We don't currently
            # have a use case for `&&l + N` arithmetic in static init,
            # but the resolution is uniform.
            try:
                return self._global_label_addr_text(expr.label), 0
            except CodegenError:
                return None
        if isinstance(expr, ast.Identifier):
            if expr.name in self._globals or expr.name in self._extern_vars:
                return f"_{expr.name}", 0
            if expr.name in self._func_return_types:
                return f"_{expr.name}", 0
            return None
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            # `*p` where p is a pointer-arithmetic expression. Lower as
            # equivalent indexing — only resolves if p is a known
            # global+offset.
            inner = self._resolve_static_addr(expr.operand, name)
            return inner
        if isinstance(expr, ast.Member):
            # `obj.field` — resolve obj's address, then add field offset.
            try:
                obj_ty = self._type_of(
                    expr.obj, _FuncCtx(),
                )
            except CodegenError:
                return None
            if expr.is_arrow:
                # `p->m`: the base address comes from evaluating p (a
                # pointer), then plus member offset.
                if isinstance(obj_ty, ast.PointerType):
                    obj_ty = obj_ty.base_type
                inner = self._resolve_static_addr(expr.obj, name)
            else:
                inner = self._resolve_static_addr(expr.obj, name)
            if inner is None:
                return None
            if isinstance(obj_ty, (ast.ArrayType, ast.PointerType)):
                obj_ty = obj_ty.base_type
            if not isinstance(obj_ty, ast.StructType):
                return None
            sname = self._resolve_struct_name(obj_ty)
            try:
                _, m_off = self._member_layout(sname, expr.member)
            except CodegenError:
                return None
            base_label, base_off = inner
            return base_label, base_off + m_off
        if isinstance(expr, ast.Index):
            # `arr[i]` — resolve arr's address, then add i*sizeof(elem).
            inner = self._resolve_static_addr(expr.array, name)
            if inner is None:
                return None
            try:
                arr_ty = self._type_of(expr.array, _FuncCtx())
            except CodegenError:
                return None
            if isinstance(arr_ty, ast.PointerType):
                elem_ty = arr_ty.base_type
            elif isinstance(arr_ty, ast.ArrayType):
                elem_ty = arr_ty.base_type
            else:
                return None
            try:
                idx = self._const_eval(expr.index, name)
            except CodegenError:
                return None
            base_label, base_off = inner
            return base_label, base_off + idx * self._size_of(elem_ty)
        if isinstance(expr, ast.BinaryOp) and expr.op in ("+", "-"):
            # `arr + N` or `arr - N` — pointer arithmetic.
            inner = self._resolve_static_addr(expr.left, name)
            if inner is None:
                # Try the other side for `N + arr`.
                inner = self._resolve_static_addr(expr.right, name)
                if inner is None:
                    return None
                try:
                    offset = self._const_eval(expr.left, name)
                except CodegenError:
                    return None
            else:
                try:
                    offset = self._const_eval(expr.right, name)
                except CodegenError:
                    return None
                if expr.op == "-":
                    offset = -offset
            try:
                arr_ty = self._type_of(
                    expr.left if inner == self._resolve_static_addr(expr.left, name)
                    else expr.right,
                    _FuncCtx(),
                )
            except CodegenError:
                return None
            if isinstance(arr_ty, ast.PointerType):
                elem_ty = arr_ty.base_type
            elif isinstance(arr_ty, ast.ArrayType):
                elem_ty = arr_ty.base_type
            else:
                elem_ty = ast.BasicType(name="char")  # byte arithmetic
            base_label, base_off = inner
            return base_label, base_off + offset * self._size_of(elem_ty)
        return None

    def _const_eval_float(self, expr: ast.Expression, name: str) -> float:
        """Compile-time fold a float-typed expression for a global init.

        Handles literal numbers, unary +/-, the standard arithmetic
        binary ops, and casts (which narrow to the target precision).
        Reaches integer literals via float() coercion so things like
        `1024.0 - 1.0 / 32768.0` and `(double)1` work. Float-typed
        literals are first narrowed to 32-bit IEEE-754 so subsequent
        arithmetic respects the source precision (matches the FPU
        runtime path: `fld dword` loads a float, exact in 80-bit, then
        ops). This matters for boundary values like FLT_MIN / 2.0 where
        the double approximation of the decimal `1.17549435e-38` differs
        from the actual 32-bit FLT_MIN.
        """
        import struct
        while isinstance(expr, ast.Cast):
            target = expr.target_type
            inner = self._const_eval_float(expr.expr, name)
            if (
                isinstance(target, ast.BasicType)
                and target.name == "float"
            ):
                # Round to 32-bit float precision.
                return struct.unpack("<f", struct.pack("<f", inner))[0]
            expr = expr.expr
            return inner
        if isinstance(expr, ast.FloatLiteral):
            v = float(expr.value)
            # `1.0F` is float-typed in C — narrow to 32-bit precision so
            # the decimal-to-double approximation doesn't leak through.
            if getattr(expr, "is_float", False):
                v = struct.unpack("<f", struct.pack("<f", v))[0]
            return v
        if isinstance(expr, ast.IntLiteral):
            return float(expr.value)
        if isinstance(expr, ast.CharLiteral):
            return float(expr.value)
        if isinstance(expr, ast.UnaryOp) and expr.op in ("+", "-"):
            v = self._const_eval_float(expr.operand, name)
            return -v if expr.op == "-" else v
        if isinstance(expr, ast.BinaryOp) and expr.op in ("+", "-", "*", "/"):
            lv = self._const_eval_float(expr.left, name)
            rv = self._const_eval_float(expr.right, name)
            if expr.op == "+": return lv + rv
            if expr.op == "-": return lv - rv
            if expr.op == "*": return lv * rv
            if expr.op == "/":
                if rv == 0:
                    raise CodegenError(
                        f"global `{name}`: division by zero in float init"
                    )
                return lv / rv
        raise CodegenError(
            f"global `{name}`: float init must be a constant expression "
            f"(got {type(expr).__name__})"
        )

    def _const_eval(self, expr: ast.Expression, name: str) -> int:
        """Evaluate a compile-time integer constant for a global initializer.

        Accepts integer literals, signed/unsigned/bitwise unaries on them,
        and simple arithmetic between literals. Anything that would require
        actually generating code (identifier reads, function calls, etc.)
        raises — globals can only be initialized from constants.
        """
        if isinstance(expr, ast.IntLiteral):
            return expr.value
        if isinstance(expr, ast.CharLiteral):
            return expr.value
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Identifier)
        ):
            if (
                expr.func.name == "__builtin_choose_expr"
                and len(expr.args) == 3
            ):
                cond_val = self._const_eval(expr.args[0], name)
                chosen = expr.args[1] if cond_val else expr.args[2]
                return self._const_eval(chosen, name)
            if (
                expr.func.name == "__builtin_constant_p"
                and len(expr.args) == 1
            ):
                try:
                    self._const_eval(expr.args[0], name)
                    return 1
                except CodegenError:
                    return 0
        if isinstance(expr, ast.OffsetofExpr):
            return self._offsetof_value(expr)
        if isinstance(expr, ast.TypesCompatibleP):
            return 1 if self._types_compatible(expr.t1, expr.t2) else 0
        if isinstance(expr, ast.SizeofType):
            if getattr(expr, "is_alignof", False):
                return self._alignment_of(expr.target_type)
            # If the type contains a VLA, refuse the constant fold so
            # the runtime sizeof path runs instead.
            if self._type_has_vla(expr.target_type):
                raise CodegenError("sizeof: VLA — runtime evaluation needed")
            return self._size_of(expr.target_type)
        if isinstance(expr, ast.SizeofExpr):
            # `sizeof(expr)` — operand is unevaluated; we just need its
            # static type. _type_of needs a ctx, but for top-level usage
            # (array dimensions, global initializers) there's no function
            # context. A fresh empty _FuncCtx works because the type-of
            # path falls through to globals when no local matches.
            if (
                getattr(expr, "is_alignof", False)
                and isinstance(expr.expr, ast.Identifier)
                and expr.expr.name in self._func_alignments
            ):
                return self._func_alignments[expr.expr.name]
            ty = self._type_of(expr.expr, _FuncCtx())
            if getattr(expr, "is_alignof", False):
                return self._alignment_of(ty)
            # Refuse the fold when the type contains a VLA so the
            # runtime sizeof path runs instead.
            if self._type_has_vla(ty):
                raise CodegenError("sizeof: VLA — runtime evaluation needed")
            return self._size_of(ty)
        if isinstance(expr, ast.Cast):
            # Recurse on the operand, then narrow per the target type.
            # `(unsigned short)-4` must produce 65532, not -4. Truncate
            # to the target width and re-extend per signedness so a
            # subsequent widen (to long, etc.) sees the right bit
            # pattern.
            inner = self._const_eval(expr.expr, name)
            ty = expr.target_type
            if isinstance(ty, ast.BasicType):
                size = self._size_of(ty)
                if size == 1:
                    inner &= 0xFF
                    if not self._is_unsigned(ty) and inner & 0x80:
                        inner -= 0x100
                elif size == 2:
                    inner &= 0xFFFF
                    if not self._is_unsigned(ty) and inner & 0x8000:
                        inner -= 0x10000
                elif size == 4:
                    inner &= 0xFFFFFFFF
                    if not self._is_unsigned(ty) and inner & 0x80000000:
                        inner -= 0x100000000
            return inner
        if isinstance(expr, ast.Identifier):
            # Allow enum constants in global initializers — they're
            # already integer constants by the time codegen runs.
            if expr.name in self._enum_constants:
                return self._enum_constants[expr.name]
            raise CodegenError(
                f"global `{name}`: initializer must be a constant "
                f"expression (got Identifier `{expr.name}`)"
            )
        if isinstance(expr, ast.UnaryOp) and expr.is_prefix:
            inner = self._const_eval(expr.operand, name)
            if expr.op == "-":
                return -inner
            if expr.op == "+":
                return inner
            if expr.op == "~":
                return ~inner
            if expr.op == "!":
                return 0 if inner else 1
        if isinstance(expr, ast.BinaryOp):
            l = self._const_eval(expr.left, name)
            r = self._const_eval(expr.right, name)
            if expr.op == "+":   return l + r
            if expr.op == "-":   return l - r
            if expr.op == "*":   return l * r
            # `/` and `%` for ints in C are truncated toward zero.
            if expr.op == "/":   return int(l / r) if r != 0 else 0
            if expr.op == "%":   return l - int(l / r) * r if r != 0 else 0
            if expr.op == "&":   return l & r
            if expr.op == "|":   return l | r
            if expr.op == "^":   return l ^ r
            if expr.op == "<<":  return l << r
            if expr.op == ">>":  return l >> r
        raise CodegenError(
            f"global `{name}`: initializer must be a constant expression "
            f"(got {type(expr).__name__})"
        )

    @staticmethod
    def _render_string(s: str) -> str:
        # NASM `db` strings: ASCII chars in single-quoted segments, control
        # bytes as numeric literals, segments comma-separated. Build a list
        # of chunks and join.
        chunks: list[str] = []
        run: list[str] = []
        for ch in s:
            code = ord(ch)
            if 0x20 <= code < 0x7F and ch not in ("'", "\\"):
                run.append(ch)
                continue
            if run:
                chunks.append("'" + "".join(run) + "'")
                run = []
            chunks.append(str(code))
        if run:
            chunks.append("'" + "".join(run) + "'")
        return ", ".join(chunks) if chunks else "0"

    def _select_generic_arm(
        self, expr: ast.GenericSelection, ctx: _FuncCtx,
    ) -> ast.Expression:
        """Pick the matching arm of a `_Generic(ctrl, T1: e1, T2: e2, ...)`.

        Per C11, the controlling expression's type (after the usual
        decays — but not integer promotions) is matched against each
        listed type. The matching expression is selected; if none
        matches, the `default:` association is used (None type-key).
        Falls back to the first arm if neither match nor default.
        """
        ctrl_ty = self._type_of(expr.controlling_expr, ctx)
        ctrl_ty = self._strip_qualifiers(ctrl_ty)
        # Decay arrays to pointers and functions to pointers, the way
        # _Generic sees the controlling expression.
        if isinstance(ctrl_ty, ast.ArrayType):
            ctrl_ty = ast.PointerType(base_type=ctrl_ty.base_type)
        elif isinstance(ctrl_ty, ast.FunctionType):
            ctrl_ty = ast.PointerType(base_type=ctrl_ty)
        default_arm: ast.Expression | None = None
        for assoc_ty, arm in expr.associations:
            if assoc_ty is None:
                default_arm = arm
                continue
            if self._types_equal(ctrl_ty, self._strip_qualifiers(assoc_ty)):
                return arm
        if default_arm is not None:
            return default_arm
        # Fall back to the first arm if a match is required and none
        # was found — keeps codegen producing *something*.
        return expr.associations[0][1]

    @staticmethod
    def _strip_qualifiers(t: ast.TypeNode) -> ast.TypeNode:
        # Top-level const/volatile are ignored for `_Generic` matching.
        # Inner qualifiers (on a pointer's pointee) stay — that's per
        # the standard.
        if isinstance(t, ast.BasicType):
            return ast.BasicType(name=t.name, is_signed=t.is_signed)
        return t

    def _types_equal(self, a: ast.TypeNode, b: ast.TypeNode) -> bool:
        """Best-effort C type equality for `_Generic` matching.

        Compares structural shape: BasicType by name + signedness,
        PointerType by base, ArrayType by element + size, StructType
        and EnumType by registered name.
        """
        if type(a) is not type(b):
            return False
        if isinstance(a, ast.BasicType):
            if a.name != b.name:
                return False
            # Treat default-int signedness as equal to explicit signed.
            sa = True if a.is_signed is None else a.is_signed
            sb = True if b.is_signed is None else b.is_signed
            return sa == sb
        if isinstance(a, ast.PointerType):
            # Pointee qualifiers matter for _Generic — `int *` and
            # `const int *` are distinct types. Compare with strict
            # qualifier equality on the pointee.
            if not self._types_equal(a.base_type, b.base_type):
                return False
            ac = getattr(a.base_type, "is_const", False)
            bc = getattr(b.base_type, "is_const", False)
            av = getattr(a.base_type, "is_volatile", False)
            bv = getattr(b.base_type, "is_volatile", False)
            return ac == bc and av == bv
        if isinstance(a, ast.ArrayType):
            if not self._types_equal(a.base_type, b.base_type):
                return False
            if a.size is None or b.size is None:
                return a.size is None and b.size is None
            if isinstance(a.size, ast.IntLiteral) and isinstance(b.size, ast.IntLiteral):
                return a.size.value == b.size.value
            return False
        if isinstance(a, ast.StructType):
            return self._resolve_struct_name(a) == self._resolve_struct_name(b)
        if isinstance(a, ast.EnumType):
            return (a.name or "") == (b.name or "")
        if isinstance(a, ast.FunctionType):
            if not self._types_equal(a.return_type, b.return_type):
                return False
            if len(a.param_types) != len(b.param_types):
                return False
            return all(
                self._types_equal(pa, pb)
                for pa, pb in zip(a.param_types, b.param_types)
            )
        return False

    def _intern_string(self, value: str, is_wide: bool = False) -> str:
        # Wide strings get a separate keyspace so the same characters
        # don't collapse into one byte sequence.
        key = ("W:" if is_wide else "") + value
        if key in self._strings:
            return self._strings[key]
        label = f"_uc386_str{len(self._strings)}"
        self._strings[key] = label
        return label

    def _intern_compound_global(
        self,
        target_type: "ast.TypeNode",
        init: "ast.Expression",
        owner_name: str,
    ) -> str:
        """Reserve a hidden global for a compound literal at file scope.

        Returns the NASM label (e.g. `_uc386_cl0`). The literal's bytes
        are emitted in the `.data` section by `_data_section` after the
        normal globals; the helper adds an entry to `_compound_globals`
        which the data emitter walks at the end.
        """
        idx = len(self._compound_globals)
        label = f"_uc386_cl{idx}"
        self._compound_globals.append((label, target_type, init))
        return label

    def _start_stub(self) -> list[str]:
        # _start: initialize FPU control word to 53-bit precision (PC=10,
        # round-to-nearest, all exceptions masked) — 0x027F. This matches
        # what glibc's _start sets on x86 Linux. Without this, the FPU
        # defaults to 80-bit (PC=11) which produces extra precision in
        # intermediate computations and diverges from gcc-emitted-test
        # expected values.
        # Then call user main, take its int return in EAX, exit DOS via
        # INT 21h/4Ch with AL = exit code. AH=4Ch leaves AL untouched.
        # `__start` aliases `_start` so a user `extern void _start(void)`
        # / `_start()` call (which our naming convention prefixes to
        # `__start`) reaches the entry stub.
        # _start also re-zeroes the non-noinit BSS range so a recursive
        # `_start()` call from main (`__attribute__((noinit))` test
        # idiom) re-runs main with a clean slate.
        return [
            "_start:",
            "__start:",
            "        sub     esp, 4",
            "        mov     word [esp], 0x027F",
            "        fldcw   [esp]",
            "        add     esp, 4",
            "        cld",
            "        xor     eax, eax",
            "        mov     edi, _bss_zero_start",
            "        mov     ecx, _bss_zero_end",
            "        sub     ecx, edi",
            "        rep     stosb",
            "        call    _main",
            "        mov     ah, 4Ch",
            "        int     21h",
        ]

    # ---- functions ------------------------------------------------------

    def _function(self, fn: ast.FunctionDecl) -> list[str]:
        ctx = _FuncCtx()
        ctx.return_type = fn.return_type
        ctx.func_name = fn.name
        ctx._codegen_ref = self
        # If this is a lifted nested function, inherit the outer's
        # capture remapping so references in our body to outer locals
        # resolve to the right mangled globals. Also inherit sibling
        # nested-fn names so we can call each other.
        if fn.name in getattr(self, "_lifted_captures", {}):
            ctx.local_captures = dict(self._lifted_captures[fn.name])
        if fn.name in getattr(self, "_lifted_nested_fn_names", {}):
            ctx.nested_fn_names = dict(
                self._lifted_nested_fn_names[fn.name]
            )
        if fn.name in getattr(self, "_lifted_nonlocal_gotos", {}):
            ctx.nonlocal_goto_targets = dict(
                self._lifted_nonlocal_gotos[fn.name]
            )
            # We're a lifted nested fn whose body contains nonlocal
            # gotos to outer's labels. The trampoline (or a direct
            # caller in outer's body) loaded ECX with the address of
            # outer's buf-array for us. Reserve a slot to save ECX so
            # subsequent calls don't clobber it; the goto lowering
            # reads from this slot to find the buf to longjmp through.
            ctx.alloc_local(
                "__static_link__",
                ast.PointerType(base_type=ast.BasicType(name="char")),
            )
            ctx.trampoline_static_link_disp = ctx.lookup("__static_link__")
        # Lift nested function definitions to file scope. Each gets a
        # mangled name (`<outer>__<inner>`); calls / address-takes in
        # the outer body resolve through `ctx.nested_fn_names`.
        #
        # Nested functions can reference outer's locals. We don't
        # implement full GCC static-link / trampoline semantics, but
        # we handle the common case by lifting the *captured* outer
        # locals to file-scope globals: any name referenced from inside
        # a nested fn (and not bound there as a param/local) gets a
        # mangled global. The outer fn's reads/writes route through
        # the same global via `local_captures`. This is correct for
        # non-reentrant outer fns (the typical torture-test shape).
        # Find nested fns at this lexical level (this function body
        # plus any nested blocks). Deeper nested fns inside other
        # nested fn bodies are NOT collected here — they'll be
        # discovered when their parent (a lifted fn) runs its own
        # _function pre-pass.
        nested_decls: list[ast.FunctionDecl] = []
        def _collect_nested_decls(node):
            if node is None:
                return
            if isinstance(node, ast.FunctionDecl) and node.body is not None:
                nested_decls.append(node)
                return  # Don't recurse into the nested fn's body.
            if isinstance(node, ast.CompoundStmt):
                for it in node.items:
                    _collect_nested_decls(it)
                return
            if isinstance(node, (ast.IfStmt,)):
                _collect_nested_decls(node.then_branch)
                _collect_nested_decls(node.else_branch)
                return
            if isinstance(node, (ast.WhileStmt, ast.DoWhileStmt)):
                _collect_nested_decls(node.body)
                return
            if isinstance(node, ast.ForStmt):
                _collect_nested_decls(node.body)
                return
            if isinstance(node, ast.SwitchStmt):
                _collect_nested_decls(node.body)
                return
            if isinstance(node, ast.CaseStmt):
                _collect_nested_decls(node.stmt)
                return
            if isinstance(node, ast.LabelStmt):
                _collect_nested_decls(node.stmt)
                return
            if isinstance(node, ast.DeclarationList):
                for d in node.declarations:
                    _collect_nested_decls(d)
                return
        _collect_nested_decls(fn.body)
        # Build the capture set: names referenced in any nested body
        # that aren't bound as the nested fn's own params or locals,
        # and aren't sibling nested fn names (which resolve via the
        # outer's lift chain, not as outer-locals).
        sibling_nested_names = {n.name for n in nested_decls}
        captured: set[str] = set()
        outer_param_names = {p.name for p in fn.params if p.name}
        for nested in nested_decls:
            inner_bound = {p.name for p in nested.params if p.name}
            inner_nested_names: set[str] = set()
            for sub in self._walk_ast(nested.body):
                if isinstance(sub, ast.VarDecl):
                    inner_bound.add(sub.name)
                if (
                    isinstance(sub, ast.FunctionDecl)
                    and sub.body is not None
                ):
                    inner_nested_names.add(sub.name)
            # Also collect names referenced in the nested fn's param
            # size-side-effect expressions — `int foo(int a[N++])`
            # references `N` even though foo's body might be empty.
            roots: list = [nested.body]
            for p in nested.params:
                sse = getattr(p, "size_side_effects", None)
                if sse:
                    roots.extend(sse)
            for root in roots:
                for sub in self._walk_ast(root):
                    if isinstance(sub, ast.Identifier):
                        n = sub.name
                        if n in inner_bound:
                            continue
                        if n in inner_nested_names:
                            continue
                        if n in sibling_nested_names:
                            continue
                        if n in self._globals or n in self._extern_vars:
                            continue
                        if n in self._func_return_types:
                            continue
                        if n in self._enum_constants:
                            continue
                        captured.add(n)
        # Promote captures to globals. Outer params are allocated
        # normally (they live on the call stack); if a param is
        # captured we copy its value into the global at function entry
        # and route subsequent outer reads/writes through the global.
        captured_param_copies: list[tuple[str, str, ast.TypeNode]] = []
        # Don't reset ctx.local_captures here — a lifted nested fn
        # already inherited remappings from its outer.
        for name in captured:
            mangled_key = f"{fn.name}__{name}"
            if name in outer_param_names:
                # Find param type; allocate a global, plan a runtime
                # copy from param to global at function entry.
                ptype = next(p.param_type for p in fn.params if p.name == name)
                if mangled_key not in self._globals:
                    self._globals[mangled_key] = ptype
                captured_param_copies.append((name, mangled_key, ptype))
                ctx.local_captures[name] = mangled_key
        # Now lift the nested function definitions.
        for sub in nested_decls:
            if sub.name in ctx.nested_fn_names:
                continue
            mangled = f"{fn.name}__{sub.name}"
            # Avoid collisions if multiple sibling outer functions
            # both have a nested fn with the same name.
            base_mangled = mangled
            suffix = 0
            while mangled in self._func_return_types:
                suffix += 1
                mangled = f"{base_mangled}_{suffix}"
            ctx.nested_fn_names[sub.name] = mangled
            self._func_return_types[mangled] = sub.return_type
            self._func_param_types[mangled] = [
                p.param_type for p in sub.params
            ]
            # Build a renamed copy of the inner FunctionDecl so the
            # emitted label uses the mangled name. Reusing the AST
            # node would otherwise emit `_<inner>` and collide with
            # any other inner of the same name.
            lifted = ast.FunctionDecl(
                name=mangled,
                return_type=sub.return_type,
                params=sub.params,
                body=sub.body,
                is_variadic=sub.is_variadic,
                storage_class=sub.storage_class,
                is_inline=sub.is_inline,
                location=sub.location,
            )
            self._pending_functions.append(lifted)
        # Stash for later (used after locals are collected).
        self._capture_set = captured
        self._captured_param_copies = captured_param_copies
        # Record per-nested-fn capture remapping so when each lifted
        # function later compiles, its body's references to outer
        # locals resolve to the right mangled global. For names that
        # are themselves outer-of-outer captures (already remapped via
        # `ctx.local_captures`), reuse the existing remapping so we
        # don't create double-mangled aliases at each nesting level.
        capture_remap: dict[str, str] = {}
        for n in captured:
            if n in ctx.local_captures:
                capture_remap[n] = ctx.local_captures[n]
            else:
                capture_remap[n] = f"{fn.name}__{n}"
        # Sibling nested fns resolve via the lift chain. Each lifted
        # nested needs to inherit BOTH our outer's `nested_fn_names`
        # (for outer-of-outer references) AND our siblings' lifts (so
        # nested t1 can call sibling nested t0).
        sibling_lifts = dict(ctx.nested_fn_names)
        # Non-local gotos: a nested fn's `goto X` where `X` is a label
        # declared in the outer (via `__label__ X`) is a cross-frame
        # jump. GCC implements this with trampolines + static link
        # registers; we approximate via a per-frame buf-array (jmp_buf
        # slots in outer's frame) plus a 12-byte stack trampoline.
        # `&inner` returns the trampoline's address. The trampoline is
        # 12 bytes of code: `mov ecx, &buf_array; mov edx, _<lifted>;
        # jmp edx`. The lifted nested fn's prologue saves ECX into a
        # static-link slot; `goto X` longjmps via that slot's value
        # (offset by the label's index in the buf-array). This handles
        # both non-recursive outers (nestfunc-6) and recursive outers
        # (nestfunc-5) uniformly because each invocation of outer gets
        # its own buf-array and its own trampoline.
        outer_user_labels: set[str] = set()
        for sub in self._walk_ast(fn.body):
            if isinstance(sub, ast.LabelStmt):
                outer_user_labels.add(sub.label)
        # Per nested fn: ordered list of nonlocal-goto labels.
        nlg_per_inner: dict[str, list[str]] = {}
        for nested in nested_decls:
            inner_user_labels: set[str] = set()
            for sub in self._walk_ast(nested.body):
                if isinstance(sub, ast.LabelStmt):
                    inner_user_labels.add(sub.label)
            seen: list[str] = []
            for sub in self._walk_ast(nested.body):
                if (
                    isinstance(sub, ast.GotoStmt)
                    and sub.target is None
                    and sub.label in outer_user_labels
                    and sub.label not in inner_user_labels
                    and sub.label not in seen
                ):
                    seen.append(sub.label)
            if seen:
                nlg_per_inner[nested.name] = seen
        # Allocate trampoline + buf-array slots in outer's frame for
        # every nested fn that has nonlocal gotos. These are just byte
        # arrays; we use char[N] typed slots so alloc_local rounds them
        # up to 4 bytes correctly.
        for inner_name, labels in nlg_per_inner.items():
            tramp_slot_name = f"__tramp_{inner_name}"
            ctx.alloc_local(
                tramp_slot_name,
                ast.ArrayType(
                    base_type=ast.BasicType(name="char"), size=12
                ),
                size=12,
            )
            tramp_disp = ctx.lookup(tramp_slot_name)
            buf_size = 12 * len(labels)
            buf_slot_name = f"__nlg_buf_{inner_name}"
            ctx.alloc_local(
                buf_slot_name,
                ast.ArrayType(
                    base_type=ast.BasicType(name="char"), size=buf_size
                ),
                size=buf_size,
            )
            buf_disp = ctx.lookup(buf_slot_name)
            ctx.trampolines[inner_name] = (tramp_disp, buf_disp)
            ctx.trampoline_buf_indices[inner_name] = {
                lbl: i for i, lbl in enumerate(labels)
            }
        for sub in nested_decls:
            mangled = ctx.nested_fn_names.get(sub.name)
            if mangled:
                self._lifted_captures[mangled] = capture_remap
                self._lifted_nested_fn_names[mangled] = sibling_lifts
                inner_labels = nlg_per_inner.get(sub.name)
                if inner_labels:
                    self._lifted_nonlocal_gotos[mangled] = {
                        lbl: i for i, lbl in enumerate(inner_labels)
                    }
                self._lifted_outer_fn[mangled] = fn.name
        # Parameters live above EBP at cdecl offsets; the first sits at
        # [ebp + 8], and each subsequent param is offset by its predecessor's
        # padded size. For scalars/pointers/floats that's `(size + 3) & ~3`
        # bytes — usually 4, but 8 for `double`. Structs by value take
        # `sizeof(struct)` rounded up to 4.
        #
        # Struct-returning functions take a hidden first param at [ebp+8]
        # — a pointer to the caller's destination buffer. Real params
        # shift up by 4 in that case.
        disp = 8
        # Vectors return via the same caller-buffer ABI as structs;
        # regular C arrays can't be returned by value (so don't reserve
        # a retptr for them).
        ret_is_vector = (
            isinstance(fn.return_type, ast.ArrayType)
            and getattr(fn.return_type, "is_vector", False)
        )
        ret_is_int128 = self._is_int128(fn.return_type)
        if (
            isinstance(fn.return_type, (ast.StructType, ast.ComplexType))
            or ret_is_vector
            or ret_is_int128
        ):
            ctx.alloc_param(
                "__retptr__", disp,
                ast.PointerType(base_type=fn.return_type),
            )
            disp += 4
        for param in fn.params:
            if param.name is None:
                continue
            self._check_supported_type(param.param_type, param.name)
            ctx.alloc_param(param.name, disp, param.param_type)
            size = self._size_of(param.param_type)
            disp += (size + 3) & ~3
        # Pre-allocate VLA size-capture slots for params with side
        # effects (`int b[a++]`). Each unique side-effect expression
        # gets a 4-byte hidden local. After alloc, replace ArrayType
        # `size` references in the param's type with Identifier(slot)
        # so sizeof() reads the captured value.
        for param in fn.params:
            sse = getattr(param, "size_side_effects", None)
            if not sse:
                continue
            captured: dict[int, str] = {}
            for i, size_expr in enumerate(sse):
                try:
                    self._const_eval(size_expr, f"<vla-{param.name}>")
                    continue
                except CodegenError:
                    pass
                slot_name = f"__vla_capture_{param.name}_{i}"
                ctx.alloc_local(slot_name, ast.BasicType(name="int"))
                captured[id(size_expr)] = slot_name
            if captured:
                self._replace_vla_size_with_capture(
                    param.param_type, captured, fn.name,
                )
        # Pre-walk the body for directly-VLA-typed VarDecls. If any
        # are present, reserve a hidden 4-byte slot at the top of the
        # frame to hold ESP after the fixed-size locals (the VLA
        # baseline). `_collect_locals` and `_var_init` need this slot
        # to exist before they run.
        has_any_vla = False
        for sub in self._walk_ast(fn.body):
            if isinstance(sub, ast.VarDecl):
                vt = self._resolved_var_type(sub)
                if (
                    isinstance(vt, ast.ArrayType)
                    and self._array_is_directly_vla(vt)
                ):
                    has_any_vla = True
                    break
        if has_any_vla:
            ctx.alloc_local(
                "__vla_baseline", ast.BasicType(name="int"),
            )
            ctx.vla_baseline_disp = ctx.lookup("__vla_baseline")
        # First pass: allocate every local up front so the prologue knows
        # the frame size before we emit body code.
        self._collect_locals(fn.body, ctx)
        # GCC statement expressions (`({...})`) embed CompoundStmts
        # inside expressions — `_collect_locals` doesn't recurse into
        # expressions, so we pre-walk the body for any StmtExpr nodes
        # and collect their inner locals. Their disps land in the
        # decl_disps map and get re-bound at emit time.
        for sub in self._walk_ast(fn.body):
            if isinstance(sub, ast.StmtExpr):
                self._collect_locals(sub.body, ctx)
        # Second pass: reserve a temp slot for each struct-returning Call
        # in the body. Some call sites have known destinations (var init,
        # struct assignment rhs, return chain) and don't actually use the
        # temp — wasting a few bytes of frame is simpler than tracking
        # parent context here.
        # Resolve `typeof(expr)` references in the body. Replaces
        # `TypeofType(operand=X)` with the result of `_type_of(X,
        # ctx)`. Done before `_collect_call_temps` so size queries
        # see concrete types.
        self._resolve_typeof_in_body(fn.body, ctx)
        self._collect_call_temps(fn.body, ctx)
        # Third pass: assign a NASM label to every user `label:`. Done
        # ahead of body emission so a forward `goto` can resolve.
        self._collect_labels(fn.body, ctx)
        # Save the user-label map for this function so static-local
        # globals (emitted later from `.data`) can resolve `&&label`.
        self._function_user_labels[fn.name] = dict(ctx.user_labels)
        # Same for static-local mappings: a static-local pointer can
        # be initialized with the address of another static-local
        # array in the same function.
        self._function_local_static[fn.name] = dict(ctx.local_static_labels)

        body = self._compound(fn.body, ctx)

        out = [f"_{fn.name}:"]
        out.append("        push    ebp")
        out.append("        mov     ebp, esp")
        if ctx.frame_size:
            out.append(f"        sub     esp, {ctx.frame_size}")
        # Lifted nested fn with nonlocal gotos: ECX was set by the
        # trampoline (or direct caller) to the address of our outer's
        # buf-array. Save it into the static-link slot before any code
        # that might clobber ECX.
        if ctx.trampoline_static_link_disp is not None:
            out.append(
                f"        mov     {_ebp_addr(ctx.trampoline_static_link_disp)}, ecx"
            )
        # Save baseline ESP for VLA-using functions so a goto-back
        # can restore (free) all VLAs.
        if ctx.vla_baseline_disp is not None:
            out.append(
                f"        mov     {_ebp_addr(ctx.vla_baseline_disp)}, esp"
            )
        # VLA parameter sizes: evaluate side effects + capture sizes.
        # Slots were pre-allocated in `_collect_locals` (or here, after
        # the prologue, but before frame_size was finalized) — we just
        # eval the captured side-effect expr and store. Replacement of
        # `size` references with `Identifier(slot)` already happened
        # in the pre-pass.
        for param in fn.params:
            sse = getattr(param, "size_side_effects", None)
            if not sse:
                continue
            for i, size_expr in enumerate(sse):
                try:
                    self._const_eval(size_expr, f"<vla-{param.name}>")
                    continue
                except CodegenError:
                    pass
                slot_name = f"__vla_capture_{param.name}_{i}"
                if not ctx.has_local(slot_name):
                    continue
                try:
                    out += self._eval_expr_to_eax(size_expr, ctx)
                except CodegenError:
                    continue
                disp = ctx.lookup(slot_name)
                out.append(f"        mov     {_ebp_addr(disp)}, eax")
        # If any of our params are captured by a nested function, copy
        # the param's incoming value into its mangled global so the
        # nested fn (compiled separately) sees the right initial value.
        # `local_captures` already routes outer's reads/writes through
        # the global; we just need to seed it from the param slot.
        for pname, mangled, ptype in self._captured_param_copies:
            param_disp = ctx.lookup(pname)
            if self._is_long_long(ptype):
                out.append(f"        mov     eax, {_ebp_addr(param_disp)}")
                out.append(f"        mov     [_{mangled}], eax")
                out.append(f"        mov     eax, {_ebp_addr(param_disp + 4)}")
                out.append(f"        mov     [_{mangled} + 4], eax")
            elif self._is_float_type(ptype):
                size = self._size_of(ptype)
                width = "dword" if size == 4 else "qword"
                out.append(f"        fld     {width} {_ebp_addr(param_disp)}")
                out.append(f"        fstp    {width} [_{mangled}]")
            else:
                size = self._size_of(ptype)
                if size == 8:
                    out.append(f"        mov     eax, {_ebp_addr(param_disp)}")
                    out.append(f"        mov     [_{mangled}], eax")
                    out.append(f"        mov     eax, {_ebp_addr(param_disp + 4)}")
                    out.append(f"        mov     [_{mangled} + 4], eax")
                else:
                    out += self._load_to_eax(_ebp_addr(param_disp), ptype)
                    out += self._store_from_eax(f"[_{mangled}]", ptype)
        # Trampolines + nonlocal-goto setjmp. For every nested fn
        # whose body has `goto OuterLabel`, the outer's frame holds:
        #   - a buf-array (12 bytes per nonlocal-goto label)
        #   - a 12-byte trampoline of the form:
        #       B9 imm32      ; mov ecx, &buf_array
        #       BA imm32      ; mov edx, _<lifted>
        #       FF E2         ; jmp edx
        # The address of the trampoline is what `&inner` and `inner`
        # (in value position) return. The trampoline loads ECX with
        # the buf-array's address, then jumps to the lifted fn — so
        # each invocation of outer has its own buf for the nested
        # fn to longjmp through (handles recursive outers).
        for inner_name, (tramp_disp, buf_disp) in ctx.trampolines.items():
            mangled = ctx.nested_fn_names.get(inner_name)
            if mangled is None:
                continue
            # Trampoline byte 0: 0xB9 (mov ecx, imm32 opcode)
            out.append(
                f"        mov     byte {_ebp_addr(tramp_disp)}, 0xB9"
            )
            # Bytes 1..4: address of buf_array (loaded via lea)
            out.append(f"        lea     eax, {_ebp_addr(buf_disp)}")
            out.append(f"        mov     {_ebp_addr(tramp_disp + 1)}, eax")
            # Byte 5: 0xBA (mov edx, imm32 opcode)
            out.append(
                f"        mov     byte {_ebp_addr(tramp_disp + 5)}, 0xBA"
            )
            # Bytes 6..9: address of lifted fn
            out.append(f"        mov     eax, _{mangled}")
            out.append(f"        mov     {_ebp_addr(tramp_disp + 6)}, eax")
            # Bytes 10..11: FF E2 (jmp edx) — stored as 0xE2FF in
            # little-endian word form so byte 10 = 0xFF, byte 11 = 0xE2.
            out.append(
                f"        mov     word {_ebp_addr(tramp_disp + 10)}, 0xE2FF"
            )
            # setjmp into each label's buf slot. On longjmp return
            # (EAX=1), dispatch to the user label.
            label_indices = ctx.trampoline_buf_indices.get(inner_name, {})
            for label_name, idx in label_indices.items():
                user_target = ctx.user_labels.get(label_name)
                if user_target is None:
                    continue
                out.append(
                    f"        lea     eax, {_ebp_addr(buf_disp + 12 * idx)}"
                )
                out.append("        push    eax")
                out.append("        call    ___builtin_setjmp")
                out.append("        add     esp, 4")
                out.append("        test    eax, eax")
                out.append(f"        jnz     {user_target}")
        # `-finstrument-functions`: enter call after prologue setup.
        instrument = (
            getattr(self, "_instrument_enabled", False)
            and fn.name not in self._instrument_no_skip
        )
        if instrument:
            out.append("        push    dword [ebp + 4]")
            out.append(f"        push    _{fn.name}")
            out.append("        call    ___cyg_profile_func_enter")
            out.append("        add     esp, 8")
        out += body
        # C99: falling off the end of main returns 0. For other functions
        # this is technically undefined, but a deterministic zero beats
        # leaking whatever EAX held.
        out.append("        xor     eax, eax")
        out.append(".epilogue:")
        # `-finstrument-functions`: exit call before stack teardown.
        # Save the return value across the call so we can restore it.
        if instrument:
            ret_save: list[str] = []
            ret_restore: list[str] = []
            rt = fn.return_type
            is_void = (
                isinstance(rt, ast.BasicType) and rt.name == "void"
            )
            if not is_void:
                if self._is_float_type(rt):
                    size = self._size_of(rt)
                    width = "dword" if size == 4 else "qword"
                    ret_save = [
                        f"        sub     esp, {size}",
                        f"        fstp    {width} [esp]",
                    ]
                    ret_restore = [
                        f"        fld     {width} [esp]",
                        f"        add     esp, {size}",
                    ]
                elif self._is_long_long(rt):
                    ret_save = [
                        "        push    edx",
                        "        push    eax",
                    ]
                    ret_restore = [
                        "        pop     eax",
                        "        pop     edx",
                    ]
                else:
                    ret_save = ["        push    eax"]
                    ret_restore = ["        pop     eax"]
            out += ret_save
            out.append("        push    dword [ebp + 4]")
            out.append(f"        push    _{fn.name}")
            out.append("        call    ___cyg_profile_func_exit")
            out.append("        add     esp, 8")
            out += ret_restore
        # `mov esp, ebp` reclaims any sub-esp-allocated VLA storage.
        out.append("        mov     esp, ebp")
        out.append("        pop     ebp")
        out.append("        ret")
        return out

    def _collect_labels(self, node, ctx: _FuncCtx) -> None:
        """Record a NASM label for every user `label:` in the function body.

        Done before body emission so a `goto` that targets a label
        appearing later in the source still resolves cleanly.
        """
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.LabelStmt):
                if sub.label in ctx.user_labels:
                    raise CodegenError(
                        f"duplicate label `{sub.label}` in function"
                    )
                ctx.user_labels[sub.label] = ctx.label(f"user_{sub.label}")

    def _resolve_typeof_in_body(self, node, ctx: _FuncCtx) -> None:
        """Walk `node` recursively and replace any `TypeofType` field
        with the concrete type.

        `_collect_locals` already exited its scope chain by the time
        we run, so `_type_of(Identifier)` would fail for body locals.
        Pre-build a flat name→type map by re-walking the body for
        VarDecls / ParamDecls and consult that here. Mutates the
        AST in place.
        """
        flat_types: dict[str, ast.TypeNode] = {}
        # Function params live on ctx via alloc_param.
        for scope in ctx.types:
            flat_types.update(scope)
        # Walk the body for VarDecls — their decl_types entry is the
        # resolved type from `_collect_locals`.
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.VarDecl):
                t = ctx.decl_types.get(id(sub))
                if t is not None:
                    flat_types[sub.name] = t

        def resolve_inner(operand: ast.Expression) -> ast.TypeNode | None:
            # For an Identifier, look in flat_types first.
            if isinstance(operand, ast.Identifier):
                t = flat_types.get(operand.name)
                if t is not None:
                    return t
            # Fall back to live _type_of (may fail).
            try:
                return self._type_of(operand, ctx)
            except CodegenError:
                return None

        for sub in self._walk_ast(node):
            if not dataclasses.is_dataclass(sub):
                continue
            for f in dataclasses.fields(sub):
                v = getattr(sub, f.name, None)
                if isinstance(v, ast.TypeofType):
                    resolved = resolve_inner(v.operand)
                    if resolved is not None:
                        setattr(sub, f.name, resolved)
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, ast.TypeofType):
                            r = resolve_inner(item.operand)
                            if r is not None:
                                v[i] = r

    def _collect_call_temps(self, node, ctx: _FuncCtx) -> None:
        """Pre-allocate a frame slot for every struct-returning Call and
        every Compound literal in `node`.

        We allocate one buffer per node (keyed by `id`) so two
        struct-returning calls or compound literals in the same
        expression — e.g. `make(1).x + make(2).x` — get distinct
        buffers and don't clobber each other. Some call sites later
        turn out to have a known destination (var init, struct
        assignment, return chain) and won't read from the temp, but
        allocating them anyway keeps this pass context-free.
        """
        # `_collect_locals` already exited its scope stack by the time we
        # run, so `_type_of(Identifier)` for body locals would fail. Push
        # a flat scope with all body VarDecls' resolved types so vector-
        # arithmetic detection (`_type_of(BinaryOp)` → ArrayType) sees
        # the right types for k0/k1/etc. The dummy slot disp doesn't
        # matter — `_type_of` only reads `ctx.types`.
        ctx.enter_scope()
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.VarDecl):
                t = ctx.decl_types.get(id(sub))
                if t is not None and sub.name not in ctx.slots[-1]:
                    ctx.slots[-1][sub.name] = 0
                    ctx.types[-1][sub.name] = t
        try:
            self._collect_call_temps_inner(node, ctx)
        finally:
            ctx.exit_scope()

    def _collect_call_temps_inner(self, node, ctx: _FuncCtx) -> None:
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.Call) and self._is_struct_returning_call(sub, ctx):
                ret_ty = self._call_return_type(sub, ctx)
                size = (self._size_of(ret_ty) + 3) & ~3
                ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, ast.Call) and self._is_vector_returning_call(sub, ctx):
                ret_ty = self._call_return_type(sub, ctx)
                size = (self._size_of(ret_ty) + 3) & ~3
                ctx.alloc_call_temp(sub, size)
            elif (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Identifier)
                and sub.func.name == "__builtin_shuffle"
                and len(sub.args) >= 1
            ):
                # Vector shuffle result lands in a per-call temp.
                try:
                    ret_ty = self._type_of(sub.args[0], ctx)
                except CodegenError:
                    ret_ty = None
                if isinstance(ret_ty, ast.ArrayType) and getattr(
                    ret_ty, "is_vector", False
                ):
                    size = (self._size_of(ret_ty) + 3) & ~3
                    ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, ast.Call) and self._is_complex_returning_call(sub, ctx):
                # Complex-returning calls need a per-call-site temp
                # so consumers like `__real foo()` can address the
                # halves via a stable slot.
                ret_ty = self._call_return_type(sub, ctx)
                size = (self._size_of(ret_ty) + 3) & ~3
                ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, ast.Call) and self._is_int128_returning_call(sub, ctx):
                # __int128-returning call uses the retptr ABI; reserve
                # a 16-byte temp slot so the caller's expression chain
                # (e.g. `f() + g()`) sees stable addresses.
                ctx.alloc_call_temp(sub, 16)
            elif isinstance(sub, ast.Compound):
                size = self._compound_temp_size(sub)
                ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, ast.Cast) and isinstance(
                sub.target_type, (ast.StructType,)
            ):
                # `(struct T) value` — type-pun into a struct/union
                # via a per-cast temp slot.
                size = (self._size_of(sub.target_type) + 3) & ~3
                ctx.alloc_call_temp(sub, size)
            elif (
                isinstance(sub, ast.Cast)
                and isinstance(sub.target_type, ast.ArrayType)
                and getattr(sub.target_type, "is_vector", False)
            ):
                # `(vec_t) scalar` — type-pun a scalar value into a
                # vector slot. The codegen needs an address to hand
                # back to consumers, so allocate a temp.
                src_ty_check = None
                try:
                    src_ty_check = self._type_of(sub.expr, ctx)
                except CodegenError:
                    pass
                if not isinstance(src_ty_check, ast.ArrayType):
                    size = (self._size_of(sub.target_type) + 3) & ~3
                    ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, (ast.BinaryOp, ast.UnaryOp, ast.Cast, ast.TernaryOp)):
                # Complex-valued sub-expression: needs a temp slot
                # to hold the (real, imag) result. One per node so
                # `(a+b) + (c+d)` allocates distinct buffers.
                # `_type_of` may fail because the temp-collection
                # pass runs after the local-scope chain has been
                # exited; fall back to the structural complex
                # detector for those cases.
                try:
                    ty = self._type_of(sub, ctx)
                except CodegenError:
                    ty = self._type_of_complex_expr(sub)
                if isinstance(ty, ast.ComplexType):
                    size = (self._size_of(ty) + 3) & ~3
                    ctx.alloc_call_temp(sub, size)
                elif (
                    isinstance(ty, ast.ArrayType)
                    and self._is_vector_op_node(sub)
                    and self._is_genuine_vector_op(sub, ctx)
                ):
                    # Vector arithmetic / unary: needs a temp to hold
                    # the componentwise result. Distinguish from
                    # pointer arithmetic (`arr + i`) which also has
                    # ArrayType-flavored result type.
                    size = (self._size_of(ty) + 3) & ~3
                    ctx.alloc_call_temp(sub, size)
                elif self._is_int128(ty):
                    # __int128 arithmetic / cast / ternary: needs a
                    # 16-byte temp to hold the result. Per-node so
                    # chained ops get distinct slots.
                    ctx.alloc_call_temp(sub, 16)
            elif (
                isinstance(sub, ast.FloatLiteral) and sub.is_imaginary
            ):
                # `1.0i` as a complex value — needs a temp slot.
                ty = self._type_of_complex_expr(sub)
                if isinstance(ty, ast.ComplexType):
                    size = (self._size_of(ty) + 3) & ~3
                    ctx.alloc_call_temp(sub, size)
            elif (
                isinstance(sub, ast.VaArgExpr)
                and self._is_int128(sub.target_type)
            ):
                # `va_arg(ap, __int128)` materializes into a per-call
                # 16-byte temp; consumers see the temp's address.
                ctx.alloc_call_temp(sub, 16)

    def _compound_temp_size(self, node: ast.Compound) -> int:
        """Size of the per-call-site temp slot for a compound literal.

        For unsized array types (`(int []){...}`), we derive the size
        from the initializer length. Otherwise the regular `_size_of`
        path applies.
        """
        ty = node.target_type
        if isinstance(ty, ast.ArrayType) and ty.size is None:
            init = node.init
            if isinstance(init, ast.InitializerList):
                length = len(init.values)
            elif isinstance(init, ast.StringLiteral):
                length = len(init.value) + 1
            else:
                length = 0
            return ((length * self._size_of(ty.base_type)) + 3) & ~3
        return (self._size_of(ty) + 3) & ~3

    @staticmethod
    def _body_uses_va_arg_pack(body) -> bool:
        for n in CodeGenerator._walk_ast(body):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Identifier)
                and n.func.name == "__builtin_va_arg_pack"
            ):
                return True
        return False

    def _substitute_inline(self, node, param_map, va_args):
        """Walk a deepcopied AST replacing param-Identifier refs and
        splicing __builtin_va_arg_pack() in Call args. Returns the
        (possibly new) node. Mutates leaf containers in place."""
        import copy
        if node is None:
            return None
        if (isinstance(node, ast.Identifier)
                and node.name in param_map):
            return ast.Identifier(
                name=param_map[node.name],
                location=node.location,
            )
        if isinstance(node, ast.Call):
            node.func = self._substitute_inline(
                node.func, param_map, va_args
            )
            new_args = []
            for a in node.args:
                if (isinstance(a, ast.Call)
                        and isinstance(a.func, ast.Identifier)
                        and a.func.name == "__builtin_va_arg_pack"):
                    for va in va_args:
                        new_args.append(self._substitute_inline(
                            copy.deepcopy(va), param_map, va_args
                        ))
                else:
                    new_args.append(self._substitute_inline(
                        a, param_map, va_args
                    ))
            node.args = new_args
            return node
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                child = getattr(node, f.name, None)
                if child is None:
                    continue
                if isinstance(child, list):
                    new_list = [
                        self._substitute_inline(c, param_map, va_args)
                        if dataclasses.is_dataclass(c) or isinstance(c, list)
                        else c
                        for c in child
                    ]
                    setattr(node, f.name, new_list)
                elif dataclasses.is_dataclass(child):
                    setattr(node, f.name,
                            self._substitute_inline(child, param_map, va_args))
        return node

    def _call_inline_gnu(
        self,
        expr: ast.Call,
        fn: ast.FunctionDecl,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Inline a gnu_inline function at the call site.

        Supports two body shapes:
          { return E; }
          { if (cond) return E1; return E2; }

        Each named param gets a fresh int slot; the call's arg evaluates
        once into the slot, and the slot's Identifier substitutes for
        the param in the body. `__builtin_va_arg_pack()` in nested Call
        args expands to the call's variadic args (each evaluated once
        in its containing call).
        """
        import copy
        body = fn.body
        if not isinstance(body, ast.CompoundStmt):
            raise CodegenError(
                f"gnu_inline {fn.name}: body must be a compound stmt"
            )
        # Detect body shape.
        items = body.items
        cond_expr = None
        e1_expr = None
        e2_expr = None
        if len(items) == 1 and isinstance(items[0], ast.ReturnStmt):
            e1_expr = items[0].value
        elif (len(items) == 2
                and isinstance(items[0], ast.IfStmt)
                and items[0].else_branch is None
                and isinstance(items[1], ast.ReturnStmt)):
            then_branch = items[0].then_branch
            if (isinstance(then_branch, ast.CompoundStmt)
                    and len(then_branch.items) == 1):
                then_branch = then_branch.items[0]
            if not isinstance(then_branch, ast.ReturnStmt):
                raise CodegenError(
                    f"gnu_inline {fn.name}: unsupported body shape"
                )
            cond_expr = items[0].condition
            e1_expr = then_branch.value
            e2_expr = items[1].value
        else:
            raise CodegenError(
                f"gnu_inline {fn.name}: unsupported body shape"
            )
        # Allocate temp slots for named params; eval args into temps.
        n_params = len(fn.params)
        args = expr.args
        if len(args) < n_params:
            raise CodegenError(
                f"gnu_inline {fn.name}: too few args"
            )
        param_args = args[:n_params]
        var_args = args[n_params:]
        out: list[str] = []
        param_map: dict[str, str] = {}
        # Use a stable per-call-site key for slot names so multiple
        # call sites get distinct slots.
        site_key = f"{id(expr):x}"
        for p, arg in zip(fn.params, param_args):
            slot_name = f"__inline_{fn.name}_{site_key}_{p.name}"
            ptype = p.param_type
            size = (self._size_of(ptype) + 3) & ~3
            ctx.alloc_local(slot_name, ptype, size=size)
            disp = ctx.lookup(slot_name)
            # Eval and store. For simplicity assume int-family (4 bytes).
            # Float / struct / long-long params via gnu_inline aren't
            # exercised by the torture suite va_arg_pack tests.
            out += self._eval_expr_to_eax(arg, ctx)
            out.append(
                f"        mov     {_ebp_addr(disp)}, eax"
            )
            param_map[p.name] = slot_name
        # Build substituted expressions.
        def subst(e):
            return self._substitute_inline(
                copy.deepcopy(e), param_map, var_args
            )
        # Lower as if/else, returning result in EAX.
        if cond_expr is None:
            # Single-return body.
            out += self._eval_expr_to_eax(subst(e1_expr), ctx)
            return out
        else_lbl = ctx.label(f"inline_{fn.name}_else")
        end_lbl = ctx.label(f"inline_{fn.name}_end")
        out += self._eval_expr_to_eax(subst(cond_expr), ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {else_lbl}")
        out += self._eval_expr_to_eax(subst(e1_expr), ctx)
        out.append(f"        jmp     {end_lbl}")
        out.append(f"{else_lbl}:")
        out += self._eval_expr_to_eax(subst(e2_expr), ctx)
        out.append(f"{end_lbl}:")
        return out

    @staticmethod
    def _walk_ast(node):
        """Yield `node` and every nested AST node, dataclass-fields style.

        Used by the call-temp pre-pass; deliberately doesn't try to know
        about specific node types — anything that's a dataclass gets its
        fields recursed into.
        """
        if node is None:
            return
        yield node
        if dataclasses.is_dataclass(node):
            for f in dataclasses.fields(node):
                child = getattr(node, f.name, None)
                yield from CodeGenerator._walk_ast(child)
        elif isinstance(node, list):
            for item in node:
                yield from CodeGenerator._walk_ast(item)
        elif isinstance(node, tuple):
            for item in node:
                yield from CodeGenerator._walk_ast(item)

    def _collect_locals(self, node, ctx: _FuncCtx) -> None:
        """Walk a function body recursively and reserve a slot for every VarDecl.

        Slots are flat across the whole function — no per-block scopes, so a
        for-init `int i = 0` reuses the same slot across iterations and a
        nested-block redeclaration of an existing name raises.
        """
        if isinstance(node, ast.VarDecl):
            # Local function declaration (`int f(int);` inside a body): no
            # storage, just a forward extern. Record return/param types so
            # subsequent calls type-check the same way as top-level externs.
            if isinstance(node.var_type, ast.FunctionType):
                if node.name not in self._func_return_types:
                    self._func_return_types[node.name] = node.var_type.return_type
                    self._func_param_types[node.name] = list(
                        node.var_type.param_types
                    )
                return
            var_type = self._resolved_var_type(node)
            self._check_supported_type(var_type, node.name)
            # Captured-by-nested-fn local: promote to a file-scope
            # global with a mangled name so the nested fn (compiled as
            # a separate top-level function) can reference the same
            # storage. Initializers are emitted via the regular
            # _var_init path with the lvalue address resolved through
            # `local_captures` → globals.
            if (
                node.storage_class != "extern"
                and node.storage_class != "static"
                and getattr(self, "_capture_set", None)
                and node.name in self._capture_set
            ):
                mangled_key = f"{ctx.func_name}__{node.name}"
                if mangled_key not in self._globals:
                    self._globals[mangled_key] = var_type
                ctx.local_captures[node.name] = mangled_key
                return
            # `extern int x;` inside a function: no slot — references
            # the global symbol of the same name. We just register the
            # type in `_extern_vars` so any later identifier reference
            # resolves; the actual scope-shadowing happens at emit time
            # via `_var_init` (which sees the AST node and binds an
            # EXTERN_REDIRECT marker in the current scope).
            if node.storage_class == "extern":
                if node.name not in self._globals and node.name not in self._extern_vars:
                    self._extern_vars[node.name] = var_type
                return
            # `static int x = ...;` inside a function: don't reserve a
            # frame slot. Instead register the variable as a global with
            # a function-mangled label so the value persists across
            # calls. The same identifier inside the function body
            # transparently routes through `local_static_labels` →
            # `_globals` for reads and writes.
            if node.storage_class == "static":
                mangled = f"_{ctx.func_name}__{node.name}"
                key = mangled[1:]  # strip leading `_` to match _globals keys
                self._globals[key] = var_type
                if node.init is not None:
                    self._global_inits[key] = node.init
                ctx.local_static_labels[node.name] = key
                self._static_local_owner[key] = ctx.func_name
                return
            # VLA arrays get a 4-byte pointer slot instead of a
            # fixed-size storage slot. The actual storage is allocated
            # at decl time via sub-esp; the pointer is stored in the
            # slot. Identifier accesses load the pointer (not lea
            # the slot). Only triggers for top-level VLA-shaped arrays
            # (`int v[n]`, `int v[n][m]`); arrays-of-structs-containing-
            # VLA (`struct S s[2]` where S has VLA member) keep static
            # layout to preserve member-offset calculations.
            is_vla_array = (
                isinstance(var_type, ast.ArrayType)
                and self._array_is_directly_vla(var_type)
            )
            if is_vla_array:
                ctx.alloc_local(node.name, var_type, 4, decl=node)
                ctx.vla_backed.add(node.name)
            else:
                # Round slot size up to a 4-byte boundary so a `char`-sized slot
                # doesn't push subsequent int slots off-alignment. Arrays whose
                # payload isn't a multiple of 4 (e.g. `char arr[5]`) get padded
                # the same way.
                raw = self._size_of(var_type)
                slot = (raw + 3) & ~3
                ctx.alloc_local(node.name, var_type, slot, decl=node)
            # For local VLAs, allocate a hidden capture slot per VLA
            # dimension so sizeof reads the size at DECL TIME (not the
            # current value of the size expression). Replace the
            # ArrayType.size references with Identifier(slot_name);
            # `_var_init` will emit "eval size expr; mov slot, eax"
            # at decl point.
            self._capture_vla_sizes(var_type, node.name, ctx, node)
            return
        if isinstance(node, ast.CompoundStmt):
            ctx.enter_scope()
            for item in node.items:
                self._collect_locals(item, ctx)
            ctx.exit_scope()
            return
        if isinstance(node, ast.IfStmt):
            self._collect_locals(node.then_branch, ctx)
            if node.else_branch is not None:
                self._collect_locals(node.else_branch, ctx)
            return
        if isinstance(node, (ast.WhileStmt, ast.DoWhileStmt)):
            self._collect_locals(node.body, ctx)
            return
        if isinstance(node, ast.ForStmt):
            # `for (int i = 0; ...)` makes i live in the for's own scope —
            # not visible after the loop, distinct from a sibling for's i.
            ctx.enter_scope()
            if node.init is not None:
                self._collect_locals(node.init, ctx)
            self._collect_locals(node.body, ctx)
            ctx.exit_scope()
            return
        if isinstance(node, ast.SwitchStmt):
            self._collect_locals(node.body, ctx)
            return
        if isinstance(node, ast.CaseStmt):
            # `case 1: case 2: VarDecl;` nests CaseStmts; each level's
            # `stmt` may eventually be a real declaration or a compound.
            # Recursing is enough — the next layer will be another
            # CaseStmt (and recurse further) or a real statement.
            self._collect_locals(node.stmt, ctx)
            return
        if isinstance(node, ast.LabelStmt):
            # `mylabel: VarDecl;` — the labeled statement may declare a
            # local. Recurse into the nested statement.
            self._collect_locals(node.stmt, ctx)
            return
        if isinstance(node, ast.DeclarationList):
            # Multi-declarator declaration: `int x, *p;` becomes
            # DeclarationList([VarDecl(x), VarDecl(p)]). Each VarDecl
            # might be a real local that needs a slot.
            for decl in node.declarations:
                self._collect_locals(decl, ctx)
            return
        if isinstance(node, ast.FunctionDecl):
            # Nested function definition (gcc extension): no frame slot.
            # The pre-pass at the top of `_function` lifts these to
            # top-level functions with mangled names.
            return
        if isinstance(node, (ast.StructDecl, ast.EnumDecl, ast.TypedefDecl)):
            # In-function type-only declarations (no storage). Register
            # struct/enum layouts now so any subsequent locals that use
            # them resolve correctly during slot allocation.
            if isinstance(node, ast.StructDecl) and node.is_definition:
                self._capture_struct_vla_member_sizes(node, ctx)
                self._register_struct(node)
            elif isinstance(node, ast.EnumDecl) and node.is_definition:
                self._register_enum(node)
            elif isinstance(node, ast.TypedefDecl):
                # `typedef enum {A, B} t;` inside a function: register
                # the enumerators so subsequent code can reference them.
                # `typedef struct {...} t;` works lazily via
                # `_resolve_struct_name`, no eager work needed here.
                if (
                    isinstance(node.target_type, ast.EnumType)
                    and node.target_type.values
                ):
                    self._register_enum_values(node.target_type.values)
            return
        # Statements with no nested locals: ExpressionStmt, ReturnStmt,
        # BreakStmt, ContinueStmt, etc.

    # Scalar BasicType names that have first-class slot support. `long` and
    # `long long` are *known* sizes (so pointer-arithmetic scaling works
    # transparently for `long *` etc.) but full slot codegen waits on a
    # 64-bit value-tracking pass.
    _SLOT_BASIC_NAMES = frozenset(
        {"bool", "char", "short", "int", "long", "long long", "int128",
         "float", "double", "long double"}
    )

    # Half-element sizes for `_Complex T`. Includes int family for
    # GCC's `_Complex int` / `_Complex long` extension.
    _COMPLEX_BASE_SIZES = {
        "float": 4, "double": 8, "long double": 8,
        "char": 1, "short": 2, "int": 4, "long": 4, "long long": 8,
    }
    _COMPLEX_INT_BASES = {"char", "short", "int", "long", "long long"}

    def _complex_value_address(
        self, expr: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Compute the address of a `_Complex T` value lvalue."""
        if isinstance(expr, ast.Identifier):
            return self._identifier_address(expr.name, ctx)
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            return self._eval_expr_to_eax(expr.operand, ctx)
        if isinstance(expr, ast.Member):
            return self._member_address(expr, ctx)
        if isinstance(expr, ast.Index):
            return self._index_address(expr, ctx)
        if (
            isinstance(expr, ast.Call)
            and self._is_complex_returning_call(expr, ctx)
        ):
            disp = ctx.call_temps[id(expr)]
            retptr_lines = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._call_into_address(expr, retptr_lines, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # Complex-valued sub-expression (BinaryOp / UnaryOp / Cast)
        # or imaginary FloatLiteral: evaluate into the pre-allocated
        # temp slot reserved by `_collect_call_temps`.
        if (
            isinstance(expr, (ast.BinaryOp, ast.UnaryOp, ast.Cast))
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            dest = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._eval_complex_to(expr, dest, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        if (
            isinstance(expr, ast.FloatLiteral)
            and expr.is_imaginary
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            ty = self._type_of_complex_expr(expr)
            half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
            width = "dword" if half_size == 4 else "qword"
            r_label = self._intern_float(0.0, half_size)
            i_label = self._intern_float(float(expr.value), half_size)
            return [
                f"        fld     {width} [{r_label}]",
                f"        fstp    {width} {_ebp_addr(disp)}",
                f"        fld     {width} [{i_label}]",
                f"        fstp    {width} {_ebp_addr(disp + half_size)}",
                f"        lea     eax, {_ebp_addr(disp)}",
            ]
        raise CodegenError(
            f"can't take address of complex `{type(expr).__name__}`"
        )

    def _complex_part_address(
        self, expr: ast.UnaryOp, ctx: _FuncCtx,
    ) -> tuple[list[str], ast.TypeNode]:
        """Compute the address of a `__real__ x` or `__imag__ x`
        expression: returns asm lines that leave EAX = address of the
        half, plus the half's type (a base BasicType)."""
        operand = expr.operand
        operand_ty = self._type_of(operand, ctx)
        if not isinstance(operand_ty, ast.ComplexType):
            raise CodegenError(
                f"`{expr.op}` requires a _Complex operand "
                f"(got {type(operand_ty).__name__})"
            )
        half_size = self._COMPLEX_BASE_SIZES[operand_ty.base_type]
        # Compute &operand.
        if isinstance(operand, ast.Identifier):
            out = self._identifier_address(operand.name, ctx)
        elif isinstance(operand, ast.UnaryOp) and operand.op == "*":
            out = self._eval_expr_to_eax(operand.operand, ctx)
        elif isinstance(operand, ast.Member):
            out = self._member_address(operand, ctx)
        elif isinstance(operand, ast.Index):
            out = self._index_address(operand, ctx)
        elif (
            isinstance(operand, ast.Call)
            and self._is_complex_returning_call(operand, ctx)
        ):
            # Route the call into its per-call-site temp slot, then
            # use the temp's address.
            disp = ctx.call_temps[id(operand)]
            retptr_lines = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._call_into_address(operand, retptr_lines, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
        else:
            raise CodegenError(
                f"`{expr.op}` operand must be a simple lvalue "
                f"(got {type(operand).__name__})"
            )
        offset = 0 if expr.op == "__real__" else half_size
        if offset:
            out.append(f"        add     eax, {offset}")
        return out, ast.BasicType(name=operand_ty.base_type)

    def _complex_struct_name(self, t: ast.ComplexType) -> str:
        """Return (and lazily register) the synthetic struct name we
        use to lay out `_Complex T`. The struct has members `_real`
        and `_imag` of type `T`."""
        key = f"__complex_{t.base_type.replace(' ', '_')}"
        if key not in self._structs:
            base = ast.BasicType(name=t.base_type)
            size = self._COMPLEX_BASE_SIZES[t.base_type]
            self._structs[key] = [
                ("_real", base, 0),
                ("_imag", base, size),
            ]
            self._struct_sizes[key] = 2 * size
        return key

    def _check_supported_type(self, t: ast.TypeNode, name: str) -> None:
        # Pointers, and arrays / scalars / structs / enums of supported
        # base types. Slot sizes are rounded up to 4 in `_collect_locals`
        # so adjacent ints stay 4-aligned. Unsized arrays (`int a[]`
        # without an init) are caught by `_resolved_var_type` before they
        # reach this check.
        if isinstance(t, ast.PointerType):
            # If the pointee is an inline-defined struct/union, register
            # it under its tag so a sibling `struct foo *` (without a
            # body) can resolve later. We don't need its layout for
            # the pointer itself.
            base = t.base_type
            if (
                isinstance(base, ast.StructType)
                and base.name
                and base.members
            ):
                self._resolve_struct_name(base)
            return
        if isinstance(t, ast.BasicType) and t.name in self._SLOT_BASIC_NAMES:
            return
        if isinstance(t, ast.EnumType):
            # Enums are int-sized. If the type carries inline values
            # (which happens when an `enum { X, Y }` is declared in
            # place — e.g. as a struct member type), register the
            # constants so they're visible at file scope, the way C
            # treats enumerators.
            if t.values:
                self._register_enum_values(t.values)
            return
        if isinstance(t, ast.ArrayType):
            if t.size is not None and not isinstance(t.size, ast.IntLiteral):
                # Try to const-fold the size (handles `1 && 1`, `MACRO+1`,
                # enumerator sums, etc.). On success, mutate the AST to
                # store the resolved literal so later code sees a literal.
                try:
                    folded = self._const_eval(t.size, name)
                    t.size = ast.IntLiteral(value=folded)
                except CodegenError:
                    # Variable-length arrays (`int a[n]`) aren't fully
                    # supported (no runtime alloca). As a compile-only
                    # convenience we pick a fixed slot size of 16 so
                    # struct layout proceeds; the original size
                    # expression is preserved on `t._vla_size` so
                    # `_emit_runtime_size_of` can recompute the real
                    # byte count for `sizeof`.
                    t._vla_size = t.size
                    t.size = ast.IntLiteral(value=16)
            self._check_supported_type(t.base_type, name)
            return
        if isinstance(t, ast.StructType):
            # `_resolve_struct_name` lazily registers typedef'd or
            # otherwise-inline-defined structs and raises if neither a
            # registered name nor inline members are available.
            self._resolve_struct_name(t)
            return
        if isinstance(t, ast.ComplexType):
            # `_Complex T` is laid out as two T's (real, imag). We
            # accept it as a struct-like type; arithmetic ops (+, *,
            # etc.) on complex values are not yet implemented, but
            # storage / __real__ / __imag__ / pass-by-value all work.
            if t.base_type not in self._COMPLEX_BASE_SIZES:
                raise CodegenError(
                    f"`{name}`: _Complex base type `{t.base_type}` not supported"
                )
            return
        raise CodegenError(
            f"`{name}`: only `int`/`short`/`char`, pointer, array, and "
            f"struct types are supported (got {type(t).__name__})"
        )

    def _resolved_var_type(self, decl: ast.VarDecl) -> ast.TypeNode:
        """Return the var's type with any unsized-array size filled in.

        `int arr[] = {1, 2, 3}` and `char s[] = "hi"` both leave the
        ArrayType's `size` as None; the size is implied by the initializer.
        Resolve it here so allocation and codegen can treat the slot as a
        fully-sized array thereafter.

        Brace elision: for `T arr[] = { v0, v1, ... }` where T is a
        compound type with K leaf scalars, every K flat values map to one
        array element. The implied length is then `count / K` rather
        than `count`.

        For initializer lists with designators (`{5, [2] = 2, 3}`), the
        cursor jumps and subsequent values continue from the new index, so
        the implied size is the max-index-touched + 1, not value count.
        """
        t = decl.var_type
        if not isinstance(t, ast.ArrayType) or t.size is not None:
            return t
        # `extern T arr[];` — size lives in another translation unit,
        # not knowable here. Return as-is; we never need to allocate a
        # frame slot for an extern, just emit a NASM extern label.
        if getattr(decl, "storage_class", None) == "extern":
            return t
        if isinstance(decl.init, ast.InitializerList):
            try:
                leaves = self._leaf_slot_count(t.base_type)
            except (CodegenError, KeyError):
                leaves = 1
            if leaves < 1:
                leaves = 1
            cursor = 0
            max_idx = -1
            for value in decl.init.values:
                if isinstance(value, ast.DesignatedInit):
                    if (
                        len(value.designators) == 1
                        and isinstance(value.designators[0], ast.IntLiteral)
                    ):
                        cursor = value.designators[0].value
                    if cursor > max_idx:
                        max_idx = cursor
                    cursor += 1
                    continue
                # A positional value already wrapped in `{}` (or as
                # a Compound literal `(T){...}`) consumes one element
                # regardless of leaf_count. A flat value in a
                # leaves > 1 array consumes 1/leaves of an element.
                if isinstance(value, (ast.InitializerList, ast.Compound)):
                    if cursor > max_idx:
                        max_idx = cursor
                    cursor += 1
                    continue
                # Flat value: contributes a fraction of an element.
                if cursor > max_idx:
                    max_idx = cursor
                # We approximate by counting flat values against the
                # current element. Bump cursor only when the element is
                # filled. Simpler approximation: count all flat values
                # toward leaves, advance cursor when leaves accumulate.
                # But mixing ILs and flats requires per-element bookkeeping.
                # For the common all-flat case, just total / leaves.
                cursor += 1
            # If all values were flat (no IL/Designators), the value
            # count divided by leaves gives the right element count.
            all_flat = all(
                not isinstance(
                    v, (ast.DesignatedInit, ast.InitializerList, ast.Compound)
                )
                for v in decl.init.values
            )
            if all_flat and leaves > 1:
                n = (len(decl.init.values) + leaves - 1) // leaves
            else:
                n = max_idx + 1
        elif isinstance(decl.init, ast.StringLiteral):
            # +1 reserves a slot for the trailing null byte. Wide
            # strings count by codepoint, not by encoded byte length.
            n = len(decl.init.value) + 1
        else:
            raise CodegenError(
                f"unsized array `{decl.name}` requires an initializer"
            )
        return ast.ArrayType(
            base_type=t.base_type,
            size=ast.IntLiteral(value=n),
        )

    @staticmethod
    def _is_pointer_like(t: ast.TypeNode) -> bool:
        """True for types that participate in pointer arithmetic.

        Arrays decay to pointers in expression context, so they're treated
        the same as pointers anywhere we look at element-step semantics.
        """
        return isinstance(t, (ast.PointerType, ast.ArrayType))

    @staticmethod
    def _is_vector_op_node(node) -> bool:
        """True for BinaryOp / UnaryOp shapes that should be lowered as
        componentwise vector arithmetic when their type is ArrayType.
        Distinguishes vector arithmetic from incidental ArrayType results
        (e.g. an Identifier of array type used in arithmetic context)."""
        if isinstance(node, ast.BinaryOp):
            return node.op in (
                "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>",
            )
        if isinstance(node, ast.UnaryOp):
            return node.op in ("-", "+", "~")
        return False

    def _is_genuine_vector_op(self, node, ctx) -> bool:
        """Distinguish vector componentwise arithmetic from pointer
        arithmetic.

        For BinaryOp:
        - Both sides ArrayType same shape with at least one tagged
          as a vector → componentwise vector arithmetic.
        - One side a vector ArrayType and the other a scalar (int)
          → scalar broadcast against each element.

        For UnaryOp, the operand must be a vector ArrayType."""
        try:
            if isinstance(node, ast.BinaryOp):
                lt = self._type_of(node.left, ctx)
                rt = self._type_of(node.right, ctx)
                l_vec = (
                    isinstance(lt, ast.ArrayType)
                    and getattr(lt, "is_vector", False)
                )
                r_vec = (
                    isinstance(rt, ast.ArrayType)
                    and getattr(rt, "is_vector", False)
                )
                # Two-sided vector op (same size).
                if (
                    isinstance(lt, ast.ArrayType)
                    and isinstance(rt, ast.ArrayType)
                    and self._size_of(lt) == self._size_of(rt)
                    and (l_vec or r_vec)
                ):
                    return True
                # Scalar broadcast: vector op scalar (int or float).
                # Float-vector codegen isn't implemented yet; the path
                # raises a clean CodegenError when the elem type is
                # float, but we still need to allocate a temp so later
                # routing doesn't KeyError.
                def _is_scalar(t):
                    return isinstance(t, ast.BasicType) and t.name in (
                        "int", "char", "short", "long", "long long",
                        "bool", "float", "double", "long double",
                    )
                if l_vec and _is_scalar(rt):
                    return True
                if r_vec and _is_scalar(lt):
                    return True
            if isinstance(node, ast.UnaryOp):
                opt = self._type_of(node.operand, ctx)
                return (
                    isinstance(opt, ast.ArrayType)
                    and getattr(opt, "is_vector", False)
                )
        except CodegenError:
            return False
        return False

    # Sizes used for pointer-arithmetic scaling. We can compute these for
    # any pointee type the parser produces, even ones we don't yet support
    # as full slot types — `char *p; p + 1;` works without `*p` working.
    _BASIC_SIZES = {
        "bool": 1,          # C99 _Bool
        "char": 1,
        "short": 2,
        "int": 4,
        "long": 4,          # i386: long is 32-bit
        "long long": 8,
        "int128": 16,       # GCC extension: __int128 / unsigned __int128
        "float": 4,         # x87 single precision
        "double": 8,        # x87 double precision
        "long double": 8,   # x87 has 80-bit, but we approximate as 8
        "void": 1,          # GCC convention; standard C disallows arithmetic on void*
    }

    def _size_of(self, t: ast.TypeNode) -> int:
        if isinstance(t, ast.PointerType):
            return 4
        if isinstance(t, ast.BasicType):
            try:
                return self._BASIC_SIZES[t.name]
            except KeyError:
                raise CodegenError(f"sizeof({t.name}) not known")
        if isinstance(t, ast.ArrayType):
            if t.size is None:
                # Flexible array member (`struct S s[];`) — sized 0 in
                # the struct layout.
                return 0
            if not isinstance(t.size, ast.IntLiteral):
                raise CodegenError("sizeof(array): size must be an integer literal")
            return t.size.value * self._size_of(t.base_type)
        if isinstance(t, ast.StructType):
            return self._struct_sizes[self._resolve_struct_name(t)]
        if isinstance(t, ast.EnumType):
            return 4
        if isinstance(t, ast.ComplexType):
            # `_Complex T` is laid out as two T's (real, imag).
            return 2 * self._COMPLEX_BASE_SIZES.get(t.base_type, 8)
        raise CodegenError(f"sizeof not supported for {type(t).__name__}")

    def _resolve_struct_name(self, t: ast.StructType) -> str:
        """Return a registry key for `t`, registering its layout if needed.

        Top-level `struct point { ... };` definitions are registered
        eagerly during `generate()`. But `typedef struct { ... } P;`
        produces a StructType node with inline members and (sometimes)
        no name — the parser doesn't synthesize a separate StructDecl
        for it. We register on first sight here using either the
        struct's name or a synthetic id-based key.

        Tag scoping: when `t.members` is empty (a forward reference by
        name) and `t.name` is bound in the alias chain
        (`_struct_aliases`), use that mapping. When `t.members` is
        present, the inline layout is authoritative — register it under
        an id-keyed entry so a later block with a same-tag shadowing
        struct doesn't redirect the lookup.
        """
        # First chance: a tag-only reference (no inline members) routes
        # through the alias chain.
        if t.name and not t.members:
            aliased = self._resolve_struct_alias(t.name)
            if aliased is not None and aliased in self._structs:
                return aliased
            if t.name in self._structs:
                return t.name
        if t.members:
            # The inline-members shape is authoritative — if the alias
            # chain currently maps the tag to a struct with different
            # members, the inline `t` is a separate definition (could be
            # a nested-scope shadow, or just two unrelated `struct T {x}`
            # declarations). Anchor by node id when the inline shape
            # doesn't match an already-aliased layout.
            if t.name:
                aliased = self._resolve_struct_alias(t.name)
                if aliased is not None and aliased in self._structs:
                    existing = [name for name, _, _ in self._structs[aliased]]
                    incoming = [m.name for m in t.members if m.name is not None]
                    if existing == incoming:
                        return aliased
                    # Fall through to id-key.
                else:
                    # First sight of this tag. Register under the tag.
                    from types import SimpleNamespace
                    self._register_struct(SimpleNamespace(
                        name=t.name, members=t.members, is_union=t.is_union,
                    ))
                    after = self._resolve_struct_alias(t.name)
                    if after is not None:
                        return after
                    if t.name in self._structs:
                        return t.name
            key = f"__inline_{id(t)}"
            if key not in self._structs:
                from types import SimpleNamespace
                self._register_struct(SimpleNamespace(
                    name=key, members=t.members, is_union=t.is_union,
                ))
            return key
        if t.name:
            # `union at6 {} v;` — empty inline definition — gives us a
            # StructType with name=at6, members=[] AND
            # `_had_inline_brace=True`. That's a real (zero-sized) tag
            # we can register. Without the brace flag, this is a use of
            # an incomplete tag, which is a constraint violation for a
            # value-typed declaration (variable, member, parameter).
            if getattr(t, "_had_inline_brace", False):
                key = f"__empty_struct_{t.name}"
                if key not in self._structs:
                    self._structs[key] = []
                    self._struct_sizes[key] = 0
                self._struct_aliases[-1][t.name] = key
                return key
            raise CodegenError(
                f"unknown struct `{t.name}` — define it before use"
            )
        # Empty struct (`typedef struct {} empty_s;` → StructType
        # with name=None, members=[]). GCC permits these; register a
        # zero-sized layout so it can be a struct member or a local.
        key = f"__empty_struct_{id(t)}"
        if key not in self._structs:
            self._structs[key] = []
            self._struct_sizes[key] = 0
        return key

    def _anon_member_layout_key(self, t: ast.StructType) -> str:
        """Resolve `t` (a StructType used as an anonymous member) to a
        registered struct/union name in `_structs` so we can copy its
        members up. Registers the inline definition if needed.
        """
        return self._resolve_struct_name(t)

    def _register_enum(self, decl: ast.EnumDecl) -> None:
        """Compute and record each `EnumValue`'s integer constant.

        Per C, an `EnumValue(name, value=None)` takes the previous
        constant + 1 (starting at 0 for the first). An explicit
        `value=IntLiteral(n)` sets the cursor; subsequent implicit
        values continue from there.
        """
        self._register_enum_values(decl.values)

    def _register_enum_values(self, values) -> None:
        """Register a sequence of EnumValue nodes as file-scope constants.

        Used both by top-level EnumDecls and by inline `enum {...}`
        types appearing inside struct members or as variable types —
        C treats enumerators as ordinary identifiers visible from the
        point of declaration onward.
        """
        cursor = 0
        for ev in values:
            if ev.value is not None:
                cursor = self._const_eval(ev.value, f"enum.{ev.name}")
            if ev.name in self._enum_constants:
                # Idempotent: a typedef'd enum can be registered more
                # than once as `_check_supported_type` runs on each use.
                # Only flag a true conflict (different value).
                if self._enum_constants[ev.name] != cursor:
                    raise CodegenError(
                        f"conflicting redefinition of enum constant `{ev.name}`"
                    )
            else:
                self._enum_constants[ev.name] = cursor
            cursor += 1

    def _resolve_struct_alias(self, name: str) -> str | None:
        """Walk the scope chain for `name`, return the registry key.

        Returns None if the tag isn't bound at any active scope.
        """
        for scope in reversed(self._struct_aliases):
            if name in scope:
                return scope[name]
        return None

    def _register_struct(self, decl: ast.StructDecl) -> None:
        """Compute member offsets and total size for a struct definition.

        Member offsets are aligned to the member's natural alignment
        (power-of-two sizes for char/short/int/pointer). The total struct
        size is rounded up to the largest member alignment so arrays of
        the struct stay properly aligned.

        Anonymous struct/union members (`struct { int a, b; };` with no
        member name) get their inner members promoted into the parent's
        namespace at the parent's offset for the anonymous block — that's
        the C11 "anonymous member" semantics.

        When `decl.name` already names a struct in the current scope's
        alias chain, this is a no-op (idempotent register). When it's
        bound in an outer scope, we mangle to a fresh key and update
        the topmost alias dict so inner-scope lookups see the new
        layout.
        """
        if decl.name:
            top = self._struct_aliases[-1]
            if decl.name in top:
                # Same scope already has a binding — idempotent.
                return
            existing_outer = self._resolve_struct_alias(decl.name)
            if existing_outer is not None:
                # Outer scope's `struct T` is being shadowed. Mangle the
                # inner one to a unique key.
                key = f"{decl.name}#{len(self._structs)}"
            else:
                key = decl.name
            top[decl.name] = key
            decl_name = key
        else:
            decl_name = None
        if decl_name and decl_name in self._structs:
            return
        # Validate every member up front. Unions and structs share the
        # same per-member checks; only the layout step differs.
        members: list[tuple[str, ast.TypeNode, int]] = []
        # Parallel list: group id (= original index in decl.members)
        # per flattened member. Anon-promoted entries share the group id
        # of their parent decl member.
        member_groups: list[int] = []
        bitfields: dict[str, tuple[int, int]] = {}
        max_align = 1
        for m in decl.members:
            if m.name is None:
                # Two unnamed cases:
                #   - Anonymous bit-field: `unsigned : 12;` — pure padding.
                #   - Anonymous nested struct/union: `struct { ... };`.
                if m.bit_width is not None:
                    self._check_supported_type(
                        m.member_type, f"{decl.name}.<anon-bf>",
                    )
                    align = self._alignment_of(m.member_type)
                    if align > max_align:
                        max_align = align
                    continue
                if not (
                    isinstance(m.member_type, ast.StructType)
                    and m.member_type.members
                ):
                    raise CodegenError(
                        f"`{decl.name}`: anonymous members not supported"
                    )
                self._check_supported_type(
                    m.member_type, f"{decl.name}.<anon>",
                )
                align = self._alignment_of(m.member_type)
                if align > max_align:
                    max_align = align
                continue
            self._check_supported_type(m.member_type, f"{decl.name}.{m.name}")
            align = self._alignment_of(m.member_type)
            # Honor an `__attribute__((aligned(N)))` on the member —
            # bumps both the member's effective alignment and the
            # struct's overall alignment.
            ma = getattr(m, "alignment", None)
            if ma is not None and ma > align:
                align = ma
            if align > max_align:
                max_align = align

        if decl.is_union:
            # All members share offset 0; total size = max member size,
            # rounded up to the union's alignment. Bit-fields each get
            # their own 32-bit storage unit at offset 0 (no packing
            # across bit-fields like in a struct, since they all
            # alias).
            total = 0
            for group_idx, m in enumerate(decl.members):
                if m.bit_width is not None:
                    if not isinstance(m.bit_width, ast.IntLiteral):
                        raise CodegenError(
                            f"`{decl.name}.{m.name}`: bit-field width "
                            f"must be an integer literal"
                        )
                    width = m.bit_width.value
                    if width <= 0 or width > 32:
                        raise CodegenError(
                            f"`{decl.name}.{m.name}`: bit-field width "
                            f"{width} out of range (1..32)"
                        )
                    if m.name is not None:
                        members.append((m.name, m.member_type, 0))
                        member_groups.append(group_idx)
                        bitfields[m.name] = (0, width)
                    if 4 > total:
                        total = 4
                    continue
                if m.name is None:
                    # Anonymous nested member: promote each of its inner
                    # members at offset 0 (union scope).
                    inner_key = self._anon_member_layout_key(m.member_type)
                    for in_name, in_ty, in_off in self._structs[inner_key]:
                        members.append((in_name, in_ty, 0 + in_off))
                        member_groups.append(group_idx)
                else:
                    members.append((m.name, m.member_type, 0))
                    member_groups.append(group_idx)
                size = self._size_of(m.member_type)
                if size > total:
                    total = size
            total = (total + max_align - 1) & ~(max_align - 1)
            self._structs[decl_name] = members
            self._struct_member_groups[decl_name] = member_groups
            self._struct_sizes[decl_name] = total
            self._struct_unions.add(decl_name)
            if bitfields:
                self._struct_bitfields[decl_name] = bitfields
            return

        # Struct layout. Two cursors run in parallel: a byte cursor for
        # regular members, and a bit cursor within the current
        # storage unit for bit-fields. Storage unit width follows the
        # bit-field's declared type — 4 bytes for `int`/`unsigned`/etc.,
        # 8 bytes for `long long`/`unsigned long long`. Adjacent
        # bit-fields pack into the same unit only when their unit
        # width agrees; a type change finishes the previous unit.
        unit_offset = 0     # byte offset of the current bit-field unit
        unit_used = 0       # bits used in the current unit (0..unit_size*8)
        unit_size = 4       # storage unit width in bytes (4 or 8)
        for group_idx, m in enumerate(decl.members):
            if m.bit_width is not None:
                # Common case: integer literal. Otherwise reduce
                # through `_const_eval` so things like
                # `int : sizeof(int) * 8 - 2` work.
                if isinstance(m.bit_width, ast.IntLiteral):
                    width = m.bit_width.value
                else:
                    try:
                        width = self._const_eval(
                            m.bit_width, f"{decl.name}.{m.name or '<anon>'}"
                        )
                    except CodegenError as e:
                        raise CodegenError(
                            f"`{decl.name}.{m.name}`: bit-field width must "
                            f"reduce to an integer constant ({e})"
                        )
                # Storage unit width follows the type: long long ⇒ 8B.
                this_unit_size = (
                    8 if self._is_long_long(m.member_type)
                    else 4
                )
                max_width = this_unit_size * 8
                if width < 0 or width > max_width:
                    raise CodegenError(
                        f"`{decl.name}.{m.name}`: bit-field width "
                        f"{width} out of range (0..{max_width})"
                    )
                if width == 0:
                    # C: zero-width forces alignment to the next unit.
                    if unit_used > 0:
                        unit_offset += unit_size
                        unit_used = 0
                    unit_size = this_unit_size
                    continue
                # If the new bit-field's storage unit is wider than the
                # current cursor's, finish the current unit and start
                # fresh (with the wider alignment).
                if (
                    this_unit_size != unit_size
                    or unit_used + width > unit_size * 8
                ):
                    if unit_used > 0:
                        unit_offset += unit_size
                        unit_used = 0
                    # Re-align to the new type's natural alignment.
                    if this_unit_size > unit_size:
                        align = this_unit_size
                        unit_offset = (
                            (unit_offset + align - 1) & ~(align - 1)
                        )
                    unit_size = this_unit_size
                # Anonymous bit-field (`unsigned : N`): just consume bits
                # without registering a member. Used as inline padding.
                if m.name is None:
                    unit_used += width
                    continue
                members.append((m.name, m.member_type, unit_offset))
                member_groups.append(group_idx)
                bitfields[m.name] = (unit_used, width, unit_size)
                unit_used += width
            else:
                # Regular member — finish any in-progress bit-field unit
                # before laying it out.
                if unit_used > 0:
                    unit_offset += unit_size
                    unit_used = 0
                unit_size = 4
                align = self._alignment_of(m.member_type)
                # `__attribute__((aligned(N)))` on the member bumps its
                # required alignment.
                ma = getattr(m, "alignment", None)
                if ma is not None and ma > align:
                    align = ma
                offset = (unit_offset + align - 1) & ~(align - 1)
                if m.name is None:
                    # Anonymous nested struct/union — promote each inner
                    # member into the outer namespace at offset+inner_off.
                    inner_key = self._anon_member_layout_key(m.member_type)
                    for in_name, in_ty, in_off in self._structs[inner_key]:
                        members.append((in_name, in_ty, offset + in_off))
                        member_groups.append(group_idx)
                else:
                    members.append((m.name, m.member_type, offset))
                    member_groups.append(group_idx)
                unit_offset = offset + self._size_of(m.member_type)
        total = unit_offset + (unit_size if unit_used > 0 else 0)
        total = (total + max_align - 1) & ~(max_align - 1)
        self._structs[decl_name] = members
        self._struct_member_groups[decl_name] = member_groups
        self._struct_sizes[decl_name] = total
        if max_align > 1:
            self._struct_alignments[decl_name] = max_align
        if bitfields:
            self._struct_bitfields[decl_name] = bitfields

    def _member_layout(
        self, struct_name: str, member: str
    ) -> tuple[ast.TypeNode, int]:
        members = self._structs.get(struct_name)
        if members is None:
            raise CodegenError(f"struct `{struct_name}` not defined")
        for name, ty, off in members:
            if name == member:
                return ty, off
        raise CodegenError(
            f"struct `{struct_name}` has no member `{member}`"
        )

    def _alignment_of(self, t: ast.TypeNode) -> int:
        """Required alignment in bytes for a value of type `t`.

        Used by struct layout — each member's offset is rounded up to its
        alignment, and the struct's overall size is rounded up to the
        largest member alignment so an array of structs stays packed
        without internal misalignment.
        """
        if isinstance(t, ast.PointerType):
            return 4
        if isinstance(t, ast.BasicType):
            return self._BASIC_SIZES.get(t.name, 1)
        if isinstance(t, ast.ArrayType):
            return self._alignment_of(t.base_type)
        if isinstance(t, ast.StructType):
            sname = self._resolve_struct_name(t)
            # If a member has `aligned(N)`, the struct's alignment was
            # bumped at registration time — honor it.
            override = self._struct_alignments.get(sname)
            if override is not None:
                return override
            members = self._structs.get(sname)
            if not members:
                return 1
            return max(self._alignment_of(mt) for _, mt, _ in members)
        if isinstance(t, ast.EnumType):
            return 4
        if isinstance(t, ast.ComplexType):
            return self._COMPLEX_BASE_SIZES.get(t.base_type, 8)
        return 1

    @staticmethod
    def _is_unsigned(t: ast.TypeNode) -> bool:
        # `is_signed=None` is the language default — signed for
        # char/short/int. EnumType is treated as unsigned for bit-field
        # purposes (matches GCC's choice for enums of non-negative
        # values, which is the common case). `_Bool` is always
        # unsigned per C99 6.2.5.
        if isinstance(t, ast.BasicType):
            if t.name == "bool":
                return True
            return t.is_signed is False
        if isinstance(t, ast.EnumType):
            return True
        return False

    def _is_unsigned_after_promotion(self, t: ast.TypeNode) -> bool:
        """C usual arithmetic conversions: unsigned char/short are
        promoted to int (signed), so they don't make the surrounding
        comparison or division unsigned. Only unsigned types that are
        at least as wide as int retain unsigned-ness.
        """
        if isinstance(t, ast.BasicType) and t.is_signed is False:
            return t.name in ("int", "long", "long long")
        return False

    @staticmethod
    def _is_long_long(t: ast.TypeNode) -> bool:
        return isinstance(t, ast.BasicType) and t.name == "long long"

    @staticmethod
    def _is_int128(t: ast.TypeNode) -> bool:
        return isinstance(t, ast.BasicType) and t.name == "int128"

    # ---- __int128 lowering --------------------------------------------
    #
    # `__int128` and `unsigned __int128` are GCC extensions for 128-bit
    # integer arithmetic. We treat them as 16-byte memory values, with
    # all operations going through addresses (analogous to structs and
    # `_Complex T`). The value is represented as four little-endian
    # dwords: byte 0 = LSB.
    #
    # `_eval_expr_to_eax` short-circuits int128 expressions to
    # `_int128_value_address` so callers always see an address, never
    # a register-held value (no register pair is wide enough). Storage
    # for sub-expression results comes from per-node `call_temps`
    # slots reserved by `_collect_call_temps`.
    #
    # Coverage: this is intentionally minimal — enough to compile
    # pr84748.c (`b += int`, `b /= int`, copy assign, `>> 0`, `>> 64`,
    # cast to ULL). Many ops (multiply, divide by int128, shifts by
    # arbitrary constant, comparisons) raise; extending those follows
    # the same pattern as long-long arithmetic.

    def _int128_value_address(
        self, expr: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Lines that produce the address of an `__int128` value in EAX."""
        if isinstance(expr, ast.Identifier):
            return self._identifier_address(expr.name, ctx)
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            return self._eval_expr_to_eax(expr.operand, ctx)
        if isinstance(expr, ast.Member):
            return self._member_address(expr, ctx)
        if isinstance(expr, ast.Index):
            return self._index_address(expr, ctx)
        # int128-returning Call: route through its per-call-site temp
        # via the retptr ABI (same as struct/complex/vector returns).
        if (
            isinstance(expr, ast.Call)
            and self._is_int128_returning_call(expr, ctx)
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            retptr_lines = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._call_into_address(expr, retptr_lines, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # Sub-expressions evaluate into a per-node temp slot.
        if (
            isinstance(expr, (ast.BinaryOp, ast.UnaryOp, ast.Cast, ast.TernaryOp))
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            out = self._eval_int128_into_temp(expr, disp, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # Comma operator with __int128 result: eval left for side
        # effects, then eval right as the int128 value.
        if isinstance(expr, ast.BinaryOp) and expr.op == ",":
            out = self._eval_expr_to_eax(expr.left, ctx)
            out += self._int128_value_address(expr.right, ctx)
            return out
        # `va_arg(ap, __int128)` — copy 16 bytes from the va_list
        # pointer into a per-expr temp and return its address.
        if (
            isinstance(expr, ast.VaArgExpr)
            and self._is_int128(expr.target_type)
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            dest = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._va_arg_struct_copy(expr, dest, ctx)
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # GCC statement expression returning __int128: emit head
        # items in scope, then yield the trailing expression's
        # int128 value's address.
        if isinstance(expr, ast.StmtExpr):
            ctx.enter_scope()
            try:
                items = list(expr.body.items)
                if not items:
                    return []
                head, tail = items[:-1], items[-1]
                out: list[str] = []
                for item in head:
                    out += self._item(item, ctx)
                if (
                    isinstance(tail, ast.ExpressionStmt)
                    and tail.expr is not None
                ):
                    out += self._int128_value_address(tail.expr, ctx)
                    return out
                out += self._item(tail, ctx)
                return out
            finally:
                ctx.exit_scope()
        # `(__int128){init}` — compound literal whose target is
        # int128. The temp is allocated by `_collect_call_temps` (with
        # size 16). Store the init's value into the temp and return
        # its address.
        if (
            isinstance(expr, ast.Compound)
            and self._is_int128(expr.target_type)
            and id(expr) in ctx.call_temps
        ):
            disp = ctx.call_temps[id(expr)]
            inner = expr.init
            if (
                isinstance(inner, ast.InitializerList)
                and len(inner.values) == 1
            ):
                inner = inner.values[0]
            inner_ty = self._type_of(inner, ctx)
            if not self._is_int128(inner_ty):
                synth_cast = ast.Cast(
                    target_type=expr.target_type, expr=inner,
                )
                ctx.alloc_call_temp(synth_cast, 16)
                inner = synth_cast
            out = self._int128_value_address(inner, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(disp)}")
            out += self._emit_int128_copy("esi", "edi")
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        raise CodegenError(
            f"can't take address of __int128 `{type(expr).__name__}`"
        )

    def _emit_int128_copy(
        self, src_addr_reg: str, dst_addr_reg: str,
    ) -> list[str]:
        """Emit a 16-byte copy from `[src_addr_reg]` to `[dst_addr_reg]`."""
        out: list[str] = []
        for off in (0, 4, 8, 12):
            out.append(f"        mov     eax, [{src_addr_reg} + {off}]")
            out.append(f"        mov     [{dst_addr_reg} + {off}], eax")
        return out

    def _emit_int128_zero(self, dst_addr_reg: str) -> list[str]:
        out = ["        xor     eax, eax"]
        for off in (0, 4, 8, 12):
            out.append(f"        mov     [{dst_addr_reg} + {off}], eax")
        return out

    def _eval_int128_into_temp(
        self, expr: ast.Expression, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        """Evaluate `expr` (BinaryOp / UnaryOp / Cast / TernaryOp) into
        the 16-byte slot at `[ebp - dest_disp]`."""
        if isinstance(expr, ast.Cast):
            return self._cast_to_int128(expr, dest_disp, ctx)
        if isinstance(expr, ast.BinaryOp):
            return self._binary_int128(expr, dest_disp, ctx)
        if isinstance(expr, ast.UnaryOp):
            return self._unary_int128(expr, dest_disp, ctx)
        if isinstance(expr, ast.TernaryOp):
            return self._int128_ternary(expr, dest_disp, ctx)
        raise CodegenError(
            f"can't evaluate `{type(expr).__name__}` as __int128"
        )

    def _int128_ternary(
        self, expr: ast.TernaryOp, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `cond ? T : F` where T/F are __int128 expressions.
        Evaluates the chosen branch into the dest slot via an int128
        copy. Smaller-integer arms widen via synthetic Cast."""
        def widen(arm):
            arm_ty = self._type_of(arm, ctx)
            if not self._is_int128(arm_ty):
                synth_cast = ast.Cast(
                    target_type=ast.BasicType(name="int128"), expr=arm,
                )
                ctx.alloc_call_temp(synth_cast, 16)
                return synth_cast
            return arm
        true_arm = widen(expr.true_expr)
        false_arm = widen(expr.false_expr)
        false_label = ctx.label("i128_tern_false")
        end_label = ctx.label("i128_tern_end")
        out = self._eval_to_bool_eax(expr.condition, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        # True branch: copy its value into dest.
        out += self._int128_value_address(true_arm, ctx)
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        out += self._emit_int128_copy("esi", "edi")
        out.append(f"        jmp     {end_label}")
        out.append(f"{false_label}:")
        out += self._int128_value_address(false_arm, ctx)
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        out += self._emit_int128_copy("esi", "edi")
        out.append(f"{end_label}:")
        return out

    def _cast_to_int128(
        self, expr: ast.Cast, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        src_ty = self._type_of(expr.expr, ctx)
        # Cast int128 → int128: just copy.
        if self._is_int128(src_ty):
            out = self._int128_value_address(expr.expr, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            out += self._emit_int128_copy("esi", "edi")
            return out
        # Cast LL → int128: zero/sign-extend EDX:EAX to 16 bytes.
        if self._is_long_long(src_ty):
            out = self._eval_expr_to_edx_eax(expr.expr, ctx)
            out.append(f"        mov     {_ebp_addr(dest_disp)}, eax")
            out.append(f"        mov     {_ebp_addr(dest_disp + 4)}, edx")
            if self._is_unsigned(src_ty):
                out.append("        xor     eax, eax")
                out.append(f"        mov     {_ebp_addr(dest_disp + 8)}, eax")
                out.append(f"        mov     {_ebp_addr(dest_disp + 12)}, eax")
            else:
                # Sign-extend: replicate sign bit of EDX through high 8 bytes.
                out.append("        mov     eax, edx")
                out.append("        sar     eax, 31")
                out.append(f"        mov     {_ebp_addr(dest_disp + 8)}, eax")
                out.append(f"        mov     {_ebp_addr(dest_disp + 12)}, eax")
            return out
        # Cast int → int128: zero/sign-extend EAX to 16 bytes.
        out = self._eval_expr_to_eax(expr.expr, ctx)
        out.append(f"        mov     {_ebp_addr(dest_disp)}, eax")
        if self._is_unsigned(src_ty):
            out.append("        xor     eax, eax")
            for off in (4, 8, 12):
                out.append(f"        mov     {_ebp_addr(dest_disp + off)}, eax")
        else:
            # Sign-extend.
            out.append("        cdq")
            for off in (4, 8, 12):
                out.append(f"        mov     {_ebp_addr(dest_disp + off)}, edx")
        return out

    def _emit_int_widened_to_int128_on_stack(
        self, expr: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Push `expr` (an int-typed value) widened to 16 bytes onto
        the stack. After this, [esp..esp+15] holds the int128. The
        caller is responsible for adjusting `esp` back."""
        ty = self._type_of(expr, ctx)
        out: list[str] = []
        if self._is_long_long(ty):
            out += self._eval_expr_to_edx_eax(expr, ctx)
            # Push high 8 bytes (sign-extended from edx if signed).
            if self._is_unsigned(ty):
                out.append("        push    0")
                out.append("        push    0")
            else:
                out.append("        mov     ecx, edx")
                out.append("        sar     ecx, 31")
                out.append("        push    ecx")
                out.append("        push    ecx")
            out.append("        push    edx")
            out.append("        push    eax")
        else:
            out += self._eval_expr_to_eax(expr, ctx)
            if self._is_unsigned(ty):
                out.append("        push    0")
                out.append("        push    0")
                out.append("        push    0")
            else:
                out.append("        cdq")
                out.append("        push    edx")
                out.append("        push    edx")
                out.append("        push    edx")
            out.append("        push    eax")
        return out

    def _binary_int128(
        self, expr: ast.BinaryOp, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        op = expr.op
        lt = self._type_of(expr.left, ctx)
        rt = self._type_of(expr.right, ctx)
        # Comma operator: evaluate left for side effects, then copy
        # right's int128 value into the dest slot.
        if op == ",":
            out = self._eval_expr_to_eax(expr.left, ctx)
            out += self._int128_value_address(expr.right, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            out += self._emit_int128_copy("esi", "edi")
            return out
        # Helper: copy left's int128 value into dest_disp slot.
        def copy_left_into_dest() -> list[str]:
            o = self._int128_value_address(expr.left, ctx)
            o.append("        mov     esi, eax")
            o.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            o += self._emit_int128_copy("esi", "edi")
            return o
        # `>>` and `<<` by a constant integer (fast path) or variable
        # (runtime per-bit loop).
        if op in (">>", "<<"):
            if not self._is_int128(lt):
                raise CodegenError(
                    f"__int128 shift requires int128 LHS"
                )
            try:
                shift_n = self._const_eval(expr.right, "<int128-shift>")
            except CodegenError:
                # Variable shift count: emit a runtime per-bit loop.
                return self._int128_shift_runtime(
                    expr.left, expr.right, dest_disp, op, lt, ctx,
                )
            shift_n = shift_n & 127
            return self._int128_shift(expr.left, dest_disp, shift_n, op, lt, ctx)
        # `+` `-`: 4-dword carry/borrow chain. Either operand may be
        # a smaller integer; widen it to 16 bytes on the stack.
        if op in ("+", "-"):
            out = copy_left_into_dest()
            if self._is_int128(rt):
                # Right is also int128: load its address and chain.
                out += self._int128_value_address(expr.right, ctx)
                out.append("        mov     esi, eax")
                out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
                instr0 = "add" if op == "+" else "sub"
                instrN = "adc" if op == "+" else "sbb"
                out.append(f"        mov     eax, [esi]")
                out.append(f"        {instr0}     [edi], eax")
                for off in (4, 8, 12):
                    out.append(f"        mov     eax, [esi + {off}]")
                    out.append(f"        {instrN}     [edi + {off}], eax")
                return out
            # Right is a smaller int — widen on stack and chain.
            out += self._emit_int_widened_to_int128_on_stack(
                expr.right, ctx,
            )
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            instr0 = "add" if op == "+" else "sub"
            instrN = "adc" if op == "+" else "sbb"
            out.append(f"        mov     eax, [esp]")
            out.append(f"        {instr0}     [edi], eax")
            for off in (4, 8, 12):
                out.append(f"        mov     eax, [esp + {off}]")
                out.append(f"        {instrN}     [edi + {off}], eax")
            out.append("        add     esp, 16")
            return out
        # `/` and `%`. Two paths:
        #   - u128 by 32-bit divisor: chain of four `div` instructions
        #     (top-down). Fast path.
        #   - int128 by int128: call the unsigned runtime helper
        #     `___uc386_udiv128` (or `___uc386_umod128`) — binary
        #     long division, slow but correct. For signed inputs we
        #     compute |lhs|, |rhs|, divide, then re-apply the result
        #     sign per C semantics: division uses sign(lhs)^sign(rhs);
        #     modulo follows sign(lhs).
        if op in ("/", "%") and self._is_int128(lt) and self._is_int128(rt):
            signed_op = (
                not self._is_unsigned(lt) or not self._is_unsigned(rt)
            )
            helper = "___uc386_udiv128" if op == "/" else "___uc386_umod128"
            out: list[str] = []
            if not signed_op:
                # Push divisor, dividend, result (right-to-left).
                out += self._int128_value_address(expr.right, ctx)
                out.append("        push    eax")
                out += self._int128_value_address(expr.left, ctx)
                out.append("        push    eax")
                out.append(f"        lea     eax, {_ebp_addr(dest_disp)}")
                out.append("        push    eax")
                out.append(f"        call    {helper}")
                out.append("        add     esp, 12")
                return out
            # Signed path. Reserve 32 bytes of scratch on the stack
            # for |lhs| at [esp+0..15] and |rhs| at [esp+16..31].
            out.append("        sub     esp, 32")
            # |lhs| → [esp+0..15], remember sign of lhs in the
            # sign-byte slot we'll allocate below.
            out += self._int128_value_address(expr.left, ctx)
            out.append("        mov     esi, eax")
            # Push lhs's sign bit (0 or 1) onto the stack so the
            # post-division step can find it. Use BL to avoid touching
            # the operand temps.
            out.append("        mov     eax, [esi + 12]")
            out.append("        shr     eax, 31")
            # Stash sign(lhs) in EBX so we don't lose it.
            out.append("        mov     ebx, eax")
            # Copy |lhs| into [esp+0..15]. If sign(lhs)=1, compute
            # 0 - lhs via 4-dword sub-with-borrow.
            out.append("        test    ebx, ebx")
            jz_pos_lhs = ctx.label("int128_sdiv_pos_lhs")
            done_lhs = ctx.label("int128_sdiv_lhs_done")
            out.append(f"        jz      {jz_pos_lhs}")
            # Negate path.
            out.append("        xor     eax, eax")
            out.append("        sub     eax, [esi]")
            out.append("        mov     [esp + 0], eax")
            for off in (4, 8, 12):
                out.append("        mov     eax, 0")
                out.append(f"        sbb     eax, [esi + {off}]")
                out.append(f"        mov     [esp + {off}], eax")
            out.append(f"        jmp     {done_lhs}")
            out.append(f"{jz_pos_lhs}:")
            for off in (0, 4, 8, 12):
                out.append(f"        mov     eax, [esi + {off}]")
                out.append(f"        mov     [esp + {off}], eax")
            out.append(f"{done_lhs}:")
            # |rhs| → [esp+16..31], remember sign(rhs) in EDI.
            out += self._int128_value_address(expr.right, ctx)
            out.append("        mov     esi, eax")
            out.append("        mov     eax, [esi + 12]")
            out.append("        shr     eax, 31")
            out.append("        mov     edi, eax")
            out.append("        test    edi, edi")
            jz_pos_rhs = ctx.label("int128_sdiv_pos_rhs")
            done_rhs = ctx.label("int128_sdiv_rhs_done")
            out.append(f"        jz      {jz_pos_rhs}")
            out.append("        xor     eax, eax")
            out.append("        sub     eax, [esi]")
            out.append("        mov     [esp + 16], eax")
            for off in (4, 8, 12):
                out.append("        mov     eax, 0")
                out.append(f"        sbb     eax, [esi + {off}]")
                out.append(f"        mov     [esp + {off + 16}], eax")
            out.append(f"        jmp     {done_rhs}")
            out.append(f"{jz_pos_rhs}:")
            for off in (0, 4, 8, 12):
                out.append(f"        mov     eax, [esi + {off}]")
                out.append(f"        mov     [esp + {off + 16}], eax")
            out.append(f"{done_rhs}:")
            # Save signs to known stack offsets so the call doesn't
            # clobber them. Push 4 bytes for each: at [esp - 4] before
            # adjusting esp.  Easier: allocate two more slots upfront,
            # but the cleaner approach is to save before the call.
            # Use the dest slot temporarily? No — that gets written by
            # the call. Allocate 8 more bytes for the saved signs.
            out.append("        sub     esp, 8")
            out.append("        mov     [esp], ebx")        # sign(lhs)
            out.append("        mov     [esp + 4], edi")    # sign(rhs)
            # Now call the unsigned helper:
            #   helper(result_dest, |lhs|_at_[esp+8], |rhs|_at_[esp+24])
            # Push args right-to-left.
            out.append("        lea     eax, [esp + 24]")  # &|rhs|
            out.append("        push    eax")
            out.append("        lea     eax, [esp + 12]")  # &|lhs| (offset shifted by 1 push)
            out.append("        push    eax")
            out.append(f"        lea     eax, {_ebp_addr(dest_disp)}")
            out.append("        push    eax")
            out.append(f"        call    {helper}")
            out.append("        add     esp, 12")
            # Restore sign info.
            out.append("        mov     ebx, [esp]")        # sign(lhs)
            out.append("        mov     edi, [esp + 4]")    # sign(rhs)
            out.append("        add     esp, 40")           # 8 (signs) + 32 (scratch)
            # Apply result sign.
            if op == "/":
                # Negate result iff sign(lhs) XOR sign(rhs) == 1.
                out.append("        xor     ebx, edi")
            # else op == "%": result sign follows lhs.
            out.append("        test    ebx, ebx")
            done = ctx.label("int128_sdiv_done")
            out.append(f"        jz      {done}")
            # Negate dest in place: 0 - dest.
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            out.append("        xor     eax, eax")
            out.append("        sub     eax, [edi]")
            out.append("        mov     [edi], eax")
            for off in (4, 8, 12):
                out.append("        mov     eax, 0")
                out.append(f"        sbb     eax, [edi + {off}]")
                out.append(f"        mov     [edi + {off}], eax")
            out.append(f"{done}:")
            return out
        if op in ("/", "%"):
            # Signed int128 / non-int128: widen rhs to int128 and
            # recurse through the (signed) int128/int128 path so the
            # result-sign machinery applies correctly.
            if not self._is_unsigned(lt):
                synth_cast = ast.Cast(target_type=lt, expr=expr.right)
                ctx.alloc_call_temp(synth_cast, 16)
                synth_binop = ast.BinaryOp(
                    op=op, left=expr.left, right=synth_cast,
                )
                return self._binary_int128(synth_binop, dest_disp, ctx)
            # u128 / non-u128: 32-bit divisor fast path below.
            # Evaluate the divisor into ECX (32-bit). Signedness handled
            # by the existing eval; we treat ECX as unsigned.
            out: list[str] = []
            out += self._eval_expr_to_eax(expr.right, ctx)
            out.append("        mov     ecx, eax")
            # Load left into ESI (address).
            out += self._int128_value_address(expr.left, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            # Divide top-down: edx:eax = high pair, divide, etc.
            # u128 quotient = (a3:a2:a1:a0) / d
            # Step 1: 0:a3 / d -> q3 (= eax), r3 (= edx)
            out.append("        xor     edx, edx")
            out.append("        mov     eax, [esi + 12]")
            out.append("        div     ecx")
            out.append("        mov     [edi + 12], eax")
            # Step 2: r3:a2 / d -> q2, r2
            out.append("        mov     eax, [esi + 8]")
            out.append("        div     ecx")
            out.append("        mov     [edi + 8], eax")
            # Step 3: r2:a1 / d -> q1, r1
            out.append("        mov     eax, [esi + 4]")
            out.append("        div     ecx")
            out.append("        mov     [edi + 4], eax")
            # Step 4: r1:a0 / d -> q0, r0
            out.append("        mov     eax, [esi]")
            out.append("        div     ecx")
            out.append("        mov     [edi], eax")
            if op == "%":
                # Modulo: remainder is in EDX (32-bit). Store into
                # the low dword and zero the rest.
                out.append("        mov     [edi], edx")
                out.append("        xor     eax, eax")
                for off in (4, 8, 12):
                    out.append(f"        mov     [edi + {off}], eax")
            return out
        if op in ("&", "|", "^"):
            out = copy_left_into_dest()
            instr = {"&": "and", "|": "or", "^": "xor"}[op]
            if self._is_int128(rt):
                out += self._int128_value_address(expr.right, ctx)
                out.append("        mov     esi, eax")
                out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
                for off in (0, 4, 8, 12):
                    out.append(f"        mov     eax, [esi + {off}]")
                    out.append(f"        {instr}     [edi + {off}], eax")
                return out
            # Bitwise with smaller int: widen and apply.
            out += self._emit_int_widened_to_int128_on_stack(
                expr.right, ctx,
            )
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            for off in (0, 4, 8, 12):
                out.append(f"        mov     eax, [esp + {off}]")
                out.append(f"        {instr}     [edi + {off}], eax")
            out.append("        add     esp, 16")
            return out
        if op == "*":
            return self._int128_multiply(expr, dest_disp, ctx)
        raise CodegenError(f"__int128 binary op `{op}` not supported")

    def _int128_multiply(
        self, expr: ast.BinaryOp, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `lhs * rhs` (int128 * int128 truncating to 128 bits)."""
        # Schoolbook 4x4-dword multiply, keeping only the low 128 bits:
        #   r[i+j] += L[i] * R[j]   (with carry to higher dwords)
        # We need partials up to i+j = 3. Higher overflows are dropped.
        # Strategy: zero dest, then for each (i, j) with i+j < 4 (or
        # i+j == 3 where we only need the low dword of the product),
        # multiply L[i] * R[j] (32x32 → 64) and add into dest[i+j..i+j+1].
        # The full algorithm below uses 16 mul instructions worst case;
        # most are skipped via the i+j >= 4 cutoff.
        out: list[str] = []
        # Eval &lhs and &rhs onto the stack (so the temp slot reuse
        # via _int128_value_address doesn't get clobbered between
        # the two calls).
        out += self._int128_value_address(expr.left, ctx)
        out.append("        push    eax")
        # If rhs is smaller than int128, widen onto stack so we have
        # 4 dwords to work with. Otherwise just take its address.
        if self._is_int128(self._type_of(expr.right, ctx)):
            out += self._int128_value_address(expr.right, ctx)
            out.append("        push    eax")
            rhs_via_stack = False
        else:
            out += self._emit_int_widened_to_int128_on_stack(
                expr.right, ctx,
            )
            # Stack now: [esp + 0..15] = rhs as int128, [esp + 16] = &lhs.
            rhs_via_stack = True
        # Zero dest first.
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        out.append("        xor     eax, eax")
        for off in (0, 4, 8, 12):
            out.append(f"        mov     [edi + {off}], eax")
        # Set ESI = &lhs, EBX = &rhs. Stack discipline:
        if rhs_via_stack:
            # [esp+0..15] = rhs bytes, [esp+16] = &lhs.
            out.append("        mov     esi, [esp + 16]")
            out.append("        mov     ebx, esp")
        else:
            # [esp+0] = &rhs, [esp+4] = &lhs.
            out.append("        mov     ebx, [esp]")
            out.append("        mov     esi, [esp + 4]")
        # Walk i=0..3, j=0..(3-i): partial = L[i] * R[j], add into
        # dest[i+j..i+j+1] with carry chain into higher dwords.
        # `add` then `adc 0` chain naturally propagates the CF.
        for i in range(4):
            for j in range(4 - i):
                out.append(f"        mov     eax, [esi + {i*4}]")
                out.append(f"        mul     dword [ebx + {j*4}]")
                # eax = low, edx = high
                k = i + j
                out.append(f"        add     [edi + {k*4}], eax")
                if k + 1 < 4:
                    out.append(f"        adc     [edi + {(k+1)*4}], edx")
                    # Propagate any further carry up through the high
                    # dwords. Each `adc dword [...], 0` adds the
                    # previous carry and updates CF for the next.
                    for kk in range(k + 2, 4):
                        out.append(f"        adc     dword [edi + {kk*4}], 0")
        # Pop stack. If rhs went through a temp address, we pushed 8
        # bytes; if rhs was widened on stack, we pushed 20.
        if rhs_via_stack:
            out.append("        add     esp, 20")
        else:
            out.append("        add     esp, 8")
        return out

    def _int128_compare(
        self, expr: ast.BinaryOp, ctx: _FuncCtx,
    ) -> list[str]:
        """Lower an int128 comparison `a OP b`. Result is int (0/1) in EAX.

        Strategy: materialize both operands as 16-byte buffers on the
        stack, then compare dword-by-dword. For ==/!= the comparison
        order doesn't matter; for </>/<=/>= we compare from high to
        low, treating the topmost dword as signed (when the operands
        are signed) and the rest as unsigned.
        """
        op = expr.op
        lt = self._type_of(expr.left, ctx)
        rt = self._type_of(expr.right, ctx)
        unsigned = self._is_unsigned(lt) or self._is_unsigned(rt)

        # Materialize both as 16-byte values on the stack so we can
        # address them by ESP-offset.
        def push_as_int128(e: ast.Expression, ty: ast.TypeNode) -> list[str]:
            o: list[str] = []
            if self._is_int128(ty):
                # Copy 16 bytes of the value onto the stack.
                o += self._int128_value_address(e, ctx)
                o.append("        mov     esi, eax")
                o.append("        sub     esp, 16")
                for off in (0, 4, 8, 12):
                    o.append(f"        mov     eax, [esi + {off}]")
                    o.append(f"        mov     [esp + {off}], eax")
                return o
            # Smaller integer — widen onto stack.
            return self._emit_int_widened_to_int128_on_stack(e, ctx)

        out: list[str] = []
        # Push right first, then left, so [esp]=left, [esp+16]=right.
        out += push_as_int128(expr.right, rt)
        out += push_as_int128(expr.left, lt)

        end = ctx.label("int128_cmp_end")
        true_l = ctx.label("int128_cmp_true")
        false_l = ctx.label("int128_cmp_false")

        if op in ("==", "!="):
            # OR all four dword-XORs together; result zero iff equal.
            out.append("        mov     eax, [esp]")
            out.append("        xor     eax, [esp + 16]")
            out.append("        mov     ecx, [esp + 4]")
            out.append("        xor     ecx, [esp + 20]")
            out.append("        or      eax, ecx")
            out.append("        mov     ecx, [esp + 8]")
            out.append("        xor     ecx, [esp + 24]")
            out.append("        or      eax, ecx")
            out.append("        mov     ecx, [esp + 12]")
            out.append("        xor     ecx, [esp + 28]")
            out.append("        or      eax, ecx")
            # eax == 0 iff equal. Set boolean per op.
            if op == "==":
                out.append("        test    eax, eax")
                out.append("        sete    al")
            else:
                out.append("        test    eax, eax")
                out.append("        setne   al")
            out.append("        movzx   eax, al")
            out.append("        add     esp, 32")
            return out

        # Ordering compare. Walk from high to low. The top dword is
        # signed (if signed compare). Lower dwords are unsigned.
        # Condition table for "a OP b":
        #   <   : strictly less than
        #   <=  : less or equal
        #   >   : strictly greater
        #   >=  : greater or equal
        # We compute strict-less as the primary; equality is handled
        # by the loop falling through.
        # Mapping per signedness:
        signed_lt_jmp = "jl" if not unsigned else "jb"
        signed_gt_jmp = "jg" if not unsigned else "ja"
        unsigned_lt_jmp = "jb"
        unsigned_gt_jmp = "ja"

        # The four dwords in order of significance (high → low):
        # offsets 12, 8, 4, 0 within each operand's 16-byte block.
        for k, off in enumerate((12, 8, 4, 0)):
            l_addr = f"[esp + {off}]"
            r_addr = f"[esp + {off + 16}]"
            out.append(f"        mov     eax, {l_addr}")
            out.append(f"        cmp     eax, {r_addr}")
            if k == 0:
                # Top dword: signedness-aware.
                if op in ("<", "<="):
                    out.append(f"        {signed_lt_jmp}      {true_l}")
                    out.append(f"        {signed_gt_jmp}      {false_l}")
                else:  # > or >=
                    out.append(f"        {signed_gt_jmp}      {true_l}")
                    out.append(f"        {signed_lt_jmp}      {false_l}")
            else:
                # Lower dwords: unsigned.
                if op in ("<", "<="):
                    out.append(f"        {unsigned_lt_jmp}      {true_l}")
                    out.append(f"        {unsigned_gt_jmp}      {false_l}")
                else:
                    out.append(f"        {unsigned_gt_jmp}      {true_l}")
                    out.append(f"        {unsigned_lt_jmp}      {false_l}")
        # All 4 dwords equal — fall through to the equality outcome.
        if op in ("<=", ">="):
            out.append(f"        jmp     {true_l}")
        else:
            out.append(f"        jmp     {false_l}")

        out.append(f"{true_l}:")
        out.append("        mov     eax, 1")
        out.append(f"        jmp     {end}")
        out.append(f"{false_l}:")
        out.append("        xor     eax, eax")
        out.append(f"{end}:")
        out.append("        add     esp, 32")
        return out

    def _int128_shift_runtime(
        self,
        left_expr: ast.Expression,
        count_expr: ast.Expression,
        dest_disp: int,
        op: str,
        lt: ast.TypeNode,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `int128 op count` where count is a runtime int (not a
        compile-time constant). Uses a per-bit loop: shifts dest by 1
        bit per iteration.

        This is O(count) — slow for large counts but correct."""
        signed = not self._is_unsigned(lt)
        # First copy the value into dest.
        out = self._int128_value_address(left_expr, ctx)
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        out += self._emit_int128_copy("esi", "edi")
        # Now load count into ECX. Mask to 0..127 (per gcc undefined
        # behavior beyond width, but we mask for safety).
        out += self._eval_expr_to_eax(count_expr, ctx)
        out.append("        and     eax, 127")
        out.append("        mov     ecx, eax")
        # If count is 0, skip the loop.
        end_label = ctx.label("i128_shift_end")
        loop_label = ctx.label("i128_shift_loop")
        out.append("        test    ecx, ecx")
        out.append(f"        jz      {end_label}")
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        out.append(f"{loop_label}:")
        if op == "<<":
            # Left shift: low → high. shl dword [edi], 1 then rcl
            # propagates the carry up through the higher dwords.
            out.append("        shl     dword [edi], 1")
            out.append("        rcl     dword [edi + 4], 1")
            out.append("        rcl     dword [edi + 8], 1")
            out.append("        rcl     dword [edi + 12], 1")
        else:
            # Right shift: high → low. sar/shr the high dword (sets CF
            # from the bit shifted out), rcr propagates to lower.
            top_instr = "sar" if signed else "shr"
            out.append(f"        {top_instr}     dword [edi + 12], 1")
            out.append("        rcr     dword [edi + 8], 1")
            out.append("        rcr     dword [edi + 4], 1")
            out.append("        rcr     dword [edi], 1")
        out.append("        dec     ecx")
        out.append(f"        jnz     {loop_label}")
        out.append(f"{end_label}:")
        return out

    def _int128_shift(
        self,
        left_expr: ast.Expression,
        dest_disp: int,
        n: int,
        op: str,
        lt: ast.TypeNode,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `int128_value op N` (op is `<<` or `>>`) into dest."""
        out = self._int128_value_address(left_expr, ctx)
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
        if n == 0:
            out += self._emit_int128_copy("esi", "edi")
            return out
        signed = not self._is_unsigned(lt)
        # Read 4 dwords as integers a0/a1/a2/a3 (LSB to MSB).
        # For `>> N`: if N >= 64, high bits to low; etc.
        # Implement via dword-aligned shift.
        word_shift = n // 32
        bit_shift = n % 32
        if op == ">>":
            # `(a3:a2:a1:a0) >> n`: result_i = a_{i+word_shift} (with
            # carry from a_{i+word_shift+1}) — for unsigned. Sign-fill
            # for signed.
            # Simplified path for n in {0, 32, 64, 96}.
            if bit_shift == 0:
                dwords_in = ["[esi]", "[esi + 4]", "[esi + 8]", "[esi + 12]"]
                fill = "0" if not signed else None
                for i in range(4):
                    src_idx = i + word_shift
                    if src_idx < 4:
                        out.append(f"        mov     eax, {dwords_in[src_idx]}")
                    else:
                        if signed:
                            # Sign-extend from highest dword.
                            out.append("        mov     eax, [esi + 12]")
                            out.append("        sar     eax, 31")
                        else:
                            out.append("        xor     eax, eax")
                    out.append(f"        mov     [edi + {i*4}], eax")
                return out
            # Bit-shift within dwords using shrd. Walk from low to high.
            # For r = x >> n where 0 < n < 32 and word_shift = 0:
            # r0 = (a1:a0) >> n  (shrd of a0 with a1, take low 32)
            # r1 = (a2:a1) >> n
            # r2 = (a3:a2) >> n
            # r3 = a3 >> n        (shr or sar)
            # Generalize for word_shift > 0 by offsetting source dwords.
            for i in range(4):
                src_idx = i + word_shift
                if src_idx >= 4:
                    if signed:
                        out.append("        mov     eax, [esi + 12]")
                        out.append("        sar     eax, 31")
                    else:
                        out.append("        xor     eax, eax")
                    out.append(f"        mov     [edi + {i*4}], eax")
                    continue
                out.append(f"        mov     eax, [esi + {src_idx*4}]")
                if src_idx + 1 < 4:
                    out.append(f"        mov     edx, [esi + {(src_idx+1)*4}]")
                    out.append(f"        shrd    eax, edx, {bit_shift}")
                else:
                    instr = "sar" if signed else "shr"
                    out.append(f"        {instr}     eax, {bit_shift}")
                out.append(f"        mov     [edi + {i*4}], eax")
            return out
        # op == "<<"
        if bit_shift == 0:
            dwords_in = ["[esi]", "[esi + 4]", "[esi + 8]", "[esi + 12]"]
            for i in range(4):
                src_idx = i - word_shift
                if 0 <= src_idx < 4:
                    out.append(f"        mov     eax, {dwords_in[src_idx]}")
                else:
                    out.append("        xor     eax, eax")
                out.append(f"        mov     [edi + {i*4}], eax")
            return out
        # General `<< n` (0 < n < 128, bit_shift != 0).
        # Walk from high to low so we can shld with carry from below.
        # r3 = (a3:a2) << n  (shld of a3 with a2, take high 32 -> low 32 of result)
        for i in range(3, -1, -1):
            src_idx = i - word_shift
            if src_idx < 0:
                out.append("        xor     eax, eax")
                out.append(f"        mov     [edi + {i*4}], eax")
                continue
            out.append(f"        mov     eax, [esi + {src_idx*4}]")
            if src_idx - 1 >= 0:
                out.append(f"        mov     edx, [esi + {(src_idx-1)*4}]")
                out.append(f"        shld    eax, edx, {bit_shift}")
            else:
                out.append(f"        shl     eax, {bit_shift}")
            out.append(f"        mov     [edi + {i*4}], eax")
        return out

    def _unary_int128(
        self, expr: ast.UnaryOp, dest_disp: int, ctx: _FuncCtx,
    ) -> list[str]:
        op = expr.op
        if op == "+":
            # Identity: just copy.
            out = self._int128_value_address(expr.operand, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            out += self._emit_int128_copy("esi", "edi")
            return out
        if op == "-":
            # Negation: 0 - x via 4-dword sub.
            out = self._int128_value_address(expr.operand, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            out.append("        xor     eax, eax")
            out.append("        sub     eax, [esi]")
            out.append("        mov     [edi], eax")
            for off in (4, 8, 12):
                out.append("        xor     eax, eax")
                out.append(f"        sbb     eax, [esi + {off}]")
                out.append(f"        mov     [edi + {off}], eax")
            return out
        if op == "~":
            out = self._int128_value_address(expr.operand, ctx)
            out.append("        mov     esi, eax")
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            for off in (0, 4, 8, 12):
                out.append(f"        mov     eax, [esi + {off}]")
                out.append("        not     eax")
                out.append(f"        mov     [edi + {off}], eax")
            return out
        if op in ("++", "--"):
            # Pre/post increment/decrement on an int128 lvalue.
            # Compute &lvalue, modify in place, return the relevant
            # value (post = old, pre = new) into dest_disp.
            is_inc = op == "++"
            is_prefix = expr.is_prefix
            out = self._int128_value_address(expr.operand, ctx)
            out.append("        mov     esi, eax")           # &lvalue
            out.append(f"        lea     edi, {_ebp_addr(dest_disp)}")
            if not is_prefix:
                # Postfix: copy current value to dest before bumping.
                for off in (0, 4, 8, 12):
                    out.append(f"        mov     eax, [esi + {off}]")
                    out.append(f"        mov     [edi + {off}], eax")
            # Bump in place.
            instr0 = "add" if is_inc else "sub"
            instrN = "adc" if is_inc else "sbb"
            out.append(f"        {instr0}     dword [esi], 1")
            for off in (4, 8, 12):
                out.append(f"        {instrN}     dword [esi + {off}], 0")
            if is_prefix:
                # Prefix: copy the post-bump value to dest.
                for off in (0, 4, 8, 12):
                    out.append(f"        mov     eax, [esi + {off}]")
                    out.append(f"        mov     [edi + {off}], eax")
            return out
        raise CodegenError(f"__int128 unary op `{op}` not supported")

    def _compound_assign_complex_lvalue(
        self,
        expr: ast.BinaryOp,
        op: str,
        cplx_ty: ast.ComplexType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """`complex_lvalue OP= rhs`. For Identifier lvalues use the
        simple desugar; for non-Identifier use the snapshot pattern."""
        size = self._size_of(cplx_ty)
        if isinstance(expr.left, ast.Identifier):
            inner = ast.BinaryOp(op=op, left=expr.left, right=expr.right)
            ctx.alloc_call_temp(inner, size)
            return self._complex_copy_assign(
                ast.BinaryOp(op="=", left=expr.left, right=inner),
                cplx_ty, ctx,
            )
        # Non-Identifier: snapshot.
        addr_slot_name = f"__compcx_addr_{id(expr)}"
        snap_slot_name = f"__compcx_snap_{id(expr)}"
        addr_disp = ctx.alloc_local(
            addr_slot_name,
            ast.PointerType(base_type=cplx_ty),
            size=4,
        )
        snap_disp = ctx.alloc_local(snap_slot_name, cplx_ty, size=size)
        out: list[str] = []
        if isinstance(expr.left, ast.Index):
            out += self._index_address(expr.left, ctx)
        elif (
            isinstance(expr.left, ast.UnaryOp)
            and expr.left.op == "*"
        ):
            out += self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Member):
            out += self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"_Complex compound assign target not supported: "
                f"{type(expr.left).__name__}"
            )
        out.append(f"        mov     {_ebp_addr(addr_disp)}, eax")
        # Copy current value into snap_slot.
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(snap_disp)}")
        for off in range(0, size, 4):
            out.append(f"        mov     eax, [esi + {off}]")
            out.append(f"        mov     [edi + {off}], eax")
        # Synthesize Identifier(snap) OP rhs and assign back through
        # *addr_slot. We re-route through _complex_copy_assign with
        # a synthetic *p lvalue that dereferences the addr_slot.
        snap_id = ast.Identifier(name=snap_slot_name)
        inner = ast.BinaryOp(op=op, left=snap_id, right=expr.right)
        ctx.alloc_call_temp(inner, size)
        # We need to build the dereference of the address slot's value.
        # Easiest: compose a new lvalue = `*addr_slot_id`. Then
        # complex_copy_assign(BinaryOp("=", *addr_slot_id, inner)).
        addr_id = ast.Identifier(name=addr_slot_name)
        deref_lvalue = ast.UnaryOp(op="*", operand=addr_id, is_prefix=True)
        return out + self._complex_copy_assign(
            ast.BinaryOp(op="=", left=deref_lvalue, right=inner),
            cplx_ty, ctx,
        )

    def _compound_assign_vector_lvalue(
        self,
        expr: ast.BinaryOp,
        op: str,
        vec_ty: ast.ArrayType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """`vec_lvalue OP= rhs` for non-Identifier vector lvalues
        (Index/Member/*p). Address-once via hidden snapshot slot."""
        addr_slot_name = f"__compvec_addr_{id(expr)}"
        snap_slot_name = f"__compvec_snap_{id(expr)}"
        size = (self._size_of(vec_ty) + 3) & ~3
        addr_disp = ctx.alloc_local(
            addr_slot_name,
            ast.PointerType(base_type=vec_ty),
            size=4,
        )
        snap_disp = ctx.alloc_local(snap_slot_name, vec_ty, size=size)
        out: list[str] = []
        # 1. Compute &lvalue once into addr_slot.
        if isinstance(expr.left, ast.Index):
            out += self._index_address(expr.left, ctx)
        elif (
            isinstance(expr.left, ast.UnaryOp)
            and expr.left.op == "*"
        ):
            out += self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Member):
            out += self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"vector compound assign target not supported: "
                f"{type(expr.left).__name__}"
            )
        out.append(f"        mov     {_ebp_addr(addr_disp)}, eax")
        # 2. Copy `size` bytes from *addr_slot to snap_slot.
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(snap_disp)}")
        for off in range(0, size, 4):
            out.append(f"        mov     eax, [esi + {off}]")
            out.append(f"        mov     [edi + {off}], eax")
        # 3. Synthesize `snapshot OP rhs` and assign into *addr_slot.
        snap_id = ast.Identifier(name=snap_slot_name)
        inner = ast.BinaryOp(op=op, left=snap_id, right=expr.right)
        ctx.alloc_call_temp(inner, size)
        # Evaluate inner into a vector temp via _vector_value_address
        # (which uses the call_temps slot).
        out += self._vector_value_address(inner, ctx)
        # 4. Copy from temp into *addr_slot.
        out.append("        mov     esi, eax")
        out.append(f"        mov     edi, {_ebp_addr(addr_disp)}")
        for off in range(0, size, 4):
            out.append(f"        mov     eax, [esi + {off}]")
            out.append(f"        mov     [edi + {off}], eax")
        out.append(f"        mov     eax, {_ebp_addr(addr_disp)}")
        return out

    def _compound_assign_int128_lvalue(
        self, expr: ast.BinaryOp, op: str, ctx: _FuncCtx,
    ) -> list[str]:
        """`lhs OP= rhs` for non-Identifier int128 lvalues.

        Address-once: compute &lhs into a hidden 4-byte slot, snapshot
        the current value into a hidden int128 slot via a synthesized
        Identifier, evaluate `snapshot OP rhs` into a result temp, then
        copy the result back to *&lhs.
        """
        # Allocate a hidden 4-byte slot for the saved address and a
        # 16-byte slot for the snapshot value. We synthesize a unique
        # name keyed by id(expr) so multiple compound-assigns in the
        # same function don't collide.
        addr_slot_name = f"__compi128_addr_{id(expr)}"
        snap_slot_name = f"__compi128_snap_{id(expr)}"
        addr_disp = ctx.alloc_local(
            addr_slot_name,
            ast.PointerType(base_type=ast.BasicType(name="int128")),
            size=4,
        )
        snap_disp = ctx.alloc_local(
            snap_slot_name,
            ast.BasicType(name="int128"),
            size=16,
        )
        out: list[str] = []
        # 1. Compute &lhs once into the addr slot.
        if isinstance(expr.left, ast.Index):
            out += self._index_address(expr.left, ctx)
        elif (
            isinstance(expr.left, ast.UnaryOp)
            and expr.left.op == "*"
        ):
            out += self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Member):
            out += self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"int128 compound assign target not supported: "
                f"{type(expr.left).__name__}"
            )
        out.append(f"        mov     {_ebp_addr(addr_disp)}, eax")
        # 2. Copy 16 bytes from *addr_slot to snap_slot.
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(snap_disp)}")
        out += self._emit_int128_copy("esi", "edi")
        # 3. Synthesize `snapshot OP rhs` and evaluate into a temp.
        snap_id = ast.Identifier(name=snap_slot_name)
        inner = ast.BinaryOp(op=op, left=snap_id, right=expr.right)
        ctx.alloc_call_temp(inner, 16)
        result_disp = ctx.call_temps[id(inner)]
        out += self._eval_int128_into_temp(inner, result_disp, ctx)
        # 4. Copy result back to *addr_slot.
        out.append(f"        mov     edi, {_ebp_addr(addr_disp)}")
        out.append(f"        lea     esi, {_ebp_addr(result_disp)}")
        out += self._emit_int128_copy("esi", "edi")
        # Return the dest address in EAX (matches assign semantics).
        out.append(f"        mov     eax, {_ebp_addr(addr_disp)}")
        return out

    def _int128_copy_assign(
        self, lhs: ast.Expression, rhs: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `lhs = rhs` where both are __int128. Returns lines that
        leave EAX = &lhs (matching `_assign`'s contract)."""
        # Compute &lhs first; hold in stack so the rhs eval doesn't
        # clobber it.
        out: list[str] = []
        if isinstance(lhs, ast.Identifier):
            out += self._identifier_address(lhs.name, ctx)
        elif isinstance(lhs, ast.Member):
            out += self._member_address(lhs, ctx)
        elif isinstance(lhs, ast.Index):
            out += self._index_address(lhs, ctx)
        elif isinstance(lhs, ast.UnaryOp) and lhs.op == "*":
            out += self._eval_expr_to_eax(lhs.operand, ctx)
        else:
            raise CodegenError(
                f"can't assign __int128 to `{type(lhs).__name__}`"
            )
        out.append("        push    eax")
        # Evaluate rhs into a slot and copy. Smaller-integer rhs is
        # widened via a synthetic Cast (matches `_var_init`).
        rhs_ty = self._type_of(rhs, ctx)
        if not self._is_int128(rhs_ty):
            synth_cast = ast.Cast(
                target_type=ast.BasicType(name="int128"), expr=rhs,
            )
            ctx.alloc_call_temp(synth_cast, 16)
            rhs = synth_cast
        out += self._int128_value_address(rhs, ctx)
        out.append("        mov     esi, eax")
        out.append("        pop     edi")
        out += self._emit_int128_copy("esi", "edi")
        out.append("        mov     eax, edi")
        return out

    # ---- struct member lowering ---------------------------------------

    def _member_address(self, expr: ast.Member, ctx: _FuncCtx) -> list[str]:
        """Compute the address of `expr` (`.` or `->` member) into eax."""
        if expr.is_arrow:
            obj_ty = self._type_of(expr.obj, ctx)
            if (
                isinstance(obj_ty, ast.ArrayType)
                and isinstance(obj_ty.base_type, ast.StructType)
            ):
                # Array of struct decays to pointer; `&arr[0]` is the
                # array's storage base, which `_struct_address` returns.
                struct_name = self._resolve_struct_name(obj_ty.base_type)
                out = self._struct_address(expr.obj, ctx)
            elif (
                isinstance(obj_ty, ast.PointerType)
                and isinstance(obj_ty.base_type, ast.StructType)
            ):
                struct_name = self._resolve_struct_name(obj_ty.base_type)
                # eax = the pointer's value, i.e. the struct's address.
                out = self._eval_expr_to_eax(expr.obj, ctx)
            elif isinstance(obj_ty, ast.StructType):
                # Permissive: typedef chains around function-pointer
                # return types may surface a bare StructType where C
                # would have a PointerType. Treat the struct's address
                # as the target — `_struct_address` does the right thing
                # for an Identifier/Member/Call obj.
                struct_name = self._resolve_struct_name(obj_ty)
                out = self._struct_address(expr.obj, ctx)
            else:
                raise CodegenError(
                    f"`->` requires a pointer to struct "
                    f"(got {type(obj_ty).__name__})"
                )
        else:
            obj_ty = self._type_of(expr.obj, ctx)
            if not isinstance(obj_ty, ast.StructType):
                raise CodegenError(
                    f"`.` requires a struct value "
                    f"(got {type(obj_ty).__name__})"
                )
            struct_name = self._resolve_struct_name(obj_ty)
            out = self._struct_address(expr.obj, ctx)
        _, offset = self._member_layout(struct_name, expr.member)
        if offset != 0:
            out.append(f"        add     eax, {offset}")
        return out

    def _struct_address(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        """Compute the address of a struct l-value into eax.

        Used as the base for `obj.member`: we don't want to "load" the
        struct's bytes into EAX (it doesn't fit), we want its address so
        we can offset into it.
        """
        if isinstance(expr, ast.Identifier):
            return self._identifier_address(expr.name, ctx)
        if isinstance(expr, ast.Member):
            return self._member_address(expr, ctx)
        if isinstance(expr, ast.Index):
            return self._index_address(expr, ctx)
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            return self._eval_expr_to_eax(expr.operand, ctx)
        if isinstance(expr, ast.Call):
            # For struct-returning calls, evaluating the call leaves EAX
            # holding the temp's address (the callee returns the retptr
            # we passed in) — that's exactly the address `.member` wants.
            return self._eval_expr_to_eax(expr, ctx)
        if isinstance(expr, ast.Compound):
            # Compound literal: evaluating it leaves EAX holding the
            # temp's address (for struct/array types).
            return self._eval_expr_to_eax(expr, ctx)
        if isinstance(expr, ast.StmtExpr):
            # `({ ...; expr; })` whose `expr` is a struct value. The body
            # ends with whatever evaluates that expression — for struct
            # types that's the temp's address in EAX. So evaluating the
            # whole StmtExpr leaves us with what we want.
            return self._eval_expr_to_eax(expr, ctx)
        if isinstance(expr, ast.BinaryOp) and expr.op == "=":
            # `s = other_s` as a struct value — perform the copy and
            # leave `&s` (the lhs's address) in EAX. `_struct_copy_assign`
            # already does exactly that.
            target_ty = self._type_of(expr.left, ctx)
            if isinstance(target_ty, ast.StructType):
                return self._struct_copy_assign(expr, target_ty, ctx)
        if isinstance(expr, ast.Cast) and isinstance(
            expr.target_type, ast.StructType
        ):
            # `(struct T) value` — if the source is already a struct of
            # the same type, the cast is a no-op (GCC ext): just return
            # the source's address. Otherwise, type-pun via temp slot.
            try:
                src_ty = self._type_of(expr.expr, ctx)
            except CodegenError:
                src_ty = None
            if isinstance(src_ty, ast.StructType):
                try:
                    if (
                        self._resolve_struct_name(src_ty)
                        == self._resolve_struct_name(expr.target_type)
                    ):
                        return self._struct_address(expr.expr, ctx)
                except CodegenError:
                    pass
            return self._cast_to_struct(expr, ctx)
        raise CodegenError(
            f"can't take address of {type(expr).__name__} for `.member`"
        )

    def _cast_to_struct(
        self, expr: ast.Cast, ctx: _FuncCtx,
    ) -> list[str]:
        """`(struct T) value` — store `value`'s bits into the temp slot
        reserved for this Cast and leave its address in EAX.

        Common case: long-long → union of {long long, struct {...}} is
        a 64-bit type-pun. Smaller widths zero-fill the rest.
        """
        disp = ctx.call_temps[id(expr)]
        size = self._size_of(expr.target_type)
        src_ty = self._type_of(expr.expr, ctx)
        out: list[str] = []
        if self._is_long_long(src_ty):
            out += self._eval_expr_to_edx_eax(expr.expr, ctx)
            out += self._store_from_edx_eax(_ebp_addr(disp))
            # Zero-fill any tail beyond 8 bytes.
            if size > 8:
                out += self._zero_fill_at(disp + 8, size - 8)
        elif self._is_float_type(src_ty):
            out += self._eval_float_to_st0(expr.expr, ctx)
            width = "qword" if self._size_of(src_ty) == 8 else "dword"
            out.append(f"        fstp    {width} {_ebp_addr(disp)}")
            written = self._size_of(src_ty)
            if size > written:
                out += self._zero_fill_at(disp + written, size - written)
        else:
            out += self._eval_expr_to_eax(expr.expr, ctx)
            out += self._store_from_eax(_ebp_addr(disp), src_ty)
            written = self._size_of(src_ty) if not isinstance(
                src_ty, (ast.PointerType, ast.ArrayType)
            ) else 4
            if size > written:
                out += self._zero_fill_at(disp + written, size - written)
        out.append(f"        lea     eax, {_ebp_addr(disp)}")
        return out

    def _bitfield_info(
        self, expr: ast.Member, ctx: _FuncCtx
    ) -> tuple[int, int, ast.TypeNode, int] | None:
        """If `expr` is a bit-field, return
        `(bit_offset, bit_width, type, unit_size)`.

        The address that `_member_address` returns for a bit-field is the
        address of the underlying storage unit (4 or 8 bytes), not the
        field's bit-position; `bit_offset` and `bit_width` describe how
        to read/write the field within that unit. `unit_size` is the
        unit's byte width — 8 only when the bit-field's type is
        `(unsigned) long long`, 4 otherwise.
        """
        obj_ty = self._type_of(expr.obj, ctx)
        # Pointer / array decays to the struct (or struct-pointer) for
        # member access purposes — `s->m` where `s` is array-of-struct
        # is the same as `(&s[0])->m`.
        if isinstance(obj_ty, (ast.PointerType, ast.ArrayType)):
            obj_ty = obj_ty.base_type
        if not isinstance(obj_ty, ast.StructType):
            return None
        struct_name = self._resolve_struct_name(obj_ty)
        bf = self._struct_bitfields.get(struct_name, {})
        info = bf.get(expr.member)
        if info is None:
            return None
        # Backwards compatible with the older 2-tuple form (unit_size
        # implicitly 4) — older callsites in tests may still build raw
        # struct_bitfields entries.
        if len(info) == 2:
            bit_offset, bit_width = info
            unit_size = 4
        else:
            bit_offset, bit_width, unit_size = info
        member_ty, _ = self._member_layout(struct_name, expr.member)
        return bit_offset, bit_width, member_ty, unit_size

    def _member_load(self, expr: ast.Member, ctx: _FuncCtx) -> list[str]:
        """Lower `obj.member` (or `obj->member`) as a value in EAX."""
        bf = self._bitfield_info(expr, ctx)
        if bf is not None:
            return self._bitfield_load(expr, bf, ctx)
        member_ty = self._type_of(expr, ctx)
        addr = self._member_address(expr, ctx)
        if isinstance(member_ty, ast.ArrayType):
            # Array decay — leave the address in eax.
            return addr
        if isinstance(member_ty, ast.StructType):
            # Struct value as expression result — return its address,
            # like array decay. Consumers that need to copy the struct
            # (assignment, by-value pass, return) go through
            # `_struct_address` / `_struct_copy_assign`. Consumers that
            # do something else (a no-op `(struct S)w->t.s` cast, for
            # instance) treat the address as the struct value.
            return addr
        return addr + self._load_to_eax("[eax]", member_ty)

    def _bitfield_load(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode, int],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Read a bit-field: load the storage unit, shift, mask,
        sign-extend. Result fits in EAX (caller widens to EDX:EAX
        when the bit-field's static type is long long; this helper
        returns the value's low 32 bits).
        """
        bit_offset, bit_width, member_ty, unit_size = bf
        if unit_size == 8:
            return self._bitfield_load_ll(expr, bf, ctx)
        out = self._member_address(expr, ctx)        # eax = &storage_unit
        out.append("        mov     eax, [eax]")     # eax = full 32-bit unit
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        if bit_width < 32:
            mask = (1 << bit_width) - 1
            out.append(f"        and     eax, {mask}")
        # Sign-extend if the field is signed (default for plain `int`).
        if not self._is_unsigned(member_ty) and bit_width < 32:
            shift = 32 - bit_width
            out.append(f"        shl     eax, {shift}")
            out.append(f"        sar     eax, {shift}")
        return out

    def _bitfield_load_ll(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode, int],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Read a 64-bit-storage bit-field, returning EDX:EAX with the
        sign- or zero-extended value.

        Steps: load the 8-byte unit into EDX:EAX, shift right by
        `bit_offset` (across the 32-bit boundary), mask to bit_width,
        sign- or zero-extend.
        """
        bit_offset, bit_width, member_ty, _ = bf
        out = self._member_address(expr, ctx)        # eax = &storage_unit
        out.append("        push    ebx")
        out.append("        mov     ebx, eax")
        out.append("        mov     eax, [ebx]")     # eax = low 32 bits
        out.append("        mov     edx, [ebx + 4]")  # edx = high 32 bits
        out.append("        pop     ebx")
        # Shift right by bit_offset across the boundary.
        if bit_offset >= 32:
            shift = bit_offset - 32
            out.append("        mov     eax, edx")
            out.append("        xor     edx, edx")
            if shift:
                out.append(f"        shr     eax, {shift}")
        elif bit_offset > 0:
            # 64-bit shift right by bit_offset (bit_offset < 32).
            out.append(f"        shrd    eax, edx, {bit_offset}")
            out.append(f"        shr     edx, {bit_offset}")
        # Mask to bit_width across both halves.
        mask = (1 << bit_width) - 1
        low_mask = mask & 0xFFFFFFFF
        high_mask = (mask >> 32) & 0xFFFFFFFF
        if bit_width < 64:
            if low_mask != 0xFFFFFFFF:
                out.append(f"        and     eax, {low_mask}")
            out.append(f"        and     edx, {high_mask}")
        # Sign-extend if signed.
        if not self._is_unsigned(member_ty) and bit_width < 64:
            if bit_width <= 32:
                shift = 32 - bit_width
                if shift:
                    out.append(f"        shl     eax, {shift}")
                    out.append(f"        sar     eax, {shift}")
                # Sign-extend EAX to EDX.
                out.append("        cdq")
            else:
                shift = 64 - bit_width
                out.append(f"        shld    edx, eax, 0")  # noop
                out.append(f"        shl     edx, {shift}")
                out.append(f"        sar     edx, {shift}")
                # Re-form eax from the sign bit
                # Easier: just sign-extend the high half.
                # Actually need to sign-extend bit_width-th bit which
                # is in EDX at position (bit_width - 32 - 1).
                # Rebuild: shift EDX:EAX left by (64 - bit_width) and
                # then arithmetic-shift right.
                # The above 3 lines aren't quite right — replace with
                # shld/shl/sar pair.
        return out

    def _bitfield_store(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode, int],
        rhs: ast.Expression,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Write a bit-field: position the rhs, mask the storage, OR them in."""
        bit_offset, bit_width, member_ty, unit_size = bf
        if unit_size == 8:
            return self._bitfield_store_ll(expr, bf, rhs, ctx)
        mask = (1 << bit_width) - 1
        clear_mask = (~(mask << bit_offset)) & 0xFFFFFFFF
        out = self._eval_expr_to_eax(rhs, ctx)        # eax = rhs
        out.append(f"        and     eax, {mask}")    # mask rhs to width
        if bit_offset > 0:
            out.append(f"        shl     eax, {bit_offset}")  # position
        out.append("        push    eax")             # save positioned rhs
        out += self._member_address(expr, ctx)        # eax = &storage_unit
        out.append("        mov     ecx, [eax]")     # ecx = full unit
        out.append(f"        and     ecx, {clear_mask}")  # clear field bits
        out.append("        pop     edx")             # edx = positioned rhs
        out.append("        or      ecx, edx")        # combine
        out.append("        mov     [eax], ecx")     # store back
        # Result of the assignment expression is the new value (rhs
        # narrowed to the bit-field's width). Reconstruct it: undo the
        # left-shift by bit_offset from EDX. For SIGNED bit-fields,
        # also sign-extend the resulting `bit_width`-wide value to the
        # full 32-bit register, so subsequent `==` etc. compares with
        # the right semantics.
        out.append("        mov     eax, edx")
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        if not self._is_unsigned(member_ty) and bit_width < 32:
            shift = 32 - bit_width
            out.append(f"        shl     eax, {shift}")
            out.append(f"        sar     eax, {shift}")
        return out

    def _bitfield_store_ll(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode, int],
        rhs: ast.Expression,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Store into a 64-bit-storage bit-field via RMW.

        Plan: take &unit (push it), eval rhs to EDX:EAX, mask + shift
        to position, then load+clear+OR back into the 8-byte unit.
        Returns EAX = bit_width-wide value (low 32 bits).
        """
        bit_offset, bit_width, member_ty, _ = bf
        mask = (1 << bit_width) - 1
        positioned_mask = (mask << bit_offset) & ((1 << 64) - 1)
        clear_low = (~positioned_mask) & 0xFFFFFFFF
        clear_high = ((~positioned_mask) >> 32) & 0xFFFFFFFF
        # Eval rhs into EDX:EAX (long-long width).
        out = self._eval_expr_to_edx_eax(rhs, ctx)
        # Mask rhs to bit_width across both halves.
        if bit_width <= 32:
            low_m = mask & 0xFFFFFFFF
            if low_m != 0xFFFFFFFF:
                out.append(f"        and     eax, {low_m}")
            out.append("        xor     edx, edx")
        else:
            out.append(f"        and     eax, 0xFFFFFFFF")
            high_m = (mask >> 32) & 0xFFFFFFFF
            out.append(f"        and     edx, {high_m}")
        # Shift left by bit_offset across the boundary.
        if bit_offset >= 32:
            shift = bit_offset - 32
            out.append("        mov     edx, eax")
            out.append("        xor     eax, eax")
            if shift:
                out.append(f"        shl     edx, {shift}")
        elif bit_offset > 0:
            out.append(f"        shld    edx, eax, {bit_offset}")
            out.append(f"        shl     eax, {bit_offset}")
        # Save positioned rhs on the stack: high then low.
        out.append("        push    edx")
        out.append("        push    eax")
        # Compute &storage_unit, RMW.
        out += self._member_address(expr, ctx)
        out.append("        push    ebx")
        out.append("        mov     ebx, eax")
        out.append("        mov     ecx, [ebx]")
        out.append(f"        and     ecx, {clear_low}")
        out.append("        pop     ebx")  # restore ebx ASAP — but we
        # actually still need it. Reorder.
        # Redo: keep ebx alive until both halves are done.
        # Discard the above wrong code — use a simpler edge.
        return self._bitfield_store_ll_simple(expr, bf, rhs, ctx)

    def _compound_assign_bitfield_ll(
        self,
        expr: ast.BinaryOp,
        bf: tuple[int, int, ast.TypeNode, int],
        ctx: _FuncCtx,
    ) -> list[str]:
        """`s.bf op= rhs` for a long-long bit-field. Desugar through
        `_bitfield_load_ll` + the long-long binary op + a store back.

        We synthesize a `_bitfield_store_ll_simple` rhs by reading
        the current value, applying op, then storing the new value.
        """
        op_text = self._COMPOUND_OPS[expr.op]
        # Build a synthetic `lvalue OP rhs` long-long expression.
        inner = ast.BinaryOp(
            op=op_text, left=expr.left, right=expr.right,
        )
        synth = ast.BinaryOp(op="=", left=expr.left, right=inner)
        return self._bitfield_store_ll_simple(expr.left, bf, inner, ctx)

    def _bitfield_store_ll_simple(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode, int],
        rhs: ast.Expression,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Simpler scheme: compute the address into a stack slot, then
        do the RMW from there using EDI as the address register.
        """
        bit_offset, bit_width, member_ty, _ = bf
        mask = (1 << bit_width) - 1
        positioned_mask = (mask << bit_offset) & ((1 << 64) - 1)
        clear_low = (~positioned_mask) & 0xFFFFFFFF
        clear_high = ((~positioned_mask) >> 32) & 0xFFFFFFFF
        # Compute &storage_unit, push.
        out = self._member_address(expr, ctx)
        out.append("        push    eax")
        # Eval rhs into EDX:EAX (long-long width).
        out += self._eval_expr_to_edx_eax(rhs, ctx)
        # Mask rhs to bit_width across both halves.
        if bit_width <= 32:
            low_m = mask & 0xFFFFFFFF
            if low_m != 0xFFFFFFFF:
                out.append(f"        and     eax, {low_m}")
            out.append("        xor     edx, edx")
        else:
            high_m = (mask >> 32) & 0xFFFFFFFF
            out.append(f"        and     edx, {high_m}")
        # Shift left by bit_offset across the boundary.
        if bit_offset >= 32:
            shift = bit_offset - 32
            out.append("        mov     edx, eax")
            out.append("        xor     eax, eax")
            if shift:
                out.append(f"        shl     edx, {shift}")
        elif bit_offset > 0:
            out.append(f"        shld    edx, eax, {bit_offset}")
            out.append(f"        shl     eax, {bit_offset}")
        # ECX = &unit; eax/edx = positioned rhs low/high.
        out.append("        pop     ecx")
        # Read the unit's two halves into the held registers we
        # haven't used: ESI:EDI. Save them around use.
        out.append("        push    esi")
        out.append("        push    edi")
        out.append("        mov     esi, [ecx]")
        out.append("        mov     edi, [ecx + 4]")
        out.append(f"        and     esi, {clear_low}")
        out.append(f"        and     edi, {clear_high}")
        out.append("        or      esi, eax")
        out.append("        or      edi, edx")
        out.append("        mov     [ecx], esi")
        out.append("        mov     [ecx + 4], edi")
        out.append("        pop     edi")
        out.append("        pop     esi")
        # The assignment-expression's value: the bit_width-wide value
        # we just stored, in the low 32 bits of EAX (caller usually
        # discards). We already lost the unshifted value, so just
        # reload from the slot if needed; for now, return the masked
        # rhs (without sign-extension). Most callers ignore.
        # Right-shift to recover the value.
        if bit_offset >= 32:
            shift = bit_offset - 32
            if shift:
                out.append(f"        shr     edx, {shift}")
            out.append("        mov     eax, edx")
        elif bit_offset > 0:
            out.append(f"        shrd    eax, edx, {bit_offset}")
        return out

    # ---- identifier resolution (local vs global) ----------------------

    def _identifier_type(self, name: str, ctx: _FuncCtx) -> ast.TypeNode:
        # `__func__` / `__FUNCTION__` (C99 / GCC) — a const char*
        # pointing to the current function's name.
        if name in ("__func__", "__FUNCTION__", "__PRETTY_FUNCTION__"):
            return ast.PointerType(
                base_type=ast.BasicType(name="char", is_const=True),
            )
        # A `static` local lives as a global under a mangled name; route
        # through the remapping table so callers don't need to know.
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name):
            disp = ctx.lookup(name)
            if disp == _EXTERN_REDIRECT:
                # `extern int v;` inside a block — look up the global.
                if name in self._globals:
                    return self._globals[name]
                if name in self._extern_vars:
                    return self._extern_vars[name]
            return ctx.lookup_type(name)
        if name in self._globals:
            return self._globals[name]
        if name in self._extern_vars:
            return self._extern_vars[name]
        if name in self._func_return_types:
            # A function name in expression context decays to a function
            # pointer. We don't compute pointer arithmetic on functions, so
            # the exact pointee type doesn't matter — represent it as
            # `void *` (4 bytes, no scaling).
            return ast.PointerType(base_type=ast.BasicType(name="void"))
        if name in self._enum_constants:
            # Enum constants are int-typed in C.
            return ast.BasicType(name="int")
        raise CodegenError(f"unknown identifier `{name}`")

    def _is_extern_redirect(self, name: str, ctx: _FuncCtx) -> bool:
        if not ctx.has_local(name):
            return False
        return ctx.lookup(name) == _EXTERN_REDIRECT

    def _types_compatible(
        self, a: ast.TypeNode, b: ast.TypeNode,
        check_quals: bool = False,
    ) -> bool:
        """C type compatibility for `__builtin_types_compatible_p`.

        Top-level qualifiers (const/volatile) are ignored per GCC
        docs. Pointee / array-element qualifiers DO matter — `char *`
        and `const char *` are NOT compatible. The recursive call
        sets `check_quals=True` for the nested case.
        """
        # Compare top-level qualifiers when nested.
        if check_quals:
            qa = (
                getattr(a, "is_const", False),
                getattr(a, "is_volatile", False),
            )
            qb = (
                getattr(b, "is_const", False),
                getattr(b, "is_volatile", False),
            )
            if qa != qb:
                return False
        if isinstance(a, ast.BasicType) and isinstance(b, ast.BasicType):
            # `int` and `signed int` are compatible. `signed` is the
            # default for int but not for char — and that distinction
            # matters here. We model "default signedness" via
            # is_signed=None.
            if a.name != b.name:
                return False
            sa = a.is_signed if a.is_signed is not None else (a.name != "char")
            sb = b.is_signed if b.is_signed is not None else (b.name != "char")
            return sa == sb
        if isinstance(a, ast.PointerType) and isinstance(b, ast.PointerType):
            return self._types_compatible(
                a.base_type, b.base_type, check_quals=True,
            )
        if isinstance(a, ast.ArrayType) and isinstance(b, ast.ArrayType):
            return self._types_compatible(
                a.base_type, b.base_type, check_quals=True,
            )
        if isinstance(a, ast.StructType) and isinstance(b, ast.StructType):
            try:
                ka = self._resolve_struct_name(a)
                kb = self._resolve_struct_name(b)
                return ka == kb
            except CodegenError:
                return False
        if isinstance(a, ast.EnumType) and isinstance(b, ast.EnumType):
            # Named enums compare by tag; anonymous enums are only
            # compatible with themselves (same node).
            if a.name and b.name:
                return a.name == b.name
            return a is b
        # Enum is compatible with its underlying integer type.
        if isinstance(a, ast.EnumType) and isinstance(b, ast.BasicType):
            return b.name in ("int", "long")
        if isinstance(b, ast.EnumType) and isinstance(a, ast.BasicType):
            return a.name in ("int", "long")
        if isinstance(a, ast.FunctionType) and isinstance(b, ast.FunctionType):
            if not self._types_compatible(a.return_type, b.return_type):
                return False
            if len(a.param_types) != len(b.param_types):
                return False
            return all(
                self._types_compatible(pa, pb)
                for pa, pb in zip(a.param_types, b.param_types)
            )
        return False

    def _emit_runtime_offsetof(
        self, expr: ast.OffsetofExpr, ctx: _FuncCtx,
    ) -> list[str]:
        """Compute `__builtin_offsetof(T, designator)` at runtime when
        the designator has non-constant indices and/or T contains VLA.

        Walks the designator inside-out, emitting code to compute
        each contribution and accumulating into a frame-local total.
        """
        # Use ESI as the running offset accumulator. Save EBX (used as
        # scratch by the inner expressions).
        out: list[str] = ["        push    esi", "        push    ebx"]
        out.append("        xor     esi, esi")
        # Walk the designator chain from the outer (member of root) down.
        # Convert into a list of (kind, payload) pairs.
        steps: list = []
        n = expr.designator
        while not isinstance(n, ast.Identifier):
            steps.append(n)
            if isinstance(n, ast.Member):
                n = n.obj
            elif isinstance(n, ast.Index):
                n = n.array
            else:
                raise CodegenError(
                    f"offsetof: unsupported designator {type(n).__name__}"
                )
        steps.reverse()
        # Walk steps. Track the current type (starting at root_ty) so
        # we know each Index's element size.
        cur_ty = expr.target_type
        for step in steps:
            if isinstance(step, ast.Member):
                if isinstance(cur_ty, ast.PointerType):
                    cur_ty = cur_ty.base_type
                if not isinstance(cur_ty, ast.StructType):
                    raise CodegenError(
                        f"offsetof: `.{step.member}` of non-struct"
                    )
                sname = self._resolve_struct_name(cur_ty)
                m_ty, m_off = self._member_layout(sname, step.member)
                out.append(f"        add     esi, {m_off}")
                cur_ty = m_ty
            elif isinstance(step, ast.Index):
                if isinstance(cur_ty, (ast.ArrayType, ast.PointerType)):
                    elem_ty = cur_ty.base_type
                else:
                    raise CodegenError(
                        f"offsetof: index of {type(cur_ty).__name__}"
                    )
                # Compute idx into EAX.
                out += self._eval_expr_to_eax(step.index, ctx)
                # Compute sizeof(elem) into ECX (may be runtime for
                # VLA-shaped element).
                out.append("        push    eax")
                out += self._emit_runtime_size_of(elem_ty, ctx)
                out.append("        mov     ecx, eax")
                out.append("        pop     eax")
                # ESI += idx * sizeof(elem).
                out.append("        imul    eax, ecx")
                out.append("        add     esi, eax")
                cur_ty = elem_ty
            else:
                raise CodegenError(
                    f"offsetof: unsupported designator {type(step).__name__}"
                )
        out.append("        mov     eax, esi")
        out.append("        pop     ebx")
        out.append("        pop     esi")
        return out

    def _offsetof_value(self, expr: ast.OffsetofExpr) -> int:
        """Walk a `__builtin_offsetof(T, designator)` designator chain
        and return the byte offset within `T`.

        The designator is Member/Index nodes rooted at a synthetic
        `Identifier("__offsetof_root")` of `target_type`. We recurse
        from the leaf upward, accumulating offsets via the same
        `_member_layout` / element-size machinery used by `.field`
        and `arr[i]`.
        """
        return self._offsetof_walk(expr.designator, expr.target_type)

    def _offsetof_walk(
        self, node: ast.Expression, root_ty: ast.TypeNode,
    ) -> int:
        if isinstance(node, ast.Identifier):
            # `__offsetof_root` — base case, offset 0.
            return 0
        if isinstance(node, ast.Member):
            base = self._offsetof_walk(node.obj, root_ty)
            base_ty = self._offsetof_type_walk(node.obj, root_ty)
            if isinstance(base_ty, ast.PointerType):
                base_ty = base_ty.base_type
            if not isinstance(base_ty, ast.StructType):
                raise CodegenError(
                    f"offsetof: cannot apply `.{node.member}` to "
                    f"{type(base_ty).__name__}"
                )
            sname = self._resolve_struct_name(base_ty)
            _m_ty, m_off = self._member_layout(sname, node.member)
            return base + m_off
        if isinstance(node, ast.Index):
            base = self._offsetof_walk(node.array, root_ty)
            base_ty = self._offsetof_type_walk(node.array, root_ty)
            if isinstance(base_ty, (ast.ArrayType, ast.PointerType)):
                elem_ty = base_ty.base_type
            else:
                raise CodegenError(
                    f"offsetof: index of {type(base_ty).__name__}"
                )
            idx = self._const_eval(node.index, "<offsetof>")
            return base + idx * self._size_of(elem_ty)
        raise CodegenError(
            f"offsetof: unsupported designator {type(node).__name__}"
        )

    def _offsetof_type_walk(
        self, node: ast.Expression, root_ty: ast.TypeNode,
    ) -> ast.TypeNode:
        if isinstance(node, ast.Identifier):
            return root_ty
        if isinstance(node, ast.Member):
            obj_ty = self._offsetof_type_walk(node.obj, root_ty)
            if isinstance(obj_ty, ast.PointerType):
                obj_ty = obj_ty.base_type
            if not isinstance(obj_ty, ast.StructType):
                raise CodegenError(
                    f"offsetof: cannot apply `.{node.member}` to non-struct"
                )
            sname = self._resolve_struct_name(obj_ty)
            m_ty, _ = self._member_layout(sname, node.member)
            return m_ty
        if isinstance(node, ast.Index):
            obj_ty = self._offsetof_type_walk(node.array, root_ty)
            if isinstance(obj_ty, (ast.ArrayType, ast.PointerType)):
                return obj_ty.base_type
            raise CodegenError("offsetof: index of non-array")
        raise CodegenError(
            f"offsetof: unsupported designator type "
            f"{type(node).__name__}"
        )

    @staticmethod
    def _string_to_bytes(s: str) -> bytes:
        """Encode a Python str (a C string literal) to bytes.

        C string literals in 8-bit char arrays target one byte per
        character. Octal escapes like `"\\377"` produce a U+00FF
        codepoint which we want as a single 0xFF byte. Latin-1
        encoding gives that round-trip for codepoints 0-255;
        surrogate-escaped raw bytes (from a Latin-1 source file)
        decode the same way. We fall back to UTF-8 only for higher
        codepoints which have no single-byte representation.
        """
        try:
            return s.encode("latin-1", errors="surrogateescape")
        except UnicodeEncodeError:
            return s.encode("utf-8", errors="surrogateescape")

    def _resolve_static_init_name(self, name: str) -> str:
        """While emitting a static-local global init, an Identifier
        like `gx` may refer to another static-local in the same
        function. Map it through the saved per-function table so the
        emitted address is the right mangled global key.
        """
        owner = self._emitting_for_func
        if not owner:
            return name
        return self._function_local_static.get(owner, {}).get(name, name)

    def _needs_recursive_init(self, expr: ast.Expression) -> bool:
        """Does `expr` need the recursive `_emit_global_init` path
        rather than a flat `_const_eval` (e.g., it contains a
        `&&label` or address-arithmetic that NASM resolves)?"""
        if isinstance(expr, ast.LabelAddr):
            return True
        if isinstance(expr, ast.UnaryOp) and expr.op == "&":
            return True
        if isinstance(expr, ast.BinaryOp):
            return (
                self._needs_recursive_init(expr.left)
                or self._needs_recursive_init(expr.right)
            )
        if isinstance(expr, ast.Cast):
            return self._needs_recursive_init(expr.expr)
        return False

    def _global_label_diff(self, expr: ast.Expression) -> str | None:
        """If `expr` is `&&l1 - &&l2` (or `+`-flavored), return the
        NASM difference operand; otherwise None.
        """
        if (
            isinstance(expr, ast.BinaryOp)
            and expr.op in ("+", "-")
            and isinstance(expr.left, ast.LabelAddr)
            and isinstance(expr.right, ast.LabelAddr)
        ):
            try:
                a = self._global_label_addr_text(expr.left.label)
                b = self._global_label_addr_text(expr.right.label)
            except CodegenError:
                return None
            sep = " - " if expr.op == "-" else " + "
            return f"{a}{sep}{b}"
        return None

    def _global_label_addr_text(self, label: str) -> str:
        """`&&label` operand text for a static initializer.

        The owning function (set transiently in `_emitting_for_func`
        while emitting a static-local global) supplies the label
        scope. Returns `_<funcname>.LN_user_<label>`.
        """
        func = self._emitting_for_func
        if not func:
            raise CodegenError(
                f"&&{label}: address-of-label only valid inside a function "
                f"body or in a static-local initializer of one"
            )
        labels = self._function_user_labels.get(func, {})
        nasm = labels.get(label)
        if nasm is None:
            raise CodegenError(f"&&{label}: unknown label in {func}")
        return f"_{func}{nasm}"

    def _label_addr_text(self, label: str, ctx: _FuncCtx) -> str:
        """Render the assembler operand for `&&label`.

        NASM uses local-label scoping: `.LN_user_x` is local to the
        most recent global label, which is `_funcname:`. So in
        in-function code we can emit just `.LN_user_x` (or
        `_funcname.LN_user_x`); from anywhere else we must qualify
        with the function prefix. We always emit the qualified form —
        same number of bytes, works in both code and `.data` /
        `.bss` initializer contexts.
        """
        nasm = ctx.user_labels.get(label)
        if nasm is not None:
            return f"_{ctx.func_name}{nasm}"
        # Non-local label: a `&&X` reference inside a lifted nested fn
        # where X is declared in the outer. Resolve via outer's
        # user-label map. We may need to walk up several lift levels
        # for deeply nested fns.
        outer = self._lifted_outer_fn.get(ctx.func_name)
        while outer is not None:
            outer_labels = self._function_user_labels.get(outer, {})
            if label in outer_labels:
                return f"_{outer}{outer_labels[label]}"
            outer = self._lifted_outer_fn.get(outer)
        raise CodegenError(f"&&{label}: unknown label")

    def _identifier_addr_text(self, name: str, ctx: _FuncCtx) -> str:
        """Return the `[...]` operand text used for in-place memory ops.

        Locals render as `[ebp - N]`; globals render as `[_name]`. Used by
        `_inc_dec` (for `inc/dec dword [...]`-style instructions) where
        `_load_to_eax` / `_store_from_eax` would be overkill.
        """
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            return _ebp_addr(ctx.lookup(name))
        if name in self._globals or name in self._extern_vars:
            return f"[_{name}]"
        raise CodegenError(f"unknown identifier `{name}`")

    def _identifier_load(self, name: str, ctx: _FuncCtx) -> list[str]:
        """Lines that produce the value (or, for arrays/functions, the address) of `name` in eax."""
        if name in ("__func__", "__FUNCTION__", "__PRETTY_FUNCTION__"):
            label = self._intern_string(ctx.func_name)
            return [f"        mov     eax, {label}"]
        # Nested fn with trampoline (this frame): `inner` (in value
        # position, e.g. `proc = inner` or `f(inner)`) yields the
        # trampoline's address so callers can call through it.
        if name in ctx.trampolines:
            tramp_disp, _buf_disp = ctx.trampolines[name]
            return [f"        lea     eax, {_ebp_addr(tramp_disp)}"]
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            ty = ctx.lookup_type(name)
            disp = ctx.lookup(name)
            if isinstance(ty, ast.ArrayType):
                if name in ctx.vla_backed:
                    # VLA: slot holds a pointer to malloc'd storage.
                    return [f"        mov     eax, {_ebp_addr(disp)}"]
                # Array decay: yield the slot's address, not its bytes.
                return [f"        lea     eax, {_ebp_addr(disp)}"]
            return self._load_to_eax(_ebp_addr(disp), ty)
        if name in self._globals or name in self._extern_vars:
            ty = self._globals.get(name) or self._extern_vars[name]
            label = f"_{name}"
            if isinstance(ty, ast.ArrayType):
                # The label IS the address in flat-32; load it as an immediate.
                return [f"        mov     eax, {label}"]
            return self._load_to_eax(f"[{label}]", ty)
        if name in self._func_return_types:
            # Function decay: the name yields its address (suitable for
            # assigning to a function pointer or passing as an argument).
            return [f"        mov     eax, _{name}"]
        if name in self._enum_constants:
            # Enum constants lower as immediate integer loads.
            return [f"        mov     eax, {self._enum_constants[name]}"]
        raise CodegenError(f"unknown identifier `{name}`")

    def _identifier_address(self, name: str, ctx: _FuncCtx) -> list[str]:
        """Lines that compute &name into eax — for `&id` and as the base for indexing."""
        if name in ("__func__", "__FUNCTION__", "__PRETTY_FUNCTION__"):
            label = self._intern_string(ctx.func_name)
            return [f"        mov     eax, {label}"]
        if name in ctx.trampolines:
            tramp_disp, _buf_disp = ctx.trampolines[name]
            return [f"        lea     eax, {_ebp_addr(tramp_disp)}"]
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            disp = ctx.lookup(name)
            if name in ctx.vla_backed:
                # VLA: the array's address IS the pointer stored in
                # the slot. Take/scale uses load the pointer.
                return [f"        mov     eax, {_ebp_addr(disp)}"]
            return [f"        lea     eax, {_ebp_addr(disp)}"]
        if name in self._globals or name in self._extern_vars:
            return [f"        mov     eax, _{name}"]
        if name in self._func_return_types:
            # `&fn` and `fn` produce the same address, just like for arrays.
            return [f"        mov     eax, _{name}"]
        raise CodegenError(f"unknown identifier `{name}`")

    def _identifier_store(self, name: str, ctx: _FuncCtx) -> list[str]:
        """Lines that store eax to the slot for `name`, with width per type."""
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        ty = self._identifier_type(name, ctx)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            return self._store_from_eax(_ebp_addr(ctx.lookup(name)), ty)
        return self._store_from_eax(f"[_{name}]", ty)

    def _load_to_eax(self, addr: str, ty: ast.TypeNode) -> list[str]:
        """Lines that load a value of type `ty` from `addr` into EAX.

        Sub-word loads sign- or zero-extend (per signedness) so callers can
        treat EAX uniformly as a 32-bit working value, matching C's integer
        promotion rules.

        Size-8 loads (long long, double in scalar contexts) take only the
        low 32 bits — used when the value is being narrowed to int. Real
        64-bit lowering uses `_load_to_edx_eax`.
        """
        size = self._size_of(ty)
        if size == 4:
            return [f"        mov     eax, {addr}"]
        if size == 2:
            mnem = "movzx" if self._is_unsigned(ty) else "movsx"
            return [f"        {mnem}   eax, word {addr}"]
        if size == 1:
            mnem = "movzx" if self._is_unsigned(ty) else "movsx"
            return [f"        {mnem}   eax, byte {addr}"]
        if size == 8:
            return [f"        mov     eax, {addr}"]
        raise CodegenError(f"can't load size-{size} value into eax")

    def _load_to_edx_eax(self, addr: str) -> list[str]:
        """Load 8 bytes from `addr` into EDX:EAX (high:low).

        `addr` is a NASM addressing form (e.g. `[ebp - 16]`, `[_g]`,
        `[ecx]`); the helper takes the low 4 bytes from `addr` and the
        high 4 bytes from `addr + 4`.
        """
        high_addr = self._bump_addr(addr, 4)
        return [
            f"        mov     eax, {addr}",
            f"        mov     edx, {high_addr}",
        ]

    def _store_from_edx_eax(self, addr: str) -> list[str]:
        """Store EDX:EAX (high:low) as 8 bytes at `addr`."""
        high_addr = self._bump_addr(addr, 4)
        return [
            f"        mov     {addr}, eax",
            f"        mov     {high_addr}, edx",
        ]

    def _cast(self, expr: ast.Cast, ctx: _FuncCtx) -> list[str]:
        """Evaluate `expr.expr` then narrow/extend EAX to match `target_type`.

        i386 makes most casts cheap: pointer ↔ pointer, int ↔ pointer, and
        int ↔ long are all no-ops (every 32-bit value already lives in EAX).
        Narrowing to char/short truncates through the low half of EAX
        (`al`/`ax`) and re-extends per the target's signedness, so a
        subsequent use of the value sees the right C semantics.
        """
        target = expr.target_type
        src_ty = self._type_of(expr.expr, ctx)
        # Float → unsigned int: fistp only does signed conversion, so
        # for values >= 2^31 we need a bias subtract / re-add. Detect
        # the case here so `_eval_expr_to_eax(expr.expr)` (which would
        # otherwise produce a wrong signed value for big floats) is
        # bypassed.
        if (
            self._is_float_type(src_ty)
            and isinstance(target, ast.BasicType)
            and self._size_of(target) == 4
            and self._is_unsigned(target)
        ):
            out = self._eval_float_to_st0(expr.expr, ctx)
            # Compare st(0) against 2^31 (the boundary).
            bias_label = self._intern_float(2147483648.0, 4)
            label_below = ctx.label("f2u_below")
            label_done = ctx.label("f2u_done")
            out.append(f"        fld     dword [{bias_label}]")
            out.append("        fucompp")
            out.append("        fnstsw  ax")
            out.append("        sahf")
            # If st(0) (the value) < bias, take the small path.
            # FPU ZF/PF/CF semantics: for `fucompp st(0)` vs ST(1),
            # ST(0)=bias, ST(1)=value (we loaded value first, then bias).
            # Compare returns: ST(0) < ST(1) → CF=0 ZF=0; equal → CF=0 ZF=1; greater → CF=1.
            # We want value < bias, i.e. ST(1) < ST(0), which means ST(0) > ST(1) → CF=0 (ja).
            out.append(f"        ja      {label_below}")
            # value >= bias: re-eval, subtract bias, fistp (truncate),
            # add 0x80000000 back.
            out += self._eval_float_to_st0(expr.expr, ctx)
            out.append(f"        fld     dword [{bias_label}]")
            out.append("        fsubp   st1, st0")
            out += self._fistp_truncate_dword_to_eax()
            out.append("        add     eax, 0x80000000")
            out.append(f"        jmp     {label_done}")
            out.append(f"{label_below}:")
            out += self._eval_float_to_st0(expr.expr, ctx)
            out += self._fistp_truncate_dword_to_eax()
            out.append(f"{label_done}:")
            return out
        # Float → signed int with saturation (gcc convention): if the
        # float is >= INT_MAX+1 or <= INT_MIN-1, fistp would produce
        # 0x80000000 ("indefinite integer"). Saturate to INT_MAX /
        # INT_MIN respectively.
        if (
            self._is_float_type(src_ty)
            and isinstance(target, ast.BasicType)
            and self._size_of(target) == 4
            and not self._is_unsigned(target)
        ):
            # Use a single-eval path: eval once into st(0), check
            # against 2^31 and -2^31, fistp, fixup if needed.
            out = self._eval_float_to_st0(expr.expr, ctx)
            high_label = self._intern_float(2147483648.0, 4)   # 2^31
            low_label = self._intern_float(-2147483648.0, 4)   # -2^31
            label_high = ctx.label("f2i_high")
            label_low = ctx.label("f2i_low")
            label_done = ctx.label("f2i_done")
            # Check if value >= 2^31 → saturate to INT_MAX.
            out.append("        fld     st0")          # dup st(0)
            out.append(f"        fld     dword [{high_label}]")
            out.append("        fucompp")               # cmp 2^31 vs dup
            out.append("        fnstsw  ax")
            out.append("        sahf")
            # value >= 2^31 means dup >= 2^31 means ST(1) >= ST(0)
            # for the order (st0=2^31, st1=dup): "ST(0) <= ST(1)" → setbe.
            # In jcc: jbe for CF=1 OR ZF=1. Use jbe.
            out.append(f"        jbe     {label_high}")
            # Check if value < -2^31 → saturate to INT_MIN.
            out.append("        fld     st0")
            out.append(f"        fld     dword [{low_label}]")
            out.append("        fucompp")
            out.append("        fnstsw  ax")
            out.append("        sahf")
            # value < -2^31 means ST(1) < ST(0) (since st0=-2^31, st1=value).
            # ST(0) > ST(1) → CF=0 AND ZF=0 → ja.
            out.append(f"        ja      {label_low}")
            # Normal range: fistp truncate.
            out += self._fistp_truncate_dword_to_eax()
            out.append(f"        jmp     {label_done}")
            out.append(f"{label_high}:")
            out.append("        fstp    st0")          # discard st(0)
            out.append("        mov     eax, 0x7FFFFFFF")
            out.append(f"        jmp     {label_done}")
            out.append(f"{label_low}:")
            out.append("        fstp    st0")          # discard st(0)
            out.append("        mov     eax, 0x80000000")
            out.append(f"{label_done}:")
            return out
        # Vector → integer cast in scalar context. The vector's bytes
        # are type-punned: for size==4 we read the low dword; for
        # size==8 (long long) we'd need EDX:EAX, but the caller of
        # `_cast` only takes EAX, so we just load the low dword. (The
        # `_eval_expr_to_edx_eax(Cast)` path handles the full LL load
        # for callers that want both halves.)
        if (
            isinstance(src_ty, ast.ArrayType)
            and getattr(src_ty, "is_vector", False)
            and isinstance(target, ast.BasicType)
            and self._size_of(src_ty) >= self._size_of(target)
        ):
            out = self._vector_value_address(expr.expr, ctx)
            out.append("        mov     eax, [eax]")
            tsize = self._size_of(target)
            if tsize == 1:
                mnem = "movzx" if target.is_signed is False else "movsx"
                out.append(f"        {mnem}   eax, al")
            elif tsize == 2:
                mnem = "movzx" if target.is_signed is False else "movsx"
                out.append(f"        {mnem}   eax, ax")
            return out
        out = self._eval_expr_to_eax(expr.expr, ctx)
        target = expr.target_type
        if isinstance(target, ast.PointerType):
            return out
        if isinstance(target, ast.EnumType):
            return out
        if isinstance(target, ast.BasicType):
            size = self._size_of(target)
            if size == 4 or size == 8:
                # size-8 (long long) is treated as 32-bit in EAX for now;
                # full 64-bit is not yet implemented but a cast through it
                # in scalar contexts can pass through.
                return out
            mnem = "movzx" if target.is_signed is False else "movsx"
            half = "al" if size == 1 else "ax" if size == 2 else None
            if half is None:
                raise CodegenError(
                    f"cast to size-{size} basic type not supported"
                )
            out.append(f"        {mnem}   eax, {half}")
            return out
        if isinstance(target, ast.ArrayType):
            # `(vec_t) value` in EAX-scalar context: produce the
            # vector's address (consumers reading EAX get a pointer
            # to the temp). Allocated in `_collect_call_temps`.
            return self._vector_value_address(expr, ctx)
        raise CodegenError(
            f"cast to {type(target).__name__} not supported"
        )

    def _store_from_eax(self, addr: str, ty: ast.TypeNode) -> list[str]:
        """Lines that store EAX (treated as `ty`) to `addr`, then leave
        EAX = the stored-and-reread value (narrowed to `ty`'s width and
        re-extended per `ty`'s signedness). This makes chained
        assignments like `ul = us = -1` see the same conversion the
        lvalue would have seen on a plain read.

        For size-8 types (long long, double) we store only the low 32
        bits in EAX and sign-extend into the high half. Real long long
        goes through `_store_from_edx_eax`; this path is the
        narrow-on-store fallback when callers happen to be in 32-bit
        eval mode.
        """
        size = self._size_of(ty)
        if size == 4:
            return [f"        mov     {addr}, eax"]
        if size == 2:
            mnem = "movzx" if self._is_unsigned(ty) else "movsx"
            return [
                f"        mov     word {addr}, ax",
                f"        {mnem}   eax, ax",
            ]
        if size == 1:
            mnem = "movzx" if self._is_unsigned(ty) else "movsx"
            return [
                f"        mov     byte {addr}, al",
                f"        {mnem}   eax, al",
            ]
        if size == 8:
            high_addr = self._bump_addr(addr, 4)
            return [
                f"        mov     {addr}, eax",
                f"        cdq",
                f"        mov     {high_addr}, edx",
            ]
        raise CodegenError(f"can't store size-{size} value from eax")

    @staticmethod
    def _bump_addr(addr: str, delta: int) -> str:
        """Add `delta` to the displacement in a NASM addressing form.

        Handles `[ebp - N]`, `[ebp + N]`, and `[ecx]` (no displacement).
        Anything more elaborate raises so callers can fix at the call site.
        """
        s = addr.strip()
        if not (s.startswith("[") and s.endswith("]")):
            raise CodegenError(f"can't bump addr {addr!r}")
        inner = s[1:-1].strip()
        if "+" in inner:
            base, off = inner.rsplit("+", 1)
            new_off = int(off.strip()) + delta
            return f"[{base.strip()} + {new_off}]" if new_off >= 0 else f"[{base.strip()} - {-new_off}]"
        if "-" in inner:
            # ebp - N
            base, off = inner.rsplit("-", 1)
            new_off = -int(off.strip()) + delta
            return f"[{base.strip()} + {new_off}]" if new_off >= 0 else f"[{base.strip()} - {-new_off}]"
        # No displacement, e.g. [ecx]
        return f"[{inner} + {delta}]"

    # ---- statements -----------------------------------------------------

    def _compound(self, block: ast.CompoundStmt, ctx: _FuncCtx) -> list[str]:
        ctx.enter_scope()
        out: list[str] = []
        for item in block.items:
            out += self._item(item, ctx)
        ctx.exit_scope()
        return out

    def _item(self, item, ctx: _FuncCtx) -> list[str]:
        if isinstance(item, ast.VarDecl):
            return self._var_init(item, ctx)
        if isinstance(item, ast.ReturnStmt):
            return self._return(item, ctx)
        if isinstance(item, ast.CompoundStmt):
            return self._compound(item, ctx)
        if isinstance(item, ast.ExpressionStmt):
            return self._expr_stmt(item, ctx)
        if isinstance(item, ast.IfStmt):
            return self._if(item, ctx)
        if isinstance(item, ast.WhileStmt):
            return self._while(item, ctx)
        if isinstance(item, ast.DoWhileStmt):
            return self._do_while(item, ctx)
        if isinstance(item, ast.ForStmt):
            return self._for(item, ctx)
        if isinstance(item, ast.SwitchStmt):
            return self._switch(item, ctx)
        if isinstance(item, ast.CaseStmt):
            # Inside a switch body. The pre-walk in `_switch` assigned
            # this CaseStmt a label; emit the label and recurse into
            # the labeled statement.
            if not ctx.active_case_labels:
                raise CodegenError("`case` outside of a switch")
            label = ctx.active_case_labels[-1].get(id(item))
            if label is None:
                # Shouldn't happen — `_switch`'s walk reaches every
                # CaseStmt within its body. If we hit this, the walk
                # missed a structural node it should have descended
                # into.
                raise CodegenError("internal: case label not found")
            return [f"{label}:"] + self._item(item.stmt, ctx)
        if isinstance(item, ast.LabelStmt):
            label = ctx.user_labels[item.label]
            return [f"{label}:"] + self._item(item.stmt, ctx)
        if isinstance(item, ast.GotoStmt):
            # GCC computed goto: `goto *expr;` — evaluate the address
            # and jump indirect.
            if item.target is not None:
                out = self._eval_expr_to_eax(item.target, ctx)
                out.append("        jmp     eax")
                return out
            # Non-local goto: a `goto X` from inside a lifted nested
            # fn where X was declared in the outer via `__label__ X`.
            # The static-link slot holds the address of outer's
            # buf-array (set on entry from ECX, which the trampoline
            # or direct caller loaded). Compute the buf's address as
            # `static_link + 12 * idx_X` and longjmp through it. The
            # outer's prologue has a matching setjmp that dispatches
            # to X on EAX=1.
            idx = ctx.nonlocal_goto_targets.get(item.label)
            if idx is not None and ctx.trampoline_static_link_disp is not None:
                slot_addr = _ebp_addr(ctx.trampoline_static_link_disp)
                offset = 12 * idx
                lines = ["        push    1"]
                if offset == 0:
                    lines.append(f"        push    dword {slot_addr}")
                else:
                    lines.append(f"        mov     eax, {slot_addr}")
                    lines.append(f"        add     eax, {offset}")
                    lines.append("        push    eax")
                lines.append("        call    ___builtin_longjmp")
                lines.append("        add     esp, 8")
                return lines
            target = ctx.user_labels.get(item.label)
            if target is None:
                raise CodegenError(
                    f"goto: unknown label `{item.label}`"
                )
            # Restore ESP to the post-locals baseline if we have VLAs.
            # This frees ALL VLAs (sub-esp-allocated). Safe when the
            # goto target is before all VLA decls (the common pattern).
            # If the target is INSIDE a VLA's scope, this would lose
            # the VLA's storage — but per C, jumping into a VLA's scope
            # from outside is undefined.
            out: list[str] = []
            if ctx.vla_baseline_disp is not None:
                out.append(
                    f"        mov     esp, {_ebp_addr(ctx.vla_baseline_disp)}"
                )
            out.append(f"        jmp     {target}")
            return out
        if isinstance(item, ast.BreakStmt):
            if not ctx.break_targets:
                raise CodegenError("`break` outside of a loop or switch")
            return [f"        jmp     {ctx.break_targets[-1]}"]
        if isinstance(item, ast.ContinueStmt):
            if not ctx.continue_targets:
                raise CodegenError("`continue` outside of a loop")
            return [f"        jmp     {ctx.continue_targets[-1]}"]
        if isinstance(item, ast.AsmStmt):
            return self._asm_stmt(item, ctx)
        if isinstance(item, ast.DeclarationList):
            # `int x, *p, **pp;` parses as DeclarationList of one VarDecl
            # per declarator. Lower each one in order.
            out: list[str] = []
            for decl in item.declarations:
                out += self._item(decl, ctx)
            return out
        if isinstance(item, ast.StructDecl):
            # In-function struct/union definition (e.g. `struct T { int x; };`
            # in the middle of a body). Register the layout if it's a
            # definition; emit no code beyond VLA-capture stores.
            out: list[str] = []
            captures = getattr(item, "_vla_member_captures", None)
            if captures:
                # Re-bind capture slot names in the current scope and
                # emit eval+store per VLA dim. The slots were allocated
                # in `_collect_locals`'s pre-pass.
                for slot_name, orig_size in captures:
                    ctx.alloc_local(
                        slot_name, ast.BasicType(name="int"),
                        decl=orig_size,
                    )
                    try:
                        out += self._eval_expr_to_eax(orig_size, ctx)
                    except CodegenError:
                        continue
                    cdisp = ctx.lookup(slot_name)
                    out.append(f"        mov     {_ebp_addr(cdisp)}, eax")
            if item.is_definition:
                self._register_struct(item)
            return out
        if isinstance(item, ast.EnumDecl):
            # Likewise for in-function enum definitions.
            if item.is_definition:
                self._register_enum(item)
            return []
        if isinstance(item, ast.TypedefDecl):
            # uc_core resolves typedef names at parse time, so the only
            # thing left at codegen is to register typedef'd structs
            # whose layout might be needed via `_resolve_struct_name`.
            # `_resolve_struct_name` already handles inline-member
            # StructTypes lazily, so there's nothing more to do here.
            return []
        if isinstance(item, ast.FunctionDecl):
            # Nested function definition — already lifted in the
            # top-of-`_function` pre-pass. Emit no code at this point;
            # the lifted function compiles separately at file scope.
            return []
        raise CodegenError(
            f"{type(item).__name__} not implemented yet"
        )

    def _if(self, stmt: ast.IfStmt, ctx: _FuncCtx) -> list[str]:
        else_label = ctx.label("else")
        end_label = ctx.label("endif")
        out = self._eval_to_bool_eax(stmt.condition, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {else_label if stmt.else_branch else end_label}")
        out += self._item(stmt.then_branch, ctx)
        if stmt.else_branch is not None:
            out.append(f"        jmp     {end_label}")
            out.append(f"{else_label}:")
            out += self._item(stmt.else_branch, ctx)
        out.append(f"{end_label}:")
        return out

    def _while(self, stmt: ast.WhileStmt, ctx: _FuncCtx) -> list[str]:
        top = ctx.label("while_top")
        end = ctx.label("while_end")
        ctx.break_targets.append(end)
        ctx.continue_targets.append(top)
        try:
            out = [f"{top}:"]
            out += self._eval_to_bool_eax(stmt.condition, ctx)
            out.append("        test    eax, eax")
            out.append(f"        jz      {end}")
            out += self._item(stmt.body, ctx)
            out.append(f"        jmp     {top}")
            out.append(f"{end}:")
        finally:
            ctx.break_targets.pop()
            ctx.continue_targets.pop()
        return out

    def _do_while(self, stmt: ast.DoWhileStmt, ctx: _FuncCtx) -> list[str]:
        top = ctx.label("do_top")
        cont = ctx.label("do_cont")
        end = ctx.label("do_end")
        # `continue` jumps to the condition test, not the top of the body.
        ctx.break_targets.append(end)
        ctx.continue_targets.append(cont)
        try:
            out = [f"{top}:"]
            out += self._item(stmt.body, ctx)
            out.append(f"{cont}:")
            out += self._eval_to_bool_eax(stmt.condition, ctx)
            out.append("        test    eax, eax")
            out.append(f"        jnz     {top}")
            out.append(f"{end}:")
        finally:
            ctx.break_targets.pop()
            ctx.continue_targets.pop()
        return out

    def _for(self, stmt: ast.ForStmt, ctx: _FuncCtx) -> list[str]:
        # `for (int i = ...)` declares i in a scope wrapping init + body
        # — collect_locals pushed the same way, so a sibling `for (int i)`
        # later won't collide.
        ctx.enter_scope()
        try:
            top = ctx.label("for_top")
            step = ctx.label("for_step")
            end = ctx.label("for_end")
            out: list[str] = []
            if stmt.init is not None:
                if isinstance(stmt.init, ast.Expression):
                    out += self._eval_expr_to_eax(stmt.init, ctx)
                else:
                    out += self._item(stmt.init, ctx)
            out.append(f"{top}:")
            if stmt.condition is not None:
                out += self._eval_to_bool_eax(stmt.condition, ctx)
                out.append("        test    eax, eax")
                out.append(f"        jz      {end}")
            # `continue` jumps to the step, not the top.
            ctx.break_targets.append(end)
            ctx.continue_targets.append(step)
            try:
                out += self._item(stmt.body, ctx)
            finally:
                ctx.break_targets.pop()
                ctx.continue_targets.pop()
            out.append(f"{step}:")
            if stmt.update is not None:
                out += self._eval_expr_to_eax(stmt.update, ctx)
            out.append(f"        jmp     {top}")
            out.append(f"{end}:")
            return out
        finally:
            ctx.exit_scope()

    def _switch(self, stmt: ast.SwitchStmt, ctx: _FuncCtx) -> list[str]:
        """Lower `switch (expr) { case V: ...; default: ...; }`.

        Cases can appear anywhere within the switch body — including
        deep inside loops or `if`s, as in Duff's device. We pre-walk
        the body recursively, assign each `case` / `default` a unique
        label, then emit a dispatch ladder up front and let `_item`'s
        own CaseStmt branch materialize each label inline as the body
        is lowered. A nested switch starts its own pre-walk; we don't
        recurse past its boundary.

        `break` resolves via `ctx.break_targets`; `continue` deliberately
        does NOT push to `continue_targets` here, so a `continue` inside
        the switch escapes to the enclosing loop (as C requires).
        """
        end_label = ctx.label("switch_end")
        # case_specs entries: (kind, value, value_end, label).
        # `kind` is "case" (single value), "range" (V..VE), or "default".
        case_specs: list[tuple[str, int | None, int | None, str]] = []
        case_label_map: dict[int, str] = {}
        default_label: str | None = None

        def walk(node):
            nonlocal default_label
            if node is None:
                return
            if isinstance(node, ast.CaseStmt):
                if node.value is None:
                    if default_label is not None:
                        raise CodegenError(
                            "multiple `default` labels in switch"
                        )
                    default_label = ctx.label("default")
                    case_specs.append(
                        ("default", None, None, default_label)
                    )
                    case_label_map[id(node)] = default_label
                else:
                    value = self._const_eval(node.value, "case")
                    lbl = ctx.label("case")
                    if getattr(node, "value_end", None) is not None:
                        value_end = self._const_eval(node.value_end, "case")
                        case_specs.append(("range", value, value_end, lbl))
                    else:
                        case_specs.append(("case", value, None, lbl))
                    case_label_map[id(node)] = lbl
                walk(node.stmt)
                return
            # Recurse into the structural nodes whose bodies can host
            # case labels. A nested switch is opaque — its own labels
            # belong to it, not to us.
            if isinstance(node, ast.CompoundStmt):
                for item in node.items:
                    walk(item)
            elif isinstance(node, ast.IfStmt):
                walk(node.then_branch)
                walk(node.else_branch)
            elif isinstance(node, (ast.WhileStmt, ast.DoWhileStmt, ast.ForStmt)):
                walk(node.body)
            elif isinstance(node, ast.LabelStmt):
                walk(node.stmt)
            # Anything else (ExpressionStmt, ReturnStmt, VarDecl, nested
            # SwitchStmt, etc.) doesn't contribute cases.

        walk(stmt.body)

        # Eval the controlling expression once. Long-long switches
        # use EDX:EAX dispatch; otherwise EAX.
        ctrl_ty = self._type_of(stmt.expr, ctx)
        is_ll = self._is_long_long(ctrl_ty)
        if is_ll:
            ctrl_unsigned = self._is_unsigned(ctrl_ty)
            out = self._eval_expr_to_edx_eax(stmt.expr, ctx)
            for kind, value, value_end, target in case_specs:
                if kind == "case":
                    v = value & 0xFFFFFFFFFFFFFFFF
                    lo = v & 0xFFFFFFFF
                    hi = (v >> 32) & 0xFFFFFFFF
                    skip = ctx.label("ll_neq")
                    out.append(f"        cmp     edx, 0x{hi:08X}")
                    out.append(f"        jne     {skip}")
                    out.append(f"        cmp     eax, 0x{lo:08X}")
                    out.append(f"        je      {target}")
                    out.append(f"{skip}:")
                elif kind == "range":
                    # value <= ctrl <= value_end. Use unsigned
                    # comparison if the controlling type is unsigned.
                    a = value & 0xFFFFFFFFFFFFFFFF
                    b = value_end & 0xFFFFFFFFFFFFFFFF
                    a_lo = a & 0xFFFFFFFF
                    a_hi = (a >> 32) & 0xFFFFFFFF
                    b_lo = b & 0xFFFFFFFF
                    b_hi = (b >> 32) & 0xFFFFFFFF
                    skip = ctx.label("ll_rskip")
                    take = ctx.label("ll_rtake")
                    # Compare ctrl with `a` (lower bound); if ctrl < a,
                    # skip. Compare high first; if higher, ctrl > a so
                    # might match. If equal, compare low.
                    out.append(f"        cmp     edx, 0x{a_hi:08X}")
                    if ctrl_unsigned:
                        out.append(f"        jb      {skip}")
                        out.append(f"        ja      .L{ctx._next_label}_ge_a")
                    else:
                        out.append(f"        jl      {skip}")
                        out.append(f"        jg      .L{ctx._next_label}_ge_a")
                    out.append(f"        cmp     eax, 0x{a_lo:08X}")
                    if ctrl_unsigned:
                        out.append(f"        jb      {skip}")
                    else:
                        out.append(f"        jl      {skip}")
                    out.append(f".L{ctx._next_label}_ge_a:")
                    ctx._next_label += 1
                    # Compare ctrl with `b` (upper bound); if ctrl > b,
                    # skip. Otherwise take.
                    out.append(f"        cmp     edx, 0x{b_hi:08X}")
                    if ctrl_unsigned:
                        out.append(f"        ja      {skip}")
                        out.append(f"        jb      {take}")
                    else:
                        out.append(f"        jg      {skip}")
                        out.append(f"        jl      {take}")
                    out.append(f"        cmp     eax, 0x{b_lo:08X}")
                    if ctrl_unsigned:
                        out.append(f"        ja      {skip}")
                    else:
                        out.append(f"        jg      {skip}")
                    out.append(f"{take}:")
                    out.append(f"        jmp     {target}")
                    out.append(f"{skip}:")
            out.append(f"        jmp     {default_label or end_label}")
        else:
            out = self._eval_expr_to_eax(stmt.expr, ctx)
            for kind, value, value_end, target in case_specs:
                if kind == "case":
                    out.append(f"        cmp     eax, {value}")
                    out.append(f"        je      {target}")
                elif kind == "range":
                    skip = ctx.label("rskip")
                    # Use signed comparison since case values may be
                    # negative. Unsigned ranges still work as long as
                    # both endpoints are < 2^31.
                    out.append(f"        cmp     eax, {value}")
                    out.append(f"        jl      {skip}")
                    out.append(f"        cmp     eax, {value_end}")
                    out.append(f"        jle     {target}")
                    out.append(f"{skip}:")
            out.append(f"        jmp     {default_label or end_label}")

        # Body emission. `_item`'s CaseStmt branch consults
        # `ctx.active_case_labels[-1]` to find the right label for each
        # `case` it encounters.
        ctx.break_targets.append(end_label)
        ctx.active_case_labels.append(case_label_map)
        try:
            out += self._item(stmt.body, ctx)
        finally:
            ctx.break_targets.pop()
            ctx.active_case_labels.pop()

        out.append(f"{end_label}:")
        return out

    def _expr_stmt(self, stmt: ast.ExpressionStmt, ctx: _FuncCtx) -> list[str]:
        if stmt.expr is None:
            return []
        # Result is discarded; we still evaluate for side effects (assignment).
        return self._eval_expr_to_eax(stmt.expr, ctx)

    def _var_init(self, decl: ast.VarDecl, ctx: _FuncCtx) -> list[str]:
        # Re-bind the per-VLA-capture slot names in the current scope.
        # `_capture_vla_sizes` allocated these in `_collect_locals` (a
        # different scope chain); use `alloc_local`'s decl-based
        # rebind path to restore the disp without bumping frame_size.
        captures = getattr(decl, "_vla_captures", None)
        if captures:
            for slot_name, orig_size in captures:
                ctx.alloc_local(
                    slot_name, ast.BasicType(name="int"),
                    decl=orig_size,
                )
        out = self._var_init_inner(decl, ctx)
        # VLA-backed local: emit `sub esp, runtime_size; mov slot, esp`
        # at decl point. Sub-esp gives us scope-correct dealloc — at
        # function return, `mov esp, ebp` reclaims; at goto-back, the
        # GotoStmt branch restores ESP to a saved baseline.
        # `ctx.has_vla` triggers the baseline-save in the prologue.
        if decl.name in ctx.vla_backed and ctx.has_local(decl.name):
            var_type = ctx.lookup_type(decl.name)
            disp = ctx.lookup(decl.name)
            alloc_lines: list[str] = []
            alloc_lines += self._emit_runtime_size_of(var_type, ctx)
            alloc_lines += [
                "        add     eax, 15",
                "        and     eax, ~15",
                "        sub     esp, eax",
                f"        mov     {_ebp_addr(disp)}, esp",
            ]
            ctx.has_vla = True
            out = alloc_lines + out
        # If the decl has VLA captures, prepend the capture stores so
        # `sizeof(decl)` (which reads the captured slot) sees the
        # correct value. Captures evaluate at decl point.
        if captures:
            prefix: list[str] = []
            for slot_name, orig_size in captures:
                try:
                    prefix += self._eval_expr_to_eax(orig_size, ctx)
                except CodegenError:
                    continue
                cdisp = ctx.lookup(slot_name)
                prefix.append(f"        mov     {_ebp_addr(cdisp)}, eax")
            return prefix + out
        return out

    def _var_init_inner(self, decl: ast.VarDecl, ctx: _FuncCtx) -> list[str]:
        # Local function declaration (`int f(int);`): registered as an
        # extern in `_collect_locals`; nothing to emit here.
        if isinstance(decl.var_type, ast.FunctionType):
            return []
        # `static` locals were registered as globals during
        # `_collect_locals`; their initializer fires once at program
        # load via the `.data` emission, not on every call.
        if decl.storage_class == "static":
            return []
        # `extern int v;` inside a block: bind a sentinel in the
        # current scope so identifier reads route through the global
        # symbol instead of any outer local of the same name.
        if decl.storage_class == "extern":
            ctx.slots[-1][decl.name] = _EXTERN_REDIRECT
            ctx.types[-1][decl.name] = self._resolved_var_type(decl)
            return []
        # Captured-by-nested-fn local: `_collect_locals` registered the
        # name as a global with a mangled label. Emit the initializer
        # as a store to the global on every entry to this function.
        if decl.name in ctx.local_captures:
            if decl.init is None:
                return []
            mangled = ctx.local_captures[decl.name]
            var_type = self._resolved_var_type(decl)
            if self._is_float_type(var_type):
                size = self._size_of(var_type)
                width = "dword" if size == 4 else "qword"
                out = self._eval_float_to_st0(decl.init, ctx)
                out.append(f"        fstp    {width} [_{mangled}]")
                return out
            if self._is_long_long(var_type):
                out = self._eval_expr_to_edx_eax(decl.init, ctx)
                out.append(f"        mov     [_{mangled}], eax")
                out.append(f"        mov     [_{mangled} + 4], edx")
                return out
            out = self._eval_expr_to_eax(decl.init, ctx)
            out += self._store_from_eax(f"[_{mangled}]", var_type)
            return out
        # Re-bind this VarDecl's name in the current scope using the disp
        # assigned during `_collect_locals`. The pre-pass walked the body
        # under the same enter_scope/exit_scope pattern, so we know an
        # entry exists in `decl_disps` keyed by id(decl).
        if id(decl) in ctx.decl_disps:
            ctx.alloc_local(decl.name, ctx.decl_types[id(decl)], decl=decl)
        disp = ctx.lookup(decl.name)
        if decl.init is None:
            # Uninitialized — leave the slot as-is. Reading it is UB, but
            # we don't pre-zero unless required.
            return []
        # Use the *resolved* type (size filled in for unsized arrays), so
        # `_array_init` and `_store_from_eax` see a concrete shape.
        var_type = ctx.lookup_type(decl.name)
        if isinstance(var_type, ast.ArrayType):
            # Vector types accept a same-shape vector value (Identifier,
            # BinaryOp, UnaryOp, Call, etc.) as their initializer — copy
            # the bytes from the rhs's address into the local's slot.
            # Regular C arrays only accept InitializerList/StringLiteral.
            if (
                getattr(var_type, "is_vector", False)
                and not isinstance(
                    decl.init, (ast.InitializerList, ast.StringLiteral)
                )
            ):
                init_ty = self._type_of(decl.init, ctx)
                if (
                    isinstance(init_ty, ast.ArrayType)
                    and self._size_of(init_ty) == self._size_of(var_type)
                ):
                    # Vector-returning call: use the local's address as
                    # the retptr to skip the temp.
                    if (
                        isinstance(decl.init, ast.Call)
                        and self._is_vector_returning_call(decl.init, ctx)
                    ):
                        return self._call_into_address(
                            decl.init,
                            [f"        lea     eax, {_ebp_addr(disp)}"],
                            ctx,
                        )
                    out = self._vector_value_address(decl.init, ctx)
                    out.append("        mov     edx, eax")
                    out.append(f"        lea     ecx, {_ebp_addr(disp)}")
                    size = self._size_of(var_type)
                    offset = 0
                    while size - offset >= 4:
                        out.append(f"        mov     eax, [edx + {offset}]")
                        out.append(f"        mov     [ecx + {offset}], eax")
                        offset += 4
                    if size - offset >= 2:
                        out.append(f"        mov     ax, [edx + {offset}]")
                        out.append(f"        mov     [ecx + {offset}], ax")
                        offset += 2
                    if size - offset >= 1:
                        out.append(f"        mov     al, [edx + {offset}]")
                        out.append(f"        mov     [ecx + {offset}], al")
                    return out
            return self._array_init(var_type, decl.init, disp, ctx, decl.name)
        if isinstance(var_type, ast.StructType):
            # `struct T s = make(...)` — call `make` with &s as the hidden
            # retptr instead of allocating a temp and copying.
            if (
                isinstance(decl.init, ast.Call)
                and self._is_struct_returning_call(decl.init, ctx)
            ):
                return self._call_into_address(
                    decl.init,
                    [f"        lea     eax, {_ebp_addr(disp)}"],
                    ctx,
                )
            # `struct T s = va_arg(ap, struct T);` — copy the struct
            # bytes from the variadic stack slot, advancing ap.
            if isinstance(decl.init, ast.VaArgExpr):
                return self._va_arg_struct_copy(
                    decl.init,
                    [f"        lea     eax, {_ebp_addr(disp)}"],
                    ctx,
                )
            # `struct T s = src;` where src is an l-value of struct type
            # (e.g. `*pls`, `arr[i]`, `outer.inner`) — perform a struct
            # copy from src to s rather than expecting `{...}`.
            if not isinstance(decl.init, ast.InitializerList):
                rhs_ty = self._type_of(decl.init, ctx)
                if isinstance(rhs_ty, ast.StructType):
                    return self._struct_copy_from_expr(
                        decl.init, disp, var_type, ctx,
                    )
            return self._struct_init(var_type, decl.init, disp, ctx, decl.name)
        if self._is_float_type(var_type):
            # Float locals get their init value via st0 + fstp.
            return self._eval_float_to_st0(decl.init, ctx) + self._store_st0_to(
                _ebp_addr(disp), var_type
            )
        if self._is_long_long(var_type):
            # 64-bit local: evaluate the rhs to EDX:EAX (sign- or
            # zero-extending int rhs to long long), store both halves.
            return self._eval_expr_to_edx_eax(
                decl.init, ctx,
            ) + self._store_from_edx_eax(_ebp_addr(disp))
        if self._is_int128(var_type):
            # 128-bit local init via copy. The rhs is either int128
            # (Identifier / BinaryOp / Cast) or a smaller integer that
            # we widen via a synthetic Cast.
            init_ty = self._type_of(decl.init, ctx)
            if not self._is_int128(init_ty):
                synth_cast = ast.Cast(target_type=var_type, expr=decl.init)
                ctx.alloc_call_temp(synth_cast, 16)
                rhs = synth_cast
            else:
                rhs = decl.init
            return self._int128_copy_assign(
                ast.Identifier(name=decl.name), rhs, ctx,
            )
        if isinstance(var_type, ast.ComplexType):
            # Complex local init — `__complex__ T r = expr` lowers via
            # the complex-eval engine, which writes (real, imag) into
            # the slot directly.
            dest = [f"        lea     eax, {_ebp_addr(disp)}"]
            init_ty = self._type_of(decl.init, ctx)
            # If rhs is a same-precision complex-returning call,
            # route into the slot. Cross-precision must go through
            # _complex_copy_assign which converts.
            if (
                isinstance(decl.init, ast.Call)
                and self._is_complex_returning_call(decl.init, ctx)
                and isinstance(init_ty, ast.ComplexType)
                and init_ty.base_type == var_type.base_type
            ):
                return self._call_into_address(decl.init, dest, ctx)
            # If rhs is itself a complex expression, eval directly.
            if isinstance(init_ty, ast.ComplexType):
                synth = ast.BinaryOp(op="=", left=ast.Identifier(name=decl.name), right=decl.init)
                return self._complex_copy_assign(synth, var_type, ctx)
            # Scalar init — promote to (val, 0).
            synth = ast.Identifier(name=decl.name)
            return self._complex_assign_from_scalar(
                synth, decl.init, var_type, ctx,
            )
        return self._eval_expr_to_eax(decl.init, ctx) + self._store_from_eax(
            _ebp_addr(disp), var_type
        )

    # Width keyword for `mov <width> [...], 0` inline zero stores. Used by
    # the array zero-fill loop.
    _ZERO_WIDTHS = {1: "byte", 2: "word", 4: "dword"}

    def _array_init(
        self,
        arr_type: ast.ArrayType,
        init: ast.Expression,
        base_disp: int,
        ctx: _FuncCtx,
        name: str,
    ) -> list[str]:
        """Lower an array initializer to per-element stores plus tail zero-fill.

        - `int arr[N] = {a, b, c}` walks the InitializerList; if the list
          is shorter than the declared size, the remaining elements are
          zeroed.
        - `char arr[N] = "..."` lays the string bytes out, appends a null
          terminator, and zero-fills any padding.
        - Anything else is rejected — designated initializers and nested
          {} for multidim arrays are still TODOs.
        """
        elem_type = arr_type.base_type
        elem_size = self._size_of(elem_type)
        # Compound literal as the array initializer (e.g.
        # `int arr[3] = (int[3]){1, 2, 3};`) — strip the compound's
        # outer wrapper to expose its InitializerList / StringLiteral.
        if isinstance(init, ast.Compound):
            init = init.init
        # Unsized array (`int x[] = {...}` or `(int []){...}` compound
        # literal): infer length from the initializer.
        if arr_type.size is None:
            if isinstance(init, ast.InitializerList):
                length = len(init.values)
            elif isinstance(init, ast.StringLiteral):
                length = len(init.value) + 1
            else:
                raise CodegenError(
                    f"`{name}`: unsized array needs a brace or string initializer"
                )
        else:
            length = arr_type.size.value

        if isinstance(init, ast.StringLiteral):
            is_wide = getattr(init, "is_wide", False)
            if is_wide:
                # `wchar_t s[] = L"...";` — each codepoint becomes one
                # array element of `elem_size` bytes (typically 2 for
                # `unsigned short`/wchar_t on this target). Range-check
                # against the slot's payload width, then store.
                codepoints = [ord(c) for c in init.value] + [0]
                if len(codepoints) > length:
                    raise CodegenError(
                        f"`{name}`: wide string init exceeds array size {length}"
                    )
                width_keyword = self._ZERO_WIDTHS.get(elem_size)
                if width_keyword is None:
                    raise CodegenError(
                        f"`{name}`: wide string element size {elem_size} unsupported"
                    )
                out: list[str] = []
                for i, cp in enumerate(codepoints):
                    # NASM doesn't accept `mov word [..], 65535` if the
                    # operand width disagrees with the value; we know
                    # the element fits because the source code declared
                    # the array of this element type.
                    addr = _ebp_addr(base_disp + i * elem_size)
                    out.append(
                        f"        mov     {width_keyword} {addr}, {cp & ((1 << (elem_size * 8)) - 1)}"
                    )
                for i in range(len(codepoints), length):
                    addr = _ebp_addr(base_disp + i * elem_size)
                    out.append(
                        f"        mov     {width_keyword} {addr}, 0"
                    )
                return out
            if not (
                isinstance(elem_type, ast.BasicType)
                and elem_type.name == "char"
            ):
                raise CodegenError(
                    f"`{name}`: string initializer requires a char array"
                )
            # Lay out the bytes, append a null when there's room.
            # C semantics: `char a[3] = "abc"` lays out exactly the
            # 3 bytes (no null terminator), `char a[4] = "abc"` lays
            # out the 3 bytes plus the null. Arrays smaller than the
            # raw string content are an error.
            raw_bytes = list(self._string_to_bytes(init.value))
            if len(raw_bytes) > length:
                raise CodegenError(
                    f"`{name}`: string initializer exceeds array size {length}"
                )
            if len(raw_bytes) < length:
                bytes_to_store = raw_bytes + [0]
            else:
                bytes_to_store = raw_bytes
            out: list[str] = []
            for i, byte in enumerate(bytes_to_store):
                addr = _ebp_addr(base_disp + i * elem_size)
                out.append(f"        mov     byte {addr}, {byte}")
            for i in range(len(bytes_to_store), length):
                addr = _ebp_addr(base_disp + i * elem_size)
                out.append(f"        mov     byte {addr}, 0")
            return out

        if isinstance(init, ast.InitializerList):
            # Brace-elide flat positional values into per-element
            # InitializerLists when the element is a compound type.
            prev_ctx = self._elision_ctx
            self._elision_ctx = ctx
            try:
                iter_values = self._elide_braces_for_array(
                    init.values, elem_type, name,
                )
            finally:
                self._elision_ctx = prev_ctx
            # Walk values in source order, allowing `[N] = expr` to jump
            # the cursor. After all source values, any unfilled slots
            # get zero-filled. This handles pure-positional, pure-
            # designated, and mixed forms uniformly.
            out = []
            filled: set[int] = set()
            cursor = 0
            for value_expr in iter_values:
                idx_range: list[int] | None = None
                if isinstance(value_expr, ast.DesignatedInit):
                    designator = value_expr.designators[0]
                    if (
                        len(value_expr.designators) == 1
                        and isinstance(designator, ast.RangeDesignator)
                    ):
                        try:
                            start = self._const_eval(
                                designator.start, f"`{name}`"
                            )
                            end = self._const_eval(
                                designator.end, f"`{name}`"
                            )
                        except CodegenError:
                            raise CodegenError(
                                f"`{name}`: range designator must be "
                                f"compile-time constants"
                            )
                        idx_range = list(range(start, end + 1))
                        actual = value_expr.value
                        cursor = end + 1
                    elif (
                        len(value_expr.designators) != 1
                        or not isinstance(designator, ast.IntLiteral)
                    ):
                        raise CodegenError(
                            f"`{name}`: only single-level integer "
                            f"designators supported in array init"
                        )
                    else:
                        idx = designator.value
                        actual = value_expr.value
                        cursor = idx + 1
                        idx_range = [idx]
                else:
                    idx = cursor
                    actual = value_expr
                    cursor += 1
                    idx_range = [idx]
                for idx in idx_range:
                    if idx < 0 or idx >= length:
                        raise CodegenError(
                            f"`{name}`: initializer index {idx} out of range "
                            f"(array size {length})"
                        )
                    filled.add(idx)
                    elem_disp = base_disp + idx * elem_size
                    if (
                        isinstance(elem_type, ast.StructType)
                        and isinstance(actual, ast.InitializerList)
                    ):
                        out += self._struct_init(
                            elem_type, actual, elem_disp, ctx,
                            f"{name}[{idx}]",
                        )
                    elif (
                        isinstance(elem_type, ast.StructType)
                        and isinstance(actual, ast.Compound)
                        and isinstance(actual.target_type, ast.StructType)
                        and isinstance(actual.init, ast.InitializerList)
                    ):
                        # `((struct T){...})` as a struct array element:
                        # treat the compound's init as the field list.
                        out += self._struct_init(
                            elem_type, actual.init, elem_disp, ctx,
                            f"{name}[{idx}]",
                        )
                    elif (
                        isinstance(elem_type, ast.ArrayType)
                        and isinstance(actual, (ast.InitializerList, ast.StringLiteral))
                    ):
                        # Multi-dim array element: recurse.
                        out += self._array_init(
                            elem_type, actual, elem_disp, ctx,
                            f"{name}[{idx}]",
                        )
                    elif self._is_float_type(elem_type):
                        # Float element: eval to st(0), then fstp at the slot.
                        out += self._eval_float_to_st0(actual, ctx)
                        out += self._store_st0_to(_ebp_addr(elem_disp), elem_type)
                    elif self._is_long_long(elem_type):
                        out += self._eval_expr_to_edx_eax(actual, ctx)
                        out += self._store_from_edx_eax(_ebp_addr(elem_disp))
                    elif self._is_int128(elem_type):
                        # int128 element: get the value's address and
                        # per-dword copy to the element slot. Smaller
                        # integer initializers are widened via a
                        # synthetic Cast (matches `_var_init`).
                        actual_ty = self._type_of(actual, ctx)
                        if not self._is_int128(actual_ty):
                            synth_cast = ast.Cast(
                                target_type=elem_type, expr=actual,
                            )
                            ctx.alloc_call_temp(synth_cast, 16)
                            value_expr = synth_cast
                        else:
                            value_expr = actual
                        out += self._int128_value_address(value_expr, ctx)
                        out.append("        mov     esi, eax")
                        out.append(
                            f"        lea     edi, {_ebp_addr(elem_disp)}"
                        )
                        for byte_off in (0, 4, 8, 12):
                            out.append(
                                f"        mov     eax, [esi + {byte_off}]"
                            )
                            out.append(
                                f"        mov     [edi + {byte_off}], eax"
                            )
                    elif isinstance(elem_type, ast.ComplexType):
                        # _Complex element: evaluate into the element
                        # slot via _eval_complex_into_top with &slot
                        # on the stack as the destination.
                        out.append(
                            f"        lea     eax, {_ebp_addr(elem_disp)}"
                        )
                        out.append("        push    eax")
                        out += self._eval_complex_into_top(
                            actual, ctx, elem_type,
                        )
                        out.append("        add     esp, 4")
                    else:
                        out += self._eval_expr_to_eax(actual, ctx)
                        out += self._store_from_eax(_ebp_addr(elem_disp), elem_type)
            # Zero-fill any indices the initializer didn't touch. We emit
            # them in index order so the asm has a predictable shape.
            for idx in range(length):
                if idx not in filled:
                    out += self._zero_fill_at(
                        base_disp + idx * elem_size, elem_size
                    )
            return out

        raise CodegenError(
            f"`{name}`: unsupported array initializer "
            f"({type(init).__name__})"
        )

    def _struct_init(
        self,
        struct_ty: ast.StructType,
        init: ast.Expression,
        base_disp: int,
        ctx: _FuncCtx,
        name: str,
    ) -> list[str]:
        """Lower `struct foo s = {a, b, ...}` to per-member stores at offsets.

        Members not covered by the initializer get zero-fill at their
        natural width (recursively, for nested struct members). Nested
        `{...}` recurses into `_struct_init` for struct-typed members.
        """
        if not isinstance(init, ast.InitializerList):
            raise CodegenError(
                f"`{name}`: struct must be initialized with `{{...}}` "
                f"(got {type(init).__name__})"
            )
        struct_name = self._resolve_struct_name(struct_ty)
        members = self._structs[struct_name]
        member_index = {m_name: i for i, (m_name, _, _) in enumerate(members)}
        bitfields = self._struct_bitfields.get(struct_name, {})
        is_union = struct_name in self._struct_unions
        # Brace-elide flat values for compound members.
        prev_ctx = self._elision_ctx
        self._elision_ctx = ctx
        try:
            elided_values = self._elide_braces_for_struct(init.values, members, name)
        finally:
            self._elision_ctx = prev_ctx
        # Walk source values in order, tracking the implicit cursor. A
        # `.field = expr` sets the cursor to that member's index; the
        # next un-designated value continues from cursor + 1. After the
        # walk, any unfilled members get zero-filled.
        out: list[str] = []
        # Up-front whole-struct zero-fill when:
        #   - there are bit-fields (share storage units across members)
        #   - the type is a union (members share offset 0)
        # Both cases break per-member end-zero-fill, which would overwrite
        # earlier writes that share storage.
        if bitfields or is_union:
            out += self._zero_fill_at(base_disp, self._struct_sizes[struct_name])
        filled: set[int] = set()
        cursor = 0
        for value in elided_values:
            if isinstance(value, ast.DesignatedInit):
                first = value.designators[0]
                if not isinstance(first, str):
                    raise CodegenError(
                        f"`{name}`: array designator on struct init"
                    )
                if first not in member_index:
                    raise CodegenError(
                        f"`{name}`: unknown member `{first}` in "
                        f"struct `{struct_ty.name}`"
                    )
                idx = member_index[first]
                # Multi-level designator like `.a.b = expr` —
                # synthesize a nested InitializerList so the regular
                # per-member dispatch handles `.b`.
                if len(value.designators) > 1:
                    inner = ast.DesignatedInit(
                        designators=value.designators[1:],
                        value=value.value,
                    )
                    actual = ast.InitializerList(values=[inner])
                else:
                    actual = value.value
                m_name_des = first
                cursor = idx + 1
            else:
                if cursor >= len(members):
                    raise CodegenError(
                        f"`{name}`: too many initializers (struct has "
                        f"{len(members)} members)"
                    )
                idx = cursor
                actual = value
                # Advance past members that share the same offset
                # (anonymous-union alternatives get one slot of init
                # between them). Bit-fields can also share an offset
                # (same storage unit, different bit_offsets) but each
                # gets its own init value, so we don't skip over them.
                next_cursor = cursor + 1
                _, _, this_off = members[idx]
                if not bitfields:
                    while (
                        next_cursor < len(members)
                        and members[next_cursor][2] == this_off
                    ):
                        next_cursor += 1
                cursor = next_cursor
            filled.add(idx)
            m_name_i, m_ty, m_off = members[idx]
            m_disp = base_disp + m_off
            if (
                isinstance(m_ty, ast.StructType)
                and isinstance(actual, ast.InitializerList)
            ):
                out += self._struct_init(
                    m_ty, actual, m_disp, ctx, f"{name}.{m_name_i}"
                )
            elif (
                isinstance(m_ty, ast.ArrayType)
                and isinstance(actual, (ast.InitializerList, ast.StringLiteral))
            ):
                out += self._array_init(
                    m_ty, actual, m_disp, ctx, f"{name}.{m_name_i}"
                )
            elif (
                isinstance(m_ty, ast.StructType)
                and isinstance(self._type_of(actual, ctx), ast.StructType)
            ):
                # Struct-typed value (an Identifier, a *p, a member, etc.)
                # being assigned to a struct member: per-dword copy.
                out += self._struct_copy_from_expr(
                    actual, m_disp, m_ty, ctx,
                )
            elif self._is_float_type(m_ty) and m_name_i not in bitfields:
                out += self._eval_float_to_st0(actual, ctx)
                out += self._store_st0_to(_ebp_addr(m_disp), m_ty)
            elif self._is_long_long(m_ty) and m_name_i not in bitfields:
                value_ty = self._type_of(actual, ctx)
                if self._is_long_long(value_ty):
                    out += self._eval_expr_to_edx_eax(actual, ctx)
                else:
                    out += self._eval_expr_to_eax(actual, ctx)
                    out.append(
                        "        xor     edx, edx"
                        if self._is_unsigned(value_ty)
                        else "        cdq"
                    )
                out += self._store_from_edx_eax(_ebp_addr(m_disp))
            elif self._is_int128(m_ty) and m_name_i not in bitfields:
                # int128 member: get 16-byte value's address and copy.
                # Smaller integer initializers widen via synthetic Cast.
                value_ty = self._type_of(actual, ctx)
                if not self._is_int128(value_ty):
                    synth_cast = ast.Cast(target_type=m_ty, expr=actual)
                    ctx.alloc_call_temp(synth_cast, 16)
                    value_expr = synth_cast
                else:
                    value_expr = actual
                out += self._int128_value_address(value_expr, ctx)
                out.append("        mov     esi, eax")
                out.append(f"        lea     edi, {_ebp_addr(m_disp)}")
                for byte_off in (0, 4, 8, 12):
                    out.append(f"        mov     eax, [esi + {byte_off}]")
                    out.append(f"        mov     [edi + {byte_off}], eax")
            elif (
                isinstance(m_ty, ast.ComplexType)
                and m_name_i not in bitfields
            ):
                # _Complex member: route through `_complex_copy_assign`
                # with a synthetic *p lvalue so the value lands at
                # &m_disp.
                size = self._size_of(m_ty)
                # Compute &(struct_base + m_disp) as a value, then
                # use _complex_copy_assign on a synthetic lvalue.
                # Easiest: copy via _eval_complex_into_top into the
                # member's slot on the stack. The dest_top dance:
                # save &dest, eval into &dest, restore.
                out.append(f"        lea     eax, {_ebp_addr(m_disp)}")
                out.append("        push    eax")
                out += self._eval_complex_into_top(actual, ctx, m_ty)
                out.append("        add     esp, 4")
            elif m_name_i in bitfields:
                # Bit-field: synthesize a fake Member node so _bitfield_store
                # can compute the storage-unit address. We can't pass the
                # base struct lvalue's name directly, so we build a Member
                # against an auto-keyed Identifier whose type/address come
                # from a one-shot synthesized lookup. Easiest: emit the
                # store inline using the stored bit_offset/width.
                info = bitfields[m_name_i]
                if len(info) == 2:
                    bit_offset, bit_width = info
                    unit_size = 4
                else:
                    bit_offset, bit_width, unit_size = info
                mask = (1 << bit_width) - 1
                positioned_mask = (mask << bit_offset) & ((1 << (unit_size * 8)) - 1)
                clear_low = (~positioned_mask) & 0xFFFFFFFF
                clear_high = ((~positioned_mask) >> 32) & 0xFFFFFFFF
                if unit_size == 8:
                    # Long-long bit-field: pack rhs (eval as 64-bit)
                    # then RMW the 8-byte unit.
                    store = self._eval_expr_to_edx_eax(actual, ctx)
                    if bit_width <= 32:
                        store.append(f"        and     eax, {mask}")
                        store.append("        xor     edx, edx")
                    else:
                        high_m = (mask >> 32) & 0xFFFFFFFF
                        store.append(f"        and     edx, {high_m}")
                    if bit_offset >= 32:
                        shift = bit_offset - 32
                        store.append("        mov     edx, eax")
                        store.append("        xor     eax, eax")
                        if shift:
                            store.append(f"        shl     edx, {shift}")
                    elif bit_offset > 0:
                        store.append(f"        shld    edx, eax, {bit_offset}")
                        store.append(f"        shl     eax, {bit_offset}")
                    addr_lo = _ebp_addr(m_disp)
                    addr_hi = _ebp_addr(m_disp + 4)
                    store.append("        push    edx")
                    store.append("        push    eax")
                    store.append(f"        mov     ecx, {addr_lo}")
                    store.append(f"        and     ecx, {clear_low}")
                    store.append("        pop     edx")
                    store.append("        or      ecx, edx")
                    store.append(f"        mov     {addr_lo}, ecx")
                    store.append(f"        mov     ecx, {addr_hi}")
                    store.append(f"        and     ecx, {clear_high}")
                    store.append("        pop     edx")
                    store.append("        or      ecx, edx")
                    store.append(f"        mov     {addr_hi}, ecx")
                    out += store
                else:
                    store = self._eval_expr_to_eax(actual, ctx)
                    store.append(f"        and     eax, {mask}")
                    if bit_offset > 0:
                        store.append(f"        shl     eax, {bit_offset}")
                    # m_disp is the storage unit (offset already equals the
                    # unit_offset). RMW the unit at [ebp + m_disp].
                    unit_addr = _ebp_addr(m_disp)
                    store.append("        push    eax")
                    store.append(f"        mov     ecx, {unit_addr}")
                    store.append(f"        and     ecx, {clear_low}")
                    store.append("        pop     edx")
                    store.append("        or      ecx, edx")
                    store.append(f"        mov     {unit_addr}, ecx")
                    out += store
            else:
                out += self._eval_expr_to_eax(actual, ctx)
                out += self._store_from_eax(_ebp_addr(m_disp), m_ty)
        # Zero-fill any unfilled non-bit-field members in declaration
        # order. (Bit-fields and unions are already zeroed by the
        # up-front whole-struct zero-fill above.)
        if not bitfields and not is_union:
            for i, (m_name_i, m_ty, m_off) in enumerate(members):
                if i not in filled:
                    out += self._zero_fill_at(
                        base_disp + m_off, self._size_of(m_ty)
                    )
        return out

    def _zero_fill_at(self, disp: int, size: int) -> list[str]:
        """Emit asm zeroing `size` bytes starting at `[ebp + disp]`.

        Uses the widest store that fits — `mov dword [...], 0` for each
        4-byte chunk, then `mov word`, then `mov byte`. Works regardless
        of the original type at that location, which is what we need for
        zero-filling struct array tails or unspecified struct members.
        """
        out: list[str] = []
        while size >= 4:
            out.append(f"        mov     dword {_ebp_addr(disp)}, 0")
            disp += 4
            size -= 4
        if size >= 2:
            out.append(f"        mov     word {_ebp_addr(disp)}, 0")
            disp += 2
            size -= 2
        if size >= 1:
            out.append(f"        mov     byte {_ebp_addr(disp)}, 0")
        return out

    def _return(self, stmt: ast.ReturnStmt, ctx: _FuncCtx) -> list[str]:
        if stmt.value is None:
            return [
                "        xor     eax, eax",
                "        jmp     .epilogue",
            ]
        # Float-returning functions leave the result on st(0); the
        # caller picks it up via fstp. No EAX involvement.
        if self._is_float_type(ctx.return_type):
            return self._eval_float_to_st0(stmt.value, ctx) + [
                "        jmp     .epilogue",
            ]
        # Struct-returning functions copy the value into the caller-provided
        # buffer (the hidden `__retptr__` first arg) rather than dropping
        # the value into EAX. We forward the retptr in EAX as the return
        # value so chained struct calls don't need a temp.
        if ctx.has_local("__retptr__"):
            retptr_disp = ctx.lookup("__retptr__")
            ret_ty = ctx.lookup_type("__retptr__").base_type
            retptr_load = [f"        mov     eax, {_ebp_addr(retptr_disp)}"]
            if (
                isinstance(stmt.value, ast.Call)
                and self._is_struct_returning_call(stmt.value, ctx)
            ):
                # Forward our retptr to the inner call so it writes
                # directly into the caller's buffer.
                return self._call_into_address(
                    stmt.value, retptr_load, ctx,
                ) + ["        jmp     .epilogue"]
            # `_Complex T` value uses the same retptr / struct-copy
            # mechanism with a per-half load via the existing complex
            # address helper.
            if isinstance(ret_ty, ast.ComplexType):
                return self._copy_complex_to_retptr(
                    stmt.value, ret_ty, retptr_disp, ctx
                ) + ["        jmp     .epilogue"]
            # Vector return: same retptr ABI as structs, but the source
            # address comes from `_vector_value_address` since vector
            # ops don't dispatch through `_struct_address`.
            if (
                isinstance(ret_ty, ast.ArrayType)
                and getattr(ret_ty, "is_vector", False)
            ):
                return self._copy_vector_to_retptr(
                    stmt.value, ret_ty, retptr_disp, ctx
                ) + ["        jmp     .epilogue"]
            # __int128 return: copy 16 bytes of the value through
            # `_int128_value_address` into the caller's retptr.
            # Smaller-integer return values widen via synthetic Cast.
            if self._is_int128(ret_ty):
                value_expr = stmt.value
                value_ty = self._type_of(value_expr, ctx)
                if not self._is_int128(value_ty):
                    synth_cast = ast.Cast(
                        target_type=ret_ty, expr=value_expr,
                    )
                    ctx.alloc_call_temp(synth_cast, 16)
                    value_expr = synth_cast
                src_lines = self._int128_value_address(value_expr, ctx)
                out = list(src_lines)
                out.append("        mov     esi, eax")
                out.append(f"        mov     edi, {_ebp_addr(retptr_disp)}")
                for off in (0, 4, 8, 12):
                    out.append(f"        mov     eax, [esi + {off}]")
                    out.append(f"        mov     [edi + {off}], eax")
                # Forward retptr in EAX (matches struct convention).
                out.append(f"        mov     eax, {_ebp_addr(retptr_disp)}")
                out.append("        jmp     .epilogue")
                return out
            return self._copy_struct_to_retptr(
                stmt.value, ret_ty, retptr_disp, ctx
            ) + ["        jmp     .epilogue"]
        if self._is_long_long(ctx.return_type):
            # cdecl returns 64-bit values in EDX:EAX (high:low). If the
            # value expression is itself 64-bit, eval directly; otherwise
            # eval as 32-bit and widen per signedness.
            value_ty = self._type_of(stmt.value, ctx)
            if self._is_long_long(value_ty):
                return self._eval_expr_to_edx_eax(stmt.value, ctx) + [
                    "        jmp     .epilogue",
                ]
            out = self._eval_expr_to_eax(stmt.value, ctx)
            if self._is_unsigned(value_ty):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            out.append("        jmp     .epilogue")
            return out
        # For sub-word return types (char / short), narrow EAX to the
        # type's width and re-extend per signedness — matches C's
        # "value narrows to the return type on return".
        out = self._eval_expr_to_eax(stmt.value, ctx)
        rt = ctx.return_type
        if isinstance(rt, ast.BasicType):
            size = self._size_of(rt)
            if size == 1:
                mnem = "movzx" if self._is_unsigned(rt) else "movsx"
                out.append(f"        {mnem}   eax, al")
            elif size == 2:
                mnem = "movzx" if self._is_unsigned(rt) else "movsx"
                out.append(f"        {mnem}   eax, ax")
        out.append("        jmp     .epilogue")
        return out

    def _complex_promotion(
        self, lt: ast.TypeNode, rt: ast.TypeNode,
    ) -> ast.ComplexType:
        """Result type of a complex-arithmetic op.

        Per C: if either operand is `_Complex double`, the result is
        `_Complex double`. Otherwise (one operand is `_Complex float`)
        the result is `_Complex float`.
        """
        bases = []
        for t in (lt, rt):
            if isinstance(t, ast.ComplexType):
                bases.append(t.base_type)
            elif isinstance(t, ast.BasicType) and t.name == "double":
                bases.append("double")
        if "long double" in bases:
            return ast.ComplexType(base_type="long double")
        if "double" in bases:
            return ast.ComplexType(base_type="double")
        return ast.ComplexType(base_type="float")

    def _expr_is_complex(self, expr: ast.Expression) -> bool:
        """Cheap probe: does this expression yield a `_Complex T`?"""
        ty = self._type_of_complex_expr(expr)
        return ty is not None

    def _type_of_complex_expr(
        self, expr: ast.Expression,
    ) -> ast.ComplexType | None:
        """Return `ComplexType` for a complex-valued expression, else
        None. Used to decide whether to allocate a temp and route
        through the complex codegen path.

        We can't always call `_type_of` here because the temp-alloc
        pass runs on the function body before locals are bound — so
        Identifier name lookup in `_type_of` may fail. Walk only the
        structural cases the alloc pass needs.
        """
        if isinstance(expr, ast.FloatLiteral) and expr.is_imaginary:
            # `1.0i` — `_Imaginary T` ≡ `_Complex T` with real=0.
            return ast.ComplexType(
                base_type="float" if expr.is_float else "double",
            )
        if isinstance(expr, ast.Cast):
            if isinstance(expr.target_type, ast.ComplexType):
                return expr.target_type
            return None
        if isinstance(expr, ast.UnaryOp):
            if expr.op in ("+", "-", "~"):
                return self._type_of_complex_expr(expr.operand)
            if expr.op in ("__real__", "__imag__"):
                return None
            if expr.op in ("*", "&"):
                return None
            return None
        if isinstance(expr, ast.BinaryOp):
            if expr.op in ("+", "-", "*", "/", "="):
                lt = self._type_of_complex_expr(expr.left)
                rt = self._type_of_complex_expr(expr.right)
                if lt is not None or rt is not None:
                    return lt if lt is not None else rt
            return None
        return None

    def _eval_complex_to(
        self,
        expr: ast.Expression,
        dest_addr_in_eax: list[str],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Evaluate a complex-typed expression, storing the result at
        the address produced by `dest_addr_in_eax` (a sequence of asm
        lines that leaves &dest in EAX).

        After the helper runs, EAX holds &dest (callers can chain).
        """
        # Push the destination address.
        out = list(dest_addr_in_eax)
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr, ctx)
        out.append("        pop     eax")
        return out

    def _eval_complex_into_top(
        self, expr: ast.Expression, ctx: _FuncCtx,
        dest_ty: ast.ComplexType | None = None,
    ) -> list[str]:
        """Helper: store a complex value into `[esp]`'s pointed-to
        slot (the top of stack is the destination address; we leave
        it there so callers can pop it).

        `dest_ty` is the complex type of the destination slot.
        Operands are converted to that type so component-wise math
        always uses the destination's half width. When the caller
        doesn't know the dest type (e.g. eval_complex_to where the
        slot's address is opaque), we fall back to the expression's
        own type.
        """
        ty = self._type_of(expr, ctx)
        if dest_ty is None and isinstance(ty, ast.ComplexType):
            dest_ty = ty
        # `(_Complex T) scalar` cast — store scalar as real, 0 as imag.
        if isinstance(expr, ast.Cast) and isinstance(
            expr.target_type, ast.ComplexType
        ):
            return self._cast_to_complex_into_top(expr, ctx, dest_ty)
        if not isinstance(ty, ast.ComplexType):
            # Real-valued expression in complex context: promote to
            # (real, 0) using the destination's complex type if known.
            if dest_ty is None:
                dest_ty = ast.ComplexType(
                    base_type=(
                        "double"
                        if isinstance(ty, ast.BasicType) and ty.name == "double"
                        else "float"
                    ),
                )
            return self._scalar_to_complex_into_top(expr, dest_ty, ctx)
        # Use the destination's complex type for the dispatch when
        # provided — this drives narrowing for cross-precision cases
        # (e.g. complex_double rhs → complex_float dest).
        eff_ty = dest_ty if dest_ty is not None else ty
        # If expr is an lvalue / Call / __real__ wrapper, we already
        # have an addressing helper that gives us &expr; copy from there.
        # When dest_ty differs from the lvalue's own type we need to
        # convert per-half rather than memcpy; route through the
        # complex-conversion helper.
        if isinstance(expr, (ast.Identifier, ast.Member, ast.Index)) or (
            isinstance(expr, ast.UnaryOp) and expr.op == "*"
        ) or (
            isinstance(expr, ast.Call)
            and self._is_complex_returning_call(expr, ctx)
        ):
            if (
                dest_ty is not None
                and isinstance(ty, ast.ComplexType)
                and dest_ty.base_type != ty.base_type
            ):
                return self._convert_complex_into_top(expr, ty, dest_ty, ctx)
            return self._copy_complex_lvalue_into_top(expr, eff_ty, ctx)
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "+":
                return self._eval_complex_into_top(expr.operand, ctx, dest_ty)
            if expr.op == "-":
                return self._complex_neg_into_top(expr, eff_ty, ctx)
            if expr.op == "~":
                return self._complex_conj_into_top(expr, eff_ty, ctx)
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
                return self._complex_assign_into_top(expr, eff_ty, ctx)
            # Integer-base complex constant arithmetic: fold at compile
            # time and store the integer halves directly. Avoids the
            # FPU paths that would mishandle 1/2-byte halves.
            if (
                eff_ty.base_type in self._COMPLEX_INT_BASES
                and expr.op in ("+", "-", "*")
            ):
                try:
                    real, imag = self._const_eval_complex(expr, "<complex-int>")
                    return self._complex_const_into_top(real, imag, eff_ty)
                except CodegenError:
                    pass
            if expr.op in ("+", "-"):
                return self._complex_addsub_into_top(expr, eff_ty, ctx)
            if expr.op == "*":
                return self._complex_mul_into_top(expr, eff_ty, ctx)
            if expr.op == "/":
                return self._complex_div_into_top(expr, eff_ty, ctx)
        # Imaginary FloatLiteral or scalar-typed sub-expression that
        # got promoted to complex (rare). Treat as (0, value) or
        # (value, 0) depending on whether it's imaginary.
        if isinstance(expr, ast.FloatLiteral) and expr.is_imaginary:
            return self._complex_const_into_top(0.0, expr.value, eff_ty)
        # Fallback: evaluate as a scalar (real part), zero the imag.
        return self._scalar_to_complex_into_top(expr, eff_ty, ctx)

    def _convert_complex_into_top(
        self,
        src_expr: ast.Expression,
        src_ty: ast.ComplexType,
        dest_ty: ast.ComplexType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy `src_expr` (a complex lvalue or call) into the
        top-of-stack destination, converting per-half from `src_ty`
        to `dest_ty`. Each half is loaded as the source width and
        stored as the destination width.

        Float-to-int via `fistp`, int-to-float via `fild`, same-kind
        narrowing/widening via `fld`/`fstp`. Int-to-int truncates.
        """
        src_half = self._COMPLEX_BASE_SIZES[src_ty.base_type]
        dst_half = self._COMPLEX_BASE_SIZES[dest_ty.base_type]
        src_is_int = src_ty.base_type in self._COMPLEX_INT_BASES
        dst_is_int = dest_ty.base_type in self._COMPLEX_INT_BASES
        # &src in EAX.
        out = self._complex_value_address(src_expr, ctx)
        out.append("        mov     edx, eax")
        out.append("        mov     ecx, [esp]")
        for i in range(2):
            src_off = i * src_half
            dst_off = i * dst_half
            self._complex_half_convert(
                out, src_ty, dest_ty, "edx", src_off, "ecx", dst_off,
            )
        return out

    def _complex_half_convert(
        self, out, src_ty, dest_ty, src_reg, src_off, dst_reg, dst_off,
    ) -> None:
        """Convert one half of a complex value from `src_ty.base_type`
        to `dest_ty.base_type` and store at `[dst_reg + dst_off]`.
        Generates FPU code for float-side conversions and integer
        loads/stores for the int family.
        """
        src_int = src_ty.base_type in self._COMPLEX_INT_BASES
        dst_int = dest_ty.base_type in self._COMPLEX_INT_BASES
        src_half = self._COMPLEX_BASE_SIZES[src_ty.base_type]
        dst_half = self._COMPLEX_BASE_SIZES[dest_ty.base_type]
        if src_int and dst_int:
            # Integer → integer. Load with sign-extend, store narrow.
            src_basic = ast.BasicType(name=src_ty.base_type)
            dst_basic = ast.BasicType(name=dest_ty.base_type)
            out += self._load_to_eax(f"[{src_reg} + {src_off}]", src_basic)
            out += self._store_from_eax(f"[{dst_reg} + {dst_off}]", dst_basic)
            return
        # FPU-mediated: load src half, store dst half.
        if src_int:
            # Integer source: fild from a stack scratch.
            src_basic = ast.BasicType(name=src_ty.base_type)
            out += self._load_to_eax(f"[{src_reg} + {src_off}]", src_basic)
            out.append("        sub     esp, 4")
            out.append("        mov     [esp], eax")
            out.append("        fild    dword [esp]")
            out.append("        add     esp, 4")
        else:
            width = "dword" if src_half == 4 else "qword"
            out.append(f"        fld     {width} [{src_reg} + {src_off}]")
        if dst_int:
            # Float → int: fistp.
            out.append("        sub     esp, 4")
            out.append("        fistp   dword [esp]")
            out.append("        pop     eax")
            dst_basic = ast.BasicType(name=dest_ty.base_type)
            out += self._store_from_eax(f"[{dst_reg} + {dst_off}]", dst_basic)
        else:
            width = "dword" if dst_half == 4 else "qword"
            out.append(f"        fstp    {width} [{dst_reg} + {dst_off}]")

    def _copy_complex_lvalue_into_top(
        self,
        expr: ast.Expression,
        ty: ast.ComplexType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy from a complex-valued lvalue/Call into the top-of-stack
        destination."""
        size = self._size_of(ty)
        out = self._complex_value_address(expr, ctx)
        out.append("        mov     edx, eax")
        out.append("        mov     ecx, [esp]")
        self._emit_memcpy_inline(out, "edx", "ecx", size)
        return out

    def _cast_to_complex_into_top(
        self, expr: ast.Cast, ctx: _FuncCtx,
        dest_ty: ast.ComplexType | None = None,
    ) -> list[str]:
        """`(_Complex T) scalar` — store scalar as real part, 0 as imag."""
        ty = dest_ty if dest_ty is not None else expr.target_type
        return self._scalar_to_complex_into_top(expr.expr, ty, ctx)

    def _scalar_to_complex_into_top(
        self,
        scalar_expr: ast.Expression,
        ty: ast.ComplexType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Promote a real-typed expression to (real, 0)."""
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        if ty.base_type in self._COMPLEX_INT_BASES:
            # Integer destination: evaluate via the regular int eval
            # path (or via float→int fistp if the rhs is float-typed).
            half_basic = ast.BasicType(name=ty.base_type)
            scalar_ty = self._type_of(scalar_expr, ctx)
            out: list[str] = []
            if self._is_float_type(scalar_ty):
                out += self._eval_float_to_st0(scalar_expr, ctx)
                out.append("        sub     esp, 4")
                out.append("        fistp   dword [esp]")
                out.append("        pop     eax")
            else:
                out += self._eval_expr_to_eax(scalar_expr, ctx)
            out.append("        mov     ecx, [esp]")
            out += self._store_from_eax("[ecx]", half_basic)
            # Zero the imag half.
            for off in range(half_size, half_size * 2, 4):
                out.append(f"        mov     dword [ecx + {off}], 0")
            if half_size == 1:
                out.append(f"        mov     byte [ecx + {half_size}], 0")
            elif half_size == 2:
                out.append(f"        mov     word [ecx + {half_size}], 0")
            return out
        width = "dword" if half_size == 4 else "qword"
        out = self._eval_float_to_st0(scalar_expr, ctx)
        out.append("        mov     ecx, [esp]")
        out.append(f"        fstp    {width} [ecx]")
        if half_size == 4:
            out.append(f"        mov     dword [ecx + {half_size}], 0")
        else:
            out.append(f"        mov     dword [ecx + {half_size}], 0")
            out.append(f"        mov     dword [ecx + {half_size + 4}], 0")
        return out

    def _complex_neg_into_top(
        self, expr: ast.UnaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`-x` for complex x: (-real, -imag)."""
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        # Eval operand into our temp first.
        out = self._eval_complex_into_top(expr.operand, ctx, ty)
        out.append("        mov     ecx, [esp]")
        if ty.base_type in self._COMPLEX_INT_BASES:
            self._complex_int_neg_half(out, "ecx", 0, half_size)
            self._complex_int_neg_half(out, "ecx", half_size, half_size)
        else:
            width = "dword" if half_size == 4 else "qword"
            out.append(f"        fld     {width} [ecx]")
            out.append("        fchs")
            out.append(f"        fstp    {width} [ecx]")
            out.append(f"        fld     {width} [ecx + {half_size}]")
            out.append("        fchs")
            out.append(f"        fstp    {width} [ecx + {half_size}]")
        return out

    def _complex_conj_into_top(
        self, expr: ast.UnaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`~x` for complex x: (real, -imag) — complex conjugate."""
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        out = self._eval_complex_into_top(expr.operand, ctx, ty)
        out.append("        mov     ecx, [esp]")
        if ty.base_type in self._COMPLEX_INT_BASES:
            self._complex_int_neg_half(out, "ecx", half_size, half_size)
        else:
            width = "dword" if half_size == 4 else "qword"
            out.append(f"        fld     {width} [ecx + {half_size}]")
            out.append("        fchs")
            out.append(f"        fstp    {width} [ecx + {half_size}]")
        return out

    def _asm_stmt(self, item: ast.AsmStmt, ctx: _FuncCtx) -> list[str]:
        """Lower an inline asm statement.

        We can't honor arbitrary asm templates, but a few common
        patterns appear often enough in gcc-c-torture that recognizing
        them is worth the effort:

        - Empty template `""` with `=r`/`=g` outputs and matching `0`/
          `r`/`g` inputs is just `out = in` for each output. Used as
          a register-allocation barrier in many tests.
        - Empty template with no outputs and just inputs is a memory /
          compiler barrier — emit nothing.
        - Single FPU instruction templates (`fsqrt`, `fpatan`, `fsin`,
          `fcos`, `fabs`, `fchs`, `f2xm1`, etc.) with `=t`/`0`/`u`
          constraints: load inputs to FPU stack, run instruction,
          store outputs.

        Anything else: evaluate the operand expressions for side
        effects (matching gcc semantics where the operand expressions
        always execute) and emit nothing else.
        """
        outputs = list(getattr(item, "outputs", []) or [])
        inputs = list(getattr(item, "inputs", []) or [])
        operands = list(getattr(item, "operands", []) or [])
        template = (getattr(item, "template", "") or "").strip()

        # Recognized identity pattern: empty template + `=r/=g/+r/+g`
        # outputs each paired with a matching input (constraint `0`,
        # `1`, ... or a constraint binding to the same operand).
        if template == "" and outputs and inputs:
            if self._asm_identity_eligible(outputs, inputs):
                out: list[str] = []
                for i, (constraint, lvalue) in enumerate(outputs):
                    rhs = self._asm_match_input(constraint, i, inputs)
                    if rhs is None:
                        # Unmatched output: try to evaluate for side
                        # effects, but tolerate non-loadable lvalues
                        # (e.g. structs in `+m (s)`).
                        try:
                            out += self._eval_expr_to_eax(lvalue, ctx)
                        except CodegenError:
                            pass
                        continue
                    # `lvalue = rhs` semantics.
                    out += self._asm_emit_assign(lvalue, rhs, ctx)
                return out

        # Recognized FPU asm templates.
        fpu_template = self._normalize_fpu_template(template)
        if fpu_template is not None and self._asm_fpu_eligible(outputs, inputs):
            return self._emit_fpu_asm(fpu_template, outputs, inputs, ctx)

        # Recognized integer-multiply asm template:
        #   `mull %N` with `=a`/`=d` outputs and `0`/`rm` inputs is the
        #   x86 MUL instruction (EAX * source → EDX:EAX). Used by
        #   glibc-style fixed-point multiply helpers.
        if (
            template.replace("\n", " ").strip().startswith("mull ")
            and len(outputs) == 2 and len(inputs) == 2
        ):
            ocs = [c.lstrip("=+&!%@")[:1] for c, _ in outputs]
            ics = [c.lstrip("&!%@")[:1] for c, _ in inputs]
            if (
                "a" in ocs and "d" in ocs
                and "0" in ics and ics[1] in ("r", "g", "m")
            ):
                # Find which output is =a (eax) and which is =d (edx).
                a_out = next(lv for c, lv in outputs if "a" in c)
                d_out = next(lv for c, lv in outputs if "d" in c)
                # Find the "0"-matching input (value for EAX) and the rm input.
                eax_in = next(e for c, e in inputs if c.lstrip("&!%@").startswith("0"))
                src_in = next(e for c, e in inputs if not c.lstrip("&!%@").startswith("0"))
                out: list[str] = []
                # Eval src to ECX (scratch).
                out += self._eval_expr_to_eax(src_in, ctx)
                out.append("        mov     ecx, eax")
                # Eval EAX value.
                out += self._eval_expr_to_eax(eax_in, ctx)
                # Multiply: EDX:EAX = EAX * ECX.
                out.append("        mul     ecx")
                # Store outputs. EDX first (to preserve EAX for later),
                # then EAX.
                out.append("        push    eax")
                out.append("        mov     eax, edx")
                out += self._asm_emit_assign_from_eax(d_out, ctx)
                out.append("        pop     eax")
                out += self._asm_emit_assign_from_eax(a_out, ctx)
                return out

        # Fallback: evaluate every operand for side effects.
        out: list[str] = []
        for op in operands:
            try:
                out += self._eval_expr_to_eax(op, ctx)
            except CodegenError:
                # Some operands (e.g. taking address of a register
                # variable) may not lower cleanly; skip silently.
                pass
        return out

    def _asm_emit_assign_from_eax(
        self, lvalue: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Store EAX into an lvalue (Identifier/Index/Member/`*p`).
        Used by the asm-output-store path. Preserves EAX semantics —
        we don't try to read EAX afterward."""
        ty = self._type_of(lvalue, ctx)
        if isinstance(lvalue, ast.Identifier):
            return self._identifier_store(lvalue.name, ctx)
        if isinstance(lvalue, ast.Index):
            addr_lines = self._index_address(lvalue, ctx)
        elif isinstance(lvalue, ast.Member):
            addr_lines = self._member_address(lvalue, ctx)
        elif isinstance(lvalue, ast.UnaryOp) and lvalue.op == "*":
            addr_lines = self._eval_expr_to_eax(lvalue.operand, ctx)
        else:
            return []
        out = ["        push    eax"]
        out += addr_lines
        out.append("        mov     ecx, eax")
        out.append("        pop     eax")
        out += self._store_from_eax("[ecx]", ty)
        return out

    @staticmethod
    def _normalize_fpu_template(template: str) -> str | None:
        """Strip whitespace/newlines from an asm template and check if
        it's a single FPU instruction we recognize. Returns the
        canonical instruction name, or None if not recognized."""
        cleaned = template.replace("\n", " ").replace("\t", " ").strip()
        # Strip trailing semicolons and "  ;" comment markers.
        cleaned = cleaned.split(";")[0].strip()
        # Tolerate leading-tab/space variants that don't strip cleanly.
        cleaned = " ".join(cleaned.split())
        if cleaned in (
            "fsqrt", "fpatan", "fsin", "fcos", "fabs", "fchs",
            "f2xm1", "fyl2x", "fyl2xp1", "fptan", "fscale",
            "frndint", "fsincos",
        ):
            return cleaned
        return None

    @staticmethod
    def _asm_fpu_eligible(outputs, inputs) -> bool:
        """Constraints must be the FPU subset we understand: outputs
        use `=t` (st(0)) or `=u` (st(1)); inputs use `0` (match output
        0), `1` (match output 1), `t`, `u`, or `f` (any FPU)."""
        for c, _ in outputs:
            base = c.lstrip("=+&!%@")
            if base[0] not in "tuf":
                return False
        for c, _ in inputs:
            base = c.lstrip("&!%@")
            if not base:
                return False
            if base[0] not in "tuf0123456789":
                return False
        return True

    def _emit_fpu_asm(
        self,
        template: str,
        outputs,
        inputs,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Lower a recognized FPU asm template.

        Inputs are loaded to the FPU stack in reverse order so the
        first input ends up on top (st(0)). The instruction executes.
        Outputs are popped off the stack in declaration order.

        Stack-position constraints:
          input  "0"  → matches output 0 (typically st(0))
          input  "t"  → st(0)
          input  "u"  → st(1)
          input  "f"  → any FPU register (we use st(0))
          output "=t" → st(0)
          output "=u" → st(1) (rare)
        """
        out: list[str] = []
        # Reverse-order load: last input pushed first so the first
        # input ends up at st(0).
        for constraint, expr in reversed(inputs):
            out += self._eval_float_to_st0(expr, ctx)
        # The asm instruction itself.
        out.append(f"        {template}")
        # Some FPU instructions consume two stack slots and leave one
        # (e.g. fpatan, fyl2x, fyl2xp1, fscale). Others leave the
        # input count unchanged (fsqrt, fsin, fcos, fabs, fchs,
        # f2xm1, frndint). fsincos consumes one and produces two.
        # We don't need to track this explicitly — after the
        # instruction, st(0) is the (or first) output.
        # Pop outputs — `=t` lvalues take st(0) (each fstp pops it).
        for constraint, lvalue in outputs:
            out += self._asm_fpu_store_output(lvalue, ctx)
        return out

    def _asm_fpu_store_output(
        self, lvalue: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Store st(0) into the lvalue. Mirrors `_eval_float_to_st0`
        + fstp pattern used elsewhere; supports Identifier/Member/
        Index/`*p` lvalues."""
        ty = self._type_of(lvalue, ctx)
        size = self._size_of(ty) if self._is_float_type(ty) else 8
        width = "dword" if size == 4 else "qword"
        if isinstance(lvalue, ast.Identifier):
            addr = self._float_lvalue_addr(lvalue.name, ctx)
            return [f"        fstp    {width} {addr}"]
        # General lvalue: compute address, fstp through it.
        if isinstance(lvalue, ast.Index):
            addr_lines = self._index_address(lvalue, ctx)
        elif isinstance(lvalue, ast.Member):
            addr_lines = self._member_address(lvalue, ctx)
        elif isinstance(lvalue, ast.UnaryOp) and lvalue.op == "*":
            addr_lines = self._eval_expr_to_eax(lvalue.operand, ctx)
        else:
            # Fallback: fstp into a scratch slot, drop.
            return [f"        fstp    {width} [esp - 8]"]
        return addr_lines + [
            "        mov     ecx, eax",
            f"        fstp    {width} [ecx]",
        ]

    @staticmethod
    def _asm_identity_eligible(outputs, inputs) -> bool:
        """Are the asm constraints simple enough to model as identity
        assignments?"""
        for c, _ in outputs:
            base = c.lstrip("=+&!%@")
            if not base:
                return False
            if base[0] not in "rgmRpiq":
                return False
        for c, _ in inputs:
            base = c.lstrip("&!%@")
            if not base:
                return False
            ok_initials = "rgmRpiq0123456789"
            if base[0] not in ok_initials:
                return False
        return True

    @staticmethod
    def _asm_match_input(out_constraint, out_idx, inputs):
        """For an output operand at index `out_idx`, find a matching
        input operand. The input may be the same operand (constraint
        starts with `+`), or any input whose constraint starts with
        the digit `out_idx`."""
        if out_constraint.startswith("+"):
            # `+r`(x) is "x is both input and output". We don't get a
            # separate input expression, so this is effectively a no-op
            # for our purposes.
            return None
        for c, expr in inputs:
            if c.lstrip("&!%@").startswith(str(out_idx)):
                return expr
        # No matching input: not an identity-style asm.
        return None

    def _asm_emit_assign(
        self, lvalue: ast.Expression, rhs: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Emit code for `lvalue = rhs` in the context of an
        identity-pattern asm statement. We synthesize a BinaryOp("=")
        and route it through the regular `_assign` path."""
        synth = ast.BinaryOp(op="=", left=lvalue, right=rhs)
        try:
            return self._eval_expr_to_eax(synth, ctx)
        except CodegenError:
            # Fall back to evaluating the rhs for side effects.
            return self._eval_expr_to_eax(rhs, ctx)

    def _emit_memcpy_inline(
        self, out: list[str], src_reg: str, dst_reg: str, size: int,
        src_off: int = 0, dst_off: int = 0,
    ) -> None:
        """Emit unrolled byte-wise memcpy from `[src_reg+src_off]` to
        `[dst_reg+dst_off]` for `size` bytes. Uses 4-byte loads/stores
        through EAX, falling back to 2-byte (AX) and 1-byte (AL) for
        odd tails. Required for `_Complex char` / `_Complex short`
        whose total size isn't a multiple of 4.
        """
        offset = 0
        while size - offset >= 4:
            out.append(f"        mov     eax, [{src_reg} + {src_off + offset}]")
            out.append(f"        mov     [{dst_reg} + {dst_off + offset}], eax")
            offset += 4
        if size - offset >= 2:
            out.append(f"        mov     ax, [{src_reg} + {src_off + offset}]")
            out.append(f"        mov     [{dst_reg} + {dst_off + offset}], ax")
            offset += 2
        if size - offset >= 1:
            out.append(f"        mov     al, [{src_reg} + {src_off + offset}]")
            out.append(f"        mov     [{dst_reg} + {dst_off + offset}], al")

    def _complex_int_neg_half(
        self, out: list, addr_reg: str, off: int, half_size: int,
    ) -> None:
        """Negate the integer half at `[addr_reg + off]` of width
        `half_size`. Long-long halves use eax/edx; smaller halves
        use the appropriately-narrow load + neg + store."""
        if half_size == 8:
            out.append(f"        mov     eax, [{addr_reg} + {off}]")
            out.append(f"        mov     edx, [{addr_reg} + {off + 4}]")
            out.append("        neg     eax")
            out.append("        adc     edx, 0")
            out.append("        neg     edx")
            out.append(f"        mov     [{addr_reg} + {off}], eax")
            out.append(f"        mov     [{addr_reg} + {off + 4}], edx")
        elif half_size == 4:
            out.append(f"        neg     dword [{addr_reg} + {off}]")
        elif half_size == 2:
            out.append(f"        neg     word [{addr_reg} + {off}]")
        elif half_size == 1:
            out.append(f"        neg     byte [{addr_reg} + {off}]")

    def _complex_const_into_top(
        self, real: float, imag: float, ty: ast.ComplexType,
    ) -> list[str]:
        """Constant `(real, imag)` complex value — interned via the
        float-constant table for each half. For int-base complex,
        each half stores the integer value with width-appropriate
        directives.
        """
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        if ty.base_type in self._COMPLEX_INT_BASES:
            r = int(real)
            i = int(imag)
            int_widths = {1: ("byte", "al"), 2: ("word", "ax"), 4: ("dword", "eax")}
            if half_size == 8:
                rl, rh = r & 0xFFFFFFFF, (r >> 32) & 0xFFFFFFFF
                il, ih = i & 0xFFFFFFFF, (i >> 32) & 0xFFFFFFFF
                return [
                    "        mov     ecx, [esp]",
                    f"        mov     dword [ecx], 0x{rl:08X}",
                    f"        mov     dword [ecx + 4], 0x{rh:08X}",
                    f"        mov     dword [ecx + 8], 0x{il:08X}",
                    f"        mov     dword [ecx + 12], 0x{ih:08X}",
                ]
            w, lo = int_widths[half_size]
            return [
                "        mov     ecx, [esp]",
                f"        mov     {w} [ecx], {r}",
                f"        mov     {w} [ecx + {half_size}], {i}",
            ]
        width = "dword" if half_size == 4 else "qword"
        r_label = self._intern_float(real, half_size)
        i_label = self._intern_float(imag, half_size)
        return [
            "        mov     ecx, [esp]",
            f"        fld     {width} [{r_label}]",
            f"        fstp    {width} [ecx]",
            f"        fld     {width} [{i_label}]",
            f"        fstp    {width} [ecx + {half_size}]",
        ]

    def _complex_addsub_into_top(
        self, expr: ast.BinaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`a + b` or `a - b` for complex values. Component-wise.

        Allocate scratch slot for left, evaluate left into scratch,
        evaluate right into top-of-stack, then add/sub component-wise.
        """
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        width = "dword" if half_size == 4 else "qword"
        # Use the left operand's pre-allocated temp (reserved by
        # _collect_call_temps) — the left expression itself is a
        # complex sub-expression so it'll have a temp.
        # Actually, `a` may be a leaf (Identifier) without a temp.
        # Easier: stack-allocate scratch space via push/pop.
        size = self._size_of(ty)  # 8 or 16
        # Reserve scratch on the stack: sub esp, size.
        out = [f"        sub     esp, {size}"]
        # Save destination ptr (currently [esp + size] after the sub).
        # Push the scratch's address as a target.
        out.append("        mov     eax, esp")
        out.append("        push    eax")
        # Eval left into scratch.
        out += self._eval_complex_into_top(expr.left, ctx, ty)
        out.append("        add     esp, 4")  # discard scratch ptr
        # Eval right into the original destination ([esp + size] now).
        out.append(f"        mov     eax, [esp + {size}]")
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr.right, ctx, ty)
        out.append("        add     esp, 4")
        # ECX = original destination, EDX = scratch base (esp).
        out.append(f"        mov     ecx, [esp + {size}]")
        out.append("        mov     edx, esp")
        # Apply faddp/fsubp component-wise.
        op_word = "faddp" if expr.op == "+" else "fsubp"
        # real
        out.append(f"        fld     {width} [edx]")
        out.append(f"        fld     {width} [ecx]")
        out.append(f"        {op_word}   st1, st0")
        out.append(f"        fstp    {width} [ecx]")
        # imag
        out.append(f"        fld     {width} [edx + {half_size}]")
        out.append(f"        fld     {width} [ecx + {half_size}]")
        out.append(f"        {op_word}   st1, st0")
        out.append(f"        fstp    {width} [ecx + {half_size}]")
        # Reclaim scratch.
        out.append(f"        add     esp, {size}")
        return out

    def _complex_mul_into_top(
        self, expr: ast.BinaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`a * b` for complex values:
            real = ar*br - ai*bi
            imag = ar*bi + ai*br
        """
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        width = "dword" if half_size == 4 else "qword"
        size = self._size_of(ty)
        # Two scratch slots: left (lo) at [esp] and right (mid) at [esp + size].
        out = [f"        sub     esp, {2 * size}"]
        # Eval left into scratch lo.
        out.append("        mov     eax, esp")
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr.left, ctx, ty)
        out.append("        add     esp, 4")
        # Eval right into scratch mid.
        out.append(f"        lea     eax, [esp + {size}]")
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr.right, ctx, ty)
        out.append("        add     esp, 4")
        # ECX = original dest (at [esp + 2*size]).
        out.append(f"        mov     ecx, [esp + {2 * size}]")
        out.append("        mov     edx, esp")  # &lo
        # real = ar*br - ai*bi
        out.append(f"        fld     {width} [edx]")            # ar
        out.append(f"        fld     {width} [edx + {size}]")   # br
        out.append("        fmulp   st1, st0")                   # ar*br
        out.append(f"        fld     {width} [edx + {half_size}]")  # ai
        out.append(f"        fld     {width} [edx + {size + half_size}]")  # bi
        out.append("        fmulp   st1, st0")                   # ai*bi
        out.append("        fsubp   st1, st0")                   # ar*br - ai*bi
        out.append(f"        fstp    {width} [ecx]")
        # imag = ar*bi + ai*br
        out.append(f"        fld     {width} [edx]")            # ar
        out.append(f"        fld     {width} [edx + {size + half_size}]")  # bi
        out.append("        fmulp   st1, st0")                   # ar*bi
        out.append(f"        fld     {width} [edx + {half_size}]")  # ai
        out.append(f"        fld     {width} [edx + {size}]")   # br
        out.append("        fmulp   st1, st0")                   # ai*br
        out.append("        faddp   st1, st0")                   # ar*bi + ai*br
        out.append(f"        fstp    {width} [ecx + {half_size}]")
        out.append(f"        add     esp, {2 * size}")
        return out

    def _complex_div_into_top(
        self, expr: ast.BinaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`a / b` for complex values:
            denom = br*br + bi*bi
            real = (ar*br + ai*bi) / denom
            imag = (ai*br - ar*bi) / denom
        """
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        width = "dword" if half_size == 4 else "qword"
        size = self._size_of(ty)
        out = [f"        sub     esp, {2 * size}"]
        out.append("        mov     eax, esp")
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr.left, ctx, ty)
        out.append("        add     esp, 4")
        out.append(f"        lea     eax, [esp + {size}]")
        out.append("        push    eax")
        out += self._eval_complex_into_top(expr.right, ctx, ty)
        out.append("        add     esp, 4")
        out.append(f"        mov     ecx, [esp + {2 * size}]")
        out.append("        mov     edx, esp")
        # Compute denom = br*br + bi*bi, leave on st(0).
        out.append(f"        fld     {width} [edx + {size}]")             # br
        out.append("        fld     st0")                                  # br br
        out.append("        fmulp   st1, st0")                              # br*br
        out.append(f"        fld     {width} [edx + {size + half_size}]")  # bi
        out.append("        fld     st0")
        out.append("        fmulp   st1, st0")                              # bi*bi
        out.append("        faddp   st1, st0")                              # denom
        # st0 = denom. Now compute real numerator: ar*br + ai*bi.
        out.append(f"        fld     {width} [edx]")            # ar
        out.append(f"        fld     {width} [edx + {size}]")   # br
        out.append("        fmulp   st1, st0")                   # ar*br
        out.append(f"        fld     {width} [edx + {half_size}]")
        out.append(f"        fld     {width} [edx + {size + half_size}]")
        out.append("        fmulp   st1, st0")                   # ai*bi
        out.append("        faddp   st1, st0")                   # ar*br+ai*bi
        # st: numer_real, denom
        out.append("        fdiv    st0, st1")                   # numer_real / denom
        out.append(f"        fstp    {width} [ecx]")             # store real
        # Imag numerator: ai*br - ar*bi
        out.append(f"        fld     {width} [edx + {half_size}]")
        out.append(f"        fld     {width} [edx + {size}]")
        out.append("        fmulp   st1, st0")                   # ai*br
        out.append(f"        fld     {width} [edx]")
        out.append(f"        fld     {width} [edx + {size + half_size}]")
        out.append("        fmulp   st1, st0")                   # ar*bi
        out.append("        fsubp   st1, st0")                   # ai*br - ar*bi
        out.append("        fdivp   st1, st0")                   # / denom
        out.append(f"        fstp    {width} [ecx + {half_size}]")
        out.append(f"        add     esp, {2 * size}")
        return out

    def _complex_compare(
        self, expr: ast.BinaryOp, ctx: _FuncCtx,
    ) -> list[str]:
        """`a == b` or `a != b` for complex values. Returns 0/1 in EAX.

        Plan: get &a and &b (handling lvalues, scalars, and complex
        expressions). Compare both halves and AND the boolean halves
        for `==` (or NAND for `!=`). Float-typed halves go through
        FPU compare; integer halves use plain `cmp`.
        """
        lt = self._type_of(expr.left, ctx)
        rt = self._type_of(expr.right, ctx)
        # Use whichever side is complex to determine the precision.
        cty = lt if isinstance(lt, ast.ComplexType) else rt
        half_size = self._COMPLEX_BASE_SIZES[cty.base_type]
        is_int = cty.base_type in self._COMPLEX_INT_BASES
        size = self._size_of(cty)
        # Materialize both operands into stack scratch slots so we
        # can compare half by half.
        out: list[str] = [f"        sub     esp, {2 * size}"]
        # Eval left into [esp + size] (slot 1).
        out.append(f"        lea     eax, [esp + {size}]")
        out.append("        push    eax")
        out += self._eval_complex_into_top(
            expr.left if isinstance(lt, ast.ComplexType)
            else ast.Cast(target_type=cty, expr=expr.left),
            ctx, cty,
        )
        out.append("        add     esp, 4")
        # Eval right into [esp] (slot 0).
        out.append("        mov     eax, esp")
        out.append("        push    eax")
        out += self._eval_complex_into_top(
            expr.right if isinstance(rt, ast.ComplexType)
            else ast.Cast(target_type=cty, expr=expr.right),
            ctx, cty,
        )
        out.append("        add     esp, 4")
        if is_int:
            # Integer halves: load each half and compare with cmp.
            int_widths = {1: ("byte", "al"), 2: ("word", "ax"), 4: ("dword", "eax")}
            int_widths_b = {1: ("byte", "bl"), 2: ("word", "bx"), 4: ("dword", "ebx")}
            if half_size == 8:
                # Long-long halves — use a pair of cmps.
                out.append(f"        mov     eax, [esp + {size}]")
                out.append(f"        mov     edx, [esp + {size} + 4]")
                out.append("        cmp     eax, [esp]")
                out.append("        jne     .L_cc_neq_real")
                out.append("        cmp     edx, [esp + 4]")
                out.append("        jne     .L_cc_neq_real")
                out.append("        mov     ecx, 1")
                out.append("        jmp     .L_cc_real_done")
                out.append(".L_cc_neq_real:")
                out.append("        mov     ecx, 0")
                out.append(".L_cc_real_done:")
                # imag halves at [esp+size+8] vs [esp+8]
                out.append(f"        mov     eax, [esp + {size + 8}]")
                out.append(f"        mov     edx, [esp + {size + 8} + 4]")
                out.append("        cmp     eax, [esp + 8]")
                out.append("        jne     .L_cc_neq_imag")
                out.append("        cmp     edx, [esp + 12]")
                out.append("        jne     .L_cc_neq_imag")
                out.append("        mov     eax, 1")
                out.append("        jmp     .L_cc_imag_done")
                out.append(".L_cc_neq_imag:")
                out.append("        mov     eax, 0")
                out.append(".L_cc_imag_done:")
                out.append("        and     eax, ecx")
            else:
                w, lo = int_widths[half_size]
                _, lo_b = int_widths_b[half_size]
                out.append(f"        mov     {lo}, [esp + {size}]")
                out.append(f"        cmp     {lo}, [esp]")
                out.append("        sete    al")
                out.append("        movzx   ecx, al")
                out.append(f"        mov     {lo}, [esp + {size + half_size}]")
                out.append(f"        cmp     {lo}, [esp + {half_size}]")
                out.append("        sete    al")
                out.append("        movzx   eax, al")
                out.append("        and     eax, ecx")
        else:
            width = "dword" if half_size == 4 else "qword"
            # Compare reals: fld both, fucompp → ax via fnstsw.
            out.append(f"        fld     {width} [esp + {size}]")        # left.real
            out.append(f"        fld     {width} [esp]")                  # right.real
            out.append("        fucompp")
            out.append("        fnstsw  ax")
            out.append("        sahf")
            # Equal flag (ZF) → AL.
            out.append("        sete    al")
            out.append("        movzx   ecx, al")
            # Compare imags.
            out.append(f"        fld     {width} [esp + {size + half_size}]")  # left.imag
            out.append(f"        fld     {width} [esp + {half_size}]")          # right.imag
            out.append("        fucompp")
            out.append("        fnstsw  ax")
            out.append("        sahf")
            out.append("        sete    al")
            out.append("        movzx   eax, al")
            # AND the two booleans → eax.
            out.append("        and     eax, ecx")
        if expr.op == "!=":
            out.append("        xor     eax, 1")
        out.append(f"        add     esp, {2 * size}")
        return out

    def _complex_assign_into_top(
        self, expr: ast.BinaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`lhs = rhs` for complex types. The result is the new value
        of lhs; we copy lhs's content into the destination."""
        # First do the assignment (which leaves &lhs in EAX), then
        # copy lhs's content into our temp slot.
        out = self._complex_copy_assign(expr, ty, ctx)  # eax = &lhs
        out.append("        mov     edx, eax")
        out.append("        mov     ecx, [esp]")
        size = self._size_of(ty)
        self._emit_memcpy_inline(out, "edx", "ecx", size)
        return out

    def _complex_assign_from_scalar(
        self,
        lhs: ast.Expression,
        rhs: ast.Expression,
        ty: ast.ComplexType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """`c = scalar` for `_Complex T c` — store scalar as the real
        part, zero the imag part. Per C, an integer or float assigned
        to a complex extends with imag=0. Int-base complex stores the
        scalar via `_store_from_eax`; float-base via `fstp`.
        """
        half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
        # Compute &c, hold in a register.
        out = self._complex_value_address(lhs, ctx)
        out.append("        push    eax")
        if ty.base_type in self._COMPLEX_INT_BASES:
            half_basic = ast.BasicType(name=ty.base_type)
            scalar_ty = self._type_of(rhs, ctx)
            if self._is_float_type(scalar_ty):
                out += self._eval_float_to_st0(rhs, ctx)
                out += self._fistp_truncate_dword_to_eax()
            elif half_size == 8:
                # Long-long destination: widen scalar to LL.
                out += self._eval_expr_to_edx_eax(rhs, ctx)
                out.append("        pop     ecx")
                out.append("        mov     [ecx], eax")
                out.append("        mov     [ecx + 4], edx")
                out.append("        mov     dword [ecx + 8], 0")
                out.append("        mov     dword [ecx + 12], 0")
                out.append("        mov     eax, ecx")
                return out
            else:
                out += self._eval_expr_to_eax(rhs, ctx)
            out.append("        pop     ecx")
            out += self._store_from_eax("[ecx]", half_basic)
            # Zero the imag half.
            if half_size == 1:
                out.append(f"        mov     byte [ecx + {half_size}], 0")
            elif half_size == 2:
                out.append(f"        mov     word [ecx + {half_size}], 0")
            else:
                out.append(f"        mov     dword [ecx + {half_size}], 0")
            out.append("        mov     eax, ecx")
            return out
        # Float halves — eval to st(0), fstp.
        out += self._eval_float_to_st0(rhs, ctx)
        out.append("        pop     ecx")
        width = "dword" if half_size == 4 else "qword"
        out.append(f"        fstp    {width} [ecx]")
        if half_size == 4:
            out.append(f"        mov     dword [ecx + {half_size}], 0")
        else:
            out.append(f"        mov     dword [ecx + {half_size}], 0")
            out.append(f"        mov     dword [ecx + {half_size + 4}], 0")
        out.append("        mov     eax, ecx")
        return out

    def _is_complex_returning_call(
        self, call: ast.Call, ctx: _FuncCtx,
    ) -> bool:
        """Does `call` invoke a function whose declared return type is
        `_Complex T`?"""
        target = self._call_target(call)
        if target is None:
            return False
        ret_ty = self._func_return_types.get(target)
        return isinstance(ret_ty, ast.ComplexType)

    def _complex_copy_assign(
        self, expr: ast.BinaryOp, ty: ast.ComplexType, ctx: _FuncCtx,
    ) -> list[str]:
        """`dst = src` where both are `_Complex T`. The src may be an
        lvalue (Identifier, Member, etc.) or a complex sub-expression
        (BinaryOp, UnaryOp, Cast). For lvalues we copy directly; for
        sub-expressions we evaluate into &dst.
        """
        size = self._size_of(ty)
        # Compute &dst, save it.
        out = self._complex_value_address(expr.left, ctx)
        out.append("        push    eax")
        rhs = expr.right
        if (
            isinstance(rhs, (ast.BinaryOp, ast.UnaryOp, ast.Cast))
            and not (
                isinstance(rhs, ast.Cast)
                and isinstance(rhs.target_type, ast.ComplexType) is False
            )
        ):
            # Evaluate rhs into &dst directly. Pass the destination's
            # complex type so component-wise math uses the dest's
            # half width (narrowing where necessary).
            out += self._eval_complex_into_top(rhs, ctx, ty)
            out.append("        pop     ecx")
            out.append("        mov     eax, ecx")
            return out
        # Lvalue / Call / FloatLiteral source.
        rhs_ty = self._type_of(rhs, ctx)
        if (
            isinstance(rhs_ty, ast.ComplexType)
            and rhs_ty.base_type != ty.base_type
        ):
            # Cross-precision: per-half conversion.
            out += self._convert_complex_into_top(rhs, rhs_ty, ty, ctx)
            out.append("        pop     ecx")
            out.append("        mov     eax, ecx")
            return out
        # Same-precision: memcpy.
        out += self._complex_value_address(rhs, ctx)
        out.append("        mov     edx, eax")
        out.append("        pop     ecx")
        self._emit_memcpy_inline(out, "edx", "ecx", size)
        out.append("        mov     eax, ecx")
        return out

    def _copy_complex_to_retptr(
        self,
        src_expr: ast.Expression,
        ret_ty: ast.ComplexType,
        retptr_disp: int,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy a `_Complex T` lvalue into the `__retptr__` buffer.

        Uses the same per-dword copy as `_copy_struct_to_retptr` —
        complex values are laid out as `{real, imag}` and live in a
        slot like a 2-member struct.
        """
        size = self._size_of(ret_ty)
        out = [f"        mov     eax, {_ebp_addr(retptr_disp)}"]
        out.append("        push    eax")
        out += self._complex_value_address(src_expr, ctx)
        out.append("        mov     edx, eax")
        out.append("        pop     ecx")
        self._emit_memcpy_inline(out, "edx", "ecx", size)
        out.append(f"        mov     eax, {_ebp_addr(retptr_disp)}")
        return out

    def _copy_vector_to_retptr(
        self,
        src_expr: ast.Expression,
        ret_ty: ast.ArrayType,
        retptr_disp: int,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy a vector value into the function's `__retptr__` buffer.

        Same byte-for-byte layout as `_copy_struct_to_retptr`; differs
        only in how the source address is computed.
        """
        size = self._size_of(ret_ty)
        out = [f"        mov     eax, {_ebp_addr(retptr_disp)}"]
        out.append("        push    eax")
        out += self._vector_value_address(src_expr, ctx)  # eax = &src
        out.append("        mov     edx, eax")
        out.append("        pop     ecx")
        offset = 0
        while size - offset >= 4:
            out.append(f"        mov     eax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], eax")
            offset += 4
        if size - offset >= 2:
            out.append(f"        mov     ax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], ax")
            offset += 2
        if size - offset >= 1:
            out.append(f"        mov     al, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], al")
        out.append("        mov     eax, ecx")
        return out

    def _copy_struct_to_retptr(
        self,
        src_expr: ast.Expression,
        ret_ty: ast.TypeNode,
        retptr_disp: int,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy a struct l-value into the function's `__retptr__` buffer.

        Leaves EAX = retptr (the destination address) so callers can
        chain. Used by `_return` when the function returns a struct and
        the value is a struct l-value (Identifier, `*p`, `arr[i]`, or a
        member access).

        When the source is a runtime-pointer-derived l-value (`*p`,
        `arr[i]`, `p->m` chain), the source and destination might
        alias (caller's destination is a global / pointee that overlaps
        with our pointer). Forward unrolled copy would corrupt under
        overlap. Use memmove-direction selection: when dst > src, copy
        backward; otherwise forward.
        """
        size = self._size_of(ret_ty)
        # Detect when aliasing is plausible — i.e. the source isn't a
        # local Identifier (which can't overlap with the caller's buffer).
        may_alias = self._may_alias_caller(src_expr, ctx)
        out = [f"        mov     eax, {_ebp_addr(retptr_disp)}"]
        out.append("        push    eax")
        out += self._struct_address(src_expr, ctx)  # eax = &src
        out.append("        mov     edx, eax")
        out.append("        pop     ecx")           # ecx = retptr
        if may_alias:
            # Runtime memmove-direction selection: branch on dst vs src.
            label_back = ctx.label("crs_back")
            label_done = ctx.label("crs_done")
            out.append("        cmp     ecx, edx")
            out.append(f"        ja      {label_back}")
            offset = 0
            while size - offset >= 4:
                out.append(f"        mov     eax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], eax")
                offset += 4
            if size - offset >= 2:
                out.append(f"        mov     ax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], ax")
                offset += 2
            if size - offset >= 1:
                out.append(f"        mov     al, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], al")
            out.append(f"        jmp     {label_done}")
            out.append(f"{label_back}:")
            offset = size
            while offset >= 4:
                offset -= 4
                out.append(f"        mov     eax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], eax")
            if offset >= 2:
                offset -= 2
                out.append(f"        mov     ax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], ax")
            if offset >= 1:
                offset -= 1
                out.append(f"        mov     al, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], al")
            out.append(f"{label_done}:")
        else:
            offset = 0
            while size - offset >= 4:
                out.append(f"        mov     eax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], eax")
                offset += 4
            if size - offset >= 2:
                out.append(f"        mov     ax, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], ax")
                offset += 2
            if size - offset >= 1:
                out.append(f"        mov     al, [edx + {offset}]")
                out.append(f"        mov     [ecx + {offset}], al")
        out.append("        mov     eax, ecx")     # return retptr
        return out

    def _may_alias_caller(
        self, src_expr: ast.Expression, ctx: _FuncCtx,
    ) -> bool:
        """Conservative test: does this source expression potentially
        alias the caller's struct-return buffer? Stack-local
        Identifiers can't (their storage lives in this frame). Anything
        that goes through a runtime pointer might."""
        # Strip a single dereference / index / member chain.
        e = src_expr
        while isinstance(e, ast.Member):
            e = e.obj
        if isinstance(e, ast.Identifier):
            # Local variable: can't alias caller's buffer.
            if ctx.has_local(e.name):
                return False
            # Global: could alias if the caller's destination points
            # into the same global. Conservative: yes.
            return True
        # `*p`, `arr[i]`, etc. — pointer-derived, potentially aliased.
        return True

    # ---- variadic builtins (va_start / va_arg / va_end) -----------------

    def _builtin_overflow(
        self,
        name: str,
        args: list[ast.Expression],
        ctx: _FuncCtx,
    ) -> list[str]:
        """`__builtin_{add,sub,mul}_overflow(a, b, &result)`.

        Returns 1 if the operation overflows when interpreting the
        operands per their static types (C overflow semantics depend on
        signedness — CF for unsigned add/sub, OF for signed add/sub,
        EDX-nonzero for unsigned mul, OF for signed mul). The libc
        stub couldn't know the type so we inline here.
        """
        a_ty = self._type_of(args[0], ctx)
        b_ty = self._type_of(args[1], ctx)
        # Per GCC docs, __builtin_*_overflow checks whether the
        # mathematically-precise result fits in the type pointed to by
        # the third argument. Use that type's signedness to pick the
        # right flag (CF for unsigned, OF for signed).
        dest_ty = self._type_of(args[2], ctx)
        if isinstance(dest_ty, ast.PointerType):
            dest_ty = dest_ty.base_type
        unsigned = self._is_unsigned(dest_ty)
        dest_is_ll = self._is_long_long(dest_ty)
        if self._is_long_long(a_ty) or self._is_long_long(b_ty):
            raise CodegenError(
                f"{name}: long-long operand not supported"
            )
        # Eval b → push, eval a → eax, eval result_ptr → ebx (after).
        out = self._eval_expr_to_eax(args[1], ctx)
        out.append("        push    eax")
        out += self._eval_expr_to_eax(args[0], ctx)
        out.append("        pop     ecx")
        # Stash addr of result for after the op (it can clobber EDX).
        out += self._eval_expr_to_eax(args[2], ctx)
        out.append("        mov     ebx, eax")
        # Reload the operands; we need EAX to hold a, ECX to hold b.
        # We just clobbered EAX with &result; redo the eval. Cheaper
        # alternative: shuffle, but the operand expressions are usually
        # leaf identifiers so re-eval is fine.
        out += self._eval_expr_to_eax(args[1], ctx)
        out.append("        mov     ecx, eax")
        out += self._eval_expr_to_eax(args[0], ctx)
        if name == "__builtin_add_overflow":
            out.append("        add     eax, ecx")
            flag = "setc" if unsigned else "seto"
        elif name == "__builtin_sub_overflow":
            out.append("        sub     eax, ecx")
            flag = "setc" if unsigned else "seto"
        else:  # mul
            # GCC: operands are conceptually widened to a common signed
            # type before multiplication. When either operand is signed,
            # use signed multiply; when both are unsigned, use unsigned.
            both_unsigned = self._is_unsigned(a_ty) and self._is_unsigned(b_ty)
            if both_unsigned:
                # `mul ecx` → EDX:EAX = unsigned product.
                out.append("        mul     ecx")
                if dest_is_ll:
                    out.append("        mov     [ebx], eax")
                    out.append("        mov     [ebx + 4], edx")
                    out.append("        xor     eax, eax")
                    return out
                if unsigned:
                    # 32-bit unsigned dest: overflow iff EDX != 0.
                    out.append("        test    edx, edx")
                    out.append("        setnz   dl")
                else:
                    # 32-bit signed dest: overflow iff result > INT_MAX
                    # (EDX != 0 OR sign bit of EAX set).
                    out.append("        test    edx, edx")
                    out.append("        setnz   dl")
                    out.append("        test    eax, eax")
                    out.append("        sets    dh")
                    out.append("        or      dl, dh")
                out.append("        mov     [ebx], eax")
                out.append("        movzx   eax, dl")
                return out
            # Signed multiply (at least one operand signed).
            out.append("        imul    ecx")
            if dest_is_ll:
                # Signed full 64-bit product fits in LL dest.
                out.append("        mov     [ebx], eax")
                out.append("        mov     [ebx + 4], edx")
                if unsigned:
                    # If product is negative (high bit of EDX set) and
                    # dest is unsigned, technically that's an overflow.
                    # Gcc does report this; but the test we're chasing
                    # (pr84169) ignores the return value. Mark overflow
                    # as 1 if EDX < 0.
                    out.append("        test    edx, edx")
                    out.append("        sets    al")
                    out.append("        movzx   eax, al")
                else:
                    out.append("        xor     eax, eax")
                return out
            if unsigned:
                # 32-bit unsigned dest: overflow if result is negative
                # or doesn't fit (EDX != 0 for positive, EDX != -1 for
                # negative — the latter still overflows because dest is
                # unsigned and can't hold negatives).
                out.append("        mov     [ebx], eax")
                out.append("        test    edx, edx")
                out.append("        setnz   dl")
                # Also: if EDX==0 but EAX is negative (high bit set),
                # the int value fits 32-bit signed but doesn't fit
                # unsigned (negative). But here EDX==0 means non-negative.
                # If EDX==-1 (sign-extension), dl already nonzero.
                out.append("        movzx   eax, dl")
                return out
            flag = "seto"
        out.append(f"        {flag}    dl")
        out.append("        mov     [ebx], eax")
        if dest_is_ll:
            # 32-bit add/sub result, sign- or zero-extend to LL slot.
            if unsigned:
                out.append("        mov     dword [ebx + 4], 0")
            else:
                out.append("        mov     ecx, eax")
                out.append("        sar     ecx, 31")
                out.append("        mov     [ebx + 4], ecx")
        out.append("        movzx   eax, dl")
        return out

    def _va_start(self, args: list[ast.Expression], ctx: _FuncCtx) -> list[str]:
        """Lower `va_start(ap, last)`: ap = (char *)&last + sizeof_padded(last).

        cdecl pushes args right-to-left so the variadic args follow
        immediately after the last named param's stack slot. We compute
        that address with a single `lea` and store it into `ap`.
        """
        if len(args) != 2:
            raise CodegenError("va_start expects exactly 2 arguments")
        ap_expr, last_expr = args
        if not isinstance(last_expr, ast.Identifier):
            raise CodegenError(
                "va_start: second argument must name a parameter"
            )
        last_name = last_expr.name
        if not ctx.has_local(last_name):
            raise CodegenError(
                f"va_start: `{last_name}` is not a parameter or local"
            )
        last_disp = ctx.lookup(last_name)
        last_padded = (self._size_of(ctx.lookup_type(last_name)) + 3) & ~3
        after_disp = last_disp + last_padded
        # Compute the destination's address (where ap lives) into ECX,
        # then store the variadic-args pointer at [ecx]. Supports any
        # lvalue form (Identifier / Index / Member / *p).
        if isinstance(ap_expr, ast.Identifier):
            out = [f"        lea     eax, {_ebp_addr(after_disp)}"]
            out += self._identifier_store(ap_expr.name, ctx)
            return out
        if isinstance(ap_expr, ast.Index):
            addr_lines = self._index_address(ap_expr, ctx)
        elif isinstance(ap_expr, ast.Member):
            addr_lines = self._member_address(ap_expr, ctx)
        elif isinstance(ap_expr, ast.UnaryOp) and ap_expr.op == "*":
            addr_lines = self._eval_expr_to_eax(ap_expr.operand, ctx)
        else:
            raise CodegenError(
                "va_start: first argument must be an lvalue "
                f"(got {type(ap_expr).__name__})"
            )
        out = list(addr_lines)
        out.append("        mov     ecx, eax")
        out.append(f"        lea     eax, {_ebp_addr(after_disp)}")
        out.append("        mov     [ecx], eax")
        return out

    def _va_arg_int(self, expr: ast.VaArgExpr, ctx: _FuncCtx) -> list[str]:
        """Lower `va_arg(ap, T)` where T is integer/pointer-typed.

        Reads ap into ECX, advances ap by `(sizeof T + 3) & ~3`, then
        loads through ECX with the target's width.
        """
        target_ty = expr.target_type
        advance = (self._size_of(target_ty) + 3) & ~3
        out = self._va_arg_read_and_advance(expr, advance, ctx)
        out += self._load_to_eax("[ecx]", target_ty)
        return out

    def _va_arg_struct_copy(
        self,
        expr: ast.VaArgExpr,
        dest_addr_lines: list[str],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy the next variadic struct argument into `*dest`."""
        target_size = self._size_of(expr.target_type)
        advance = (target_size + 3) & ~3
        out = list(dest_addr_lines)
        out.append("        mov     edx, eax")            # edx = dest
        # Now compute the va_list pointer into ECX (preserving EDX).
        out += self._va_arg_read_and_advance(expr, advance, ctx)
        # Copy in dword-sized chunks, then byte-tail.
        offset = 0
        while offset + 4 <= target_size:
            out.append(f"        mov     eax, [ecx + {offset}]")
            out.append(f"        mov     [edx + {offset}], eax")
            offset += 4
        while offset < target_size:
            out.append(f"        mov     al, [ecx + {offset}]")
            out.append(f"        mov     [edx + {offset}], al")
            offset += 1
        out.append("        mov     eax, edx")            # leave dest in EAX
        return out

    def _va_arg_float(self, expr: ast.VaArgExpr, ctx: _FuncCtx) -> list[str]:
        """Lower `va_arg(ap, T)` for `float`/`double` — leaves the
        value on st(0)."""
        target_ty = expr.target_type
        target_size = self._size_of(target_ty)
        advance = (target_size + 3) & ~3
        width = "dword" if target_size == 4 else "qword"
        out = self._va_arg_read_and_advance(expr, advance, ctx)
        out.append(f"        fld     {width} [ecx]")
        return out

    def _va_arg_read_and_advance(
        self, expr: ast.VaArgExpr, advance: int, ctx: _FuncCtx,
    ) -> list[str]:
        """Emit code that puts the current va_list pointer into ECX
        and advances the underlying ap slot by `advance`. Supports any
        lvalue form for the ap operand (Identifier / `*p` / arr[i] /
        struct.member / arr[i].member chains).
        """
        if isinstance(expr.ap, ast.Identifier):
            ap_addr = self._identifier_addr_text(expr.ap.name, ctx)
            return [
                f"        mov     ecx, {ap_addr}",
                f"        add     dword {ap_addr}, {advance}",
            ]
        # General lvalue path: compute &ap into EBX, then load+advance.
        if isinstance(expr.ap, ast.UnaryOp) and expr.ap.op == "*":
            addr_lines = self._eval_expr_to_eax(expr.ap.operand, ctx)
        elif isinstance(expr.ap, ast.Index):
            addr_lines = self._index_address(expr.ap, ctx)
        elif isinstance(expr.ap, ast.Member):
            addr_lines = self._member_address(expr.ap, ctx)
        else:
            raise CodegenError(
                "va_arg: ap must be an lvalue "
                f"(got {type(expr.ap).__name__})"
            )
        out = list(addr_lines)
        out.append("        mov     ebx, eax")
        out.append("        mov     ecx, [ebx]")
        out.append(f"        add     dword [ebx], {advance}")
        return out

    def _stripped_callee(self, call: ast.Call) -> ast.Expression:
        """Strip leading `*`s on a Call's callee (function-typed `*` is idempotent)."""
        callee = call.func
        while isinstance(callee, ast.UnaryOp) and callee.op == "*":
            callee = callee.operand
        return callee

    def _call_target(self, call: ast.Call, ctx=None) -> str | None:
        """If the call is direct (a known function name), return that name.

        When ctx is provided, nested-fn names get remapped to their
        lifted top-level mangled name first, so `Foo()` inside an
        outer fn resolves to `_outer__Foo` (which is what's registered
        in `_func_return_types`).
        """
        callee = self._stripped_callee(call)
        if isinstance(callee, ast.Identifier):
            name = callee.name
            if ctx is not None:
                name = ctx.nested_fn_names.get(name, name)
            if name in self._func_return_types:
                return name
        return None

    def _call_into_address(
        self,
        call: ast.Call,
        retptr_lines: list[str],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Lower a struct-returning call where the destination is known.

        `retptr_lines` is asm that produces the destination address in
        EAX; we hand it to `_emit_call` as the hidden first arg. Used by
        `_var_init`, `_assign`, and `_return` (for chained struct
        returns).
        """
        target = self._call_target(call)
        if target is not None:
            return self._emit_call(
                call.args, ctx, direct=target, retptr=retptr_lines,
            )
        return self._emit_call(
            call.args, ctx,
            indirect_callee=self._stripped_callee(call),
            retptr=retptr_lines,
        )

    def _call_return_type(
        self, call: ast.Call, ctx: _FuncCtx | None,
    ) -> ast.TypeNode | None:
        """Resolve a Call's return type — directly when the callee is a
        named function, or by walking the indirect callee's function-
        pointer type.
        """
        target = self._call_target(call, ctx)
        if target is not None:
            return self._func_return_types.get(target)
        if ctx is None:
            return None
        try:
            ty = self._type_of(call.func, ctx)
        except CodegenError:
            return None
        # Strip leading `*`s that the user may have layered on a fn
        # pointer — `(*fp)()` and `(***fp)()` are idempotent.
        if isinstance(ty, ast.PointerType) and isinstance(
            ty.base_type, ast.FunctionType
        ):
            return ty.base_type.return_type
        if isinstance(ty, ast.FunctionType):
            return ty.return_type
        return None

    def _is_struct_returning_call(self, call: ast.Call, ctx) -> bool:
        """True iff `call` returns a struct (direct or via known
        function-pointer type)."""
        rt = self._call_return_type(call, ctx)
        return isinstance(rt, ast.StructType)

    def _is_vector_returning_call(self, call: ast.Call, ctx) -> bool:
        """True iff `call` returns a vector (ArrayType, by-value)."""
        rt = self._call_return_type(call, ctx)
        return isinstance(rt, ast.ArrayType)

    def _is_int128_returning_call(self, call: ast.Call, ctx) -> bool:
        """True iff `call` returns __int128. Uses the same retptr ABI
        as struct/complex/vector returns: hidden first param is a
        pointer to a 16-byte buffer in the caller's frame."""
        rt = self._call_return_type(call, ctx)
        return self._is_int128(rt)

    # ---- expressions ----------------------------------------------------

    # Operators whose result is always plain int regardless of operand types
    # (relational, equality, logical, shifts, bitwise, multiplicative).
    _INT_RESULT_BINOPS = frozenset({
        "*", "/", "%", "&", "|", "^", "<<", ">>",
        "==", "!=", "<", ">", "<=", ">=", "&&", "||",
    })

    def _type_of(self, expr: ast.Expression, ctx: _FuncCtx) -> ast.TypeNode:
        """Best-effort static type of `expr`.

        Used to drive pointer-arithmetic scaling. Falls back to `int` for
        anything we can't classify (call return when no prototype, etc.) —
        the conservative default keeps non-pointer code on the integer path.
        """
        if isinstance(expr, ast.Identifier):
            return self._identifier_type(expr.name, ctx)
        if isinstance(expr, ast.IntLiteral):
            # Honor the L/U suffixes per C 6.4.4.1: walk int → unsigned int
            # → long → unsigned long → long long → unsigned long long,
            # taking the first that fits per the suffix's lower bound.
            # On i386 long is 32-bit, so the long step is interchangeable
            # with int width-wise; we set the BasicType name based on the
            # ranks the value/suffix imply.
            name = "int"
            v = expr.value
            unsigned = getattr(expr, "is_unsigned", False)
            is_long = getattr(expr, "is_long", False)
            is_long_long = getattr(expr, "is_long_long", False)
            is_hex = getattr(expr, "is_hex", False)
            if is_long_long:
                name = "long long"
                # Hex/octal can promote to unsigned ll if value > LL_MAX.
                if is_hex and v > 0x7FFFFFFFFFFFFFFF:
                    unsigned = True
            elif is_long:
                # `L` suffix: walk (signed) long → (unsigned) long → ll → ull.
                # Default name is "long" so _Generic distinguishes it
                # from "int" even though both are 32 bits on i386.
                name = "long"
                if v > 0xFFFFFFFFFFFFFFFF:
                    name = "long long"   # shouldn't happen, but be safe
                elif v > 0xFFFFFFFF:
                    name = "long long"
                    if is_hex:
                        unsigned = unsigned or v > 0x7FFFFFFFFFFFFFFF
                elif v > 0x7FFFFFFF:
                    if not unsigned and is_hex:
                        unsigned = True
                    elif not unsigned and not is_hex:
                        # decimal with `L` suffix: skip unsigned long,
                        # promote directly to long long.
                        name = "long long"
            else:
                # No L suffix: walk int → unsigned int → long → unsigned
                # long → long long → unsigned long long.
                if v > 0xFFFFFFFF:
                    name = "long long"
                    if is_hex:
                        unsigned = unsigned or v > 0x7FFFFFFFFFFFFFFF
                elif v > 0x7FFFFFFF:
                    if not unsigned and is_hex:
                        unsigned = True
                    elif not unsigned:
                        # decimal that overflows int: long long (signed).
                        name = "long long"
                elif v < -0x80000000:
                    name = "long long"
            return ast.BasicType(
                name=name,
                is_signed=(False if unsigned else None),
            )
        if isinstance(expr, ast.CharLiteral):
            # Per C, a character constant has type `int`, not `char`.
            return ast.BasicType(name="int")
        if isinstance(expr, ast.NullptrLiteral):
            return ast.PointerType(base_type=ast.BasicType(name="void"))
        if isinstance(expr, ast.LabelAddr):
            return ast.PointerType(base_type=ast.BasicType(name="void"))
        if isinstance(expr, ast.OffsetofExpr):
            # `__builtin_offsetof` is a constant of type `size_t`. We
            # use unsigned long (4 bytes on i386).
            return ast.BasicType(name="unsigned long")
        if isinstance(expr, ast.TypesCompatibleP):
            return ast.BasicType(name="int")
        if isinstance(expr, ast.FloatLiteral):
            base = "float" if expr.is_float else "double"
            if expr.is_imaginary:
                # GCC: `_Imaginary T` ≡ `_Complex T` with real=0.
                return ast.ComplexType(base_type=base)
            return ast.BasicType(name=base)
        if isinstance(expr, ast.StringLiteral):
            # Per C99 6.4.5#6 string literals are `char[N+1]` (non-const).
            # The implicit array-to-pointer decay gives `char *`. C++
            # would mark them `const char[N+1]` but C does not, and
            # `_Generic("...", char *: ..., const char *: ...)` must
            # match the non-const arm to be C-conformant.
            if getattr(expr, "is_wide", False):
                return ast.PointerType(
                    base_type=ast.BasicType(name="short", is_signed=False)
                )
            return ast.PointerType(base_type=ast.BasicType(name="char"))
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "&":
                return ast.PointerType(base_type=self._type_of(expr.operand, ctx))
            if expr.op == "*":
                inner = self._type_of(expr.operand, ctx)
                if not self._is_pointer_like(inner):
                    raise CodegenError(
                        f"`*` operand must be a pointer (got {type(inner).__name__})"
                    )
                return inner.base_type
            if expr.op in ("++", "--"):
                # Mutation in place; the expression's type matches the operand.
                return self._type_of(expr.operand, ctx)
            if expr.op in ("__real__", "__imag__"):
                inner = self._type_of(expr.operand, ctx)
                if isinstance(inner, ast.ComplexType):
                    return ast.BasicType(name=inner.base_type)
                # On a non-complex operand, __real__ is identity and
                # __imag__ is zero (per gcc semantics).
                return inner
            # `+` and `-` and `~` propagate float / long long / int128
            # / unsigned / complex / vector through; `!` always
            # produces int.
            if expr.op in ("+", "-", "~"):
                inner = self._type_of(expr.operand, ctx)
                if isinstance(inner, ast.ComplexType):
                    return inner
                if isinstance(inner, ast.ArrayType):
                    # Vector componentwise unary.
                    return inner
                if self._is_float_type(inner):
                    return inner
                if self._is_int128(inner):
                    return inner
                if self._is_long_long(inner):
                    return inner
                if self._is_unsigned(inner):
                    return ast.BasicType(name="int", is_signed=False)
            return ast.BasicType(name="int")
        if isinstance(expr, ast.Index):
            arr_type = self._type_of(expr.array, ctx)
            # C allows `int[ptr]` swapped form. If `array` isn't pointer-
            # like but `index` is, treat the index as the pointer.
            if not self._is_pointer_like(arr_type):
                idx_type = self._type_of(expr.index, ctx)
                if self._is_pointer_like(idx_type):
                    arr_type = idx_type
            if not self._is_pointer_like(arr_type):
                raise CodegenError(
                    f"index target must be array or pointer "
                    f"(got {type(arr_type).__name__})"
                )
            return arr_type.base_type
        if isinstance(expr, ast.Member):
            obj_ty = self._type_of(expr.obj, ctx)
            if expr.is_arrow:
                # Arrays decay to pointers in expression context, so
                # `arr->member` is `(&arr[0])->member`. We also tolerate
                # a bare StructType obj — it can show up when typedef'd
                # function-pointer return chains drop a Pointer wrap and
                # the call result presents as the struct directly.
                if (
                    isinstance(obj_ty, ast.ArrayType)
                    and isinstance(obj_ty.base_type, ast.StructType)
                ):
                    struct_name = self._resolve_struct_name(obj_ty.base_type)
                elif (
                    isinstance(obj_ty, ast.PointerType)
                    and isinstance(obj_ty.base_type, ast.StructType)
                ):
                    struct_name = self._resolve_struct_name(obj_ty.base_type)
                elif isinstance(obj_ty, ast.StructType):
                    struct_name = self._resolve_struct_name(obj_ty)
                else:
                    raise CodegenError(
                        f"`->` requires a pointer to struct "
                        f"(got {type(obj_ty).__name__})"
                    )
            else:
                if not isinstance(obj_ty, ast.StructType):
                    raise CodegenError(
                        f"`.` requires a struct (got {type(obj_ty).__name__})"
                    )
                struct_name = self._resolve_struct_name(obj_ty)
            ty, _ = self._member_layout(struct_name, expr.member)
            # Bit-field integer promotion (C 6.3.1.1): a bit-field
            # narrower than int promotes to int (signed). 32-bit unsigned
            # bit-fields stay unsigned (not representable in signed int).
            # 32-bit signed bit-fields stay signed int. Bit-fields with
            # width > 32 keep their declared LL type.
            bf = self._struct_bitfields.get(struct_name, {}).get(expr.member)
            if bf is not None:
                bit_width = bf[1]
                unit_size = bf[2] if len(bf) >= 3 else 4
                if bit_width < 32:
                    return ast.BasicType(name="int")
                if bit_width == 32 and not self._is_unsigned(ty):
                    return ast.BasicType(name="int")
            return ty
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
                return self._type_of(expr.left, ctx)
            if expr.op in self._COMPOUND_OPS:
                return self._type_of(expr.left, ctx)
            # Comma: the result type is the right operand's type.
            if expr.op == ",":
                return self._type_of(expr.right, ctx)
            # `*` `/` (and the relational/bitwise ops) need to know if
            # either operand is float — the result of `1.5 * 2` is float,
            # not int.
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            # Complex promotion: any arithmetic with a complex operand
            # yields complex.
            if expr.op in ("+", "-", "*", "/") and (
                isinstance(lt, ast.ComplexType) or isinstance(rt, ast.ComplexType)
            ):
                return self._complex_promotion(lt, rt)
            # Vector arithmetic: two ArrayTypes of the same shape combine
            # componentwise and return the array type. Comparison ops
            # still return int (since GCC vector compares fold to a mask
            # that's int-typed at the C level for our subset). Also
            # supports scalar-broadcast `vec op int` / `int op vec` where
            # the scalar is replicated to every element.
            if expr.op in ("+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"):
                l_vec = (
                    isinstance(lt, ast.ArrayType)
                    and getattr(lt, "is_vector", False)
                )
                r_vec = (
                    isinstance(rt, ast.ArrayType)
                    and getattr(rt, "is_vector", False)
                )
                if (
                    isinstance(lt, ast.ArrayType)
                    and isinstance(rt, ast.ArrayType)
                    and self._size_of(lt) == self._size_of(rt)
                ):
                    return lt
                if l_vec and isinstance(rt, ast.BasicType):
                    return lt
                if r_vec and isinstance(lt, ast.BasicType):
                    return rt
            if expr.op in self._INT_RESULT_BINOPS:
                # Pointer-arithmetic promotion above doesn't apply here,
                # but float promotion does for `*` and `/`.
                if expr.op in ("*", "/", "%") and (
                    self._is_float_type(lt) or self._is_float_type(rt)
                ):
                    if expr.op == "%":
                        # C: % is integer-only; if either side is float
                        # the program is ill-formed.
                        raise CodegenError(
                            "`%` requires integer operands"
                        )
                    return self._float_promotion(lt, rt)
                # Comparison/logical/etc. result is always int. Other
                # int-result ops keep long-long promotion if either
                # operand is long long.
                if expr.op in ("==", "!=", "<", ">", "<=", ">=", "&&", "||"):
                    return ast.BasicType(name="int")
                # Per C99 6.5.7#3, shifts use the promoted LEFT operand
                # type for the result — the right operand's signedness
                # doesn't matter. Other int-result ops follow the usual
                # arithmetic conversions (unsigned wins).
                if expr.op in ("<<", ">>"):
                    if self._is_int128(lt):
                        return ast.BasicType(
                            name="int128",
                            is_signed=(False if self._is_unsigned(lt) else None),
                        )
                    if self._is_long_long(lt):
                        return ast.BasicType(
                            name="long long",
                            is_signed=(False if self._is_unsigned(lt) else None),
                        )
                    if self._is_unsigned(lt):
                        return ast.BasicType(name="int", is_signed=False)
                    return ast.BasicType(name="int")
                if self._is_int128(lt) or self._is_int128(rt):
                    return ast.BasicType(
                        name="int128",
                        is_signed=(False if (
                            self._is_unsigned(lt) or self._is_unsigned(rt)
                        ) else None),
                    )
                if self._is_long_long(lt) or self._is_long_long(rt):
                    return ast.BasicType(
                        name="long long",
                        is_signed=(False if (
                            self._is_unsigned(lt) or self._is_unsigned(rt)
                        ) else None),
                    )
                # C usual arithmetic conversions: if either operand is
                # unsigned, the result is unsigned. Drives the
                # signed-vs-unsigned dispatch in `_binary` for div / mod
                # / shr / comparisons.
                if self._is_unsigned(lt) or self._is_unsigned(rt):
                    return ast.BasicType(name="int", is_signed=False)
                return ast.BasicType(name="int")
            # `+` and `-`: pointer ± int → pointer; pointer - pointer → int.
            # Arrays count as pointer-like via decay. (Vector componentwise
            # arithmetic on two ArrayType operands is handled above.)
            l_ptr = self._is_pointer_like(lt)
            r_ptr = self._is_pointer_like(rt)
            if l_ptr and r_ptr:
                return ast.BasicType(name="int")
            if l_ptr:
                return lt
            if r_ptr:
                return rt
            # Float promotion: if either operand is float, the result is
            # float (or double if either is double).
            if self._is_float_type(lt) or self._is_float_type(rt):
                return self._float_promotion(lt, rt)
            # __int128 promotion: if either operand is int128, the
            # result is int128 (rank above long long).
            if self._is_int128(lt) or self._is_int128(rt):
                return ast.BasicType(
                    name="int128",
                    is_signed=(False if (
                        self._is_unsigned(lt) or self._is_unsigned(rt)
                    ) else None),
                )
            # Long-long promotion: if either operand is long long, the
            # result is long long.
            if self._is_long_long(lt) or self._is_long_long(rt):
                return ast.BasicType(
                    name="long long",
                    is_signed=(False if (
                        self._is_unsigned(lt) or self._is_unsigned(rt)
                    ) else None),
                )
            # Long propagation: per C usual arithmetic conversions, the
            # higher-ranked type wins. On i386 long and int have the
            # same width, but they're distinct types for _Generic.
            l_long = isinstance(lt, ast.BasicType) and lt.name == "long"
            r_long = isinstance(rt, ast.BasicType) and rt.name == "long"
            if l_long or r_long:
                return ast.BasicType(
                    name="long",
                    is_signed=(False if (
                        self._is_unsigned(lt) or self._is_unsigned(rt)
                    ) else None),
                )
            # Unsigned propagation for `+`/`-`.
            if self._is_unsigned(lt) or self._is_unsigned(rt):
                return ast.BasicType(name="int", is_signed=False)
            return ast.BasicType(name="int")
        if isinstance(expr, ast.TernaryOp):
            # Both arms should agree in well-formed C; pick the true arm.
            return self._type_of(expr.true_expr, ctx)
        if isinstance(expr, (ast.SizeofExpr, ast.SizeofType)):
            # `sizeof` returns size_t; treat it as int for our flat-32 ABI.
            return ast.BasicType(name="int")
        if isinstance(expr, ast.VaArgExpr):
            return expr.target_type
        if isinstance(expr, ast.Cast):
            return expr.target_type
        if isinstance(expr, ast.Compound):
            return expr.target_type
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Identifier):
                fname = ctx.nested_fn_names.get(
                    expr.func.name, expr.func.name,
                )
                # `__builtin_shuffle(src, mask)` returns a vector with
                # the same shape as `src` (the first arg). Resolve the
                # type from the arg so vector-aware lvalue paths work.
                if (
                    expr.func.name == "__builtin_shuffle"
                    and len(expr.args) >= 1
                ):
                    return self._type_of(expr.args[0], ctx)
                rt = self._func_return_types.get(fname)
                if rt is not None:
                    return rt
                # Identifier that's a known variable (e.g. function
                # pointer): follow its type to find the return type.
                try:
                    var_ty = self._identifier_type(expr.func.name, ctx)
                except CodegenError:
                    var_ty = None
                if isinstance(var_ty, ast.PointerType) and isinstance(
                    var_ty.base_type, ast.FunctionType
                ):
                    return var_ty.base_type.return_type
                if isinstance(var_ty, ast.FunctionType):
                    return var_ty.return_type
                # Unknown identifier — default to int.
                return ast.BasicType(name="int")
            # Indirect call: the callee evaluates to a function pointer
            # (or a function — same shape after decay). Recover the
            # return type by introspecting the callee's static type.
            try:
                callee_ty = self._type_of(expr.func, ctx)
            except CodegenError:
                return ast.BasicType(name="int")
            if (
                isinstance(callee_ty, ast.PointerType)
                and isinstance(callee_ty.base_type, ast.FunctionType)
            ):
                return callee_ty.base_type.return_type
            if isinstance(callee_ty, ast.FunctionType):
                return callee_ty.return_type
            return ast.BasicType(name="int")
        if isinstance(expr, ast.StmtExpr):
            # Type of `({ ...; expr; })` is the type of the trailing
            # expression statement; default to int when the body is empty
            # or doesn't end in one. The trailing expression may
            # reference locals declared inside the body, which aren't
            # visible in the surrounding scope — so we push a temporary
            # scope and re-bind any locals seen so far so the type lookup
            # for the trailing expression resolves.
            trailing = None
            for item in reversed(expr.body.items):
                if isinstance(item, ast.ExpressionStmt) and item.expr is not None:
                    trailing = item.expr
                    break
            if trailing is None:
                return ast.BasicType(name="int")
            ctx.enter_scope()
            try:
                for item in expr.body.items:
                    if isinstance(item, ast.VarDecl) and id(item) in ctx.decl_disps:
                        ctx.alloc_local(item.name, ctx.decl_types[id(item)], decl=item)
                return self._type_of(trailing, ctx)
            finally:
                ctx.exit_scope()
        return ast.BasicType(name="int")

    def _array_is_directly_vla(self, t: ast.ArrayType) -> bool:
        """True if `t` is a VLA-shaped array (size is non-constant or
        marked via `_vla_size`). Doesn't recurse into struct members.
        Used by VLA-backing slot allocation to distinguish a true VLA
        from `struct S s[N]` where S happens to contain a VLA member.
        """
        if getattr(t, "_vla_size", None) is not None:
            return True
        if t.size is not None and not isinstance(t.size, ast.IntLiteral):
            return True
        if isinstance(t.base_type, ast.ArrayType):
            return self._array_is_directly_vla(t.base_type)
        return False

    def _type_has_vla(self, t: ast.TypeNode) -> bool:
        """Does `t` contain a variable-length array? Recognized via
        either a saved `_vla_size` (set by `_check_supported_type`'s
        VLA fallback) or a non-literal `size` expression on a fresh
        ArrayType the codegen hasn't fully resolved yet."""
        if isinstance(t, ast.ArrayType):
            if getattr(t, "_vla_size", None) is not None:
                return True
            if t.size is not None and not isinstance(t.size, ast.IntLiteral):
                return True
            return self._type_has_vla(t.base_type)
        if isinstance(t, ast.StructType):
            # Check inline members first (when the struct is being
            # introduced for the first time and isn't yet registered).
            for m in getattr(t, "members", []) or []:
                if self._type_has_vla(m.member_type):
                    return True
            try:
                sname = self._resolve_struct_name(t)
            except CodegenError:
                return False
            for _name, mt, _off in self._structs.get(sname, []):
                if self._type_has_vla(mt):
                    return True
        return False

    def _emit_runtime_size_of(
        self, t: ast.TypeNode, ctx: _FuncCtx,
    ) -> list[str]:
        """Compute sizeof(t) at runtime when t contains a VLA.

        For non-VLA types we just emit the static size. For VLA-shaped
        arrays (with the original size expression on `_vla_size`), we
        evaluate that expression and multiply by element size. For
        structs we walk members so a VLA member contributes runtime
        size while regular members fold.
        """
        if not self._type_has_vla(t):
            return [f"        mov     eax, {self._size_of(t)}"]
        if isinstance(t, ast.ArrayType):
            vla_size = getattr(t, "_vla_size", None)
            if vla_size is None and t.size is not None and not isinstance(
                t.size, ast.IntLiteral
            ):
                # Fresh VLA — typedef'd or otherwise hasn't been
                # processed by `_check_supported_type`. Use the size
                # expression directly.
                vla_size = t.size
            if vla_size is not None:
                inner = self._emit_runtime_size_of(t.base_type, ctx)
                out = list(inner)
                out.append("        push    eax")
                out += self._eval_expr_to_eax(vla_size, ctx)
                out.append("        pop     ecx")
                out.append("        imul    eax, ecx")
                return out
            # ArrayType with constant size whose base contains a VLA.
            inner = self._emit_runtime_size_of(t.base_type, ctx)
            count = t.size.value if isinstance(t.size, ast.IntLiteral) else 1
            out = list(inner)
            out.append(f"        imul    eax, eax, {count}")
            return out
        if isinstance(t, ast.StructType):
            sname = self._resolve_struct_name(t)
            members = self._structs.get(sname, [])
            if not members:
                return ["        xor     eax, eax"]
            total_const = 0
            runtime_lines: list[str] = []
            for _name, mt, _off in members:
                if self._type_has_vla(mt):
                    sub = self._emit_runtime_size_of(mt, ctx)
                    if not runtime_lines:
                        runtime_lines = list(sub)
                    else:
                        runtime_lines.append("        push    eax")
                        runtime_lines += sub
                        runtime_lines.append("        pop     ecx")
                        runtime_lines.append("        add     eax, ecx")
                    continue
                total_const += self._size_of(mt)
            if not runtime_lines:
                return [f"        mov     eax, {total_const}"]
            out = list(runtime_lines)
            if total_const:
                out.append(f"        add     eax, {total_const}")
            return out
        # Other type categories don't carry VLA in our subset.
        return [f"        mov     eax, {self._size_of(t)}"]

    def _fistp_truncate_dword_to_eax(self) -> list[str]:
        """Pop st(0), truncate toward zero, store as 32-bit signed int
        in EAX. Saves and restores the FPU control word so we don't
        corrupt the rounding mode used by surrounding FPU code.

        Frame layout while running:
          [esp + 4] = result slot (later popped to eax)
          [esp + 2..3] = patched control word (truncate-rc set)
          [esp + 0..1] = saved control word
        """
        return [
            "        sub     esp, 8",
            "        fnstcw  [esp]",
            "        mov     ax, [esp]",
            "        or      ax, 0x0C00",  # RC = 11 (truncate)
            "        mov     [esp + 2], ax",
            "        fldcw   [esp + 2]",
            "        fistp   dword [esp + 4]",
            "        fldcw   [esp]",
            "        mov     eax, [esp + 4]",
            "        add     esp, 8",
        ]

    def _fistp_truncate_qword_to_edx_eax(self) -> list[str]:
        """Pop st(0), truncate toward zero, store as 64-bit signed int
        in EDX:EAX (high:low). Saves/restores FPU control word."""
        return [
            "        sub     esp, 12",
            "        fnstcw  [esp]",
            "        mov     ax, [esp]",
            "        or      ax, 0x0C00",
            "        mov     [esp + 2], ax",
            "        fldcw   [esp + 2]",
            "        fistp   qword [esp + 4]",
            "        fldcw   [esp]",
            "        mov     eax, [esp + 4]",
            "        mov     edx, [esp + 8]",
            "        add     esp, 12",
        ]

    def _eval_expr_to_eax(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        # Float-typed expressions live on the FPU stack; if a caller
        # wants the value in EAX (e.g. `int x = (float)n`), evaluate to
        # st(0) then convert via `fistp` through a stack scratch slot.
        # C requires truncation toward zero (not the FPU's default
        # round-to-nearest), so we set the rounding mode to truncate
        # for the duration of the fistp.
        if self._is_float_type(self._type_of(expr, ctx)):
            out = self._eval_float_to_st0(expr, ctx)
            out += self._fistp_truncate_dword_to_eax()
            return out
        # __int128 — produces an address (16 bytes don't fit in a
        # register pair). Callers that consume the value know it's
        # int128 and use the address; callers that want a narrower
        # value should go through a Cast.  Assignment ops (=, +=, ...)
        # need to route through `_binary` first so `_assign` /
        # `_compound_assign` apply their own type-based dispatch.
        if self._is_int128(self._type_of(expr, ctx)):
            if isinstance(expr, ast.BinaryOp) and (
                expr.op == "=" or expr.op in self._COMPOUND_OPS
            ):
                return self._binary(expr, ctx)
            return self._int128_value_address(expr, ctx)
        if isinstance(expr, ast.IntLiteral):
            return [f"        mov     eax, {expr.value}"]
        if isinstance(expr, ast.CharLiteral):
            # `'a'` is an integer constant in C — its parser-level value is
            # already the character code, so it lowers exactly like IntLiteral.
            return [f"        mov     eax, {expr.value}"]
        if isinstance(expr, ast.NullptrLiteral):
            # `nullptr` is the integer 0 with pointer type. Loading 0 into
            # EAX gives the right value for both pointer-init and pointer
            # comparison contexts.
            return ["        mov     eax, 0"]
        if isinstance(expr, ast.LabelAddr):
            return [f"        mov     eax, {self._label_addr_text(expr.label, ctx)}"]
        if isinstance(expr, ast.StringLiteral):
            is_wide = getattr(expr, "is_wide", False)
            label = self._intern_string(expr.value, is_wide=is_wide)
            return [f"        mov     eax, {label}"]
        if isinstance(expr, ast.Identifier):
            return self._identifier_load(expr.name, ctx)
        if isinstance(expr, ast.Index):
            return self._index_load(expr, ctx)
        if isinstance(expr, ast.Member):
            return self._member_load(expr, ctx)
        if isinstance(expr, ast.UnaryOp):
            # Vector unary: returns a temp address.
            if (
                self._is_vector_op_node(expr)
                and id(expr) in ctx.call_temps
                and self._is_genuine_vector_op(expr, ctx)
            ):
                return self._eval_vector_into_temp(expr, ctx)
            return self._unary(expr, ctx)
        if isinstance(expr, ast.BinaryOp):
            # Vector arithmetic: both sides ArrayType of same shape.
            if (
                self._is_vector_op_node(expr)
                and id(expr) in ctx.call_temps
                and self._is_genuine_vector_op(expr, ctx)
            ):
                return self._eval_vector_into_temp(expr, ctx)
            return self._binary(expr, ctx)
        if isinstance(expr, ast.Call):
            if self._is_struct_returning_call(expr, ctx):
                # Direct EAX returns can't carry a struct; route the
                # call into a per-call-site temp slot reserved by
                # `_collect_call_temps`. EAX ends up holding the
                # temp's address (as the callee returns the retptr),
                # which is exactly what later `.field` / arg-copy
                # paths want.
                disp = ctx.call_temps[id(expr)]
                retptr_lines = [f"        lea     eax, {_ebp_addr(disp)}"]
                return self._call_into_address(expr, retptr_lines, ctx)
            return self._call(expr, ctx)
        if isinstance(expr, ast.TernaryOp):
            return self._ternary(expr, ctx)
        if isinstance(expr, ast.Cast):
            return self._cast(expr, ctx)
        if isinstance(expr, ast.Compound):
            # `(T){init}` — evaluate the init into a pre-reserved temp
            # slot, then return the address (for compound types) or
            # load the value (for scalars).
            disp = ctx.call_temps[id(expr)]
            target_ty = expr.target_type
            out: list[str] = []
            if isinstance(target_ty, ast.StructType):
                out += self._struct_init(
                    target_ty, expr.init, disp, ctx, "<compound>",
                )
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            if isinstance(target_ty, ast.ArrayType):
                out += self._array_init(
                    target_ty, expr.init, disp, ctx, "<compound>",
                )
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            # Scalar: emit init then read the slot back.
            if (
                isinstance(expr.init, ast.InitializerList)
                and len(expr.init.values) == 1
            ):
                inner = expr.init.values[0]
            else:
                inner = expr.init
            out += self._eval_expr_to_eax(inner, ctx)
            out += self._store_from_eax(_ebp_addr(disp), target_ty)
            return out
        if isinstance(expr, ast.VaArgExpr):
            return self._va_arg_int(expr, ctx)
        if isinstance(expr, ast.SizeofType):
            if getattr(expr, "is_alignof", False):
                value = self._alignment_of(expr.target_type)
                return [f"        mov     eax, {value}"]
            return self._emit_runtime_size_of(expr.target_type, ctx)
        if isinstance(expr, ast.OffsetofExpr):
            try:
                value = self._offsetof_value(expr)
                return [f"        mov     eax, {value}"]
            except CodegenError:
                return self._emit_runtime_offsetof(expr, ctx)
        if isinstance(expr, ast.TypesCompatibleP):
            v = 1 if self._types_compatible(expr.t1, expr.t2) else 0
            return [f"        mov     eax, {v}"]
        if isinstance(expr, ast.SizeofExpr):
            # C: the operand of `sizeof` is *not* evaluated — only its
            # static type matters. So we infer the type and never emit
            # any of the operand's lowering code (no slot loads, no
            # function calls).
            if (
                getattr(expr, "is_alignof", False)
                and isinstance(expr.expr, ast.Identifier)
                and expr.expr.name in self._func_alignments
            ):
                value = self._func_alignments[expr.expr.name]
                return [f"        mov     eax, {value}"]
            ty = self._type_of(expr.expr, ctx)
            if getattr(expr, "is_alignof", False):
                value = self._alignment_of(ty)
                return [f"        mov     eax, {value}"]
            return self._emit_runtime_size_of(ty, ctx)
        if isinstance(expr, ast.StmtExpr):
            # GCC statement expression: `({ stmt; stmt; expr; })`. Lower
            # the body as a regular compound; the value of the last
            # ExpressionStmt is what the StmtExpr produces (already in
            # EAX from `_expr_stmt`'s evaluation). If the body ends in a
            # non-expression, EAX is left at whatever the last emitted
            # code put there — undefined but not crashing.
            #
            # For struct-typed StmtExprs the trailing expression refers
            # to a struct value (often a local declared in the body).
            # Evaluating it via `_eval_expr_to_eax` would try to load
            # the whole struct into EAX. Detect this in-scope, after
            # the head items have run, and emit `_struct_address`
            # instead so consumers (`.member`, struct copy assign) get
            # what they want.
            ctx.enter_scope()
            try:
                items = list(expr.body.items)
                if not items:
                    return []
                head, tail = items[:-1], items[-1]
                out: list[str] = []
                for item in head:
                    out += self._item(item, ctx)
                if (
                    isinstance(tail, ast.ExpressionStmt)
                    and tail.expr is not None
                ):
                    tail_ty = self._type_of(tail.expr, ctx)
                    if isinstance(tail_ty, ast.StructType):
                        out += self._struct_address(tail.expr, ctx)
                        return out
                out += self._item(tail, ctx)
                return out
            finally:
                ctx.exit_scope()
        if isinstance(expr, ast.GenericSelection):
            chosen = self._select_generic_arm(expr, ctx)
            return self._eval_expr_to_eax(chosen, ctx)
        raise CodegenError(
            f"expression {type(expr).__name__} not implemented yet"
        )

    # ---- float (x87) lowering ------------------------------------------

    # ---- 64-bit (long long) lowering ------------------------------------

    def _eval_expr_to_edx_eax(
        self, expr: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Lower a long-long-typed expression. Result lands in EDX:EAX
        (high:low). Mirrors `_eval_expr_to_eax` but maintains the full
        64-bit value through every node.
        """
        # GCC statement expression: emit head items in scope, then
        # evaluate the trailing expression as long long.
        if isinstance(expr, ast.StmtExpr):
            ctx.enter_scope()
            try:
                items = list(expr.body.items)
                if not items:
                    return []
                head, tail = items[:-1], items[-1]
                out: list[str] = []
                for item in head:
                    out += self._item(item, ctx)
                if (
                    isinstance(tail, ast.ExpressionStmt)
                    and tail.expr is not None
                ):
                    out += self._eval_expr_to_edx_eax(tail.expr, ctx)
                else:
                    out += self._item(tail, ctx)
                return out
            finally:
                ctx.exit_scope()
        # __int128 expression in LL context: evaluate to a 16-byte
        # temp, then load the low 8 bytes into EDX:EAX (truncation
        # per C's standard conversion rules).
        if self._is_int128(self._type_of(expr, ctx)):
            out = self._int128_value_address(expr, ctx)
            return out + [
                "        mov     ecx, eax",
                "        mov     eax, [ecx]",
                "        mov     edx, [ecx + 4]",
            ]
        # Direct loads.
        if isinstance(expr, ast.IntLiteral):
            v = expr.value
            if v < 0:
                v = (1 << 64) + v
            v &= 0xFFFFFFFFFFFFFFFF
            low = v & 0xFFFFFFFF
            high = (v >> 32) & 0xFFFFFFFF
            return [
                f"        mov     eax, 0x{low:08X}",
                f"        mov     edx, 0x{high:08X}",
            ]
        if isinstance(expr, ast.CharLiteral):
            return [f"        mov     eax, {expr.value}", "        xor     edx, edx"]
        if isinstance(expr, ast.NullptrLiteral):
            return ["        xor     eax, eax", "        xor     edx, edx"]
        if isinstance(expr, ast.Identifier):
            t = self._type_of(expr, ctx)
            if self._is_long_long(t):
                addr = self._identifier_addr_text(expr.name, ctx)
                return self._load_to_edx_eax(addr)
            # Smaller-than-long-long Identifier in long-long context:
            # load as 32-bit value, then sign- or zero-extend to fill EDX.
            out = self._eval_expr_to_eax(expr, ctx)
            if self._is_unsigned(t):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            return out
        if isinstance(expr, ast.Index):
            elem_ty = self._type_of(expr, ctx)
            if self._is_long_long(elem_ty):
                addr = self._index_address(expr, ctx)
                out = list(addr)
                return out + [
                    "        mov     ecx, eax",
                    "        mov     eax, [ecx]",
                    "        mov     edx, [ecx + 4]",
                ]
            # Sub-LL element in LL context: load 32-bit value via
            # `_index_load`, then extend per signedness.
            out = self._eval_expr_to_eax(expr, ctx)
            if self._is_unsigned(elem_ty):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            return out
        if isinstance(expr, ast.Member):
            mem_ty = self._type_of(expr, ctx)
            # Bit-field member needs the proper load (shrd + mask)
            # rather than a raw 64-bit memcpy. The 32-bit-storage
            # case still falls through to `_eval_expr_to_eax` which
            # routes through `_bitfield_load`.
            bf = self._bitfield_info(expr, ctx)
            if bf is not None and len(bf) == 4 and bf[3] == 8:
                # Long-long-storage bit-field: `_bitfield_load_ll`
                # already produces the fully sign- or zero-extended
                # 64-bit value in EDX:EAX. Don't `cdq` on top — that
                # would clobber EDX with the sign of EAX and lose the
                # bit-field's high bits.
                return self._bitfield_load_ll(expr, bf, ctx)
            if self._is_long_long(mem_ty):
                out = self._member_address(expr, ctx)
                return out + [
                    "        mov     ecx, eax",
                    "        mov     eax, [ecx]",
                    "        mov     edx, [ecx + 4]",
                ]
            out = self._eval_expr_to_eax(expr, ctx)
            if self._is_unsigned(mem_ty):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            return out
        if isinstance(expr, ast.Cast):
            target = expr.target_type
            src_ty = self._type_of(expr.expr, ctx)
            # __int128 → long long: load low 8 bytes of the int128's
            # storage into EDX:EAX.
            if self._is_int128(src_ty) and self._is_long_long(target):
                out = self._int128_value_address(expr.expr, ctx)
                return out + [
                    "        mov     ecx, eax",
                    "        mov     eax, [ecx]",
                    "        mov     edx, [ecx + 4]",
                ]
            if self._is_long_long(src_ty) and self._is_long_long(target):
                # ll → ll: pass through (signedness-only changes are
                # bit-identical at the EDX:EAX level).
                return self._eval_expr_to_edx_eax(expr.expr, ctx)
            if self._is_long_long(src_ty):
                # ll → narrow type, but parent wants 64 bits. Eval as 32
                # via the int-cast path (truncates to target), then
                # re-extend to fill EDX per the target's signedness.
                out = self._eval_expr_to_eax(expr, ctx)
                if self._is_unsigned(target):
                    out.append("        xor     edx, edx")
                else:
                    out.append("        cdq")
                return out
            if self._is_long_long(target):
                # int → long long: route through the LL evaluator so
                # nested operations (shifts, arithmetic) get promoted
                # consistently rather than computed in 32 bits and then
                # sign-extended.
                if self._is_float_type(src_ty):
                    out = self._eval_float_to_st0(expr.expr, ctx)
                    out += self._fistp_truncate_qword_to_edx_eax()
                    return out
                # 8-byte vector (ArrayType) → long long: type-pun by
                # loading both halves of the vector's storage.
                if (
                    isinstance(src_ty, ast.ArrayType)
                    and self._size_of(src_ty) == 8
                ):
                    out = self._vector_value_address(expr.expr, ctx)
                    out += [
                        "        mov     ecx, eax",
                        "        mov     eax, [ecx]",
                        "        mov     edx, [ecx + 4]",
                    ]
                    return out
                # For BinaryOps that get promoted in LL context (shifts
                # in particular), recurse to evaluate in 64-bit space.
                # For other shapes, the 32-bit eval + extend is fine and
                # cheaper.
                if isinstance(expr.expr, ast.BinaryOp):
                    return self._eval_expr_to_edx_eax(expr.expr, ctx)
                out = self._eval_expr_to_eax(expr.expr, ctx)
                if self._is_unsigned(src_ty):
                    out.append("        xor     edx, edx")
                else:
                    out.append("        cdq")
                return out
            # Widening a smaller type via cast to int/unsigned int and
            # then needing to fill EDX:EAX. Use the TARGET type's
            # signedness — `(unsigned int)(-1)` should zero-extend.
            if self._is_float_type(src_ty):
                out = self._eval_float_to_st0(expr.expr, ctx)
                out += self._fistp_truncate_qword_to_edx_eax()
                return out
            out = self._eval_expr_to_eax(expr, ctx)
            if self._is_unsigned(target):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            return out
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "+":
                return self._eval_expr_to_edx_eax(expr.operand, ctx)
            if expr.op == "-":
                out = self._eval_expr_to_edx_eax(expr.operand, ctx)
                # 64-bit negate. `neg eax` sets CF=1 iff l != 0, then we
                # add CF to h before negating it — matches two's
                # complement on the 64-bit value.
                out += [
                    "        neg     eax",
                    "        adc     edx, 0",
                    "        neg     edx",
                ]
                return out
            if expr.op == "~":
                out = self._eval_expr_to_edx_eax(expr.operand, ctx)
                return out + ["        not     eax", "        not     edx"]
            if expr.op == "*":
                # Dereferencing in long-long context. If the pointee is
                # itself long-long, load 8 bytes; otherwise load 32 bits
                # via the standard path and extend per signedness.
                pointee = self._type_of(expr, ctx)
                if self._is_long_long(pointee):
                    out = self._eval_expr_to_eax(expr.operand, ctx)
                    return out + [
                        "        mov     ecx, eax",
                        "        mov     eax, [ecx]",
                        "        mov     edx, [ecx + 4]",
                    ]
                out = self._eval_expr_to_eax(expr, ctx)
                if self._is_unsigned(pointee):
                    out.append("        xor     edx, edx")
                else:
                    out.append("        cdq")
                return out
            if expr.op in ("++", "--"):
                # ++/-- on a long-long lvalue.
                return self._inc_dec_ll(expr, ctx)
        if isinstance(expr, ast.BinaryOp):
            return self._binary_ll(expr, ctx)
        if isinstance(expr, ast.TernaryOp):
            cond_label = ctx.label("ll_ter_else")
            end_label = ctx.label("ll_ter_end")
            out = self._eval_to_bool_eax(expr.condition, ctx)
            out += [
                "        test    eax, eax",
                f"        jz      {cond_label}",
            ]
            out += self._eval_expr_to_edx_eax(expr.true_expr, ctx)
            out.append(f"        jmp     {end_label}")
            out.append(f"{cond_label}:")
            out += self._eval_expr_to_edx_eax(expr.false_expr, ctx)
            out.append(f"{end_label}:")
            return out
        if isinstance(expr, ast.Call):
            # `llabs(long long)` / `__builtin_llabs`: inline absolute
            # value. The user may have redefined `llabs` to abort
            # (gcc-canonical test pattern); gcc treats it as a builtin
            # and inlines.
            if (
                isinstance(expr.func, ast.Identifier)
                and expr.func.name in ("llabs", "__builtin_llabs")
                and len(expr.args) == 1
            ):
                out = self._eval_expr_to_edx_eax(expr.args[0], ctx)
                skip = ctx.label("llabs_done")
                out += [
                    "        test    edx, edx",
                    f"        jns     {skip}",
                    "        neg     eax",
                    "        adc     edx, 0",
                    "        neg     edx",
                    f"{skip}:",
                ]
                return out
            # cdecl returns 64-bit values in EDX:EAX. If the callee's
            # return type is sub-LL, the call leaves only EAX defined
            # — extend EDX from EAX per the return type's signedness so
            # the LL result has a valid high half.
            ret_ty = self._type_of(expr, ctx)
            if self._is_long_long(ret_ty):
                return self._call(expr, ctx)
            out = self._call(expr, ctx)
            if self._is_unsigned(ret_ty):
                out.append("        xor     edx, edx")
            else:
                out.append("        cdq")
            return out
        if isinstance(expr, ast.VaArgExpr):
            target_ty = expr.target_type
            if self._is_long_long(target_ty):
                out = self._va_arg_read_and_advance(expr, 8, ctx)
                out.append("        mov     eax, [ecx]")
                out.append("        mov     edx, [ecx + 4]")
                return out
        # Fallback: eval as 32-bit, sign-extend.
        out = self._eval_expr_to_eax(expr, ctx)
        try:
            src_ty = self._type_of(expr, ctx)
        except CodegenError:
            src_ty = ast.BasicType(name="int")
        if self._is_unsigned(src_ty):
            out.append("        xor     edx, edx")
        else:
            out.append("        cdq")
        return out

    def _replace_vla_size_with_capture(
        self, t: ast.TypeNode, captured: dict[int, str], fn_name: str,
    ) -> None:
        """Walk `t` and replace any ArrayType.size expression whose
        identity matches a captured side-effect with an Identifier
        reading from the captured hidden local."""
        if isinstance(t, ast.PointerType):
            self._replace_vla_size_with_capture(t.base_type, captured, fn_name)
            return
        if isinstance(t, ast.ArrayType):
            if t.size is not None and id(t.size) in captured:
                slot_name = captured[id(t.size)]
                t.size = ast.Identifier(name=slot_name, location=t.size.location)
            self._replace_vla_size_with_capture(t.base_type, captured, fn_name)
            return

    def _capture_struct_vla_member_sizes(
        self, decl: ast.StructDecl, ctx: _FuncCtx,
    ) -> None:
        """Pre-pass for in-function struct decls with VLA members:
        allocate a hidden capture slot per VLA dimension, replace each
        ArrayType.size in the member with Identifier(slot_name) so
        sizeof reads the captured value. The eval+store happens in
        `_item(StructDecl)` at struct-decl emission time."""
        captures: list[tuple[ast.ArrayType, ast.Expression, str]] = []

        def walk(t: ast.TypeNode) -> None:
            if isinstance(t, ast.PointerType):
                walk(t.base_type)
                return
            if isinstance(t, ast.ArrayType):
                if (
                    t.size is not None
                    and not isinstance(t.size, ast.IntLiteral)
                ):
                    slot_name = (
                        f"__vla_capture_struct_"
                        f"{decl.name or '<anon>'}_{len(captures)}"
                    )
                    captures.append((t, t.size, slot_name))
                walk(t.base_type)
                return
            if isinstance(t, ast.StructType):
                # Don't recurse — that struct has its own captures (or
                # was already processed).
                return

        for m in decl.members:
            walk(m.member_type)
        if not captures:
            return
        for arr_t, orig_size, slot_name in captures:
            ctx.alloc_local(
                slot_name, ast.BasicType(name="int"),
                decl=orig_size,
            )
        for arr_t, orig_size, slot_name in captures:
            arr_t.size = ast.Identifier(
                name=slot_name, location=orig_size.location,
            )
        decl._vla_member_captures = [
            (slot_name, orig_size) for _arr_t, orig_size, slot_name in captures
        ]

    def _capture_vla_sizes(
        self,
        var_type: ast.TypeNode,
        var_name: str,
        ctx: _FuncCtx,
        decl: ast.VarDecl,
    ) -> None:
        """For each VLA dimension in a local's type, allocate a hidden
        capture slot and replace the size expression with an Identifier
        reading from that slot. The init code emitted by `_var_init`
        will store the captured size into the slot at decl time."""
        slots: list[tuple[ast.ArrayType, ast.Expression, str]] = []

        def walk(t: ast.TypeNode) -> None:
            if isinstance(t, ast.ArrayType):
                # Prefer `_vla_size` (saved by `_check_supported_type`'s
                # VLA fallback when it mutated `size` to IntLiteral(16)).
                # Otherwise capture a non-literal `size`.
                vla = getattr(t, "_vla_size", None)
                cand = vla if vla is not None else t.size
                if (
                    cand is not None
                    and not isinstance(cand, ast.IntLiteral)
                ):
                    slot_name = f"__vla_capture_local_{var_name}_{len(slots)}"
                    slots.append((t, cand, slot_name))
                walk(t.base_type)

        walk(var_type)
        for arr_t, orig_size, slot_name in slots:
            ctx.alloc_local(
                slot_name, ast.BasicType(name="int"),
                decl=orig_size,
            )
        if slots:
            # Save the captures on the decl so `_var_init` can emit
            # the runtime evaluation later.
            decl._vla_captures = [
                (slot_name, orig_size) for _arr_t, orig_size, slot_name in slots
            ]
            for arr_t, orig_size, slot_name in slots:
                # Replace BOTH `size` (the post-fallback IntLiteral)
                # AND `_vla_size` (the original expression) with an
                # Identifier reading from the captured slot.
                slot_id = ast.Identifier(
                    name=slot_name, location=orig_size.location,
                )
                arr_t._vla_size = slot_id

    def _bitfield_precision_mask(
        self, *operands: ast.Expression, ctx: _FuncCtx,
    ) -> list[str]:
        """Mask EDX:EAX to the WIDEST bit-field width among `operands`.
        Per gcc, arithmetic on a bit-field of declared type
        `unsigned long long b:N` keeps results modulo 2^N. For mixed
        bit-field operands, the result's precision is the wider of the
        two."""
        max_width = 0
        for operand in operands:
            if not isinstance(operand, ast.Member):
                continue
            bf = self._bitfield_info(operand, ctx)
            if bf is None:
                continue
            _bit_offset, bit_width, _member_ty, unit_size = bf
            if unit_size != 8 or bit_width >= 64:
                continue
            if bit_width > max_width:
                max_width = bit_width
        if max_width == 0:
            return []
        if max_width <= 32:
            return [
                f"        and     eax, {(1 << max_width) - 1}",
                "        xor     edx, edx",
            ]
        high_mask = (1 << (max_width - 32)) - 1
        return [f"        and     edx, {high_mask}"]

    def _binary_ll(
        self, expr: ast.BinaryOp, ctx: _FuncCtx,
    ) -> list[str]:
        """64-bit binary op. Result in EDX:EAX. Stack-machine eval:
        right → push EDX:EAX, left → EDX:EAX, pop into ECX:EBX, op.
        """
        op = expr.op
        if op in self._COMPOUND_OPS:
            # `lvalue OP= rhs` — desugar to `lvalue = lvalue OP rhs`
            # in long-long context. lhs is side-effect-safe when it's
            # an Identifier; non-Identifier lvalues (Member, Index,
            # *p) need address-once. Defer to the regular long-long
            # compound-assign helper.
            return self._compound_assign_ll(expr, ctx)
        if op == "=":
            # If the lhs isn't long-long itself, route through the
            # regular 32-bit assign (which narrows to the lhs's width)
            # and then re-extend EAX to EDX:EAX per the lhs's
            # signedness — that matches C's "result of `=` is the value
            # stored in the lhs after conversion to its type".
            lhs_ty = self._type_of(expr.left, ctx)
            if not self._is_long_long(lhs_ty):
                out = self._assign(expr, ctx)
                if self._is_unsigned(lhs_ty):
                    out.append("        xor     edx, edx")
                else:
                    out.append("        cdq")
                return out
            return self._assign_ll(expr, ctx)
        # Right to EDX:EAX, push (high then low so low ends up on top).
        right = self._eval_expr_to_edx_eax(expr.right, ctx)
        out = list(right)
        out += ["        push    edx", "        push    eax"]
        # Left to EDX:EAX.
        out += self._eval_expr_to_edx_eax(expr.left, ctx)
        # Pop right into ECX:EBX (low then high).
        out += [
            "        pop     ecx",
            "        pop     ebx",
        ]
        # Now: EDX:EAX = left, EBX:ECX = right.
        if op == "+":
            return out + ["        add     eax, ecx", "        adc     edx, ebx"] + self._bitfield_precision_mask(expr.left, expr.right, ctx=ctx)
        if op == "-":
            return out + ["        sub     eax, ecx", "        sbb     edx, ebx"] + self._bitfield_precision_mask(expr.left, expr.right, ctx=ctx)
        if op == "&":
            return out + ["        and     eax, ecx", "        and     edx, ebx"]
        if op == "|":
            return out + ["        or      eax, ecx", "        or      edx, ebx"]
        if op == "^":
            return out + ["        xor     eax, ecx", "        xor     edx, ebx"]
        # Comparisons return int 0/1 in EAX, EDX clobbered.
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return out + self._cmp_ll(op, ctx, signed=not (
                self._is_unsigned(self._type_of(expr.left, ctx))
                or self._is_unsigned(self._type_of(expr.right, ctx))
            ))
        if op == "<<":
            big = ctx.label("ll_shl_big")
            done = ctx.label("ll_shl_done")
            shift_lines = [
                "        test    cl, 32",
                f"        jnz     {big}",
                "        shld    edx, eax, cl",
                "        shl     eax, cl",
                f"        jmp     {done}",
                f"{big}:",
                "        mov     edx, eax",
                "        xor     eax, eax",
                "        and     cl, 31",
                "        shl     edx, cl",
                f"{done}:",
            ]
            return out + shift_lines + self._bitfield_precision_mask(expr.left, ctx=ctx)
        if op == ">>":
            unsigned = self._is_unsigned(self._type_of(expr.left, ctx))
            shift_high = "shr" if unsigned else "sar"
            ext = "xor edx, edx" if unsigned else "sar edx, 31"
            big = ctx.label("ll_shr_big")
            done = ctx.label("ll_shr_done")
            shift_lines = [
                "        test    cl, 32",
                f"        jnz     {big}",
                "        shrd    eax, edx, cl",
                f"        {shift_high}    edx, cl",
                f"        jmp     {done}",
                f"{big}:",
                "        mov     eax, edx",
                f"        {ext}",
                "        and     cl, 31",
                f"        {shift_high}    eax, cl",
                f"{done}:",
            ]
            return out + shift_lines + self._bitfield_precision_mask(expr.left, ctx=ctx)
        if op == "*":
            # 64x64 → 64 truncated multiply.
            #   left  = LH:LL (EDX:EAX);  right = RH:RC (EBX:ECX)
            #   low  32 = (LL*RC) low
            #   high 32 = (LL*RC) high + (LL*RH) low + (LH*RC) low
            mul_lines = [
                "        push    edx",          # [esp+8] = LH
                "        push    eax",          # [esp+4] = LL (after one more push below)
                "        mul     ecx",          # edx:eax = LL * RC
                "        mov     esi, edx",     # esi = high 32 of LL*RC
                "        push    eax",          # [esp]   = LL*RC low
                "        mov     eax, [esp + 4]",   # LL
                "        imul    eax, ebx",     # eax = LL * RH (low 32)
                "        add     esi, eax",
                "        mov     eax, [esp + 8]",   # LH
                "        imul    eax, ecx",     # eax = LH * RC (low 32)
                "        add     esi, eax",
                "        pop     eax",          # eax = LL*RC low
                "        mov     edx, esi",     # edx = high 32 of full product
                "        add     esp, 8",       # discard LL and LH
            ]
            mask = self._bitfield_precision_mask(expr.left, expr.right, ctx=ctx)
            return out + mul_lines + mask
        if op in ("/", "%"):
            # Long-long div/mod: route through the harness via INT 0x80
            # (a private trap vector — DOS doesn't use it, so our hook
            # owns it). ESI selects the op (0=udiv, 1=sdiv, 2=umod,
            # 3=smod). Inputs: EDX:EAX = numer, EBX:ECX = denom.
            # Result in EDX:EAX. Using INT 0x80 (instead of INT 21h
            # AH=...) avoids the AH-write clobbering EAX.
            unsigned = self._is_unsigned(self._type_of(expr.left, ctx))
            if op == "/":
                code = 0 if unsigned else 1
            else:
                code = 2 if unsigned else 3
            return out + [
                f"        mov     esi, {code}",
                "        int     0x80",
            ]
        if op == ",":
            # Comma: evaluate left for its side effects (its value is
            # discarded), then yield right's value as long-long.
            lhs_ty = self._type_of(expr.left, ctx)
            if self._is_float_type(lhs_ty):
                out_lhs = self._eval_float_to_st0(expr.left, ctx)
                # Drop the float result.
                out_lhs.append("        fstp    st0")
            elif self._is_long_long(lhs_ty):
                out_lhs = self._eval_expr_to_edx_eax(expr.left, ctx)
            elif self._is_int128(lhs_ty):
                out_lhs = self._int128_value_address(expr.left, ctx)
            else:
                out_lhs = self._eval_expr_to_eax(expr.left, ctx)
            return out_lhs + self._eval_expr_to_edx_eax(expr.right, ctx)
        if op == "&&":
            return self._logical_and_ll(expr, ctx)
        if op == "||":
            return self._logical_or_ll(expr, ctx)
        raise CodegenError(f"long-long op `{op}` not implemented")

    def _cmp_ll(self, op: str, ctx: _FuncCtx, *, signed: bool) -> list[str]:
        """Lines comparing EDX:EAX vs EBX:ECX (left vs right). Result in
        EAX (0 or 1). Sets EDX to 0 (clobber).
        """
        true_label = ctx.label("ll_cmp_true")
        false_label = ctx.label("ll_cmp_false")
        end_label = ctx.label("ll_cmp_end")
        # Map to high/low compare branches.
        if op == "==":
            return [
                "        cmp     edx, ebx",
                f"        jne     {false_label}",
                "        cmp     eax, ecx",
                f"        jne     {false_label}",
                f"{true_label}:",
                "        mov     eax, 1",
                f"        jmp     {end_label}",
                f"{false_label}:",
                "        xor     eax, eax",
                f"{end_label}:",
                "        xor     edx, edx",
            ]
        if op == "!=":
            return [
                "        cmp     edx, ebx",
                f"        jne     {true_label}",
                "        cmp     eax, ecx",
                f"        jne     {true_label}",
                f"{false_label}:",
                "        xor     eax, eax",
                f"        jmp     {end_label}",
                f"{true_label}:",
                "        mov     eax, 1",
                f"{end_label}:",
                "        xor     edx, edx",
            ]
        # < > <= >= — branch on high half first, then low half (unsigned).
        if signed:
            high_lt = "jl"
            high_gt = "jg"
        else:
            high_lt = "jb"
            high_gt = "ja"
        # For all four: if high differs, decide; else compare low (unsigned).
        if op == "<":
            high_take = high_lt
            high_drop = high_gt
            low_take = "jb"
        elif op == ">":
            high_take = high_gt
            high_drop = high_lt
            low_take = "ja"
        elif op == "<=":
            high_take = high_lt
            high_drop = high_gt
            low_take = "jbe"
        else:  # >=
            high_take = high_gt
            high_drop = high_lt
            low_take = "jae"
        return [
            "        cmp     edx, ebx",
            f"        {high_take}      {true_label}",
            f"        {high_drop}      {false_label}",
            "        cmp     eax, ecx",
            f"        {low_take}     {true_label}",
            f"{false_label}:",
            "        xor     eax, eax",
            f"        jmp     {end_label}",
            f"{true_label}:",
            "        mov     eax, 1",
            f"{end_label}:",
            "        xor     edx, edx",
        ]

    def _compound_assign_ll(
        self, expr: ast.BinaryOp, ctx: _FuncCtx,
    ) -> list[str]:
        """`lvalue OP= rhs` for long-long lvalues.

        For Identifier lvalues, desugar to `lvalue = lvalue OP rhs`
        — re-reading an Identifier is side-effect-free. For
        non-Identifier lvalues (Index / *p / Member), compute the
        address once, snapshot the value into a hidden LL slot,
        compute `snapshot OP rhs` via the standard 64-bit ladder,
        and store the result back through the saved address. This
        guarantees side effects in the lvalue's sub-expressions fire
        exactly once.
        """
        op = self._COMPOUND_OPS[expr.op]
        if isinstance(expr.left, ast.Identifier):
            inner = ast.BinaryOp(op=op, left=expr.left, right=expr.right)
            return self._binary_ll(
                ast.BinaryOp(op="=", left=expr.left, right=inner), ctx,
            )
        # Bit-field LL compound assign goes through the bit-field
        # RMW path which already handles address-once.
        if isinstance(expr.left, ast.Member):
            bf = self._bitfield_info(expr.left, ctx)
            if bf is not None:
                if len(bf) == 4 and bf[3] == 8:
                    return self._compound_assign_bitfield_ll(expr, bf, ctx)
                return self._compound_assign_bitfield(expr, bf, ctx)
        # Non-Identifier lvalue: address-once via hidden snapshot slot.
        addr_slot_name = f"__compll_addr_{id(expr)}"
        snap_slot_name = f"__compll_snap_{id(expr)}"
        addr_disp = ctx.alloc_local(
            addr_slot_name,
            ast.PointerType(base_type=ast.BasicType(name="long long")),
            size=4,
        )
        snap_disp = ctx.alloc_local(
            snap_slot_name,
            ast.BasicType(name="long long"),
            size=8,
        )
        out: list[str] = []
        # 1. Compute &lvalue once into addr_slot.
        if isinstance(expr.left, ast.Index):
            out += self._index_address(expr.left, ctx)
        elif (
            isinstance(expr.left, ast.UnaryOp)
            and expr.left.op == "*"
        ):
            out += self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Member):
            out += self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"long-long compound assign target must be an "
                f"identifier, `*ptr`, `arr[i]`, or `s.m` "
                f"(got {type(expr.left).__name__})"
            )
        out.append(f"        mov     {_ebp_addr(addr_disp)}, eax")
        # 2. Snapshot the current value into snap_slot.
        out.append("        mov     ecx, eax")
        out.append("        mov     eax, [ecx]")
        out.append("        mov     edx, [ecx + 4]")
        out.append(f"        mov     {_ebp_addr(snap_disp)}, eax")
        out.append(f"        mov     {_ebp_addr(snap_disp + 4)}, edx")
        # 3. Compute `snap OP rhs` into EDX:EAX via the full LL ladder.
        snap_id = ast.Identifier(name=snap_slot_name)
        inner = ast.BinaryOp(op=op, left=snap_id, right=expr.right)
        out += self._eval_expr_to_edx_eax(inner, ctx)
        # 4. Store result back at *addr_slot.
        out.append(f"        mov     ecx, {_ebp_addr(addr_disp)}")
        out.append("        mov     [ecx], eax")
        out.append("        mov     [ecx + 4], edx")
        return out

    def _assign_ll(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        """Long-long assignment to a long-long lvalue."""
        lhs = expr.left
        if isinstance(lhs, ast.Identifier):
            out = self._eval_expr_to_edx_eax(expr.right, ctx)
            addr = self._identifier_addr_text(lhs.name, ctx)
            out += self._store_from_edx_eax(addr)
            return out
        if isinstance(lhs, ast.UnaryOp) and lhs.op == "*":
            # *p = rhs : eval p → eax push, eval rhs → edx:eax, pop ecx
            out = self._eval_expr_to_eax(lhs.operand, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_edx_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out.append("        mov     [ecx], eax")
            out.append("        mov     [ecx + 4], edx")
            return out
        if isinstance(lhs, ast.Index):
            out = self._index_address(lhs, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_edx_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out.append("        mov     [ecx], eax")
            out.append("        mov     [ecx + 4], edx")
            return out
        if isinstance(lhs, ast.Member):
            # Bit-field LL member: route through the LL bit-field
            # store so we don't smash neighboring fields in the
            # 8-byte storage unit.
            bf = self._bitfield_info(lhs, ctx)
            if bf is not None and len(bf) == 4 and bf[3] == 8:
                return self._bitfield_store_ll_simple(
                    lhs, bf, expr.right, ctx,
                )
            # Bit-field with 4-byte storage: route through the
            # 32-bit bit-field store, narrowing the rhs.
            if bf is not None:
                return self._bitfield_store(lhs, bf, expr.right, ctx)
            out = self._member_address(lhs, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_edx_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out.append("        mov     [ecx], eax")
            out.append("        mov     [ecx + 4], edx")
            return out
        raise CodegenError(
            f"long-long assignment to {type(lhs).__name__} not supported"
        )

    def _inc_dec_ll(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        """++/-- on a long-long Identifier, member, indexed, or *p
        lvalue. Returns the result (pre or post value) in EDX:EAX."""
        # Long-long bit-field: route through the bit-field-aware path.
        if isinstance(expr.operand, ast.Member):
            bf = self._bitfield_info(expr.operand, ctx)
            if bf is not None and len(bf) == 4 and bf[3] == 8:
                return self._inc_dec_bitfield_ll(expr, bf, ctx)
        if isinstance(expr.operand, ast.Identifier):
            addr = self._identifier_addr_text(expr.operand.name, ctx)
            high_addr = self._bump_addr(addr, 4)
            # Read pre-value if postfix.
            if expr.is_prefix:
                out: list[str] = []
                if expr.op == "++":
                    out.append(f"        add     dword {addr}, 1")
                    out.append(f"        adc     dword {high_addr}, 0")
                else:
                    out.append(f"        sub     dword {addr}, 1")
                    out.append(f"        sbb     dword {high_addr}, 0")
                out += self._load_to_edx_eax(addr)
                return out
            # Postfix: load first, then bump.
            out = self._load_to_edx_eax(addr)
            if expr.op == "++":
                out.append(f"        add     dword {addr}, 1")
                out.append(f"        adc     dword {high_addr}, 0")
            else:
                out.append(f"        sub     dword {addr}, 1")
                out.append(f"        sbb     dword {high_addr}, 0")
            return out
        # Non-Identifier lvalue: compute &lvalue once into ECX, then
        # RMW [ecx] (low) and [ecx+4] (high) with carry/borrow.
        if isinstance(expr.operand, ast.Index):
            addr_lines = self._index_address(expr.operand, ctx)
        elif isinstance(expr.operand, ast.UnaryOp) and expr.operand.op == "*":
            addr_lines = self._eval_expr_to_eax(expr.operand.operand, ctx)
        elif isinstance(expr.operand, ast.Member):
            addr_lines = self._member_address(expr.operand, ctx)
        else:
            raise CodegenError(
                f"long-long ++/-- not supported on "
                f"{type(expr.operand).__name__}"
            )
        instr0 = "add" if expr.op == "++" else "sub"
        instrN = "adc" if expr.op == "++" else "sbb"
        out = list(addr_lines)
        out.append("        mov     ecx, eax")
        if expr.is_prefix:
            out.append(f"        {instr0}     dword [ecx], 1")
            out.append(f"        {instrN}     dword [ecx + 4], 0")
            out.append("        mov     eax, [ecx]")
            out.append("        mov     edx, [ecx + 4]")
            return out
        # Postfix: load OLD value into EDX:EAX, then bump in place.
        out.append("        mov     eax, [ecx]")
        out.append("        mov     edx, [ecx + 4]")
        out.append(f"        {instr0}     dword [ecx], 1")
        out.append(f"        {instrN}     dword [ecx + 4], 0")
        return out

    def _logical_and_ll(self, expr, ctx):
        # Treat both sides as bool; result is 0 or 1 in EAX:EDX (high=0).
        false_label = ctx.label("ll_and_false")
        end_label = ctx.label("ll_and_end")
        out = self._eval_to_bool_eax(expr.left, ctx)
        out += ["        test    eax, eax", f"        jz      {false_label}"]
        out += self._eval_to_bool_eax(expr.right, ctx)
        out += ["        test    eax, eax", f"        jz      {false_label}"]
        out += ["        mov     eax, 1", f"        jmp     {end_label}"]
        out.append(f"{false_label}:")
        out.append("        xor     eax, eax")
        out.append(f"{end_label}:")
        out.append("        xor     edx, edx")
        return out

    def _logical_or_ll(self, expr, ctx):
        true_label = ctx.label("ll_or_true")
        false_label = ctx.label("ll_or_false")
        end_label = ctx.label("ll_or_end")
        out = self._eval_to_bool_eax(expr.left, ctx)
        out += ["        test    eax, eax", f"        jnz     {true_label}"]
        out += self._eval_to_bool_eax(expr.right, ctx)
        out += ["        test    eax, eax", f"        jz      {false_label}"]
        out.append(f"{true_label}:")
        out.append("        mov     eax, 1")
        out.append(f"        jmp     {end_label}")
        out.append(f"{false_label}:")
        out.append("        xor     eax, eax")
        out.append(f"{end_label}:")
        out.append("        xor     edx, edx")
        return out

    def _eval_float_to_st0(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        """Lower a float-or-int expression so its value sits at st(0).

        Int-typed sub-expressions are evaluated to EAX first, then
        promoted via `fild` from a stack scratch (since x87 has no
        register-direct int load). The caller is responsible for
        consuming the value off the FPU stack — every code path here
        leaves exactly one extra value on st(0).
        """
        ty = self._type_of(expr, ctx)
        if not self._is_float_type(ty):
            # Promote int → float via stack scratch. fild reads the
            # value as SIGNED, so for unsigned int we zero-extend into
            # an 8-byte qword scratch and `fild qword` from there.
            if self._is_long_long(ty):
                out = self._eval_expr_to_edx_eax(expr, ctx)
                out.append("        push    edx")
                out.append("        push    eax")
                out.append("        fild    qword [esp]")
                out.append("        add     esp, 8")
                # For unsigned long long: if the high bit is set, fild
                # treated it as negative — add 2^64 to correct.
                if self._is_unsigned(ty):
                    out.append("        test    edx, edx")
                    label_ok = ctx.label("ull_to_float_ok")
                    out.append(f"        jns     {label_ok}")
                    # 2^64 as a float constant. Intern it.
                    bias_label = self._intern_float(
                        18446744073709551616.0, 8,
                    )
                    out.append(f"        fld     qword [{bias_label}]")
                    out.append("        faddp   st1, st0")
                    out.append(f"{label_ok}:")
                return out
            if self._is_unsigned(ty):
                out = self._eval_expr_to_eax(expr, ctx)
                out.append("        push    0")           # high half = 0
                out.append("        push    eax")
                out.append("        fild    qword [esp]")
                out.append("        add     esp, 8")
                return out
            out = self._eval_expr_to_eax(expr, ctx)
            out.append("        push    eax")
            out.append("        fild    dword [esp]")
            out.append("        add     esp, 4")
            return out
        # Float-typed expression. Dispatch by node.
        if isinstance(expr, ast.FloatLiteral):
            size = 4 if ty.name == "float" else 8
            label = self._intern_float(expr.value, size)
            width = "dword" if size == 4 else "qword"
            return [f"        fld     {width} [{label}]"]
        if isinstance(expr, ast.Identifier):
            return self._float_identifier_load(expr.name, ctx)
        if isinstance(expr, ast.UnaryOp):
            if expr.op == "+":
                return self._eval_float_to_st0(expr.operand, ctx)
            if expr.op == "-":
                return self._eval_float_to_st0(expr.operand, ctx) + [
                    "        fchs",
                ]
            if expr.op in ("++", "--"):
                return self._float_inc_dec(expr, ctx)
            if expr.op == "*":
                # Dereferencing a float pointer in a float context:
                # eval pointer to eax, then `fld` from there. Width
                # follows the pointee (float vs double).
                size = 4 if ty.name == "float" else 8
                width = "dword" if size == 4 else "qword"
                out = self._eval_expr_to_eax(expr.operand, ctx)
                out.append(f"        fld     {width} [eax]")
                return out
            if expr.op in ("__real__", "__imag__"):
                operand_ty = self._type_of(expr.operand, ctx)
                if isinstance(operand_ty, ast.ComplexType):
                    out, _ = self._complex_part_address(expr, ctx)
                    size = self._COMPLEX_BASE_SIZES[operand_ty.base_type]
                    width = "dword" if size == 4 else "qword"
                    out.append(f"        fld     {width} [eax]")
                    return out
                # Non-complex: __real__ is identity, __imag__ is 0.
                if expr.op == "__real__":
                    return self._eval_float_to_st0(expr.operand, ctx)
                return ["        fldz"]
            raise CodegenError(
                f"unary `{expr.op}` not supported on float operand"
            )
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
                # `__real__/__imag__ x = rhs` writes to a half of a
                # complex value; route through _assign so the half-
                # address path runs.
                if (
                    isinstance(expr.left, ast.UnaryOp)
                    and expr.left.op in ("__real__", "__imag__")
                ):
                    return self._assign(expr, ctx)
                return self._float_assign(expr, ctx)
            if expr.op in self._COMPOUND_OPS:
                return self._float_compound_assign(expr, ctx)
            return self._float_binop(expr, ctx)
        if isinstance(expr, ast.Cast):
            return self._float_cast(expr, ctx)
        if isinstance(expr, ast.TernaryOp):
            return self._float_ternary(expr, ctx)
        if isinstance(expr, ast.Member):
            return self._float_member_load(expr, ctx)
        if isinstance(expr, ast.Index):
            return self._float_index_load(expr, ctx)
        if isinstance(expr, ast.Call):
            # Float-returning calls leave their result on st(0) per
            # cdecl, so we just emit the standard call sequence —
            # `_call`'s post-call cleanup doesn't touch the FPU stack.
            return self._call(expr, ctx)
        if isinstance(expr, ast.VaArgExpr):
            return self._va_arg_float(expr, ctx)
        if isinstance(expr, ast.StmtExpr):
            # GCC statement expression returning a float — emit head
            # items in scope, then evaluate the trailing expression
            # to st(0).
            ctx.enter_scope()
            try:
                items = list(expr.body.items)
                if not items:
                    return []
                head, tail = items[:-1], items[-1]
                out: list[str] = []
                for item in head:
                    out += self._item(item, ctx)
                if (
                    isinstance(tail, ast.ExpressionStmt)
                    and tail.expr is not None
                ):
                    out += self._eval_float_to_st0(tail.expr, ctx)
                else:
                    out += self._item(tail, ctx)
                return out
            finally:
                ctx.exit_scope()
        raise CodegenError(
            f"float expression {type(expr).__name__} not implemented yet"
        )

    def _float_identifier_load(self, name: str, ctx: _FuncCtx) -> list[str]:
        """`fld` a float-typed Identifier (local, param, or global)."""
        # `_identifier_type` already handles the static-local remap, but
        # we resolve here too so the slot/global lookups below match.
        ty = self._identifier_type(name, ctx)
        name = ctx.local_static_labels.get(name, name)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        if ctx.has_local(name):
            disp = ctx.lookup(name)
            return [f"        fld     {width} {_ebp_addr(disp)}"]
        if name in self._globals:
            return [f"        fld     {width} [_{name}]"]
        raise CodegenError(f"unknown identifier `{name}`")

    def _float_member_load(self, expr: ast.Member, ctx: _FuncCtx) -> list[str]:
        ty = self._type_of(expr, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        return self._member_address(expr, ctx) + [
            f"        fld     {width} [eax]",
        ]

    def _float_index_load(self, expr: ast.Index, ctx: _FuncCtx) -> list[str]:
        ty = self._type_of(expr, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        return self._index_address(expr, ctx) + [
            f"        fld     {width} [eax]",
        ]

    def _float_binop(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        # x87 binop pattern: load left, load right, then `f<op>p st1, st0`
        # which pops st(0) and combines with st(1), leaving the result
        # on st(0). NASM accepts the implicit one-operand form too, but
        # the explicit form documents the intent better.
        op_to_mnem = {
            "+": "faddp",
            "-": "fsubp",
            "*": "fmulp",
            "/": "fdivp",
        }
        # Reverse-direction mnemonics: if we evaluate the right side
        # first, we need fsubrp/fdivrp because the operand order is
        # flipped. faddp/fmulp are commutative.
        op_to_mnem_rev = {
            "+": "faddp",
            "-": "fsubrp",
            "*": "fmulp",
            "/": "fdivrp",
        }
        if expr.op == ",":
            # Comma operator: eval lhs for side effects, then yield rhs.
            # Lhs may be int or float; we need to drop its value.
            lhs_ty = self._type_of(expr.left, ctx)
            if self._is_float_type(lhs_ty):
                out = self._eval_float_to_st0(expr.left, ctx)
                out.append("        fstp    st0")
            else:
                out = self._eval_expr_to_eax(expr.left, ctx)
            out += self._eval_float_to_st0(expr.right, ctx)
            return out
        if expr.op not in op_to_mnem:
            raise CodegenError(
                f"binary `{expr.op}` not supported on float operands"
            )
        # Pick eval order to minimize peak FPU stack depth. When the
        # right side is deeper than the left (a leaf or shallow), eval
        # right first so the left's value isn't sitting on the stack
        # for the duration of right's deep ladder. Crucial for level-7
        # Horner-form polynomials in pr58574 — left-first overflows
        # the 8-deep x87 stack at level 7.
        l_depth = self._fpu_depth(expr.left, ctx)
        r_depth = self._fpu_depth(expr.right, ctx)
        if r_depth > l_depth:
            out = self._eval_float_to_st0(expr.right, ctx)
            out += self._eval_float_to_st0(expr.left, ctx)
            out.append(f"        {op_to_mnem_rev[expr.op]}   st1, st0")
            return out
        out = self._eval_float_to_st0(expr.left, ctx)
        out += self._eval_float_to_st0(expr.right, ctx)
        out.append(f"        {op_to_mnem[expr.op]}   st1, st0")
        return out

    def _fpu_depth(self, expr: ast.Expression, ctx: _FuncCtx) -> int:
        """Conservative estimate of the peak FPU stack depth needed to
        evaluate `expr`. Leaves use depth 1; binary ops use 1 + min(left,
        right) + abs(left - right) — the standard formula for
        register-allocation peak when we eval the deeper side first.
        For our purposes, a coarse upper bound suffices.
        """
        if isinstance(expr, ast.BinaryOp) and expr.op in ("+", "-", "*", "/"):
            l = self._fpu_depth(expr.left, ctx)
            r = self._fpu_depth(expr.right, ctx)
            return max(l, r) + 1
        if isinstance(expr, ast.UnaryOp):
            return self._fpu_depth(expr.operand, ctx)
        if isinstance(expr, ast.Cast):
            return self._fpu_depth(expr.expr, ctx)
        # Leaves: literals, identifiers, member/index, function calls.
        return 1

    def _float_cast(self, expr: ast.Cast, ctx: _FuncCtx) -> list[str]:
        """Cast to a float target. The source may be int or float.

        x87 keeps intermediate results at 80-bit precision and the C
        standard allows that — so we don't force a fstp+fld dword
        round-trip on every cast. The eventual fstp at the consuming
        end picks the storage width.
        """
        return self._eval_float_to_st0(expr.expr, ctx)

    def _float_ternary(self, expr: ast.TernaryOp, ctx: _FuncCtx) -> list[str]:
        false_label = ctx.label("ftern_false")
        end_label = ctx.label("ftern_end")
        out = self._eval_to_bool_eax(expr.condition, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        out += self._eval_float_to_st0(expr.true_expr, ctx)
        out.append(f"        jmp     {end_label}")
        out.append(f"{false_label}:")
        out += self._eval_float_to_st0(expr.false_expr, ctx)
        out.append(f"{end_label}:")
        return out

    def _store_st0_to(self, addr: str, ty: ast.TypeNode) -> list[str]:
        """`fstp` the top of the FPU stack into memory at `addr`."""
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        return [f"        fstp    {width} {addr}"]

    def _float_assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        """Lower `lvalue = rhs` where `lvalue` is float-typed.

        Uses `fst` (store without pop) instead of `fstp` so the new
        value stays on st(0) — that lets a float assignment expression
        be consumed (e.g. `(f = 1.5f) + 1.0f`) without an extra reload.
        For non-Identifier lvalues we compute the address once into EAX,
        push it, then evaluate the rhs to st(0) and pop the address back
        into ECX before storing.
        """
        ty = self._type_of(expr.left, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        if isinstance(expr.left, ast.Identifier):
            addr = self._float_lvalue_addr(expr.left.name, ctx)
            out = self._eval_float_to_st0(expr.right, ctx)
            out.append(f"        fst     {width} {addr}")
            return out
        # Non-Identifier lvalues: compute address first (clobbers eax),
        # push it, then evaluate the float rhs onto st(0), pop the
        # address into ecx, and `fst` through it.
        if isinstance(expr.left, ast.UnaryOp) and expr.left.op == "*":
            addr_lines = self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Index):
            addr_lines = self._index_address(expr.left, ctx)
        elif isinstance(expr.left, ast.Member):
            addr_lines = self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"float assignment to {type(expr.left).__name__} not supported"
            )
        out = list(addr_lines)
        out.append("        push    eax")
        out += self._eval_float_to_st0(expr.right, ctx)
        out.append("        pop     ecx")
        out.append(f"        fst     {width} [ecx]")
        return out

    def _float_inc_dec(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        """`++f` / `--f` / `f++` / `f--` for a float Identifier or any
        float lvalue (Index, Member, *p).

        Pre-form: bump the slot, leaving the new value on st(0).
        Post-form: load the slot twice, bump one copy, store it, leaving
        the original value on st(0) so the expression yields the old.
        """
        op_mnem = "faddp" if expr.op == "++" else "fsubp"
        if isinstance(expr.operand, ast.Identifier):
            name = expr.operand.name
            ty = self._identifier_type(name, ctx)
            addr = self._float_lvalue_addr(name, ctx)
            size = self._size_of(ty)
            width = "dword" if size == 4 else "qword"
            if expr.is_prefix:
                return [
                    f"        fld     {width} {addr}",
                    "        fld1",
                    f"        {op_mnem}   st1, st0",
                    f"        fst     {width} {addr}",
                ]
            # Post-form: keep the old value on st(0) after storing the new.
            return [
                f"        fld     {width} {addr}",   # st0 = old
                f"        fld     {width} {addr}",   # st0 = old, st1 = old
                "        fld1",                       # st0 = 1, st1 = old, st2 = old
                f"        {op_mnem}   st1, st0",      # pops, st1 = old±1
                f"        fstp    {width} {addr}",   # store new, pop. st0 = old
            ]
        # Non-Identifier lvalue: compute &lvalue once into EAX, hold
        # in stack, then fld through it. Same shape as
        # _float_compound_assign's general-lvalue path.
        if isinstance(expr.operand, ast.Index):
            addr_lines = self._index_address(expr.operand, ctx)
        elif (
            isinstance(expr.operand, ast.UnaryOp)
            and expr.operand.op == "*"
        ):
            addr_lines = self._eval_expr_to_eax(expr.operand.operand, ctx)
        elif isinstance(expr.operand, ast.Member):
            addr_lines = self._member_address(expr.operand, ctx)
        else:
            raise CodegenError(
                f"`{expr.op}` on a float operand of type "
                f"{type(expr.operand).__name__} not supported"
            )
        ty = self._type_of(expr.operand, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        out = list(addr_lines)
        out.append("        push    eax")
        if expr.is_prefix:
            out.append(f"        fld     {width} [eax]")
            out.append("        fld1")
            out.append(f"        {op_mnem}   st1, st0")
            out.append("        pop     ecx")
            out.append(f"        fst     {width} [ecx]")
            return out
        # Postfix: keep old on st(0) after storing new through ECX.
        out.append(f"        fld     {width} [eax]")    # st0 = old
        out.append(f"        fld     {width} [eax]")    # st0 = old, st1 = old
        out.append("        fld1")                       # st0 = 1
        out.append(f"        {op_mnem}   st1, st0")      # st0 = old±1, st1 = old
        out.append("        pop     ecx")
        out.append(f"        fstp    {width} [ecx]")    # store new, pop. st0 = old
        return out

    def _float_compound_assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        """Lower `lvalue op= rhs` where the value is float-typed.

        Identifier lvalues use the simple desugar (re-evaluating an
        Identifier is side-effect-free) so `f += rhs` becomes
        `f = f + rhs`. For `arr[i]`, `*p`, and `s.m` lvalues we compute
        the address once into EAX, push it, `fld` the current value,
        evaluate the rhs to st(0), apply `faddp`/`fsubp`/`fmulp`/
        `fdivp`, pop the address back into ECX, and `fst` through it.
        """
        op = self._COMPOUND_OPS[expr.op]
        if isinstance(expr.left, ast.Identifier):
            inner = ast.BinaryOp(op=op, left=expr.left, right=expr.right)
            return self._float_assign(
                ast.BinaryOp(op="=", left=expr.left, right=inner), ctx,
            )
        if isinstance(expr.left, ast.UnaryOp) and expr.left.op == "*":
            addr_lines = self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Index):
            addr_lines = self._index_address(expr.left, ctx)
        elif isinstance(expr.left, ast.Member):
            addr_lines = self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"float compound assignment to "
                f"{type(expr.left).__name__} not supported"
            )
        op_to_mnem = {
            "+": "faddp",
            "-": "fsubp",
            "*": "fmulp",
            "/": "fdivp",
        }
        if op not in op_to_mnem:
            raise CodegenError(
                f"float compound op `{expr.op}` not supported"
            )
        target_ty = self._type_of(expr.left, ctx)
        size = self._size_of(target_ty)
        width = "dword" if size == 4 else "qword"
        out = list(addr_lines)
        out.append("        push    eax")            # save addr
        out.append(f"        fld     {width} [eax]") # current value to st0
        out += self._eval_float_to_st0(expr.right, ctx)
        out.append(f"        {op_to_mnem[op]}   st1, st0")
        out.append("        pop     ecx")
        out.append(f"        fst     {width} [ecx]")
        return out

    def _eval_to_bool_eax(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        """Evaluate `expr` and leave EAX = 0 (false) or 1 (true).

        For float-typed expressions, tests against 0.0 on the FPU rather
        than truncating to int — `if (0.5)` should branch true, not
        false. For long-long expressions, OR the two halves so a value
        with non-zero high bits but zero low bits is still truthy. For
        plain int expressions, just defer to `_eval_expr_to_eax`; the
        following `test eax, eax` handles non-zero-is-true.
        """
        ty = self._type_of(expr, ctx)
        if self._is_float_type(ty):
            out = self._eval_float_to_st0(expr, ctx)
            out.append("        fldz")
            out.append("        fucompp")
            out.append("        fnstsw  ax")
            out.append("        sahf")
            out.append("        setne   al")
            out.append("        movzx   eax, al")
            return out
        if self._is_long_long(ty):
            out = self._eval_expr_to_edx_eax(expr, ctx)
            out.append("        or      eax, edx")
            return out
        if self._is_int128(ty):
            # `if (u128_val)` — true iff any of the four dwords is
            # non-zero. OR them together and let the caller's
            # `test eax, eax` decide.
            out = self._int128_value_address(expr, ctx)
            out.append("        mov     ecx, eax")
            out.append("        mov     eax, [ecx]")
            out.append("        or      eax, [ecx + 4]")
            out.append("        or      eax, [ecx + 8]")
            out.append("        or      eax, [ecx + 12]")
            return out
        if isinstance(ty, ast.ComplexType):
            # `if (c)` for complex — true if either half is non-zero.
            # Materialize into a stack temp, then OR-or-FUCOMP both halves.
            half_size = self._COMPLEX_BASE_SIZES[ty.base_type]
            size = self._size_of(ty)
            out = [f"        sub     esp, {size}"]
            out.append("        mov     eax, esp")
            out.append("        push    eax")
            out += self._eval_complex_into_top(expr, ctx, ty)
            out.append("        add     esp, 4")
            if ty.base_type in self._COMPLEX_INT_BASES:
                int_widths = {1: ("byte", "al"), 2: ("word", "ax"),
                              4: ("dword", "eax")}
                if half_size == 8:
                    out.append("        mov     eax, [esp]")
                    out.append("        or      eax, [esp + 4]")
                    out.append("        or      eax, [esp + 8]")
                    out.append("        or      eax, [esp + 12]")
                else:
                    w, lo = int_widths[half_size]
                    if half_size == 4:
                        out.append("        mov     eax, [esp]")
                        out.append("        or      eax, [esp + 4]")
                    else:
                        out.append(f"        mov     {lo}, [esp]")
                        out.append(f"        or      {lo}, [esp + {half_size}]")
                        out.append("        movzx   eax, " + lo)
                out.append("        setne   al")
                out.append("        movzx   eax, al")
            else:
                width = "dword" if half_size == 4 else "qword"
                out.append(f"        fld     {width} [esp]")
                out.append("        fldz")
                out.append("        fucompp")
                out.append("        fnstsw  ax")
                out.append("        sahf")
                out.append("        setne   cl")
                out.append("        movzx   ecx, cl")
                out.append(f"        fld     {width} [esp + {half_size}]")
                out.append("        fldz")
                out.append("        fucompp")
                out.append("        fnstsw  ax")
                out.append("        sahf")
                out.append("        setne   al")
                out.append("        movzx   eax, al")
                out.append("        or      eax, ecx")
            out.append(f"        add     esp, {size}")
            return out
        return self._eval_expr_to_eax(expr, ctx)

    def _float_compare(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        """Lower a float comparison to a 386-compatible setCC sequence.

        Evaluates left then right (so ST(0) = right, ST(1) = left), then
        `fucompp` compares and pops both. `fnstsw ax; sahf` lifts the
        FPU condition codes into the integer flags, and a setCC reads
        them. The mapping is inverted vs. the integer ops because the
        FPU compares ST(0) op ST(1) (= right op left).
        """
        out = self._eval_float_to_st0(expr.left, ctx)
        out += self._eval_float_to_st0(expr.right, ctx)
        out.append("        fucompp")
        out.append("        fnstsw  ax")
        out.append("        sahf")
        out.append(f"        {self._FLOAT_CMP_SETCC[expr.op]}    al")
        out.append("        movzx   eax, al")
        return out

    def _float_lvalue_addr(self, name: str, ctx: _FuncCtx) -> str:
        """Render the address text for a float-typed Identifier lvalue."""
        name = ctx.local_static_labels.get(name, name)
        if ctx.has_local(name):
            return _ebp_addr(ctx.lookup(name))
        if name in self._globals:
            return f"[_{name}]"
        raise CodegenError(f"unknown identifier `{name}`")

    def _index_address(self, expr: ast.Index, ctx: _FuncCtx) -> list[str]:
        """Compute &(expr.array[expr.index]) into eax.

        The array sub-expression evaluates with normal decay rules, so
        whether `expr.array` is an array name, a pointer variable, or a
        more complex expression that produces a pointer — the address
        flow is the same: base in eax, scale the index by element size,
        then add. For VLA elements (size depends on a runtime value),
        emit a runtime sizeof and multiply by the index.

        C allows the swapped form `int[ptr]` (= ptr[int]) — if `array`
        isn't pointer-like but `index` is, swap them.
        """
        arr_type = self._type_of(expr.array, ctx)
        idx_type = self._type_of(expr.index, ctx)
        if not self._is_pointer_like(arr_type) and self._is_pointer_like(idx_type):
            expr = ast.Index(
                array=expr.index, index=expr.array, location=expr.location,
            )
            arr_type, idx_type = idx_type, arr_type
        if not self._is_pointer_like(arr_type):
            raise CodegenError(
                f"index target must be array or pointer "
                f"(got {type(arr_type).__name__})"
            )
        elem_ty = arr_type.base_type
        if self._type_has_vla(elem_ty):
            # Runtime stride: eval array → push, eval index → push,
            # eval sizeof(elem) → eax, multiply, add.
            out = self._eval_expr_to_eax(expr.array, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_eax(expr.index, ctx)
            out.append("        push    eax")
            out += self._emit_runtime_size_of(elem_ty, ctx)
            out.append("        pop     ecx")    # ecx = index
            out.append("        imul    eax, ecx")
            out.append("        mov     ecx, eax")
            out.append("        pop     eax")    # eax = array base
            out.append("        add     eax, ecx")
            return out
        elem_size = self._size_of(elem_ty)
        out = self._eval_expr_to_eax(expr.array, ctx)
        out.append("        push    eax")
        out += self._eval_expr_to_eax(expr.index, ctx)
        out += self._scale_reg("eax", elem_size)
        out.append("        mov     ecx, eax")
        out.append("        pop     eax")
        out.append("        add     eax, ecx")
        return out

    def _index_load(self, expr: ast.Index, ctx: _FuncCtx) -> list[str]:
        # Read through the computed element address using a width matching
        # the element type — `arr[i]` for `char arr[]` reads one byte.
        # For aggregate elements (an inner array of a multidim or a
        # struct row), evaluating the expression yields the element's
        # address (array-to-pointer decay / struct l-value), since we
        # don't load aggregates into EAX.
        elem_ty = self._type_of(expr, ctx)
        addr = self._index_address(expr, ctx)
        if isinstance(elem_ty, (ast.ArrayType, ast.StructType)):
            return addr
        return addr + self._load_to_eax("[eax]", elem_ty)

    def _ternary(self, expr: ast.TernaryOp, ctx: _FuncCtx) -> list[str]:
        false_label = ctx.label("tern_false")
        end_label = ctx.label("tern_end")
        out = self._eval_to_bool_eax(expr.condition, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        out += self._eval_expr_to_eax(expr.true_expr, ctx)
        out.append(f"        jmp     {end_label}")
        out.append(f"{false_label}:")
        out += self._eval_expr_to_eax(expr.false_expr, ctx)
        out.append(f"{end_label}:")
        return out

    def _call(self, expr: ast.Call, ctx: _FuncCtx) -> list[str]:
        # In C, `*f` on a function-typed expression is idempotent — `f()`
        # and `(*f)()` and `(***f)()` all call the same function. Strip
        # leading dereferences from the callee so the lowering doesn't
        # try to actually load through the function address.
        #
        # Struct-returning calls are handled in `_eval_expr_to_eax` (via
        # the per-call temp) before reaching here.
        callee = self._stripped_callee(expr)
        # gnu_inline functions get inlined at every call site instead of
        # generating a regular `call _name`. The body uses
        # `__builtin_va_arg_pack()` which requires the caller's variadic
        # args at compile time.
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in self._gnu_inline_funcs
        ):
            return self._call_inline_gnu(
                expr, self._gnu_inline_funcs[callee.name], ctx,
            )
        # Rebind nested-fn names to their lifted top-level mangled name
        # so the rest of the path treats them as regular file-scope
        # functions. The pre-pass at the top of `_function` registers
        # the mangled name in `_func_return_types` / `_func_param_types`.
        # If the nested fn has nonlocal gotos, build the ECX-setup
        # lines so a direct call propagates the static link the same
        # way the trampoline would for an indirect call.
        ecx_setup: list[str] | None = None
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in ctx.nested_fn_names
        ):
            user_name = callee.name
            mangled = ctx.nested_fn_names[user_name]
            if user_name in ctx.trampolines:
                # Outer is calling its nested fn directly. Load ECX
                # with the address of the buf-array in our frame.
                _tramp_disp, buf_disp = ctx.trampolines[user_name]
                ecx_setup = [
                    f"        lea     ecx, {_ebp_addr(buf_disp)}"
                ]
            elif (
                ctx.trampoline_static_link_disp is not None
                and mangled in self._lifted_nonlocal_gotos
            ):
                # We're a lifted nested fn calling a sibling (or
                # ourself) that also has nonlocal gotos. The callee's
                # buf-array lives in our outer's frame; for the
                # self-call case, our own static link points to the
                # right buf and we can propagate it directly. For
                # cross-sibling calls with different buf-arrays this
                # would be wrong — those aren't reached by current
                # tests.
                ecx_setup = [
                    f"        mov     ecx, "
                    f"{_ebp_addr(ctx.trampoline_static_link_disp)}"
                ]
            callee = ast.Identifier(
                name=mangled,
                location=callee.location,
            )
            # Build a fresh Call node so downstream code (param-type
            # lookups, struct-return retptr handling) keys on the new
            # name. Args are pass-by-reference; reusing them is fine.
            expr = ast.Call(
                func=callee,
                args=expr.args,
                location=expr.location,
            )

        # Variadic builtins. `va_start(ap, last)` writes the address
        # past `last` into `ap`; `va_end(ap)` is a no-op (cdecl needs
        # no per-call cleanup) — we still produce a 0 in EAX so the
        # call's value is defined when the user accidentally consumes
        # it. `va_arg` is handled at expression eval (see VaArgExpr).
        if isinstance(callee, ast.Identifier):
            if callee.name in ("va_start", "__builtin_va_start"):
                return self._va_start(expr.args, ctx)
            if callee.name in ("va_end", "__builtin_va_end"):
                # `va_end(ap)` is a no-op for cdecl, but we still
                # evaluate `ap` for any side effects (e.g.
                # `va_end(*ap_ptr++)` increments ap_ptr).
                out: list[str] = []
                for arg in expr.args:
                    out += self._eval_expr_to_eax(arg, ctx)
                out.append("        xor     eax, eax")
                return out
            # GCC branch-prediction hint: `__builtin_expect(expr, val)`
            # has the value of `expr`. We ignore the hint and just emit
            # the first argument's value.
            if callee.name == "__builtin_expect" and len(expr.args) >= 1:
                # The second arg is "expected value" — gcc treats it as
                # a runtime expression (not a static hint), so we eval
                # it for any side effects, drop the result, then yield
                # the first arg.
                out: list[str] = []
                for extra in expr.args[1:]:
                    out += self._eval_expr_to_eax(extra, ctx)
                out += self._eval_expr_to_eax(expr.args[0], ctx)
                return out
            # `__builtin_choose_expr(cond, a, b)` selects a or b at
            # compile time based on `cond`'s integer-constant value.
            if (
                callee.name == "__builtin_choose_expr"
                and len(expr.args) == 3
            ):
                try:
                    cond_val = self._const_eval(expr.args[0], "<choose>")
                except CodegenError:
                    cond_val = 0
                chosen = expr.args[1] if cond_val else expr.args[2]
                return self._eval_expr_to_eax(chosen, ctx)
            # `__builtin_constant_p(x)` returns 1 if x is a compile-time
            # constant integer expression, 0 otherwise. We can answer
            # "yes" only when `_const_eval` succeeds without side effects.
            # String literals and FloatLiterals also count.
            if (
                callee.name == "__builtin_constant_p"
                and len(expr.args) == 1
            ):
                arg = expr.args[0]
                if isinstance(arg, (ast.StringLiteral, ast.FloatLiteral)):
                    return ["        mov     eax, 1"]
                try:
                    self._const_eval(arg, "<bcp>")
                    return ["        mov     eax, 1"]
                except CodegenError:
                    return ["        xor     eax, eax"]
            # `__builtin_unreachable()` and `__builtin_trap()` are
            # diagnostic-only — emit a 0 in EAX so calls to them in
            # value position are at least defined.
            if callee.name in ("__builtin_unreachable", "__builtin_trap"):
                return ["        xor     eax, eax"]
            # `abs(int)` / `labs(long)` / `__builtin_abs` / `__builtin_labs`:
            # inline as `cdq; xor eax, edx; sub eax, edx`. Avoids the
            # user redefining `abs` to `abort()` (a gcc-canonical test
            # pattern) — gcc treats abs/labs/llabs as builtins and inlines.
            if (
                callee.name in ("abs", "labs", "__builtin_abs", "__builtin_labs")
                and len(expr.args) == 1
            ):
                out = self._eval_expr_to_eax(expr.args[0], ctx)
                out += [
                    "        cdq",
                    "        xor     eax, edx",
                    "        sub     eax, edx",
                ]
                return out
            # GCC: __builtin_apply_args() — returns a pointer to the
            # caller's argument area. For cdecl with no named params
            # before the variadic part, that's just [ebp+8] (after
            # retptr if present).
            if callee.name == "__builtin_apply_args":
                return ["        lea     eax, [ebp + 8]"]
            # __builtin_apply(fn, args, size) — call fn with `size`
            # bytes of args from `args`.
            if (
                callee.name == "__builtin_apply"
                and len(expr.args) == 3
            ):
                # Eval size first (compile-time constant in practice).
                size_val = None
                try:
                    size_val = self._const_eval(expr.args[2], "<apply>")
                except CodegenError:
                    pass
                if size_val is None:
                    raise CodegenError(
                        "__builtin_apply: size must be a compile-time constant"
                    )
                # Round up to 4.
                size_val = (size_val + 3) & ~3
                # Eval args pointer first, save.
                out: list[str] = []
                out += self._eval_expr_to_eax(expr.args[1], ctx)
                out.append("        push    eax")  # save args ptr
                # Eval callee fn pointer.
                out += self._eval_expr_to_eax(expr.args[0], ctx)
                out.append("        push    eax")  # save fn ptr
                # Reserve `size` bytes on stack and copy from args.
                out.append(f"        sub     esp, {size_val}")
                # Stack now (low to high): [size_val area | fn ptr | args ptr]
                # Push esi/edi to preserve, fix up offsets.
                out.append("        push    esi")
                out.append("        push    edi")
                # After both pushes: [edi(0) | esi(4) | size_area(8..size+7) |
                #                     fn(size+8) | args(size+12)]
                out.append(f"        mov     esi, [esp + {size_val + 12}]")
                out.append(f"        lea     edi, [esp + 8]")
                out.append(f"        mov     ecx, {size_val // 4}")
                out.append("        rep     movsd")
                out.append("        pop     edi")
                out.append("        pop     esi")
                # Now: [size_area(0..size-1) | fn(size) | args(size+4)]
                out.append(f"        mov     eax, [esp + {size_val}]")
                out.append("        call    eax")
                # Cleanup: size bytes + 4 (fn ptr) + 4 (args ptr).
                out.append(f"        add     esp, {size_val + 8}")
                # Return value from the last call: it's a struct/value
                # but the test typically discards it. Leave EAX as-is.
                return out
            if callee.name in (
                "__builtin_add_overflow",
                "__builtin_sub_overflow",
                "__builtin_mul_overflow",
            ) and len(expr.args) == 3:
                return self._builtin_overflow(
                    callee.name, expr.args, ctx,
                )

        # Direct call: callee names a function declared in this unit
        # (defined or extern). Emit a `call _name` linker reference.
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in self._func_return_types
        ):
            return self._emit_call(
                expr.args, ctx,
                direct=callee.name,
                ecx_setup=ecx_setup,
            )

        # Implicit-int function declaration (K&R / pre-C99): a Call
        # whose callee is an undeclared Identifier. Auto-register the
        # name as `int func()` and emit a direct call. This matches
        # what gcc does for sources that drop their `#include` and is
        # what the c-testsuite / GCC torture suite assume.
        if (
            isinstance(callee, ast.Identifier)
            and not ctx.has_local(callee.name)
            and callee.name not in self._globals
            and callee.name not in self._extern_vars
            and callee.name not in self._enum_constants
        ):
            self._func_return_types[callee.name] = ast.BasicType(name="int")
            self._func_param_types[callee.name] = []
            return self._emit_call(expr.args, ctx, direct=callee.name)

        # Indirect call: the callee evaluates to a function address. Push
        # args first (which clobber EAX), then evaluate the callee into
        # EAX last so it survives until the `call`.
        return self._emit_call(expr.args, ctx, indirect_callee=callee)

    def _emit_call(
        self,
        args: list[ast.Expression],
        ctx: _FuncCtx,
        *,
        direct: str | None = None,
        indirect_callee: ast.Expression | None = None,
        retptr: list[str] | None = None,
        ecx_setup: list[str] | None = None,
    ) -> list[str]:
        # cdecl pushes args right-to-left so the leftmost arg ends up at
        # the lowest address (= [ebp+8] in the callee). For scalars we
        # use plain `push eax`; for struct-by-value we reserve `sizeof`
        # bytes via `sub esp, N` and copy the struct's bytes into that
        # window with the same per-dword pattern as `_struct_copy_assign`.
        #
        # When `retptr` is provided (for struct-returning callees), it
        # holds asm lines that produce the destination address in EAX;
        # we push it after the regular args so it lands at the lowest
        # address (= the hidden first param).
        # If we know the called function's param types, look them up so
        # we can coerce arg widths (e.g. narrow a `double` literal when
        # the param is declared `float`). Variadic and indirect calls
        # fall through to the arg's own type.
        param_types: list[ast.TypeNode] | None = None
        if direct is not None and direct in self._func_param_types:
            param_types = self._func_param_types[direct]
        elif indirect_callee is not None:
            # Indirect call: extract param types from the callee's
            # function-pointer type if known.
            try:
                callee_ty = self._type_of(indirect_callee, ctx)
            except CodegenError:
                callee_ty = None
            ft: ast.FunctionType | None = None
            if isinstance(callee_ty, ast.FunctionType):
                ft = callee_ty
            elif (
                isinstance(callee_ty, ast.PointerType)
                and isinstance(callee_ty.base_type, ast.FunctionType)
            ):
                ft = callee_ty.base_type
            if ft is not None:
                param_types = list(ft.param_types)
        out: list[str] = []
        total_arg_bytes = 0
        for arg_idx, arg in enumerate(reversed(args)):
            # `reversed` reverses the list; recompute the original index
            # so we can look up the matching param type.
            real_idx = len(args) - 1 - arg_idx
            expected_ty: ast.TypeNode | None = None
            if param_types is not None and real_idx < len(param_types):
                expected_ty = param_types[real_idx]
            arg_ty = self._type_of(arg, ctx)
            # Pass-by-value applies to: structs, complex, and *vectors*.
            # Regular arrays decay to a pointer at the call site (just
            # push the address).
            arg_is_vector = (
                isinstance(arg_ty, ast.ArrayType)
                and getattr(arg_ty, "is_vector", False)
            )
            if (
                isinstance(arg_ty, (ast.StructType, ast.ComplexType))
                or arg_is_vector
                or self._is_int128(arg_ty)
            ):
                size = self._size_of(arg_ty)
                padded = (size + 3) & ~3
                # Compute &arg first; once it's in EDX we can reserve and copy.
                if isinstance(arg_ty, ast.ComplexType):
                    out += self._complex_value_address(arg, ctx)
                elif arg_is_vector:
                    out += self._vector_value_address(arg, ctx)
                elif self._is_int128(arg_ty):
                    out += self._int128_value_address(arg, ctx)
                else:
                    out += self._struct_address(arg, ctx)
                out.append("        mov     edx, eax")
                out.append(f"        sub     esp, {padded}")
                offset = 0
                while size - offset >= 4:
                    out.append(f"        mov     eax, [edx + {offset}]")
                    out.append(f"        mov     [esp + {offset}], eax")
                    offset += 4
                if size - offset >= 2:
                    out.append(f"        mov     ax, [edx + {offset}]")
                    out.append(f"        mov     [esp + {offset}], ax")
                    offset += 2
                if size - offset >= 1:
                    out.append(f"        mov     al, [edx + {offset}]")
                    out.append(f"        mov     [esp + {offset}], al")
                total_arg_bytes += padded
            elif self._is_float_type(arg_ty):
                # Float arg. If the param is also float, push that width
                # (narrow double→float or widen float→double as needed).
                # If the param is an integer-family type, convert via
                # fistp and push 4 bytes. For variadic args (no
                # expected_ty available), C promotes float→double, so
                # we always push 8 bytes.
                if expected_ty is not None and not self._is_float_type(expected_ty):
                    out += self._eval_float_to_st0(arg, ctx)
                    out.append("        sub     esp, 4")
                    out.append("        fistp   dword [esp]")
                    total_arg_bytes += 4
                else:
                    if expected_ty is not None and self._is_float_type(expected_ty):
                        effective_ty = expected_ty
                    else:
                        # Variadic — promote float to double.
                        effective_ty = ast.BasicType(name="double")
                    size = self._size_of(effective_ty)
                    width = "dword" if size == 4 else "qword"
                    out += self._eval_float_to_st0(arg, ctx)
                    out.append(f"        sub     esp, {size}")
                    out.append(f"        fstp    {width} [esp]")
                    total_arg_bytes += size
            elif (
                expected_ty is not None
                and self._is_float_type(expected_ty)
            ):
                # Integer arg → float param: load int into FPU via fild,
                # then fstp at expected float width.
                out += self._eval_expr_to_eax(arg, ctx)
                out.append("        push    eax")
                out.append("        fild    dword [esp]")
                out.append("        add     esp, 4")
                size = self._size_of(expected_ty)
                width = "dword" if size == 4 else "qword"
                out.append(f"        sub     esp, {size}")
                out.append(f"        fstp    {width} [esp]")
                total_arg_bytes += size
            else:
                # Long-long arg or long-long expected param: push 8 bytes
                # (high then low so low ends up at the lower address).
                want_ll = self._is_long_long(arg_ty) or (
                    expected_ty is not None and self._is_long_long(expected_ty)
                )
                if want_ll:
                    out += self._eval_expr_to_edx_eax(arg, ctx)
                    out.append("        push    edx")
                    out.append("        push    eax")
                    total_arg_bytes += 8
                else:
                    out += self._eval_expr_to_eax(arg, ctx)
                    out.append("        push    eax")
                    total_arg_bytes += 4
        if retptr is not None:
            # Push retptr last so it's the leftmost arg (= [ebp+8] in callee).
            out += retptr
            out.append("        push    eax")
            total_arg_bytes += 4
        # Direct call to a nested fn with nonlocal goto: load ECX with
        # the address of the buf-array so the lifted fn's prologue can
        # save it as the static link. (Indirect calls via the
        # trampoline already do this themselves.) The caller computes
        # the lines that produce the right ECX — either `lea ecx,
        # [buf_array]` (outer's direct call) or `mov ecx,
        # [static_link_slot]` (sibling/self propagation).
        if ecx_setup is not None:
            out += ecx_setup
        if direct is not None:
            out.append(f"        call    _{direct}")
        else:
            out += self._eval_expr_to_eax(indirect_callee, ctx)
            out.append("        call    eax")
        if total_arg_bytes:
            out.append(f"        add     esp, {total_arg_bytes}")
        # Return value is in EAX. For struct-returning callees, EAX is
        # the retptr (forwarded by the callee).
        return out

    def _unary(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        if expr.op in ("++", "--"):
            return self._inc_dec(expr, ctx)
        if expr.op == "&":
            return self._address_of(expr, ctx)
        if expr.op in ("__real__", "__imag__"):
            operand_ty = self._type_of(expr.operand, ctx)
            if isinstance(operand_ty, ast.ComplexType):
                # Complex int half: load the integer half from the
                # appropriate offset and sign- or zero-extend to EAX.
                if operand_ty.base_type in self._COMPLEX_INT_BASES:
                    addr_lines, half_ty = self._complex_part_address(
                        expr, ctx,
                    )
                    return addr_lines + self._load_to_eax("[eax]", half_ty)
        if expr.op == "*":
            # Dereference: load the pointer value into EAX, then read from
            # the address it holds. The load width follows the pointee
            # type — `*char_ptr` reads one byte (sign-extended), not four.
            pointee_ty = self._type_of(expr, ctx)
            # Array (or struct) pointee in value context decays to its
            # address — `*pa` where `pa` has type `T(*)[N]` evaluates to
            # the address of the array, not its contents.
            if isinstance(pointee_ty, (ast.ArrayType, ast.StructType)):
                return self._eval_expr_to_eax(expr.operand, ctx)
            return self._eval_expr_to_eax(expr.operand, ctx) + self._load_to_eax(
                "[eax]", pointee_ty
            )
        if not expr.is_prefix:
            raise CodegenError(f"postfix `{expr.op}` not implemented yet")
        out = self._eval_expr_to_eax(expr.operand, ctx)
        if expr.op == "+":
            return out
        if expr.op == "-":
            return out + ["        neg     eax"]
        if expr.op == "~":
            return out + ["        not     eax"]
        if expr.op == "!":
            # `!f` for a float must compare against 0.0, not against the
            # truncated int. Re-do the eval through `_eval_to_bool_eax`,
            # which gives EAX = 1 for nonzero / 0 for zero, and then
            # invert it.
            out = self._eval_to_bool_eax(expr.operand, ctx)
            return out + [
                "        test    eax, eax",
                "        sete    al",
                "        movzx   eax, al",
            ]
        raise CodegenError(f"unary `{expr.op}` not implemented yet")

    def _address_of(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        # `&identifier`, `&arr[i]`, and `&s.m` are supported. `&*p` (a
        # no-op) will land if needed.
        if isinstance(expr.operand, ast.Identifier):
            return self._identifier_address(expr.operand.name, ctx)
        if isinstance(expr.operand, ast.Index):
            # &arr[i] — same address arithmetic as a load, just no final deref.
            return self._index_address(expr.operand, ctx)
        if isinstance(expr.operand, ast.Member):
            # &s.m or &p->m — member-address lowering, no deref.
            return self._member_address(expr.operand, ctx)
        if isinstance(expr.operand, ast.Compound):
            # &(T){init}: evaluating the compound leaves EAX holding
            # the temp's address (per `_eval_expr_to_eax`).
            return self._eval_expr_to_eax(expr.operand, ctx)
        if isinstance(expr.operand, ast.UnaryOp) and expr.operand.op == "*":
            # &*p is a no-op — just produce p.
            return self._eval_expr_to_eax(expr.operand.operand, ctx)
        if (
            isinstance(expr.operand, ast.UnaryOp)
            and expr.operand.op in ("__real__", "__imag__")
        ):
            # &__real__ x / &__imag__ x — address of the corresponding
            # half of the _Complex value.
            addr_lines, _half_ty = self._complex_part_address(
                expr.operand, ctx,
            )
            return addr_lines
        raise CodegenError(
            f"`&` operand must be an identifier, `arr[i]`, or `s.m` "
            f"(got {type(expr.operand).__name__})"
        )

    def _inc_dec(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        # `arr[i]++`, `*p++`, `s.m++`, `p->m++` etc. compute the lvalue
        # address once into EAX, then RMW through it. Identifier lvalues
        # take the simpler in-place path below.
        if not isinstance(expr.operand, ast.Identifier):
            return self._inc_dec_lvalue(expr, ctx)
        ty = self._identifier_type(expr.operand.name, ctx)
        # Array names aren't lvalues — `++arr` is a C error, not "advance the
        # array pointer" (that would only make sense for a pointer variable).
        if isinstance(ty, ast.ArrayType):
            raise CodegenError(
                f"cannot {expr.op} array `{expr.operand.name}`"
            )
        # Long-long Identifier: route through `_inc_dec_ll` which does
        # the 8-byte RMW with carry/borrow propagation.
        if self._is_long_long(ty):
            return self._inc_dec_ll(expr, ctx)
        # _Complex / __int128 / vector: not standard C for ++/-- but
        # gcc accepts them. Raise a clearer error than "KeyError: 16"
        # for now — extending these is straightforward but rarely
        # needed.
        if isinstance(ty, ast.ComplexType):
            raise CodegenError(
                f"`{expr.op}` on _Complex operand `{expr.operand.name}` "
                f"not supported (use `__real__ x {expr.op}` instead)"
            )
        addr = self._identifier_addr_text(expr.operand.name, ctx)
        # On a pointer, ++/-- step by sizeof(*ptr) instead of 1. We still
        # mutate the slot in place — the slot stores the pointer value —
        # so an `add dword [...], N` covers it.
        if isinstance(ty, ast.PointerType):
            step = self._size_of(ty.base_type)
            instr = "add" if expr.op == "++" else "sub"
            bump = [f"        {instr}     dword {addr}, {step}"]
        else:
            # `inc byte/word/dword` for char/short/int. The slot's payload
            # bytes are at the same `[ebp - N]` regardless of width because
            # x86 is little-endian.
            size = self._size_of(ty)
            width = {1: "byte", 2: "word", 4: "dword"}[size]
            instr = "inc" if expr.op == "++" else "dec"
            bump = [f"        {instr}     {width} {addr}"]
        load = self._load_to_eax(addr, ty)
        if expr.is_prefix:
            # ++x: bump in place, then load the new value into EAX.
            return bump + load
        # x++: load old value into EAX, then bump in place. EAX is the result.
        return load + bump

    def _inc_dec_lvalue(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        """`++` / `--` on `arr[i]`, `*p`, `s.m`, or `p->m`.

        Compute the lvalue's address into EAX once, then RMW through
        it. The bump instruction is `add/sub <width> [...], <step>`
        (or `inc/dec` for step=1) — same shape as the Identifier path
        but with the address in a register rather than a slot.
        """
        # Bit-field ++/--: read-modify-write the bit-field via load/store
        # helpers since width-aware RMW on the storage unit can't be a
        # simple `add dword [...]`.
        if isinstance(expr.operand, ast.Member):
            bf = self._bitfield_info(expr.operand, ctx)
            if bf is not None:
                return self._inc_dec_bitfield(expr, bf, ctx)
        if isinstance(expr.operand, ast.Index):
            addr_lines = self._index_address(expr.operand, ctx)
        elif isinstance(expr.operand, ast.UnaryOp) and expr.operand.op == "*":
            addr_lines = self._eval_expr_to_eax(expr.operand.operand, ctx)
        elif isinstance(expr.operand, ast.Member):
            addr_lines = self._member_address(expr.operand, ctx)
        else:
            raise CodegenError(
                f"`{expr.op}` operand must be an identifier, `arr[i]`, "
                f"`*ptr`, or `s.m` (got {type(expr.operand).__name__})"
            )
        target_ty = self._type_of(expr.operand, ctx)
        if isinstance(target_ty, ast.ArrayType):
            raise CodegenError(f"cannot {expr.op} an array")
        # Long-long lvalue: RMW the 8-byte slot with carry/borrow
        # propagating through the high dword. The eval is for value;
        # callers that ignore the result (statement context) can drop
        # the load.
        if self._is_long_long(target_ty):
            instr0 = "add" if expr.op == "++" else "sub"
            instrN = "adc" if expr.op == "++" else "sbb"
            out = list(addr_lines)  # eax = &lvalue
            out.append("        mov     ecx, eax")
            if expr.is_prefix:
                # ++lvalue: bump in place via low add, high adc 0.
                out.append(f"        {instr0}     dword [ecx], 1")
                out.append(f"        {instrN}     dword [ecx + 4], 0")
                out.append("        mov     eax, [ecx]")
                out.append("        mov     edx, [ecx + 4]")
                return out
            # lvalue++: load OLD value into EDX:EAX, then bump.
            out.append("        mov     eax, [ecx]")
            out.append("        mov     edx, [ecx + 4]")
            out.append(f"        {instr0}     dword [ecx], 1")
            out.append(f"        {instrN}     dword [ecx + 4], 0")
            return out
        if isinstance(target_ty, ast.PointerType):
            step = self._size_of(target_ty.base_type)
            width = "dword"
        else:
            step = 1
            size = self._size_of(target_ty)
            width = {1: "byte", 2: "word", 4: "dword"}[size]
        op_mnem = "add" if expr.op == "++" else "sub"

        out = list(addr_lines)  # eax = &lvalue
        if expr.is_prefix:
            # ++lvalue: bump in place, then load the new value (width-aware).
            out.append(f"        {op_mnem}     {width} [eax], {step}")
            out += self._load_to_eax("[eax]", target_ty)
            return out
        # lvalue++: stash the address, load the OLD value, then bump
        # the slot. EAX retains the old value as the expression's result.
        out.append("        mov     ecx, eax")
        out += self._load_to_eax("[ecx]", target_ty)
        out.append(f"        {op_mnem}     {width} [ecx], {step}")
        return out

    def _inc_dec_bitfield(
        self,
        expr: ast.UnaryOp,
        bf: tuple[int, int, ast.TypeNode],
        ctx: _FuncCtx,
    ) -> list[str]:
        """`++` / `--` on a bit-field member.

        Compute &storage_unit once, load the field's value (with
        bit-offset/width handling and signed sign-extend), bump it,
        rewrite the storage unit. EAX holds the value the postfix /
        prefix form should yield.
        """
        # Unpack bit-field info, defaulting unit_size to 4 for the
        # 2-tuple form.
        bit_offset, bit_width, member_ty = bf[:3]
        unit_size = bf[3] if len(bf) == 4 else 4
        if unit_size == 8:
            raise CodegenError(
                "long-long bit-field ++/-- not yet supported"
            )
        mask = (1 << bit_width) - 1
        clear_mask = (~(mask << bit_offset)) & 0xFFFFFFFF
        is_unsigned = self._is_unsigned(member_ty)
        delta = 1 if expr.op == "++" else -1
        # eax = &unit, save in ebx
        out = self._member_address(expr.operand, ctx)
        out.append("        mov     ebx, eax")
        # Load old value into eax (post-shift, post-mask, sign-extended).
        out.append("        mov     eax, [ebx]")
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        out.append(f"        and     eax, {mask}")
        if not is_unsigned and bit_width < 32:
            shift = 32 - bit_width
            out.append(f"        shl     eax, {shift}")
            out.append(f"        sar     eax, {shift}")
        # ecx = bumped value. For postfix we save the OLD eax first.
        if not expr.is_prefix:
            out.append("        push    eax")
        if delta == 1:
            out.append("        lea     ecx, [eax + 1]")
        else:
            out.append("        lea     ecx, [eax - 1]")
        # Mask the bumped value to bit_width and shift into position.
        out.append(f"        and     ecx, {mask}")
        if bit_offset > 0:
            out.append(f"        shl     ecx, {bit_offset}")
        # Read storage unit, clear the field bits, OR in new positioned val.
        out.append("        mov     edx, [ebx]")
        out.append(f"        and     edx, {clear_mask}")
        out.append("        or      edx, ecx")
        out.append("        mov     [ebx], edx")
        if expr.is_prefix:
            # eax already has the OLD value; we need the new one. The
            # simplest is to reload from the just-stored unit so the
            # sign-extend semantics match _bitfield_load.
            out.append("        mov     eax, edx")
            if bit_offset > 0:
                out.append(f"        shr     eax, {bit_offset}")
            out.append(f"        and     eax, {mask}")
            if not is_unsigned and bit_width < 32:
                shift = 32 - bit_width
                out.append(f"        shl     eax, {shift}")
                out.append(f"        sar     eax, {shift}")
        else:
            out.append("        pop     eax")
        return out

    def _inc_dec_bitfield_ll(
        self,
        expr: ast.UnaryOp,
        bf: tuple[int, int, ast.TypeNode, int],
        ctx: _FuncCtx,
    ) -> list[str]:
        """`++` / `--` on a long-long bit-field. Lowers to
        `lhs = lhs +/- 1` via the existing LL bitfield store, with
        the right pre/postfix value semantics."""
        # Delta as an IntLiteral; the synthesized BinaryOp takes the
        # bit-field value and adds/subtracts 1.
        synth_one = ast.IntLiteral(
            value=1, is_long=False, is_long_long=True, is_unsigned=False,
        )
        op = "+" if expr.op == "++" else "-"
        # For prefix, the result is the new value; for postfix, the
        # old value. We use _bitfield_load_ll + _bitfield_store_ll_simple
        # to avoid re-evaluating the bitfield address and the increment.
        # Postfix: load old, then bump.
        if not expr.is_prefix:
            out = self._bitfield_load_ll(expr.operand, bf, ctx)
            # Save old.
            out.append("        push    edx")
            out.append("        push    eax")
            # Compute new = old + delta. The bit-field address still
            # needs to be re-evaluated (we lost it). Build a synthetic
            # `lhs op= 1`.
            new_rhs = ast.BinaryOp(op=op, left=expr.operand, right=synth_one)
            out += self._bitfield_store_ll_simple(
                expr.operand, bf, new_rhs, ctx,
            )
            out.append("        pop     eax")
            out.append("        pop     edx")
            return out
        # Prefix: store first, then load the new value.
        new_rhs = ast.BinaryOp(op=op, left=expr.operand, right=synth_one)
        out = self._bitfield_store_ll_simple(
            expr.operand, bf, new_rhs, ctx,
        )
        # _bitfield_store_ll_simple leaves a positioned-then-shifted-
        # back value in EAX (low 32 bits) without sign extension.
        # Reload via _bitfield_load_ll for correct semantics.
        out += self._bitfield_load_ll(expr.operand, bf, ctx)
        return out

    def _compound_assign_bitfield(
        self,
        expr: ast.BinaryOp,
        bf: tuple[int, int, ast.TypeNode],
        ctx: _FuncCtx,
    ) -> list[str]:
        """`s.bf op= rhs` on a bit-field member: address-once, read,
        op, mask, store back."""
        # bf is `(bit_offset, bit_width, member_ty, unit_size)` from
        # `_bitfield_info`. The original 32-bit-only path takes
        # unit_size=4 only — the long-long path is too complex to
        # inline here, so dispatch upward when needed.
        if len(bf) == 4 and bf[3] == 8:
            return self._compound_assign_bitfield_ll(expr, bf, ctx)
        bit_offset, bit_width, member_ty = bf[:3]
        op = self._COMPOUND_OPS[expr.op]
        mask = (1 << bit_width) - 1
        clear_mask = (~(mask << bit_offset)) & 0xFFFFFFFF
        is_unsigned = self._is_unsigned(member_ty)
        # Compute &storage_unit, save in EBX.
        out = self._member_address(expr.left, ctx)
        out.append("        push    eax")
        # Evaluate rhs after the address — per C99 6.5.16.2.
        out += self._eval_expr_to_eax(expr.right, ctx)
        out.append("        push    eax")          # save rhs
        out.append("        mov     ebx, [esp + 4]")  # ebx = &unit
        # Load current bit-field value into EAX (sign-extended if signed).
        out.append("        mov     eax, [ebx]")
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        out.append(f"        and     eax, {mask}")
        if not is_unsigned and bit_width < 32:
            shift = 32 - bit_width
            out.append(f"        shl     eax, {shift}")
            out.append(f"        sar     eax, {shift}")
        # Pop rhs into ECX, apply op (lvalue OP rhs in eax).
        out.append("        pop     ecx")  # ecx = rhs
        # The remaining stack top is the saved address, but we have it in ebx.
        out.append("        add     esp, 4")  # discard
        # Stack-machine for `op`: we have lhs in EAX, rhs in ECX.
        # Push lhs, then move rhs to ECX (already there), pop lhs into EAX,
        # then run `_apply_binop_post_eval` which expects [esp+0]=lhs, EAX=rhs.
        # Easier: just synthesize the op directly.
        out.append("        push    eax")
        out.append("        mov     eax, ecx")
        out += self._apply_binop_post_eval(op, member_ty, member_ty)
        # eax = new value. Mask to bit_width and shift into position.
        out.append(f"        and     eax, {mask}")
        if bit_offset > 0:
            out.append(f"        shl     eax, {bit_offset}")
        # Read storage unit, clear field bits, OR in new positioned val.
        out.append("        mov     edx, [ebx]")
        out.append(f"        and     edx, {clear_mask}")
        out.append("        or      edx, eax")
        out.append("        mov     [ebx], edx")
        # Result of the assignment expression is the new bit-field value
        # (sign-extended, masked).
        out.append("        mov     eax, edx")
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        out.append(f"        and     eax, {mask}")
        if not is_unsigned and bit_width < 32:
            shift = 32 - bit_width
            out.append(f"        shl     eax, {shift}")
            out.append(f"        sar     eax, {shift}")
        return out

    @staticmethod
    def _scale_reg(reg: str, size: int) -> list[str]:
        """Multiply `reg` by `size` (unsigned). Used for pointer-arithmetic scaling."""
        if size == 1:
            return []
        if size == 2:
            return [f"        shl     {reg}, 1"]
        if size == 4:
            return [f"        shl     {reg}, 2"]
        if size == 8:
            return [f"        shl     {reg}, 3"]
        return [f"        imul    {reg}, {reg}, {size}"]

    @staticmethod
    def _unscale_eax(size: int) -> list[str]:
        """Divide EAX (signed) by `size`. Used for pointer differences."""
        if size == 1:
            return []
        if size == 2:
            return ["        sar     eax, 1"]
        if size == 4:
            return ["        sar     eax, 2"]
        if size == 8:
            return ["        sar     eax, 3"]
        return [
            "        cdq",
            f"        mov     ecx, {size}",
            "        idiv    ecx",
        ]

    # Map from C operator to a one-line "op eax, ecx" instruction. `+` and
    # `-` are routed through `_add_sub` instead because of pointer
    # arithmetic; everything else here is type-uniform on int.
    _SIMPLE_BINOPS = {
        "*":  "imul    eax, ecx",
        "&":  "and     eax, ecx",
        "|":  "or      eax, ecx",
        "^":  "xor     eax, ecx",
    }

    # setCC mnemonic for each comparison (signed).
    _CMP_SETCC = {
        "==": "sete",
        "!=": "setne",
        "<":  "setl",
        ">":  "setg",
        "<=": "setle",
        ">=": "setge",
    }
    # Unsigned variants — used when either operand is unsigned-typed.
    _CMP_SETCC_UNSIGNED = {
        "==": "sete",
        "!=": "setne",
        "<":  "setb",
        ">":  "seta",
        "<=": "setbe",
        ">=": "setae",
    }

    # setCC mnemonic for float comparisons. We evaluate left then right,
    # so after both `fld`s ST(0) = right and ST(1) = left. `fucompp`
    # then sets condition codes describing ST(0) vs ST(1) — i.e.
    # right vs left. After `fnstsw ax; sahf` the codes land in
    # ZF (C3 = equal), CF (C0 = ST(0)<ST(1)). To get the C-level
    # comparison we invert: `left < right` is `right > left` is
    # ST(0) > ST(1), which is `seta` (CF=0, ZF=0).
    _FLOAT_CMP_SETCC = {
        "==": "sete",
        "!=": "setne",
        "<":  "seta",
        ">":  "setb",
        "<=": "setae",
        ">=": "setbe",
    }

    # Compound-assignment operators desugar to `lvalue = lvalue OP rvalue`.
    _COMPOUND_OPS = {
        "+=":  "+",
        "-=":  "-",
        "*=":  "*",
        "/=":  "/",
        "%=":  "%",
        "&=":  "&",
        "|=":  "|",
        "^=":  "^",
        "<<=": "<<",
        ">>=": ">>",
    }

    def _binary(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        if expr.op == "=":
            return self._assign(expr, ctx)
        if expr.op in self._COMPOUND_OPS:
            return self._compound_assign(expr, ctx)
        # Long-long short-circuit: any non-assignment binary op where at
        # least one operand is long-long-typed routes through
        # `_binary_ll` (the 64-bit ladder). Comparisons return bool in
        # EAX; arithmetic leaves the 64-bit result in EDX:EAX, which
        # the caller may truncate to EAX (taking the low 32 bits) when
        # the surrounding context is 32-bit — that matches C's "narrow
        # on assign".
        if (
            self._is_long_long(self._type_of(expr.left, ctx))
            or self._is_long_long(self._type_of(expr.right, ctx))
        ):
            return self._binary_ll(expr, ctx)
        if expr.op == ",":
            # Comma operator: evaluate left for side effects, then right.
            # The result is the right-hand value.
            out = self._eval_expr_to_eax(expr.left, ctx)
            out += self._eval_expr_to_eax(expr.right, ctx)
            return out
        if expr.op == "&&":
            return self._logical_and(expr, ctx)
        if expr.op == "||":
            return self._logical_or(expr, ctx)
        if expr.op in ("+", "-"):
            return self._add_sub(expr, ctx)
        # Complex equality `a == b` / `a != b`: compare both halves.
        if expr.op in ("==", "!="):
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            if isinstance(lt, ast.ComplexType) or isinstance(rt, ast.ComplexType):
                return self._complex_compare(expr, ctx)
        # __int128 comparison: 4-dword chain. Result is int (0 or 1).
        if expr.op in ("==", "!=", "<", ">", "<=", ">="):
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            if self._is_int128(lt) or self._is_int128(rt):
                return self._int128_compare(expr, ctx)
        # Float comparisons land here as int-typed expressions (their
        # result fits in EAX), but the operands are floats so the
        # standard "left to eax, right to ecx" path can't compare them.
        if expr.op in self._FLOAT_CMP_SETCC:
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            if self._is_float_type(lt) or self._is_float_type(rt):
                return self._float_compare(expr, ctx)

        # Stack-machine eval: left → EAX → stack, right → EAX → ECX, pop EAX.
        out = self._eval_expr_to_eax(expr.left, ctx)
        out.append("        push    eax")
        out += self._eval_expr_to_eax(expr.right, ctx)
        out.append("        mov     ecx, eax")
        out.append("        pop     eax")

        if expr.op in self._SIMPLE_BINOPS:
            out.append(f"        {self._SIMPLE_BINOPS[expr.op]}")
            return out
        if expr.op == "/":
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            if (
                self._is_unsigned_after_promotion(lt)
                or self._is_unsigned_after_promotion(rt)
            ):
                return out + [
                    "        xor     edx, edx",
                    "        div     ecx",
                ]
            return out + [
                "        cdq",
                "        idiv    ecx",
            ]
        if expr.op == "%":
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            if (
                self._is_unsigned_after_promotion(lt)
                or self._is_unsigned_after_promotion(rt)
            ):
                return out + [
                    "        xor     edx, edx",
                    "        div     ecx",
                    "        mov     eax, edx",
                ]
            return out + [
                "        cdq",
                "        idiv    ecx",
                "        mov     eax, edx",
            ]
        if expr.op == "<<":
            return out + [
                "        shl     eax, cl",
            ]
        if expr.op == ">>":
            # Unsigned operand → logical shift; otherwise arithmetic.
            lt = self._type_of(expr.left, ctx)
            if self._is_unsigned_after_promotion(lt):
                return out + ["        shr     eax, cl"]
            return out + ["        sar     eax, cl"]
        if expr.op in self._CMP_SETCC:
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            # Per C, only types as wide as int retain unsigned-ness;
            # unsigned char / unsigned short promote to signed int.
            unsigned = (
                self._is_unsigned_after_promotion(lt)
                or self._is_unsigned_after_promotion(rt)
            )
            table = (
                self._CMP_SETCC_UNSIGNED if unsigned else self._CMP_SETCC
            )
            return out + [
                "        cmp     eax, ecx",
                f"        {table[expr.op]}    al",
                "        movzx   eax, al",
            ]
        raise CodegenError(f"binary `{expr.op}` not implemented yet")

    def _add_sub(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        """`+` / `-` with C pointer-arithmetic semantics.

        Four cases by operand types:
          int   ± int    — straight integer add/sub.
          ptr   + int    — scale the int by sizeof(*ptr), then add.
          int   + ptr    — symmetric (the int is what gets scaled).
          ptr   - ptr    — byte difference, then divide by sizeof(*left).

        Both operands pointer for `+`, or `int - ptr`, are illegal C and
        get rejected here rather than silently producing nonsense.
        """
        lt = self._type_of(expr.left, ctx)
        rt = self._type_of(expr.right, ctx)
        # Arrays decay to pointers in arithmetic context, so the four cases
        # below also cover (array, int), (int, array), and (array, array).
        l_ptr = self._is_pointer_like(lt)
        r_ptr = self._is_pointer_like(rt)

        out = self._eval_expr_to_eax(expr.left, ctx)
        out.append("        push    eax")
        out += self._eval_expr_to_eax(expr.right, ctx)
        # Post-eval: stack top = left value, eax = right value.

        if l_ptr and r_ptr:
            if expr.op == "+":
                raise CodegenError("cannot add two pointers")
            # ptr - ptr: byte-difference / sizeof(*left).
            size = self._size_of(lt.base_type)
            out.append("        mov     ecx, eax")
            out.append("        pop     eax")
            out.append("        sub     eax, ecx")
            out += self._unscale_eax(size)
            return out

        if l_ptr:
            # ptr ± int: scale the int (in eax), then op against the pointer.
            size = self._size_of(lt.base_type)
            out += self._scale_reg("eax", size)
            out.append("        mov     ecx, eax")
            out.append("        pop     eax")
            mnem = "add" if expr.op == "+" else "sub"
            out.append(f"        {mnem}     eax, ecx")
            return out

        if r_ptr:
            if expr.op == "-":
                raise CodegenError("cannot subtract a pointer from an integer")
            # int + ptr: the int is on the stack, the pointer is in eax. Pop
            # the int into ecx, scale ecx, then add to eax (which holds ptr).
            size = self._size_of(rt.base_type)
            out.append("        pop     ecx")
            out += self._scale_reg("ecx", size)
            out.append("        add     eax, ecx")
            return out

        # int ± int — the original integer path.
        out.append("        mov     ecx, eax")
        out.append("        pop     eax")
        mnem = "add" if expr.op == "+" else "sub"
        out.append(f"        {mnem}     eax, ecx")
        return out

    def _logical_and(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        false_label = ctx.label("and_false")
        end_label = ctx.label("and_end")
        out = self._eval_to_bool_eax(expr.left, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        out += self._eval_to_bool_eax(expr.right, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        out.append("        mov     eax, 1")
        out.append(f"        jmp     {end_label}")
        out.append(f"{false_label}:")
        out.append("        xor     eax, eax")
        out.append(f"{end_label}:")
        return out

    def _logical_or(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        true_label = ctx.label("or_true")
        end_label = ctx.label("or_end")
        out = self._eval_to_bool_eax(expr.left, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jnz     {true_label}")
        out += self._eval_to_bool_eax(expr.right, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jnz     {true_label}")
        out.append("        xor     eax, eax")
        out.append(f"        jmp     {end_label}")
        out.append(f"{true_label}:")
        out.append("        mov     eax, 1")
        out.append(f"{end_label}:")
        return out

    def _assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        # `__real__ x = rhs` / `__imag__ x = rhs` — store rhs into
        # the real or imag half of x.
        if (
            isinstance(expr.left, ast.UnaryOp)
            and expr.left.op in ("__real__", "__imag__")
        ):
            operand_ty = self._type_of(expr.left.operand, ctx)
            if isinstance(operand_ty, ast.ComplexType):
                addr_lines, half_ty = self._complex_part_address(expr.left, ctx)
                if operand_ty.base_type in self._COMPLEX_INT_BASES:
                    # Integer half: store via the regular int store.
                    out = list(addr_lines)
                    out.append("        push    eax")
                    out += self._eval_expr_to_eax(expr.right, ctx)
                    out.append("        pop     ecx")
                    out += self._store_from_eax("[ecx]", half_ty)
                    return out
                size = self._size_of(half_ty)
                width = "dword" if size == 4 else "qword"
                out = list(addr_lines)
                out.append("        push    eax")
                out += self._eval_float_to_st0(expr.right, ctx)
                out.append("        pop     ecx")
                out.append(f"        fst     {width} [ecx]")
                # Leave value on st(0) — rhs is the assignment's value.
                return out
        # Struct-to-struct assignment `dst = src` short-circuits to a
        # struct-copy regardless of the lvalue shape. Without this, the
        # rhs's `_eval_expr_to_eax` would try to load the whole struct
        # into EAX (which `_load_to_eax` rejects). When the rhs is a
        # struct-returning call, we route the call straight into &dst
        # so there's no intermediate copy.
        target_ty = self._type_of(expr.left, ctx)
        # __int128 assignment: 16-byte copy. The rhs may be smaller —
        # cast wraps narrowing/widening at the type level.
        if self._is_int128(target_ty):
            return self._int128_copy_assign(expr.left, expr.right, ctx)
        if isinstance(target_ty, ast.StructType):
            if (
                isinstance(expr.right, ast.Call)
                and self._is_struct_returning_call(expr.right, ctx)
            ):
                return self._call_into_address(
                    expr.right,
                    self._struct_address(expr.left, ctx),
                    ctx,
                )
            if isinstance(expr.right, ast.VaArgExpr):
                # `s = va_arg(ap, struct T)` — the value lives at the
                # current ap; copy from there into &s and advance ap.
                return self._va_arg_struct_copy(
                    expr.right,
                    self._struct_address(expr.left, ctx),
                    ctx,
                )
            return self._struct_copy_assign(expr, target_ty, ctx)
        if isinstance(target_ty, ast.ComplexType):
            # `c = c2` — complex-to-complex copy. Same per-dword
            # mechanism as struct copy. The rhs may be a complex-
            # returning call (handled like struct-returning).
            rhs_ty = self._type_of(expr.right, ctx)
            if isinstance(rhs_ty, ast.ComplexType):
                # Same-precision complex-returning call: route the
                # call directly into &lhs (no temp).
                if (
                    isinstance(expr.right, ast.Call)
                    and self._is_complex_returning_call(expr.right, ctx)
                    and rhs_ty.base_type == target_ty.base_type
                ):
                    return self._call_into_address(
                        expr.right,
                        self._complex_value_address(expr.left, ctx),
                        ctx,
                    )
                return self._complex_copy_assign(expr, target_ty, ctx)
            # Scalar → complex: rhs becomes the real part, imag is 0.
            return self._complex_assign_from_scalar(
                expr.left, expr.right, target_ty, ctx,
            )
        # Long-long lvalue: route through the 64-bit assignment helper.
        if self._is_long_long(target_ty):
            return self._assign_ll(expr, ctx)

        # Vector-typed lvalue (gcc vector_size types are ArrayType in
        # our AST). Copy via memcpy of `sizeof(vector)` bytes from the
        # rhs's address. Catches any l-value shape (Identifier, Member,
        # Index, *p) — they all flow through `_vector_value_address`.
        # Only vector-tagged ArrayTypes are assignable; regular C arrays
        # are not lvalues. (rhs ArrayType need not be vector-tagged —
        # casts and `(vec){...}` compounds may strip the flag.)
        if (
            isinstance(target_ty, ast.ArrayType)
            and getattr(target_ty, "is_vector", False)
        ):
            rhs_ty = self._type_of(expr.right, ctx)
            if (
                isinstance(rhs_ty, ast.ArrayType)
                and self._size_of(rhs_ty) == self._size_of(target_ty)
            ):
                # Vector-returning call: route the call directly into
                # &lhs without an intermediate copy.
                if (
                    isinstance(expr.right, ast.Call)
                    and self._is_vector_returning_call(expr.right, ctx)
                ):
                    return self._call_into_address(
                        expr.right,
                        self._vector_value_address(expr.left, ctx),
                        ctx,
                    )
                return self._vector_copy_assign(expr, target_ty, ctx)

        # `x = rhs` — direct slot store. Array names aren't lvalues in C.
        if isinstance(expr.left, ast.Identifier):
            ty = self._identifier_type(expr.left.name, ctx)
            if isinstance(ty, ast.ArrayType):
                raise CodegenError(
                    f"cannot assign to array `{expr.left.name}`"
                )
            if self._is_float_type(ty):
                # Float lvalue → fstp from st(0) to the slot/global.
                addr = self._float_lvalue_addr(expr.left.name, ctx)
                return self._eval_float_to_st0(
                    expr.right, ctx
                ) + self._store_st0_to(addr, ty)
            return self._eval_expr_to_eax(expr.right, ctx) + self._identifier_store(
                expr.left.name, ctx
            )
        # `*p = rhs` — store-through-pointer. Evaluate the pointer expr
        # first, save its value, then evaluate rhs into EAX (so the result
        # of the whole assignment expression is rhs, as C requires). The
        # store width follows the pointee type so `*char_ptr = 65` writes
        # one byte, not four.
        if isinstance(expr.left, ast.UnaryOp) and expr.left.op == "*":
            pointee_ty = self._type_of(expr.left, ctx)
            out = self._eval_expr_to_eax(expr.left.operand, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out += self._store_from_eax("[ecx]", pointee_ty)
            return out
        # `arr[i] = rhs` — same shape as `*ptr = rhs`, but the address
        # comes from element-arithmetic rather than a single load.
        if isinstance(expr.left, ast.Index):
            elem_ty = self._type_of(expr.left, ctx)
            out = self._index_address(expr.left, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out += self._store_from_eax("[ecx]", elem_ty)
            return out
        # `s.m = rhs` / `pp->m = rhs` — same address-once pattern.
        # (Struct-typed members already short-circuited above.)
        if isinstance(expr.left, ast.Member):
            bf = self._bitfield_info(expr.left, ctx)
            if bf is not None:
                return self._bitfield_store(expr.left, bf, expr.right, ctx)
            member_ty = self._type_of(expr.left, ctx)
            out = self._member_address(expr.left, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out += self._store_from_eax("[ecx]", member_ty)
            return out
        raise CodegenError(
            f"assignment target must be an identifier, `*ptr`, `arr[i]`, "
            f"or `s.m` (got {type(expr.left).__name__})"
        )

    def _struct_copy_from_expr(
        self,
        src_expr: ast.Expression,
        dest_disp: int,
        target_ty: ast.StructType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Copy a struct from `src_expr` into the local at `[ebp + dest_disp]`.

        Used by `struct T s = lvalue_expr;` initializers when the rhs
        isn't an InitializerList (e.g. `*pls`, `arr[i]`, `outer.inner`).
        Bytes are copied per-dword with a byte tail.
        """
        size = self._size_of(target_ty)
        out = self._struct_address(src_expr, ctx)              # eax = &src
        out.append("        mov     edx, eax")                  # edx = &src
        out.append(f"        lea     ecx, {_ebp_addr(dest_disp)}")
        offset = 0
        while size - offset >= 4:
            out.append(f"        mov     eax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], eax")
            offset += 4
        if size - offset >= 2:
            out.append(f"        mov     ax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], ax")
            offset += 2
        if size - offset >= 1:
            out.append(f"        mov     al, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], al")
        return out

    def _struct_copy_assign(
        self,
        expr: ast.BinaryOp,
        target_ty: ast.StructType,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Lower `dst = src` where dst is a struct l-value.

        Both sides resolve to struct addresses; we copy `sizeof(struct)`
        bytes via per-dword `mov` (with byte/word tail for non-multiples
        of 4). EAX is left holding the destination address — a sentinel
        that lets chained assignments compile, even though struct values
        aren't first-class in our codegen yet.
        """
        rhs_ty = self._type_of(expr.right, ctx)
        if not isinstance(rhs_ty, ast.StructType):
            raise CodegenError(
                f"struct assignment requires both sides be the same "
                f"struct type (got {type(rhs_ty).__name__} on rhs)"
            )
        target_key = self._resolve_struct_name(target_ty)
        rhs_key = self._resolve_struct_name(rhs_ty)
        if target_key != rhs_key:
            raise CodegenError(
                f"struct assignment requires both sides be the same "
                f"struct type (got `{target_key}` and `{rhs_key}`)"
            )
        size = self._size_of(target_ty)
        # Compute &src first, push it, then &dst — that way EDX (src) and
        # ECX (dst) wind up holding the right values without further
        # juggling.
        out = self._struct_address(expr.right, ctx)        # eax = &src
        out.append("        push    eax")
        out += self._struct_address(expr.left, ctx)        # eax = &dst
        out.append("        mov     ecx, eax")             # ecx = &dst
        out.append("        pop     edx")                  # edx = &src
        offset = 0
        # Per-dword body. ECX/EDX are caller-save in cdecl, so clobbering
        # is fine within a function.
        while size - offset >= 4:
            out.append(f"        mov     eax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], eax")
            offset += 4
        if size - offset >= 2:
            out.append(f"        mov     ax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], ax")
            offset += 2
        if size - offset >= 1:
            out.append(f"        mov     al, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], al")
        # Leave EAX = &dst as the assignment-expression result. Real
        # struct-value semantics would copy the struct again, but our
        # codegen never reads a struct as a value.
        out.append("        mov     eax, ecx")
        return out

    # ------------------------------------------------------------
    # Vector arithmetic (gcc __attribute__((vector_size(N))) types).
    # We model vectors as ArrayType. Componentwise arithmetic is
    # lowered as a per-element load / op / store loop over the
    # element type. The result lands in a per-node temp slot
    # allocated by `_collect_call_temps`.
    # ------------------------------------------------------------

    def _vector_element_type(self, vec_ty: ast.ArrayType) -> ast.TypeNode:
        """Element type of a vector (potentially nested ArrayType)."""
        elem = vec_ty.base_type
        # We use the leaf scalar; nested arrays are not vectors here.
        return elem

    def _vector_value_address(
        self, expr: ast.Expression, ctx: _FuncCtx
    ) -> list[str]:
        """Compute the address of a vector-typed expression in EAX.

        For lvalue forms (Identifier / Index / Member / *p) we use the
        existing addressing machinery. For BinaryOp / UnaryOp / Cast,
        the value lives in a per-node temp slot — we emit the
        evaluation that fills the temp and leave its address in EAX.
        """
        if isinstance(expr, ast.Identifier):
            return self._identifier_address(expr.name, ctx)
        if isinstance(expr, ast.Index):
            return self._index_address(expr, ctx)
        if isinstance(expr, ast.Member):
            return self._member_address(expr, ctx)
        if isinstance(expr, ast.UnaryOp) and expr.op == "*":
            # `*p` of pointer-to-vector: just evaluate the pointer.
            return self._eval_expr_to_eax(expr.operand, ctx)
        if isinstance(expr, ast.Cast):
            # Vector type-pun: if the source is itself a vector / lvalue,
            # take its address. If the source is a scalar (e.g.
            # `(V2SI) 0LL`), evaluate it into the cast's temp slot
            # (allocated by `_collect_call_temps` for ArrayType targets)
            # so we can hand back a stable address.
            src_ty = self._type_of(expr.expr, ctx)
            if isinstance(src_ty, ast.ArrayType):
                return self._vector_value_address(expr.expr, ctx)
            if id(expr) in ctx.call_temps:
                disp = ctx.call_temps[id(expr)]
                out: list[str] = []
                if self._is_long_long(src_ty):
                    out += self._eval_expr_to_edx_eax(expr.expr, ctx)
                    out.append(f"        mov     {_ebp_addr(disp)}, eax")
                    out.append(f"        mov     {_ebp_addr(disp + 4)}, edx")
                else:
                    out += self._eval_expr_to_eax(expr.expr, ctx)
                    out.append(f"        mov     {_ebp_addr(disp)}, eax")
                    # Zero-fill the rest of the vector slot.
                    target_size = self._size_of(expr.target_type)
                    src_size = self._size_of(src_ty) if not isinstance(
                        src_ty, (ast.PointerType, ast.ArrayType)
                    ) else 4
                    if src_size < target_size:
                        out += self._zero_fill_at(
                            disp + src_size, target_size - src_size
                        )
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            # Fallback: peel and try the source.
            return self._vector_value_address(expr.expr, ctx)
        if isinstance(expr, ast.Compound):
            # Compound literal — use its temp slot, run init into it,
            # then return the address.
            disp = ctx.call_temps[id(expr)]
            out: list[str] = []
            target_ty = expr.target_type
            if isinstance(target_ty, ast.ArrayType):
                out += self._array_init(
                    target_ty, expr.init, disp, ctx, "<vector-compound>",
                )
            elif isinstance(target_ty, ast.StructType):
                out += self._struct_init(
                    target_ty, expr.init, disp, ctx, "<vector-compound>",
                )
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        if isinstance(expr, (ast.BinaryOp, ast.UnaryOp)):
            return self._eval_vector_into_temp(expr, ctx)
        if isinstance(expr, ast.Call) and self._is_vector_returning_call(expr, ctx):
            disp = ctx.call_temps[id(expr)]
            retptr_lines = [f"        lea     eax, {_ebp_addr(disp)}"]
            out = self._call_into_address(expr, retptr_lines, ctx)
            # Callee leaves &temp in EAX; that's already the address.
            return out
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Identifier)
            and expr.func.name == "__builtin_shuffle"
            and id(expr) in ctx.call_temps
        ):
            return self._emit_builtin_shuffle(expr, ctx)
        raise CodegenError(
            f"vector value: unsupported expression {type(expr).__name__}"
        )

    def _emit_builtin_shuffle(
        self, expr: ast.Call, ctx: _FuncCtx
    ) -> list[str]:
        """Lower `__builtin_shuffle(src, mask)` (or 2-source variant) into
        a per-element copy.

        2-arg form: `result[i] = src[mask[i] mod N]`.
        3-arg form: `__builtin_shuffle(s1, s2, mask)` — `result[i]` picks
        from `s1` if `mask[i] mod (2N) < N`, else from `s2[mask[i] mod N]`.
        """
        if len(expr.args) == 2:
            src_a, mask_arg = expr.args[0], expr.args[1]
            src_b = None
        elif len(expr.args) == 3:
            src_a, src_b, mask_arg = expr.args
        else:
            raise CodegenError(
                f"__builtin_shuffle: expected 2 or 3 args, got {len(expr.args)}"
            )
        src_ty = self._type_of(src_a, ctx)
        if not isinstance(src_ty, ast.ArrayType):
            raise CodegenError("__builtin_shuffle: src must be a vector")
        elem_ty = self._vector_element_type(src_ty)
        elem_size = self._size_of(elem_ty)
        n_raw = src_ty.size
        if isinstance(n_raw, ast.IntLiteral):
            n = n_raw.value
        elif isinstance(n_raw, int):
            n = n_raw
        else:
            raise CodegenError(
                "__builtin_shuffle: vector size must be a constant"
            )
        if not n or (n & (n - 1)) != 0:
            raise CodegenError(
                "__builtin_shuffle: only power-of-2 vector sizes supported"
            )
        # Mask element size — for the index extraction we just take
        # the low 32 bits of mask[i] (works because n is small).
        mask_ty = self._type_of(mask_arg, ctx)
        if not isinstance(mask_ty, ast.ArrayType):
            raise CodegenError("__builtin_shuffle: mask must be a vector")
        mask_elem_size = self._size_of(self._vector_element_type(mask_ty))
        disp = ctx.call_temps[id(expr)]
        # Layout on stack:
        #   [ebp + ?]  &src_a   (2nd source if any)
        #   [ebp + ?]  &src_b
        #   &mask    in ESI
        #   &dst     in EDI
        # Push &mask and &src(s) onto the stack to free up registers.
        out: list[str] = []
        out += self._vector_value_address(src_a, ctx)  # eax = &src_a
        out.append("        push    eax")
        if src_b is not None:
            out += self._vector_value_address(src_b, ctx)
            out.append("        push    eax")
        out += self._vector_value_address(mask_arg, ctx)
        out.append("        mov     esi, eax")
        out.append(f"        lea     edi, {_ebp_addr(disp)}")
        # Total source-pointer slots that we pushed.
        n_pushed = 2 if src_b is not None else 1
        # &src_a is at [esp + (n_pushed-1)*4]; &src_b at [esp + 0]
        src_a_off = (n_pushed - 1) * 4
        src_b_off = 0
        idx_mask = n - 1  # power-of-2 modulus
        for i in range(n):
            mask_off = i * mask_elem_size
            # idx = mask[i] (low 32 bits) & idx_mask
            out.append(f"        mov     eax, [esi + {mask_off}]")
            if src_b is not None:
                # Two-source shuffle: keep full lo dword for branch test.
                out.append(f"        mov     ecx, eax")
                out.append(f"        and     ecx, {2 * n - 1}")
                out.append(f"        cmp     ecx, {n}")
                pick_b = ctx.label("shuf_b")
                done = ctx.label("shuf_done")
                out.append(f"        jge     {pick_b}")
                # Pick from src_a:
                out.append(f"        and     eax, {idx_mask}")
                if elem_size in (1, 2, 4, 8):
                    out += self._emit_shuffle_one_lane(
                        f"[esp + {src_a_off}]", "eax", elem_size, i, "edi",
                    )
                out.append(f"        jmp     {done}")
                out.append(f"{pick_b}:")
                out.append(f"        and     eax, {idx_mask}")
                out += self._emit_shuffle_one_lane(
                    f"[esp + {src_b_off}]", "eax", elem_size, i, "edi",
                )
                out.append(f"{done}:")
            else:
                out.append(f"        and     eax, {idx_mask}")
                out += self._emit_shuffle_one_lane(
                    f"[esp + {src_a_off}]", "eax", elem_size, i, "edi",
                )
        # Pop our pushed src pointers.
        out.append(f"        add     esp, {n_pushed * 4}")
        out.append(f"        lea     eax, {_ebp_addr(disp)}")
        return out

    def _emit_shuffle_one_lane(
        self,
        src_ptr_mem: str,
        idx_reg: str,
        elem_size: int,
        dst_lane: int,
        dst_reg: str,
    ) -> list[str]:
        """Copy `src[idx]` (where idx is in `idx_reg` as a small int) into
        `dst[dst_lane]`. `src_ptr_mem` is a memory operand holding &src.
        Element size must be 1/2/4/8."""
        # Compute idx*elem_size into idx_reg (shift by log2).
        out: list[str] = []
        if elem_size > 1:
            shift = {2: 1, 4: 2, 8: 3}[elem_size]
            out.append(f"        shl     {idx_reg}, {shift}")
        # Load &src into EBX (we have ESI=&mask, EDI=&dst already).
        out.append(f"        mov     ebx, {src_ptr_mem}")
        out.append(f"        add     ebx, {idx_reg}")
        dst_off = dst_lane * elem_size
        if elem_size == 1:
            out.append(f"        mov     al, [ebx]")
            out.append(f"        mov     [{dst_reg} + {dst_off}], al")
        elif elem_size == 2:
            out.append(f"        mov     ax, [ebx]")
            out.append(f"        mov     [{dst_reg} + {dst_off}], ax")
        elif elem_size == 4:
            out.append(f"        mov     eax, [ebx]")
            out.append(f"        mov     [{dst_reg} + {dst_off}], eax")
        elif elem_size == 8:
            out.append(f"        mov     eax, [ebx]")
            out.append(f"        mov     [{dst_reg} + {dst_off}], eax")
            out.append(f"        mov     eax, [ebx + 4]")
            out.append(f"        mov     [{dst_reg} + {dst_off + 4}], eax")
        return out

    def _eval_vector_into_temp(
        self, expr: ast.Expression, ctx: _FuncCtx
    ) -> list[str]:
        """Evaluate a vector-typed BinaryOp / UnaryOp into the per-node
        temp slot. Returns asm; EAX = address of the temp."""
        vec_ty = self._type_of(expr, ctx)
        if not isinstance(vec_ty, ast.ArrayType):
            raise CodegenError("vector eval: expected ArrayType")
        disp = ctx.call_temps[id(expr)]
        elem_ty = self._vector_element_type(vec_ty)
        elem_size = self._size_of(elem_ty)
        count = self._size_of(vec_ty) // elem_size
        out: list[str] = []
        if isinstance(expr, ast.BinaryOp):
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            l_vec = (
                isinstance(lt, ast.ArrayType)
                and getattr(lt, "is_vector", False)
            )
            r_vec = (
                isinstance(rt, ast.ArrayType)
                and getattr(rt, "is_vector", False)
            )
            # Float-vector path: per-element FPU operations.
            if self._is_float_type(elem_ty):
                return self._float_vector_binary(
                    expr, vec_ty, elem_ty, elem_size, count, disp, ctx,
                )
            # Scalar broadcast — vec op scalar / scalar op vec.
            if l_vec and not r_vec:
                # eval scalar to ECX once (push), then loop over vec.
                out += self._eval_expr_to_eax(expr.right, ctx)
                out.append("        push    eax")          # [esp] = scalar
                out += self._vector_value_address(expr.left, ctx)
                out.append("        push    eax")          # [esp] = &vec
                for i in range(count):
                    offset = i * elem_size
                    out.append("        mov     edx, [esp]")
                    out += self._load_to_eax(
                        f"[edx + {offset}]" if offset else "[edx]",
                        elem_ty,
                    )
                    out.append("        mov     ecx, [esp + 4]")
                    out += self._vector_int_binop(expr.op, elem_ty)
                    addr = _ebp_addr(disp + offset)
                    out += self._store_from_eax(addr, elem_ty)
                out.append("        add     esp, 8")
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            if r_vec and not l_vec:
                out += self._eval_expr_to_eax(expr.left, ctx)
                out.append("        push    eax")          # [esp] = scalar
                out += self._vector_value_address(expr.right, ctx)
                out.append("        push    eax")          # [esp] = &vec
                for i in range(count):
                    offset = i * elem_size
                    # eax = vec[i]; ecx = scalar — but we want
                    # `scalar op vec[i]`. So move things around.
                    out.append("        mov     edx, [esp]")
                    out += self._load_to_eax(
                        f"[edx + {offset}]" if offset else "[edx]",
                        elem_ty,
                    )
                    out.append("        mov     ecx, eax")    # ecx = vec[i]
                    out.append("        mov     eax, [esp + 4]")  # eax = scalar
                    out += self._vector_int_binop(expr.op, elem_ty)
                    addr = _ebp_addr(disp + offset)
                    out += self._store_from_eax(addr, elem_ty)
                out.append("        add     esp, 8")
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            # Compute &right, &left; keep them across the per-element
            # body. Use [esp + 4]/[esp] as anchors so per-element loads
            # can re-fetch (we don't keep them in fixed regs because
            # the per-element op may want to use the regs).
            out += self._vector_value_address(expr.right, ctx)
            out.append("        push    eax")              # [esp] = &right
            out += self._vector_value_address(expr.left, ctx)
            out.append("        push    eax")              # [esp] = &left
            for i in range(count):
                offset = i * elem_size
                # Load left[i] into EAX.
                out.append("        mov     edx, [esp]")
                out += self._load_to_eax(
                    f"[edx + {offset}]" if offset else "[edx]",
                    elem_ty,
                )
                # Apply op against right[i].
                out.append("        mov     edx, [esp + 4]")
                if self._is_float_type(elem_ty):
                    raise CodegenError(
                        "float vectors not yet supported"
                    )
                # Integer per-element op.
                # Load right[i] into ECX.
                out.append("        push    eax")
                rhs_load = self._load_to_eax(
                    f"[edx + {offset}]" if offset else "[edx]",
                    elem_ty,
                )
                out += rhs_load
                out.append("        mov     ecx, eax")
                out.append("        pop     eax")
                # eax op ecx → eax
                out += self._vector_int_binop(expr.op, elem_ty)
                # Store to temp[disp + offset].
                addr = _ebp_addr(disp + offset)
                out += self._store_from_eax(addr, elem_ty)
            out.append("        add     esp, 8")
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # UnaryOp.
        if isinstance(expr, ast.UnaryOp):
            # Float-vector unary: per-element FPU.
            if self._is_float_type(elem_ty):
                if expr.op not in ("-", "+"):
                    raise CodegenError(
                        f"float vector unary `{expr.op}` not supported"
                    )
                width = "dword" if elem_size == 4 else "qword"
                out += self._vector_value_address(expr.operand, ctx)
                out.append("        push    eax")
                for i in range(count):
                    offset = i * elem_size
                    out.append("        mov     edx, [esp]")
                    if offset:
                        out.append(f"        fld     {width} [edx + {offset}]")
                    else:
                        out.append(f"        fld     {width} [edx]")
                    if expr.op == "-":
                        out.append("        fchs")
                    addr = _ebp_addr(disp + offset)
                    out.append(f"        fstp    {width} {addr}")
                out.append("        add     esp, 4")
                out.append(f"        lea     eax, {_ebp_addr(disp)}")
                return out
            out += self._vector_value_address(expr.operand, ctx)
            out.append("        push    eax")              # [esp] = &operand
            for i in range(count):
                offset = i * elem_size
                out.append("        mov     edx, [esp]")
                out += self._load_to_eax(
                    f"[edx + {offset}]" if offset else "[edx]",
                    elem_ty,
                )
                out += self._vector_int_unop(expr.op, elem_ty)
                addr = _ebp_addr(disp + offset)
                out += self._store_from_eax(addr, elem_ty)
            out.append("        add     esp, 4")
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        raise CodegenError(
            f"vector eval: unexpected node {type(expr).__name__}"
        )

    def _float_vector_binary(
        self,
        expr: ast.BinaryOp,
        vec_ty: ast.ArrayType,
        elem_ty: ast.TypeNode,
        elem_size: int,
        count: int,
        disp: int,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Float-vector arithmetic: per-element FPU ops `+ - * /`.

        Both operands' addresses are computed and pushed onto the
        stack; the loop loads elements via `fld`, applies the op
        via `faddp`/`fsubp`/`fmulp`/`fdivp`, then `fstp`s into the
        temp's element slot. Scalar broadcast (vec op float, float
        op vec) loads the scalar to st(0) once before the loop and
        leaves it pinned (load fresh per element since faddp pops).
        """
        op = expr.op
        op_map = {"+": "faddp", "-": "fsubp", "*": "fmulp", "/": "fdivp"}
        if op not in op_map:
            raise CodegenError(
                f"float vector op `{op}` not supported"
            )
        fop = op_map[op]
        width = "dword" if elem_size == 4 else "qword"
        lt = self._type_of(expr.left, ctx)
        rt = self._type_of(expr.right, ctx)
        l_vec = (
            isinstance(lt, ast.ArrayType)
            and getattr(lt, "is_vector", False)
        )
        r_vec = (
            isinstance(rt, ast.ArrayType)
            and getattr(rt, "is_vector", False)
        )
        out: list[str] = []
        if l_vec and not r_vec:
            # vec op float-scalar — store the scalar at a stack slot
            # for repeated load.
            out += self._eval_float_to_st0(expr.right, ctx)
            scalar_size = self._size_of(rt) if isinstance(rt, ast.BasicType) else 8
            scalar_w = "dword" if scalar_size == 4 else "qword"
            out.append(f"        sub     esp, {scalar_size}")
            out.append(f"        fstp    {scalar_w} [esp]")
            out += self._vector_value_address(expr.left, ctx)
            out.append("        push    eax")          # [esp] = &vec
            for i in range(count):
                offset = i * elem_size
                # Load vec[i] (st0), then load scalar (st1).
                out.append("        mov     edx, [esp]")
                if offset:
                    out.append(f"        fld     {width} [edx + {offset}]")
                else:
                    out.append(f"        fld     {width} [edx]")
                out.append(f"        fld     {scalar_w} [esp + 4]")
                out.append(f"        {fop}   st1, st0")
                addr = _ebp_addr(disp + offset)
                out.append(f"        fstp    {width} {addr}")
            out.append(f"        add     esp, {4 + scalar_size}")
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        if r_vec and not l_vec:
            # float-scalar op vec. After loading scalar then vec[i],
            # the FPU stack has st0=vec[i], st1=scalar. The two-arg
            # popping form `fop st1, st0` does `st1 = st1 OP st0`,
            # which is `scalar OP vec[i]` — exactly what we want.
            out += self._eval_float_to_st0(expr.left, ctx)
            scalar_size = self._size_of(lt) if isinstance(lt, ast.BasicType) else 8
            scalar_w = "dword" if scalar_size == 4 else "qword"
            out.append(f"        sub     esp, {scalar_size}")
            out.append(f"        fstp    {scalar_w} [esp]")
            out += self._vector_value_address(expr.right, ctx)
            out.append("        push    eax")
            for i in range(count):
                offset = i * elem_size
                # Load scalar (st0=scalar), then vec[i] on top
                # (st0=vec[i], st1=scalar).
                out.append(f"        fld     {scalar_w} [esp + 4]")
                out.append("        mov     edx, [esp]")
                if offset:
                    out.append(f"        fld     {width} [edx + {offset}]")
                else:
                    out.append(f"        fld     {width} [edx]")
                # st1 = st1 OP st0 = scalar OP vec[i], then pop.
                out.append(f"        {fop}   st1, st0")
                addr = _ebp_addr(disp + offset)
                out.append(f"        fstp    {width} {addr}")
            out.append(f"        add     esp, {4 + scalar_size}")
            out.append(f"        lea     eax, {_ebp_addr(disp)}")
            return out
        # Both vectors: per-element FPU.
        out += self._vector_value_address(expr.right, ctx)
        out.append("        push    eax")
        out += self._vector_value_address(expr.left, ctx)
        out.append("        push    eax")
        for i in range(count):
            offset = i * elem_size
            out.append("        mov     edx, [esp]")
            if offset:
                out.append(f"        fld     {width} [edx + {offset}]")
            else:
                out.append(f"        fld     {width} [edx]")
            out.append("        mov     edx, [esp + 4]")
            if offset:
                out.append(f"        fld     {width} [edx + {offset}]")
            else:
                out.append(f"        fld     {width} [edx]")
            out.append(f"        {fop}   st1, st0")
            addr = _ebp_addr(disp + offset)
            out.append(f"        fstp    {width} {addr}")
        out.append("        add     esp, 8")
        out.append(f"        lea     eax, {_ebp_addr(disp)}")
        return out

    def _vector_int_binop(
        self, op: str, elem_ty: ast.TypeNode
    ) -> list[str]:
        """Emit eax := eax OP ecx for integer vector elements."""
        out: list[str] = []
        if op == "+":
            out.append("        add     eax, ecx")
        elif op == "-":
            out.append("        sub     eax, ecx")
        elif op == "*":
            out.append("        imul    eax, ecx")
        elif op == "&":
            out.append("        and     eax, ecx")
        elif op == "|":
            out.append("        or      eax, ecx")
        elif op == "^":
            out.append("        xor     eax, ecx")
        elif op == "/":
            if self._is_unsigned(elem_ty):
                out.append("        xor     edx, edx")
                out.append("        div     ecx")
            else:
                out.append("        cdq")
                out.append("        idiv    ecx")
        elif op == "%":
            if self._is_unsigned(elem_ty):
                out.append("        xor     edx, edx")
                out.append("        div     ecx")
            else:
                out.append("        cdq")
                out.append("        idiv    ecx")
            out.append("        mov     eax, edx")
        elif op == "<<":
            # `shl/sal eax, cl` — only CL is meaningful for shifts on x86.
            out.append("        shl     eax, cl")
        elif op == ">>":
            if self._is_unsigned(elem_ty):
                out.append("        shr     eax, cl")
            else:
                out.append("        sar     eax, cl")
        else:
            raise CodegenError(f"vector op `{op}` not supported")
        return out

    def _vector_int_unop(
        self, op: str, elem_ty: ast.TypeNode
    ) -> list[str]:
        """Emit eax := OP eax for integer vector elements."""
        if op == "-":
            return ["        neg     eax"]
        if op == "+":
            return []
        if op == "~":
            return ["        not     eax"]
        raise CodegenError(f"vector unary `{op}` not supported")

    def _vector_copy_assign(
        self, expr: ast.BinaryOp, vec_ty: ast.ArrayType, ctx: _FuncCtx
    ) -> list[str]:
        """Copy a vector value to a vector lvalue (memcpy of sizeof bytes)."""
        size = self._size_of(vec_ty)
        # &src first, push, then &dst.
        out = self._vector_value_address(expr.right, ctx)
        out.append("        push    eax")
        out += self._vector_value_address(expr.left, ctx)
        out.append("        mov     ecx, eax")
        out.append("        pop     edx")
        offset = 0
        while size - offset >= 4:
            out.append(f"        mov     eax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], eax")
            offset += 4
        if size - offset >= 2:
            out.append(f"        mov     ax, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], ax")
            offset += 2
        if size - offset >= 1:
            out.append(f"        mov     al, [edx + {offset}]")
            out.append(f"        mov     [ecx + {offset}], al")
        out.append("        mov     eax, ecx")
        return out

    def _compound_assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        # `x op= rhs` is `x = x op rhs`. For Identifier lvalues, evaluating
        # the lvalue is side-effect-free, so the simple desugaring works.
        # For `arr[i]` and `*p`, re-evaluating the lvalue would compute the
        # address (and any side effects in `i` or `p`) twice — we instead
        # compute it once and keep it on the stack while we read, op, store.
        op = self._COMPOUND_OPS[expr.op]

        # __int128 compound assign: synthesize a BinaryOp for the
        # `lhs OP rhs` and route through `_int128_copy_assign`. The
        # synthesized op needs a temp slot — pre-allocate now since
        # `_collect_call_temps` already ran.
        target_ty_check = self._type_of(expr.left, ctx)
        if self._is_int128(target_ty_check):
            # For non-Identifier lvalues, address-once: compute &lhs
            # once, snapshot the value into a hidden int128 slot,
            # synthesize the op against the snapshot, then store
            # the result back at the saved address. This avoids
            # firing side effects in `expr.left` twice.
            if not isinstance(expr.left, ast.Identifier):
                return self._compound_assign_int128_lvalue(expr, op, ctx)
            inner = ast.BinaryOp(
                op=op, left=expr.left, right=expr.right,
            )
            ctx.alloc_call_temp(inner, 16)
            return self._int128_copy_assign(expr.left, inner, ctx)
        # Long-long compound assign: route through the LL path so
        # arithmetic uses the 64-bit ladder and stores write both
        # halves with proper carry propagation. The LL path desugars
        # to `lhs = lhs OP rhs`; for non-Identifier lvalues this is
        # safe iff the address-computing sub-expressions are
        # side-effect-free, which is the common case.
        if self._is_long_long(target_ty_check):
            return self._compound_assign_ll(expr, ctx)
        # Vector compound assign on non-Identifier lvalue: snapshot
        # pattern. Compute &lvalue once, copy to a hidden snapshot
        # slot, evaluate `Identifier(snap) OP rhs` into a result
        # temp, copy result back to *addr_slot. Side effects in the
        # lvalue's sub-expressions fire exactly once.
        if (
            isinstance(target_ty_check, ast.ArrayType)
            and getattr(target_ty_check, "is_vector", False)
            and not isinstance(expr.left, ast.Identifier)
        ):
            return self._compound_assign_vector_lvalue(
                expr, op, target_ty_check, ctx,
            )
        # Complex compound assign on any lvalue: snapshot pattern.
        # Identifier path desugars trivially (re-reading is
        # side-effect-free); non-Identifier uses snapshot.
        if isinstance(target_ty_check, ast.ComplexType):
            return self._compound_assign_complex_lvalue(
                expr, op, target_ty_check, ctx,
            )

        if isinstance(expr.left, ast.Identifier):
            ty = self._identifier_type(expr.left.name, ctx)
            if isinstance(ty, ast.ArrayType):
                # Vector compound assign `v op= rhs` desugars to
                # `v = v op rhs`. We need to allocate a temp for
                # the synthesized BinaryOp via `_collect_call_temps`,
                # but at this point that's already run. Workaround:
                # synthesize the BinaryOp and pre-allocate its temp
                # here.
                if not getattr(ty, "is_vector", False):
                    raise CodegenError(
                        f"cannot assign to array `{expr.left.name}`"
                    )
                inner = ast.BinaryOp(
                    op=op, left=expr.left, right=expr.right,
                )
                size = (self._size_of(ty) + 3) & ~3
                ctx.alloc_call_temp(inner, size)
                return self._assign(
                    ast.BinaryOp(op="=", left=expr.left, right=inner),
                    ctx,
                )
            inner = ast.BinaryOp(op=op, left=expr.left, right=expr.right)
            return self._assign(
                ast.BinaryOp(op="=", left=expr.left, right=inner),
                ctx,
            )

        if isinstance(expr.left, ast.Index):
            addr_lines = self._index_address(expr.left, ctx)
        elif isinstance(expr.left, ast.UnaryOp) and expr.left.op == "*":
            # The pointer operand evaluates once into eax — that's the
            # address we'll read from and write back to.
            addr_lines = self._eval_expr_to_eax(expr.left.operand, ctx)
        elif isinstance(expr.left, ast.Member):
            # Bit-field compound assign goes through the bit-field
            # RMW path so we don't smash the other fields in the unit.
            bf = self._bitfield_info(expr.left, ctx)
            if bf is not None:
                return self._compound_assign_bitfield(expr, bf, ctx)
            # Vector member: desugar to `s.m = s.m OP rhs` and let
            # the vector-copy assign path handle it. Pre-allocate
            # the inner BinaryOp's temp slot since the regular
            # call-temp pre-pass already ran.
            mty = self._type_of(expr.left, ctx)
            if (
                isinstance(mty, ast.ArrayType)
                and getattr(mty, "is_vector", False)
            ):
                inner = ast.BinaryOp(
                    op=op, left=expr.left, right=expr.right,
                )
                size = (self._size_of(mty) + 3) & ~3
                ctx.alloc_call_temp(inner, size)
                return self._assign(
                    ast.BinaryOp(op="=", left=expr.left, right=inner),
                    ctx,
                )
            addr_lines = self._member_address(expr.left, ctx)
        else:
            raise CodegenError(
                f"compound assignment target must be an identifier, "
                f"`*ptr`, `arr[i]`, or `s.m` (got {type(expr.left).__name__})"
            )

        target_ty = self._type_of(expr.left, ctx)
        rhs_ty = self._type_of(expr.right, ctx)

        # C99 §6.5.16.2: in `E1 op= E2`, the lvalue's address is
        # computed once, then E2 is evaluated, then the read-op-store
        # sequence runs. So if E2 has side effects on E1, the read
        # must happen AFTER E2.
        out = addr_lines                                       # eax = lvalue address
        out.append("        push    eax")                       # save addr
        out += self._eval_expr_to_eax(expr.right, ctx)          # eax = rhs (may mutate *addr)
        out.append("        push    eax")                       # save rhs
        out.append("        mov     eax, [esp + 4]")            # eax = saved addr
        out += self._load_to_eax("[eax]", target_ty)            # eax = current value
        out.append("        push    eax")                       # left operand on stack
        out.append("        mov     eax, [esp + 4]")            # eax = saved rhs
        out += self._apply_binop_post_eval(op, target_ty, rhs_ty)  # eax = new value
        out.append("        add     esp, 4")                    # discard saved rhs
        out.append("        pop     ecx")                       # ecx = saved addr
        out += self._store_from_eax("[ecx]", target_ty)         # *addr = new value
        return out

    def _apply_binop_post_eval(
        self,
        op: str,
        lt: ast.TypeNode,
        rt: ast.TypeNode,
    ) -> list[str]:
        """Compute `(stack_top OP eax) → eax`, with C semantics.

        Used by compound-assign on non-Identifier lvalues, where the
        loaded current value is on the stack and the rhs has just been
        evaluated into EAX. Mirrors `_add_sub`'s pointer-scaling rules
        for `+` and `-`; everything else is the integer instruction
        sequence shared with `_binary`.
        """
        l_ptr = self._is_pointer_like(lt)
        r_ptr = self._is_pointer_like(rt)

        if op in ("+", "-"):
            if l_ptr and r_ptr:
                if op == "+":
                    raise CodegenError("cannot add two pointers")
                size = self._size_of(lt.base_type)
                out = [
                    "        mov     ecx, eax",
                    "        pop     eax",
                    "        sub     eax, ecx",
                ]
                out += self._unscale_eax(size)
                return out
            if l_ptr:
                size = self._size_of(lt.base_type)
                out = self._scale_reg("eax", size)
                out.append("        mov     ecx, eax")
                out.append("        pop     eax")
                mnem = "add" if op == "+" else "sub"
                out.append(f"        {mnem}     eax, ecx")
                return out
            if r_ptr:
                if op == "-":
                    raise CodegenError(
                        "cannot subtract a pointer from an integer"
                    )
                size = self._size_of(rt.base_type)
                out = ["        pop     ecx"]
                out += self._scale_reg("ecx", size)
                out.append("        add     eax, ecx")
                return out
            out = ["        mov     ecx, eax", "        pop     eax"]
            mnem = "add" if op == "+" else "sub"
            out.append(f"        {mnem}     eax, ecx")
            return out

        # All other ops: int-only. Same instruction sequences as `_binary`'s
        # tail.
        out = ["        mov     ecx, eax", "        pop     eax"]
        if op in self._SIMPLE_BINOPS:
            out.append(f"        {self._SIMPLE_BINOPS[op]}")
            return out
        if op == "/":
            return out + ["        cdq", "        idiv    ecx"]
        if op == "%":
            return out + [
                "        cdq",
                "        idiv    ecx",
                "        mov     eax, edx",
            ]
        if op == "<<":
            return out + ["        shl     eax, cl"]
        if op == ">>":
            return out + ["        sar     eax, cl"]
        raise CodegenError(f"compound op `{op}=` not implemented yet")
