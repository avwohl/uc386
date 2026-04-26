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
        # Struct definitions: name → list of (member_name, member_type,
        # offset). `_struct_sizes` is the corresponding total size in bytes
        # (rounded up to struct alignment).
        self._structs: dict[str, list[tuple[str, ast.TypeNode, int]]] = {}
        self._struct_sizes: dict[str, int] = {}
        # Names of structs that were declared with `union` so init/copy
        # paths know members share storage.
        self._struct_unions: set[str] = set()
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

        functions = [
            d for d in top_decls
            if isinstance(d, ast.FunctionDecl) and d.body is not None
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
        # Build the function-return-type and param-type tables from
        # every declaration in the unit (defined or extern). Calls to
        # unknown names default to int in `_type_of`; arg coercion in
        # `_emit_call` only fires when the param types are known.
        self._func_return_types = {}
        self._func_param_types = {}
        for d in top_decls:
            if isinstance(d, ast.FunctionDecl):
                self._func_return_types[d.name] = d.return_type
                self._func_param_types[d.name] = [
                    p.param_type for p in d.params
                ]
            elif isinstance(d, ast.VarDecl) and isinstance(d.var_type, ast.FunctionType):
                self._func_return_types[d.name] = d.var_type.return_type
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
            elif isinstance(d, ast.TypedefDecl):
                # Typedef'd enums register their constants at file scope
                # (e.g. `typedef enum { X, Y } T;` declares X and Y).
                if (
                    isinstance(d.target_type, ast.EnumType)
                    and d.target_type.values
                ):
                    self._register_enum_values(d.target_type.values)
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
        for d in top_decls:
            if isinstance(d, ast.VarDecl) and not isinstance(d.var_type, ast.FunctionType):
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
                if d.init is not None:
                    if d.name in self._global_inits:
                        raise CodegenError(
                            f"global `{d.name}` has multiple initializers"
                        )
                    self._global_inits[d.name] = d.init
        extern_list = sorted(set(extern_list))

        lines: list[str] = []
        lines += self._header(extern_list)
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
        while self._pending_functions:
            fn = self._pending_functions.pop(0)
            function_blocks.append(self._function(fn))
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
            out.append(f"_{name}:")
            out += self._emit_global_init(ty, init, name)
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
        for value, label in sorted(self._strings.items(), key=lambda kv: kv[1]):
            out.append(f"{label}:")
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
        uninit = sorted(
            name for name in self._globals if name not in self._global_inits
        )
        if not uninit:
            return []
        out = ["        section .bss"]
        for name in uninit:
            ty = self._globals[name]
            size = self._size_of(ty)
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
        # Scalar globals with a literal init.
        if isinstance(ty, ast.BasicType):
            if self._is_float_type(ty):
                # Float globals can initialize from a float or int literal.
                # NASM accepts the decimal form in `dd` / `dq` and converts
                # to IEEE-754; a plain integer like `100` becomes `100.0`.
                directive = "dd" if ty.name == "float" else "dq"
                if isinstance(init, ast.FloatLiteral):
                    return [f"        {directive}      {init.value!r}"]
                if isinstance(init, ast.IntLiteral):
                    return [f"        {directive}      {float(init.value)!r}"]
                if isinstance(init, ast.UnaryOp) and init.op == "-":
                    inner = init.operand
                    if isinstance(inner, ast.FloatLiteral):
                        return [f"        {directive}      {(-inner.value)!r}"]
                    if isinstance(inner, ast.IntLiteral):
                        return [f"        {directive}      {float(-inner.value)!r}"]
                raise CodegenError(
                    f"global `{name}`: float init must be a numeric literal"
                )
            value = self._const_eval(init, name)
            directive = self._DATA_DIRECTIVE[self._size_of(ty)]
            return [f"        {directive}      {value}"]
        if isinstance(ty, ast.PointerType):
            # Allow `int *p = 0` (null pointer literal) and
            # `int *p = &other_global` / `char *s = "literal"` — both
            # become `dd <label>` so the linker resolves the address.
            if (
                isinstance(init, ast.UnaryOp)
                and init.op == "&"
                and isinstance(init.operand, ast.Identifier)
                and (
                    init.operand.name in self._globals
                    or init.operand.name in self._func_return_types
                )
            ):
                return [f"        dd      _{init.operand.name}"]
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
            if (
                isinstance(init, ast.Identifier)
                and init.name in self._func_return_types
            ):
                return [f"        dd      _{init.name}"]
            # `int *p = some_other_global;` — array name decays to address.
            if (
                isinstance(init, ast.Identifier)
                and init.name in self._globals
                and isinstance(self._globals[init.name], ast.ArrayType)
            ):
                return [f"        dd      _{init.name}"]
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
            return self._emit_global_array_init(ty, init, name)
        if isinstance(ty, ast.StructType):
            return self._emit_global_struct_init(ty, init, name)
        raise CodegenError(
            f"global `{name}`: unsupported type {type(ty).__name__}"
        )

    def _emit_global_struct_init(
        self,
        struct_ty: ast.StructType,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        if not isinstance(init, ast.InitializerList):
            raise CodegenError(
                f"global `{name}`: struct init must be `{{...}}`"
            )
        sname = self._resolve_struct_name(struct_ty)
        members = self._structs[sname]
        total = self._struct_sizes[sname]
        member_index = {mn: i for i, (mn, _, _) in enumerate(members)}
        bitfields = self._struct_bitfields.get(sname, {})
        if bitfields:
            return self._emit_global_bitfield_struct_init(
                sname, members, bitfields, total, init, name,
            )

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
                next_cursor = cursor + 1
                _, _, this_off = members[idx]
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
            # Find the run of members sharing this offset.
            j = i + 1
            while j < len(members) and members[j][2] == m_off:
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

        Walks the InitializerList and members in parallel. For each
        bit-field member, OR its value (masked to bit_width and shifted
        to bit_offset) into the value at unit_offset. Non-bit-field
        members are not currently supported in the same struct as bit
        fields at global scope.
        """
        # Map storage_unit_offset → packed dword value.
        units: dict[int, int] = {}
        # Walk values in declaration order. Designators not supported here.
        cursor = 0
        for value in init.values:
            if cursor >= len(members):
                raise CodegenError(
                    f"global `{name}`: too many initializers"
                )
            m_name_i, m_ty_i, m_off = members[cursor]
            cursor += 1
            if m_name_i not in bitfields:
                raise CodegenError(
                    f"global `{name}`: mixed bit-field and regular "
                    f"members not supported in bit-field struct init"
                )
            bit_offset, bit_width = bitfields[m_name_i]
            v = self._const_eval(value, f"{name}.{m_name_i}")
            mask = (1 << bit_width) - 1
            v_masked = (v & mask) << bit_offset
            units[m_off] = units.get(m_off, 0) | v_masked
        out: list[str] = []
        emit_cursor = 0
        for off in sorted(units.keys()):
            if off > emit_cursor:
                out.append(f"        times {off - emit_cursor} db 0")
                emit_cursor = off
            out.append(f"        dd      0x{units[off] & 0xFFFFFFFF:08X}")
            emit_cursor = off + 4
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
            # Flat value — gather up to leaf_count consecutive flat
            # values into an InitializerList for one element.
            group = []
            j = i
            while j < len(values) and j - i < leaf_count:
                vj = values[j]
                if isinstance(vj, (ast.DesignatedInit, ast.InitializerList)):
                    break
                group.append(vj)
                j += 1
            if len(group) == leaf_count:
                out.append(ast.InitializerList(values=group))
                i = j
            else:
                # Couldn't form a complete group — pass through, let
                # the downstream type-check raise a clearer error.
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
            if not (
                isinstance(elem_type, ast.BasicType) and elem_type.name == "char"
            ):
                raise CodegenError(
                    f"global `{name}`: string init requires a char array"
                )
            raw_bytes = list(init.value.encode())
            # C: a string literal initializing a char array of the exact
            # length drops the trailing null. If the array is larger,
            # the null is included and remaining slots zero-fill.
            if len(raw_bytes) > length:
                raise CodegenError(
                    f"global `{name}`: string init exceeds array size {length}"
                )
            if len(raw_bytes) == length:
                # No room for the null terminator — emit just the bytes.
                return [f"        db      {self._render_string(init.value)}"]
            # Render the printable run + explicit null + zero-pad.
            chars_with_null = raw_bytes + [0]
            parts = [self._render_string(init.value), "0"]
            parts.extend(["0"] * (length - len(chars_with_null)))
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
                if isinstance(value, ast.DesignatedInit):
                    if (
                        len(value.designators) != 1
                        or not isinstance(value.designators[0], ast.IntLiteral)
                    ):
                        raise CodegenError(
                            f"global `{name}`: only single-level integer "
                            f"designators supported in array init"
                        )
                    idx = value.designators[0].value
                    actual = value.value
                    cursor = idx + 1
                else:
                    idx = cursor
                    actual = value
                    cursor += 1
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
                values = []
                for i in range(length):
                    if i in slots:
                        values.append(str(self._const_eval(slots[i], name)))
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
        if isinstance(expr, ast.SizeofType):
            return self._size_of(expr.target_type)
        if isinstance(expr, ast.SizeofExpr):
            # `sizeof(expr)` — operand is unevaluated; we just need its
            # static type. _type_of needs a ctx, but for top-level usage
            # (array dimensions, global initializers) there's no function
            # context. A fresh empty _FuncCtx works because the type-of
            # path falls through to globals when no local matches.
            ty = self._type_of(expr.expr, _FuncCtx())
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
            return self._types_equal(a.base_type, b.base_type)
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

    def _intern_string(self, value: str) -> str:
        if value in self._strings:
            return self._strings[value]
        label = f"_uc386_str{len(self._strings)}"
        self._strings[value] = label
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
        # _start: call user main, take its int return in EAX, exit DOS via
        # INT 21h/4Ch with AL = exit code. AH=4Ch leaves AL untouched.
        return [
            "_start:",
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
            for sub in self._walk_ast(nested.body):
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
        for sub in nested_decls:
            mangled = ctx.nested_fn_names.get(sub.name)
            if mangled:
                self._lifted_captures[mangled] = capture_remap
                self._lifted_nested_fn_names[mangled] = sibling_lifts
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
        if isinstance(fn.return_type, ast.StructType):
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
        self._collect_call_temps(fn.body, ctx)
        # Third pass: assign a NASM label to every user `label:`. Done
        # ahead of body emission so a forward `goto` can resolve.
        self._collect_labels(fn.body, ctx)

        body = self._compound(fn.body, ctx)

        out = [f"_{fn.name}:"]
        out.append("        push    ebp")
        out.append("        mov     ebp, esp")
        if ctx.frame_size:
            out.append(f"        sub     esp, {ctx.frame_size}")
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
        out += body
        # C99: falling off the end of main returns 0. For other functions
        # this is technically undefined, but a deterministic zero beats
        # leaking whatever EAX held.
        out.append("        xor     eax, eax")
        out.append(".epilogue:")
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
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.Call) and self._is_struct_returning_call(sub, None):
                ret_ty = self._func_return_types[self._call_target(sub)]
                size = (self._size_of(ret_ty) + 3) & ~3
                ctx.alloc_call_temp(sub, size)
            elif isinstance(sub, ast.Compound):
                size = (self._size_of(sub.target_type) + 3) & ~3
                ctx.alloc_call_temp(sub, size)

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
                return
            # Round slot size up to a 4-byte boundary so a `char`-sized slot
            # doesn't push subsequent int slots off-alignment. Arrays whose
            # payload isn't a multiple of 4 (e.g. `char arr[5]`) get padded
            # the same way.
            raw = self._size_of(var_type)
            slot = (raw + 3) & ~3
            ctx.alloc_local(node.name, var_type, slot, decl=node)
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
        {"bool", "char", "short", "int", "long", "long long",
         "float", "double", "long double"}
    )

    def _check_supported_type(self, t: ast.TypeNode, name: str) -> None:
        # Pointers, and arrays / scalars / structs / enums of supported
        # base types. Slot sizes are rounded up to 4 in `_collect_locals`
        # so adjacent ints stay 4-aligned. Unsized arrays (`int a[]`
        # without an init) are caught by `_resolved_var_type` before they
        # reach this check.
        if isinstance(t, ast.PointerType):
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
                    # Variable-length arrays (`int a[n]`) aren't truly
                    # supported — there's no run-time `alloca` step. As a
                    # compile-only convenience we pick a fixed slot size
                    # so codegen proceeds; the program won't run correctly
                    # but it won't crash the compiler either.
                    t.size = ast.IntLiteral(value=16)
            self._check_supported_type(t.base_type, name)
            return
        if isinstance(t, ast.StructType):
            # `_resolve_struct_name` lazily registers typedef'd or
            # otherwise-inline-defined structs and raises if neither a
            # registered name nor inline members are available.
            self._resolve_struct_name(t)
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
                # A positional value already wrapped in `{}` consumes one
                # element regardless of leaf_count. A flat value in a
                # leaves > 1 array consumes 1/leaves of an element.
                if isinstance(value, ast.InitializerList):
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
                not isinstance(v, (ast.DesignatedInit, ast.InitializerList))
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
            if align > max_align:
                max_align = align

        if decl.is_union:
            # All members share offset 0; total size = max member size,
            # rounded up to the union's alignment. (Bitfields in unions
            # are unusual; we don't try to support them here.)
            total = 0
            for m in decl.members:
                if m.bit_width is not None:
                    raise CodegenError(
                        f"`{decl.name}`: bit-fields in unions not supported"
                    )
                if m.name is None:
                    # Anonymous nested member: promote each of its inner
                    # members at offset 0 (union scope).
                    inner_key = self._anon_member_layout_key(m.member_type)
                    for in_name, in_ty, in_off in self._structs[inner_key]:
                        members.append((in_name, in_ty, 0 + in_off))
                else:
                    members.append((m.name, m.member_type, 0))
                size = self._size_of(m.member_type)
                if size > total:
                    total = size
            total = (total + max_align - 1) & ~(max_align - 1)
            self._structs[decl_name] = members
            self._struct_sizes[decl_name] = total
            self._struct_unions.add(decl_name)
            return

        # Struct layout. Two cursors run in parallel: a byte cursor for
        # regular members, and a bit cursor within the current 32-bit
        # storage unit for bit-fields. Adjacent bit-fields pack into the
        # same unit; a regular member or a bit-field that would overflow
        # the current unit forces a new unit.
        unit_offset = 0     # byte offset of the current bit-field unit
        unit_used = 0       # bits used in the current unit (0..32)
        for m in decl.members:
            if m.bit_width is not None:
                if not isinstance(m.bit_width, ast.IntLiteral):
                    raise CodegenError(
                        f"`{decl.name}.{m.name}`: bit-field width must "
                        f"be an integer literal"
                    )
                width = m.bit_width.value
                if width < 0 or width > 32:
                    raise CodegenError(
                        f"`{decl.name}.{m.name}`: bit-field width "
                        f"{width} out of range (0..32)"
                    )
                if width == 0:
                    # C: zero-width forces alignment to the next unit.
                    if unit_used > 0:
                        unit_offset += 4
                        unit_used = 0
                    continue
                if unit_used + width > 32:
                    unit_offset += 4
                    unit_used = 0
                # Anonymous bit-field (`unsigned : N`): just consume bits
                # without registering a member. Used as inline padding.
                if m.name is None:
                    unit_used += width
                    continue
                members.append((m.name, m.member_type, unit_offset))
                bitfields[m.name] = (unit_used, width)
                unit_used += width
            else:
                # Regular member — finish any in-progress bit-field unit
                # before laying it out.
                if unit_used > 0:
                    unit_offset += 4
                    unit_used = 0
                align = self._alignment_of(m.member_type)
                offset = (unit_offset + align - 1) & ~(align - 1)
                if m.name is None:
                    # Anonymous nested struct/union — promote each inner
                    # member into the outer namespace at offset+inner_off.
                    inner_key = self._anon_member_layout_key(m.member_type)
                    for in_name, in_ty, in_off in self._structs[inner_key]:
                        members.append((in_name, in_ty, offset + in_off))
                else:
                    members.append((m.name, m.member_type, offset))
                unit_offset = offset + self._size_of(m.member_type)
        total = unit_offset + (4 if unit_used > 0 else 0)
        total = (total + max_align - 1) & ~(max_align - 1)
        self._structs[decl_name] = members
        self._struct_sizes[decl_name] = total
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
            members = self._structs.get(self._resolve_struct_name(t))
            if not members:
                return 1
            return max(self._alignment_of(mt) for _, mt, _ in members)
        if isinstance(t, ast.EnumType):
            return 4
        return 1

    @staticmethod
    def _is_unsigned(t: ast.TypeNode) -> bool:
        # `is_signed=None` is the language default — signed for
        # char/short/int. EnumType is treated as unsigned for bit-field
        # purposes (matches GCC's choice for enums of non-negative
        # values, which is the common case).
        if isinstance(t, ast.BasicType):
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
        raise CodegenError(
            f"can't take address of {type(expr).__name__} for `.member`"
        )

    def _bitfield_info(
        self, expr: ast.Member, ctx: _FuncCtx
    ) -> tuple[int, int, ast.TypeNode] | None:
        """If `expr` is a bit-field, return `(bit_offset, bit_width, type)`.

        The address that `_member_address` returns for a bit-field is the
        address of the underlying 32-bit storage unit, not the field's
        bit-position; `bit_offset` and `bit_width` then describe how to
        read/write the field within that unit.
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
        bit_offset, bit_width = info
        member_ty, _ = self._member_layout(struct_name, expr.member)
        return bit_offset, bit_width, member_ty

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
            # Reading a nested struct as a value requires struct-copy
            # codegen; defer until that slice lands. Until then, only
            # uses that immediately take an address (`.inner.field`) work,
            # and those go through `_struct_address` not `_member_load`.
            raise CodegenError(
                "reading a nested struct as a value is not supported yet"
            )
        return addr + self._load_to_eax("[eax]", member_ty)

    def _bitfield_load(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode],
        ctx: _FuncCtx,
    ) -> list[str]:
        """Read a bit-field: load the storage unit, shift, mask, sign-extend."""
        bit_offset, bit_width, member_ty = bf
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

    def _bitfield_store(
        self,
        expr: ast.Member,
        bf: tuple[int, int, ast.TypeNode],
        rhs: ast.Expression,
        ctx: _FuncCtx,
    ) -> list[str]:
        """Write a bit-field: position the rhs, mask the storage, OR them in."""
        bit_offset, bit_width, member_ty = bf
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

    # ---- identifier resolution (local vs global) ----------------------

    def _identifier_type(self, name: str, ctx: _FuncCtx) -> ast.TypeNode:
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
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            ty = ctx.lookup_type(name)
            disp = ctx.lookup(name)
            if isinstance(ty, ast.ArrayType):
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
        name = ctx.local_static_labels.get(name, name)
        name = ctx.local_captures.get(name, name)
        name = ctx.nested_fn_names.get(name, name)
        if ctx.has_local(name) and not self._is_extern_redirect(name, ctx):
            return [f"        lea     eax, {_ebp_addr(ctx.lookup(name))}"]
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
            # value >= bias: re-eval, subtract bias, fistp, add 0x80000000.
            out += self._eval_float_to_st0(expr.expr, ctx)
            out.append(f"        fld     dword [{bias_label}]")
            out.append("        fsubp   st1, st0")
            out.append("        sub     esp, 4")
            out.append("        fistp   dword [esp]")
            out.append("        pop     eax")
            out.append("        add     eax, 0x80000000")
            out.append(f"        jmp     {label_done}")
            out.append(f"{label_below}:")
            out += self._eval_float_to_st0(expr.expr, ctx)
            out.append("        sub     esp, 4")
            out.append("        fistp   dword [esp]")
            out.append("        pop     eax")
            out.append(f"{label_done}:")
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
            target = ctx.user_labels.get(item.label)
            if target is None:
                raise CodegenError(
                    f"goto: unknown label `{item.label}`"
                )
            return [f"        jmp     {target}"]
        if isinstance(item, ast.BreakStmt):
            if not ctx.break_targets:
                raise CodegenError("`break` outside of a loop or switch")
            return [f"        jmp     {ctx.break_targets[-1]}"]
        if isinstance(item, ast.ContinueStmt):
            if not ctx.continue_targets:
                raise CodegenError("`continue` outside of a loop")
            return [f"        jmp     {ctx.continue_targets[-1]}"]
        if isinstance(item, ast.AsmStmt):
            # Inline asm is treated as a no-op; we don't honor the
            # template, constraints, or clobbers. Tests that use it
            # purely as an optimization barrier (most of them) work
            # because uc386 doesn't aggressively optimize across the
            # statement boundary.
            return []
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
            # definition; emit no code.
            if item.is_definition:
                self._register_struct(item)
            return []
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
        case_specs: list[tuple[str, int | None, str]] = []
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
                    case_specs.append(("default", None, default_label))
                    case_label_map[id(node)] = default_label
                else:
                    value = self._const_eval(node.value, "case")
                    lbl = ctx.label("case")
                    case_specs.append(("case", value, lbl))
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

        # Eval the controlling expression once.
        out = self._eval_expr_to_eax(stmt.expr, ctx)
        for kind, value, target in case_specs:
            if kind == "case":
                out.append(f"        cmp     eax, {value}")
                out.append(f"        je      {target}")
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
            # Lay out the bytes, append a null. ASCII string literals only
            # for now — escape handling lives in `_render_string` for the
            # `.data` path; for inline init we just pull byte values.
            bytes_to_store = list(init.value.encode()) + [0]
            if len(bytes_to_store) > length:
                raise CodegenError(
                    f"`{name}`: string initializer exceeds array size {length}"
                )
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
                if isinstance(value_expr, ast.DesignatedInit):
                    if (
                        len(value_expr.designators) != 1
                        or not isinstance(value_expr.designators[0], ast.IntLiteral)
                    ):
                        raise CodegenError(
                            f"`{name}`: only single-level integer "
                            f"designators supported in array init"
                        )
                    idx = value_expr.designators[0].value
                    actual = value_expr.value
                    cursor = idx + 1
                else:
                    idx = cursor
                    actual = value_expr
                    cursor += 1
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
                if (
                    len(value.designators) != 1
                    or not isinstance(value.designators[0], str)
                ):
                    raise CodegenError(
                        f"`{name}`: only single-level `.field` "
                        f"designators supported in struct init"
                    )
                m_name_des = value.designators[0]
                if m_name_des not in member_index:
                    raise CodegenError(
                        f"`{name}`: unknown member `{m_name_des}` in "
                        f"struct `{struct_ty.name}`"
                    )
                idx = member_index[m_name_des]
                actual = value.value
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
                # between them: the value goes to members[idx], and
                # the rest at this offset are not separately consumed).
                next_cursor = cursor + 1
                _, _, this_off = members[idx]
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
            elif self._is_float_type(m_ty):
                out += self._eval_float_to_st0(actual, ctx)
                out += self._store_st0_to(_ebp_addr(m_disp), m_ty)
            elif self._is_long_long(m_ty):
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
            elif m_name_i in bitfields:
                # Bit-field: synthesize a fake Member node so _bitfield_store
                # can compute the storage-unit address. We can't pass the
                # base struct lvalue's name directly, so we build a Member
                # against an auto-keyed Identifier whose type/address come
                # from a one-shot synthesized lookup. Easiest: emit the
                # store inline using the stored bit_offset/width.
                bit_offset, bit_width = bitfields[m_name_i]
                mask = (1 << bit_width) - 1
                clear_mask = (~(mask << bit_offset)) & 0xFFFFFFFF
                store = self._eval_expr_to_eax(actual, ctx)
                store.append(f"        and     eax, {mask}")
                if bit_offset > 0:
                    store.append(f"        shl     eax, {bit_offset}")
                # m_disp is the storage unit (offset already equals the
                # unit_offset). RMW the unit at [ebp + m_disp].
                unit_addr = _ebp_addr(m_disp)
                store.append("        push    eax")
                store.append(f"        mov     ecx, {unit_addr}")
                store.append(f"        and     ecx, {clear_mask}")
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
        """
        size = self._size_of(ret_ty)
        # eax = retptr first (destination); push it for after the source
        # eval, which clobbers EAX.
        out = [f"        mov     eax, {_ebp_addr(retptr_disp)}"]
        out.append("        push    eax")
        out += self._struct_address(src_expr, ctx)  # eax = &src
        out.append("        mov     edx, eax")
        out.append("        pop     ecx")           # ecx = retptr
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
            if unsigned:
                # `mul ecx` → EDX:EAX = unsigned product. Overflow iff
                # EDX != 0. Capture before EDX is clobbered.
                out.append("        mul     ecx")
                out.append("        test    edx, edx")
                out.append("        setnz   dl")
                out.append("        mov     [ebx], eax")
                out.append("        movzx   eax, dl")
                return out
            out.append("        imul    ecx")
            flag = "seto"
        out.append(f"        {flag}    dl")
        out.append("        mov     [ebx], eax")
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

    def _call_target(self, call: ast.Call) -> str | None:
        """If the call is direct (a known function name), return that name."""
        callee = self._stripped_callee(call)
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in self._func_return_types
        ):
            return callee.name
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

    def _is_struct_returning_call(self, call: ast.Call, ctx: _FuncCtx) -> bool:
        """True iff `call` invokes a known function that returns a struct.

        Indirect calls through function pointers always return False —
        we don't track function-pointer return types yet, so we treat
        them as "not struct" to avoid spurious detection.
        """
        target = self._call_target(call)
        if target is None:
            return False
        return isinstance(self._func_return_types[target], ast.StructType)

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
                if v > 0xFFFFFFFFFFFFFFFF:
                    name = "long long"   # shouldn't happen, but be safe
                elif v > 0xFFFFFFFF:
                    name = "long long"
                    if is_hex:
                        unsigned = unsigned or v > 0x7FFFFFFFFFFFFFFF
                elif v > 0x7FFFFFFF:
                    # On i386 long is 32-bit, so this is unsigned long
                    # (which is just unsigned int, same width).
                    if not unsigned and is_hex:
                        unsigned = True
                    elif not unsigned and not is_hex:
                        # decimal with `L` suffix: skip unsigned long,
                        # promote directly to long long.
                        name = "long long"
                # name stays "int" / "long" otherwise; we treat them as
                # the same width here.
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
        if isinstance(expr, ast.FloatLiteral):
            return ast.BasicType(name="float" if expr.is_float else "double")
        if isinstance(expr, ast.StringLiteral):
            return ast.PointerType(base_type=ast.BasicType(name="char", is_const=True))
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
            # `+` and `-` and `~` propagate float / long long / unsigned
            # through; `!` always produces int.
            if expr.op in ("+", "-", "~"):
                inner = self._type_of(expr.operand, ctx)
                if self._is_float_type(inner):
                    return inner
                if self._is_long_long(inner):
                    return inner
                if self._is_unsigned(inner):
                    return ast.BasicType(name="int", is_signed=False)
            return ast.BasicType(name="int")
        if isinstance(expr, ast.Index):
            arr_type = self._type_of(expr.array, ctx)
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
            return ty
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
                return self._type_of(expr.left, ctx)
            if expr.op in self._COMPOUND_OPS:
                return self._type_of(expr.left, ctx)
            # `*` `/` (and the relational/bitwise ops) need to know if
            # either operand is float — the result of `1.5 * 2` is float,
            # not int.
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
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
            # Arrays count as pointer-like via decay.
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
            # Long-long promotion: if either operand is long long, the
            # result is long long.
            if self._is_long_long(lt) or self._is_long_long(rt):
                return ast.BasicType(
                    name="long long",
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

    def _eval_expr_to_eax(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        # Float-typed expressions live on the FPU stack; if a caller
        # wants the value in EAX (e.g. `int x = (float)n`), evaluate to
        # st(0) then convert via `fistp` through a stack scratch slot.
        if self._is_float_type(self._type_of(expr, ctx)):
            out = self._eval_float_to_st0(expr, ctx)
            out.append("        sub     esp, 4")
            out.append("        fistp   dword [esp]")
            out.append("        pop     eax")
            return out
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
        if isinstance(expr, ast.StringLiteral):
            label = self._intern_string(expr.value)
            return [f"        mov     eax, {label}"]
        if isinstance(expr, ast.Identifier):
            return self._identifier_load(expr.name, ctx)
        if isinstance(expr, ast.Index):
            return self._index_load(expr, ctx)
        if isinstance(expr, ast.Member):
            return self._member_load(expr, ctx)
        if isinstance(expr, ast.UnaryOp):
            return self._unary(expr, ctx)
        if isinstance(expr, ast.BinaryOp):
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
            return [f"        mov     eax, {self._size_of(expr.target_type)}"]
        if isinstance(expr, ast.SizeofExpr):
            # C: the operand of `sizeof` is *not* evaluated — only its
            # static type matters. So we infer the type and never emit
            # any of the operand's lowering code (no slot loads, no
            # function calls).
            return [
                f"        mov     eax, {self._size_of(self._type_of(expr.expr, ctx))}"
            ]
        if isinstance(expr, ast.StmtExpr):
            # GCC statement expression: `({ stmt; stmt; expr; })`. Lower
            # the body as a regular compound; the value of the last
            # ExpressionStmt is what the StmtExpr produces (already in
            # EAX from `_expr_stmt`'s evaluation). If the body ends in a
            # non-expression, EAX is left at whatever the last emitted
            # code put there — undefined but not crashing.
            return self._compound(expr.body, ctx)
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
                # int → long long: extend per the SOURCE's signedness
                # (since the value's bit pattern comes from the source).
                if self._is_float_type(src_ty):
                    out = self._eval_float_to_st0(expr.expr, ctx)
                    out += [
                        "        sub     esp, 8",
                        "        fistp   qword [esp]",
                        "        pop     eax",
                        "        pop     edx",
                    ]
                    return out
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
                out += [
                    "        sub     esp, 8",
                    "        fistp   qword [esp]",
                    "        pop     eax",
                    "        pop     edx",
                ]
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
            # cdecl returns 64-bit values in EDX:EAX. The standard call
            # path leaves the low half in EAX; we need to make sure EDX
            # is preserved as the high half.
            return self._call(expr, ctx)
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

    def _binary_ll(
        self, expr: ast.BinaryOp, ctx: _FuncCtx,
    ) -> list[str]:
        """64-bit binary op. Result in EDX:EAX. Stack-machine eval:
        right → push EDX:EAX, left → EDX:EAX, pop into ECX:EBX, op.
        """
        op = expr.op
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
            return out + ["        add     eax, ecx", "        adc     edx, ebx"]
        if op == "-":
            return out + ["        sub     eax, ecx", "        sbb     edx, ebx"]
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
            return out + [
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
        if op == ">>":
            unsigned = self._is_unsigned(self._type_of(expr.left, ctx))
            shift_high = "shr" if unsigned else "sar"
            ext = "xor edx, edx" if unsigned else "sar edx, 31"
            big = ctx.label("ll_shr_big")
            done = ctx.label("ll_shr_done")
            return out + [
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
        if op == "*":
            # 64x64 → 64 truncated multiply.
            #   left  = LH:LL (EDX:EAX);  right = RH:RC (EBX:ECX)
            #   low  32 = (LL*RC) low
            #   high 32 = (LL*RC) high + (LL*RH) low + (LH*RC) low
            return out + [
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
            # Comma: discard left's value, keep right.
            return self._eval_expr_to_edx_eax(expr.right, ctx) if not False else out
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
        """++/-- on a long-long Identifier lvalue."""
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError("long-long ++/-- on non-Identifier not supported")
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
            raise CodegenError(
                f"unary `{expr.op}` not supported on float operand"
            )
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
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
        out = self._eval_float_to_st0(expr.left, ctx)
        out += self._eval_float_to_st0(expr.right, ctx)
        out.append(f"        {op_to_mnem[expr.op]}   st1, st0")
        return out

    def _float_cast(self, expr: ast.Cast, ctx: _FuncCtx) -> list[str]:
        """Cast to a float target. The source may be int or float."""
        target = expr.target_type
        source_ty = self._type_of(expr.expr, ctx)
        if self._is_float_type(source_ty):
            # Float-to-float cast — for our purposes (single x87 path),
            # the bit-pattern conversion would happen on store, but on
            # st(0) the value is in 80-bit form regardless. Just leave
            # it on st(0); the eventual fstp picks the width.
            return self._eval_float_to_st0(expr.expr, ctx)
        # Int-to-float — `_eval_float_to_st0` already promotes via fild
        # when the operand is int-typed.
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
        """`++f` / `--f` / `f++` / `f--` for a float Identifier.

        Pre-form: bump the slot, leaving the new value on st(0).
        Post-form: load the slot twice, bump one copy, store it, leaving
        the original value on st(0) so the expression yields the old.
        """
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError(
                f"`{expr.op}` on a float requires an identifier"
            )
        name = expr.operand.name
        ty = self._identifier_type(name, ctx)
        addr = self._float_lvalue_addr(name, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        op_mnem = "faddp" if expr.op == "++" else "fsubp"
        if expr.is_prefix:
            return [
                f"        fld     {width} {addr}",
                "        fld1",
                f"        {op_mnem}   st1, st0",
                f"        fst     {width} {addr}",
            ]
        # Post-form: keep the old value on st(0) after storing the new.
        return [
            f"        fld     {width} {addr}",   # st0 = old (the value we yield)
            f"        fld     {width} {addr}",   # st0 = old, st1 = old
            "        fld1",                       # st0 = 1, st1 = old, st2 = old
            f"        {op_mnem}   st1, st0",      # pops, st1 = old±1. Now st0 = old±1, st1 = old.
            f"        fstp    {width} {addr}",   # store new, pop. st0 = old.
        ]

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
        then add.
        """
        arr_type = self._type_of(expr.array, ctx)
        if not self._is_pointer_like(arr_type):
            raise CodegenError(
                f"index target must be array or pointer "
                f"(got {type(arr_type).__name__})"
            )
        elem_size = self._size_of(arr_type.base_type)
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
        # Rebind nested-fn names to their lifted top-level mangled name
        # so the rest of the path treats them as regular file-scope
        # functions. The pre-pass at the top of `_function` registers
        # the mangled name in `_func_return_types` / `_func_param_types`.
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in ctx.nested_fn_names
        ):
            callee = ast.Identifier(
                name=ctx.nested_fn_names[callee.name],
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
            if callee.name == "va_start":
                return self._va_start(expr.args, ctx)
            if callee.name == "va_end":
                return ["        xor     eax, eax"]
            # GCC branch-prediction hint: `__builtin_expect(expr, val)`
            # has the value of `expr`. We ignore the hint and just emit
            # the first argument's value.
            if callee.name == "__builtin_expect" and len(expr.args) >= 1:
                return self._eval_expr_to_eax(expr.args[0], ctx)
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
            if (
                callee.name == "__builtin_constant_p"
                and len(expr.args) == 1
            ):
                try:
                    self._const_eval(expr.args[0], "<bcp>")
                    return ["        mov     eax, 1"]
                except CodegenError:
                    return ["        xor     eax, eax"]
            # `__builtin_unreachable()` and `__builtin_trap()` are
            # diagnostic-only — emit a 0 in EAX so calls to them in
            # value position are at least defined.
            if callee.name in ("__builtin_unreachable", "__builtin_trap"):
                return ["        xor     eax, eax"]
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
            return self._emit_call(expr.args, ctx, direct=callee.name)

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
            if isinstance(arg_ty, ast.StructType):
                size = self._size_of(arg_ty)
                padded = (size + 3) & ~3
                # Compute &arg first; once it's in EDX we can reserve and copy.
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
            width = {1: "byte", 2: "word", 4: "dword", 8: "dword"}[size]
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
        if isinstance(target_ty, ast.PointerType):
            step = self._size_of(target_ty.base_type)
            width = "dword"
        else:
            step = 1
            size = self._size_of(target_ty)
            width = {1: "byte", 2: "word", 4: "dword", 8: "dword"}[size]
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
        bit_offset, bit_width, member_ty = bf
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
        # Struct-to-struct assignment `dst = src` short-circuits to a
        # struct-copy regardless of the lvalue shape. Without this, the
        # rhs's `_eval_expr_to_eax` would try to load the whole struct
        # into EAX (which `_load_to_eax` rejects). When the rhs is a
        # struct-returning call, we route the call straight into &dst
        # so there's no intermediate copy.
        target_ty = self._type_of(expr.left, ctx)
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
        # Long-long lvalue: route through the 64-bit assignment helper.
        if self._is_long_long(target_ty):
            return self._assign_ll(expr, ctx)

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

    def _compound_assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        # `x op= rhs` is `x = x op rhs`. For Identifier lvalues, evaluating
        # the lvalue is side-effect-free, so the simple desugaring works.
        # For `arr[i]` and `*p`, re-evaluating the lvalue would compute the
        # address (and any side effects in `i` or `p`) twice — we instead
        # compute it once and keep it on the stack while we read, op, store.
        op = self._COMPOUND_OPS[expr.op]

        if isinstance(expr.left, ast.Identifier):
            ty = self._identifier_type(expr.left.name, ctx)
            if isinstance(ty, ast.ArrayType):
                raise CodegenError(
                    f"cannot assign to array `{expr.left.name}`"
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
