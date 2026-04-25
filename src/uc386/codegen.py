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


def _ebp_addr(disp: int) -> str:
    """Render an EBP-relative address. Locals have negative disp, params positive."""
    if disp < 0:
        return f"[ebp - {-disp}]"
    return f"[ebp + {disp}]"


class _FuncCtx:
    """Per-function lowering state: locals, params, label gen, loop stack."""

    def __init__(self) -> None:
        # Maps a name to its signed displacement from EBP. Locals are
        # negative (below EBP), params are positive (above the saved EBP
        # and return address — first param is at +8 in cdecl).
        self.slots: dict[str, int] = {}
        # Parallel map from name to its declared TypeNode. Used by
        # `_type_of` to drive pointer-arithmetic scaling.
        self.types: dict[str, "ast.TypeNode"] = {}
        self.frame_size: int = 0          # bytes reserved by `sub esp, frame_size`
        self._next_label: int = 0
        # Stack of (continue_target, break_target) for the enclosing loops.
        # Stack of jump targets for control-flow keywords. Loops push to
        # both stacks; switches push only to `break_targets`. Splitting
        # the stacks lets `continue` inside a switch resolve to the
        # enclosing loop, the way C requires.
        self.break_targets: list[str] = []
        self.continue_targets: list[str] = []
        # Per-call-site temp slots for struct-returning calls. Keyed by
        # `id(call_node)` so each Call expression in the function gets
        # its own buffer (so `make(1).x + make(2).x` works).
        self.call_temps: dict[int, int] = {}
        # User-declared `label:` → NASM label mapping for goto. Pre-walked
        # before body emission so forward gotos can resolve.
        self.user_labels: dict[str, str] = {}

    def alloc_local(self, name: str, ty: "ast.TypeNode", size: int = 4) -> int:
        if name in self.slots:
            raise CodegenError(f"redeclaration of `{name}`")
        # Each local sits at the next 4-byte slot below EBP.
        self.frame_size += size
        self.slots[name] = -self.frame_size
        self.types[name] = ty
        return self.slots[name]

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
        if name in self.slots:
            raise CodegenError(f"duplicate parameter `{name}`")
        self.slots[name] = disp
        self.types[name] = ty
        return self.slots[name]

    def lookup(self, name: str) -> int:
        if name not in self.slots:
            raise CodegenError(f"unknown identifier `{name}`")
        return self.slots[name]

    def lookup_type(self, name: str) -> "ast.TypeNode":
        if name not in self.types:
            raise CodegenError(f"unknown identifier `{name}`")
        return self.types[name]

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

    def generate(self, unit: ast.TranslationUnit) -> str:
        functions = [
            d for d in unit.declarations
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
        for d in unit.declarations:
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
        # Build the function-return-type table from every declaration in
        # the unit (defined or extern). Calls to unknown names default to
        # int in `_type_of`.
        self._func_return_types = {}
        for d in unit.declarations:
            if isinstance(d, ast.FunctionDecl):
                self._func_return_types[d.name] = d.return_type
            elif isinstance(d, ast.VarDecl) and isinstance(d.var_type, ast.FunctionType):
                self._func_return_types[d.name] = d.var_type.return_type

        # Reset struct + globals state. Structs need to be registered
        # before globals because a global of struct type will look up the
        # struct's size during validation.
        self._structs = {}
        self._struct_sizes = {}
        self._struct_bitfields = {}
        self._enum_constants = {}
        self._float_constants = {}
        for d in unit.declarations:
            if isinstance(d, ast.StructDecl) and d.is_definition:
                self._register_struct(d)
            elif isinstance(d, ast.EnumDecl) and d.is_definition:
                self._register_enum(d)
        # `StructType` references inside a containing decl (e.g.,
        # `struct point origin;` at top level) carry an empty members
        # list; the layout is owned by `_structs[name]`. Inline struct
        # definitions inside another decl aren't yet supported.

        # Register top-level VarDecls (non-function-type) as globals. The
        # resolved type fills in inferred array sizes the same way local
        # var-decls do.
        self._globals = {}
        self._global_inits = {}
        for d in unit.declarations:
            if isinstance(d, ast.VarDecl) and not isinstance(d.var_type, ast.FunctionType):
                resolved = self._resolved_var_type(d)
                self._check_supported_type(resolved, d.name)
                self._globals[d.name] = resolved
                if d.init is not None:
                    self._global_inits[d.name] = d.init

        lines: list[str] = []
        lines += self._header(extern_list)
        lines += self._start_stub()
        function_blocks: list[list[str]] = []
        for fn in functions:
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
        ):
            return []
        out = ["        section .data"]
        # Stable order for reproducible output.
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
        for name in initialized_globals:
            ty = self._globals[name]
            init = self._global_inits[name]
            out.append(f"_{name}:")
            out += self._emit_global_init(ty, init, name)
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
        return isinstance(t, ast.BasicType) and t.name in ("float", "double")

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
            value = self._const_eval(init, name)
            directive = self._DATA_DIRECTIVE[self._size_of(ty)]
            return [f"        {directive}      {value}"]
        if isinstance(ty, ast.PointerType):
            # Allow `int *p = 0` (null pointer). Pointer-to-other-global
            # init is rejected pending a clearer use case.
            value = self._const_eval(init, name)
            return [f"        dd      {value}"]
        if isinstance(ty, ast.ArrayType):
            return self._emit_global_array_init(ty, init, name)
        raise CodegenError(
            f"global `{name}`: unsupported type {type(ty).__name__}"
        )

    def _emit_global_array_init(
        self,
        arr_ty: ast.ArrayType,
        init: ast.Expression,
        name: str,
    ) -> list[str]:
        elem_type = arr_ty.base_type
        elem_size = self._size_of(elem_type)
        directive = self._DATA_DIRECTIVE[elem_size]
        length = arr_ty.size.value

        if isinstance(init, ast.StringLiteral):
            if not (
                isinstance(elem_type, ast.BasicType) and elem_type.name == "char"
            ):
                raise CodegenError(
                    f"global `{name}`: string init requires a char array"
                )
            chars = list(init.value.encode()) + [0]
            if len(chars) > length:
                raise CodegenError(
                    f"global `{name}`: string init exceeds array size {length}"
                )
            # Render the printable run + explicit nulls + zero-pad in one db.
            parts = [self._render_string(init.value), "0"]
            parts.extend(["0"] * (length - len(chars)))
            return [f"        db      {', '.join(parts)}"]

        if isinstance(init, ast.InitializerList):
            if len(init.values) > length:
                raise CodegenError(
                    f"global `{name}`: too many initializers "
                    f"(got {len(init.values)}, array size {length})"
                )
            values = [str(self._const_eval(v, name)) for v in init.values]
            values.extend(["0"] * (length - len(values)))
            return [f"        {directive}      {', '.join(values)}"]

        raise CodegenError(
            f"global `{name}`: unsupported array initializer "
            f"({type(init).__name__})"
        )

    # NASM directives keyed by element size in bytes.
    _DATA_DIRECTIVE = {1: "db", 2: "dw", 4: "dd"}

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

    def _intern_string(self, value: str) -> str:
        if value in self._strings:
            return self._strings[value]
        label = f"_uc386_str{len(self._strings)}"
        self._strings[value] = label
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
        # Parameters live above EBP at cdecl offsets; the first sits at
        # [ebp + 8], and each subsequent param is offset by its predecessor's
        # padded size (4 for scalars / pointers, sizeof(struct) rounded
        # up to 4 for structs passed by value).
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
            if isinstance(param.param_type, ast.StructType):
                size = self._size_of(param.param_type)
                disp += (size + 3) & ~3
            else:
                disp += 4
        # First pass: allocate every local up front so the prologue knows
        # the frame size before we emit body code.
        self._collect_locals(fn.body, ctx)
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
        """Pre-allocate a frame slot for every struct-returning Call in `node`.

        We allocate one buffer per Call expression (keyed by `id(call)`)
        so two struct-returning calls in the same expression — e.g.
        `make(1).x + make(2).x` — get distinct buffers and don't clobber
        each other. Some call sites later turn out to have a known
        destination (var init, struct assignment, return chain) and
        won't read from the temp, but allocating them anyway keeps this
        pass context-free.
        """
        for sub in self._walk_ast(node):
            if isinstance(sub, ast.Call) and self._is_struct_returning_call(sub, None):
                ret_ty = self._func_return_types[self._call_target(sub)]
                size = (self._size_of(ret_ty) + 3) & ~3
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
            var_type = self._resolved_var_type(node)
            self._check_supported_type(var_type, node.name)
            # Round slot size up to a 4-byte boundary so a `char`-sized slot
            # doesn't push subsequent int slots off-alignment. Arrays whose
            # payload isn't a multiple of 4 (e.g. `char arr[5]`) get padded
            # the same way.
            raw = self._size_of(var_type)
            slot = (raw + 3) & ~3
            ctx.alloc_local(node.name, var_type, slot)
            return
        if isinstance(node, ast.CompoundStmt):
            for item in node.items:
                self._collect_locals(item, ctx)
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
            if node.init is not None:
                self._collect_locals(node.init, ctx)
            self._collect_locals(node.body, ctx)
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
        # Statements with no nested locals: ExpressionStmt, ReturnStmt,
        # BreakStmt, ContinueStmt, etc.

    # Scalar BasicType names that have first-class slot support. `long` and
    # `long long` are *known* sizes (so pointer-arithmetic scaling works
    # transparently for `long *` etc.) but full slot codegen waits on a
    # 64-bit value-tracking pass.
    _SLOT_BASIC_NAMES = frozenset(
        {"bool", "char", "short", "int", "float", "double"}
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
            # Enums are int-sized; we don't validate that the named enum
            # exists because anonymous enums (no name) and uses-before-
            # definition both occur naturally.
            return
        if isinstance(t, ast.ArrayType):
            if t.size is not None and not isinstance(t.size, ast.IntLiteral):
                raise CodegenError(
                    f"`{name}`: array size must be an integer literal"
                )
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

    @staticmethod
    def _resolved_var_type(decl: ast.VarDecl) -> ast.TypeNode:
        """Return the var's type with any unsized-array size filled in.

        `int arr[] = {1, 2, 3}` and `char s[] = "hi"` both leave the
        ArrayType's `size` as None; the size is implied by the initializer.
        Resolve it here so allocation and codegen can treat the slot as a
        fully-sized array thereafter.
        """
        t = decl.var_type
        if not isinstance(t, ast.ArrayType) or t.size is not None:
            return t
        if isinstance(decl.init, ast.InitializerList):
            n = len(decl.init.values)
        elif isinstance(decl.init, ast.StringLiteral):
            # +1 reserves a slot for the trailing null byte.
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
        """
        if t.name and t.name in self._structs:
            return t.name
        if t.members:
            key = t.name or f"__anon_struct_{id(t)}"
            if key not in self._structs:
                # Synthesize a minimal StructDecl-shaped object so we can
                # reuse `_register_struct`. SimpleNamespace is enough; the
                # method only reads name / members / is_union.
                from types import SimpleNamespace
                self._register_struct(SimpleNamespace(
                    name=key, members=t.members, is_union=t.is_union,
                ))
            return key
        if t.name:
            raise CodegenError(
                f"unknown struct `{t.name}` — define it before use"
            )
        raise CodegenError("anonymous struct without inline members")

    def _register_enum(self, decl: ast.EnumDecl) -> None:
        """Compute and record each `EnumValue`'s integer constant.

        Per C, an `EnumValue(name, value=None)` takes the previous
        constant + 1 (starting at 0 for the first). An explicit
        `value=IntLiteral(n)` sets the cursor; subsequent implicit
        values continue from there.
        """
        cursor = 0
        for ev in decl.values:
            if ev.value is not None:
                cursor = self._const_eval(ev.value, f"enum {decl.name or '?'}.{ev.name}")
            if ev.name in self._enum_constants:
                raise CodegenError(
                    f"duplicate enum constant `{ev.name}`"
                )
            self._enum_constants[ev.name] = cursor
            cursor += 1

    def _register_struct(self, decl: ast.StructDecl) -> None:
        """Compute member offsets and total size for a struct definition.

        Member offsets are aligned to the member's natural alignment
        (power-of-two sizes for char/short/int/pointer). The total struct
        size is rounded up to the largest member alignment so arrays of
        the struct stay properly aligned.
        """
        if decl.name in self._structs:
            # Idempotent: tolerate multiple `struct foo { ... }` definitions
            # if they happen to repeat. Conflicting definitions aren't
            # detected here.
            return
        # Validate every member up front. Unions and structs share the
        # same per-member checks; only the layout step differs.
        members: list[tuple[str, ast.TypeNode, int]] = []
        bitfields: dict[str, tuple[int, int]] = {}
        max_align = 1
        for m in decl.members:
            if m.name is None:
                raise CodegenError(
                    f"`{decl.name}`: anonymous members not supported"
                )
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
                members.append((m.name, m.member_type, 0))
                size = self._size_of(m.member_type)
                if size > total:
                    total = size
            total = (total + max_align - 1) & ~(max_align - 1)
            self._structs[decl.name] = members
            self._struct_sizes[decl.name] = total
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
                members.append((m.name, m.member_type, offset))
                unit_offset = offset + self._size_of(m.member_type)
        total = unit_offset + (4 if unit_used > 0 else 0)
        total = (total + max_align - 1) & ~(max_align - 1)
        self._structs[decl.name] = members
        self._struct_sizes[decl.name] = total
        if bitfields:
            self._struct_bitfields[decl.name] = bitfields

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
        # `is_signed=None` is the language default — signed for char/short/int.
        return isinstance(t, ast.BasicType) and t.is_signed is False

    # ---- struct member lowering ---------------------------------------

    def _member_address(self, expr: ast.Member, ctx: _FuncCtx) -> list[str]:
        """Compute the address of `expr` (`.` or `->` member) into eax."""
        if expr.is_arrow:
            obj_ty = self._type_of(expr.obj, ctx)
            if not (
                isinstance(obj_ty, ast.PointerType)
                and isinstance(obj_ty.base_type, ast.StructType)
            ):
                raise CodegenError(
                    f"`->` requires a pointer to struct "
                    f"(got {type(obj_ty).__name__})"
                )
            struct_name = self._resolve_struct_name(obj_ty.base_type)
            # eax = the pointer's value, i.e. the struct's address.
            out = self._eval_expr_to_eax(expr.obj, ctx)
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
        if isinstance(obj_ty, ast.PointerType):
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
        bit_offset, bit_width, _ = bf
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
        # Result of the assignment expression is the new value (rhs as
        # narrowed to the field's width). Recompute it cheaply: edx
        # already has the positioned form, but we want the unpositioned
        # value — for simplicity, leave EAX = the rhs as-positioned in
        # EDX. This is a fudge; chained `(f.x = v).y` isn't really
        # meaningful for bit-fields.
        out.append("        mov     eax, edx")
        if bit_offset > 0:
            out.append(f"        shr     eax, {bit_offset}")
        return out

    # ---- identifier resolution (local vs global) ----------------------

    def _identifier_type(self, name: str, ctx: _FuncCtx) -> ast.TypeNode:
        if name in ctx.slots:
            return ctx.lookup_type(name)
        if name in self._globals:
            return self._globals[name]
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

    def _identifier_addr_text(self, name: str, ctx: _FuncCtx) -> str:
        """Return the `[...]` operand text used for in-place memory ops.

        Locals render as `[ebp - N]`; globals render as `[_name]`. Used by
        `_inc_dec` (for `inc/dec dword [...]`-style instructions) where
        `_load_to_eax` / `_store_from_eax` would be overkill.
        """
        if name in ctx.slots:
            return _ebp_addr(ctx.lookup(name))
        if name in self._globals:
            return f"[_{name}]"
        raise CodegenError(f"unknown identifier `{name}`")

    def _identifier_load(self, name: str, ctx: _FuncCtx) -> list[str]:
        """Lines that produce the value (or, for arrays/functions, the address) of `name` in eax."""
        if name in ctx.slots:
            ty = ctx.lookup_type(name)
            disp = ctx.lookup(name)
            if isinstance(ty, ast.ArrayType):
                # Array decay: yield the slot's address, not its bytes.
                return [f"        lea     eax, {_ebp_addr(disp)}"]
            return self._load_to_eax(_ebp_addr(disp), ty)
        if name in self._globals:
            ty = self._globals[name]
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
        if name in ctx.slots:
            return [f"        lea     eax, {_ebp_addr(ctx.lookup(name))}"]
        if name in self._globals:
            return [f"        mov     eax, _{name}"]
        if name in self._func_return_types:
            # `&fn` and `fn` produce the same address, just like for arrays.
            return [f"        mov     eax, _{name}"]
        raise CodegenError(f"unknown identifier `{name}`")

    def _identifier_store(self, name: str, ctx: _FuncCtx) -> list[str]:
        """Lines that store eax to the slot for `name`, with width per type."""
        ty = self._identifier_type(name, ctx)
        if name in ctx.slots:
            return self._store_from_eax(_ebp_addr(ctx.lookup(name)), ty)
        return self._store_from_eax(f"[_{name}]", ty)

    def _load_to_eax(self, addr: str, ty: ast.TypeNode) -> list[str]:
        """Lines that load a value of type `ty` from `addr` into EAX.

        Sub-word loads sign- or zero-extend (per signedness) so callers can
        treat EAX uniformly as a 32-bit working value, matching C's integer
        promotion rules.
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
        raise CodegenError(f"can't load size-{size} value into eax")

    def _cast(self, expr: ast.Cast, ctx: _FuncCtx) -> list[str]:
        """Evaluate `expr.expr` then narrow/extend EAX to match `target_type`.

        i386 makes most casts cheap: pointer ↔ pointer, int ↔ pointer, and
        int ↔ long are all no-ops (every 32-bit value already lives in EAX).
        Narrowing to char/short truncates through the low half of EAX
        (`al`/`ax`) and re-extends per the target's signedness, so a
        subsequent use of the value sees the right C semantics.
        """
        out = self._eval_expr_to_eax(expr.expr, ctx)
        target = expr.target_type
        if isinstance(target, ast.PointerType):
            return out
        if isinstance(target, ast.BasicType):
            size = self._size_of(target)
            if size == 4:
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
        """Lines that store EAX (treated as `ty`) to `addr`.

        For sub-word types we narrow via the low half of EAX (`ax`/`al`),
        leaving the higher bytes of the slot untouched — the load helper
        above only reads the meaningful bytes anyway.
        """
        size = self._size_of(ty)
        if size == 4:
            return [f"        mov     {addr}, eax"]
        if size == 2:
            return [f"        mov     word {addr}, ax"]
        if size == 1:
            return [f"        mov     byte {addr}, al"]
        raise CodegenError(f"can't store size-{size} value from eax")

    # ---- statements -----------------------------------------------------

    def _compound(self, block: ast.CompoundStmt, ctx: _FuncCtx) -> list[str]:
        out: list[str] = []
        for item in block.items:
            out += self._item(item, ctx)
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
        raise CodegenError(
            f"{type(item).__name__} not implemented yet"
        )

    def _if(self, stmt: ast.IfStmt, ctx: _FuncCtx) -> list[str]:
        else_label = ctx.label("else")
        end_label = ctx.label("endif")
        out = self._eval_expr_to_eax(stmt.condition, ctx)
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
            out += self._eval_expr_to_eax(stmt.condition, ctx)
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
            out += self._eval_expr_to_eax(stmt.condition, ctx)
            out.append("        test    eax, eax")
            out.append(f"        jnz     {top}")
            out.append(f"{end}:")
        finally:
            ctx.break_targets.pop()
            ctx.continue_targets.pop()
        return out

    def _for(self, stmt: ast.ForStmt, ctx: _FuncCtx) -> list[str]:
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
            out += self._eval_expr_to_eax(stmt.condition, ctx)
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

    def _switch(self, stmt: ast.SwitchStmt, ctx: _FuncCtx) -> list[str]:
        """Lower `switch (expr) { case V: ...; default: ...; }`.

        The expression evaluates once into EAX. We flatten the body into
        a sequence of label declarations and ordinary statements so that
        chained `case 1: case 2: stmt;` (which the parser nests as
        `CaseStmt(1, CaseStmt(2, stmt))`) and a `case`'s nested labeled
        statement are all handled uniformly. Then we emit a dispatch
        ladder of `cmp eax, V; je .case_V` followed by a tail jump to
        the default (or end), then walk the flattened entries to
        materialize each label inline above the statements that follow.

        `break` resolves via `ctx.break_targets`; `continue` deliberately
        does NOT push to `continue_targets` here, so a `continue` inside
        the switch escapes to the enclosing loop (as C requires).
        """
        if not isinstance(stmt.body, ast.CompoundStmt):
            raise CodegenError("switch body must be a compound statement")
        end_label = ctx.label("switch_end")

        # Flatten: each entry is one of
        #   ("case", value, label)        — case label declaration
        #   ("default", None, label)      — default label declaration
        #   ("body", None, stmt)          — ordinary statement
        # Adjacent `case 1: case 2: stmt;` produces case/case/body in that order.
        entries: list[tuple[str, int | None, object]] = []
        default_label: str | None = None

        def expand(node):
            nonlocal default_label
            while isinstance(node, ast.CaseStmt):
                if node.value is None:
                    if default_label is not None:
                        raise CodegenError(
                            "multiple `default` labels in switch"
                        )
                    default_label = ctx.label("default")
                    entries.append(("default", None, default_label))
                else:
                    value = self._const_eval(node.value, "case")
                    entries.append(("case", value, ctx.label("case")))
                node = node.stmt
            entries.append(("body", None, node))

        for item in stmt.body.items:
            if isinstance(item, ast.CaseStmt):
                expand(item)
            else:
                entries.append(("body", None, item))

        # Eval the controlling expression once.
        out = self._eval_expr_to_eax(stmt.expr, ctx)

        # Dispatch ladder. The tail `jmp` catches any value not matched
        # by a case; if there's no `default`, fall through to end_label.
        for kind, value, target in entries:
            if kind == "case":
                out.append(f"        cmp     eax, {value}")
                out.append(f"        je      {target}")
        out.append(f"        jmp     {default_label or end_label}")

        # Body emission. Case/default entries materialize labels; body
        # entries lower their statement.
        ctx.break_targets.append(end_label)
        try:
            for kind, _, target in entries:
                if kind in ("case", "default"):
                    out.append(f"{target}:")
                else:
                    out += self._item(target, ctx)
        finally:
            ctx.break_targets.pop()

        out.append(f"{end_label}:")
        return out

    def _expr_stmt(self, stmt: ast.ExpressionStmt, ctx: _FuncCtx) -> list[str]:
        if stmt.expr is None:
            return []
        # Result is discarded; we still evaluate for side effects (assignment).
        return self._eval_expr_to_eax(stmt.expr, ctx)

    def _var_init(self, decl: ast.VarDecl, ctx: _FuncCtx) -> list[str]:
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
            return self._struct_init(var_type, decl.init, disp, ctx, decl.name)
        if self._is_float_type(var_type):
            # Float locals get their init value via st0 + fstp.
            return self._eval_float_to_st0(decl.init, ctx) + self._store_st0_to(
                _ebp_addr(disp), var_type
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
        length = arr_type.size.value

        if isinstance(init, ast.StringLiteral):
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
            # Walk values in source order, allowing `[N] = expr` to jump
            # the cursor. After all source values, any unfilled slots
            # get zero-filled. This handles pure-positional, pure-
            # designated, and mixed forms uniformly.
            out = []
            filled: set[int] = set()
            cursor = 0
            for value_expr in init.values:
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
        members = self._structs[self._resolve_struct_name(struct_ty)]
        member_index = {m_name: i for i, (m_name, _, _) in enumerate(members)}
        # Walk source values in order, tracking the implicit cursor. A
        # `.field = expr` sets the cursor to that member's index; the
        # next un-designated value continues from cursor + 1. After the
        # walk, any unfilled members get zero-filled.
        out: list[str] = []
        filled: set[int] = set()
        cursor = 0
        for value in init.values:
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
                cursor += 1
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
            else:
                out += self._eval_expr_to_eax(actual, ctx)
                out += self._store_from_eax(_ebp_addr(m_disp), m_ty)
        # Zero-fill any unfilled members in declaration order.
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
        # Struct-returning functions copy the value into the caller-provided
        # buffer (the hidden `__retptr__` first arg) rather than dropping
        # the value into EAX. We forward the retptr in EAX as the return
        # value so chained struct calls don't need a temp.
        if "__retptr__" in ctx.slots:
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
        return self._eval_expr_to_eax(stmt.value, ctx) + [
            "        jmp     .epilogue",
        ]

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
            return ast.BasicType(name="int")
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
            # `+` and `-` propagate float through; `~ !` always produce int.
            if expr.op in ("+", "-"):
                inner = self._type_of(expr.operand, ctx)
                if self._is_float_type(inner):
                    return inner
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
                if not (
                    isinstance(obj_ty, ast.PointerType)
                    and isinstance(obj_ty.base_type, ast.StructType)
                ):
                    raise CodegenError(
                        f"`->` requires a pointer to struct "
                        f"(got {type(obj_ty).__name__})"
                    )
                struct_name = self._resolve_struct_name(obj_ty.base_type)
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
            return ast.BasicType(name="int")
        if isinstance(expr, ast.TernaryOp):
            # Both arms should agree in well-formed C; pick the true arm.
            return self._type_of(expr.true_expr, ctx)
        if isinstance(expr, (ast.SizeofExpr, ast.SizeofType)):
            # `sizeof` returns size_t; treat it as int for our flat-32 ABI.
            return ast.BasicType(name="int")
        if isinstance(expr, ast.Cast):
            return expr.target_type
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Identifier):
                rt = self._func_return_types.get(expr.func.name)
                if rt is not None:
                    return rt
            return ast.BasicType(name="int")
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
        raise CodegenError(
            f"expression {type(expr).__name__} not implemented yet"
        )

    # ---- float (x87) lowering ------------------------------------------

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
            # Promote int → float via stack scratch.
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
            raise CodegenError(
                f"unary `{expr.op}` not supported on float operand"
            )
        if isinstance(expr, ast.BinaryOp):
            return self._float_binop(expr, ctx)
        if isinstance(expr, ast.Cast):
            return self._float_cast(expr, ctx)
        if isinstance(expr, ast.TernaryOp):
            return self._float_ternary(expr, ctx)
        if isinstance(expr, ast.Member):
            return self._float_member_load(expr, ctx)
        if isinstance(expr, ast.Index):
            return self._float_index_load(expr, ctx)
        raise CodegenError(
            f"float expression {type(expr).__name__} not implemented yet"
        )

    def _float_identifier_load(self, name: str, ctx: _FuncCtx) -> list[str]:
        """`fld` a float-typed Identifier (local, param, or global)."""
        ty = self._identifier_type(name, ctx)
        size = self._size_of(ty)
        width = "dword" if size == 4 else "qword"
        if name in ctx.slots:
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
        out = self._eval_expr_to_eax(expr.condition, ctx)
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
        if name in ctx.slots:
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
        out = self._eval_expr_to_eax(expr.condition, ctx)
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

        # Direct call: callee names a function declared in this unit
        # (defined or extern). Emit a `call _name` linker reference.
        if (
            isinstance(callee, ast.Identifier)
            and callee.name in self._func_return_types
        ):
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
        out: list[str] = []
        total_arg_bytes = 0
        for arg in reversed(args):
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
        raise CodegenError(
            f"`&` operand must be an identifier, `arr[i]`, or `s.m` "
            f"(got {type(expr.operand).__name__})"
        )

    def _inc_dec(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError(
                f"`{expr.op}` operand must be an identifier "
                f"(got {type(expr.operand).__name__})"
            )
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
            width = {1: "byte", 2: "word", 4: "dword"}[size]
            instr = "inc" if expr.op == "++" else "dec"
            bump = [f"        {instr}     {width} {addr}"]
        load = self._load_to_eax(addr, ty)
        if expr.is_prefix:
            # ++x: bump in place, then load the new value into EAX.
            return bump + load
        # x++: load old value into EAX, then bump in place. EAX is the result.
        return load + bump

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
            return out + [
                "        cdq",
                "        idiv    ecx",
            ]
        if expr.op == "%":
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
            # Signed int → arithmetic shift. Unsigned will branch here once
            # type info is plumbed through codegen.
            return out + [
                "        sar     eax, cl",
            ]
        if expr.op in self._CMP_SETCC:
            return out + [
                "        cmp     eax, ecx",
                f"        {self._CMP_SETCC[expr.op]}    al",
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
        out = self._eval_expr_to_eax(expr.left, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jz      {false_label}")
        out += self._eval_expr_to_eax(expr.right, ctx)
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
        out = self._eval_expr_to_eax(expr.left, ctx)
        out.append("        test    eax, eax")
        out.append(f"        jnz     {true_label}")
        out += self._eval_expr_to_eax(expr.right, ctx)
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
            return self._struct_copy_assign(expr, target_ty, ctx)

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

        out = addr_lines                                       # eax = lvalue address
        out.append("        push    eax")                       # save addr for the store
        out += self._load_to_eax("[eax]", target_ty)            # eax = current value
        out.append("        push    eax")                       # left operand on stack
        out += self._eval_expr_to_eax(expr.right, ctx)          # eax = rhs
        out += self._apply_binop_post_eval(op, target_ty, rhs_ty)  # eax = new value
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
