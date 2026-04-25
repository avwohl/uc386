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
- Control flow: `if`/`else`, `while`, `do`/`while`, `for`, `break`, `continue`.
- Function calls — direct calls only (callee must be an identifier).
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
- String literals: interned per-translation-unit, emitted as
  null-terminated bytes in `.data` with labels like `_uc386_strN`.
- `extern` declarations (FunctionDecls without bodies) emit NASM
  `extern _name` at the top so calls can be resolved by the linker.

Anything else raises CodegenError.
"""

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
        self.loops: list[tuple[str, str]] = []

    def alloc_local(self, name: str, ty: "ast.TypeNode", size: int = 4) -> int:
        if name in self.slots:
            raise CodegenError(f"redeclaration of `{name}`")
        # Each local sits at the next 4-byte slot below EBP.
        self.frame_size += size
        self.slots[name] = -self.frame_size
        self.types[name] = ty
        return self.slots[name]

    def alloc_param(self, name: str, index: int, ty: "ast.TypeNode", size: int = 4) -> int:
        # cdecl: caller pushed args right-to-left, then `call` pushed the
        # return address, then we pushed EBP. So the first arg lives at
        # [ebp + 8], the second at [ebp + 12], etc.
        if name in self.slots:
            raise CodegenError(f"duplicate parameter `{name}`")
        self.slots[name] = 8 + index * size
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

        lines: list[str] = []
        lines += self._header(extern_list)
        lines += self._start_stub()
        function_blocks: list[list[str]] = []
        for fn in functions:
            function_blocks.append(self._function(fn))
        for block in function_blocks:
            lines.append("")
            lines += block
        if self._strings:
            lines.append("")
            lines += self._data_section()
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
        out = ["        section .data"]
        # Stable order for reproducible output.
        for value, label in sorted(self._strings.items(), key=lambda kv: kv[1]):
            out.append(f"{label}:")
            out.append(f"        db      {self._render_string(value)}, 0")
        return out

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
        # Parameters live above EBP at fixed cdecl offsets; register them
        # before the locals so a body that references a parameter lowers
        # correctly. `int` is the only param type we handle today.
        for i, param in enumerate(fn.params):
            if param.name is None:
                continue
            self._check_supported_type(param.param_type, param.name)
            ctx.alloc_param(param.name, i, param.param_type)
        # First pass: allocate every local up front so the prologue knows
        # the frame size before we emit body code.
        self._collect_locals(fn.body, ctx)

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
        # Statements with no nested locals: ExpressionStmt, ReturnStmt,
        # BreakStmt, ContinueStmt, etc.

    # Scalar BasicType names that have first-class slot support. `long` and
    # `long long` are *known* sizes (so pointer-arithmetic scaling works
    # transparently for `long *` etc.) but full slot codegen waits on a
    # 64-bit value-tracking pass.
    _SLOT_BASIC_NAMES = frozenset({"char", "short", "int"})

    @classmethod
    def _check_supported_type(cls, t: ast.TypeNode, name: str) -> None:
        # Pointers, and arrays / scalars of supported base types. Slot sizes
        # are rounded up to 4 in `_collect_locals` so adjacent ints stay
        # 4-aligned. Unsized arrays (`int a[]` without an init) are caught
        # by `_resolved_var_type` before they reach this check.
        if isinstance(t, ast.PointerType):
            return
        if isinstance(t, ast.BasicType) and t.name in cls._SLOT_BASIC_NAMES:
            return
        if isinstance(t, ast.ArrayType):
            if t.size is not None and not isinstance(t.size, ast.IntLiteral):
                raise CodegenError(
                    f"`{name}`: array size must be an integer literal"
                )
            cls._check_supported_type(t.base_type, name)
            return
        raise CodegenError(
            f"`{name}`: only `int`/`short`/`char`, pointer, and array "
            f"types are supported (got {type(t).__name__})"
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
        "char": 1,
        "short": 2,
        "int": 4,
        "long": 4,          # i386: long is 32-bit
        "long long": 8,
        "void": 1,          # GCC convention; standard C disallows arithmetic on void*
    }

    @classmethod
    def _size_of(cls, t: ast.TypeNode) -> int:
        if isinstance(t, ast.PointerType):
            return 4
        if isinstance(t, ast.BasicType):
            try:
                return cls._BASIC_SIZES[t.name]
            except KeyError:
                raise CodegenError(f"sizeof({t.name}) not known")
        if isinstance(t, ast.ArrayType):
            if not isinstance(t.size, ast.IntLiteral):
                raise CodegenError("sizeof(array): size must be an integer literal")
            return t.size.value * cls._size_of(t.base_type)
        raise CodegenError(f"sizeof not supported for {type(t).__name__}")

    @staticmethod
    def _is_unsigned(t: ast.TypeNode) -> bool:
        # `is_signed=None` is the language default — signed for char/short/int.
        return isinstance(t, ast.BasicType) and t.is_signed is False

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
        if isinstance(item, ast.BreakStmt):
            if not ctx.loops:
                raise CodegenError("`break` outside of a loop")
            return [f"        jmp     {ctx.loops[-1][1]}"]
        if isinstance(item, ast.ContinueStmt):
            if not ctx.loops:
                raise CodegenError("`continue` outside of a loop")
            return [f"        jmp     {ctx.loops[-1][0]}"]
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
        ctx.loops.append((top, end))
        try:
            out = [f"{top}:"]
            out += self._eval_expr_to_eax(stmt.condition, ctx)
            out.append("        test    eax, eax")
            out.append(f"        jz      {end}")
            out += self._item(stmt.body, ctx)
            out.append(f"        jmp     {top}")
            out.append(f"{end}:")
        finally:
            ctx.loops.pop()
        return out

    def _do_while(self, stmt: ast.DoWhileStmt, ctx: _FuncCtx) -> list[str]:
        top = ctx.label("do_top")
        cont = ctx.label("do_cont")
        end = ctx.label("do_end")
        # `continue` jumps to the condition test, not the top of the body.
        ctx.loops.append((cont, end))
        try:
            out = [f"{top}:"]
            out += self._item(stmt.body, ctx)
            out.append(f"{cont}:")
            out += self._eval_expr_to_eax(stmt.condition, ctx)
            out.append("        test    eax, eax")
            out.append(f"        jnz     {top}")
            out.append(f"{end}:")
        finally:
            ctx.loops.pop()
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
        ctx.loops.append((step, end))
        try:
            out += self._item(stmt.body, ctx)
        finally:
            ctx.loops.pop()
        out.append(f"{step}:")
        if stmt.update is not None:
            out += self._eval_expr_to_eax(stmt.update, ctx)
        out.append(f"        jmp     {top}")
        out.append(f"{end}:")
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
            if len(init.values) > length:
                raise CodegenError(
                    f"`{name}`: too many initializers (got "
                    f"{len(init.values)}, array size {length})"
                )
            out = []
            for i, value_expr in enumerate(init.values):
                addr = _ebp_addr(base_disp + i * elem_size)
                out += self._eval_expr_to_eax(value_expr, ctx)
                out += self._store_from_eax(addr, elem_type)
            zero_width = self._ZERO_WIDTHS[elem_size]
            for i in range(len(init.values), length):
                addr = _ebp_addr(base_disp + i * elem_size)
                out.append(f"        mov     {zero_width} {addr}, 0")
            return out

        raise CodegenError(
            f"`{name}`: unsupported array initializer "
            f"({type(init).__name__})"
        )

    def _return(self, stmt: ast.ReturnStmt, ctx: _FuncCtx) -> list[str]:
        if stmt.value is None:
            return [
                "        xor     eax, eax",
                "        jmp     .epilogue",
            ]
        return self._eval_expr_to_eax(stmt.value, ctx) + [
            "        jmp     .epilogue",
        ]

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
            return ctx.lookup_type(expr.name)
        if isinstance(expr, ast.IntLiteral):
            return ast.BasicType(name="int")
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
            # `+ - ~ !` produce int.
            return ast.BasicType(name="int")
        if isinstance(expr, ast.Index):
            arr_type = self._type_of(expr.array, ctx)
            if not self._is_pointer_like(arr_type):
                raise CodegenError(
                    f"index target must be array or pointer "
                    f"(got {type(arr_type).__name__})"
                )
            return arr_type.base_type
        if isinstance(expr, ast.BinaryOp):
            if expr.op == "=":
                return self._type_of(expr.left, ctx)
            if expr.op in self._COMPOUND_OPS:
                return self._type_of(expr.left, ctx)
            if expr.op in self._INT_RESULT_BINOPS:
                return ast.BasicType(name="int")
            # `+` and `-`: pointer ± int → pointer; pointer - pointer → int.
            # Arrays count as pointer-like via decay.
            lt = self._type_of(expr.left, ctx)
            rt = self._type_of(expr.right, ctx)
            l_ptr = self._is_pointer_like(lt)
            r_ptr = self._is_pointer_like(rt)
            if l_ptr and r_ptr:
                return ast.BasicType(name="int")
            if l_ptr:
                return lt
            if r_ptr:
                return rt
            return ast.BasicType(name="int")
        if isinstance(expr, ast.TernaryOp):
            # Both arms should agree in well-formed C; pick the true arm.
            return self._type_of(expr.true_expr, ctx)
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Identifier):
                rt = self._func_return_types.get(expr.func.name)
                if rt is not None:
                    return rt
            return ast.BasicType(name="int")
        return ast.BasicType(name="int")

    def _eval_expr_to_eax(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        if isinstance(expr, ast.IntLiteral):
            return [f"        mov     eax, {expr.value}"]
        if isinstance(expr, ast.StringLiteral):
            label = self._intern_string(expr.value)
            return [f"        mov     eax, {label}"]
        if isinstance(expr, ast.Identifier):
            disp = ctx.lookup(expr.name)
            ty = ctx.lookup_type(expr.name)
            # Array decay: an array name in expression context yields the
            # address of its first element, not the bytes at that slot.
            if isinstance(ty, ast.ArrayType):
                return [f"        lea     eax, {_ebp_addr(disp)}"]
            return self._load_to_eax(_ebp_addr(disp), ty)
        if isinstance(expr, ast.Index):
            return self._index_load(expr, ctx)
        if isinstance(expr, ast.UnaryOp):
            return self._unary(expr, ctx)
        if isinstance(expr, ast.BinaryOp):
            return self._binary(expr, ctx)
        if isinstance(expr, ast.Call):
            return self._call(expr, ctx)
        if isinstance(expr, ast.TernaryOp):
            return self._ternary(expr, ctx)
        raise CodegenError(
            f"expression {type(expr).__name__} not implemented yet"
        )

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
        elem_ty = self._type_of(expr, ctx)
        return self._index_address(expr, ctx) + self._load_to_eax("[eax]", elem_ty)

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
        if not isinstance(expr.func, ast.Identifier):
            raise CodegenError(
                f"only direct calls are supported "
                f"(got {type(expr.func).__name__})"
            )
        # cdecl: push args right-to-left so the first arg ends up at the
        # lowest address (= [ebp+8] in the callee). C leaves inter-arg
        # evaluation order unspecified, so right-to-left is fine.
        out: list[str] = []
        for arg in reversed(expr.args):
            out += self._eval_expr_to_eax(arg, ctx)
            out.append("        push    eax")
        out.append(f"        call    _{expr.func.name}")
        if expr.args:
            out.append(f"        add     esp, {4 * len(expr.args)}")
        # Return value is in EAX.
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
        # `&identifier` and `&arr[i]` are supported. `&*p` (a no-op) and
        # `&(struct.field)` will land when those constructs do.
        if isinstance(expr.operand, ast.Identifier):
            disp = ctx.lookup(expr.operand.name)
            return [f"        lea     eax, {_ebp_addr(disp)}"]
        if isinstance(expr.operand, ast.Index):
            # &arr[i] — same address arithmetic as a load, just no final deref.
            return self._index_address(expr.operand, ctx)
        raise CodegenError(
            f"`&` operand must be an identifier or `arr[i]` "
            f"(got {type(expr.operand).__name__})"
        )

    def _inc_dec(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError(
                f"`{expr.op}` operand must be an identifier "
                f"(got {type(expr.operand).__name__})"
            )
        ty = ctx.lookup_type(expr.operand.name)
        # Array names aren't lvalues — `++arr` is a C error, not "advance the
        # array pointer" (that would only make sense for a pointer variable).
        if isinstance(ty, ast.ArrayType):
            raise CodegenError(
                f"cannot {expr.op} array `{expr.operand.name}`"
            )
        disp = ctx.lookup(expr.operand.name)
        addr = _ebp_addr(disp)
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
        # `x = rhs` — direct slot store. Array names aren't lvalues in C.
        if isinstance(expr.left, ast.Identifier):
            ty = ctx.lookup_type(expr.left.name)
            if isinstance(ty, ast.ArrayType):
                raise CodegenError(
                    f"cannot assign to array `{expr.left.name}`"
                )
            disp = ctx.lookup(expr.left.name)
            return self._eval_expr_to_eax(expr.right, ctx) + self._store_from_eax(
                _ebp_addr(disp), ty
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
        raise CodegenError(
            f"assignment target must be an identifier, `*ptr`, or `arr[i]` "
            f"(got {type(expr.left).__name__})"
        )

    def _compound_assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        # `x op= rhs` is `x = x op rhs`. For Identifier lvalues, evaluating
        # the lvalue is side-effect-free, so it's safe to compute it twice
        # via a synthesized BinaryOp. Pointer/array lvalues will need a
        # different lowering when those land.
        if not isinstance(expr.left, ast.Identifier):
            raise CodegenError(
                f"compound assignment target must be an identifier "
                f"(got {type(expr.left).__name__})"
            )
        inner = ast.BinaryOp(
            op=self._COMPOUND_OPS[expr.op],
            left=expr.left,
            right=expr.right,
        )
        return self._assign(
            ast.BinaryOp(op="=", left=expr.left, right=inner),
            ctx,
        )
