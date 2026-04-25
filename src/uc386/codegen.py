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
- Pointer locals/params (`int *p`), `&x` (Identifier only), `*expr`,
  and store-through-pointer assignment `*p = rhs`. Pointer arithmetic
  not implemented yet — needs type-info plumbing for size scaling.

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
        self.frame_size: int = 0          # bytes reserved by `sub esp, frame_size`
        self._next_label: int = 0
        # Stack of (continue_target, break_target) for the enclosing loops.
        self.loops: list[tuple[str, str]] = []

    def alloc_local(self, name: str, size: int = 4) -> int:
        if name in self.slots:
            raise CodegenError(f"redeclaration of `{name}`")
        # Each local sits at the next 4-byte slot below EBP.
        self.frame_size += size
        self.slots[name] = -self.frame_size
        return self.slots[name]

    def alloc_param(self, name: str, index: int, size: int = 4) -> int:
        # cdecl: caller pushed args right-to-left, then `call` pushed the
        # return address, then we pushed EBP. So the first arg lives at
        # [ebp + 8], the second at [ebp + 12], etc.
        if name in self.slots:
            raise CodegenError(f"duplicate parameter `{name}`")
        self.slots[name] = 8 + index * size
        return self.slots[name]

    def lookup(self, name: str) -> int:
        if name not in self.slots:
            raise CodegenError(f"unknown identifier `{name}`")
        return self.slots[name]

    def label(self, hint: str) -> str:
        self._next_label += 1
        return f".L{self._next_label}_{hint}"


class CodeGenerator:
    """i386/MS-DOS backend."""

    def __init__(self, module_name: str = "main"):
        self.module_name = module_name

    # ---- top level ------------------------------------------------------

    def generate(self, unit: ast.TranslationUnit) -> str:
        functions = [
            d for d in unit.declarations
            if isinstance(d, ast.FunctionDecl) and d.body is not None
        ]
        if not any(fn.name == "main" for fn in functions):
            raise CodegenError("uc386 requires a `main` function definition")

        lines: list[str] = []
        lines += self._header()
        lines += self._start_stub()
        for fn in functions:
            lines.append("")
            lines += self._function(fn)
        return "\n".join(lines) + "\n"

    def _header(self) -> list[str]:
        return [
            f"; uc386 codegen output",
            f"; module: {self.module_name}",
            f"        bits 32",
            f"        section .text",
            f"        global _start",
            "",
        ]

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
            ctx.alloc_param(param.name, i)
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
            self._check_supported_type(node.var_type, node.name)
            ctx.alloc_local(node.name)
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

    @staticmethod
    def _check_supported_type(t: ast.TypeNode, name: str) -> None:
        # Both ints and pointers occupy a single 4-byte slot in flat-32.
        # short/char/long-long codegen comes later.
        if isinstance(t, ast.PointerType):
            return
        if isinstance(t, ast.BasicType) and t.name == "int":
            return
        raise CodegenError(
            f"`{name}`: only `int` and pointer types are supported "
            f"(got {type(t).__name__})"
        )

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
        return self._eval_expr_to_eax(decl.init, ctx) + [
            f"        mov     {_ebp_addr(disp)}, eax",
        ]

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

    def _eval_expr_to_eax(self, expr: ast.Expression, ctx: _FuncCtx) -> list[str]:
        if isinstance(expr, ast.IntLiteral):
            return [f"        mov     eax, {expr.value}"]
        if isinstance(expr, ast.Identifier):
            disp = ctx.lookup(expr.name)
            return [f"        mov     eax, {_ebp_addr(disp)}"]
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
            # the address it holds. Operand may be any expression that
            # produces a 32-bit address.
            return self._eval_expr_to_eax(expr.operand, ctx) + [
                "        mov     eax, [eax]",
            ]
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
        # Only `&identifier` for now — no `&*p` (which is a no-op anyway),
        # no `&arr[i]` (arrays don't exist yet). Pointer-to-pointer falls
        # out naturally because the slot is just 4 bytes.
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError(
                f"`&` operand must be an identifier "
                f"(got {type(expr.operand).__name__})"
            )
        disp = ctx.lookup(expr.operand.name)
        return [f"        lea     eax, {_ebp_addr(disp)}"]

    def _inc_dec(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        if not isinstance(expr.operand, ast.Identifier):
            raise CodegenError(
                f"`{expr.op}` operand must be an identifier "
                f"(got {type(expr.operand).__name__})"
            )
        disp = ctx.lookup(expr.operand.name)
        addr = _ebp_addr(disp)
        instr = "inc" if expr.op == "++" else "dec"
        if expr.is_prefix:
            # ++x: bump in place, then load the new value into EAX.
            return [
                f"        {instr}     dword {addr}",
                f"        mov     eax, {addr}",
            ]
        # x++: load old value into EAX, then bump in place. EAX is the result.
        return [
            f"        mov     eax, {addr}",
            f"        {instr}     dword {addr}",
        ]

    # Map from C operator to a one-line "op eax, ecx" instruction.
    _SIMPLE_BINOPS = {
        "+":  "add     eax, ecx",
        "-":  "sub     eax, ecx",
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
        # `x = rhs` — direct slot store.
        if isinstance(expr.left, ast.Identifier):
            disp = ctx.lookup(expr.left.name)
            return self._eval_expr_to_eax(expr.right, ctx) + [
                f"        mov     {_ebp_addr(disp)}, eax",
            ]
        # `*p = rhs` — store-through-pointer. Evaluate the pointer expr
        # first, save its value, then evaluate rhs into EAX (so the result
        # of the whole assignment expression is rhs, as C requires).
        if isinstance(expr.left, ast.UnaryOp) and expr.left.op == "*":
            out = self._eval_expr_to_eax(expr.left.operand, ctx)
            out.append("        push    eax")
            out += self._eval_expr_to_eax(expr.right, ctx)
            out.append("        pop     ecx")
            out.append("        mov     [ecx], eax")
            return out
        raise CodegenError(
            f"assignment target must be an identifier or `*ptr` "
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
