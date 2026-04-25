"""x86-32 (i386) code generator for MS-DOS targets.

NASM Intel-syntax output. The emitted file assembles with `nasm -fobj`
and links against a DOS/4GW-compatible linker.

NASM was chosen as the assembler target because it's open source, ships
on every modern dev box, supports OMF object output for DOS toolchains
(`-fobj`), and uses Intel syntax that matches the rest of the
Watcom-era ecosystem.

Current scope:
- `int main(void) { ... }` and other functions returning int.
- Integer-literal returns and bare `return;`.
- `int` locals with arbitrary initializer expressions.
- Reading a local in any expression position.
- Assignment to a local (`x = expr;`) as an expression statement.
- Unary `+ - ~ !` and binary `+ - * / % & | ^ << >> == != < > <= >=`.

Anything else raises CodegenError.
"""

from uc_core import ast


class CodegenError(NotImplementedError):
    """Raised when the AST contains a construct codegen can't handle yet."""


class _FuncCtx:
    """Per-function lowering state: local variable layout."""

    def __init__(self) -> None:
        self.locals: dict[str, int] = {}  # name -> positive offset; address is [ebp - offset]
        self.frame_size: int = 0          # bytes reserved by `sub esp, frame_size`

    def alloc(self, name: str, size: int = 4) -> int:
        if name in self.locals:
            raise CodegenError(f"redeclaration of local `{name}`")
        # Each local sits at the next 4-byte slot. For ints (4 bytes) this
        # is also natural alignment.
        self.frame_size += size
        self.locals[name] = self.frame_size
        return self.locals[name]

    def lookup(self, name: str) -> int:
        if name not in self.locals:
            raise CodegenError(f"unknown identifier `{name}`")
        return self.locals[name]


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

    def _collect_locals(self, block: ast.CompoundStmt, ctx: _FuncCtx) -> None:
        for item in block.items:
            if isinstance(item, ast.VarDecl):
                self._check_int_type(item.var_type, item.name)
                ctx.alloc(item.name)
            elif isinstance(item, ast.CompoundStmt):
                self._collect_locals(item, ctx)
            # Other statement types: nothing to allocate yet.

    @staticmethod
    def _check_int_type(t: ast.TypeNode, name: str) -> None:
        if not isinstance(t, ast.BasicType) or t.name != "int":
            raise CodegenError(
                f"local `{name}`: only `int` locals are supported "
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
        raise CodegenError(
            f"{type(item).__name__} not implemented yet"
        )

    def _expr_stmt(self, stmt: ast.ExpressionStmt, ctx: _FuncCtx) -> list[str]:
        if stmt.expr is None:
            return []
        # Result is discarded; we still evaluate for side effects (assignment).
        return self._eval_expr_to_eax(stmt.expr, ctx)

    def _var_init(self, decl: ast.VarDecl, ctx: _FuncCtx) -> list[str]:
        offset = ctx.lookup(decl.name)
        if decl.init is None:
            # Uninitialized — leave the slot as-is. Reading it is UB, but
            # we don't pre-zero unless required.
            return []
        return self._eval_expr_to_eax(decl.init, ctx) + [
            f"        mov     [ebp - {offset}], eax",
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
            offset = ctx.lookup(expr.name)
            return [f"        mov     eax, [ebp - {offset}]"]
        if isinstance(expr, ast.UnaryOp):
            return self._unary(expr, ctx)
        if isinstance(expr, ast.BinaryOp):
            return self._binary(expr, ctx)
        raise CodegenError(
            f"expression {type(expr).__name__} not implemented yet"
        )

    def _unary(self, expr: ast.UnaryOp, ctx: _FuncCtx) -> list[str]:
        if not expr.is_prefix:
            raise CodegenError("postfix ++/-- not implemented yet")
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

    def _binary(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        if expr.op == "=":
            return self._assign(expr, ctx)

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

    def _assign(self, expr: ast.BinaryOp, ctx: _FuncCtx) -> list[str]:
        if not isinstance(expr.left, ast.Identifier):
            raise CodegenError(
                f"assignment target must be an identifier "
                f"(got {type(expr.left).__name__})"
            )
        offset = ctx.lookup(expr.left.name)
        return self._eval_expr_to_eax(expr.right, ctx) + [
            f"        mov     [ebp - {offset}], eax",
        ]
