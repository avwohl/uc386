"""x86-32 (i386) code generator for MS-DOS targets.

Phase 0: emit NASM Intel-syntax assembly for the smallest interesting C
program — `int main(void) { return N; }`. The emitted file assembles
with `nasm -fobj` and links against a DOS/4GW-compatible linker.

NASM was chosen as the assembler target because it's open source, ships
on every modern dev box, supports OMF object output for DOS toolchains
(`-fobj`), and uses Intel syntax that matches the rest of the
Watcom-era ecosystem.

Anything beyond `return <int-literal>;` raises NotImplementedError —
this is Phase 0, not full codegen.
"""

from uc_core import ast


class CodegenError(NotImplementedError):
    """Raised when the AST contains a construct Phase 0 codegen can't handle."""


class CodeGenerator:
    """i386/MS-DOS backend (Phase 0: int main + integer-literal return)."""

    def __init__(self, module_name: str = "main"):
        self.module_name = module_name

    def generate(self, unit: ast.TranslationUnit) -> str:
        functions = [
            d for d in unit.declarations
            if isinstance(d, ast.FunctionDecl) and d.body is not None
        ]
        if not any(fn.name == "main" for fn in functions):
            raise CodegenError("uc386 Phase 0 requires a `main` function definition")

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
        # _start: call user main, take its int return in EAX, exit DOS
        # via INT 21h/4Ch with AL = exit code. AH=4Ch leaves AL untouched.
        return [
            "_start:",
            "        call    _main",
            "        mov     ah, 4Ch",
            "        int     21h",
        ]

    def _function(self, fn: ast.FunctionDecl) -> list[str]:
        out = [f"_{fn.name}:"]
        out.append("        push    ebp")
        out.append("        mov     ebp, esp")
        out += self._compound(fn.body)
        # C99: falling off the end of main returns 0. For other functions
        # this is technically undefined, but a deterministic zero beats
        # leaking whatever EAX held.
        out.append("        xor     eax, eax")
        out.append(".epilogue:")
        out.append("        mov     esp, ebp")
        out.append("        pop     ebp")
        out.append("        ret")
        return out

    def _compound(self, block: ast.CompoundStmt) -> list[str]:
        out: list[str] = []
        for item in block.items:
            out += self._stmt(item)
        return out

    def _stmt(self, stmt) -> list[str]:
        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is None:
                return [
                    "        xor     eax, eax",
                    "        jmp     .epilogue",
                ]
            if isinstance(stmt.value, ast.IntLiteral):
                return [
                    f"        mov     eax, {stmt.value.value}",
                    "        jmp     .epilogue",
                ]
            raise CodegenError(
                f"return value must be an integer literal in Phase 0 "
                f"(got {type(stmt.value).__name__})"
            )
        if isinstance(stmt, ast.CompoundStmt):
            return self._compound(stmt)
        raise CodegenError(
            f"statement type {type(stmt).__name__} not implemented in Phase 0"
        )
