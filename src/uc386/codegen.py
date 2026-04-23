"""x86-32 (i386) code generator for MS-DOS targets.

Skeleton. Implements uc_core.backend.CodeGenerator but emits a TODO
comment instead of real assembly. Exists so the end-to-end pipeline
(preprocess -> lex -> parse -> AST optimize -> codegen) can be wired
up and tested before the actual codegen work begins.
"""

from uc_core import ast


class CodeGenerator:
    """i386/MS-DOS backend (stub)."""

    def __init__(self, module_name: str = "main"):
        self.module_name = module_name

    def generate(self, unit: ast.TranslationUnit) -> str:
        decls = [d for d in unit.declarations if isinstance(d, ast.FunctionDecl)]
        lines = [
            "; uc386 codegen output (STUB)",
            f"; module: {self.module_name}",
            f"; function declarations parsed: {len(decls)}",
            "",
        ]
        for fn in decls:
            lines.append(f";   {fn.name}")
        lines.append("")
        lines.append("; TODO: emit real i386 assembly")
        return "\n".join(lines) + "\n"
