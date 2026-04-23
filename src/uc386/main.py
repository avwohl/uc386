#!/usr/bin/env python3
"""uc386 - C23 compiler for i386/MS-DOS.

Driver: preprocess + lex + parse + AST-optimize via uc_core, then pass
to the uc386 backend. The backend is currently a stub.
"""

import argparse
import sys
from pathlib import Path

from uc_core.lexer import Lexer, LexerError
from uc_core.parser import Parser, ParseError
from uc_core import ast as ast_module
from uc_core.preprocessor import Preprocessor, PreprocessorError, Macro
from uc_core.ast_optimizer import ASTOptimizer

from .codegen import CodeGenerator

I386_DOS_PREDEFINES = {
    "__UC386__": "1",
    "__UC386_VERSION__": "1",
    "__I386__": "1",
    "__i386__": "1",
    "__MSDOS__": "1",
    "__DOS__": "1",
}


def main() -> int:
    ap = argparse.ArgumentParser(prog="uc386", description="C23 compiler for i386/MS-DOS")
    ap.add_argument("input", nargs="+", help="Input C source file(s)")
    ap.add_argument("-o", "--output", help="Output assembly file (default: input.asm)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-I", "--include", action="append", default=[], metavar="DIR")
    ap.add_argument("-D", "--define", action="append", default=[], metavar="NAME[=VALUE]")
    ap.add_argument("-E", "--preprocess-only", action="store_true")
    ap.add_argument("-P", "--no-preprocess", action="store_true")
    ap.add_argument("--no-ast-optimize", action="store_true")
    args = ap.parse_args()

    input_paths = [Path(f) for f in args.input]
    for p in input_paths:
        if not p.exists():
            print(f"uc386: error: {p}: No such file", file=sys.stderr)
            return 1

    output_path = Path(args.output) if args.output else input_paths[0].with_suffix(".asm")

    try:
        asts = []
        for p in input_paths:
            source = p.read_text()
            if not args.no_preprocess:
                pp = Preprocessor(args.include, target_predefines=I386_DOS_PREDEFINES)
                for define in args.define:
                    if "=" in define:
                        name, value = define.split("=", 1)
                        pp.macros[name] = pp.macros.get(name) or Macro(name, body=value)
                    else:
                        pp.macros[define] = Macro(define, body="1")
                source = pp.preprocess(source, str(p))
                if args.preprocess_only:
                    print(source)
                    continue
            tokens = list(Lexer(source, str(p)).tokenize())
            asts.append(Parser(tokens).parse())

        if args.preprocess_only:
            return 0

        if len(asts) == 1:
            unit = asts[0]
        else:
            unit = ast_module.TranslationUnit(declarations=[])
            for u in asts:
                unit.declarations.extend(u.declarations)

        if not args.no_ast_optimize:
            unit = ASTOptimizer(3).optimize(unit)

        gen = CodeGenerator(module_name=input_paths[0].stem)
        code = gen.generate(unit)
        output_path.write_text(code)

        if args.verbose:
            print(f"uc386: wrote {output_path}")
        return 0

    except (PreprocessorError, LexerError) as e:
        print(f"uc386: {e}", file=sys.stderr)
        return 1
    except ParseError as e:
        print(f"uc386: {e.location}: {e.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
