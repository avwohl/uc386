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
from uc_core.type_config import TypeConfig, WATCOM_FLAT32

from .codegen import CodeGenerator

I386_DOS_PREDEFINES = {
    "__UC386__": "1",
    "__UC386_VERSION__": "1",
    "__I386__": "1",
    "__i386__": "1",
    "__MSDOS__": "1",
    "__DOS__": "1",
    "__ILP32__": "1",      # int/long/pointer all 32-bit
    # GCC builtin type macros — used by lots of period code.
    "__SIZE_TYPE__": "unsigned long",
    "__PTRDIFF_TYPE__": "long",
    "__INTPTR_TYPE__": "long",
    "__UINTPTR_TYPE__": "unsigned long",
    "__INT8_TYPE__": "signed char",
    "__UINT8_TYPE__": "unsigned char",
    "__INT16_TYPE__": "short",
    "__UINT16_TYPE__": "unsigned short",
    "__INT32_TYPE__": "long",
    "__UINT32_TYPE__": "unsigned long",
    "__INT64_TYPE__": "long long",
    "__UINT64_TYPE__": "unsigned long long",
    "__WCHAR_TYPE__": "unsigned short",
    "__WINT_TYPE__": "int",
    "__CHAR16_TYPE__": "unsigned short",
    "__CHAR32_TYPE__": "unsigned long",
    # GCC predefines this for sources that probe.
    "__GNUC__": "4",
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
    ap.add_argument("--int", dest="int_bits", type=int, choices=[16, 32],
                    help="int width in bits (default: 32 — Watcom flat-32)")
    ap.add_argument("--long", dest="long_bits", type=int, choices=[32, 64],
                    help="long width in bits (default: 32)")
    ap.add_argument("--long-long", dest="long_long_bits", type=int, choices=[64],
                    help="long long width in bits (default: 64)")
    ap.add_argument("--ptr", dest="ptr_bits", type=int, choices=[32],
                    help="pointer width in bits (default: 32 — flat-32 only)")
    args = ap.parse_args()

    type_config = TypeConfig(
        char_size=WATCOM_FLAT32.char_size,
        short_size=WATCOM_FLAT32.short_size,
        int_size=(args.int_bits // 8) if args.int_bits else WATCOM_FLAT32.int_size,
        long_size=(args.long_bits // 8) if args.long_bits else WATCOM_FLAT32.long_size,
        long_long_size=(args.long_long_bits // 8) if args.long_long_bits else WATCOM_FLAT32.long_long_size,
        ptr_size=(args.ptr_bits // 8) if args.ptr_bits else WATCOM_FLAT32.ptr_size,
        float_size=WATCOM_FLAT32.float_size,
        double_size=WATCOM_FLAT32.double_size,
        long_double_size=WATCOM_FLAT32.long_double_size,
    )

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
                pp_predefines = {**I386_DOS_PREDEFINES, **type_config.predefined_macros()}
                pp = Preprocessor(args.include, target_predefines=pp_predefines)
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
            unit = ASTOptimizer(3, type_config=type_config).optimize(unit)

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
