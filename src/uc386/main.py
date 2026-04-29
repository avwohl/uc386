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
    "__INT_LEAST8_TYPE__": "signed char",
    "__UINT_LEAST8_TYPE__": "unsigned char",
    "__INT_LEAST16_TYPE__": "short",
    "__UINT_LEAST16_TYPE__": "unsigned short",
    "__INT_LEAST32_TYPE__": "long",
    "__UINT_LEAST32_TYPE__": "unsigned long",
    "__INT_LEAST64_TYPE__": "long long",
    "__UINT_LEAST64_TYPE__": "unsigned long long",
    "__INT_FAST8_TYPE__": "signed char",
    "__UINT_FAST8_TYPE__": "unsigned char",
    "__INT_FAST16_TYPE__": "int",
    "__UINT_FAST16_TYPE__": "unsigned int",
    "__INT_FAST32_TYPE__": "int",
    "__UINT_FAST32_TYPE__": "unsigned int",
    "__INT_FAST64_TYPE__": "long long",
    "__UINT_FAST64_TYPE__": "unsigned long long",
    "__INTMAX_TYPE__": "long long",
    "__UINTMAX_TYPE__": "unsigned long long",
    "__builtin_va_list": "char *",
    "__WCHAR_TYPE__": "unsigned short",
    "__WINT_TYPE__": "int",
    "__CHAR16_TYPE__": "unsigned short",
    "__CHAR32_TYPE__": "unsigned long",
    # GCC predefines this for sources that probe.
    "__GNUC__": "4",
    # IEEE-754 float / double limits — used by torture tests as
    # `__FLT_MAX__` etc. Approximated as decimal literals in the
    # source so the lexer parses them back as float/double values.
    "__FLT_MAX__": "3.40282347e+38F",
    "__FLT_MIN__": "1.17549435e-38F",
    "__FLT_EPSILON__": "1.19209290e-07F",
    "__DBL_MAX__": "1.7976931348623157e+308",
    "__DBL_MIN__": "2.2250738585072014e-308",
    "__DBL_EPSILON__": "2.2204460492503131e-16",
    "__LDBL_MAX__": "1.7976931348623157e+308L",
    "__LDBL_MIN__": "2.2250738585072014e-308L",
    "__LDBL_EPSILON__": "2.2204460492503131e-16L",
    "__INT_MAX__": "2147483647",
    "__SHRT_MAX__": "32767",
    "__SCHAR_MAX__": "127",
    "__LONG_MAX__": "2147483647L",
    "__LONG_LONG_MAX__": "9223372036854775807LL",
    "__CHAR_BIT__": "8",
    "__SCHAR_MIN__": "(-128)",
    "__INT_MIN__": "(-2147483648)",
    "__SHRT_MIN__": "(-32768)",
    "__LONG_MIN__": "(-2147483648L)",
    "__LONG_LONG_MIN__": "(-9223372036854775807LL-1)",
    "__FLT_DIG__": "6",
    "__FLT_MANT_DIG__": "24",
    "__DBL_DIG__": "15",
    "__DBL_MANT_DIG__": "53",
    "__LDBL_DIG__": "15",
    "__LDBL_MANT_DIG__": "53",
    "__FLT_RADIX__": "2",
    "__FLT_MAX_EXP__": "128",
    "__FLT_MIN_EXP__": "(-125)",
    "__DBL_MAX_EXP__": "1024",
    "__DBL_MIN_EXP__": "(-1021)",
    "__SIZEOF_POINTER__": "4",
    "__SIZEOF_INT__": "4",
    "__SIZEOF_LONG__": "4",
    "__SIZEOF_LONG_LONG__": "8",
    "__SIZEOF_SHORT__": "2",
    "__SIZEOF_FLOAT__": "4",
    "__SIZEOF_DOUBLE__": "8",
    "__SIZEOF_LONG_DOUBLE__": "8",
    "__SIZEOF_SIZE_T__": "4",
    "__SIZEOF_PTRDIFF_T__": "4",
    "__SIZEOF_WCHAR_T__": "2",
    "__SIZEOF_WINT_T__": "4",
    # Endianness predefines — i386 is little-endian.
    "__BYTE_ORDER__": "1234",
    "__ORDER_LITTLE_ENDIAN__": "1234",
    "__ORDER_BIG_ENDIAN__": "4321",
    "__ORDER_PDP_ENDIAN__": "3412",
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
    ap.add_argument("--no-peephole", action="store_true",
                    help="Disable asm-level peephole optimization")
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
            # C sources sometimes contain Latin-1 / extended ASCII
            # bytes (e.g. embedded `\377` characters in string
            # initializers). Read with errors='surrogateescape' so
            # those bytes survive and the lexer's char-by-char path
            # treats them as ordinary high-bit characters.
            source = p.read_text(errors="surrogateescape")
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

        gen = CodeGenerator(module_name=input_paths[0].stem,
                            peephole=not args.no_peephole)
        code = gen.generate(unit)
        output_path.write_text(code)

        if args.verbose:
            print(f"uc386: wrote {output_path}")
            if gen.peephole_stats:
                print("  peephole optimizations:")
                for name, count in sorted(gen.peephole_stats.items()):
                    print(f"    {name}: {count}")
        return 0

    except (PreprocessorError, LexerError) as e:
        print(f"uc386: {e}", file=sys.stderr)
        return 1
    except ParseError as e:
        print(f"uc386: {e.location}: {e.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
