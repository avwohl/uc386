#!/usr/bin/env python3
"""uc386 - C23 compiler for i386/MS-DOS.

Driver: preprocess + lex + parse + AST-optimize via uc_core, then pass
to the uc386 backend. The backend is currently a stub.
"""

import argparse
import sys
from pathlib import Path

from uc_core.frontend import parse as _frontend_parse
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
    # GCC predefines these for sources that probe. Period code (Build
    # engine, etc.) often uses both __GNUC__ and __GNUC_MINOR__ in the
    # same expression (`sprintf("%d.%d", __GNUC__, __GNUC_MINOR__)`),
    # so leaving __GNUC_MINOR__ undefined turns it into an unknown
    # identifier at codegen time.
    "__GNUC__": "4",
    "__GNUC_MINOR__": "0",
    "__GNUC_PATCHLEVEL__": "0",
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
    # GCC fall-through attribute. MicroPython's MP_FALLTHROUGH is
    # conditional on __GNUC__ >= 7; when it's defined to
    # ``__attribute__((fallthrough));`` the source-level attribute
    # strip handles it. When source uses the macro without including
    # mpconfig.h, we'd see the bare identifier mid-switch which the
    # parser can't recover from — predefine it to nothing here as a
    # safety net.
    "MP_FALLTHROUGH": "",
    # Endianness predefines — i386 is little-endian.
    "__BYTE_ORDER__": "1234",
    "__ORDER_LITTLE_ENDIAN__": "1234",
    "__ORDER_BIG_ENDIAN__": "4321",
    "__ORDER_PDP_ENDIAN__": "3412",
}


def _mangling_prefix(path: Path) -> str:
    """Stable per-file prefix used to mangle file-scope statics.
    Hash collision risk is low; only the basename's stem is used so
    builds are reproducible regardless of build directory layout."""
    stem = "".join(c if (c.isalnum() or c == "_") else "_" for c in path.stem)
    return f"__static_{stem}__"


def _mangle_static_globals(unit, prefix: str) -> None:
    """Rename file-scope `static` decls in `unit` to `<prefix><name>`,
    and rewrite intra-TU references to match. Walks the auto-AST
    generically via dataclasses.fields to avoid maintaining a
    per-node-type table.

    Scope-aware: when a function parameter or block-local variable
    shadows a file-scope static, references within that scope refer
    to the local, not the static — so we skip the rename. The
    scenario that surfaced this requirement: crypto_misc.c declares
    `static const uint8_t map[128]` (a base64 lookup table) and
    includes py/obj.h's `static inline mp_map_slot_is_filled(const
    mp_map_t *map, ...)` whose body references parameter `map`. A
    naive lexical rewrite turns those parameter refs into the
    file-scope byte array, breaking the inline at the type level.
    """
    import dataclasses
    from uc_core.codegen_helpers import decl_storage_class, declarator_ident
    from uc_core.c23_parser import Token as _Token

    def _innermost_declarator(node):
        """Walk a declarator chain (PointerDeclarator, ArrayDeclarator,
        FnDeclarator, GroupDeclarator) to the leaf Declarator carrying
        the Token name. Returns the leaf or None."""
        while node is not None:
            if isinstance(node, ast_module.Declarator):
                return node
            inner = getattr(node, "inner", None)
            if inner is None:
                return None
            node = inner
        return None

    def _function_param_idents(declarator):
        """Yield the parameter name Tokens (from ParamDecl declarators)
        on the outermost FnDeclarator of a function-typed declarator."""
        node = declarator
        while node is not None:
            if isinstance(node, (ast_module.FnDeclarator,
                                 ast_module.FnDeclaratorEmpty)):
                params = getattr(node, "params", None) or []
                if isinstance(params, ast_module.VariadicParams):
                    params = params.params
                for p in params:
                    if isinstance(p, ast_module.ParamDecl):
                        leaf = _innermost_declarator(p.declarator)
                        if leaf is not None:
                            yield leaf.name
                return
            node = getattr(node, "inner", None)

    # Collect file-scope statics. The auto-AST stores top-level decls
    # as ast.Declaration (with N declarators) and ast.FunctionDef.
    statics: set[str] = set()
    for d in unit.items:
        if isinstance(d, ast_module.Declaration):
            if decl_storage_class(d.decl_specs) != "static":
                continue
            for init_decl in d.declarators or []:
                inner = init_decl
                if isinstance(inner, (ast_module.InitDeclarator,
                                      ast_module.InitDeclaratorWithInit)):
                    inner = inner.declarator
                nm = declarator_ident(inner)
                if nm:
                    statics.add(nm)
        elif isinstance(d, ast_module.FunctionDef):
            if decl_storage_class(d.decl_specs) != "static":
                continue
            nm = declarator_ident(d.declarator)
            if nm:
                statics.add(nm)
    if not statics:
        return

    def _rename_token(tok, shadowed: frozenset[str]):
        """If ``tok`` names a static and isn't shadowed, return a new
        Token with the prefix prepended to .text. Otherwise return
        ``tok`` unchanged. Tokens are frozen dataclasses; callers
        re-assign the returned Token onto the parent node."""
        if tok is None:
            return tok
        text = getattr(tok, "text", None)
        if text in statics and text not in shadowed:
            return _Token(
                name=tok.name, text=prefix + text,
                line=tok.line, column=tok.column,
                offset=tok.offset, file_id=tok.file_id,
            )
        return tok

    def walk(node, shadowed: frozenset[str]):
        if node is None:
            return
        if isinstance(node, list):
            for child in node:
                walk(child, shadowed)
            return
        if not dataclasses.is_dataclass(node):
            return

        # Function definition: rewrite the function's own name, then
        # collect its parameter names as the shadow set for the body.
        if isinstance(node, ast_module.FunctionDef):
            leaf = _innermost_declarator(node.declarator)
            if leaf is not None:
                leaf.name = _rename_token(leaf.name, shadowed)
            param_names: set[str] = set()
            for ptok in _function_param_idents(node.declarator):
                if ptok is not None:
                    param_names.add(ptok.text)
            # Walk decl_specs + declarator (excluding the leaf we just
            # handled) under the outer scope, then body under the
            # extended scope.
            for spec in node.decl_specs or []:
                walk(spec, shadowed)
            walk(node.declarator, shadowed)
            new_shadowed = shadowed | frozenset(param_names)
            if node.body is not None:
                walk(node.body, new_shadowed)
            return

        # Declaration: rewrite each declarator's leaf name, then walk
        # the init expressions under the current scope. Multi-decl
        # `int a, b = a;` reads the outer `a` for `b`'s init (C 6.7.6).
        if isinstance(node, ast_module.Declaration):
            for init_decl in node.declarators or []:
                inner = init_decl
                if isinstance(inner, (ast_module.InitDeclarator,
                                      ast_module.InitDeclaratorWithInit)):
                    inner = init_decl.declarator
                leaf = _innermost_declarator(inner)
                if leaf is not None:
                    leaf.name = _rename_token(leaf.name, shadowed)
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name, None), shadowed)
            return

        # CompoundStmt: collect locally-declared names into the
        # shadow set as we walk (source-order: a decl's initializer
        # can still reference the static if the local hasn't been
        # declared yet at that point).
        if isinstance(node, ast_module.CompoundStmt):
            local_shadow = set(shadowed)
            for item in node.items or []:
                walk(item, frozenset(local_shadow))
                if isinstance(item, ast_module.Declaration):
                    for init_decl in item.declarators or []:
                        inner = init_decl
                        if isinstance(inner, (ast_module.InitDeclarator,
                                              ast_module.InitDeclaratorWithInit)):
                            inner = init_decl.declarator
                        leaf = _innermost_declarator(inner)
                        if leaf is not None and leaf.name.text:
                            local_shadow.add(leaf.name.text)
                elif isinstance(item, ast_module.FunctionDef):
                    leaf = _innermost_declarator(item.declarator)
                    if leaf is not None and leaf.name.text:
                        local_shadow.add(leaf.name.text)
            return

        # Identifier: rewrite the name token if it matches a static
        # and isn't shadowed.
        if isinstance(node, ast_module.Identifier):
            node.name = _rename_token(node.name, shadowed)

        for f in dataclasses.fields(node):
            walk(getattr(node, f.name, None), shadowed)

    for d in unit.items:
        walk(d, frozenset())


def main() -> int:
    ap = argparse.ArgumentParser(prog="uc386", description="C23 compiler for i386/MS-DOS")
    ap.add_argument("input", nargs="+", help="Input C source file(s)")
    ap.add_argument("-o", "--output", help="Output assembly file (default: input.asm)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-I", "--include", action="append", default=[], metavar="DIR")
    ap.add_argument("--include-file", action="append", default=[], metavar="FILE",
                    help="Force-include FILE at the start of every source. "
                         "Useful for headers that period code expects to be "
                         "always available (gcc -include equivalent).")
    ap.add_argument("-D", "--define", action="append", default=[], metavar="NAME[=VALUE]")
    ap.add_argument("-E", "--preprocess-only", action="store_true")
    ap.add_argument("-P", "--no-preprocess", action="store_true")
    ap.add_argument("--no-ast-optimize", action="store_true")
    ap.add_argument("--no-peephole", action="store_true",
                    help="Disable asm-level peephole optimization")
    ap.add_argument("--no-asm-dce", action="store_true",
                    help="Disable post-codegen asm dead-code elimination")
    ap.add_argument("--no-embed-runtime", action="store_true",
                    help="Don't embed libc runtime into output asm. "
                         "When disabled, output is user-only with extern "
                         "declarations; runtime gets bundled at runtime "
                         "by dos_emu (legacy behavior).")
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
                # `--include-file` (gcc -include semantics): prepend
                # `#include "..."` directives to the source so the named
                # files get processed before anything else. Lets us pull
                # in headers that period code expects to "always be in
                # scope" even when upstream forgot to #include them.
                if args.include_file:
                    prefix = "".join(f'#include "{f}"\n' for f in args.include_file)
                    source = prefix + source
                source = pp.preprocess(source, str(p))
                if args.preprocess_only:
                    print(source)
                    continue
            asts.append(_frontend_parse(source, str(p)))

        if args.preprocess_only:
            return 0

        if len(asts) == 1:
            unit = asts[0]
        else:
            # Multi-TU mode: file-scope `static` decls have internal
            # linkage in C, so identical names from different TUs are
            # legitimate. We naive-merge into one TranslationUnit, so
            # mangle each TU's static names with a per-file prefix to
            # keep them distinct (and rewrite intra-TU references to
            # match).
            for path, u in zip(input_paths, asts):
                _mangle_static_globals(u, _mangling_prefix(path))
            unit = ast_module.TranslationUnit(items=[])
            merged_vector_names: set[str] = set()
            for u in asts:
                unit.items.extend(u.items)
                merged_vector_names |= getattr(
                    u, "_vector_typedef_names", set(),
                )
            unit._vector_typedef_names = merged_vector_names

        if not args.no_ast_optimize:
            unit = ASTOptimizer(3, type_config=type_config).optimize(unit)

        gen = CodeGenerator(module_name=input_paths[0].stem,
                            peephole=not args.no_peephole)
        code = gen.generate(unit)
        if not args.no_asm_dce:
            from uc386.asm_dce import dce as asm_dce
            code = asm_dce(code)
        # Embed libc into the asm at compile time so subsequent
        # peephole + asm DCE passes optimize across user + runtime.
        # Without this, libc functions never get peephole'd and any
        # cascade DCE through libc is missed. (When --no-embed-runtime
        # is set, output is user-only; dos_emu bundles at runtime.)
        if not args.no_embed_runtime:
            from uc386.dos_emu import bundle_text
            code = bundle_text(code)
            # Re-run peephole over the combined asm so libc functions
            # benefit from the same optimization passes user code does.
            if not args.no_peephole:
                from uc386.peephole import optimize as peephole_optimize
                code = peephole_optimize(code)
            # Re-run asm DCE to catch cascades: libc functions whose
            # only callers were dropped by the user-side DCE, or that
            # peephole exposed as unreachable.
            if not args.no_asm_dce:
                code = asm_dce(code)
        output_path.write_text(code)

        if args.verbose:
            print(f"uc386: wrote {output_path}")
            if gen.peephole_stats:
                print("  peephole optimizations:")
                for name, count in sorted(gen.peephole_stats.items()):
                    print(f"    {name}: {count}")
        return 0

    except PreprocessorError as e:
        print(f"uc386: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
