# GNU gawk port notes

**Upstream**: <https://ftp.gnu.org/gnu/gawk/gawk-5.4.0.tar.gz>
**License**: GPL-3.0-or-later
**Size**: ~50K LoC across `awkgram.c`, `eval.c`, `builtin.c`, `re.c`,
plus a shipped gnulib subset (~10K LoC) and a regex engine (~3K LoC).

This is **the** tall port of `docs/addons.txt`. Heavy gnulib
dependency, configure-script-driven feature detection, locale +
mbtowc / wcwidth / iconv, dynamic-extension dlopen plumbing.

## Strategy

Two-pass approach:

1. **Subset build**. Identify a "core gawk" subset that excludes:
   - extension/ (dlopen-loaded extensions — needs fork/exec we don't have)
   - mpfr/gmp big-number support (compile-time disabled via `-DNUMBER=double`)
   - gettext / locale (compile-time disabled via `-DENABLE_NLS=0`)
   - mbtowc / wcwidth (compile-time `-DMB_LEN_MAX=1` + stub functions)
   - readline (already optional; `-DREADLINE=0`)

   What's left: lexer, parser, AST evaluator, regex engine,
   built-in printf/sub/gsub/match. ~25K LoC.

2. **gnulib stubs**. gawk pulls in gnulib's `xalloc.c`, `dirname-lgpl.c`,
   `getopt.c`, `localename.c`, etc. Most of these have one-line uc386
   replacements (xmalloc → malloc-or-die wrapper, dirname → string
   manipulation, getopt → roll-our-own). Stub each in
   `addons/gnu/gawk/shim/`.

## Concrete blockers (anticipated)

| Blocker             | Fix                                                |
|---------------------|----------------------------------------------------|
| `regex.h`           | Port glibc's regex.c (~3K LoC) or use POSIX-2 BRE  |
| `<setjmp.h>`        | We have it (libc supports setjmp / longjmp)        |
| `getopt_long`       | Roll a 50-line replacement; gawk uses single-letter|
| `<errno.h>` semantics | Have errno but no per-errno strerror text        |
| `mbtowc` / `wcwidth`| Stub: 1 byte = 1 char, no UTF-8 awareness          |
| `dlopen`            | Drop extension support (subset build)              |
| `setlocale`         | Stub returning "C"                                 |
| `time` / `strftime` | We have signal.h; need clock + strftime — stub     |
| `qsort_r` (gnulib)  | We have plain `qsort`; gnulib has shim already     |

## Smaller alternative: BWK awk

Brian Kernighan's "one true awk" (`addons/gnu/awk-bwk/`) is the
historical reference implementation and a much smaller target —
~6K LoC, no gnulib. License: free. If gawk port stalls, awk-bwk is
a viable fallback that also satisfies the spirit of "include awk in
the FOSS installer."

## Status

Scaffolding only — `fetch.sh` works, `build.sh` is a stub that prints
"awaiting subset patches." First real attempt is several iterations
away.
