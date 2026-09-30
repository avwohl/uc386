# Status and highlights

**Status: working and released — `pip install uc386` (0.2.1 on
[PyPI](https://pypi.org/project/uc386/)).** Measured
against two reference suites under our DOS emulator (compile →
assemble → run → diff): **215 / 220**
[c-testsuite](https://github.com/c-testsuite/c-testsuite) and, with
the `--kr` pre-pass (see below), **1397 / 1514**
[gcc-c-torture](https://github.com/llvm/llvm-test-suite) executable
tests passing. The frontend defaults to **strict C23**; the
gcc-c-torture corpus is pre-ANSI and GNU-heavy, so it is run with
`--kr` enabled. The remaining ~117 are GCC extensions and scoped
features rather than standard-C miscompiles: nested functions and
`__label__` (which need a static-chain ABI / closure conversion),
`__attribute__((aligned(N)))` in struct layout, extended inline
`__asm__` with operand constraints, `_Complex` struct members,
`-finstrument-functions`, and a few large-frame / file-I/O edges —
tracked, not claimed as passing. C99 VLAs and variably-modified
types, designated initializers, and `offsetof` designators all
landed during the campaign, and the standard-C codegen-corner
miscompiles have been driven out (see [`STANDARD_C_BACKLOG.md`](../STANDARD_C_BACKLOG.md)).

**K&R / implicit-int compatibility (`--kr`).** Pre-ANSI sources —
implicit-`int` returns (`main() { … }`) and K&R old-style parameter
lists (`f(a, b) int a; char *b; { … }`) — are not valid C23 and the
strict grammar rejects them — as is the GNU **computed-goto /
labels-as-values** extension (`&&label`, `goto *expr`). Passing
`--kr` enables a source-level pre-pass (in
[uc_core](https://github.com/avwohl/uc_core)) that rewrites these
shapes into equivalent standard C before parsing (computed goto
lowers to a `switch` dispatch). It is **off by default and only
engages on files that fail the strict parse**, so modern code is
parsed exactly once and pays zero cost. Use it for legacy/pre-ANSI
or GNU-C codebases; the conformance runners enable it for the
K&R-heavy torture corpus.

The frontend (parsing, preprocessing, AST-level optimization) lives
in [uc_core](https://github.com/avwohl/uc_core); this repo owns the
driver, the x86-32 NASM emitter, and the DOS runtime bindings.

**Highlights** — beyond the reference suites, uc386 compiles real
third-party C programs into runnable DOS executables:

- **Real `.exe` output.** Produces self-contained, DOS/32A-bound DOS
  `.exe` files (not just flat binaries), boot-tested under DOSBox in
  CI: correct errorlevels, command-line argument parsing, and
  `printf`/file I/O through genuine DOS handles. The `.exe` pipeline
  lives in `addons/harness/` and needs a source checkout plus
  [`upyle`](https://github.com/avwohl/pyle); the PyPI package ships
  the compiler and its libc, which stop at `.asm`.
- **DOOM** (id Software's 1993 shooter) compiles and boots
  end-to-end, running through engine startup until it exits cleanly
  on the expected "WAD file not found".
- **MicroPython** (a small Python interpreter) compiles into a
  working DOS Python REPL — expressions, functions, classes, list
  comprehensions, exceptions, and the common builtins. Packaged
  separately as
  [freedos_micro_python](https://github.com/avwohl/freedos_micro_python).
  It is our toughest end-to-end test of the compiler.
- **awk** — Kernighan's "one true awk" runs arithmetic, regexes,
  aggregation, and string functions.
- **GNU utilities** — 17 in-tree programs (`cat`, `wc`, `true`,
  `head`, `tail`, three `sbase` ports, …) build and pass
  parametrized regression tests against per-addon manifests.

See [`addons/STATUS.md`](../addons/STATUS.md) for the full per-addon report and
[`path-a-mz-le.md`](path-a-mz-le.md) for the `.exe` build path.

**File positioning and stream state work.** `fseek`, `ftell`, `rewind`,
`clearerr`, `feof` and `ferror` are real: seeking goes through INT 21h
AH=0x42, and per-stream EOF/error state lives in a handle-indexed table,
so `while (!feof(f))` terminates, `ftell` reports the true position, and
`ferror` distinguishes a read error from end-of-file. These were
no-op stubs until recently — a stub that returns a plausible wrong
answer is worse than one that fails — and `tests/test_stdio_position.py`
now pins the behaviour.

`errno` is populated too: DOS reports failure as a code in AX, which
the libc now translates (invalid handle → `EBADF`, access denied →
`EACCES`, not found → `ENOENT`, …), so `strerror` returns a real
message and `perror` prints `path: No such file or directory` rather
than a fixed `": error"`.

Console output is line-buffered. Every character used to be its own
INT 21h — measured, printing 2,000 bytes cost **2,000 DOS calls; it
now costs 2**. Output is flushed on newline, when the 1024-byte buffer
fills, by `fflush`/`fclose`, and at exit, so nothing is dropped;
`setvbuf` honors all three modes for real. The trade is size: programs
that print carry ~120–300 bytes more (`echo` 148 → 264), which is why
the exit-time flush is emitted only for programs that actually print —
`true` is still 18 bytes.

`popen`/`pclose` remain the real gap: they always fail, DOS having no
pipe API without a shell layer. Details in
[`addons/gnu/UPSTREAM.md`](../addons/gnu/UPSTREAM.md).
