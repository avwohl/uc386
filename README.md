# uc386

> ### 🤖 No Primate policy
>
> **The code in this repository is written by AI. No primate wrote it.**
>
> Code: AI. Documentation: AI. Deciding what to build, what is correct,
> and what ships: primate.
>
> Stated up front so nobody has to work it out later. This is a
> disclosure, not a boast — and nothing below asks to be taken on trust.
> Every number is measured, the suites are public, and the whole thing
> rebuilds from source. Run it yourself and judge the output, not the
> byline.

C23 compiler targeting the Intel 386 (i386 / x86-32) processor under a
DOS extender — specifically the **flat 32-bit Watcom / DOS/4GW-era** C
that early-to-mid-1990s PC games were written in.

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
miscompiles have been driven out (see `STANDARD_C_BACKLOG.md`).

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

See `addons/STATUS.md` for the full per-addon report and
`docs/path-a-mz-le.md` for the `.exe` build path.

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
[`addons/gnu/UPSTREAM.md`](addons/gnu/UPSTREAM.md).

## Size — measured, not asserted

The "tiny output" claim, checked against the period reference
compiler instead of asserted. Every column below was **reproduced
on one macOS/arm64 host** by `python -m addons.harness.compare`
(Open Watcom V2 has no native macOS build, so its DOS-hosted
`wcc386`/`wlink` run under DOSBox-X via `addons/harness/
watcom_dosbox.py`; DJGPP is the gcc-12.2 osx cross under Rosetta).
Bytes of the on-disk executable; full table in
[`addons/results.md`](addons/results.md):

| program | uc386 .bin | uc386 .exe | Watcom | DJGPP |
|---------|-----------:|-----------:|-------:|------:|
| true    |         18 |     32,847 |  5,420 | 147,914 |
| echo    |        264 |     32,911 | 11,286 | 150,212 |
| factor  |      2,022 |     32,981 | 20,538 | 179,614 |
| wc      |      1,861 |     32,992 | 20,158 | 179,092 |

Reading this honestly:

- **`.bin` is not a DOS program.** It has no MZ header and runs
  only under `uc386.dos_emu`/a custom loader. It is the right
  metric for *codegen+DCE tightness* (and there uc386 is in a
  class of its own — tens of bytes), but it is not what you ship.
- **`.exe` is what you ship**, and it carries a **~32.8 KB DOS/32A
  extender floor** — every `.exe` in the table is that floor plus a
  few hundred bytes of program. Against that real-DOS artifact,
  **Open Watcom is smaller: ~6× on `true`, ~1.6× on `wc`/`factor`**
  (its DOS/4GW clib + mature linker beat our extender floor); the
  two converge as real code grows. uc386 beats **DJGPP ~4.5–5.5×**.
- **The floor is a deliberate correctness trade.** `--extender=pmodew`
  halves it (~16.8 KB), but PMODE/W's real-mode call path hangs on
  any DOS call that touches a physical sector, so a PMODE/W build
  cannot do disk I/O on real DOS. DOS/32A is the default because a
  working `.exe` beats a smaller broken one; PMODE/W stays available
  for programs that only touch stdout.
- So: uc386's *code generation* is extremely compact; its current
  *DOS packaging* is not yet competitive with Watcom's. Both
  statements are true and the table shows which is which — no
  single "390× smaller" headline.

## Goal

Compile representative public-source DOS games **unmodified**:

- Descent (Parallax, 1995 — Watcom)
- Duke Nukem 3D / Build engine (3D Realms, 1996 — Watcom)
- Rise of the Triad (Apogee, 1994 — Watcom)
- Heretic / Hexen (Raven, 1994–95 — Watcom)

These all share one compiler (Watcom C/C++) and one memory model
(flat 32-bit under DOS/4GW). That's the target.

**Non-goals:** 16-bit real-mode with near/far/huge memory models
(Wolf3D-era code). uc386 will *parse* the 16-bit keywords so that
shared period headers don't choke, but won't honor their semantics —
all pointers are 32-bit flat.

## Design

The uc80/uc386 family shares a single C23 frontend
([uc_core](https://github.com/avwohl/uc_core), itself uplox-driven).
This project contributes only:

- `main.py` — driver (CLI, I/O, embedding, post-processing)
- `codegen.py` — x86-32 NASM code generator
- `lib/i386_dos_libc.asm` — the DOS libc, plus `lib/include/` headers
- `runtime.py` — placeholder for Python-side runtime bindings (the
  real libc is the `.asm` above)
- `dos_emu.py` — i386 emulator harness for testing flat-binary output
- `dos_emu_netsim.py` — simulated network for the INT 0x83 packet-driver shim
- `dosiz_run.py` — alternate harness dispatching to `../dosiz`
  (in-process dosbox-staging, full DPMI 0.9)
- `harness.py` — selects between the two via `UC386_HARNESS`
- `addons/harness/` — the `.asm` → `.obj` → MZ+LE `.exe` pipeline
  (source checkout only; not shipped on PyPI)

The NASM-text peephole optimizer, the assembly-level dead-code
eliminator, and the libc symbol splitter used to live here; they were
factored out into [upeep386](https://github.com/avwohl/upeep386) and
are now a dependency rather than part of this repo.

Every front-end improvement (new C23 feature, AST optimization, DOS-era
syntax tolerance) lands in uc_core and benefits both targets
automatically.

## Install

From PyPI:

```
pip install uc386
```

That gets you the `uc386` driver, the bundled `i386_dos_libc.asm`,
and the `lib/include/` headers, and it pulls the frontend
(`uc_core`, `uplox`) and the asm-level optimizer (`upeep386`)
automatically. To assemble + run the output you also need `nasm`
(system package) and, for the `dos_emu` test harness, `pip install
unicorn`.

**The driver has no default include path**, so `#include <stdio.h>`
fails until you point `-I` at the installed headers:

```sh
UC386_INC=$(python -c "import uc386,os;print(os.path.join(os.path.dirname(uc386.__file__),'lib','include'))")
uc386 hello.c -o hello.asm -I "$UC386_INC"
```

`examples/hello.c` declares its one prototype inline specifically so
it compiles with no `-I` at all.

That install compiles C to `.asm`. Building a bootable DOS `.exe`
additionally needs `pip install upyle` and the `addons/harness/`
tree, which ships only in the source checkout — see
[`docs/path-a-mz-le.md`](docs/path-a-mz-le.md).

Source checkout for development:

```
sudo apt-get install -y python3 python3-venv nasm    # Debian/Ubuntu
git clone https://github.com/avwohl/uc386 && cd uc386
python3 -m venv .venv && . .venv/bin/activate
pip install pytest unicorn upyle -e .
pytest tests/          # 498 passed, 1 skipped
```

To co-develop the frontend or the optimizer, clone them as siblings
and install those editable too — see
[`CLAUDE.md`](CLAUDE.md) for that layout.

macOS (Homebrew) and Fedora/RHEL (dnf) instructions, plus the
optional toolchains for addon builds (bison/flex) and the
DJGPP / OpenWatcom comparison columns, are documented in
[`docs/INSTALL.md`](docs/INSTALL.md).

## Related Projects

- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android phones and tablets. It emulates the RomWBW HBIOS interface and a VT100 terminal.
- [cpmemu](https://github.com/avwohl/cpmemu) - Z80/CP/M emulator for Linux and Windows, with Z80 and 8080 CPU cores. It translates the BDOS and BIOS calls of CP/M 2.2 programs to the host file system.
- [dosiz](https://github.com/avwohl/dosiz) - MS-DOS emulator for Linux. It uses the dosbox-staging CPU core and translates system calls in the manner of cpmemu. It is the intended test host for uc386.
- [pyle](https://github.com/avwohl/pyle) - OMF to MZ+LE linker written in pure Python. It builds the DOS `.exe` files of uc386 and needs no Open Watcom. The repository is `pyle` but the package is `upyle`.
- [qxDOS](https://github.com/avwohl/qxDOS) - DOS emulator app for iOS and macOS with a SwiftUI interface. DOSBox Staging supplies the emulated i386 hardware.
- [uc80](https://github.com/avwohl/uc80) - C compiler for the Z80 processor and CP/M. This sibling backend shares the C23 frontend of uc_core.
- [uc_core](https://github.com/avwohl/uc_core) - Shared C23 frontend and AST optimizer for the uc80 and uc386 compilers.
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Linux toolchain that is compatible with Microsoft MACRO-80. It has an assembler, a linker, a librarian, and a disassembler. It is the Z80 equivalent of what uc386 needs for i386.
- [upeep386](https://github.com/avwohl/upeep386) - Peephole optimizer, assembly dead-code eliminator, and libc symbol splitter for i386. uc386 depends on it.
- [upeepz80](https://github.com/avwohl/upeepz80) - Peephole optimizer for Z80 compilers. It was the template for upeep386.
- [uplox](https://github.com/avwohl/uplox) - LR(1) and GLR parser generator. It writes the lexer and parser tables for the C23 frontend of uc_core from `examples/c23.uplox`.

## License

GPL-3.0-or-later.
