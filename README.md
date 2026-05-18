# uc386

C23 compiler targeting the Intel 386 (i386 / x86-32) processor under a
DOS extender — specifically the **flat 32-bit Watcom / DOS/4GW-era** C
that early-to-mid-1990s PC games were written in.

**Status: working — in testing ahead of a general release.** Passes
both reference suites at 100%: all 1514 executable
[gcc-c-torture](https://github.com/llvm/llvm-test-suite) tests and
all 220 [c-testsuite](https://github.com/c-testsuite/c-testsuite)
tests compile, assemble, and run correctly under our DOS emulator.
The frontend (parsing, preprocessing, AST-level optimization) lives
in [uc_core](https://github.com/avwohl/uc_core); this repo owns the
driver, the x86-32 NASM emitter, and the DOS runtime bindings.

**Highlights** — beyond the reference suites, uc386 compiles real
third-party C programs into runnable DOS executables:

- **Real `.exe` output.** Produces self-contained, PMODE/W-bound DOS
  `.exe` files (not just flat binaries), boot-tested under DOSBox in
  CI: correct errorlevels, command-line argument parsing, and
  `printf`/file I/O through genuine DOS handles.
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
- **GNU utilities** — 16 in-tree coreutils-style programs (`cat`,
  `wc`, `true`, …) build and pass parametrized regression tests.

See `addons/STATUS.md` for the full per-addon report and
`docs/path-a-mz-le.md` for the `.exe` build path.

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
- `peephole.py` — NASM-text peephole optimizer
- `asm_dce.py` — assembly-level dead-code elimination from `_start` / `_main`
- `libc_split.py` — selective inclusion of `lib/i386_dos_libc.asm` symbols
- `runtime.py` — MS-DOS runtime library bindings (INT 21h wrappers, stubs)
- `dos_emu.py` — i386 emulator harness for testing flat-binary output
- `dos_emu_netsim.py` — simulated network for the INT 0x83 packet-driver shim

Every front-end improvement (new C23 feature, AST optimization, DOS-era
syntax tolerance) lands in uc_core and benefits both targets
automatically.

## Install

> **Note:** not yet ready on PyPI — install from the GitHub repository for now.

From PyPI:

```
pip install uc386
```

That gets you the `uc386` driver, the bundled `i386_dos_libc.asm`,
and the `lib/include/` headers. To assemble + run the output you
also need `nasm` (system package) and, for the `dos_emu` test
harness, `pip install unicorn`.

Source checkout for development:

```
sudo apt-get install -y python3 python3-venv nasm    # Debian/Ubuntu
python3 -m venv .venv && . .venv/bin/activate
pip install pytest unicorn "uc_core @ git+https://github.com/avwohl/uc_core@main" -e .
pytest tests/
```

macOS (Homebrew) and Fedora/RHEL (dnf) instructions, plus the
optional toolchains for addon builds (bison/flex) and the
DJGPP / OpenWatcom comparison columns, are documented in
[`docs/INSTALL.md`](docs/INSTALL.md).

## Related Projects

- [cpmdroid](https://github.com/avwohl/cpmdroid) - Z80/CP/M emulator for Android with RomWBW HBIOS compatibility and VT100 terminal
- [cpmemu](https://github.com/avwohl/cpmemu) - CP/M 2.2 emulator with Z80/8080 CPU emulation and BDOS/BIOS translation to Unix filesystem
- [dosemu](https://github.com/avwohl/dosemu) - MS-DOS emulator for Linux: dosbox-staging CPU + cpmemu-style syscall translation (intended test host for uc386)
- [qxDOS](https://github.com/avwohl/qxDOS) - DOS emulator for iPad and Mac — DOSBox-based with SwiftUI interface
- [uc80](https://github.com/avwohl/uc80) - C23 compiler targeting Z80 processor and CP/M; sibling backend sharing the uc_core frontend
- [uc_core](https://github.com/avwohl/uc_core) - Shared C23 frontend and AST optimizer used by uc80 and uc386
- [um80_and_friends](https://github.com/avwohl/um80_and_friends) - Microsoft MACRO-80 compatible toolchain for Linux: assembler, linker, librarian, disassembler (the Z80 analogue of what uc386 needs for i386)
- [upeepz80](https://github.com/avwohl/upeepz80) - Z80 peephole optimizer (template for an eventual upeep386)
- [uplox](https://github.com/avwohl/uplox) - Parser/lexer-table generator that produces uc_core's C23 frontend (from `examples/c23.uplox`)

## License

GPL-3.0-or-later.
