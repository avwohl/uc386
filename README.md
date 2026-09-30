# uc386

> **AI — no code written by a primate.**

C23 compiler targeting the Intel 386 (i386 / x86-32) processor under a
DOS extender — specifically the **flat 32-bit Watcom / DOS/4GW-era** C
that early-to-mid-1990s PC games were written in.

**Status: working and released — `pip install uc386` (0.2.1 on
[PyPI](https://pypi.org/project/uc386/)).** Measured under our DOS
emulator (compile → assemble → run → diff): **215 / 220**
[c-testsuite](https://github.com/c-testsuite/c-testsuite) and, with
`--kr`, **1397 / 1514**
[gcc-c-torture](https://github.com/llvm/llvm-test-suite) executable
tests pass. [docs/status.md](https://github.com/avwohl/uc386/blob/main/docs/status.md) lists the remaining
failures.

The frontend (parsing, preprocessing, AST-level optimization) lives
in [uc_core](https://github.com/avwohl/uc_core); this repo owns the
driver, the x86-32 NASM emitter, and the DOS runtime bindings.

## Features

- Strict C23 by default. `--kr` rewrites pre-ANSI K&R code and GNU
  computed goto into standard C before parsing.
- Self-contained DOS `.exe` files bound to the DOS/32A extender,
  boot-tested under DOSBox in CI.
- Compiles real third-party programs: DOOM, MicroPython, Kernighan's
  awk, and 17 GNU utilities.
- Compact code generation: `true` is 18 bytes as a flat `.bin`. A DOS
  `.exe` carries a ~32.8 KB DOS/32A extender floor
  ([docs/size.md](https://github.com/avwohl/uc386/blob/main/docs/size.md)).
- Target: unmodified Watcom-era DOS games such as Descent and Duke
  Nukem 3D ([docs/design.md](https://github.com/avwohl/uc386/blob/main/docs/design.md)).

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
[`docs/path-a-mz-le.md`](https://github.com/avwohl/uc386/blob/main/docs/path-a-mz-le.md).

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
[`CLAUDE.md`](https://github.com/avwohl/uc386/blob/main/CLAUDE.md) for that layout.

macOS (Homebrew) and Fedora/RHEL (dnf) instructions, plus the
optional toolchains for addon builds (bison/flex) and the
DJGPP / OpenWatcom comparison columns, are documented in
[`docs/INSTALL.md`](https://github.com/avwohl/uc386/blob/main/docs/INSTALL.md).

## Documentation

- [docs/status.md](https://github.com/avwohl/uc386/blob/main/docs/status.md) - suite results, `--kr`, highlights, libc status
- [docs/size.md](https://github.com/avwohl/uc386/blob/main/docs/size.md) - executable sizes against Open Watcom and DJGPP
- [docs/design.md](https://github.com/avwohl/uc386/blob/main/docs/design.md) - goal, non-goals, and what this repo contributes
- [docs/INSTALL.md](https://github.com/avwohl/uc386/blob/main/docs/INSTALL.md) - per-platform install and optional toolchains
- [docs/path-a-mz-le.md](https://github.com/avwohl/uc386/blob/main/docs/path-a-mz-le.md) - the MZ+LE `.exe` build path
- [docs/dosiz-integration.md](https://github.com/avwohl/uc386/blob/main/docs/dosiz-integration.md) - dosiz as a test runner
- [addons/STATUS.md](https://github.com/avwohl/uc386/blob/main/addons/STATUS.md) - per-addon report
- [STANDARD_C_BACKLOG.md](https://github.com/avwohl/uc386/blob/main/STANDARD_C_BACKLOG.md) - standard-C backlog
- [CHANGELOG.md](https://github.com/avwohl/uc386/blob/main/CHANGELOG.md) - release changes
- [docs/changes.md](https://github.com/avwohl/uc386/blob/main/docs/changes.md) - historical development log

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
