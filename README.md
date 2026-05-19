# uc386

C23 compiler targeting the Intel 386 (i386 / x86-32) processor under a
DOS extender — specifically the **flat 32-bit Watcom / DOS/4GW-era** C
that early-to-mid-1990s PC games were written in.

**Status: 100% on both reference test suites.** All 1514 executable
[gcc-c-torture](https://github.com/llvm/llvm-test-suite) tests
compile, assemble, and run correctly under our DOS emulator;
all 220 [c-testsuite](https://github.com/c-testsuite/c-testsuite)
tests pass. The frontend (parsing, preprocessing, AST-level
optimization) lives in [uc_core](https://github.com/avwohl/uc_core);
this repo owns the driver, the x86-32 NASM emitter, and the DOS
runtime bindings. See `CLAUDE.md` for the per-slice development log.

**Highlights**: uc386 also produces real DOS `.exe` files — `addons/harness/exe.py`
drives `nasm -f obj` → `wlink system pmodew` to build self-contained
`.exe` (PMODE/W bound, ~17 KB stub floor) for any in-tree addon.
Validated end-to-end in CI: `true.exe` boots PMODE/W under DOSBox
0.74-3, runs the 32-bit code, exits with the correct errorlevel
(`false.exe` → 1, `true.exe` → 0); `argv_pr.exe alpha beta` parses the
DOS PSP via the bridge stub's ES-at-entry trick and reports
`argc=3 / argv[1]='alpha' / argv[2]='beta'`; `factor.exe 2 12 60 97`
emits multi-arg printf output (`2: 2 / 12: 2 2 3 / 60: 2 2 3 5 /
97: 97`) via the legacy in-asm format engine; `myecho.exe hello dos`
writes `hello dos\n` via libc fputs through real DOS handles. All
14 manifest-driven addons build .exe successfully (basename, cat,
dirname, echo, factor, false, head, open_test, strtol_test, tail,
true, wc, yes, argv_probe — .exe ~17 KB; all 17 also pass their
behavioral manifest test under `addons/test_gnu_addons.py`).
See `docs/path-a-mz-le.md`. DOOM boots end-to-end through uc386 → NASM → dos_emu
(reaches W_InitFiles after V_Init / M_LoadDefaults / Z_Init; exits 1
at WAD-not-found as expected; smoke-tested via
`addons/games/doom/test_doom_smoke.py`). MicroPython is a
fully-functional Python REPL, packaged separately as
[freedos_micro_python](https://github.com/avwohl/freedos_micro_python)
(`pip install freedos_micro_python`). The bundled
`freedos-micropython port` CLI builds an i386 DOS binary that
evaluates expressions, defines functions and classes, runs list
comprehensions, handles exceptions, and dispatches ~25 named builtins
(`print`, `min`, `max`, `sum`, `sorted`, `bin`, `hex`, `oct`, `len`,
`range`, `repr`, `type`, `isinstance`, ...). The MP smoke test
(`tests/test_micropython_integration.py`) re-uses uc386's `dos_emu`
to pin the REPL banner and is the toughest end-to-end exercise we
have for the compiler. BWK awk runs
arithmetic, regex, aggregation, and string functions
(`addons/gnu/awk-bwk/test_awk_smoke.py`). 16 in-tree GNU utilities
(`true`, `cat`, `wc`, ...) get parametrized regression coverage via
`addons/test_gnu_addons.py`. See `addons/STATUS.md` for the full
per-addon report.

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
| true    |         18 |     16,907 |  5,420 | 147,914 |
| echo    |        148 |     16,915 | 11,286 | 150,212 |
| factor  |      1,858 |     16,989 | 20,538 | 179,614 |
| wc      |      1,529 |     16,928 | 20,158 | 179,092 |

Reading this honestly:

- **`.bin` is not a DOS program.** It has no MZ header and runs
  only under `uc386.dos_emu`/a custom loader. It is the right
  metric for *codegen+DCE tightness* (and there uc386 is in a
  class of its own — tens of bytes), but it is not what you ship.
- **`.exe` is what you ship**, and it carries a ~17 KB PMODE/W
  extender floor. Against that real-DOS artifact, **Open Watcom
  is ~2–3× smaller on tiny programs** (its DOS/4GW clib + mature
  linker beat our extender floor); the two converge as real code
  grows. uc386 beats **DJGPP ~9×** and host **gcc ~2×**.
- So: uc386's *code generation* is extremely compact; its current
  *DOS packaging* (PMODE/W) is not yet competitive with Watcom's
  on small binaries. Both statements are true and the table shows
  which is which — no single "390× smaller" headline.

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

## Roadmap

### Phase 0 — hello world (current)
Emit enough assembly for `int main(){return 0;}` and a DOS INT 21h/4Ch
exit. Pick an assembler target (candidates: NASM, MASM, hand-rolled
`um386` paralleling `um80`).

### Phase 1 — syntactic tolerance for DOS-era cruft
Parse-and-ignore the non-standard keywords/pragmas that period headers
use. In flat-32 these are mostly no-ops — we just need the parser to
not choke on them. Lands in **uc_core** (shared with uc80). Includes:

- **Type qualifiers to ignore**: `near`, `far`, `huge`, `__near`,
  `__far`, `__huge`, `_cs`, `_ds`, `_es`, `_ss`, `_seg`, `__based(...)`
- **Calling-convention keywords**: `__cdecl`, `__pascal`, `__stdcall`,
  `__fastcall`, `__syscall`, `__watcall` (plus bare and `_`-prefixed
  variants). Accepted; all compile to the same ABI in Phase 1.
- **Function attributes**: `__interrupt`, `interrupt`, `__loadds`,
  `__saveregs`, `__export`
- **Pragmas to drop**: `hdrstop`, `hdrfile`, `warn`, `warning`,
  `intrinsic`, `function`, `check_stack`, `code_seg`, `data_seg`,
  `alloc_text`, `disable_message`, `argsused`, `inline`, `library`,
  `startup`, `exit`. (`pack` stays honored.)

### Phase 2 — Watcom real (the big one)
The survey says `#pragma aux` is the single feature that unlocks
Descent, Duke3D, ROTT, and Heretic/Hexen. It has two forms:

1. Describe calling convention for a named function:
   `#pragma aux f parm [eax] [edx] value [eax] modify [ecx];`
2. Define an inline-asm function body:
   `#pragma aux f = "add eax, edx" parm [eax] [edx] value [eax];`

Also in this phase: `__watcall` as a real ABI (first 4 args in
`EAX/EDX/EBX/ECX`), `_asm { }` Intel-syntax inline blocks.

### Phase 3 — optional gcc-compat
If we want Doom's public source (Linux port, DJGPP-style) or Quake
(also DJGPP): GCC-style `asm(...)`, `__attribute__((...))`, GAS `.S`
input.

### Phase 4 — integer codegen
32-bit int, 16-bit short, 32-bit pointer. Reuses
`uc_core.ast_optimizer` once TypeConfig lands in uc_core.

### Phase 5 — libc subset
printf / putchar / puts / file I/O via DOS INT 21h.

### Phase 6 — testing
Via [dosemu](https://github.com/avwohl/dosemu) or similar.

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
