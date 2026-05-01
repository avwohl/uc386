# Rise of the Triad port notes

ROTT (Apogee 1994) is a Doom-engine derivative with significantly
more Watcom-specific assembly and DOS DPMI hooks than the id Tech 1
games. Source release:

- Original: <https://github.com/icculus/rott>
- License: GPL-2.0 (since 2002 release).

Blockers:

- Heavier `#pragma aux` use than Doom — Phase 2 territory.
- DOS DPMI sound code that's not stubbed in any port; needs manual
  no-opping inside the build script's `-D` set.

## Status (2026-05-01)

`fetch.sh` works (videogamepreservation/rott mirror).
`build.sh` is a per-file triage harness like Duke3D's.

Triage result: **50 of 53** game-side .C files compile cleanly via
uc386. Driven by:
- 9 new DOS-platform libc headers: `dos.h`, `bios.h`, `conio.h`,
  `i86.h`, `mem.h`, `libc.h`, `process.h`, `direct.h`, `graph.h`.
  Each is real declarations + stub semantics — dos_emu doesn't
  simulate ports / IRQs / video, so the runtime impls return
  0 / no-op.
- `dos.h` Watcom REGS layout (`.x` / `.w` / `.h` views), DOS
  time/date/diskinfo structs, and `_HARDERR_*` macros.
- `sys/stat.h` adds Watcom-style `S_IREAD` / `S_IWRITE` aliases.
- `__WATCOMC__=1100` and `__386__` predefined in build.sh so
  `memcheck.h`'s compiler-dispatch picks up the Watcom branch
  (not `#error Unknown compiler`).
- uc_core preprocessor: backslash-in-include normalization
  (`<sys\stat.h>`) + case-insensitive include lookup
  (period DOS code freely mixes case).

Remaining 3 fails:
- `RT_TEXT.C:1471` — uc_core parser corner case (file is large).
- `TEXTURE.C` — `scan_t` typedef missing (used without
  `#include`; likely a Watcom-builtin or similar).
- `RT_SOUND.C` — bails silently; needs investigation.

Earlier fails resolved this session:
- `_rt_build.h` / `rt_spball.h` upstream typos: fetch.sh now
  patches the `#include` lines after extraction.
- `S_IREAD` and `errno` unknown: `<io.h>` shim now pulls in
  `<sys/stat.h>` and `<errno.h>`, matching DOS-era expectations.
- `byte` / `int32` / `fixed` undefined: build.sh predefines
  via `-D` flags, plus `--include-file dos.h` brings in the
  `uchar` typedef chain.

The deeper engine blocker (`#pragma aux` for `mulscale<n>` etc.)
isn't surfacing here because most of the affected files compile
through anyway — `pragmas.h`-style fallbacks. Multi-file linkage
is the next ticket; expect new errors once the merged AST sees
inter-file inconsistencies.
