# Duke Nukem 3D / Build engine port notes

**Upstream**: jfduke3d (Jonathon Fowler's caretaker fork), based on
3D Realms' GPL'd release (2003).
**License**: GPL-2.0
**Tree of interest**: `upstream/source/` (game) + `upstream/build/`
(engine).

## Why this is the hardest of the games

Build engine is **the** showcase for `#pragma aux` Watcom inline-asm
fixed-point primitives. Functions like `mulscale<n>(x, y)` and
`scale(x, y, z)` are defined as 1–3 instruction asm bodies via
`#pragma aux NAME = "..." parm [...] value [...]`. Without uc_core
Phase 2, the parser ignores the asm body and the linker errors on
unresolved symbols (or the parser parses the surrounding C and the
program runs with garbage where the asm should be).

Phase 2 is the **#pragma aux + `__watcall` ABI** slice in the README's
roadmap. Until that lands, Build engine is at best partially
buildable — the engine framework compiles, the math primitives don't.

## What works without Phase 2

- Game-side C that doesn't touch the engine math primitives
- Top-level menu / file I/O — straight POSIX
- Sound system replacement (we'd no-op it anyway under dos_emu)

## What needs Phase 2

- All the `kvxlist.c` / `engine.c` math primitives
- `__watcall` ABI for engine entry points (needs codegen support)

## Status (2026-04-30 triage)

`fetch.sh` now uses `git clone --recurse-submodules` so the
`jfbuild` / `jfaudiolib` / `jfmact` siblings actually populate
(the github tarball doesn't include submodules). Verified locally:
22 .c files in `upstream/src/` after fetch.

First-pass triage with the same setup that built DOOM:

```
version.c   — compiles clean (smoke).
all others  — bail at  jfbuild/include/compat.h:147
              "#error Unknown endianness"
```

The Build engine's compat.h does an OS-keyed endianness lookup
(`__linux`, `__APPLE__`, `_WIN32`, BSDs). uc386 doesn't claim any
of those, so neither `B_LITTLE_ENDIAN` nor `B_BIG_ENDIAN` gets
defined and the `#error` fires. Cleanest fix: add `-D B_LITTLE_ENDIAN=1
-D B_BIG_ENDIAN=0` to build.sh (or define an
`__UC386_LITTLE_ENDIAN__` predefine and patch compat.h's chain).

Beyond endianness, the deeper blocker is unchanged: `#pragma aux`
math primitives (`mulscale<n>`, `scale`) in `jfbuild/engine.c`.
Phase 2 of uc_core is the prereq there. The game-side C in
`upstream/src/` is approachable in isolation (it's mostly straight
C99) but it calls into the engine's primitives, so any meaningful
build needs both.

Next iteration: pin down endianness + try compiling enough Build
engine sources to see how many `#pragma aux` gates we hit before
dropping into runtime work.
