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

## Status

Blocked on uc_core. `build.sh` exits with status 1 today.
