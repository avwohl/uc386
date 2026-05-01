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

## Status (2026-04-30 triage v2)

After several rounds of compiler + libc fixes (see commits this
session), per-file triage stands at:

**Game-side (`upstream/src/`):** 15 of 16 clean.
- Bail: `menues.c` — assigns to a `char *s[]` after a same-name
  `static const char *s[]` from a sibling `case` block. uc386 has
  flat function scope, so it doesn't see the inner `s` as
  block-local. uc_core scope-fix is the right answer; for now
  exclude this file from the linked build.

**Build engine (`upstream/jfbuild/src/`):** 19 of 26 clean —
includes `engine.c` (the renderer!) and `pragmas.c` (the math
primitives, even though some are still placeholders without
`#pragma aux`).
- Bail (platform headers, expected): `compat.c` (dirent.h),
  `kplib.c` (io.h), `mmulti.c` (netinet/in.h), `sdlayer2.c`
  (SDL.h), `startgtk_editor.c` (gtk/gtk.h)
- Bail (Windows-only): `startwin_editor.c`, `winlayer.c`

**35 of 42 source files compile through uc386 today.** Surprise:
much of the Build engine's "needs Phase 2" is actually fine
without `#pragma aux` because `pragmas.c` provides plain-C
fallbacks. The renderer (`engine.c`) compiles. The hard yards
between here and a runnable Duke3D binary are now: (1) write
`stubs.c` for the platform-specific subsystems we excluded
(input, video, sound, networking, file I/O); (2) link everything
in multi-file mode and triage cross-TU issues; (3) provide
GRP/CON loading shims under `vfiles_init`.

Compiler improvements this iteration:
- `char arr[N] = {"string"}` brace-around-string init unwrap
- `__GNUC_MINOR__` / `__GNUC_PATCHLEVEL__` predefines
- `div_t div(int, int)` returns by value (libc shim + standard
  header signature)
- duke3d build.sh added a per-file triage mode
