# Heretic / Hexen port notes

Heretic and Hexen share Doom's engine (Raven Software extended id's
codebase). Public source releases:

- Heretic: <https://github.com/chocolate-doom/chocolate-doom> (chocolate
  port preserves DOS-era semantics)
- Original Raven release: <https://github.com/id-Software/Heretic>

**License**: GPL-2.0 (id Tech 1 codebase since 1999).

Build prospects track Doom's — same engine, same blockers. Once Doom
builds, these are mostly recompiles + per-game asset paths.

## Status (2026-05-01)

`fetch.sh` works (pulls chocolate-doom which carries both Heretic and
Hexen). `build.sh` is a per-file triage harness like the other games'.

`uc386_config/` carries three hand-written shims:
- `config.h` — autotools stand-in (PACKAGE_NAME, HAVE_DECL_*, etc.)
- `SDL_endian.h` — identity LE byte-swaps (uc386 is little-endian)
- `SDL.h` — minimal opaque types so SDL_Event*-typed APIs parse

**44 of 47 src/heretic/*.c sources compile cleanly** through uc386
after the uc_core preprocessor improvements (uc_core@63912fd) and
the SDL.h shim. Remaining 3 fails are deeper SDL2 API references.

The same engine + libc work as Doom carries through — `doom_stubs.c`
is the next deliverable to actually link a bin. The remaining work:
1. Expand SDL.h shim or make it pointer-only opaque
2. Add a heretic-flavored stubs.c (or share doom's with #ifdef)
3. Verify multi-file linkage works end-to-end
