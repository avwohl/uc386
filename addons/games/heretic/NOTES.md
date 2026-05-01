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

**ALL 47 of 47 src/heretic/*.c sources compile cleanly** through
uc386 once `--include-file stdarg.h` is passed (chocolate-doom's
txt_main.h declares TXT_vsnprintf with va_list but doesn't include
stdarg.h itself).

**Multi-file build confirmed working** (uc386@3534ad7): all 40
non-platform source files merge through a single uc386 invocation
with `--include-file stdarg.h` (chocolate-doom's txt_main.h
forgets to include stdarg.h itself) and `PROGRAM_PREFIX=""` in
config.h. Output asm is small only because our stub `main()`
doesn't call into Heretic; asm-DCE strips the unreferenced
symbols. Switching the stub to `D_DoomMain()` would surface the
remaining link-time work (I_* stubs, runtime).

The same engine + libc work as Doom carries through — a
`doom_stubs.c`-style file is the last deliverable to actually
boot a bin.
