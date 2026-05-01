# Hexen port notes

Hexen — Raven's id-Tech-1-derived engine, separate from Heretic's.

**Upstream**: <https://github.com/id-Software/HEXEN>
**License**: GPL-2.0.

## Status (2026-05-01)

`fetch.sh` works (shares chocolate-doom upstream with Heretic).
`uc386_config` is a symlink to `../heretic/uc386_config` so the
generated `config.h` and `SDL_endian.h` shims are shared.
`build.sh` is a per-file triage harness like the other games'.

**ALL 48 of 48 hexen sources compile cleanly** through uc386 with
`--include-file stdarg.h`. The pieces that came together:
- uc_core preprocessor improvements (uc_core@63912fd):
  multi-line macro merge in _preprocess_included, comment-aware
  paren tracking, trailing-comment strip in _process_define
- SDL.h opaque shims (SDL_Window/Renderer/Texture/Surface/...)
- SDL_gamecontroller.h enum-only shim
- PROGRAM_PREFIX="" in config.h
- uc386 `--include-file` flag (gcc -include equivalent) so
  stdarg.h gets pulled in before chocolate-doom's textscreen
  headers reference va_list

Next: a hexen_stubs.c (I_* runtime, similar to doom_stubs.c) to
actually link.
