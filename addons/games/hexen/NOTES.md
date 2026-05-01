# Hexen port notes

Hexen — Raven's id-Tech-1-derived engine, separate from Heretic's.

**Upstream**: <https://github.com/id-Software/HEXEN>
**License**: GPL-2.0.

## Status (2026-05-01)

`fetch.sh` works (shares chocolate-doom upstream with Heretic).
`uc386_config` is a symlink to `../heretic/uc386_config` so the
generated `config.h` and `SDL_endian.h` shims are shared.
`build.sh` is a per-file triage harness like the other games'.

**44 of 48 hexen sources compile cleanly** through uc386 after the
uc_core preprocessor improvements (uc_core@63912fd):
- multi-line macro merge in _preprocess_included
- comment-aware paren tracking in _has_unclosed_macro_call
- comment-first scan in _expand_macros_once and _parse_macro_args
- trailing-comment strip in _process_define

Remaining 4 fails are SDL.h missing (chocolate-doom uses SDL2
directly for input). To build Hexen end-to-end we'd need an SDL.h
shim (similar to the SDL_endian.h we already have) plus the
matching doom_stubs.c-style I_* implementations.
