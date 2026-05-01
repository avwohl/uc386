# Heretic / Hexen port notes

Heretic and Hexen share Doom's engine (Raven Software extended id's
codebase). Public source releases:

- Heretic: <https://github.com/chocolate-doom/chocolate-doom> (chocolate
  port preserves DOS-era semantics)
- Original Raven release: <https://github.com/id-Software/Heretic>

**License**: GPL-2.0 (id Tech 1 codebase since 1999).

Build prospects track Doom's — same engine, same blockers. Once Doom
builds, these are mostly recompiles + per-game asset paths.

## Status (2026-04-30)

`fetch.sh` works (pulls chocolate-doom which carries both Heretic and
Hexen). `uc386_config/config.h` is a hand-written stand-in for the
autotools-generated config.h chocolate-doom expects — claims
HAVE_DECL_STRCASECMP/STRNCASECMP, leaves all the optional features
(fluidsynth, libsamplerate, libpng, ALSA) undefined.

First-pass triage: `info.c` (the giant frame-state table) gets past
doomtype.h's `#include "config.h"` and starts parsing the
`state_t states[NUMSTATES]` literal but bails ~500 entries in with
"Expected type specifier" — likely uc386 closing the array literal
early. Worth a uc_core ticket.

Underlying engine is Doom's, so all the `doom_stubs.c` work applies
unchanged — `I_*` functions, `lseek`, `fstat`, the libc additions.
The remaining work after the parser fix: patch chocolate-doom's
autotools-driven Makefile-isms out of the source list, and decide
whether to share or fork doom_stubs.c for heretic's slightly
different sound/midi calls.
