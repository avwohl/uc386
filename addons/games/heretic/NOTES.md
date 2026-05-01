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

`uc386_config/` carries four hand-written shims:
- `config.h` — autotools stand-in (PACKAGE_NAME, HAVE_DECL_*,
  PROGRAM_PREFIX, etc.)
- `SDL_endian.h` — identity LE byte-swaps (uc386 is little-endian)
- `SDL.h` — opaque SDL_Event/Window/Renderer/Surface types + a
  subset of constants
- `SDL_gamecontroller.h` — opaque controller / joystick types

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

The same engine + libc work as Doom carries through, BUT
chocolate-doom has a much bigger platform abstraction layer
than original DOOM:
- 76 distinct `I_*` functions (vs DOOM's 30)
- 30 `TXT_*` textscreen functions
- 8 `SDL_*` calls

A heretic_stubs.c in the spirit of doom_stubs.c is the last
deliverable to link a bin — but it's a multi-day port, not a
single-tick task. Doom's stubs.c (~30 functions) is the model.

## Boot-link gap, quantified (2026-05-01)

A holistic compile of the 47 sources + a stub `main → D_DoomMain`
through uc386 produces a 2.5 MB / 84,397-line `.asm` and surfaces
exactly **265 unique extern symbols** that NASM's flat-binary mode
cannot resolve (`binary output format does not support external
references`). Triaging by `call _name` references in the asm:

- **138 functions** the engine calls but doesn't define — every
  `I_*` (47), `M_*` (28), `W_*` (16), `TXT_*` (10), `V_*` (9),
  `D_*` (8), `Z_*` (6), `DEH_*` (6), plus 8 misc.
- **127 variables** referenced as globals (or referenced only via
  address-take, never `call`) — 74 `key_*` (input bindings), 47
  `I_*`-namespaced globals like `I_VideoBuffer` and lookup tables
  like `finecosine` / `finesine` / `finetangent`, plus 6 timing
  ints and the rest as misc.

A naive auto-generated stubs.c (every function `int X(void) { return
0; }`, every variable `int X[1]`) re-introduces a real-vs-stub type
collision (heretic sources declare some of these `extern fixed_t
foo[N]`; my stub `int X[1]` clashes with `fixed_t X[N]` on multi-TU
merge). The right fix is the same labor as Doom's stubs.c: write
real definitions one by one, sourced from chocolate-doom's i_*.c /
txt_*.c, with the actual prototypes and array sizes. ~3-5 days of
mechanical porting work.

**Hexen is downstream of Heretic** — the same 265 stubs cover
Hexen too once we factor out the per-game lump table differences.
