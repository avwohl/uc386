# DOOM port notes

**Upstream**: <https://github.com/id-Software/DOOM>
**License**: GPL-2.0 (since 1999 re-release; original 1997 source was
under id's "research only" license, replaced by GPL).
**Tree of interest**: `linuxdoom-1.10/`

## What we expect to need

DOOM linuxdoom-1.10 is mostly straight C with these dependencies:

- `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<ctype.h>` — have these
- `<unistd.h>` — minimal; `read`/`write`/`close` provided
- `<fcntl.h>` — basic flags; need `O_BINARY` (no-op under uc386)
- `<sys/types.h>`, `<sys/stat.h>` — `stat()` is missing
- `<errno.h>` — global errno provided by recent libc additions
- Linux X11 / sound: `i_video.c`, `i_sound.c` need stubs that draw
  to a frame buffer / no-op respectively
- Allocator stress: DOOM's zone allocator is its own; libc malloc is
  only used briefly

## Expected blockers (today)

1. `stat(2)` — used to check WAD file existence. Add as libc stub or
   patch DOOM to skip the stat probe.
2. `gettimeofday(2)` / `clock(3)` — DOOM uses for frame timing. Stub
   out to a counter that increments per call.
3. `signal(SIGINT, …)` — works, but our handler dispatch is limited.
4. `i_video.c` X11 calls — replace with a uc386-side stub that
   collects a 320×200×8 frame buffer and prints a hash to stdout
   (so the test rig can validate without a display).
5. The DOS-only `dmx.c` (sound) is NOT in `linuxdoom-1.10`; use the
   Linux subset to avoid DPMI sound entirely.

## Status (2026-04-30)

`fetch.sh` works (~66 C files pulled from id-Software/DOOM master).
`build.sh` excludes `i_net.c` / `i_sound.c` / `i_video.c` / `i_system.c`
(BSD sockets / Linux DSP / X11 — need stub replacements rather than
upstream compile) and gets 58 sources into the compiler.

Blockers cleared so far:
- `<values.h>`, `<alloca.h>`, `<malloc.h>`, `R_OK`/`access()` — libc
- File-scope `static` name collisions (every .c has `static const char
  rcsid[]`) — fixed via per-file mangling pass in main.py before TU merge
- Anonymous-struct type identity across TUs — `_resolve_struct_name`
  now uses a structural fingerprint (MD5 of member shape) instead of
  Python `id(t)`, so the same `typedef struct {...}` in a header
  collapses to one registered layout across all including TUs.
- Float-typed initializers in integer globals (`(.2 * FRACUNIT)` style
  fixed-point constants) — `_emit_global_init` falls back to the float
  evaluator when the int evaluator gives up; truncates toward zero.
- Bit-shift subexpressions in float const-eval — `_const_eval_float`
  delegates to `_const_eval` when its early integer try succeeds,
  picking up `<<`, `>>`, `&`, `|`, `^` for free.
- `(int)"string"` and `(int)&global` in int-typed global init slots —
  recognized as a label diff, lays down `dd <label>`. DOOM uses this
  in m_misc.c chatmacro defaults.
- Strength reduction `x * 2^n -> x << n` was firing on float operands
  (when `other` was a `UnaryOp(-, FloatLiteral)`, not a top-level
  FloatLiteral). uc_core ast_optimizer now bails on any subtree
  containing a FloatLiteral / float-Cast.

Status: all 58 sources compile end-to-end through uc386 to a single
76 K-line .asm file. Remaining work to actually run is a
**doom_stubs.c** file providing definitions for the 30+ `_I_*`
externs (the platform-specific functions we excluded:
`I_StartFrame`, `I_GetTime`, `I_Error`, `I_InitGraphics`, etc.) plus
a couple of libc additions (`fstat`, `mkdir`, `sscanf`,
`sndserver_filename`/`mb_used` variable stubs).

Once that exists, NASM produces a single flat `.bin` and the question
becomes "does it run under dos_emu" — likely a fresh batch of runtime
issues, but qualitatively different (no longer a compile-time wall).
