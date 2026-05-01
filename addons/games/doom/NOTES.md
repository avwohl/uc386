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

Status: **DOOM boots end-to-end under dos_emu.**

Build pipeline (`./build.sh`):
1. uc386 compiles 58 doom sources + `stubs.c` into a 2 MB .asm
2. NASM assembles to a 301 KB flat .bin
3. dos_emu loads + runs the .bin

Boot output as of 2026-04-30:

```
Game mode indeterminate.
                 Public DOOM - v1.10
V_Init: allocate screens.
M_LoadDefaults: Load system defaults.
Z_Init: Init zone memory allocation daemon.
W_Init: Init WADfiles.
I_Error: W_InitFiles: no files found
```

Exit point is `W_InitFiles` (not a uc386 limitation — we just don't
ship a WAD; we can't, license-wise). With a user-supplied WAD via
`vfiles_init`, DOOM would proceed into `R_Init` (rendering), then
`P_Init` (gameplay), then the title-screen tic loop. Reaching the
title screen requires a video stub that captures the 320x200x8
framebuffer (currently `I_FinishUpdate` is no-op); reaching gameplay
also requires the input pump (`I_StartTic` → produce ticcmds).

Compiler-side blockers cleared this session: 6 codegen / optimizer
fixes (above), 4 new libc headers, `lseek` + `strcasecmp` in libc
asm + dos_emu INT 21h AH=0x42 handler, and `getenv` recognizing
HOME / DOOMWADDIR.
