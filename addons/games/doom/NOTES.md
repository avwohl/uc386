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
- `<values.h>` — added libc shim forwarding to `<limits.h>` / `<float.h>`
- `<alloca.h>` — added (alloca → malloc, leaks bounded by use-pattern)
- `<malloc.h>` — added (forwards to `<stdlib.h>`)
- `R_OK`/`access()` — added to `<unistd.h>`
- File-scope `static` name collisions (every .c has `static const char
  rcsid[]`) — fixed via per-file mangling pass in main.py before TU merge

Current first remaining blocker:
- Anonymous-struct type identity across TUs. When `typedef struct {...}`
  in a header is included by two .c files, the merged AST has two
  distinct struct-type nodes for the same logical type, so struct
  assignment fails ("got `__inline_X` and `__inline_Y`"). Fix path:
  unify named typedef'd structs at TU-merge time, OR teach codegen to
  compare struct types structurally instead of by identity. uc_core
  type-system change.

After that there will be more — but the work is now incremental
ticket-by-ticket rather than uniformly blocked. ETA-to-first-build is
no longer "weeks" — it's "however many of these tickets we close."
