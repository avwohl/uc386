# Descent port notes

Descent (Parallax 1995) — first true 3D shooter, Watcom flat-32 DOS
game. Source available from the d1x / dxx-rebirth caretaker forks:

- Source: <https://github.com/dxx-rebirth/dxx-rebirth>
  (modernized; needs SDL — too far from a clean DOS rebuild)
- Original DOS source release: 1998, hosted at <https://archive.org>
  under "Descent source code release".
- License: original release was source-available with a non-redistribution
  clause; later releases under MIT-style.

This is the single hardest target — the entire 3D math + DPMI setup
is the demonstration vehicle for Watcom flat-32 in 1995. Phase 2's
`__watcall` and `#pragma aux` are required; we'll likely also need
real interrupt-service-routine support that we currently parse-and-drop.

## Status (2026-05-01)

`fetch.sh` clones dxx-rebirth (modernized fork, builds today) but
that tree is **C++** end-to-end — 112 `.cpp` files plus 7 plain `.c`
support utilities. uc386 is C-only, so even with the new period-DOS
header shims, the game body never reaches the parser.

Two paths to revive this target:
1. Switch fetch.sh to the **1998 Parallax DOS source release**
   (plain C with Watcom `#pragma aux`). Hosted in mirrors like
   archive.org's "Descent source code release"; no clean GitHub
   home that we know of. Needs uc_core Phase 2 (`#pragma aux`).
2. Add C++ frontend to uc_core. Out of scope; uc386 is C-only.

Status: longest-horizon target. Other DPMI-era games are
prioritized first.
