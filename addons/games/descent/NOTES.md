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

Status: longest-horizon target. Documented here for completeness.
