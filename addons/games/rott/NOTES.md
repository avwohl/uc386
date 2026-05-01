# Rise of the Triad port notes

ROTT (Apogee 1994) is a Doom-engine derivative with significantly
more Watcom-specific assembly and DOS DPMI hooks than the id Tech 1
games. Source release:

- Original: <https://github.com/icculus/rott>
- License: GPL-2.0 (since 2002 release).

Blockers:

- Heavier `#pragma aux` use than Doom — Phase 2 territory.
- DOS DPMI sound code that's not stubbed in any port; needs manual
  no-opping inside the build script's `-D` set.

Status: pending Doom (engine baseline) + uc_core Phase 2.
