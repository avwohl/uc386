# DPMI-era game ports

Build scripts AND built executables. Each game in this directory
ships in the **abandonware installer** (the second of the two
installers built on release) — see `addons/README.md` phase 8.

The original `docs/addons.txt` first read "we don't ship the built
executables due to licensing" — that's been revised. The public
source releases for these games are GPL-or-similar; the resulting
binaries are derivative works under those licenses, so shipping
them is fine. What we still **don't** ship: the data files (WAD,
GRP, RTL, etc.), which remain proprietary to the original
publisher. Users supply their own.

Each game directory contains:

- `fetch.sh` — downloads the upstream source archive into `upstream/`.
  Idempotent: skips if `upstream/` already exists.
- `build.sh` — invokes `uc386` on the upstream sources. Sets `-I`,
  `-D`, and the source list per the game's actual layout.
- `NOTES.md` — what we know about porting this game: what works,
  what doesn't, what libc functions are missing, license caveats.

Building is a **multi-day uc386 stress test** — these games average
30–80 KLoC of dense Watcom-flavored C with `#pragma aux` blocks,
inline asm, and assumptions about flat-32 `__watcall` calling
conventions that uc386 still parses-and-flattens (Phase 1) but
hasn't fully implemented (Phase 2).

Status today (2026-05-01):

| Game     | Source available    | Per-file triage | Boots? | Notes                                                                     |
|----------|---------------------|-----------------|--------|---------------------------------------------------------------------------|
| Doom     | yes (id Software)   | **58 / 58**     | **yes** | Boots end-to-end through dos_emu — exits at WAD-not-found (no WAD shipped) |
| Heretic  | yes (chocolate-doom)| **44 / 47**     | no     | Remaining 3 want richer SDL2 API; same engine as Doom, can share stubs.c    |
| Hexen    | yes (chocolate-doom)| **45 / 48**     | no     | Same SDL.h gap as Heretic                                                  |
| Duke3D   | yes (jfduke3d)      | **34 / 42**     | no     | game-side 15/16 + Build engine 19/26 — engine.c renderer compiles!         |
| ROTT     | yes (Apogee)        | **46 / 53**     | no     | Watcom DOS source — needed 9 new period libc headers                       |
| Descent  | yes (dxx-rebirth)   | n/a (C++)       | no     | dxx-rebirth is C++, uc386 is C-only; would need 1998 source release        |

**~227 source files** from period DOS games compile cleanly through
uc386 today. The "build scripts" no longer just document what's
missing — they're per-file triage harnesses that produce a
histogram of remaining errors. Each error is a concrete ticket;
many turn out to be small (2-line predefines, header shims).

The biggest remaining lever is **multi-file linkage**: each game's
files compile in isolation but haven't been linked together yet.
That'll surface cross-TU issues (struct identity already fixed for
Doom; others may follow). After that, runtime stubs to actually
boot — `doom_stubs.c` is the existing template.

## License

Each game's source is released under its own license — typically
GPL-2.0 (or similar) for the *code* but restrictive on the *data
files* (WAD, GRP, etc.). We ship the **binaries** (derivative
works of the public source under its license) but **not the
data files**. Users supply their own data files, drop them
alongside the binary, and run locally.

The fetch scripts download from each upstream's stated public URL.
If an upstream takes their archive offline, the corresponding
fetch.sh becomes a documentation artifact (the URL still tells you
what we tried to use).
