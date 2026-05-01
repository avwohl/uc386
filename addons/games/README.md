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

Status today (2026-04-30):

| Game            | Source available  | Compiles? | Runs? | Notes |
|-----------------|-------------------|-----------|-------|-------|
| Doom            | yes (id Software) | not yet   | no    | Linux source compiles, DOS-source via dmx.c needs DPMI int handling |
| Duke3D          | yes (3D Realms)   | not yet   | no    | Build engine + game.c — heavy `#pragma aux`, blocked on Phase 2 |
| Heretic         | yes (Raven)       | not yet   | no    | Same Doom-derived engine; likely tracks Doom progress |
| Hexen           | yes (Raven)       | not yet   | no    | Same |
| ROTT            | yes (Apogee)      | not yet   | no    | Watcom-specific extensions |
| Descent         | yes (Parallax)    | not yet   | no    | Source release covers DOS only — heavy DPMI |

The "build scripts" exist as scaffolding; running them today produces
compile errors that document what's missing. Each error becomes a
ticket against uc_core (Phase 2 `#pragma aux`) or this repo (libc
extensions, codegen for specific patterns).

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
