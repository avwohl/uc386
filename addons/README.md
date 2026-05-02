# uc386 addons

Optional packages built **on top of** uc386: GNU userland utilities,
DPMI-era games, and DOS installers. The compiler itself lives in
`../src/uc386/`; this directory is the "things you can build with it"
shelf.

**Latest release**: <https://github.com/avwohl/uc386/releases/tag/v0.1.1-dev>
(superseding v0.1.0-dev). Ships:

- `uc386-foss-addons-v0.1.1-dev.tar.gz` — 16 GNU userland binaries
  + BWK awk (~62 KB total).
- `uc386-games-build-scripts-v0.1.1-dev.tar.gz` — fetch.sh /
  build.sh / NOTES.md / stubs.c / config shims for Doom + 5
  games. 19 KB after the package.py upstream-exclude fix.
- `results.md` — uc386 vs gcc / Watcom / DJGPP across all 13
  FOSS addons (uc386 is 50–10,000× smaller).

See `STATUS.md` for the per-item completion against `docs/addons.txt`.

## Layout

```
addons/
├── README.md          ← this file (the plan)
├── harness/           ← Python build/test driver shared by everything
│   └── build.py
├── gnu/               ← GNU-licensed userland (coreutils, gawk, …)
│   └── <util>/        ← per-tool: source, build script, expected outputs
├── games/             ← DPMI-era game build scripts (no shipped binaries)
│   └── <game>/        ← per-game: fetch.sh, build.sh, NOTES
├── installer/         ← DOS installers built on release
│   ├── foss/          ← bundles built FOSS binaries (GPL et al)
│   └── abandonware/   ← ships only build scripts; user fetches sources
└── results.md         ← size comparison: uc386 vs gcc/watcom/djgpp
```

## Phases (this is what /loop docs/addons.txt is iterating over)

The work splits into nine tracked tasks (`TaskList` in the harness).
Order reflects dependency, not priority — the foundation has to land
first or none of the rest is testable.

1. **Foundation** — directory layout, `harness/build.py`, plan (this
   file). Smoke tests on `examples/hello.c`.
2. **Trivial in-tree utilities** — `true`, `false`, `echo`, `cat`,
   `yes` written from scratch in `addons/gnu/<name>/main.c`. Each
   exercises a slice of the runtime (exit codes, string output,
   stdin→stdout, looping with cap). No external download yet.
3. **argv plumbing** — wire `argv` from `dos_emu.run()` through a
   DOS-PSP-style command tail at `[PSP+0x80]` (or pre-pushed
   stack args) so `cat file1 file2` actually receives its args.
4. **Real GNU coreutils** — download upstream, compile a sensible
   subset unmodified or with minimal patches. Skip tools that have
   no DOS analog (`chroot`, `groups`, `id`, `who`, …).
5. **gawk** — explicit ask in `docs/addons.txt`. Likely the
   tallest single port; needs a regex implementation we may have
   to add to libc.
6. **Comparison build matrix** — for each FOSS addon, also build
   with gcc (host ELF, baseline), Watcom (`wcc386` → flat-32
   `.exe`), and djgpp (DJGPP → COFF → DOS). Record sizes side by
   side in `results.md`.
7. **Game build scripts** — Doom (id Software open release), Duke3D
   / Build (3D Realms), Heretic / Hexen (Raven), ROTT (Apogee).
   Scripts download upstream sources and build locally; we **do
   not ship the binaries** because the source-release archives'
   licensing is ambiguous-to-restrictive on redistribution.
8. **DOS installers** — two separate installers shipped via GitHub
   releases:
   - **FOSS installer**: built GNU userland binaries (clear GPL
     provenance, OK to redistribute). The release tarball also
     ships per-addon source + `manifest.toml` + a top-level
     `test_addons.py` that exercises each binary under
     `uc386.dos_emu` and asserts stdout/exit against the manifest.
   - **Abandonware installer**: built game binaries from each game's
     public-source release. Original source releases (Doom, Duke3D,
     Heretic, Hexen, ROTT) ship under GPL or similar; the resulting
     binaries are derivative works under those licenses. We DO
     ship the binaries — but **never the data files** (WAD, GRP,
     etc.), which remain proprietary. Users supply their own.
     Build scripts also ride along so users can rebuild from source
     if they don't trust our binaries. Today's tarball ships
     `bin/doom/doom.bin` (the only game that boots end-to-end);
     the rest will join as their link-time gaps close.

9. **MicroPython port** — `addons/gnu/micropython/`. Today a
   fully-functional Python REPL: `build_port.sh` produces a
   ~169 KB i386 DOS binary that evaluates expressions, runs
   `def`/`class`/list comprehensions, handles `try/except`, and
   dispatches ~25 named builtins (`print`, `min`, `max`, `sum`,
   `sorted`, `bin`, `hex`, `oct`, `len`, `range`, `repr`, `type`,
   `isinstance`, ...). 12 smoke tests in
   `test_micropython_smoke.py` pin the wins.
   145 / 145 sources (`upstream/py/` + `upstream/shared/`) compile
   cleanly through uc386 → NASM in the per-file triage; the
   multi-TU build links a 1.94 MB .asm to flat .bin via NASM.
   See `addons/gnu/micropython/NOTES.md`.

## Running the harness

```
.venv/bin/python -m addons.harness.build  # show usage
.venv/bin/python -m addons.harness.build smoke      # build hello.c
.venv/bin/python -m addons.harness.build gnu echo   # build addons/gnu/echo
.venv/bin/python -m addons.harness.build gnu --all  # all gnu/* dirs
```

## License notes

`addons/gnu/*` source written in this repo is GPL-3.0-or-later (matches
parent project). Anything we *download* keeps its upstream license —
the harness records it in each `addons/gnu/<util>/LICENSE` if available.
The abandonware installer ships only our build scripts, never the
upstream game sources.
