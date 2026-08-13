# uc386 addons

Optional packages built **on top of** uc386: GNU userland utilities,
DPMI-era games, and DOS installers. The compiler itself lives in
`../src/uc386/`; this directory is the "things you can build with it"
shelf.

**Releases**: <https://github.com/avwohl/uc386/releases> (latest tag
v0.2.0). `addons/harness/package.py` builds the tarballs:

- the FOSS addon tarball — the in-tree GNU userland binaries
  (`.bin`, plus `.exe` variants under `exe/`) + BWK awk.
- the games build-script tarball — fetch.sh / build.sh / NOTES.md /
  stubs.c / config shims for Doom + 5 games. No upstream sources
  and no game binaries ride along.
- `results.md` — uc386 vs gcc / Watcom / DJGPP. Read it with its
  own caveats: uc386's *flat `.bin`* is 50–10,000× smaller, but the
  shippable `.exe` carries a ~32.8 KB DOS/32A extender floor and is
  *larger* than Watcom's on small programs.

See `STATUS.md` for the per-item completion against `docs/addons.txt`.

## Layout

```
addons/
├── README.md          ← this file
├── STATUS.md          ← per-item completion report
├── harness/           ← Python build/test driver shared by everything
│   ├── build.py       ← compile+run addons under dos_emu
│   ├── exe.py         ← .asm → nasm OMF → upyle → MZ+LE .exe
│   ├── compare.py     ← size comparison, writes results.md
│   ├── package.py     ← release tarballs
│   ├── watcom_dosbox.py ← DOS-hosted Open Watcom under DOSBox-X
│   └── test_addons.py
├── gnu/               ← GNU-licensed userland (coreutils, awk, …)
│   └── <util>/        ← per-tool: source, manifest.toml, expected outputs
├── games/             ← DPMI-era game build scripts (no shipped binaries)
│   └── <game>/        ← per-game: fetch.sh, build.sh, NOTES
├── test_gnu_addons.py ← parametrized regression run over the manifests
└── results.md         ← size comparison: uc386 vs gcc/watcom/djgpp
```

The release installers are built by `package.py` at release time;
there is no checked-in `installer/` tree.

## Phases

The work split into nine tracked tasks. Order reflects dependency,
not priority — the foundation had to land first or none of the rest
was testable.

1. ✓ **Foundation** — directory layout, `harness/build.py`, plan
   (this file). Smoke test on `examples/hello.c`.
2. ✓ **Trivial in-tree utilities** — `true`, `false`, `echo`, `cat`,
   `yes` written from scratch in `addons/gnu/<name>/main.c`. Each
   exercises a slice of the runtime (exit codes, string output,
   stdin→stdout, looping with cap).
3. ✓ **argv plumbing** — `argv` reaches `main` both under
   `dos_emu.run()` and in a real `.exe`, where the bridge stub reads
   the DOS PSP command tail at `[es:0x80]`. Pinned by the
   `argv_probe` addon.
4. ◑ **Real GNU coreutils** — three `sbase` ports (`sbase-cat`,
   `sbase-head`, `sbase-tee`) plus BWK awk build from upstream
   source. A broader coreutils sweep is not done.
5. ✗ **gawk** — `addons/gnu/gawk/` is a doc-only stub (NOTES +
   fetch/build scripts, no working port). Still the tallest single
   port; needs a regex implementation in libc.
6. ✓ **Comparison build matrix** — gcc (host ELF baseline), Watcom
   (`wcc386`), and DJGPP columns recorded in `results.md` by
   `harness/compare.py`.
7. ✓ **Game build scripts** — Doom (id Software open release),
   Duke3D / Build (3D Realms), Heretic / Hexen (Raven), ROTT
   (Apogee). Scripts download upstream sources and build locally;
   we **do not ship the upstream sources**. Only Doom currently
   boots end-to-end — see `STATUS.md` and each `games/*/NOTES.md`
   for how far the others get.
8. ✓ **DOS installers** — two separate installers shipped via GitHub
   releases, built by `harness/package.py`:
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

9. ✓ **MicroPython port** — **moved out of this tree.** It now lives
   in its own repo,
   [freedos_micro_python](https://github.com/avwohl/freedos_micro_python).
   A working DOS Python REPL: expressions, `def`/`class`, list
   comprehensions, `try/except`, and the common builtins. It remains
   uc386's toughest end-to-end test; `tests/test_micropython_integration.py`
   is what's left here.

## Running the harness

```
.venv/bin/python -m addons.harness.build            # show usage
.venv/bin/python -m addons.harness.build smoke      # build+run examples/hello.c
.venv/bin/python -m addons.harness.build gnu echo   # one addon
.venv/bin/python -m addons.harness.build gnu all    # all of gnu/ (17/17)
.venv/bin/python -m addons.harness.build games      # game build scripts
```

Regression run over every manifest, as CI does it:

```
.venv/bin/pytest addons/test_gnu_addons.py          # 17 passed
```

## License notes

`addons/gnu/*` source written in this repo is GPL-3.0-or-later (matches
parent project). Anything we *download* keeps its upstream license —
the harness records it in each `addons/gnu/<util>/LICENSE` if available.
The abandonware installer ships only our build scripts, never the
upstream game sources.
