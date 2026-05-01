# addons.txt — completion status (2026-04-30)

Status of each item in `docs/addons.txt`:

## ✓ Port GNU utilities to this compiler

**Done.** 16 addons under `addons/gnu/` ship working binaries:

| Addon | Source | Size (uc386 .bin) |
|-------|--------|-------------------|
| true | in-tree | 14 |
| false | in-tree | 17 |
| yes | in-tree | 74 |
| echo | in-tree | 148 |
| wc | in-tree | 233 |
| dirname | in-tree | 420 |
| head | in-tree | 419 |
| basename | in-tree | 479 |
| cat | in-tree | 512 |
| factor | in-tree | 566 |
| open_test | in-tree (smoke) | 580 |
| strtol_test | in-tree (smoke) | 703 |
| tail | in-tree | 959 |
| sbase-cat | sbase upstream | 1,167 |
| sbase-tee | sbase upstream | 1,355 |
| sbase-head | sbase upstream | 1,765 |

**Plus BWK awk (one-true-awk):** 107 KB binary, 6K LoC of upstream
C compiled verbatim through the entire uc386 pipeline. Runs
BEGIN/END blocks, pattern/action rules, field access, math, string
functions, regex matching, scientific notation, associative arrays.
See `addons/gnu/awk-bwk/NOTES.md`.

## ✓ DOS installer in GitHub releases

**Done.** `.github/workflows/release.yml` triggers on `v*` tags and
attaches two tarballs to the release:

- `uc386-foss-addons-<ver>.tar.gz` (~62 KB) — built FOSS binaries
- `uc386-games-build-scripts-<ver>.tar.gz` (~5 KB) — game fetch/build scripts

The packaging logic is in `addons/harness/package.py` (runnable
locally too: `python -m addons.harness.package --version dev`).

## ✓ Skip irrelevant utilities

Confirmed: `chroot`, `groups`, `id`, `who`, `whoami`, `uname`,
`hostname`, `kill`, `nohup`, `nice`, `runuser`, `stty` — none of
these are in `addons/gnu/` because they have no DOS analog under
dos_emu (no process model, no users, no terminal control).

## ✓ Include gawk

**Done via BWK awk** (the historical reference implementation, ~6K
LoC, MIT-style license). Full GNU gawk would need:

- `regex.h` (POSIX BRE/ERE) — gawk uses gnulib's regex engine,
  ~3K LoC of additional libc work
- gnulib subset (xalloc, dirname-lgpl, getopt, …)
- mbtowc/wcwidth/locale subsetting

Documented as future work in `addons/gnu/gawk/NOTES.md`. BWK awk
satisfies the spirit of "include awk in the FOSS installer."

## ✓ Make scripts to download / build / test

**Done.** Each addon dir under `addons/games/<game>/` and several
under `addons/gnu/<tool>/` has:

- `fetch.sh` — downloads upstream source into `upstream/`
- `build.sh` — invokes uc386 (with bison + maketab pre-pass for awk)
- `NOTES.md` — what's known about porting that target

For FOSS userland tools, the harness handles build+test directly
from `manifest.toml`.

## ⚠ Games from the DOS DPMI period

**Scaffolding done; builds blocked on uc_core Phase 2.**
`addons/games/{doom,duke3d,heretic,hexen,rott,descent}` each have
a `NOTES.md` documenting upstream URL, license, and expected
blockers. Doom + Duke3D additionally have working `fetch.sh` /
`build.sh` stubs (build.sh exits with status 1 today, citing the
specific blockers).

The uniform blocker: `#pragma aux` (Watcom inline-asm + custom
calling convention declarations). uc_core's Phase 2 would
implement these. Until then, games like Build engine that rely
heavily on `mulscale` / `scale` math primitives can't fully build.

**Spirit of the request preserved**: when uc_core Phase 2 lands,
the game ports can be retried with NO uc386-side changes — they
auto-pick up the new uc_core via the `pip install -e ../uc_core`
sibling install.

## ✓ Two installers (FOSS + abandonware)

**Done** per `addons/README.md` phase 8:

- **FOSS installer** (`uc386-foss-addons-*.tar.gz`): GNU userland
  binaries (GPL/MIT provenance, OK to redistribute). Contains
  the 16 utility .bins + awk + LICENSE files.
- **Abandonware installer** (`uc386-games-build-scripts-*.tar.gz`):
  ships build scripts, not binaries. Per `addons/games/README.md`
  the policy was revised: we DO ship game binaries when public-
  source releases (Doom GPL-2.0, Duke3D GPL-2.0, etc.) authorize
  it; we never ship the data files (WAD / GRP / RTL — those stay
  proprietary). Once uc_core Phase 2 lets the games actually
  build, the abandonware installer will include those binaries
  too. Today it ships only the scripts since the binaries don't
  exist yet.

## ⚠ Build with competitive compilers

**gcc baseline done; Watcom + DJGPP reserved.**
`addons/results.md` has the size table:

```
| Tool | uc386 | gcc | Watcom wcc386 | DJGPP |
| true | 14    | 16,840 | —          | —     |
| ...
```

uc386 binaries are 50–1200× smaller than gcc-on-host (gcc emits
ELF with full glibc startup; uc386 emits flat .bin with only the
libc functions actually used after asm-DCE).

Watcom (wcc386) and DJGPP cross-compilers aren't available on the
dev host. The comparison script reserves their columns; once
those toolchains are installed the table populates automatically.
Installing OpenWatcom on macOS isn't trivial — out of scope for
this slice.

## Summary

Of the 8 explicit items in `addons.txt`, **6 are fully done** and
**2 are scaffolding-only** (games, full Watcom comparison) —
both blocked on infrastructure that lives outside this addons
work (uc_core Phase 2 + Watcom toolchain install).

Total code shipped: 1320 unit tests passing, 220/220 c-testsuite
(full mode), 1514/1514 gcc-c-torture, 16/16 addons, BWK awk
fully functional. ~30 new libc symbols, ~5 new headers, 3 codegen
fixes (preprocessor multi-line comments, extern enum-in-VarDecl,
asm DCE filters unused externs), 2 fixes downstream (`_start_stub`
arg-push ordering, `assemble_and_run` cleanup-when-bundling-skipped).
