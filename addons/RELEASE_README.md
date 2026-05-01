# uc386 FOSS addons — release tarball

This tarball ships **built binaries + source + a test runner** for
the GNU/MIT-licensed userland that runs on top of the [uc386][uc]
C compiler. Every binary is a flat-32 i386 image emitted by uc386
through NASM; load it under `uc386.dos_emu` (or any flat-binary
emulator that initialises EAX=argc / EBX=&argv[0]).

[uc]: https://github.com/avwohl/uc386

## Layout

```
uc386-foss/
  *.bin             ← 16 utility binaries + awk.bin
  src/<name>/       ← per-addon: manifest.toml + sources
  src/_sbase_shim/  ← shared sbase headers (used by sbase-cat etc.)
  src/awk-bwk/      ← BWK awk fetch.sh / build.sh / NOTES
  src/awk-bwk/upstream/ ← one-true-awk source tree (when awk.bin ships)
  src/gawk/         ← GNU gawk (doc-only — see NOTES.md for blockers)
  test_addons.py    ← walks src/<name>/manifest.toml, runs each .bin
  README.md         ← this file
  LICENSE / SBASE-LICENSE / AWK-LICENSE
```

## Run the binaries

Each `<name>.bin` is a flat-binary i386 program. Three paths to
run them:

1. **`uc386.dos_emu` (Python)**:

   ```python
   from uc386.dos_emu import run
   res = run("echo.bin", argv=["echo", "hi"])
   print(res.stdout, res.exit_code)
   ```

2. **`test_addons.py`** (golden tests, requires `pip install uc386`):

   ```sh
   python test_addons.py            # all addons
   python test_addons.py --name wc  # one addon
   python test_addons.py -v         # show stdout/exit on PASS
   ```

   Each addon's expected argv / stdin / stdout / exit code lives in
   `src/<name>/manifest.toml`. The runner compares stdout + exit
   code to those fields.

3. **DOS / DOSBox** — the binaries are flat-32 with a uc386
   `_start_stub` entry, not COM/EXE format. They run under
   `dos_emu` natively; for real DOS you'd need a flat-32 loader
   shim (out of scope for this tarball).

## Rebuild from source

The `src/<name>/` tree contains everything needed to rebuild each
addon. Most are single-file C programs; `awk-bwk` ships upstream
fetch.sh + build.sh that pull the [one-true-awk][bwk] master and
compile it through uc386.

[bwk]: https://github.com/onetrueawk/awk

```sh
git clone https://github.com/avwohl/uc386
cd uc386 && python3.12 -m venv .venv && .venv/bin/pip install -e .

# Then for any in-tree addon: drop its sources + manifest into
# addons/gnu/<name>/ in the uc386 checkout (or replace this
# tarball's sources back into a checkout) and run:
.venv/bin/python -m addons.harness.build gnu --all
```

For the upstream awk port:

```sh
cd src/awk-bwk
./fetch.sh        # downloads upstream into upstream/
./build.sh        # bison + maketab + uc386 → build/awk.bin
```

## Inventory

| Addon       | Source         | Notes                                         |
|-------------|----------------|-----------------------------------------------|
| true        | in-tree        | exit 0                                        |
| false       | in-tree        | exit 1                                        |
| yes         | in-tree        | "y\n" × 1000 (capped — no SIGPIPE on dos_emu) |
| echo        | in-tree        | argv joining + `-n`                           |
| cat         | in-tree        | argv files + stdin (`-`)                      |
| wc          | in-tree        | -lwc over stdin                               |
| head        | in-tree        | -n N                                          |
| tail        | in-tree        | -n N                                          |
| basename    | in-tree        | strip dir + suffix                            |
| dirname     | in-tree        | strip last path component                     |
| factor      | in-tree        | trial-division prime factorisation            |
| open_test   | in-tree        | smoke for fopen/fread                         |
| strtol_test | in-tree        | smoke for strtol parsing                      |
| sbase-cat   | sbase upstream | via `_sbase_shim/util.c`                      |
| sbase-head  | sbase upstream | via `_sbase_shim/util.c`                      |
| sbase-tee   | sbase upstream | via `_sbase_shim/util.c`                      |
| awk         | one-true-awk   | full BWK awk (~107 KB — 6 KLoC source)        |

## Licenses

- **In-tree addons** (basename, cat, dirname, echo, factor, false,
  head, open_test, strtol_test, tail, true, wc, yes): GPL-3.0
  (matches the parent uc386 repo). See `LICENSE`.
- **sbase-cat / sbase-head / sbase-tee**: ISC (sbase upstream
  license). See `SBASE-LICENSE` and `src/_sbase_shim/LICENSE`.
- **awk-bwk**: Lucent free / public-source licence (BWK awk
  upstream). See `AWK-LICENSE`. When `awk.bin` ships, the
  matching one-true-awk source tree ships under
  `src/awk-bwk/upstream/` so the binary always travels with the
  source it was built from.
- **gawk** (folder is documentation-only — no binary ships):
  full GNU gawk would need a regex engine and gnulib subset; see
  `src/gawk/NOTES.md`. The shipped `awk` (BWK) covers the spirit
  of "include awk" from `docs/addons.txt`.
