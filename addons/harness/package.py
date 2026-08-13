#!/usr/bin/env python3
"""Package addon binaries into release archives.

Two outputs:
  uc386-foss-addons-<ver>.tar.gz       — built FOSS addon binaries
                                         (true/false/cat/wc/.../sbase-tee).
                                         Source provenance: this repo +
                                         sbase upstream (MIT/GPL).
  uc386-games-build-scripts-<ver>.tar.gz — fetch.sh / build.sh / NOTES.md
                                         for Doom, Duke3D, Heretic, etc.
                                         Users run the scripts to fetch
                                         upstream source and build locally.

Run from repo root:
    .venv/bin/python -m addons.harness.package --version v0.1.0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADDONS_ROOT = REPO_ROOT / "addons"


def build_one_addon(name: str) -> Path | None:
    """Build an addon end-to-end and return its .bin path (or None on fail).

    Compiles via uc386 (libc bundled at compile time → self-contained
    .asm), then nasm-assembles to a flat .bin. Bypasses the harness
    so the assembled .bin survives (the harness's `assemble_and_run`
    cleanup deletes both the bin and the bundled asm).
    """
    # Read the manifest to find sources + extra_cflags.
    addon_dir = ADDONS_ROOT / "gnu" / name
    manifest = addon_dir / "manifest.toml"
    if not manifest.exists():
        return None
    import tomllib
    data = tomllib.loads(manifest.read_text())
    sources = [str(addon_dir / s) for s in data["sources"]]
    extra = list(data.get("extra_cflags", []))

    out_dir = REPO_ROOT / "build" / "package" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    asm_path = out_dir / f"{name}.asm"
    bin_path = out_dir / f"{name}.bin"

    cmd = [
        sys.executable, "-m", "uc386.main",
        *sources, "-o", str(asm_path),
        "-I", str(REPO_ROOT / "src" / "uc386" / "lib" / "include"),
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0 or not asm_path.exists():
        return None
    proc = subprocess.run(
        ["nasm", "-f", "bin", str(asm_path), "-o", str(bin_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return bin_path


def list_addons() -> list[str]:
    out = []
    for sub in sorted((ADDONS_ROOT / "gnu").iterdir()):
        if sub.is_dir() and not sub.name.startswith("_"):
            if (sub / "manifest.toml").exists():
                out.append(sub.name)
    return out


def _pack_addon_sources(tar: tarfile.TarFile, addon_dir: Path) -> None:
    """Include per-addon sources + scripts under uc386-foss/src/<name>/.

    Skips derived directories (`build/`, `__pycache__/`). The shipped
    layout mirrors `addons/gnu/<name>/` so users can read the source,
    re-run fetch.sh / build.sh, or run test_addons.py against the
    manifests. **Includes `upstream/` when it exists** so the FOSS
    tarball ships the exact upstream source corresponding to any
    shipped binary (GPL §3 / sbase MIT / one-true-awk Lucent license
    all require source-with-binary). For addons whose upstream is not
    yet fetched, the fetch.sh + build.sh scripts ship instead.
    """
    EXCLUDE = {"build", "__pycache__"}
    for child in addon_dir.rglob("*"):
        if not child.is_file():
            continue
        rel = child.relative_to(addon_dir)
        if any(p in EXCLUDE for p in rel.parts):
            continue
        tar.add(child, arcname=f"uc386-foss/src/{addon_dir.name}/{rel}")


def package_foss(version: str) -> Path:
    """Bundle built FOSS addon binaries + sources + test runner.

    Tarball layout:
        uc386-foss/
          README.md, LICENSE, SBASE-LICENSE, AWK-LICENSE
          test_addons.py           — runs <name>.bin under dos_emu
                                     against src/<name>/manifest.toml
          <name>.bin                — built binaries (17 + awk)
          exe/<name>.exe            — DOS/32A-bound .exe, when the
                                      release pipeline built them
          src/<name>/manifest.toml  — argv / stdin / expected stdout
          src/<name>/*.c            — addon source
          src/_sbase_shim/util.{c,h}, LICENSE — shared sbase helpers
          src/awk-bwk/{fetch,build}.sh, NOTES.md — upstream port
          src/gawk/{fetch,build}.sh, NOTES.md   — doc-only stub
    """
    out_dir = REPO_ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"uc386-foss-addons-{version}.tar.gz"

    addons = list_addons()
    print(f"Building {len(addons)} FOSS addons …")

    with tarfile.open(out_path, "w:gz") as tar:
        for name in addons:
            bin_path = build_one_addon(name)
            if bin_path is None:
                print(f"  {name}: SKIP (build failed)")
                continue
            arcname = f"uc386-foss/{name}.bin"
            tar.add(bin_path, arcname=arcname)
            print(f"  {name}: {bin_path.stat().st_size:,} bytes")
            # If a corresponding .exe was built earlier in the
            # release pipeline (release.yml's "Build .exe variants"
            # step), ship it under exe/<name>.exe. Skip silently
            # when missing so dev runs (no Watcom on macOS) still
            # work.
            exe_path = REPO_ROOT / "build" / "exe" / f"{name}.exe"
            if exe_path.exists():
                tar.add(exe_path, arcname=f"uc386-foss/exe/{name}.exe")
                print(f"    + {name}.exe: {exe_path.stat().st_size:,} bytes")

        # Per-addon sources, manifests, and scripts under src/.
        # Includes every gnu/* dir (the manifest-driven 16, the
        # _sbase_shim shared headers, awk-bwk's fetch+build scripts,
        # and the gawk doc-only stub).
        gnu_root = ADDONS_ROOT / "gnu"
        for sub in sorted(gnu_root.iterdir()):
            if not sub.is_dir():
                continue
            _pack_addon_sources(tar, sub)

        # Test runner: ships at the top level so users can `python
        # test_addons.py` from the unpacked tarball directory.
        test_runner = REPO_ROOT / "addons" / "harness" / "test_addons.py"
        if test_runner.exists():
            tar.add(test_runner, arcname="uc386-foss/test_addons.py")

        # Ship the release-tailored README (action-oriented, references
        # test_addons.py and src/) instead of the dev-side addons/README.md.
        readme = REPO_ROOT / "addons" / "RELEASE_README.md"
        license_ = REPO_ROOT / "LICENSE"
        if readme.exists():
            tar.add(readme, arcname="uc386-foss/README.md")
        if license_.exists():
            tar.add(license_, arcname="uc386-foss/LICENSE")
        # Sbase shim license (covers sbase-* binaries).
        sbase_lic = REPO_ROOT / "addons" / "gnu" / "_sbase_shim" / "LICENSE"
        if sbase_lic.exists():
            tar.add(sbase_lic, arcname="uc386-foss/SBASE-LICENSE")
        # awk-bwk is a special case: built via build.sh, not the
        # uc386-on-.c path. Include the built binary if present.
        awk_bin = REPO_ROOT / "addons" / "gnu" / "awk-bwk" / "build" / "awk.bin"
        if awk_bin.exists():
            tar.add(awk_bin, arcname="uc386-foss/awk.bin")
            print(f"  awk: {awk_bin.stat().st_size:,} bytes")
        awk_lic = REPO_ROOT / "addons" / "gnu" / "awk-bwk" / "upstream" / "LICENSE"
        if awk_lic.exists():
            tar.add(awk_lic, arcname="uc386-foss/AWK-LICENSE")
        # micropython is the same special case: multi-TU build via
        # build_port.sh; ship the binary if it's been built. The
        # smoke test goes alongside so users can validate the bin
        # under their own dos_emu install.
        mp_bin = REPO_ROOT / "addons" / "gnu" / "micropython" / "build" / "micropython.bin"
        if mp_bin.exists():
            tar.add(mp_bin, arcname="uc386-foss/micropython.bin")
            print(f"  micropython: {mp_bin.stat().st_size:,} bytes")
        mp_lic = REPO_ROOT / "addons" / "gnu" / "micropython" / "upstream" / "LICENSE"
        if mp_lic.exists():
            tar.add(mp_lic, arcname="uc386-foss/MICROPYTHON-LICENSE")

    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def package_games_scripts(version: str) -> Path:
    """Bundle game fetch+build scripts and any built binaries.

    Today only Doom boots end-to-end (under dos_emu, exits at WAD-
    not-found because we don't ship WADs). Its binary, if present at
    addons/games/doom/build/doom.bin, ships in the tarball alongside
    its scaffolding. Other games triage clean per-file but don't yet
    link/boot — they ship as scripts only until they do.
    """
    out_dir = REPO_ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"uc386-games-build-scripts-{version}.tar.gz"

    games_root = ADDONS_ROOT / "games"
    print(f"Bundling game build scripts from {games_root} …")

    # Skip derived `build/` and `__pycache__/`. `upstream/` is shipped
    # when present — every game we host is GPL or otherwise requires
    # source-with-binary, so when we ship a binary the source has to
    # ride along. For games where upstream/ is empty (developer hasn't
    # run fetch.sh, or CI didn't fetch this game), only the
    # scaffolding ships.
    EXCLUDE_DIRS = {"build", "__pycache__"}

    # Games whose .bin we DO ship when it exists in build/. Listed
    # explicitly so a stale per-game build/ doesn't silently leak
    # in once a different game starts producing one.
    SHIP_BIN = {"doom"}

    with tarfile.open(out_path, "w:gz") as tar:
        for sub in sorted(games_root.iterdir()):
            if not sub.is_dir():
                continue
            # First pass: pick up any symlinks at the top of each game
            # dir (hexen/uc386_config -> ../heretic/uc386_config). The
            # default rglob doesn't follow symlinked directories, so
            # without this the link gets dropped entirely.
            for child in sub.iterdir():
                if child.is_symlink():
                    rel = child.relative_to(games_root)
                    tar.add(child, arcname=f"uc386-games/{rel}",
                            recursive=False)  # preserve the link itself
            for child in sub.rglob("*"):
                if not child.is_file():
                    continue
                # Skip anything inside an excluded directory at any depth.
                if any(p in EXCLUDE_DIRS for p in child.relative_to(sub).parts):
                    continue
                rel = child.relative_to(games_root)
                tar.add(child, arcname=f"uc386-games/{rel}")
            # Second pass: explicitly ship the built binary if this
            # game is on the SHIP_BIN list and the build artifact
            # exists. Lives under bin/<game>/ to keep it separate
            # from the scaffolding so users see at a glance which
            # games come pre-built.
            if sub.name in SHIP_BIN:
                game_bin = sub / "build" / f"{sub.name}.bin"
                if game_bin.exists():
                    arc = f"uc386-games/bin/{sub.name}/{sub.name}.bin"
                    tar.add(game_bin, arcname=arc)
                    print(f"  {sub.name}.bin: "
                          f"{game_bin.stat().st_size:,} bytes")

    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(prog="addons.harness.package")
    ap.add_argument("--version", default="dev",
                    help="release version tag (default: dev)")
    ap.add_argument("--foss-only", action="store_true",
                    help="only build the FOSS-addons tarball")
    ap.add_argument("--games-only", action="store_true",
                    help="only build the games-scripts tarball")
    args = ap.parse_args()

    if args.foss_only and args.games_only:
        print("--foss-only and --games-only are mutually exclusive",
              file=sys.stderr)
        return 2

    if not args.games_only:
        package_foss(args.version)
    if not args.foss_only:
        package_games_scripts(args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
