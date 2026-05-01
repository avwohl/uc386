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
        "-I", str(REPO_ROOT / "lib" / "include"),
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


def package_foss(version: str) -> Path:
    """Bundle all built FOSS addon binaries into a release tarball."""
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
        # Include README + LICENSE
        readme = REPO_ROOT / "addons" / "README.md"
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

    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


def package_games_scripts(version: str) -> Path:
    """Bundle game fetch+build scripts (no binaries, just sources of
    instructions). Users run them locally to fetch each game's
    public-source release and build."""
    out_dir = REPO_ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"uc386-games-build-scripts-{version}.tar.gz"

    games_root = ADDONS_ROOT / "games"
    print(f"Bundling game build scripts from {games_root} …")

    with tarfile.open(out_path, "w:gz") as tar:
        for sub in sorted(games_root.iterdir()):
            if not sub.is_dir():
                continue
            for child in sub.rglob("*"):
                if not child.is_file():
                    continue
                rel = child.relative_to(games_root)
                tar.add(child, arcname=f"uc386-games/{rel}")
        readme = games_root / "README.md"
        if readme.exists():
            # already covered by the loop, but ensure top-level
            pass

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
