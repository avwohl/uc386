#!/usr/bin/env python3
"""Run each FOSS addon binary against its manifest under dos_emu.

This is the "test script" shipped in the FOSS release tarball. It
walks `src/<name>/manifest.toml`, locates `<name>.bin` next to the
script (or under `bin/`), runs it through `uc386.dos_emu.run` with
the manifest's argv + stdin + vfiles, and compares stdout + exit
code to the expected values.

Usage:
    cd <unpacked uc386-foss tarball>
    python test_addons.py [-v] [--name <addon>]

Requires `uc386` to be importable (so `dos_emu` is available). The
manifest.toml format is the same one the dev harness uses.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def _load_manifest(path: Path) -> dict:
    data = tomllib.loads(path.read_text())
    base = path.parent
    vfiles_raw = data.get("vfiles", {}) or {}
    vfiles: dict[bytes, bytes] = {}
    for name, val in vfiles_raw.items():
        if isinstance(val, dict) and "file" in val:
            vfiles[name.encode("utf-8")] = (base / val["file"]).read_bytes()
        else:
            vfiles[name.encode("utf-8")] = str(val).encode("utf-8")
    return {
        "name": data.get("name") or base.name,
        "argv": list(data.get("argv", [])),
        "stdin": data.get("stdin", "").encode("utf-8"),
        "expect_stdout": data.get("expect_stdout"),
        "expect_exit": int(data.get("expect_exit", 0)),
        "timeout": float(data.get("timeout_seconds", 10.0)),
        "ilim": int(data.get("instruction_limit", 200_000_000)),
        "vfiles": vfiles,
    }


def _find_bin(here: Path, name: str) -> Path | None:
    for cand in (here / f"{name}.bin", here / "bin" / f"{name}.bin"):
        if cand.exists():
            return cand
    # awk-bwk: manifest names "awk-bwk" but the bin ships as awk.bin.
    if name.startswith("awk"):
        for cand in (here / "awk.bin", here / "bin" / "awk.bin"):
            if cand.exists():
                return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--name", default=None,
                    help="run only the named addon")
    ap.add_argument("--root", default=None,
                    help="tarball root (default: dir containing this script)")
    args = ap.parse_args()

    try:
        from uc386.dos_emu import run
    except ImportError:
        print(
            "test_addons.py: uc386 is not importable.\n"
            "Install with `pip install uc386` (or point PYTHONPATH at a"
            " uc386 checkout's src/), then re-run.",
            file=sys.stderr,
        )
        return 2
    # uc386.dos_emu imports lazily; the actual unicorn import happens
    # inside run(). Probe early so the user gets a clean message
    # instead of finding out per-addon.
    try:
        import unicorn  # noqa: F401
    except ImportError:
        print(
            "test_addons.py: the `unicorn` engine isn't installed.\n"
            "Install with `pip install unicorn`, then re-run.",
            file=sys.stderr,
        )
        return 2

    here = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    src_root = here / "src"
    if not src_root.is_dir():
        print(f"test_addons.py: no src/ next to {here}", file=sys.stderr)
        return 2

    manifests = sorted(src_root.glob("*/manifest.toml"))
    if args.name:
        manifests = [m for m in manifests if m.parent.name == args.name]
        if not manifests:
            print(f"test_addons.py: no manifest for {args.name}",
                  file=sys.stderr)
            return 1

    fails = 0
    skips = 0
    passes = 0
    for mp in manifests:
        m = _load_manifest(mp)
        name = m["name"]
        bin_path = _find_bin(here, name)
        if bin_path is None:
            print(f"  {name}: SKIP (no <{name}>.bin)")
            skips += 1
            continue

        try:
            res = run(
                bin_path,
                argv=m["argv"] or None,
                stdin_bytes=m["stdin"],
                timeout_seconds=m["timeout"],
                instruction_limit=m["ilim"],
                vfiles_init=m["vfiles"] or None,
            )
        except Exception as e:
            print(f"  {name}: RUN FAIL — {type(e).__name__}: {e}")
            fails += 1
            continue

        if res.timed_out:
            print(f"  {name}: TIMEOUT")
            fails += 1
            continue
        if res.error:
            print(f"  {name}: RUN ERROR — {res.error}")
            fails += 1
            continue

        ok = res.exit_code == m["expect_exit"]
        if m["expect_stdout"] is not None:
            ok = ok and res.stdout == m["expect_stdout"]

        if ok:
            print(f"  {name}: PASS")
            passes += 1
            if args.verbose:
                print(f"    stdout={res.stdout!r}")
                print(f"    exit={res.exit_code}")
        else:
            print(f"  {name}: FAIL exit={res.exit_code} (want {m['expect_exit']})")
            if args.verbose or m["expect_stdout"] is not None:
                got = res.stdout
                want = m["expect_stdout"]
                if want is not None and got != want:
                    print(f"    got  stdout={got[:120]!r}{'…' if len(got) > 120 else ''}")
                    print(f"    want stdout={want[:120]!r}{'…' if len(want) > 120 else ''}")
            fails += 1

    # Run a tiny smoke check on the upstream-port binaries the FOSS
    # tarball ships outside the manifest set: awk.bin and
    # micropython.bin. These don't have a manifest.toml because their
    # build path (fetch.sh + build.sh + multi-TU compile) doesn't
    # plug into the per-source harness, but a one-shot probe through
    # dos_emu still gives end-to-end confidence.
    awk_bin = here / "awk.bin"
    if awk_bin.exists():
        try:
            res = run(
                awk_bin, argv=["awk", "BEGIN { print 2*3 }"],
                stdin_bytes=b"", timeout_seconds=10.0,
                instruction_limit=200_000_000,
            )
            ok = res.exit_code == 0 and res.stdout == "6\n" and not res.timed_out
        except Exception as e:
            print(f"  awk: RUN FAIL — {type(e).__name__}: {e}")
            fails += 1
        else:
            if ok:
                print("  awk: PASS (BEGIN { print 2*3 } → 6)")
                passes += 1
            else:
                print(
                    f"  awk: FAIL exit={res.exit_code} stdout={res.stdout!r}"
                )
                fails += 1

    mp_bin = here / "micropython.bin"
    if mp_bin.exists():
        try:
            res = run(
                mp_bin, stdin_bytes=b"\x04",
                timeout_seconds=15.0, instruction_limit=2_000_000_000,
            )
            # Ctrl-D-only path: REPL boots, prints banner + prompt,
            # then exits cleanly.
            ok = (
                res.exit_code == 0
                and not res.timed_out
                and "MicroPython" in res.stdout
                and ">>> " in res.stdout
            )
        except Exception as e:
            print(f"  micropython: RUN FAIL — {type(e).__name__}: {e}")
            fails += 1
        else:
            if ok:
                print("  micropython: PASS (REPL banner + clean Ctrl-D exit)")
                passes += 1
            else:
                tail = res.stdout[-160:]
                print(
                    f"  micropython: FAIL exit={res.exit_code} "
                    f"timed_out={res.timed_out} tail={tail!r}"
                )
                fails += 1

    total = passes + fails
    print()
    print(f"== {passes}/{total} passed ({skips} skipped) ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
