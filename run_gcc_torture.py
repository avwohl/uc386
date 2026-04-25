#!/usr/bin/env python3
"""Run GCC torture tests (execute/) against uc386.

Tests use abort() on failure, exit(0) on success — no stdout diff
needed, just check whether the program aborted or exited cleanly.

Expects the upstream tests at:

    `../external/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute`

(LLVM bundles a checked-in copy of GCC's torture suite.)

Like the other runners, defaults to `--compile-only` until the i386
assemble + link + run pipeline is wired up.
"""

import argparse
import subprocess
import sys
from pathlib import Path

UC386_DIR = Path(__file__).parent
LIB_INCLUDE = UC386_DIR / "lib" / "include"
TORTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "external" / "llvm-test-suite"
    / "SingleSource" / "Regression" / "C"
    / "gcc-c-torture" / "execute"
)

DEFAULT_TIMEOUT = 10
MAX_COM_SIZE = 128_000


def find_tests(patterns=None, limit=None):
    tests: list[Path] = []
    if patterns:
        for pat in patterns:
            for f in sorted(TORTURE_DIR.glob(pat)):
                if f.suffix == ".c" and f not in tests:
                    tests.append(f)
    else:
        tests = sorted(TORTURE_DIR.glob("*.c"))
    if limit:
        tests = tests[:limit]
    return tests


def run_test(c_file: Path, *, compile_only: bool = True) -> tuple[str, str]:
    asm_file = Path("/tmp") / c_file.with_suffix(".asm").name
    cc_cmd = [
        sys.executable, "-m", "uc386.main", str(c_file), "-o", str(asm_file),
        "-I", str(LIB_INCLUDE),
    ]
    try:
        result = subprocess.run(
            cc_cmd, capture_output=True, text=True,
            cwd=UC386_DIR, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "compile", "compile timed out"
    if result.returncode != 0:
        return "compile", result.stderr.strip()[:200]
    if compile_only:
        return "pass", ""

    com_file = asm_file.with_suffix(".com")
    try:
        result = subprocess.run(
            ["nasm", "-f", "bin", str(asm_file), "-o", str(com_file)],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "asm", "nasm timed out"
    if result.returncode != 0:
        return "asm", result.stderr.strip()[:200]
    if com_file.stat().st_size > MAX_COM_SIZE:
        return "skip", f"binary too large ({com_file.stat().st_size} bytes)"
    return "skip", "run stage not yet wired (need DOS extender + libc + emulator)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "patterns", nargs="*",
        help="glob patterns to match (e.g. '93000*.c'). Default: all *.c.",
    )
    parser.add_argument("--limit", type=int, help="stop after N tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=True)
    parser.add_argument("--full", dest="compile_only", action="store_false")
    args = parser.parse_args()

    if not TORTURE_DIR.exists():
        sys.exit(
            f"GCC torture tests not found at {TORTURE_DIR}.\n"
            f"Clone the LLVM test suite (it ships GCC's torture suite):\n"
            f"  git clone --depth=1 https://github.com/llvm/llvm-test-suite.git "
            f"{TORTURE_DIR.parents[3]}",
        )

    tests = find_tests(args.patterns, args.limit)
    if not tests:
        sys.exit("No tests found.")
    print(f"Running {len(tests)} tests...")

    buckets: dict[str, list[str]] = {}
    for c_file in tests:
        status, msg = run_test(c_file, compile_only=args.compile_only)
        buckets.setdefault(status, []).append(c_file.name)
        if args.verbose or status not in ("pass", "skip"):
            print(f"{c_file.name}: {status.upper()}")
            if args.verbose and msg:
                print(f"  {msg.splitlines()[0]}")

    print()
    print("=" * 50)
    total = sum(len(v) for v in buckets.values())
    print(f"Total:   {total}")
    print(f"Pass:    {len(buckets.get('pass', []))}")
    for k, v in buckets.items():
        if k != "pass" and v:
            print(f"{k.title():8} {len(v)}: {v[:5]}{'...' if len(v) > 5 else ''}")


if __name__ == "__main__":
    main()
