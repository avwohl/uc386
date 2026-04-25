#!/usr/bin/env python3
"""Run the Fujitsu compiler-test-suite C tests against uc386.

Expects the upstream tests at `../external/CompilerTestSuite`:

    git clone https://github.com/AcademySoftwareFoundation/CompilerTestSuite.git ../external/CompilerTestSuite

The suite has ~30K single-source tests grouped under `C/`. Each test
has a `.reference_output` file with the expected stdout. Tests use
`#ifdef` to handle 32 / 64-bit long differences.

Like `run_ctests.py`, this defaults to `--compile-only` because the
i386 assemble → link → run pipeline isn't wired yet. The full mode
will compile, NASM-assemble, link with a (yet-to-be-built) DOS
extender, run under dosemu/dosbox, and diff stdout against the
reference output.
"""

import argparse
import subprocess
import sys
from pathlib import Path

UC386_DIR = Path(__file__).parent
LIB_INCLUDE = UC386_DIR / "lib" / "include"
FUJITSU_DIR = Path(__file__).resolve().parent.parent / "external" / "CompilerTestSuite" / "C"

DEFAULT_TIMEOUT = 5
MAX_COM_SIZE = 128_000


def find_tests(dirs=None, limit=None):
    """Find single-source test files with reference output."""
    tests = []
    if dirs:
        roots = [FUJITSU_DIR / d for d in dirs]
    else:
        roots = sorted(p for p in FUJITSU_DIR.iterdir() if p.is_dir())
    for root in roots:
        if not root.is_dir():
            continue
        for c_file in sorted(root.glob("*.c")):
            ref = c_file.with_suffix(".reference_output")
            if ref.exists():
                tests.append((c_file, ref))
            if limit and len(tests) >= limit:
                return tests
    return tests


def run_test(c_file: Path, ref_file: Path, *, compile_only: bool = True) -> tuple[str, str]:
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

    sys.path.insert(0, str(UC386_DIR / "src"))
    from uc386.dos_emu import assemble_and_run
    try:
        emu_res = assemble_and_run(
            asm_file, timeout_seconds=10, instruction_limit=200_000_000,
        )
    except Exception as e:
        return "asm", f"emu: {type(e).__name__}: {e}"

    if emu_res.error:
        return "run", emu_res.error[:300]
    if emu_res.timed_out:
        return "timeout", f"timeout (after {emu_res.instructions_executed} insns)"

    expected = ref_file.read_text() if ref_file.exists() else ""
    if expected != emu_res.stdout:
        return "output", f"got {emu_res.stdout!r} expected {expected!r}"
    if emu_res.exit_code != 0:
        return "run", f"exit {emu_res.exit_code}"
    return "pass", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="*", help="subset of suite dirs to test (e.g. 0001 0003)")
    parser.add_argument("--limit", type=int, help="stop after N tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=True)
    parser.add_argument("--full", dest="compile_only", action="store_false")
    args = parser.parse_args()

    if not FUJITSU_DIR.exists():
        sys.exit(
            f"Fujitsu suite not found at {FUJITSU_DIR}.\n"
            f"Clone with:\n"
            f"  git clone https://github.com/AcademySoftwareFoundation/CompilerTestSuite.git "
            f"{FUJITSU_DIR.parent}",
        )

    tests = find_tests(args.dirs, args.limit)
    if not tests:
        sys.exit("No tests found.")
    print(f"Running {len(tests)} tests...")

    buckets: dict[str, list[str]] = {
        k: [] for k in ("pass", "compile", "asm", "skip")
    }
    for c_file, ref_file in tests:
        status, msg = run_test(
            c_file, ref_file, compile_only=args.compile_only,
        )
        buckets.setdefault(status, []).append(str(c_file.relative_to(FUJITSU_DIR)))
        if args.verbose or status not in ("pass", "skip"):
            print(f"{c_file.relative_to(FUJITSU_DIR)}: {status.upper()}")
            if args.verbose and msg:
                print(f"  {msg.splitlines()[0] if msg else ''}")

    print()
    print("=" * 50)
    total = sum(len(v) for v in buckets.values())
    print(f"Total:   {total}")
    print(f"Pass:    {len(buckets['pass'])}")
    for k, v in buckets.items():
        if k != "pass" and v:
            print(f"{k.title():8} {len(v)}: {v[:5]}{'...' if len(v) > 5 else ''}")


if __name__ == "__main__":
    main()
