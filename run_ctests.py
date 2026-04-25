#!/usr/bin/env python3
"""Run c-testsuite tests for uc386.

This is the i386/MS-DOS port of uc80's c-testsuite runner. It expects
the upstream tests cloned at `../external/c-testsuite` (sibling to
`uc386` and `uc_core`):

    git clone https://github.com/c-testsuite/c-testsuite.git ../external/c-testsuite

The full pipeline is compile → assemble → link → run → diff:

  1. compile:  python -m uc386.main file.c -o file.asm  (works today)
  2. assemble: nasm -f bin file.asm -o file.com         (works today)
  3. link:     n/a (the .asm contains its own _start and DOS exit stub)
  4. run:      dosbox / dosemu / qemu-system-i386       (NOT WIRED YET)
  5. diff:     compare stdout to <test>.c.expected      (waits on 4)

Until step 4 is wired, the runner defaults to `--compile-only` and
just checks that uc386 accepts each source. The remaining stages
print "skip" with a clear reason.

Notes shared with uc80 (most are still relevant for i386):
- 00040: 8-queens. Slow; skip in --compile-only since it doesn't apply
        to compile-only timing, but listed for the runtime stage.
- 00174 / 00178 / 00195: float tests; uc386 has full FPU codegen so
        these should compile.
- 00216: range designators [1...5] — not supported by uc_core's parser.
- 00220: wide characters (wchar_t, L"...") — partial uc_core support.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

UC386_DIR = Path(__file__).parent
LIB_INCLUDE = UC386_DIR / "lib" / "include"
TEST_SUITE_DIR = Path(__file__).resolve().parent.parent / "external" / "c-testsuite" / "tests" / "single-exec"
# Optional per-platform .c overrides, mirroring uc80's tests/c-testsuite-z80
# layout. Drop a `<num>.c` here to substitute a tweaked source.
PLATFORM_DIR = UC386_DIR / "tests" / "c-testsuite-i386"

# Tests that need longer timeouts when the run stage lands.
SLOW_TESTS = {
    "00040": 600,  # 8-queens — O(n!)
    "00041": 60,   # prime sieve — many div/mods
}

# Known-skip with reason. Keep this empty by default; only add entries
# after a human checks a test is truly impossible (not just failing).
SKIP_TESTS: dict[str, str] = {}

# Maximum .com size (16-bit DOS .com cap is 65280 bytes; with a DOS
# extender we have more headroom but this number stays sane).
MAX_COM_SIZE = 128_000


def resolve_source(c_file: Path, test_num: str) -> Path:
    """Return the source we should actually compile for `test_num`.

    If `tests/c-testsuite-i386/<test_num>.c` exists, prefer it over
    the upstream version — that's where uc386-specific adaptations
    live (e.g. tweaked formatting expectations).
    """
    if test_num:
        platform = PLATFORM_DIR / f"{test_num}.c"
        if platform.exists():
            return platform
    return c_file


def resolve_expected(c_file: Path, test_num: str) -> Path | None:
    """Return the expected-output file for this test, if any."""
    if test_num:
        platform = PLATFORM_DIR / f"{test_num}.c.expected"
        if platform.exists():
            return platform
    upstream = c_file.with_suffix(".c.expected")
    return upstream if upstream.exists() else None


def run_test(
    c_file: Path,
    test_num: str,
    *,
    verbose: bool = False,
    compile_only: bool = True,
) -> tuple[str, str]:
    """Compile (and optionally assemble/link/run) one test.

    Returns (status, message). Status is one of: pass, compile, asm,
    link, run, output, timeout, skip, unknown.
    """
    if test_num in SKIP_TESTS:
        return "skip", SKIP_TESTS[test_num]

    source = resolve_source(c_file, test_num)
    asm_file = Path("/tmp") / c_file.with_suffix(".asm").name

    cc_cmd = [
        sys.executable, "-m", "uc386.main", str(source), "-o", str(asm_file),
        "-I", str(LIB_INCLUDE),
    ]
    try:
        result = subprocess.run(
            cc_cmd, capture_output=True, text=True,
            cwd=UC386_DIR, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "compile", "compile timed out after 15s"
    if result.returncode != 0:
        return "compile", result.stderr.strip()[:400]

    if compile_only:
        return "pass", ""

    # ---- assemble + run via the unicorn-engine harness ----
    # Lazy import so the compile-only path doesn't pay for unicorn.
    sys.path.insert(0, str(UC386_DIR / "src"))
    from uc386.dos_emu import assemble_and_run

    timeout = SLOW_TESTS.get(test_num, 10)
    insn_limit = 2_000_000_000 if test_num in SLOW_TESTS else 200_000_000
    try:
        emu_res = assemble_and_run(
            asm_file, timeout_seconds=timeout, instruction_limit=insn_limit,
        )
    except Exception as e:
        return "asm", f"emu: {type(e).__name__}: {e}"

    if emu_res.error:
        return "run", emu_res.error[:300]
    if emu_res.timed_out:
        return "timeout", f"timeout (after {emu_res.instructions_executed} insns)"

    expected = resolve_expected(c_file, test_num)
    if expected is None:
        # No expected output — pass iff exit code is 0.
        if emu_res.exit_code != 0:
            return "run", f"exit {emu_res.exit_code}"
        return "pass", ""

    expected_text = expected.read_text()
    actual_text = emu_res.stdout
    if expected_text != actual_text:
        return "output", f"got {actual_text!r} expected {expected_text!r}"
    if emu_res.exit_code != 0:
        return "run", f"exit {emu_res.exit_code}"
    return "pass", ""


def main():
    parser = argparse.ArgumentParser(
        description="Run c-testsuite against uc386",
    )
    parser.add_argument("tests", nargs="*", help="specific test numbers (e.g. 00001)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=220)
    parser.add_argument(
        "--compile-only", action="store_true", default=True,
        help="default. Stop after `python -m uc386.main` — "
             "checks that uc386 accepts each source.",
    )
    parser.add_argument(
        "--full", dest="compile_only", action="store_false",
        help="run compile+assemble+link+run+diff. Most stages aren't "
             "wired yet; expect 'skip' for each test until the "
             "DOS-extender pipeline lands.",
    )
    args = parser.parse_args()

    if not TEST_SUITE_DIR.exists():
        sys.exit(
            f"c-testsuite not found at {TEST_SUITE_DIR}.\n"
            f"Clone it with:\n"
            f"  git clone https://github.com/c-testsuite/c-testsuite.git "
            f"{TEST_SUITE_DIR.parent.parent}",
        )

    if args.tests:
        test_nums = args.tests
    else:
        test_nums = [f"{i:05d}" for i in range(args.start, args.end + 1)]

    buckets: dict[str, list[str]] = {
        k: [] for k in (
            "pass", "compile", "asm", "link", "run",
            "output", "timeout", "skip", "unknown",
        )
    }

    for num in test_nums:
        c_file = TEST_SUITE_DIR / f"{num}.c"
        if not c_file.exists():
            continue
        status, msg = run_test(
            c_file, num, verbose=args.verbose,
            compile_only=args.compile_only,
        )
        buckets[status].append(num)
        if args.verbose or status not in ("pass", "skip"):
            print(f"{num}: {status.upper()}")
            if args.verbose and msg:
                for line in msg.splitlines()[:5]:
                    print(f"  {line}")

    total = sum(len(v) for v in buckets.values())
    print()
    print("=" * 50)
    print(f"Total:    {total}")
    print(f"Pass:     {len(buckets['pass'])}")
    for k in ("compile", "asm", "link", "run", "output", "timeout", "skip", "unknown"):
        if buckets[k]:
            print(f"{k.title():9} {len(buckets[k])}: {buckets[k][:20]}"
                  + ("..." if len(buckets[k]) > 20 else ""))


if __name__ == "__main__":
    main()
