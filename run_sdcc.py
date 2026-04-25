#!/usr/bin/env python3
"""Run SDCC regression tests against uc386.

Expects the upstream tests at `../external/sdcc-regression`. uc80
ships them under that path; for uc386 you'd typically copy or
symlink the same source tree.

The SDCC suite uses a test framework (testfwk.c/testfwk.h) with
ASSERT() macros. Some tests are templates with {placeholder}
substitutions that generate multiple test instances.

This is a skeleton — `run_ctests.py` is the runner that actually
works in `--compile-only` mode today; this one matches the uc80
shape but the template-expansion + run pipeline isn't wired yet.
"""

import argparse
import sys
from pathlib import Path

UC386_DIR = Path(__file__).parent
SDCC_DIR = Path(__file__).resolve().parent.parent / "external" / "sdcc-regression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=True)
    parser.add_argument("--full", dest="compile_only", action="store_false")
    args = parser.parse_args()

    if not SDCC_DIR.exists():
        sys.exit(
            f"SDCC regression tests not found at {SDCC_DIR}.\n"
            f"See uc80's run_sdcc.py for the upstream layout — copy "
            f"that tree into {SDCC_DIR} (it's not a public github repo; "
            f"it ships with SDCC's source distribution).",
        )

    print(
        "SDCC runner not yet ported in full. The template-expansion "
        "logic (multiple test instances per .c file) and the test "
        "framework wrapper need adaptation for the i386 ABI before "
        "this can produce useful results. See run_ctests.py for the "
        "minimum viable runner shape.",
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
