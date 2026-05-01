#!/bin/sh
# Build Duke Nukem 3D via uc386. Expected to fail today — Build engine
# uses heavy `#pragma aux` for fixed-point math primitives and sound
# DPMI hooks, both blocked on uc_core Phase 2 work.
set -eu

cd "$(dirname "$0")"
if [ ! -d upstream ]; then
    echo "duke3d: run ./fetch.sh first." >&2
    exit 1
fi

echo "duke3d: build today expected to fail at uc_core Phase 2 (#pragma aux)."
echo "duke3d: see NOTES.md for the blocker list."
exit 1
