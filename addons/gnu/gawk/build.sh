#!/bin/sh
# Attempt to build a subset of gawk via uc386. Today this is a stub —
# see NOTES.md for the gnulib + regex porting work that has to land
# before this script does anything useful.
set -eu

cd "$(dirname "$0")"
if [ ! -d upstream ]; then
    echo "gawk: run ./fetch.sh first." >&2
    exit 1
fi

echo "gawk: build awaiting subset patches + gnulib stubs."
echo "gawk: see NOTES.md for the porting strategy."
exit 1
