#!/bin/sh
# Build BWK awk via uc386. Pre-generates awkgram.tab.{c,h} via bison and
# proctab.c via the host-built maketab helper, then invokes uc386 on
# the full source set.
set -eu

cd "$(dirname "$0")"
if [ ! -d upstream ]; then
    echo "awk-bwk: run ./fetch.sh first." >&2
    exit 1
fi

REPO="$(cd ../../.. && pwd)"
# In CI we don't have .venv; fall back to whatever python is on PATH
# (the workflow installs the right packages globally).
if [ -n "${PYTHON:-}" ]; then
    :  # explicit override wins
elif [ -x "$REPO/.venv/bin/python" ]; then
    PYTHON="$REPO/.venv/bin/python"
else
    PYTHON="$(command -v python3.12 || command -v python3 || command -v python)"
fi
# The headers moved to src/uc386/lib/include when lib/ was packaged
# for PyPI; $REPO/lib/include has not existed since. Every addon build
# script kept pointing at the old path, so uc386 could not find even
# stdio.h and each build died on its first #include. Fall back to the
# installed package so this works outside a source checkout too.
if [ -d "$REPO/src/uc386/lib/include" ]; then
    INCLUDE="$REPO/src/uc386/lib/include"
elif [ -d "$REPO/lib/include" ]; then
    INCLUDE="$REPO/lib/include"          # pre-packaging layout
else
    INCLUDE="$("$PYTHON" -c 'import pathlib, uc386; print(pathlib.Path(uc386.__file__).parent / "lib" / "include")')"
fi
SHIM="$REPO/addons/gnu/_sbase_shim"
SRC="$(pwd)/upstream"
OUT="$(pwd)/build"
mkdir -p "$OUT"

# 1. Run bison on awkgram.y to produce awkgram.tab.{c,h}.
if [ ! -f "$SRC/awkgram.tab.c" ]; then
    echo "awk-bwk: running bison on awkgram.y …"
    (cd "$SRC" && bison -d -o awkgram.tab.c awkgram.y)
fi

# 2. Build maketab on host, run it to generate proctab.c.
if [ ! -f "$SRC/proctab.c" ]; then
    echo "awk-bwk: building host maketab …"
    cc -O2 -o "$OUT/maketab" "$SRC/maketab.c"
    echo "awk-bwk: running maketab to produce proctab.c …"
    (cd "$SRC" && "$OUT/maketab" awkgram.tab.h > proctab.c)
fi

# 3. Invoke uc386 on the full source set.
echo "awk-bwk: compiling via uc386 …"
SOURCES="$SRC/awkgram.tab.c $SRC/b.c $SRC/main.c $SRC/parse.c $SRC/proctab.c $SRC/tran.c $SRC/lib.c $SRC/run.c $SRC/lex.c"
"$PYTHON" -m uc386.main $SOURCES \
    -I "$INCLUDE" \
    -I "$SRC" \
    -D HAS_ISBLANK \
    -o "$OUT/awk.asm" || {
    echo "awk-bwk: compile failed — see error above." >&2
    exit 1
}

echo "awk-bwk: wrote $OUT/awk.asm"
echo "awk-bwk: assembling …"
nasm -f bin "$OUT/awk.asm" -o "$OUT/awk.bin"
echo "awk-bwk: built $OUT/awk.bin"
