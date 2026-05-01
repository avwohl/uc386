#!/bin/sh
# Build DOOM via uc386. Today this is best-effort: it expects a lot
# of compile errors. Each error is a ticket against uc_core / uc386 /
# libc until enough features land.
set -eu

cd "$(dirname "$0")"
if [ ! -d upstream/linuxdoom-1.10 ]; then
    echo "doom: run ./fetch.sh first." >&2
    exit 1
fi

REPO="$(cd ../../.. && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
INCLUDE="$REPO/lib/include"

# DOOM compiles N source files, links to one executable. We invoke
# uc386 in multi-file mode (it merges TUs in main.py).
SOURCES="$(find upstream/linuxdoom-1.10 -name '*.c' | sort)"
OUT="$(pwd)/build/doom.asm"
mkdir -p "$(dirname "$OUT")"

echo "doom: compiling $(echo "$SOURCES" | wc -l) sources …"
"$PYTHON" -m uc386.main $SOURCES \
    -I "$INCLUDE" \
    -I upstream/linuxdoom-1.10 \
    -D NORMALUNIX -D LINUX \
    -o "$OUT" || {
    echo "doom: compile failed — ticket against uc_core/uc386." >&2
    exit 1
}

echo "doom: wrote $OUT"
echo "doom: assembling via nasm …"
nasm -f bin "$OUT" -o "${OUT%.asm}.bin"
echo "doom: built ${OUT%.asm}.bin"
