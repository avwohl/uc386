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
if [ -n "${PYTHON:-}" ]; then
    :
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

# DOOM compiles N source files, links to one executable. We invoke
# uc386 in multi-file mode (it merges TUs in main.py).
#
# EXCLUDE_RX skips sources that need platform-specific subsystems we
# stub out at higher level instead of trying to compile (BSD sockets,
# Linux sound DSP, X11). We still need a couple of these as object-
# level shims (i_video / i_sound replacement), but the upstream files
# themselves don't compile under uc386.
EXCLUDE_RX='/(i_net|i_sound|i_video|i_system)\.c$'
SOURCES="$(find upstream/linuxdoom-1.10 -name '*.c' | grep -Ev "$EXCLUDE_RX" | sort) stubs.c"
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
# `-w-error=label-redef-late`: large programs (DOOM is 60K+ lines of
# generated asm) hit NASM's multi-pass convergence corner cases —
# short-vs-long jump promotion can shift labels between passes. NASM
# 3.x makes this an error by default; we want the warning, not the
# fail. The binary's still correct after convergence.
nasm -f bin -w-error=label-redef-late "$OUT" -o "${OUT%.asm}.bin"
echo "doom: built ${OUT%.asm}.bin"
ls -lh "${OUT%.asm}.bin"
