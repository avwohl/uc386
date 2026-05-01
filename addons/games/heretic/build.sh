#!/bin/bash
# Build Heretic via uc386. chocolate-doom-style modernized port; uses
# the heretic-owned uc386_config (config.h + SDL_endian.h shim) plus
# the same uc_core preprocessor improvements that landed in 63912fd
# (multi-line macro merge, comment-aware paren tracking, trailing-
# comment strip in #define).
set -u

cd "$(dirname "$0")"
if [ ! -d upstream/src/heretic ]; then
    echo "heretic: run ./fetch.sh first." >&2
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
INCLUDE="$REPO/lib/include"

cat > /tmp/heretic_stub_main.c << 'EOF'
int main(void) { return 0; }
EOF

OK=0
FAIL=0
for src in upstream/src/heretic/*.c; do
    out=$("$PYTHON" -m uc386.main "$src" /tmp/heretic_stub_main.c \
        -I "$INCLUDE" \
        -I uc386_config \
        -I upstream/src \
        -I upstream/src/heretic \
        -I upstream/textscreen \
        --include-file stdarg.h \
        -o /tmp/heretic_one.asm 2>&1) && rc=0 || rc=$?
    name="${src##*/}"
    if [ $rc -eq 0 ]; then
        printf "  %-25s OK\n" "$name"
        OK=$((OK + 1))
    else
        err=$(echo "$out" | grep -E "uc386:|uc386\.codegen|ParseError|^.*\.h:[0-9]+:" | head -1 | tr -s ' ')
        printf "  %-25s %s\n" "$name" "$err"
        FAIL=$((FAIL + 1))
    fi
done
echo "heretic: $OK clean, $FAIL bailed."