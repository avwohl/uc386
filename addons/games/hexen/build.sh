#!/bin/bash
# Build Hexen via uc386. Same engine family as Heretic / Doom; uses
# the chocolate-doom modernized port, which means it builds on top
# of the heretic uc386_config (config.h + SDL_endian.h shim) via
# the symlink at ./uc386_config.
set -u

cd "$(dirname "$0")"
if [ ! -d upstream/src/hexen ]; then
    echo "hexen: run ./fetch.sh first." >&2
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

cat > /tmp/hexen_stub_main.c << 'EOF'
int main(void) { return 0; }
EOF

OK=0
FAIL=0
for src in upstream/src/hexen/*.c; do
    out=$("$PYTHON" -m uc386.main "$src" /tmp/hexen_stub_main.c \
        -I "$INCLUDE" \
        -I uc386_config \
        -I upstream/src \
        -I upstream/src/hexen \
        -o /tmp/hexen_one.asm 2>&1) && rc=0 || rc=$?
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
echo "hexen: $OK clean, $FAIL bailed."