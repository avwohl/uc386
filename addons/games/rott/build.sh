#!/bin/bash
# Build Rise of the Triad via uc386. ROTT is much harder than Doom
# because it carries period DOS infrastructure end-to-end (memcheck.h,
# Watcom #pragma aux assembly primitives, BIOS/INT 21h direct calls).
# Today this is a triage harness — every per-file compile is allowed
# to fail; we want a histogram of distinct first-error messages so the
# uc_core/uc386 backlog shapes up.
set -u

cd "$(dirname "$0")"
if [ ! -d upstream/rott ]; then
    echo "rott: run ./fetch.sh first." >&2
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

# ROTT was written for Watcom 10.x with the DOS/4GW DPMI extender.
# We claim Watcom-compatibility so memcheck.h's `#elif defined(__WATCOMC__)`
# branch lights up rather than the `#error Unknown compiler` fallback.
# 1100 is Watcom 11.0; ROTT's tests treat anything ≥1000 the same.
CFLAGS="-D __WATCOMC__=1100 -D __386__"

cat > /tmp/rott_stub_main.c << 'EOF'
int main(void) { return 0; }
EOF

OK=0
FAIL=0
for src in upstream/rott/*.C; do
    name="${src##*/}"
    out=$("$PYTHON" -m uc386.main "$src" /tmp/rott_stub_main.c \
        -I "$INCLUDE" \
        -I upstream/rott \
        -I upstream/rottcom/ROTTSER \
        -I upstream/rottcom/ROTTIPX \
        $CFLAGS \
        -o /tmp/rott_one.asm 2>&1) && rc=0 || rc=$?
    if [ $rc -eq 0 ]; then
        printf "  %-25s OK\n" "$name"
        OK=$((OK + 1))
    else
        err=$(echo "$out" | grep -E "uc386:|uc386\.codegen|ParseError|^.*\.h:[0-9]+:|#error" | head -1 | tr -s ' ')
        printf "  %-25s %s\n" "$name" "$err"
        FAIL=$((FAIL + 1))
    fi
done
echo "rott: $OK clean, $FAIL bailed."
echo "rott: triage only — game-side build needs uc_core Phase 2 (#pragma aux)."