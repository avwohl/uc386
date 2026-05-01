#!/bin/sh
# Per-file triage of the upstream/py/ core sources through uc386.
#
# This is NOT a full build (yet) — MicroPython needs a port shim
# (mp_hal_stdout_tx_*, mp_hal_delay_*, a fixed-region heap for GC,
# argv plumbing) before it can produce a runnable image. Today this
# script answers the prerequisite question: how many of the ~50
# platform-independent .c files in py/ compile cleanly through
# uc386 → NASM-ready .asm? An error histogram surfaces the top
# blockers as concrete tickets.
#
# Outputs:
#   build/<name>.asm   — for every PASS
#   build/<name>.err   — uc386 stderr for every FAIL
#   build/triage.txt   — per-source PASS/FAIL summary
#   build/errors.txt   — first-line error histogram (sorted by count)
set -eu

cd "$(dirname "$0")"

if [ ! -d upstream ]; then
    echo "micropython: run ./fetch.sh first." >&2
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
SRC_DIR="upstream/py"
PORT_DIR="upstream/ports/minimal"   # supplies mpconfigport.h + mphalport.h

mkdir -p build build/genhdr

# Stub the upstream-generated headers that py/*.c expects. A real
# build runs upstream/py/makeqstrdefs.py + makemoduledefs.py to
# emit these from the source tree; for triage we ship empty stubs
# so the preprocessor finds them and we see compile-class failures
# (uc386 limitations) instead of a wall of missing-header errors.
if [ ! -f build/genhdr/qstrdefs.generated.h ]; then
    # Emit a triage qstr table by grep over upstream/py/. Real builds
    # use upstream's tools/makeqstrdefs.py which preprocesses each TU
    # to find MP_QSTR_x macro uses; for triage we approximate with a
    # grep over the source tree, which over-includes (any string that
    # parses as an identifier becomes a qstr) but keeps the mapping
    # complete enough that the enum in py/qstr.h covers every
    # reference downstream code makes.
    {
        echo "QDEF0(MP_QSTRnull, 0, 0, \"\")"
        echo "QDEF0(MP_QSTR_, 0, 0, \"\")"
        grep -rhoE "MP_QSTR_[A-Za-z_][A-Za-z0-9_]*" upstream/py/*.c upstream/py/*.h \
            | sort -u \
            | awk '{ name = substr($0, 8); print "QDEF0(" $0 ", 0, 0, \"" name "\")" }'
    } > build/genhdr/qstrdefs.generated.h
fi
[ -f build/genhdr/moduledefs.h ] || cat > build/genhdr/moduledefs.h <<'EOF'
// Empty stub so py/objmodule.c can include it cleanly.
// A real build emits MP_REGISTER_MODULE entries here.
EOF
[ -f build/genhdr/mpversion.h ] || cat > build/genhdr/mpversion.h <<'EOF'
// Triage stub.
#define MICROPY_GIT_TAG "uc386-triage"
#define MICROPY_GIT_HASH "0000000"
#define MICROPY_BUILD_DATE "2026-05-01"
EOF
[ -f build/genhdr/root_pointers.h ] || cat > build/genhdr/root_pointers.h <<'EOF'
// Empty stub. Real build emits MP_REGISTER_ROOT_POINTER entries here.
EOF

# Triage stub: a one-line main() so uc386 (which requires a main
# function in every translation unit it compiles) accepts library
# .c files. We append it to each src on the fly. This answers
# "would this file compile if it were linked into a real port?"
# without the multi-file plumbing a real port needs.
TRIAGE_MAIN="build/_triage_main.c"
cat > "$TRIAGE_MAIN" <<'EOF'
// Synthetic main so uc386 has an entry-point during per-file triage.
// Real ports/<port>/main.c supplies its own main + mp_init/mp_deinit.
int main(int argc, char **argv) { (void)argc; (void)argv; return 0; }
EOF

TRIAGE="build/triage.txt"
ERR_HIST="build/errors.txt"
: > "$TRIAGE"
: > "$ERR_HIST"

PASS=0
FAIL=0
TOTAL=0
for src in "$SRC_DIR"/*.c; do
    [ -f "$src" ] || continue
    TOTAL=$((TOTAL + 1))
    name="$(basename "$src" .c)"
    if "$PYTHON" -m uc386.main "$TRIAGE_MAIN" "$src" \
            -o "build/${name}.asm" \
            -I "$INCLUDE" \
            -I "upstream" \
            -I "$PORT_DIR" \
            -I "build" \
            > "build/${name}.out" 2> "build/${name}.err"; then
        PASS=$((PASS + 1))
        echo "$name: OK" >> "$TRIAGE"
        rm -f "build/${name}.err" "build/${name}.out"
    else
        FAIL=$((FAIL + 1))
        first_line="$(head -1 "build/${name}.err" 2>/dev/null || echo unknown)"
        echo "$name: FAIL  $first_line" >> "$TRIAGE"
        # Strip filename + line numbers from the leading error so
        # the histogram clusters by error class.
        echo "$first_line" \
            | sed -E 's#^[^:]+:[0-9]+:[0-9]+:?##; s#^[^:]+:[0-9]+:?##' \
            | sed -E 's#  +# #g; s#^ +##' \
            >> "$ERR_HIST"
    fi
done

echo
echo "== py/ triage: $PASS pass / $FAIL fail / $TOTAL total =="
echo
echo "Top error classes (count × class):"
sort "$ERR_HIST" | uniq -c | sort -rn | head -15
