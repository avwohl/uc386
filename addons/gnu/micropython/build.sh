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
    # Emit a triage qstr table by grep over upstream/py/ +
    # upstream/shared/. Real builds use upstream's
    # tools/makeqstrdefs.py which preprocesses each TU to find
    # MP_QSTR_x macro uses; for triage we approximate with a grep
    # over the source tree, which over-includes (any string that
    # parses as an identifier becomes a qstr) but keeps the mapping
    # complete enough that the enum in py/qstr.h covers every
    # reference downstream code (py/, shared/runtime, etc.) makes.
    #
    # The first entry (MP_QSTRnull) goes in the static pool as
    # QDEF0; everything else goes in the main pool as QDEF1.
    # Reason: `mp_qstr_const_pool` (the main pool) has
    # `is_sorted = true` and the binary search in
    # `qstr_find_strn` does `pool->len - 1` as its upper bound —
    # which underflows to 0xFFFFFFFF when `pool->len == 0`,
    # then loops the binary search into out-of-bounds reads.
    # A real upstream build always has at least one QDEF1 entry,
    # so the empty-main-pool case never happens. We avoid it by
    # routing every grep'd qstr to QDEF1.
    {
        echo "QDEF0(MP_QSTRnull, 0, 0, \"\")"
        echo "QDEF1(MP_QSTR_, 0, 0, \"\")"
        # `MP_QSTR_<name>` — skip the 8-char `MP_QSTR_` prefix (M-P-_-Q-S-T-R-_)
        # to recover the actual qstr string. Earlier the prefix-strip used 7
        # chars and left a stray leading underscore (`__repl_print__` came
        # out as `___repl_print__`), making LOAD_NAME for builtins fail and
        # value-print compile crash inside mp_obj_equal_not_equal.
        #
        # Emit `length(name)` as the third QDEF1 field. qstr_find_strn's
        # post-binary-search linear sweep checks lengths[at] == str_len
        # before doing memcmp. Length 0 (the previous heuristic) made
        # every static qstr fail the length check, causing
        # `qstr_find_strn("__name__")` to return MP_QSTRnull even
        # though the qstr was in the pool — so the bytecode compiler
        # invented a fresh dynamic qstr for every identifier and
        # LOAD_NAME against the static-init dict_main key (whose qstr
        # id is the static 67) would never match.
        grep -rhoE "MP_QSTR_[A-Za-z_][A-Za-z0-9_]*" \
                upstream/py/ upstream/shared/ \
            | sort -u \
            | awk '{ name = substr($0, 9); print "QDEF1(" $0 ", 0, " length(name) ", \"" name "\")" }'
    } > build/genhdr/qstrdefs.generated.h
fi
[ -f build/genhdr/moduledefs.h ] || cat > build/genhdr/moduledefs.h <<'EOF'
// Triage stub. A real build runs upstream/py/makemoduledefs.py over
// the source tree to emit per-module MP_ROM_QSTR / MP_ROM_PTR entries
// followed by `#define MICROPY_REGISTERED_MODULES <list>`. With no
// registered modules in the triage config we just define the macro
// empty so py/objmodule.c's `mp_builtin_module_table[] = { ... }`
// reduces to an empty array initializer.
#define MICROPY_REGISTERED_MODULES
#define MICROPY_REGISTERED_EXTENSIBLE_MODULES
EOF
[ -f build/genhdr/mpversion.h ] || cat > build/genhdr/mpversion.h <<'EOF'
// Triage stub.
#define MICROPY_GIT_TAG "uc386-triage"
#define MICROPY_GIT_HASH "0000000"
#define MICROPY_BUILD_DATE "2026-05-01"
EOF
if [ ! -f build/genhdr/root_pointers.h ]; then
    # Real builds run upstream/py/makeqstrdefs.py with mode=root_pointer
    # to scan all C sources for `MP_REGISTER_ROOT_POINTER(<decl>);`
    # declarations and emit them as struct fields of `_mp_state_vm_t`
    # (via py/mpstate.h's `#include "genhdr/root_pointers.h"`). For
    # triage we approximate with grep — the macro pattern is regular,
    # we just take everything between the parens and emit it as a
    # struct member terminated with a semicolon.
    {
        echo "// Triage stub. Real build emits MP_REGISTER_ROOT_POINTER entries here."
        grep -rhE "^MP_REGISTER_ROOT_POINTER\(.*\);" \
                upstream/py/ upstream/shared/ \
            | sed -E 's#^MP_REGISTER_ROOT_POINTER\((.*)\);#    \1;#' \
            | sort -u
    } > build/genhdr/root_pointers.h
fi

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

# Section accounting so the per-section pass/fail is visible.
PY_PASS=0; PY_FAIL=0; PY_TOTAL=0
SH_PASS=0; SH_FAIL=0; SH_TOTAL=0

triage_one() {
    src="$1"
    section="$2"        # used only for the section-count update
    name_prefix="$3"    # disambiguates basenames between sections
    [ -f "$src" ] || return 0
    TOTAL=$((TOTAL + 1))
    name="${name_prefix}$(basename "$src" .c)"
    if "$PYTHON" -m uc386.main "$TRIAGE_MAIN" "$src" \
            -o "build/${name}.asm" \
            -I "$INCLUDE" \
            -I "upstream" \
            -I "$PORT_DIR" \
            -I "build" \
            > "build/${name}.out" 2> "build/${name}.err"; then
        PASS=$((PASS + 1))
        if [ "$section" = py ]; then PY_PASS=$((PY_PASS + 1)); PY_TOTAL=$((PY_TOTAL + 1)); fi
        if [ "$section" = sh ]; then SH_PASS=$((SH_PASS + 1)); SH_TOTAL=$((SH_TOTAL + 1)); fi
        echo "$name: OK" >> "$TRIAGE"
        rm -f "build/${name}.err" "build/${name}.out"
    else
        FAIL=$((FAIL + 1))
        if [ "$section" = py ]; then PY_FAIL=$((PY_FAIL + 1)); PY_TOTAL=$((PY_TOTAL + 1)); fi
        if [ "$section" = sh ]; then SH_FAIL=$((SH_FAIL + 1)); SH_TOTAL=$((SH_TOTAL + 1)); fi
        first_line="$(head -1 "build/${name}.err" 2>/dev/null || echo unknown)"
        echo "$name: FAIL  $first_line" >> "$TRIAGE"
        # Strip filename + line numbers from the leading error so
        # the histogram clusters by error class.
        echo "$first_line" \
            | sed -E 's#^[^:]+:[0-9]+:[0-9]+:?##; s#^[^:]+:[0-9]+:?##' \
            | sed -E 's#  +# #g; s#^ +##' \
            >> "$ERR_HIST"
    fi
}

# py/ — the platform-independent core (132 sources today).
for src in "$SRC_DIR"/*.c; do
    triage_one "$src" py ""
done

# shared/{libc,readline,runtime,timeutils,netutils}/ — extra sources
# real ports pull in alongside py/. The minimal port uses the first
# three; richer ports (esp32, rp2, etc.) also pull in timeutils +
# netutils. Keeping them in the same triage answers "how close is a
# full port to compiling cleanly" not just "how clean is py/".
for shared_src in \
        upstream/shared/libc/printf.c \
        upstream/shared/libc/string0.c \
        upstream/shared/libc/__errno.c \
        upstream/shared/libc/abort_.c \
        upstream/shared/readline/readline.c \
        upstream/shared/runtime/pyexec.c \
        upstream/shared/runtime/stdout_helpers.c \
        upstream/shared/runtime/interrupt_char.c \
        upstream/shared/runtime/sys_stdio_mphal.c \
        upstream/shared/timeutils/timeutils.c \
        upstream/shared/netutils/netutils.c \
        upstream/shared/netutils/trace.c \
        upstream/shared/netutils/dhcpserver.c; do
    [ -f "$shared_src" ] || continue
    # name_prefix=shared_ so e.g. shared/libc/printf.c doesn't collide
    # with py/ — there is no collision today, but the prefix keeps
    # the name space clean and makes the section visible in triage.txt.
    triage_one "$shared_src" sh "shared_"
done

echo
echo "== triage: $PASS pass / $FAIL fail / $TOTAL total =="
echo "    py/                                          $PY_PASS / $PY_TOTAL"
echo "    shared/{libc,readline,runtime,timeutils,netutils}/  $SH_PASS / $SH_TOTAL"
echo
echo "Top error classes (count × class):"
sort "$ERR_HIST" | uniq -c | sort -rn | head -15
