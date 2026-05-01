#!/bin/bash
# Build Duke Nukem 3D via uc386. Today this is best-effort: the goal
# is to triage how many compile-time blockers remain past the
# endianness gate, NOT to produce a runnable .bin (the Build engine
# math primitives need #pragma aux which is uc_core Phase 2 work).
# (Don't `set -e` — every per-file compile is allowed to fail; that's
# the whole point of triage mode.)
set -u

cd "$(dirname "$0")"
if [ ! -d upstream/jfbuild/include ]; then
    echo "duke3d: run ./fetch.sh first." >&2
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

# Game-side sources (upstream/src/) plus a minimal main shim. Excludes
# files that need OS-specific subsystems we'd stub at higher level —
# startgtk_game.c (GTK), startwin_game.c (Win32), the SDL appicon
# resources under rsrc/.
EXCLUDE_RX='/(startgtk_game|startwin_game|rsrc/[^/]+)\.c$'
GAME_SOURCES="$(find upstream/src -name '*.c' | grep -Ev "$EXCLUDE_RX" | sort)"

# Triage mode: try compiling each game-side TU individually and report
# how it lands. We expect most to fail — what we want is a histogram
# of distinct error messages, NOT a successful build.
echo "duke3d: triaging $(echo "$GAME_SOURCES" | wc -l) game-side sources …"
echo "stub-main int main(void) { return 0; }" > /tmp/duke_stub_main.c
cat > /tmp/duke_stub_main.c << 'EOF'
/* Minimal main shim so uc386 has an entry point in single-file
   triage. Real entry is in upstream/src/game.c. */
int main(void) { return 0; }
EOF

triage() {
    local label="$1"; shift
    local sources="$@"
    local OK=0 FAIL=0
    echo
    echo "=== $label ==="
    for src in $sources; do
        out=$("$PYTHON" -m uc386.main "$src" /tmp/duke_stub_main.c \
            -I "$INCLUDE" \
            -I upstream/src \
            -I upstream/jfbuild/include \
            -I upstream/jfaudiolib/include \
            -I upstream/jfmact \
            -D B_LITTLE_ENDIAN=1 -D B_BIG_ENDIAN=0 -D USE_OPENGL=0 \
            -o /tmp/duke_one.asm 2>&1) && rc=0 || rc=$?
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
    echo "  $label: $OK clean, $FAIL bailed."
}

triage "Game-side (upstream/src)" "$GAME_SOURCES"

# Build engine — exclude files that are platform-specific to OSes
# we don't target. None of these are reachable on DOS:
#   gl/polymost*       — OpenGL (jfduke3d's hardware renderer)
#   gtkbits/startgtk*  — GTK Linux editor frontends
#   startwin*/winlayer/winbits — Win32 editor + DDraw layer
#   sdlayer*           — SDL backend (we have no SDL on flat-DOS)
#   mmulti             — BSD sockets (netinet/in.h)
#   defs               — defs.c is a stub that #error-s without DDB
ENGINE_EXCLUDE='/(gl[^/]*|polymost[^/]*|gtkbits|sdlayer[^/]*|startgtk[^/]*|startwin[^/]*|winbits|winlayer|mmulti|defs)\.c$|polymosttex|mdsprite'
ENGINE_SOURCES="$(find upstream/jfbuild/src -maxdepth 1 -name '*.c' | grep -Ev "$ENGINE_EXCLUDE" | sort)"
triage "Build engine (jfbuild/src)" "$ENGINE_SOURCES"

echo
echo "duke3d: triage only — not a complete build. Multi-file link is"
echo "the next step once individual triage settles."