#!/bin/sh
# Fetch MicroPython upstream into upstream/.
# Idempotent: a second run is a no-op once upstream/ exists.
set -eu
cd "$(dirname "$0")"

# B-Con public-domain crypto reference impls used by hashlib's
# md5/sha1 (sha256 is already in upstream's tarball). Pull these
# even when upstream/ already exists, so an older fetch that
# pre-dates the crypto-algorithms additions self-heals on next run.
fetch_b_con_crypto() {
    CA_BASE="https://raw.githubusercontent.com/B-Con/crypto-algorithms/master"
    mkdir -p upstream/lib/crypto-algorithms
    for f in md5.c md5.h sha1.c sha1.h; do
        if [ ! -f "upstream/lib/crypto-algorithms/$f" ]; then
            echo "micropython: fetching crypto-algorithms/$f …"
            curl -fsSL "$CA_BASE/$f" -o "upstream/lib/crypto-algorithms/$f"
        fi
    done
}

# lwIP — submodule in upstream/.gitmodules but not pulled by the
# tarball. Fetch a pinned release tarball into upstream/lib/lwip/
# alongside the existing crypto-algorithms stash. STABLE-2_2_1 is
# the latest tagged release as of mid-2026 and is what our port
# integration targets.
fetch_lwip() {
    LWIP_TAG="STABLE-2_2_1_RELEASE"
    LWIP_DIR="upstream/lib/lwip"
    if [ -d "$LWIP_DIR/src/core" ]; then
        return 0
    fi
    echo "micropython: fetching lwIP $LWIP_TAG …"
    LWIP_TMP="$(mktemp -d)"
    trap 'rm -rf "$LWIP_TMP"' RETURN 2>/dev/null || true
    curl -fsSL \
        "https://github.com/lwip-tcpip/lwip/archive/refs/tags/${LWIP_TAG}.tar.gz" \
        -o "$LWIP_TMP/lwip.tgz"
    tar -xzf "$LWIP_TMP/lwip.tgz" -C "$LWIP_TMP"
    mkdir -p "$LWIP_DIR"
    cp -r "$LWIP_TMP"/lwip-*/src "$LWIP_DIR/"
    rm -rf "$LWIP_TMP"
}

# Patch upstream's `mod_lwip_reset` to register our loopback packet
# pump as the poll callback (instead of nulling it). LWIP_NETIF_LOOPBACK=1
# with NO_SYS=1 needs manual netif_poll(netif_default) per tick to
# deliver packets; uc386dos_loopback_poll (in uc386-dos/lwip_uc386dos.c)
# does that. Idempotent: skips when the patched line is already present.
patch_modlwip_loopback_poll() {
    F="upstream/extmod/modlwip.c"
    if [ ! -f "$F" ]; then return 0; fi
    if grep -q "uc386dos_loopback_poll" "$F"; then return 0; fi
    if ! grep -q "lwip_poll_list.poll = NULL;" "$F"; then
        echo "micropython: warn: modlwip.c reset shape changed — skipping loopback patch." >&2
        return 0
    fi
    echo "micropython: patching modlwip.c mod_lwip_reset for loopback poll …"
    awk '
        /static mp_obj_t mod_lwip_reset/ { in_reset = 1 }
        in_reset && /lwip_poll_list\.poll = NULL;/ {
            print "    extern void uc386dos_loopback_poll(void *arg);"
            print "    lwip_poll_list.poll = uc386dos_loopback_poll;"
            print "    lwip_poll_list.poll_arg = NULL;"
            in_reset = 0
            next
        }
        in_reset && /^}/ { in_reset = 0 }
        { print }
    ' "$F" > "$F.tmp" && mv "$F.tmp" "$F"
}

# Inject printf checkpoints into ports/minimal/main.c so the rig
# can see how far MP startup gets when stdout is redirected through
# PMODE/W INT 21h AH=40 → file. The DOSBox-X rig runs a known-good
# echo.exe baseline that prints fine; MP.EXE produces no observable
# bytes despite returning. This bisects which startup step is the
# silent one. Idempotent: skip if the markers are already in place.
patch_main_startup_markers() {
    F="upstream/ports/minimal/main.c"
    if [ ! -f "$F" ]; then return 0; fi
    if grep -q "mp-startup-marker" "$F"; then return 0; fi
    echo "micropython: patching ports/minimal/main.c with startup markers …"
    awk '
        /^int main\(int argc, char \*\*argv\) \{/ {
            print
            print "    /* mp-startup-marker injected by addons/gnu/micropython/fetch.sh */"
            print "    printf(\"[mp-main-entered]\\n\"); fflush(stdout);"
            in_main = 1
            next
        }
        in_main && /mp_stack_ctrl_init\(\);/ {
            print "    printf(\"[mp-before-stack-ctrl]\\n\"); fflush(stdout);"
            print
            next
        }
        in_main && /mp_init\(\);/ {
            print "    printf(\"[mp-before-mp-init]\\n\"); fflush(stdout);"
            print
            print "    printf(\"[mp-after-mp-init]\\n\"); fflush(stdout);"
            next
        }
        in_main && /pyexec_friendly_repl\(\);/ {
            print "    printf(\"[mp-before-repl]\\n\"); fflush(stdout);"
            print
            print "    printf(\"[mp-after-repl]\\n\"); fflush(stdout);"
            next
        }
        in_main && /mp_deinit\(\);/ {
            print "    printf(\"[mp-before-deinit]\\n\"); fflush(stdout);"
            print
            in_main = 0
            next
        }
        { print }
    ' "$F" > "$F.tmp" && mv "$F.tmp" "$F"
}

if [ -d upstream ]; then
    echo "micropython: upstream/ already present — skipping main fetch."
    fetch_b_con_crypto
    fetch_lwip
    patch_modlwip_loopback_poll
    patch_main_startup_markers
    exit 0
fi

# Pinned to a specific upstream commit so the source we ship in the
# tarball alongside the binary always matches what was built. The
# port currently targets v1.26-era APIs; bump with
# `git ls-remote https://github.com/micropython/micropython HEAD`
# and re-run the build cycle when promoting to a newer revision.
SHA="9f396bba8d675ffb53f7fb047def21c7a581948e"  # 2025-08-01
URL="https://github.com/micropython/micropython/archive/${SHA}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "micropython: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/micropython.tar.gz"
tar -xzf "$TMP/micropython.tar.gz" -C "$TMP"
mv "$TMP"/micropython-* upstream
echo "micropython: upstream tree at addons/gnu/micropython/upstream/"

fetch_b_con_crypto
fetch_lwip
patch_modlwip_loopback_poll
patch_main_startup_markers
