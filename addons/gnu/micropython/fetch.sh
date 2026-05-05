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

if [ -d upstream ]; then
    echo "micropython: upstream/ already present — skipping main fetch."
    fetch_b_con_crypto
    fetch_lwip
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
