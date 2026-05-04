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

if [ -d upstream ]; then
    echo "micropython: upstream/ already present — skipping main fetch."
    fetch_b_con_crypto
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
