#!/bin/sh
# Fetch GNU gawk source. Idempotent: skips if upstream/ exists.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "gawk: upstream/ already populated; skip fetch."
    exit 0
fi

VERSION="${GAWK_VERSION:-5.4.0}"
URL="https://ftp.gnu.org/gnu/gawk/gawk-${VERSION}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "gawk: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/gawk.tar.gz"
tar -xzf "$TMP/gawk.tar.gz" -C "$TMP"
mv "$TMP/gawk-${VERSION}" upstream
echo "gawk: upstream tree at addons/gnu/gawk/upstream/"
