#!/bin/sh
# Fetch one-true-awk source from GitHub.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "awk-bwk: upstream/ already populated; skip fetch."
    exit 0
fi

URL="https://github.com/onetrueawk/awk/archive/refs/heads/master.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "awk-bwk: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/awk.tar.gz"
tar -xzf "$TMP/awk.tar.gz" -C "$TMP"
mv "$TMP"/awk-* upstream
echo "awk-bwk: upstream tree at addons/gnu/awk-bwk/upstream/"
