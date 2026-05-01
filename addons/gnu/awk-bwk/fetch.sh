#!/bin/sh
# Fetch one-true-awk source from GitHub.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "awk-bwk: upstream/ already populated; skip fetch."
    exit 0
fi

# Pinned to a specific upstream commit so the source we ship in the
# FOSS tarball alongside awk.bin always matches the binary, and
# re-running fetch.sh in the future returns the same tree. Bump by
# replacing the SHA below with `git ls-remote https://github.com/onetrueawk/awk HEAD`.
SHA="5739fd79bcfc75ba7526773d0cf634521f8aca3c"  # 2026-04-26
URL="https://github.com/onetrueawk/awk/archive/${SHA}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "awk-bwk: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/awk.tar.gz"
tar -xzf "$TMP/awk.tar.gz" -C "$TMP"
mv "$TMP"/awk-* upstream
echo "awk-bwk: upstream tree at addons/gnu/awk-bwk/upstream/"
