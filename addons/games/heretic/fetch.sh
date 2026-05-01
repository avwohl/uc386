#!/bin/sh
# Fetch Heretic source. Raven Software released the 1996 Heretic
# source under GPL in 2008, but never to GitHub directly — common
# mirrors live in third-party caretaker repos. The chocolate-doom
# tree carries a Heretic build target that's a faithful port of
# the original engine; we point at that for now.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "heretic: upstream/ already populated; skip fetch."
    exit 0
fi

# Pinned upstream commit. Bump by replacing the SHA below with
# `git ls-remote https://github.com/chocolate-doom/chocolate-doom HEAD`.
SHA="9e731e2b2b03d361a477f4c0ce4da830c1a71312"  # 2026-04-29
URL="https://github.com/chocolate-doom/chocolate-doom/archive/${SHA}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "heretic: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/cdoom.tar.gz"
tar -xzf "$TMP/cdoom.tar.gz" -C "$TMP"
mv "$TMP"/chocolate-doom-* upstream
echo "heretic: upstream tree at addons/games/heretic/upstream/"
echo "heretic: src/heretic/ is the game-specific code; src/ has the shared id-Tech-1 base."
