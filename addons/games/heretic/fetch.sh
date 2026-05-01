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

URL="https://github.com/chocolate-doom/chocolate-doom/archive/refs/heads/master.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "heretic: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/cdoom.tar.gz"
tar -xzf "$TMP/cdoom.tar.gz" -C "$TMP"
mv "$TMP"/chocolate-doom-* upstream
echo "heretic: upstream tree at addons/games/heretic/upstream/"
echo "heretic: src/heretic/ is the game-specific code; src/ has the shared id-Tech-1 base."
