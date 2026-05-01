#!/bin/sh
# Fetch Hexen source. Same situation as Heretic — Raven's 2008 GPL
# release predates GitHub and the canonical-feeling source today
# lives in chocolate-doom's heretic/hexen tree. fetch.sh shares
# the same upstream as heretic/ to keep one canonical clone.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "hexen: upstream/ already populated; skip fetch."
    exit 0
fi

URL="https://github.com/chocolate-doom/chocolate-doom/archive/refs/heads/master.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "hexen: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/cdoom.tar.gz"
tar -xzf "$TMP/cdoom.tar.gz" -C "$TMP"
mv "$TMP"/chocolate-doom-* upstream
echo "hexen: upstream tree at addons/games/hexen/upstream/"
echo "hexen: src/hexen/ is the game-specific code; src/ has the shared id-Tech-1 base."
