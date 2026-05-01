#!/bin/sh
# Fetch Duke Nukem 3D source. 3D Realms released the Duke3D + Build
# engine source under GPL in 2003; the modern caretaker is the
# eduke32 community fork.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "duke3d: upstream/ already populated; skip fetch."
    exit 0
fi

# Original 3D Realms release (preferred — minimum modifications).
URL="https://github.com/jonof/jfduke3d/archive/refs/heads/master.tar.gz"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "duke3d: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/duke3d.tar.gz"
tar -xzf "$TMP/duke3d.tar.gz" -C "$TMP"
mv "$TMP"/jfduke3d-* upstream
echo "duke3d: upstream tree at addons/games/duke3d/upstream/"
echo "duke3d: build/ subdir is the Build engine; source/ is game logic."
