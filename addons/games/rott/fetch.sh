#!/bin/sh
# Fetch Rise of the Triad source. Apogee released the 1994 source
# under GPL-2.0 in 2002. The cleanest mirror today is in the
# videogamepreservation org.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "rott: upstream/ already populated; skip fetch."
    exit 0
fi

URL="https://github.com/videogamepreservation/rott/archive/refs/heads/master.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "rott: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/rott.tar.gz"
tar -xzf "$TMP/rott.tar.gz" -C "$TMP"
mv "$TMP"/rott-* upstream
echo "rott: upstream tree at addons/games/rott/upstream/"
echo "rott: ROTT-source/ holds the original DOS C; ROTT-Audio/ has audio/data tools."
