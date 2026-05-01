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

# Patch known upstream typos: a few `#include "long_name.h"` lines
# reference filenames that don't exist in any case (the actual files
# are 8.3 truncated like _RT_BUIL.H from the DOS-era FAT layout).
# Rewrite to the names that actually exist.
echo "rott: patching upstream filename typos …"
sed -i '' 's|"_rt_build\.h"|"_rt_buil.h"|g' upstream/rott/RT_BUILD.C
sed -i '' 's|"rt_spball\.h"|"rt_spbal.h"|g' upstream/rott/RT_IN.C

echo "rott: upstream tree at addons/games/rott/upstream/"
echo "rott: ROTT-source/ holds the original DOS C; ROTT-Audio/ has audio/data tools."
