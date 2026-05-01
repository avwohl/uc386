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

# RT_TEXT.C declares `char word[WORDLIMIT]` as a local in HandleWord
# but RT_DEF.H typedefs `word` as `unsigned short int`. uc386's parser
# doesn't handle local-variable shadowing of a typedef; rename the
# local to `wordbuf` for lines 395-435 of HandleWord.
echo "rott: patching RT_TEXT.C HandleWord local-typedef shadow …"
python3 -c '
import re
src = open("upstream/rott/RT_TEXT.C").read()
lines = src.split("\n")
for i in range(394, 435):
    if i < len(lines):
        lines[i] = re.sub(r"\bword\b", "wordbuf", lines[i])
open("upstream/rott/RT_TEXT.C", "w").write("\n".join(lines))
'

# RT_TEXT.C also uses `pic_t` (defined in lumpy.h, not transitively
# pulled in). Add the include.
sed -i '' 's|#include "memcheck.h"|#include "lumpy.h"\
#include "memcheck.h"|' upstream/rott/RT_TEXT.C

echo "rott: upstream tree at addons/games/rott/upstream/"
echo "rott: ROTT-source/ holds the original DOS C; ROTT-Audio/ has audio/data tools."
