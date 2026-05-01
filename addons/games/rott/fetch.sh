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
# (sed -i.bak works on both GNU sed and BSD/macOS sed; bare -i differs.)
echo "rott: patching upstream filename typos …"
sed -i.bak 's|"_rt_build\.h"|"_rt_buil.h"|g' upstream/rott/RT_BUILD.C
sed -i.bak 's|"rt_spball\.h"|"rt_spbal.h"|g' upstream/rott/RT_IN.C

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

# RT_TEXT.C is missing a pile of transitive includes. Upstream relied
# on Watcom's order-of-definitions across the per-TU compile; uc386
# parses each TU strict-header-resolved. Add what RT_TEXT.C actually
# uses: pic_t (lumpy.h), ticcount (isr.h), PU_CACHE/zone (z_zone.h),
# px/py/rowon/leftmargin (rt_menu.h), VW_UpdateScreen (rt_view.h),
# VWB_DrawPic (rt_draw.h).
sed -i.bak 's|#include "memcheck.h"|#include "lumpy.h"\
#include "isr.h"\
#include "z_zone.h"\
#include "rt_menu.h"\
#include "rt_view.h"\
#include "rt_draw.h"\
#include "memcheck.h"|' upstream/rott/RT_TEXT.C

# Drop the .bak files sed left behind so the upstream/ tree we ship
# in the games tarball doesn't carry confusing duplicates.
rm -f upstream/rott/RT_BUILD.C.bak upstream/rott/RT_IN.C.bak \
      upstream/rott/RT_TEXT.C.bak

echo "rott: upstream tree at addons/games/rott/upstream/"
echo "rott: ROTT-source/ holds the original DOS C; ROTT-Audio/ has audio/data tools."
