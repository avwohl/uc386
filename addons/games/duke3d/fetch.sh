#!/bin/sh
# Fetch Duke Nukem 3D source. 3D Realms released the Duke3D + Build
# engine source under GPL in 2003; the modern caretaker is the
# jonof/jfduke3d community fork.
#
# jfduke3d depends on three sibling submodules — jfbuild (engine),
# jfaudiolib (sound), jfmact (input) — that the GitHub tarball
# doesn't include. Use `git clone --recursive` so they actually
# resolve, otherwise the duke3d compile bails on `Cannot find
# include file: build.h` immediately.
set -eu

cd "$(dirname "$0")"
if [ -d upstream/jfbuild/include ]; then
    echo "duke3d: upstream/ already populated; skip fetch."
    exit 0
fi

URL="https://github.com/jonof/jfduke3d.git"
# Pinned upstream commit (parent jfduke3d only — submodules pin via
# whatever they're recorded at on this commit). Bump with
# `git ls-remote https://github.com/jonof/jfduke3d HEAD`.
SHA="55c5f9592d2a78e427a2bbdb13ce3c4c9ccf6f04"  # 2026-01-02

if [ -d upstream ]; then
    echo "duke3d: upstream/ exists but submodules empty; re-cloning."
    rm -rf upstream
fi

echo "duke3d: cloning $URL @ $SHA with submodules …"
# --depth=1 only fetches HEAD, so we can't checkout an older SHA from
# a depth-1 clone. Use a full clone here; jfduke3d is small (a few MB).
git clone --recurse-submodules "$URL" upstream
( cd upstream && git checkout "$SHA" && \
  git submodule update --init --recursive )

# Patch: menues.c uses a block-scope `static const char *s[]` in
# `case 8` that uc386's flat function-scope can't disambiguate from
# an `int s` reused as a counter in `case 9`. Rename the inner array
# to `weaponswitch_names` so the two `s` references are no longer
# the same identifier.
# (sed -i.bak works on both GNU sed and BSD/macOS sed; bare -i differs.)
echo "duke3d: patching menues.c block-scope shadow …"
sed -i.bak 's|static const char \*s\[\] = { "Off", "New"|static const char *weaponswitch_names[] = { "Off", "New"|' upstream/src/menues.c
sed -i.bak 's|gametextpal(d,yy, s\[ud.weaponswitch\]|gametextpal(d,yy, weaponswitch_names[ud.weaponswitch]|' upstream/src/menues.c
rm -f upstream/src/menues.c.bak

# Patch: kplib.c expects __int64 + _lrotl as Watcom intrinsics under
# __DOS__ (uc386 predefines __DOS__=1, so the in-file
# `#if !defined(_WIN32) && !defined(__DOS__)` typedef branch is
# inactive). Inject the typedef + a portable _lrotl fallback right
# after `#include <stdint.h>`.
echo "duke3d: patching kplib.c __int64 / _lrotl shim for __DOS__ …"
python3 -c '
src = open("upstream/jfbuild/src/kplib.c").read()
shim = ("#include <stdint.h>\n"
        "typedef long long __int64;\n"
        "static int _lrotl(int i, int sh) "
        "{ return (int)(((unsigned)i << sh) | ((unsigned)i >> (32 - sh))); }")
src = src.replace("#include <stdint.h>", shim, 1)
open("upstream/jfbuild/src/kplib.c", "w").write(src)
'

echo "duke3d: upstream tree at addons/games/duke3d/upstream/"
echo "duke3d: src/ is game logic; jfbuild/ is the Build engine."
