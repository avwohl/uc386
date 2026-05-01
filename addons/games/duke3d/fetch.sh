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

if [ -d upstream ]; then
    echo "duke3d: upstream/ exists but submodules empty; re-cloning."
    rm -rf upstream
fi

echo "duke3d: cloning $URL with submodules …"
git clone --depth=1 --recurse-submodules --shallow-submodules "$URL" upstream
echo "duke3d: upstream tree at addons/games/duke3d/upstream/"
echo "duke3d: src/ is game logic; jfbuild/ is the Build engine."
