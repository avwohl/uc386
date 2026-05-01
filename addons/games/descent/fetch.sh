#!/bin/sh
# Fetch Descent source. The Parallax 1998 release lived on archive.org
# and various community mirrors; today the actively-maintained
# caretaker fork is dxx-rebirth, which carries both Descent 1 and 2
# under the original 1998 source-available license.
#
# Note: dxx-rebirth is heavily modernized (SDL, OpenGL, MSVC/clang
# friendly). Building it through uc386 is the longest-horizon target
# in this set — see NOTES.md.
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "descent: upstream/ already populated; skip fetch."
    exit 0
fi

# Pinned upstream commit. Bump with
# `git ls-remote https://github.com/dxx-rebirth/dxx-rebirth HEAD`.
SHA="b749eadb4080f596ce90ef7b2be97d7c1213567f"  # 2026-04-12
URL="https://github.com/dxx-rebirth/dxx-rebirth/archive/${SHA}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "descent: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/descent.tar.gz"
tar -xzf "$TMP/descent.tar.gz" -C "$TMP"
mv "$TMP"/dxx-rebirth-* upstream
echo "descent: upstream tree at addons/games/descent/upstream/"
echo "descent: similar/ is Descent 1 game-specific code; main/ has the engine."
