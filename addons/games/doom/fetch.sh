#!/bin/sh
# Fetch the original DOOM source release from id Software's public
# archive (released GPL 1997). Targets the linuxdoom-1.10 tree —
# the original DOS source had `dmx.c` (sound DPMI) which is the
# trickier port; the Linux subset is closer to "plain C99 + libc".
set -eu

cd "$(dirname "$0")"
if [ -d upstream ]; then
    echo "doom: upstream/ already populated; skip fetch."
    exit 0
fi

URL="https://github.com/id-Software/DOOM/archive/refs/heads/master.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "doom: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/doom.tar.gz"
tar -xzf "$TMP/doom.tar.gz" -C "$TMP"
mv "$TMP"/DOOM-* upstream
echo "doom: upstream tree at addons/games/doom/upstream/"
echo "doom: linuxdoom-1.10 has the cleanest sources (no DPMI sound)."
