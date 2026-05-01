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

# Pinned to a specific upstream commit so the source we ship in the
# games tarball alongside doom.bin always matches the binary. id's
# repo is essentially frozen (last commit 2024-01) so this pin will
# rarely move. Bump with `git ls-remote https://github.com/id-Software/DOOM HEAD`.
SHA="a77dfb96cb91780ca334d0d4cfd86957558007e0"  # 2024-01-16
URL="https://github.com/id-Software/DOOM/archive/${SHA}.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "doom: fetching $URL …"
curl -fsSL "$URL" -o "$TMP/doom.tar.gz"
tar -xzf "$TMP/doom.tar.gz" -C "$TMP"
mv "$TMP"/DOOM-* upstream
echo "doom: upstream tree at addons/games/doom/upstream/"
echo "doom: linuxdoom-1.10 has the cleanest sources (no DPMI sound)."
