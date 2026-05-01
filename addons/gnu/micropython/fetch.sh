#!/bin/sh
# Download MicroPython upstream (latest master) into upstream/.
# Idempotent: a second run is a no-op once upstream/ exists.
set -eu
cd "$(dirname "$0")"

if [ -d upstream ]; then
    echo "micropython: upstream/ already present — skipping fetch."
    exit 0
fi

echo "micropython: cloning github.com/micropython/micropython master …"
git clone --depth 1 https://github.com/micropython/micropython.git upstream
echo "micropython: $(cd upstream && git log -1 --format='%H %s')"
