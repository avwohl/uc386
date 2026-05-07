#!/usr/bin/env bash
# uc386 MicroPython end-to-end TLS rig.
#
# Boots FreeDOS in QEMU with NE2000 + SLIRP user-mode networking,
# runs MP.EXE under PMODE/W with a tiny test script (TLSTEST.PY)
# that initialises lwIP, gets a DHCP lease, opens a TCP socket to
# the host (reachable as 10.0.2.2 via SLIRP), and does a real TLS
# handshake against tls_server.py running on the host. The server
# responds with a fixed marker, the script prints PASS, the rig
# extracts the floppy's TLSOUT.TXT and greps for the marker.
#
# This is the wire-level analogue of the dos_emu library smoke
# tests (`test_micropython_ssl_*`). The library tests prove the
# axtls integration parses certs and sets up SSLContext; this
# rig proves it can actually finish a handshake against a peer.
#
# Prereqs:
#   - qemu-system-i386 (brew install qemu / apt install qemu-system-x86)
#   - mtools (brew install mtools / apt install mtools)
#   - openssl (system)
#   - python3 (with stdlib ssl)
#   - MP.EXE built via addons/harness/exe.py (see ../README.md)
#
# Usage:
#   ./run-tls-rig.sh                    # uses ../dosbox-x-rig/MP.EXE
#   ./run-tls-rig.sh /path/to/MP.EXE
#
# Output: prints the captured TLSOUT.TXT and exits 0 on PASS, 1 on
# anything else.

set -eu

cd "$(dirname "$0")"

MP_EXE="${1:-../dosbox-x-rig/MP.EXE}"
NE2000_COM="../dosbox-x-rig/NE2000.COM"
TLS_PORT="${TLS_PORT:-8443}"

if [ ! -f "$MP_EXE" ]; then
    echo "MP.EXE not found at $MP_EXE — build via addons/harness/exe.py" >&2
    exit 1
fi
if [ ! -f "$NE2000_COM" ]; then
    echo "NE2000.COM not found at $NE2000_COM — needed for guest packet driver" >&2
    exit 1
fi

# 1. Generate a fresh self-signed cert (CN=10.0.2.2) every run. The
#    guest will load this exact PEM via load_verify_locations and
#    verify_mode=CERT_REQUIRED, so the handshake exercises the full
#    chain-validation path against a known-good peer.
echo "[rig] generating test cert ..."
openssl req -x509 -nodes -newkey rsa:2048 \
    -subj "/CN=uc386-tls-rig" \
    -addext "subjectAltName=IP:10.0.2.2,IP:127.0.0.1" \
    -days 30 \
    -keyout test-server.key -out test-server.crt 2>/dev/null

# Single PEM with just the cert — what the guest needs as a CA bundle.
cp test-server.crt test-server-ca.pem

# 2. Build the FAT12 floppy: FreeDOS minimal + MP.EXE + NE2000.COM
#    + AUTOEXEC.BAT + TLSTEST.PY + TESTCA.PEM (the cert).
FREEDOS_IMG=/tmp/freedos.img
TEST_IMG="$(pwd)/freedos-tls.img"
LOG="$(pwd)/qemu-tls.log"

if [ ! -f "$FREEDOS_IMG" ]; then
    echo "[rig] fetching FreeDOS boot floppy ..."
    curl -fsSL -o "$FREEDOS_IMG" \
        https://raw.githubusercontent.com/codercowboy/freedosbootdisks/master/bootdisks/freedos.boot.disk.1.4MB.img
fi

echo "[rig] building FAT12 image ..."
cp "$FREEDOS_IMG" "$TEST_IMG"
mcopy -i "$TEST_IMG" -o "$MP_EXE"      ::MP.EXE
mcopy -i "$TEST_IMG" -o "$NE2000_COM"  ::NE2000.COM
mcopy -i "$TEST_IMG" -o ./TLSTEST.PY   ::TLSTEST.PY
mcopy -i "$TEST_IMG" -o ./test-server-ca.pem ::TESTCA.PEM

# AUTOEXEC.BAT: load the packet driver at INT 0x60 / IRQ 9 / IO 0x300
# (matches QEMU's ne2k_isa default + pktdrv_uc386dos.c's IVT scan
# target), then run MP.EXE with the test script as stdin and stdout
# captured in TLSOUT.TXT. AFTER MP exits the 'DONE' marker prints to
# COM1 (no CTTY — DOS console is screen) so the rig can detect
# completion via the serial log.
AUTOEXEC=$(mktemp)
{
    printf '@echo off\r\n'
    printf 'NE2000 0x60 9 0x300 > NUL\r\n'
    printf 'echo === rig start === > COM1\r\n'
    printf 'MP.EXE < TLSTEST.PY > TLSOUT.TXT\r\n'
    printf 'echo === rig done === > COM1\r\n'
} > "$AUTOEXEC"
mcopy -i "$TEST_IMG" -o "$AUTOEXEC" ::AUTOEXEC.BAT
rm -f "$AUTOEXEC"

# 3. Start the host TLS server in the background. Reaped at exit.
echo "[rig] starting tls_server.py on port $TLS_PORT ..."
python3 ./tls_server.py --port "$TLS_PORT" --max-seconds 90 \
    > tls-server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    rc=$?
    [ -n "${QEMU_PID:-}" ] && kill -KILL "$QEMU_PID" 2>/dev/null || true
    [ -n "${SERVER_PID:-}" ] && kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$QEMU_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    return $rc
}
trap cleanup EXIT INT TERM

# 4. Boot QEMU with NE2000 (ne2k_isa, the Crynwr driver target) on
#    its conventional IO 0x300 / IRQ 9, plugged into a SLIRP user-mode
#    netdev. SLIRP's built-in DHCP server hands the guest 10.0.2.15
#    with gateway 10.0.2.2 — that's the host's view of itself, where
#    the TLS server is listening. No hostfwd needed; the guest reaches
#    the server by talking to the gateway.
echo "[rig] booting QEMU + FreeDOS + NE2000 (log: $LOG) ..."
qemu-system-i386 \
    -display none \
    -serial stdio \
    -fda "$TEST_IMG" \
    -boot a \
    -m 16 \
    -cpu pentium \
    -netdev user,id=net0 \
    -device ne2k_isa,netdev=net0,iobase=0x300,irq=9 \
    -no-reboot \
    < /dev/null > "$LOG" 2>&1 &
QEMU_PID=$!

# Watch for the "rig done" marker in COM1 output. Cap at 180s.
for _ in $(seq 1 180); do
    sleep 1
    if grep -q "rig done" "$LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$QEMU_PID" 2>/dev/null; then
        break
    fi
done
kill "$QEMU_PID" 2>/dev/null || true
sleep 1
kill -KILL "$QEMU_PID" 2>/dev/null || true
wait "$QEMU_PID" 2>/dev/null || true

# 5. Pull TLSOUT.TXT off the floppy.
mcopy -i "$TEST_IMG" -n -o ::TLSOUT.TXT ./tls-out.txt 2>/dev/null || true

echo
echo "=== Captured COM1 (qemu-tls.log) ==="
cat "$LOG"
echo
echo "=== TLSOUT.TXT (MP stdout) ==="
if [ -f ./tls-out.txt ]; then
    cat ./tls-out.txt
else
    echo "(no TLSOUT.TXT produced)"
fi
echo
echo "=== tls-server.log (host TLS server) ==="
cat tls-server.log

echo
echo "=== Result ==="
if [ -f ./tls-out.txt ] && grep -q "TLSTEST: PASS" ./tls-out.txt; then
    echo "PASS: end-to-end TLS handshake completed; uc386 MP read decrypted bytes from the host TLS server."
    exit 0
else
    echo "FAIL: see logs above"
    exit 1
fi
