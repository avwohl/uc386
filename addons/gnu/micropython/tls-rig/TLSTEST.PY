# MicroPython side of the uc386 TLS rig. Boots up networking via the
# Crynwr ne2000 packet driver + lwIP, opens a TCP socket to the SLIRP
# host gateway (10.0.2.2) on port 8443, runs a TLS handshake against
# the host's tls_server.py with the cert it issued, sends a known
# string, and prints the server's reply.
#
# Output goes to TLSOUT.TXT via AUTOEXEC.BAT's `> TLSOUT.TXT` redirect.
# After QEMU exits, the rig pulls TLSOUT.TXT off the FAT image and
# greps for the success marker.

import sys
import time

print('TLSTEST: start')

import lwip
import uc386_net
lwip.reset()
print('TLSTEST: eth_init')
uc386_net.eth_init()  # DHCP via SLIRP — gives us 10.0.2.15

# Wait for DHCP. lwIP runs NO_SYS=1, so we have to drive its state
# machine forward by calling lwip.callback() which pumps the poll
# hook (NE2000 RX -> netif input) and sys_check_timeouts (DHCP retry
# pacing). Without this, blocking lwIP calls work because they pump
# internally, but a sleep+poll loop like the one we want here does
# not — DHCP would never complete.
ip = ''
for tick in range(200):
    lwip.callback()
    info = uc386_net.eth_status()
    ip = info[0]
    up = info[3]
    if up and ip and ip != '0.0.0.0':
        print('TLSTEST: ip', ip, 'gw', info[2])
        break
    # Tiny sleep so we don't spin too tight; the BIOS tick is ~55ms.
    time.sleep_ms(50)
else:
    print('TLSTEST: dhcp_failed')
    sys.exit(1)

print('TLSTEST: connecting to 10.0.2.2:8443')
import socket
addr = socket.getaddrinfo('10.0.2.2', 8443)[0][-1]
s = socket.socket()
try:
    s.connect(addr)
except OSError as e:
    print('TLSTEST: connect_failed', e)
    sys.exit(2)
print('TLSTEST: tcp_connected')

# Load the same cert the rig handed to tls_server.py — we trust this
# specific server. CERT_REQUIRED makes axtls walk the chain and check
# the parsed notBefore / notAfter window against time() (real DOS RTC).
with open('TESTCA.PEM', 'rb') as f:
    ca = f.read()

import ssl
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.verify_mode = ssl.CERT_REQUIRED
ctx.load_verify_locations(cadata=ca)
print('TLSTEST: ctx ready, verify_mode', ctx.verify_mode)

try:
    tls = ctx.wrap_socket(s)
except OSError as e:
    print('TLSTEST: handshake_failed', e)
    sys.exit(3)
print('TLSTEST: handshake_ok')

tls.write(b'hi-from-uc386')
data = tls.read(64)
print('TLSTEST: received', data)

if b'TLS_RIG_OK' in data:
    print('TLSTEST: PASS')
else:
    print('TLSTEST: FAIL_unexpected_response')
