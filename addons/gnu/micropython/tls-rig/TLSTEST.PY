import sys
import time
import lwip
import uc386_net
import socket
import ssl
print('TLSTEST: start')
print('TLSTEST: pre_lwip_reset')
_ = lwip.reset()
print('TLSTEST: post_lwip_reset')
print('TLSTEST: pre_eth_init')
rc = uc386_net.eth_init(True)
print('TLSTEST: post_eth_init')
# Bypass print formatting — write raw bytes via sys.stdout.write
# to rule out any concat / format / encoding side effects.
sys.stdout.write('TLSTEST: rc_eq_zero=' + str(rc == 0) + '\n')
sys.stdout.write('TLSTEST: rc_eq_minus1=' + str(rc == -1) + '\n')
sys.stdout.write('TLSTEST: rc_eq_minus2=' + str(rc == -2) + '\n')
sys.stdout.write('TLSTEST: rc_eq_minus3=' + str(rc == -3) + '\n')
sys.stdout.write('TLSTEST: rc_lt_zero=' + str(rc < 0) + '\n')
sys.stdout.write('TLSTEST: rc_gt_million=' + str(rc > 1000000) + '\n')
ip = ''
up = False
for tick in range(60):
    _ = lwip.callback()
    info = uc386_net.eth_status()
    ip = info[0]
    up = info[3]
    if up and ip and ip != '0.0.0.0':
        break
    time.sleep_ms(100)
print('TLSTEST: ifup', up, 'ip', ip)
if not up or not ip or ip == '0.0.0.0':
    print('TLSTEST: dhcp_failed')
    sys.exit(1)
addr = socket.getaddrinfo('10.0.2.2', 8443)[0][-1]
s = socket.socket()
try:
    s.connect(addr)
except OSError as e:
    print('TLSTEST: connect_failed', e)
    sys.exit(2)
print('TLSTEST: tcp_connected')
with open('TESTCA.PEM', 'rb') as f:
    ca = f.read()
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ctx.verify_mode = ssl.CERT_REQUIRED
_ = ctx.load_verify_locations(cadata=ca)
print('TLSTEST: ctx_ready')
try:
    tls = ctx.wrap_socket(s)
except OSError as e:
    print('TLSTEST: handshake_failed', e)
    sys.exit(3)
print('TLSTEST: handshake_ok')
_ = tls.write(b'hi-from-uc386')
data = tls.read(64)
print('TLSTEST: received', data)
if b'TLS_RIG_OK' in data:
    print('TLSTEST: PASS')
else:
    print('TLSTEST: FAIL_unexpected_response')
