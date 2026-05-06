"""Simulated network attached to dos_emu's INT 0x83 packet-driver shim.

The uc386 binary calls `ethdrv_init/send/recv` (lib/i386_dos_libc.asm),
which lower to `INT 0x83` with AH = subfunction. dos_emu.on_int reads
ESI/EDI/ECX, hands the bytes to a `NetworkSimulator` instance, and
mirrors the response back through ECX.

The simulator plays the role of the host gateway: it answers ARPs for
its own IP, runs a tiny DHCP server (DISCOVER -> OFFER, REQUEST -> ACK),
and echoes ICMP pings. Anything unrecognized is silently dropped — its
job is to give lwIP enough back-and-forth to get a lease, not to be a
production stack.

Frame layout:
  Ethernet header (14B): dst[6] | src[6] | ethertype[2 BE]
  IPv4 (20B no opts):    ver/ihl, dscp, total_len, id, flags/frag,
                          ttl, proto, hdr_chk, src[4], dst[4]
  UDP (8B):              src_port, dst_port, len, chk
  DHCP (240B fixed +     op, htype, hlen, hops, xid, secs, flags,
   options):              ciaddr, yiaddr, siaddr, giaddr, chaddr[16],
                          sname[64], file[128], magic[4], options[...]
"""

from __future__ import annotations

import struct
from typing import Optional


def _checksum(data: bytes, init: int = 0) -> int:
    """16-bit one's-complement sum (RFC 1071). Used for IPv4/ICMP/UDP."""
    s = init
    if len(data) & 1:
        data = data + b"\x00"
    for i in range(0, len(data), 2):
        s += (data[i] << 8) | data[i + 1]
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _udp_checksum(src_ip: bytes, dst_ip: bytes, udp_dgram: bytes) -> int:
    """UDP checksum with IPv4 pseudo-header."""
    pseudo = src_ip + dst_ip + b"\x00\x11" + struct.pack(">H", len(udp_dgram))
    return _checksum(pseudo + udp_dgram)


# Hardcoded test fabric — single uc386 client, single host gateway.
# The lease is fixed (no real allocation pool); we hand out the same
# IP regardless of XID so deterministic tests can pin the value.
DEFAULT_OUR_MAC      = bytes([0x02, 0x00, 0xDE, 0xAD, 0xBE, 0xEF])  # uc386 program
DEFAULT_HOST_MAC     = bytes([0x02, 0x00, 0xCA, 0xFE, 0xBA, 0xBE])  # gateway
DEFAULT_OFFERED_IP   = bytes([10, 0, 2, 15])
DEFAULT_GATEWAY_IP   = bytes([10, 0, 2, 2])
DEFAULT_DNS_IP       = bytes([10, 0, 2, 3])
DEFAULT_NETMASK      = bytes([255, 255, 255, 0])
DHCP_MAGIC = bytes([99, 130, 83, 99])

# Where in low memory we plant the fake Crynwr "PKT DRVR" signature.
# Far enough past the PSP (0xF1000 end) and well past any address
# the smoke-test heap touches that the binary won't trample it.
PKTDRV_HANDLER_LINEAR = 0x000FE000
PKTDRV_INT_NUM        = 0x60


class NetworkSimulator:
    """Single-client virtual NIC. Plug into `dos_emu.run(net=...)`."""

    def __init__(
        self,
        our_mac: bytes = DEFAULT_OUR_MAC,
        host_mac: bytes = DEFAULT_HOST_MAC,
        offered_ip: bytes = DEFAULT_OFFERED_IP,
        gateway_ip: bytes = DEFAULT_GATEWAY_IP,
        dns_ip: bytes = DEFAULT_DNS_IP,
        netmask: bytes = DEFAULT_NETMASK,
        crynwr_int_num: Optional[int] = PKTDRV_INT_NUM,
    ) -> None:
        assert len(our_mac) == 6 and len(host_mac) == 6
        for b in (offered_ip, gateway_ip, dns_ip, netmask):
            assert len(b) == 4
        self.our_mac = our_mac
        self.host_mac = host_mac
        self.offered_ip = offered_ip
        self.gateway_ip = gateway_ip
        self.dns_ip = dns_ip
        self.netmask = netmask
        self.rx_queue: list[bytes] = []
        self.tx_log: list[bytes] = []
        # When set, dos_emu plants a "PKT DRVR" signature in low
        # memory and answers DPMI INT 0x31 fn 0x0200 + Crynwr INT
        # 0x60 calls so the binary's pktdrv_init() succeeds. Set
        # to None to force the binary onto the INT 0x83 sim path.
        self.crynwr_int_num: Optional[int] = crynwr_int_num
        # Filled in after access_type — the linear address of the
        # binary's receiver. Reserved for the DPMI-thunk-based RX
        # path that lives behind a TODO today.
        self.pktdrv_receiver_addr: int = 0
        self.pktdrv_handle: int = 1
        # Filled in by AH=0x99 (uc386dos extension): linear addrs of
        # pktdrv_rx_buf / pktdrv_rx_pending / pktdrv_rx_len. Lets the
        # harness post inbound frames straight into the binary's RX
        # slot. Replaced by a real-mode-callback DPMI thunk on real
        # hardware.
        self.pktdrv_rx_buf_addr: int = 0
        self.pktdrv_rx_pending_addr: int = 0
        self.pktdrv_rx_len_addr: int = 0

    # ---- INT 0x83 entry points ------------------------------------

    def init_mac(self) -> bytes:
        """AH=0: caller-supplied 6-byte buf gets the program's MAC."""
        return self.our_mac

    def send_frame(self, frame: bytes) -> int:
        """AH=1: program is shipping a frame. Returns 0 on success."""
        self.tx_log.append(frame)
        if len(frame) < 14:
            return 0
        ethertype = struct.unpack(">H", frame[12:14])[0]
        if ethertype == 0x0806:  # ARP
            self._handle_arp(frame[14:])
        elif ethertype == 0x0800:  # IPv4
            self._handle_ipv4(frame)
        # Other ethertypes (0x86DD IPv6, 0x8100 vlan, ...) — silently drop.
        return 0

    def recv_frame(self, maxlen: int) -> Optional[bytes]:
        """AH=2: dequeue next frame. Returns bytes (≤ maxlen) or None."""
        while self.rx_queue:
            f = self.rx_queue.pop(0)
            if len(f) <= maxlen:
                return f
            # Frame too big for caller's buffer — drop. (No fragmentation
            # protocol at the eth layer; oversize means a misconfig.)
        return None

    # ---- ARP -------------------------------------------------------

    def _handle_arp(self, payload: bytes) -> None:
        if len(payload) < 28:
            return
        (htype, ptype, hlen, plen, op) = struct.unpack(">HHBBH", payload[:8])
        if htype != 1 or ptype != 0x0800 or hlen != 6 or plen != 4:
            return
        if op != 1:  # only handle requests; replies are absorbed silently.
            return
        sha = payload[8:14]
        spa = payload[14:18]
        # tha = payload[18:24]  # unused on requests
        tpa = payload[24:28]
        if tpa != self.gateway_ip:
            # We only own the gateway IP; ignore queries for other addrs.
            return
        # Build ARP reply.
        reply = struct.pack(">HHBBH", 1, 0x0800, 6, 4, 2)
        reply += self.host_mac + self.gateway_ip + sha + spa
        self._enqueue_eth(dst=sha, src=self.host_mac,
                          ethertype=0x0806, payload=reply)

    # ---- IPv4 ------------------------------------------------------

    def _handle_ipv4(self, frame: bytes) -> None:
        ip_hdr_start = 14
        if len(frame) < ip_hdr_start + 20:
            return
        ver_ihl = frame[ip_hdr_start]
        if (ver_ihl >> 4) != 4:
            return
        ihl = (ver_ihl & 0x0F) * 4
        if ihl < 20 or len(frame) < ip_hdr_start + ihl:
            return
        proto = frame[ip_hdr_start + 9]
        src_ip = frame[ip_hdr_start + 12:ip_hdr_start + 16]
        dst_ip = frame[ip_hdr_start + 16:ip_hdr_start + 20]
        payload = frame[ip_hdr_start + ihl:]
        if proto == 17:  # UDP
            self._handle_udp(src_ip, dst_ip, payload)
        elif proto == 1:  # ICMP
            self._handle_icmp(src_ip, dst_ip, payload)
        # TCP/IGMP/etc — drop.

    # ---- UDP / DHCP -----------------------------------------------

    def _handle_udp(self, src_ip: bytes, dst_ip: bytes, dgram: bytes) -> None:
        if len(dgram) < 8:
            return
        sport, dport, length, _chk = struct.unpack(">HHHH", dgram[:8])
        body = dgram[8:length]
        # DHCP client → server: dport=67. We're the server.
        if dport == 67 and sport == 68:
            self._handle_dhcp(body)

    def _handle_dhcp(self, msg: bytes) -> None:
        if len(msg) < 240:
            return
        op = msg[0]
        if op != 1:  # 1=BOOTREQUEST; 2=BOOTREPLY (we don't accept)
            return
        xid = msg[4:8]
        chaddr = msg[28:28 + 6]
        if msg[236:240] != DHCP_MAGIC:
            return
        opts = self._parse_dhcp_options(msg[240:])
        msg_type = opts.get(53, b"")
        if not msg_type:
            return
        mt = msg_type[0]
        if mt == 1:    # DHCPDISCOVER -> DHCPOFFER
            self._send_dhcp_reply(xid, chaddr, dhcp_msg_type=2)
        elif mt == 3:  # DHCPREQUEST  -> DHCPACK
            self._send_dhcp_reply(xid, chaddr, dhcp_msg_type=5)
        # 4=DECLINE, 7=RELEASE, 8=INFORM — ignore.

    @staticmethod
    def _parse_dhcp_options(opts: bytes) -> dict[int, bytes]:
        out: dict[int, bytes] = {}
        i = 0
        while i < len(opts):
            tag = opts[i]
            if tag == 0:    # PAD
                i += 1
                continue
            if tag == 255:  # END
                break
            if i + 1 >= len(opts):
                break
            length = opts[i + 1]
            if i + 2 + length > len(opts):
                break
            out[tag] = opts[i + 2:i + 2 + length]
            i += 2 + length
        return out

    def _send_dhcp_reply(self, xid: bytes, chaddr: bytes,
                         dhcp_msg_type: int) -> None:
        # BOOTREPLY shell.
        reply = bytearray(240)
        reply[0] = 2                    # op = BOOTREPLY
        reply[1] = 1                    # htype = ethernet
        reply[2] = 6                    # hlen
        reply[3] = 0                    # hops
        reply[4:8] = xid
        reply[8:10] = b"\x00\x00"       # secs
        reply[10:12] = b"\x00\x00"      # flags
        reply[12:16] = b"\x00\x00\x00\x00"  # ciaddr
        reply[16:20] = self.offered_ip       # yiaddr
        reply[20:24] = self.gateway_ip       # siaddr (next-server)
        reply[24:28] = b"\x00\x00\x00\x00"   # giaddr
        reply[28:34] = chaddr
        reply[44:108] = b"\x00" * 64    # sname
        reply[108:236] = b"\x00" * 128  # file
        reply[236:240] = DHCP_MAGIC
        # Options.
        opts = bytearray()
        opts += bytes([53, 1, dhcp_msg_type])         # DHCP Message Type
        opts += bytes([54, 4]) + self.gateway_ip      # Server Identifier
        opts += bytes([51, 4, 0, 0, 0x0E, 0x10])      # Lease 3600s
        opts += bytes([1, 4]) + self.netmask          # Subnet Mask
        opts += bytes([3, 4]) + self.gateway_ip       # Router
        opts += bytes([6, 4]) + self.dns_ip           # DNS
        opts += bytes([255])                          # END
        if len(opts) % 2:
            opts += b"\x00"
        msg = bytes(reply) + bytes(opts)
        # Wrap in UDP 67 -> 68 directed at the broadcast eth address
        # (lwIP's DHCP client doesn't have an IP yet, so broadcast).
        udp_len = 8 + len(msg)
        udp_hdr_pre = struct.pack(">HHHH", 67, 68, udp_len, 0)
        udp_chk = _udp_checksum(self.gateway_ip, b"\xff\xff\xff\xff",
                                udp_hdr_pre + msg)
        udp = struct.pack(">HHHH", 67, 68, udp_len, udp_chk) + msg
        self._send_ipv4(self.gateway_ip, b"\xff\xff\xff\xff", 17, udp,
                        eth_dst=b"\xff\xff\xff\xff\xff\xff")

    # ---- ICMP ------------------------------------------------------

    def _handle_icmp(self, src_ip: bytes, dst_ip: bytes,
                     icmp_msg: bytes) -> None:
        if len(icmp_msg) < 8:
            return
        if icmp_msg[0] != 8:  # only echo-request
            return
        if dst_ip != self.gateway_ip:
            return
        # Build echo-reply: type 0, recompute checksum.
        body = bytearray(icmp_msg)
        body[0] = 0
        body[2:4] = b"\x00\x00"
        chk = _checksum(bytes(body))
        body[2:4] = struct.pack(">H", chk)
        self._send_ipv4(self.gateway_ip, src_ip, 1, bytes(body))

    # ---- frame builders -------------------------------------------

    def _send_ipv4(self, src_ip: bytes, dst_ip: bytes, proto: int,
                   payload: bytes,
                   eth_dst: Optional[bytes] = None) -> None:
        total_len = 20 + len(payload)
        hdr = bytearray(20)
        hdr[0] = 0x45               # ver=4 ihl=5
        hdr[1] = 0x00               # dscp/ecn
        hdr[2:4] = struct.pack(">H", total_len)
        hdr[4:6] = b"\x00\x00"      # id
        hdr[6:8] = b"\x00\x00"      # flags/frag (we don't fragment)
        hdr[8] = 64                 # ttl
        hdr[9] = proto
        hdr[10:12] = b"\x00\x00"    # checksum (filled below)
        hdr[12:16] = src_ip
        hdr[16:20] = dst_ip
        hdr[10:12] = struct.pack(">H", _checksum(bytes(hdr)))
        if eth_dst is None:
            # Default: send to the program's MAC (we know it).
            eth_dst = self.our_mac
        self._enqueue_eth(dst=eth_dst, src=self.host_mac,
                          ethertype=0x0800,
                          payload=bytes(hdr) + payload)

    def _enqueue_eth(self, dst: bytes, src: bytes,
                     ethertype: int, payload: bytes) -> None:
        frame = dst + src + struct.pack(">H", ethertype) + payload
        # Pad to minimum eth frame (60B = 64 - 4 byte FCS, but lwIP
        # doesn't expect FCS so 60 is the practical minimum).
        if len(frame) < 60:
            frame = frame + b"\x00" * (60 - len(frame))
        self.rx_queue.append(frame)
