// Crynwr "FTP Software" packet-driver bindings (INT 0x60–0x7F).
//
// Detection: walk the IVT looking for the 8-byte signature
// "PKT DRVR" at offset +3 of the handler routine (the leading two
// bytes are a `JMP SHORT +8` over the signature). The IVT lives at
// linear 0x0000:0xNNNN — under PMODE/W's flat 32-bit model the
// real-mode IVT is reachable via the same flat address space, just
// at the bottom of memory.
//
// Calls used:
//   AH=0x01 driver_info     (probe — confirms a handle is valid)
//   AH=0x02 access_type     (register a receiver callback)
//   AH=0x04 send_pkt        (transmit a frame)
//   AH=0x06 get_address     (read the NIC's MAC)
//   AH=0x05 release_type    (unregister at shutdown — not yet wired)
//
// Receive is callback-driven: on packet arrival, the driver calls
// our handler twice — first with AX=0 to get a buffer pointer, then
// with AX=1 once it's copied the bytes in. We funnel both into a
// small RX queue (`pktdrv_rx_queue`) that ethdrv_recv polls.
//
// Real-DOS deployment notes (TODO): the AX=0/AX=1 callback runs in
// real mode; PMODE/W requires us to allocate a real-mode trampoline
// via DPMI INT 31h fn 0x0303 that ferries the call into 32-bit
// protected mode and translates the seg:offset return into a flat
// linear pointer. Under dos_emu the receiver is invoked directly as
// a 32-bit cdecl function — the emulator does the impedance match.

#include <stddef.h>
#include <string.h>

extern unsigned char pktdrv_int_invoke(unsigned int int_num,
                                       unsigned int regs_in_out[8]);

// regs_in_out indices, mirrored in the asm wrapper.
#define R_EAX 0
#define R_EBX 1
#define R_ECX 2
#define R_EDX 3
#define R_ESI 4
#define R_EDI 5
#define R_DS  6
#define R_ES  7

static const unsigned char PKT_SIG[8] = "PKT DRVR";

static int pktdrv_int_num   = 0;   // 0 = not detected
static int pktdrv_handle    = -1;  // access_type result
static unsigned char pktdrv_mac_cache[6];

// RX ring: the receiver callback is split into TWO calls per packet
// (give-buffer / packet-copied). We hand back `pktdrv_rx_buf` from
// give-buffer, then mark `pktdrv_rx_pending = 1` once the copy is
// done. ethdrv_recv (in lwip_uc386dos.c) polls the pending flag,
// drains the buffer, clears the flag — single-slot ring is enough
// because lwip.callback() pumps after every TX and dos_emu is
// single-threaded so packets land one at a time.
#define PKTDRV_MAX_FRAME 1518
static unsigned char pktdrv_rx_buf[PKTDRV_MAX_FRAME];
volatile int          pktdrv_rx_pending  = 0;
volatile unsigned int pktdrv_rx_len      = 0;

// DPMI INT 0x31 fn 0x0200 — Get Real Mode Interrupt Vector.
// On entry: AX=0x0200, BL=int_num.
// On exit:  CX=segment, DX=offset, CF clear on success.
//
// Reading the real-mode IVT directly from a flat 32-bit binary
// would only be valid if the loader maps the real-mode IVT into our
// flat address space (uc386's dos_emu happens to do that — the IVT
// lives in [0,0x400)). Real DOS via PMODE/W is in protected mode
// with paging, and the real-mode IVT is *not* mapped at 0:0; the
// official path is DPMI 0x0200, which both PMODE/W and dos_emu can
// implement uniformly.
static unsigned int pktdrv_probe_slot(unsigned int int_num) {
    unsigned int regs[8] = {0};
    regs[R_EAX] = 0x0200;
    regs[R_EBX] = int_num & 0xFF;
    unsigned char carry = pktdrv_int_invoke(0x31, regs);
    if (carry) {
        return 0;
    }
    unsigned int seg = regs[R_ECX] & 0xFFFF;
    unsigned int off = regs[R_EDX] & 0xFFFF;
    unsigned int linear = seg * 16 + off;
    if (linear == 0 || linear >= 0x110000) {
        return 0;
    }
    unsigned char *handler = (unsigned char *)linear;
    // Real Crynwr drivers begin with `JMP SHORT 0x08` followed by
    // the 8-byte signature. Some implementations differ on the JMP
    // form; just match the signature at +3.
    for (int i = 0; i < 8; i++) {
        if (handler[3 + i] != PKT_SIG[i]) {
            return 0;
        }
    }
    return linear;
}

// Find a Crynwr packet driver in the IVT. Returns the INT number
// it's installed on, or 0 if none. Caches in pktdrv_int_num.
int pktdrv_detect(void) {
    if (pktdrv_int_num != 0) {
        return pktdrv_int_num;
    }
    for (unsigned int i = 0x60; i < 0x80; i++) {
        if (pktdrv_probe_slot(i) != 0) {
            pktdrv_int_num = (int)i;
            return pktdrv_int_num;
        }
    }
    return 0;
}

// AH=0x06 get_address — fetch the NIC's MAC into `out[6]`.
static int pktdrv_get_addr(unsigned char out[6]) {
    if (pktdrv_int_num == 0 || pktdrv_handle < 0) {
        return -1;
    }
    unsigned int regs[8] = {0};
    regs[R_EAX] = 0x0600;                          // AH=06, AL=0
    regs[R_EBX] = (unsigned int)pktdrv_handle;
    regs[R_ECX] = 6;
    regs[R_EDI] = (unsigned int)(unsigned long)out;
    regs[R_ES]  = 0;                               // flat — dos_emu treats
                                                   // ES:DI as linear EDI.
    unsigned char carry = pktdrv_int_invoke(
        (unsigned int)pktdrv_int_num, regs);
    if (carry) {
        return -1;
    }
    memcpy(pktdrv_mac_cache, out, 6);
    return 0;
}

// AH=0x02 access_type — register `receiver` as the packet handler
// for `ethertype`. Use ethertype=0x0000 with type_len=0 to catch
// every protocol (broadcast netif sees ARP + IPv4 alike).
static int pktdrv_access(unsigned int linear_receiver) {
    unsigned int regs[8] = {0};
    regs[R_EAX] = 0x0200;     // AH=02
    regs[R_EBX] = 0;          // if_type=0 (any)
    regs[R_ECX] = 0;          // type_len=0 -> match all ethertypes
    regs[R_EDX] = 0;          // if_number=0
    regs[R_ESI] = 0;          // type bytes (ignored when len=0)
    regs[R_EDI] = linear_receiver;
    regs[R_DS]  = 0;
    regs[R_ES]  = 0;
    unsigned char carry = pktdrv_int_invoke(
        (unsigned int)pktdrv_int_num, regs);
    if (carry) {
        return -1;
    }
    pktdrv_handle = (int)(regs[R_EAX] & 0xFFFF);
    return 0;
}

// 32-bit receiver. Crynwr semantics on real DOS go through a DPMI
// real-mode-callback thunk that maps AX/CX/ES:DI to/from this cdecl
// signature; dos_emu calls us directly with the same convention.
//   phase==0: caller wants a buffer for `len` bytes. Return a flat
//             pointer into pktdrv_rx_buf or NULL to drop.
//   phase==1: caller has finished the copy. Mark the slot full.
unsigned char *uc386dos_pktdrv_receiver(int phase, unsigned int len) {
    if (phase == 0) {
        if (pktdrv_rx_pending || len > sizeof(pktdrv_rx_buf)) {
            return NULL;
        }
        pktdrv_rx_len = len;
        return pktdrv_rx_buf;
    }
    // phase == 1: packet has been copied.
    pktdrv_rx_pending = 1;
    return NULL;
}

// Public init: detect, register, fetch MAC. Returns 0 on success.
int pktdrv_init(unsigned char mac[6]) {
    if (pktdrv_detect() == 0) {
        return -1;
    }
    if (pktdrv_access((unsigned int)(unsigned long)
                      &uc386dos_pktdrv_receiver) != 0) {
        pktdrv_int_num = 0;  // fall through to alt path
        return -2;
    }
    if (pktdrv_get_addr(mac) != 0) {
        pktdrv_int_num = 0;
        return -3;
    }
    return 0;
}

// AH=0x04 send_pkt — transmit a frame. `buf` is flat-linear; the
// driver under PMODE/W's DPMI host translates DS:SI internally.
int pktdrv_send(const unsigned char *buf, unsigned int len) {
    if (pktdrv_int_num == 0) {
        return -1;
    }
    unsigned int regs[8] = {0};
    regs[R_EAX] = 0x0400;
    regs[R_ECX] = len;
    regs[R_ESI] = (unsigned int)(unsigned long)buf;
    regs[R_DS]  = 0;
    unsigned char carry = pktdrv_int_invoke(
        (unsigned int)pktdrv_int_num, regs);
    return carry ? -1 : 0;
}

// Drain the RX slot if one is pending. Returns the byte count
// written (clamped to maxlen) or 0 if nothing's queued. Truncates
// silently on overflow.
unsigned int pktdrv_recv(unsigned char *out, unsigned int maxlen) {
    if (!pktdrv_rx_pending) {
        return 0;
    }
    unsigned int n = pktdrv_rx_len;
    if (n > maxlen) {
        n = maxlen;
    }
    memcpy(out, pktdrv_rx_buf, n);
    pktdrv_rx_pending = 0;
    pktdrv_rx_len = 0;
    return n;
}

// True when a Crynwr driver was successfully attached at init.
int pktdrv_is_active(void) { return pktdrv_int_num != 0; }
