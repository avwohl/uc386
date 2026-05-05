// uc386-dos lwIP port glue. Provides:
//   - sys_now() — millisecond tick from BIOS (INT 1Ah AH=0).
//   - lwip_uc386dos_init() — calls lwip_init() once + adds the
//     loopback netif. Called from main.c at boot.
//   - mp_module_socket — re-export of lwIP's mp_module_lwip under
//     `MP_QSTR_socket` so `import socket` lights up the BSD-style
//     surface modlwip.c provides.
//
// We don't ship modnetwork.c — that's the higher-level network
// configuration interface used by ports with WLAN/Ethernet drivers.
// For loopback-only Phase 1 testing, the raw socket surface
// (modlwip.c → MP_QSTR_socket) is sufficient.

#include <string.h>

#include "py/runtime.h"
#include "lwip/init.h"
#include "lwip/timeouts.h"
#include "lwip/netif.h"
#if LWIP_HAVE_LOOPIF
#include "lwip/sys.h"
#endif

// `bios_ticks()` from lib/i386_dos_libc.asm — INT 1Ah AH=0 read of
// the BIOS tick counter (~18.2 Hz). Multiply by 55 for an
// approximate millisecond clock; lwIP's timeouts only need
// monotonic ms, not absolute wall time, so the small drift is fine.
extern unsigned bios_ticks(void);

uint32_t sys_now(void) {
    return bios_ticks() * 55u;
}

#if LWIP_HAVE_LOOPIF
// One-shot lwIP init. Call from `main()` after mp_init.
// Loopback netif is auto-added by lwip_init() when LWIP_HAVE_LOOPIF
// is on, so we just call lwip_init() and we're done.
void lwip_uc386dos_init(void) {
    lwip_init();
}
#endif

// Drive periodic timer checks. Call from the REPL idle loop or a
// dedicated tick. modlwip.c expects sys_check_timeouts() to be
// pumped regularly so TCP retransmits / DNS timeouts fire.
void lwip_uc386dos_poll(void) {
    sys_check_timeouts();
}

// Stub for `mp_mod_network_prefer_dns_use_ip_version` — modlwip.c
// reads it inside getaddrinfo() to pick AF_INET vs AF_INET6 ordering.
// modnetwork.c (which we don't compile) defines it as a real global;
// we ship the stub here defaulting to IPv4 preferred. If we ever
// turn LWIP_IPV6 on, switch this to 6 or wire modnetwork properly.
int mp_mod_network_prefer_dns_use_ip_version = 4;

// modlwip.c also references the hostname buffer that modnetwork.c
// otherwise owns. Provide a stub matching the symbol shape.
char mod_network_hostname_data[16 + 1] = "uc386-dos";

// Expose `mp_module_lwip` as `socket` too. modlwip.c registers
// itself under MP_QSTR_lwip and MP_QSTR_socket via `MP_REGISTER_*`
// markers — but our hand-rolled moduledefs.h doesn't process those.
// The `socket` registration goes through UCDOS_MOD_ENTRY_SOCKET in
// build.sh's heredoc; we don't need any C-side glue for that.
