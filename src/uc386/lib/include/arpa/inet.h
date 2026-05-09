/* arpa/inet.h — minimal shim for axtls and other code that pulls
 * htons/htonl/ntohs/ntohl from the POSIX-flavoured location. The full
 * networking surface (struct sockaddr, inet_pton, ...) lives in lwIP
 * — this header only provides the byte-order helpers axtls touches.
 *
 * i386 is little-endian, so host-to-network is always a swap.
 */
#ifndef _ARPA_INET_H
#define _ARPA_INET_H

#include <stdint.h>

static inline uint16_t __uc386_bswap16(uint16_t x) {
    return (uint16_t)((x >> 8) | (x << 8));
}

static inline uint32_t __uc386_bswap32(uint32_t x) {
    return ((x & 0xFF000000u) >> 24)
         | ((x & 0x00FF0000u) >>  8)
         | ((x & 0x0000FF00u) <<  8)
         | ((x & 0x000000FFu) << 24);
}

#define htons(x) __uc386_bswap16((uint16_t)(x))
#define ntohs(x) __uc386_bswap16((uint16_t)(x))
#define htonl(x) __uc386_bswap32((uint32_t)(x))
#define ntohl(x) __uc386_bswap32((uint32_t)(x))

#endif /* _ARPA_INET_H */
