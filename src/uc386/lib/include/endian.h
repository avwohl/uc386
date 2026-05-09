/* endian.h — GNU-style byte-order helpers. uc386 targets i386 which
 * is little-endian; *toh* macros are byte-swaps for big-endian and
 * no-ops for little-endian. axtls's sha512.c uses be64toh() in its
 * message-schedule expansion. */
#ifndef _ENDIAN_H
#define _ENDIAN_H

#include <stdint.h>

#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN    4321
#define __BYTE_ORDER    __LITTLE_ENDIAN

static inline uint16_t __uc386_bswap16h(uint16_t x) {
    return (uint16_t)((x >> 8) | (x << 8));
}

static inline uint32_t __uc386_bswap32h(uint32_t x) {
    return ((x & 0xFF000000u) >> 24)
         | ((x & 0x00FF0000u) >>  8)
         | ((x & 0x0000FF00u) <<  8)
         | ((x & 0x000000FFu) << 24);
}

static inline uint64_t __uc386_bswap64h(uint64_t x) {
    return ((uint64_t)__uc386_bswap32h((uint32_t)x) << 32)
         | (uint64_t)__uc386_bswap32h((uint32_t)(x >> 32));
}

/* Big-endian to/from host. On i386 (little-endian) the *be* forms swap. */
#define htobe16(x) __uc386_bswap16h((uint16_t)(x))
#define htobe32(x) __uc386_bswap32h((uint32_t)(x))
#define htobe64(x) __uc386_bswap64h((uint64_t)(x))
#define be16toh(x) __uc386_bswap16h((uint16_t)(x))
#define be32toh(x) __uc386_bswap32h((uint32_t)(x))
#define be64toh(x) __uc386_bswap64h((uint64_t)(x))

/* Little-endian to/from host. On i386 these are no-ops. */
#define htole16(x) ((uint16_t)(x))
#define htole32(x) ((uint32_t)(x))
#define htole64(x) ((uint64_t)(x))
#define le16toh(x) ((uint16_t)(x))
#define le32toh(x) ((uint32_t)(x))
#define le64toh(x) ((uint64_t)(x))

#endif /* _ENDIAN_H */
