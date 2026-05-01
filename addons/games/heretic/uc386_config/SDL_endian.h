/* SDL_endian.h — minimal shim for chocolate-doom under uc386.
 *
 * chocolate-doom uses SDL2's endian helpers in i_swap.h. uc386 isn't
 * SDL-aware, but the actual semantics on i386 (always little-endian)
 * collapse to identity. Provide just what i_swap.h consumes.
 */
#ifndef _UC386_SDL_ENDIAN_H
#define _UC386_SDL_ENDIAN_H

#define SDL_LIL_ENDIAN  1234
#define SDL_BIG_ENDIAN  4321
#define SDL_BYTEORDER   SDL_LIL_ENDIAN

/* Identity on little-endian — i_swap.h calls SDL_SwapLE* which is the
 * "no-swap on little-endian, byteswap on big-endian" form. */
#define SDL_Swap16(x)   ((unsigned short)(((unsigned short)(x) << 8) | \
                                          ((unsigned short)(x) >> 8)))
#define SDL_Swap32(x)   ((unsigned int)(((unsigned int)(x) << 24) | \
                                        (((unsigned int)(x) << 8) & 0x00FF0000) | \
                                        (((unsigned int)(x) >> 8) & 0x0000FF00) | \
                                        ((unsigned int)(x) >> 24)))

#define SDL_SwapBE16(x) SDL_Swap16(x)
#define SDL_SwapBE32(x) SDL_Swap32(x)
#define SDL_SwapLE16(x) (x)
#define SDL_SwapLE32(x) (x)

#endif /* _UC386_SDL_ENDIAN_H */
