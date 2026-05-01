/* alloca.h — stack-allocated buffers.
 *
 * uc386 doesn't currently support __builtin_alloca; we approximate by
 * forwarding to malloc(). Programs that rely on automatic free at
 * function exit will leak — but they'll function correctly otherwise.
 * Doom uses alloca() in r_data.c for one-shot scratch buffers; the
 * leak is bounded.
 */
#ifndef _ALLOCA_H
#define _ALLOCA_H

#include <stddef.h>
#include <stdlib.h>

#define alloca(size)  malloc(size)

#endif /* _ALLOCA_H */
