/* sys/select.h — fd_set + select() shim for the uc386 libc.
 *
 * uc386's port targets aren't built around BSD select() (FreeDOS
 * has no kernel-side fd multiplexing, MS-DOS likewise). This header
 * exists so code that includes <sys/select.h> for fd_set / FD_SET /
 * FD_ZERO compiles cleanly. The macros are no-ops; select() is
 * declared but no implementation is provided — callers that
 * actually wire it up will get a link error. In practice, MP-port
 * users force their session into non-blocking mode so this never
 * runs at runtime.
 */
#ifndef _SYS_SELECT_H
#define _SYS_SELECT_H

#include <sys/time.h>     /* struct timeval */
#include <sys/types.h>    /* time_t, etc. */

#ifndef FD_SETSIZE
#define FD_SETSIZE 64
#endif

typedef struct {
    unsigned long fds_bits[(FD_SETSIZE + 31) / 32];
} fd_set;

#define FD_ZERO(set)        do { int _i; for (_i = 0; _i < (FD_SETSIZE + 31) / 32; _i++) (set)->fds_bits[_i] = 0; } while (0)
#define FD_SET(fd, set)     ((void)((set)->fds_bits[(fd) >> 5] |= (1UL << ((fd) & 31))))
#define FD_CLR(fd, set)     ((void)((set)->fds_bits[(fd) >> 5] &= ~(1UL << ((fd) & 31))))
#define FD_ISSET(fd, set)   (((set)->fds_bits[(fd) >> 5] >> ((fd) & 31)) & 1)

int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
           struct timeval *timeout);

#endif
