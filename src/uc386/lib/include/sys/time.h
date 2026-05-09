/* sys/time.h — POSIX timeval shim. axtls uses this for session-id
 * randomness (gettimeofday seeds a weak RNG) and not much else when
 * built in MP's SKELETON_MODE config. Real entropy comes from the
 * port-supplied PLATFORM_RNG path.
 */
#ifndef _SYS_TIME_H
#define _SYS_TIME_H

#include <time.h>  /* time_t */

struct timeval {
    time_t tv_sec;
    long   tv_usec;
};

int gettimeofday(struct timeval *tv, void *tz);

#endif /* _SYS_TIME_H */
