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

/* Interval timers. DOS delivers no SIGALRM, so setitimer() is a
 * benign no-op (portable code such as git's progress.c arms an
 * ITIMER_REAL tick for the progress spinner; without delivery the
 * spinner just never refreshes — correct, only cosmetic). Defined
 * in the libc rather than via git's NO_SETITIMER /
 * NO_STRUCT_ITIMERVAL knobs so there is exactly one `struct
 * itimerval` in the whole-program link (a second, git-provided one
 * would alias-conflict the way zlib's gz_header_s did). */
#define ITIMER_REAL    0
#define ITIMER_VIRTUAL 1
#define ITIMER_PROF    2

struct itimerval {
    struct timeval it_interval;
    struct timeval it_value;
};

int setitimer(int which, const struct itimerval *new_value,
               struct itimerval *old_value);
int getitimer(int which, struct itimerval *cur_value);

#endif /* _SYS_TIME_H */
