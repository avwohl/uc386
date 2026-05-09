/* sys/types.h - minimal stub for uc386. Just provides POSIX-ish size_t,
 * ssize_t, off_t, time_t, pid_t with i386-typical widths. Enough for
 * test programs that include the header but don't actually use system
 * calls.
 */
#ifndef _SYS_TYPES_H
#define _SYS_TYPES_H

#include <stddef.h>

typedef long ssize_t;
typedef long off_t;
typedef long time_t;
typedef int pid_t;
typedef unsigned int uid_t;
typedef unsigned int gid_t;
typedef unsigned int mode_t;
typedef unsigned int dev_t;
typedef unsigned long ino_t;
typedef long blkcnt_t;
typedef long blksize_t;
typedef int nlink_t;

#endif /* _SYS_TYPES_H */
