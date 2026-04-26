/* sys/stat.h - stub for uc386. */
#ifndef _SYS_STAT_H
#define _SYS_STAT_H

#include <sys/types.h>

struct stat {
    dev_t   st_dev;
    ino_t   st_ino;
    mode_t  st_mode;
    nlink_t st_nlink;
    uid_t   st_uid;
    gid_t   st_gid;
    dev_t   st_rdev;
    off_t   st_size;
    blksize_t st_blksize;
    blkcnt_t  st_blocks;
    time_t  st_atime;
    time_t  st_mtime;
    time_t  st_ctime;
};

#define S_IFMT   0170000
#define S_IFREG  0100000
#define S_IFDIR  0040000
#define S_IFCHR  0020000
#define S_IFBLK  0060000
#define S_IFIFO  0010000

#define S_IRUSR 0400
#define S_IWUSR 0200
#define S_IXUSR 0100
#define S_IRWXU 0700

int stat(const char *path, struct stat *buf);
int fstat(int fd, struct stat *buf);
int lstat(const char *path, struct stat *buf);
int chmod(const char *path, mode_t mode);
int mkdir(const char *path, mode_t mode);

#endif /* _SYS_STAT_H */
