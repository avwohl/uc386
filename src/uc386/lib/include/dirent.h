/* dirent.h — POSIX directory iteration. dos_emu has no real
 * directory model, but enough programs include this header that
 * we provide opaque-typed stubs that link cleanly.
 */
#ifndef _DIRENT_H
#define _DIRENT_H

#include <stddef.h>
#include <sys/types.h>

/* d_type values — POSIX optional. */
#define DT_UNKNOWN 0
#define DT_FIFO    1
#define DT_CHR     2
#define DT_DIR     4
#define DT_BLK     6
#define DT_REG     8
#define DT_LNK     10
#define DT_SOCK    12
#define DT_WHT     14

struct dirent {
    ino_t    d_ino;
    off_t    d_off;
    unsigned short d_reclen;
    unsigned char  d_type;
    char     d_name[260];
};

/* Opaque DIR — the runtime version returns NULL for opendir, so
 * the rest of the API is never reached. */
typedef struct DIR DIR;

DIR *opendir(const char *name);
struct dirent *readdir(DIR *dirp);
int closedir(DIR *dirp);
void rewinddir(DIR *dirp);

#endif /* _DIRENT_H */
