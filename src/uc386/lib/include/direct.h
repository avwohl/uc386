/* direct.h — Microsoft-style directory I/O. Forwards to the
 * POSIX equivalents in <unistd.h> + <sys/stat.h>.
 */
#ifndef _DIRECT_H
#define _DIRECT_H

#include <unistd.h>
#include <sys/stat.h>

/* Microsoft underscored aliases. */
int  _chdir(const char *path);
int  _mkdir(const char *path);
int  _rmdir(const char *path);
char *_getcwd(char *buf, int maxlen);

#endif /* _DIRECT_H */
