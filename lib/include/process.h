/* process.h — DOS process/spawn API. Under dos_emu there's no
 * subprocess model; the spawn / exec families are stubs that
 * return -1.
 */
#ifndef _PROCESS_H
#define _PROCESS_H

#include <stddef.h>

#define P_WAIT      0
#define P_NOWAIT    1
#define P_OVERLAY   2
#define P_NOWAITO   3
#define P_DETACH    4

int system(const char *command);

int execl(const char *path, const char *arg0, ...);
int execlp(const char *path, const char *arg0, ...);
int execv(const char *path, char *const argv[]);
int execvp(const char *path, char *const argv[]);

int spawnl(int mode, const char *path, const char *arg0, ...);
int spawnv(int mode, const char *path, char *const argv[]);
int spawnvp(int mode, const char *path, char *const argv[]);

void exit(int status);
void abort(void);

#endif /* _PROCESS_H */
