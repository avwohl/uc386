/* sys/wait.h — POSIX process-wait macros.
 *
 * uc386 has no fork/exec — these are stubs so programs that include
 * this header parse cleanly. Code that actually CALLS waitpid will
 * link against an unimplemented stub.
 */
#ifndef _SYS_WAIT_H
#define _SYS_WAIT_H

#include <sys/types.h>

#define WIFEXITED(status)    (((status) & 0xFF) == 0)
#define WEXITSTATUS(status)  (((status) >> 8) & 0xFF)
#define WIFSIGNALED(status)  (((status) & 0x7F) > 0 && ((status) & 0x7F) < 0x7F)
#define WTERMSIG(status)     ((status) & 0x7F)
#define WIFSTOPPED(status)   (((status) & 0xFF) == 0x7F)
#define WSTOPSIG(status)     WEXITSTATUS(status)

#define WNOHANG     1
#define WUNTRACED   2

pid_t wait(int *status);
pid_t waitpid(pid_t pid, int *status, int options);

#endif /* _SYS_WAIT_H */
