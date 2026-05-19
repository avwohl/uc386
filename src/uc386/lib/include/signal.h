/* signal.h - Signal handling for uc80/CP-M */
#ifndef _SIGNAL_H
#define _SIGNAL_H

/* Signal handler function type */
typedef void (*sig_handler_t)(int);

/* C-standard integer type accessible as an atomic entity even with
 * asynchronous signals. DOS has no async delivery, but portable
 * code still declares flags as `volatile sig_atomic_t` (e.g. git's
 * progress.c `static volatile sig_atomic_t progress_update;`). */
typedef int sig_atomic_t;

/* Special signal handler values */
#define SIG_DFL ((sig_handler_t)0)  /* Default signal handling */
#define SIG_IGN ((sig_handler_t)1)  /* Ignore signal */
#define SIG_ERR ((sig_handler_t)-1) /* Error return */

/* Signal numbers */
#define SIGABRT 1  /* Abnormal termination */
#define SIGFPE  2  /* Floating-point exception */
#define SIGILL  3  /* Illegal instruction */
#define SIGINT  4  /* Interrupt */
#define SIGSEGV 5  /* Segmentation violation */
#define SIGTERM 6  /* Termination request */
/* POSIX signals referenced by portable code (e.g. git's
 * run-command.c). DOS has no real asynchronous signals — these
 * exist so such code compiles and links; delivery is a no-op. */
#define SIGHUP   7
#define SIGQUIT  8
#define SIGPIPE  9
#define SIGKILL 10
#define SIGCHLD 11
#define SIGCONT 12
#define SIGSTOP 13
#define SIGUSR1 14
#define SIGUSR2 15
#define SIGALRM 16   /* interval-timer expiry (no real delivery on DOS) */

/* Number of signals. _NSIG is the glibc-internal spelling; NSIG is
 * the BSD/POSIX-ish name portable code uses to size signal tables
 * (e.g. git's run-command.c: `for (sig = 1; sig < NSIG; sig++)`).
 * One past the highest signal number defined above. */
#define _NSIG 17
#define NSIG  _NSIG

/* POSIX signal-set type + sigprocmask `how` values. DOS has no
 * signal masks; sigset_t is an opaque word and the operations are
 * benign no-ops (single-threaded emulator, no async delivery, so
 * "blocking" is vacuously satisfied). */
typedef unsigned long sigset_t;
#define SIG_BLOCK   0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

int sigemptyset(sigset_t *set);
int sigfillset(sigset_t *set);
int sigaddset(sigset_t *set, int signo);
int sigdelset(sigset_t *set, int signo);
int sigismember(const sigset_t *set, int signo);
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);

/* POSIX sigaction(). DOS delivers no asynchronous signals, so this
 * just records/returns the disposition (benign no-op delivery) —
 * portable code like git's progress.c installs a SIGALRM handler
 * via sigaction() for the progress spinner. SA_* flag values are
 * arbitrary (never acted on); only the type/symbols need to exist
 * so such code compiles and links. */
#define SA_NOCLDSTOP 0x0001
#define SA_NOCLDWAIT 0x0002
#define SA_SIGINFO   0x0004
#define SA_RESTART   0x0010
#define SA_NODEFER   0x0020
#define SA_RESETHAND 0x0040
#define SA_ONSTACK   0x0080

struct sigaction {
    sig_handler_t sa_handler;
    sigset_t      sa_mask;
    int           sa_flags;
    void        (*sa_sigaction)(int, void *, void *);
};

int sigaction(int signum, const struct sigaction *act,
              struct sigaction *oldact);

/* Set signal handler - returns previous handler or SIG_ERR on error */
sig_handler_t signal(int sig, sig_handler_t handler);

/* Raise a signal - returns 0 on success, non-zero on error */
int raise(int sig);

#endif /* _SIGNAL_H */
