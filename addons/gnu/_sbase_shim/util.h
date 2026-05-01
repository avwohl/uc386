/* Minimal sbase util.h shim for uc386 — provides what sbase tools
 * actually use across the ports here (ARGBEGIN/ARGEND macros, argv0,
 * eprintf, weprintf, ecalloc, writeall, concat). NOT a complete sbase
 * compatibility layer — only the subset needed.
 *
 * Each addon's manifest references this via:
 *   sources = ["../_sbase_shim/util.c", "<tool>.c"]
 *   extra_cflags = ["-I", "addons/gnu/_sbase_shim"]
 */
#ifndef SBASE_UTIL_H
#define SBASE_UTIL_H

#include <stddef.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>   /* ssize_t */

extern char *argv0;

/* arg.h ARGBEGIN/ARGEND family — verbatim from sbase's arg.h, MIT.
 * Implements suckless's "arg.h" idiom for parsing single-letter
 * options without using getopt(). The user's switch (case 'a':, …)
 * lands inside the inner for-loop here, after `argc_` is set to the
 * current option letter.
 */
#define ARGBEGIN	for (argv0 = *argv, argv++, argc--;\
				argv[0] && argv[0][0] == '-'\
				&& argv[0][1];\
				argc--, argv++) {\
			char argc_;\
			char **argv_;\
			int brk_;\
			if (argv[0][1] == '-' && argv[0][2] == '\0') {\
				argv++;\
				argc--;\
				break;\
			}\
			for (brk_ = 0, argv[0]++, argv_ = argv;\
					argv[0][0] && !brk_;\
					argv[0]++) {\
				if (argv_ != argv)\
					break;\
				argc_ = argv[0][0];\
				switch (argc_)

#define ARGEND			}\
		}

#define ARGC()		argc_

/* EARGF — extract a "required" argument for a single-letter option:
 * either inline (`-nN`) or as the next argv element (`-n N`). The `x`
 * is a fallback expression invoked when neither form is present
 * (typically `usage()` which calls `eprintf` and exits).
 */
#define EARGF(x)	((argv[0][1] == '\0' && argv[1] == NULL)?\
			((x), abort(), (char *)0) :\
			(brk_ = 1, (argv[0][1] != '\0')?\
				(&argv[0][1]) :\
				(argc--, argv++, argv[0])))

/* Declarations for libc functions sbase ports often need beyond what
 * stdio.h gives us. */
ssize_t getline(char **lineptr, size_t *n, FILE *stream);
int ferror(FILE *fp);
int fclose(FILE *fp);

/* I/O syscalls used by sbase tools. ssize_t comes from <sys/types.h>
 * via stdio.h. Declarations parallel the libc symbols (lib/i386_dos_libc.asm).
 */
ssize_t read(int fd, void *buf, size_t n);
ssize_t write(int fd, const void *buf, size_t n);
int close(int fd);
int open(const char *path, int flags, ...);

void eprintf(const char *fmt, ...);
void weprintf(const char *fmt, ...);
void *ecalloc(size_t nmemb, size_t size);
ssize_t writeall(int fd, const void *buf, size_t n);
int concat(int fd1, const char *s1, int fd2, const char *s2);
int fshut(FILE *fp, const char *fname);

#endif
