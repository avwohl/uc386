/* Minimal sbase util.c shim for uc386. Backs the declarations in
 * util.h. eprintf / weprintf simplified relative to upstream sbase
 * (no automatic strerror append on `:` suffix, no setlocale wrapping).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "util.h"

/* Programs that include arg.h-style ARGBEGIN expect a global argv0
 * pointer they can read. ARGBEGIN sets it from argv[0] before
 * processing flags. Default value here covers the rare case where
 * a program's main never enters ARGBEGIN. */
char *argv0 = (char *)"sbase";

void
weprintf(const char *fmt, ...)
{
	va_list ap;
	if (argv0) {
		fputs(argv0, stderr);
		fputs(": ", stderr);
	}
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
}

void
eprintf(const char *fmt, ...)
{
	va_list ap;
	if (argv0) {
		fputs(argv0, stderr);
		fputs(": ", stderr);
	}
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
	exit(1);
}

void *
ecalloc(size_t nmemb, size_t size)
{
	void *p = calloc(nmemb, size);
	if (!p) {
		eprintf("calloc:");
	}
	return p;
}

ssize_t
writeall(int fd, const void *buf, size_t n)
{
	const char *p = (const char *)buf;
	size_t left = n;
	while (left > 0) {
		ssize_t w = write(fd, p, left);
		if (w < 0) {
			return -1;
		}
		if (w == 0) {
			break;
		}
		p += w;
		left -= w;
	}
	return (ssize_t)(n - left);
}

/* concat(fd1, s1, fd2, s2): copy fd1's contents to fd2, with s1/s2
 * just diagnostic name strings. Returns 0 on success, -1 on read
 * error, -2 on write error (matches sbase behavior).
 */
int
concat(int fd1, const char *s1, int fd2, const char *s2)
{
	char buf[BUFSIZ];
	ssize_t n;
	while ((n = read(fd1, buf, sizeof(buf))) > 0) {
		if (writeall(fd2, buf, n) < 0) {
			weprintf("write %s:", s2);
			return -2;
		}
	}
	if (n < 0) {
		weprintf("read %s:", s1);
		return -1;
	}
	return 0;
}
