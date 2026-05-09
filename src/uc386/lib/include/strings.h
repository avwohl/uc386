/* strings.h — BSD-flavored string functions for uc386.
 * (Note: distinct from string.h. Some POSIX programs include both.)
 */
#ifndef _STRINGS_H
#define _STRINGS_H

#include <stddef.h>

int strcasecmp(const char *s1, const char *s2);
int strncasecmp(const char *s1, const char *s2, size_t n);

/* Legacy; modern code uses memset/memcpy. */
void bzero(void *s, size_t n);
void bcopy(const void *src, void *dst, size_t n);

#endif /* _STRINGS_H */
