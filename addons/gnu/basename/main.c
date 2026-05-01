/* `basename PATH [SUFFIX]`: print everything after the last `/` in
 * PATH; if SUFFIX is given and PATH ends with it, strip it too.
 *
 * Pure string manipulation — useful for shell-script cousins built
 * on top of uc386 binaries.
 */
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fputs("basename: missing operand\n", stderr);
        return 1;
    }
    const char *path = argv[1];
    /* Strip trailing slashes (POSIX), but only if the result is non-empty. */
    size_t end = strlen(path);
    while (end > 1 && path[end - 1] == '/') {
        end--;
    }
    /* Find the last '/' before `end`. */
    size_t last_slash = 0;
    int has_slash = 0;
    for (size_t i = 0; i < end; i++) {
        if (path[i] == '/') {
            last_slash = i;
            has_slash = 1;
        }
    }
    size_t start = has_slash ? (last_slash + 1) : 0;
    size_t len = end - start;
    /* Optional SUFFIX strip. */
    if (argc >= 3) {
        const char *suf = argv[2];
        size_t slen = strlen(suf);
        if (slen <= len && memcmp(path + end - slen, suf, slen) == 0
            && len > slen) {
            len -= slen;
        }
    }
    fwrite(path + start, 1, len, stdout);
    putchar('\n');
    return 0;
}
