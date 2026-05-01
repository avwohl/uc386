/* `dirname PATH`: print everything up to (but not including) the
 * last '/' in PATH. POSIX rules:
 *   dirname /usr/bin/foo  → /usr/bin
 *   dirname /usr/         → /
 *   dirname foo           → .
 *   dirname /             → /
 */
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fputs("dirname: missing operand\n", stderr);
        return 1;
    }
    const char *path = argv[1];
    size_t end = strlen(path);
    /* Strip trailing slashes (but keep one if path is all slashes). */
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
    if (!has_slash) {
        fputs(".\n", stdout);
        return 0;
    }
    if (last_slash == 0) {
        fputs("/\n", stdout);
        return 0;
    }
    /* Strip trailing slashes again on the directory part. */
    while (last_slash > 1 && path[last_slash - 1] == '/') {
        last_slash--;
    }
    fwrite(path, 1, last_slash, stdout);
    putchar('\n');
    return 0;
}
