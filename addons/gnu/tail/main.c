/* `tail -n N`: print the last N lines of stdin (default 10).
 *
 * Reads stdin into a circular buffer of line pointers. Allocates
 * enough memory for N lines; for very large N or pathologically long
 * lines the libc heap (1 MB by default) is the limit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 1024

int main(int argc, char **argv) {
    int n = 10;
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (argv[i][0] == '-' && argv[i][1] >= '0' && argv[i][1] <= '9') {
            n = atoi(argv[i] + 1);
        }
    }
    if (n <= 0) {
        return 0;
    }
    /* Ring buffer of n line slots. */
    char **buf = (char **)calloc((size_t)n, sizeof(char *));
    if (!buf) {
        return 1;
    }
    char line[MAX_LINE];
    int head = 0;
    int count = 0;
    while (fgets(line, MAX_LINE, stdin)) {
        size_t len = strlen(line);
        if (buf[head]) {
            free(buf[head]);
        }
        buf[head] = (char *)malloc(len + 1);
        if (!buf[head]) {
            return 1;
        }
        memcpy(buf[head], line, len + 1);
        head = (head + 1) % n;
        if (count < n) {
            count++;
        }
    }
    int start = (count < n) ? 0 : head;
    for (i = 0; i < count; i++) {
        int idx = (start + i) % n;
        fputs(buf[idx], stdout);
    }
    return 0;
}
