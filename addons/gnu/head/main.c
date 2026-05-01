/* `head -n N`: print the first N lines of stdin (default 10).
 *
 * Supported:
 *   head            → first 10 lines of stdin
 *   head -n 5       → first 5 lines of stdin
 *   head -5         → first 5 lines of stdin (BSD-style)
 *
 * File arguments come once tasks 4-5 land — for now, stdin only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int n = 10;
    int i;
    int lines = 0;
    int c;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (argv[i][0] == '-' && argv[i][1] >= '0' && argv[i][1] <= '9') {
            n = atoi(argv[i] + 1);
        }
    }
    while (lines < n && (c = getchar()) != EOF) {
        putchar(c);
        if (c == '\n') {
            lines++;
        }
    }
    return 0;
}
