/* `cat`: copy named files (or stdin if none given) to stdout.
 *
 * Behavior:
 *   cat            → copy stdin to stdout
 *   cat F1 F2 ...  → copy each file in turn
 *   cat - F        → '-' means stdin (POSIX); included files are
 *                    read in order as given.
 *
 * Exits 0 if all files were read; exits 1 if any fopen failed (a
 * diagnostic line is written to stderr first).
 */
#include <stdio.h>
#include <string.h>

static int copy_stream(FILE *in) {
    int c;
    while ((c = fgetc(in)) != EOF) {
        if (putchar(c) == EOF) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    int i;
    int rc = 0;
    if (argc <= 1) {
        return copy_stream(stdin);
    }
    for (i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == '\0') {
            if (copy_stream(stdin)) {
                rc = 1;
            }
            continue;
        }
        FILE *fp = fopen(argv[i], "r");
        if (!fp) {
            fputs("cat: cannot open ", stderr);
            fputs(argv[i], stderr);
            fputc('\n', stderr);
            rc = 1;
            continue;
        }
        if (copy_stream(fp)) {
            rc = 1;
        }
        fclose(fp);
    }
    return rc;
}
