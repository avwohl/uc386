/* Smoke test for strtol (decimal-only minimal version) / strdup /
 * getenv / strerror / fflush.
 *
 * Today's _strtol is a thin wrapper around _atoi (sign + decimal
 * digits, decimal-base only, best-effort endptr). Hex / octal / base
 * != 10 require a richer implementation that doesn't yet exist —
 * the libc.asm-side full strtol triggered a NASM phase-error flap
 * when bundled into BWK awk. Restoring it is a future slice (likely
 * by writing strtol in C and compiling through uc386 to produce
 * stable asm).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char *end;
    long a, b, f;
    char *p;
    char *dup;
    int rc;

    /* Basic decimal */
    a = strtol("42", &end, 10);
    printf("dec=%ld end=%c\n", a, *end ? *end : '0');

    /* Negative with leading whitespace */
    b = strtol("  -1234abc", &end, 10);
    printf("neg=%ld rem=%s\n", b, end);

    /* Plain decimal */
    f = strtol("99", NULL, 10);
    printf("dec99=%ld\n", f);

    /* getenv: empty environment under dos_emu */
    p = getenv("PATH");
    printf("env=%s\n", p ? p : "(null)");

    /* strdup */
    dup = strdup("hello");
    printf("dup=%s\n", dup);
    free(dup);

    /* strerror — maps errno to a real message (2 == ENOENT) */
    printf("err=%s\n", strerror(2));

    /* fflush — should return 0 */
    rc = fflush(stdout);
    printf("fflush=%d\n", rc);

    return 0;
}
