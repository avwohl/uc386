/* Deliberately self-contained: declaring printf inline instead of
   including <stdio.h> keeps this compilable with a bare
   `uc386 examples/hello.c -o hello.asm`, with no -I flag and no
   dependency on where the bundled headers landed. */
int printf(const char *fmt, ...);

int main(void) {
    printf("Hello, DOS!\n");
    return 0;
}
