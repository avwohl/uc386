/* doom_stubs.c — uc386-side replacements for the platform-specific
 * symbols that linuxdoom-1.10's i_video.c / i_sound.c / i_system.c /
 * i_net.c would normally provide. Those upstream files would pull in
 * BSD sockets, X11, Linux DSP — none of which exist under dos_emu —
 * so we exclude them at build time and provide these no-op
 * replacements instead.
 *
 * Behaviour: just enough to let DOOM start, allocate its zone, and
 * stop without segfaulting. No graphics, no audio, no input. The goal
 * here is "the .bin assembles and the entry point is reachable" —
 * actually rendering Doom needs a real video stub that talks to a
 * frame buffer + a uc386-side input pump.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>

/* Forward decls for the doom types these stubs reference. We don't
 * include doomdef.h because we want this file build-able even if the
 * upstream headers aren't on -I (e.g. compiling stubs alone for
 * testing). Sizes match the doom/linux defs. */
typedef unsigned char byte;
typedef int fixed_t;

typedef struct {
    char    forwardmove;
    char    sidemove;
    short   angleturn;
    short   consistancy;
    unsigned char chatchar;
    unsigned char buttons;
} ticcmd_t;

/* ---- I_system.c replacements ---- */

void I_Init(void) { /* no-op */ }

/* doom asks for a chunk of memory it owns end-to-end. Match the
 * linux build's behaviour: 6 MB allocation, default heap. */
byte *I_ZoneBase(int *size) {
    *size = 6 * 1024 * 1024;
    return (byte *) malloc(*size);
}

/* doom polls I_GetTime in its tic loop. Without a real timer we'd
 * spin forever; bump a counter so the game can make progress. */
static int _i_get_time_counter = 0;
int I_GetTime(void) {
    return ++_i_get_time_counter;
}

ticcmd_t *I_BaseTiccmd(void) {
    static ticcmd_t emptycmd;
    return &emptycmd;
}

byte *I_AllocLow(int length) {
    byte *p = (byte *) malloc(length);
    if (p) memset(p, 0, length);
    return p;
}

void I_Tactile(int on, int off, int total) { (void)on; (void)off; (void)total; }

void I_Quit(void) { exit(0); }

void I_Error(char *error, ...) {
    va_list ap;
    va_start(ap, error);
    fprintf(stderr, "I_Error: ");
    vfprintf(stderr, error, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

/* ---- I_video.c replacements ---- */

void I_InitGraphics(void)        { /* no-op */ }
void I_ShutdownGraphics(void)    { /* no-op */ }
void I_StartFrame(void)          { /* no-op */ }
void I_StartTic(void)            { /* no-op */ }
void I_SetPalette(byte *pal)     { (void)pal; }
void I_UpdateNoBlit(void)        { /* no-op */ }
void I_FinishUpdate(void)        { /* no-op */ }
void I_WaitVBL(int count)        { (void)count; }
void I_ReadScreen(byte *scr)     { (void)scr; }

/* ---- I_sound.c replacements ---- */

void I_InitSound(void)               { /* no-op */ }
void I_UpdateSound(void)             { /* no-op */ }
void I_SubmitSound(void)             { /* no-op */ }
void I_ShutdownSound(void)           { /* no-op */ }
void I_SetChannels(void)             { /* no-op */ }

/* sfxinfo_t is opaque to us; the parameter is passed by pointer. */
int I_GetSfxLumpNum(void *sfxinfo)   { (void)sfxinfo; return -1; }

int I_StartSound(int id, int vol, int sep, int pitch, int priority) {
    (void)id; (void)vol; (void)sep; (void)pitch; (void)priority;
    return -1;
}
void I_StopSound(int handle)         { (void)handle; }
int  I_SoundIsPlaying(int handle)    { (void)handle; return 0; }
void I_UpdateSoundParams(int h, int v, int s, int p) {
    (void)h; (void)v; (void)s; (void)p;
}

void I_InitMusic(void)               { /* no-op */ }
void I_ShutdownMusic(void)           { /* no-op */ }
void I_SetMusicVolume(int volume)    { (void)volume; }
void I_PauseSong(int handle)         { (void)handle; }
void I_ResumeSong(int handle)        { (void)handle; }
int  I_RegisterSong(void *data)      { (void)data; return 0; }
void I_PlaySong(int handle, int looping) { (void)handle; (void)looping; }
void I_StopSong(int handle)          { (void)handle; }
void I_UnRegisterSong(int handle)    { (void)handle; }

/* ---- I_net.c replacements ---- */

void I_InitNetwork(void) { /* single-player only — no setup needed */ }
void I_NetCmd(void)      { /* no-op */ }

/* ---- m_misc.c referents that the SNDSERV branch wants ---- */

char *sndserver_filename = "sndserver";
int   mb_used = 6;

/* ---- libc gaps ---- */

/* fstat: doom uses this purely to learn a file's size. lseek the fd
 * to SEEK_END, capture, restore. Sets st_size only; the rest stays
 * zero (callers don't peek). */
int fstat(int fd, struct stat *buf) {
    long cur = lseek(fd, 0, SEEK_CUR);
    if (cur < 0) return -1;
    long end = lseek(fd, 0, SEEK_END);
    lseek(fd, cur, SEEK_SET);
    if (end < 0) return -1;
    memset(buf, 0, sizeof(*buf));
    buf->st_size = end;
    return 0;
}

/* mkdir: doom's only call is `mkdir("c:\\doomdata", 0)`. dos_emu
 * doesn't let us write outside the host's working dir anyway —
 * pretend success; if a subsequent open() fails, that'll surface
 * the real problem. */
int mkdir(const char *path, mode_t mode) {
    (void)path; (void)mode;
    return 0;
}

/* sscanf: minimal implementation covering the patterns doom uses —
 * %d, %i, %x, %c, with optional field width. Not POSIX-complete; do
 * not ship as the real libc sscanf. */
static const char *_skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
    return s;
}

int sscanf(const char *str, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int matched = 0;
    while (*fmt) {
        if (*fmt == ' ') {
            str = _skip_ws(str);
            fmt++;
            continue;
        }
        if (*fmt != '%') {
            if (*str != *fmt) break;
            str++; fmt++;
            continue;
        }
        fmt++; /* past '%' */
        if (*fmt == '*') { /* skip-and-discard not in doom; bail */
            break;
        }
        char conv = *fmt++;
        if (conv == 'd' || conv == 'i' || conv == 'x') {
            int base = (conv == 'x') ? 16 : (conv == 'i' ? 0 : 10);
            int sign = 1;
            str = _skip_ws(str);
            if (*str == '-') { sign = -1; str++; }
            else if (*str == '+') { str++; }
            if (base == 0) {
                if (str[0] == '0' && (str[1] == 'x' || str[1] == 'X')) {
                    base = 16; str += 2;
                } else if (str[0] == '0') {
                    base = 8; str++;
                } else {
                    base = 10;
                }
            } else if (conv == 'x' && str[0] == '0' &&
                       (str[1] == 'x' || str[1] == 'X')) {
                str += 2;
            }
            int val = 0, any = 0;
            for (;;) {
                int d;
                if (*str >= '0' && *str <= '9') d = *str - '0';
                else if (*str >= 'a' && *str <= 'f') d = *str - 'a' + 10;
                else if (*str >= 'A' && *str <= 'F') d = *str - 'A' + 10;
                else break;
                if (d >= base) break;
                val = val * base + d;
                str++; any = 1;
            }
            if (!any) break;
            int *out = va_arg(ap, int *);
            *out = sign * val;
            matched++;
        } else if (conv == 'c') {
            char *out = va_arg(ap, char *);
            if (!*str) break;
            *out = *str++;
            matched++;
        } else {
            break; /* unsupported conversion */
        }
    }
    va_end(ap);
    return matched;
}
