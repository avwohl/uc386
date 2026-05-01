/* graph.h — Watcom DOS graphics library. Period games use it to
 * set video modes, draw pixels, etc. dos_emu has no real video
 * subsystem; these are declarations only. The runtime stubs no-op.
 */
#ifndef _GRAPH_H
#define _GRAPH_H

/* Video modes (subset). */
#define _VRES16COLOR    0x12  /* 640x480x16 */
#define _MRES256COLOR   0x13  /* 320x200x256 — DOOM/ROTT default */
#define _DEFAULTMODE    0xFFFF
#define _TEXTC80        0x03

/* Coordinate types. */
struct xycoord { short xcoord, ycoord; };
struct rccoord { short row, col; };
struct videoconfig {
    short numxpixels, numypixels;
    short numtextcols, numtextrows;
    short numcolors;
    short bitsperpixel;
    short numvideopages;
    short mode, adapter, monitor, memory;
};

short _setvideomode(short mode);
short _getvideomode(void);
void _clearscreen(short area);
short _setpixel(short x, short y);
short _getpixel(short x, short y);
short _setcolor(short color);
short _getcolor(void);
struct xycoord _moveto(short x, short y);
short _lineto(short x, short y);
short _rectangle(short fill, short x1, short y1, short x2, short y2);

#endif /* _GRAPH_H */
