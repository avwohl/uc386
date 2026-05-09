/* conio.h — console I/O for period code. Most period games do
 * keypress polling via kbhit() + getch(). dos_emu has no real
 * keyboard input; these return 0 / EOF.
 */
#ifndef _CONIO_H
#define _CONIO_H

int kbhit(void);
int getch(void);
int getche(void);
int putch(int c);
int cprintf(const char *fmt, ...);
int cputs(const char *s);

void clrscr(void);
void gotoxy(int x, int y);
void textcolor(int color);
void textbackground(int color);

#endif /* _CONIO_H */
