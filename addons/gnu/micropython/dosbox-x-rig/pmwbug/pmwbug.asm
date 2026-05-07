; ============================================================
; Minimal DOSBox-X PMODE/W INT 21h AH=0x40 caller-EIP bug repro
; ============================================================
;
; Standalone NASM 32-bit source — no external dependencies. Two
; `INT 21h AH=0x40 BX=1` (DOS write to stdout) call sites differ
; ONLY in caller's CS:EIP. Both invoke the same `_writer` helper
; with identical fd / buffer / count. Under DOSBox-X the high-EIP
; call writes NUL bytes to stdout; under classic DOSBox 0.74-3,
; QEMU+FreeDOS, or real DOS hardware, both produce the expected
; ASCII text.
;
; Build (Linux, with Open Watcom V2 + NASM installed):
;
;   nasm -f obj -o pmwbug.obj pmwbug.asm
;   wlink system pmodew name PMWBUG.EXE \
;       file pmwbug.obj \
;       option stack=64k option start=_start \
;       option stub=$WATCOM/binw/pmodew.exe
;
; Run (capture stdout to a file via DOS redirection so we can see
; the actual byte values, not what the terminal happens to render):
;
;   PMWBUG.EXE > out.txt
;
; Expected (classic DOSBox / QEMU+FreeDOS / real hardware):
;
;   [low CS:EIP — should print under any DOS]
;   [high CS:EIP — fails under DOSBox-X only]
;
; Actual under DOSBox-X (e.g. 2026.05.02 SDL2):
;
;   [low CS:EIP — should print under any DOS]
;   <61 NUL bytes here, no newline either>
; ============================================================

        bits    32

        section _TEXT use32 class=CODE

        global  _start

; ------------------------------------------------------------
; _start: entry point set via wlink `option start=_start`. PMODE/W
; enters in flat 32-bit protected mode with SS:ESP set up.
; ------------------------------------------------------------

_start:
        ; First write: caller is _start at low CS:EIP. Same writer,
        ; same fd, same args as the second call below — only the
        ; CS:EIP at the call instruction differs.
        push    dword msg_low_len
        push    msg_low
        push    1                       ; fd = stdout
        call    _writer
        add     esp, 12

        ; Jump past padding to the high-CS:EIP call site.
        jmp     .high_call

        ; Padding to push the next call site past the
        ; observed-failing threshold (~256 KB). 768 KB is well
        ; clear of any 256/512 KB boundary uncertainty.
        times   768 * 1024 db 0x90      ; nop

.high_call:
        ; Second write: caller is _start at high CS:EIP. EVERY OTHER
        ; thing about this call is identical to the one above —
        ; same _writer, same fd, same buffer category, same count
        ; (different string content so we can tell which one
        ; produced output, but same length category, same DS).
        push    dword msg_high_len
        push    msg_high
        push    1
        call    _writer
        add     esp, 12

        ; Exit cleanly via INT 21h AH=0x4C (DOS terminate, rc=0).
        mov     ax, 0x4c00
        int     0x21

; ------------------------------------------------------------
; _writer: cdecl-style 3-arg INT 21h AH=0x40 wrapper. Lives AFTER
; the padding so its own EIP is the same on both invocations —
; ensuring any caller-EIP-dependent emulator bug pivots on the
; call site, not on _writer itself.
; ------------------------------------------------------------

        global  _writer
_writer:
        push    ebp
        mov     ebp, esp
        push    ebx
        mov     ebx, [ebp + 8]          ; fd
        mov     edx, [ebp + 12]         ; buffer (flat linear addr)
        mov     ecx, [ebp + 16]         ; count
        mov     ah, 0x40                ; DOS: write to handle
        int     0x21
        pop     ebx
        leave
        ret

; ------------------------------------------------------------
; Test strings.
; ------------------------------------------------------------

        section _DATA use32 class=DATA

msg_low:
        db      "[low CS:EIP -- should print under any DOS]", 10
msg_low_len equ $ - msg_low

msg_high:
        db      "[high CS:EIP -- fails under DOSBox-X only]", 10
msg_high_len equ $ - msg_high
