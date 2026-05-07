; ============================================================
; Minimal MP.EXE-shaped DOSBox-X PMODE/W INT 21h bug repro:
; the user-code half of a two-obj LE binary built via the
; exe.py bridge.
;
; Mimics MP.EXE structurally:
;   - bridge.obj (provided by exe.py) at low VA — does its
;     diagnostic markers, then calls _main.
;   - user.obj (this file) at high VA — _main calls _write,
;     _write does INT 21h.
;
; Build:
;   python -m addons.harness.exe \
;       addons/gnu/micropython/dosbox-x-rig/pmwbug/pmwbug_user.asm \
;       -o addons/gnu/micropython/dosbox-x-rig/pmwbug/PMWBUG2.EXE
;
; The exe.py bridge auto-detects `_write:` and exports it as a
; global so the bridge can also reference it (via `extern _write`).
; That cross-obj reference is the same one MP.EXE has.
; ============================================================

        bits    32

        section _TEXT use32 class=CODE

        global  _main
        global  _write

; Padding to push _main + _write to high VA inside user.obj.
; The bridge stub is ~2 KB; with 2 MB of padding here, _main
; lands at ~2 MB into the code segment — same neighborhood as
; MP.EXE's _main at runtime VA ~0x1c3000.
        times   2 * 1024 * 1024 db 0x90 ; nop

; ------------------------------------------------------------
; _main: called by the bridge's _pmodew_start. The bridge does
; its diagnostic markers first; once we see [bridge-pre-call-main]
; the next event is _main running.
; ------------------------------------------------------------

_main:
        push    ebp
        mov     ebp, esp

        ; Call _write — same as MP.EXE's main does for its
        ; "[mp-main-entered]" marker. This is the high-CS:EIP
        ; depth-3 call that writes NUL bytes under DOSBox-X.
        push    dword msg_high_len
        push    msg_high
        push    1                       ; fd = stdout
        call    _write
        add     esp, 12

        xor     eax, eax
        leave
        ret

; ------------------------------------------------------------
; _write(fd, buf, count): cdecl. The literal MP.EXE _write body
; (per the disassembly cited in README.md): push ebp / mov ebp,
; esp / move args into BX/EDX/ECX / mov AH,0x40 / int 0x21 / ...
; ------------------------------------------------------------

_write:
        push    ebp
        mov     ebp, esp
        mov     ebx, [ebp + 8]
        mov     edx, [ebp + 12]
        mov     ecx, [ebp + 16]
        mov     ah, 0x40
        int     0x21
        leave
        ret

; ------------------------------------------------------------
; BSS to match MP.EXE's substantial uninitialized data layout.
; ------------------------------------------------------------

        section _BSS use32 class=BSS
        global  _bss_zero_start
        global  _bss_zero_end
_bss_zero_start:
_bss_filler:
        resb    1 * 1024 * 1024
_bss_zero_end:

; ------------------------------------------------------------
; Marker string. The MP.EXE marker that fails is the literal
; "[mp-main-entered]\n" — keeping the same length and category
; here so any size-keyed dispatch in DOSBox-X behaves identically.
; ------------------------------------------------------------

        section _DATA use32 class=DATA

msg_high:
        db      "[mp-main-entered]", 10
msg_high_len equ $ - msg_high
