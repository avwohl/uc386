#!/usr/bin/env python3
"""Pipeline: uc386 .asm → MZ+LE .exe that runs on FreeDOS / DOSBox / dosiz.

Today this orchestrates external tools rather than emitting the LE
format directly:

    1. NASM (`-f obj`) turns uc386's NASM-syntax .asm into a 32-bit
       OMF (Object Module Format) .obj file. NASM's OMF backend
       produces Watcom-compatible objects with USE32 segments.

    2. Open Watcom's `wlink` consumes the .obj and produces an MZ+LE
       executable. The `system causeway` directive bundles the
       CauseWay DOS extender (~10 KB free stub) into the .exe so the
       result runs unmodified on FreeDOS / DOSBox / dosiz / real DOS,
       no separate `dos4gw.exe` redistribution required.

The pipeline isn't free of caveats — uc386's libc was written
assuming flat-bin layout under dos_emu (INT 21h calls reach our
Python harness directly). Under DOS/4GW or CauseWay those same
INT 21h calls get reflected back to real-mode DOS by the extender,
which means the *extender* loads our binary — so its protected-mode
stack, segment selectors, and PSP are owned by the extender.

Watcom availability: Linux + Windows have native builds. macOS does
not (per the comment in `compare.py`). On macOS the function returns
None and the harness must skip — `compare.py` does this for the same
reason.

Usage:
    python -m addons.harness.exe addons/gnu/echo/main.c -o echo.exe

After build, the .exe runs under DOSBox:
    dosbox echo.exe

Or under dosiz (`../dosiz/dosiz echo.exe` once the LE-loader is
wired up — see `docs/dosiz-integration.md`).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LIB_INCLUDE = REPO_ROOT / "src" / "uc386" / "lib" / "include"


# The PMODE/W bridge stub. Linked into every .exe build; provides:
#   - Real DOS handles (0/1/2) for stdin/stdout/stderr (libc's
#     0xF0/F1/F2 dos_emu sentinels are stripped by the asm rewriter).
#     Without this, fputs/fwrite/printf via INT 21h AH=0x40 silently
#     drop output (BX=0xF1 is an invalid DOS handle).
#   - argv setup: argc=1 + argv[0]="program" placeholder. PMODE/W
#     doesn't pass argc/argv in any register and its protected-mode
#     PSP doesn't carry the cmdline tail at PSP+0x80. Reading the
#     real cmdline requires Watcom CRT internals — see Phase 7
#     section of docs/path-a-mz-le.md for the full failure log.
#
# Programs that don't read argv work cleanly. Programs that do see
# argc=1 + a placeholder argv[0].
BRIDGE_ASM = """
        section _DATA use32 class=DATA
        global _stdin
        global _stdout
        global _stderr
_stdin:  dd 0
_stdout: dd 1
_stderr: dd 2

        global _pmodew_argc
        global _pmodew_argv
_pmodew_argc:         dd 0
_pmodew_argv:         dd _pmodew_argv_array

        ; Up to 32 args + NULL terminator.
_pmodew_argv_array: times 33 dd 0
        ; Buffer for the parsed/null-terminated cmdline (PSP tail
        ; max is 127 bytes; round to 128 for alignment).
_pmodew_argv_buffer: times 128 db 0
        ; Buffer for argv[0] (the program path, found by walking the
        ; DOS environment block). Falls back to a placeholder if
        ; the env-walk fails (DPMI selector alloc fails, etc.).
_pmodew_argv0_buffer: times 128 db 0
_pmodew_argv0_placeholder: db "program", 0

        ; Diagnostic markers — raw INT 21h AH=0x40 BX=1 (write to
        ; fd 1) so they respect DOS file-handle redirection in the
        ; rig's `MP.EXE > MP_OUT.TXT`. printf() can't be used here
        ; because libc's _printf goes through INT 21h AH=02h
        ; (console output), which under `dosbox-x -silent` lands on
        ; the suppressed console rather than in MP_OUT.TXT.
_bridge_marker_entered: db "[bridge-entered]", 10
_bridge_marker_argv:    db "[bridge-argv-done]", 10
_bridge_marker_jump:    db "[bridge-pre-jump]", 10
_bridge_marker_postfpu: db "[bridge-post-fpu]", 10
_bridge_marker_prebss:  db "[bridge-pre-bss-zero]", 10
_bridge_marker_postbss: db "[bridge-post-bss-zero]", 10
_bridge_marker_premain: db "[bridge-pre-call-main]", 10
_bridge_marker_postmain: db "[bridge-post-main]", 10
_bridge_marker_diag:    db "[bridge-diag-stub]", 10
_bridge_marker_postdiag: db "[bridge-post-diag]", 10
_bridge_marker_dump_main7: db "[main+7]=", 0
_bridge_marker_dump_str:   db "[str-bytes]=", 0
_bridge_marker_dump_mainaddr: db "[main_addr]=", 0
_bridge_marker_dump_main0:  db "[main+0]=", 0
_bridge_marker_dump_main4:  db "[main+4]=", 0
_bridge_marker_dump_main11: db "[main+11]=", 0
_bridge_marker_dump_main14: db "[main+14]=", 0
_bridge_marker_dump_writetest: db "[direct-write]=", 0
_bridge_marker_dump_writeaddr: db "[write_addr]=", 0
_bridge_marker_dump_write0: db "[write+0]=", 0
_bridge_marker_dump_write4: db "[write+4]=", 0
_bridge_marker_dump_stackwrite: db "[stack-write]=", 0
_bridge_marker_dump_esp: db "[esp]=", 0
_bridge_marker_dump_likemain: db "[mainlike]=", 0
_bridge_marker_dump_userwrite: db "[user-write]=", 0
_bridge_hex_buf:           times 12 db 0    ; 8 hex chars + LF + slack

        section _TEXT use32 class=CODE
        global _pmodew_start
        extern _main
        extern _bss_zero_start
        extern _bss_zero_end

; _bridge_emit(EDX=msg ptr, ECX=byte count): write to fd 1 + commit.
; Caller saves EDX/ECX if needed across the call. We save EAX/EBX
; internally. The commit-file (AH=0x68) forces DOS to flush its file
; buffer for handle 1 to disk — without it, DOSBox-X can buffer the
; bridge's marker writes in DOS's internal cache and they never make
; it to RIG.LOG before dosbox-x is killed by the rig's timeout.
_bridge_emit:
        push    eax
        push    ebx
        mov     ah, 0x40
        mov     bx, 1
        int     0x21
        mov     ah, 0x68
        mov     bx, 1
        int     0x21
        pop     ebx
        pop     eax
        ret

; _bridge_emit_hex32(EAX=value): write 8 lowercase hex chars + LF
; to fd 1. Trashes EAX, EBX, ECX, EDX.
_bridge_emit_hex32:
        push    edi
        mov     edi, _bridge_hex_buf
        mov     ecx, 8
.loop:
        rol     eax, 4
        mov     edx, eax
        and     edx, 0x0F
        cmp     dl, 9
        jbe     .digit
        add     dl, 'a' - 10 - '0'
.digit:
        add     dl, '0'
        mov     [edi], dl
        inc     edi
        loop    .loop
        mov     byte [edi], 10           ; LF
        ; Now write 9 bytes (8 hex + LF) via _bridge_emit
        pop     edi
        mov     edx, _bridge_hex_buf
        mov     ecx, 9
        call    _bridge_emit
        ret

; _bridge_write_stack(fd, buf, count) cdecl: literal copy of _write's
; body — push ebp / mov ebp, esp / read args from stack / INT 21h.
; Used to test whether _main's failure is intrinsic to stack-based
; arg passing OR specific to the codegen-emitted _write in the user
; .obj.
_bridge_write_stack:
        push    ebp
        mov     ebp, esp
        mov     ebx, [ebp + 8]
        mov     edx, [ebp + 12]
        mov     ecx, [ebp + 16]
        mov     ah, 0x40
        int     0x21
        movzx   eax, ax
        leave
        ret

; _bridge_emit_str0(EDX=ptr to NUL-terminated string): writes string
; (without NUL) to fd 1. Trashes ECX, AL.
_bridge_emit_str0:
        push    edi
        mov     edi, edx
        xor     ecx, ecx
.scan:
        cmp     byte [edi], 0
        je      .done
        inc     edi
        inc     ecx
        jmp     .scan
.done:
        pop     edi
        call    _bridge_emit
        ret

; _diag_main(): in-bridge stub that mimics what main() should do
; (write a marker, return). Called from the bridge BEFORE the real
; _main so we can verify the call-prolog-write-ret plumbing works
; at all from inside this binary's text segment. If
; [bridge-diag-stub] prints but [mp-main-entered] doesn't, the
; failure is specific to the codegen-emitted _main (its prolog,
; argv plumbing, or first call into libc — not the bridge → main
; plumbing itself).
_diag_main:
        enter   4, 0
        mov     edx, _bridge_marker_diag
        mov     ecx, 19
        call    _bridge_emit
        xor     eax, eax
        leave
        ret

; _diag_main_writelike(argc, argv) cdecl: mirrors _main EXACTLY —
; same enter prolog, same push 18 / push str_addr / push 1 / call
; pattern. The only difference vs the real _main is that this
; calls _bridge_write_stack (in bridge .obj) instead of _write
; (in user .obj). If this prints "[mp-main-entered]" but the real
; _main produces NULs, the bug is something specific to the user
; .obj's _write that doesn't manifest with bridge .obj's identical
; copy of the same code (very strange — possibly a wlink
; section-layout bug, or DOSBox-X-side cache issue).
_diag_main_writelike:
        enter   4, 0
        push    18
        push    dword [_main + 7]
        push    1
        call    _bridge_write_stack
        add     esp, 12
        xor     eax, eax
        leave
        ret

; _bridge_marker(label addr, byte count): syntactic sugar shorthand.
; NASM doesn't have proper "function" macros that can be reused
; per-site without name collisions, so each site does the explicit
; mov + call. Keeping the helper out-of-band lets us share the
; inner write+commit logic without per-site stack churn.

_pmodew_start:
        ; Marker 1: PMODE/W reached our stub at all. If this doesn't
        ; print, the LE binary isn't loading (extender bailed before
        ; transferring control). If it does, the bridge runs but a
        ; later step (argv tokenization, BSS init, main, write())
        ; is the one that aborts.
        mov     edx, _bridge_marker_entered
        mov     ecx, 17
        call    _bridge_emit

        ; OpenWatcom's cstrt386.asm (the standard 32-bit CRT for
        ; DOS/4GW + PMODE/W + generic DPMI hosts) uses `mov esi,es`
        ; to reach the PSP under generic-DOS — at entry, ES holds the
        ; PSP selector that the extender set up. The cmdline tail is
        ; at [es:0x80] (length byte) and [es:0x81..] (the tail).
        ;
        ; Earlier attempts overwrote ES via `mov es, 0x21` which
        ; broke PMODE/W. The fix: USE the ES PMODE/W gave us at
        ; entry, don't replace it.

        ; Step 1: copy cmdline length + tail via [es:offset].
        ; Buffer copy with es-override per byte.
        movzx   ecx, byte [es:0x80]   ; cmdline length
        cmp     ecx, 127
        jbe     .len_ok
        mov     ecx, 127
.len_ok:
        mov     edi, _pmodew_argv_buffer
        mov     edx, 0x81
        test    ecx, ecx
        jz      .copy_done
.copy_loop:
        mov     al, [es:edx]
        mov     [edi], al
        inc     edx
        inc     edi
        dec     ecx
        jnz     .copy_loop
.copy_done:
        mov     byte [edi], 0          ; NUL-terminate

        ; Step 2: tokenize. argv[0] = placeholder. Walk the buffer
        ; splitting on space/tab/CR into argv[1..].
        mov     edi, _pmodew_argv_array
        mov     dword [edi], _pmodew_argv0_placeholder
        add     edi, 4
        mov     ecx, 1                 ; argc starts at 1
        mov     esi, _pmodew_argv_buffer

.skip_ws:
        cmp     ecx, 32
        jge     .tokenize_done
        mov     al, [esi]
        test    al, al
        jz      .tokenize_done
        cmp     al, ' '
        je      .skip_one
        cmp     al, 9
        je      .skip_one
        cmp     al, 13
        je      .tokenize_done
        ; start of token: record pointer
        mov     [edi], esi
        add     edi, 4
        inc     ecx

.in_token:
        inc     esi
        mov     al, [esi]
        test    al, al
        jz      .tokenize_done
        cmp     al, ' '
        je      .end_token
        cmp     al, 9
        je      .end_token
        cmp     al, 13
        je      .end_token
        jmp     .in_token

.end_token:
        mov     byte [esi], 0          ; null-terminate token
        inc     esi
        jmp     .skip_ws

.skip_one:
        inc     esi
        jmp     .skip_ws

.tokenize_done:
        mov     dword [edi], 0         ; argv[argc] = NULL
        mov     [_pmodew_argc], ecx

        ; Step 3: try to find the real program path for argv[0] by
        ; walking the DOS environment block. PSP+0x2C holds the env
        ; segment as a word. After the env vars (each NUL-terminated,
        ; doubly NUL-terminated as a group), DOS appends a 16-bit
        ; count followed by the program path string. Allocate a
        ; fresh DPMI selector for the env segment and walk it.
        ;
        ; Best-effort: if any DPMI step fails, argv[0] stays as the
        ; "program" placeholder.
        movzx   eax, word [es:0x2C]    ; env segment
        test    eax, eax
        jz      .argv0_done            ; no env block
        push    eax
        push    ebx
        mov     bx, ax
        mov     ax, 0x0002             ; DPMI: Segment to Descriptor
        int     0x31
        jc      .argv0_alloc_failed
        movzx   eax, ax                ; env selector
        push    fs
        mov     fs, ax
        ; Skip env vars: each is NUL-terminated; group ends at double NUL.
        xor     edx, edx
.env_walk:
        cmp     edx, 0x7FFE            ; clamp env-walk distance
        jae     .env_walk_done
        movzx   eax, byte [fs:edx]
        inc     edx
        test    eax, eax
        jnz     .env_walk
        ; Saw NUL. Check if next byte is also NUL.
        movzx   eax, byte [fs:edx]
        test    eax, eax
        jnz     .env_walk              ; single NUL — keep walking
        ; Double NUL found. Skip past it and the count word.
        inc     edx                    ; past second NUL
        add     edx, 2                 ; past 16-bit count
        ; Now [fs:edx] is the start of the program path string.
        ; Bail if the first byte is NUL — PMODE/W's env block may
        ; not include the program path (empirically: the env-walk
        ; completes cleanly but the path string is empty). Keep
        ; the placeholder in that case.
        movzx   eax, byte [fs:edx]
        test    eax, eax
        jz      .env_walk_done
        mov     edi, _pmodew_argv0_buffer
        mov     ecx, 127               ; max bytes to copy
.path_copy:
        movzx   eax, byte [fs:edx]
        mov     [edi], al
        test    al, al
        jz      .path_done
        inc     edx
        inc     edi
        dec     ecx
        jnz     .path_copy
.path_done:
        mov     byte [edi], 0          ; ensure NUL-terminated
        ; Replace argv[0] placeholder with the real path.
        mov     dword [_pmodew_argv_array], _pmodew_argv0_buffer
.env_walk_done:
        pop     fs
.argv0_alloc_failed:
        pop     ebx
        pop     eax
.argv0_done:

        ; Marker 2: argv tokenization + env walk completed.
        mov     edx, _bridge_marker_argv
        mov     ecx, 19
        call    _bridge_emit

        ; Marker 3: about to begin the codegen-style startup
        ; (FPU init → BSS zero → call _main → exit). Each step
        ; below has a paired marker so we can pin the silent step.
        mov     edx, _bridge_marker_jump
        mov     ecx, 18
        call    _bridge_emit

        ; --- FPU init (mirrors codegen _start's fldcw) -----------
        sub     esp, 4
        mov     word [esp], 0x027F
        fldcw   [esp]
        add     esp, 4
        mov     edx, _bridge_marker_postfpu
        mov     ecx, 18
        call    _bridge_emit

        ; --- BSS zero -------------------------------------------
        ; PMODE/W's loader already zero-fills the BSS region at
        ; program load (per the LE/PMODE/W spec — `bss_size` in
        ; the LE header drives the allocate+zero). The redundant
        ; rep stosb in the codegen's _start was needed for dos_emu
        ; mode (where recursive _start() calls re-zero the BSS for
        ; "noinit" idiom tests), but in .exe mode it's just
        ; touching what should already be zero.
        ;
        ; Empirically (mp-rig run 25465153840): doing a 280 KB
        ; rep stosb on the multi-TU MicroPython binary's BSS range
        ; silently aborts the program — [bridge-pre-bss-zero]
        ; prints, [bridge-post-bss-zero] doesn't, MP.EXE returns
        ; to DOS without main() ever running. Skipping the redundant
        ; zero (loader already did it) lets execution continue.
        mov     edx, _bridge_marker_prebss
        mov     ecx, 22
        call    _bridge_emit

        ; (rep stosb intentionally omitted — see comment above)

        mov     edx, _bridge_marker_postbss
        mov     ecx, 23
        call    _bridge_emit

        ; --- Diagnostic stub: verify call mechanism BEFORE _main --
        ; If [bridge-diag-stub] prints but [mp-main-entered] doesn't,
        ; the bridge → C-style-function-with-enter-prolog-and-libc-
        ; INT-21h plumbing works. The failure is then specific to
        ; the codegen-emitted _main's body, not the call mechanism.
        call    _diag_main
        mov     edx, _bridge_marker_postdiag
        mov     ecx, 19
        call    _bridge_emit

        ; --- LE FIXUP / runtime addressing diagnostics ----------
        ;
        ; (Previously a ~150-line block of `mov eax, [_main + N];
        ; call _bridge_emit_hex32` and the like — debug code added
        ; while chasing wlink fixup behavior in mp-rig runs
        ; 25475877877 / 25476601480 / 25477796902. Those issues are
        ; long resolved.)
        ;
        ; Removed because under paged DPMI hosts (CWSDPMI) the
        ; dereferences of `[_main]`, `[_main + 7]`, etc. PF when
        ; the loader didn't map those exact pages — the diagnostic
        ; doesn't survive a host swap. Git history (commit f2faf76
        ; and surroundings) preserves the originals for reference.
        ;
        ; If you're chasing a fixup / call-target bug again, copy
        ; the relevant block back in temporarily — but gate it
        ; behind a build flag so a release-style build doesn't
        ; trip the next host's paging.

        ; Dump ESP so we know where main's stack will land.
        mov     edx, _bridge_marker_dump_esp
        call    _bridge_emit_str0
        mov     eax, esp
        call    _bridge_emit_hex32

        ; --- Mainlike test: REMOVED -----------------------------
        ; Previously a call to _diag_main_writelike — a stub that
        ; pushed `[_main + 7]` (the rel32 displacement of _main's
        ; first call) as an argument and invoked _bridge_write_stack
        ; with it. Useful in 2024 for distinguishing _main-side _write
        ; vs bridge-side _write bugs in a wlink/codegen issue that's
        ; long resolved.
        ;
        ; Removed because under paged DPMI hosts (CWSDPMI) the bytes
        ; at `_main + 7` no longer reliably form a valid pointer —
        ; the value gets passed to _bridge_write_stack's memcpy and
        ; PFs on unmapped pages. The diagnostic doesn't survive a
        ; host swap. Git history (commit f2faf76 and surroundings)
        ; preserves the original for reference.

        ; --- User-write test: REMOVED ---------------------------
        ; Run 25477796902 confirmed: calling user.obj's _write
        ; DIRECTLY from the bridge (via indirect call to address
        ; computed from _main's rel32) DOES produce
        ; "[mp-main-entered]" correctly. So user.obj's _write IS
        ; reachable and works when called from any caller. The
        ; bug is therefore specific to _main → _write — same _write,
        ; same args, different result based on the CALLER's runtime
        ; address.
        ;
        ; The test was incompatible with echo (its _main+14 doesn't
        ; contain a valid rel32 → call landed at garbage and faulted).
        ; Removed; the [user-write] datum is preserved in commit
        ; f2faf76's run logs.

        ; --- Call _main (mirrors codegen _start's call _main) ----
        mov     edx, _bridge_marker_premain
        mov     ecx, 23
        call    _bridge_emit

        ; CRITICAL: load ES from DS before user code runs. PMODE/W
        ; (and DOS/32A) hand control to the LE entry point with ES
        ; pointing at the PSP selector (256-byte limit). libc's
        ; memset/memcpy use `rep stosb`/`rep movsb` which both write
        ; through ES:EDI. With ES != DS, those writes either land in
        ; the wrong region or fault silently — the symptom is that
        ; memset(buf, X, N) leaves `buf` untouched but the program
        ; otherwise runs fine. Verified empirically: lwIP's
        ; pbuf_copy_partial → memcpy filled a static tx_buf with
        ; zeros (ES != DS write went elsewhere) and the NE2000
        ; driver shipped a 60-byte zero frame instead of an ARP
        ; request. Setting ES = DS here fixes every libc string/mem
        ; primitive at once.
        push    ds
        pop     es

        ; --- Install PM INT 0x80 divmod handler -----------------
        ; uc386 lowers 64-bit `/` and `%` into `int 0x80` with
        ; EDX:EAX = numerator, EBX:ECX = denominator,
        ; ESI low byte = op (0=udiv, 1=sdiv, 2=umod, 3=smod),
        ; result returned in EDX:EAX. dos_emu intercepts this in
        ; Python (uc386/src/uc386/dos_emu.py:683-721). On real DOS
        ; the IDT entry for 0x80 is a no-op IRETD, so every 64-bit
        ; / and % silently leaves EDX:EAX unchanged — the symptom
        ; is integer formatting that emits the same digit byte over
        ; and over (23 of them, the format-buffer size). Install a
        ; PM handler via DPMI fn 0x0205 (Set PM Interrupt Vector).
        mov     ax, 0x0205
        mov     bl, 0x80
        mov     cx, cs
        mov     edx, _bridge_int80_handler
        int     0x31

        ; cdecl: argc/argv on stack at [ebp+8]/[ebp+12] in main.
        ; Use an INDIRECT call (call eax with eax=_main) instead of
        ; `call _main` (rel32). Last run showed rel32 target lands
        ; at the right address with the right bytes, yet _main's
        ; behavior differs from a direct INT 21h with same args.
        ; If indirect-call produces "[mp-main-entered]" but direct
        ; doesn't, rel32 has some subtle issue. If both hang, the
        ; issue is downstream of the call (something in _main's
        ; runtime state that differs from the bridge's).
        push    dword [_pmodew_argv]
        push    dword [_pmodew_argc]
        mov     eax, _main
        call    eax
        add     esp, 8

        ; Preserve main's return code through the marker call —
        ; _bridge_emit only saves EAX/EBX internally so EAX is
        ; safe across it, but the AH=4Ch exit needs AL=return.
        push    eax
        mov     edx, _bridge_marker_postmain
        mov     ecx, 19
        call    _bridge_emit
        pop     eax

        ; Exit DOS via INT 21h AH=4Ch with AL = main's return code.
        mov     ah, 0x4C
        int     0x21

; -----------------------------------------------------------------
; PM INT 0x80 handler — 64-bit divmod intrinsic.
; Contract (matches dos_emu.py:683-721 byte-for-byte):
;   IN:  EDX:EAX = numerator   (high:low)
;        EBX:ECX = denominator (high:low)
;        ESI low = op (0=udiv, 1=sdiv, 2=umod, 3=smod)
;   OUT: EDX:EAX = result (quotient for div, remainder for mod)
; All other GP regs preserved across the interrupt.
;
; Algorithm: binary long division, 64 iterations. Signed ops
; normalize to unsigned (abs both operands), divide, then apply
; the correct sign back. Mod sign follows numerator (C99 truncated
; division). Divide-by-zero leaves EDX:EAX at original numerator
; — matches "no handler installed" behavior; dos_emu errors in
; this case, but production code shouldn't be dividing by zero.
; -----------------------------------------------------------------
_bridge_int80_handler:
        push    ebp
        push    edi
        push    esi
        push    ecx
        push    ebx
        sub     esp, 16
        ; Local scratch layout:
        ;   [esp+0]  = op (low byte of ESI on entry)
        ;   [esp+4]  = sign_flags: bit0 = num_neg, bit1 = den_neg
        ;   [esp+8]  = quot_low  (built MSB-first via shift+or)
        ;   [esp+12] = quot_high
        mov     dword [esp + 8], 0
        mov     dword [esp + 12], 0
        mov     ebp, esi
        and     ebp, 0xFF
        mov     [esp + 0], ebp
        mov     dword [esp + 4], 0

        ; Divide-by-zero guard: if EBX:ECX == 0, bail (EDX:EAX
        ; still hold the original numerator).
        mov     edi, ebx
        or      edi, ecx
        jz      .b80_restore

        ; If signed op, normalize numerator + denominator to abs.
        test    ebp, 1
        jz      .b80_unsigned
        test    edx, edx
        jns     .b80_num_pos
        not     edx
        neg     eax
        sbb     edx, -1
        or      dword [esp + 4], 1
.b80_num_pos:
        test    ebx, ebx
        jns     .b80_den_pos
        not     ebx
        neg     ecx
        sbb     ebx, -1
        or      dword [esp + 4], 2
.b80_den_pos:

.b80_unsigned:
        ; Long division: rem = EDI:ESI (start 0), num = EDX:EAX
        ; (shifted out bit-by-bit), den = EBX:ECX, quot built on
        ; the stack.
        xor     esi, esi
        xor     edi, edi
        mov     ebp, 64
.b80_lloop:
        ; num <<= 1, capturing bit 63 into CF.
        shl     eax, 1
        rcl     edx, 1
        ; rem = (rem << 1) | CF
        rcl     esi, 1
        rcl     edi, 1
        ; Tentative rem -= den.
        sub     esi, ecx
        sbb     edi, ebx
        jc      .b80_undo
        ; rem >= den: keep subtract, set quot bit (quot = (quot<<1)|1).
        shl     dword [esp + 8], 1
        rcl     dword [esp + 12], 1
        or      dword [esp + 8], 1
        jmp     .b80_cont
.b80_undo:
        add     esi, ecx
        adc     edi, ebx
        shl     dword [esp + 8], 1
        rcl     dword [esp + 12], 1
.b80_cont:
        dec     ebp
        jnz     .b80_lloop

        ; Select output: quot (op bit 1 = 0) or rem (op bit 1 = 1).
        mov     ebp, [esp + 0]
        test    ebp, 2
        jnz     .b80_pick_mod
        mov     eax, [esp + 8]
        mov     edx, [esp + 12]
        jmp     .b80_signed_adjust
.b80_pick_mod:
        mov     eax, esi
        mov     edx, edi

.b80_signed_adjust:
        test    ebp, 1
        jz      .b80_restore
        mov     ecx, [esp + 4]
        test    ebp, 2
        jnz     .b80_mod_sign
        ; Quot sign = num_neg XOR den_neg
        mov     bl, cl
        shr     bl, 1
        xor     bl, cl
        and     bl, 1
        jz      .b80_restore
        not     edx
        neg     eax
        sbb     edx, -1
        jmp     .b80_restore
.b80_mod_sign:
        ; Mod sign = num_neg (truncated division, C99).
        test    cl, 1
        jz      .b80_restore
        not     edx
        neg     eax
        sbb     edx, -1

.b80_restore:
        add     esp, 16
        pop     ebx
        pop     ecx
        pop     esi
        pop     edi
        pop     ebp
        iretd
"""


# Same Watcom-discovery pattern as `compare.py` (CI sets WATCOM env;
# dev hosts on Linux typically install via `~/.local/opt/watcom`).
WATCOM_CANDIDATES = [
    "wlink",
    str(Path.home() / ".local/opt/watcom/binl64/wlink"),
    str(Path.home() / ".local/opt/watcom/binl/wlink"),
]
if env := os.environ.get("WATCOM"):
    WATCOM_CANDIDATES.insert(0, str(Path(env) / "binl64/wlink"))
    WATCOM_CANDIDATES.insert(1, str(Path(env) / "binl/wlink"))


def _which_first(candidates: list[str]) -> str | None:
    for c in candidates:
        if "/" in c:
            if Path(c).is_file() and os.access(c, os.X_OK):
                return c
        else:
            found = shutil.which(c)
            if found:
                return found
    return None


def build_exe(
    asm_path: Path,
    out_path: Path,
    *,
    extender: str = "pmodew",
    extra_obj_files: list[Path] | None = None,
    dos32a_stub_path: Path | None = None,
) -> tuple[bool, str]:
    """Run nasm + wlink to turn `asm_path` into `out_path` (.exe).

    Returns (ok, message). The message is human-readable on failure
    (preserved stderr from whichever tool died) or empty on success.

    `extender` controls the wlink `system <X>` directive:
        - "pmodew"   : bundles PMODE/W (BSD-ish) — self-contained
                       .exe, ~9 KB stub overhead. Default.
        - "causeway" : LE binary that needs cwstub.exe alongside.
                       (verified empirically: `system causeway`
                       does not bind the extender — it produces a
                       371-byte stub-only .exe whose MZ stub prints
                       "This is a CauseWay executable" and exits.)
        - "dos4g"    : LE binary that needs dos4gw.exe alongside.

    `extra_obj_files` are additional .obj files to link in (e.g. a
    libc shim that bridges between uc386's calling convention and
    DOS/4GW's startup expectations — not yet written, see
    `docs/path-a-mz-le.md` for the plan)."""
    if shutil.which("nasm") is None:
        return False, "nasm not found — install with apt/brew"
    wlink = _which_first(WATCOM_CANDIDATES)
    # wlink is OPTIONAL — when missing (typical on macOS, where Open
    # Watcom has no native build), the linker step falls back to pyle
    # (pure Python, ships in the same package). The fallback only
    # supports the pmodew + dos32a extenders; for causeway/dos4g,
    # wlink is still required so we error out below if it's missing.
    if wlink is None and extender not in ("pmodew", "dos32a"):
        return False, (
            f"wlink not found — extender={extender!r} requires Open Watcom. "
            f"Use --extender=pmodew or --extender=dos32a to use the pyle "
            f"fallback (pure Python). "
            f"Or install Open Watcom V2 and set WATCOM=<install-dir>."
        )

    # uc386 emits `section .text` / `section .data` / `section .bss`
    # without the OMF-specific `use32 class=...` modifiers. NASM's
    # `-f obj` defaults to USE16 segments, which makes the resulting
    # OMF declare 32-bit code as 16-bit. wlink links it cleanly but
    # the LE-loader runs it with the D-bit clear → CPU treats every
    # instruction as 16-bit and execution wanders off into garbage
    # (DOSBox: "Illegal read from 4cb4f3*").
    # Rewrite each section line to include `use32` + an OMF class
    # before NASM sees it.
    #
    # Note on argv: uc386's `_start` does `push ebx; push eax` to
    # convert dos_emu's register-passed argc/argv into cdecl on the
    # stack. Under PMODE/W those registers contain extender-internal
    # state, so the pushes pass garbage to _main. Empirically:
    # `echo hello dos > out.txt` produces `exe hello dos` (argv has
    # 4 elements with argv[1]="exe" — looks like PMODE/W's command-
    # line parser is contributing something through a side channel).
    # Stripping the pushes (tested in Phase 7) didn't change the
    # output, so PMODE/W isn't placing argc/argv on the stack at
    # entry either — argv reaches _main via some channel uc386
    # doesn't read. Real fix needs a bridge stub that:
    #   1. parses PSP+0x80 (real-mode cmdline tail) via DPMI INT 31h
    #   2. allocates argv[] in the LE data segment
    #   3. sets EAX=argc, EBX=&argv[0]
    #   4. jumps to _start
    # That's a separate addons/harness/exe_argv_bridge.asm. For now
    # `.exe` programs that don't read argv work correctly (true,
    # false, yes, factor with default input).
    asm_text = asm_path.read_text()
    rewritten = []
    for line in asm_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("section .text"):
            rewritten.append("        section _TEXT use32 class=CODE")
            continue
        if stripped.startswith("section .data"):
            rewritten.append("        section _DATA use32 class=DATA")
            continue
        if stripped.startswith("section .bss"):
            rewritten.append("        section _BSS use32 class=BSS")
            continue
        # Strip libc's _stdin/_stdout/_stderr definitions — the
        # bridge stub redefines them with real DOS handles 0/1/2
        # instead of the dos_emu sentinels 0xF0/F1/F2.
        if stripped.startswith(("_stdin:", "_stdout:", "_stderr:")) \
                and "dd 0xF" in stripped:
            continue
        rewritten.append(line)
    # Declare the stripped stream globals as externs so user code
    # like `push dword [_stdout]` still assembles. The definitions
    # come from the bridge stub at link time.
    rewritten.insert(0, "        extern _stderr")
    rewritten.insert(0, "        extern _stdout")
    rewritten.insert(0, "        extern _stdin")
    # The bridge stub now drives the full startup (FPU init, BSS
    # zero, call _main, exit) so it can place diagnostic markers
    # between each step. That requires the linker to resolve
    # _main, _bss_zero_start, _bss_zero_end across object-file
    # boundaries — codegen defines them as labels but doesn't
    # export them. Inject the globals here so wlink can wire the
    # references from the bridge to the codegen-emitted bodies.
    #
    # Codegen drops the BSS labels entirely when a TU has no
    # uninitialized non-noinit globals (the rep stosb is also
    # skipped via _needs_bss_init). For those small TUs (e.g.
    # echo.exe) we have to emit a degenerate stub here so the
    # bridge's `extern` references still resolve — _start ==
    # _end means the rep stosb loop counts to zero.
    has_bss_labels = any(
        line.lstrip().startswith("_bss_zero_start:")
        for line in rewritten
    )
    # _write is only present in TUs that pull in the libc bundle's
    # write() (e.g. MP.EXE — main.c uses write() for the startup
    # markers). echo.exe goes through fputs/putchar which don't pull
    # _write. Only export _write when the asm actually defines it,
    # otherwise wlink errors on the bridge's `extern _write`.
    has_write = any(line.lstrip().startswith("_write:") for line in rewritten)
    if has_write:
        rewritten.insert(0, "        global _write")
    rewritten.insert(0, "        global _main")
    if has_bss_labels:
        rewritten.insert(0, "        global _bss_zero_end")
        rewritten.insert(0, "        global _bss_zero_start")
    else:
        rewritten.append("        section _BSS use32 class=BSS")
        rewritten.append("        global _bss_zero_start")
        rewritten.append("        global _bss_zero_end")
        rewritten.append("_bss_zero_start:")
        rewritten.append("_bss_zero_end:")
    asm_for_omf = out_path.with_suffix(".omf.asm")
    asm_for_omf.write_text("\n".join(rewritten) + "\n")

    obj_path = out_path.with_suffix(".obj")
    proc = subprocess.run(
        ["nasm", "-f", "obj", "-o", str(obj_path), str(asm_for_omf)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, f"nasm rc={proc.returncode}: {proc.stderr[:400]}"

    # Phase 7 bridge: dos_emu uses 0xF0/F1/F2 as magic stdin/out/err
    # values (so `fp == NULL` doesn't accidentally match stdin); real
    # DOS via PMODE/W needs raw fd 0/1/2 for INT 21h AH=0x40 (write).
    # A 7-byte mismatch silently breaks every fputs / fwrite / fprintf
    # call — `myecho.exe hello dos > out.txt` produces 767 spaces and
    # no actual content. Patch the globals at PMODE/W entry, then jump
    # to the codegen-emitted _start. argv parsing (PSP+0x80 via DPMI
    # INT 31h) lands in the same stub once stdout is verified working.
    bridge_asm = out_path.with_suffix(".bridge.asm")
    bridge_asm.write_text(BRIDGE_ASM)
    bridge_obj = out_path.with_suffix(".bridge.obj")
    proc = subprocess.run(
        ["nasm", "-f", "obj", "-o", str(bridge_obj), str(bridge_asm)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False, f"nasm bridge rc={proc.returncode}: {proc.stderr[:400]}"

    # If wlink is missing (typical on macOS), fall back to pyle — our
    # pure-Python OMF→MZ+LE linker. pyle handles the same bridge.obj
    # + user.obj + PMODE/W stub combination wlink does, just without
    # needing Open Watcom installed. Verified end-to-end against
    # bit-identical pmodew stub bytes carved from a reference MP.EXE.
    if wlink is None and extender == "pmodew":
        from . import pyle
        stub = Path(__file__).resolve().parent / "pmodew_stub.bin"
        if not stub.is_file():
            return False, (
                f"pyle fallback: missing PMODE/W stub at {stub}. "
                f"Carve from a wlink-built .exe and place there."
            )
        try:
            objects = [pyle.parse_omf(p) for p in (obj_path, bridge_obj)]
            image = pyle.link(objects)
            pyle.write_le(image, stub.read_bytes(), "_pmodew_start", out_path)
        except Exception as exc:
            return False, f"pyle: {exc}"
        return True, ""

    # DOS/32A pyle path. Unlike PMODE/W, DOS/32A's stub isn't a
    # ready-to-prepend blob — it's a standalone DOS32A.EXE that needs
    # SUNSYS Bind's transform applied first. pyle.bind_dos32a_stub
    # does that transform in pure Python. The caller supplies the
    # path to DOS32A.EXE (not bundled because it's a 27 KB binary
    # licensed under zlib that the host project would have to vendor
    # — easier to fetch on demand from archive.org).
    if wlink is None and extender == "dos32a":
        from . import pyle
        if dos32a_stub_path is None or not dos32a_stub_path.is_file():
            return False, (
                f"dos32a: pass --stub-binary <path/to/DOS32A.EXE>. "
                f"Fetch from https://archive.org/details/dos32a-912-bin."
            )
        try:
            raw_stub = dos32a_stub_path.read_bytes()
            stub_bytes = pyle.bind_dos32a_stub(raw_stub)
            objects = [pyle.parse_omf(p) for p in (obj_path, bridge_obj)]
            image = pyle.link(objects)
            pyle.write_le(
                image, stub_bytes, "_pmodew_start", out_path,
                explicit_stack_object=True,
            )
        except Exception as exc:
            return False, f"pyle (dos32a): {exc}"
        return True, ""

    # wlink wants WATCOM in env so it can find its stub library.
    env = os.environ.copy()
    if "WATCOM" not in env:
        env["WATCOM"] = str(Path(wlink).parent.parent)

    # Locate the extender stub binary so wlink can BIND it as the
    # MZ portion of the .exe (the file becomes self-contained: real
    # DOS / FreeDOS / DOSBox load the MZ stub, which is the extender
    # itself, which then loads the LE payload that follows).
    # Without `option stub=...`, `system <X>` produces a 371-byte LE
    # whose MZ portion just prints "This is a X executable" and
    # exits — verified empirically in CI.
    stub_name = {
        "pmodew": "pmodew.exe",
        "causeway": "cwstub.exe",
        "dos4g": "dos4gw.exe",
    }.get(extender)
    stub_path: Path | None = None
    if stub_name:
        # Watcom ships these under $WATCOM/binw/ (the 16-bit DOS
        # binaries — the stubs themselves are real-mode .exe).
        candidates = [
            Path(env["WATCOM"]) / "binw" / stub_name,
            Path(env["WATCOM"]) / "binnt" / stub_name,
        ]
        for p in candidates:
            if p.is_file():
                stub_path = p
                break

    cmd = [
        wlink, "system", extender,
        "name", str(out_path),
        "file", str(obj_path),
        # `option stack=64k` allocates a 64-KB protected-mode stack
        # at link time. Without it wlink prints `W1014: stack segment
        # not found` and the .exe runs with a stack at whatever
        # garbage address the LE-loader picks — DOSBox reports
        # "Illegal read from <addr>" when the program tries to push.
        "option", "stack=64k",
        # `option start=_pmodew_start` enters via the bridge stub
        # (fixes stdin/out/err sentinels, future home of argv setup),
        # which falls through to the codegen-emitted `_start` (FPU
        # init, BSS init, call _main, INT 21h AH=4Ch exit).
        "option", "start=_pmodew_start",
        "file", str(bridge_obj),
    ]
    if stub_path is not None:
        # wlink's `option stub=...` directive writes <stub-file>
        # bytes verbatim as the .exe's MZ portion, then writes the
        # LE payload after it.
        cmd.extend(["option", f"stub={stub_path}"])
    for extra in extra_obj_files or []:
        cmd.extend(["file", str(extra)])
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0 or not out_path.exists():
        return False, (
            f"wlink rc={proc.returncode}: "
            f"stdout={proc.stdout[:400]} stderr={proc.stderr[:400]}"
        )

    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="addons.harness.exe",
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("source", help=".c source to compile, OR .asm to skip uc386")
    ap.add_argument("-o", "--output", required=True, help="output .exe path")
    ap.add_argument(
        "--extender", default="pmodew",
        choices=["pmodew", "causeway", "dos4g", "dos32a"],
        help="DOS extender to bundle (default: pmodew)",
    )
    ap.add_argument(
        "--stub-binary", type=Path, default=None,
        help="Path to DOS32A.EXE (required when --extender=dos32a).",
    )
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out = Path(args.output).resolve()

    # If a .c is provided, run uc386 first to produce the .asm.
    if src.suffix == ".c":
        asm_path = out.with_suffix(".asm")
        proc = subprocess.run(
            [sys.executable, "-m", "uc386.main", str(src),
             "-o", str(asm_path), "-I", str(LIB_INCLUDE)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"uc386 rc={proc.returncode}: {proc.stderr[:400]}\n"
            )
            return 1
    elif src.suffix == ".asm":
        asm_path = src
    else:
        sys.stderr.write(f"unrecognised extension: {src.suffix}\n")
        return 2

    ok, msg = build_exe(
        asm_path, out,
        extender=args.extender,
        dos32a_stub_path=args.stub_binary,
    )
    if not ok:
        sys.stderr.write(f"exe build failed: {msg}\n")
        return 1
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
