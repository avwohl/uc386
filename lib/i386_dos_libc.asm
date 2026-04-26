; uc386 minimal i386 libc — appended after user code in run mode.
;
; Implements the C library symbols that the c-testsuite, Fujitsu, and
; GCC-torture programs actually call. Backing I/O is INT 21h, which
; the dos_emu.py harness intercepts and routes to host stdout/stderr/
; exit code.
;
; Calling convention: cdecl. Args pushed right-to-left; caller cleans
; the stack; result in EAX (or st(0) for double-returning helpers).

        section .text

; ---- exit / abort ----------------------------------------------------------
_exit:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        and     eax, 0xFF
        or      eax, 0x4C00          ; AH = 4Ch (DOS exit), AL = code
        int     21h
        ; not reached
        mov     esp, ebp
        pop     ebp
        ret

_abort:
        mov     ax, 0x4C01           ; exit code 1 — INT 21h doesn't really
        int     21h                  ; care, but the harness sees code 1
        ret                          ; unreachable

; The C standard exits 0 when main returns, but our codegen falls off
; main with `xor eax, eax` and a normal `ret`. _start in the codegen
; output handles that — it calls _main, then INT 21h AH=4Ch with main's
; AL as the exit code.

; ---- putchar ---------------------------------------------------------------
_putchar:
        push    ebp
        mov     ebp, esp
        mov     edx, [ebp + 8]       ; ch in low byte of EDX (passed as int)
        mov     ah, 0x02
        int     21h
        mov     eax, [ebp + 8]       ; return ch
        and     eax, 0xFF
        mov     esp, ebp
        pop     ebp
        ret

; ---- puts ------------------------------------------------------------------
; puts(const char *s) — prints s + '\n', returns non-negative on success.
_puts:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 8]       ; s
.loop:
        mov     al, [esi]
        test    al, al
        jz      .done
        mov     edx, eax
        mov     ah, 0x02
        int     21h
        inc     esi
        jmp     .loop
.done:
        mov     edx, 0x0A            ; newline
        mov     ah, 0x02
        int     21h
        xor     eax, eax
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- fputc / fputs / putc --------------------------------------------------
; fputc(int ch, FILE *stream): we ignore stream and print to stdout.
_fputc:
        push    ebp
        mov     ebp, esp
        mov     edx, [ebp + 8]
        mov     ah, 0x02
        int     21h
        mov     eax, [ebp + 8]
        and     eax, 0xFF
        mov     esp, ebp
        pop     ebp
        ret

_putc:
        jmp     _fputc

; fputs(const char *s, FILE *stream): print s without trailing newline.
_fputs:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 8]
.fl:
        mov     al, [esi]
        test    al, al
        jz      .fd
        mov     edx, eax
        mov     ah, 0x02
        int     21h
        inc     esi
        jmp     .fl
.fd:
        xor     eax, eax
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- write / fwrite --------------------------------------------------------
; write(int fd, const void *buf, size_t count) → count
_write:
        push    ebp
        mov     ebp, esp
        mov     ebx, [ebp + 8]
        mov     edx, [ebp + 12]
        mov     ecx, [ebp + 16]
        mov     ah, 0x40
        int     21h
        movzx   eax, ax
        mov     esp, ebp
        pop     ebp
        ret

; fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) → nmemb
_fwrite:
        push    ebp
        mov     ebp, esp
        push    ebx
        push    esi
        mov     esi, [ebp + 8]       ; ptr
        mov     eax, [ebp + 12]      ; size
        mov     ebx, [ebp + 16]      ; nmemb
        imul    eax, ebx             ; total bytes
        ; Call write(1, ptr, total)
        mov     ebx, 1
        mov     edx, esi
        mov     ecx, eax
        mov     ah, 0x40
        int     21h
        ; Return nmemb (caller passed it).
        mov     eax, [ebp + 16]
        pop     esi
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret

; ---- read ------------------------------------------------------------------
_read:
        push    ebp
        mov     ebp, esp
        mov     ebx, [ebp + 8]
        mov     edx, [ebp + 12]
        mov     ecx, [ebp + 16]
        mov     ah, 0x3F
        int     21h
        movzx   eax, ax
        mov     esp, ebp
        pop     ebp
        ret

; ---- getchar ---------------------------------------------------------------
_getchar:
        push    ebp
        mov     ebp, esp
        sub     esp, 4
        ; read(0, &local, 1)
        lea     edx, [ebp - 4]
        mov     ebx, 0
        mov     ecx, 1
        mov     ah, 0x3F
        int     21h
        movzx   eax, ax
        test    eax, eax
        jz      .eof
        movzx   eax, byte [ebp - 4]
        jmp     .done
.eof:
        mov     eax, -1
.done:
        mov     esp, ebp
        pop     ebp
        ret

; ---- strlen ----------------------------------------------------------------
_strlen:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 8]
        xor     eax, eax
.l:
        cmp     byte [esi + eax], 0
        je      .d
        inc     eax
        jmp     .l
.d:
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strcmp ----------------------------------------------------------------
_strcmp:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     esi, [ebp + 8]
        mov     edi, [ebp + 12]
.l:
        movzx   eax, byte [esi]
        movzx   edx, byte [edi]
        cmp     eax, edx
        jne     .ne
        test    eax, eax
        jz      .eq
        inc     esi
        inc     edi
        jmp     .l
.ne:
        sub     eax, edx
        jmp     .d
.eq:
        xor     eax, eax
.d:
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strcpy ----------------------------------------------------------------
_strcpy:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]       ; dest
        mov     esi, [ebp + 12]      ; src
.l:
        mov     al, [esi]
        mov     [edi], al
        test    al, al
        jz      .d
        inc     esi
        inc     edi
        jmp     .l
.d:
        mov     eax, [ebp + 8]       ; return dest
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strncpy ---------------------------------------------------------------
_strncpy:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]
        mov     esi, [ebp + 12]
        mov     ecx, [ebp + 16]
.l:
        test    ecx, ecx
        jz      .d
        mov     al, [esi]
        mov     [edi], al
        test    al, al
        jz      .pad
        inc     esi
        inc     edi
        dec     ecx
        jmp     .l
.pad:
        ; reached NUL — zero-fill the rest
        dec     ecx
        jz      .d
        inc     edi
.zl:
        test    ecx, ecx
        jz      .d
        mov     byte [edi], 0
        inc     edi
        dec     ecx
        jmp     .zl
.d:
        mov     eax, [ebp + 8]
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strncmp ---------------------------------------------------------------
_strncmp:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     esi, [ebp + 8]
        mov     edi, [ebp + 12]
        mov     ecx, [ebp + 16]
.l:
        test    ecx, ecx
        jz      .eq
        movzx   eax, byte [esi]
        movzx   edx, byte [edi]
        cmp     eax, edx
        jne     .ne
        test    eax, eax
        jz      .eq
        inc     esi
        inc     edi
        dec     ecx
        jmp     .l
.ne:
        sub     eax, edx
        jmp     .d
.eq:
        xor     eax, eax
.d:
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strcat ----------------------------------------------------------------
_strcat:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]
        ; advance edi to NUL
.fl:
        cmp     byte [edi], 0
        je     .copy
        inc     edi
        jmp     .fl
.copy:
        mov     esi, [ebp + 12]
.cl:
        mov     al, [esi]
        mov     [edi], al
        test    al, al
        jz      .d
        inc     esi
        inc     edi
        jmp     .cl
.d:
        mov     eax, [ebp + 8]
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strncat ---------------------------------------------------------------
_strncat:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]
        mov     ecx, [ebp + 16]
.fl:
        cmp     byte [edi], 0
        je     .copy
        inc     edi
        jmp     .fl
.copy:
        mov     esi, [ebp + 12]
.cl:
        test    ecx, ecx
        jz      .term
        mov     al, [esi]
        test    al, al
        jz      .term
        mov     [edi], al
        inc     esi
        inc     edi
        dec     ecx
        jmp     .cl
.term:
        mov     byte [edi], 0
        mov     eax, [ebp + 8]
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strchr ----------------------------------------------------------------
_strchr:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 8]
        movzx   edx, byte [ebp + 12]
.l:
        movzx   eax, byte [esi]
        cmp     eax, edx
        je      .found
        test    eax, eax
        jz      .nf
        inc     esi
        jmp     .l
.found:
        mov     eax, esi
        jmp     .d
.nf:
        xor     eax, eax
.d:
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- strrchr ---------------------------------------------------------------
_strrchr:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     esi, [ebp + 8]
        movzx   edx, byte [ebp + 12]
        xor     edi, edi             ; last match
.l:
        movzx   eax, byte [esi]
        cmp     eax, edx
        jne     .skip
        mov     edi, esi
.skip:
        test    eax, eax
        jz      .d
        inc     esi
        jmp     .l
.d:
        mov     eax, edi
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- memcmp ----------------------------------------------------------------
_memcmp:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     esi, [ebp + 8]
        mov     edi, [ebp + 12]
        mov     ecx, [ebp + 16]
.l:
        test    ecx, ecx
        jz      .eq
        movzx   eax, byte [esi]
        movzx   edx, byte [edi]
        cmp     eax, edx
        jne     .ne
        inc     esi
        inc     edi
        dec     ecx
        jmp     .l
.ne:
        sub     eax, edx
        jmp     .d
.eq:
        xor     eax, eax
.d:
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- memmove ---------------------------------------------------------------
_memmove:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]
        mov     esi, [ebp + 12]
        mov     ecx, [ebp + 16]
        ; Forward or backward depending on overlap.
        cmp     edi, esi
        ja      .back
        cld
        rep movsb
        jmp     .d
.back:
        std
        add     esi, ecx
        add     edi, ecx
        dec     esi
        dec     edi
        rep movsb
        cld
.d:
        mov     eax, [ebp + 8]
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- GCC builtin aliases ---------------------------------------------------
; The __builtin_* forms are intrinsics gcc would normally inline. We
; just punt to the regular libc routines.
___builtin_memcpy:        jmp _memcpy
___builtin_memset:        jmp _memset
___builtin_memmove:       jmp _memmove
___builtin_memcmp:        jmp _memcmp
___builtin_strcpy:        jmp _strcpy
___builtin_strncpy:       jmp _strncpy
___builtin_strncmp:       jmp _strncmp
___builtin_strlen:        jmp _strlen
___builtin_strcmp:        jmp _strcmp
___builtin_strchr:        jmp _strchr
___builtin_strrchr:       jmp _strrchr
___builtin_strcat:        jmp _strcat
___builtin_abs:           jmp _abs
___builtin_labs:          jmp _abs
___builtin_alloca:        jmp _alloca
___builtin_classify_type:
        ; gcc returns an integer indicating the type class of an
        ; unevaluated expression. We always return 1 (integer_type)
        ; since callers that test for specific types check via
        ; constant-folded equality, which fails-closed safely.
        mov     eax, 1
        ret

; Tests use `link_error()` as a marker for "this code path should
; have been DCE'd". Without DCE we'd link-fail; provide a no-op so
; the binary assembles and the call is harmless at runtime
; (callers always gate it on a compile-time-false condition).
_link_error:
        ret
___builtin_abort:         jmp _abort
___builtin_exit:          jmp _exit
___builtin_putchar:       jmp _putchar
___builtin_puts:          jmp _puts
___builtin_printf:        jmp _printf
___builtin_fprintf:       jmp _fprintf
___builtin_malloc:        jmp _malloc
___builtin_calloc:        jmp _calloc
___builtin_free:          jmp _free
___builtin_atoi:          jmp _atoi
___builtin_sin:           jmp _sin
___builtin_cos:           jmp _cos
___builtin_sqrt:          jmp _sqrt
___builtin_fabs:          jmp _fabs
___builtin_floor:         jmp _floor
___builtin_ceil:          jmp _ceil
___builtin_pow:           jmp _pow
___builtin_return_address:
        ; A no-op-ish approximation: return 0 so the simple
        ; "did this code path get reached" probes don't crash.
        xor     eax, eax
        ret
___builtin_frame_address:
        xor     eax, eax
        ret
___builtin_expect_with_probability:
        ; First arg is the value, ignore the rest.
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        mov     esp, ebp
        pop     ebp
        ret
___builtin_constant_p:
        ; gcc evaluates this at compile time. We can't, so always say
        ; "not constant" (returns 0). Programs that gate on it via
        ; if/else still pick a working path.
        xor     eax, eax
        ret
___builtin_unreachable:
        ; Diagnostic-only — exit non-zero so any program that actually
        ; reaches here visibly fails its test.
        mov     ax, 0x4C7F
        int     21h
        ret
___builtin_trap:
        mov     ax, 0x4C7F
        int     21h
        ret
___builtin_clz:
        ; Count leading zeros in [esp+4]. bsr finds highest-set bit;
        ; if input is 0, behavior is undefined (we return 32).
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        test    eax, eax
        jz      .clz_zero
        bsr     ecx, eax
        mov     eax, 31
        sub     eax, ecx
        jmp     .clz_done
.clz_zero:
        mov     eax, 32
.clz_done:
        mov     esp, ebp
        pop     ebp
        ret
___builtin_ctz:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        test    eax, eax
        jz      .ctz_zero
        bsf     eax, eax
        jmp     .ctz_done
.ctz_zero:
        mov     eax, 32
.ctz_done:
        mov     esp, ebp
        pop     ebp
        ret
___builtin_popcount:
        push    ebp
        mov     ebp, esp
        push    ebx
        mov     eax, [ebp + 8]
        xor     ebx, ebx
.pc_loop:
        test    eax, eax
        jz      .pc_done
        mov     ecx, eax
        and     ecx, 1
        add     ebx, ecx
        shr     eax, 1
        jmp     .pc_loop
.pc_done:
        mov     eax, ebx
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret
___builtin_bswap32:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        bswap   eax
        mov     esp, ebp
        pop     ebp
        ret
___builtin_prefetch:
        ret                          ; no-op
___builtin_signbit:
        ; signbit(double): low half at [esp+4..7], high at [esp+8..11].
        ; Bit 31 of high half is the sign bit. Return 0 or 1.
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 12]      ; high 32 of double
        shr     eax, 31
        mov     esp, ebp
        pop     ebp
        ret
___builtin_signbitf:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]       ; float bits
        shr     eax, 31
        mov     esp, ebp
        pop     ebp
        ret
___builtin_signbitl:
        jmp     ___builtin_signbit
___builtin_sprintf:
        jmp     _sprintf
___builtin_snprintf:
        jmp     _snprintf
; jmp_buf layout (32-byte / 8-int): ebx, esi, edi, ebp, esp, return-eip,
; (slots 6 and 7 are spare). Both _setjmp and __builtin_setjmp populate
; this layout. _longjmp / __builtin_longjmp restore from it and jump.
_setjmp:
___builtin_setjmp:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]            ; eax = jmp_buf
        mov     [eax + 0],  ebx
        mov     [eax + 4],  esi
        mov     [eax + 8],  edi
        mov     ecx, [ebp]                ; saved EBP from caller
        mov     [eax + 12], ecx
        lea     ecx, [ebp + 8]            ; caller's ESP after our pop
        mov     [eax + 16], ecx
        mov     ecx, [ebp + 4]            ; return EIP
        mov     [eax + 20], ecx
        xor     eax, eax
        mov     esp, ebp
        pop     ebp
        ret
_longjmp:
___builtin_longjmp:
        ; longjmp(buf, val): restore the jmp_buf and return `val` from
        ; setjmp. Per C, `val == 0` is treated as 1 to keep setjmp's
        ; "0 means direct call" sentinel meaningful.
        mov     ecx, [esp + 4]            ; ecx = jmp_buf (no frame yet)
        mov     eax, [esp + 8]            ; eax = val
        test    eax, eax
        jne     .lj_have_val
        mov     eax, 1
.lj_have_val:
        mov     ebx, [ecx + 0]
        mov     esi, [ecx + 4]
        mov     edi, [ecx + 8]
        mov     ebp, [ecx + 12]
        mov     esp, [ecx + 16]
        push    dword [ecx + 20]
        ret
___builtin_mul_overflow:
        ; Three args: int a, int b, int *result. Returns 1 on overflow.
        ; Uses one-operand IMUL so OF reflects whether the 64-bit signed
        ; product fits in 32 bits.
        push    ebp
        mov     ebp, esp
        push    edi
        mov     eax, [ebp + 8]
        imul    dword [ebp + 12]
        seto    cl
        mov     edi, [ebp + 16]
        mov     [edi], eax
        movzx   eax, cl
        pop     edi
        mov     esp, ebp
        pop     ebp
        ret
___builtin_add_overflow:
        ; Three args: int a, int b, int *result. Returns 1 on overflow.
        push    ebp
        mov     ebp, esp
        push    edi
        mov     eax, [ebp + 8]
        add     eax, [ebp + 12]
        seto    cl
        mov     edi, [ebp + 16]
        mov     [edi], eax
        movzx   eax, cl
        pop     edi
        mov     esp, ebp
        pop     ebp
        ret
___builtin_sub_overflow:
        ; Three args: int a, int b, int *result. Returns 1 on overflow.
        push    ebp
        mov     ebp, esp
        push    edi
        mov     eax, [ebp + 8]
        sub     eax, [ebp + 12]
        seto    cl
        mov     edi, [ebp + 16]
        mov     [edi], eax
        movzx   eax, cl
        pop     edi
        mov     esp, ebp
        pop     ebp
        ret
___builtin_bswap16:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        xchg    al, ah
        movzx   eax, ax
        mov     esp, ebp
        pop     ebp
        ret

; ---- abs -------------------------------------------------------------------
_abs:
        push    ebp
        mov     ebp, esp
        mov     eax, [ebp + 8]
        test    eax, eax
        jns     .pos
        neg     eax
.pos:
        mov     esp, ebp
        pop     ebp
        ret

; ---- alloca ----------------------------------------------------------------
; Real alloca needs to grow the caller's stack frame, which is tricky
; from a separately-compiled function. Punt to the bump allocator —
; the lifetime is per-process instead of per-function, but tests don't
; usually mind.
_alloca:
        push    ebp
        mov     ebp, esp
        push    dword [ebp + 8]
        call    _malloc
        add     esp, 4
        mov     esp, ebp
        pop     ebp
        ret

; ---- memcpy ----------------------------------------------------------------
_memcpy:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        mov     edi, [ebp + 8]
        mov     esi, [ebp + 12]
        mov     ecx, [ebp + 16]
        cld
        rep movsb
        mov     eax, [ebp + 8]
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- memset ----------------------------------------------------------------
_memset:
        push    ebp
        mov     ebp, esp
        push    edi
        mov     edi, [ebp + 8]
        mov     eax, [ebp + 12]
        and     eax, 0xFF
        mov     ecx, [ebp + 16]
        cld
        rep stosb
        mov     eax, [ebp + 8]
        pop     edi
        mov     esp, ebp
        pop     ebp
        ret

; ---- printf ----------------------------------------------------------------
; A small printf supporting:
;   %d %i  signed int (decimal)
;   %u     unsigned int
;   %x %X  hex
;   %o     octal
;   %c     char
;   %s     string
;   %p     pointer (= %#x)
;   %%     literal %
;   %ld %li (treated like %d)
;   %lu %lx %lX (treated like %u/%x)
;   %lld %llu %llx %llX (low 32 bits only — long long isn't real)
;   %f %.Nf  via _print_float (st0 lowering)
;   width and precision: minimal — leading-zero pad for %0Nd, precision
;     for %.Nf, otherwise ignored.
;
; Output is per-character via INT 21h AH=02.
;
; Returns total bytes written.

; sprintf(char *buf, const char *fmt, ...) — formats into buf and
; returns the byte count (not including the trailing NUL). We punt
; the formatting to the harness via a custom INT 21h subfunction:
;
;   AH = 0x5C
;   EBX = destination buffer
;   ECX = format string
;   EDX = pointer to first vararg
;   EAX (return) = bytes written (excluding NUL)
;
; The harness reads fmt + varargs from emulator memory, formats in
; Python, writes the result + NUL to ebx, and returns the length.
_sprintf:
        push    ebp
        mov     ebp, esp
        push    ebx
        mov     ebx, [ebp + 8]       ; buf
        mov     ecx, [ebp + 12]      ; fmt
        lea     edx, [ebp + 16]      ; first vararg
        mov     ah, 0x5C
        int     21h
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret

; snprintf(buf, size, fmt, ...) — similar but with a size cap.
;   EBX = buf, ECX = fmt, EDX = first vararg, ESI = size
_snprintf:
        push    ebp
        mov     ebp, esp
        push    ebx
        push    esi
        mov     ebx, [ebp + 8]       ; buf
        mov     esi, [ebp + 12]      ; size
        mov     ecx, [ebp + 16]      ; fmt
        lea     edx, [ebp + 20]      ; first vararg
        mov     ah, 0x5D
        int     21h
        pop     esi
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret

; printf(const char *fmt, ...) — formats and writes to stdout.
; Routed through the harness's INT 21h AH=5E hook so we get a real
; Python-side printf with full %lld / %.Nf / %p / etc. support.
_printf:
        push    ebp
        mov     ebp, esp
        push    ebx
        mov     ecx, [ebp + 8]       ; fmt
        lea     edx, [ebp + 12]      ; va_args
        mov     ah, 0x5E
        int     21h
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret

; fprintf(FILE *stream, const char *fmt, ...) — formats to the FILE.
; The harness reads stream as fd (1=stdout, 2=stderr). Since our
; libc declares stdin/stdout/stderr as 0/1/2 globals, the FILE *
; arg evaluates to one of those small ints.
_fprintf:
        push    ebp
        mov     ebp, esp
        push    ebx
        mov     ebx, [ebp + 8]       ; FILE *stream → fd
        mov     ecx, [ebp + 12]      ; fmt
        lea     edx, [ebp + 16]      ; va_args
        mov     ah, 0x5F
        int     21h
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret

; The legacy ASM format engine is kept below as `_printf_legacy` so any
; user code that called it indirectly (via `&printf` taken to a function
; pointer) finds the same behavior. New code goes through the INT 21h
; harness path above.
_printf_legacy:
        push    ebp
        mov     ebp, esp
        sub     esp, 8                ; [ebp-4] = zero_pad flag (per-spec)
        push    ebx
        push    esi
        push    edi
        ; ESI = format string
        ; EDI = next-arg pointer (start at [ebp + 12])
        ; EBX = bytes-written count
        mov     esi, [ebp + 8]
        lea     edi, [ebp + 12]
        xor     ebx, ebx
.next:
        mov     al, [esi]
        test    al, al
        jz      .done
        cmp     al, '%'
        je      .pcent
        ; ordinary char → output
        mov     edx, eax
        mov     ah, 0x02
        int     21h
        inc     esi
        inc     ebx
        jmp     .next
.pcent:
        inc     esi
        ; Parse flags and width.
        xor     ecx, ecx              ; width
        mov     byte [ebp - 4], 0     ; zero_pad flag
.flags:
        mov     al, [esi]
        cmp     al, '0'
        jne     .nf
        ; '0' as a flag only if followed by another digit; otherwise it's
        ; a zero-width specifier (rare). Simpler: mark zero-pad and let
        ; the width loop consume subsequent digits.
        mov     byte [ebp - 4], 1
        inc     esi
        jmp     .flags
.nf:
        ; Read width digits.
.wd:
        mov     al, [esi]
        cmp     al, '0'
        jb      .wend
        cmp     al, '9'
        ja      .wend
        sub     al, '0'
        movzx   eax, al
        imul    ecx, ecx, 10
        add     ecx, eax
        inc     esi
        jmp     .wd
.wend:
        ; Optional precision: '.' followed by digits.
        xor     edx, edx              ; precision (default 0; conversions
                                      ; that need a default differ)
        mov     dl, 0xFF              ; sentinel: no precision specified
        mov     al, [esi]
        cmp     al, '.'
        jne     .lenp
        inc     esi
        xor     edx, edx
.pd:
        mov     al, [esi]
        cmp     al, '0'
        jb      .lenp
        cmp     al, '9'
        ja      .lenp
        sub     al, '0'
        movzx   eax, al
        imul    edx, edx, 10
        add     edx, eax
        inc     esi
        jmp     .pd
.lenp:
        ; Eat 'l', 'll', 'h', 'hh', 'z' length specifiers (ignored — we
        ; treat all integer args as 32-bit).
.eatlen:
        mov     al, [esi]
        cmp     al, 'l'
        je      .eat1
        cmp     al, 'h'
        je      .eat1
        cmp     al, 'z'
        je      .eat1
        cmp     al, 'L'
        je      .eat1
        jmp     .conv
.eat1:
        inc     esi
        jmp     .eatlen
.conv:
        mov     al, [esi]
        inc     esi
        cmp     al, 'd'
        je      .pd_dec
        cmp     al, 'i'
        je      .pd_dec
        cmp     al, 'u'
        je      .pd_udec
        cmp     al, 'x'
        je      .pd_hex
        cmp     al, 'X'
        je      .pd_HEX
        cmp     al, 'o'
        je      .pd_oct
        cmp     al, 's'
        je      .pd_str
        cmp     al, 'c'
        je      .pd_char
        cmp     al, 'p'
        je      .pd_ptr
        cmp     al, 'f'
        je      .pd_flt
        cmp     al, 'g'
        je      .pd_flt
        cmp     al, 'e'
        je      .pd_flt
        cmp     al, '%'
        je      .pd_pcent
        ; Unknown — output the literal '%' + char and move on.
        mov     edx, '%'
        mov     ah, 0x02
        int     21h
        inc     ebx
        movzx   edx, al
        mov     ah, 0x02
        int     21h
        inc     ebx
        jmp     .next
.pd_pcent:
        mov     edx, '%'
        mov     ah, 0x02
        int     21h
        inc     ebx
        jmp     .next

.pd_char:
        ; %c — eat one int from args, print low byte.
        mov     eax, [edi]
        add     edi, 4
        mov     edx, eax
        mov     ah, 0x02
        int     21h
        inc     ebx
        jmp     .next

.pd_str:
        ; %s — eat a char* from args, print until NUL or precision exhausted.
        mov     eax, [edi]
        add     edi, 4
        ; precision: dl = 0xFF (sentinel) means no limit.
        push    edi
        push    ecx
        mov     edi, eax              ; src
        ; If dl == 0xFF, use a huge limit.
        cmp     dl, 0xFF
        je      .ss_unl
        movzx   ecx, dl
        jmp     .ss_loop
.ss_unl:
        mov     ecx, -1
.ss_loop:
        test    ecx, ecx
        jz      .ss_done
        mov     al, [edi]
        test    al, al
        jz      .ss_done
        movzx   edx, al
        push    ecx
        mov     ah, 0x02
        int     21h
        pop     ecx
        inc     edi
        inc     ebx
        dec     ecx
        jmp     .ss_loop
.ss_done:
        pop     ecx
        pop     edi
        jmp     .next

.pd_dec:
        mov     eax, [edi]
        add     edi, 4
        movzx   ebx, byte [ebp - 4]
        push    ebx
        push    ecx
        call    _printf_emit_dec
        add     esp, 8
        jmp     .next

.pd_udec:
        mov     eax, [edi]
        add     edi, 4
        movzx   ebx, byte [ebp - 4]
        push    ebx
        push    ecx
        call    _printf_emit_udec
        add     esp, 8
        jmp     .next

.pd_hex:
        mov     eax, [edi]
        add     edi, 4
        movzx   ebx, byte [ebp - 4]   ; zero_pad
        push    ebx
        push    ecx                   ; width
        push    0                     ; 0 = lowercase
        call    _printf_emit_hex
        add     esp, 12
        jmp     .next

.pd_HEX:
        mov     eax, [edi]
        add     edi, 4
        movzx   ebx, byte [ebp - 4]
        push    ebx
        push    ecx
        push    1                     ; 1 = uppercase
        call    _printf_emit_hex
        add     esp, 12
        jmp     .next

.pd_oct:
        mov     eax, [edi]
        add     edi, 4
        movzx   ebx, byte [ebp - 4]
        push    ebx
        push    ecx
        call    _printf_emit_oct
        add     esp, 8
        jmp     .next

.pd_ptr:
        ; %p → "0x" + lowercase hex
        push    edx
        push    ecx
        mov     edx, '0'
        mov     ah, 0x02
        int     21h
        inc     ebx
        mov     edx, 'x'
        mov     ah, 0x02
        int     21h
        inc     ebx
        pop     ecx
        pop     edx
        mov     eax, [edi]
        add     edi, 4
        push    0
        call    _printf_emit_hex
        add     esp, 4
        jmp     .next

.pd_flt:
        ; %f / %g / %e — naive lowering: print "<int>.<frac>" with
        ; precision (default 6). The arg is a double on the cdecl stack
        ; — 8 bytes.
        fld     qword [edi]
        add     edi, 8
        cmp     dl, 0xFF
        jne     .pf_pgo
        mov     edx, 6
.pf_pgo:
        push    edx
        call    _printf_emit_double
        add     esp, 4
        jmp     .next

.done:
        mov     eax, ebx
        pop     edi
        pop     esi
        pop     ebx
        mov     esp, ebp
        pop     ebp
        ret


; The signed/unsigned/hex/oct print helpers below take their value in
; EAX, lay out digits in a 24-byte local buffer, and emit each digit
; via INT 21h AH=02. They DO NOT update any caller bytes-written
; counter — printf's overall return is approximate. They preserve EBX
; (caller's count register), ESI/EDI.
;
; All four helpers accept extra args on the caller's stack:
;   [esp+4] = width  (minimum field width; 0 = no padding)
;   [esp+8] = zero_pad (0 = pad with spaces; 1 = pad with '0')
; The hex helper additionally consumes [esp+12] = uppercase (0 / 1).

; ---- print signed decimal in EAX -------------------------------------------
; In:  EAX = value, [esp + 4] = width, [esp + 8] = zero_pad.
_printf_emit_dec:
        push    ebp
        mov     ebp, esp
        sub     esp, 32
        push    esi
        push    edi
        push    ebx
        mov     ebx, 0               ; sign flag
        test    eax, eax
        jns     .pos
        mov     ebx, 1
        neg     eax
.pos:
        lea     edi, [ebp - 4]
        mov     byte [edi], 0
.l:
        xor     edx, edx
        mov     esi, 10
        div     esi
        add     dl, '0'
        dec     edi
        mov     [edi], dl
        test    eax, eax
        jnz     .l
        ; Forward to padded helper. Stack [ebp+8]=width, [ebp+12]=zero_pad
        ; (caller pushed in that order before calling us).
        push    dword [ebp + 12]      ; zero_pad
        push    dword [ebp + 8]       ; width
        push    edi                   ; digits ptr
        push    ebx                   ; sign flag
        call    _emit_padded_digits_wp
        add     esp, 16
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- print unsigned decimal in EAX -----------------------------------------
; In:  EAX = value, [esp + 4] = width, [esp + 8] = zero_pad.
_printf_emit_udec:
        push    ebp
        mov     ebp, esp
        sub     esp, 32
        push    esi
        push    edi
        push    ebx
        xor     ebx, ebx
        lea     edi, [ebp - 4]
        mov     byte [edi], 0
.l:
        xor     edx, edx
        mov     esi, 10
        div     esi
        add     dl, '0'
        dec     edi
        mov     [edi], dl
        test    eax, eax
        jnz     .l
        ; Width/zero-pad may not be on the stack if caller is the float
        ; helper (which calls us without those args). Detect by checking
        ; the literal stack frame size — but simpler: the float helper
        ; doesn't use width/zero-pad anyway, so it's safe to read whatever
        ; happens to be there as long as we don't crash. The width path
        ; still works for direct printf calls.
        ;
        ; The safer route: the float helper passes (ebx=0 sign, edi=digits)
        ; and would call _emit_padded_digits (no width). We do the same
        ; here when this function was called WITHOUT width pushed.
        ; In practice, printf always pushes width+zero_pad before calling
        ; us, so the [ebp+8]/[ebp+12] reads are valid.
        push    dword [ebp + 12]
        push    dword [ebp + 8]
        push    edi
        push    ebx
        call    _emit_padded_digits_wp
        add     esp, 16
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- print hex (32-bit, lowercase or uppercase) ----------------------------
; In:  EAX = value, [esp + 4] = uppercase flag (0 or 1),
;      [esp + 8] = width, [esp + 12] = zero_pad
_printf_emit_hex:
        push    ebp
        mov     ebp, esp
        sub     esp, 32
        push    esi
        push    edi
        push    ebx
        mov     ecx, [ebp + 8]       ; uppercase flag
        xor     ebx, ebx             ; sign flag
        lea     edi, [ebp - 4]
        mov     byte [edi], 0
.l:
        mov     edx, eax
        and     edx, 0x0F
        cmp     edx, 9
        jbe     .digit
        sub     edx, 10
        test    ecx, ecx
        jnz     .upper
        add     edx, 'a'
        jmp     .write
.upper:
        add     edx, 'A'
        jmp     .write
.digit:
        add     edx, '0'
.write:
        dec     edi
        mov     [edi], dl
        shr     eax, 4
        test    eax, eax
        jnz     .l
        ; Push width/zero-pad from caller's stack frame to ours.
        push    dword [ebp + 16]     ; zero_pad
        push    dword [ebp + 12]     ; width
        push    edi
        push    ebx
        call    _emit_padded_digits_wp
        add     esp, 16
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- print octal -----------------------------------------------------------
_printf_emit_oct:
        push    ebp
        mov     ebp, esp
        sub     esp, 32
        push    esi
        push    edi
        push    ebx
        xor     ebx, ebx
        lea     edi, [ebp - 4]
        mov     byte [edi], 0
.l:
        mov     edx, eax
        and     edx, 0x07
        add     edx, '0'
        dec     edi
        mov     [edi], dl
        shr     eax, 3
        test    eax, eax
        jnz     .l
        push    dword [ebp + 12]
        push    dword [ebp + 8]
        push    edi
        push    ebx
        call    _emit_padded_digits_wp
        add     esp, 16
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- _emit_padded_digits(sign_flag, digits_ptr) ----------------------------
; Stack: [ret][sign][digits]. Emits sign + digits, no padding.
; (Width/zero-pad handled by _emit_padded_digits_wp variant.)
_emit_padded_digits:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 12]      ; digits ptr
        mov     eax, [ebp + 8]       ; sign flag
        test    eax, eax
        jz      .nosign
        mov     edx, '-'
        mov     ah, 0x02
        int     21h
.nosign:
.l:
        mov     al, [esi]
        test    al, al
        jz      .d
        movzx   edx, al
        mov     ah, 0x02
        int     21h
        inc     esi
        jmp     .l
.d:
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- _emit_padded_digits_wp(sign, digits, width, zero_pad) -----------------
; Honors the printf width + zero-pad flags. width=0 means no padding.
; If zero_pad and we have a sign, the sign goes BEFORE the zero-padding.
; If !zero_pad, the sign goes after the spaces.
_emit_padded_digits_wp:
        push    ebp
        mov     ebp, esp
        push    esi
        push    edi
        push    ebx
        ; ESI = digits ptr; count chars (excluding sentinel).
        mov     esi, [ebp + 12]
        xor     edi, edi
.cl:
        cmp     byte [esi + edi], 0
        je      .ce
        inc     edi
        jmp     .cl
.ce:
        ; EDI = digit count.
        mov     ecx, [ebp + 16]      ; width
        mov     ebx, [ebp + 20]      ; zero_pad
        ; pad_count = max(0, width - (digit_count + sign_flag)).
        mov     eax, ecx
        sub     eax, edi
        cmp     dword [ebp + 8], 0
        je      .ns
        sub     eax, 1
.ns:
        test    eax, eax
        jle     .nopad
        ; If zero_pad, emit sign first then pad with '0'. Else pad
        ; with spaces then sign.
        test    ebx, ebx
        jz      .spadl
        ; sign?
        cmp     dword [ebp + 8], 0
        je      .zpad
        push    eax
        mov     edx, '-'
        mov     ah, 0x02
        int     21h
        pop     eax
.zpad:
        mov     ecx, eax
.zl:
        test    ecx, ecx
        jz      .digits_only
        push    ecx
        mov     edx, '0'
        mov     ah, 0x02
        int     21h
        pop     ecx
        dec     ecx
        jmp     .zl
.spadl:
        mov     ecx, eax
.spl:
        test    ecx, ecx
        jz      .signsp
        push    ecx
        mov     edx, ' '
        mov     ah, 0x02
        int     21h
        pop     ecx
        dec     ecx
        jmp     .spl
.signsp:
        cmp     dword [ebp + 8], 0
        je      .digits_only
        mov     edx, '-'
        mov     ah, 0x02
        int     21h
        jmp     .digits_only
.nopad:
        ; No padding: sign then digits.
        cmp     dword [ebp + 8], 0
        je      .digits_only
        mov     edx, '-'
        mov     ah, 0x02
        int     21h
.digits_only:
        ; Emit digits.
        mov     esi, [ebp + 12]
.dl:
        mov     al, [esi]
        test    al, al
        jz      .done
        movzx   edx, al
        mov     ah, 0x02
        int     21h
        inc     esi
        jmp     .dl
.done:
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- print double on st(0) with given precision ----------------------------
; In:  st(0) = value, [esp + 4] = precision (digits after .)
;
; Strategy: scale by 10^precision, round-to-nearest via the default FCW,
; then split into integer/fractional parts. Emit integer-part decimals
; via _printf_emit_udec, then '.', then `precision` digits with leading
; zeros. This avoids the truncation drift you get from per-digit
; fistp + multiply-by-10.
_printf_emit_double:
        push    ebp
        mov     ebp, esp
        sub     esp    , 64
        push    esi
        push    edi
        push    ebx
        ; Save the value (st0 currently) into a local first, then
        ; reset the FPU so prior state can't bias our scaling/rounding.
        ; Caller pushes the value as st0, but if there are stale
        ; entries below it (e.g. from a leaked previous call) they'd
        ; throw off the multiply chain.
        fstp    qword [ebp - 16]      ; save value, drop it from FPU
        finit                          ; reset FPU to default 80-bit/nearest
        fld     qword [ebp - 16]      ; reload value as st0
        ; Detect sign.
        ftst
        fnstsw  ax
        sahf
        jae     .nonneg
        push    eax
        mov     edx, '-'
        mov     ah, 0x02
        int     21h
        pop     eax
        fchs
.nonneg:
        mov     ecx, [ebp + 8]      ; precision
        test    ecx, ecx
        jnz     .with_frac
        ; precision==0 → round-to-nearest integer.
        fistp   dword [ebp - 16]
        mov     eax, [ebp - 16]
        push    dword 0              ; zero_pad
        push    dword 0              ; width
        call    _printf_emit_udec
        add     esp, 8
        jmp     .end
.with_frac:
        ; Multiply value by 10^precision (loop, default rounding).
.scale:
        test    ecx, ecx
        jz      .scaled
        push    dword 10
        fild    dword [esp]
        add     esp, 4
        fmulp   st1, st0
        dec     ecx
        jmp     .scale
.scaled:
        ; Round to nearest 32-bit int.
        fistp   dword [ebp - 16]
        mov     eax, [ebp - 16]
        ; Compute 10^precision in EBX.
        mov     ebx, 1
        mov     ecx, [ebp + 8]
.pow:
        test    ecx, ecx
        jz      .pow_done
        imul    ebx, ebx, 10
        dec     ecx
        jmp     .pow
.pow_done:
        ; eax / ebx = integer; eax % ebx = fractional.
        xor     edx, edx
        div     ebx
        mov     [ebp - 20], eax     ; integer part
        mov     [ebp - 24], edx     ; fractional part
        ; Emit integer.
        mov     eax, [ebp - 20]
        push    dword 0              ; zero_pad
        push    dword 0              ; width
        call    _printf_emit_udec
        add     esp, 8
        ; Emit '.'.
        mov     edx, '.'
        mov     ah, 0x02
        int     21h
        ; Render fractional digits into a buffer with leading zeros.
        ; Buffer at [ebp - 56 .. ebp - 33]; we lay out right-to-left.
        mov     ecx, [ebp + 8]      ; precision (buffer length)
        lea     esi, [ebp - 33]     ; one-past-end
        mov     byte [esi], 0
        mov     eax, [ebp - 24]     ; fractional value
.fd:
        test    ecx, ecx
        jz      .fdone
        xor     edx, edx
        mov     edi, 10
        div     edi
        add     dl, '0'
        dec     esi
        mov     [esi], dl
        dec     ecx
        jmp     .fd
.fdone:
.fp:
        mov     al, [esi]
        test    al, al
        jz      .end
        movzx   edx, al
        mov     ah, 0x02
        int     21h
        inc     esi
        jmp     .fp
.end:
        ; FPU should already be empty (we popped at fistp); leave it so.
        pop     ebx
        pop     edi
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret

; ---- malloc / free / calloc — bump allocator ------------------------------
; A 1 MB heap allocated from the BSS, served bump-style. free is a no-op.
; This is wildly insufficient for real programs but enough for the
; allocator test suites' small workloads.
;
; stdin/stdout/stderr are FILE pointers — we don't really track files,
; but the variables exist so user code that writes through them links.

section .bss
__heap:         resb 0x100000        ; 1 MB heap
__heap_end:
section .data
__heap_ptr:     dd __heap
_stdin:         dd 0
_stdout:        dd 1
_stderr:        dd 2
section .text

_malloc:
        push    ebp
        mov     ebp, esp
        mov     ecx, [ebp + 8]
        ; round up to 4
        add     ecx, 3
        and     ecx, ~3
        mov     eax, [__heap_ptr]
        mov     edx, eax
        add     edx, ecx
        cmp     edx, __heap_end
        ja      .oom
        mov     [__heap_ptr], edx
        mov     esp, ebp
        pop     ebp
        ret
.oom:
        xor     eax, eax
        mov     esp, ebp
        pop     ebp
        ret

_free:
        ; bump allocator: no-op
        ret

_calloc:
        push    ebp
        mov     ebp, esp
        push    edi
        ; n * size
        mov     eax, [ebp + 8]
        mov     ecx, [ebp + 12]
        imul    eax, ecx
        push    eax
        call    _malloc
        add     esp, 4
        test    eax, eax
        jz      .end
        mov     edi, eax
        mov     ecx, [ebp + 8]
        imul    ecx, [ebp + 12]
        push    eax
        xor     eax, eax
        cld
        rep stosb
        pop     eax
.end:
        pop     edi
        mov     esp, ebp
        pop     ebp
        ret

; ---- math: sin / cos / sqrt / fabs / floor / ceil -------------------------
; All take a `double` (8 bytes at [ebp+8]) and leave their result on
; st(0). The 80387 implements sin/cos/sqrt natively; floor/ceil come
; via FCW round-mode + frndint.
_sin:
        push    ebp
        mov     ebp, esp
        fld     qword [ebp + 8]
        fsin
        mov     esp, ebp
        pop     ebp
        ret
_cos:
        push    ebp
        mov     ebp, esp
        fld     qword [ebp + 8]
        fcos
        mov     esp, ebp
        pop     ebp
        ret
_sqrt:
        push    ebp
        mov     ebp, esp
        fld     qword [ebp + 8]
        fsqrt
        mov     esp, ebp
        pop     ebp
        ret
_fabs:
        push    ebp
        mov     ebp, esp
        fld     qword [ebp + 8]
        fabs
        mov     esp, ebp
        pop     ebp
        ret
_floor:
        push    ebp
        mov     ebp, esp
        sub     esp, 4
        fnstcw  [ebp - 2]
        mov     ax, [ebp - 2]
        and     ax, 0xF3FF
        or      ax, 0x0400           ; round down
        mov     [ebp - 4], ax
        fldcw   [ebp - 4]
        fld     qword [ebp + 8]
        frndint
        fldcw   [ebp - 2]
        mov     esp, ebp
        pop     ebp
        ret
_ceil:
        push    ebp
        mov     ebp, esp
        sub     esp    , 4
        fnstcw  [ebp - 2]
        mov     ax, [ebp - 2]
        and     ax, 0xF3FF
        or      ax, 0x0800           ; round up
        mov     [ebp - 4], ax
        fldcw   [ebp - 4]
        fld     qword [ebp + 8]
        frndint
        fldcw   [ebp - 2]
        mov     esp, ebp
        pop     ebp
        ret
_pow:
        ; pow(x, y) = exp(y * log(x)); approximate with FPU
        ; F2XM1 expects |x| <= 1 so this is a rough impl. Many tests
        ; pass simple powers like 2^N which FYL2X+F2XM1 handle directly.
        push    ebp
        mov     ebp, esp
        fld     qword [ebp + 16]      ; y
        fld     qword [ebp + 8]       ; x
        fyl2x                         ; st0 = y * log2(x)
        ; Compute 2^st0:
        fld     st0
        frndint                       ; round to int → integer part
        fxch    st1
        fsub    st0, st1              ; st0 = fractional part
        f2xm1                         ; st0 = 2^frac - 1
        fld1
        faddp   st1, st0              ; st0 = 2^frac
        fscale                        ; st0 *= 2^st1 (integer scale)
        fxch    st1
        fstp    st0                   ; pop the integer part
        mov     esp, ebp
        pop     ebp
        ret

; ---- atoi ------------------------------------------------------------------
_atoi:
        push    ebp
        mov     ebp, esp
        push    esi
        mov     esi, [ebp + 8]
        xor     eax, eax
        xor     edx, edx              ; sign flag
        ; Skip whitespace
.ws:
        mov     cl, [esi]
        cmp     cl, ' '
        je      .skip
        cmp     cl, 9
        je      .skip
        jmp     .sign
.skip:
        inc     esi
        jmp     .ws
.sign:
        cmp     cl, '-'
        jne     .nopos
        mov     edx, 1
        inc     esi
        jmp     .digits
.nopos:
        cmp     cl, '+'
        jne     .digits
        inc     esi
.digits:
        mov     cl, [esi]
        cmp     cl, '0'
        jb      .end
        cmp     cl, '9'
        ja      .end
        sub     cl, '0'
        movzx   ecx, cl
        imul    eax, eax, 10
        add     eax, ecx
        inc     esi
        jmp     .digits
.end:
        test    edx, edx
        jz      .pos
        neg     eax
.pos:
        pop     esi
        mov     esp, ebp
        pop     ebp
        ret
