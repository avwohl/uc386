@echo off
rem ============================================================
rem DOSBox-X autoexec for the MicroPython network test rig.
rem
rem Loads the Crynwr ne2000 packet driver at INT 0x60 (matches
rem pktdrv_uc386dos.c's IVT scan target), then runs MP.EXE under
rem the bundled DOS extender. SLIRP's DHCP server hands out
rem 10.0.2.15, same lease the dos_emu sim does, so the
rem MicroPython REPL's DHCP probe should succeed identically in
rem both environments.
rem
rem All output is teed into RIG.LOG so run.sh can read it after
rem DOSBox-X exits in -silent mode.
rem ============================================================

echo --- DOSBox-X test rig --- > RIG.LOG
mem /c | find "available" >> RIG.LOG

if not exist NE2000.COM goto :no_pktdrv
echo Loading Crynwr ne2000 packet driver at INT 0x60 ... >> RIG.LOG
NE2000 0x60 9 0x300 >> RIG.LOG
goto :run_mp

:no_pktdrv
echo NE2000.COM not found in C: >> RIG.LOG
goto :run_mp

:run_mp
if not exist MP.EXE goto :no_exe
echo --- before-mp marker --- >> RIG.LOG
if exist SCRIPT.PY goto :run_scripted
echo Running MP.EXE interactively ... >> RIG.LOG
MP.EXE >> RIG.LOG
goto :after_mp

:run_scripted
echo Running MP.EXE with SCRIPT.PY ... >> RIG.LOG
rem Capture mp.exe stdout into a separate file so a partial
rem (truncated mid-write) RIG.LOG still shows the "after-mp"
rem marker — tells us whether MP.EXE returned at all vs. hung.
MP.EXE < SCRIPT.PY > MP_OUT.TXT
type MP_OUT.TXT >> RIG.LOG
goto :after_mp

:after_mp
echo --- after-mp marker --- >> RIG.LOG
goto :done

:no_exe
echo MP.EXE not found in C: >> RIG.LOG
echo Build it on a Linux box via addons/harness/exe.py >> RIG.LOG

:done
echo --- Test rig finished --- >> RIG.LOG
