@echo off
title TicketLens - Local AI Ticket Analytics
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem Read the version from src\__init__.py (the single source of truth) rather than
rem hardcoding it here, where it silently drifted from 1.1.0 to a stale "v1.0.0".
rem Parsed with findstr so this works before the venv/Python exists.
set "APPVER=unknown"
for /f "tokens=2 delims==" %%v in ('findstr /b /c:"__version__" "src\__init__.py" 2^>nul') do (
    set "APPVER=%%~v"
)
set "APPVER=%APPVER: =%"
set "APPVER=%APPVER:"=%"

echo.
echo  ============================================================
echo    TicketLens   v%APPVER%
echo    Local AI Ticket Analytics
echo    Privacy-first ticket clustering and insights - your ticket
echo    data stays on this machine (AI models download once, on
echo    first run, from Hugging Face).
echo    Free and open source, Apache-2.0 licensed.
echo    https://github.com/AneekHait/TicketLens
echo  ============================================================
echo.

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.deps_installed"
set "LLAMA_MARK=%VENV%\.llama_ok"
set "REPAIR=%~dp0repair_llm.ps1"

REM --- Validate any existing venv actually runs on THIS machine ----------------
REM A .venv copied from another machine/user hardcodes an absolute interpreter
REM path in pyvenv.cfg and will not run here. Detect a broken/copied venv and
REM rebuild it rather than failing later with a confusing error. Deleting the
REM venv also drops the .deps_installed marker, so dependencies reinstall.
if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>&1
    if errorlevel 1 (
        echo [TicketLens] The existing .venv is not usable on this machine ^(likely copied^) - rebuilding...
        rmdir /s /q "%VENV%"
    )
)

REM --- First run (or rebuild): create the virtual environment ------------------
if not exist "%VPY%" (
    call :check_python
    if errorlevel 1 (
        echo.
        pause
        exit /b 1
    )
    echo [TicketLens] Creating virtual environment in %VENV% ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo Install Python 3.11-3.14 ^(3.12 recommended^) and ensure it is on PATH.
        echo.
        pause
        exit /b 1
    )
)

REM --- First run: install dependencies INTO the venv (venv pip works) ----------
if not exist "%MARKER%" (
    echo [TicketLens] Installing dependencies into %VENV% - this can take several minutes...
    "%VPY%" -m pip install --upgrade pip

    REM Prefer a bundled wheel matching this CPU profile (avx2/sse) so we never
    REM install the prebuilt AVX-512 wheel that crashes on CPUs without AVX-512.
    "%VPY%" -c "import ctypes;print('avx2' if ctypes.windll.kernel32.IsProcessorFeaturePresent(40) else 'sse')" > "%TEMP%\ts_cpuprof.txt" 2>nul
    set "CPUPROF="
    set /p CPUPROF=<"%TEMP%\ts_cpuprof.txt"
    del "%TEMP%\ts_cpuprof.txt" >nul 2>&1
    set "LLAMA_WHEEL="
    if defined CPUPROF if exist "wheels\!CPUPROF!\llama_cpp_python-*.whl" (
        for %%w in ("wheels\!CPUPROF!\llama_cpp_python-*.whl") do set "LLAMA_WHEEL=%%~fw"
    )
    if defined LLAMA_WHEEL (
        echo [TicketLens] Using bundled llama.cpp wheel for this CPU ^(!CPUPROF!^).
        "%VPY%" -m pip install --no-deps "!LLAMA_WHEEL!"
        REM --only-binary guards against a silent source build if the wheel above
        REM didn't take; the pinned llama version is already satisfied otherwise.
        "%VPY%" -m pip install -r requirements.txt --only-binary llama-cpp-python
    ) else (
        "%VPY%" -m pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Dependency installation failed - see the messages above.
        echo Fix the issue and re-run this file to retry.
        echo.
        pause
        exit /b 1
    )
    echo ok>"%MARKER%"
    echo [TicketLens] Setup complete.
)

REM --- First run per machine: offer to build a CPU-optimized engine wheel -------
REM Building is user-consented (never automatic). Asked once per machine; empty
REM input / non-interactive launch = No. If Yes, we compile a reusable wheel in
REM the foreground so its progress streams live in this window.
set "WHEEL_PROMPT=%VENV%\.wheel_prompted"
set "NEED_WHEEL_PROMPT=1"
if exist "%WHEEL_PROMPT%" (
    set "PROMPTHOST="
    set /p PROMPTHOST=<"%WHEEL_PROMPT%"
    if /I "!PROMPTHOST!"=="%COMPUTERNAME%" set "NEED_WHEEL_PROMPT=0"
)
if "%NEED_WHEEL_PROMPT%"=="1" (
    echo.
    echo [TicketLens] The local AI ^(LLM^) engine can be compiled specifically for
    echo               this machine's CPU. It needs Visual Studio C++ Build Tools
    echo               and takes a few minutes. A bundled wheel is used if you skip;
    echo               you can also build later with build_wheel.bat.
    set "BUILDW="
    set /p BUILDW=Build a llama.cpp wheel optimized for this machine now? [Y/N]:
    REM Redirect-first so a computer name ending in a digit isn't parsed as "N>".
    >"%WHEEL_PROMPT%" echo %COMPUTERNAME%
    if /I "!BUILDW!"=="Y" (
        if exist "%REPAIR%" powershell -NoProfile -ExecutionPolicy Bypass -File "%REPAIR%" -ForceBuild
    )
)

REM --- Per-machine AI engine (llama.cpp) health check ----------------------------
REM Verify-only: confirm the engine loads (bundled/prebuilt/just-built wheel) and
REM stamp a per-machine marker. This NEVER compiles - building is owned solely by
REM the prompt above and build_wheel.bat. The marker is stamped with the computer
REM name so the check runs once per machine and re-runs if copied to another PC.
set "NEED_LLAMA_CHECK=1"
if exist "%LLAMA_MARK%" (
    set "MARKHOST="
    set /p MARKHOST=<"%LLAMA_MARK%"
    if /I "!MARKHOST!"=="%COMPUTERNAME%" set "NEED_LLAMA_CHECK=0"
)

if "%NEED_LLAMA_CHECK%"=="1" (
    if exist "%REPAIR%" (
        echo [TicketLens] Verifying the local AI engine for this machine ^(one-time^)...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%REPAIR%" -VerifyOnly
        if errorlevel 1 (
            echo [TicketLens] NOTE: the local LLM is not enabled on this machine yet.
            echo                Clustering runs with keyword-based labels. To enable it,
            echo                run build_wheel.bat ^(compiles an engine for this CPU^).
        )
    )
)

REM --- Launch -------------------------------------------------------------------
echo [TicketLens] Launching...
"%VPY%" main.py
set "EXITCODE=%ERRORLEVEL%"
endlocal & exit /b %EXITCODE%

REM ============================================================================
REM Subroutine: verify a supported global Python (3.11-3.14, 64-bit) is on PATH.
REM Called only when the venv must be created. Sets errorlevel 1 on failure.
REM ============================================================================
:check_python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    echo Install CPython 3.11-3.14 ^(3.12 recommended, 64-bit^) from https://python.org
    echo and make sure "python --version" works, then re-run this file.
    exit /b 1
)
set "PYVER="
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set "PYMAJ=%%a"
    set "PYMIN=%%b"
)
if not "!PYMAJ!"=="3" (
    echo ERROR: Python !PYVER! is not supported. Install CPython 3.11-3.14 ^(3.12 recommended^).
    exit /b 1
)
if !PYMIN! LSS 11 (
    echo ERROR: Python !PYVER! is too old. TicketLens needs 3.11-3.14 ^(pandas 3.x requires Python 3.11+^).
    exit /b 1
)
if !PYMIN! GTR 14 (
    echo [TicketLens] WARNING: Python !PYVER! is newer than the tested range ^(3.11-3.14^).
    echo                Some native packages may lack prebuilt wheels; continuing anyway.
)
exit /b 0
