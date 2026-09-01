@echo off
title TicketLens - Build local AI engine wheel for this machine
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo  ============================================================
echo    TicketLens - Build the local AI (llama.cpp) engine
echo    Compiles llama-cpp-python for THIS CPU's instruction set and
echo    caches a reusable wheel under wheels\^<profile^>\ (avx2 or sse).
echo    Needs Visual Studio C++ Build Tools; takes a few minutes.
echo  ============================================================
echo.

set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
    echo ERROR: virtual environment not found at %VPY%.
    echo Run run.bat once first to create it, then re-run this file.
    echo.
    pause
    exit /b 1
)

REM -ForceBuild always compiles (interactive: may offer to install Build Tools).
REM Progress streams live because pip runs with -v in this foreground console.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0repair_llm.ps1" -ForceBuild
set "EC=%ERRORLEVEL%"

echo.
if "%EC%"=="0" (
    echo [TicketLens] Done. A reusable wheel is cached under wheels\ and installed.
    echo               You can commit/ship it for other machines with the same CPU class.
) else (
    echo [TicketLens] Build did not complete ^(exit %EC%^). See the messages above.
)
echo.
pause
exit /b %EC%
