@echo off
title TicketLens - Install OpenVINO acceleration (experimental)
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo    TicketLens - OpenVINO INT8 acceleration (experimental)
echo    Optional. Speeds up embeddings ~3.6-4.2x on CPUs with VNNI
echo    for BERT-family models + Qwen3. PyTorch stays the default.
echo  ============================================================
echo.

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"

REM optimum-intel is installed from git because tagged releases lag the latest
REM transformers. "main" tracks the newest code (works with transformers 5.x but
REM is a moving target). For a REPRODUCIBLE build, set OPTIMUM_INTEL_REF to a
REM specific tag or commit, e.g.:  set "OPTIMUM_INTEL_REF=v1.26.0"
if not defined OPTIMUM_INTEL_REF set "OPTIMUM_INTEL_REF=main"

if not exist "%VPY%" (
    echo ERROR: virtual environment not found. Run run.bat once first.
    echo.
    pause
    exit /b 1
)

echo [TicketLens] Installing OpenVINO support into %VENV% ...
echo [TicketLens] Step 1/3: optimum-intel (%OPTIMUM_INTEL_REF%, --no-deps) ...
"%VPY%" -m pip install --no-deps "git+https://github.com/huggingface/optimum-intel.git@%OPTIMUM_INTEL_REF%"
if errorlevel 1 goto :failed

echo [TicketLens] Step 2/3: optimum 2.2.0 (--no-deps) ...
"%VPY%" -m pip install --no-deps "optimum==2.2.0"
if errorlevel 1 goto :failed

echo [TicketLens] Step 3/3: OpenVINO runtime libraries ...
"%VPY%" -m pip install -r requirements-openvino.txt
if errorlevel 1 goto :failed

echo.
echo [TicketLens] OpenVINO support installed. Restart TicketLens, then pick
echo               "OpenVINO INT8 (CPU)" under Embedding Model -^> Acceleration.
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: OpenVINO install failed - see the messages above.
echo The app still works on PyTorch; you can re-run this file to retry.
echo.
pause
exit /b 1
