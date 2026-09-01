# repair_llm.ps1
#
# Makes the local LLM (llama-cpp-python) load correctly on the CURRENT machine,
# using a reusable, CPU-profile-specific wheel.
#
# Why: the prebuilt "cpu" wheel can be compiled with CPU instructions (e.g.
# AVX-512) that some processors lack. Loading a model then crashes with
# "Windows Error 0xc000001d" (STATUS_ILLEGAL_INSTRUCTION). We ship a wheel per
# instruction-set profile (wheels\avx2, wheels\sse) and, if needed, build one
# from source for this CPU.
#
# Modes:
#   (default)     verify the engine; if it can't load, build from source.
#   -VerifyOnly   verify only (test current -> reuse the profile wheel -> clean
#                 reinstall prebuilt). NEVER compiles. Exit 0 if healthy, else 1.
#   -ForceBuild   always compile a fresh wheel for this CPU (streams progress).
#   -Unattended   never prompt (skip the interactive VS Build Tools install).
#
# Exit code 0 = the LLM loads on this machine (marker stamped). Non-zero = could
# not be enabled (the app then falls back to keyword-only labels).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\repair_llm.ps1 -VerifyOnly
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\repair_llm.ps1 -ForceBuild

[CmdletBinding()]
param(
    # Keep this in sync with llama-cpp-python in requirements.txt.
    [string]$LlamaVersion = "0.3.32",
    [switch]$Unattended,
    [switch]$VerifyOnly,
    [switch]$ForceBuild
)

$ErrorActionPreference = "Stop"

# --- Resolve paths -----------------------------------------------------------
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py     = Join-Path $Root ".venv\Scripts\python.exe"
$Marker = Join-Path $Root ".venv\.llama_ok"

if (-not (Test-Path $Py)) {
    Write-Host "ERROR: venv Python not found at $Py" -ForegroundColor Red
    Write-Host "Run run.bat once first to create the .venv, then re-run this script."
    exit 1
}

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# Stamp the marker with this machine's name and exit successfully.
function Complete-Healthy {
    Set-Content -Path $Marker -Value $env:COMPUTERNAME -Encoding ASCII -NoNewline
    Write-Host "`nSUCCESS: the local AI engine loads on this machine." -ForegroundColor Green
    exit 0
}

# Pick the smallest local .gguf model to use as a real load smoke-test.
function Get-SmokeModel {
    $m = Get-ChildItem -Path (Join-Path $Root 'models') -Filter *.gguf -File -Recurse -ErrorAction SilentlyContinue |
         Sort-Object Length | Select-Object -First 1
    if ($m) { return $m.FullName } else { return $null }
}

# Returns $true only if llama-cpp-python can actually load a model here.
function Test-LlamaLoads {
    $model = Get-SmokeModel
    if (-not $model) {
        # NOTE: with no .gguf present (e.g. a fresh "lite" copy before any model
        # has downloaded) we can only verify the import. The AVX-512 illegal-
        # instruction crash happens on MODEL LOAD, not import, so this check can
        # pass while a real load later still fails. Once a model downloads, the
        # next launch (new machine marker) re-runs this with a real load test.
        Write-Host "  No .gguf model present yet; testing import only."
        & $Py -c "import llama_cpp" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    Write-Host "  Smoke-testing model load: $model"
    & $Py -c "import sys; from llama_cpp import Llama; Llama(model_path=sys.argv[1], n_ctx=512, verbose=False); print('LOAD_OK')" $model 2>$null
    return ($LASTEXITCODE -eq 0)
}

# Locate a Visual Studio install that includes the C++ x64 toolset.
function Find-VsPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($p) { return $p }
    }
    return $null
}

# Newest reusable wheel already built/shipped for this CPU profile, or $null.
function Get-CachedWheel {
    if (Test-Path $WheelDir) {
        return Get-ChildItem $WheelDir -Filter "llama_cpp_python-$LlamaVersion-*.whl" -File -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending | Select-Object -First 1
    }
    return $null
}

# Force-reinstall a local wheel into the venv (no deps). Returns $true on success.
function Install-Wheel($whl) {
    & $Py -m pip install --force-reinstall --no-deps --no-cache-dir $whl.FullName
    return ($LASTEXITCODE -eq 0)
}

# --- Detect CPU SIMD support (decides build flags + wheel profile) ----------
# Single-line -c with no inner double quotes so it survives PowerShell 5.1 native
# argument quoting (a multi-line f-string here-string does NOT). AVX2 = feature 40.
Write-Step "Detecting CPU instruction-set support"
$cpuProfile = (& $Py -c "import ctypes;print('avx2' if ctypes.windll.kernel32.IsProcessorFeaturePresent(40) else 'sse')").Trim()
$hasAVX2 = ($cpuProfile -eq 'avx2')
$WheelDir = Join-Path $Root "wheels\$cpuProfile"
Write-Host "  CPU profile: $cpuProfile  (AVX2=$hasAVX2; wheel cache: wheels\$cpuProfile)"

# numpy<2.5 keeps numba happy; quick no-op when already satisfied.
Write-Step "Ensuring numpy < 2.5 (numba compatibility)"
& $Py -m pip install "numpy<2.5" --no-cache-dir --quiet

# ============================================================================
# VERIFY (skipped only for -ForceBuild)
# ============================================================================
if (-not $ForceBuild) {
    Write-Step "Testing current llama-cpp-python"
    if (Test-LlamaLoads) { Complete-Healthy }

    $cached = Get-CachedWheel
    if ($cached) {
        Write-Step "Reusing prebuilt wheel for this CPU ($cpuProfile): $($cached.Name)"
        if ((Install-Wheel $cached) -and (Test-LlamaLoads)) { Complete-Healthy }
    }

    Write-Step "Clean reinstall of the prebuilt CPU wheel"
    & $Py -m pip install "llama-cpp-python==$LlamaVersion" --force-reinstall --no-cache-dir --no-deps `
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python
    if (Test-LlamaLoads) { Complete-Healthy }

    if ($VerifyOnly) {
        Write-Host "`nThe local AI engine could not be enabled without building." -ForegroundColor Yellow
        Write-Host "Run build_wheel.bat to compile a wheel for this machine (needs C++ build tools)."
        exit 1
    }
    Write-Host "Prebuilt wheel still fails -> building from source for this CPU." -ForegroundColor Yellow
}

# ============================================================================
# BUILD (for -ForceBuild, or the default fall-through)
# ============================================================================
Write-Step "Locating Visual Studio C++ build tools"
$vsPath = Find-VsPath
if (-not $vsPath) {
    Write-Host "Visual Studio C++ Build Tools are required to build the engine for this CPU." -ForegroundColor Yellow
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($Unattended) {
        Write-Host "Unattended mode: skipping the interactive Build Tools install." -ForegroundColor Yellow
        Write-Host "To enable the local LLM later, run build_wheel.bat and accept the install,"
        Write-Host "or install the tools yourself:"
        Write-Host '  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    }
    elseif ($winget) {
        $answer = Read-Host "Install them now automatically with winget? (large download, may prompt for admin) [Y/N]"
        if ($answer -match '^(y|yes)$') {
            Write-Step "Installing Visual Studio Build Tools (C++ workload) - this can take a while"
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
                --accept-source-agreements --accept-package-agreements `
                --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            $vsPath = Find-VsPath
        } else {
            Write-Host "Skipped automatic install." -ForegroundColor Yellow
        }
    } else {
        Write-Host "winget is not available here, so it cannot be installed automatically." -ForegroundColor Yellow
    }
}
if (-not $vsPath) {
    Write-Host "ERROR: Visual Studio C++ Build Tools not found - cannot build." -ForegroundColor Red
    Write-Host "Install them, then re-run build_wheel.bat:"
    Write-Host '  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    Write-Host "  (or download https://aka.ms/vs/17/release/vs_BuildTools.exe and pick 'Desktop development with C++')"
    exit 1
}
Write-Host "  Found: $vsPath"
try {
    Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
    Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
} catch {
    Write-Host "ERROR: could not initialise the VS build environment: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Build a REUSABLE wheel into wheels\<profile> with safe SIMD flags. -v streams
# the cmake/ninja/compiler progress to the console (instead of a silent spinner).
Write-Step "Building llama-cpp-python $LlamaVersion for this CPU ($cpuProfile) - a few minutes"
if ($hasAVX2) {
    $env:CMAKE_ARGS = "-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=OFF"
    Write-Host "  Building with AVX2 (AVX-512 disabled)."
} else {
    $env:CMAKE_ARGS = "-DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX512=OFF"
    Write-Host "  No AVX2 detected -> SSE-only build (slower but safe)."
}
$env:FORCE_CMAKE = "1"
New-Item -ItemType Directory -Force $WheelDir | Out-Null
& $Py -m pip wheel "llama-cpp-python==$LlamaVersion" --no-binary llama-cpp-python --no-deps --no-cache-dir -v -w $WheelDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: source build failed (see the log above)." -ForegroundColor Red
    exit 1
}

$whl = Get-CachedWheel
if (-not $whl) {
    Write-Host "ERROR: build finished but no wheel was found in $WheelDir." -ForegroundColor Red
    exit 1
}
Write-Step "Installing the freshly built wheel: $($whl.Name)"
if (-not (Install-Wheel $whl)) {
    Write-Host "ERROR: installing the built wheel failed." -ForegroundColor Red
    exit 1
}

Write-Step "Verifying the freshly built engine"
if (Test-LlamaLoads) {
    Write-Host "Wheel is reusable on any $cpuProfile machine: $($whl.FullName)"
    Complete-Healthy
}

Write-Host "FAILED: the engine still cannot load after building." -ForegroundColor Red
exit 1
