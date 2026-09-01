<#
    check.ps1 - run the checks that must pass before building a distributable.

    One command to run before opening a PR or running build_dist.ps1. GitHub Actions
    (.github/workflows/ci.yml) runs these same two checks on every push and PR, so a
    green run here should mean a green run there.

    NOTE: keep this file pure ASCII. Windows PowerShell 5.1 reads .ps1 as ANSI when
    there is no BOM, so non-ASCII punctuation breaks parsing.

    Usage:
        powershell -ExecutionPolicy Bypass -File check.ps1
        powershell -ExecutionPolicy Bypass -File check.ps1 -SkipLint
#>
[CmdletBinding()]
param(
    [switch]$SkipLint,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

# Prefer the project venv, so this checks what run.bat actually ships.
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
if (Test-Path $venvPy) { $py = $venvPy } else { $py = "python" }
Write-Host "Interpreter: $py"
& $py --version

$failures = @()

# --- lint ------------------------------------------------------------------
if (-not $SkipLint) {
    Write-Host ""
    Write-Host "== ruff =="
    & $py -m ruff check src tests main.py --output-format concise
    if ($LASTEXITCODE -ne 0) {
        $failures += "ruff"
    } else {
        Write-Host "ruff: clean"
    }
} else {
    Write-Host "ruff: skipped"
}

# --- tests -----------------------------------------------------------------
if (-not $SkipTests) {
    Write-Host ""
    Write-Host "== pytest =="
    # pytest.ini pins --basetemp, so the exit code is trustworthy here. Without it,
    # teardown hit a PermissionError on the pytest-current symlink and returned 1 on a
    # fully passing suite.
    & $py -m pytest tests -q
    if ($LASTEXITCODE -ne 0) {
        $failures += "pytest"
    } else {
        Write-Host "pytest: passed"
    }
} else {
    Write-Host "pytest: skipped"
}

# --- report ----------------------------------------------------------------
Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "FAILED: $($failures -join ', ')"
    exit 1
}
Write-Host "All checks passed. Safe to run build_dist.ps1."
