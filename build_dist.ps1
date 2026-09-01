<#
    build_dist.ps1 - package TicketLens into a distributable zip.

    Replaces a hand-rolled manual zip. The version comes from src/__init__.py (the
    single source of truth, the same value run.bat prints and the About dialog shows),
    so the archive name can never disagree with the app's reported version.

    NOTE: keep this file pure ASCII. Windows PowerShell 5.1 reads .ps1 as ANSI when
    there is no BOM, so non-ASCII punctuation (em dashes, arrows) breaks parsing.

    Usage:
        powershell -ExecutionPolicy Bypass -File build_dist.ps1
        powershell -ExecutionPolicy Bypass -File build_dist.ps1 -OutDir "C:\somewhere"
#>
[CmdletBinding()]
param(
    # Where to write the zip. Defaults to the repo's parent directory so the archive
    # never lands inside the tree it is packaging.
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
if (-not $OutDir) { $OutDir = Split-Path $repo -Parent }

# --- version (single source of truth) ---------------------------------------
$initPath = Join-Path $repo "src\__init__.py"
if (-not (Test-Path $initPath)) { throw "Cannot find $initPath" }
$verLine = Select-String -Path $initPath -Pattern '^__version__' | Select-Object -First 1
if (-not $verLine) { throw "No __version__ found in $initPath" }
$version = ($verLine.Line -split '=', 2)[1].Trim().Trim('"').Trim("'")
if (-not $version) { throw "Could not parse a version from: $($verLine.Line)" }
Write-Host "TicketLens version: $version"

$zipName = "TicketLens-v$version.zip"
$zipPath = Join-Path $OutDir $zipName

# --- what to leave out ------------------------------------------------------
# Everything here is either recreated on the target machine at first run, or is
# local/developer state that must not ship.
$excludeDirs  = @('.venv', '.git', '.git-rewrite', '.pytest_cache', '.pytest_tmp',
                  '.vscode', '.idea', '.claude', '.github', 'cache', 'logs', 'models',
                  '__pycache__')
# .tsz = a saved session, which embeds the ticket frame -- same reason .xlsx is here.
$excludeExt   = @('.pyc', '.pyo', '.xlsx', '.xls', '.csv', '.zip', '.gguf', '.npy',
                  '.tsz')
# Per-user state: gitignored and not shipped. config/user_settings.json carries
# developer overrides (model path, cluster size, stopwords) that must not be
# silently inherited by every recipient of the distributable.
$excludeFiles = @('config\user_settings.json', 'config\custom_stopwords.txt')

$files = Get-ChildItem -Path $repo -Recurse -File | Where-Object {
    $rel   = $_.FullName.Substring($repo.Length).TrimStart('\')
    $parts = ($rel -split '\\')
    # Drop the filename so a *file* named like an excluded dir is not skipped.
    $dirs  = if ($parts.Count -gt 1) { $parts[0..($parts.Count - 2)] } else { @() }
    $inExcludedDir = $false
    foreach ($d in $dirs) { if ($excludeDirs -contains $d) { $inExcludedDir = $true; break } }
    (-not $inExcludedDir) -and ($excludeExt -notcontains $_.Extension.ToLower()) `
        -and ($excludeFiles -notcontains $rel)
}

if (-not $files) { throw "No files matched, refusing to build an empty archive." }

if (Test-Path $zipPath) {
    Write-Host "Replacing existing $zipName"
    Remove-Item $zipPath -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')
try {
    foreach ($f in $files) {
        $entry = $f.FullName.Substring($repo.Length).TrimStart('\').Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $f.FullName, $entry, 'Optimal') | Out-Null
    }
} finally {
    $zip.Dispose()
}

# --- report + sanity checks -------------------------------------------------
$sizeMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
$fileCount = $files.Count
Write-Host ""
Write-Host "Built $zipName"
Write-Host "  size  : $sizeMb MB"
Write-Host "  files : $fileCount"
Write-Host "  path  : $zipPath"

$read = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $names = $read.Entries | ForEach-Object { $_.FullName }
    # These must be present or the distributable simply will not run.
    # src/ui/* is the presentation layer gui.py imports at module scope, so a
    # missing file there is an ImportError on launch, not a degraded feature.
    # LICENSE and NOTICE are an Apache-2.0 redistribution obligation: shipping the
    # zip without them is a licensing defect, so fail the build rather than warn.
    $required = @('main.py', 'run.bat', 'run.sh', 'src/gui.py', 'src/__init__.py',
                  'requirements.txt', 'assets/check.svg',
                  'LICENSE', 'NOTICE', 'README.md',
                  'src/ui/__init__.py', 'src/ui/theme.py', 'src/ui/widgets.py',
                  'src/ui/dialogs.py', 'src/ui/pages.py')
    foreach ($r in $required) {
        if ($names -notcontains $r) { throw "MISSING from the archive: $r" }
    }
    $svgs = @($names | Where-Object { $_ -like '*.svg' })
    Write-Host "  assets: $($svgs.Count) svg file(s)"

    $verEntry = $read.Entries | Where-Object { $_.FullName -eq 'src/__init__.py' }
    if ($verEntry) {
        $sr = New-Object System.IO.StreamReader($verEntry.Open())
        try { $content = $sr.ReadToEnd() } finally { $sr.Dispose() }
        if ($content -notmatch [regex]::Escape($version)) {
            throw "Archived src/__init__.py does not report version $version"
        }
        Write-Host "  version: archived src/__init__.py reports $version"
    }
    # Per-user state must never ship: leaks developer settings or data paths.
    $banned = @('config/user_settings.json', 'config/custom_stopwords.txt')
    foreach ($b in $banned) {
        if ($names -contains $b) { throw "MUST NOT be in the archive: $b" }
    }
    # Editor / agent tooling state is not part of the app and grows over time.
    $tooling = @($names | Where-Object { $_ -like '.claude/*' -or $_ -like '.vscode/*' `
                                          -or $_ -like '.github/*' })
    if ($tooling) { throw "MUST NOT be in the archive: $($tooling -join ', ')" }
} finally {
    $read.Dispose()
}
Write-Host "OK"
