# [I6 / P3-COVERAGE-HEATMAP - 2026-05-20] Wrapper PowerShell del coverage
# heatmap. Espejo de run_coverage.sh para devs en Windows sin WSL/git-bash.
#
# Uso:
#   .\scripts\run_coverage.ps1                  # html + terminal summary
#   .\scripts\run_coverage.ps1 -Term            # solo terminal (sin html)
#   .\scripts\run_coverage.ps1 -ExtraArgs "-k", "history"   # filtro pytest

param(
    [switch]$Term,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

# Check pytest-cov
$hasPytestCov = $false
try {
    & python -c "import pytest_cov" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasPytestCov = $true }
} catch {}

if (-not $hasPytestCov) {
    Write-Host "==> pytest-cov no instalado, instalando..."
    & pip install --quiet pytest-cov
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FALLO: pip install pytest-cov - intenta manualmente: pip install pytest-cov"
        exit 1
    }
}

# Report flags
if ($Term) {
    $covReportFlags = @("--cov-report=term-missing")
} else {
    $covReportFlags = @("--cov-report=term-missing:skip-covered", "--cov-report=html:htmlcov")
}

# Default filter `not e2e` si no se pasa -m
$hasMarker = $false
foreach ($a in $ExtraArgs) {
    if ($a -eq "-m") { $hasMarker = $true; break }
}
if (-not $hasMarker) {
    $ExtraArgs = $ExtraArgs + @("-m", "not e2e")
}

Write-Host "==> pytest --cov=. $($covReportFlags -join ' ') $($ExtraArgs -join ' ')"
& pytest `
    --cov=. `
    --cov-config=.coveragerc `
    @covReportFlags `
    @ExtraArgs

$rc = $LASTEXITCODE
Write-Host ""
if ($rc -eq 0 -and -not $Term) {
    Write-Host "==> Coverage report HTML: file:///$($RepoRoot.Path.Replace('\','/'))/htmlcov/index.html"
}
exit $rc
