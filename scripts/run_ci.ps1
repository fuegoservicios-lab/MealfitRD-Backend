# [P1-LIVE-2 · 2026-05-12] Wrapper local del CI gate (PowerShell).
#
# Reproduce los 3 jobs de .github/workflows/ci.yml en el entorno local:
#   1. pytest del bundle parser-based + funcional (excluyendo e2e).
#   2. vitest del frontend.
#   3. vite build production.
#
# Uso:
#   pwsh -File scripts/run_ci.ps1
#   pwsh -File scripts/run_ci.ps1 -SkipBackend     # solo frontend
#   pwsh -File scripts/run_ci.ps1 -SkipFrontend    # solo backend
#   pwsh -File scripts/run_ci.ps1 -SkipBuild       # tests sin build prod
#
# Exit code:
#   0 si los 3 jobs (no-skipped) pasaron, 1 si alguno falló.
#
# Recomendado: invocar antes de cada `git push` (manual o vía hook
# pre-push tras `git init`).

param(
    [switch]$SkipBackend,
    [switch]$SkipFrontend,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$failed = @()

function Run-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    try {
        & $Block
        if ($LASTEXITCODE -ne 0) {
            $script:failed += $Label
            Write-Host "    FAIL ($Label) exit=$LASTEXITCODE" -ForegroundColor Red
        } else {
            Write-Host "    PASS ($Label)" -ForegroundColor Green
        }
    } catch {
        $script:failed += $Label
        Write-Host "    FAIL ($Label): $_" -ForegroundColor Red
    }
}

if (-not $SkipBackend) {
    Run-Step "Backend pytest" {
        Push-Location "$repoRoot/backend"
        try {
            $py = if (Test-Path "venv/bin/python.exe") { "venv/bin/python.exe" }
                  elseif (Test-Path "venv/Scripts/python.exe") { "venv/Scripts/python.exe" }
                  else { "python" }
            & $py -m pytest tests/ -v --tb=short -m "not e2e" -x
        } finally {
            Pop-Location
        }
    }
}

if (-not $SkipFrontend) {
    Run-Step "Frontend vitest" {
        Push-Location "$repoRoot/frontend"
        try {
            npm test
        } finally {
            Pop-Location
        }
    }
}

if (-not $SkipBuild) {
    Run-Step "Frontend vite build" {
        Push-Location "$repoRoot/frontend"
        try {
            npm run build
        } finally {
            Pop-Location
        }
    }
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "All CI jobs PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "CI FAIL on: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
