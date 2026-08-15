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

# [P2-CI-PYTHON-PROBE · 2026-08-15] Elegir el interprete PROBANDOLO, no viendo si
# el fichero existe.
#
# LA CASCADA ANTERIOR ERA `venv/bin/python.exe` -> `venv/Scripts/python.exe` ->
# `python`, y se quedaba en la PRIMERA que existiera. En esta maquina existe
# `backend/venv/bin/python.exe` (Python 3.12.11) y tiene `pytest`... pero NO tiene
# `fastapi`, `langgraph`, `psycopg` ni `pydantic`: es un venv a medio provisionar.
# Con el, la suite no falla -- ni siquiera colecciona.
#
# Eso importa mas de lo que parece, porque `deploy-mealfit.ps1` documenta que el
# backend esta fuera del gate por tener "baseline roja: 43 fallos". Medida el
# 2026-08-14 con el entorno REAL (conda `mealfit`): 17.898 passed, 0 failed. O sea
# que parte de esa "baseline roja" podia ser el interprete equivocado y no tests
# malos. No lo afirmo del todo -- no reproduje la cifra de 43 -- pero mientras la
# eleccion sea "el primer fichero que exista", el gate no puede distinguir "los
# tests fallan" de "este python no tiene las dependencias", y esa ambiguedad es la
# que convierte una deuda en permanente.
#
# Ahora cada candidato se PRUEBA con un import barato antes de aceptarlo, y si
# ninguno sirve el paso falla RUIDOSAMENTE diciendo que configures MEALFIT_PYTHON
# -- en vez de correr con uno roto y culpar a los tests.
function Resolve-BackendPython {
    param([string]$BackendDir)

    $candidatos = @()
    # Escotilla explicita: gana siempre. Para CI, contenedores y maquinas raras.
    if ($env:MEALFIT_PYTHON) { $candidatos += $env:MEALFIT_PYTHON }
    $candidatos += @(
        (Join-Path $BackendDir "venv/Scripts/python.exe"),   # venv de Windows
        (Join-Path $BackendDir "venv/bin/python.exe"),       # venv "unix" en Windows
        (Join-Path $BackendDir "venv/bin/python"),
        "python",
        "python3"
    )
    # Entornos conda del usuario (este repo usa `mealfit`, ver la memoria del proyecto).
    foreach ($raiz in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\miniforge3", "$env:USERPROFILE\anaconda3")) {
        $candidatos += (Join-Path $raiz "envs\mealfit\python.exe")
    }

    $descartados = @()
    foreach ($c in $candidatos) {
        if (-not $c) { continue }

        # Resolver ANTES de invocar. Con `$ErrorActionPreference = "Stop"` (arriba),
        # invocar una ruta inexistente es un error TERMINANTE que ni `*> $null`
        # silencia: el fallo ocurre al resolver el comando, no en los flujos del
        # proceso hijo. Sin esto, la primera ruta ausente aborta la busqueda entera.
        $exe = $null
        if ($c -match '[\\/]') {
            if (Test-Path -LiteralPath $c) { $exe = $c }
        } else {
            $exe = (Get-Command $c -CommandType Application -ErrorAction SilentlyContinue |
                    Select-Object -First 1).Source
        }
        if (-not $exe) { $descartados += "$c (no existe)"; continue }

        # La sonda: un import que solo pasa si el entorno esta provisionado de verdad.
        # `pytest` NO sirve de sonda por si solo -- el venv a medias lo tiene, y es
        # exactamente por eso que la cascada vieja lo daba por bueno.
        $global:LASTEXITCODE = 0
        try { & $exe -c "import fastapi, pydantic, pytest" *> $null } catch { $global:LASTEXITCODE = 1 }
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    python: $exe" -ForegroundColor DarkGray
            return $exe
        }
        $descartados += "$exe (sin fastapi/pydantic/pytest)"
    }

    throw ("No encuentro un Python capaz de correr la suite del backend.`n" +
           "Probados (ninguno pudo importar fastapi+pydantic+pytest):`n  " +
           ($descartados -join "`n  ") + "`n`n" +
           "Configura la escotilla:  `$env:MEALFIT_PYTHON = 'C:\ruta\a\python.exe'")
}

if (-not $SkipBackend) {
    Run-Step "Backend pytest" {
        Push-Location "$repoRoot/backend"
        try {
            $py = Resolve-BackendPython -BackendDir "$repoRoot/backend"
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
