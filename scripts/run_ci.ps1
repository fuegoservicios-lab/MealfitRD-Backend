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

            # [P2-CI-PYTEST-PARALLEL · 2026-08-19] El gate corria 18.9k tests EN SERIE
            # (~19-20 min por deploy; el dueno lo llamo, con razon, exagerado). Con
            # pytest-xdist a 3 workers + --dist loadfile la MISMA suite corre en ~9 min
            # (medido: 18.855 tests en 8:51). Lo aprendido midiendo, para que nadie lo
            # re-descubra a golpes:
            #   - PYTHONHASHSEED=0 es OBLIGATORIO: sin el, cada worker colecciona tests
            #     en orden distinto (parametrize sobre set/dict) y xdist aborta con
            #     "Different tests were collected".
            #   - -n 8 revienta la RAM (16 GB, cada worker importa el backend entero:
            #     MemoryError en la coleccion). 3 workers es el techo seguro medido.
            #   - --dist loadfile: los tests de este repo asumen ejecucion por-archivo.
            #   - CUARENTENA (fase serial de abajo): 2 archivos de la familia
            #     renewal/chunk matan el worker con salida LIMPIA a mitad de test
            #     (victimas, no culpables: cada archivo pasa solo bajo xdist; el veneno
            #     es estado acumulado de archivos previos en el worker — sospecha
            #     principal: presion de memoria tardia) + 3 tests paralelo-hostiles de
            #     clase conocida (identidad de objeto cross-modulo, timing de hilo de
            #     fondo bajo contencion, assert de exactamente-un-warning con logger
            #     compartido). Si un run futuro aborta con INTERNALERROR crashitem,
            #     anade ESE archivo aqui — no vuelvas a serie completa.
            #   - Sin -v: 18.9k lineas verbose a consola cuestan minutos reales.
            # Escotilla: MEALFIT_CI_PYTEST_WORKERS=1 (o 0/serial) vuelve al modo serie
            # historico completo sin tocar codigo.
            $workers = $env:MEALFIT_CI_PYTEST_WORKERS
            if (-not $workers) { $workers = "3" }
            $env:PYTHONHASHSEED = "0"

            $quarantineFiles = @(
                "tests/test_chunked_generation.py",
                "tests/test_renewal_15d.py",
                "tests/test_p1_17_purge_graph_cache.py"
            )
            $quarantineTests = @(
                "tests/test_p1_audit_hist_7_lesson_whitelist_ssot.py",
                "tests/test_p1_bg_thread_timeout.py",
                "tests/test_p2_cap_log_level.py"
            )

            if ($workers -eq "1" -or $workers -eq "0" -or $workers -eq "serial") {
                & $py -m pytest tests/ --tb=short -m "not e2e" -x
            } else {
                # FASE A - bulk paralelo SIN -x: 47 archivos de la suite stubbean
                # sys.modules (patron estructural), asi que bajo xdist CUALQUIER test
                # puede caer como victima aleatoria del veneno de un vecino de worker.
                # Cazar victimas no converge (3 corridas = 3 victimas distintas). El
                # diseno honesto: el bulk enumera, y un fallo solo tumba el gate si
                # falla TAMBIEN en el re-juicio SERIAL de la fase C (mismo estandar de
                # verdad que el gate serie historico; un fallo real falla en ambos).
                $ignoreArgs = @()
                foreach ($f in $quarantineFiles + $quarantineTests) { $ignoreArgs += "--ignore"; $ignoreArgs += $f }
                & $py -m pytest tests/ --tb=short -m "not e2e" -q `
                    -n $workers --dist loadfile --max-worker-restart=4 @ignoreArgs
                $bulkExit = $LASTEXITCODE

                # FASE B - cuarentena en serie (los 3 archivos que matan workers +
                # los paralelo-hostiles conocidos). -x: aqui un fallo es real.
                & $py -m pytest @($quarantineFiles + $quarantineTests) --tb=short -m "not e2e" -x -p no:cacheprovider
                if ($LASTEXITCODE -ne 0) { throw "pytest (cuarentena serial) fallo (exit $LASTEXITCODE)" }

                # FASE C - re-juicio serial de los fallos del bulk (si los hubo).
                # OJO: va DESPUES de la fase B en codigo pero usa el cache --lf de la
                # fase A... y la fase B ya lo piso. Por eso el orden real es: correr
                # B con -p no:cacheprovider para NO tocar el cache de A.
                if ($bulkExit -ne 0) {
                    & $py -m pytest --lf --last-failed-no-failures none --tb=short -m "not e2e" -x
                    if ($LASTEXITCODE -ne 0) {
                        throw "pytest: fallos del bulk CONFIRMADOS en serie (exit $LASTEXITCODE) - regresion real"
                    }
                }
            }
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
