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
                # [P1-CI-SERIE-INCONCLUSIVE - 2026-08-20] La misma distincion que la
                # rama paralela, que hasta hoy solo vivia alli.
                #
                # Esto era un `pytest -x` pelado, asi que TODO exit != 0 se reportaba
                # como "los tests FALLARON". El 2026-08-20 la suite murio por falta de
                # memoria (3,7 GB libres de 15,7) y el gate dijo justo eso: mande a
                # buscar un test roto que no existia y perdi una corrida de 20 minutos.
                # Un veredicto que confunde "abortó" con "hay una regresion" no es solo
                # ruido: dirige mal la investigacion.
                #
                # Solo el exit 1 significa "fallaron tests" (medido). 2 (interrumpido),
                # 3 (error interno), 4 (uso) y 5 (nada coleccionado) son NO
                # CONCLUYENTES. Aqui no hay a que caer -- la serie YA es el modo mas
                # conservador -- asi que se reintenta una vez y, si insiste, se falla
                # DICIENDO QUE ABORTO. Fallar sigue siendo lo correcto: nunca desplegar
                # sin veredicto. Lo que cambia es que el operador sepa que buscar.
                $serieExit = -1
                $serieConcluyente = $false
                foreach ($intento in 1, 2) {
                    & $py -m pytest tests/ --tb=short -m "not e2e" -x | Tee-Object -Variable serieCap
                    $serieExit = $LASTEXITCODE
                    $serieTexto = ($serieCap | Out-String)
                    $serieConcluyente = (($serieExit -eq 0) -or ($serieExit -eq 1)) -and ($serieTexto -notmatch "INTERNALERROR")
                    if ($serieConcluyente) { break }
                    Write-Host "    [gate] SERIE NO CONCLUYENTE (exit $serieExit) - intento $intento de 2" -ForegroundColor Yellow
                }
                if (-not $serieConcluyente) {
                    throw ("pytest (serie) NO CONCLUYENTE tras 2 intentos (exit $serieExit): " +
                           "la suite ABORTO, no fallaron tests. No busques una regresion todavia. " +
                           "Causa tipica en esta maquina: memoria (la serie necesita ~2,5 GB; " +
                           "con el editor y el navegador abiertos quedan ~3,7 GB de 15,7). " +
                           "Cierra aplicaciones y reintenta.")
                }
                if ($serieExit -ne 0) {
                    throw "pytest (serie) fallo (exit $serieExit) - regresion real"
                }
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

                # [P1-CI-GATE-INCONCLUSIVE - 2026-08-19] El diseno de 3 fases trataba
                # "exit != 0" como "el bulk enumero fallos". No siempre: xdist aborta
                # con INTERNALERROR (KeyError: <WorkerController gw0>, visto 3 veces el
                # 2026-08-19) y ahi la sesion NO termina, asi que cacheprovider no
                # escribe el cache y `--lf` se queda VACIO o RANCIO.
                #
                # Medido, no supuesto: con el cache vacio,
                #   pytest --lf --last-failed-no-failures none
                # DESELECCIONA TODO y sale 0. O sea que la fase C daba verde sin
                # ejecutar un solo test, y el gate del deploy reportaba PASS habiendo
                # abortado la suite. Un falso verde en la puerta de produccion es peor
                # que el ruido que este bloque venia a evitar.
                #
                # La senal limpia es el CODIGO DE SALIDA, y tampoco hacia falta
                # adivinarlo: pytest usa 1 para "hubo fallos" y 3 para "error interno"
                # (medido). Solo el 1 deja una lista de fallos en la que confiar; 2
                # (interrumpido), 3 (interno), 4 (uso) y 5 (nada coleccionado) son
                # veredictos NO CONCLUYENTES. El texto INTERNALERROR se comprueba
                # ademas por si una version futura saliera con 1 tras abortar.
                #
                # No concluyente => un reintento (el aborto suele ser transitorio) y,
                # si vuelve a abortar, SERIE COMPLETA. Nunca un pase. El caso
                # patologico cuesta tiempo; un deploy verde sin suite cuesta produccion.
                # Se captura a VARIABLE y no a fichero: `Tee-Object -FilePath` sobre una
                # ruta fija de %TEMP% se choco consigo mismo ("el proceso no puede acceder
                # al archivo"), y una ruta compartida entre corridas es una clase de bug
                # gratuita en un gate. `-Variable` sigue mostrando la salida en vivo -- que
                # es el motivo de usar Tee y no una redireccion -- y preserva $LASTEXITCODE.
                $bulkExit = -1
                $bulkConcluyente = $false
                foreach ($intento in 1, 2) {
                    & $py -m pytest tests/ --tb=short -m "not e2e" -q `
                        -n $workers --dist loadfile --max-worker-restart=4 @ignoreArgs |
                        Tee-Object -Variable bulkCap
                    $bulkExit = $LASTEXITCODE
                    $bulkTexto = ($bulkCap | Out-String)
                    $bulkConcluyente = (($bulkExit -eq 0) -or ($bulkExit -eq 1)) -and ($bulkTexto -notmatch "INTERNALERROR")
                    if ($bulkConcluyente) { break }
                    Write-Host "    [gate] FASE A NO CONCLUYENTE (exit $bulkExit) - intento $intento de 2" -ForegroundColor Yellow
                }

                # FASE B - cuarentena en serie (los 3 archivos que matan workers +
                # los paralelo-hostiles conocidos). -x: aqui un fallo es real.
                & $py -m pytest @($quarantineFiles + $quarantineTests) --tb=short -m "not e2e" -x -p no:cacheprovider
                if ($LASTEXITCODE -ne 0) { throw "pytest (cuarentena serial) fallo (exit $LASTEXITCODE)" }

                # FASE C - re-juicio serial de los fallos del bulk (si los hubo).
                # OJO: va DESPUES de la fase B en codigo pero usa el cache --lf de la
                # fase A... y la fase B ya lo piso. Por eso el orden real es: correr
                # B con -p no:cacheprovider para NO tocar el cache de A.
                $serieCompleta = -not $bulkConcluyente
                if ($bulkConcluyente -and ($bulkExit -eq 1)) {
                    & $py -m pytest --lf --last-failed-no-failures none --tb=short -m "not e2e" -x |
                        Tee-Object -Variable lfCap
                    if ($LASTEXITCODE -ne 0) {
                        throw "pytest: fallos del bulk CONFIRMADOS en serie (exit $LASTEXITCODE) - regresion real"
                    }
                    # Segundo cinturon: el bulk dijo "hubo fallos" pero el re-juicio no
                    # ejecuto NADA => el cache estaba vacio o rancio. Salir verde de aqui
                    # seria el mismo falso verde descrito arriba, sin INTERNALERROR de
                    # por medio.
                    $lfTexto = ($lfCap | Out-String)
                    if ($lfTexto -notmatch "\d+\s+(passed|failed|error)") {
                        Write-Host "    [gate] FASE C no re-juzgo NADA (cache --lf vacio o rancio)" -ForegroundColor Yellow
                        $serieCompleta = $true
                    }
                }

                if ($serieCompleta) {
                    Write-Host "    [gate] cayendo a la SERIE COMPLETA: el bulk no dejo un veredicto fiable." -ForegroundColor Yellow
                    & $py -m pytest tests/ --tb=short -m "not e2e" -x
                    if ($LASTEXITCODE -ne 0) { throw "pytest (serie completa tras bulk no concluyente) fallo (exit $LASTEXITCODE)" }
                }
            }
        } finally {
            Pop-Location
        }
    }
}

if (-not $SkipFrontend) {
    # [P1-CI-I18N-GATE - 2026-08-20] El chequeo de catalogos NO estaba en el gate.
    #
    # El 2026-08-20 el dueno reporto OCHO superficies distintas en espanol con la app
    # en ingles: dias de la semana, fecha del modal, titulo del plan, pestanas, fecha
    # de la tarjeta, slots de Recetas, submenu "Mas informacion", nombres de plan y el
    # splash. Todas llegaron por CAPTURA, ninguna por CI -- porque `i18n:check` solo
    # corria cuando alguien se acordaba de escribirlo.
    #
    # Va ANTES de vitest a proposito: tarda ~2 s y su fallo es de una linea, mientras
    # que la suite tarda ~2 min. Fallar rapido y con la causa a la vista.
    #
    # EN MODO ESTRICTO, y esa es la decision que importa. MEDIDO quitando una traduccion
    # de `en-US.json`:
    #
    #     npm run i18n:check          -> exit 0     <- no habria cazado NADA de hoy
    #     npm run i18n:check:strict   -> exit 1
    #
    # Sin `--strict` una clave sin traduccion NO tumba nada: el texto cae al espanol y
    # la pantalla queda a medias en silencio -- literalmente la forma de todos los
    # reportes de hoy. Anadir el paso en permisivo habria sido teatro: un guard que da
    # verde justo en el caso que motivo ponerlo. El repo esta hoy
    # al 100% en los 4 idiomas (0 huerfanas, 0 faltantes), asi que encenderlo no cuesta
    # deuda: solo obliga a que una cadena nueva nazca traducida.
    #
    # Escotilla sin editar codigo (convencion del repo): MEALFIT_CI_I18N_STRICT=0 baja
    # a modo permisivo. Sirve para una tanda larga a medio traducir; no para desplegar.
    Run-Step "Frontend i18n" {
        Push-Location "$repoRoot/frontend"
        try {
            $strict = $env:MEALFIT_CI_I18N_STRICT
            if ($strict -eq "0" -or $strict -eq "false") {
                Write-Host "    [gate] i18n en modo PERMISIVO (MEALFIT_CI_I18N_STRICT=$strict)" -ForegroundColor Yellow
                npm run i18n:check
            } else {
                npm run i18n:check:strict
            }
        } finally {
            Pop-Location
        }
    }

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
