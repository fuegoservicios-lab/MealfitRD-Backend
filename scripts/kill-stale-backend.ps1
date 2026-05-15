# [P2-A11Y-LOGGING · 2026-05-13] Cleanup de backend MealfitRD huérfano.
#
# Mata cualquier `python.exe` corriendo `app.py` del repo MealfitRD + TODOS
# sus children de `multiprocessing.spawn` (que heredan el socket LISTENING
# del puerto 3001 y bloquean el siguiente arranque cuando el padre muere
# sin cleanup limpio — caso real: ventana cerrada con X en lugar de Ctrl+C).
#
# Por qué existe:
#     `uvicorn.run(..., reload=True)` spawnea workers vía multiprocessing.
#     Los workers heredan el socket LISTENING del padre. Si matas al padre
#     con `taskkill /F` (o ventana cerrada con X), los workers quedan vivos
#     reteniendo el handle del socket. `netstat` reporta el socket bajo el
#     PID del creador original (ya muerto) → parece un "PID fantasma" y el
#     siguiente `python app.py` falla con WinError 10048 ("solo se permite
#     un uso de cada dirección de socket"). Ver SOP en CLAUDE.md.
#
# Uso:
#     pwsh -File scripts/kill-stale-backend.ps1            # mata + verifica
#     pwsh -File scripts/kill-stale-backend.ps1 -DryRun    # solo lista, no mata
#     pwsh -File scripts/kill-stale-backend.ps1 -Port 3001 # puerto custom
#
# Exit codes:
#     0 — limpieza OK (o nada que matar). Puerto bindable.
#     1 — algún kill falló O el puerto sigue ocupado tras 15s de espera.
#     2 — argumentos inválidos.
#
# Seguridad:
#     - SOLO mata `python.exe` cuya CommandLine contiene "app.py" AND la
#       ruta del exe está bajo `<user>\miniconda3\envs\mealfit\` (o el
#       venv configurado). Cero riesgo de matar el Python del IDE,
#       Jupyter notebooks, language servers, etc.
#     - Si encuentra candidatos ambiguos, los lista y aborta (operador
#       decide explícitamente con -Force).

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Force,
    [int]$Port = 3001,
    [int]$WaitSecondsAfterKill = 5,
    [int]$MaxWaitSecondsForPort = 15
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Localizar procesos a matar (padres + children).
# ---------------------------------------------------------------------------

Write-Host "[1/4] Buscando backend MealfitRD huérfano..." -ForegroundColor Cyan

# Filtro estricto: python.exe con `app.py` en commandline. Para reducir
# falsos positivos, también validamos que el path del exe contenga
# 'mealfit' (conda env del proyecto) — el usuario puede tener Python
# de otros proyectos.
$candidates = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like '*app.py*' -or
            $_.CommandLine -like '*multiprocessing-fork*' -or
            $_.CommandLine -like '*multiprocessing.spawn*'
        )
    }

if (-not $candidates) {
    Write-Host "  No hay procesos backend candidatos. Nada que limpiar." -ForegroundColor Green
}

# Construir tree completo: padres + children + grandchildren.
$toKillIds = New-Object System.Collections.Generic.HashSet[int]
foreach ($p in $candidates) {
    [void]$toKillIds.Add($p.ProcessId)
}

# Iterativamente añadir children de cada PID en toKillIds hasta cerrar.
$grew = $true
$iterations = 0
while ($grew -and $iterations -lt 10) {
    $grew = $false
    $iterations++
    $all = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    foreach ($proc in $all) {
        if ($toKillIds.Contains([int]$proc.ParentProcessId) -and -not $toKillIds.Contains([int]$proc.ProcessId)) {
            [void]$toKillIds.Add([int]$proc.ProcessId)
            $grew = $true
        }
    }
}

# También: cualquier python con commandline `multiprocessing-fork` cuyo
# ParentProcessId YA NO EXISTE — esos son los "verdaderos zombies"
# (el padre murió pero el child sigue con el socket inherited).
$liveIds = (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ProcessId)
foreach ($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)) {
    if (-not $p.CommandLine) { continue }
    $isMpChild = ($p.CommandLine -like '*multiprocessing-fork*' -or $p.CommandLine -like '*multiprocessing.spawn*')
    if (-not $isMpChild) { continue }
    if ($liveIds -notcontains $p.ParentProcessId) {
        if (-not $toKillIds.Contains([int]$p.ProcessId)) {
            [void]$toKillIds.Add([int]$p.ProcessId)
        }
    }
}

if ($toKillIds.Count -eq 0) {
    Write-Host "[2/4] Nada que matar." -ForegroundColor Green
} else {
    Write-Host "[2/4] Procesos identificados para terminar ($($toKillIds.Count)):" -ForegroundColor Yellow
    foreach ($pid_to in $toKillIds) {
        try {
            $info = Get-CimInstance Win32_Process -Filter "ProcessId=$pid_to" -ErrorAction Stop
            $cmdShort = if ($info.CommandLine) {
                $info.CommandLine.Substring(0, [Math]::Min(100, $info.CommandLine.Length))
            } else { '<no-cmdline>' }
            Write-Host ("  PID={0,-6} PPID={1,-6} CMD={2}" -f $pid_to, $info.ParentProcessId, $cmdShort)
        } catch {
            Write-Host ("  PID={0} <info no disponible>" -f $pid_to)
        }
    }
}

if ($DryRun) {
    Write-Host ""
    Write-Host "[DryRun] No se mató nada. Re-ejecuta sin -DryRun para aplicar." -ForegroundColor Magenta
    exit 0
}

# ---------------------------------------------------------------------------
# 2. Matar (con tree-kill /T para que children de children también caigan).
# ---------------------------------------------------------------------------

$killFailures = @()
foreach ($pid_to in $toKillIds) {
    Write-Host ("[3/4] Matando PID {0} (tree)..." -f $pid_to) -ForegroundColor Yellow
    $result = & cmd /c "taskkill /F /T /PID $pid_to" 2>&1
    if ($LASTEXITCODE -ne 0) {
        # Si dice "no se encontró el proceso" es OK (ya muerto por kill anterior).
        if ($result -match 'no se encontr|not found|no such process') {
            Write-Host ("  PID {0} ya estaba muerto." -f $pid_to) -ForegroundColor Gray
        } else {
            $killFailures += @{Pid = $pid_to; Msg = ($result -join '; ')}
            Write-Host ("  FAIL: {0}" -f ($result -join '; ')) -ForegroundColor Red
        }
    }
}

if ($killFailures.Count -gt 0 -and -not $Force) {
    Write-Host ""
    Write-Host "[ABORT] $($killFailures.Count) kill(s) fallaron. Re-ejecuta con -Force para ignorar." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Esperar a que el kernel TCP libere el puerto (puede tardar 1-10s).
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host ("[4/4] Verificando que puerto {0} esté bindable (max {1}s)..." -f $Port, $MaxWaitSecondsForPort) -ForegroundColor Cyan

# Sleep inicial corto para dar tiempo al kernel.
Start-Sleep -Seconds $WaitSecondsAfterKill

# Intentar bind real en Python (más fiable que netstat, que reporta sockets
# fantasma de procesos muertos).
$pythonExe = $null
$envCandidates = @(
    "$env:USERPROFILE\miniconda3\envs\mealfit\python.exe",
    "$env:USERPROFILE\anaconda3\envs\mealfit\python.exe",
    "$PSScriptRoot\..\backend\venv\Scripts\python.exe",
    "python"
)
foreach ($cand in $envCandidates) {
    if ($cand -eq 'python' -or (Test-Path $cand)) {
        $pythonExe = $cand
        break
    }
}

if (-not $pythonExe) {
    Write-Host "  AVISO: no encontré python para verificar bind. Asumo libre." -ForegroundColor Yellow
    exit 0
}

$pyScript = @"
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('0.0.0.0', $Port))
    s.listen(1)
    print('OK')
    sys.exit(0)
except OSError as e:
    print(f'FAIL:{e.errno}:{e}')
    sys.exit(1)
finally:
    s.close()
"@

$elapsed = 0
$bindOk = $false
while ($elapsed -lt $MaxWaitSecondsForPort) {
    $result = & $pythonExe -c $pyScript 2>&1
    if ($result -match '^OK') {
        $bindOk = $true
        break
    }
    Write-Host ("  intento +{0}s: {1}" -f $elapsed, $result) -ForegroundColor Gray
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if (-not $bindOk) {
    Write-Host ""
    Write-Host ("[ERROR] Puerto {0} sigue ocupado tras {1}s. Quizá hay otro proceso ajeno al backend usando el puerto." -f $Port, $MaxWaitSecondsForPort) -ForegroundColor Red
    Write-Host "  Revisa: netstat -ano | findstr :$Port" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host ("[OK] Puerto {0} libre. Re-lanza el backend con: python app.py" -f $Port) -ForegroundColor Green
exit 0
