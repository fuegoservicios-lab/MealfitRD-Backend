"""[P2-AUDIT-NEW-3 · 2026-05-12] Boot grace window para scheduler_missed_*.

Bug original (audit comprehensivo 2026-05-12):
    Tras restart del backend, APScheduler dispara EVENT_JOB_MISSED para
    TODOS los jobs cuyo `next_run_time` pasó durante el downtime. El
    listener `_scheduler_alert_listener` los upsertea ciegamente a
    `system_alerts` → burst de 15-20 alerts `scheduler_missed_*`
    triggered en <1s post-boot.

    Observado vivo en MealFitRD (2026-05-12 00:38:51 audit):
        19 alerts `scheduler_missed_*` triggered en 90 segundos, todas
        del mismo evento de restart. Los autoheals (P0-LIVE-1 hard-floor
        + EVENT_JOB_EXECUTED auto-resolve P1-NEW-2) las cierran en <5min
        cuando los jobs re-ejecutan exitosamente — pero entre tanto SRE
        recibe Sentry breadcrumbs falsos positivos y system_alerts
        acumula churn por cada deploy.

Fix:
    Knob `MEALFIT_SCHEDULER_BOOT_GRACE_MIN` (default 2, clamp [0, 15]).
    `_APP_START_TIME` se setea al entrar a `lifespan()` startup. El
    listener compara `time.time() - _APP_START_TIME` contra el grace;
    si está dentro:
      - MISSED → log INFO + return (NO upsert, NO breadcrumb).
      - ERROR → SIGUE emitiendo (ERROR temprano es señal genuina, NO ruido).

    Knob=0 desactiva la supresión (back-compat). Clamp upper 15min
    cubre deploys rolling extensos.

Lo que este test enforza:
    A) Constante `_SCHEDULER_BOOT_GRACE_S` derivada del knob existe en app.py.
    B) Variable módulo-level `_APP_START_TIME: float = 0.0` declarada.
    C) `lifespan` setea `_APP_START_TIME = time.time()` early en startup
       (antes de `register_plan_chunk_scheduler`).
    D) `_scheduler_alert_listener` invoca el guard de grace para el path
       EVENT_JOB_MISSED — comparando uptime con grace ANTES del upsert.
    E) El guard NO se aplica al path EVENT_JOB_ERROR (asimetría intencional).
    F) Knob está clampado a [0, 15] (no permite valores negativos ni
       arbitrarios que rompan UX operacional).

Tooltip-anchor: P2-AUDIT-NEW-3-BOOT-GRACE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"


@pytest.fixture(scope="module")
def app_src() -> str:
    return _APP_PY.read_text(encoding="utf-8")


def test_boot_grace_constants_declared(app_src: str) -> None:
    """A) Constantes derivadas del knob existen al top del módulo."""
    assert re.search(
        r"_SCHEDULER_BOOT_GRACE_MIN\s*=\s*max\s*\(\s*0\s*,\s*min\s*\(\s*_env_int\s*\(\s*[\"']MEALFIT_SCHEDULER_BOOT_GRACE_MIN[\"']",
        app_src,
    ), (
        "P2-AUDIT-NEW-3 violation: `_SCHEDULER_BOOT_GRACE_MIN` no está "
        "definido como `max(0, min(_env_int('MEALFIT_SCHEDULER_BOOT_GRACE_MIN', ...), ...))`. "
        "Clamp manual requerido — `_env_int` no clampa por default."
    )
    assert re.search(
        r"_SCHEDULER_BOOT_GRACE_S\s*=\s*_SCHEDULER_BOOT_GRACE_MIN\s*\*\s*60",
        app_src,
    ), "P2-AUDIT-NEW-3 violation: `_SCHEDULER_BOOT_GRACE_S` no se deriva de `_SCHEDULER_BOOT_GRACE_MIN * 60`."


def test_app_start_time_module_var(app_src: str) -> None:
    """B) `_APP_START_TIME: float = 0.0` declarado a nivel módulo."""
    assert re.search(
        r"_APP_START_TIME\s*:\s*float\s*=\s*0\.0",
        app_src,
    ), (
        "P2-AUDIT-NEW-3 violation: `_APP_START_TIME: float = 0.0` no "
        "existe como variable módulo-level. El listener necesita un "
        "valor por default seguro (pre-startup) para que la comparación "
        "`time.time() - _APP_START_TIME` retorne uptime grande → "
        "permita alerts si jamás se llama lifespan (e.g. tests, scripts)."
    )


def test_lifespan_sets_start_time_early(app_src: str) -> None:
    """C) `lifespan` setea `_APP_START_TIME = time.time()` con `global`
    statement, antes del bloque `if HAS_SCHEDULER and scheduler:` que
    registra crons.
    """
    # Localizar lifespan body
    lifespan_match = re.search(
        r"async def lifespan\(app:\s*FastAPI\)\s*:\s*\n",
        app_src,
    )
    assert lifespan_match, "No se encontró `async def lifespan(app: FastAPI):` en app.py."

    body_start = lifespan_match.end()
    # Cortar al `if HAS_SCHEDULER and scheduler:` que marca el inicio del
    # bloque de registro de crons.
    cron_block_pos = app_src.find("if HAS_SCHEDULER and scheduler:", body_start)
    assert cron_block_pos > body_start, (
        "No se encontró `if HAS_SCHEDULER and scheduler:` después de "
        "`lifespan(...)`. El test asume ese marker como upper-bound."
    )

    body = app_src[body_start:cron_block_pos]

    # `global _APP_START_TIME` declarado
    assert re.search(
        r"\bglobal\s+_APP_START_TIME\b",
        body,
    ), (
        "P2-AUDIT-NEW-3 violation: `lifespan` debe declarar "
        "`global _APP_START_TIME` antes de asignarle (sin `global` "
        "se crea local y el listener nunca ve el bump)."
    )

    # `_APP_START_TIME = time.time()` presente en body
    assert re.search(
        r"_APP_START_TIME\s*=\s*time\.time\(\)",
        body,
    ), (
        "P2-AUDIT-NEW-3 violation: `lifespan` no setea "
        "`_APP_START_TIME = time.time()` antes del registro de crons. "
        "Si el setter va después, los primeros eventos MISSED no entran "
        "en grace y se siguen upserteando."
    )


def test_missed_path_guarded_by_grace(app_src: str) -> None:
    """D) El path EVENT_JOB_MISSED en `_scheduler_alert_listener` invoca
    el guard de grace y hace `return` cuando uptime < grace.

    El guard debe usar `_APP_START_TIME > 0` (evita falsos positivos
    cuando lifespan no corrió) Y `_SCHEDULER_BOOT_GRACE_S > 0` (kill
    switch via knob=0).
    """
    # Localizar bloque del listener
    listener_match = re.search(
        r"def _scheduler_alert_listener\(event\)\s*:",
        app_src,
    )
    assert listener_match, "No se encontró `_scheduler_alert_listener` en app.py."

    body_start = listener_match.end()
    # Cortar al cierre de la función (siguiente top-level `def `, `async def `,
    # o end of file).
    next_def = re.search(r"\n(?:def|async def)\s+\w+\s*\(", app_src[body_start:])
    body_end = body_start + next_def.start() if next_def else len(app_src)
    body = app_src[body_start:body_end]

    # Localizar el branch MISSED
    missed_branch_match = re.search(
        r"if\s+code\s*==\s*EVENT_JOB_MISSED\s*:",
        body,
    )
    assert missed_branch_match, "No se encontró branch `if code == EVENT_JOB_MISSED:` en el listener."

    # El siguiente branch (elif EVENT_JOB_ERROR) marca el final del MISSED block.
    error_branch_match = re.search(
        r"elif\s+code\s*==\s*EVENT_JOB_ERROR\s*:",
        body[missed_branch_match.end():],
    )
    assert error_branch_match, "No se encontró `elif code == EVENT_JOB_ERROR:` tras el MISSED block."

    missed_block = body[
        missed_branch_match.end():
        missed_branch_match.end() + error_branch_match.start()
    ]

    # El guard de grace debe estar presente: chequeo de `_APP_START_TIME > 0`
    # AND `_SCHEDULER_BOOT_GRACE_S > 0`.
    assert "_APP_START_TIME > 0" in missed_block, (
        "P2-AUDIT-NEW-3 violation: el branch MISSED no verifica "
        "`_APP_START_TIME > 0` antes de aplicar grace. Sin esto, "
        "tests/scripts que importan app.py sin lifespan tendrían "
        "uptime infinito (`time.time() - 0`) y nunca suprimirían."
    )
    assert "_SCHEDULER_BOOT_GRACE_S > 0" in missed_block, (
        "P2-AUDIT-NEW-3 violation: el branch MISSED no verifica "
        "`_SCHEDULER_BOOT_GRACE_S > 0`. Sin esto, `knob=0` (kill switch) "
        "no desactiva la supresión."
    )
    assert "time.time() - _APP_START_TIME" in missed_block, (
        "P2-AUDIT-NEW-3 violation: el branch MISSED no calcula uptime "
        "con `time.time() - _APP_START_TIME`. Sin este cálculo, el "
        "guard no puede comparar contra el grace window."
    )
    assert "return" in missed_block, (
        "P2-AUDIT-NEW-3 violation: el branch MISSED no hace `return` "
        "en el guard — sin return la ejecución sigue al upsert y la "
        "alert se inserta de todos modos."
    )


def test_error_path_NOT_guarded_by_grace(app_src: str) -> None:
    """E) El path EVENT_JOB_ERROR NO debe estar bajo el guard de grace
    (asimetría intencional: ERROR siempre es señal genuina).

    El test asegura que dentro del bloque `elif code == EVENT_JOB_ERROR`
    NO hay un `if _APP_START_TIME > 0 ... return` antes del upsert.
    """
    listener_match = re.search(
        r"def _scheduler_alert_listener\(event\)\s*:",
        app_src,
    )
    assert listener_match, "Listener no encontrado."
    body_start = listener_match.end()
    next_def = re.search(r"\n(?:def|async def)\s+\w+\s*\(", app_src[body_start:])
    body_end = body_start + next_def.start() if next_def else len(app_src)
    body = app_src[body_start:body_end]

    error_branch_match = re.search(
        r"elif\s+code\s*==\s*EVENT_JOB_ERROR\s*:",
        body,
    )
    assert error_branch_match, "Branch ERROR no encontrado."

    # El bloque ERROR termina con el siguiente `elif code ==` o `else:`
    next_branch = re.search(
        r"\n\s+elif\s+code\s*==|^\s+else\s*:",
        body[error_branch_match.end():],
        re.MULTILINE,
    )
    error_block = body[
        error_branch_match.end():
        error_branch_match.end() + (next_branch.start() if next_branch else len(body) - error_branch_match.end())
    ]

    # El branch ERROR NO debe contener un `return` condicionado por grace.
    # Buscamos el patrón completo `if _APP_START_TIME > 0 ... : ... return`
    # — si aparece dentro del block, el test falla.
    guard_pattern = re.compile(
        r"if\s+_APP_START_TIME\s*>\s*0[^\n]*:",
    )
    assert not guard_pattern.search(error_block), (
        "P2-AUDIT-NEW-3 violation: el branch EVENT_JOB_ERROR tiene un "
        "guard de grace (`if _APP_START_TIME > 0...`). Asimetría: "
        "ERROR siempre debe emitir alert, incluso temprano post-boot "
        "(es señal genuina de bug, no de burst de re-arranque). Si "
        "necesitas suprimir errores boot-time específicos, hazlo a "
        "nivel del job (try/except dentro del job) — no del listener."
    )


def test_knob_clamp_bounds(app_src: str) -> None:
    """F) Knob está clampado a [0, 15] explícitamente en el código.

    El default de 2 min es razonable; un operador podría querer subirlo
    durante deploys lentos pero no debe poder setear valores enormes
    (e.g., 600 min = 10h) que oculten alerts genuinos post-warmup.
    """
    # Patrón estricto: `max(0, min(_env_int('MEALFIT_SCHEDULER_BOOT_GRACE_MIN', N), M))`
    # con M numérico al final.
    pattern = re.compile(
        r"_SCHEDULER_BOOT_GRACE_MIN\s*=\s*max\s*\(\s*0\s*,\s*min\s*\(\s*"
        r"_env_int\s*\(\s*[\"']MEALFIT_SCHEDULER_BOOT_GRACE_MIN[\"']\s*,\s*(\d+)\s*\)\s*,\s*(\d+)\s*\)\s*\)",
    )
    m = pattern.search(app_src)
    assert m, (
        "P2-AUDIT-NEW-3 violation: no se encontró el clamp "
        "`max(0, min(_env_int(MEALFIT_SCHEDULER_BOOT_GRACE_MIN, <default>), <upper>))` "
        "en app.py. Sin clamp, un operador podría setear 600 min y "
        "ocultar alerts genuinos post-warmup."
    )
    default_val = int(m.group(1))
    upper_val = int(m.group(2))
    assert 0 <= default_val <= 5, (
        f"Default del knob ({default_val}) fuera de rango razonable [0,5]. "
        f"2 min cubre el burst observado; >5 oculta MISSED genuinos."
    )
    assert upper_val == 15, (
        f"Upper bound del clamp ({upper_val}) ≠ 15 min. Si subes este "
        f"techo, actualiza este test Y CLAUDE.md (deploys que tarden "
        f">15min son anomalía operacional, no caso normal)."
    )
