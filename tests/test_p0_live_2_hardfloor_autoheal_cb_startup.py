"""[P0-LIVE-1 + P0-LIVE-2 · 2026-05-11] Hard-floor autoheal asyncio task
(independiente de APScheduler) + startup-run del CB sweep en `lifespan`.

Gap cerrado (audit live 2026-05-11):
    APScheduler en prod tuvo 19 jobs missed simultáneos a las 18:08:52 UTC
    (cascade alert CRITICAL abierta >35min). Crons designed-as-autoheal
    (`_resolve_stale_scheduler_alerts` P0-AUDIT-1+P2-LIVE-1,
    `_sweep_stale_llm_circuit_breakers` P2-NEW-D, `_alert_scheduler_cascade_missed`
    P0-NEW-2) viven DENTRO del scheduler — cuando el pool está saturado
    el autoheal también está MISSED. Resultado observado:

      - `scheduler_cascade_missed` (CRITICAL) abierto sin auto-resolve.
      - CB row `llm_circuit_breaker:gemini-3.1-pro-preview` con
        `is_open=true` durante 4.4 días pese a `MEALFIT_CB_KV_STALENESS_HOURS=2`.
      - pipeline_metrics 0 ticks en 48h para 4 crons baseline (`_sweep_stale_*`,
        `_chunk_heartbeat_baseline`, `_alert_chunk_pantry_snapshots_stale_tick`,
        `_scheduler_cascade_autoheal`).

Fix (P0-LIVE-1):
    `_hardfloor_autoheal_loop` async coroutine corre en el event loop de
    FastAPI/uvicorn, NO en el ThreadPoolExecutor del scheduler. `asyncio.to_thread`
    despacha los sweeps sync a worker threads del event loop default pool.
    Cada `MEALFIT_HARDFLOOR_AUTOHEAL_INTERVAL_S` (default 300s, clamp [60,1800])
    invoca:
      - `_resolve_stale_scheduler_alerts` (incluye sweep cascade P2-LIVE-1 y
        hard-cap edad absoluta P2-NEW-E).
      - `_sweep_stale_llm_circuit_breakers` (P2-NEW-D).
    Emite tick observable `_hardfloor_autoheal_tick` por iteración (espejo
    P3-LIVE-1) → confirma live que el loop sigue vivo aunque APScheduler
    esté saturado.
    Knob kill-switch `MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED` (default True).
    Task guardado en `app.state.hardfloor_autoheal_task`; cancelado
    en shutdown ANTES de cerrar pools/scheduler.

Fix (P0-LIVE-2):
    `lifespan` invoca `_sweep_stale_llm_circuit_breakers()` UNA vez al startup
    DESPUÉS del autoheal P0-NEW-1 y ANTES del start del scheduler. Garantiza
    una limpieza por deploy/restart aunque el cron periódico esté missed.
    Patrón espejo P0-NEW-1-AUTOHEAL (sweep scheduler alerts en startup).

Drift detection:
    - `_hardfloor_autoheal_loop` renombrada/borrada → falla.
    - El loop pierde la llamada a `_resolve_stale_scheduler_alerts` o
      `_sweep_stale_llm_circuit_breakers` → falla.
    - El loop pierde el tick observable `_hardfloor_autoheal_tick` → falla.
    - El loop no se crea via `asyncio.create_task` en `lifespan` → falla.
    - El task no se cancela en shutdown (post-yield) → falla.
    - Startup-run del CB sweep desaparece de `lifespan` (P0-LIVE-2 regression)
      → falla.
    - Knobs `MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED` /
      `MEALFIT_HARDFLOOR_AUTOHEAL_INTERVAL_S` no leídos via `_env_*` (auto-registro
      `_KNOBS_REGISTRY`) → falla.

Tooltip-anchor: P0-LIVE-1-START | P0-LIVE-2-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_APP = _BACKEND / "app.py"


def _read_function_body(source: str, fn_name: str) -> str:
    """Devuelve el cuerpo de la primera función `def`/`async def` con `fn_name`.

    Robusta a `async def` y a indent — corta en la próxima top-level `def `,
    `async def `, `class ` o `@` decorator.
    """
    pattern = re.compile(
        rf"^(?:async\s+)?def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(
        r"^(async\s+def |def |class |@)",
        re.MULTILINE,
    )
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


@pytest.fixture(scope="module")
def app_source() -> str:
    return _APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def hardfloor_body(app_source: str) -> str:
    body = _read_function_body(app_source, "_hardfloor_autoheal_loop")
    assert body, (
        "[P0-LIVE-1] `_hardfloor_autoheal_loop` desapareció de app.py. "
        "Es el coroutine que corre los sweeps en el event loop independiente "
        "del scheduler — sin él, los autoheal vuelven a depender de APScheduler "
        "y la cascada se autosostiene. Si renombras, actualiza este test."
    )
    return body


@pytest.fixture(scope="module")
def lifespan_body(app_source: str) -> str:
    body = _read_function_body(app_source, "lifespan")
    assert body, "[P0-LIVE-1+2] `lifespan` no encontrado en app.py."
    return body


# ---------------------------------------------------------------------------
# P0-LIVE-1: hard-floor autoheal loop
# ---------------------------------------------------------------------------

def test_p0_live_1_loop_invokes_resolve_stale_scheduler_alerts(hardfloor_body: str):
    """El loop debe invocar `_resolve_stale_scheduler_alerts` via
    `asyncio.to_thread` (no bloquea el event loop)."""
    assert "_resolve_stale_scheduler_alerts" in hardfloor_body, (
        "[P0-LIVE-1] El loop ya no invoca `_resolve_stale_scheduler_alerts`. "
        "Esa es la razón principal del task: cerrar alerts scheduler_* cuando "
        "el cron P0-AUDIT-1/P2-LIVE-1 está missed por saturación."
    )
    assert "asyncio.to_thread" in hardfloor_body, (
        "[P0-LIVE-1] El loop debe usar `asyncio.to_thread` para los sweeps "
        "sync — invocarlos directo bloquearía el event loop."
    )


def test_p0_live_1_loop_invokes_cb_sweep(hardfloor_body: str):
    """El loop debe invocar `_sweep_stale_llm_circuit_breakers` también
    (defense-in-depth con el startup-run de P0-LIVE-2)."""
    assert "_sweep_stale_llm_circuit_breakers" in hardfloor_body, (
        "[P0-LIVE-1] El loop ya no invoca `_sweep_stale_llm_circuit_breakers`. "
        "El startup-run (P0-LIVE-2) cierra el primer tick; el loop mantiene "
        "frescas las filas mientras el cron P2-NEW-D está missed."
    )


def test_p0_live_1_loop_emits_observable_tick(hardfloor_body: str):
    """El loop debe emitir `_hardfloor_autoheal_tick` a pipeline_metrics
    (espejo P3-LIVE-1 / P2-NEW-D). Sin esto, no podemos distinguir
    'loop vivo y sin trabajo' de 'loop muerto'."""
    assert "_hardfloor_autoheal_tick" in hardfloor_body, (
        "[P0-LIVE-1] Tick observable `_hardfloor_autoheal_tick` desaparecido. "
        "Es la única señal de que el loop sigue vivo independiente de APScheduler."
    )
    assert "pipeline_metrics" in hardfloor_body, (
        "[P0-LIVE-1] El tick debe insertarse en `pipeline_metrics`."
    )


def test_p0_live_1_loop_uses_asyncio_sleep(hardfloor_body: str):
    """El loop debe pausar via `asyncio.sleep(interval_s)` — `time.sleep`
    bloquearía el event loop."""
    assert "asyncio.sleep" in hardfloor_body, (
        "[P0-LIVE-1] El loop debe usar `asyncio.sleep(interval_s)` para "
        "ceder el event loop. `time.sleep` bloquearía todas las requests."
    )


def test_p0_live_1_loop_handles_cancellation(hardfloor_body: str):
    """El loop debe propagar/loguear `asyncio.CancelledError` para
    shutdown limpio."""
    assert "CancelledError" in hardfloor_body, (
        "[P0-LIVE-1] El loop debe manejar `asyncio.CancelledError` para "
        "shutdown limpio (cancelado en `lifespan` post-yield)."
    )


def test_p0_live_1_task_created_in_lifespan(lifespan_body: str):
    """`lifespan` debe crear el task via `asyncio.create_task` y guardarlo
    en `app.state` para poder cancelarlo en shutdown."""
    assert "asyncio.create_task" in lifespan_body, (
        "[P0-LIVE-1] `lifespan` debe lanzar el hard-floor loop con "
        "`asyncio.create_task` — sin esto el coroutine nunca corre."
    )
    assert "_hardfloor_autoheal_loop" in lifespan_body, (
        "[P0-LIVE-1] `lifespan` debe referenciar `_hardfloor_autoheal_loop` "
        "para instanciarlo."
    )
    assert "hardfloor_autoheal_task" in lifespan_body, (
        "[P0-LIVE-1] El task debe guardarse en `app.state.hardfloor_autoheal_task` "
        "para cancelarse en shutdown."
    )


def test_p0_live_1_task_cancelled_on_shutdown(lifespan_body: str):
    """El task debe cancelarse post-`yield` (sección shutdown de `lifespan`)
    ANTES del shutdown del scheduler para que no se quede colgado escribiendo
    al pool ya cerrado."""
    # Separa pre/post yield. `yield` debe existir en lifespan.
    parts = lifespan_body.split("yield", 1)
    assert len(parts) == 2, (
        "[P0-LIVE-1] `lifespan` no contiene `yield` — estructura inválida "
        "(context manager FastAPI)."
    )
    post_yield = parts[1]
    assert ".cancel()" in post_yield, (
        "[P0-LIVE-1] El task hard-floor no se cancela post-yield. "
        "Sin cancel(), el shutdown deja la coroutine corriendo contra "
        "pools ya cerrados."
    )
    assert "hardfloor_autoheal_task" in post_yield, (
        "[P0-LIVE-1] La sección shutdown debe referenciar el task guardado."
    )


@pytest.mark.parametrize("knob", [
    "MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED",
    "MEALFIT_HARDFLOOR_AUTOHEAL_INTERVAL_S",
])
def test_p0_live_1_knobs_registered_via_env_helpers(app_source: str, knob: str):
    """Los knobs P0-LIVE-1 deben leerse via `_env_bool` o `_env_int`
    (auto-registro en `_KNOBS_REGISTRY` — convención del repo)."""
    pattern_bool = rf'_env_bool\(\s*["\']{re.escape(knob)}["\']'
    pattern_int = rf'_env_int\(\s*["\']{re.escape(knob)}["\']'
    assert re.search(pattern_bool, app_source) or re.search(pattern_int, app_source), (
        f"[P0-LIVE-1] Knob `{knob}` no leído via `_env_bool`/`_env_int`. "
        f"Sin auto-registro en `_KNOBS_REGISTRY` queda invisible al snapshot."
    )


def test_p0_live_1_interval_clamped(app_source: str):
    """El intervalo debe clamparse a `[60, 1800]` para que un knob mal
    configurado no genere thrashing (<60s burst contra DB) ni dormancia
    larga (>1800s donde el cascade reaparece antes del próximo tick)."""
    # Busca el patrón `max(60, min(_hardfloor_interval, 1800))` o equivalente.
    m = re.search(
        r"_hardfloor_interval\s*=\s*max\(\s*60\s*,\s*min\(\s*_hardfloor_interval\s*,\s*1800\s*\)\s*\)",
        app_source,
    )
    assert m, (
        "[P0-LIVE-1] El intervalo del hard-floor no está clamado a [60, 1800]. "
        "Patrón esperado: `_hardfloor_interval = max(60, min(_hardfloor_interval, 1800))`."
    )


# ---------------------------------------------------------------------------
# P0-LIVE-2: startup-run del CB sweep
# ---------------------------------------------------------------------------

def test_p0_live_2_startup_cb_sweep_in_lifespan(lifespan_body: str):
    """`lifespan` debe invocar `_sweep_stale_llm_circuit_breakers()` UNA vez
    en startup ANTES del `yield` (mirror P0-NEW-1-AUTOHEAL)."""
    parts = lifespan_body.split("yield", 1)
    assert len(parts) == 2, "[P0-LIVE-2] `lifespan` sin yield."
    pre_yield = parts[0]
    assert "_sweep_stale_llm_circuit_breakers" in pre_yield, (
        "[P0-LIVE-2] `lifespan` ya no invoca `_sweep_stale_llm_circuit_breakers` "
        "en startup. Sin esto, un deploy/restart con CB rows stale no las limpia "
        "hasta el primer tick del cron P2-NEW-D — que puede estar missed."
    )


def test_p0_live_2_startup_sweep_best_effort(lifespan_body: str):
    """El startup CB sweep debe estar wrappeado en try/except (best-effort,
    NO debe abortar el startup si falla)."""
    parts = lifespan_body.split("yield", 1)
    pre_yield = parts[0]
    # Captura la región alrededor de la llamada
    m = re.search(
        r"try:.*?_sweep_stale_llm_circuit_breakers\(\).*?except",
        pre_yield,
        re.DOTALL,
    )
    assert m, (
        "[P0-LIVE-2] El startup CB sweep debe estar dentro de un bloque "
        "`try/except` (best-effort). Un fallo del sweep NO debe tumbar el "
        "startup — el sweep es defense-in-depth, no startup-critical."
    )


def test_p0_live_2_marker_bumped(app_source: str):
    """`_LAST_KNOWN_PFIX` debe reflejar P0-LIVE-2 (o posterior). Sin bump,
    `/health/version` no expone que el fix está vivo."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*["\'](P[0-9].*?·\s*\d{4}-\d{2}-\d{2})["\']',
        app_source,
    )
    assert m, "[P0-LIVE-2] `_LAST_KNOWN_PFIX` no encontrado o malformado en app.py."
    marker = m.group(1)
    # Aceptamos P0-LIVE-2 o cualquier P-fix posterior con fecha >= 2026-05-11.
    # El check de fecha lo hace `test_p3_1_last_known_pfix_freshness`; aquí
    # solo exigimos que NO sea el marker pre-P0-LIVE (P3-NEW-E).
    assert "P3-NEW-E" not in marker or "LIVE" in marker, (
        f"[P0-LIVE-2] `_LAST_KNOWN_PFIX={marker}` no fue bumpeado tras "
        f"P0-LIVE-1/2. Sube el marker al cerrar cada P-fix mergeado."
    )
