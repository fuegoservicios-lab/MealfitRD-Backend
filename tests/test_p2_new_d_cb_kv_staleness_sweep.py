"""[P2-NEW-D · 2026-05-11] Sweep periódico que resetea rows stale en
`app_kv_store` con keys `llm_circuit_breaker%` cuyo `last_failure`
excede `MEALFIT_CB_KV_STALENESS_HOURS` (default 2h).

Gap cerrado:
    `LLMCircuitBreaker.can_proceed()` retorna True una vez `time.time() -
    last_failure > reset_timeout` — runtime CORRECTO. La fila en
    `app_kv_store` queda `is_open=true` con contadores stale hasta que
    algún caller exitoso llame `record_success()`. Si tras un outage
    el modelo deja de routearse (ej. `gemini-3.1-pro-preview` sin
    perfil clínico activo durante semanas), nadie reinicia la fila.

    Audit 2026-05-11 detectó row `llm_circuit_breaker:gemini-3.1-pro-preview`
    con `is_open=true, failures=6, last_failure=2026-05-07` (4.4 días) —
    SRE leyendo `app_kv_store` concluye "modelo caído" cuando
    funcionalmente está disponible.

Drift detection:
    - Función `_sweep_stale_llm_circuit_breakers` renombrada/borrada → falla.
    - UPDATE pierde el filtro de pattern `'llm_circuit_breaker%'` → falla.
    - UPDATE pierde el filtro de staleness `(value->>'last_failure')::float < ...` → falla.
    - UPDATE pierde el filtro idempotente (`is_open='true' OR failures>0`) → falla.
    - Knobs `MEALFIT_CB_KV_STALENESS_HOURS` / `_SWEEP_INTERVAL_MIN` no
      leídos via `_env_int` (auto-registro `_KNOBS_REGISTRY`) → falla.
    - Cron no registrado en `register_plan_chunk_scheduler` → falla.
    - Tick observable (`pipeline_metrics._sweep_stale_llm_circuit_breakers_tick`)
      desaparece → falla (rompe convención P2-B-OBS/P3-LIVE-1).

Tooltip-anchor: P2-NEW-D-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"


def _read_function_body(source: str, fn_name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(r"^(def |class |@)", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


@pytest.fixture(scope="module")
def cron_source() -> str:
    return _CRON.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Función definida
# ---------------------------------------------------------------------------
def test_sweep_function_defined(cron_source: str):
    assert re.search(
        r"^def\s+_sweep_stale_llm_circuit_breakers\s*\(",
        cron_source,
        re.MULTILINE,
    ), (
        "P2-NEW-D violation: `_sweep_stale_llm_circuit_breakers` no está "
        "definida en cron_tasks.py. Sin ella, rows `llm_circuit_breaker:*` "
        "con `is_open=true` quedan visibles indefinidamente en "
        "`app_kv_store` aunque el runtime ya retorne True via `can_proceed()`."
    )


# ---------------------------------------------------------------------------
# 2. UPDATE filtra pattern key + staleness + idempotencia
# ---------------------------------------------------------------------------
def test_update_filters_pattern_and_staleness(cron_source: str):
    body = _read_function_body(cron_source, "_sweep_stale_llm_circuit_breakers")
    assert body, "Función ausente — cubre test #1."

    assert re.search(
        r"key\s+LIKE\s+'llm_circuit_breaker%%'",
        body,
    ), (
        "P2-NEW-D violation: UPDATE no filtra `key LIKE 'llm_circuit_breaker%%'`. "
        "Sin este pattern el sweep tocaría keys ortogonales (rag_*, "
        "expected_last_known_pfix, etc.)."
    )
    # Staleness filter usando epoch.
    assert re.search(
        r"value->>'last_failure'\s*\)::float\s*\n?\s*<\s*\(",
        body,
    ) or re.search(
        r"value->>'last_failure'.{0,40}::float.{0,80}EPOCH",
        body,
        re.DOTALL,
    ), (
        "P2-NEW-D violation: UPDATE no filtra "
        "`(value->>'last_failure')::float < (EPOCH threshold)`. Sin esto "
        "resetearía CBs que recién se abrieron, interfiriendo con el "
        "runtime activo del runtime breaker."
    )
    # Idempotencia: no tocar rows ya en estado canónico cero.
    assert re.search(
        r"value->>'is_open'\s*=\s*'true'",
        body,
    ), (
        "P2-NEW-D violation: UPDATE no filtra "
        "`value->>'is_open' = 'true'` (ni gate idempotente equivalente). "
        "Sin esto contaría como swept rows ya en estado canónico cero, "
        "inflando métricas y emitiendo UPDATE no-op."
    )


# ---------------------------------------------------------------------------
# 3. Payload de reset = estado canónico cero (mismo que _atomic_reset_db)
# ---------------------------------------------------------------------------
def test_update_payload_is_canonical_zero_state(cron_source: str):
    """El UPDATE debe escribir el mismo payload que
    `LLMCircuitBreaker._atomic_reset_db()` (graph_orchestrator.py:1431).
    Cualquier divergencia abre asimetría entre el reset por success y el
    reset por sweep (mismo objeto, dos shapes)."""
    body = _read_function_body(cron_source, "_sweep_stale_llm_circuit_breakers")
    assert re.search(
        r'"failures":\s*0',
        body,
    ), "Payload no incluye `\"failures\": 0`."
    assert re.search(
        r'"last_failure":\s*0',
        body,
    ), "Payload no incluye `\"last_failure\": 0`."
    assert re.search(
        r'"is_open":\s*false',
        body,
    ), "Payload no incluye `\"is_open\": false`."


# ---------------------------------------------------------------------------
# 4. Knobs auto-registrados via _env_int
# ---------------------------------------------------------------------------
def test_knobs_registered_via_env_int(cron_source: str):
    """`MEALFIT_CB_KV_STALENESS_HOURS` y
    `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` DEBEN leerse via
    `_env_int` para auto-registrarse en `_KNOBS_REGISTRY` y aparecer en
    `/health/version`."""
    body = _read_function_body(cron_source, "_sweep_stale_llm_circuit_breakers")
    assert re.search(
        r'_env_int\s*\(\s*["\']MEALFIT_CB_KV_STALENESS_HOURS["\']',
        body,
    ), (
        "P2-NEW-D violation: `MEALFIT_CB_KV_STALENESS_HOURS` no leído via "
        "`_env_int` dentro de la función. Reemplazar cualquier "
        "`os.environ.get(...)` raw."
    )
    register_body = _read_function_body(cron_source, "register_plan_chunk_scheduler")
    assert re.search(
        r'_env_int\s*\(\s*["\']MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN["\']',
        register_body,
    ), (
        "P2-NEW-D violation: `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` "
        "no leído via `_env_int` en `register_plan_chunk_scheduler`."
    )


# ---------------------------------------------------------------------------
# 5. Cron registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------
def test_cron_registered(cron_source: str):
    register_body = _read_function_body(cron_source, "register_plan_chunk_scheduler")
    assert re.search(
        r'id\s*=\s*["\']sweep_stale_llm_circuit_breakers["\']',
        register_body,
    ), (
        "P2-NEW-D violation: cron `sweep_stale_llm_circuit_breakers` "
        "no registrado en `register_plan_chunk_scheduler`. Sin registro "
        "la función nunca corre."
    )
    assert "_sweep_stale_llm_circuit_breakers" in register_body, (
        "P2-NEW-D violation: register_plan_chunk_scheduler tiene el id "
        "pero no pasa la función `_sweep_stale_llm_circuit_breakers` "
        "como callable."
    )


# ---------------------------------------------------------------------------
# 6. Tick observable a pipeline_metrics (patrón P2-B-OBS)
# ---------------------------------------------------------------------------
def test_tick_observable_emitted(cron_source: str):
    """El sweep DEBE emitir un tick observable a `pipeline_metrics`
    con `node='_sweep_stale_llm_circuit_breakers_tick'` SIEMPRE
    (no solo cuando hay rows reseteadas). Patrón P2-B-OBS/P3-LIVE-1."""
    body = _read_function_body(cron_source, "_sweep_stale_llm_circuit_breakers")
    assert re.search(
        r'"_sweep_stale_llm_circuit_breakers_tick"',
        body,
    ), (
        "P2-NEW-D violation: el sweep no emite tick observable "
        "`_sweep_stale_llm_circuit_breakers_tick` a `pipeline_metrics`. "
        "Sin tick, el cron es invisible cuando `reset_count=0` — "
        "rompe la convención P2-B-OBS / P3-LIVE-1 (cron debe ser "
        "verificable independiente del outcome)."
    )
    # Metadata debe incluir reset_count para correlacionar con logs.
    assert '"reset_count"' in body, (
        "P2-NEW-D violation: metadata del tick no incluye `reset_count`. "
        "Sin este field, post-mortem no puede correlacionar con logs."
    )
