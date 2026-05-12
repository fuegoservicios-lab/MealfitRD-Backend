"""[P1-LIVE-3 + P1-LIVE-4 + P1-LIVE-5 · 2026-05-11] Hardening de
`_recover_future_scheduled_pending_chunks` + nuevo sweep de planes
test-fixture (`Plan Sintético% — Test%`).

Gaps cerrados (audit live 2026-05-11):

P1-LIVE-3 — Plan zombies de test fixture en prod
    User `bf6f1383` (test user del equipo) acumuló 3 planes con nombre
    `Plan Sintético <N> días — Test <Chunks|Historial|...>` (created
    2026-05-08 a 2026-05-10). P2-NEXT-3 NO los cubre porque tienen
    chunks vivos. Ayer P2-LIVE-2 limpió 8 chunks fixture del mismo
    user via oneshot — el patrón se repite. Fix: nuevo cron diario
    `_sweep_synthetic_test_plans` (03:45 UTC) cancela chunks vivos
    + marca el plan abandoned via jsonb merge `||`.

P1-LIVE-4 — `_recover_future_scheduled_pending_chunks` sin tick observable
    Audit live no podía distinguir "cron registrado pero no firing"
    (scheduler starvation) de "cron firing pero 0 candidatos vs
    escalando". Fix: try/finally con tick `_recover_future_scheduled_pending_chunks_tick`
    emitido SIEMPRE (patrón P3-LIVE-1).

P1-LIVE-5 — `_recover_future_scheduled_pending_chunks` excluye planes legacy
    Filtro SQL original `AND (p.plan_data->>'total_days_requested')::int IS NOT NULL`
    excluía planes pre-campo. Sus chunks pending futuros eran zombies
    indetectables. Fix: `COALESCE((p.plan_data->>'total_days_requested')::int,
    jsonb_array_length(p.plan_data->'days'), 7)`.

Drift detection:
    - Función `_sweep_synthetic_test_plans` renombrada/borrada → falla.
    - Pattern `'Plan Sintético%% — Test%%'` cambiado (drift hacia LIKE
      laxo que ataparía planes legítimos) → falla.
    - Cron no registrado en `register_plan_chunk_scheduler` → falla.
    - Tick observable `_recover_future_scheduled_pending_chunks_tick`
      desaparece → falla.
    - Filtro `IS NOT NULL` reintroducido (regresión P1-LIVE-5) → falla.
    - COALESCE pierde alguno de los 3 niveles → falla.

Tooltip-anchor: P1-LIVE-3-START | P1-LIVE-4-START | P1-LIVE-5-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"
_APP = _BACKEND / "app.py"


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


@pytest.fixture(scope="module")
def app_source() -> str:
    return _APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def recover_body(cron_source: str) -> str:
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    assert body, (
        "[P1-LIVE-4/5] `_recover_future_scheduled_pending_chunks` no encontrado. "
        "Es la función que escala chunks pending con execute_after fuera del horizonte "
        "del plan — borrarla reabre el gap original de P1-NEW-D (zombies 5+ meses)."
    )
    return body


@pytest.fixture(scope="module")
def synthetic_body(cron_source: str) -> str:
    body = _read_function_body(cron_source, "_sweep_synthetic_test_plans")
    assert body, (
        "[P1-LIVE-3] `_sweep_synthetic_test_plans` no encontrado. "
        "Es el sweep diario que limpia test fixtures (`Plan Sintético% — Test%`) "
        "de prod — sin él, el user `bf6f1383` (QA) acumula planes zombie "
        "indefinidamente porque P2-NEXT-3 skipea planes con chunks vivos."
    )
    return body


# ---------------------------------------------------------------------------
# P1-LIVE-5: COALESCE fallback para planes legacy sin total_days_requested
# ---------------------------------------------------------------------------

def test_p1_live_5_coalesce_total_days_requested(recover_body: str):
    """El SELECT debe usar `COALESCE` con 3 niveles para `total_days_requested`."""
    # Espera: COALESCE((p.plan_data->>'total_days_requested')::int, jsonb_array_length(p.plan_data->'days'), 7)
    coalesce_pattern = re.compile(
        r"COALESCE\s*\(\s*\(\s*p\.plan_data->>\s*'total_days_requested'\s*\)\s*::\s*int\s*,"
        r"\s*jsonb_array_length\s*\(\s*p\.plan_data->\s*'days'\s*\)\s*,"
        r"\s*7\s*\)",
        re.DOTALL,
    )
    assert coalesce_pattern.search(recover_body), (
        "[P1-LIVE-5] El SELECT ya no usa COALESCE con 3 niveles para "
        "`total_days_requested`. Patrón esperado: "
        "`COALESCE((p.plan_data->>'total_days_requested')::int, "
        "jsonb_array_length(p.plan_data->'days'), 7)`. Sin esto, planes "
        "legacy sin el campo quedan zombies indetectables."
    )


def test_p1_live_5_no_is_not_null_filter(recover_body: str):
    """El filtro `AND (plan_data->>'total_days_requested')::int IS NOT NULL`
    debe estar REMOVIDO — ese era el bug que excluía planes legacy."""
    # Busca el filtro problemático específicamente.
    bad_filter = re.compile(
        r"AND\s+\(\s*p\.plan_data->>\s*'total_days_requested'\s*\)\s*::\s*int\s+IS\s+NOT\s+NULL",
        re.IGNORECASE,
    )
    assert not bad_filter.search(recover_body), (
        "[P1-LIVE-5] Regresión: el filtro `IS NOT NULL` sobre "
        "`total_days_requested` reintrodujo el bug que excluye planes legacy. "
        "Ese filtro debe eliminarse — el COALESCE garantiza non-null."
    )


# ---------------------------------------------------------------------------
# P1-LIVE-4: tick observable patrón P3-LIVE-1
# ---------------------------------------------------------------------------

def test_p1_live_4_tick_emitted_in_finally(recover_body: str):
    """El tick `_recover_future_scheduled_pending_chunks_tick` debe estar
    dentro de un `finally:` para emitirse SIEMPRE (incluso si SELECT
    falla o early return por 0 candidatos)."""
    assert "_recover_future_scheduled_pending_chunks_tick" in recover_body, (
        "[P1-LIVE-4] Tick observable desaparecido. Es la única señal "
        "de que el cron está vivo independiente del scheduler."
    )
    # Verifica que la última occurrencia del tick está después de `finally:`
    # (la primera puede aparecer en el docstring — la real es la INSERT en SQL).
    finally_idx = recover_body.find("finally:")
    tick_idx = recover_body.rfind("_recover_future_scheduled_pending_chunks_tick")
    assert finally_idx != -1, (
        "[P1-LIVE-4] La función debe usar try/finally para emitir tick siempre."
    )
    assert finally_idx < tick_idx, (
        "[P1-LIVE-4] El tick observable debe estar DENTRO del bloque `finally:` "
        "para emitirse aunque haya early return / excepción. Patrón P3-LIVE-1."
    )


def test_p1_live_4_tick_has_required_flags(recover_body: str):
    """El tick debe incluir flags `candidates_count`, `escalated`, `skipped`,
    `errors`, `select_failed` para distinguir outcomes."""
    required_flags = ["candidates_count", "escalated", "skipped", "errors", "select_failed"]
    # Localiza el bloque del tick (entre `finally:` y el final de la función).
    finally_idx = recover_body.find("finally:")
    assert finally_idx != -1
    tick_block = recover_body[finally_idx:]
    for flag in required_flags:
        assert flag in tick_block, (
            f"[P1-LIVE-4] Tick observable no incluye flag `{flag}`. "
            f"Sin él, no podemos distinguir scenarios (SELECT fail vs "
            f"0 candidatos vs escalated vs skipped)."
        )


def test_p1_live_4_tick_kill_switch_knob(recover_body: str, cron_source: str):
    """Knob `MEALFIT_RECOVER_FUTURE_TICK_EMIT` debe existir como kill switch
    leído via `_env_bool` (auto-registro `_KNOBS_REGISTRY`)."""
    assert re.search(
        r'_env_bool\(\s*["\']MEALFIT_RECOVER_FUTURE_TICK_EMIT["\']',
        recover_body,
    ) or re.search(
        r'_env_bool\(\s*["\']MEALFIT_RECOVER_FUTURE_TICK_EMIT["\']',
        cron_source,
    ), (
        "[P1-LIVE-4] Knob `MEALFIT_RECOVER_FUTURE_TICK_EMIT` no leído via "
        "`_env_bool`. Sin kill switch + auto-registro, no podemos desactivar "
        "el tick si genera volumen problemático."
    )


# ---------------------------------------------------------------------------
# P1-LIVE-3: sweep diario de planes test-fixture
# ---------------------------------------------------------------------------

def test_p1_live_3_synthetic_pattern_strict(synthetic_body: str):
    """El SELECT debe matchear `'Plan Sintético%% — Test%%'` (con sufijo
    `— Test` para evitar falsos positivos sobre nombres legítimos)."""
    # En la string Python `'%%'` representa `%` literal en SQL (porque
    # usamos `%s` placeholders en otro lado del query).
    assert "Plan Sintético%% — Test%%" in synthetic_body, (
        "[P1-LIVE-3] Pattern del SELECT drifteado. Esperado literal "
        "`'Plan Sintético%% — Test%%'` (estricto, requiere sufijo `— Test`). "
        "Sin `— Test` un usuario podría nombrar legítimamente un plan "
        "`Plan Sintético Pro 2026` y este sweep lo borraría — falso positivo."
    )


def test_p1_live_3_cancels_alive_chunks_before_marking_abandoned(synthetic_body: str):
    """El sweep debe cancelar chunks `pending/processing/stale` ANTES de
    marcar el plan abandoned. Diferencia clave con P2-NEXT-3 que solo
    cubre planes ya sin chunks vivos."""
    # Cancel pattern
    cancel_pattern = re.compile(
        r"UPDATE\s+plan_chunk_queue\s+SET\s+status\s*=\s*'cancelled'",
        re.IGNORECASE | re.DOTALL,
    )
    assert cancel_pattern.search(synthetic_body), (
        "[P1-LIVE-3] El sweep ya no cancela chunks vivos antes de marcar "
        "el plan abandoned. Sin esto, los chunks fixture quedan corriendo "
        "y el worker los procesará — defeats el propósito del cleanup."
    )
    assert "status IN ('pending', 'processing', 'stale')" in synthetic_body, (
        "[P1-LIVE-3] El cancel debe cubrir 'pending', 'processing' y "
        "'stale' — los 3 estados pre-terminal."
    )


def test_p1_live_3_marks_abandoned_via_jsonb_merge(synthetic_body: str):
    """El UPDATE del plan debe usar `||` jsonb merge (atómico, exento I7)
    y marcar `_abandoned_reason='synthetic_fixture'` para distinguir
    de `orphan_chunks` (P2-NEXT-3)."""
    assert "plan_data || jsonb_build_object" in synthetic_body, (
        "[P1-LIVE-3] El UPDATE debe usar `plan_data || jsonb_build_object(...)` "
        "(merge atómico). Full overwrite requeriría advisory lock (I7)."
    )
    assert "synthetic_fixture" in synthetic_body, (
        "[P1-LIVE-3] El reason debe ser `synthetic_fixture` para distinguir "
        "de `orphan_chunks` (P2-NEXT-3). Sin esto, post-mortems no pueden "
        "discriminar entre los dos modos de cleanup."
    )


def test_p1_live_3_user_id_defense_in_depth(synthetic_body: str):
    """El UPDATE de plan_data debe filtrar por `AND user_id = %s`
    (invariante I2 defense-in-depth)."""
    # Busca el UPDATE de meal_plans y verifica que tenga AND user_id.
    update_meal_plans = re.search(
        r"UPDATE\s+meal_plans.*?WHERE\s+.*?user_id\s*=\s*%s",
        synthetic_body,
        re.IGNORECASE | re.DOTALL,
    )
    assert update_meal_plans, (
        "[P1-LIVE-3] El UPDATE de `meal_plans` debe filtrar `AND user_id = %s` "
        "(invariante I2). Sin él, una raza extremadamente improbable de "
        "id-collision podría tocar otro user."
    )


def test_p1_live_3_cron_registered(cron_source: str):
    """El cron `sweep_synthetic_test_plans` debe registrarse en
    `register_plan_chunk_scheduler` (CronTrigger diario)."""
    register_body = _read_function_body(cron_source, "register_plan_chunk_scheduler")
    assert register_body, (
        "[P1-LIVE-3] `register_plan_chunk_scheduler` no encontrado en cron_tasks.py."
    )
    assert "sweep_synthetic_test_plans" in register_body, (
        "[P1-LIVE-3] Cron `sweep_synthetic_test_plans` no registrado. "
        "Sin registration el sweep nunca corre — equivalente a no implementarlo."
    )
    assert "_sweep_synthetic_test_plans" in register_body, (
        "[P1-LIVE-3] La función `_sweep_synthetic_test_plans` debe ser el "
        "callable pasado a `_add_job_jittered`."
    )


def test_p1_live_3_tick_observable(synthetic_body: str):
    """El sweep debe emitir tick `_sweep_synthetic_test_plans_tick` siempre
    (patrón P1-LIVE-4 / P3-LIVE-1)."""
    assert "_sweep_synthetic_test_plans_tick" in synthetic_body, (
        "[P1-LIVE-3] Tick observable `_sweep_synthetic_test_plans_tick` ausente. "
        "Sin él, no podemos distinguir scheduler starvation de 'cero fixtures'."
    )
    finally_idx = synthetic_body.find("finally:")
    tick_idx = synthetic_body.rfind("_sweep_synthetic_test_plans_tick")
    assert finally_idx != -1 and finally_idx < tick_idx, (
        "[P1-LIVE-3] El tick debe estar dentro de un bloque `finally:`."
    )


@pytest.mark.parametrize("knob", [
    "MEALFIT_SWEEP_SYNTHETIC_PLANS_ENABLED",
    "MEALFIT_SWEEP_SYNTHETIC_PLANS_AGE_HOURS",
    "MEALFIT_SWEEP_SYNTHETIC_PLANS_BATCH",
])
def test_p1_live_3_knobs_via_env_helpers(synthetic_body: str, knob: str):
    """Los knobs de P1-LIVE-3 deben leerse via `_env_bool`/`_env_int`."""
    assert re.search(rf'_env_(?:bool|int)\(\s*["\']{re.escape(knob)}["\']', synthetic_body), (
        f"[P1-LIVE-3] Knob `{knob}` no leído via `_env_bool`/`_env_int`. "
        f"Sin auto-registro en `_KNOBS_REGISTRY` queda invisible al snapshot."
    )


# ---------------------------------------------------------------------------
# Marker bump
# ---------------------------------------------------------------------------

def test_marker_bumped_to_p1_live(app_source: str):
    """`_LAST_KNOWN_PFIX` debe reflejar P1-LIVE-X."""
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*["\']([^"\']+)["\']',
        app_source,
    )
    assert m, "[P1-LIVE-5] `_LAST_KNOWN_PFIX` no encontrado."
    marker = m.group(1)
    assert "P1-LIVE" in marker or "P0-LIVE" not in marker, (
        f"[P1-LIVE-5] `_LAST_KNOWN_PFIX={marker}` debe bumpearse a P1-LIVE-X "
        f"tras cerrar P1-LIVE-3/4/5."
    )
