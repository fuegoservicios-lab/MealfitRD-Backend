"""[P1-NEW-D · 2026-05-11] Cron `_recover_future_scheduled_pending_chunks`
detecta chunks `pending` con `execute_after` distante en el futuro Y fuera
del horizonte temporal del plan (plan_start + (total_days_requested + 7d)),
y los escala vía `_escalate_unrecoverable_chunk` con
`escalation_reason='execute_after_beyond_plan_window'`.

Gap cerrado:
    `_recover_failed_chunks_for_long_plans` cubre SOLO `status='failed'`.
    `_sweep_meal_plans_without_chunks` (P2-NEXT-3) EXCLUYE planes con
    chunks `pending`. Resultado: un chunk `pending` con `execute_after`
    a N meses queda inmune a ambos crons y zombifica al plan.

    Audit 2026-05-11 detectó chunk `6e0756e5-c70e-42a5-b219-5dcb94ee7680`
    con `execute_after = 2026-10-24` (~5.5 meses) sobre plan creado
    2026-05-08 (`98d902e3-...`). Sin este P-fix, el chunk dispara en
    Octubre con snapshot stale → fallo silencioso garantizado.

Drift detection:
    - Función `_recover_future_scheduled_pending_chunks` renombrada/borrada → falla.
    - SELECT relaja filtros (omite `status='pending'`, `attempts=0`,
      `dead_lettered_at IS NULL`, o el horizonte temporal) → falla.
    - CAS guard `WHERE status='pending' AND dead_lettered_at IS NULL`
      al transicionar pending→failed desaparece → falla (race con worker
      reabierta).
    - Cron no registrado en `register_plan_chunk_scheduler` → falla.
    - Nueva razón `execute_after_beyond_plan_window` desaparece del
      whitelist `ESCALATION_REASONS` → falla.
    - Branch de copy en `_escalate_unrecoverable_chunk` para esta razón
      desaparece → falla (caería al default genérico, perdiendo el
      deeplink dedicado del banner frontend).

Tooltip-anchor: P1-NEW-D-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"
_CONSTANTS = _BACKEND / "constants.py"


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
def constants_source() -> str:
    return _CONSTANTS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Razón `execute_after_beyond_plan_window` en ESCALATION_REASONS
# ---------------------------------------------------------------------------
def test_new_escalation_reason_in_whitelist(constants_source: str):
    """`execute_after_beyond_plan_window` debe figurar en el tuple
    canónico `ESCALATION_REASONS`. Sin esto,
    `_escalate_unrecoverable_chunk` rechaza la razón (P2-NEW-3 validation)
    y aborta early — el chunk queda en `failed` sin dead_letter ni alert."""
    assert '"execute_after_beyond_plan_window"' in constants_source, (
        "P1-NEW-D violation: razón canónica "
        "`execute_after_beyond_plan_window` no presente en ESCALATION_REASONS. "
        "Añadir al tuple en constants.py (junto a `recovery_exhausted`, "
        "`unrecoverable_missing_anchor`, etc.)."
    )


# ---------------------------------------------------------------------------
# 2. Función _recover_future_scheduled_pending_chunks existe
# ---------------------------------------------------------------------------
def test_recover_function_defined(cron_source: str):
    """`_recover_future_scheduled_pending_chunks` debe estar definida."""
    assert re.search(
        r"^def\s+_recover_future_scheduled_pending_chunks\s*\(",
        cron_source,
        re.MULTILINE,
    ), (
        "P1-NEW-D violation: `_recover_future_scheduled_pending_chunks` "
        "no está definida en cron_tasks.py. Sin ella, chunks `pending` "
        "con `execute_after` distante en el futuro (fuera del horizonte "
        "del plan) zombifican planes indefinidamente."
    )


# ---------------------------------------------------------------------------
# 3. SELECT filtra status='pending' + attempts=0 + dead_lettered_at IS NULL
#    + execute_after > NOW() + horizon_days
# ---------------------------------------------------------------------------
def test_select_filters_eligibility(cron_source: str):
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    assert body, "Función ausente — cubre test #2."

    assert re.search(r"status\s*=\s*['\"]pending['\"]", body), (
        "P1-NEW-D violation: SELECT no filtra `status = 'pending'`. "
        "Sin esto el cron escalaría chunks ya en `processing`/`failed`/"
        "`completed`, corrompiendo flujo del worker."
    )
    assert re.search(r"attempts\s*=\s*0", body), (
        "P1-NEW-D violation: SELECT no filtra `attempts = 0`. Sin esto "
        "podríamos escalar chunks que ya están en retry backoff legítimo."
    )
    assert re.search(r"dead_lettered_at\s+IS\s+NULL", body, re.IGNORECASE), (
        "P1-NEW-D violation: SELECT no filtra `dead_lettered_at IS NULL`. "
        "Sin esto reinvocaríamos escalación sobre chunks ya dead-lettered, "
        "duplicando push notifications."
    )
    assert re.search(
        r"execute_after\s*>\s*NOW\s*\(\s*\)\s*\+\s*make_interval\s*\(\s*days\s*=>",
        body,
        re.IGNORECASE,
    ), (
        "P1-NEW-D violation: SELECT no filtra `execute_after > NOW() + "
        "make_interval(days => HORIZON)`. Sin el horizonte temporal, "
        "escalaríamos chunks con execute_after vencido o próximo "
        "(que el worker debería procesar normalmente)."
    )


# ---------------------------------------------------------------------------
# 4. Knobs auto-registrados via _env_int
# ---------------------------------------------------------------------------
def test_knobs_registered(cron_source: str):
    """Los 2 knobs (`MEALFIT_CHUNK_FUTURE_HORIZON_DAYS`,
    `MEALFIT_CHUNK_FUTURE_HORIZON_BATCH`) DEBEN leerse vía `_env_int`
    para auto-registrarse en `_KNOBS_REGISTRY` y aparecer en
    `/health/version` (P3-NEW-D pattern)."""
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    assert re.search(
        r'_env_int\s*\(\s*["\']MEALFIT_CHUNK_FUTURE_HORIZON_DAYS["\']',
        body,
    ), (
        "P1-NEW-D violation: `MEALFIT_CHUNK_FUTURE_HORIZON_DAYS` no leído "
        "via `_env_int` (auto-registro en _KNOBS_REGISTRY). Reemplazar "
        "cualquier `os.environ.get(...)` raw."
    )
    assert re.search(
        r'_env_int\s*\(\s*["\']MEALFIT_CHUNK_FUTURE_HORIZON_BATCH["\']',
        body,
    ), (
        "P1-NEW-D violation: `MEALFIT_CHUNK_FUTURE_HORIZON_BATCH` no "
        "leído via `_env_int`. Mismo motivo."
    )


# ---------------------------------------------------------------------------
# 5. CAS guard pending→failed con filtro `status='pending' AND
#    dead_lettered_at IS NULL`
# ---------------------------------------------------------------------------
def test_cas_guard_on_state_transition(cron_source: str):
    """El UPDATE que flippea status `pending`→`failed` DEBE incluir
    `WHERE id=%s AND status='pending' AND dead_lettered_at IS NULL`.
    Sin esto, race con worker (que también filtra pending para pickup)
    podría hacer doble-procesamiento o sobreescribir status='processing'."""
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    # Buscar UPDATE … SET status = 'failed' … WHERE … status = 'pending' …
    upd = re.search(
        r"UPDATE\s+plan_chunk_queue\s+SET\s+status\s*=\s*['\"]failed['\"]"
        r".*?WHERE\s+id\s*=\s*%s"
        r".*?status\s*=\s*['\"]pending['\"]"
        r".*?dead_lettered_at\s+IS\s+NULL",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    assert upd, (
        "P1-NEW-D violation: UPDATE pending→failed sin CAS guard "
        "completo (`WHERE id=%s AND status='pending' AND "
        "dead_lettered_at IS NULL`). Riesgo: race con worker "
        "(`process_plan_chunk_queue`) que también filtra pending; sin "
        "CAS podríamos pisar `status='processing'` o re-escalar chunk "
        "ya dead-lettered."
    )


# ---------------------------------------------------------------------------
# 6. Llamada a _escalate_unrecoverable_chunk con la nueva razón
# ---------------------------------------------------------------------------
def test_calls_escalate_with_new_reason(cron_source: str):
    """Tras el CAS exitoso, el código DEBE invocar
    `_escalate_unrecoverable_chunk(...,
    escalation_reason='execute_after_beyond_plan_window')`."""
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    assert re.search(
        r"_escalate_unrecoverable_chunk\s*\(",
        body,
    ), (
        "P1-NEW-D violation: la función no invoca "
        "`_escalate_unrecoverable_chunk(...)`. Sin esta llamada, el "
        "chunk transitado a `failed` queda sin dead_lettered_at + sin "
        "push + sin alert + sin _user_action_required en plan_data — "
        "es decir, invisible al usuario."
    )
    assert re.search(
        r'escalation_reason\s*=\s*["\']execute_after_beyond_plan_window["\']',
        body,
    ), (
        "P1-NEW-D violation: el call a `_escalate_unrecoverable_chunk` "
        "no pasa `escalation_reason='execute_after_beyond_plan_window'`. "
        "Cae al else default 'recovery_exhausted' con copy genérico que "
        "no explica al usuario qué pasó (su plan se acortó vs LLM falló)."
    )


# ---------------------------------------------------------------------------
# 7. Copy block dedicado en _escalate_unrecoverable_chunk
# ---------------------------------------------------------------------------
def test_copy_block_present_in_escalate(cron_source: str):
    """`_escalate_unrecoverable_chunk` DEBE tener un branch dedicado
    `elif escalation_reason == "execute_after_beyond_plan_window":`.
    Sin el branch, la razón cae al else default y el deeplink del
    banner queda `recovery_exhausted=1` — equivocado para este caso."""
    body = _read_function_body(cron_source, "_escalate_unrecoverable_chunk")
    assert body, "_escalate_unrecoverable_chunk no encontrada."
    assert re.search(
        r'elif\s+escalation_reason\s*==\s*["\']execute_after_beyond_plan_window["\']',
        body,
    ), (
        "P1-NEW-D violation: `_escalate_unrecoverable_chunk` no contiene "
        "branch `elif escalation_reason == 'execute_after_beyond_plan_window':`. "
        "Sin el branch dedicado, el copy y la URL del deeplink caen al "
        "else default ('Tu plan necesita atención' + "
        "'?recovery_exhausted=1'), confundiendo al usuario y al banner "
        "del frontend que filtra por `action_required`."
    )
    # El branch debe setear action_url con el query param dedicado.
    assert re.search(
        r"action_required\s*=\s*execute_after_beyond_window",
        body,
    ), (
        "P1-NEW-D violation: el branch no setea "
        "`action_url='/dashboard?action_required=execute_after_beyond_window'`. "
        "Frontend banner se decide por este parámetro; sin él el banner "
        "no aparece o aparece con copy equivocado."
    )


# ---------------------------------------------------------------------------
# 8. Cron registrado en register_plan_chunk_scheduler
# ---------------------------------------------------------------------------
def test_cron_registered_in_scheduler(cron_source: str):
    """`register_plan_chunk_scheduler` DEBE registrar el cron con
    id='recover_future_scheduled_pending_chunks'."""
    register_body = _read_function_body(cron_source, "register_plan_chunk_scheduler")
    assert register_body, "register_plan_chunk_scheduler no encontrada."
    assert re.search(
        r'id\s*=\s*["\']recover_future_scheduled_pending_chunks["\']',
        register_body,
    ), (
        "P1-NEW-D violation: cron `recover_future_scheduled_pending_chunks` "
        "no está registrado en `register_plan_chunk_scheduler`. Sin "
        "registro la función nunca corre — el gap audit 2026-05-11 "
        "queda abierto."
    )
    # Y la función registrada debe ser `_recover_future_scheduled_pending_chunks`.
    assert re.search(
        r"_recover_future_scheduled_pending_chunks",
        register_body,
    ), (
        "P1-NEW-D violation: register_plan_chunk_scheduler tiene el id "
        "pero no invoca `_recover_future_scheduled_pending_chunks` "
        "como callable. Verificar pasaje a `_add_job_jittered`."
    )


# ---------------------------------------------------------------------------
# 9. Cálculo del horizonte del plan usa gracia +7 días (simetría con
#    `_recover_failed_chunks_for_long_plans:9209`)
# ---------------------------------------------------------------------------
def test_plan_window_uses_grace_plus_7_days(cron_source: str):
    """El cálculo `plan_window_end = plan_start + (total_days_requested + 7) * 1 día`
    debe usar la misma gracia +7d que `_recover_failed_chunks_for_long_plans`.
    Cualquier divergencia abre asimetría entre los dos crons hermanos."""
    body = _read_function_body(cron_source, "_recover_future_scheduled_pending_chunks")
    assert re.search(
        r"total_days_requested\s*\)\s*\+\s*7",
        body,
    ) or re.search(
        r"int\s*\(\s*total_days_requested\s*\)\s*\+\s*7",
        body,
    ), (
        "P1-NEW-D violation: cálculo del horizonte del plan no usa "
        "gracia `total_days_requested + 7`. Romper la simetría con "
        "`_recover_failed_chunks_for_long_plans:9209` abre asimetría "
        "entre crons hermanos — un chunk legítimo en el día 8 (gracia) "
        "podría ser escalado por uno y recuperado por el otro."
    )
