"""[P1-23] Tests para los guards de resolución de alertas en
`_alert_chunks_paused_indefinitely`.

Bug original (audit P1-23):
  El cron emitía warning Phase 1 (`system_alerts.alert_type =
  'chunk_paused_indefinitely'`) cuando un chunk llevaba >= 12h pausado por
  `_pause_reason='missing_prior_lessons'`. Marcaba la warning como
  resuelta SOLO en la rama unblock-success (~línea 11800 de cron_tasks.py).

  Dos paths quedaban con la warning dangling indefinidamente:

  1. **Escalate path**: cuando el chunk se escalaba a dead_letter tras
     fallar el último intento de unblock, la warning quedaba abierta. Y
     peor: `_alert_new_dead_lettered_chunks` emite UNA NUEVA alerta
     critical con alert_key `dead_lettered_chunks_recent`, así que en el
     dashboard aparecían DOS señales activas para el mismo chunk.

  2. **Recovery por otro path**: si el usuario añadió pantry, otro cron
     resumió el chunk (status='pending'), o el plan fue soft-deleted, la
     warning seguía marcada como `triggered_at` activa porque ESTE cron
     ya no la reseleccionaba (filter `_pause_reason='missing_prior_lessons'`
     y `dead_lettered_at IS NULL`).

Fix:
  1. **Self-healing sweep al inicio**: UPDATE `system_alerts` resolviendo
     todas las alertas `chunk_paused_indefinitely:*` cuyo alert_key NO
     corresponda actualmente a un chunk pausado (status='pending_user_
     action' AND `_pause_reason='missing_prior_lessons'` AND no dead-letter).
     Best-effort, no bloquea el resto del cron si falla.
  2. **Resolve-on-escalate**: tras `_escalate_unrecoverable_chunk` exitoso,
     UPDATE `system_alerts SET resolved_at = NOW() WHERE alert_key = %s`.
     Cierra la ventana inmediata; la sweep también la cubriría en el
     próximo tick pero con menos delay.

Cobertura:
  - test_self_healing_sweep_runs_at_start_of_cron
  - test_sweep_filters_by_alert_type_chunk_paused_indefinitely
  - test_sweep_uses_not_exists_subquery_against_plan_chunk_queue
  - test_sweep_failure_does_not_abort_cron
  - test_resolve_on_escalate_emits_update_with_alert_key
  - test_resolve_on_escalate_uses_correct_alert_key_format
  - test_resolve_on_escalate_failure_does_not_abort_cron
  - test_documentation_p1_23_present
"""
import inspect
import json
from unittest.mock import patch

import pytest

import cron_tasks
from cron_tasks import _alert_chunks_paused_indefinitely


_SRC = inspect.getsource(cron_tasks._alert_chunks_paused_indefinitely)


def _make_candidate(
    *,
    task_id="task-1",
    user_id="user-1",
    plan_id="plan-1",
    week_number=2,
    paused_seconds=25 * 3600,  # 25h: phase 2 (escalate)
    total_days_requested=15,
    pipeline_snapshot=None,
    plan_data=None,
):
    snap = pipeline_snapshot if pipeline_snapshot is not None else {
        "_pause_reason": "missing_prior_lessons",
        "_p1_1_expected_lessons": 4,
        "_p1_1_actual_lessons": 1,
        "_p1_1_rebuilt_lessons": 1,
    }
    pd = plan_data if plan_data is not None else {"days": []}
    return {
        "task_id": task_id,
        "user_id": user_id,
        "meal_plan_id": plan_id,
        "week_number": week_number,
        "days_offset": 3,
        "days_count": 4,
        "pipeline_snapshot": snap,
        "dead_lettered_at": None,
        "paused_seconds": paused_seconds,
        "total_days_requested": total_days_requested,
        "plan_data": pd,
    }


# ---------------------------------------------------------------------------
# 1. Self-healing sweep.
# ---------------------------------------------------------------------------
def test_self_healing_sweep_runs_at_start_of_cron():
    """La sweep debe ejecutarse SIEMPRE al inicio del cron, antes de
    SELECT-ear candidatos. Verificamos vía orden de calls."""
    write_calls = []

    def _capture(sql, params=None, **kw):
        write_calls.append(sql)

    with patch("cron_tasks.execute_sql_query", return_value=[]), \
         patch("cron_tasks.execute_sql_write", side_effect=_capture), \
         patch("cron_tasks._ensure_quality_alert_schema"):
        _alert_chunks_paused_indefinitely()

    sweep_calls = [
        s for s in write_calls
        if "UPDATE system_alerts" in s
        and "alert_type = 'chunk_paused_indefinitely'" in s
        and "NOT EXISTS" in s
    ]
    assert sweep_calls, (
        f"P1-23: la self-healing sweep debe ejecutarse al inicio del cron. "
        f"Writes vistos: {[s[:120] for s in write_calls]}"
    )


def test_sweep_filters_by_alert_type_chunk_paused_indefinitely():
    """La sweep debe filtrar por `alert_type = 'chunk_paused_indefinitely'`
    para no resolver alertas de otros tipos accidentalmente."""
    assert "alert_type = 'chunk_paused_indefinitely'" in _SRC, (
        "P1-23: la sweep debe filtrar por alert_type específico — sin esto "
        "podría resolver alertas de OTROS tipos (e.g., dead_lettered_chunks_"
        "recent, degraded_rate_high, memory_summary_failures) accidentalmente."
    )


def test_sweep_uses_not_exists_subquery_against_plan_chunk_queue():
    """La sweep debe usar `NOT EXISTS` contra `plan_chunk_queue` para
    determinar si la alerta tiene chunk pausado correspondiente."""
    # Buscamos el patrón en el source.
    assert "NOT EXISTS" in _SRC, (
        "P1-23: la sweep debe usar NOT EXISTS para identificar alertas "
        "huérfanas."
    )
    assert "plan_chunk_queue" in _SRC, (
        "P1-23: la sweep debe consultar plan_chunk_queue como fuente de "
        "verdad del estado actual del chunk."
    )
    # El subquery debe replicar los criterios del SELECT principal:
    # status='pending_user_action', dead_lettered_at IS NULL,
    # _pause_reason='missing_prior_lessons'.
    assert "status = 'pending_user_action'" in _SRC, (
        "P1-23: el subquery NOT EXISTS debe verificar status="
        "'pending_user_action'."
    )
    assert "missing_prior_lessons" in _SRC, (
        "P1-23: el subquery debe verificar _pause_reason='missing_prior_lessons'."
    )


def test_sweep_failure_does_not_abort_cron():
    """Si la sweep falla (best-effort), el resto del cron debe seguir
    ejecutándose. Sin esto, una BD blip aborta TODO el ciclo de alerta."""
    candidate = _make_candidate(paused_seconds=13 * 3600)  # phase 1 only

    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        # Hacer fallar la sweep inicial (UPDATE con alert_type filter)
        # pero permitir el resto.
        if (
            "UPDATE system_alerts" in sql
            and "alert_type = 'chunk_paused_indefinitely'" in sql
            and "NOT EXISTS" in sql
        ):
            raise RuntimeError("simulated sweep DB failure")
        write_calls.append((sql, params))

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"):
        # No debe lanzar.
        _alert_chunks_paused_indefinitely()

    # El cron debió seguir hasta emitir el INSERT de Phase 1.
    insert_alerts = [
        sql for (sql, _) in write_calls
        if "INSERT INTO system_alerts" in sql
    ]
    assert insert_alerts, (
        "P1-23: el cron debe seguir emitiendo Phase 1 INSERT incluso si "
        "la sweep falló (best-effort)."
    )


# ---------------------------------------------------------------------------
# 2. Resolve-on-escalate.
# ---------------------------------------------------------------------------
def test_resolve_on_escalate_emits_update_with_alert_key():
    """Tras `_escalate_unrecoverable_chunk` exitoso, debe emitirse un
    UPDATE sobre `system_alerts` con WHERE alert_key = %s para resolver
    la warning Phase 1. Sin esto, la warning queda dangling junto con
    la nueva alerta critical de dead-letter."""
    candidate = _make_candidate(paused_seconds=25 * 3600)  # phase 2 escalate
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        write_calls.append((sql, params))

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_esc, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[]), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=[]):
        _alert_chunks_paused_indefinitely()

    mock_esc.assert_called_once()
    # Buscar el UPDATE per-chunk con WHERE alert_key (NO la sweep inicial).
    resolve_per_chunk = [
        (sql, params) for (sql, params) in write_calls
        if "UPDATE system_alerts" in sql
        and "resolved_at = NOW()" in sql
        and "WHERE alert_key" in sql
        and params is not None
    ]
    assert resolve_per_chunk, (
        f"P1-23: tras escalate debe emitirse UPDATE por alert_key. "
        f"Writes: {[s[:120] for s, _ in write_calls]}"
    )


def test_resolve_on_escalate_uses_correct_alert_key_format():
    """El alert_key del UPDATE debe matchear EXACTAMENTE el del INSERT
    Phase 1: `chunk_paused_indefinitely:{plan_id}:{week_number}`. Si el
    formato diverge, el UPDATE nunca encuentra row."""
    candidate = _make_candidate(
        plan_id="plan-xyz",
        week_number=7,
        paused_seconds=30 * 3600,
    )
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        write_calls.append((sql, params))

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._escalate_unrecoverable_chunk"), \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[]), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=[]):
        _alert_chunks_paused_indefinitely()

    resolve_per_chunk = [
        (sql, params) for (sql, params) in write_calls
        if "UPDATE system_alerts" in sql
        and "resolved_at = NOW()" in sql
        and "WHERE alert_key" in sql
        and params is not None
    ]
    assert resolve_per_chunk
    expected_key = "chunk_paused_indefinitely:plan-xyz:7"
    assert resolve_per_chunk[0][1][0] == expected_key, (
        f"P1-23: alert_key esperado={expected_key!r}, "
        f"obtenido={resolve_per_chunk[0][1][0]!r}"
    )


def test_resolve_on_escalate_failure_does_not_abort_cron():
    """Si el resolve falla (best-effort), el cron debe seguir procesando
    los siguientes candidatos. Sin esto, una alert resolve fallida bloquea
    el procesamiento de chunks subsiguientes en el batch."""
    candidate_a = _make_candidate(
        task_id="task-a", plan_id="plan-a", week_number=1,
        paused_seconds=30 * 3600,
    )
    candidate_b = _make_candidate(
        task_id="task-b", plan_id="plan-b", week_number=2,
        paused_seconds=30 * 3600,
    )
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        # Hacer fallar SOLO el resolve per-chunk del task-a, no la sweep
        # ni los demás writes.
        if (
            "UPDATE system_alerts" in sql
            and "WHERE alert_key" in sql
            and params
            and params[0] == "chunk_paused_indefinitely:plan-a:1"
        ):
            raise RuntimeError("simulated resolve failure")
        write_calls.append((sql, params))

    escalate_calls = []

    def fake_escalate(**kwargs):
        escalate_calls.append(kwargs)

    with patch(
        "cron_tasks.execute_sql_query", return_value=[candidate_a, candidate_b]
    ), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._ensure_quality_alert_schema"), \
         patch("cron_tasks._escalate_unrecoverable_chunk", side_effect=fake_escalate), \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[]), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=[]):
        _alert_chunks_paused_indefinitely()

    # Ambos chunks deben haber sido escalados (loop sobrevive al resolve fail).
    escalated_plans = sorted(c["plan_id"] for c in escalate_calls)
    assert escalated_plans == ["plan-a", "plan-b"], (
        f"P1-23: el resolve fallido del primer chunk no debe abortar el "
        f"procesamiento del siguiente. Escalated: {escalated_plans}"
    )


# ---------------------------------------------------------------------------
# 3. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_23_present():
    """Comentario `[P1-23]` debe documentar los guards de resolve."""
    full_src = inspect.getsource(cron_tasks)
    assert "[P1-23]" in full_src, (
        "P1-23: falta marker que documente sweep + resolve-on-escalate."
    )


def test_documentation_mentions_orphaned_or_dangling():
    """El comentario debe explicar QUÉ caso captura: alertas huérfanas /
    dangling. Esto ayuda al lector a no eliminar la sweep pensando que es
    redundante con el resolve original (que solo cubre unblock-success)."""
    full_src = inspect.getsource(cron_tasks)
    idx = full_src.find("[P1-23]")
    assert idx > -1
    window = full_src[idx : idx + 2500]
    needles = ["huérfana", "dangling", "stale", "self-healing", "supersede"]
    assert any(n in window.lower() for n in needles), (
        "P1-23: el comentario debe explicar el rationale (alertas huérfanas / "
        "self-healing / supersede)."
    )
