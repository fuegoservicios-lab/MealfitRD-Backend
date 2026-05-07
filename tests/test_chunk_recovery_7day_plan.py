"""
[P0-1-RECOVERY] Tests para `_recover_failed_chunks_for_long_plans`.

Valida los tres invariantes nuevos del cron de recuperación:
  A. Planes de 7 días entran al SCOPE de recovery (antes solo 15d+).
  B. Cada recovery incrementa `learning_metrics.recovery_attempts` en plan_chunk_queue.
  C. Tras `CHUNK_MAX_RECOVERY_ATTEMPTS` ciclos, el chunk se escala (dead_letter +
     anotación en plan_data + push notification al usuario), SIN re-encolarlo.
  D. La query SQL filtra por `updated_at < NOW() - min_age` para no atajar
     el backoff exponencial normal.
"""
import json
import sys
from unittest.mock import patch, MagicMock

# El módulo importa langgraph indirectamente vía graph_orchestrator; lo neutralizamos
# en entornos sin la dependencia.
sys.modules.setdefault('langgraph', MagicMock())
sys.modules.setdefault('langgraph.graph', MagicMock())
sys.modules.setdefault('langgraph.graph.message', MagicMock())

from cron_tasks import _recover_failed_chunks_for_long_plans  # noqa: E402
from constants import (  # noqa: E402
    CHUNK_MAX_RECOVERY_ATTEMPTS,
    CHUNK_RECOVERY_MIN_AGE_MINUTES,
    CHUNK_RECOVERY_BATCH_LIMIT,
)


def _make_failed_row(week_number: int = 2, recovery_attempts: int = 0, total_days: int = 7):
    return {
        "id": f"task-{week_number}",
        "user_id": "user-abc",
        "meal_plan_id": "plan-xyz",
        "week_number": week_number,
        "days_offset": 3,
        "days_count": 4,
        "pipeline_snapshot": json.dumps({
            "form_data": {"_plan_start_date": "2026-05-01"}
        }),
        "learning_metrics": {"recovery_attempts": recovery_attempts} if recovery_attempts else {},
        "total_days": total_days,
        # [P0-1-RECOVERY/A] columna unificada: el cron usa COALESCE(plan_data->>'grocery_start_date',
        # meal_plans.created_at). El test no diferencia el origen.
        "plan_start_effective": "2026-05-01",
    }


def test_a_7day_plan_chunk_is_recoverable():
    """[P0-1] Un chunk failed de un plan de 7 días aparece en la query y se re-encola."""
    captured_queries = []

    def fake_query(sql, params=None, fetch_all=False, fetch_one=False):
        captured_queries.append((sql, params))
        return [_make_failed_row(week_number=2)]

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query) as mock_q, \
         patch("cron_tasks.execute_sql_write") as mock_w, \
         patch("cron_tasks._enqueue_plan_chunk") as mock_enqueue, \
         patch("cron_tasks._detect_and_escalate_stuck_chunks"), \
         patch("cron_tasks._dispatch_push_notification") as mock_push, \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate:
        _recover_failed_chunks_for_long_plans()

    # Re-encolado exactamente una vez como catchup, NO escalado.
    assert mock_enqueue.call_count == 1, "El chunk debió re-encolarse"
    assert mock_enqueue.call_args.kwargs.get("chunk_kind") == "catchup"
    assert mock_escalate.call_count == 0, "No debió escalar (recovery_attempts < cap)"
    assert mock_push.call_count == 0, "No debió enviar push (no es escalación)"

    # Query SELECT pidió min_age + límite de batch correctos.
    select_calls = [c for c in captured_queries if "SELECT" in c[0] and "plan_chunk_queue" in c[0]]
    assert select_calls, "Debió ejecutarse la query SELECT de candidatos"
    sql_text, sql_params = select_calls[0]
    assert "make_interval(mins => %s)" in sql_text, "Query debe filtrar por min_age"
    assert "(p.plan_data->>'total_days_requested')::int >= 7" in sql_text, "Scope debe ser 7d+"
    # [P0-1-RECOVERY/A] El filtro de fecha debe usar COALESCE con created_at como fallback.
    assert "COALESCE(" in sql_text and "p.created_at" in sql_text, \
        "Query debe usar COALESCE(grocery_start_date, p.created_at) para tolerar planes sin start_date"
    assert sql_params[0] == CHUNK_RECOVERY_MIN_AGE_MINUTES
    assert sql_params[1] == CHUNK_RECOVERY_BATCH_LIMIT


def test_b_recovery_attempts_counter_incremented():
    """[P0-1] El cron escribe recovery_attempts=N+1 en learning_metrics antes de re-encolar."""
    rows = [_make_failed_row(week_number=2, recovery_attempts=0)]

    update_calls = []

    def fake_write(sql, params=None, returning=False):
        update_calls.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._enqueue_plan_chunk"), \
         patch("cron_tasks._detect_and_escalate_stuck_chunks"):
        _recover_failed_chunks_for_long_plans()

    bump_updates = [
        c for c in update_calls
        if "UPDATE plan_chunk_queue" in c[0] and "recovery_attempts" in c[0]
    ]
    assert bump_updates, "Debió ejecutarse el UPDATE de recovery_attempts"
    sql_text, sql_params = bump_updates[0]
    assert sql_params[0] == 1, f"Debe persistir recovery_attempts=1, fue {sql_params[0]}"
    assert sql_params[1] == "task-2"


def test_c_escalation_after_max_recovery_attempts():
    """[P0-1] Tras CHUNK_MAX_RECOVERY_ATTEMPTS ciclos, escalar (dead_letter + push) y NO re-encolar."""
    rows = [_make_failed_row(week_number=2, recovery_attempts=CHUNK_MAX_RECOVERY_ATTEMPTS)]

    escalation_writes = []

    def fake_write(sql, params=None, returning=False):
        escalation_writes.append((sql, params))
        return None

    with patch("cron_tasks.execute_sql_query", return_value=rows), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._enqueue_plan_chunk") as mock_enqueue, \
         patch("cron_tasks._detect_and_escalate_stuck_chunks"), \
         patch("cron_tasks._dispatch_push_notification") as mock_push:
        _recover_failed_chunks_for_long_plans()

    assert mock_enqueue.call_count == 0, "Chunk agotado NO debe re-encolarse"
    assert mock_push.call_count == 1, "Debe enviarse push de escalación al usuario"
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["user_id"] == "user-abc"
    assert "regener" in push_kwargs["body"].lower() or "regener" in push_kwargs["title"].lower()

    dead_letter_updates = [
        c for c in escalation_writes
        if "UPDATE plan_chunk_queue" in c[0] and "dead_lettered_at" in c[0]
    ]
    assert dead_letter_updates, "Debe marcarse dead_lettered_at en el chunk escalado"

    plan_data_updates = [
        c for c in escalation_writes
        if "UPDATE meal_plans" in c[0] and "_recovery_exhausted_chunks" in c[0]
    ]
    assert plan_data_updates, "Debe anotarse week_number escalado en plan_data"


def test_d_no_candidates_short_circuits_safely():
    """[P0-1] Sin candidatos, no se hacen escrituras pero sí se corre el detector de stuck chunks."""
    with patch("cron_tasks.execute_sql_query", return_value=[]), \
         patch("cron_tasks.execute_sql_write") as mock_write, \
         patch("cron_tasks._enqueue_plan_chunk") as mock_enqueue, \
         patch("cron_tasks._detect_and_escalate_stuck_chunks") as mock_stuck:
        _recover_failed_chunks_for_long_plans()

    assert mock_enqueue.call_count == 0
    assert mock_write.call_count == 0
    assert mock_stuck.call_count == 1, "Debe correr _detect_and_escalate_stuck_chunks aunque no haya candidatos"
