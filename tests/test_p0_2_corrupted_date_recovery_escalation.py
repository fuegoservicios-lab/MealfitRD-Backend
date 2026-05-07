"""[P0-2 FIX] Tests para escalación de chunks pausados con fecha de inicio corrupta.

Cubre el escenario "fecha corrupta no parseable":
  - Plan con `_plan_start_date` o `grocery_start_date` presentes pero con datos
    inválidos (ej. cadena gibberish, año imposible) que `safe_fromisoformat`
    rechaza. La cascada del gate intenta `created_at` pero también está dañado.
    El gate `_check_chunk_learning_ready` devuelve
    `reason='unrecoverable_corrupted_date'`.
  - Pre-fix: este reason caía al bloque genérico de deferrals
    (CHUNK_LEARNING_READY_MAX_DEFERRALS) y, tras agotar, terminaba en
    `_force_variety` o pausa genérica `empty_pantry` — el plan se generaba con
    datos incorrectos o quedaba congelado sin escalar.
  - Post-fix: el worker espejo de `missing_start_date_no_anchor` cuenta
    `_anchor_recovery_attempts` y escala a dead_letter al exceder MAX. El
    recovery cron re-intenta parsear los anchors en cada tick y reanuda si
    alguno se vuelve válido.

Ejecutar:
    cd backend && python -m pytest tests/test_p0_2_corrupted_date_recovery_escalation.py -v
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. _escalate_unrecoverable_chunk acepta el nuevo escalation_reason
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_escalate_with_corrupted_date_uses_specific_copy(mock_write, mock_push):
    """Con escalation_reason='unrecoverable_corrupted_date', el push debe llevar
    copy específico ('Tu plan necesita regenerarse' + 'datos inválidos en la
    fecha') y deeplink distinto del missing_anchor."""
    from cron_tasks import _escalate_unrecoverable_chunk

    _escalate_unrecoverable_chunk(
        task_id="task-corrupt-1",
        user_id="user-corrupt-1",
        plan_id="plan-corrupt-1",
        week_number=2,
        recovery_attempts=3,
        escalation_reason="unrecoverable_corrupted_date",
    )

    mock_push.assert_called_once()
    push_kwargs = mock_push.call_args.kwargs
    assert push_kwargs["title"] == "Tu plan necesita regenerarse"
    assert "datos inválidos" in push_kwargs["body"].lower()
    assert push_kwargs["url"] == "/dashboard?action_required=corrupted_date"


@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks.execute_sql_write")
def test_escalate_with_corrupted_date_persists_user_action_required(
    mock_write, mock_push
):
    """`_user_action_required` y `_recovery_exhausted_chunks` deben persistirse
    con reason='unrecoverable_corrupted_date' para que el frontend muestre el
    banner."""
    from cron_tasks import _escalate_unrecoverable_chunk

    _escalate_unrecoverable_chunk(
        task_id="task-corrupt-2",
        user_id="user-corrupt-2",
        plan_id="plan-corrupt-2",
        week_number=3,
        recovery_attempts=3,
        escalation_reason="unrecoverable_corrupted_date",
    )

    plan_update_call = next(
        c for c in mock_write.call_args_list
        if "UPDATE meal_plans" in c.args[0]
    )
    sql = plan_update_call.args[0]
    params = plan_update_call.args[1]
    assert "_user_action_required" in sql
    assert "_recovery_exhausted_chunks" in sql
    # reason aparece dos veces (recovery_exhausted_chunks entry + user_action_required)
    assert params.count("unrecoverable_corrupted_date") == 2


# ---------------------------------------------------------------------------
# 2. _recover_pantry_paused_chunks: reanuda si un anchor parseable apareció
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_resumes_chunk_when_corrupted_date_becomes_parseable(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Si plan_data tenía un anchor corrupto y luego una escritura externa
    (renovación, fix manual) lo reemplaza por una fecha válida, el chunk se
    reanuda como `pending` y NO escala."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-corrupt-row-1",
        "user_id": "user-CR-1",
        "meal_plan_id": "plan-CR-1",
        "week_number": 2,
        "pipeline_snapshot": {"_pantry_pause_reason": "unrecoverable_corrupted_date"},
        "paused_seconds": 600,
    }
    # psd corrupto, gsd reemplazado por fecha válida.
    anchor_row = {
        "psd": "garbage-not-a-date",
        "gsd": "2026-04-15",
        "created_at": None,
        "attempts": 1,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    # No debe escalar (anchor parseable encontrado).
    mock_escalate.assert_not_called()

    # Debe haber un UPDATE plan_chunk_queue con status='pending'.
    pending_updates = [
        c for c in mock_write.call_args_list
        if "UPDATE plan_chunk_queue" in c.args[0] and "'pending'" in c.args[0]
    ]
    assert len(pending_updates) >= 1, (
        "Chunk debe reanudarse como pending cuando un anchor parseable aparece"
    )

    # El snapshot debe llevar la marca de resolución.
    pending_payload = pending_updates[0].args[1][0]
    assert "corrupted_date_recovered" in pending_payload


# ---------------------------------------------------------------------------
# 3. _recover_pantry_paused_chunks: incrementa contador si la corrupción persiste
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_increments_attempts_when_all_anchors_unparseable(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Todos los anchors corruptos y attempts<MAX → incrementa contador, NO escala."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-corrupt-row-2",
        "user_id": "user-CR-2",
        "meal_plan_id": "plan-CR-2",
        "week_number": 3,
        "pipeline_snapshot": {"_pantry_pause_reason": "unrecoverable_corrupted_date"},
        "paused_seconds": 600,
    }
    # Todos los anchors corruptos.
    anchor_row = {
        "psd": "not-a-date",
        "gsd": "2099-13-99",
        "created_at": "also-garbage",
        "attempts": 0,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    # Debe haber UPDATE incrementando attempts.
    persist_updates = [
        c for c in mock_write.call_args_list
        if "_anchor_recovery_attempts" in c.args[0]
    ]
    assert len(persist_updates) == 1
    assert persist_updates[0].args[1][0] == 1, "attempts debe incrementarse a 1"

    # NO debe escalar todavía.
    mock_escalate.assert_not_called()


# ---------------------------------------------------------------------------
# 4. _recover_pantry_paused_chunks: escala al exceder umbral
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_escalates_corrupted_date_when_attempts_exceed_max(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Anchors corruptos y attempts>=MAX → llama a _escalate_unrecoverable_chunk
    con escalation_reason='unrecoverable_corrupted_date'."""
    from cron_tasks import _recover_pantry_paused_chunks
    from constants import CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS

    paused_row = {
        "id": "chunk-corrupt-row-3",
        "user_id": "user-CR-3",
        "meal_plan_id": "plan-CR-3",
        "week_number": 4,
        "pipeline_snapshot": {"_pantry_pause_reason": "unrecoverable_corrupted_date"},
        "paused_seconds": 99999,
    }
    anchor_row = {
        "psd": "not-a-date-anywhere",
        "gsd": "neither-this",
        "created_at": "garbage",
        "attempts": CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS - 1,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        if "_anchor_recovery_attempts" in sql:
            return anchor_row
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    mock_escalate.assert_called_once()
    call_kwargs = mock_escalate.call_args.kwargs
    assert call_kwargs["escalation_reason"] == "unrecoverable_corrupted_date"
    assert call_kwargs["plan_id"] == "plan-CR-3"
    assert call_kwargs["week_number"] == 4
    assert call_kwargs["recovery_attempts"] == CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# 5. _recover_pantry_paused_chunks: chunk sin meal_plan_id no rompe
# ---------------------------------------------------------------------------
@patch("cron_tasks._dispatch_push_notification")
@patch("cron_tasks._escalate_unrecoverable_chunk")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recover_skips_corrupted_date_chunk_when_meal_plan_id_null(
    mock_query, mock_write, mock_inv, mock_escalate, mock_push
):
    """Defense: meal_plan_id=NULL en la fila no debe crashear ni escalar."""
    from cron_tasks import _recover_pantry_paused_chunks

    paused_row = {
        "id": "chunk-corrupt-row-4",
        "user_id": "user-CR-4",
        "meal_plan_id": None,
        "week_number": 2,
        "pipeline_snapshot": {"_pantry_pause_reason": "unrecoverable_corrupted_date"},
        "paused_seconds": 600,
    }

    def query_side_effect(sql, *args, **kwargs):
        if "SELECT id, user_id, meal_plan_id" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_side_effect

    _recover_pantry_paused_chunks()

    mock_escalate.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Smoke test del invariante: el código fuente del worker tiene la rama
# ---------------------------------------------------------------------------
def test_worker_has_unrecoverable_corrupted_date_branch():
    """Contrato: el worker `process_chunk_task` (vía `_chunk_worker`) debe
    tener una rama explícita para `_defer_reason == "unrecoverable_corrupted_date"`
    que pause con `_pause_reason="unrecoverable_corrupted_date"`. Sin esa rama,
    el reason cae al deferrals genérico y el plan termina con _force_variety —
    exactamente el bug que P0-2 cierra.
    """
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cron_tasks.py",
    )
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()

    assert '_defer_reason == "unrecoverable_corrupted_date"' in source, (
        "El worker no tiene branch para 'unrecoverable_corrupted_date'. Antes "
        "del fix P0-2, este reason caía al bloque genérico de deferrals y el "
        "plan terminaba con _force_variety o pausa empty_pantry sin escalar."
    )
    assert 'pause_snapshot["_pause_reason"] = "unrecoverable_corrupted_date"' in source, (
        "El worker no estampa _pause_reason='unrecoverable_corrupted_date' en "
        "el snapshot de pausa — el recovery cron no podrá distinguir el caso."
    )
    assert 'pause_reason == "unrecoverable_corrupted_date"' in source, (
        "El recovery cron `_recover_pantry_paused_chunks` no tiene branch "
        "para `unrecoverable_corrupted_date`. Sin esa rama, el chunk caería "
        "al fallback genérico de empty_pantry y derivaría a flexible_mode tras "
        "el TTL — generando con datos inválidos."
    )
