"""[P0-4] Tests para el guard proactivo de nevera vacía al enqueue de chunks.

Antes el worker descubría reactivamente que `current_pantry_ingredients` estaba
por debajo de `CHUNK_MIN_FRESH_PANTRY_ITEMS` solo al picking del chunk (línea
~13193 de cron_tasks.py); el plan quedaba en estado "fantasma" en la UI durante
horas/días hasta el pickup. Ahora `_enqueue_plan_chunk` prueba `get_user_inventory_net`
al ENQUEUE: si los items vivos no alcanzan el mínimo, flipea el chunk a
`pending_user_action` con `reason='empty_pantry_proactive'` y dispara push
inmediato. El recovery cron re-probea cada tick y reanuda en cuanto el usuario
restockea.

Casos:
  1. _enqueue_plan_chunk con week_number > 1 + nevera vacía → flip a
     pending_user_action con reason='empty_pantry_proactive'.
  2. _enqueue_plan_chunk con nevera suficiente → no flip.
  3. _enqueue_plan_chunk con week_number == 1 → no probe (initial chunk).
  4. _enqueue_plan_chunk con chunk_kind == 'initial_plan' → no probe.
  5. _enqueue_plan_chunk con CHUNK_PANTRY_PROACTIVE_GUARD=False → comportamiento
     legacy.
  6. _enqueue_plan_chunk cuando live fetch falla → no flip (conservador).
  7. _enqueue_plan_chunk en reactivación de chunk failed → no probe.
  8. P0-4 toma precedencia sobre P0-3 (si pantry vacío, no se ejecuta zero-log).
  9. Recovery cron: pantry restockeado → reanuda chunk.
 10. Recovery cron: pantry sigue vacío → no reanuda, cae a TTL.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))


# ---------- _enqueue_plan_chunk: integración del guard proactivo ----------

@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_pauses_chunk_when_pantry_empty(
    mock_write, mock_query, mock_resolve, mock_detect_zl, mock_pause
):
    """Chunk no-inicial con week_number=2 + nevera vacía → flip a
    pending_user_action con reason='empty_pantry_proactive', y P0-3 no corre."""
    import cron_tasks
    from constants import CHUNK_MIN_FRESH_PANTRY_ITEMS

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, -300, "snapshot")
    mock_query.return_value = {"id": "chunk-p04-empty", "status": "pending", "inserted": True}

    # nevera vacía: 1 item solo (< min=3)
    with patch("db_inventory.get_user_inventory_net", return_value=["sal"]):
        snapshot = {"form_data": {"_plan_start_date": start_dt.isoformat()}}
        cron_tasks._enqueue_plan_chunk(
            user_id="u-p04-empty",
            meal_plan_id="plan-p04-empty",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot=snapshot,
            chunk_kind="rolling_refill",
        )

    # Helper de pausa debe ser invocado con reason='empty_pantry_proactive'.
    assert mock_pause.called, "esperaba _pause_chunk_for_pantry_refresh invocado"
    pause_args = mock_pause.call_args
    # El reason puede llegar como kwarg o último positional
    reason = pause_args.kwargs.get("reason") or (
        pause_args.args[4] if len(pause_args.args) > 4 else None
    )
    assert reason == "empty_pantry_proactive", (
        f"esperaba reason='empty_pantry_proactive'; got {reason!r}"
    )
    # P0-3 zero-log probe NO debe correr (P0-4 retornó early).
    assert not mock_detect_zl.called, (
        "P0-4 debe tomar precedencia: P0-3 no debe correr cuando pantry vacío"
    )


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_no_pause_when_pantry_sufficient(
    mock_write, mock_query, mock_resolve, mock_detect_zl, mock_pause
):
    """Nevera con items suficientes → no flip, y P0-3 corre normalmente."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "chunk-p04-ok", "status": "pending", "inserted": True}
    mock_detect_zl.return_value = None  # zero-log no detectado

    # Nevera con ingredientes meaningful (≥3): pollo, arroz, brócoli, cebolla
    with patch(
        "db_inventory.get_user_inventory_net",
        return_value=["500g pollo", "300g arroz", "200g brocoli", "1 cebolla"],
    ):
        cron_tasks._enqueue_plan_chunk(
            user_id="u-p04-ok",
            meal_plan_id="plan-p04-ok",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )

    assert not mock_pause.called, "no debe pausar cuando nevera tiene items suficientes"
    # P0-3 SÍ debe ejecutarse (chunk no-inicial, primer INSERT)
    assert mock_detect_zl.called


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_for_initial_plan(
    mock_write, mock_query, mock_resolve, mock_pause
):
    """chunk_kind == 'initial_plan' → no probe (validation in routers/plans.py)."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-init", "status": "pending", "inserted": True}

    with patch("db_inventory.get_user_inventory_net") as mock_inv:
        cron_tasks._enqueue_plan_chunk(
            user_id="u-init",
            meal_plan_id="plan-init",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="initial_plan",
        )
        assert not mock_inv.called, "initial_plan no debe probar nevera al enqueue"
    assert not mock_pause.called


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_for_first_week(
    mock_write, mock_query, mock_resolve, mock_pause
):
    """week_number == 1 → no probe."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-w1", "status": "pending", "inserted": True}

    with patch("db_inventory.get_user_inventory_net") as mock_inv:
        cron_tasks._enqueue_plan_chunk(
            user_id="u-w1",
            meal_plan_id="plan-w1",
            week_number=1,
            days_offset=0,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )
        assert not mock_inv.called
    assert not mock_pause.called


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_when_flag_disabled(
    mock_write, mock_query, mock_resolve, mock_pause
):
    """CHUNK_PANTRY_PROACTIVE_GUARD=False → comportamiento legacy."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-off", "status": "pending", "inserted": True}

    with patch("cron_tasks.CHUNK_PANTRY_PROACTIVE_GUARD", False):
        with patch("db_inventory.get_user_inventory_net") as mock_inv:
            cron_tasks._enqueue_plan_chunk(
                user_id="u-off",
                meal_plan_id="plan-off",
                week_number=2,
                days_offset=3,
                days_count=3,
                pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
                chunk_kind="rolling_refill",
            )
            assert not mock_inv.called
    assert not mock_pause.called


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._detect_proactive_zero_log_at_boundary")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_conservative_when_live_fetch_fails(
    mock_write, mock_query, mock_resolve, mock_detect_zl, mock_pause
):
    """Si live fetch falla, NO pausamos — preferimos dejar el chunk en pending y
    que el worker decida con su fallback. Aceptar info parcial sería peor UX."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-fail", "status": "pending", "inserted": True}
    mock_detect_zl.return_value = None

    with patch("db_inventory.get_user_inventory_net", side_effect=Exception("DB blip")):
        cron_tasks._enqueue_plan_chunk(
            user_id="u-fail",
            meal_plan_id="plan-fail",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )

    assert not mock_pause.called, "live fetch fail no debe disparar pausa"
    # P0-3 SÍ debe correr (no fue bloqueado por P0-4)
    assert mock_detect_zl.called


@patch("cron_tasks._pause_chunk_for_pantry_refresh")
@patch("cron_tasks._resolve_chunk_start_anchor")
@patch("cron_tasks.execute_sql_query")
@patch("cron_tasks.execute_sql_write")
def test_enqueue_skips_probe_for_chunk_reactivation(
    mock_write, mock_query, mock_resolve, mock_pause
):
    """UPSERT que retorna inserted=False (reactivación de chunk failed) no
    debe probar — ya tuvo una pasada anterior."""
    import cron_tasks

    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    mock_resolve.return_value = (start_dt, 0, "snapshot")
    mock_query.return_value = {"id": "c-react", "status": "pending", "inserted": False}

    with patch("db_inventory.get_user_inventory_net") as mock_inv:
        cron_tasks._enqueue_plan_chunk(
            user_id="u-react",
            meal_plan_id="plan-react",
            week_number=2,
            days_offset=3,
            days_count=3,
            pipeline_snapshot={"form_data": {"_plan_start_date": start_dt.isoformat()}},
            chunk_kind="rolling_refill",
        )
        assert not mock_inv.called
    assert not mock_pause.called


# ---------- Recovery cron: empty_pantry_proactive ----------

@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_resumes_chunk_when_pantry_restocked(
    mock_query, mock_write, mock_get_inv, mock_activate
):
    """Recovery cron + pantry actualizado → reanuda chunk como 'pending' con
    snapshot enriquecido."""
    import cron_tasks

    paused_row = {
        "id": "chunk-rec-rest",
        "user_id": "u-rest",
        "meal_plan_id": "plan-rest",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "empty_pantry_proactive",
            "_pantry_pause_started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=20)
            ).isoformat(),
            "_pantry_pause_reminders": 0,
            "_pantry_pause_ttl_hours": 12,
            "_pantry_pause_reminder_hours": 6,
            "form_data": {},
        },
        "paused_seconds": 1200,
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_router
    # Live fetch ahora con items suficientes
    mock_get_inv.return_value = ["500g pollo", "300g arroz", "200g brocoli", "1 cebolla"]

    cron_tasks._recover_pantry_paused_chunks()

    # Confirmar UPDATE que reanuda como 'pending' con form_data poblado
    resume_calls = [
        c for c in mock_write.call_args_list
        if c.args
        and "status = 'pending'" in c.args[0]
        and "execute_after = NOW()" in c.args[0]
    ]
    assert resume_calls, (
        f"esperaba UPDATE reanudando como pending; "
        f"recibí: {[c.args[0][:100] for c in mock_write.call_args_list]}"
    )
    payload = json.loads(resume_calls[0].args[1][0])
    assert payload["_pantry_pause_resolution"] == "pantry_restocked"
    assert "current_pantry_ingredients" in payload["form_data"]
    assert len(payload["form_data"]["current_pantry_ingredients"]) >= 3


@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_does_not_resume_when_pantry_still_empty(
    mock_query, mock_write, mock_get_inv, mock_activate
):
    """Recovery cron + pantry sigue vacío → no reanuda; cae al TTL/reminders."""
    import cron_tasks

    paused_row = {
        "id": "chunk-rec-empty",
        "user_id": "u-empty",
        "meal_plan_id": "plan-empty",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "empty_pantry_proactive",
            "_pantry_pause_started_at": (
                datetime.now(timezone.utc) - timedelta(minutes=20)
            ).isoformat(),
            "_pantry_pause_reminders": 0,
            "_pantry_pause_ttl_hours": 12,
            "_pantry_pause_reminder_hours": 6,
            "form_data": {},
        },
        "paused_seconds": 1200,  # 20 min, debajo de TTL
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_router
    mock_get_inv.return_value = ["sal"]  # Sigue vacío

    cron_tasks._recover_pantry_paused_chunks()

    # NO debe haber UPDATE reanudando como 'pending' con execute_after=NOW().
    resume_calls = [
        c for c in mock_write.call_args_list
        if c.args
        and "status = 'pending'" in c.args[0]
        and "execute_after = NOW()" in c.args[0]
    ]
    assert not resume_calls, "no debe reanudar chunk con pantry aún vacío"
    # Y no debe haber escalado a flexible_mode aún (TTL no agotado).
    assert not mock_activate.called


@patch("cron_tasks._activate_flexible_mode")
@patch("cron_tasks.get_user_inventory_net")
@patch("cron_tasks.execute_sql_write")
@patch("cron_tasks.execute_sql_query")
def test_recovery_does_not_resume_when_live_fetch_fails(
    mock_query, mock_write, mock_get_inv, mock_activate
):
    """Live fetch lanza excepción → no reanuda; espera próximo tick."""
    import cron_tasks

    paused_row = {
        "id": "chunk-rec-blip",
        "user_id": "u-blip",
        "meal_plan_id": "plan-blip",
        "week_number": 2,
        "pipeline_snapshot": {
            "_pantry_pause_reason": "empty_pantry_proactive",
            "_pantry_pause_started_at": datetime.now(timezone.utc).isoformat(),
            "_pantry_pause_reminders": 0,
            "_pantry_pause_ttl_hours": 12,
            "_pantry_pause_reminder_hours": 6,
            "form_data": {},
        },
        "paused_seconds": 600,
    }

    def query_router(sql, params=None, fetch_one=False, fetch_all=False, **kwargs):
        if "FROM plan_chunk_queue" in sql and "pending_user_action" in sql:
            return [paused_row]
        return None

    mock_query.side_effect = query_router
    mock_get_inv.side_effect = Exception("DB blip")

    cron_tasks._recover_pantry_paused_chunks()

    resume_calls = [
        c for c in mock_write.call_args_list
        if c.args
        and "status = 'pending'" in c.args[0]
        and "execute_after = NOW()" in c.args[0]
    ]
    assert not resume_calls
