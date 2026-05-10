"""[P1-CHUNKS-3] `_alert_chunks_paused_indefinitely` cubre el TTL='indefinido' del
state-machine para chunks pausados con `_pause_reason='missing_prior_lessons'`.

Tests sobre el cron en aislamiento: mock de `execute_sql_query` y
`execute_sql_write` para verificar que las dos fases se aplican correctamente:
  - Fase 1 (>=12h): alerta `system_alerts` per-chunk con severity='warning'.
  - Fase 2 (>=24h): reintenta unblock; éxito → flip a 'pending';
                    fracaso → `_escalate_unrecoverable_chunk(reason='missing_prior_lessons_unrecoverable')`.
  - Idempotencia: con `_force_unblock_attempted_at` ya seteado, no re-corre el synthesize.

Las funciones colaboradoras (`_rebuild_recent_chunk_lessons_from_queue`,
`_regenerate_recent_chunk_lessons_from_plan_days`, `_escalate_unrecoverable_chunk`,
`_record_chunk_lesson_telemetry`) se mockean para aislar la lógica de escalación.
(`_ensure_quality_alert_schema` fue eliminada en P3-B 2026-05-08 — el schema
ahora vive en `supabase/migrations/p2_new_e_consolidate_runtime_ddl.sql`.)
"""
import json
from unittest.mock import patch, MagicMock


def _make_candidate(
    *,
    task_id="task-1",
    user_id="user-1",
    plan_id="plan-1",
    week_number=2,
    days_offset=3,
    days_count=4,
    paused_seconds=13 * 3600,  # 13h: phase 1
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
        "days_offset": days_offset,
        "days_count": days_count,
        "pipeline_snapshot": snap,
        "dead_lettered_at": None,
        "paused_seconds": paused_seconds,
        "total_days_requested": total_days_requested,
        "plan_data": pd,
    }


def test_no_candidates_no_op():
    """Query vacía → ninguna llamada a write per-chunk.

    [P1-23] La self-healing sweep al inicio del cron sí emite UN UPDATE
    sobre `system_alerts` (resolver alertas huérfanas cuyo chunk ya no
    está pausado). Verificamos que cualquier write SOLO sea esa sweep,
    NO un INSERT/UPDATE per-chunk."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    writes = []

    def _capture(sql, params=None, **kw):
        writes.append((sql, params))

    with patch("cron_tasks.execute_sql_query", return_value=[]) as mock_q, \
         patch("cron_tasks.execute_sql_write", side_effect=_capture) as _mw:
        _alert_chunks_paused_indefinitely()

    assert mock_q.called
    # [P3-B 2026-05-08] removed `assert mock_schema.called` — la función
    # `_ensure_quality_alert_schema` fue eliminada (P3-B). Schema garantizado
    # por la migración SSOT, no por una llamada runtime.
    # Cualquier write debe ser la sweep P1-23 (UPDATE sobre alert_type =
    # 'chunk_paused_indefinitely' con NOT EXISTS subquery).
    non_sweep = [
        (sql, p) for (sql, p) in writes
        if not (
            "UPDATE system_alerts" in sql
            and "alert_type = 'chunk_paused_indefinitely'" in sql
            and "NOT EXISTS" in sql
        )
    ]
    assert non_sweep == [], (
        f"Sin candidatos solo la P1-23 sweep debería escribir; vio "
        f"otros writes: {[s[:120] for s, _ in non_sweep]!r}"
    )


def test_phase1_alert_only_for_chunk_in_12h_window():
    """Chunk con 13h de pausa → solo INSERT en system_alerts, sin escalate."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    candidate = _make_candidate(paused_seconds=13 * 3600)
    insert_sqls = []

    def fake_write(sql, params=None, **kwargs):
        insert_sqls.append(sql)

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue") as mock_rebuild, \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days") as mock_regen:
        _alert_chunks_paused_indefinitely()

    # Fase 2 NO debe correr (paused < 24h).
    mock_escalate.assert_not_called()
    mock_rebuild.assert_not_called()
    mock_regen.assert_not_called()

    # Debe haber al menos un INSERT en system_alerts.
    assert any("INSERT INTO system_alerts" in sql for sql in insert_sqls), (
        f"Esperaba INSERT en system_alerts, vio: {insert_sqls!r}"
    )


def test_phase2_unblock_succeeds_when_lessons_recoverable():
    """Chunk con 25h pausado y suficientes lecciones reconstruibles → flip a 'pending'."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    candidate = _make_candidate(
        paused_seconds=25 * 3600,
        total_days_requested=15,  # window_cap = 4
        plan_data={"days": [{"day": d, "meals": []} for d in range(1, 8)]},
    )
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        write_calls.append((sql, params))

    # Devolvemos suficientes lecciones combinadas (>= window_cap=4 para 15 días)
    fake_combined = [
        {"chunk": i, "synthesized_from_plan_days": False} for i in range(1, 5)
    ]

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=fake_combined), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=fake_combined), \
         patch("cron_tasks._record_chunk_lesson_telemetry"):
        _alert_chunks_paused_indefinitely()

    # Escalate NO debe haberse llamado (unblock OK).
    mock_escalate.assert_not_called()

    # Debe haber un UPDATE flipping status a 'pending'.
    flip_to_pending = [
        sql for (sql, _params) in write_calls
        if "UPDATE plan_chunk_queue" in sql and "status = 'pending'" in sql
    ]
    assert flip_to_pending, (
        f"Esperaba UPDATE flipping a 'pending'; vio: {[s[:120] for s, _ in write_calls]!r}"
    )

    # Debe haber un UPDATE escribiendo `_recent_chunk_lessons` en plan_data.
    persist_lessons = [
        sql for (sql, _params) in write_calls
        if "UPDATE meal_plans" in sql and "_recent_chunk_lessons" in sql
    ]
    assert persist_lessons, (
        f"Esperaba persistencia de _recent_chunk_lessons; vio: {[s[:120] for s, _ in write_calls]!r}"
    )


def test_phase2_escalates_when_unblock_insufficient():
    """Chunk con 25h y lecciones insuficientes → `_escalate_unrecoverable_chunk` llamado."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    candidate = _make_candidate(
        paused_seconds=25 * 3600,
        total_days_requested=15,  # window_cap = 4
        plan_data={"days": []},   # plan_data sin contenido → synthesize falla
    )
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        write_calls.append((sql, params))

    # rebuild devuelve 0 lecciones; regenerate también — caemos a escalate.
    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[]), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=[]):
        _alert_chunks_paused_indefinitely()

    # Escalate llamado con la nueva reason.
    mock_escalate.assert_called_once()
    call_kwargs = mock_escalate.call_args.kwargs
    assert call_kwargs["escalation_reason"] == "missing_prior_lessons_unrecoverable"
    assert call_kwargs["task_id"] == "task-1"
    assert call_kwargs["plan_id"] == "plan-1"
    assert call_kwargs["week_number"] == 2

    # Debe haber sellado `_force_unblock_attempted_at` en el snapshot.
    sealed = [
        params for (sql, params) in write_calls
        if "UPDATE plan_chunk_queue" in sql and "pipeline_snapshot" in sql
        and "status =" not in sql
    ]
    assert sealed, "Esperaba un UPDATE de pipeline_snapshot sellando attempt timestamp."
    # El primer param es el JSON del snapshot.
    snap_str = sealed[0][0]
    snap_obj = json.loads(snap_str)
    assert "_force_unblock_attempted_at" in snap_obj
    assert snap_obj["_force_unblock_lessons_count"] == 0


def test_phase2_idempotent_when_already_attempted():
    """Snapshot con `_force_unblock_attempted_at` → no re-corre el rebuild/synthesize."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    snap = {
        "_pause_reason": "missing_prior_lessons",
        "_p1_1_expected_lessons": 4,
        "_p1_1_actual_lessons": 1,
        "_p1_1_rebuilt_lessons": 1,
        "_force_unblock_attempted_at": "2025-01-01T00:00:00+00:00",
        "_force_unblock_lessons_count": 0,
    }
    candidate = _make_candidate(
        paused_seconds=30 * 3600,  # bien pasado el escalate threshold
        pipeline_snapshot=snap,
    )

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue") as mock_rebuild, \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days") as mock_regen:
        _alert_chunks_paused_indefinitely()

    # Synthesize NO se invoca de nuevo (idempotencia).
    mock_rebuild.assert_not_called()
    mock_regen.assert_not_called()
    # Pero SÍ va al escalate (fase 2 final).
    mock_escalate.assert_called_once()
    assert mock_escalate.call_args.kwargs["escalation_reason"] == "missing_prior_lessons_unrecoverable"


def test_phase1_alert_metadata_includes_diagnostic_fields():
    """La metadata de la alerta incluye los campos diagnósticos para SRE."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    snap = {
        "_pause_reason": "missing_prior_lessons",
        "_p1_1_expected_lessons": 5,
        "_p1_1_actual_lessons": 2,
        "_p1_1_rebuilt_lessons": 3,
    }
    candidate = _make_candidate(
        paused_seconds=15 * 3600,
        pipeline_snapshot=snap,
        total_days_requested=21,
    )
    captured_alert_params = []

    def fake_write(sql, params=None, **kwargs):
        if "INSERT INTO system_alerts" in sql:
            captured_alert_params.append(params)

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write):
        _alert_chunks_paused_indefinitely()

    assert len(captured_alert_params) == 1, (
        f"Esperaba exactamente 1 INSERT en system_alerts; vio {len(captured_alert_params)}"
    )
    params = captured_alert_params[0]
    # Layout (alert_key, title, message, metadata_json, affected_users_json)
    assert params[0] == "chunk_paused_indefinitely:plan-1:2"
    metadata = json.loads(params[3])
    assert metadata["task_id"] == "task-1"
    assert metadata["plan_id"] == "plan-1"
    assert metadata["week_number"] == 2
    assert metadata["expected_lessons"] == 5
    assert metadata["actual_lessons"] == 2
    assert metadata["rebuilt_lessons"] == 3
    assert metadata["total_days_requested"] == 21
    assert metadata["paused_hours"] == 15.0


def test_phase1_does_not_select_dead_lettered_chunks():
    """El SELECT debe filtrar `dead_lettered_at IS NULL` para no re-procesar."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    captured_select_sql = []

    def fake_query(sql, params=None, **kwargs):
        captured_select_sql.append(sql)
        return []

    with patch("cron_tasks.execute_sql_query", side_effect=fake_query), \
         patch("cron_tasks.execute_sql_write"):
        _alert_chunks_paused_indefinitely()

    assert len(captured_select_sql) == 1
    sql = captured_select_sql[0]
    assert "status = 'pending_user_action'" in sql
    assert "_pause_reason" in sql and "missing_prior_lessons" in sql
    assert "dead_lettered_at IS NULL" in sql


def test_phase2_unblock_resolves_alert():
    """Tras unblock OK, la alerta previa se marca como `resolved_at = NOW()`."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    candidate = _make_candidate(
        paused_seconds=25 * 3600,
        total_days_requested=15,
    )
    write_calls = []

    def fake_write(sql, params=None, **kwargs):
        write_calls.append((sql, params))

    fake_combined = [
        {"chunk": i, "synthesized_from_plan_days": False} for i in range(1, 5)
    ]

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write", side_effect=fake_write), \
         patch("cron_tasks._escalate_unrecoverable_chunk"), \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=fake_combined), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=fake_combined), \
         patch("cron_tasks._record_chunk_lesson_telemetry"):
        _alert_chunks_paused_indefinitely()

    # [P1-23] Hay dos tipos de UPDATEs sobre system_alerts:
    #  (a) la sweep al inicio del cron (alert_type filter, NOT EXISTS, params=None)
    #  (b) la resolve per-chunk tras unblock-success (WHERE alert_key = %s)
    # Filtramos al tipo (b) — el que valida el path de resolve original.
    resolved_per_chunk = [
        (sql, params) for (sql, params) in write_calls
        if "UPDATE system_alerts" in sql
        and "resolved_at = NOW()" in sql
        and "WHERE alert_key" in sql
        and params is not None
    ]
    assert resolved_per_chunk, (
        f"Esperaba UPDATE marcando system_alerts.resolved_at por alert_key tras unblock; "
        f"vio: {[s[:120] for s, _ in write_calls]!r}"
    )
    # El alert_key debe matchear el del INSERT inicial.
    assert resolved_per_chunk[0][1][0] == "chunk_paused_indefinitely:plan-1:2"


def test_phase2_escalation_uses_correct_recovery_attempts():
    """En el escalate path, `recovery_attempts` debe leerse de `_p1_1_rebuilt_lessons`."""
    from cron_tasks import _alert_chunks_paused_indefinitely

    snap = {
        "_pause_reason": "missing_prior_lessons",
        "_p1_1_expected_lessons": 4,
        "_p1_1_actual_lessons": 1,
        "_p1_1_rebuilt_lessons": 2,
    }
    candidate = _make_candidate(
        paused_seconds=25 * 3600,
        pipeline_snapshot=snap,
    )

    with patch("cron_tasks.execute_sql_query", return_value=[candidate]), \
         patch("cron_tasks.execute_sql_write"), \
         patch("cron_tasks._escalate_unrecoverable_chunk") as mock_escalate, \
         patch("cron_tasks._rebuild_recent_chunk_lessons_from_queue", return_value=[]), \
         patch("cron_tasks._regenerate_recent_chunk_lessons_from_plan_days", return_value=[]):
        _alert_chunks_paused_indefinitely()

    mock_escalate.assert_called_once()
    assert mock_escalate.call_args.kwargs["recovery_attempts"] == 2
