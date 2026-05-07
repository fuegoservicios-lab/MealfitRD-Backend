"""[P1-6] Observabilidad de learning loss — `_rebuild_last_chunk_learning_from_queue`.

Antes del fix:
    Cuando el rebuild fallaba (SQL exception, JSON corrupto, schema inválido),
    devolvía None silentemente. El chunk siguiente arrancaba con
    `_last_chunk_learning` vacío sin que nadie lo supiera. El equipo de
    operaciones se enteraba sólo cuando los usuarios se quejaban de planes
    repetitivos — síntoma post-mortem que requería análisis manual.

Después del fix:
    Cada corrupción detectada llama `_record_learning_loss(meal_plan_id, week,
    reason, user_id)` que:
      1. Persiste `_learning_corrupted_chunks: [{chunk, reason, timestamp}]`
         en `meal_plans.plan_data` (atomically, capped a 50 entries).
      2. Inserta evento `learning_rebuild_failed` en `chunk_lesson_telemetry`
         con `metadata.reason` para agregación cross-window.
      3. Loguea ERROR estructurado.

    Razones canónicas (`_P1_6_LEARNING_LOSS_REASONS`):
      - 'select_failed': excepción en SELECT (DB blip transitorio).
      - 'json_corrupted': learning_metrics presente pero no parseable como JSON.
      - 'schema_invalid': JSON parseable pero falla _validate_lesson_schema.

    `/admin/metrics` agrega learning_loss totals por reason en la ventana.

NOTA: el path "no row found" NO es learning loss — es comportamiento normal
cuando el chunk previo aún no completó (caller cae a síntesis desde plan_data.days).
"""
from unittest.mock import patch, MagicMock

import pytest


def test_canonical_reasons_constant():
    """Las razones canónicas deben mantenerse estables — `/admin/metrics`
    y crons de alerta filtran por estas strings exactas."""
    from cron_tasks import _P1_6_LEARNING_LOSS_REASONS
    expected = {"select_failed", "json_corrupted", "schema_invalid"}
    assert set(_P1_6_LEARNING_LOSS_REASONS) == expected, (
        "Cambiar las razones canónicas rompe queries de /admin/metrics y "
        "alertas downstream. Si añades una razón, agrégala aquí Y a "
        "_P1_6_LEARNING_LOSS_REASONS en cron_tasks.py."
    )


def test_record_learning_loss_persists_to_plan_data():
    """`_record_learning_loss` debe llamar `update_plan_data_atomic` para
    persistir el flag `_learning_corrupted_chunks` en plan_data. Si falla esa
    persistencia, el siguiente chunk del mismo plan no sabrá que su
    predecesor tuvo corrupción."""
    import cron_tasks
    captured = {}

    def _capture_mutator(plan_id, mutator, **kw):
        captured["plan_id"] = plan_id
        # Simular plan_data vacío inicial → el mutator lo poblara.
        result = mutator({})
        captured["mutated"] = result
        return result

    with patch("db_plans.update_plan_data_atomic", side_effect=_capture_mutator), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry"):
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=3, reason="json_corrupted", user_id="u-1",
        )

    assert captured["plan_id"] == "plan-uuid"
    losses = captured["mutated"].get("_learning_corrupted_chunks")
    assert isinstance(losses, list) and len(losses) == 1
    entry = losses[0]
    assert entry["chunk"] == 3
    assert entry["reason"] == "json_corrupted"
    assert "timestamp" in entry


def test_record_learning_loss_appends_to_existing_list():
    """Si el plan_data ya tiene `_learning_corrupted_chunks`, el helper debe
    APPEND, no sobrescribir. Sin esto, sólo veríamos la última corrupción y
    perderíamos el patrón."""
    import cron_tasks
    captured = {}

    def _capture_mutator(plan_id, mutator, **kw):
        existing = {
            "_learning_corrupted_chunks": [
                {"chunk": 1, "reason": "select_failed", "timestamp": "2026-01-01T00:00:00+00:00"},
                {"chunk": 2, "reason": "json_corrupted", "timestamp": "2026-01-02T00:00:00+00:00"},
            ]
        }
        captured["mutated"] = mutator(existing)
        return captured["mutated"]

    with patch("db_plans.update_plan_data_atomic", side_effect=_capture_mutator), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry"):
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=3, reason="schema_invalid", user_id="u-1",
        )

    losses = captured["mutated"]["_learning_corrupted_chunks"]
    assert len(losses) == 3
    assert [l["chunk"] for l in losses] == [1, 2, 3]
    assert losses[-1]["reason"] == "schema_invalid"


def test_record_learning_loss_caps_list_to_50_entries():
    """Cap FIFO a 50 entries: planes patológicos podrían acumular cientos de
    fallos consecutivos; sin cap el JSON crecería sin límite."""
    import cron_tasks
    captured = {}

    def _capture_mutator(plan_id, mutator, **kw):
        # 60 entries pre-existentes: el helper debe append y truncar a 50 (mantiene los más recientes).
        existing = {
            "_learning_corrupted_chunks": [
                {"chunk": i, "reason": "select_failed", "timestamp": f"2026-01-{i:02d}T00:00:00+00:00"}
                for i in range(1, 61)
            ]
        }
        captured["mutated"] = mutator(existing)
        return captured["mutated"]

    with patch("db_plans.update_plan_data_atomic", side_effect=_capture_mutator), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry"):
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=99, reason="json_corrupted", user_id="u-1",
        )

    losses = captured["mutated"]["_learning_corrupted_chunks"]
    assert len(losses) == 50, f"Esperaba cap de 50, got {len(losses)}"
    # El más reciente (el que acabamos de agregar) debe estar.
    assert losses[-1]["chunk"] == 99
    # Los más antiguos (chunks 1-11) se descartaron por FIFO.
    assert all(l["chunk"] >= 12 for l in losses)


def test_record_learning_loss_calls_telemetry_with_user_id():
    """Cuando user_id se proporciona, el evento se inserta en
    chunk_lesson_telemetry (event='learning_rebuild_failed') con
    metadata.reason para agregación en /admin/metrics."""
    import cron_tasks
    captured_telemetry = {}

    def _capture_telemetry(**kwargs):
        captured_telemetry.update(kwargs)
        return True

    with patch("db_plans.update_plan_data_atomic"), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry", side_effect=_capture_telemetry):
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=5, reason="schema_invalid", user_id="u-7",
        )

    assert captured_telemetry["event"] == "learning_rebuild_failed"
    assert captured_telemetry["user_id"] == "u-7"
    assert captured_telemetry["meal_plan_id"] == "plan-uuid"
    assert captured_telemetry["week_number"] == 5
    assert captured_telemetry["metadata"] == {"reason": "schema_invalid"}


def test_record_learning_loss_skips_telemetry_without_user_id():
    """Si user_id es None (ej. helper invocado desde un contexto sin user),
    el INSERT a chunk_lesson_telemetry se skipea (el schema requiere user_id NOT NULL)
    pero plan_data SÍ se actualiza — no perdemos la señal por completo."""
    import cron_tasks
    persisted = {"called": False}
    telemetry_called = {"called": False}

    def _capture_persist(*a, **kw):
        persisted["called"] = True
        return {}

    def _capture_telemetry(**kwargs):
        telemetry_called["called"] = True
        return True

    with patch("db_plans.update_plan_data_atomic", side_effect=_capture_persist), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry", side_effect=_capture_telemetry):
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=2, reason="json_corrupted", user_id=None,
        )

    assert persisted["called"], "Persist a plan_data debe ocurrir aunque user_id=None."
    assert not telemetry_called["called"], (
        "Telemetría a chunk_lesson_telemetry debe skipearse sin user_id "
        "(schema requiere user_id NOT NULL)."
    )


def test_record_learning_loss_persistence_failure_does_not_crash():
    """Best-effort: si `update_plan_data_atomic` lanza (DB blip, plan no existe,
    etc.), el helper continúa al telemetry path sin propagar la excepción.
    El log warning queda como evidencia."""
    import cron_tasks

    def _raise_persist(*a, **kw):
        raise RuntimeError("simulated DB blip")

    telemetry_called = {"called": False}

    def _capture_telemetry(**kwargs):
        telemetry_called["called"] = True
        return True

    with patch("db_plans.update_plan_data_atomic", side_effect=_raise_persist), \
         patch.object(cron_tasks, "_record_chunk_lesson_telemetry", side_effect=_capture_telemetry):
        # No debe lanzar.
        cron_tasks._record_learning_loss(
            "plan-uuid", week_number=2, reason="select_failed", user_id="u-1",
        )

    assert telemetry_called["called"], (
        "Telemetría debe ejecutarse aunque la persistencia a plan_data falle — "
        "son dos sistemas distintos, queremos best-effort en ambos."
    )


def test_rebuild_records_loss_on_select_exception():
    """Path 1 de corrupción: SELECT a plan_chunk_queue lanza excepción
    (DB unreachable, permisos)."""
    import cron_tasks

    def _raise_select(*a, **kw):
        raise RuntimeError("connection refused")

    captured = []

    def _capture_loss(meal_plan_id, week, reason, user_id=None):
        captured.append({"plan": meal_plan_id, "week": week, "reason": reason, "user_id": user_id})

    with patch.object(cron_tasks, "execute_sql_query", side_effect=_raise_select), \
         patch.object(cron_tasks, "_record_learning_loss", side_effect=_capture_loss):
        result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-uuid", target_week=3, user_id="u-1",
        )

    assert result is None
    assert len(captured) == 1
    assert captured[0]["reason"] == "select_failed"
    assert captured[0]["plan"] == "plan-uuid"
    assert captured[0]["week"] == 3
    assert captured[0]["user_id"] == "u-1"


def test_rebuild_does_not_record_loss_on_no_row():
    """Path "no data yet": chunk previo aún no completó. NO es learning loss
    — el caller cae a síntesis desde plan_data.days, comportamiento normal."""
    import cron_tasks
    captured = []

    def _capture_loss(*a, **kw):
        captured.append((a, kw))

    with patch.object(cron_tasks, "execute_sql_query", return_value=None), \
         patch.object(cron_tasks, "_record_learning_loss", side_effect=_capture_loss):
        result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-uuid", target_week=3, user_id="u-1",
        )

    assert result is None
    assert len(captured) == 0, (
        "'No row found' es comportamiento NORMAL (chunk previo aún no completó). "
        "Telemetrarlo como loss inflaría falsos positivos en /admin/metrics."
    )


def test_rebuild_records_loss_on_json_corrupted():
    """Path 2 de corrupción: learning_metrics presente pero no parseable."""
    import cron_tasks

    fake_row = {
        "week_number": 2,
        "status": "completed",
        "learning_metrics": "this is not valid JSON {{{",
    }
    captured = []

    def _capture_loss(meal_plan_id, week, reason, user_id=None):
        captured.append({"plan": meal_plan_id, "week": week, "reason": reason})

    with patch.object(cron_tasks, "execute_sql_query", return_value=fake_row), \
         patch.object(cron_tasks, "_record_learning_loss", side_effect=_capture_loss):
        result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-uuid", target_week=2, user_id="u-1",
        )

    assert result is None
    assert len(captured) == 1
    assert captured[0]["reason"] == "json_corrupted"


def test_rebuild_records_loss_on_schema_invalid():
    """Path 3 de corrupción: JSON parseable pero falla
    `_validate_lesson_schema` (NaN, inf, type mismatch)."""
    import cron_tasks

    # learning_metrics con sample_repeats como str (debería ser list) → schema invalid.
    bad_lm = {
        "learning_repeat_pct": float("nan"),  # NaN → schema invalid
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "sample_repeated_bases": [],
        "sample_repeats": [],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }
    fake_row = {
        "week_number": 4,
        "status": "completed",
        "learning_metrics": bad_lm,
    }
    captured = []

    def _capture_loss(meal_plan_id, week, reason, user_id=None):
        captured.append({"plan": meal_plan_id, "week": week, "reason": reason})

    with patch.object(cron_tasks, "execute_sql_query", return_value=fake_row), \
         patch.object(cron_tasks, "_record_learning_loss", side_effect=_capture_loss):
        result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-uuid", target_week=4, user_id="u-1",
        )

    assert result is None
    assert len(captured) == 1, (
        f"Esperaba 1 learning loss event para schema_invalid, got {len(captured)}: "
        f"{captured}. ¿Cambió la lógica de _validate_lesson_schema?"
    )
    assert captured[0]["reason"] == "schema_invalid"


def test_rebuild_does_not_record_loss_on_success():
    """Path feliz: rebuild devuelve dict válido → NO se registra learning loss."""
    import cron_tasks
    good_lm = {
        "learning_repeat_pct": 25.0,
        "ingredient_base_repeat_pct": 10.0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "sample_repeated_bases": [],
        "sample_repeats": [],
        "sample_rejection_hits": [],
        "sample_allergy_hits": [],
    }
    fake_row = {
        "week_number": 2,
        "status": "completed",
        "learning_metrics": good_lm,
    }
    captured = []

    def _capture_loss(*a, **kw):
        captured.append((a, kw))

    with patch.object(cron_tasks, "execute_sql_query", return_value=fake_row), \
         patch.object(cron_tasks, "_record_learning_loss", side_effect=_capture_loss):
        result = cron_tasks._rebuild_last_chunk_learning_from_queue(
            "plan-uuid", target_week=2, user_id="u-1",
        )

    assert result is not None
    assert result["chunk"] == 2
    assert len(captured) == 0


def test_rebuild_function_signature_includes_user_id():
    """Regression guard: el parámetro `user_id` debe ser opcional para
    backward compat (algunos call sites legacy pueden invocar sin él)."""
    import cron_tasks
    import inspect
    sig = inspect.signature(cron_tasks._rebuild_last_chunk_learning_from_queue)
    assert "user_id" in sig.parameters, (
        "user_id debe ser parámetro de _rebuild_last_chunk_learning_from_queue "
        "para que el helper pueda emitir telemetría con user identity."
    )
    user_id_param = sig.parameters["user_id"]
    assert user_id_param.default is None, (
        "user_id debe tener default None para preservar backward compat con "
        "call sites legacy que aún no lo pasan."
    )


def test_admin_metrics_includes_learning_loss_section():
    """Regression guard del endpoint: `/admin/metrics` debe exponer la sección
    `learning_loss` con `total_events`, `plans_with_loss`, `users_with_loss`,
    `by_reason`. Sin esto, ops no puede dashboard-ear la métrica."""
    from routers import plans as plans_router
    import inspect
    src = inspect.getsource(plans_router.api_admin_metrics)
    assert '"learning_loss"' in src, (
        "Endpoint /admin/metrics debe incluir la key 'learning_loss' en el "
        "response. Sin esto el dashboard no la verá."
    )
    assert "learning_rebuild_failed" in src, (
        "El query debe filtrar por event='learning_rebuild_failed' (string "
        "canónico que el helper inserta)."
    )
    assert "by_reason" in src, (
        "Breakdown por reason permite distinguir blips transitorios "
        "(select_failed) de corrupción real (json_corrupted, schema_invalid)."
    )
