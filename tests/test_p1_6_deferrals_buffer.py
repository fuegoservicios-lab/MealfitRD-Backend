"""
Tests P1-6: Buffer local jsonl para deferrals telemetry cuando DB falla.

Antes: si la INSERT a `chunk_deferrals` lanzaba excepción (DB blip, schema
corrupto), el deferral se perdía silenciosamente. `_detect_chronic_deferrals`
no podía ver el patrón si DB tuvo un outage durante la ráfaga.

Cambios P1-6:
  - Constantes: CHUNK_DEFERRALS_BUFFER_PATH, _FLUSH_INTERVAL_MINUTES, _MAX_RECORDS.
  - `_append_deferral_to_buffer(record)` escribe al jsonl con FIFO cap.
  - `_record_chunk_deferral` llama al buffer cuando INSERT falla.
  - `_flush_pending_deferrals()` cron re-intenta INSERTs y mantiene archivo
     solo con records que aún no se pudieron persistir.

Cubre:
  1. _append: archivo nuevo + cap FIFO.
  2. _flush: archivo no existe → no-op.
  3. _flush: registros válidos → INSERT ok → archivo eliminado.
  4. _flush: registro con meal_plan_id None → descartar (NOT NULL en schema).
  5. _flush: error de schema (NOT NULL/syntax) → descartar permanentemente.
  6. _flush: error de DB transitorio → mantener para retry.
  7. _record_chunk_deferral: fallo de INSERT → buffer recibe el record.
  8. _record_chunk_deferral con meal_plan_id None: NO se buffea (descarta inline).
  9. JSON corrupto en archivo: descartar línea sin crashear.
"""
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def tmp_buffer_path(tmp_path):
    """Path temporal para el buffer; se limpia automáticamente."""
    buf = tmp_path / "deferrals_pending.jsonl"
    yield str(buf)
    # cleanup automático con tmp_path


# ---------------------------------------------------------------------------
# _append_deferral_to_buffer
# ---------------------------------------------------------------------------
def test_p1_6_append_creates_file_with_record(tmp_buffer_path):
    import cron_tasks
    record = {
        "user_id": "user-1",
        "meal_plan_id": "plan-1",
        "week_number": 2,
        "reason": "test_reason",
        "buffered_at": "2026-05-01T00:00:00+00:00",
    }
    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        result = cron_tasks._append_deferral_to_buffer(record)
    assert result is True
    assert os.path.exists(tmp_buffer_path)
    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["user_id"] == "user-1"
    assert parsed["meal_plan_id"] == "plan-1"


def test_p1_6_append_appends_to_existing_file(tmp_buffer_path):
    import cron_tasks
    rec1 = {"user_id": "u1", "meal_plan_id": "p1", "week_number": 1, "reason": "r1"}
    rec2 = {"user_id": "u2", "meal_plan_id": "p2", "week_number": 2, "reason": "r2"}
    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        cron_tasks._append_deferral_to_buffer(rec1)
        cron_tasks._append_deferral_to_buffer(rec2)
    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["user_id"] == "u1"
    assert json.loads(lines[1])["user_id"] == "u2"


def test_p1_6_append_fifo_cap_drops_oldest(tmp_buffer_path):
    """Si el archivo ya tiene >= MAX_RECORDS, las líneas más viejas se descartan."""
    import cron_tasks
    # Pre-poblar con 5 records
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"user_id": f"u{i}", "meal_plan_id": f"p{i}", "week_number": 1, "reason": "old"}) + "\n")

    new_rec = {"user_id": "u_new", "meal_plan_id": "p_new", "week_number": 99, "reason": "new"}
    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        with patch("constants.CHUNK_DEFERRALS_BUFFER_MAX_RECORDS", 3):
            cron_tasks._append_deferral_to_buffer(new_rec)

    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # Después del cap (3) y append, debe haber 3 records: los 2 más recientes pre-existentes + el nuevo
    assert len(lines) == 3
    assert json.loads(lines[-1])["user_id"] == "u_new"
    assert json.loads(lines[0])["user_id"] == "u3"  # los más viejos (u0, u1, u2) se cayeron


# ---------------------------------------------------------------------------
# _flush_pending_deferrals
# ---------------------------------------------------------------------------
def test_p1_6_flush_no_file_is_noop(tmp_buffer_path):
    import cron_tasks
    # Asegurar que no existe
    if os.path.exists(tmp_buffer_path):
        os.remove(tmp_buffer_path)
    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()
    assert stats == {"flushed": 0, "remaining": 0, "discarded_invalid": 0}


@patch("cron_tasks.execute_sql_write")
def test_p1_6_flush_succeeds_and_removes_file(mock_write, tmp_buffer_path):
    """Si todos los INSERTs son ok, el archivo debe eliminarse."""
    import cron_tasks
    # Pre-poblar
    records = [
        {"user_id": "u1", "meal_plan_id": "p1", "week_number": 1, "reason": "r1", "buffered_at": "2026-05-01T00:00:00+00:00"},
        {"user_id": "u2", "meal_plan_id": "p2", "week_number": 2, "reason": "r2", "buffered_at": "2026-05-01T00:01:00+00:00"},
    ]
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["flushed"] == 2
    assert stats["remaining"] == 0
    assert not os.path.exists(tmp_buffer_path), "Archivo debe eliminarse tras flush total"
    assert mock_write.call_count == 2


@patch("cron_tasks.execute_sql_write")
def test_p1_6_flush_discards_records_with_null_meal_plan_id(mock_write, tmp_buffer_path):
    """meal_plan_id None → descarta sin intentar INSERT (NOT NULL en schema)."""
    import cron_tasks
    records = [
        {"user_id": "u1", "meal_plan_id": None, "week_number": 1, "reason": "r1"},  # invalid
        {"user_id": "u2", "meal_plan_id": "p2", "week_number": 2, "reason": "r2"},  # valid
    ]
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 1
    assert stats["flushed"] == 1
    # Solo se intentó UN INSERT (el válido); el inválido se descartó antes.
    assert mock_write.call_count == 1


@patch("cron_tasks.execute_sql_write")
def test_p1_6_flush_discards_on_schema_violation(mock_write, tmp_buffer_path):
    """INSERT lanza 'violates not-null constraint' → descartar permanentemente."""
    import cron_tasks
    mock_write.side_effect = Exception("null value in column violates not-null constraint")

    rec = {"user_id": "u1", "meal_plan_id": "p1", "week_number": 1, "reason": "r1"}
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 1
    assert stats["remaining"] == 0
    assert not os.path.exists(tmp_buffer_path)


@patch("cron_tasks.execute_sql_write")
def test_p1_6_flush_keeps_records_on_transient_error(mock_write, tmp_buffer_path):
    """Errores transitorios (timeout, conexión) → mantener record para próximo flush."""
    import cron_tasks
    mock_write.side_effect = Exception("connection timeout")

    rec = {"user_id": "u1", "meal_plan_id": "p1", "week_number": 1, "reason": "r1"}
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["remaining"] == 1
    assert stats["flushed"] == 0
    assert os.path.exists(tmp_buffer_path), "Record debe seguir en archivo para retry"
    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert len(lines) == 1


def test_p1_6_flush_handles_corrupted_json_lines(tmp_buffer_path):
    """Líneas JSON corruptas se descartan sin crashear."""
    import cron_tasks
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write('{"valid": false, but malformed\n')

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 2
    assert not os.path.exists(tmp_buffer_path)


# ---------------------------------------------------------------------------
# Integración: _record_chunk_deferral → buffer
# ---------------------------------------------------------------------------
@patch("cron_tasks.execute_sql_write")
def test_p1_6_record_failure_persists_to_buffer(mock_write, tmp_buffer_path):
    """Cuando _record_chunk_deferral falla, el record debe persistirse al buffer."""
    import cron_tasks
    mock_write.side_effect = Exception("DB outage")

    # Resetear contador para no spamear logs de tests previos
    cron_tasks._chunk_deferral_telemetry_failures["count"] = 0

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        result = cron_tasks._record_chunk_deferral(
            user_id="user-buffered",
            meal_plan_id="plan-buffered",
            week_number=3,
            reason="test_buffered_reason",
            days_until_prev_end=1,
        )

    assert result is False  # INSERT falló
    assert os.path.exists(tmp_buffer_path), "Record debe haber sido buffered"
    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["user_id"] == "user-buffered"
    assert parsed["meal_plan_id"] == "plan-buffered"
    assert parsed["week_number"] == 3
    assert parsed["reason"] == "test_buffered_reason"
    assert parsed["days_until_prev_end"] == 1
    assert "buffered_at" in parsed


@patch("cron_tasks.execute_sql_write")
def test_p1_6_record_failure_with_null_meal_plan_id_skips_buffer(mock_write, tmp_buffer_path):
    """meal_plan_id=None NO se buffea (descarta inline antes que el flush lo descarte)."""
    import cron_tasks
    mock_write.side_effect = Exception("DB outage")
    cron_tasks._chunk_deferral_telemetry_failures["count"] = 0

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        result = cron_tasks._record_chunk_deferral(
            user_id="user-null-plan",
            meal_plan_id=None,  # NOT NULL violation futura
            week_number=1,
            reason="r1",
        )

    assert result is False
    # El buffer NO debe existir porque el record fue descartado antes de buffer
    assert not os.path.exists(tmp_buffer_path), (
        "meal_plan_id=None no debe buffearse (se descarta inline)"
    )


# ---------------------------------------------------------------------------
# Sanity check de imports / scheduler
# ---------------------------------------------------------------------------
def test_p1_6_module_exports():
    """Verifica que los símbolos públicos estén disponibles."""
    import cron_tasks
    import constants
    assert hasattr(cron_tasks, "_append_deferral_to_buffer")
    assert hasattr(cron_tasks, "_flush_pending_deferrals")
    assert hasattr(constants, "CHUNK_DEFERRALS_BUFFER_PATH")
    assert hasattr(constants, "CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES")
    assert hasattr(constants, "CHUNK_DEFERRALS_BUFFER_MAX_RECORDS")


def test_p1_6_scheduler_registration():
    """`register_plan_chunk_scheduler` registra el job 'flush_pending_deferrals'."""
    import cron_tasks
    mock_scheduler = MagicMock()
    mock_scheduler.get_job.return_value = None  # no jobs preexistentes
    cron_tasks.register_plan_chunk_scheduler(mock_scheduler)

    add_job_calls = mock_scheduler.add_job.call_args_list
    flush_jobs = [
        c for c in add_job_calls
        if c.kwargs.get("id") == "flush_pending_deferrals"
        or (len(c.args) > 0 and getattr(c.args[0], "__name__", "") == "_flush_pending_deferrals")
    ]
    assert len(flush_jobs) >= 1, (
        f"Esperaba job 'flush_pending_deferrals'; no encontrado en {[c.kwargs for c in add_job_calls]}"
    )
