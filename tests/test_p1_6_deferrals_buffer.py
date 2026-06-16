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
import uuid
from unittest.mock import patch, MagicMock
import pytest


# [P1-NEON-DB-MIGRATION · 2026-06-12] El flush (`_flush_pending_deferrals`) y
# `_record_chunk_deferral` endurecieron un guard `_is_valid_uuid(user_id/
# meal_plan_id)`: records con IDs no-UUID se descartan ANTES del INSERT (bajo
# PostgREST/MagicMock cualquier string pasaba; contra Postgres real las columnas
# son `uuid` NOT NULL). Los fixtures que verifican el path "válido → INSERT"
# deben usar UUIDs reales para que el record sobreviva el guard y ejercite el
# `execute_sql_write` mockeado. Helpers deterministas para legibilidad de asserts.
_UID_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p16-user-1"))
_UID_2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p16-user-2"))
_PID_1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p16-plan-1"))
_PID_2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "p16-plan-2"))


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


def test_p1_6_append_does_not_cap_on_each_append(tmp_buffer_path):
    """[P0-8] `_append_deferral_to_buffer` ya NO aplica el FIFO cap en cada append.

    El refactor P0-8 cambió el append de read-modify-write (reescribir TODO el
    archivo + cap en cada llamada) a append simple O(1) (`open(path, "a")` +
    flush + fsync). El cap se DELEGA al sweep `_flush_pending_deferrals`. Por eso
    un append sobre un archivo con 5 records pre-existentes deja 6 líneas — el
    cap NO se aplica aquí aunque `CHUNK_DEFERRALS_BUFFER_MAX_RECORDS` sea menor.
    Ver `test_p1_6_flush_applies_fifo_cap_drops_oldest` para el cap real.
    """
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
    # Append simple: los 5 pre-existentes + el nuevo = 6 (cap NO aplicado en append)
    assert len(lines) == 6
    assert json.loads(lines[-1])["user_id"] == "u_new"
    assert json.loads(lines[0])["user_id"] == "u0"  # los viejos siguen ahí


@patch("cron_tasks.execute_sql_write")
def test_p1_6_flush_applies_fifo_cap_drops_oldest(mock_write, tmp_buffer_path):
    """[P0-8] El FIFO cap se aplica durante el flush, no en cada append.

    Cuando el outage entre flushes deja el buffer por encima del cap, el sweep
    `_flush_pending_deferrals` trunca a las N más recientes (las más viejas se
    descartan — FIFO). Mockeamos un error transitorio para que los records se
    MANTENGAN (no se persistan) y el truncado del cap sea observable en el
    archivo reescrito.
    """
    import cron_tasks
    mock_write.side_effect = Exception("connection timeout")  # transitorio → mantener

    # UUIDs reales (el guard `_is_valid_uuid` descarta IDs no-UUID antes del INSERT).
    # Uno por record para poder identificar cuáles sobreviven el cap.
    uids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"p16-fifo-user-{i}")) for i in range(5)]
    pids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"p16-fifo-plan-{i}")) for i in range(5)]
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({
                "user_id": uids[i], "meal_plan_id": pids[i], "week_number": 1,
                "reason": f"old{i}", "buffered_at": "2026-05-01T00:00:00+00:00",
            }) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        with patch("constants.CHUNK_DEFERRALS_BUFFER_MAX_RECORDS", 3):
            stats = cron_tasks._flush_pending_deferrals()

    with open(tmp_buffer_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # Tras el cap (3) en el flush: las 3 más recientes (índices 2,3,4) sobreviven.
    assert len(lines) == 3
    assert stats["remaining"] == 3
    assert json.loads(lines[0])["user_id"] == uids[2]   # u0, u1 (más viejos) se cayeron
    assert json.loads(lines[-1])["user_id"] == uids[4]  # el más reciente sobrevive


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
    # Pre-poblar (UUIDs reales: el guard `_is_valid_uuid` del flush descarta IDs
    # no-UUID antes del INSERT — ver nota de cabecera P1-NEON-DB-MIGRATION).
    records = [
        {"user_id": _UID_1, "meal_plan_id": _PID_1, "week_number": 1, "reason": "r1", "buffered_at": "2026-05-01T00:00:00+00:00"},
        {"user_id": _UID_2, "meal_plan_id": _PID_2, "week_number": 2, "reason": "r2", "buffered_at": "2026-05-01T00:01:00+00:00"},
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
        {"user_id": _UID_1, "meal_plan_id": None, "week_number": 1, "reason": "r1"},  # invalid: meal_plan_id None (NOT NULL)
        {"user_id": _UID_2, "meal_plan_id": _PID_2, "week_number": 2, "reason": "r2"},  # valid: UUIDs reales
    ]
    with open(tmp_buffer_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        stats = cron_tasks._flush_pending_deferrals()

    assert stats["discarded_invalid"] == 1
    assert stats["flushed"] == 1
    # Solo se intentó UN INSERT (el válido); el inválido (meal_plan_id None) se
    # descartó antes de tocar la DB.
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

    # UUIDs reales para que el record pase el guard y LLEGUE al INSERT (donde el
    # error transitorio lo mantiene para retry). Con IDs no-UUID el flush lo
    # descartaría antes de intentar el INSERT y el assert de `remaining` fallaría.
    rec = {"user_id": _UID_1, "meal_plan_id": _PID_1, "week_number": 1, "reason": "r1"}
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

    # UUIDs reales: `_record_chunk_deferral` tiene un hard guard que descarta IDs
    # no-UUID ANTES del INSERT y NO los buffea (evita contaminar el buffer con
    # basura de tests). Con UUIDs válidos el INSERT se intenta, falla por el
    # `DB outage` mockeado, y el record SÍ se buffea para retry — que es lo que
    # este test verifica.
    with patch("constants.CHUNK_DEFERRALS_BUFFER_PATH", tmp_buffer_path):
        result = cron_tasks._record_chunk_deferral(
            user_id=_UID_1,
            meal_plan_id=_PID_1,
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
    assert parsed["user_id"] == _UID_1
    assert parsed["meal_plan_id"] == _PID_1
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
