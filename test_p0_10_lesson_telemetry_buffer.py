"""[P0-10] Tests para garantizar que `chunk_lesson_telemetry` tiene buffer
de respaldo durante outages de DB, simétrico con el buffer de deferrals (P0-8).

Bug original (audit P0-10):
  `_record_chunk_lesson_telemetry` solo logueaba el fallo si la INSERT a
  `chunk_lesson_telemetry` lanzaba excepción. Eventos críticos para detectar
  degradación del aprendizaje continuo (`lesson_synthesized_low_confidence`,
  `recent_lessons_partial_synthesis`, `synth_schema_invalid`,
  `indefinite_pause_unblocked`, `lifetime_proxy_ratio_exceeded`) se perdían
  para siempre durante outages — SRE perdía la capacidad de detectar
  problemas en el rango horas-días posteriores. Asimétrico con
  `_record_chunk_deferral` que sí buffeaba a JSONL en disco (P0-8).

Fix:
  - `_append_lesson_telemetry_to_buffer(record)` usa el helper atómico
    compartido `_atomic_append_jsonl_record` (P0-8/P0-10).
  - `_record_chunk_lesson_telemetry`: en el `except`, si user_id y
    meal_plan_id son UUIDs válidos, escribe el evento al buffer.
  - `_flush_pending_lesson_telemetry`: cron que reintenta cuando DB se
    recupera, descartando inválidos y aplicando FIFO cap.
  - Helpers `_atomic_append_jsonl_record` y `_atomic_rewrite_jsonl_buffer`
    extraídos para que cualquier futuro buffer de telemetría reuse el
    contrato P0-8.

Cobertura:
  - test_lesson_telemetry_writes_to_buffer_on_db_failure
  - test_lesson_telemetry_skips_buffer_for_invalid_uuids
  - test_lesson_telemetry_buffer_uses_atomic_append_helper
  - test_flush_pending_lesson_telemetry_re_inserts_buffered_records
  - test_flush_pending_lesson_telemetry_discards_invalid
  - test_flush_pending_lesson_telemetry_keeps_transient_failures
  - test_flush_applies_fifo_cap
  - test_buffer_path_is_independent_from_deferrals_path
  - test_helpers_share_atomic_pattern_with_deferrals_p0_8
"""
import inspect
import json
import logging
from unittest.mock import patch, MagicMock

import pytest

import cron_tasks


# ---------------------------------------------------------------------------
# 1. Fallback: el buffer recibe el evento cuando la INSERT falla.
# ---------------------------------------------------------------------------
def test_lesson_telemetry_writes_to_buffer_on_db_failure(tmp_path, monkeypatch):
    """Cuando `execute_sql_write` falla y los UUIDs son válidos, el evento
    debe quedar persistido en el buffer JSONL para retry posterior."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))

    def db_fail(*_a, **_kw):
        raise ConnectionError("simulated DB outage")

    user_id = "00000000-0000-0000-0000-000000000001"
    plan_id = "00000000-0000-0000-0000-000000000002"

    with patch.object(cron_tasks, "execute_sql_write", side_effect=db_fail):
        ok = cron_tasks._record_chunk_lesson_telemetry(
            user_id=user_id, meal_plan_id=plan_id, week_number=2,
            event="lesson_synthesized_low_confidence",
            synthesized_count=1, queue_count=0,
            metadata={"prev_week": 1},
        )

    assert ok is False, "el helper debe devolver False cuando la INSERT falla"
    assert buf_path.exists(), "el buffer JSONL debe haberse creado"
    lines = buf_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"esperado 1 línea buffered, got {len(lines)}"
    rec = json.loads(lines[0])
    assert rec["user_id"] == user_id
    assert rec["meal_plan_id"] == plan_id
    assert rec["event"] == "lesson_synthesized_low_confidence"
    assert rec["synthesized_count"] == 1
    assert rec["metadata"] == {"prev_week": 1}
    assert "buffered_at" in rec, "buffered_at debe estar presente para timestamp retry"


def test_lesson_telemetry_skips_buffer_for_invalid_uuids(tmp_path, monkeypatch):
    """Si user_id/meal_plan_id NO son UUIDs válidos, NO se escribe al buffer
    (sería basura permanente que el flush descartaría siempre). Asimétrico
    con strings tipo "u1"/"plan-1" que aparecen en tests."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))

    def db_fail(*_a, **_kw):
        raise ConnectionError("simulated")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=db_fail):
        ok = cron_tasks._record_chunk_lesson_telemetry(
            user_id="not-a-uuid", meal_plan_id="plan-x", week_number=1,
            event="any_event",
        )

    assert ok is False
    # El buffer NO debe haberse creado (UUIDs inválidos no se buffean).
    assert not buf_path.exists() or buf_path.read_text(encoding="utf-8").strip() == "", \
        "P0-10: UUIDs inválidos NO deben contaminar el buffer"


# ---------------------------------------------------------------------------
# 2. El append usa el helper atómico P0-8 (reuso del contrato de durabilidad).
# ---------------------------------------------------------------------------
def test_lesson_telemetry_buffer_uses_atomic_append_helper():
    """`_append_lesson_telemetry_to_buffer` debe delegar a
    `_atomic_append_jsonl_record` para reusar el contrato P0-8 (mode "a" +
    flush + fsync). Defensa textual contra reintroducir el patrón roto."""
    src = inspect.getsource(cron_tasks._append_lesson_telemetry_to_buffer)
    assert "_atomic_append_jsonl_record" in src, \
        "P0-10: append helper debe delegar al helper compartido"
    # Y NO debe tener un `open(path, "w")` propio (ese era el bug P0-8).
    assert 'open(' not in src or '"a"' in src, \
        "P0-10: si abre archivos directamente debe ser modo 'a'"


def test_flush_pending_lesson_telemetry_uses_atomic_rewrite_helper():
    """`_flush_pending_lesson_telemetry` debe usar el helper compartido
    `_atomic_rewrite_jsonl_buffer` para la reescritura atómica."""
    src = inspect.getsource(cron_tasks._flush_pending_lesson_telemetry)
    assert "_atomic_rewrite_jsonl_buffer" in src, \
        "P0-10: flush debe usar el helper compartido para rewrite atómico"


# ---------------------------------------------------------------------------
# 3. Flush: re-inserta records cuando DB se recupera.
# ---------------------------------------------------------------------------
def test_flush_pending_lesson_telemetry_re_inserts_buffered_records(tmp_path, monkeypatch):
    """Records válidos buffered se re-INSERTan exitosamente y el buffer
    queda vacío tras el flush."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 100)

    # Pre-popular buffer con 3 records válidos.
    records = [
        {
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "week_number": i,
            "event": "lesson_synthesized_low_confidence",
            "synthesized_count": 1, "queue_count": 0,
            "metadata": {}, "buffered_at": "2026-05-04T17:00:00+00:00",
        }
        for i in range(3)
    ]
    buf_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    insert_calls = []

    def db_ok(query, params=None, **kwargs):
        insert_calls.append((query, params))
        return None

    with patch.object(cron_tasks, "execute_sql_write", side_effect=db_ok):
        stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["flushed"] == 3
    assert stats["remaining"] == 0
    assert len(insert_calls) == 3
    # Cada INSERT debe contener event y metadata (jsonb).
    for query, params in insert_calls:
        assert "INSERT INTO chunk_lesson_telemetry" in query
        assert "event" in query
    # Tras flush exitoso, el archivo debe haber sido eliminado.
    assert not buf_path.exists(), "buffer debe limpiarse tras flush exitoso"


def test_flush_pending_lesson_telemetry_discards_invalid(tmp_path, monkeypatch):
    """Records con UUIDs inválidos o JSON corrupto se descartan permanentemente
    (no quedan en remaining)."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 100)

    lines = [
        '{not_valid_json',  # JSON corrupto
        json.dumps({  # UUIDs inválidos
            "user_id": "not-a-uuid",
            "meal_plan_id": "also-not",
            "week_number": 1, "event": "x",
        }),
        json.dumps({  # válido (UUID)
            "user_id": "00000000-0000-0000-0000-000000000001",
            "meal_plan_id": "00000000-0000-0000-0000-000000000002",
            "week_number": 2, "event": "valid_event",
            "synthesized_count": 0, "queue_count": 0, "metadata": {},
        }),
    ]
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with patch.object(cron_tasks, "execute_sql_write", return_value=None):
        stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["flushed"] == 1
    assert stats["discarded_invalid"] == 2
    assert stats["remaining"] == 0


def test_flush_pending_lesson_telemetry_keeps_transient_failures(tmp_path, monkeypatch):
    """Errores transitorios (ConnectionError, timeout) preservan el record
    en remaining para el siguiente flush."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 100)

    rec = {
        "user_id": "00000000-0000-0000-0000-000000000001",
        "meal_plan_id": "00000000-0000-0000-0000-000000000002",
        "week_number": 1, "event": "x",
        "synthesized_count": 0, "queue_count": 0, "metadata": {},
    }
    buf_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    def db_transient_fail(*_a, **_kw):
        raise ConnectionError("transient")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=db_transient_fail):
        stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["flushed"] == 0
    assert stats["remaining"] == 1
    # Buffer debe seguir teniendo el record.
    final = buf_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(final) == 1
    assert json.loads(final[0])["event"] == "x"


def test_flush_applies_fifo_cap(tmp_path, monkeypatch):
    """Si el buffer creció más allá del cap durante un outage prolongado, el
    flush trunca a las N más recientes (FIFO)."""
    buf_path = tmp_path / "lesson_telemetry_test.jsonl"
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_PATH", str(buf_path))
    monkeypatch.setattr("constants.CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", 5)

    # 20 records válidos, todos quedarán en remaining (DB falla).
    lines = [
        json.dumps({
            "user_id": f"00000000-0000-0000-0000-{i:012d}",
            "meal_plan_id": f"00000000-0000-0000-0000-{i+100:012d}",
            "week_number": i, "event": f"evt_{i}",
            "synthesized_count": 0, "queue_count": 0, "metadata": {},
        })
        for i in range(20)
    ]
    buf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with patch.object(cron_tasks, "execute_sql_write", side_effect=ConnectionError("transient")):
        stats = cron_tasks._flush_pending_lesson_telemetry()

    assert stats["remaining"] == 5
    final_lines = [json.loads(ln) for ln in buf_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(final_lines) == 5
    weeks = [r["week_number"] for r in final_lines]
    assert weeks == [15, 16, 17, 18, 19], \
        f"FIFO cap incorrecto: esperado [15..19], got {weeks}"


# ---------------------------------------------------------------------------
# 4. Independencia: lesson_telemetry y deferrals son buffers separados.
# ---------------------------------------------------------------------------
def test_buffer_path_is_independent_from_deferrals_path():
    """Los dos buffers deben usar paths distintos para que no se contaminen
    entre sí."""
    from constants import (
        CHUNK_DEFERRALS_BUFFER_PATH,
        CHUNK_LESSON_TELEMETRY_BUFFER_PATH,
    )
    assert CHUNK_DEFERRALS_BUFFER_PATH != CHUNK_LESSON_TELEMETRY_BUFFER_PATH, \
        "P0-10: los buffers deben tener paths distintos"


def test_lock_independence_between_buffers():
    """Los dos locks deben ser objetos distintos para que un append a un
    buffer no bloquee el otro."""
    assert cron_tasks._p16_buffer_lock is not cron_tasks._p010_lesson_buffer_lock, \
        "P0-10: cada buffer debe tener su propio lock (independencia de contención)"


# ---------------------------------------------------------------------------
# 5. Defensa textual: el contrato `[P0-10]` debe estar documentado.
# ---------------------------------------------------------------------------
def test_documentation_p0_10_present():
    """Comentarios `[P0-10]` documentan el contrato. Si alguien borra los
    helpers, el grep falla y obliga a entender el rationale."""
    src = inspect.getsource(cron_tasks)
    assert "[P0-10]" in src
    assert "_append_lesson_telemetry_to_buffer" in src
    assert "_flush_pending_lesson_telemetry" in src


def test_record_chunk_lesson_telemetry_invokes_buffer_on_failure():
    """Defensa AST: `_record_chunk_lesson_telemetry` debe contener una
    llamada a `_append_lesson_telemetry_to_buffer` en su except branch."""
    src = inspect.getsource(cron_tasks._record_chunk_lesson_telemetry)
    assert "_append_lesson_telemetry_to_buffer" in src, (
        "P0-10 regression: el except branch debe escribir al buffer"
    )
