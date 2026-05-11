"""[P2-NEW-2 · 2026-05-10] At-write whitelist para `chunk_lesson_telemetry.event`.

Bug original (audit 2026-05-10):
  `_record_chunk_lesson_telemetry` (cron_tasks.py:~12224) aceptaba cualquier
  string en `event` y confiaba en la CHECK constraint runtime de DB (P1-5:
  regex de formato). Un typo (`leson_synthesized_low_confidence`) pasaría
  el regex pero el read-path (`/lessons-counts`) lo ignoraría como "no
  clasificado" — la row queda persistida pero el chip miente al usuario.

Fix:
  Validar `event ∈ CHUNK_LESSON_TELEMETRY_VALID_EVENTS` ANTES del INSERT.
  Si no matchea: log error con contexto + return False sin persistir.

Cobertura:
  1. Constant `CHUNK_LESSON_TELEMETRY_VALID_EVENTS` existe en constants
     con la unión de lecciones (LESSON_COUNT_EVENT_WHITELIST) + métricas
     mecánicas.
  2. Cada event emitido en cron_tasks.py existe en la whitelist (drift
     detection parser-based).
  3. Helper rechaza event inválido + loguea con tag P2-NEW-2/INVALID-EVENT.
  4. Helper acepta cada event válido (no falsea-positivos).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


# ---------------------------------------------------------------------------
# 1. Constant existe + es superset de LESSON_COUNT_EVENT_WHITELIST
# ---------------------------------------------------------------------------
def test_whitelist_constant_exists():
    from constants import CHUNK_LESSON_TELEMETRY_VALID_EVENTS
    assert isinstance(CHUNK_LESSON_TELEMETRY_VALID_EVENTS, tuple)
    assert len(CHUNK_LESSON_TELEMETRY_VALID_EVENTS) >= 4, (
        "CHUNK_LESSON_TELEMETRY_VALID_EVENTS debe contener al menos los 4 "
        "events de LESSON_COUNT_EVENT_WHITELIST."
    )


def test_whitelist_is_superset_of_lesson_count_whitelist():
    """Toda lección semántica DEBE estar también en la write whitelist —
    sin esto, una lección legítima sería rechazada at-write."""
    from constants import (
        CHUNK_LESSON_TELEMETRY_VALID_EVENTS,
        LESSON_COUNT_EVENT_WHITELIST,
    )
    missing = set(LESSON_COUNT_EVENT_WHITELIST) - set(CHUNK_LESSON_TELEMETRY_VALID_EVENTS)
    assert not missing, (
        f"LESSON_COUNT_EVENT_WHITELIST tiene events que NO están en la write "
        f"whitelist: {sorted(missing)}. Esos events serían rechazados antes "
        f"de persistir, perdiendo telemetría de lecciones."
    )


# ---------------------------------------------------------------------------
# 2. Drift detection: cada event emitido en cron_tasks.py está en la whitelist
# ---------------------------------------------------------------------------
def test_all_emitted_events_in_whitelist():
    """Parsea cron_tasks.py extrayendo todos los `event="..."` literal en
    call sites y verifica que cada uno esté en CHUNK_LESSON_TELEMETRY_VALID_EVENTS."""
    from constants import CHUNK_LESSON_TELEMETRY_VALID_EVENTS
    src = _CRON_PY.read_text(encoding="utf-8")
    # Match `event="literal_lowercase_underscored"` evitando matches dentro
    # de docstrings con comillas mixtas — el patrón `event = "` (con espacios)
    # cubre kwarg-call style; sin espacios cubre keyword-call.
    matches = set(re.findall(
        r"event\s*=\s*[\"']([a-z][a-z0-9_]+)[\"']",
        src,
    ))
    # Filtrar los matches que claramente NO son call sites del helper
    # (heurística: nombres de función/variable que matchean pero no son events).
    # Los valores válidos del SUT son todos minúsculas con underscore y al
    # menos un underscore presente — sirvió para filtrar e.g. `event="completed"`
    # de otras tablas que también tienen columna `event`. En esta pasada
    # confiamos en la whitelist como veredicto: si el match no está en la
    # whitelist, falla loud — el dev decide si es legítimo + lo añade, o
    # si es un valor de OTRA tabla y lo excluye del regex.
    forbidden = matches - set(CHUNK_LESSON_TELEMETRY_VALID_EVENTS)
    # Permitir excepciones conocidas (events de otras tablas — chunk_deferrals,
    # plan_chunk_metrics, etc. — que comparten el nombre del kwarg pero no
    # son destinados a chunk_lesson_telemetry).
    _KNOWN_NON_TELEMETRY_EVENTS: set[str] = set()
    # Añadir aquí si surge un falso positivo con justificación
    # (events de otras tablas que comparten el kwarg `event=`).
    real_forbidden = forbidden - _KNOWN_NON_TELEMETRY_EVENTS
    assert not real_forbidden, (
        f"Hay events emitidos en cron_tasks.py que NO están en "
        f"CHUNK_LESSON_TELEMETRY_VALID_EVENTS (constants.py): "
        f"{sorted(real_forbidden)}. Si son legítimos para chunk_lesson_telemetry, "
        f"añadirlos a la tupla; si pertenecen a otra tabla, añadir excepción "
        f"a `_KNOWN_NON_TELEMETRY_EVENTS` en este test con comentario."
    )


# ---------------------------------------------------------------------------
# 3. Helper rechaza event inválido sin persistir
# ---------------------------------------------------------------------------
def test_invalid_event_returns_false_without_insert(monkeypatch, caplog):
    """`_record_chunk_lesson_telemetry` con event no listado debe devolver
    False y NO ejecutar INSERT — verificable porque stubeo execute_sql_write
    para fallar si se invoca."""
    import cron_tasks as ct
    import logging

    called = {"count": 0}

    def _stub_sql_write(*args, **kwargs):
        called["count"] += 1
        raise AssertionError(
            "execute_sql_write fue invocado para event inválido — "
            "el guard P2-NEW-2 falló."
        )

    monkeypatch.setattr(ct, "execute_sql_write", _stub_sql_write)
    caplog.set_level(logging.ERROR, logger="cron_tasks")

    result = ct._record_chunk_lesson_telemetry(
        user_id="00000000-0000-0000-0000-000000000001",
        meal_plan_id="00000000-0000-0000-0000-000000000002",
        week_number=1,
        event="leson_synthesized_low_confidence",  # typo
    )

    assert result is False
    assert called["count"] == 0, "INSERT no debe haberse intentado."
    assert any("[P2-NEW-2/INVALID-EVENT]" in rec.message for rec in caplog.records), (
        f"Falta log `[P2-NEW-2/INVALID-EVENT]`. Logs: "
        f"{[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 4. Helper acepta cada event válido (no falsea positivos)
# ---------------------------------------------------------------------------
def test_each_valid_event_accepted(monkeypatch):
    """Para cada event en CHUNK_LESSON_TELEMETRY_VALID_EVENTS, el helper debe
    permitir el flow hasta el INSERT (que stubea a no-op)."""
    import cron_tasks as ct
    from constants import CHUNK_LESSON_TELEMETRY_VALID_EVENTS

    inserts = []

    def _stub_sql_write(sql, params):
        # Captura sin fallar — simula INSERT exitoso.
        inserts.append({"sql": sql, "event_param": params[3]})
        return None

    monkeypatch.setattr(ct, "execute_sql_write", _stub_sql_write)

    for ev in CHUNK_LESSON_TELEMETRY_VALID_EVENTS:
        result = ct._record_chunk_lesson_telemetry(
            user_id="00000000-0000-0000-0000-000000000001",
            meal_plan_id="00000000-0000-0000-0000-000000000002",
            week_number=1,
            event=ev,
        )
        assert result is True, f"Event válido {ev!r} fue rechazado."

    # Cada event debe haber alcanzado el INSERT.
    seen_events = {ins["event_param"] for ins in inserts}
    assert seen_events == set(CHUNK_LESSON_TELEMETRY_VALID_EVENTS)
