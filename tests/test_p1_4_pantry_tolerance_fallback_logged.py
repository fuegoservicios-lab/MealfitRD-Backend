"""[P1-4] Tests para el logging estructurado y telemetría de fallbacks de
`_get_pantry_tolerance_for_user` / `_resolve_pantry_tolerance`.

Antes los fallbacks (DB error, valor no-numérico, clamp por override fuera
de rango) eran silenciosos o solo `logger.debug` sin tag estructurado y sin
contador agregado. Un usuario con override 1.30 que sufría DB blip caía al
default 1.05 sin que dashboards detectaran la degradación.

Después:
  - Cada fallback inesperado emite `[P1-4/PANTRY/TOLERANCE] Fallback <source> ...`
  - Se incrementa el ring buffer `_PANTRY_TOLERANCE_FALLBACKS` (24h rolling).
  - El endpoint `/api/system/pantry-tolerance-health` lo expone agregado.

Ejecutar:
    cd backend && python -m pytest tests/test_p1_4_pantry_tolerance_fallback_logged.py -v
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _reset_fallback_buffer():
    """Limpia el ring buffer antes de cada test para aislamiento."""
    from cron_tasks import _PANTRY_TOLERANCE_FALLBACKS
    _PANTRY_TOLERANCE_FALLBACKS.clear()
    yield
    _PANTRY_TOLERANCE_FALLBACKS.clear()


# ---------------------------------------------------------------------------
# 1. _resolve_pantry_tolerance retorna source correcto en cada caso
# ---------------------------------------------------------------------------
@patch("cron_tasks.execute_sql_query")
def test_resolve_returns_user_override_source_when_value_differs(mock_query):
    """Override del usuario distinto al default → source='user_override'."""
    from cron_tasks import _resolve_pantry_tolerance

    mock_query.return_value = {"pantry_tolerance": 1.30}

    val, source = _resolve_pantry_tolerance("user-1", default=1.05)

    assert val == 1.30
    assert source == "user_override"


@patch("cron_tasks.execute_sql_query")
def test_resolve_returns_default_match_when_user_value_equals_default(mock_query):
    from cron_tasks import _resolve_pantry_tolerance

    mock_query.return_value = {"pantry_tolerance": 1.05}

    val, source = _resolve_pantry_tolerance("user-1", default=1.05)

    assert val == 1.05
    assert source == "user_override_default_match"


@patch("cron_tasks.execute_sql_query")
def test_resolve_returns_default_no_override_when_value_null(mock_query):
    """Columna existe pero es NULL → source='default_no_override' (esperado)."""
    from cron_tasks import _resolve_pantry_tolerance

    mock_query.return_value = {"pantry_tolerance": None}

    val, source = _resolve_pantry_tolerance("user-1", default=1.05)

    assert val == 1.05
    assert source == "default_no_override"


@patch("cron_tasks.execute_sql_query")
def test_resolve_returns_default_no_row_when_user_missing(mock_query):
    """SELECT exitoso pero no hay fila (defensivo)."""
    from cron_tasks import _resolve_pantry_tolerance

    mock_query.return_value = None

    val, source = _resolve_pantry_tolerance("user-1", default=1.05)

    assert val == 1.05
    assert source == "default_no_row"


# ---------------------------------------------------------------------------
# 2. Fallbacks "inesperados" emiten log estructurado y registran counter
# ---------------------------------------------------------------------------
@patch("cron_tasks.execute_sql_query")
def test_db_error_logs_warning_and_increments_counter(mock_query, caplog):
    """SELECT lanza excepción → log warning con tag P1-4 + counter +1."""
    from cron_tasks import _resolve_pantry_tolerance, _PANTRY_TOLERANCE_FALLBACKS
    import logging

    mock_query.side_effect = RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        val, source = _resolve_pantry_tolerance("user-A", default=1.05)

    assert val == 1.05
    assert source == "fallback_db_error"
    # Counter incrementado.
    assert len(_PANTRY_TOLERANCE_FALLBACKS) == 1
    ts, recorded_source, recorded_user = _PANTRY_TOLERANCE_FALLBACKS[0]
    assert recorded_source == "fallback_db_error"
    assert recorded_user == "user-A"
    # Log estructurado con tag.
    matching = [r for r in caplog.records if "[P1-4/PANTRY/TOLERANCE]" in r.message]
    assert matching, "Log estructurado [P1-4/PANTRY/TOLERANCE] esperado"
    assert "db_error" in matching[0].message


@patch("cron_tasks.execute_sql_query")
def test_non_numeric_logs_warning_and_increments_counter(mock_query, caplog):
    """Valor no-parseable a float → fallback_non_numeric."""
    from cron_tasks import _resolve_pantry_tolerance, _PANTRY_TOLERANCE_FALLBACKS
    import logging

    mock_query.return_value = {"pantry_tolerance": "definitely-not-a-number"}

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        val, source = _resolve_pantry_tolerance("user-B", default=1.05)

    assert val == 1.05
    assert source == "fallback_non_numeric"
    assert len(_PANTRY_TOLERANCE_FALLBACKS) == 1
    assert _PANTRY_TOLERANCE_FALLBACKS[0][1] == "fallback_non_numeric"
    matching = [r for r in caplog.records if "[P1-4/PANTRY/TOLERANCE]" in r.message]
    assert matching


@patch("cron_tasks.execute_sql_query")
def test_clamped_value_logs_warning_and_increments_counter(mock_query, caplog):
    """Override fuera de [MIN, MAX] → clamped + log + counter."""
    from cron_tasks import (
        _resolve_pantry_tolerance,
        _PANTRY_TOLERANCE_FALLBACKS,
    )
    from constants import CHUNK_PANTRY_TOLERANCE_MAX
    import logging

    out_of_range = CHUNK_PANTRY_TOLERANCE_MAX + 0.5  # 2.0 si MAX=1.5
    mock_query.return_value = {"pantry_tolerance": out_of_range}

    with caplog.at_level(logging.WARNING, logger="cron_tasks"):
        val, source = _resolve_pantry_tolerance("user-C", default=1.05)

    assert val == CHUNK_PANTRY_TOLERANCE_MAX
    assert source == "fallback_clamped"
    assert len(_PANTRY_TOLERANCE_FALLBACKS) == 1
    assert _PANTRY_TOLERANCE_FALLBACKS[0][1] == "fallback_clamped"
    matching = [r for r in caplog.records if "[P1-4/PANTRY/TOLERANCE]" in r.message]
    assert matching


# ---------------------------------------------------------------------------
# 3. Casos esperados (no-override, no-row) NO incrementan el counter
# ---------------------------------------------------------------------------
@patch("cron_tasks.execute_sql_query")
def test_default_no_override_does_not_increment_counter(mock_query):
    from cron_tasks import _resolve_pantry_tolerance, _PANTRY_TOLERANCE_FALLBACKS

    mock_query.return_value = {"pantry_tolerance": None}

    _resolve_pantry_tolerance("user-D", default=1.05)

    assert len(_PANTRY_TOLERANCE_FALLBACKS) == 0


@patch("cron_tasks.execute_sql_query")
def test_user_override_does_not_increment_counter(mock_query):
    from cron_tasks import _resolve_pantry_tolerance, _PANTRY_TOLERANCE_FALLBACKS

    mock_query.return_value = {"pantry_tolerance": 1.20}

    _resolve_pantry_tolerance("user-E", default=1.05)

    assert len(_PANTRY_TOLERANCE_FALLBACKS) == 0


# ---------------------------------------------------------------------------
# 4. Ring buffer: poda eventos viejos
# ---------------------------------------------------------------------------
def test_record_fallback_prunes_events_older_than_window():
    """Eventos más viejos que la ventana de 24h se descartan al añadir nuevos."""
    from cron_tasks import (
        _record_pantry_tolerance_fallback,
        _PANTRY_TOLERANCE_FALLBACKS,
        _PANTRY_TOLERANCE_FALLBACK_WINDOW_SECONDS,
    )
    import time as _p14_time

    # Inyectar evento "antiguo" (25h atrás).
    old_ts = _p14_time.time() - _PANTRY_TOLERANCE_FALLBACK_WINDOW_SECONDS - 3600
    _PANTRY_TOLERANCE_FALLBACKS.append((old_ts, "fallback_db_error", "old-user"))

    # Añadir evento nuevo → poda automática.
    _record_pantry_tolerance_fallback("fallback_db_error", "new-user")

    # El antiguo debe haber sido podado.
    user_ids = [e[2] for e in _PANTRY_TOLERANCE_FALLBACKS]
    assert "old-user" not in user_ids
    assert "new-user" in user_ids


def test_record_fallback_caps_at_max_records():
    """Si excede MAX_RECORDS, descarta los más antiguos."""
    from cron_tasks import (
        _record_pantry_tolerance_fallback,
        _PANTRY_TOLERANCE_FALLBACKS,
        _PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS,
    )
    import time as _p14_time

    # Llenar el buffer al cap.
    now = _p14_time.time()
    for i in range(_PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS):
        _PANTRY_TOLERANCE_FALLBACKS.append((now - 0.001, "fallback_db_error", f"user-{i}"))

    pre_len = len(_PANTRY_TOLERANCE_FALLBACKS)
    assert pre_len == _PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS

    # Añadir uno más → poda al cap.
    _record_pantry_tolerance_fallback("fallback_db_error", "user-overflow")

    assert len(_PANTRY_TOLERANCE_FALLBACKS) <= _PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS


# ---------------------------------------------------------------------------
# 5. API pública _get_pantry_tolerance_for_user sigue retornando float
# ---------------------------------------------------------------------------
@patch("cron_tasks.execute_sql_query")
def test_public_api_returns_float_unchanged(mock_query):
    """Back-compat: la firma pública sigue siendo `-> float`."""
    from cron_tasks import _get_pantry_tolerance_for_user

    mock_query.return_value = {"pantry_tolerance": 1.25}

    result = _get_pantry_tolerance_for_user("user-1")

    assert isinstance(result, float)
    assert result == 1.25
