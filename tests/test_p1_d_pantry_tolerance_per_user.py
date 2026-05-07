"""[P1-D] Tests para `_get_pantry_tolerance_for_user` (override per-usuario).

Cubre:
  1. Valor en rango → se devuelve sin clampar.
  2. Valor < MIN → clamp a MIN con warning.
  3. Valor > MAX → clamp a MAX con warning.
  4. NULL / sin row → default global.
  5. Valor no-numérico (corrupto) → default global.
  6. Excepción en SQL (e.g., columna missing en deploy híbrido) → default global,
     no propaga la excepción (no bloqueante).

Ejecutar:
    cd backend && python -m pytest tests/test_p1_d_pantry_tolerance_per_user.py -v
"""
from unittest.mock import patch
import pytest

from cron_tasks import _get_pantry_tolerance_for_user
from constants import (
    CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE,
    CHUNK_PANTRY_TOLERANCE_MIN,
    CHUNK_PANTRY_TOLERANCE_MAX,
)


@patch("cron_tasks.execute_sql_query")
def test_valid_value_returned_unchanged(mock_query):
    mock_query.return_value = {"pantry_tolerance": 1.20}
    assert _get_pantry_tolerance_for_user("user-1") == 1.20


@patch("cron_tasks.execute_sql_query")
def test_null_returns_default(mock_query):
    """row con pantry_tolerance NULL → default global."""
    mock_query.return_value = {"pantry_tolerance": None}
    assert _get_pantry_tolerance_for_user("user-1") == CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE


@patch("cron_tasks.execute_sql_query")
def test_no_row_returns_default(mock_query):
    """SELECT no devuelve row → default global."""
    mock_query.return_value = None
    assert _get_pantry_tolerance_for_user("user-1") == CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE


@patch("cron_tasks.execute_sql_query")
def test_value_below_min_clamps_to_min(mock_query):
    mock_query.return_value = {"pantry_tolerance": 0.50}
    result = _get_pantry_tolerance_for_user("user-1")
    assert result == CHUNK_PANTRY_TOLERANCE_MIN


@patch("cron_tasks.execute_sql_query")
def test_value_above_max_clamps_to_max(mock_query):
    mock_query.return_value = {"pantry_tolerance": 3.00}
    result = _get_pantry_tolerance_for_user("user-1")
    assert result == CHUNK_PANTRY_TOLERANCE_MAX


@patch("cron_tasks.execute_sql_query")
def test_non_numeric_value_returns_default(mock_query):
    """Si la columna trae basura (e.g., string corrupto post-migración manual),
    el helper cae al default sin propagar."""
    mock_query.return_value = {"pantry_tolerance": "loose"}
    assert _get_pantry_tolerance_for_user("user-1") == CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE


@patch("cron_tasks.execute_sql_query")
def test_sql_exception_returns_default_without_propagating(mock_query):
    """Columna missing (deploy híbrido sin la migración aún), DB blip → no
    bloquear chunk; default global."""
    mock_query.side_effect = RuntimeError("column 'pantry_tolerance' does not exist")
    result = _get_pantry_tolerance_for_user("user-1")
    assert result == CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE


@patch("cron_tasks.execute_sql_query")
def test_explicit_default_param_overrides_global(mock_query):
    """`default=` explícito permite tests/callers pasar otro fallback."""
    mock_query.return_value = None
    assert _get_pantry_tolerance_for_user("user-1", default=1.10) == 1.10


@patch("cron_tasks.execute_sql_query")
def test_boundary_values_pass(mock_query):
    """Valores en los bordes exactos no se clampean."""
    mock_query.return_value = {"pantry_tolerance": CHUNK_PANTRY_TOLERANCE_MIN}
    assert _get_pantry_tolerance_for_user("user-1") == CHUNK_PANTRY_TOLERANCE_MIN

    mock_query.return_value = {"pantry_tolerance": CHUNK_PANTRY_TOLERANCE_MAX}
    assert _get_pantry_tolerance_for_user("user-1") == CHUNK_PANTRY_TOLERANCE_MAX


@patch("cron_tasks.execute_sql_query")
def test_decimal_string_from_db_is_coerced(mock_query):
    """psycopg/Supabase a veces devuelve NUMERIC como Decimal o string. El helper
    debe coerce a float."""
    from decimal import Decimal
    mock_query.return_value = {"pantry_tolerance": Decimal("1.15")}
    assert _get_pantry_tolerance_for_user("user-1") == 1.15

    mock_query.return_value = {"pantry_tolerance": "1.25"}
    assert _get_pantry_tolerance_for_user("user-1") == 1.25
