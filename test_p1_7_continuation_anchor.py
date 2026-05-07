"""
Tests P1-7: Rolling refill continuation anchor en temporal gate.

Antes: el primer chunk del rolling refill calculaba `_prev_end_date` con la
fórmula:
    _prev_end_date = new_plan_start_iso + (prev_offset + prev_count - 1)
donde `new_plan_start_iso` ~= today (ancla del refill) y `prev_offset/count`
venían del plan_chunk_queue del chunk previo del PLAN ORIGINAL (offsets
relativos al `_plan_start_date` original). Eso producía fechas futuras
incorrectas (e.g., today + 8 días) y diferia el chunk innecesariamente.

Cambios P1-7:
  1. Snapshot del rolling refill marca `_is_continuation=True` y
     `_continuation_anchor_iso=new_plan_start_iso`.
  2. En `_check_chunk_learning_ready`, si esos markers están y el chunk previo
     en plan_chunk_queue NO tiene `chunk_kind='rolling_refill'`, el gate usa
     `_prev_end_date = anchor_iso - 1 día` (último día real del plan original).
  3. Para chunks subsiguientes del refill (cuyo prev SÍ es 'rolling_refill'),
     el cálculo legacy es correcto y se preserva.

Cubre:
  1. Continuation con prev=initial_plan → anchor override aplicado.
  2. Continuation con prev=rolling_refill → cálculo legacy preservado.
  3. Sin _is_continuation → cálculo legacy.
  4. _continuation_anchor_iso ausente → cálculo legacy.
  5. anchor_iso malformado → fallback al cálculo legacy + warning.
  6. week_number=1 (chunk inicial) → no aplica anchor (continuation requiere week>=2).
"""
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

import pytest


def _build_plan_data_with_prior_days(days_count=6):
    """plan_data con días previos suficientes para cruzar el bloque previous_chunk_days."""
    return {
        "days": [
            {"day": d, "meals": [{"name": f"M{d}", "ingredients": ["pollo"]}]}
            for d in range(1, days_count + 1)
        ],
    }


def _stub_resolve_window(prev_offset, prev_count):
    """Helper: devuelve un mock que el gate use como _resolve_previous_chunk_window."""
    return MagicMock(return_value=(prev_offset, prev_count))


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_p1_7_continuation_with_initial_plan_prev_uses_anchor(mock_query, mock_now):
    """
    Escenario: rolling refill chunk_4 tras plan original (chunks 1-3 completed).
    new_plan_start_iso = 2026-05-01 (today). Plan original empezaba 2026-04-25.
    Chunk 3 original: offset=6, count=3.

    Sin fix: _prev_end_date = 2026-05-01 + (6+3-1) = 2026-05-09 → diferimos 8d.
    Con fix: anchor=2026-05-01, _prev_end_date = 2026-04-30 → ya pasó, gate dispara.
    """
    from cron_tasks import _check_chunk_learning_ready
    today = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    mock_now.return_value = today

    snapshot = {
        "form_data": {
            "_plan_start_date": "2026-05-01",  # new_plan_start del refill
            "_is_continuation": True,
            "_continuation_anchor_iso": "2026-05-01",
        },
    }
    plan_data = _build_plan_data_with_prior_days(6)

    # Mocks de execute_sql_query: orden de llamadas que el gate hace.
    # 1. SELECT health_profile (TZ live).
    # 2. SELECT plan_chunk_queue WHERE week_number=3 (anchor lookup).
    # 3+. otras queries que pueda hacer; devolvemos None.
    def query_side(q, params=None, **kw):
        q_lower = (q or "").lower()
        if "health_profile" in q_lower:
            return {"health_profile": {"tz_offset_minutes": 0}}
        if "chunk_kind" in q_lower:
            # Chunk previo era del plan original
            return {"chunk_kind": "initial_plan"}
        return None
    mock_query.side_effect = query_side

    # Force prev window: simulamos que el chunk previo era offset=6, count=3
    with patch("cron_tasks._resolve_previous_chunk_window", _stub_resolve_window(6, 3)):
        result = _check_chunk_learning_ready(
            user_id="user-p17-1",
            meal_plan_id="plan-p17-1",
            week_number=4,
            days_offset=0,  # primer chunk del refill arranca en day 1 efectivo
            plan_data=plan_data,
            snapshot=snapshot,
        )

    # El gate debería tratar el chunk previo como ya completado (anchor=2026-05-01,
    # prev_end_date=2026-04-30, today=2026-05-01 → días faltantes = -1 → ready).
    # ratio puede no estar disponible (no hay logs en este mock), pero la rama
    # de "prev_chunk_day_not_yet_elapsed" NO debe activarse.
    reason = result.get("reason")
    assert reason != "prev_chunk_day_not_yet_elapsed", (
        f"Con anchor el chunk previo se considera completado; pero gate dijo {reason!r}"
    )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_p1_7_continuation_with_rolling_refill_prev_uses_legacy(mock_query, mock_now):
    """
    Escenario: rolling refill chunk_5 (segundo del refill). Chunk_4 (primero del refill)
    completed con offset=0, count=3. plan_start_date=2026-05-01 (refill anchor).

    El cálculo legacy es correcto: _prev_end_date = 2026-05-01 + 0+3-1 = 2026-05-03.
    Hoy es 2026-05-04 → days_until_prev_end = -1 → ready.

    El override P1-7 NO debe aplicarse porque chunk_4 es 'rolling_refill'.
    """
    from cron_tasks import _check_chunk_learning_ready
    today = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    mock_now.return_value = today

    snapshot = {
        "form_data": {
            "_plan_start_date": "2026-05-01",
            "_is_continuation": True,
            "_continuation_anchor_iso": "2026-05-01",
        },
    }
    plan_data = _build_plan_data_with_prior_days(9)

    def query_side(q, params=None, **kw):
        q_lower = (q or "").lower()
        if "health_profile" in q_lower:
            return {"health_profile": {"tz_offset_minutes": 0}}
        if "chunk_kind" in q_lower:
            return {"chunk_kind": "rolling_refill"}
        return None
    mock_query.side_effect = query_side

    with patch("cron_tasks._resolve_previous_chunk_window", _stub_resolve_window(0, 3)):
        result = _check_chunk_learning_ready(
            user_id="user-p17-2",
            meal_plan_id="plan-p17-2",
            week_number=5,
            days_offset=3,
            plan_data=plan_data,
            snapshot=snapshot,
        )

    # Resultado esperado: ready (chunk_4 del refill terminó en 2026-05-03, hoy 2026-05-04).
    # El override NO debe activarse porque chunk_4 es del refill.
    reason = result.get("reason")
    assert reason != "prev_chunk_day_not_yet_elapsed"


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_p1_7_no_continuation_marker_uses_legacy(mock_query, mock_now):
    """
    Sin `_is_continuation`, el gate usa el cálculo legacy intacto.
    """
    from cron_tasks import _check_chunk_learning_ready
    today = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    mock_now.return_value = today

    snapshot = {"form_data": {"_plan_start_date": "2026-05-01"}}
    plan_data = _build_plan_data_with_prior_days(6)

    def query_side(q, params=None, **kw):
        if "health_profile" in (q or "").lower():
            return {"health_profile": {"tz_offset_minutes": 0}}
        return None
    mock_query.side_effect = query_side

    with patch("cron_tasks._resolve_previous_chunk_window", _stub_resolve_window(0, 3)):
        result = _check_chunk_learning_ready(
            user_id="user-p17-3",
            meal_plan_id="plan-p17-3",
            week_number=2,
            days_offset=3,
            plan_data=plan_data,
            snapshot=snapshot,
        )

    # SQL calls: solo health_profile + posibles otras (no chunk_kind lookup)
    chunk_kind_calls = [
        c for c in mock_query.call_args_list
        if "chunk_kind" in (c.args[0] if c.args else "").lower()
    ]
    assert len(chunk_kind_calls) == 0, (
        "Sin _is_continuation, el gate NO debe hacer lookup de chunk_kind."
    )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_p1_7_continuation_without_anchor_uses_legacy(mock_query, mock_now):
    """`_is_continuation=True` pero sin anchor_iso → cálculo legacy."""
    from cron_tasks import _check_chunk_learning_ready
    today = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    mock_now.return_value = today

    snapshot = {
        "form_data": {
            "_plan_start_date": "2026-05-01",
            "_is_continuation": True,
            # _continuation_anchor_iso ausente
        },
    }
    plan_data = _build_plan_data_with_prior_days(6)

    def query_side(q, params=None, **kw):
        if "health_profile" in (q or "").lower():
            return {"health_profile": {"tz_offset_minutes": 0}}
        return None
    mock_query.side_effect = query_side

    with patch("cron_tasks._resolve_previous_chunk_window", _stub_resolve_window(0, 3)):
        _check_chunk_learning_ready(
            user_id="user-p17-4",
            meal_plan_id="plan-p17-4",
            week_number=2,
            days_offset=3,
            plan_data=plan_data,
            snapshot=snapshot,
        )

    chunk_kind_calls = [
        c for c in mock_query.call_args_list
        if "chunk_kind" in (c.args[0] if c.args else "").lower()
    ]
    assert len(chunk_kind_calls) == 0, (
        "Sin anchor_iso, el gate NO debe hacer lookup de chunk_kind."
    )


@patch("cron_tasks._dt_p0b_now")
@patch("cron_tasks.execute_sql_query")
def test_p1_7_malformed_anchor_falls_back_to_legacy(mock_query, mock_now):
    """anchor_iso con formato inválido → fallback a legacy sin crashear."""
    from cron_tasks import _check_chunk_learning_ready
    today = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)
    mock_now.return_value = today

    snapshot = {
        "form_data": {
            "_plan_start_date": "2026-05-01",
            "_is_continuation": True,
            "_continuation_anchor_iso": "not-a-valid-iso-date",
        },
    }
    plan_data = _build_plan_data_with_prior_days(6)

    def query_side(q, params=None, **kw):
        q_lower = (q or "").lower()
        if "health_profile" in q_lower:
            return {"health_profile": {"tz_offset_minutes": 0}}
        if "chunk_kind" in q_lower:
            return {"chunk_kind": "initial_plan"}
        return None
    mock_query.side_effect = query_side

    with patch("cron_tasks._resolve_previous_chunk_window", _stub_resolve_window(0, 3)):
        # No debe crashear pese al anchor malformado.
        result = _check_chunk_learning_ready(
            user_id="user-p17-5",
            meal_plan_id="plan-p17-5",
            week_number=2,
            days_offset=3,
            plan_data=plan_data,
            snapshot=snapshot,
        )

    # El gate completa el flow (no excepciones propagadas)
    assert isinstance(result, dict)


def test_p1_7_week_1_does_not_apply_anchor():
    """
    week_number=1 retorna ready=True desde el inicio del gate (first_chunk),
    así que el anchor lookup ni se intenta.
    """
    from cron_tasks import _check_chunk_learning_ready
    snapshot = {
        "form_data": {
            "_plan_start_date": "2026-05-01",
            "_is_continuation": True,
            "_continuation_anchor_iso": "2026-05-01",
        },
    }
    result = _check_chunk_learning_ready(
        user_id="u",
        meal_plan_id="p",
        week_number=1,
        days_offset=0,
        plan_data={"days": []},
        snapshot=snapshot,
    )
    assert result["ready"] is True
    assert result["reason"] == "first_chunk"


def test_p1_7_continuation_marker_keys_documented_in_constants():
    """Smoke check: los keys del snapshot existen como strings literales en código."""
    import cron_tasks
    src = open(cron_tasks.__file__, encoding="utf-8").read()
    assert '"_is_continuation"' in src
    assert '"_continuation_anchor_iso"' in src
