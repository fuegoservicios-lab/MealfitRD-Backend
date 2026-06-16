"""[P0-3] Gate `_check_chunk_learning_ready` consulta plan_chunk_queue.

Antes del fix:
    Si `plan_data.days` no contenía los días del chunk previo, el gate devolvía
    `ready=True, reason="missing_previous_chunk_days"` (fail-open). Esto creaba
    una race: cuando el chunk N seguía en estado activo (pending/processing/stale)
    y aún no había mergeado days a plan_data, el gate de N+1 dejaba pasar la
    dispatch con learning vacío. La pickup query de `process_plan_chunk_queue`
    serializa por `NOT EXISTS (SELECT processing del mismo plan)`, pero el gate
    también se invoca desde `_recover_pantry_paused_chunks` (~línea 4455) que
    NO pasa por la pickup query — esa ruta sí podía race.

Después del fix:
    Antes del fail-open, el gate consulta plan_chunk_queue: si hay chunks con
    `week_number < current` en estados activos (pending/processing/stale),
    devuelve `ready=False, reason="prev_chunk_still_active_in_queue"`. Si no
    hay chunks anteriores activos, mantiene el fail-open original (escenarios
    legítimos: chunk N falló permanentemente, plan corrupto, etc.) para no
    deadlockear el plan.

Estados terminales (completed/failed/cancelled) NO bloquean: el gate asume que
el housekeeping ya reconcilió el estado y plan_data.days está vacío por una
razón legítima (recovery exhausted, etc.).
"""
from unittest.mock import patch

import pytest

from cron_tasks import _check_chunk_learning_ready


def _base_kwargs(plan_data=None, week_number=2, days_offset=3):
    """Setup que dispara la rama 'previous_chunk_days vacío' sin ruido SQL.

    week_number=2 hace que _resolve_previous_chunk_window retorne (0, days_offset)
    sin consultar SQL (línea 8362-8363). _plan_start_date presente skipea el
    fallback SQL de start_date. plan_data.days vacío fuerza el flujo al nuevo
    check de queue.
    """
    return {
        "user_id": "user-test",
        "meal_plan_id": "plan-test-uuid",
        "week_number": week_number,
        "days_offset": days_offset,
        "plan_data": plan_data if plan_data is not None else {"days": []},
        "snapshot": {
            "form_data": {
                "_plan_start_date": "2026-04-01",
                "tz_offset_minutes": 0,
            },
            "totalDays": 7,
        },
    }


def test_gate_blocks_when_prior_chunks_active_in_queue():
    """plan_data.days vacío Y queue tiene chunks anteriores activos →
    ready=False con reason='prev_chunk_still_active_in_queue'."""
    with patch("cron_tasks.execute_sql_query", return_value={"n": 1}):
        result = _check_chunk_learning_ready(**_base_kwargs())
    assert result["ready"] is False
    assert result["reason"] == "prev_chunk_still_active_in_queue"
    assert result["_active_prior_chunks"] == 1


def test_gate_blocks_when_multiple_prior_chunks_active():
    """Cuenta exacta de chunks activos surface en _active_prior_chunks para
    diagnóstico operacional."""
    with patch("cron_tasks.execute_sql_query", return_value={"n": 3}):
        result = _check_chunk_learning_ready(**_base_kwargs())
    assert result["ready"] is False
    assert result["_active_prior_chunks"] == 3


def test_gate_falls_open_when_no_prior_chunks_active():
    """plan_data.days vacío PERO queue confirma que NINGÚN chunk anterior
    está activo (todos terminales o no existen) → preservar fail-open
    original con reason='missing_previous_chunk_days'. Esto evita deadlockear
    el plan cuando chunk N falló permanentemente o el plan está en recovery.
    """
    with patch("cron_tasks.execute_sql_query", return_value={"n": 0}):
        result = _check_chunk_learning_ready(**_base_kwargs())
    assert result["ready"] is True
    assert result["reason"] == "missing_previous_chunk_days"


def test_gate_falls_open_when_queue_query_returns_none():
    """Si la query de queue retorna None (DB blip o tabla vacía), tratamos
    como n=0 — no bloqueamos por fallos de SQL aislados."""
    with patch("cron_tasks.execute_sql_query", return_value=None):
        result = _check_chunk_learning_ready(**_base_kwargs())
    assert result["ready"] is True
    assert result["reason"] == "missing_previous_chunk_days"


def test_gate_falls_open_on_query_exception():
    """Si la query de queue tira excepción, fail-open con log warning. NO
    bloqueamos chunks por flaps transientes de DB; el TOCTOU guard interno
    en el merge (líneas 13100) y la pickup query siguen siendo defensa final.
    """
    def _raise(*a, **kw):
        raise RuntimeError("connection refused")
    with patch("cron_tasks.execute_sql_query", side_effect=_raise):
        result = _check_chunk_learning_ready(**_base_kwargs())
    assert result["ready"] is True
    assert result["reason"] == "missing_previous_chunk_days"


def test_gate_query_uses_active_states_only():
    """Regression guard: el SQL debe filtrar por status activos
    ('pending','processing','stale') NO por terminales. Si por refactor el
    filtro incluyera 'completed', el gate bloquearía indefinidamente cualquier
    plan multi-chunk porque chunks ya completados serían tratados como
    'todavía activos'."""
    # [test fix · P0-3/STALE-RECOVERY] El gate ahora hace DOS execute_sql_query
    # en la rama `not previous_chunk_days`: (1) el COUNT de chunks activos en
    # queue (el que queremos verificar) y, si n=0, (2) un re-read fresco
    # `SELECT plan_data FROM meal_plans` (P0-3 stale-recovery). Capturar solo la
    # última llamada agarraba la #2 → assert fallaba. Coleccionamos todas y
    # localizamos la query del filtro de estados por su contenido.
    calls = []

    def _capture(query, params, fetch_one=None):
        calls.append({"query": query, "params": params})
        return {"n": 0}

    with patch("cron_tasks.execute_sql_query", side_effect=_capture):
        _check_chunk_learning_ready(**_base_kwargs())

    queue_calls = [c for c in calls if "plan_chunk_queue" in c["query"]]
    assert queue_calls, (
        "El gate debe consultar plan_chunk_queue por estados activos antes del "
        "fail-open. No se capturó ninguna query contra plan_chunk_queue."
    )
    captured = queue_calls[0]
    q = captured["query"]
    # Estados activos que SÍ deben bloquear:
    assert "'pending'" in q
    assert "'processing'" in q
    assert "'stale'" in q
    # Estados terminales que NO deben bloquear (no aparecer en el filtro):
    assert "'completed'" not in q, (
        "El filtro NO debe incluir 'completed' — un chunk completed no debe "
        "bloquear al siguiente chunk; eso causaría deadlock perpetuo en planes "
        "multi-chunk."
    )
    assert "'cancelled'" not in q
    # Comparación por week_number con < (no <=) para no incluirse a sí mismo.
    assert "week_number < %s" in q
    # Param order: meal_plan_id, week_number.
    assert captured["params"] == ("plan-test-uuid", 2)


def test_first_chunk_skips_queue_check():
    """Chunk 1 (week_number=1) tiene su propia rama early-return ('first_chunk')
    ANTES de cualquier SQL. El queue check NO debe activarse para el primer
    chunk, que por definición no tiene anteriores."""
    captured = {"called": 0}

    def _count_calls(*a, **kw):
        captured["called"] += 1
        return None

    kwargs = _base_kwargs(week_number=1, days_offset=0)
    with patch("cron_tasks.execute_sql_query", side_effect=_count_calls):
        result = _check_chunk_learning_ready(**kwargs)

    assert result["ready"] is True
    assert result["reason"] == "first_chunk"
    assert captured["called"] == 0, (
        "Chunk 1 no debería disparar ningún SQL (gate retorna 'first_chunk' "
        "early). Si pasó por queue check, hay regression en el orden de checks."
    )


def test_gate_check_does_not_trigger_when_previous_chunk_days_present():
    """Si plan_data.days SÍ contiene los días del chunk previo (caso normal
    post-merge), el queue check NO se ejecuta — la rama temporal gate
    (calendario) toma el control. Esto evita un SQL extra en el caso
    feliz que es el path mayoritario."""
    plan_data_with_prior_days = {
        "days": [
            {"day": 1, "meals": []},
            {"day": 2, "meals": []},
            {"day": 3, "meals": []},
        ]
    }
    captured = {"queries": []}

    def _record(query, params, fetch_one=None):
        captured["queries"].append(query)
        return None

    with patch("cron_tasks.execute_sql_query", side_effect=_record):
        # No nos interesa el resultado final (la rama temporal hace más SQL),
        # solo verificar que la query del queue check NO se hizo.
        try:
            _check_chunk_learning_ready(
                **_base_kwargs(plan_data=plan_data_with_prior_days)
            )
        except Exception:
            pass  # Cualquier fallo posterior está OK; sólo nos interesa qué SQL corrió.

    queue_check_queries = [
        q for q in captured["queries"]
        if "plan_chunk_queue" in q
        and "COUNT(*)" in q.upper()
        and "week_number <" in q
    ]
    assert len(queue_check_queries) == 0, (
        "El queue check P0-3 corrió aunque plan_data.days tenía los días del "
        "chunk previo. Debería ser exclusivo de la rama 'previous_chunk_days "
        "vacío' para no añadir SQL al path feliz."
    )
