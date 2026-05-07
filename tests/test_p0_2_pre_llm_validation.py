"""[P0-2] Pre-LLM validation TOCTOU guard.

Cierra la ventana entre el pickup (`SELECT FOR UPDATE SKIP LOCKED` en
`process_plan_chunk_queue`) y el submit al LLM dentro de `_chunk_worker`.
Durante esa ventana (~2500 líneas de prep work + I/O contra inventario,
profile, learning rebuilds), otro path puede haber:
  - Cancelado el chunk vía `save_new_meal_plan_atomic` (regeneración de plan).
  - Borrado el `meal_plan` (caso edge: el usuario lo elimina vía UI).
  - Cambiado el status del chunk por housekeeping (zombie rescue, etc.).

Sin este guard, el worker gasta tokens del LLM en un chunk ya muerto cuyo
merge será descartado por el TOCTOU guard interno (~líneas 13100). El guard
pre-LLM evita ese desperdicio.
"""
from unittest.mock import patch

import pytest

from cron_tasks import _validate_chunk_pre_llm, _P0_2_CHUNK_TERMINAL_STATES


def test_terminal_states_constant_lists_all_post_pickup_states():
    """Contrato: si `save_new_meal_plan_atomic` o el housekeeping puede
    transicionar el chunk a uno de estos estados durante la ventana TOCTOU,
    debe estar listado aquí. Si se añade un nuevo estado terminal sin
    actualizar esta tupla, el guard NO abortará y se quemarán tokens.
    """
    assert set(_P0_2_CHUNK_TERMINAL_STATES) == {"cancelled", "completed", "failed"}


def test_validation_ok_when_chunk_processing_and_plan_exists():
    """Caso feliz: chunk en 'processing' (estado normal post-pickup) y plan
    existe. El guard debe retornar 'ok' para que el LLM corra."""
    fake_row = {"chunk_status": "processing", "plan_exists": "plan-uuid-1"}
    with patch("cron_tasks.execute_sql_query", return_value=fake_row):
        result = _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")
    assert result == "ok"


@pytest.mark.parametrize("terminal_status", ["cancelled", "completed", "failed"])
def test_validation_aborts_when_chunk_in_terminal_state(terminal_status):
    """Si `save_new_meal_plan_atomic` (o cualquier path) transicionó el
    chunk a estado terminal mientras el worker preparaba el LLM call, el
    guard debe abortar para no gastar tokens en un chunk muerto."""
    fake_row = {"chunk_status": terminal_status, "plan_exists": "plan-uuid-1"}
    with patch("cron_tasks.execute_sql_query", return_value=fake_row):
        result = _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")
    assert result == "chunk_terminal", (
        f"Status terminal {terminal_status!r} debe disparar abort para no "
        f"quemar tokens. Si retorna 'ok', el LLM correrá y el merge será "
        f"descartado por el TOCTOU guard interno (waste)."
    )


def test_validation_detects_deleted_meal_plan():
    """Si el usuario borró el meal_plan entre pickup y LLM call, el guard
    debe detectarlo (LEFT JOIN devuelve plan_exists=NULL) y devolver
    'plan_missing' para que el caller cancele el chunk + libere reservas."""
    fake_row = {"chunk_status": "processing", "plan_exists": None}
    with patch("cron_tasks.execute_sql_query", return_value=fake_row):
        result = _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")
    assert result == "plan_missing"


def test_validation_handles_chunk_row_disappeared():
    """Caso muy improbable (la PK del chunk no debería desaparecer), pero si
    pasa, el guard debe abortar limpiamente sin proceder al LLM."""
    with patch("cron_tasks.execute_sql_query", return_value=None):
        result = _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")
    assert result == "chunk_unknown"


def test_validation_falls_back_to_continue_on_db_error():
    """Best-effort: si la query de validación falla por flap transitorio de
    DB, NO bloqueamos el chunk. El TOCTOU guard interno (dentro del
    FOR UPDATE del merge) es la red de seguridad final. Bloquear aquí por
    flaps haría que un blip de DB borre todos los chunks en flight."""
    def _raise(*a, **kw):
        raise RuntimeError("connection refused")
    with patch("cron_tasks.execute_sql_query", side_effect=_raise):
        result = _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")
    assert result == "validation_error"


def test_query_uses_left_join_to_detect_deleted_plan():
    """El SQL debe usar LEFT JOIN (no INNER) sobre meal_plans, para que el
    caso 'plan borrado' devuelva una row con plan_exists=NULL en vez de
    cero rows. Si pasara a INNER JOIN por error de refactor, el caso de
    plan borrado caería en 'chunk_unknown' (correcto pero confunde
    diagnóstico) o peor, en 'validation_error' si fuera por excepción.
    """
    captured = {}

    def _capture(query, params, fetch_one=None):
        captured["query"] = query
        captured["params"] = params
        return {"chunk_status": "processing", "plan_exists": "plan-uuid-1"}

    with patch("cron_tasks.execute_sql_query", side_effect=_capture):
        _validate_chunk_pre_llm("chunk-1", "plan-uuid-1", "user-1")

    assert "LEFT JOIN" in captured["query"].upper(), (
        "El SQL debe usar LEFT JOIN para detectar plan borrado vía "
        "plan_exists IS NULL en lugar de devolver cero rows."
    )
    assert captured["params"] == ("chunk-1",)
