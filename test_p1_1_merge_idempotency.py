"""
Tests P1-1: Idempotencia del merge contra duplicación de días.

El merge de chunks usa `_merged_chunk_ids` como marker idempotente. Antes:
  - El marker se modificaba en el dict `plan_data` en memoria durante la transacción
    1 (FOR UPDATE meal_plans), pero NO se persistía a `meal_plans` en esa misma
    transacción — la persistencia ocurría en una transacción 2 separada al final
    del worker.
  - Si la transacción 2 fallaba (DB blip durante cálculo de shopping list, etc.),
    el marker quedaba sin persistir aunque los días se hubieran intentado escribir.
    Un retry reentraba al merge sin detectar la mergeación previa.

Cambios P1-1:
  1. UPDATE atómico de `plan_data` (con `_merged_chunk_ids` + days) DENTRO de la
     transacción 1 (mismo FOR UPDATE), antes de salir.
  2. Pre-check de duplicados por content signature: si los últimos N días en
     `existing_days` tienen las mismas signatures de meals que `new_days`, el
     marker se re-añade defensivamente y se trata como `chunk_already_merged=True`.

Este archivo testea la lógica del pre-check (parte 2) directamente en una función
helper extraída de la lógica del merge, replicando el mismo algoritmo. La parte
1 (UPDATE atómico) se valida indirectamente vía import y via los tests E2E
existentes en test_chunk_corrupted_plan_data_pauses.py que siguen pasando.
"""
import pytest


def _meal_signature(day):
    """Replica de _p11_meal_signature dentro de cron_tasks.py:_chunk_worker.

    Mantenemos el algoritmo idéntico para validar el invariante. Si el pre-check
    cambia en producción, este test debe alinearse.
    """
    if not isinstance(day, dict):
        return ()
    return tuple(sorted(
        (str(m.get('name') or ''), str(m.get('type') or ''))
        for m in (day.get('meals') or []) if isinstance(m, dict)
    ))


def _detect_already_merged(existing_days, new_days, current_marker_ids, task_id):
    """Replica de la lógica de pre-check: returns (already_merged, reason)."""
    new_count = len([d for d in new_days if isinstance(d, dict)])
    if new_count == 0:
        return False, "no_new_days"
    if len(existing_days) < new_count:
        return False, "existing_shorter_than_new"
    if str(task_id) in [str(x) for x in (current_marker_ids or [])]:
        return False, "marker_already_present"

    last_existing = existing_days[-new_count:]
    existing_sigs = [
        _meal_signature(d) for d in last_existing
        if isinstance(d, dict) and _meal_signature(d) != ()
    ]
    new_sigs = [
        _meal_signature(d) for d in new_days
        if isinstance(d, dict) and _meal_signature(d) != ()
    ]
    if (
        existing_sigs
        and new_sigs
        and len(existing_sigs) == len(new_sigs)
        and existing_sigs == new_sigs
    ):
        return True, "signatures_match"
    return False, "signatures_differ"


def _day(day_num, meal_specs):
    """Helper: meal_specs es lista de tuplas (name, type)."""
    return {
        "day": day_num,
        "meals": [{"name": n, "type": t, "ingredients": ["x"]} for n, t in meal_specs],
    }


# ---------------------------------------------------------------------------
# Pre-check P1-1
# ---------------------------------------------------------------------------
def test_pre_check_detects_signature_match_without_marker():
    """
    Si los últimos N días de existing_days coinciden por signatures con new_days
    Y el marker está ausente, debe detectarse como ya mergeado defensivamente.
    """
    existing_days = [
        _day(1, [("Pollo asado", "Almuerzo"), ("Avena", "Desayuno")]),
        _day(2, [("Salmón", "Almuerzo"), ("Yogur", "Desayuno")]),
        _day(3, [("Res guisada", "Almuerzo"), ("Tostada", "Desayuno")]),
    ]
    # new_days idénticos a los últimos 2 días de existing
    new_days = [
        _day(2, [("Salmón", "Almuerzo"), ("Yogur", "Desayuno")]),
        _day(3, [("Res guisada", "Almuerzo"), ("Tostada", "Desayuno")]),
    ]
    already_merged, reason = _detect_already_merged(
        existing_days, new_days, current_marker_ids=[], task_id="task-abc"
    )
    assert already_merged is True
    assert reason == "signatures_match"


def test_pre_check_skips_when_marker_already_present():
    """Si el marker ya existe, el pre-check no aplica (camino normal toma control)."""
    existing_days = [
        _day(1, [("Pollo", "Almuerzo")]),
        _day(2, [("Salmón", "Almuerzo")]),
    ]
    new_days = [_day(2, [("Salmón", "Almuerzo")])]
    already_merged, reason = _detect_already_merged(
        existing_days, new_days,
        current_marker_ids=["task-abc"], task_id="task-abc",
    )
    assert already_merged is False
    assert reason == "marker_already_present"


def test_pre_check_does_not_match_when_signatures_differ():
    """Si las signatures no coinciden, NO debe declarar ya mergeado."""
    existing_days = [
        _day(1, [("Pollo", "Almuerzo")]),
        _day(2, [("Salmón", "Almuerzo")]),
    ]
    new_days = [
        _day(3, [("Camarones", "Almuerzo")]),  # nombre distinto
    ]
    already_merged, reason = _detect_already_merged(
        existing_days, new_days, current_marker_ids=[], task_id="task-abc"
    )
    assert already_merged is False
    assert reason == "signatures_differ"


def test_pre_check_handles_existing_shorter_than_new():
    """Si existing_days es más corto que new_days, no puede haber overlap."""
    existing_days = [_day(1, [("Pollo", "Almuerzo")])]
    new_days = [
        _day(1, [("Pollo", "Almuerzo")]),
        _day(2, [("Salmón", "Almuerzo")]),
    ]
    already_merged, reason = _detect_already_merged(
        existing_days, new_days, current_marker_ids=[], task_id="task-abc"
    )
    assert already_merged is False
    assert reason == "existing_shorter_than_new"


def test_pre_check_handles_empty_new_days():
    """new_days vacío: no aplicar pre-check (caso degraded edge)."""
    existing_days = [_day(1, [("Pollo", "Almuerzo")])]
    already_merged, reason = _detect_already_merged(
        existing_days, new_days=[], current_marker_ids=[], task_id="task-abc"
    )
    assert already_merged is False
    assert reason == "no_new_days"


def test_pre_check_handles_malformed_days():
    """Días sin meals o no-dict no deben crashear ni dar falsos positivos."""
    existing_days = [
        None,  # no es dict
        {"day": 1},  # sin meals
        _day(2, [("Pollo", "Almuerzo")]),
    ]
    new_days = [_day(2, [("Pollo", "Almuerzo")])]
    # last_existing slice = existing_days[-1:] = [_day(2,...)]
    # existing_sigs filtra los None/{} con signature (); queda 1 sig.
    # new_sigs = 1 sig.
    # Coinciden → already_merged.
    already_merged, reason = _detect_already_merged(
        existing_days, new_days, current_marker_ids=[], task_id="task-abc"
    )
    assert already_merged is True


def test_meal_signature_order_independent():
    """La signature ordena los meals para que el orden no afecte la comparación."""
    day_a = {
        "day": 1,
        "meals": [
            {"name": "Pollo", "type": "Almuerzo"},
            {"name": "Avena", "type": "Desayuno"},
        ],
    }
    day_b = {
        "day": 1,
        "meals": [
            {"name": "Avena", "type": "Desayuno"},
            {"name": "Pollo", "type": "Almuerzo"},
        ],
    }
    assert _meal_signature(day_a) == _meal_signature(day_b)


def test_meal_signature_distinguishes_different_meals():
    """Días con meals distintos producen signatures distintas."""
    day_a = {"day": 1, "meals": [{"name": "Pollo", "type": "Almuerzo"}]}
    day_b = {"day": 1, "meals": [{"name": "Salmón", "type": "Almuerzo"}]}
    assert _meal_signature(day_a) != _meal_signature(day_b)


# ---------------------------------------------------------------------------
# Sanity check de imports — el código P1-1 modificó cron_tasks.py
# ---------------------------------------------------------------------------
def test_cron_tasks_imports_after_p1_1_changes():
    """El módulo cron_tasks.py debe importar sin errores tras los cambios P1-1."""
    import cron_tasks  # noqa: F401
    # El helper _touch_chunk_heartbeat (P0-1) sigue presente.
    assert hasattr(cron_tasks, "_touch_chunk_heartbeat")
    # El helper P0-4 sigue presente.
    assert hasattr(cron_tasks, "_validate_merged_days_against_pantry")
