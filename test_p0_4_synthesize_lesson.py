"""[P0-4] Tests para _synthesize_last_chunk_learning_from_plan_days.

Cubre el last-resort cuando plan_chunk_queue.learning_metrics está NULL: el sistema
debe extraer señal mínima desde meal_plans.plan_data.days para que chunk N+1 no
arranque con dict vacío y regenere los mismos platos del chunk previo.

Casos: días con tag de chunk, días sin tag (planes legacy), filtrado por status,
ingredientes en formato dict vs string, ausencia de datos.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from cron_tasks import _synthesize_last_chunk_learning_from_plan_days


def _days_chunk_tagged():
    return [
        {"week_number": 1, "meals": [
            {"name": "Pollo asado con arroz", "ingredients": ["pollo", "arroz blanco"], "status": "ok"},
            {"name": "Tortilla de huevos", "ingredients": [{"name": "huevos"}, {"name": "cebolla"}], "status": "ok"},
        ]},
        {"week_number": 1, "meals": [
            {"name": "Pollo a la plancha", "ingredients": ["pollo", "brócoli"], "status": "ok"},
            {"name": "Arroz con vegetales", "ingredients": ["arroz blanco", "zanahoria"], "status": "swapped_out"},
        ]},
        {"week_number": 2, "meals": [
            {"name": "Pescado al horno", "ingredients": ["pescado", "limón"], "status": "ok"},
        ]},
    ]


def test_extracts_only_target_week():
    res = _synthesize_last_chunk_learning_from_plan_days(
        meal_plan_id="plan-x", target_week=1, prior_plan_data={"days": _days_chunk_tagged()}
    )
    assert res is not None, "debió sintetizar lección desde días tagueados como week=1"
    assert res["chunk"] == 1
    assert res["synthesized_from_plan_days"] is True
    assert res["metrics_unavailable"] is True
    assert res["low_confidence"] is True
    assert "Pollo asado con arroz" in res["repeated_meal_names"]
    assert "Tortilla de huevos" in res["repeated_meal_names"]
    assert "Pollo a la plancha" in res["repeated_meal_names"]
    assert "Pescado al horno" not in res["repeated_meal_names"], "no debe incluir week=2"
    assert "Arroz con vegetales" not in res["repeated_meal_names"], "no debe incluir swapped_out"


def test_filters_swapped_skipped_rejected():
    days = [{"week_number": 1, "meals": [
        {"name": "Plato A", "ingredients": ["a"], "status": "ok"},
        {"name": "Plato B", "ingredients": ["b"], "status": "swapped_out"},
        {"name": "Plato C", "ingredients": ["c"], "status": "skipped"},
        {"name": "Plato D", "ingredients": ["d"], "status": "rejected"},
    ]}]
    res = _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": days})
    assert res is not None
    assert res["repeated_meal_names"] == ["Plato A"]


def test_accepts_legacy_days_without_chunk_tag():
    days = [
        {"meals": [{"name": "Plato Legacy", "ingredients": ["arroz"], "status": "ok"}]},
    ]
    res = _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": days})
    assert res is not None
    assert res["synthesized_chunk_tag_present"] is False
    assert "Plato Legacy" in res["repeated_meal_names"]


def test_returns_none_when_no_days():
    assert _synthesize_last_chunk_learning_from_plan_days("p", 1, {}) is None
    assert _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": []}) is None
    assert _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": None}) is None


def test_returns_none_when_target_week_has_no_consumed_meals():
    days = [{"week_number": 2, "meals": [{"name": "X", "ingredients": ["x"], "status": "ok"}]}]
    res = _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": days})
    assert res is None, "no hay días en target_week=1, debe devolver None"


def test_handles_dict_ingredients():
    days = [{"week_number": 1, "meals": [
        {"name": "Plato", "ingredients": [{"name": "salmón"}, {"display_string": "100g de quinoa"}], "status": "ok"},
    ]}]
    res = _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": days})
    assert res is not None
    assert len(res["repeated_bases"]) >= 1, "debe extraer al menos una base de ingredientes dict"


def test_counters_are_zero_no_fake_signal():
    """Last-resort no inventa repeticiones — sólo provee nombres/bases para que el LLM evite."""
    res = _synthesize_last_chunk_learning_from_plan_days(
        "p", 1, {"days": _days_chunk_tagged()}
    )
    assert res is not None
    assert res["repeat_pct"] == 0
    assert res["ingredient_base_repeat_pct"] == 0
    assert res["rejection_violations"] == 0
    assert res["allergy_violations"] == 0
    assert res["fatigued_violations"] == 0


def test_caps_lists_to_reasonable_size():
    """Defensa contra prompt-bloat si el chunk tiene 100 platos."""
    days = [{"week_number": 1, "meals": [
        {"name": f"Plato {i}", "ingredients": [f"ing_{i}"], "status": "ok"}
        for i in range(50)
    ]}]
    res = _synthesize_last_chunk_learning_from_plan_days("p", 1, {"days": days})
    assert res is not None
    assert len(res["repeated_meal_names"]) <= 8
    assert len(res["repeated_bases"]) <= 10


if __name__ == "__main__":
    test_extracts_only_target_week()
    test_filters_swapped_skipped_rejected()
    test_accepts_legacy_days_without_chunk_tag()
    test_returns_none_when_no_days()
    test_returns_none_when_target_week_has_no_consumed_meals()
    test_handles_dict_ingredients()
    test_counters_are_zero_no_fake_signal()
    test_caps_lists_to_reasonable_size()
    print("OK: 8/8 tests pasaron")
