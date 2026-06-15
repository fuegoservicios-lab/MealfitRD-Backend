"""[P2-CLINICAL-LAYER-FAILSAFE · 2026-06-15] Fail-secure de la capa clínica determinista.

Audit P2-9/P2-12: si la DB de nutrición no carga (import roto), la capa clínica corre INCOMPLETA
(sin quantize/micros/proveniencia ni el enforcement renal per-comida) — antes esto se omitía SILENCIOSO
(solo un warning). Ahora marca dura `_clinical_layer_incomplete` que viaja a la entrega → observable.

Validación determinista (sin LLM/créditos): se fuerza la falla de IngredientNutritionDB y se verifica
el marcador; el happy-path (catálogo disponible) NO marca.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def _plan():
    return {
        "days": [{"meals": [{"name": "Almuerzo", "ingredients": ["1 taza de arroz"],
                             "protein": 20, "carbs": 30, "fats": 10, "cals": 310}]}],
        "macros": {"protein": "100g", "carbs": "200g", "fats": "60g"},
        "calories": 1800,
    }


_NUTRITION = {"total_daily_macros": {"protein_str": "100g", "carbs_str": "200g", "fats_str": "60g"},
              "target_calories": 1800}


def test_marks_incomplete_when_db_unavailable(go, monkeypatch):
    import nutrition_db

    class _BoomDB:
        def __init__(self, *a, **k):
            raise RuntimeError("db down")

    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _BoomDB)
    plan = _plan()
    plan.pop("_clinical_layer_applied", None)
    result = go._apply_deterministic_clinical_layer(plan, {"gender": "female"}, dict(_NUTRITION))
    assert result.get("_clinical_layer_incomplete") is True
    assert "db_unavailable" in str(result.get("_clinical_layer_incomplete_reason"))


def test_healthy_path_not_marked_incomplete(go):
    """Con catálogo disponible (local), la capa corre completa → sin marcador + _clinical_layer_applied."""
    plan = _plan()
    plan.pop("_clinical_layer_applied", None)
    result = go._apply_deterministic_clinical_layer(plan, {"gender": "female"}, dict(_NUTRITION))
    assert not result.get("_clinical_layer_incomplete")
    assert result.get("_clinical_layer_applied") is True


def test_marker_present(go):
    from pathlib import Path
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert "P2-CLINICAL-LAYER-FAILSAFE" in src
    assert '"_clinical_layer_incomplete"' in src
