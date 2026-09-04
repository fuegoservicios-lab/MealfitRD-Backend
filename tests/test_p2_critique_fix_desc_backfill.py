"""[P2-CRITIQUE-FIX-DESC-BACKFILL · 2026-09-04] El corrector de días ya no pierde la corrección por `desc`.

Tres generaciones seguidas del dueño (10:47, 12:32 y la regen quirúrgica post-aprobación):
«⚠️ [SELF-CRITIQUE] Error corrigiendo Día 3: 4 validation errors for SingleDayPlanModel —
meals.0.desc Field required … meals.3.desc Field required». El modelo devolvió las 4 comidas sin
descripción, pydantic tiró el día ENTERO, quedó `_critique_unresolved`, y la regen post-aprobación
repitió el mismo fallo (45-75 s + 44-64 s por plan, para nada).

Arreglo: el corrector (autocrítica, regen quirúrgica y pro-fallback) parsea con
`SingleDayCorrectionModel` (`desc` opcional) y `_backfill_corrected_day_desc` rellena: mismo nombre
que el plato original del slot ⇒ copia su descripción; plato nuevo ⇒ descripción mínima desde el
nombre y los primeros ingredientes. La generación normal sigue exigiendo `desc` (SingleDayPlanModel).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas import MealCorrectionModel, SingleDayCorrectionModel, SingleDayPlanModel

_BACKEND = Path(__file__).resolve().parents[1]
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _meal(name, desc=None, **kw):
    d = {"meal": kw.get("meal", "Desayuno"), "name": name, "prep_time": "10 min", "cals": 500,
         "ingredients": kw.get("ingredients", ["2 huevos", "50 g de avena"]), "recipe": ["Mise en place: a"]}
    if desc is not None:
        d["desc"] = desc
    return d


def test_lenient_model_accepts_meals_without_desc_but_strict_one_does_not():
    day = {"day": 3, "meals": [_meal("Avena con huevo"), _meal("Pollo guisado", meal="Almuerzo")]}
    parsed = SingleDayCorrectionModel(**day)
    assert [m.desc for m in parsed.meals] == [None, None]
    with pytest.raises(ValidationError):
        SingleDayPlanModel(**day)


def test_backfill_copies_desc_when_the_dish_is_the_same_and_synthesizes_when_new():
    import graph_orchestrator as go
    target = {"meals": [_meal("Avena con huevo", desc="Avena cremosa con huevo revuelto."),
                        _meal("Pollo guisado", desc="Pollo guisado criollo.", meal="Almuerzo")]}
    corrected = {"day": 3, "meals": [
        _meal("Avena con huevo"),                                  # mismo plato → copia
        _meal("Pescado al horno", meal="Almuerzo", ingredients=["150 g de Filete de pescado", "1 batata", "1 cda de aceite"]),  # nuevo → sintetiza
        _meal("Yogurt con fruta", meal="Merienda", desc="Ya venía con desc."),
    ]}
    n = go._backfill_corrected_day_desc(corrected, target)
    assert n == 2
    assert corrected["meals"][0]["desc"] == "Avena cremosa con huevo revuelto."
    assert corrected["meals"][1]["desc"] == "Pescado al horno, con 150 g de Filete de pescado, 1 batata, 1 cda de aceite."
    assert corrected["meals"][2]["desc"] == "Ya venía con desc."
    assert go._backfill_corrected_day_desc({"meals": [_meal("X")]}, {}) == 1


def test_the_three_correctors_use_the_lenient_model_and_backfill():
    assert _SRC.count(".with_structured_output(SingleDayCorrectionModel") == 4
    assert ".with_structured_output(SingleDayPlanModel" not in _SRC
    assert _SRC.count("_backfill_corrected_day_desc(corrected_day, target_day)") == 2
    assert "_backfill_corrected_day_desc(corrected, {})" in _SRC
    # los generadores de días siguen validando estricto
    assert "SingleDayPlanModel(**parsed_dict).model_dump()" in _SRC


def test_marker_present():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "P2-CRITIQUE-FIX-DESC-BACKFILL · 2026-09-04" in app
