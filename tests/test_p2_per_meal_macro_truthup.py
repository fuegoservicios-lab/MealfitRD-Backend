"""[P2-PER-MEAL-MACRO-TRUTHUP · 2026-06-25] Honestidad de macros por-comida.

Bug observado (cuenta angelobrito915, plan cedcc3d2): el plato "Huevos Duros con una Pizca de
Sal y Pimienta" mostraba 41g proteína / 306 kcal para 3 huevos enteros, que aportan ~19g/208kcal
(verificado contra el catálogo en el VPS). La etiqueta inflada del LLM hace que el protein-closer
SALTE la comida (`cur_p >= target` con un número mentiroso) → el día puede quedar bajo el piso de
proteína mientras el plato dice que cumple.

Fix: `_truth_up_meal_macros_from_catalog` recalcula los macros desde el catálogo (suma de
`macros_from_ingredient_string` sobre ingredients_raw) y los snapea a la verdad cuando la cobertura
es alta y la divergencia es grande. Corre ANTES del solver/closer (band-safe). Las líneas de
condimento ("al gusto") se excluyen del denominador de cobertura. Knob `MEALFIT_PER_MEAL_MACRO_TRUTHUP`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go
from graph_orchestrator import _truth_up_meal_macros_from_catalog


class _FakeDB:
    """Catálogo de prueba: resuelve algunos strings a macros reales, otros None."""
    _TABLE = {
        "3 huevos enteros": {"protein": 18.9, "carbs": 1.1, "fats": 14.3, "kcal": 208.4},
        "2 huevos": {"protein": 12.6, "carbs": 0.7, "fats": 9.5, "kcal": 138.9},
        "25g de almendras": {"protein": 6.5, "carbs": 4.0, "fats": 12.5, "kcal": 155.3},
        "150g de pechuga de pollo": {"protein": 46.5, "carbs": 0.0, "fats": 5.4, "kcal": 247.5},
    }

    def macros_from_ingredient_string(self, s):
        return self._TABLE.get(str(s).strip())


def _eggs_meal(protein_label):
    return {
        "name": "Huevos Duros con una Pizca de Sal y Pimienta",
        "meal": "Merienda",
        "protein": protein_label, "carbs": 1, "fats": 14, "cals": 306,
        "ingredients_raw": ["3 huevos enteros", "Sal al gusto", "Pimienta negra al gusto"],
    }


def test_corrige_proteina_inflada_a_la_verdad_del_catalogo():
    assert go.MACRO_TRUTHUP_ENABLED is True  # default
    m = _eggs_meal(41)  # etiqueta inflada del LLM
    changed = _truth_up_meal_macros_from_catalog(m, _FakeDB())
    assert changed is True
    assert m["protein"] == 19, f"esperaba ~19g (3 huevos), no 41g: {m['protein']}"
    assert m["cals"] == 208
    assert m["macros"][0] == "P:19g"


def test_condimentos_al_gusto_no_bajan_la_cobertura():
    """'Sal al gusto' y 'Pimienta al gusto' se excluyen del denominador → 1 línea con cantidad,
    1 resuelta = cobertura 1.0 (no 1/3=0.33). Sin esto, la comida nunca se corregiría."""
    m = _eggs_meal(41)
    assert _truth_up_meal_macros_from_catalog(m, _FakeDB()) is True


def test_no_toca_comida_ya_coherente():
    """Si la etiqueta ya coincide con los ingredientes (dentro de tolerancia), no la cambia."""
    m = _eggs_meal(19)  # ya honesta
    changed = _truth_up_meal_macros_from_catalog(m, _FakeDB())
    assert changed is False
    assert m["protein"] == 19


def test_no_toca_cuando_cobertura_baja():
    """Mayoría de ingredientes con cantidad NO resueltos → la estimación del LLM es lo mejor que hay."""
    m = {
        "name": "Plato Exótico",
        "meal": "Almuerzo",
        "protein": 40, "carbs": 50, "fats": 20, "cals": 540,
        # 1 resuelve, 2 no (con cantidad, no condimento) → cobertura 1/3 < 0.6
        "ingredients_raw": ["25g de almendras", "100g de fruta del dragón", "80g de quinoa roja andina"],
    }
    changed = _truth_up_meal_macros_from_catalog(m, _FakeDB())
    assert changed is False
    assert m["protein"] == 40


def test_knob_off_no_hace_nada(monkeypatch):
    monkeypatch.setattr(go, "MACRO_TRUTHUP_ENABLED", False)
    m = _eggs_meal(41)
    assert _truth_up_meal_macros_from_catalog(m, _FakeDB()) is False
    assert m["protein"] == 41  # intacto


def test_knob_registrado():
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_PER_MEAL_MACRO_TRUTHUP" in get_knobs_registry_snapshot()
