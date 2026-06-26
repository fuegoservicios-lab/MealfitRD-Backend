"""[P2-DISH-COHERENCE · 2026-06-25] Coherencia de platos en el motor de generación.

Bug observado (cuenta angelobrito915, plan cedcc3d2, 2026-06-25):
  1. El protein closer (`_close_protein_gap_for_meal`) decidía ligero-vs-principal leyendo el
     NOMBRE del plato, no el SLOT → una merienda cuyo nombre no traía palabra-pista ("Puñado de
     Almendras y Nueces con Ralladura de Limón") caía a la rama "principal" y recibía CARNE/PESCADO
     (115g de camarón) como proteína añadida. Incongruente con una merienda.
  2. El closer NUNCA reflejaba en el `name` la proteína que añadía como "fuente principal" → el
     plato ESCONDÍA su proteína principal (camarón invisible en el nombre).
  3. `build_variety_report` no detectaba la misma fruta repetida el mismo día (mango en desayuno
     Y merienda del Día 1).

Fix (knob `MEALFIT_CLOSER_DISH_COHERENCE`, default ON):
  - `_meal_slot_is_light`: el SLOT (merienda/desayuno) manda → lácteo/huevo, nunca carne/pescado.
  - `_reflect_added_protein_in_name`: refleja la proteína añadida en el nombre (idempotente).
  - `build_variety_report`: nuevo contador `fruit_repeats` (advisory).
  - Prompt `day_generator.py` regla 15(f): no repetir fruta el mismo día + no fruta dulce con huevo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go
from graph_orchestrator import (
    build_variety_report,
    _meal_slot_is_light,
    _reflect_added_protein_in_name,
    _close_protein_gap_for_meal,
)


def _sa(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


class _Info:
    """Stub del IngredientInfo del catálogo (lo que el closer consume de `candidates`)."""
    def __init__(self, name, protein, carbs, fats, kcal):
        self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, carbs, fats, kcal


def _candidates():
    """Pool ordenado por magrez desc (como `_safe_high_density_proteins`).
    Camarón es MÁS magro → el legacy lo prefiere para una merienda sin palabra-pista."""
    camaron = _Info("camarones", protein=24.0, carbs=0.0, fats=1.0, kcal=99.0)   # 0.24 prot/kcal
    yogur = _Info("yogur griego", protein=10.0, carbs=4.0, fats=0.4, kcal=59.0)  # 0.17 prot/kcal
    return [(0.24, "camarones", camaron), (0.17, "yogur griego", yogur)]


def _merienda():
    return {
        "name": "Puñado de Almendras y Nueces con Ralladura de Limón",
        "meal": "Merienda",
        "protein": 6, "carbs": 20, "fats": 30, "cals": 374,
        "ingredients": ["25g de almendras", "15g de nueces", "Ralladura de 1/2 limón"],
        "ingredients_raw": ["25g de almendras", "15g de nueces", "Ralladura de 1/2 limón"],
        "recipe": ["Mezcla y sirve."],
    }


# ---------------------------------------------------------------- helpers

def test_meal_slot_is_light_detecta_merienda_y_desayuno():
    assert _meal_slot_is_light({"meal": "Merienda"}, _sa) is True
    assert _meal_slot_is_light({"slot": "Desayuno"}, _sa) is True
    assert _meal_slot_is_light({"meal": "Snack PM"}, _sa) is True
    assert _meal_slot_is_light({"meal": "Almuerzo"}, _sa) is False
    assert _meal_slot_is_light({"meal": "Cena"}, _sa) is False


def test_reflect_name_appends_cuando_proteina_ausente():
    m = {"name": "Puñado de Almendras y Nueces con Ralladura de Limón"}
    changed = _reflect_added_protein_in_name(m, "yogur griego", _sa)
    assert changed is True
    assert "Yogur Griego" in m["name"]
    # El nombre ya tenía " con " → el conector es " y " (no doble "con").
    assert " y Yogur Griego" in m["name"]


def test_reflect_name_usa_con_cuando_no_hay_con_previo():
    m = {"name": "Huevos Duros"}
    changed = _reflect_added_protein_in_name(m, "yogur griego", _sa)
    assert changed is True
    assert m["name"] == "Huevos Duros con Yogur Griego"


def test_reflect_name_idempotente_si_ya_esta_la_proteina():
    m = {"name": "Ensalada Verde con Pollo a la Plancha"}
    changed = _reflect_added_protein_in_name(m, "pollo", _sa)
    assert changed is False
    assert m["name"] == "Ensalada Verde con Pollo a la Plancha"


# ---------------------------------------------------------------- closer integrado

def test_closer_merienda_recibe_lacteo_no_carne_y_nombre_lo_refleja():
    """Con el knob ON: una merienda recibe lácteo/huevo (yogur), NUNCA camarón, y el nombre
    refleja la proteína añadida (cierra el 'camarón escondido en almendras')."""
    assert go.CLOSER_DISH_COHERENCE_ENABLED is True  # default
    m = _merienda()
    added = _close_protein_gap_for_meal(m, slot_protein_target=25.0, db=None,
                                        candidates=_candidates(), fill_pct=0.92)
    assert added > 0
    ings = " ".join(m["ingredients"]).lower()
    assert "yogur" in ings, f"esperaba yogur en ingredientes, no camarón: {m['ingredients']}"
    assert "camaron" not in _sa(ings), f"NO debe meter camarón en una merienda: {m['ingredients']}"
    assert "Yogur" in m["name"], f"el nombre debe reflejar la proteína añadida: {m['name']!r}"


def test_closer_legacy_flag_off_vuelve_a_camaron_y_no_renombra(monkeypatch):
    """Con el knob OFF: comportamiento legacy (decide por nombre → carne/pescado, nombre intacto).
    Prueba que el FIX es exactamente lo que cambia el comportamiento."""
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", False)
    m = _merienda()
    name_before = m["name"]
    added = _close_protein_gap_for_meal(m, slot_protein_target=25.0, db=None,
                                        candidates=_candidates(), fill_pct=0.92)
    assert added > 0
    ings = _sa(" ".join(m["ingredients"]).lower())
    assert "camaron" in ings, f"legacy debía elegir camarón (más magro): {m['ingredients']}"
    assert m["name"] == name_before, "legacy NO refleja la proteína en el nombre"


# ---------------------------------------------------------------- variety report

def _meal(name, ings=None):
    return {"name": name, "ingredients": ings or []}


def test_variety_report_flaggea_fruta_repetida_mismo_dia():
    days = [{"day": 1, "meals": [
        _meal("Revoltillo de Huevos con Coliflor y Mango Fresco", ["huevos", "mango"]),
        _meal("Pollo a la Plancha con Arroz", ["pollo", "arroz"]),
        _meal("Mango Fresco con Queso Mozzarella", ["mango", "queso"]),
        _meal("Crema de Auyama", ["auyama"]),
    ]}]
    rep = build_variety_report({"days": days})
    assert rep["fruit_repeats"] >= 1
    assert any("mango" in i.lower() for i in rep["issues"])
    assert rep["ok"] is False


def test_variety_report_no_falso_positivo_frutas_distintas():
    days = [{"day": 1, "meals": [
        _meal("Avena con Mango", ["avena", "mango"]),
        _meal("Pollo con Arroz", ["pollo", "arroz"]),
        _meal("Yogur con Lechosa", ["yogur", "lechosa"]),
        _meal("Pescado al Horno", ["pescado"]),
    ]}]
    rep = build_variety_report({"days": days})
    assert rep["fruit_repeats"] == 0


# ---------------------------------------------------------------- prompt + knob anchors

def test_prompt_day_generator_tiene_regla_de_coherencia_de_frutas():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "prompts", "day_generator.py"), encoding="utf-8").read()
    assert "P2-DISH-COHERENCE" in src
    assert "NO repitas la MISMA fruta" in src
    assert "fruta dulce" in src.lower()


def test_knob_registrado():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_CLOSER_DISH_COHERENCE" in snap
