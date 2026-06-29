"""[audit objetivo · P2-8/P2-9/P2-10] Slot-appropriateness: merienda enforced, exclusiones tangenciales de
arroz, y corrector determinista de slot (night-rice) recipe-coherente + reusable en updates.

- P2-8 (P2-SLOT-MERIENDA): un plato fuerte disfrazado de merienda se flagea en TODAS las superficies (SSOT),
  sin romper "Arroz con leche" (postre legítimo).
- P2-9 (P2-SLOT-RICE-TANGENTIAL): "un toque de arroz" en cena no dispara el falso positivo de "arroz de noche".
- P2-10 (P2-SLOT-CORRECTOR): _night_rice_autofix reescribe ingrediente + nombre + PASOS de receta (cierra la
  incoherencia latente) y se cablea en el finalizador de updates.
"""
from __future__ import annotations

from pathlib import Path

import constants as c
import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


# ───────────────────────── P2-8: merienda ─────────────────────────
def test_merienda_flags_heavy_dish():
    assert c.slot_violations_for_meal_name("Locrio de pollo", "merienda")
    assert c.slot_violations_for_meal_name("Moro de habichuela", "merienda")
    assert c.slot_violations_for_meal_name("Sancocho", "merienda")


def test_merienda_respects_legit_snacks_and_postre():
    # "Arroz con leche" (postre) y snacks ligeros NO se flagean.
    assert c.slot_violations_for_meal_name("Arroz con leche", "merienda") == []
    assert c.slot_violations_for_meal_name("Yogurt con fruta y maní", "merienda") == []
    assert c.slot_violations_for_meal_name("Casabe con queso", "merienda") == []


def test_almuerzo_legit_dishes_clean():
    # [P2-SLOT-ALMUERZO · 2026-06-29] locrio/arroz/sopa/ensalada SIGUEN limpios en almuerzo (plato fuerte legítimo).
    assert c.slot_violations_for_meal_name("Locrio de cerdo", "almuerzo") == []
    assert c.slot_violations_for_meal_name("Arroz con habichuela y pollo", "almuerzo") == []
    assert c.slot_violations_for_meal_name("Sancocho de res", "almuerzo") == []
    assert c.slot_violations_for_meal_name("Ensalada César con pollo", "almuerzo") == []


def test_almuerzo_flags_breakfast_food_and_dessert_main():
    # [P2-SLOT-ALMUERZO · 2026-06-29] un desayuno/postre como PLATO PRINCIPAL del almuerzo = soft.
    assert c.slot_violations_for_meal_name("Panqueques de avena con miel", "almuerzo")
    assert c.slot_violations_for_meal_name("Cereal con leche", "almuerzo")
    assert c.slot_violations_for_meal_name("Helado de vainilla", "almuerzo")
    # soft (degrada en intento final, nunca cero-plan)
    v = c.slot_violations_for_meal_name("Waffles con fruta", "almuerzo")
    assert v and all(not x["hard"] for x in v)


# ───────────────────────── P2-SLOT-CENA-HEAVY-SOUP / P2-SLOT-MERIENDA-JUNK ─────────────────────────
def test_cena_flags_heavy_soup_soft():
    v = c.slot_violations_for_meal_name("Sancocho de siete carnes", "cena")
    assert v and all(not x["hard"] for x in v), "sopón pesado de noche = soft (no zero-plan)"
    assert c.slot_violations_for_meal_name("Mondongo", "cena")
    # cena ligera legítima NO se flagea
    assert c.slot_violations_for_meal_name("Pescado al horno con ensalada", "cena") == []


def test_merienda_flags_junk_but_keeps_legit_dr_snacks():
    assert c.slot_violations_for_meal_name("Pizza personal", "merienda")
    assert c.slot_violations_for_meal_name("Hamburguesa con queso", "merienda")
    # snacks dominicanos legítimos NO se flagean (empanada/pastelito/chicharrón/frituras)
    assert c.slot_violations_for_meal_name("Empanada de pollo", "merienda") == []
    assert c.slot_violations_for_meal_name("Chicharrón de cerdo", "merienda") == []
    assert c.slot_violations_for_meal_name("Tostones con queso frito", "merienda") == []


# ───────────────────────── P2-9: contexto/cantidad ─────────────────────────
def test_tangential_rice_not_flagged_in_dinner():
    assert c.slot_violations_for_meal_name("Sopa ligera de pollo con un toque de arroz", "cena") == []
    assert c.slot_violations_for_meal_name("Ensalada con crocante de arroz", "cena") == []


def test_real_rice_dish_still_flagged_in_dinner():
    assert c.slot_violations_for_meal_name("Arroz blanco con pollo", "cena")


# ───────────────────────── P2-10: corrector determinista ─────────────────────────
class _FakeDB:
    def grams_from_ingredient_string(self, s):
        return 150.0


def test_night_rice_rewrites_ingredient_name_and_recipe_steps(monkeypatch):
    """El corrector reescribe ingrediente + nombre + PASOS de receta (recipe-coherente)."""
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda m, db: True)
    days = [{"day": 1, "meals": [{
        "meal": "Cena", "name": "Arroz blanco con pollo guisado",
        "ingredients": ["1 taza de arroz", "150g Pechuga de pollo"],
        "recipe": ["Mise en place: lava el arroz.", "El Toque de Fuego: cocina el arroz con el pollo."],
    }]}]
    n = g._night_rice_autofix(days, db=_FakeDB())
    assert n == 1
    meal = days[0]["meals"][0]
    # nombre ya no dice arroz
    assert "arroz" not in meal["name"].lower()
    # ingrediente arroz reemplazado por tubérculo
    assert not any("arroz" in str(i).lower() for i in meal["ingredients"])
    # PASOS de receta ya no mencionan arroz (P2-10 recipe-coherence)
    assert not any("arroz" in str(s).lower() for s in meal["recipe"]), "los pasos deben reescribirse"


def test_night_rice_wired_in_update_finalizer():
    """El corrector se cablea en finalize_single_meal_recipe_coherence (updates)."""
    assert "_night_rice_autofix(_wrap, db)" in _GRAPH
    assert "P2-SLOT-CORRECTOR" in _GRAPH
