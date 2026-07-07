"""[P1-RECIPE-QUALITY-BATCH · 2026-07-07] Cluster de calidad de receta del review visual del plan 766893f4:

- P1-CHEESE-DUMP-FINAL: cap FINAL del queso (sweet-aware) tras todos los pases de macro → caza la
  re-inflación post-cap del closer (190g de queso en un desayuno dulce PB+mango).
- P1-STEP-SUGAR-GHOST: strip de azúcar fantasma en pasos (mencionada pero ausente de ingredients).
- P1-COMPLEMENT-WORDING: muletilla robótica "integrándolo al plato de forma coherente" eliminada.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _DB:
    import re as _re

    def macros_from_ingredient_string(self, s):
        m = self._re.match(r"(\d+(?:[.,]\d+)?)", str(s))
        gr = float(m.group(1).replace(",", ".")) if m else 0.0
        low = str(s).lower()
        if "queso" in low:  # queso graso (fat-dominante)
            return {"protein": gr * 0.11, "carbs": gr * 0.03, "fats": gr * 0.30, "kcal": gr * 3.5}
        if "yogur" in low:
            return {"protein": gr * 0.10, "carbs": gr * 0.05, "fats": gr * 0.03, "kcal": gr * 0.9}
        return None


# ─────────────────────────── Cheese-dump-final ───────────────────────────

def test_cheese_dump_sweet_desayuno_capped_to_snack():
    """Un 'Desayuno' de carácter DULCE (mantequilla de maní + mango) con 190g de queso → cap de
    MERIENDA (120g) aunque el meal_field sea Desayuno; grasa recomputada honesta."""
    import graph_orchestrator as g
    meal = {"name": "Tostadas de Mantequilla de Maní y Mango con Queso", "meal": "Desayuno",
            "ingredients": ["190 g de queso", "2 tortillas de trigo"],
            "ingredients_raw": ["190 g de queso", "2 tortillas de trigo"],
            "protein": 60, "carbs": 40, "fats": 67, "cals": 1206}
    n = g._cap_cheese_dumps_final([{"meals": [meal]}], _DB())
    assert n == 1
    assert meal["ingredients"][0].startswith("120")
    assert meal["ingredients_raw"][0].startswith("120")  # lockstep raw
    assert meal["fats"] < 67  # recompute honesto (bajó la grasa del queso)


def test_cheese_dump_savory_uses_meal_cap():
    """Comida principal salada → techo generoso de comida (180g), no el de merienda."""
    import graph_orchestrator as g
    meal = {"name": "Pollo Guisado con Arroz", "meal": "Almuerzo",
            "ingredients": ["200 g de queso"], "ingredients_raw": ["200 g de queso"],
            "protein": 30, "carbs": 40, "fats": 60, "cals": 700}
    g._cap_cheese_dumps_final([{"meals": [meal]}], _DB())
    assert meal["ingredients"][0].startswith("180")  # MEAL_CHEESE_CAP


def test_cheese_dump_yogurt_exempt_and_under_cap_untouched():
    import graph_orchestrator as g
    meal = {"name": "Merienda", "meal": "Merienda",
            "ingredients": ["200 g de yogurt griego", "80 g de queso"],
            "ingredients_raw": ["200 g de yogurt griego", "80 g de queso"],
            "protein": 30, "carbs": 20, "fats": 10, "cals": 300}
    n = g._cap_cheese_dumps_final([{"meals": [meal]}], _DB())
    assert n == 0  # yogurt exento + queso 80g < 120g cap → nada tocado
    assert meal["ingredients"] == ["200 g de yogurt griego", "80 g de queso"]


def test_cheese_dump_knob_off(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "CHEESE_DUMP_FINAL_ENABLED", False)
    meal = {"name": "Batido de Avena", "meal": "Merienda", "ingredients": ["190 g de queso"],
            "ingredients_raw": ["190 g de queso"], "protein": 40, "carbs": 20, "fats": 57, "cals": 900}
    assert g._cap_cheese_dumps_final([{"meals": [meal]}], _DB()) == 0
    assert meal["ingredients"][0].startswith("190")


# ─────────────────────────── Sugar-ghost strip ───────────────────────────

def test_sugar_stripped_when_absent_from_ingredients():
    import graph_orchestrator as g
    meal = {"ingredients": ["1 plátano verde", "1 cdta de mantequilla"],
            "recipe": ["Mise en place: pela el plátano.",
                       "Derrite la mantequilla restante con el azúcar a fuego bajo y añade la cebolla."]}
    n = g._strip_phantom_sugar_from_steps([{"meals": [meal]}])
    assert n == 1
    assert "azúcar" not in meal["recipe"][1] and "azucar" not in meal["recipe"][1].lower()
    assert "fuego bajo" in meal["recipe"][1]  # el resto del paso se preserva


def test_sugar_kept_when_in_ingredients():
    """Si el plato SÍ compra azúcar/miel, no se toca el paso (respeta postres legítimos)."""
    import graph_orchestrator as g
    meal = {"ingredients": ["1 cdta de miel", "1 taza de avena"],
            "recipe": ["Endulza la avena con la miel al servir."]}
    before = list(meal["recipe"])
    assert g._strip_phantom_sugar_from_steps([{"meals": [meal]}]) == 0
    assert meal["recipe"] == before


# ─────────────────────────── Filler wording ───────────────────────────

def test_complement_wording_robotic_muletilla_gone():
    assert "integrándolo al plato de forma coherente" not in _GO
    assert "integrándolo de forma coherente" not in _GO
    assert "P1-COMPLEMENT-WORDING" in _GO


def test_markers_anchored():
    assert "P1-CHEESE-DUMP-FINAL" in _GO
    assert "P1-STEP-SUGAR-GHOST" in _GO
    assert 'CHEESE_DUMP_FINAL_ENABLED = _env_bool("MEALFIT_CHEESE_DUMP_FINAL", True)' in _GO
    assert 'STEP_SUGAR_GHOST_STRIP_ENABLED = _env_bool("MEALFIT_STEP_SUGAR_GHOST_STRIP", True)' in _GO
