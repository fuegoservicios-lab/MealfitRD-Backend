"""[P1-RECIPE-VISIBLE-DEFECTS · 2026-07-11] Batch de los 4 defectos visibles en las capturas
del owner (plan 5424440f):

1. "mezcla la ingrediente alternativo con..." — placeholder del sanitizador
   P2-PROTEIN-VIOLATION-SANITIZE llegando al usuario → resolver final lo reemplaza con la
   proteína REAL de ingredients (artículo por género).
2. La línea mecánica "Incorpora también carne de res molida..." nacía en cascada (steps
   sanitizados sin proteína → sweep de paridad la re-pegaba) → el resolver corre ANTES del
   sweep en la cadena de finalize.
3. "55 g de carne de res molida" para "dos hamburguesas GRUESAS" / "15 g de cerdo molida"
   protagonista → piso PROTAGONISTA (label del nombre del plato) en el shrink-floor.
4. "75g de camarones cocido" en el snack dulce yogurt+lechosa+maní+miel → el pase
   carbos→proteína elegía destino por MÁS carbos sin sweet/light-guard, y sin day-aware.

tooltip-anchor: P1-RECIPE-VISIBLE-DEFECTS
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_DBP = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")


class _FakeDB:
    def macros_from_ingredient_string(self, s):
        try:
            parts = str(s).replace("g de ", "|", 1).split("|")
            if len(parts) != 2:
                return None
            g = float(parts[0].strip())
            return {"name": parts[1].strip(), "grams": g, "kcal": g * 1.5}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 1+2. Resolver del placeholder
# ---------------------------------------------------------------------------

def test_placeholder_resolved_with_real_protein_and_article():
    from graph_orchestrator import resolve_alt_ingredient_placeholders
    meal = {
        "name": "Hamburguesas de Res con Guineítos",
        "ingredients": ["55 g de carne de res molida", "100 g de berenjena"],
        "recipe": [
            "En un tazón, mezcla la ingrediente alternativo con la mitad del ajo.",
            "Cubre con la ingrediente alternativo salteada.",
        ],
    }
    n = resolve_alt_ingredient_placeholders([{"meals": [meal]}], _FakeDB())
    assert n == 2
    assert "ingrediente alternativo" not in " ".join(meal["recipe"])
    assert "mezcla la carne de res molida con" in meal["recipe"][0]
    assert "la carne de res molida salteada" in meal["recipe"][1]


def test_placeholder_left_alone_without_resolvable_protein():
    from graph_orchestrator import resolve_alt_ingredient_placeholders
    meal = {
        "name": "Ensalada Verde",
        "ingredients": ["100 g de lechuga"],
        "recipe": ["Mezcla la ingrediente alternativo con la lechuga."],
    }
    n = resolve_alt_ingredient_placeholders([{"meals": [meal]}], _FakeDB())
    assert n == 0 and "ingrediente alternativo" in meal["recipe"][0], (
        "sin proteína resoluble el placeholder se queda (mejor genérico que inventar)"
    )


def test_resolver_runs_before_step_parity_in_finalize_chain():
    i_res = _DBP.find("resolve_alt_ingredient_placeholders")
    i_par = _DBP.find("ensure_protein_step_parity")
    assert 0 < i_res < i_par, (
        "el resolver debe correr ANTES del sweep de paridad — con la mención restaurada "
        "el sweep no pega su línea mecánica encima"
    )


# ---------------------------------------------------------------------------
# 3. Piso protagonista en el shrink-floor
# ---------------------------------------------------------------------------

def test_protagonist_floor_bumps_named_protein():
    import graph_orchestrator as go
    from graph_orchestrator import _floor_subservible_portions
    meal = {
        "name": "Hamburguesas de Res con Guineítos Verdes",
        "ingredients": ["55 g de carne de res molida", "100 g de berenjena", "80 g de guineítos"],
        "ingredients_raw": ["55 g de carne de res molida", "100 g de berenjena", "80 g de guineítos"],
        "cals": 400, "protein": 15, "carbs": 40, "fats": 10,
    }
    days = [{"day": 1, "meals": [meal]}]
    touched = _floor_subservible_portions(days, day_kcal_target=2500, db=_FakeDB())
    _line = meal["ingredients"][0]
    assert "75" in _line or "75" in str(meal["ingredients"]), (
        f"la proteína PROTAGONISTA (res en el NOMBRE) debe subir al piso 75g; quedó: {_line!r}"
    )
    assert touched >= 1


def test_protagonist_floor_skips_non_protagonist_lines():
    from graph_orchestrator import _floor_subservible_portions
    meal = {
        "name": "Ensalada de Vegetales",  # sin label de proteína en el nombre
        "ingredients": ["55 g de carne de res molida", "100 g de lechuga"],
        "ingredients_raw": ["55 g de carne de res molida", "100 g de lechuga"],
        "cals": 300, "protein": 12, "carbs": 10, "fats": 8,
    }
    days = [{"day": 1, "meals": [meal]}]
    _floor_subservible_portions(days, day_kcal_target=2500, db=_FakeDB())
    assert meal["ingredients"][0].startswith("55"), (
        "res NO está en el nombre → no es protagonista → el piso de 75g no aplica (55g ≥ piso genérico)"
    )


# ---------------------------------------------------------------------------
# 4. Carb→protein swap: sweet/light/day-aware
# ---------------------------------------------------------------------------

def test_carb_swap_never_targets_sweet_or_light_meal():
    i = _GO.find("comida destino: la de MÁS carbos pero JAMÁS")
    assert i > 0, "el guard de destino del carb-swap desapareció"
    blk = _GO[i: i + 2600]
    assert "_is_sweet_meal(m, _sa)" in blk and "_meal_slot_is_light(m, _sa)" in blk
    assert "sin comida destino salada" in blk, "sin destino válido → skip honesto (no forzar combo aberrante)"
    assert "_protein_gate_labels_in_meal(_om)" in blk, (
        "day-aware: el candidato no debe repetir proteína de otra comida del día"
    )


def test_carb_swap_functional_skips_sweet_snack():
    from graph_orchestrator import _swap_excess_carbs_to_protein_for_day

    class _Info:
        def __init__(self, name, protein, carbs=0.0, fats=1.0):
            self.name, self.protein, self.carbs, self.fats = name, protein, carbs, fats

    sweet = {"name": "Yogurt Griego con Lechosa y Mantequilla de Maní", "meal": "Merienda",
             "ingredients": ["200 g de yogurt griego", "150 g de lechosa"],
             "protein": 10, "carbs": 80, "fats": 8, "cals": 450}
    savory = {"name": "Pollo Guisado con Arroz", "meal": "Almuerzo",
              "ingredients": ["120 g de pechuga de pollo", "150 g de arroz blanco"],
              "protein": 30, "carbs": 60, "fats": 10, "cals": 550}
    meals = [sweet, savory]
    _swap_excess_carbs_to_protein_for_day(
        meals, p_target_day=120, c_target_day=100, db=_FakeDB(),
        candidates=[(1.0, "camarones", _Info("camarones", 20.0))],
    )
    _sweet_blob = " ".join(str(x) for x in sweet["ingredients"]).lower()
    assert "camarones" not in _sweet_blob, (
        "P1-RECIPE-VISIBLE-DEFECTS: camarones volvió al snack dulce (caso vivo 5424440f)"
    )


def test_marker_anchored_in_source():
    assert _GO.count("P1-RECIPE-VISIBLE-DEFECTS") >= 4
