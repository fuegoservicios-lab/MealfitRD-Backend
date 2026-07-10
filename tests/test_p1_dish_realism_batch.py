"""[P1-DISH-REALISM-BATCH · 2026-07-01] Review de recetas en vivo (test Gemini, plan 2be7a10f) —
6 defectos de coherencia de PLATO con macros perfectas ("cuadra en números, nadie se lo come"):

  P1-CLOSER-COOKABLE-MIN     "10g de pechuga de pollo cocido" + paso dedicado (bypass FASE A floor 10g).
  P1-PORTION-REALISM-CAP     "505g de calamar" (P71), "1.75 taza de ají" para 1 huevo, "1.5 taza de
                             perejil" — el solver escala al clamp sin noción de plato servible.
  P2-STEP-CARB-GHOST         "Revoltillo con AVENA": avena en nombre+pasos, ausente de ingredients[].
  P2-INGREDIENT-LINE-CONSOLIDATE  "15 g de queso" + "40 g de queso" como líneas separadas.
  P2-QTYSYNC-COUNT-NOUNS     ingredientes "1 huevo entero" pero paso "batir los 3 huevos".
  P2-DISPLAY-FRACTIONS       "1 cdas", "0.5 papa mediana", "1.75 cdta" (display no-cocinable).
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import graph_orchestrator as go
from humanize_ingredients import _prettify_quantity_display

_GRAPH = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


class _StubDB:
    """macros por token — suficiente para _ingredient_macro_group y truth-up no-op."""

    def macros_from_ingredient_string(self, s):
        s = str(s).lower()
        if "calamar" in s:
            return {"protein": 80.0, "carbs": 5.0, "fats": 5.0, "kcal": 400.0}
        return None


# ── P1-CLOSER-COOKABLE-MIN ──────────────────────────────────────────────────
def _mk_candidates():
    info = SimpleNamespace(name="pechuga de pollo", protein=22.0, carbs=0.0, fats=3.0, kcal=120.0)
    return [(3.0, "pechuga de pollo", info)]


def test_cookable_min_knob_default():
    assert go.CLOSER_COOKABLE_MIN_G == 40


def test_closer_never_adds_sub_cookable_portion(monkeypatch):
    """El bypass de FASE A (enforce_min_threshold=False) ya no produce adds de 10g: todo add que
    procede se SUBE al piso cocinable (40g servibles) y sin headroom calórico se salta por completo."""
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    # gap trivial CON headroom → se sube a porción cocinable (jamás 10g)
    meal = {"name": "Bowl de guisantes", "ingredients": ["80g de guisantes"], "protein": 20,
            "carbs": 40, "fats": 8, "cals": 350, "recipe": ["Mise en place: x.", "Montaje: y."]}
    g = go._close_protein_gap_for_meal(meal, 24.0, None, _mk_candidates(),
                                       enforce_min_threshold=False)
    assert g == 0 or g >= go.CLOSER_COOKABLE_MIN_G, f"add de {g}g = porción no-cocinable (el bug del 10g)"
    if g:
        _added = str(meal["ingredients"][-1])
        m = re.match(r"(\d+)g", _added)
        assert m and int(m.group(1)) >= go.CLOSER_COOKABLE_MIN_G, f"línea no-cocinable: {_added}"
    # gap trivial SIN headroom calórico → skip total (no infla el slot)
    meal2 = {"name": "Bowl de guisantes", "ingredients": ["80g de guisantes"], "protein": 20,
             "carbs": 40, "fats": 8, "cals": 350, "recipe": ["Mise en place: x.", "Montaje: y."]}
    g2 = go._close_protein_gap_for_meal(meal2, 24.0, None, _mk_candidates(),
                                        enforce_min_threshold=False, slot_cal_target=360.0)
    assert g2 == 0, f"sin headroom (10 kcal) debió saltarse, añadió {g2}g"
    assert len(meal2["ingredients"]) == 1


def test_closer_still_closes_real_deficits(monkeypatch):
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    meal = {"name": "Bowl de guisantes", "ingredients": ["80g de guisantes"], "protein": 10,
            "carbs": 40, "fats": 8, "cals": 350, "recipe": ["Mise en place: x.", "Montaje: y."]}
    g = go._close_protein_gap_for_meal(meal, 40.0, None, _mk_candidates(),
                                       enforce_min_threshold=False)
    assert g >= go.CLOSER_COOKABLE_MIN_G, f"déficit real debe cerrarse con porción cocinable (got {g}g)"
    assert any("pechuga" in str(i) for i in meal["ingredients"])


# ── P1-PORTION-REALISM-CAP ──────────────────────────────────────────────────
def test_realism_cap_protein_grams():
    days = [{"meals": [{"name": "Calamar al vapor", "meal": "Almuerzo",
                        "ingredients": ["505g de calamar limpio en aros", "45g de arroz blanco"],
                        "ingredients_raw": ["505g de calamar limpio en aros", "45g de arroz blanco"],
                        "protein": 71, "carbs": 40, "fats": 5, "cals": 500}]}]
    n = go._cap_unrealistic_portions(days, db=_StubDB())
    assert n == 1
    lead = float(re.match(r"(\d+(?:\.\d+)?)", days[0]["meals"][0]["ingredients"][0]).group(1))
    assert lead <= go.PORTION_CAP_PROTEIN_G, f"calamar no fue recortado: {days[0]['meals'][0]['ingredients'][0]}"
    assert days[0]["meals"][0].get("_portion_realism_capped") is True
    # lockstep con raw
    lead_raw = float(re.match(r"(\d+(?:\.\d+)?)", days[0]["meals"][0]["ingredients_raw"][0]).group(1))
    assert lead_raw <= go.PORTION_CAP_PROTEIN_G


def test_realism_cap_aromatics_herbs_and_counts():
    days = [{"meals": [{"name": "Revoltillo", "meal": "Desayuno",
                        "ingredients": ["1.75 taza de ají morrón rojo picado",
                                        "1.5 taza de perejil picado",
                                        "3.5 papas medianas",
                                        "1 taza de repollo picado fino",
                                        "8 claras de huevo"],
                        "protein": 20, "carbs": 40, "fats": 10, "cals": 350}]}]
    n = go._cap_unrealistic_portions(days, db=_StubDB())
    ings = days[0]["meals"][0]["ingredients"]
    # [P2-CLARAS-CAP-KNOB-PARITY · 2026-07-05] el techo de claras dejó de ser 8.0 hardcoded y
    # sigue a MEALFIT_MAX_EGG_WHITES_PER_MEAL (6) — "8 claras" ahora SÍ se recorta (4º recorte;
    # el plato vivo "8 claras (267g)" pasaba al ras mientras el motor decía 6).
    assert n == 4, f"esperados 4 recortes (ají/perejil/papas/claras), got {n}: {ings}"
    assert float(re.match(r"([\d.]+)", ings[0]).group(1)) <= 1.0, f"aromático sin cap: {ings[0]}"
    assert float(re.match(r"([\d.]+)", ings[1]).group(1)) <= 0.25, f"hierba sin cap: {ings[1]}"
    assert float(re.match(r"([\d.]+)", ings[2]).group(1)) <= 3.0, f"papas sin cap: {ings[2]}"
    assert ings[3].startswith("1 taza de repollo"), "repollo (veg principal) NO debe caparse a 1 taza"
    assert float(re.match(r"([\d.]+)", ings[4]).group(1)) <= float(go.MAX_EGG_WHITES_PER_MEAL), \
        f"claras sobre el knob ({go.MAX_EGG_WHITES_PER_MEAL}) deben recortarse: {ings[4]}"


def test_realism_cap_idempotent():
    days = [{"meals": [{"name": "X", "meal": "Cena",
                        "ingredients": ["1 taza de cebolla picada"], "protein": 5,
                        "carbs": 10, "fats": 2, "cals": 80}]}]
    assert go._cap_unrealistic_portions(days, db=_StubDB()) == 0


# ── P2-STEP-CARB-GHOST ──────────────────────────────────────────────────────
def test_carb_ghost_materialized():
    days = [{"meals": [{"name": "Revoltillo Criollo con Queso y Avena", "meal": "Desayuno",
                        "ingredients": ["1 huevo entero", "15 g de queso"],
                        "ingredients_raw": ["1 huevo entero", "15 g de queso"],
                        "recipe": ["Mise en place: batir el huevo.",
                                   "El Toque de Fuego: cuajar 3-4 min. Cocinar la avena con agua según empaque.",
                                   "Montaje: servir junto a la avena cremosa."]}]}]
    n = go._add_missing_recipe_step_carbs(days)
    assert n == 1
    ings = days[0]["meals"][0]["ingredients"]
    assert any("avena" in str(i) for i in ings), f"avena fantasma no materializada: {ings}"
    assert len(days[0]["meals"][0]["ingredients_raw"]) == len(ings)
    # idempotente
    assert go._add_missing_recipe_step_carbs(days) == 0


def test_carb_ghost_excludes_and_rice():
    days = [{"meals": [{"name": "Panqueques de harina de avena", "meal": "Desayuno",
                        "ingredients": ["100g de harina de avena"],
                        "recipe": ["El Toque de Fuego: dora los panqueques 3 min por lado."]},
                       {"name": "Pollo guisado", "meal": "Cena",
                        "ingredients": ["150g de pollo"],
                        "recipe": ["El Toque de Fuego: guisa el pollo 25 min y cocina el arroz."]}]}]
    assert go._add_missing_recipe_step_carbs(days) == 0, \
        "'harina de avena' (exclude) y 'arroz' (slot-sensitivo, fuera de la lista) no deben materializarse"


# ── P2-INGREDIENT-LINE-CONSOLIDATE ──────────────────────────────────────────
def test_consolidate_duplicate_gram_lines():
    days = [{"meals": [{"name": "Revoltillo",
                        "ingredients": ["15 g de queso", "1 huevo entero", "40 g de queso"],
                        "ingredients_raw": ["15 g de queso", "1 huevo entero", "40 g de queso"]}]}]
    n = go._consolidate_duplicate_gram_lines(days)
    assert n == 1
    ings = days[0]["meals"][0]["ingredients"]
    assert ings == ["55g de queso", "1 huevo entero"], f"consolidación incorrecta: {ings}"
    assert days[0]["meals"][0]["ingredients_raw"] == ings


def test_consolidate_conservative():
    days = [{"meals": [{"name": "X",
                        "ingredients": ["15 g de queso", "40 g de queso fresco"],
                        "ingredients_raw": ["15 g de queso", "40 g de queso fresco"]}]}]
    assert go._consolidate_duplicate_gram_lines(days) == 0, "'queso' ≠ 'queso fresco' — no fusionar"
    # [P1-RECIPE-QUALITY-100 · 2026-07-10] SEMÁNTICA ACTUALIZADA: raw desalineado ya NO bloquea el
    # meal entero (dejaba "15 g de queso" + "40 g de queso" visibles al usuario — bollitos, plan
    # vivo 6d742f23). Ahora el DISPLAY se consolida y raw queda intacto (ya venía desincronizado).
    days2 = [{"meals": [{"name": "X",
                         "ingredients": ["15 g de queso", "40 g de queso"],
                         "ingredients_raw": ["solo una línea"]}]}]
    assert go._consolidate_duplicate_gram_lines(days2) == 1, "display SÍ consolida con raw desalineado"
    assert days2[0]["meals"][0]["ingredients"] == ["55g de queso"]
    assert days2[0]["meals"][0]["ingredients_raw"] == ["solo una línea"], "raw intacto (no lockstep)"


# ── P2-QTYSYNC-COUNT-NOUNS ──────────────────────────────────────────────────
def test_qtysync_count_nouns_rewrites_egg_count():
    meal = {"ingredients": ["1 huevo entero", "15 g de queso"],
            "recipe": ["Mise en place: Batir los 3 huevos en un recipiente. Picar la cebolla."]}
    n = go._sync_recipe_step_quantities(meal)
    assert n >= 1
    assert "1 huevo" in meal["recipe"][0], f"paso sin sincronizar: {meal['recipe'][0]}"
    assert "3 huevos" not in meal["recipe"][0]
    assert "Picar la cebolla" in meal["recipe"][0], "el resto del paso debe preservarse"


def test_qtysync_count_nouns_skips_fractions_and_matches():
    meal = {"ingredients": ["0.5 papa mediana"],
            "recipe": ["Hervir las 2 papas hasta que estén suaves."]}
    go._sync_recipe_step_quantities(meal)
    assert "2 papas" in meal["recipe"][0], "conteo fraccional (0.5) no debe reescribir el texto"
    meal2 = {"ingredients": ["2 huevos"], "recipe": ["Batir los 2 huevos."]}
    assert go._sync_recipe_step_quantities(meal2) == 0, "conteo ya correcto → no tocar"


# ── P2-DISPLAY-FRACTIONS ────────────────────────────────────────────────────
def test_display_fractions_and_plural_agreement():
    assert _prettify_quantity_display("0.5 papa mediana").startswith("½ papa")
    assert _prettify_quantity_display("1.75 cdta de orégano").startswith("1¾ cdta")
    assert _prettify_quantity_display("1 cdas de aceite de oliva") == "1 cda de aceite de oliva"
    assert _prettify_quantity_display("1 tallos de puerro picado") == "1 tallo de puerro picado"
    assert _prettify_quantity_display("3 papas medianas") == "3 papas medianas"
    # [P3-DISPLAY-GRAMMAR · 2026-07-05] retune: cantidades >1 (incluidas fraccionarias) llevan
    # plural en español ("1,5 tazas", RAE) — el "no tocar" conservador previo queda superseded.
    assert _prettify_quantity_display("1½ taza de yogurt griego") == "1½ tazas de yogurt griego"


# ── wiring ──────────────────────────────────────────────────────────────────
def test_wired_in_assemble_and_finalizers():
    assert _GRAPH.count("_cap_unrealistic_portions(") >= 4, "cap: def + assemble + finalizer single + persist"
    assert _GRAPH.count("_add_missing_recipe_step_carbs(") >= 4
    assert _GRAPH.count("_consolidate_duplicate_gram_lines(") >= 4
    i_cap = _GRAPH.find("[P1-PORTION-REALISM-CAP · 2026-07-01] (batch P1-DISH-REALISM-BATCH) Techo de porción REALISTA\n    # post-sizing y ANTES de FASE A")
    i_rep = _GRAPH.find("if REPAIR_PROTEIN_POST_CAPS:")
    assert -1 < i_cap < i_rep, "el cap debe correr ANTES del reparador de proteína (para que redistribuya)"


def test_marker_anchor():
    assert "P1-DISH-REALISM-BATCH" in _GRAPH
