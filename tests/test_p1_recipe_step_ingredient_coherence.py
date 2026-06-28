"""[P1-RECIPE-STEP-INGREDIENT-COHERENCE · 2026-06-28] Bug en vivo (plan NORMAL, 2 de 4 recetas): los PASOS de la receta
referencian un vegetal del catálogo ("calabacín") ausente de ingredients[] → no entra a la lista de compras + macros
sub-contados. Guard UNIVERSAL determinista que detecta vegetales del catálogo en recipe[] ausentes de ingredients[] y los
AGREGA (100g) ANTES del macro engine (→ allergen scan + reconcile + shopping + coherence). Macro-safe por construcción:
solo category=='Vegetales' AND kcal/100g ≤ 60 AND fuera de stop-list de hierbas/aromáticos.

+ Pulido cosmético display-only (humanize): "0.25 cda"→"1 cdta", "1 huevos"→"1 huevo" (el quantize no los toca: "cda"/
count no son unidades métricas → early-return).

Tests PUROS: detector con catálogo mockeado (sin Neon); polish sin deps.
"""
from __future__ import annotations

import re
import unicodedata

import graph_orchestrator as g
import shopping_calculator
from humanize_ingredients import _polish_countunit_display, humanize_ingredient


def _norm(s):
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"^[\d\s/.,]+\s*(g|gr|kg|mg|ml|l|lb|oz)?\b\s*(de\s+)?", "", s).strip()
    return s


_MOCK_CATALOG = [
    {"name": "Calabacín", "category": "Vegetales", "kcal_per_100g": 20.2},
    {"name": "Cebolla", "category": "Vegetales", "kcal_per_100g": 42.7},
    {"name": "Ajo", "category": "Vegetales", "kcal_per_100g": 142.7},       # kcal > 60 → excluido
    {"name": "Cilantro", "category": "Vegetales", "kcal_per_100g": 27.9},   # stop-list hierba → excluido
    {"name": "Maní", "category": "Despensa", "kcal_per_100g": 630.1},       # no Vegetales → excluido
    {"name": "Limón", "category": "Frutas", "kcal_per_100g": 46.6},         # no Vegetales → excluido
]


def _patch(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: _MOCK_CATALOG)
    monkeypatch.setattr(shopping_calculator, "normalize_name", _norm)


def test_phantom_vegetable_added(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "Revoltillo", "ingredients": ["2 huevos"],
                        "recipe": ["Lava el calabacín y rállalo.", "Agrega el calabacín y saltéalo."]}]}]
    n = g._add_missing_recipe_step_vegetables(days)
    assert n == 1
    assert any("calabac" in i.lower() for i in days[0]["meals"][0]["ingredients"])


def test_idempotent(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["2 huevos"], "recipe": ["Agrega el calabacín."]}]}]
    g._add_missing_recipe_step_vegetables(days)
    assert g._add_missing_recipe_step_vegetables(days) == 0  # 2da pasada no re-agrega


def test_already_present_not_readded(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["100g calabacín"], "recipe": ["Agrega el calabacín."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 0


def test_negation_skipped(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["2 huevos"], "recipe": ["Cocina sin calabacín."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 0


def test_dense_and_nonveg_excluded(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["1 huevo"],
                        "recipe": ["Agrega maní, limón y ajo al sofrito."]}]}]
    # maní (Despensa), limón (Frutas), ajo (kcal>60) → ninguno se agrega
    assert g._add_missing_recipe_step_vegetables(days) == 0


def test_herb_stoplist_excluded(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["1 huevo"], "recipe": ["Espolvorea cilantro picado."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 0  # cilantro en stop-list


def test_max_per_meal_cap(monkeypatch):
    _patch(monkeypatch)
    days = [{"meals": [{"name": "x", "ingredients": ["1 huevo"],
                        "recipe": ["Agrega calabacín y cebolla."]}]}]
    n = g._add_missing_recipe_step_vegetables(days, max_per_meal=1)
    assert n == 1  # cap respetado


def test_failsafe_catalog_error(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients",
                        lambda: (_ for _ in ()).throw(RuntimeError("neon down")))
    days = [{"meals": [{"name": "x", "ingredients": ["1 huevo"], "recipe": ["Agrega el calabacín."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 0  # degrada, no crashea


# ---- POLISH cosmético (puro) ----
def test_polish_quarter_tbsp():
    assert _polish_countunit_display("0.25 cda de aceite de oliva", "0.25", "cda de aceite de oliva") == "1 cdta de aceite de oliva"
    assert _polish_countunit_display("1/4 cda de aceite", "1/4", "cda de aceite") == "1 cdta de aceite"


def test_polish_singular():
    assert _polish_countunit_display("1 huevos enteros", "1", "huevos enteros") == "1 huevo entero"
    assert _polish_countunit_display("1 huevos", "1", "huevos") == "1 huevo"


def test_polish_leaves_others_intact():
    assert _polish_countunit_display("2 huevos", "2", "huevos") == "2 huevos"          # qty != 1
    assert _polish_countunit_display("0.5 cda de aceite", "0.5", "cda de aceite") == "0.5 cda de aceite"  # ½ cda OK
    assert _polish_countunit_display("1 manzana", "1", "manzana") == "1 manzana"        # ya singular


def test_humanize_e2e():
    assert humanize_ingredient("0.25 cda de aceite de oliva") == "1 cdta de aceite de oliva"
    assert humanize_ingredient("1 huevos enteros") == "1 huevo entero"
    assert humanize_ingredient("2 huevos") == "2 huevos"  # intacto


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-RECIPE-STEP-INGREDIENT-COHERENCE" in src
    assert "RECIPE_STEP_VEG_GUARD_ENABLED" in src
    assert "_add_missing_recipe_step_vegetables" in src
    assert g.RECIPE_STEP_VEG_GUARD_ENABLED is True
    # el guard corre ANTES del macro engine (el ingrediente agregado debe pasar por reconcile/allergen/shopping):
    # la LLAMADA al engine debe aparecer DESPUÉS del hook del guard.
    i_guard = src.index("_veg_added = _add_missing_recipe_step_vegetables")
    assert "_apply_macro_engine(result, days" in src[i_guard:]
