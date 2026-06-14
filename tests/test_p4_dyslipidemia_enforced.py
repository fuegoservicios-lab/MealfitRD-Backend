"""[P4-DYSLIPIDEMIA-ENFORCED · 2026-06-14] Dislipidemia pasa de prompt-only a ENFORCED añadiendo una
FILA al registro CONDITION_RULES (la demostración más limpia del motor declarativo): substitución de
fuentes de grasa saturada → versión magra de la misma categoría (mantequilla→aceite, lácteos enteros→
bajos en grasa, tocino/chicharrón→lean). Usa el engine de sustitución macro-aware existente. Tokens
estrechos (lección del bug 'soya'): NO 'manteca' desnudo / 'nata' / 'crema' que colisionarían con
mantecado / natural / crema de maní.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import condition_rules as cr
import graph_orchestrator as go

DYS = {"medicalConditions": ["Colesterol alto"]}


def _ings(plan):
    return plan["days"][0]["meals"][0]["ingredients"]


# ── El registro expone las subs de dislipidemia ──
def test_collect_substitutions_includes_dyslipidemia():
    subs = cr.collect_substitutions(DYS)
    assert subs and all(s["condition"] == "dyslipidemia" for s in subs)
    labels = {s["label"] for s in subs}
    assert "mantequilla/manteca/margarina" in labels and "queso alto en grasa" in labels


# ── El engine sustituye las fuentes de grasa saturada ──
def test_substitutes_butter_and_whole_dairy():
    plan = {"days": [{"meals": [{"ingredients": [
        "2 cda de mantequilla", "30g de queso cheddar", "1 taza de yogur entero", "100g de tocino"]}]}]}
    go._apply_condition_substitutions(plan, DYS)
    out = " | ".join(_ings(plan)).lower()
    assert "mantequilla" not in out and "aceite de oliva" in out
    assert "cheddar" not in out and "cottage" in out
    assert "yogur entero" not in out and "griego" in out
    assert "tocino" not in out and "pollo" in out


def test_cheese_swap_preserves_quantity():
    plan = {"days": [{"meals": [{"ingredients": ["60g de queso cheddar"]}]}]}
    go._apply_condition_substitutions(plan, DYS)
    repl = _ings(plan)[0]
    assert repl[0].isdigit() and "cottage" in repl.lower()   # conserva 60g


def test_whole_milk_intentionally_not_substituted():
    """[review adversaria] 'leche entera' NO se sustituye: el catálogo la conflaciona con 'descremada'
    en la misma fila → el swap sería un no-op clínico. Documentado, no un bug."""
    plan = {"days": [{"meals": [{"ingredients": ["250ml de leche entera"]}]}]}
    assert go._apply_condition_substitutions(plan, DYS) == 0


def test_respects_low_fat_negative():
    plan = {"days": [{"meals": [{"ingredients": ["250ml de leche descremada", "yogur bajo en grasa"]}]}]}
    n = go._apply_condition_substitutions(plan, DYS)
    out = " | ".join(_ings(plan)).lower()
    assert "descremada" in out and "bajo en grasa" in out   # ya magros → no se tocan


# ── Guardas de false-positive (lección 'soya') ──
def test_does_not_substitute_mantecado_dessert():
    # "manteca" desnudo matchearía "mantecado"; los tokens estrechos NO
    plan = {"days": [{"meals": [{"ingredients": ["1 bola de mantecado"]}]}]}
    assert go._apply_condition_substitutions(plan, DYS) == 0
    assert _ings(plan) == ["1 bola de mantecado"]


def test_does_not_substitute_yogur_natural_or_crema_de_mani():
    plan = {"days": [{"meals": [{"ingredients": ["yogur griego natural", "1 cda de crema de maní"]}]}]}
    n = go._apply_condition_substitutions(plan, DYS)
    assert "natural" in " ".join(_ings(plan)).lower()
    assert any("maní" in i or "mani" in i for i in _ings(plan))


def test_does_not_substitute_nut_butter():
    # 'mantequilla' matchearía 'mantequilla de maní' (grasa INSATURADA, saludable) — el negativo lo veta
    plan = {"days": [{"meals": [{"ingredients": ["2 cda de mantequilla de maní", "1 cda de mantequilla de almendra"]}]}]}
    n = go._apply_condition_substitutions(plan, DYS)
    assert n == 0
    out = " ".join(_ings(plan)).lower()
    assert "maní" in out or "mani" in out
    assert "almendra" in out


def test_does_not_substitute_plain_milk_chicken():
    # leche (sin "entera") y pollo magro no se tocan
    plan = {"days": [{"meals": [{"ingredients": ["1 taza de leche", "150g de pechuga de pollo"]}]}]}
    assert go._apply_condition_substitutions(plan, DYS) == 0


def test_no_condition_no_substitution():
    plan = {"days": [{"meals": [{"ingredients": ["2 cda de mantequilla"]}]}]}
    assert go._apply_condition_substitutions(plan, {"medicalConditions": ["Ninguna"]}) == 0


# ── Review adversaria: 'chicharron'/'tocino' desnudos NO deben tocar pollo/pavo/postre ──
@pytest.mark.parametrize("ing", [
    "150g de chicharrón de pollo",     # pollo frito criollo = POLLO (no cerdo)
    "3 lonjas de tocino de pavo",      # turkey bacon = ya magro
    "1 porción de tocino de cielo",    # postre tipo flan
    "1 cda de crema de leche de coco", # crema de coco = grasa vegetal, no lácteo
])
def test_does_not_substitute_poultry_or_dessert_or_coconut(ing):
    plan = {"days": [{"meals": [{"ingredients": [ing]}]}]}
    assert go._apply_condition_substitutions(plan, DYS) == 0
    assert _ings(plan) == [ing]


def test_still_substitutes_real_pork():
    """Control positivo: el chicharrón/tocino de CERDO (objetivo real) SÍ se sustituye."""
    plan = {"days": [{"meals": [{"ingredients": ["100g de chicharrón de cerdo", "50g de tocino"]}]}]}
    go._apply_condition_substitutions(plan, DYS)
    out = " ".join(_ings(plan)).lower()
    assert "chicharrón de cerdo" not in out and "pollo" in out


# ── Contrato de resolubilidad (M3): los reemplazos de dislipidemia deben existir en el catálogo
#    real, si no el delta de macros restaría el viejo y sumaría 0 → pérdida silenciosa de proteína. ──
def test_dyslipidemia_replacements_resolve_in_catalog():
    import nutrition_db as ndb
    db = ndb.IngredientNutritionDB()
    db._ensure_loaded()
    if not db._rows:
        pytest.skip("catálogo no disponible en el entorno de test (sin DB)")
    for s in cr.collect_substitutions(DYS):
        repl = s["replacement"]
        assert db.lookup(repl) is not None, f"reemplazo de dislipidemia NO resuelve al catálogo: {repl!r}"


# ── Comorbilidad: HTA + dislipidemia, tocino lo resuelve la precedencia ──
def test_hta_plus_dyslipidemia_merge():
    subs = cr.collect_substitutions({"medicalConditions": ["Hipertensión", "Colesterol alto"]})
    conds = {s["condition"] for s in subs}
    assert conds == {"hta", "dyslipidemia"}
