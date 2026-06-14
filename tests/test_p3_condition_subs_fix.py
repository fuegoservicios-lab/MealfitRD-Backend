"""[P3-CONDITION-SUBS-FIX · 2026-06-14] Regresión de DOS bugs del guard de sustitución por condición
(`_apply_condition_substitutions` + `_HTA_SODIUM_SUBS`), ambos hallados por review adversaria sobre
el commit P3-CONDITION-ENGINE:

  BUG 1 (macros/compra): la sustitución reemplazaba un ingrediente CUANTIFICADO con peso
  ("100g de longaniza") por una etiqueta SIN cantidad ("Pechuga de pollo") → (a) macros del plato
  quedaban describiendo el embutido viejo (el docstring mentía "Macro-preservante"), (b) la lista de
  compras derivaba peso=0 ("Al gusto de Pechuga de pollo"). FIX: para staples weight-bearing se
  preserva el prefijo de cantidad Y se ajustan los macros por delta quirúrgico macros(nuevo)-macros(viejo).

  BUG 2 (falso positivo de proteína vegetal): el token DESNUDO 'soya' (substring) borraba
  'Leche de soya', 'carne de soya', 'Soya/Tofu' (proteína legítima) reemplazándolos por un condimento
  sin proteína → socavaba el piso de proteína FS6 en pacientes HTA vegetarianos. Análogo: 'cubito'
  matcheaba 'cubitos de pollo' (pollo en cubos). FIX: tokens estrechos ('salsa de soya', 'cubito de').
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import condition_rules as cr
import graph_orchestrator as go

HTA = {"medicalConditions": ["Hipertensión"]}


def _ings(plan, day=0, meal=0):
    return plan["days"][day]["meals"][meal]["ingredients"]


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 — el reemplazo de un staple conserva la CANTIDAD (peso para la lista de compras)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("orig", ["100g de longaniza", "2 lonjas de salami", "150 g de jamón"])
def test_weight_bearing_substitution_carries_quantity(orig):
    """El reemplazo de un embutido (staple) DEBE empezar con una cantidad → la lista de compras
    conserva un peso comprable en vez de 'al gusto'."""
    plan = {"days": [{"meals": [{"ingredients": [orig], "ingredients_raw": [orig]}]}]}
    go._apply_condition_substitutions(plan, HTA)
    out = _ings(plan)
    assert len(out) == 1
    repl = str(out[0])
    assert repl[0].isdigit(), f"el reemplazo debe llevar cantidad, no '{repl}'"
    # el embutido desapareció y entró la proteína fresca
    assert "longaniza" not in repl.lower() and "salami" not in repl.lower()
    assert "pollo" in repl.lower()


def test_quantity_prefix_regex_extracts_leading_amount():
    rx = go._COND_SUB_QTY_PREFIX_RE
    assert rx is not None
    assert rx.match("100g de longaniza").group(1).strip() == "100g de"
    assert rx.match("2 lonjas de salami").group(1).strip() == "2 lonjas de"
    # sin cantidad → sin match (condimento "al gusto")
    assert rx.match("Soya/Tofu") is None


def test_seasoning_substitution_drops_offending_unit():
    """El cubito de caldo (la UNIDAD misma es lo contraindicado) NO preserva 'cubito' — cae a bare."""
    plan = {"days": [{"meals": [{"ingredients": ["1 cubito de pollo", "Arroz"]}]}]}
    go._apply_condition_substitutions(plan, HTA)
    out = _ings(plan)
    assert not any("cubito" in str(i).lower() for i in out)
    assert "Arroz" in out


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 — los macros del plato se AJUSTAN por delta (no quedan describiendo el viejo ingrediente)
# ─────────────────────────────────────────────────────────────────────────────
class _StubDB:
    """DB mínima: longaniza (graso, poca proteína) vs pechuga de pollo (magra, alta proteína)."""
    def __init__(self, *a, **k):
        pass

    def macros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "longaniza" in low:
            return {"protein": 14.0, "carbs": 2.0, "fats": 30.0, "kcal": 330.0}
        if "pechuga" in low or "pollo" in low:
            return {"protein": 31.0, "carbs": 0.0, "fats": 4.0, "kcal": 165.0}
        return None


def test_macro_delta_applied_after_substitution(monkeypatch):
    """Tras sustituir longaniza→pechuga, los macros del plato suben en proteína y bajan en grasa
    (delta quirúrgico), en vez de quedar describiendo la longaniza."""
    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", _StubDB)
    plan = {"days": [{"meals": [{
        "name": "Almuerzo", "ingredients": ["100g de longaniza"],
        "protein": 20, "carbs": 40, "fats": 35, "cals": 500,
    }]}]}
    go._apply_condition_substitutions(plan, HTA)
    m = plan["days"][0]["meals"][0]
    # delta = pollo(31/0/4/165) - longaniza(14/2/30/330)
    assert m["protein"] == 20 + (31 - 14)   # 37
    assert m["fats"] == 35 + (4 - 30)       # 9
    assert m["carbs"] == 40 + (0 - 2)       # 38
    assert m["cals"] == 500 + (165 - 330)   # 335
    assert m["macros"] == ["P:37g", "C:38g", "G:9g"]


def test_macro_delta_failsafe_when_db_unavailable(monkeypatch):
    """Si la DB no resuelve nada (offline / ingrediente desconocido), los macros previos se respetan
    (conservador) — la sustitución textual + la cantidad ya quedaron aplicadas."""
    class _EmptyDB:
        def __init__(self, *a, **k):
            pass

        def macros_from_ingredient_string(self, s):
            return None

    monkeypatch.setattr("nutrition_db.IngredientNutritionDB", _EmptyDB)
    plan = {"days": [{"meals": [{
        "ingredients": ["100g de longaniza"], "protein": 20, "carbs": 40, "fats": 35, "cals": 500,
    }]}]}
    go._apply_condition_substitutions(plan, HTA)
    m = plan["days"][0]["meals"][0]
    assert (m["protein"], m["carbs"], m["fats"], m["cals"]) == (20, 40, 35, 500)
    assert str(_ings(plan)[0]).startswith("100g de")   # cantidad preservada igual


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2 — el token 'soya'/'cubito' desnudo NO debe borrar proteína vegetal / pollo en cubos
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("protein_ing", [
    "Leche de soya sin azucar",
    "200g de carne de soya",
    "Soya/Tofu",
    "150g de tofu firme",
    "Edamame al vapor",
])
def test_hta_does_not_substitute_vegetable_protein(protein_ing):
    """Un paciente HTA vegetariano NO debe perder su fuente de proteína de soya/tofu por el guard."""
    plan = {"days": [{"meals": [{"ingredients": [protein_ing]}]}]}
    n = go._apply_condition_substitutions(plan, HTA)
    assert n == 0, f"'{protein_ing}' (proteína vegetal) NO debe sustituirse"
    assert _ings(plan) == [protein_ing]


def test_hta_does_not_substitute_diced_chicken():
    """'cubitos de pollo' (pollo en cubos = proteína) NO debe matchear el token 'cubito de'."""
    plan = {"days": [{"meals": [{"ingredients": ["200g de cubitos de pollo"]}]}]}
    n = go._apply_condition_substitutions(plan, HTA)
    assert n == 0
    assert _ings(plan) == ["200g de cubitos de pollo"]


def test_hta_still_substitutes_genuine_soy_sauce():
    """La salsa de soya (alta en sodio) SÍ se sustituye — el fix no debe sobre-corregir."""
    plan = {"days": [{"meals": [{"ingredients": ["2 cda de salsa de soya"]}]}]}
    go._apply_condition_substitutions(plan, HTA)
    assert not any("salsa de soya" in str(i).lower() for i in _ings(plan))


# ─────────────────────────────────────────────────────────────────────────────
# Contrato del flag preserve_qty en el registro
# ─────────────────────────────────────────────────────────────────────────────
def test_preserve_qty_flag_only_on_weight_bearing_staples():
    subs = cr.collect_substitutions(HTA)
    by_label = {s["label"]: s["preserve_qty"] for s in subs}
    assert by_label["embutidos"] is True
    assert by_label["pescado salado"] is True
    assert by_label["cubitos/sazón en polvo"] is False
    assert by_label["salsa de soya"] is False
    assert by_label["sazonadores salados"] is False


def test_dm2_sugar_subs_are_not_weight_bearing():
    subs = cr.collect_substitutions({"medicalConditions": ["Diabetes T2"]})
    assert subs and all(s["preserve_qty"] is False for s in subs)
