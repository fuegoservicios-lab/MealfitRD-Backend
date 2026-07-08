"""[P1-SODIUM-DAIRY-SWAP · 2026-07-07] Baseline clínico 2026-07-07: HTA solo 57% de días ≤2000mg de sodio.
El driver #1 medido NO era cubito/enlatado (lo que ya cubría la escalera cured) sino QUESO COTTAGE
(406mg Na/100g × 240-245g ≈ 1000mg/comida). El swap lácteo alto-sodio → yogur griego sin azúcar (36mg Na,
proteína ≈ igual) corta ~91% del sodio de esa línea preservando la proteína.

Contrato: (1) para perfil HTA/renal con día > techo, el cottage se swapea a yogur griego; (2) para perfil
SIN condición de sodio, NO se toca (gate clínico); (3) con el knob OFF, NO se toca (rollout A/B).
"""
import os
import re

import graph_orchestrator as g

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _DB:
    """Mock: cottage alto-sodio, yogur bajo-sodio; gramos desde '(NNN g)' o 'NNN g de'."""
    _NA = {"cottage": 4.06, "yogur": 0.36, "queso blanco": 7.51}  # mg Na por gramo

    def _grams(self, s):
        m = re.search(r"\((\d+(?:\.\d+)?)\s*g\)", str(s)) or re.search(r"(\d+(?:\.\d+)?)\s*g\b", str(s))
        return float(m.group(1)) if m else 100.0

    def grams_from_ingredient_string(self, s):
        return self._grams(s)

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        gr = self._grams(s)
        for tok, na_g in self._NA.items():
            if tok in low:
                return {"grams": gr, "sodium_mg": na_g * gr}
        return {"grams": gr, "sodium_mg": 0.0}

    def macros_from_ingredient_string(self, s):
        return {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}

    def lookup(self, s):
        return object()


def _day_with_cottage():
    # ~1000mg de un cottage doble-servido + resto → sobre el techo de 2000
    return [{"meals": [
        {"name": "Bowl de Queso Cottage con Frutas", "meal": "Desayuno",
         "ingredients": ["245 g de queso cottage", "1 taza de avena (80 g)"],
         "ingredients_raw": ["245 g de queso cottage", "1 taza de avena (80 g)"],
         "recipe": ["Sirve el queso cottage con la avena."],
         "protein": 30, "carbs": 50, "fats": 5, "cals": 400},
        {"name": "Almuerzo", "meal": "Almuerzo",
         "ingredients": ["300 g de queso blanco", "arroz"],
         "ingredients_raw": ["300 g de queso blanco", "arroz"],
         "protein": 40, "carbs": 60, "fats": 10, "cals": 550},
    ]}]


def _run(days, form_data, knob):
    prev = g.SODIUM_DAIRY_SWAP_ENABLED
    g.SODIUM_DAIRY_SWAP_ENABLED = knob
    try:
        g._day_sodium_autofix(days, form_data, _DB())
    finally:
        g.SODIUM_DAIRY_SWAP_ENABLED = prev
    return " ".join(str(i).lower() for d in days for m in d["meals"] for i in m["ingredients"])


def test_hta_day_over_ceiling_swaps_cottage_to_yogurt():
    joined = _run(_day_with_cottage(), {"medicalConditions": ["Hipertensión"]}, True)
    assert "yogur" in joined, f"el cottage debió swapearse a yogur: {joined}"


def test_non_clinical_profile_keeps_cottage():
    # perfil SIN condición de sodio → el swap lácteo NO aplica (gate clínico)
    joined = _run(_day_with_cottage(), {"medicalConditions": ["Ninguna"]}, True)
    assert "queso cottage" in joined, f"perfil sin HTA/renal NO debe perder su cottage: {joined}"


def test_knob_off_keeps_cottage():
    joined = _run(_day_with_cottage(), {"medicalConditions": ["Hipertensión"]}, False)
    assert "queso cottage" in joined, f"knob OFF: sin swap lácteo: {joined}"


def test_name_rewritten_no_phantom():
    days = _day_with_cottage()
    _run(days, {"medicalConditions": ["Hipertensión"]}, True)
    names = " ".join(str(m.get("name", "")).lower() for d in days for m in d["meals"])
    # tras swapear el cottage del desayuno, su NOMBRE no debe seguir diciendo "cottage" (fantasma)
    swapped_meal = days[0]["meals"][0]
    if "yogur" in " ".join(str(i).lower() for i in swapped_meal["ingredients"]):
        assert "cottage" not in str(swapped_meal.get("name", "")).lower(), \
            f"nombre-fantasma tras swap: {swapped_meal.get('name')}"


def test_anchored():
    assert "P1-SODIUM-DAIRY-SWAP" in _GO
    assert "SODIUM_DAIRY_SWAP_ENABLED" in _GO
    assert "_SODIUM_DAIRY_SWAP_LADDER" in _GO
