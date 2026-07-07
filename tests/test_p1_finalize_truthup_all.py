"""[P1-FINALIZE-TRUTHUP-ALL · 2026-07-07] Forense del plan vivo 4b9291fe (degradado, carbs band 0.333):
la merienda "Pera Fresca Crujiente con Maní y Queso Blanco" listaba `2 peras medianas (300 g)` en sus
ingredientes (raw + display) pero sus macros guardadas eran C3 P3 F8 k90 — las peras (verificadas,
15.23g carb/100g → ~46g) NO estaban contadas. El truth-up del finalizer solo corría CONDICIONALMENTE
(dentro de los `if` de veg/carb-ghost) y TEMPRANO → un plato desincronizado aguas arriba cuyo día no
disparaba esos pases NUNCA recibía reconciliación → el band score leía macros subcontadas → degradado.

Fix: pasada FINAL incondicional de truth-up al cierre de `finalize_plan_data_coherence`, tras todos los
pases que mutan strings. El abort-gate conservador del truth-up protege platos no-resolubles.
"""
import os
import re

import pytest

import graph_orchestrator as g


@pytest.fixture(autouse=True)
def _no_catalog_pool(monkeypatch):
    """Hermético: los pases del finalizer que consultan el catálogo global NO deben tocar el pool Neon
    (cuelga sin DB). El truth-up final usa el `db` mock que pasamos, no `get_master_ingredients`."""
    try:
        import shopping_calculator
        monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [], raising=False)
    except Exception:
        pass

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


class _Info:
    def __init__(self, kcal, protein, carbs, fats):
        self.kcal, self.protein, self.carbs, self.fats = kcal, protein, carbs, fats
        self.name = "x"
        self.density_g_per_cup = None


# macros/100g de los 3 alimentos del plato degradado (valores reales del catálogo)
_PER100 = {
    "pera": _Info(57.0, 0.36, 15.23, 0.14),
    "man": _Info(567.0, 24.4, 21.3, 49.0),      # maní
    "girasol": _Info(584.0, 20.8, 20.0, 51.5),  # semillas de girasol
}


def _resolve(s):
    low = str(s).lower()
    for tok, info in _PER100.items():
        if tok in low:
            return info
    return None


class _DB:
    """Mock mínimo: resuelve pera/maní/girasol por gramos de '(NNN g)' o 'NNN g'."""
    def macros_from_ingredient_string(self, s):
        info = _resolve(s)
        if info is None:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)\s*g", str(s))
        if not m:
            return None
        gr = float(m.group(1))
        f = gr / 100.0
        return {"name": info.name, "grams": gr, "kcal": round(info.kcal * f, 1),
                "protein": round(info.protein * f, 2), "carbs": round(info.carbs * f, 2),
                "fats": round(info.fats * f, 2)}

    def lookup(self, s):
        return _resolve(s)

    def _ingredient_macro_group(self, *a, **k):
        return None


def _pear_meal():
    # macros STALE (solo maní+girasol contados) — las peras (300g, ~46g carb) faltan.
    return {"name": "Pera Fresca Crujiente con Maní Tostado y Queso Blanco", "meal": "Merienda",
            "ingredients": ["5 g de maní tostado sin sal", "10 g de semillas de girasol",
                            "2 peras medianas (300 g)"],
            "ingredients_raw": ["2 pera mediana (300 g)", "5 g de maní tostado sin sal",
                                "10 g de semillas de girasol"],
            "recipe": ["Sirve las peras con el maní y las semillas."],
            "protein": 3, "carbs": 3, "fats": 8, "cals": 90}


def test_final_truthup_reconciles_uncounted_pears():
    """El finalizer recuenta las peras (C3 -> ~49) via la pasada final incondicional."""
    meal = _pear_meal()
    days = [{"meals": [meal]}]
    g.finalize_plan_data_coherence(days, db=_DB())
    # 300g pera (~46g) + maní 5g (~1) + girasol 10g (~2) ≈ 49g carb
    assert meal["carbs"] >= 40, f"carbs no reconciliadas: {meal['carbs']} (esperado ~49)"
    assert meal["cals"] >= 200, f"cals no reconciliadas: {meal['cals']} (esperado ~220)"


def test_truthup_all_can_be_disabled():
    """Con el knob OFF, la pasada final no corre → macros stale intactas (rollback sin redeploy).
    Parchea la constante de módulo (reload de graph_orchestrator es prohibitivamente lento/cuelga)."""
    prev = g.FINALIZE_TRUTHUP_ALL_ENABLED
    g.FINALIZE_TRUTHUP_ALL_ENABLED = False
    try:
        meal = _pear_meal()
        g.finalize_plan_data_coherence([{"meals": [meal]}], db=_DB())
        assert meal["carbs"] == 3, f"knob OFF pero truth-up corrió igual: {meal['carbs']}"
    finally:
        g.FINALIZE_TRUTHUP_ALL_ENABLED = prev


def test_anchored():
    assert "P1-FINALIZE-TRUTHUP-ALL" in _GO
    assert "FINALIZE_TRUTHUP_ALL_ENABLED" in _GO
    # la pasada final DEBE estar despues del ultimo pase que muta strings (contract-lint / display-polish),
    # e inmediatamente antes del return del finalizer.
    _fn = _GO[_GO.index("def finalize_plan_data_coherence"):]
    _fn = _fn[:_fn.index("\ndef ", 1)]
    assert "final_truthup" in _fn
    assert _fn.index("FINALIZE_TRUTHUP_ALL_ENABLED") > _fn.index("display_polish"), \
        "la pasada final de truth-up debe correr DESPUES de las mutaciones de string"
