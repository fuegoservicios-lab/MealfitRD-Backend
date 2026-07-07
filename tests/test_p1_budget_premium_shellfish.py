"""[P1-BUDGET-PREMIUM-SHELLFISH · 2026-07-07] Review del plan vivo 4e7b8dbb (24% sobre presupuesto):
el driver de costo #1 era "Cangrejo" (RD$599, RD$479/lb) pero NO estaba en la familia de mariscos de
`_BUDGET_DRIVER_FAMILIES` → la convergencia de presupuesto lo rankeaba #1 y no lo podía sustituir. Fix:
añadir cangrejo/jaiba/langostino/vieira/almeja/mejillón/lambí/ostra/concha a la familia premium (sustituto
"Filete de pescado blanco", RD$127.5/lb = ~73% más barato).
"""
import os
import re

import graph_orchestrator as g

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()

_SEAFOOD_RX = g._BUDGET_DRIVER_FAMILIES[0][0]


def _matches(name):
    return bool(re.search(rf"\b(?:{_SEAFOOD_RX})\b", name, re.IGNORECASE))


def test_premium_shellfish_in_family():
    for n in ["Cangrejo", "Jaiba", "Langostino", "Langostinos", "Camarones", "Langosta",
              "Vieira", "Almejas", "Mejillón", "Lambí", "Concha de lambí"]:
        assert _matches(n), f"{n} debe estar en la familia de mariscos premium"


def test_cheap_fish_and_staples_not_matched():
    for n in ["Filete de pescado blanco", "Pollo", "Chivo", "Huevo", "Sardinas"]:
        assert not _matches(n), f"{n} NO debe matchear la familia premium"


def test_crab_gets_substituted(monkeypatch):
    """Funcional: con Cangrejo como driver caro, la convergencia lo sustituye por pescado blanco."""
    monkeypatch.setattr(g, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(g, "_budget_build_master_price_map",
                        lambda: {"cangrejo": 479.0, "carne de cangrejo": 479.0,
                                 "filete de pescado blanco": 127.5, "pescado blanco": 127.5})
    meal = {"name": "Batata Rellena de Cangrejo", "meal": "Almuerzo",
            "ingredients": ["105 g de carne de cangrejo", "1/2 batata"],
            "ingredients_raw": ["105 g de carne de cangrejo", "1/2 batata"],
            "recipe": ["Desmenuza el cangrejo y saltéalo.", "Rellena la batata con el cangrejo."]}
    days = [{"day": 1, "meals": [meal]}]
    weekly = [{"name": "Cangrejo", "estimated_cost_rd": 599},
              {"name": "Batata", "estimated_cost_rd": 32}]
    subs = g._apply_budget_driver_aware_pass(days, {"allergies": []}, weekly)
    assert subs >= 1, "el cangrejo debió sustituirse"
    joined = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "cangrejo" not in joined, f"el cangrejo debió salir: {meal['ingredients']}"
    assert "pescado" in joined, f"debió entrar pescado blanco: {meal['ingredients']}"


def test_no_sub_when_saving_insufficient(monkeypatch):
    """Si el sustituto no ahorra ≥30%, no sustituye (guard preservado)."""
    monkeypatch.setattr(g, "BUDGET_DRIVER_AWARE_ENABLED", True)
    monkeypatch.setattr(g, "_budget_build_master_price_map",
                        lambda: {"cangrejo": 130.0, "carne de cangrejo": 130.0,
                                 "filete de pescado blanco": 127.5})  # solo ~2% más barato
    meal = {"name": "X", "meal": "Almuerzo", "ingredients": ["105 g de carne de cangrejo"],
            "ingredients_raw": ["105 g de carne de cangrejo"], "recipe": ["cocina"]}
    days = [{"day": 1, "meals": [meal]}]
    weekly = [{"name": "Cangrejo", "estimated_cost_rd": 599}]
    subs = g._apply_budget_driver_aware_pass(days, {"allergies": []}, weekly)
    assert subs == 0
    assert "cangrejo" in " ".join(meal["ingredients"]).lower()


def test_marker_anchored():
    assert "P1-BUDGET-PREMIUM-SHELLFISH" in _GO
