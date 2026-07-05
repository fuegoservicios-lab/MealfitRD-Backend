"""[P1-FRUIT-DEDUP-GATE-PARITY · 2026-07-05] Paridad detector del de-dup ↔ detector del gate.

Medido en vivo (corrida 99583727): el 🍓 de-dup reportó "1 fruta reescrita" y la re-review
post-quirúrgica volvió a rechazar FRUTA REPETIDA → intento quemado. Causa: el de-dup trackeaba
solo la PRIMERA fruta del nombre (`next(...)`) mientras el gate (`build_variety_report`) cuenta
TODAS las frutas featured mencionadas — "Batido de Mango y Fresas" dejaba las fresas invisibles
para el fix pero visibles para el gate.

Fix (regla-clase "detector espejo" de la madrugada 2026-07-04/05):
  (a) todas las frutas del nombre entran al tracking;
  (b) el rewrite cubre también ingredients_raw + PASOS (la lista de compras lee raw);
  (c) auto-verificación post-rewrite vía `_plan_has_same_day_fruit_repeat` (MISMA base del gate),
      2ª pasada si quedó repetición, y warning HONESTO si el residual persiste.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_and_verifier_anchored():
    assert "P1-FRUIT-DEDUP-GATE-PARITY" in _GO
    assert "def _plan_has_same_day_fruit_repeat" in _GO
    i = _GO.index("def dedup_featured_fruits_in_plan")
    body = _GO[i:i + 6000]
    assert "_plan_has_same_day_fruit_repeat(plan)" in body, "auto-verificación post-rewrite"
    assert "ingredients_raw" in body, "el rewrite cubre raw (lista de compras coherente)"


@pytest.fixture()
def go():
    import graph_orchestrator as g
    return g


def _meal(name, ings, recipe=None):
    return {"meal": "Merienda", "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings),
            "recipe": recipe or ["Mise en place: prepara.", "Montaje: sirve."]}


def test_verifier_matches_gate_basis(go):
    plan_rep = {"days": [{"day": 1, "meals": [
        _meal("Yogurt con Mango", ["100 g de mango"]),
        _meal("Batido de Mango", ["120 g de mango"]),
    ]}]}
    assert go._plan_has_same_day_fruit_repeat(plan_rep) is True
    plan_ok = {"days": [
        {"day": 1, "meals": [_meal("Yogurt con Mango", ["100 g de mango"])]},
        {"day": 2, "meals": [_meal("Batido de Mango", ["120 g de mango"])]},
    ]}
    assert go._plan_has_same_day_fruit_repeat(plan_ok) is False, \
        "misma fruta en DÍAS distintos está permitida (mismo contrato del gate)"


def test_second_fruit_in_multiword_name_now_visible(go):
    """El caso vivo: la 2ª fruta del nombre era invisible para el de-dup pero no para el gate."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Yogurt Griego con Fresas", ["¾ taza de yogurt", "80 g de fresas"]),
        _meal("Batido de Mango y Fresas", ["100 g de mango", "80 g de fresas", "1 taza de leche"],
              ["Mise en place: pela el mango y lava las fresas.",
               "Montaje: licúa las fresas con el mango y la leche; sirve frío."]),
    ]}]}
    n = go.dedup_featured_fruits_in_plan(plan)
    assert n >= 1
    assert go._plan_has_same_day_fruit_repeat(plan) is False, \
        "post-dedup el DETECTOR DEL GATE debe quedar limpio (paridad, no 'fix' de mentira)"
    m2 = plan["days"][0]["meals"][1]
    assert "fresa" not in str(m2["name"]).lower()
    assert not any("fresa" in s.lower() for s in m2["ingredients"]), "ingredientes reescritos"
    assert m2["ingredients"] == m2["ingredients_raw"], "raw en lockstep (lista de compras)"
    assert not any("fresa" in str(s).lower() for s in m2["recipe"]), "pasos reescritos"


def test_single_fruit_regression(go):
    """El caso original del P1-FRUIT-DEDUP sigue funcionando igual."""
    plan = {"days": [{"day": 1, "meals": [
        _meal("Avena con Guineo", ["1 guineo", "½ taza de avena"]),
        _meal("Batido de Guineo", ["1 guineo", "1 taza de leche"]),
    ]}]}
    assert go.dedup_featured_fruits_in_plan(plan) >= 1
    assert go._plan_has_same_day_fruit_repeat(plan) is False


def test_pool_exhausted_returns_honestly(go, monkeypatch):
    """Pool vacío → sin loop infinito, el plan queda con el residual y el gate decide (warning)."""
    monkeypatch.setattr(go, "_FRUIT_DEDUP_POOL", ())
    plan = {"days": [{"day": 1, "meals": [
        _meal("Yogurt con Mango", ["100 g de mango"]),
        _meal("Batido de Mango", ["120 g de mango"]),
    ]}]}
    assert go.dedup_featured_fruits_in_plan(plan) == 0
    assert go._plan_has_same_day_fruit_repeat(plan) is True, \
        "residual visible — el de-dup jamás reporta éxito que el gate desmiente"
