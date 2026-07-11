"""[P1-EGG-SAMEDAY-AUTOFIX · 2026-07-05] Huevo deja de estar excluido del autofix de proteína.

El label residual de la madrugada: huevo repetido same-day sobrevivía al 🍗 (huevo excluido
v1 — el swap textual rompía revoltillos), al egg-cap (solo mira el CAP global 3/12, no la
repetición same-day) y al corrector quirúrgico (medido en corridas 200c69f3/2e0cb836: la
re-review re-encontraba la repetición tras la cirugía).

Fix: la rama huevo del `_protein_repeat_autofix` reescribe la comida NO-protagonista (huevo
fuera del nombre, no aglutinante) con la maquinaria slot-aware compartida
(`_replace_meal_egg_lines`: merienda→yogurt griego, desayuno→queso, fuertes→pollo/queso;
guards alergia/dislike + anti-colisión pollo). Si ambas comidas son protagonistas
(revoltillo + tortilla el mismo día), decide el corrector/gate — jamás se rompe un plato
huevo-identidad.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    return g


def _meal(slot, name, ings):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": ["Prepara con huevo y sirve."]}


def _mk_day(second_name="Bowl Energético", second_ings=("1 huevo", "1 guineo")):
    return [{"day": 1, "meals": [
        _meal("Desayuno", "Revoltillo de Huevo con Mangú", ["2 huevos", "200 g de plátano"]),
        _meal("Merienda", second_name, list(second_ings)),
        _meal("Cena", "Pescado con Batata", ["150 g de filete de pescado blanco", "100 g de batata"]),
    ]}]


# ---------------------------------------------------------------------------

def test_egg_sameday_filler_rewritten(go):
    days = _mk_day()
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 1
    breakfast, snack = days[0]["meals"][0], days[0]["meals"][1]
    # el protagonista (revoltillo) queda intacto…
    assert "huevos" in " ".join(breakfast["ingredients"])
    # …y el relleno de la merienda pasa a proteína slot-aware (yogurt/queso).
    joined = " ".join(snack["ingredients"]).lower()
    assert "huevo" not in joined
    assert "yogurt griego" in joined or "queso blanco" in joined
    assert str(snack.get("_protein_autofix_applied", "")).startswith("huevo->")
    assert snack["ingredients"] == snack["ingredients_raw"]
    assert "huevo" not in " ".join(snack["recipe"]).lower(), "los pasos también se reescriben"


def test_two_protagonists_left_to_gate(go):
    # [P1-EGG-INTRINSIC-DEDUP · 2026-07-11] contrato actualizado: "decide el gate" produjo
    # 3 rechazos + entrega degradada (corr=9cc4317e). El pase (3) conserva el desayuno y
    # transplanta la cabeza del otro plato ("Tortilla..." → "Salteado...").
    days = _mk_day(second_name="Tortilla de Claras", second_ings=("4 claras", "50 g de espinaca"))
    assert go._protein_repeat_autofix(days, {}, db=object()) >= 1
    assert "huevos" in " ".join(days[0]["meals"][0]["ingredients"]), "keeper = desayuno"
    assert "claras" not in " ".join(days[0]["meals"][1]["ingredients"]), "merienda reasignada"
    assert not days[0]["meals"][1]["name"].lower().startswith("tortilla"), "cabeza transplantada"


def test_binder_dish_protected(go):
    # [P1-EGG-INTRINSIC-DEDUP · 2026-07-11] la croqueta (huevo-ligante) sigue INTOCABLE,
    # [P1-EGG-BINDER-GATE-EXEMPT · 2026-07-11] el binder NO cuenta para el same-day
    # (paridad gate↔autofix): revoltillo + croqueta el mismo día es LEGAL — sin cambios.
    days = _mk_day(second_name="Croquetas de Yuca", second_ings=("1 huevo", "100 g de yuca"))
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0, \
        "binder exento → un solo huevo real → sin repetición"
    assert "1 huevo" in days[0]["meals"][1]["ingredients"], \
        "el huevo-aglutinante de la croqueta es funcional"
    assert "huevos" in " ".join(days[0]["meals"][0]["ingredients"]), \
        "el desayuno queda intacto"


def test_idempotent(go):
    days = _mk_day()
    assert go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0


def test_meat_ladder_branch_unaffected(go):
    """La rama de carnes (pollo→pavo...) sigue intacta tras la inserción del branch huevo."""
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Pollo Guisado", ["150 g de pechuga de pollo", "150 g de arroz"]),
        _meal("Cena", "Wrap de Pollo", ["120 g de pechuga de pollo", "1 tortilla integral"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert days[0]["meals"][1]["_protein_autofix_applied"] == "pollo->pavo"


def test_marker_anchored_in_source():
    assert "P1-EGG-SAMEDAY-AUTOFIX" in _GO
