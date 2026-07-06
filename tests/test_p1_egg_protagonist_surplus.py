"""[P1-EGG-PROTAGONIST-SURPLUS · 2026-07-06] Cierra el ÚLTIMO hueco del gate same-day-protein
medido en vivo (3 rechazos consecutivos en la sesión 2026-07-06): un día con TRES comidas de
huevo protagonista — "Puré de Batata con Huevos Revueltos" + "Arepitas de Harina con Huevo
Revuelto" + "Revoltillo de Huevo". El egg-cap y el 🍗 saltaban a TODO protagonista, dejando la
repetición HUEVO al reviewer → retry quemado.

Fix: la rama huevo del `_protein_repeat_autofix` ahora distingue el huevo-INTRÍNSECO
(revoltillo/tortilla/omelet: el huevo ES el plato) del huevo-NOMBRADO renombrable ("X con
Huevos Revueltos": base de almidón + huevo como guarnición). Cuando el día aún tiene ≥2
comidas-huevo tras el pase de relleno, reasigna las nombradas conservando 1 (prioridad al
inevitable: intrínseco/aglutinante).
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
            "ingredients_raw": list(ings), "recipe": [f"Prepara {name} con huevo y sirve."]}


def _count_egg_meals(go, day):
    # espejo de la detección del gate (aliases huevo/claras sobre nombre + ingredientes).
    import re
    from constants import strip_accents as sa
    cnt = 0
    for m in day["meals"]:
        blob = sa((str(m.get("name", "")) + " "
                   + " ".join(str(i) for i in m.get("ingredients") or [])).lower())
        if re.search(r"\bhuevos?\b|\bclaras?\b", blob):
            cnt += 1
    return cnt


# ─────────── helpers de clasificación ───────────

def test_intrinsic_vs_added_classification(go):
    assert go._egg_is_intrinsic_dish("Revoltillo de Huevo con Plátano")
    assert go._egg_is_intrinsic_dish("Tortilla de Claras")
    assert go._egg_is_intrinsic_dish("Omelet de Vegetales")
    assert not go._egg_is_intrinsic_dish("Puré de Batata con Huevos Revueltos")
    assert not go._egg_is_intrinsic_dish("Arepitas con Huevo Revuelto")


# (los tests del reemplazo de frase huevo/claras viven en test_p1_egg_claras_phrase.py)


# ─────────── el caso vivo: 3 comidas-huevo, mezcla intrínseco + añadido ───────────

def test_three_egg_meals_keeps_one_reassigns_rest(go):
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Puré de Batata con Huevos Revueltos",
              ["200 g de batata", "2 huevos"]),
        _meal("Almuerzo", "Arepitas de Harina de Trigo con Huevo Revuelto",
              ["100 g de harina de trigo", "2 huevos"]),
        _meal("Cena", "Revoltillo de Huevo con Plátano Maduro",
              ["3 huevos", "150 g de plátano"]),
    ]}]
    n = go._protein_repeat_autofix(days, {}, db=object())
    assert n == 2, "reasigna las 2 nombradas (Puré, Arepitas), conserva el Revoltillo intrínseco"
    assert _count_egg_meals(go, days[0]) == 1, "el día queda con UNA comida-huevo → gate pasa"
    # el revoltillo (identidad) intacto
    revoltillo = days[0]["meals"][2]
    assert "huevo" in " ".join(revoltillo["ingredients"]).lower()
    # las nombradas: sin huevo en nombre NI ingredientes, con proteína real reflejada
    for m in days[0]["meals"][:2]:
        assert "huevo" not in m["name"].lower(), f"nombre sin huevo: {m['name']}"
        assert "huevo" not in " ".join(m["ingredients"]).lower()
        assert str(m.get("_protein_autofix_applied", "")).startswith("huevo->")
        assert any(t in m["name"].lower() for t in ("queso", "pollo", "yogur")), (
            f"la proteína nueva se refleja en el nombre: {m['name']}"
        )


def test_arepita_side_egg_reassignable_despite_binder_token(go):
    """'Arepitas con Huevo Revuelto' — arepita es binder-token, pero 'Huevo Revuelto' es una
    guarnición nombrada (modo de cocción), no el ligante → reasignable."""
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Revoltillo de Huevo", ["3 huevos"]),
        _meal("Merienda", "Arepitas con Huevo Revuelto", ["80 g de harina", "1 huevo"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert _count_egg_meals(go, days[0]) == 1
    assert "huevo" not in days[0]["meals"][1]["name"].lower()


def test_binder_egg_without_modifier_protected(go):
    """'Croquetas de Yuca' con huevo-ligante SIN modo de cocción en el nombre → protegido."""
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Revoltillo de Huevo", ["3 huevos"]),
        _meal("Cena", "Croquetas de Yuca", ["1 huevo", "150 g de yuca"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0, "el huevo-ligante es funcional"
    assert "1 huevo" in days[0]["meals"][1]["ingredients"]


def test_all_named_keeps_first(go):
    """Sin ningún huevo inevitable: 2 'X con Huevos' → conserva la 1ª, reasigna la 2ª."""
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Mangú con Huevos Fritos", ["200 g de plátano", "2 huevos"]),
        _meal("Cena", "Puré de Yuca con Huevos Revueltos", ["200 g de yuca", "2 huevos"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 1
    assert _count_egg_meals(go, days[0]) == 1, "conserva exactamente una comida-huevo"


def test_two_intrinsic_left_to_gate(go):
    """2 revoltillos/tortillas el mismo día: renombrar una a no-huevo es incoherente → gate/retry."""
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Revoltillo de Huevo con Mangú", ["2 huevos", "200 g de plátano"]),
        _meal("Cena", "Tortilla de Huevo con Espinaca", ["3 huevos", "80 g de espinaca"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0
    assert _count_egg_meals(go, days[0]) == 2, "ambos intrínsecos intactos (identidad)"


def test_knob_off_no_escalation(go, monkeypatch):
    monkeypatch.setattr(go, "EGG_PROTAGONIST_SURPLUS_ENABLED", False)
    days = [{"day": 1, "meals": [
        _meal("Desayuno", "Puré de Batata con Huevos Revueltos", ["200 g de batata", "2 huevos"]),
        _meal("Cena", "Revoltillo de Huevo", ["3 huevos"]),
    ]}]
    assert go._protein_repeat_autofix(days, {}, db=object()) == 0, "knob off → gate/retry decide"
    assert _count_egg_meals(go, days[0]) == 2


def test_marker_anchored_in_source():
    assert "P1-EGG-PROTAGONIST-SURPLUS" in _GO
