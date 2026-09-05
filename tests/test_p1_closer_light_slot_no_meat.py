# -*- coding: utf-8 -*-
"""[P1-CLOSER-LIGHT-SLOT-NO-MEAT · 2026-09-05] Dos defectos del plan vivo 2a2e2516 (prueba A v4):

1. Dos meriendas salieron con «, Camarones» pegado al título: «Vaso de toronja con almendras y mantequilla de
   maní» y «Tortilla de trigo tostada con pera y mantequilla de maní». No son «dulces» para el léxico del guard
   (toronja/pera no son marcadores), la pasta de untar sacó al queso del pool (NO-SPREAD-PLUS-CHEESE) y el huevo
   colisionaba con el desayuno ⇒ el cerrador cayó a «la más magra»: camarones. La franja (merienda/desayuno) era
   una PREFERENCIA, no una regla. Ahora vive en el SSOT `_dish_coherence_filter`: franja ligera ⇒ sin carne/pescado/marisco.
2. Pescado en las tres cenas (Diversidad 4/10): la clase 6 del endurecedor prestaba «la proteína menos usada» sin
   actualizar el contador, así que prestaba la MISMA a todos los días.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents as _sa  # noqa: E402


def _merienda(name, ingredients):
    return {"meal": "Merienda", "name": name, "ingredients": ingredients}


def test_light_slot_rejects_meat_and_seafood_even_when_not_sweet(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    m = _merienda("Vaso de toronja con almendras y mantequilla de maní",
                  ["1½ toronjas", "15 g de almendras fileteadas", "1 cdta de mantequilla de maní"])
    assert not go._is_sweet_meal(m, _sa), "el caso vivo NO es dulce para el léxico: por eso el guard dulce no lo cubría"
    ok = go._dish_coherence_filter(m, _sa)
    for meat in ("camarones", "filete de pescado blanco", "pechuga de pollo", "atun en agua", "carne de res"):
        assert not ok(meat), meat
    for light in ("yogurt griego entero", "huevos", "lentejas", "claras de huevo"):
        assert ok(light), light
    # tortilla de trigo con pera: mismo caso, segundo plato vivo
    m2 = _merienda("Tortilla de trigo tostada con pera y mantequilla de maní", ["2 tortillas de trigo", "½ pera"])
    assert not go._dish_coherence_filter(m2, _sa)("camarones")


def test_savory_hot_light_meal_admits_chicken_turkey_tuna_but_never_seafood(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    m = {"meal": "Merienda", "name": "Sándwich integral de vegetales",
         "ingredients": ["2 rebanadas de pan integral", "1/2 taza de vegetales salteados"],
         "recipe": ["MISE EN PLACE: Prepara los vegetales.", "EL TOQUE DE FUEGO: Saltea 5 min.", "MONTAJE: Arma el sándwich."]}
    ok = go._dish_coherence_filter(m, _sa)
    assert ok("pechuga de pollo") and ok("pechuga de pavo") and ok("atun en agua")
    for never in ("camarones", "filete de pescado blanco", "carne de res", "chuleta de cerdo", "calamar"):
        assert not ok(never), never


def test_main_slots_keep_legacy_behaviour(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    m = {"meal": "Almuerzo", "name": "Ensalada de garbanzos con vegetales", "ingredients": ["150 g de garbanzos", "lechuga"]}
    ok = go._dish_coherence_filter(m, _sa)
    assert ok("pechuga de pollo") and ok("camarones"), "un almuerzo sin proteína principal sigue admitiendo carne"


def test_light_rule_is_gated_by_dish_coherence_knob(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", False)
    m = _merienda("Vaso de toronja con almendras y mantequilla de maní", ["1 toronja"])
    assert go._dish_coherence_filter(m, _sa)("camarones") is True


def _days_with(pool_of):
    return {"days": [{"day": i + 1, "protein_pool": list(p), "carb_pool": [], "fruit_pool": []}
                     for i, p in enumerate(pool_of)]}


def test_class6_lends_proteins_round_robin_not_the_same_to_every_day(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY", True)
    monkeypatch.setattr(go, "HARDEN_MAIN_ARITY_TARGET", 3)
    # el reparto vivo: día 1 pescado+huevo+queso, día 2 pavo+huevo, día 3 queso+yogurt (sin huevo por el diversificador)
    skel = _days_with([
        ["Filete de pescado blanco", "Huevos", "Queso blanco fresco"],
        ["Pechuga de pavo", "Huevos", "Mantequilla de maní"],
        ["Yogurt griego entero", "Queso blanco fresco", "Frutos secos (nueces)"],
    ])
    go.harden_day_pools(skel, {}, None)
    pools = [d["protein_pool"] for d in skel["days"]]
    lent_fish_to = [i for i, p in enumerate(pools) if i != 0 and "Filete de pescado blanco" in p]
    lent_turkey_to = [i for i, p in enumerate(pools) if i != 1 and "Pechuga de pavo" in p]
    # antes: pescado prestado a los días 2 Y 3 y pavo a ninguno ⇒ pescado en las tres cenas
    assert not (len(lent_fish_to) == 2 and not lent_turkey_to), f"pescado a todos: {pools}"
    assert lent_turkey_to, f"el pavo también debe prestarse: {pools}"


def test_marker_and_anchor_present():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-CLOSER-LIGHT-SLOT-NO-MEAT" in src
    assert "_freq[_lbl] = _freq.get(_lbl, 0) + 1" in src


# ─── el título tras el cerrador / el swap de huevo ───────────────────────────────────────────────

def test_reflection_reopens_the_enumeration_instead_of_a_trailing_comma(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    m = {"name": "Vaso de toronja con almendras y mantequilla de maní"}
    assert go._reflect_added_protein_in_name(m, "yogurt griego entero", _sa) is True
    assert m["name"] == "Vaso de toronja con almendras, mantequilla de maní y Yogurt Griego Entero"
    m2 = {"name": "Queso blanco fresco con durazno y almendras"}
    go._reflect_added_protein_in_name(m2, "huevo", _sa)
    assert m2["name"] == "Queso blanco fresco con durazno, almendras y Huevo"
    # los dos casos previos del conector siguen igual
    m3 = {"name": "Revoltillo con Kale"}; go._reflect_added_protein_in_name(m3, "yogur griego", _sa)
    assert m3["name"] == "Revoltillo con Kale y Yogur Griego"
    m4 = {"name": "Batido de Frutas"}; go._reflect_added_protein_in_name(m4, "yogur griego", _sa)
    assert m4["name"] == "Batido de Frutas con Yogur Griego"


def test_egg_to_yogurt_name_sync_drops_the_dangling_participle():
    assert go._fix_egg_swap_dangling_adjectives("Batido espeso de avena con Yogurt Griego cocido") == \
        "Batido espeso de avena con Yogurt Griego"
    assert go._fix_egg_swap_dangling_adjectives("Tostadas con yogur griego entero y aguacate") == \
        "Tostadas con yogur griego entero y aguacate"
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("swap huevo→yogur sincronizó el nombre")
    assert "_nm_new = _fix_egg_swap_dangling_adjectives(_nm_new)" in src[i - 900:i], "el nombre pasa por el reparador"
