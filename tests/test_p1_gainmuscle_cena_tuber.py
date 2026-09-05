# -*- coding: utf-8 -*-
"""[P1-GAINMUSCLE-CENA-TUBER · 2026-09-05] Plan vivo 080a91c7 (prueba A v6):

(a) «40 g de arroz blanco crudo» en la CENA (habichuelas guisadas con queso): el piso kcal de ganancia muscular
    añade arroz a almuerzo/cena y su pasada FINAL corre después del guard de arroz nocturno. En la cena la
    guarnición es batata cocida.
(b) «Vaso rápido de ricotta con sandía y almendras, Huevo»: sandía no estaba en el léxico dulce, y el
    anti-duplicado de queso prefería huevo a yogurt por orden de la tupla.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents as _sa  # noqa: E402

NUT = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}   # 2497 kcal
FD = {"mainGoal": "gain_muscle"}


def _meal(slot, name, cals, carbs, prot, fats, ingredients=None):
    ings = list(ingredients or [f"{prot}g de proteina"])
    return {"meal": slot, "name": name, "cals": cals, "carbs": carbs, "protein": prot, "fats": fats,
            "ingredients": ings, "ingredients_raw": list(ings),
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: cocina.", "Montaje: sirve."]}


def test_cena_gets_batata_never_rice(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    # el almuerzo YA lleva arroz (se salta) ⇒ el piso cae a la cena
    days = [{"day": 1, "meals": [
        _meal("Almuerzo", "Pollo con arroz", 700, 80, 50, 18, ["200 g de pollo", "1 taza de arroz blanco cocido"]),
        _meal("Cena", "Habichuelas negras guisadas con queso blanco", 450, 50, 28, 12,
              ["¼ taza de habichuelas negras secas", "50 g de queso blanco fresco"]),
        _meal("Merienda", "Yogurt con mango", 300, 30, 20, 8),
    ]}]
    added = go._repair_gainmuscle_day_kcal(days, NUT, FD)
    assert added > 0
    cena = days[0]["meals"][1]
    assert any("batata cocida" in i for i in cena["ingredients"]), cena["ingredients"]
    assert not any("arroz" in i.lower() for i in cena["ingredients"]), cena["ingredients"]
    assert any("batata" in st.lower() for st in cena["recipe"]) and not any("arroz" in st.lower() for st in cena["recipe"])
    assert cena.get("_gainmuscle_kcal_floor") is True


def test_cena_with_batata_already_is_skipped_and_final_pass_consolidates(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    days = [{"day": 1, "meals": [
        _meal("Cena", "Pescado con batata", 500, 60, 35, 10, ["150 g de pescado", "40g de batata cocida"]),
    ]}]
    go._repair_gainmuscle_day_kcal(days, NUT, FD, final_pass=True)
    lines = [i for i in days[0]["meals"][0]["ingredients"] if "batata cocida" in i]
    assert len(lines) == 1 and int(lines[0].split("g")[0]) > 40, lines
    days2 = [{"day": 1, "meals": [
        _meal("Cena", "Pollo con batata horneada", 500, 60, 35, 10, ["150 g de pollo", "1 batata mediana"]),
    ]}]
    assert go._repair_gainmuscle_day_kcal(days2, NUT, FD) == 0   # ya lleva batata: no una segunda


def test_lunch_still_gets_rice(monkeypatch):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    days = [{"day": 1, "meals": [_meal("Almuerzo", "Pollo guisado con habichuelas", 700, 60, 50, 20)]}]
    assert go._repair_gainmuscle_day_kcal(days, NUT, FD) > 0
    assert any("arroz blanco cocido" in i for i in days[0]["meals"][0]["ingredients"])


def test_sweet_lexicon_knows_sandia_pera_toronja():
    for name in ("Vaso rápido de ricotta con sandía y almendras", "Tortilla de trigo tostada con pera",
                 "Vaso de toronja con almendras", "Ciruelas con cottage"):
        assert go._is_sweet_meal({"name": name}, _sa), name
    assert not go._is_sweet_meal({"name": "Pechuga a la plancha con batata"}, _sa)


class _Info:
    def __init__(self, name, protein, kcal, carbs=4.0, fats=3.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


def test_sweet_dish_with_cheese_prefers_yogurt_over_egg(monkeypatch):
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "CLOSER_NO_DUP_PROTEIN", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    candidates = [
        (2.0, "Queso cottage", _Info("Queso cottage", 11.0, 98.0)),
        (1.5, "Huevo", _Info("Huevo", 13.0, 155.0)),
        (1.0, "Yogurt griego", _Info("Yogurt griego", 10.0, 97.0)),
    ]
    meal = {"meal": "Merienda", "name": "Vaso rápido de ricotta con sandía y almendras",
            "protein": 6, "carbs": 20, "fats": 9, "cals": 185,
            "ingredients": ["4 cdas de queso ricotta", "1 taza de sandía", "10 almendras"],
            "ingredients_raw": ["4 cdas de queso ricotta", "1 taza de sandía", "10 almendras"],
            "recipe": ["Mise en place: corta la sandía.", "Montaje: sirve."]}
    g = go._close_protein_gap_for_meal(meal, 18.0, None, candidates, enforce_min_threshold=False)
    assert g > 0
    blob = " ".join(meal["ingredients"]).lower()
    assert "yogurt" in blob and "huevo" not in blob, meal["ingredients"]


def test_marker_anchor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "_GM_TUBER_FOOD, _GM_TUBER_KCAL_G" in src and "P1-GAINMUSCLE-CENA-TUBER" in src
