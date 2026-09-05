# -*- coding: utf-8 -*-
"""[P1-SNACK-MAKE-ROOM · 2026-09-05] Meriendas de 3 y 11 g de proteína (planes vivos c350dec0 día 10, 606e9017 días 2-3)
ya en su tope de calorías con fruta y frutos secos: el cerrador y el rescate devolvían 0 g por falta de headroom. Ahora
en franja ligera se encoge la fruta / los frutos secos (nunca por debajo de la mitad) para meter proteína."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents as _sa  # noqa: E402


class _DB:
    """macros por 100 g y parseo «N g de X» / «N cdas de X» (suficiente para el test)."""
    PER100 = {"higo": (0.8, 19, 0.3, 74), "membrillo": (0.4, 15, 0.1, 57), "almendra": (21, 22, 50, 579),
              "mantequilla de mani": (25, 20, 50, 588), "queso cottage": (11, 3.4, 4.3, 98), "yogurt griego": (10, 4, 5, 97)}

    def _grams(self, s):
        import re
        low = _sa(str(s).lower())
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos)\b", low)
        if m:
            return float(m.group(1).replace(",", "."))
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:cdas?|cucharadas?)\b", low)
        if m:
            return float(m.group(1).replace(",", ".")) * 15.0
        return 0.0

    def macros_from_ingredient_string(self, s):
        low = _sa(str(s).lower())
        for k, (p, c, f, kc) in self.PER100.items():
            if k in low:
                g = self._grams(s)
                if g <= 0:
                    return None
                return {"protein": p * g / 100, "carbs": c * g / 100, "fats": f * g / 100, "kcal": kc * g / 100}
        return None

    def lookup(self, s):
        return None


def _truth(meal, db):
    tot = {"protein": 0.0, "carbs": 0.0, "fats": 0.0, "kcal": 0.0}
    for i in meal.get("ingredients") or []:
        mc = db.macros_from_ingredient_string(i) or {}
        for k in tot:
            tot[k] += float(mc.get(k) or 0.0)
    meal["protein"], meal["carbs"], meal["fats"], meal["cals"] = round(tot["protein"]), round(tot["carbs"]), round(tot["fats"]), round(tot["kcal"])
    return True


class _Info:
    def __init__(self, name, protein, kcal, carbs=3.4, fats=4.3):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


def _snack():
    m = {"meal": "Merienda", "name": "Vasito grab-and-go de higo, membrillo y almendras con Queso Cottage",
         "ingredients": ["60 g de higo", "40 g de membrillo", "30 g de almendras", "30 g de queso cottage"],
         "ingredients_raw": ["60 g de higo", "40 g de membrillo", "30 g de almendras", "30 g de queso cottage"],
         "recipe": ["Mise en place: corta.", "Montaje: sirve."]}
    _truth(m, _DB())
    return m


def test_make_room_shrinks_nuts_not_below_half(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_SNACK_MAKE_ROOM", True)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", _truth)
    m = _snack()
    before = m["cals"]
    freed = go._make_room_for_protein(m, 60.0, _DB())
    assert freed >= 55, freed
    alm = next(i for i in m["ingredients"] if "almendra" in i)
    assert "30 g" not in alm and float(alm.split("g")[0]) >= 15, alm
    assert m["cals"] < before and "30 g de queso cottage" in m["ingredients"]
    # sin palanca (solo proteína) ⇒ 0
    m2 = {"meal": "Merienda", "name": "x", "ingredients": ["30 g de queso cottage"], "ingredients_raw": ["30 g de queso cottage"]}
    assert go._make_room_for_protein(m2, 60.0, _DB()) == 0.0


def test_closer_rescues_the_live_snack(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_SNACK_MAKE_ROOM", True)
    monkeypatch.setattr(go, "CLOSER_DISH_COHERENCE_ENABLED", True)
    monkeypatch.setattr(go, "PROTEIN_CLOSER_SCALE_FIRST", False)
    monkeypatch.setattr(go, "_scale_congruent_protein_line", lambda *a, **k: False)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", _truth)
    m = _snack()
    cals0 = m["cals"]
    candidates = [(1.0, "Yogurt griego", _Info("Yogurt griego", 10.0, 97.0, carbs=4.0, fats=5.0))]
    # tope de kcal = lo que ya tiene: hoy el cerrador devolvía 0
    g = go._close_protein_gap_for_meal(m, 18.0, _DB(), candidates, enforce_min_threshold=False, slot_cal_target=float(cals0))
    assert g >= go.CLOSER_COOKABLE_MIN_G, g
    assert any("yogurt" in i.lower() for i in m["ingredients"]), m["ingredients"]
    assert m["cals"] <= cals0 * 1.06, (m["cals"], cals0)


def test_knob_off_keeps_legacy(monkeypatch):
    monkeypatch.setattr(go, "LIGHT_SNACK_MAKE_ROOM", False)
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", _truth)
    m = _snack()
    assert go._make_room_for_protein(m, 60.0, _DB()) == 0.0
    assert "30 g de almendras" in m["ingredients"]


def test_anchor():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-SNACK-MAKE-ROOM" in src and src.count("_make_room_for_protein(meal, _need_k, db)") == 2
