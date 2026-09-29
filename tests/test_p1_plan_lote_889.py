# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-889 · 2026-09-29] Tope por ALIMENTO añadido: el cerrador de proteína no llena una meta alta con 200-300 g
de un solo alimento (el dueño lo aprobó «sólo si es mejor»).

Replay del escudo vivo: 11 de las 14 porciones exageradas eran edamame cocido (200-300 g en una cena); corpus: 153 de
247. El techo genérico de proteína (300 g) las dejaba pasar. Ahora el edamame tiene su tope por línea (155 g = 1 taza) y
el cerrador lo consulta al elegir: lo que no cabe va a otro alimento o a otra comida.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import topes_por_linea as ts  # noqa: E402


class _Info:
    def __init__(self, name, protein, kcal, carbs=0.0, fats=0.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


_EDAMAME = _Info("Edamame", 11.9, 121.0, 8.9, 5.2)
_HUEVO = _Info("Huevo", 12.6, 143.0, 0.7, 9.5)


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        return None if g is None else {"grams": g, "kcal": g * 1.21, "protein": g * 0.119, "carbs": g * 0.089,
                                       "fats": g * 0.052}

    def get_nutrition(self, name):
        return {"edamame": _EDAMAME, "huevo": _HUEVO}.get(str(name).lower())

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _cena(*extra):
    return {"meal": "Cena", "name": "Canoas de plátano verde rellenas de queso fresco",
            "protein": 20, "cals": 450, "carbs": 60, "fats": 12,
            "ingredients": ["½ plátano verde", "60 g de queso fresco", *extra],
            "recipe": ["El Toque de Fuego: hornea el plátano 20 min.", "Montaje: rellena y sirve."]}


def _cerrar(meal, cands):
    return go._close_protein_gap_for_meal(meal, 60.0, _DB(), [(0.0, i.name, i) for i in cands], allergies=None,
                                          fill_pct=0.92, max_add_g=300, slot_cal_target=1400.0,
                                          enforce_min_threshold=False, diet="vegetariana", country="DO", goal="gain_muscle")


def _gramos_de(meal, token):
    return sum(_DB().grams_from_ingredient_string(x) or 0 for x in meal["ingredients"] if token in x.lower())


def test_el_edamame_de_la_cena_baja_a_una_taza_como_ultima_palabra():
    dias = [{"day": 3, "meals": [_cena("285 g de edamame cocido")]}]
    assert ts.cap(dias, db=_DB()) == 1
    assert _gramos_de(dias[0]["meals"][0], "edamame") <= 155.5, dias[0]["meals"][0]["ingredients"]


def test_margen_y_pool():
    assert ts.margen_g(_cena("100 g de edamame cocido"), "Edamame", _DB()) == 55.0
    assert ts.margen_g(_cena(), "Edamame", _DB()) == 155.0
    assert ts.margen_g(_cena(), "Pechuga de pollo", _DB()) == float("inf")
    pool = [(_EDAMAME, "edamame"), (_HUEVO, "huevo")]
    assert ts.caben(_cena("130 g de edamame cocido"), pool, _DB(), 40) == [(_HUEVO, "huevo")]
    assert ts.caben(_cena("130 g de edamame cocido"), pool[:1], _DB(), 40) == pool[:1], "sin otro, el pool tal cual"


def test_el_cerrador_no_pasa_de_una_racion(monkeypatch):
    meal = _cena()
    assert _cerrar(meal, [_EDAMAME]) > 0
    assert 40 <= _gramos_de(meal, "edamame") <= 155, meal["ingredients"]
    monkeypatch.setenv("MEALFIT_EDAMAME_LINE_CAP_G", "0")            # apagado: vuelve el techo del bolt (180 g)
    meal = _cena()
    _cerrar(meal, [_EDAMAME])
    assert _gramos_de(meal, "edamame") > 155, meal["ingredients"]


def test_si_ya_no_cabe_elige_otro_y_si_no_hay_otro_no_lo_infla():
    meal = _cena("150 g de edamame cocido")
    _cerrar(meal, [_EDAMAME, _HUEVO])
    assert _gramos_de(meal, "edamame") == 150 and any("huevo" in x.lower() for x in meal["ingredients"]), meal["ingredients"]
    meal = _cena("150 g de edamame cocido")
    assert _cerrar(meal, [_EDAMAME]) == 0
    assert _gramos_de(meal, "edamame") == 150


def test_anclas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("[P1-PLAN-LOTE-889]") >= 4
    assert "tooltip-anchor: P1-PLAN-LOTE-889" in (_BACKEND / "topes_por_linea.py").read_text(encoding="utf-8")
    pf = (_BACKEND / "protein_floor_last_word.py").read_text(encoding="utf-8")
    i_cap = pf.index("go._cap_unrealistic_portions(plan_data.get(\"days\"))")
    i_889 = pf.index("__import__(\"topes_por_linea\").cap(plan_data.get(\"days\"))")
    assert i_cap < i_889, "el tope por alimento corre DESPUÉS del bump del piso, como el techo genérico"


def test_el_bump_del_piso_no_pasa_el_tope(monkeypatch):
    """ia-59: el escudo subía el edamame 10 g por pasada (105→115→125). Tras el bump, el tope por alimento dispone."""
    import protein_floor_last_word as pflw
    plan = {"days": [{"day": 1, "meals": [_cena("150 g de edamame cocido")]}]}

    def _bump(pd):
        m = pd["days"][0]["meals"][0]
        m["ingredients"][-1] = "200 g de edamame cocido"
        return True
    monkeypatch.setattr(go, "reconcile_protein_band_post_finalize", _bump)
    monkeypatch.setattr(go, "_cap_unrealistic_portions", lambda days, *a, **k: 0)
    monkeypatch.setattr(go, "refresh_delivered_macros", lambda pd: None)
    corto = {"dia": 1, "proteina_g": 120.0, "piso_g": 135.0, "falta_g": 15.0}
    monkeypatch.setattr(pflw, "medir", lambda pd, **k: {"target_g": 150.0, "piso_pct": 0.9, "piso_g": 135.0,
                                                          "cortos": [corto], "cumple": False, "dias_medidos": 1})
    real_cap = ts.cap
    monkeypatch.setattr(ts, "cap", lambda days, db=None: real_cap(days, db=_DB()))
    pflw.reencuadra_y_mide(plan)
    assert _gramos_de(plan["days"][0]["meals"][0], "edamame") <= 155.5, plan["days"][0]["meals"][0]["ingredients"]
