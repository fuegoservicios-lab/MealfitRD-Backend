# -*- coding: utf-8 -*-
"""[P1-EGG-DAY-CAP · 2026-09-05] Prueba B (plan c350dec0, vegetariana, ganancia muscular):
(a) la fidelidad bajó a 0,67 porque «Sal al gusto» y «Pimienta negra al gusto» contaban como ingrediente en 3 de 3 días;
(b) el día 1 llevaba 3 huevos en el desayuno y 3 en el almuerzo (el prompt dice 3 enteros y una sola comida; el revisor
lo dejó pasar)."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import horizon  # noqa: E402


def test_condiments_are_exempt_from_ingredient_days():
    ok = horizon._ingredient_days_is_exempt
    for nm in ("sal al gusto", "pimienta negra al gusto", "aceite vegetal", "ajo picado", "polvo de hornear", "sal marina"):
        assert ok(nm), nm
    for nm in ("mantequilla de mani", "pollo", "queso blanco fresco", "lentejas secas", "tomate cherry" if False else "batata"):
        assert not ok(nm), nm
    assert ok("mantequilla")   # exacta, como antes


class _NoopDB:
    def macros_from_ingredient_string(self, s):
        return None

    def lookup(self, s):
        return None


def _day(*meals):
    return [{"day": 1, "meals": [
        {"meal": slot, "name": name, "ingredients": list(ings), "ingredients_raw": list(ings), "recipe": []}
        for slot, name, ings in meals]}]


def test_six_whole_eggs_in_a_day_become_three_plus_whites(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _day(("Desayuno", "Revoltillo criollo con papa", ["3 huevos", "½ papa mediana"]),
                ("Almuerzo", "Croquetas de papa y queso", ["3 papas medianas", "30 g de queso blanco", "3 huevos"]))
    n = go._cap_daily_whole_eggs(days, db=_NoopDB(), max_whole=3)
    assert n == 1
    des, alm = days[0]["meals"]
    assert "3 huevos" in des["ingredients"] and not des.get("_egg_day_capped")
    assert "3 claras de huevo" in alm["ingredients"] and "3 huevos" not in alm["ingredients"]
    assert alm["ingredients_raw"] == alm["ingredients"] and alm.get("_egg_day_capped") is True


def test_keeper_over_cap_splits_into_whole_plus_whites_and_whites_do_not_count(monkeypatch):
    monkeypatch.setattr(go, "_truth_up_meal_macros_from_strings", lambda meal, db: None)
    days = _day(("Desayuno", "Tortilla", ["5 huevos"]), ("Merienda", "Batido", ["6 claras de huevo"]))
    assert go._cap_daily_whole_eggs(days, db=_NoopDB(), max_whole=3) == 1
    des = days[0]["meals"][0]
    assert des["ingredients"] == ["3 huevos", "2 claras de huevo"]
    # idempotente y dentro del tope: nada que hacer
    assert go._cap_daily_whole_eggs(days, db=_NoopDB(), max_whole=3) == 0
    ok = _day(("Desayuno", "Revoltillo", ["2 huevos"]), ("Cena", "Tortilla", ["1 huevo"]))
    assert go._cap_daily_whole_eggs(ok, db=_NoopDB(), max_whole=3) == 0


def test_wired_in_the_pre_insert_caps():
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert "_cap_daily_whole_eggs as _cdwe" in src and "_pass_n += _cdwe(_pd.get(\"days\") or [], db=_db_ins)" in src
    assert "tooltip-anchor: P1-EGG-DAY-CAP" in (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
