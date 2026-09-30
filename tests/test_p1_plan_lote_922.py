# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-922 · 2026-09-29] Las hierbas frescas son exentas del piso cocinable: «10 g de cilantro» no es una
porción inservible.

Medidor de punto fijo del escudo (913, ia-59): `_floor_subservible_portions` dropeaba «10 g de cilantro» (sin headroom)
y la receta seguía nombrándolo. Ronda 2: la primera versión además impedía que `_rebalance_day_macros_to_target` bajara
bajo el piso una línea que lo cumplía («15 g de avena» → 10 g); el A/B del escudo sobre 92 planes (sellando la política
como el router) mostró que, para sostener 15 g de maní o almendras, el grupo recortaba la proteína: 5 días entre -5 y
-9 g (mozzarella 45 → 25 g, edamame 155 → 80 g). Esa parte se quitó: la proteína vale más que la guarnición.
"""
from __future__ import annotations

import pathlib
import re
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_POR100 = {"avena": (13.0, 66.0, 7.0), "arroz": (2.7, 28.0, 0.3), "cilantro": (2.1, 3.7, 0.5), "pollo": (31.0, 0.0, 3.6)}


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        k = next((k for k in _POR100 if k in str(s).lower()), None)
        if g is None or k is None:
            return None
        p, c, f = (v * g / 100.0 for v in _POR100[k])
        return {"grams": g, "protein": p, "carbs": c, "fats": f, "kcal": 4 * p + 4 * c + 9 * f}


def _dia(ings):
    db = _DB()
    mc = [db.macros_from_ingredient_string(x) or {} for x in ings]
    tot = {k: round(sum(m.get(k, 0) for m in mc)) for k in ("protein", "carbs", "fats", "kcal")}
    return [{"meals": [{"meal": "Almuerzo", "name": "Pollo a la plancha con arroz blanco", "ingredients": list(ings),
                        "ingredients_raw": list(ings), "protein": tot["protein"], "carbs": tot["carbs"],
                        "fats": tot["fats"], "cals": tot["kcal"]}]}]


def test_el_piso_no_dropea_el_cilantro_sin_headroom():
    days = _dia(["10 g de cilantro fresco", "10 g de avena", "200 g de arroz blanco cocido", "150 g de pollo a la plancha"])
    go._floor_subservible_portions(days, day_kcal_target=100.0, db=_DB())
    ings = days[0]["meals"][0]["ingredients"]
    assert any("cilantro" in x for x in ings), ings
    assert not any("avena" in x for x in ings), "el piso sigue dropeando lo que no es hierba"


def test_las_hierbas_estan_en_los_exentos():
    for h in ("cilantro", "perejil", "cebollin", "albahaca", "tomillo"):
        assert h in go._SHRINK_FLOOR_EXEMPT_TOKENS


def test_el_rebalance_no_sostiene_la_linea_sobre_la_proteina():
    """Ronda 2: el ajuste de banda puede bajar una guarnición bajo el piso; no se sostiene a costa del grupo."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "piso_en_rebalance" not in src
    assert not (_BACKEND / "piso_en_rebalance.py").exists()
