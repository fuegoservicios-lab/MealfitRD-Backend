# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-805 · 2026-09-28] El refinador entero no baja lo que da nombre al plato por debajo de su piso.

Batería real (embarazo, 28-sep): el rebalanceo del día paró el mango de «Tostadas integrales con queso fresco, mango y
yogurt griego entero» en su piso (60 g, lote 177) y `refine_day_portions_integer` lo dejó en 30 g.
"""
from __future__ import annotations

import re

import portion_solver as ps

# kcal, proteína, carbohidrato, grasa por gramo
_POR_G = {"mango": (0.60, 0.008, 0.15, 0.004), "queso blanco": (2.64, 0.18, 0.03, 0.20),
          "pan integral": (2.50, 0.13, 0.41, 0.035), "zanahoria": (0.41, 0.009, 0.10, 0.002)}
_CAT = {"mango": "Frutas", "queso blanco": "Lácteos", "pan integral": "Granos", "zanahoria": "Vegetales"}


class _DB:
    def _fila(self, s):
        n = str(s).lower()
        return next((k for k in _POR_G if k in n), None)

    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g\s+de\s+(.+)$", str(s).lower())
        k = self._fila(s)
        if not (m and k):
            return None
        g = float(m.group(1))
        kc, p, c, f = _POR_G[k]
        return {"grams": g, "kcal": kc * g, "protein": p * g, "carbs": c * g, "fats": f * g}

    def grams_from_ingredient_string(self, s):
        mac = self.macros_from_ingredient_string(s)
        return mac["grams"] if mac else None

    def category_of(self, s):
        k = self._fila(s)
        return _CAT.get(k) if k else None

    def lookup(self, s):
        return None


def _mango(meals):
    return next(x for m in meals for x in m["ingredients"] if "mango" in x)


def _dia():
    return [{"name": "Tostadas integrales con queso fresco, mango y yogurt griego entero",
             "ingredients": ["60 g de mango", "30 g de queso blanco"], "ingredients_raw": ["60 g de mango", "30 g de queso blanco"]}]


def test_el_mango_que_da_nombre_no_baja_de_su_piso():
    meals = _dia()
    # el día se pasa de carbohidratos: la única palanca de carbohidrato es el mango
    ps.refine_day_portions_integer(meals, {"kcal": 100, "protein": 5.9, "carbs": 3.0, "fats": 6.1}, _DB())
    g = float(re.match(r"(\d+)", _mango(meals)).group(1))
    assert g >= 60, _mango(meals)


def test_lo_que_el_nombre_no_nombra_sigue_bajando():
    meals = [{"name": "Queso fresco a la plancha", "ingredients": ["60 g de pan integral", "30 g de queso blanco"],
              "ingredients_raw": ["60 g de pan integral", "30 g de queso blanco"]}]
    # el objetivo es exactamente el plato con 30 g de pan: la palanca es el pan, que el nombre no nombra
    ps.refine_day_portions_integer(meals, {"kcal": 154.2, "protein": 9.3, "carbs": 13.2, "fats": 7.05}, _DB())
    g = float(re.match(r"(\d+)", meals[0]["ingredients"][0]).group(1))
    assert g < 60, meals[0]["ingredients"]
