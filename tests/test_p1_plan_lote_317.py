# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-317 · 2026-09-25] La avena que el paso ya cocina «con agua» recibe ahí su medida.

Batería renal del 25-sep, con el lote 311 ya puesto: «en una olla pequeña, cocina la avena con agua a fuego medio durante
5-7 minutos, removiendo hasta que esté suave; deja que se enfríe. Completa el líquido con 200 ml de agua para que la avena
se cocine.» — la medida llegaba al final del paso, después de «deja que se enfríe». Ahora: «cocina la avena con 200 ml de
agua a fuego medio…»."""
from __future__ import annotations

import pathlib

import avena_liquido as al

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _bowl(fuego):
    ings = ["½ taza de avena", "10 g de yogurt griego sin azúcar", "115 g de fresas", "30 g de semillas de chía"]
    return {"name": "Bowl fresco de avena, fresas y mandarina", "meal": "Desayuno", "ingredients": list(ings),
            "ingredients_raw": list(ings),
            "recipe": ["Mise en place: lava y corta 115 g de fresas; mide ½ taza de avena y 30 g de semillas de chía.", fuego,
                       "Montaje: coloca la avena en un bowl con el yogurt y las fresas."]}


def test_el_agua_que_el_paso_ya_nombra_recibe_su_medida():
    m = _bowl("El Toque de Fuego: en una olla pequeña, cocina la avena con agua a fuego medio durante 5-7 minutos, "
              "removiendo hasta que esté suave; deja que se enfríe.")
    assert al.completar(m) == 170                         # ½ taza = 42,5 g → 4 × 42,5 = 170
    assert m["recipe"][1] == ("El Toque de Fuego: en una olla pequeña, cocina la avena con 170 ml de agua a fuego medio "
                              "durante 5-7 minutos, removiendo hasta que esté suave; deja que se enfríe."), m["recipe"][1]
    assert "Completa el líquido" not in " ".join(m["recipe"])


def test_el_agua_de_otro_alimento_no_se_toca():
    m = _bowl("El Toque de Fuego: cocina la avena en el microondas 2 min, removiendo hasta que espese; aparte, hierve el "
              "huevo en agua 10 minutos.")
    al.completar(m)
    assert "hierve el huevo en agua 10 minutos." in m["recipe"][1], m["recipe"][1]
    assert m["recipe"][1].endswith("Completa el líquido con 170 ml de agua para que la avena se cocine."), m["recipe"][1]


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-317" in (_BACKEND / "avena_liquido.py").read_text(encoding="utf-8")
