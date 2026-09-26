# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-354 · 2026-09-26] Los conteos con decimales de máquina se escriben como en la cocina.

Plan bariátrico real (porciones escaladas ×0,27): «0.27 pepino», «0.27 diente de ajo», «0.27 limón» en la lista y «Pica
0.27 diente de ajo, 0.27 de ají cubanela y 0.27 de cebolla… jugo de 0.27 limón» en los pasos."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_lista_y_pasos_a_fraccion_de_cocina():
    lista = ["½ pechuga de pollo (≈83 g)", "0.27 pepino", "0.27 tomate", "0.27 diente de ajo", "0.27 ají cubanela",
             "0.27 limón", "0.27 cebolla"]
    m = {"ingredients": list(lista), "ingredients_raw": list(lista),
         "recipe": ["Mise en place: Pica 0.27 diente de ajo, 0.27 de ají cubanela y 0.27 de cebolla; corta 0.27 tomate en "
                    "cubos y 0.27 pepino en rodajas.",
                    "Montaje: aliña con el jugo de 0.27 limón.",
                    "⚠️ Nota: 0.27 limón."]}
    pc.decimales_de_cocina(m)
    assert m["ingredients"][1:] == ["¼ pepino", "¼ tomate", "¼ diente de ajo", "¼ ají cubanela", "¼ limón", "¼ cebolla"]
    assert m["recipe"][0] == ("Mise en place: Pica ¼ diente de ajo, ¼ de ají cubanela y ¼ de cebolla; corta ¼ tomate en "
                              "cubos y ¼ pepino en rodajas."), m["recipe"][0]
    assert "jugo de ¼ limón" in m["recipe"][1]
    assert m["recipe"][2] == "⚠️ Nota: 0.27 limón."                       # las notas no se tocan
    assert m["ingredients_raw"] == lista                                  # lo que mide el motor no se toca


def test_el_paso_sigue_a_la_lista_limpia():
    m = {"ingredients": ["½ cebolla"], "recipe": ["Mise en place: pica 0.62 cebolla."]}
    pc.decimales_de_cocina(m)
    assert m["recipe"][0] == "Mise en place: pica ½ cebolla.", m["recipe"][0]


def test_lo_que_no_se_toca():
    pasos = ["Corta 2.5 cm de jengibre, hierve 1.5 min, bate 1.5 huevos y usa 0.4 pepino."]
    m = {"ingredients": ["0.4 pepino", "1.5 huevos"], "recipe": list(pasos)}
    assert pc.decimales_de_cocina(m) == 0 and m["recipe"] == pasos and m["ingredients"] == ["0.4 pepino", "1.5 huevos"]


def test_ancla():
    assert "n += conteos_de_cocina(meal)" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
