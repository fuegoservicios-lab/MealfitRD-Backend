# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-353 · 2026-09-26] Cucharadas y cucharaditas SIEMPRE a la cuadrícula de la cuchara (¼ ½ ¾).

Plan bariátrico real (batería sobre el 331): «calienta 1.82 cdtas de aceite de oliva» con «1¾ cdtas» en la lista; otro
bariátrico: «pesa 1.56 cdas de queso ricotta», «0.14 cdta de orégano dominicano», «0.12 cdta de vinagre blanco»."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_cucharas_a_la_cuadricula_en_la_lista_y_en_los_pasos():
    m = {"ingredients": ["1.82 cdtas de aceite de oliva", "0.14 cdta de orégano dominicano", "1.56 cdas de queso ricotta",
                         "0.12 cdta de vinagre blanco", "1.1 cdtas de miel"],
         "ingredients_raw": ["1.82 cdtas de aceite de oliva"],
         "recipe": ["El Toque de Fuego: calienta 1.82 cdtas de aceite de oliva y añade 0.14 cdta de orégano.",
                    "Mise en place: pesa 1.56 cdas de queso ricotta."]}
    pc.decimales_de_cocina(m)
    assert m["ingredients"] == ["1¾ cdtas de aceite de oliva", "¼ cdta de orégano dominicano", "1½ cdas de queso ricotta",
                                "¼ cdta de vinagre blanco", "1 cdta de miel"], m["ingredients"]
    assert "calienta 1¾ cdtas de aceite de oliva y añade ¼ cdta de orégano" in m["recipe"][0], m["recipe"][0]
    assert "pesa 1½ cdas de queso ricotta" in m["recipe"][1], m["recipe"][1]
    assert m["ingredients_raw"] == ["1.82 cdtas de aceite de oliva"]          # lo que mide el motor no se toca


def test_la_taza_sigue_con_su_tolerancia():
    m = {"ingredients": ["0.4 taza de avena"], "recipe": ["Mide 0.4 taza de avena."]}
    assert pc.decimales_de_cocina(m) == 0 and m["ingredients"] == ["0.4 taza de avena"]


def test_ancla():
    assert "# [P1-PLAN-LOTE-353]" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
