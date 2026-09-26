# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-337 · 2026-09-25] Las piezas contadas del paso son las de la lista.

Baterías re-medidas del 25-sep: «pica 1 diente de ajo» con «3 dientes de ajo» en la lista (22 casos), «pica ½ ají
cubanela» con «1½ ají cubanela», «exprime ½ limón» con «1 limón» — 33 de 384 comidas."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_unica_mencion_con_numero_sigue_a_la_lista():
    m = {"ingredients": ["3 dientes de ajo", "1½ ají cubanela", "1 tomate mediano", "1 cebolla"],
         "recipe": ["Mise en place: pica 1 cebolla, ½ ají cubanela, 2 tomates medianos y 1 diente de ajo.",
                    "El Toque de Fuego: sofríe el ajo y el ají 2 minutos.", "Montaje: sirve."]}
    assert pc.conteos_de_la_lista(m) == 3
    assert m["recipe"][0] == ("Mise en place: pica 1 cebolla, 1½ ajíes cubanela, 1 tomate mediano y 3 dientes de ajo.")


def test_el_articulo_concuerda():
    # replay del 25-sep: «machaca los 2 dientes de ajo» con «1 diente de ajo» en la lista salía «los 1 diente»
    m = {"ingredients": ["1 diente de ajo"], "recipe": ["El Toque de Fuego: machaca los 2 dientes de ajo con sal.",
                                                        "Montaje: sirve."]}
    assert pc.conteos_de_la_lista(m) == 1
    assert m["recipe"][0] == "El Toque de Fuego: machaca el 1 diente de ajo con sal."


def test_dos_menciones_con_numero_o_reparto_no_se_tocan():
    pasos = ["Mise en place: pica 1 tomate mediano; reserva la mitad, 1 diente de ajo, para la salsa.",
             "Montaje: corta 1 tomate pequeño en rodajas para decorar."]
    m = {"ingredients": ["1½ tomates", "3 dientes de ajo"], "recipe": list(pasos)}
    assert pc.conteos_de_la_lista(m) == 0 and m["recipe"] == pasos


def test_ancla():
    src = (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
    assert "+ conteos_de_la_lista(meal)" in src and "tooltip-anchor: P1-PLAN-LOTE-337" in src
