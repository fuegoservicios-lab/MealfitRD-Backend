# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-498 · 2026-09-27] La frase de guiso del cerrador es para la proteína.

«Añade X al guiso y cocínalo a fuego medio 12-15 minutos, hasta que esté cocido por dentro; incorpóralo con cuidado para
no deshacer el resto» salía para arroz crudo, repollo, auyama, tayota o agua (re-encadenado del corpus y batería real del
27-sep, alérgico al pescado: «Añade arroz blanco crudo al guiso y cocínala… cocida por dentro»), y tras la compra única
para las claras. Y la otra plantilla, «Cocina claras a la plancha o hervida», no se leía.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cerrador as pc  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402

_COLA = " al guiso y cocínala a fuego medio 12-15 minutos, hasta que esté cocida por dentro; incorpórala con cuidado para " \
        "no deshacer el resto."


def _paso(obj):
    return {"recipe": [f"El Toque de Fuego: calienta el aceite; cocina la cebolla 3 minutos. Añade {obj}{_COLA}"]}


def test_el_arroz_crudo_se_cuece_con_su_agua():
    m = _paso("arroz blanco crudo")
    assert pc.guiso_sin_proteina(m) == 1
    assert m["recipe"][0].endswith("Añade el arroz blanco al guiso con el doble de su volumen en agua, tapa y cocina a "
                                   "fuego bajo 18-20 minutos, hasta que esté tierno y el agua se absorba."), m["recipe"][0]


def test_la_verdura_y_el_viver_hasta_que_esten_tiernos():
    m = _paso("repollo rallado")
    pc.guiso_sin_proteina(m)
    assert "Añade el repollo rallado al guiso y cocínalo a fuego medio 8-10 minutos, hasta que esté tierno." in \
        m["recipe"][0], m["recipe"][0]
    m = _paso("yautía")
    pc.guiso_sin_proteina(m)
    assert "Añade la yautía al guiso y cocínala a fuego medio 20-25 minutos, hasta que esté tierna." in m["recipe"][0], \
        m["recipe"][0]


def test_el_agua_hierve_y_las_claras_cuajan():
    m = _paso("agua o caldo bajo en sodio")
    pc.guiso_sin_proteina(m)
    assert "Añade agua o caldo bajo en sodio al guiso y deja que hierva a fuego medio 12-15 minutos." in m["recipe"][0]
    m = _paso("claras")
    pc.guiso_sin_proteina(m)
    assert "Añade las claras batidas al guiso y remueve 2-3 minutos, hasta que cuajen." in m["recipe"][0], m["recipe"][0]


def test_la_proteina_se_queda():
    m = _paso("pechuga de pollo")
    antes = list(m["recipe"])
    assert pc.guiso_sin_proteina(m) == 0 and m["recipe"] == antes


def test_claras_no_a_la_plancha_o_hervida():
    m = {"name": "Ensalada templada con pollo", "ingredients": ["1¼ pechugas de pollo (≈288 g)"],
         "ingredients_raw": ["1¼ pechugas de pollo (≈288 g)"],
         "recipe": ["El Toque de Fuego: templa el repollo 3-4 min. Cocina pechuga de pollo a la plancha o hervida y "
                    "sírvela como proteína del plato."]}
    sf.sustituir_en_plato(m, 0, "1¼ pechugas de pollo (≈288 g)", "6 claras de huevo", "claras de huevo")
    assert "a la plancha" not in m["recipe"][0] and "hervid" not in m["recipe"][0], m["recipe"][0]
    assert "Cocina claras, hasta que cuajen y sírvelas como proteína del plato." in m["recipe"][0], m["recipe"][0]
