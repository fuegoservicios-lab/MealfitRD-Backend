# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-352 · 2026-09-26] «maní tostado SIN SAL» no es una dosis de sal.

Plan DM2 real: «0.01 g de maní tostado sin sal» en la lista y «mide 0.01 g de maní tostado sin sal» en el paso — el
quitador de trazas lo tomaba por una dosis de sal (la palabra «sal» de la negación). Si el NOMBRE del plato promete ese
alimento, la traza se queda y se escribe «1 pizca» (quitarla dejaba «Casabe… con maní» sin maní en la lista)."""
from __future__ import annotations

import pathlib

import pasos_cantidades as pc
import pulido_lineas as pl

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _comida(nombre="Casabe crujiente con queso blanco fresco, maní y huevo"):
    ings = ["½ porción de casabe (16 g)", "5 g de queso blanco fresco", "0.01 g de maní tostado sin sal", "1 clara de huevo"]
    return {"name": nombre, "ingredients": list(ings),
            "ingredients_raw": ["0.5 porción de casabe (16.12 g)", "5 g de queso blanco fresco",
                                "0.01 g de maní tostado sin sal", "1.2 claras de huevo"],
            "recipe": ["Mise en place: corta 5 g de queso blanco fresco y mide 0.01 g de maní tostado sin sal."]}


def test_la_traza_de_mani_sin_sal_sale_de_la_lista():
    m = _comida(nombre="Casabe crujiente con queso blanco fresco y huevo")
    assert pc.quitar_trazas(m) == 1
    assert not any("maní" in x for x in m["ingredients"]), m["ingredients"]
    assert not any("maní" in x for x in m["ingredients_raw"]), m["ingredients_raw"]


def test_la_traza_que_el_nombre_promete_se_queda_como_pizca():
    m = _comida()
    assert pc.quitar_trazas(m) == 0 and "0.01 g de maní tostado sin sal" in m["ingredients"]
    pl.pulir_plan({"days": [{"day": 1, "meals": [m]}]})
    assert "1 pizca de maní tostado sin sal" in m["ingredients"], m["ingredients"]
    assert "mide 1 pizca de maní tostado sin sal" in m["recipe"][0], m["recipe"][0]                   # lote 358


def test_la_sal_de_verdad_sigue_siendo_dosis():
    m = {"ingredients": ["100 g de pechuga de pollo", "0.3 g de sal", "50 g de arroz", "0.2 g de ajo en polvo"]}
    assert pc.quitar_trazas(m) == 0 and len(m["ingredients"]) == 4


def test_el_pulido_de_la_migaja():
    assert pl.pulir_linea("0.4 g de galletas de soda sin sal") == "0.4 g de galletas de soda sin sal"  # «sin sal» no es sal
    assert pl.pulir_linea("0.4 g de sal") == "1 pizca de sal"
    assert pl.pulir_linea("0.4 g de semillas de girasol sin sal") == "1 pizca de semillas de girasol sin sal"
    assert pl.pulir_linea("0.4 g de almendras tostadas") == "1 pizca de almendras tostadas"


def test_ancla():
    assert "# [P1-PLAN-LOTE-352]" in (_BACKEND / "pasos_cantidades.py").read_text(encoding="utf-8")
