# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-732 · 2026-09-28] El casabe también se cuenta por pieza.

Batería real sobre el 659 (adulto mayor con HTA, día 3, merienda): «mide 1 casabe pequeño sin sal» con «½ casabe
pequeño sin sal (15 g)» en la lista.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def _merienda():
    return {
        "name": "Casabe crujiente con mantequilla de maní, canela, guineo y yogurt griego entero",
        "ingredients": ["½ casabe pequeño sin sal (15 g)", "1 cdta de mantequilla de maní natural sin sal (4 g)",
                        "60 g de guineo", "½ cdta de canela en polvo", "¾ taza de yogurt griego entero"],
        "recipe": [
            "Mise en place: corta 60 g de guineo en cubos y mide 1 casabe pequeño sin sal, 1 cdta de mantequilla de maní "
            "natural sin sal y ½ cdta de canela en polvo.",
            "El Toque de Fuego: tuesta el casabe en sartén antiadherente seco a fuego medio 1-2 minutos por lado, hasta "
            "que esté crujiente.",
            "Montaje: unta la mantequilla de maní sobre el casabe, espolvorea la canela y corona con los cubos de guineo.",
        ],
    }


def test_el_casabe_del_paso_es_el_de_la_lista():
    m = _merienda()
    assert pc.conteos_de_la_lista(m) == 1
    assert m["recipe"][0] == ("Mise en place: corta 60 g de guineo en cubos y mide ½ casabe pequeño sin sal, 1 cdta de "
                              "mantequilla de maní natural sin sal y ½ cdta de canela en polvo.")
    assert m["recipe"][1:] == _merienda()["recipe"][1:]


def test_dos_casabes_en_plural():
    m = _merienda()
    m["ingredients"][0] = "2 casabes pequeños sin sal (60 g)"
    assert pc.conteos_de_la_lista(m) == 1
    assert "mide 2 casabes pequeños sin sal," in m["recipe"][0]


def test_la_misma_cuenta_concuerda_nombre_y_adjetivo():
    # replay de la cola: «mide 1½ casabe tostado (45 g)» con «1½ casabes tostados (45 g)» salía «1½ casabes tostado»
    m = {"name": "Casabe tostado con queso", "ingredients": ["1½ casabes tostados (45 g)", "50 g de queso blanco"],
         "recipe": ["Mise en place: mide 1½ casabe tostado (45 g), corta el queso blanco en cubos."]}
    assert pc.conteos_de_la_lista(m) == 1
    assert m["recipe"][0] == "Mise en place: mide 1½ casabes tostados (45 g), corta el queso blanco en cubos."


def test_con_la_misma_cuenta_no_se_toca():
    m = _merienda()
    m["recipe"][0] = m["recipe"][0].replace("mide 1 casabe", "mide ½ casabe")
    antes = list(m["recipe"])
    assert pc.conteos_de_la_lista(m) == 0 and m["recipe"] == antes


def test_en_gramos_en_la_lista_no_se_toca():
    m = _merienda()
    m["ingredients"][0] = "30 g de casabe"
    antes = list(m["recipe"])
    assert pc.conteos_de_la_lista(m) == 0 and m["recipe"] == antes
