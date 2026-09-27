# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-467 · 2026-09-27] Las piezas de fruta, víver y pan también siguen a la lista en el paso.

Batería real del 27-sep (perfil del dueño, días 4-7): la IA escribió «¼ lechosa» en la lista y el paso; el band-closer
subió la lista a «½ lechosa (395g)» y el paso siguió «pela y corta ¼ lechosa». El sincronizador de piezas
(`conteos_de_la_lista`) sólo conocía doce (diente, ají, tomate, limón…). En el replay de 322 planes: 19 comidas con la
pieza del paso distinta de la lista (lechosa 10, aguacate 5, rebanada 2, mango 1). Y «prepara 2 rebanada» con «2
rebanadas» en la lista: la cuenta coincide, el nombre no.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def _plato():
    return {"meal": "Merienda", "name": "Lechosa con queso blanco fresco y pan integral",
            "ingredients": ["30 g de queso blanco", "2 rebanadas de pan integral familiar", "½ lechosa (395g)"],
            "recipe": ["Mise en place: corta 30 g de queso blanco en lonjas, pela y corta ¼ lechosa, y prepara 2 "
                       "rebanada de pan integral familiar.",
                       "Montaje: coloca el queso blanco sobre el pan integral y sirve lechosa al lado."]}


def test_la_lechosa_del_paso_sigue_a_la_lista():
    m = _plato()
    assert pc.conteos_de_la_lista(m) >= 2
    assert m["recipe"][0] == ("Mise en place: corta 30 g de queso blanco en lonjas, pela y corta ½ lechosa, y prepara 2 "
                              "rebanadas de pan integral familiar."), m["recipe"][0]


def test_aguacate_mango_y_rebanada():
    for linea, paso, esperado in (
        ("½ aguacate", "Mise en place: corta ¼ aguacate en láminas.", "Mise en place: corta ½ aguacate en láminas."),
        ("1 mango", "Mise en place: pela ½ mango y córtalo.", "Mise en place: pela 1 mango y córtalo."),
        ("2 rebanadas de pan integral", "Mise en place: tuesta 1 rebanada de pan integral.",
         "Mise en place: tuesta 2 rebanadas de pan integral."),
    ):
        m = {"ingredients": [linea, "1 taza de agua"], "recipe": [paso]}
        pc.conteos_de_la_lista(m)
        assert m["recipe"][0] == esperado, (linea, m["recipe"][0])


def test_un_reparto_no_se_toca():
    m = {"ingredients": ["1 aguacate"], "recipe": ["Mise en place: corta ½ aguacate para el plato y ½ aguacate para la salsa."]}
    antes = list(m["recipe"])
    pc.conteos_de_la_lista(m)
    assert m["recipe"] == antes


def test_el_resultado_de_un_corte_no_es_la_cuenta():
    """Replay normal: «corta el pan integral en 2 rebanadas» con «1 rebanada» en la lista salía «… en 1 rebanada»."""
    m = {"ingredients": ["1 rebanada de pan integral", "35 g de aguacate"],
         "recipe": ["Mise en place: corta el pan integral en 2 rebanadas, machaca los frijoles negros."]}
    antes = list(m["recipe"])
    pc.conteos_de_la_lista(m)
    assert m["recipe"] == antes


def test_la_pista_de_peso_escala_con_la_cuenta():
    """Replay normal: «ten lista 1 rebanada (30 g)» con «2 rebanadas de pan integral (60 g)» salía «2 rebanadas (30 g)»."""
    m = {"ingredients": ["2 rebanadas de pan integral (60 g)", "1½ cdas de queso ricotta"],
         "recipe": ["Mise en place: pesa 1½ cdas de queso ricotta y ten lista 1 rebanada (30 g) de pan integral."]}
    pc.conteos_de_la_lista(m)
    assert m["recipe"][0] == ("Mise en place: pesa 1½ cdas de queso ricotta y ten lista 2 rebanadas (60 g) de pan "
                              "integral."), m["recipe"][0]
