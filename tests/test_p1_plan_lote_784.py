# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-784 · 2026-09-28] «Marina el pollo cocido» con el pollo CRUDO en la lista trae su cocción previa.

Corpus de la cola 744, perfil sin tiempo: «Montaje: marina pechuga de pollo cocido con el jugo de limón, el palmito…»
con «1½ pechugas de pollo (≈296 g)» en la lista y ningún paso que las cocine (el único fuego era la yautía), y un ceviche
que «marina el pescado blanco cocido» con los filetes crudos. El lote 407 ya añade la «💡 Cocción previa» a la proteína
que el paso usa cocida y la lista compra cruda, pero su lista de verbos no tenía «marina» ni «acompaña».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pasos_cantidades as pc  # noqa: E402


def test_marina_el_pollo_cocido_crudo_en_la_lista():
    m = {"name": "Yautía al microondas con pechuga de pollo frío en limón y palmito",
         "ingredients": ["½ pedazo de yautía (≈88 g)", "1½ pechugas de pollo (≈296 g)", "55 g de palmito escurrido"],
         "recipe": ["Mise en place: pela la yautía; corta 1½ pechugas de pollo en tiras; mide el jugo de limón.",
                    "El Toque de Fuego: cocina las rodajas de yautía en el microondas 5-6 minutos.",
                    "Montaje: marina pechuga de pollo cocido con el jugo de limón y el palmito; sirve la yautía al lado."]}
    assert pc.proteina_cocida_de_la_lista(m) == 1
    assert m["recipe"][1].startswith("💡 Cocción previa: cocina la pechuga de pollo") and "74 °C" in m["recipe"][1]


def test_ceviche_de_pescado_cocido():
    m = {"name": "Ceviche de pescado blanco con palmito",
         "ingredients": ["2 filetes de pescado (≈300 g)", "30 g de palmito"],
         "recipe": ["Mise en place: corta 2 filetes de pescado en cubos.",
                    "Montaje: marina el pescado blanco cocido con el jugo de limón y el palmito."]}
    assert pc.proteina_cocida_de_la_lista(m) == 1
    assert "63 °C" in m["recipe"][1]


def test_acompana_con_pollo_cocido_crudo_en_la_lista():
    m = {"name": "Arroz con vegetales", "ingredients": ["½ taza de arroz", "¼ pechuga de pollo (≈80 g)"],
         "recipe": ["Mise en place: lava el arroz.", "El Toque de Fuego: hierve el arroz 18 min.",
                    "Montaje: sirve el arroz. Acompaña con pechuga de pollo cocida y desmenuzada."]}
    assert pc.proteina_cocida_de_la_lista(m) == 1


def test_si_un_paso_lo_cocina_no_se_toca():
    m = {"name": "Pollo marinado", "ingredients": ["1 pechuga de pollo (≈200 g)"],
         "recipe": ["El Toque de Fuego: cocina la pechuga de pollo a la plancha 6 min por lado.",
                    "Montaje: marina el pollo cocido con limón y sirve."]}
    assert pc.proteina_cocida_de_la_lista(m) == 0
