# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-318 · 2026-09-25] La avena se reconoce cocida por la FRASE, no por 60 caracteres.

Batería de cierre (perfil del dueño), 25-sep: «en un tazón apto para microondas mezcla la avena con la leche y cocina 2-3
minutos a potencia alta, removiendo a mitad de tiempo, hasta que espese» con «35 g de avena» y «5 ml de leche». El verbo
quedaba a más de 60 caracteres de la avena y el lote 311 no completaba el líquido."""
from __future__ import annotations

import copy
import pathlib

import avena_liquido as al

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _plato(fuego):
    ings = ["35 g de avena en hojuelas", "5 ml de leche", "½ guineo maduro mediano (57 g)", "15 g de maní picado",
            "120 g de queso cottage"]
    return {"name": "Avena tibia con guineo y maní", "meal": "Desayuno", "ingredients": list(ings),
            "ingredients_raw": list(ings),
            "recipe": ["Mise en place: mide 35 g de avena en hojuelas y 5 ml de leche; corta 1 guineo maduro en rodajas.",
                       fuego, "Montaje: sirve la avena tibia y corona con el guineo y el maní."]}


def test_la_frase_que_cocina_la_avena_lejos_del_verbo():
    m = _plato("El Toque de Fuego: en un tazón apto para microondas mezcla la avena con la leche y cocina 2-3 minutos a "
               "potencia alta, removiendo a mitad de tiempo, hasta que espese y quede cremosa.")
    assert al.completar(m) == 140                          # 4 × 35 − 5 = 135 → 140
    assert "mide 35 g de avena en hojuelas y 5 ml de leche y 140 ml de agua;" in m["recipe"][0], m["recipe"][0]


def test_el_huevo_batido_no_vuelve_batido_el_plato():
    # batería de cierre, perfil con warfarina: «incorpora el huevo batido poco a poco» excluía la avena como si fuera un batido
    ings = ["30 g de avena", "¼ taza de leche", "60 g de huevo", "15 g de merey"]
    m = {"name": "Avena cremosa con huevo, aguacate y merey", "ingredients": list(ings), "ingredients_raw": list(ings),
         "recipe": ["Mise en place: mide 30 g de avena y ¼ taza de leche; bate 1 huevo.",
                    "El Toque de Fuego: cocina la avena con la leche a fuego medio durante 6-8 min, removiendo; incorpora el "
                    "huevo batido poco a poco y cocina 2-3 min más."]}
    assert al.completar(m) == 60                           # 4 × 30 − ¼ taza (60 ml)
    assert "mide 30 g de avena y ¼ taza de leche y 60 ml de agua;" in m["recipe"][0], m["recipe"][0]
    batido = dict(m, name="Batido de avena, guineo y leche")
    batido["ingredients"] = list(ings)
    assert al.completar(batido) == 0, "el batido (por su nombre) sigue fuera"


def test_la_avena_fria_o_cruda_no_se_cocina():
    for fuego in ("Montaje: sirve la avena fría con la leche; aparte, calienta el pan en la sartén 2 minutos.",
                  "Mezcla la avena cruda con el yogurt y cocina el guineo en la sartén 3 minutos."):
        m = _plato(fuego)
        antes = copy.deepcopy(m)
        assert al.completar(m) == 0 and m == antes, fuego


def test_ancla():
    assert "tooltip-anchor: P1-PLAN-LOTE-318" in (_BACKEND / "avena_liquido.py").read_text(encoding="utf-8")
