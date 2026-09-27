# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-497 · 2026-09-27] Batería REAL del perfil del dueño, días 12-15 de 30 sin congelador (código 460-496).

1. La IA pensó el pescado enlatado («escurre 2 latas de filete de pescado blanco») y la sustitución dejó «escurre 2 latas
   de sardinas» con «80 g de sardinas en lata» en la lista.
2. La IA escribió «extrae las semillas de 1 guineo… añade las semillas de guineo por encima»; con guineo → manzana quedó
   «añade las semillas de manzana»: servir pepitas de manzana.
3. El bok choy (≈10 días) seguía en la cena del día 15: ningún grupo de la tabla lo cubría.
4. «Wrap fresco de yogurt, lechuga y tomate con casabe crujiente» recibía «rellena la casabe con lo que cierra…»: el
   casabe de guarnición no envuelve nada; el wrap es la tortilla.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402
import dish_structure as ds  # noqa: E402
import sustitucion_fresca as sf  # noqa: E402


def test_las_latas_del_paso_son_la_medida_de_la_lista():
    m = {"name": "Pescado blanco guisado rápido con papa y zanahoria sobre casabe",
         "ingredients": ["¾ filete de pescado (≈105 g)"], "ingredients_raw": ["105 g de filete de pescado blanco"],
         "recipe": ["Mise en place: escurre 2 latas de filete de pescado blanco; pela y corta 1 papa mediana."]}
    sf.sustituir_en_plato(m, 0, "¾ filete de pescado (≈105 g)", "105 g de sardinas en lata", "sardinas en lata")
    assert m["recipe"][0] == "Mise en place: escurre 105 g de sardinas en lata; pela y corta 1 papa mediana.", \
        m["recipe"][0]


def test_no_se_sirven_las_semillas_de_la_manzana():
    m = {"name": "Yogurt con guineo, maní y queso cottage", "ingredients": ["1 guineo"], "ingredients_raw": ["1 guineo"],
         "recipe": ["Mise en place: mide 15 g de yogurt natural, extrae las semillas de 1 guineo y pesa 10 g de maní.",
                    "Montaje: sirve el yogurt en un tazón y añade las semillas de guineo y el maní por encima."]}
    sf.sustituir_en_plato(m, 0, "1 guineo", "90 g de manzana", "manzana")
    assert "semillas" not in m["recipe"][1], m["recipe"][1]
    assert m["recipe"][1] == "Montaje: sirve el yogurt en un tazón y añade la manzana y el maní por encima.", m["recipe"]
    assert "corta 90 g de manzana en cubos, sin semillas" in m["recipe"][0], m["recipe"][0]
    assert sf._manzana_sin_semillas("retira las semillas de la manzana") == "retira las semillas de la manzana"


def test_el_bok_choy_no_llega_al_dia_15():
    r = cu.sustituir_linea("¾ taza de bok choy picado", 14, {"need_days": 15, "allow_frozen": False})
    assert r and r[1] == "repollo" and r[0].startswith("¾ taza de repollo"), r


def test_el_wrap_es_la_tortilla_aunque_el_nombre_diga_casabe():
    nombre = ds._norm("Wrap fresco de yogurt, lechuga y tomate con casabe crujiente y queso cottage")
    assert not ds._soporte_del_nombre_378(nombre, "casabe")
    assert ds._soporte_del_nombre_378(nombre, "tortilla de trigo")
    assert ds._soporte_del_nombre_378(ds._norm("Casabe con queso"), "casabe"), "sin wrap, el pan del nombre vale"
    assert ds._soporte_del_nombre_378(ds._norm("Wrap criollo de casabe con queso fresco"), "casabe"), \
        "«Wrap … de casabe» nombra su vasija"
