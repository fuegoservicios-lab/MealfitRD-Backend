# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-560 · 2026-09-27] La tayota de una ensalada fresca no se hierve.

Batería real (rechaza pescado y berenjena, «Nada» de tiempo): el lote 540 añadía «💡 Cocción previa: hierve la tayota
10-12 minutos» a «Pollo cítrico a la plancha con maíz y ensalada fresca de tayota», y el plato pasaba de 10 a 20 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import verdura_sin_coccion as vsc  # noqa: E402


def _plato(nombre, montaje):
    return {"name": nombre,
            "ingredients": ["1½ pechugas de pollo (≈300 g)", "½ tayota pequeña", "½ tomate"],
            "recipe": ["Mise en place: corta la pechuga en filetes; corta la tayota y el tomate en láminas finas.",
                       "El Toque de Fuego: cocina la pechuga 3-4 min por lado, hasta 74 °C en la parte más gruesa.",
                       montaje]}


def test_ensalada_fresca_de_tayota_no_se_hierve():
    m = _plato("Pollo cítrico a la plancha con maíz y ensalada fresca de tayota",
               "Montaje: combina la tayota y el tomate en una ensalada; sírvela junto al pollo.")
    assert vsc.cocer(m) == 0, m["recipe"]


def test_la_ensalada_la_dicen_los_pasos():
    m = _plato("Pollo cítrico a la plancha", "Montaje: combina la tayota y el tomate en una ensalada fresca.")
    assert vsc.cocer(m) == 0, m["recipe"]


def test_la_tayota_de_un_plato_caliente_sigue_recibiendo_su_coccion():
    m = _plato("Pollo con tayota", "Montaje: sirve el pollo con la tayota y el tomate.")
    assert vsc.cocer(m) == 1
    assert any("hierve la tayota" in p for p in m["recipe"])


def test_las_vainitas_no_se_comen_crudas_ni_en_ensalada():
    m = {"name": "Pollo con ensalada de vainitas",
         "ingredients": ["150 g de pechuga de pollo", "100 g de vainitas"],
         "recipe": ["Mise en place: corta las vainitas.", "El Toque de Fuego: cocina el pollo 6-8 min hasta 74 °C.",
                    "Montaje: combina las vainitas en una ensalada."]}
    assert vsc.cocer(m) == 1
