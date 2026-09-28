# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-734 · 2026-09-28] «queso fresco batido» no hace de un bowl un batido.

Batería real sobre el 659: «Bowl fresco de mango, lechosa y queso fresco batido con yogurt griego entero» salió con
«coloca todos los ingredientes en la licuadora y licúa…» y un Montaje que acomoda el mango por separado.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import batido_nombre as bn  # noqa: E402
import graph_orchestrator as g  # noqa: E402


@pytest.mark.parametrize("nombre", [
    "Bowl fresco de mango, lechosa y queso fresco batido con yogurt griego entero",
    "Tostada integral con queso fresco batido, piña y yogurt griego entero",
    "Queso Ricotta Batido con Canela",
    "Chinola fresca con queso blanco batido y semillas de maní",
    "Yogurt natural batido con Guineo, Maní tostadas y queso cottage",
    "Tortilla de claras de huevo batidas con espinaca",
])
def test_el_adjetivo_no_es_un_batido(nombre):
    from constants import strip_accents
    assert not g._name_suggests_blended(nombre)
    assert not bn.es_batido(strip_accents(nombre.lower()))


@pytest.mark.parametrize("nombre", [
    "Batido de lechosa y avena",
    "Licuado de guineo con leche descremada",
    "Batido verde de espinaca con huevo batido",
    "Smoothie de mango y yogurt griego",
])
def test_el_sustantivo_sigue_siendo_un_batido(nombre):
    assert g._name_suggests_blended(nombre)
    assert bn.es_batido(nombre.lower())


def test_la_conjuncion_no_se_cruza():
    # «huevo al sartén y batida…»: el adjetivo no salta un «al» ni un «y»
    assert bn.sin_adjetivo("tostadas integrales con huevo al sarten y batido de guineo").count("batido") == 1


def test_el_paso_de_licuadora_usa_el_predicado():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("[P1-BLEND-STEP-REQUIRED · 2026-07-25] Un batido que no dice licuar.")
    bloque = src[i:i + 2500]
    assert '__import__("batido_nombre").es_batido(_n_blend)' in bloque
    assert not re.search(r'_re\.search\(r"\\b\(batido\|licuado', bloque)
