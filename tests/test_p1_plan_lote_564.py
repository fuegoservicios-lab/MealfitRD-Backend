# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-564 · 2026-09-27] El nombre del plato no repite un alimento.

Batería real (perfil tipo dueño): «Pescado blanco a la plancha con nabo salteado al limón, casabe y pescado blanco»;
corpus: «Wrap dominicano de queso blanco, nabo con aguacate y queso blanco y queso blanco».
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import nombre_sin_repetidos as nsr  # noqa: E402


@pytest.mark.parametrize("antes, despues", [
    ("Pescado blanco a la plancha con nabo salteado al limón, casabe y pescado blanco",
     "Pescado blanco a la plancha con nabo salteado al limón y casabe"),
    ("Wrap dominicano de queso blanco, nabo con aguacate y queso blanco y queso blanco",
     "Wrap dominicano de queso blanco y nabo con aguacate"),
    ("Yautía Majada Caliente con Queso Blanco, Queso Blanco, Vegetales Salteados al Limón y Edamame",
     "Yautía Majada Caliente con Queso Blanco, Vegetales Salteados al Limón y Edamame"),
    ("Tostadas integrales con queso blanco fresco, aguacate, queso blanco cuajado y queso blanco",
     "Tostadas integrales con queso blanco fresco y aguacate"),
])
def test_corpus(antes, despues):
    assert nsr.nombre(antes) == despues


@pytest.mark.parametrize("nombre", [
    "Pan integral con queso blanco y queso crema",
    "Mangú de plátano verde con huevo, cebolla y aguacate",
    "Arroz con habichuelas rojas y pollo guisado",
])
def test_lo_distinto_se_queda(nombre):
    assert nsr.nombre(nombre) is None


def test_ancla_en_la_cola():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("nombre_sin_repetidos").limpiar(meal)  # [P1-PLAN-LOTE-564]' in src
