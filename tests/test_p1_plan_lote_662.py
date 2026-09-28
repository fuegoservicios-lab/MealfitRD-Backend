# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-662 · 2026-09-28] El tiempo que falta va en la frase de su técnica, no al final del paso.

Batería real del 28-sep (código 636, estudiante): «…Incorpora las claras de huevo y 3 huevos, removiendo suavemente hasta
que cuajen (~18-20 min a 180 °C).» — el tiempo de horno de «asa las rodajas de plátano… o en el horno» colgado de unos
huevos en sartén. Corpus: 11 de 30 pasos con el tiempo inyectado cambian de frase, todos a la suya.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _meal(paso, name="Guiso criollo de huevos con plátano maduro asado"):
    return {"name": name, "recipe": ["Mise en place: corta.", paso, "Montaje: sirve."]}


@pytest.mark.parametrize("paso, esperado", [
    ("El Toque de Fuego: asa las rodajas de plátano maduro en una sartén antiadherente o en el horno hasta que estén "
     "tiernas y doradas. En otra sartén, calienta el aceite y sofríe la cebolla. Incorpora las claras y los huevos, "
     "removiendo suavemente hasta que cuajen.",
     "El Toque de Fuego: asa las rodajas de plátano maduro en una sartén antiadherente o en el horno hasta que estén "
     "tiernas y doradas (~18-20 min a 180 °C). En otra sartén, calienta el aceite y sofríe la cebolla. Incorpora las "
     "claras y los huevos, removiendo suavemente hasta que cuajen."),
    ("El Toque de Fuego: cuece la pasta integral en agua según el tiempo indicado en el paquete. Escúrrela y deja que se "
     "enfríe.",
     "El Toque de Fuego: cuece la pasta integral en agua según el tiempo indicado en el paquete (~12-15 min en agua "
     "hirviendo). Escúrrela y deja que se enfríe."),
])
def test_el_tiempo_va_en_su_frase(paso, esperado):
    m = _meal(paso, name="Pasta fría" if "pasta" in paso else "Guiso criollo de huevos con plátano maduro asado")
    assert go._inject_recipe_time_temp_defaults(m) is True
    assert m["recipe"][1] == esperado


def test_sin_frase_de_la_tecnica_queda_al_final_como_antes():
    m = _meal("El Toque de Fuego: calienta las 2 tortillas integrales en una sartén limpia.", name="Wrap de pollo")
    go._inject_recipe_time_temp_defaults(m)
    assert m["recipe"][1].endswith("limpia (~3-4 min por lado a fuego medio-alto).")
