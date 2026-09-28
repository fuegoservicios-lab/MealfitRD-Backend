# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-634 · 2026-09-28] Un paso no termina en «..».

Validación del 592 (estudiante, día 3): «…toma agua con la comida. Acompaña con pechuga de pollo..» — 6 de 4.231
comidas recientes, ya presente antes del shield.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import doble_punto as dp  # noqa: E402


@pytest.mark.parametrize("paso, esperado", [
    ("Montaje: sirve las lentejas y el brócoli al lado; toma agua con la comida. Acompaña con pechuga de pollo..",
     "Montaje: sirve las lentejas y el brócoli al lado; toma agua con la comida. Acompaña con pechuga de pollo."),
    ("Montaje: corona con la zanahoria rallada. Acompaña con agua..", "Montaje: corona con la zanahoria rallada. Acompaña con agua."),
    ("Montaje: sirve el guiso. . Acompaña con sardinas en lata.", "Montaje: sirve el guiso. Acompaña con sardinas en lata."),
])
def test_el_punto_doble_queda_en_uno(paso, esperado):
    m = {"name": "x", "recipe": ["Mise en place: corta.", paso]}
    assert dp.limpiar(m) == 1 and m["recipe"][1] == esperado
    assert dp.limpiar(m) == 0                                   # idempotente


def test_los_puntos_suspensivos_no_se_tocan():
    m = {"name": "x", "recipe": ["Montaje: sirve y listo... a disfrutar.", "Montaje: 1 cda. de miel."]}
    antes = list(m["recipe"])
    assert dp.limpiar(m) == 0 and m["recipe"] == antes


def test_es_lo_ultimo_que_toca_los_pasos_en_el_contrato():
    """Tiene que ir DETRÁS de toda pasada que pega frases («Acompaña con X.»): si no, el punto doble vuelve."""
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def _aplicar_meal("):]
    cuerpo = cuerpo[:cuerpo.index('if r.get("lista_reescrita"):')]
    llamadas = re.findall(r'__import__\("([a-z_]+)"\)\.([a-z_]+)\(meal', cuerpo)
    assert llamadas[-1] == ("doble_punto", "limpiar"), llamadas[-3:]
