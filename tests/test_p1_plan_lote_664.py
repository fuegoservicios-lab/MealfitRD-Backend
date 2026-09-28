# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-664 · 2026-09-28] Los gramos de un paso no contradicen la línea casera de la lista.

Batería real del 28-sep sobre el 636 (estudiante): «prepara ⅓ taza de yogurt natural sin azúcar (155 g)» con «⅓ taza de
yogurt natural sin azúcar» en la lista (~80 g en los macros) y «corta ¼ plátano maduro (100 g)» con «¼ plátano maduro»
(~70 g). Replay: 154 pasos en 5.099 comidas, todos hacia los gramos del motor.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import pista_de_la_linea as pdl  # noqa: E402

_GRAMOS = {"⅓ taza de yogurt natural sin azúcar": 82.0, "¼ plátano maduro": 70.0, "½ tomate": 75.0}


@pytest.fixture(autouse=True)
def _motor(monkeypatch):
    monkeypatch.setattr(go, "_resolve_line_food_grams", lambda s, cheap=False: ("x", _GRAMOS.get(str(s).strip())))


def test_la_pista_vieja_pasa_a_los_gramos_de_la_linea():
    m = {"name": "Avena cremosa", "ingredients": ["35 g de avena", "⅓ taza de yogurt natural sin azúcar", "¼ plátano maduro"],
         "recipe": ["Mise en place: mide 35 g de avena, prepara ⅓ taza de yogurt natural sin azúcar (155 g) y corta "
                    "¼ plátano maduro (100 g) en tajadas.", "Montaje: sirve."]}
    assert pdl.sincronizar(m) == 1
    assert m["recipe"][0] == ("Mise en place: mide 35 g de avena, prepara ⅓ taza de yogurt natural sin azúcar (≈80 g) y "
                              "corta ¼ plátano maduro (≈70 g) en tajadas.")
    assert pdl.sincronizar(m) == 0                       # idempotente


def test_una_diferencia_pequena_no_se_toca():
    m = {"name": "Guiso", "ingredients": ["½ tomate"], "recipe": ["Mise en place: pica ½ tomate (90 g)."]}
    antes = list(m["recipe"])
    assert pdl.sincronizar(m) == 0 and m["recipe"] == antes     # 90 vs 75: 15 g, dentro del ruido de densidades


def test_corre_tras_los_pesos_de_la_lista_en_el_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    llamadas = re.findall(r'__import__\("([a-z_]+)"\)\.([a-z_]+)\(meal', src)
    i = llamadas.index(("pasos_cantidades", "pesos_de_la_lista"))
    assert llamadas[i + 1] == ("pista_de_la_linea", "sincronizar")
