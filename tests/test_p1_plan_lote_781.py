# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-781 · 2026-09-28] El maní no se filetea: «maní fileteado» → «maní picado».

Auditoría de la cola 744 (5.157 comidas): 79 con «mide 10 g de maní fileteado» o «Guineo fresco con maní fileteadas y
queso cottage», resto de la sustitución por presupuesto de «almendras fileteadas».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import mani_picado as mp  # noqa: E402


def test_nombre_pasos_y_lista():
    m = {"name": "Guineo fresco con maní fileteadas y queso cottage",
         "ingredients": ["1 guineo", "10 g de maní fileteado", "40 g de queso cottage"],
         "ingredients_raw": ["1 guineo", "10 g de maní fileteado", "40 g de queso cottage"],
         "recipe": ["Mise en place: pela el guineo; mide 10 g de maní fileteado.",
                    "Montaje: sirve el guineo con el Maní fileteado por encima."]}
    assert mp.limpiar(m) == 5
    assert m["name"] == "Guineo fresco con maní picado y queso cottage"
    assert m["ingredients"][1] == "10 g de maní picado" and m["ingredients_raw"][1] == "10 g de maní picado"
    assert m["recipe"][0].endswith("mide 10 g de maní picado.")
    assert m["recipe"][1] == "Montaje: sirve el guineo con el Maní picado por encima."


def test_las_almendras_fileteadas_siguen_igual():
    m = {"name": "Yogur con almendras fileteadas", "ingredients": ["15 g de almendras fileteadas"],
         "recipe": ["Montaje: termina con las almendras fileteadas."]}
    antes = (m["name"], list(m["ingredients"]), list(m["recipe"]))
    assert mp.limpiar(m) == 0
    assert (m["name"], m["ingredients"], m["recipe"]) == (antes[0], antes[1], antes[2])


def test_enganchado_en_la_cola_del_contrato():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("mani_picado").limpiar(meal)' in src
