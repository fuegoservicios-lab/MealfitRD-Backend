# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-527 · 2026-09-27] «yogur» y «yogurt» son el mismo alimento al reflejar la proteína en el nombre.

Batería real del dueño (días 25-28): «Yogur con manzana, queso cottage y yogurt»; en el replay forzado «Avena cremosa con
yogur, manzana, queso cottage y yogurt», «Casabe crujiente con yogur, manzana, queso cottage y yogurt».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from constants import strip_accents  # noqa: E402


def test_el_yogur_del_nombre_es_el_yogurt_de_la_lista():
    m = {"name": "Yogur con manzana y queso cottage"}
    assert go._reflect_added_protein_in_name(m, "yogurt natural sin azúcar", strip_accents) is False
    assert m["name"] == "Yogur con manzana y queso cottage"
    m = {"name": "Avena cremosa con yogurt y manzana"}
    assert go._reflect_added_protein_in_name(m, "Yogur griego entero", strip_accents) is False


def test_sin_yogur_en_el_nombre_se_sigue_reflejando():
    m = {"name": "Avena cremosa con manzana"}
    assert go._reflect_added_protein_in_name(m, "yogurt griego", strip_accents) is True
    assert "yogurt" in m["name"].lower(), m["name"]


def test_canon_yogur():
    from dish_naming import canon_yogur
    assert canon_yogur("Yogurt griego") == "yogur griego"
    assert canon_yogur("yoghurt") == "yogur" and canon_yogur("yogures") == "yogur"
    assert canon_yogur("Yogur natural") == "yogur natural"
