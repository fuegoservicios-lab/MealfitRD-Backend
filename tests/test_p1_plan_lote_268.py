# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-268 · 2026-09-25] «Harina de Negrito» es crema de trigo y la ensalada de macarrones lleva mayonesa.

Batería final (mariscos + gluten): el revisor de IA rechazó «la harina de Negrito contiene trigo/gluten»; el escáner
determinista no la conocía y el filtro de catálogo (mismo vocabulario) se la OFRECÍA al celíaco. Barrido de las 349
filas del catálogo contra el escáner, clase por clase: además, «Ensalada de macarrones» (alias «ensalada de pasta con
mayonesa») pasaba limpia para el alérgico al huevo.
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


def _viola(linea, chip):
    return go._scan_allergen_violations({"days": [{"meals": [{"name": "x", "ingredients": [linea]}]}]}, [chip])


@pytest.mark.parametrize("linea", ["20 g de harina de Negrito", "1 taza de crema de Negrito", "30 g de Negrito",
                                   "40 g de farina"])
def test_negrito_es_gluten(linea):
    assert _viola(linea, "Gluten")


@pytest.mark.parametrize("linea", ["100 g de ensalada de macarrones", "1 taza de ensalada de coditos",
                                   "120 g de ensalada rusa"])
def test_ensaladas_con_mayonesa_son_huevo(linea):
    assert _viola(linea, "Huevo")


def test_la_dieta_vegana_tambien_la_ve():
    viola = go._scan_diet_violations({"days": [{"meals": [{"name": "x", "ingredients": ["120 g de ensalada rusa"]}]}]},
                                     "vegana")
    assert viola


def test_el_catalogo_ya_no_se_la_ofrece_al_celiaco():
    from constants import _get_fast_filtered_catalogs
    pools = _get_fast_filtered_catalogs(("Gluten",), (), "")
    assert not any("negrito" in str(x).lower() for pool in pools for x in pool)


@pytest.mark.parametrize("linea", ["100 g de harina de maíz precocida", "2 tortas de casabe", "100 g de harina de yuca"])
def test_lo_sin_gluten_sigue_limpio(linea):
    assert not _viola(linea, "Gluten")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 268
