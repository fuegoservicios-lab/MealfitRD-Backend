# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-247 · 2026-09-25] «Maní molido hasta obtener una crema» no es un lácteo.

Batería real del 25-sep (vegana, México): la guarda de dieta rechazó CRÍTICO esa línea como lácteo y quemó un intento.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _plan(*ings):
    return {"days": [{"meals": [{"name": "Pan integral tostado con crema de maní y fresas", "ingredients": list(ings)}]}]}


def test_la_crema_de_mani_molido_no_es_lactea_para_una_vegana():
    assert go._scan_diet_violations(_plan("25 g de maní molido hasta obtener una crema"), "vegan") == []
    assert go._scan_diet_violations(_plan("30 g de almendras trituradas hasta formar una mantequilla"), "vegan") == []


def test_ni_para_un_alergico_a_los_lacteos():
    assert go._scan_allergen_violations(_plan("25 g de maní molido hasta obtener una crema"), ["Lacteos"]) == []
    # la cocina dominicana: «maja las habichuelas … hasta lograr una crema» (batería real, 25-sep)
    assert go._scan_allergen_violations(_plan("½ taza de habichuelas rojas majadas hasta lograr una crema"),
                                        ["Lacteos"]) == []


def test_lo_lacteo_de_verdad_se_sigue_marcando():
    assert go._scan_diet_violations(_plan("½ taza de crema de leche"), "vegan")
    assert go._scan_diet_violations(_plan("Fresas con almendras y crema batida"), "vegan")
    assert go._scan_diet_violations(_plan("30 g de queso crema"), "vegan")
    assert go._scan_allergen_violations(_plan("Fresas con almendras y crema batida"), ["Lacteos"])
    # y el alérgico al maní sigue viendo su maní
    assert go._scan_allergen_violations(_plan("25 g de maní molido hasta obtener una crema"), ["Mani"])


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 247
