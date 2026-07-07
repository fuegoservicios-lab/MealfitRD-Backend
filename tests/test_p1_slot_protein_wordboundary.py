"""[P1-SLOT-PROTEIN-WORDBOUNDARY · 2026-07-07] `_detect_main_items` matcheaba proteína
por SUBSTRING → 'res' en "Queso FRESco"/"FRESas".

Observado en el benchmark de macros: `_detect_slot_incoherence` reportaba "la proteína
'res' aparece en 4 comidas del Día 1" en un día SIN res real (solo hígado de res en el
almuerzo). Causa: `_detect_main_items` usaba `alias in blob` (substring), así el alias
'res' matcheaba "Queso Fresco", "Batido de Fresas", "purés", etc. Ese reporte falso se
inyectaba al self-critique del generador → "CORRECCIONES OBLIGATORIAS" para arreglar una
repetición inexistente → regens innecesarios (quema DeepSeek, agravó el circuit breaker
durante el benchmark).

Fix: match con FRONTERA DE PALABRA (`_name_has_token`, mismo SSOT que build_variety_report).
`\bres` no matchea "fresco" (la 'f' precede) pero sí tolera plurales ("huevos", "frijoles").
tooltip-anchor: P1-SLOT-PROTEIN-WORDBOUNDARY
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _prot(name, ings=None):
    return g._detect_main_items({"name": name, "ingredients": ings or []}, g._MAIN_PROTEIN_ALIASES)


def test_marker_and_wordboundary_used():
    assert "P1-SLOT-PROTEIN-WORDBOUNDARY" in _SRC
    assert "_name_has_token(_norm_text(a), blob)" in _SRC, "debe usar frontera de palabra"


def test_res_not_detected_in_fresco():
    assert "res" not in _prot("Tortilla de Huevos con Espinaca y Queso Fresco")


def test_res_not_detected_in_fresas():
    assert "res" not in _prot("Batido de Fresas con Yogurt Griego")


def test_res_detected_in_real_beef():
    assert "res" in _prot("Hígado de res a la plancha", ["150 g de hígado de res"])


def test_plural_tolerated():
    """Frontera de palabra al INICIO → tolera plurales (huevo↔huevos)."""
    assert "huevo" in _prot("Revoltillo de Huevos con Tomate")


def test_slot_incoherence_no_false_res():
    """El caso del benchmark: un día con 'fresco'/'fresas' y solo 1 res real (almuerzo)
    NO debe generar issue de 'res en múltiples comidas'."""
    day = {"day": 1, "meals": [
        {"meal": "almuerzo", "name": "Hígado de res guisado", "ingredients": ["150 g de hígado de res"]},
        {"meal": "cena", "name": "Tortilla de Huevos con Queso Fresco", "ingredients": ["2 huevos", "queso fresco"]},
        {"meal": "merienda", "name": "Batido de Fresas", "ingredients": ["fresas", "yogurt"]},
    ]}
    issues = g._detect_slot_incoherence([day])
    assert not any("res" in _i and "aparece en" in _i for _i in issues), (
        f"no debe reportar 'res' en múltiples comidas por 'fresco'/'fresas': {issues}"
    )
