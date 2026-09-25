# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-262 · 2026-09-25] «almendras tostadas» y el wrap DE hojas no son pan.

Batería final (sin gluten + huevo) y medido contra el código: el escáner de alérgenos marcaba como GLUTEN los frutos
secos y semillas TOSTADOS y el wrap de hojas (el plan del celíaco se rechazaba), y la sustitución proactiva del gluten
convertía «10 g de semillas de sésamo tostadas» en «10 g de Casabe» y sus pasos en «semillas de sésamo casabe».
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _viola(ingrediente, alergias=("Gluten",)):
    plan = {"days": [{"day": 1, "meals": [{"name": "x", "ingredients": [ingrediente]}]}]}
    return go._scan_allergen_violations(plan, list(alergias))


@pytest.mark.parametrize("ingrediente", [
    "Almendras tostadas",
    "15 g de almendras tostadas",
    "15 g de almendras fileteadas tostadas",
    "10 g de semillas de sésamo tostadas",
    "20 g de pepitas de auyama tostadas",
    "15 g de nueces tostadas",
    "1 cda de avellanas tostadas",
    "1 nuez tostada",
    "10 g de semillas de girasol tostadas",
    "2 hojas de lechuga para wrap",
    "4 hojas de repollo para los wraps",
    "1 wrap de lechuga",
    "2 wraps de hojas de repollo",
])
def test_no_es_pan(ingrediente):
    assert _viola(ingrediente) == [], f"{ingrediente!r} no lleva gluten"


@pytest.mark.parametrize("ingrediente", [
    "2 tostadas integrales",
    "1 tostada",
    "2 tostadas de pan",
    "15 g de almendras y 2 tostadas",
    "almendras con tostadas",
    "1 wrap",
    "1 wrap integral",
    "1 wrap con lechuga",
    "1 tortilla de trigo para wrap",
    "1 tortilla con lechuga para wrap",
    "semillas de trigo tostadas",
    "20 g de avena tostada",
])
def test_el_pan_sigue_marcado(ingrediente):
    assert _viola(ingrediente), f"{ingrediente!r} puede llevar gluten: debe seguir marcado"


def test_la_excusa_no_absuelve_a_otro_termino_de_la_linea():
    v = _viola("1 rebanada de pan con almendras tostadas")
    assert v and v[0][2] == "pan"


def _sustituir(ingrediente):
    plan = {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Yogur de coco",
                                            "ingredients": [ingrediente, "150 g de yogur de coco"],
                                            "recipe": [f"Montaje: sirve el yogur con {ingrediente.lower()} por encima."]}]}]}
    p = copy.deepcopy(plan)
    n = go._apply_allergen_substitutions(p, {"allergies": ["Gluten"], "country": "DO"})
    return n, p["days"][0]["meals"][0]


@pytest.mark.parametrize("ingrediente", [
    "Almendras tostadas", "10 g de semillas de sésamo tostadas", "15 g de nueces tostadas",
    "20 g de pepitas de auyama tostadas", "1 cda de avellanas tostadas",
])
def test_la_sustitucion_no_convierte_frutos_secos_en_casabe(ingrediente):
    n, meal = _sustituir(ingrediente)
    assert n == 0 and meal["ingredients"][0] == ingrediente
    assert "casabe" not in " ".join(meal["recipe"]).lower()


def test_la_sustitucion_del_pan_sigue():
    n, meal = _sustituir("2 tostadas integrales")
    assert n == 1 and "Casabe" in meal["ingredients"][0]


def test_ancla():
    src = (_BACKEND / "excusas_vegetales.py").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-262-ADJETIVO" in src
    go_src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'excusa_contextual(f, ing_low, _m_al.start(), _m_al.end())' in go_src
    assert 'sustitucion_excusada(s, ing_norm)' in go_src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 262
