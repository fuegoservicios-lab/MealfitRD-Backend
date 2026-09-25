# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-269 · 2026-09-25] Miel para la vegana y carne oculta (gelatina, grenetina, colágeno, sopita).

Sonda contra el escáner de dieta: pasaban limpias. La miel además se le ofrecía a la vegana en el catálogo.
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


def _viola(linea, dieta):
    return go._scan_diet_violations({"days": [{"meals": [{"name": "x", "ingredients": [linea]}]}]}, dieta)


@pytest.mark.parametrize("linea, dieta", [
    ("1 cdta de miel", "vegana"), ("1 cda de miel de abeja", "vegana"),
    ("1 taza de gelatina sin azúcar", "vegana"), ("1 taza de gelatina sin azúcar", "vegetariana"),
    ("1 taza de gelatina de fresa", "pescetariana"), ("5 g de grenetina", "vegetariana"),
    ("10 g de colágeno hidrolizado", "vegetariana"), ("1 sobre de sopita", "vegetariana"),
])
def test_marcado(linea, dieta):
    assert _viola(linea, dieta), (linea, dieta)


@pytest.mark.parametrize("linea, dieta", [
    ("1 cdta de miel de agave", "vegana"), ("1 cda de miel de caña", "vegana"), ("1 cda de miel de maple", "vegana"),
    ("1 taza de gelatina de agar", "vegana"), ("1 taza de gelatina vegetal", "vegetariana"),
    ("1 sobre de sopita de vegetales", "vegetariana"), ("10 g de colágeno vegano", "vegana"),
    ("1 cdta de miel", "vegetariana"), ("1 cdta de miel", "pescetariana"),
    ("25 g de maní molido hasta obtener una crema", "vegana"),        # el 247 sigue en pie
])
def test_limpio(linea, dieta):
    assert not _viola(linea, dieta), (linea, dieta)


def test_la_excusa_esta_acotada_al_termino():
    assert _viola("2 lonjas de tocino de maple", "vegana")
    assert _viola("2 lonjas de tocino de maple", "vegetariana")


def test_el_catalogo_vegano_ya_no_ofrece_miel():
    from constants import _get_fast_filtered_catalogs
    pools = _get_fast_filtered_catalogs((), (), "vegana")
    assert not any(re.search(r"\bmiel\b", str(x).lower()) for pool in pools for x in pool)


def test_ancla():
    assert "P1-PLAN-LOTE-269-DIETA-OCULTA" in (_BACKEND / "vocabulario_dieta.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 269
