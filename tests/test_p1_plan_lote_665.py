# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-665 · 2026-09-28] Dos líneas del mismo alimento en formas distintas no se funden.

Plan vivo de producción (28-sep): «250 ml de leche descremada» + «100 g de leche descremada en polvo» → «440 ml de leche
descremada» (el polvo contado como líquido); y «habichuelas secas» + «habichuelas cocidas» sumadas en gramos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import formas_de_base as fb  # noqa: E402
import graph_orchestrator as go  # noqa: E402


@pytest.mark.parametrize("linea, forma", [
    ("100 g de leche descremada en polvo", "polvo"),
    ("250 ml de leche descremada", ""),
    ("30 g de habichuelas rojas secas", ""),       # la base del catálogo es la seca
    ("90 g de habichuelas rojas cocidas", "cocido"),
    ("50 g de arroz blanco crudo", ""),
    ("75 g de huevo cocido", ""),                      # el huevo cocido pesa lo que el crudo: se sigue fundiendo
])
def test_la_forma_que_cambia_la_base(linea, forma):
    assert fb.clave(linea) == forma


def test_la_fusion_no_junta_formas_distintas(monkeypatch):
    canon = {"leche": "Leche descremada", "habichuelas": "Habichuelas rojas", "huevo": "Huevo"}

    def _pq(s, apply_yield_multiplier=False):
        import re
        m = re.match(r"\s*(\d+)\s*(g|ml)\s+de\s+(\w+)", s)
        return (float(m.group(1)), m.group(2), canon[m.group(3)]) if m else None

    import shopping_calculator as sc
    monkeypatch.setattr(sc, "_parse_quantity", _pq)
    monkeypatch.setattr(go, "_dup_merge_line_to_grams", lambda q, u, c, **k: q)
    monkeypatch.setattr(go, "_dup_merge_format", lambda total, u, c: f"{int(total)} {u} de {c}")
    days = [{"day": 1, "meals": [
        {"name": "Avena", "ingredients": ["250 ml de leche descremada", "100 g de leche descremada en polvo"]},
        {"name": "Moro", "ingredients": ["30 g de habichuelas rojas secas", "90 g de habichuelas rojas cocidas"]},
        {"name": "Revoltillo", "ingredients": ["100 g de huevo", "50 g de huevo cocido"]},
    ]}]
    go._merge_duplicate_food_lines(days)
    m = days[0]["meals"]
    assert m[0]["ingredients"] == ["250 ml de leche descremada", "100 g de leche descremada en polvo"]
    assert m[1]["ingredients"] == ["30 g de habichuelas rojas secas", "90 g de habichuelas rojas cocidas"]
    assert m[2]["ingredients"] == ["150 g de Huevo"]          # misma base: se sigue fundiendo
