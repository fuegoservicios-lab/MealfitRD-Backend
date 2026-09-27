# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-522 · 2026-09-27] La rebanada de pan pesa lo que dice el catálogo.

Cadena completa re-encadenada (173 planes) y batería real del 27-sep: 11 de 147 comidas con pan traían la misma línea
repetida —«1 rebanada de pan integral» dos veces, o «2 rebanadas de pan integral familiar» + «1 rebanada…» + «2
rebanadas…»—. El motor de macros ya pesaba la rebanada (30 g, `density_g_per_unit` del pan de molde), pero el
resolvedor del grafo (`_dup_merge_line_to_grams`) sólo convertía g/ml/taza/unidad: «sin gramos» ⇒ el deduplicador no
sumaba y el reconciliador no comparaba.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _indice(monkeypatch):
    monkeypatch.setattr(go, "_catalog_density_index", lambda: {"pan integral familiar": (30.0, 0.0)})


def test_la_rebanada_se_pesa(monkeypatch):
    _indice(monkeypatch)
    assert go._dup_merge_line_to_grams(2, "rebanadas", "Pan integral familiar") == 60.0
    assert go._dup_merge_line_to_grams(1, "rebanada", "Pan integral familiar") == 30.0
    assert go._dup_merge_line_to_grams(2, "lonjas", "Pan integral familiar") is None, "la lonja sigue sin fusionarse"


def test_el_total_se_escribe_en_rebanadas(monkeypatch):
    _indice(monkeypatch)
    assert go._dup_merge_format(90.0, "rebanadas", "Pan integral familiar") == "3 rebanadas de Pan integral familiar"
    assert go._dup_merge_format(30.0, "rebanada", "Pan integral familiar") == "1 rebanada de Pan integral familiar"
