# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-314 · 2026-09-25] El tope de un rango («3-4 cdas») no es una cantidad.

Plan real del perfil del dueño, 25-sep: «mide ¼ taza de harina de maíz precocida y 3-3 cdas de agua». El paso decía
«3-4 cdas de agua»; el sincronizador tomaba «4 cdas de agua» como mención y la llevaba a la lista (3), dejando «3-3»."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_el_rango_queda_como_rango():
    for rango in ("3-4 cdas", "3–4 cdas", "3 a 4 cdas"):
        m = {"ingredients": ["¼ taza de harina de maíz precocida", "3 cdas de agua"],
             "recipe": [f"Mise en place: mide ¼ taza de harina de maíz precocida y {rango} de agua."]}
        go._sync_recipe_step_quantities(m)
        assert f"{rango} de agua" in m["recipe"][0], m["recipe"][0]


def test_una_cantidad_suelta_se_sigue_sincronizando():
    m = {"ingredients": ["3 cdas de agua"], "recipe": ["Mise en place: mide 5 cdas de agua tibia."]}
    go._sync_recipe_step_quantities(m)
    assert "mide 3 cdas de agua tibia" in m["recipe"][0], m["recipe"][0]


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "(?<![-–])(?<!\\d\\sa\\s)" in src and "[P1-PLAN-LOTE-314] rangos" in src
