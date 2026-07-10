"""[P2-3-BATCH-ARITHMETIC-CHECK · 2026-07-10] (recipe plausibility roadmap, item P2-3) Evidencia visual
(plan 564d6e4e): "Formar 6 tortitas ... Servir 3 tortitas de batata y queso en el plato" para una receta
de 1 porción — el batch de cocción (6) no coincide con lo servido (3), sin explicar qué pasa con el
resto. Warn-first (frecuencia baja per roadmap): detector determinista conservador
forma/hace-N ↔ sirve-M sobre los pasos de receta; NO reescribe todavía (evidencia primero).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_present():
    assert "P2-3-BATCH-ARITHMETIC-CHECK" in _GO


def test_function_defined():
    assert "def detect_batch_arithmetic_mismatch(plan_data" in _GO


def _plan(recipe_steps):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "Tortitas de batata",
                                            "recipe": list(recipe_steps)}]}]}


def test_flags_batch_serve_mismatch():
    import graph_orchestrator as g
    pd = _plan([
        "Mise en place: pela la batata y ralla la tayota.",
        "El Toque de Fuego: forma 6 tortitas y colócalas en una bandeja engrasada. Hornea 20 min.",
        "Montaje: sirve 3 tortitas de batata y queso en el plato.",
    ])
    violations = g.detect_batch_arithmetic_mismatch(pd)
    assert len(violations) == 1
    assert violations[0]["formed"] == 6
    assert violations[0]["served"] == 3


def test_passes_matching_batch_and_serve():
    import graph_orchestrator as g
    pd = _plan([
        "El Toque de Fuego: forma 4 tortitas pequeñas y colócalas en una bandeja.",
        "Montaje: sirve las 4 tortitas en el plato.",
    ])
    assert g.detect_batch_arithmetic_mismatch(pd) == []


def test_noop_without_batch_or_serve_mentions():
    import graph_orchestrator as g
    pd = _plan(["Mise en place: pela la papa.", "Montaje: sirve caliente."])
    assert g.detect_batch_arithmetic_mismatch(pd) == []


def test_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.detect_batch_arithmetic_mismatch({}) == []
    assert g.detect_batch_arithmetic_mismatch(None) == []
