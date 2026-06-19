"""[P2-STRUCTURED-RETRY-CONTEXT · 2026-06-18] (audit fresco P2-2) El retry inyectaba solo prosa
("RECHAZADO por: ..."). Ahora enriquece la directiva con los deltas de macro DETERMINISTAS por día del
plan rechazado (números concretos por celda día×macro fuera de banda) → el planner corrige celdas
específicas en vez de adivinar. Read-only + fail-safe; NO cambia el control de flujo del grafo.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# Targets: proteína 110g, carbos 200g, grasas 60g.
_NUTRITION = {"total_daily_macros": {"protein_str": "110g", "carbs_str": "200g", "fats_str": "60g"}}


def _plan(p, c, f):
    """Plan de 1 día con un meal cuyos macros entregados son (p, c, f)."""
    return {"days": [{"meals": [{"name": "Almuerzo", "protein": p, "carbs": c, "fats": f}]}]}


# ── Unit del helper determinista ──
def test_structured_context_flags_out_of_band_protein(go):
    # Proteína 70/110 = 0.64 (fuera de [0.90,1.12]); carbos/grasas en banda.
    s = go._structured_rejection_context(_plan(70, 200, 60), _NUTRITION)
    assert "Día 1" in s
    assert "proteína" in s
    assert "70g" in s and "110g" in s
    assert "carbohidratos" not in s  # en banda → no se lista
    assert "grasas" not in s


def test_structured_context_empty_when_all_in_band(go):
    s = go._structured_rejection_context(_plan(110, 200, 60), _NUTRITION)
    assert s == ""


def test_structured_context_failsafe_on_missing_data(go):
    assert go._structured_rejection_context(None, _NUTRITION) == ""
    assert go._structured_rejection_context(_plan(70, 200, 60), None) == ""
    assert go._structured_rejection_context({}, {}) == ""


def test_structured_context_caps_cells(go):
    # Muchos días fuera de banda → capeado a max_cells (8 por default), peores primero.
    days = [{"meals": [{"protein": 10, "carbs": 10, "fats": 5}]} for _ in range(10)]
    s = go._structured_rejection_context({"days": days}, _NUTRITION, max_cells=5)
    assert s.count("Día ") == 5


# ── Wiring en retry_reflection_node ──
def test_retry_directive_includes_structured_deltas(go):
    state = {"rejection_reasons": ["proteína insuficiente"], "attempt": 1,
             "plan_result": _plan(70, 200, 60), "nutrition": _NUTRITION}
    out = asyncio.run(go.retry_reflection_node(state))
    d = out["reflection_directive"]
    assert "RECHAZADO por" in d                      # prosa original preservada
    assert "DESVIACIONES DE MACRO MEDIDAS" in d      # + deltas estructurados
    assert "proteína 70g vs objetivo 110g" in d


def test_retry_directive_knob_off(go, monkeypatch):
    monkeypatch.setattr(go, "STRUCTURED_RETRY_CONTEXT_ENABLED", False)
    state = {"rejection_reasons": ["proteína insuficiente"], "attempt": 1,
             "plan_result": _plan(70, 200, 60), "nutrition": _NUTRITION}
    out = asyncio.run(go.retry_reflection_node(state))
    assert "RECHAZADO por" in out["reflection_directive"]
    assert "DESVIACIONES DE MACRO" not in out["reflection_directive"]  # knob OFF → solo prosa


def test_retry_no_reasons_no_directive(go):
    out = asyncio.run(go.retry_reflection_node({"rejection_reasons": [], "attempt": 1}))
    assert out["reflection_directive"] is None  # sin reasons → directive limpio (P1-ORQ-2)


def test_orchestrator_anchors(go):
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert 'STRUCTURED_RETRY_CONTEXT_ENABLED = _env_bool("MEALFIT_STRUCTURED_RETRY_CONTEXT", True)' in src
    assert "def _structured_rejection_context(" in src
    assert "if STRUCTURED_RETRY_CONTEXT_ENABLED:" in src
