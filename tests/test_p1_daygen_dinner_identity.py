"""[P1-DAYGEN-DINNER-IDENTITY · 2026-07-09] (opción A del análisis de calidad) Nudge para que la CENA
tenga identidad propia y NO defaultee a tortilla/revoltillo de huevo.

Forense plan ae7ab047 (gain_muscle): 3 tortillas de cena (Día 1/2/3) → rechazo cross-day-dish tolerado en
el intento final (degradado). La técnica asignada se aplicaba a "la comida principal" y el LLM la ponía en
el almuerzo, dejando la cena libre → tortilla de huevo. El nudge (additive, knob-gated, prompt-cache-safe)
instruye que la cena sea un plato fuerte real con la proteína asignada, no huevo por defecto.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "prompts", "day_generator.py"), encoding="utf-8") as f:
    _DG = f.read()


def test_marker_present():
    assert "P1-DAYGEN-DINNER-IDENTITY" in _DG


def test_knob_default_on():
    assert "MEALFIT_DAYGEN_DINNER_IDENTITY" in _DG


def test_nudge_content():
    assert "IDENTIDAD DE LA CENA" in _DG
    assert "tortilla/revoltillo" in _DG


def test_nudge_injected_into_assignment_prompt():
    """El bloque se inyecta en el string de asignación del planificador."""
    assert "{dinner_identity_block}" in _DG


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def dg():
    import prompts.day_generator as _d
    return _d


def test_context_includes_nudge_when_on(dg, monkeypatch):
    monkeypatch.setenv("MEALFIT_DAYGEN_DINNER_IDENTITY", "true")
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}, 1)
    assert "IDENTIDAD DE LA CENA" in ctx


def test_context_omits_nudge_when_off(dg, monkeypatch):
    monkeypatch.setenv("MEALFIT_DAYGEN_DINNER_IDENTITY", "false")
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}, 1)
    assert "IDENTIDAD DE LA CENA" not in ctx
