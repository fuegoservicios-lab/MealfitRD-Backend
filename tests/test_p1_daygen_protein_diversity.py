"""[P1-DAYGEN-PROTEIN-DIVERSITY · 2026-07-09] Nudge para diversificar la proteína principal y NO
sobrecargar el plan de queso (alto en sodio).

Forense plan 55b659c5 (gain_muscle, renovación en vivo 2026-07-09): 8/12 comidas usaban queso como
proteína principal (queso de freír, cottage, crema, queso blanco). Consecuencias: (1) sodio día 3 =
2410mg vs techo 2000mg → ÚNICA causa del `_quality_degraded` (reason=micro_worst_day_ceiling); (2)
monotonía; (3) proteína animal magra es superior para ganancia muscular. El nudge (additive, knob-gated,
prompt-cache-safe) instruye usar queso como proteína PRINCIPAL en ≤1 comida/día y diversificar el resto
con proteína animal magra (pollo/pescado/res/cerdo/calamar/huevo/hígado).

Espeja P1-DAYGEN-DINNER-IDENTITY.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "prompts", "day_generator.py"), encoding="utf-8") as f:
    _DG = f.read()


def test_marker_present():
    assert "P1-DAYGEN-PROTEIN-DIVERSITY" in _DG


def test_knob_default_on():
    assert "MEALFIT_DAYGEN_PROTEIN_DIVERSITY" in _DG


def test_nudge_content():
    assert "DIVERSIDAD DE PROTEÍNA" in _DG
    # Debe mencionar queso (el ingrediente a limitar) y el cap ≤1/día.
    assert "queso" in _DG.lower()


def test_nudge_injected_into_assignment_prompt():
    """El bloque se inyecta en el string de asignación del planificador."""
    assert "{protein_diversity_block}" in _DG


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def dg():
    import prompts.day_generator as _d
    return _d


def test_context_includes_nudge_when_on(dg, monkeypatch):
    monkeypatch.setenv("MEALFIT_DAYGEN_PROTEIN_DIVERSITY", "true")
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}, 1)
    assert "DIVERSIDAD DE PROTEÍNA" in ctx


def test_context_omits_nudge_when_off(dg, monkeypatch):
    monkeypatch.setenv("MEALFIT_DAYGEN_PROTEIN_DIVERSITY", "false")
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}, 1)
    assert "DIVERSIDAD DE PROTEÍNA" not in ctx


def test_lean_preference_complements_fat_budget(dg, monkeypatch):
    """[P1-DAYGEN-PROTEIN-DIVERSITY-LEAN · 2026-07-09] La diversidad debe preferir cortes MAGROS por
    defecto — la grasa embebida en un corte graso no se recorta después y revienta la banda (forense
    plan f19d55a6 intento 2, día grasas 174%)."""
    monkeypatch.setenv("MEALFIT_DAYGEN_PROTEIN_DIVERSITY", "true")
    ctx = dg.build_day_assignment_context({"assigned_technique": "Guiso", "protein_pool": ["Pollo"]}, 1)
    assert "MAGROS" in ctx
    assert "embebida" in ctx  # explica por qué (no se puede recortar después)
