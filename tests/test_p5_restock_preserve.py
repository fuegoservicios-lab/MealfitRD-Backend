"""[P5-RESTOCK-PRESERVE · 2026-06-23] Los tweaks de platos PANTRY-STRICT (swap desde la
Nevera / regenerate-day) NO deben limpiar `is_restocked`: cocinan desde la Nevera existente,
no introducen nada que el usuario deba comprar.

Bug recurrente reportado (con screenshot): el usuario con la Nevera llena (41 items) hacía un
swap/día pantry-strict → reaparecía el banner "Tu Nevera está vacía para este plan" + el botón
"Ya compré la lista". Dos vías limpiaban el flag por error:
  (1) swap single: chequeo P0-1 de substring contra el inventario daba falsos positivos por
      canonicalización ("Pechuga de pollo" vs "Pollo") → clear_is_restocked.
  (2) regenerate-day: el recalc post-día veía un falso `has_changed` → limpiaba is_restocked.

Fix: el backend señala `pantry_constrained` en el swap; el frontend NO limpia is_restocked para
platos pantry-strict; y el recalc respeta `preserve_restock` cuando la Nevera no está vacía.
"""
import os
import re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AGENT = open(os.path.join(_ROOT, "agent.py"), encoding="utf-8").read()
_PLANS = open(os.path.join(_ROOT, "routers", "plans.py"), encoding="utf-8").read()
_ASSESS = open(os.path.join(_ROOT, "..", "frontend", "src", "context", "AssessmentContext.jsx"), encoding="utf-8").read()


def test_swap_meal_emits_pantry_constrained():
    """swap_meal debe inyectar `pantry_constrained` (bool de clean_ingredients) en su retorno."""
    assert "P5-RESTOCK-PRESERVE" in _AGENT
    assert '"pantry_constrained"' in _AGENT
    assert "_pantry_constrained = bool(clean_ingredients)" in _AGENT


def test_recalc_respects_preserve_restock():
    """El recalc debe leer `preserve_restock` y, con Nevera no vacía, NO limpiar is_restocked."""
    assert 'data.get("preserve_restock"' in _PLANS
    # El gate del pop debe incluir `and not _preserve_restock`.
    assert re.search(r"and not _preserve_restock", _PLANS), \
        "el clear de is_restocked debe respetar preserve_restock"
    # preserve_restock solo aplica con inventario > 0 (el heal de pantry-vacía sigue ganando).
    assert "_inv_count_at_recalc > 0" in _PLANS


def test_frontend_skips_clear_for_pantry_strict_swap():
    """El P0-1 del swap NO debe limpiar is_restocked cuando pantry_constrained === true."""
    assert "newMealData?.pantry_constrained !== true" in _ASSESS, \
        "el clear de is_restocked debe saltarse para platos pantry-strict"


def test_regenerate_day_recalc_sends_preserve_restock():
    """regenerateDay debe mandar preserve_restock:true en el body del recalc."""
    assert "preserve_restock: true" in _ASSESS
