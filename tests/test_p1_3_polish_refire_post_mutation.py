"""[P1-3-POLISH-REFIRE · 2026-07-10] (recipe plausibility roadmap, item P1-3) El countable-polish
(`_polish_finalize_display`: "1.25 cdas"→"1¼ cdas", "0.5 toronja"→"½ toronja") vive DENTRO de
`finalize_plan_data_coherence` (fpc) — que en el shield pre-INSERT (`_finalize_plan_data_for_insert`,
db_plans.py) corre ANTES de `reconcile_protein_band_post_finalize` y `reconcile_all_macros_band_post_finalize`
(P0-1-FINAL-BAND-CLOSER). Ambos reconcile_* mutan cantidades de ingredientes (vía
`apply_update_macro_engine` → `_rebalance_day_macros_to_target`/`refine_day_portions_integer`, que por
diseño NO re-quantizan — el docstring de refine dice "sin re-quantize", asumiendo que el caller lo hace)
SIN que nada vuelva a pulir el display después. Evidencia visual (plan 564d6e4e, banda 1.00 — la banda
está perfecta, el DISPLAY no): "1.25 cdas de mantequilla de maní", "0.5 toronja", "1.5 dientes de ajo".

Fix: `refire_display_polish_post_finalize(plan_data)` — wrapper público (mismo patrón que
`refresh_clinical_band_score_post_finalize`/`clear_stale_low_band_degraded`) que llama al
`_polish_finalize_display` YA EXISTENTE sobre `plan_data['days']`. Se engancha en el shield
DESPUÉS de `reconcile_all_macros_band_post_finalize` (última mutación de cantidades) — mismo
patrón "quantize = última mutación" ya usado para banda (memoria closer_coherence 2026-06-27).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


# ───────────────────────── estructural ─────────────────────────

def test_marker_and_knob_present():
    assert "P1-3-POLISH-REFIRE" in _GO


def test_function_defined_and_delegates_to_existing_polish():
    assert "def refire_display_polish_post_finalize(plan_data" in _GO
    i = _GO.index("def refire_display_polish_post_finalize(plan_data")
    body = _GO[i:i + 1800]
    assert "_polish_finalize_display(" in body, (
        "debe delegar en el countable-polish YA EXISTENTE (P1-FINALIZE-COUNTABLE-POLISH), "
        "no reimplementar la lógica de fracciones"
    )


def test_called_in_dbplans_after_all4_closer():
    """El re-fire debe correr DESPUÉS de reconcile_all_macros_band_post_finalize (última mutación
    de cantidades de ingredientes) — antes es inútil (el closer volvería a ensuciar el display)."""
    assert "refire_display_polish_post_finalize" in _DBP
    i_rpb = _DBP.index("reconcile_protein_band_post_finalize")
    i_all4 = _DBP.index("reconcile_all_macros_band_post_finalize")
    i_refire = _DBP.index("refire_display_polish_post_finalize")
    assert i_rpb < i_all4 < i_refire, (
        "el re-fire debe correr DESPUÉS de ambos pases de reconciliación de banda "
        "(proteína y all-4), que son los que mutan cantidades de ingredientes"
    )


# ───────────────────────── funcional ─────────────────────────

def _plan_with_messy_decimals():
    return {
        "days": [{"day": 1, "meals": [{
            "meal": "Merienda", "name": "Tostadas con maní",
            "ingredients": ["1.25 cdas de mantequilla de maní", "0.5 toronja"],
            "ingredients_raw": ["1.25 cdas de mantequilla de maní", "0.5 toronja"],
        }]}]
    }


def test_functional_polishes_decimals_to_fractions(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "FINALIZE_DISPLAY_POLISH", True)
    pd = _plan_with_messy_decimals()
    n = g.refire_display_polish_post_finalize(pd)
    assert n >= 1
    joined = " ".join(pd["days"][0]["meals"][0]["ingredients"])
    assert "1.25" not in joined and "0.5 toronja" not in joined


def test_functional_noop_on_clean_plan(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "FINALIZE_DISPLAY_POLISH", True)
    pd = {"days": [{"day": 1, "meals": [{"meal": "M", "name": "N",
                                          "ingredients": ["2 huevos"], "ingredients_raw": ["2 huevos"]}]}]}
    assert g.refire_display_polish_post_finalize(pd) == 0


def test_functional_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.refire_display_polish_post_finalize({}) == 0
    assert g.refire_display_polish_post_finalize(None) == 0
    assert g.refire_display_polish_post_finalize({"days": "not-a-list"}) == 0
