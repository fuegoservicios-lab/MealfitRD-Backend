"""[P1-PLAN-QUALITY-FOLLOWUPS · 2026-07-08] Batch de follow-ups de calidad del plan (review en vivo).

FU3 (P1-CHUNK-GAINMUSCLE-PARITY): el gainmuscle day-kcal-floor (re-relleno final) solo corría en assemble/
  semana 1 → los días de bulk de semanas 2+ (chunks, ~27 de 30 días) que los recortes tardíos dejaban bajo
  banda nunca se re-rellenaban. Ahora corre en `finalize_plan_data_coherence` (path de chunks) con goal +
  target_macros que pasa el chunk worker.
FU1 (P1-CLOSER-SWEET-NO-LEGUME): en un snack DULCE (fruta+nueces) una legumbre salada (guisantes/lentejas/
  habichuelas) es tan incongruente como carne/pescado → se excluye del pool de proteína del closer.
FU4 (P1-MICRO-WORSTDAY-EXCLUDE-UNREACHABLE): omega-3/vitE/vitD (difíciles de alimentos enteros, ya con
  tarjeta de suplemento) NO marcan "degradado" un plan macro-perfecto; si queda un micro cerrable corto
  (fibra/Ca/hierro), el banner sigue.
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()
with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
    _CT = f.read()


# ═══════════════════ FU3: chunk gainmuscle parity ═══════════════════

def test_fu3_marker_and_finalizer_signature():
    assert "P1-CHUNK-GAINMUSCLE-PARITY" in _GO
    assert "def finalize_plan_data_coherence(days: list, db=None, allergies=None, target_fats=None, *," in _GO
    assert "main_goal=None, target_macros=None" in _GO


def test_fu3_gainmuscle_runs_after_cheese_final_in_finalizer():
    """En el finalizer, el re-relleno de bulk debe correr DESPUÉS de cheese-final (última pasada aditiva)."""
    i_fn = _GO.index("def finalize_plan_data_coherence")
    _next = _GO.find("\ndef ", i_fn + 10)  # cuerpo completo hasta el siguiente def a nivel de módulo
    body = _GO[i_fn:_next if _next > i_fn else i_fn + 20000]
    assert "_repair_gainmuscle_day_kcal(" in body and "final_pass=True" in body
    assert body.index("CHEESE_DUMP_FINAL_ENABLED") < body.index("P1-CHUNK-GAINMUSCLE-PARITY")


def test_fu3_chunk_worker_passes_goal_and_macros():
    assert "main_goal=plan_data.get(\"main_goal\"), target_macros=_tm_ck" in _CT
    assert "P1-CHUNK-GAINMUSCLE-PARITY" in _CT


# ═══════════════════ FU1: sweet-snack no legume ═══════════════════

def test_fu1_marker_knob_and_exclusion():
    assert "P1-CLOSER-SWEET-NO-LEGUME" in _GO
    assert 'CLOSER_SWEET_NO_LEGUME = _env_bool("MEALFIT_CLOSER_SWEET_NO_LEGUME", True)' in _GO
    # el pool dulce excluye legumbres junto con la carne
    i = _GO.index("_pool_meat_free = [(info, nlow)")
    win = _GO[i:i + 320]
    assert "_MEAT_PROTEIN_HINT" in win and "_LEGUME_PROTEIN_HINT" in win and "CLOSER_SWEET_NO_LEGUME" in win


# ═══════════════════ FU4: micro worst-day exclude unreachable ═══════════════════

def _plan_wd(low):
    return {"micronutrient_report": {"gaps": [],
            "per_day_floors": {"flagged": True, "worst_day": {"day_index": 2, "low": list(low)}}}}


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_SOFT_REJECT_ENABLED", False)  # el low_micros promedio está OFF
    monkeypatch.setattr(g, "MICRO_PERDAY_DEGRADE_ENABLED", True)
    return g


def test_fu4_marker_and_set():
    assert "P1-MICRO-WORSTDAY-EXCLUDE-UNREACHABLE" in _GO
    assert '_MICRO_WORSTDAY_EXCLUDE = ("vit_d_mcg", "omega3_g", "vit_e_mg")' in _GO


def test_fu4_unreachable_only_does_not_degrade(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_WORSTDAY_EXCLUDE_UNREACHABLE", True)
    plan = _plan_wd(["vit_e_mg", "omega3_g"])
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert not plan.get("_quality_degraded"), "worst_day solo omega3/vitE → NO degrada (banner limpio)"


def test_fu4_closeable_micro_still_degrades(go):
    plan = _plan_wd(["vit_e_mg", "omega3_g", "fiber_g"])
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert plan.get("_quality_degraded") is True
    assert plan.get("_quality_degraded_reason") == "micro_worst_day", "fibra cerrable corta → sigue el banner"


def test_fu4_knob_off_reverts(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_WORSTDAY_EXCLUDE_UNREACHABLE", False)
    plan = _plan_wd(["vit_e_mg", "omega3_g"])
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert plan.get("_quality_degraded") is True, "knob OFF → omega3/vitE vuelven a degradar (comportamiento previo)"
