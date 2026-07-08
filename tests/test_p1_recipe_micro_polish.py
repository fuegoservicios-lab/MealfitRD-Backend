"""[P1-RECIPE-MICRO-POLISH · 2026-07-08] Dos pulidos del review en vivo (plan fcb739fa).

Fix A (P1-MISE-COOK-COND): una cocción CONDICIONAL en el Mise ("Si las semillas están crudas, tuéstalas…")
  no se separaba a un 'El Toque de Fuego' porque `_MISE_COOK_VERB_RE` exigía el verbo al INICIO de la
  oración → el lint emitía "falta 'El Toque de Fuego'" → el frontend mostraba "Receta con pasos incompletos —
  regenera para detalle" en una receta que en realidad estaba completa. Fix: prefijo condicional opcional.
Fix B (P1-MICRO-WORSTDAY-MIN2): tras excluir los micros inalcanzables (omega3/vitE/vitD), exigir ≥2 micros
  CERRABLES cortos para marcar _quality_degraded — un único micro marginal (p.ej. fibra a 80%) no degrada un
  plan macro-perfecto (alinea con el umbral ≥2 del propio panel).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ═══════════════════ Fix A: mise-cook-split condicional ═══════════════════

def test_fix_a_marker_and_conditional_prefix():
    assert "P1-MISE-COOK-COND" in _GO
    i = _GO.index("_MISE_COOK_VERB_RE = _re.compile(")
    win = _GO[i:i + 500]
    assert r"si\b[^,]{0,80},\s*" in win, "prefijo condicional opcional debe estar en el regex"


def test_fix_a_splits_conditional_cooking():
    import graph_orchestrator as g
    meal = {"name": "Pina con Semillas", "recipe": [
        "Mise en place: Corta la piña en bastones. Si las semillas de calabaza están crudas, "
        "tuéstalas en una sartén seca por 2 minutos hasta que estén doradas.",
        "Montaje: Coloca los bastones y espolvorea las semillas."]}
    assert g._split_cooking_from_mise(meal) is True, "la cocción condicional del Mise debe separarse a un TdF"
    assert any(str(s).strip().lower().startswith("el toque de fuego") for s in meal["recipe"])
    assert g._recipe_step_contract_issues(meal) == [], "sin 'falta Toque de Fuego' → sin advisory 'pasos incompletos'"


def test_fix_a_pure_nocook_not_split():
    """Regresión: un plato sin cocción real (yogurt+fruta) NO debe inventar un TdF."""
    import graph_orchestrator as g
    meal = {"name": "Yogurt con Fruta", "recipe": [
        "Mise en place: Corta la fruta en cubos.", "Montaje: Sirve el yogurt con la fruta encima."]}
    assert g._split_cooking_from_mise(meal) is False


# ═══════════════════ Fix B: micro worst-day min-2 ═══════════════════

def _plan_wd(low):
    return {"micronutrient_report": {"gaps": [],
            "per_day_floors": {"flagged": True, "worst_day": {"day_index": 0, "low": list(low)}}}}


@pytest.fixture()
def go(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "MICRONUTRIENT_SOFT_REJECT_ENABLED", False)
    monkeypatch.setattr(g, "MICRO_PERDAY_DEGRADE_ENABLED", True)
    monkeypatch.setattr(g, "MICRO_WORSTDAY_EXCLUDE_UNREACHABLE", True)
    return g


def test_fix_b_marker_and_knob():
    assert "P1-MICRO-WORSTDAY-MIN2" in _GO
    assert 'MICRO_WORSTDAY_MIN2 = _env_bool("MEALFIT_MICRO_WORSTDAY_MIN2", True)' in _GO
    assert "len(_wd_low) >= (2 if MICRO_WORSTDAY_MIN2 else 1)" in _GO


def test_fix_b_single_closeable_micro_does_not_degrade(go, monkeypatch):
    """El caso vivo: worst_day=[fiber,vit_e] → excluir vit_e → solo fibra (1) → NO degrada."""
    monkeypatch.setattr(go, "MICRO_WORSTDAY_MIN2", True)
    plan = _plan_wd(["fiber_g", "vit_e_mg"])
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert not plan.get("_quality_degraded"), "1 micro cerrable marginal → banner limpio en plan macro-perfecto"


def test_fix_b_two_closeable_micros_degrade(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_WORSTDAY_MIN2", True)
    plan = _plan_wd(["fiber_g", "calcium_mg", "vit_e_mg"])  # 2 cerrables tras excluir vit_e
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert plan.get("_quality_degraded") is True
    assert plan.get("_quality_degraded_reason") == "micro_worst_day"


def test_fix_b_knob_off_reverts_to_one(go, monkeypatch):
    monkeypatch.setattr(go, "MICRO_WORSTDAY_MIN2", False)
    plan = _plan_wd(["fiber_g", "vit_e_mg"])  # excluir vit_e → [fiber] (1); con MIN2 off, 1 basta
    go._maybe_mark_panel_degraded(plan, {}, False, 1)
    assert plan.get("_quality_degraded") is True, "knob OFF → basta 1 micro (comportamiento previo de FU4)"
