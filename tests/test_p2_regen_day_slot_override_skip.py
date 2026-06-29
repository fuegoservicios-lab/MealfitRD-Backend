"""[P2-REGEN-DAY-SLOT-OVERRIDE-SKIP + P2-REGEN-DAY-BAND-WARN · 2026-06-29] (re-audit testing en vivo)

Bug observado en logs de regenerate-day (corr c2d072ff): el día regenerado salió con band_score=0.0
(TODOS los macros fuera de banda) y SIN aviso honesto. Dos causas:

1. CONFLICTO de targets: `P1-REGEN-DAY-RETARGET` calcula targets per-comida hacia el objetivo del día
   (contra el target REAL del plan, ~2141 kcal) y los pasa en `target_*`. Pero `P2-8-SWAP-SLOT-TARGET`
   (ON en prod) los RE-DERIVA con `get_nutrition_targets(meal_form)` — y el meal_form de regen NO trae
   biométricos (weight/height/age) → cae a defaults (154lb/170/25 → ~2949 kcal) → sobre-asigna cada slot
   → el día sobre-entrega → band 0.0. Fix: regen pasa `_skip_slot_target_override=True`; el slot-override
   lo respeta (el swap standalone, con biométricos en el request, lo conserva).

2. AVISO solo-déficit: el aviso honesto era solo bajo-target (proteína <90% / kcal-carbs <85%). Un día
   fuera de banda por SOBRE-entrega o mixto (band bajo sin déficit) se entregaba callado. Fix: si
   `score_macros_only < umbral` y no hubo ya un aviso, avisa honesto (dirección-agnóstico).

Test parser-based (Neon-free): ancla el flag + el gate del override + el aviso band-based.
"""
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


# ── #2: skip del slot-override en regen-day ──
def test_swap_meal_gate_respects_skip_flag():
    # El gate de P2-8-SWAP-SLOT-TARGET debe incluir `and not form_data.get("_skip_slot_target_override")`.
    assert 'and not form_data.get("_skip_slot_target_override")' in _AGENT
    assert "P2-REGEN-DAY-SLOT-OVERRIDE-SKIP" in _AGENT


def test_regen_day_sets_skip_flag_in_meal_form():
    # El meal_form de regen-day debe pasar el flag para no re-derivar el target sin biométricos.
    seg = _PLANS[_PLANS.find("def _day_mutator") - 6000: _PLANS.find("def _day_mutator")]  # cuerpo del loop, antes del mutator
    # (el meal_form se construye en el loop `for meal in meals:` antes del mutator)
    assert '"_skip_slot_target_override": True' in _PLANS, "regen-day debe señalizar skip del slot-override"
    assert "P2-REGEN-DAY-SLOT-OVERRIDE-SKIP" in _PLANS


def test_skip_flag_lives_near_regen_targets():
    """El flag debe estar en el bloque del meal_form junto a los target_* retargeteados (no suelto)."""
    idx_flag = _PLANS.find('"_skip_slot_target_override": True')
    idx_tc = _PLANS.find('"target_calories": round(float(meal.get("cals")')
    assert idx_flag != -1 and idx_tc != -1
    assert abs(idx_flag - idx_tc) < 1500, "el flag debe vivir en el meal_form de regen, junto a los targets"


# ── #1: aviso honesto gated por band-score ──
def test_band_warn_block_present():
    assert "P2-REGEN-DAY-BAND-WARN" in _PLANS
    assert "score_macros_only" in _PLANS
    assert "MEALFIT_REGEN_DAY_BAND_WARN_THRESHOLD" in _PLANS


def test_band_warn_only_when_no_prior_warning():
    """El aviso band-based NO debe pisar un aviso de déficit ya existente (es complementario)."""
    idx = _PLANS.find("P2-REGEN-DAY-BAND-WARN")
    region = _PLANS[idx - 200: idx + 1400]
    assert "_day_warning is None" in region, "el aviso band-based solo aplica si no hubo déficit ya avisado"
    assert "_day_warning = " in region, "debe asignar _day_warning (que va a day_quality_warning en la respuesta)"


def test_band_warn_after_band_score_computed():
    """El aviso band-based debe ir DESPUÉS del cómputo del band_score (lo consume)."""
    idx_score = _PLANS.find("[P2-REGEN-DAY-BAND-SCORE]")
    idx_warn = _PLANS.find("P2-REGEN-DAY-BAND-WARN")
    assert idx_score != -1 and idx_warn != -1 and idx_score < idx_warn
