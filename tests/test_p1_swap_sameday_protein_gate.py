"""[P1-SWAP-SAMEDAY-PROTEIN-GATE · 2026-07-10] Suite de variedad para la superficie de
UPDATES (swaps/regen-day) — el análisis del plan vivo del owner (9bce8fff) con los
detectores oficiales mostró que los updates degradaban un plan bueno:

  1. 'huevo' en 2 comidas del Día 1 Y del Día 2 (estado que el reviewer de generación
     RECHAZARÍA) — los swaps solo tenían un hint SOFT que el LLM ignoraba.
  2. Camarones en almuerzo Y cena del mismo día — invisibles al gate (labels sin mariscos).
  3. "Avena Cremosa" en Día 1 Y Día 3 — el gate de plato-base solo comparaba contra el
     plato REEMPLAZADO, no contra los otros días.
  4. Cena aceptada de 1,037 kcal dejó el día en +18% (banda 0.667) sin corrección — el
     swap valida su slot, nadie re-miraba el DÍA.
  5. Con 40+ items en Nevera, el LLM gravitaba a los mismos 6-8 alimentos.

tooltip-anchor: P1-SWAP-SAMEDAY-PROTEIN-GATE
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Labels de mariscos en el gate (gap 2)
# ---------------------------------------------------------------------------

def test_shellfish_visible_to_same_day_gate():
    from graph_orchestrator import _protein_gate_labels_in_text, _SAME_DAY_PROTEIN_GATE_LABELS
    assert "camarones" in _SAME_DAY_PROTEIN_GATE_LABELS, (
        "P1-VARIETY-SHELLFISH-LABELS: camarones ×2 el mismo día volvió a ser invisible "
        "(caso vivo: Camarones Guisados almuerzo + Tortilla de Camarones cena, Día 1)"
    )
    assert _protein_gate_labels_in_text("camarones guisados al ajillo con casabe") == {"camarones"}
    # 'tortilla' NO mapea a huevo por diseño (tortilla de harina existe); el huevo del caso
    # vivo entró por los INGREDIENTES del blob, no por el nombre:
    assert _protein_gate_labels_in_text("tortilla de camarones 2 huevos 40 g de queso") >= {"camarones", "huevo"}
    assert "camarones" in _protein_gate_labels_in_text("150 g de langostinos frescos")


# ---------------------------------------------------------------------------
# 2. Gate determinista same-day en el swap (gap 1)
# ---------------------------------------------------------------------------

def test_sameday_protein_gate_wired_in_swap():
    i = _AGENT.find("P1-SWAP-SAMEDAY-PROTEIN-GATE")
    assert i > 0, "el gate determinista same-day desapareció del swap (el hint soft NO basta)"
    blk = _AGENT[i: i + 3400]
    assert "_protein_gate_labels_in_text" in blk, "mismo SSOT del detector oficial (labels+aliases+word-boundary)"
    assert "same_day_other_meal_blobs" in blk, (
        "el gate necesita BLOBS (nombre+ingredientes) — con solo nombres el huevo embebido "
        "en 'Panqueques' era invisible"
    )
    assert "SWAP_SAMEDAY_PROTEIN" in blk and "raise ValueError" in blk, "retryable con directiva"
    assert 'os.environ.get("MEALFIT_SWAP_SAMEDAY_PROTEIN_GATE", "true")' in blk, "knob default ON"


def test_producers_send_blobs():
    # single-swap: el helper del router devuelve (names, blobs) y el endpoint inyecta ambos
    assert "same_day_other_meal_blobs" in _PLANS
    i_fn = _PLANS.find("def _same_day_other_meals_for_swap")
    assert "return [], []" in _PLANS[i_fn: i_fn + 1200], "fail-open ahora es tupla ([], [])"
    # regen-day: meal_form trae los blobs del estado ACTUAL del día + los retira en el fallback
    i_mf = _PLANS.find('"same_day_other_meal_blobs": [')
    assert i_mf > 0, "el meal_form del regen-day debe traer los blobs del día"
    assert '_form_relaxed.pop("same_day_other_meal_blobs", None)' in _PLANS, (
        "el fallback de factibilidad del regen-day debe retirar el gate (nunca slots imposibles)"
    )


# ---------------------------------------------------------------------------
# 3. Gate cross-día de plato-base (gap 4)
# ---------------------------------------------------------------------------

def test_crossday_base_gate_in_swap():
    i = _AGENT.find("P1-SWAP-CROSSDAY-BASE-GATE")
    assert i > 0, (
        "P1-SWAP-CROSSDAY-BASE-GATE: el gate de base volvió a comparar solo contra el plato "
        "reemplazado — 'Avena Cremosa' en Día 1 Y Día 3 (screenshot del owner)."
    )
    blk = _AGENT[i: i + 1800]
    assert "_cross_day_names" in blk and "_cross_bases_br" in blk
    # regen-day puebla cross_day_meal_names (mismo slot, otros días) para este gate
    assert '"cross_day_meal_names": [' in _PLANS
    assert '_form_relaxed.pop("cross_day_meal_names", None)' in _PLANS


# ---------------------------------------------------------------------------
# 4. Re-cuadre del día post swap-persist (gap 3)
# ---------------------------------------------------------------------------

def test_day_band_resquare_after_swap_persist():
    i = _PLANS.find("P1-SWAP-PERSIST-DAY-BAND")
    assert i > 0, (
        "P1-SWAP-PERSIST-DAY-BAND: el swap-persist volvió a dejar el DÍA sin re-mirar "
        "(cena de 1,037 kcal → día +18% sin corrección, caso vivo)."
    )
    blk = _PLANS[i: i + 2600]
    assert "apply_update_macro_engine" in blk, "motor SSOT de updates sobre view 1-día"
    assert '"days": [day]' in blk, "view 1-día (mismos dicts → mutación llega al persist)"
    assert "_sync_recipe_step_quantities" in blk, "re-sync de pasos tras re-escalar porciones"
    assert "0.99" in blk, "solo re-cuadra si el día quedó fuera de banda (idempotente si ya está)"
    assert 'os.environ.get("MEALFIT_SWAP_PERSIST_DAY_BAND", "true")' in blk, "knob default ON"


# ---------------------------------------------------------------------------
# 5. Rotación de despensa (variedad rica desde la Nevera) (gap 5)
# ---------------------------------------------------------------------------

def test_pantry_rotation_hint_in_swap_prompt():
    i = _AGENT.find("P1-SWAP-PANTRY-ROTATION")
    assert i > 0, (
        "P1-SWAP-PANTRY-ROTATION: el prompt del swap perdió la rotación de despensa — con "
        "40+ items el LLM gravita a los mismos 6-8 (avena/guineo/queso en loop)."
    )
    blk = _AGENT[i: i + 2200]
    assert "AÚN NO se usan" in blk or "AUN NO se usan" in blk
    assert "_extract_clean_name_from_display_string" in blk, "nombres limpios desde display strings de pantry"
    assert "same_day_other_meal_blobs" in blk and "_cross_day_names" in blk, (
        "lo 'usado' considera el plato actual + el día + el mismo slot de otros días"
    )
