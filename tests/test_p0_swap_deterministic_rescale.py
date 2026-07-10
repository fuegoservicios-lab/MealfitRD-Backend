"""[P0-SWAP-DETERMINISTIC-RESCALE · 2026-07-10] El swap deja de pedirle aritmética al LLM:
el solver per-ingrediente de la generación re-escala el candidato al target del slot ANTES
del guardrail de macros. + [P1-SWAP-COHERENCE-REPAIR] repara menciones no-listadas (pantry-
seguro) antes de rechazar. + [P2-SWAP-COST-INSTRUMENTATION] include_raw=True → usage real a
`llm_usage_events` (la superficie era invisible al cost-by-node).

Root cause (regenerate-day EN VIVO 2026-07-10 13:20-13:24, corr=8ce66cae, plan 9bce8fff):
4/4 slots agotaron 2 fases × 3 intentos y quedaron "slot conservado" — cero platos
entregados tras ~4 min y ~20 calls LLM sin fila de costo. Taxonomía medida:
  - 6+ rechazos por drift de macros (el guardrail exige ±15%/macro; candidato con kcal
    +1.2% murió por proteína +31% — un re-escalado per-línea lo habría entregado).
  - 4 rechazos por coherencia receta↔ingredientes ('dorado'/'pepino'/'tostada'/'arroz+
    guineítos+coliflor') — reparables añadiendo la línea (el solver re-escala después).
  - El log "Usando Plato Fallback" anunciaba un fallback que el default NO emite
    (P3-SWAP-LLM-RETRIES-422) → forensic confuso.

tooltip-anchor: P0-SWAP-DETERMINISTIC-RESCALE
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _swap_fn_region() -> str:
    i = _AGENT.index("def invoke_with_retry():")
    return _AGENT[i: i + 40_000]


# ---------------------------------------------------------------------------
# 1. Solver determinista ANTES del guardrail de macros
# ---------------------------------------------------------------------------

def test_rescale_block_wired_before_macros_guardrail():
    region = _swap_fn_region()
    i_rescale = region.find("P0-SWAP-DETERMINISTIC-RESCALE")
    assert i_rescale > 0, (
        "P0-SWAP-DETERMINISTIC-RESCALE: falta el bloque de re-escalado determinista en "
        "invoke_with_retry. Sin él, el guardrail de macros vuelve a pedirle aritmética "
        "de ±15% al LLM y a quemar retries (4/4 slots conservados el 2026-07-10)."
    )
    i_guardrail = region.find("[P1-SWAP-MACROS · 2026-05-22] Validación post-gen")
    assert i_guardrail > 0, "ancla del guardrail de macros desapareció"
    assert i_rescale < i_guardrail, (
        "el re-escalado debe correr ANTES del guardrail (repara → valida, no al revés)"
    )
    i_truthup = region.find("[P2-UPDATE-MACRO-TRUTHUP")
    assert 0 < i_truthup < i_rescale, (
        "el re-escalado corre DESPUÉS del truth-up (solver sobre números honestos)"
    )


def test_rescale_reuses_generation_solver_and_syncs_steps():
    region = _swap_fn_region()
    blk = region[region.find("P0-SWAP-DETERMINISTIC-RESCALE"):]
    blk = blk[:blk.find("[P1-SWAP-MACROS")]
    assert "_apply_macro_solver_to_meal" in blk, "debe reusar el solver SSOT de generación"
    assert "_truth_up_meal_macros_from_strings" in blk, "re-truth-up post-solver (números honestos)"
    assert "_sync_recipe_step_quantities" in blk, (
        "re-sync de menciones de cantidad en pasos post-solver (evita paso '150g' vs línea '180g')"
    )
    assert 'os.environ.get("MEALFIT_SWAP_DETERMINISTIC_RESCALE", "true")' in blk, (
        "knob de rollback sin redeploy, default ON"
    )


# ---------------------------------------------------------------------------
# 2. Coherence-repair pantry-seguro antes del reject
# ---------------------------------------------------------------------------

def test_coherence_repair_before_reject_and_pantry_safe():
    region = _swap_fn_region()
    i_repair = region.find("P1-SWAP-COHERENCE-REPAIR")
    assert i_repair > 0, (
        "P1-SWAP-COHERENCE-REPAIR: falta el repair determinista — cada mención no-listada "
        "volvería a costar un intento LLM completo."
    )
    i_reject = region.find("[P1-SWAP-RECIPE-COHERENCE] divergence detected", i_repair)
    assert i_reject > i_repair, "el repair debe intentarse ANTES del reject/log de divergencia"
    blk = region[i_repair:i_reject]
    assert "_pantry_blob" in blk and "clean_ingredients" in blk, (
        "PANTRY-SEGURO: en modo pantry solo se repara si el alimento está en la nevera "
        "(jamás inventamos compra en un swap pantry-constrained)"
    )
    assert "_validate_recipe_coh(meal_dump)" in blk, "re-validación tras el append (no confiar a ciegas)"
    assert 'os.environ.get("MEALFIT_SWAP_COHERENCE_REPAIR", "true")' in blk


# ---------------------------------------------------------------------------
# 3. Instrumentación de costo (include_raw + emit usage)
# ---------------------------------------------------------------------------

def test_swap_llm_usage_instrumented():
    assert "with_structured_output(MealModel, include_raw=True)" in _AGENT, (
        "P2-SWAP-COST-INSTRUMENTATION: include_raw=True es lo que expone usage_metadata; "
        "sin él las calls del swap son invisibles en llm_usage_events (medido 2026-07-10: "
        "~20 calls de un regenerate-day con CERO filas de costo)."
    )
    region = _swap_fn_region()
    assert "_emit_llm_usage_event_best_effort" in region, "emit del usage al invoke del swap"
    assert 'node="swap_meal"' in region, "la fila debe filtrar por node='swap_meal' en cost-by-node"
    assert "SWAP_PARSE_ERROR" in region, (
        "parsed=None (parse error) debe ser ValueError retryable de guardrail — antes contaba "
        "como CB failure con el proveedor sano"
    )


# ---------------------------------------------------------------------------
# 4. Log honesto del fallback
# ---------------------------------------------------------------------------

def test_fallback_log_is_honest():
    assert "Fallaron los intentos LLM y validador: {e}. Usando Plato Fallback." not in _AGENT, (
        "P2-SWAP-HONEST-LOG: el log viejo anunciaba un fallback que el default no emite "
        "(P3-SWAP-LLM-RETRIES-422 default OFF) — confundía el forensic."
    )
    assert "Usando Plato Fallback (knob MEALFIT_SWAP_EMIT_FALLBACK_DISH=true)" in _AGENT, (
        "el mensaje de fallback debe vivir SOLO en la rama que sí lo emite"
    )


# ---------------------------------------------------------------------------
# 5. regenerate-day: error_code honesto (no siempre culpar a la Nevera)
# ---------------------------------------------------------------------------

def test_regen_day_error_code_is_cause_aware():
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    i = src.find("P2-REGEN-DAY-HONEST-CODE")
    assert i > 0, (
        "P2-REGEN-DAY-HONEST-CODE: el 'regenerated == 0' volvió al blanket "
        "pantry_insufficient_for_goal — el toast mandaría al usuario a comprar "
        "ingredientes aunque la causa fuera el guardrail del LLM (Nevera llena, "
        "visto en vivo 2026-07-10)."
    )
    assert "_kept_reasons" in src, "el loop debe coleccionar la causa real de cada slot conservado"
    assert '"ai_exhausted_retries"' in src, (
        "cuando ningún slot falló por despensa, el code honesto es ai_exhausted_retries "
        "(copy 'reintenta', no 'agrega ítems')"
    )
    j = src.find("_pantry_reason = any(")
    assert j > 0, "el clasificador _pantry_reason desapareció del branch regenerated==0"
    blk = src[j: j + 1200]
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY" in blk and "ERRORES DE DESPENSA" in blk, (
        "la clasificación pantry debe basarse en los markers reales de los ValueError del swap"
    )
