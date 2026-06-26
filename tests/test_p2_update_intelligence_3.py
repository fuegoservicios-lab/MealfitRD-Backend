"""[P2-UPDATE-INTELLIGENCE-3 · 2026-06-24] Regresión de los P2 de la 2da re-auditoría independiente de
inteligencia (los 8 P2). P2-1 (macro truth-up) tiene su propio archivo test_p2_update_macro_truthup.py.

  P2-2 MICROS-STALE: recompute del panel de micros en swap-persist (_swap_mutator) y chat-modify
       (_apply_meal_modification), hidratando _micro_form server-side. Knob MEALFIT_UPDATE_RECOMPUTE_MICROS.
  P2-3 LEDGER-LEAK: regenerate-day decrementa el ledger D7 también por platos CONSERVADOS (ambas ramas
       de keep), no solo regenerados. Knob MEALFIT_REGEN_DAY_LEDGER_KEEP_DECREMENT.
  P2-4 CHARGE-ON-SUCCESS: api_swap_meal cobra el crédito DESPUÉS del éxito + catch del breaker/rate-limit
       (soft-fail swap_ai_unavailable, sin cobro). Knob MEALFIT_SWAP_CHARGE_ON_SUCCESS_ONLY.
  P2-5 SWAP-RECALC-RETRY: regenerateSingleMeal (frontend) retrofitea el loop 2-intentos + toast honesto.
  P2-6 CHATMODIFY-DISLIKES: chat-modify inyecta los dislikes como prohibición dura. Knob MEALFIT_UPDATE_HYDRATE_DISLIKES.
  P2-7 CHATMODIFY-GAINMUSCLE: chat-modify inyecta directiva de alta densidad para gain_muscle (_LOW_DENSITY_AS_MAIN).
  P2-8 WARN-MULTI-AXIS: regenerate-day day_quality_warning audita kcal/carbs además de proteína.

Parser-based (corre local + CI con `py`/venv).
"""
import ast
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND)


def _read(rel, base=BACKEND):
    with open(os.path.join(base, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
TOOLS = _read("tools.py")
PLANS = _read("routers/plans.py")
ORCH = _read("graph_orchestrator.py")
APP = _read("app.py")
ASSESS = _read(os.path.join("frontend", "src", "context", "AssessmentContext.jsx"), base=ROOT)


# ── P2-2: micros recompute en swap-persist + chat-modify ──────────────────────
def test_p2_2_micros_recompute_swap_persist():
    src = _func_src(PLANS, "api_swap_meal_persist")
    assert "recompute_micronutrient_report_for_plan" in src, "swap-persist debe recomputar micros"
    assert "_micro_form" in src and "get_user_profile" in src, "debe hidratar biométricos server-side (no form vacío → DRI embarazo correcto)"
    assert "P2-SWAP-MICROS-STALE" in src


def test_p2_2_micros_recompute_chatmodify():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "recompute_micronutrient_report_for_plan" in src, "chat-modify debe recomputar micros (persiste él mismo)"
    assert "P2-CHATMODIFY-MICROS-STALE" in src


# ── P2-3: ledger decrement en platos conservados ──────────────────────────────
def test_p2_3_ledger_decrement_on_kept_meals():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "_ledger_keep_decrement_on" in src, "debe gatear el decrement-on-keep por knob"
    assert "MEALFIT_REGEN_DAY_LEDGER_KEEP_DECREMENT" in src
    # el decrement por kept-meal aparece en AMBAS ramas de keep (else + ValueError) → ≥2 callsites
    # del marker, además del callsite del plato regenerado.
    assert src.count("P2-REGEN-DAY-LEDGER-LEAK") >= 2, "ambas ramas de keep deben decrementar el ledger"
    assert src.count("_decrement_ledger_by_meal(ledger, meal, _db)") >= 2, "decrement en else + ValueError"


# ── P2-4: swap cobra post-éxito + catch del breaker ───────────────────────────
def test_p2_4_swap_charges_after_success():
    src = _func_src(PLANS, "api_swap_meal")
    assert "MEALFIT_SWAP_CHARGE_ON_SUCCESS_ONLY" in src
    assert "result = swap_meal(data)" in src
    # el cobro post-éxito debe aparecer DESPUÉS de la llamada a swap_meal
    swap_call = src.index("result = swap_meal(data)")
    last_charge = src.rfind('log_api_usage(user_id, "llm_swap_meal")')
    assert last_charge > swap_call, "el cobro debe ocurrir tras swap_meal (no antes)"


def test_p2_4_swap_catches_llm_unavailable():
    src = _func_src(PLANS, "api_swap_meal")
    assert "except (LLMRateLimitedError, LLMCircuitBreakerOpen)" in src, "debe capturar caída del proveedor"
    assert "swap_ai_unavailable" in src
    assert "No se descontó tu crédito" in src
    # imports a nivel módulo
    assert "from agent import analyze_preferences_agent, swap_meal, LLMRateLimitedError, LLMCircuitBreakerOpen" in PLANS


# ── P2-6 + P2-7: chat-modify dislikes + densidad gain_muscle ───────────────────
def test_p2_6_chatmodify_injects_dislikes():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "P2-CHATMODIFY-DISLIKES" in src
    assert "MEALFIT_UPDATE_HYDRATE_DISLIKES" in src
    assert "DISGUSTOS" in src, "debe inyectar la prohibición dura de disgustos al prompt"


def test_p2_7_chatmodify_gainmuscle_density():
    src = _func_src(TOOLS, "execute_modify_single_meal")
    assert "P2-CHATMODIFY-GAINMUSCLE-DENSITY" in src
    assert "_LOW_DENSITY_AS_MAIN" in src, "debe reusar el set SSOT de proteínas de baja densidad"
    assert "MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN" in src
    assert "gain_muscle" in src


# ── P2-8: day_quality_warning multi-eje ───────────────────────────────────────
def test_p2_8_day_warning_multi_axis():
    src = _func_src(PLANS, "api_regenerate_day")
    assert "MEALFIT_REGEN_DAY_WARN_MULTI_AXIS" in src
    assert "P2-REGEN-DAY-WARN-MULTI-AXIS" in src
    assert "_new_kcal" in src, "debe auditar calorías además de proteína"
    # carbos solo para gain_muscle/bulk (evita ruido en pérdida de peso)
    assert '("gain_muscle", "bulk")' in src or "'gain_muscle', 'bulk'" in src


# ── P2-5: frontend swap recalc retry + toast honesto ──────────────────────────
def test_p2_5_frontend_swap_recalc_retry():
    assert "P5-SWAP-RECALC-RETRY" in ASSESS, "regenerateSingleMeal debe retrofitear el loop de reintentos"
    assert "_swapRecalcOnce" in ASSESS
    assert "Tu lista de compras se está actualizando" in ASSESS, "toast honesto si el recalc falla"
    # el catch silencioso (console.error) del recalc post-swap debe haberse eliminado
    assert "Error recalculando lista de compras post-swap (tras retry)" not in ASSESS


# ── marker freshness ──────────────────────────────────────────────────────────
def test_last_known_pfix_bumped():
    # [de-pin · 2026-06-26] `_LAST_KNOWN_PFIX` es single-valued → pinear "P2-UPDATE-INTELLIGENCE-3"
    # quedó stale apenas un P-fix posterior bumpeó el marker. Contrato durable del bump:
    # test_p3_1_last_known_pfix_freshness (formato + floor) + test_p2_hist_audit_14_marker_test_link.
    assert re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P\d+-[A-Z0-9-]+ · \d{4}-\d{2}-\d{2}"', APP), \
        "_LAST_KNOWN_PFIX debe existir con formato `Pn-... · YYYY-MM-DD`"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
