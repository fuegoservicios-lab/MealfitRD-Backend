"""[P1-RECIPE-EXPAND-FAILSIGNAL · 2026-05-30] Audit profundo de Recetas.

Contrato anclado (parser-based, corre DB-less con --noconftest):

  #1/#9 expand failure-signal: `expand_recipe_agent` devuelve `None` en fallo
        (NO la receta original) y solo acepta listas no-vacías con pasos string.
        El endpoint `/recipe/expand`, ante `None`, devuelve `success=False` +
        `expansion_failed=True`, SIN marcar isExpanded, SIN persistir, SIN cobrar.
  #2    dedup pre-check usa `isinstance(..., list)` (era `str` → código muerto).
  #3    `log_api_usage("llm_recipe_expand")` se invoca DESPUÉS de
        `expand_recipe_agent` (cobro tras éxito, no antes).
  #10   `expand_recipe_agent` usa el knob `_recipe_expand_model_name()`.
  #4    macro-balancing (`assemble_plan_node`) append el disclaimer como
        ELEMENTO de lista (`recipe_list + [disclaimer]`), NO stringifica.
  #6    `_build_filtered_edge_recipe_day` emite claves CANÓNICAS
        (`meal`/`desc`/`recipe`/`cals`), no `type`/`description`/`instructions`.
  #5    swap resetea `isExpanded` (frontend AssessmentContext + backend mutator).
  frontend: coerción `toRecipeSteps` defensiva contra `recipe` string.

Si un rename rompe un tooltip-anchor, este test falla ANTES que producción.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _ROOT / "backend"
_AI_HELPERS = _BACKEND / "ai_helpers.py"
_PLANS = _BACKEND / "routers" / "plans.py"
_ORCH = _BACKEND / "graph_orchestrator.py"
_CRON = _BACKEND / "cron_tasks.py"
_RECIPES_JSX = _ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"
_ASSESS_CTX = _ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"


@pytest.fixture(scope="module")
def ai_src() -> str:
    return _AI_HELPERS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS.read_text(encoding="utf-8")


def _fn_body(src: str, fn_name: str) -> str:
    start = src.find(f"def {fn_name}(")
    assert start > 0, f"{fn_name} no encontrada"
    after = src[start:]
    # Hasta el siguiente def/@router a col 0 o decorador @router.
    m = re.search(r"\n(?:@router\.|def |async def )", after[1:])
    end = (1 + m.start()) if m else len(after)
    return after[:end]


def _expand_body(src: str) -> str:
    start = src.find("def api_expand_recipe(")
    assert start > 0
    after = src[start + 30:]
    nm = re.search(r"\n@router\.", after)
    end = (start + 30 + nm.start()) if nm else (start + 30 + len(after))
    return src[start:end]


# ---------------------------------------------------------------------------
# #1/#9 — expand_recipe_agent señaliza fallo con None
# ---------------------------------------------------------------------------
def test_expand_agent_returns_none_on_failure(ai_src: str):
    body = _fn_body(ai_src, "expand_recipe_agent")
    # El except y el fallback devuelven None (NO la receta original).
    assert "return None" in body, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: expand_recipe_agent ya no "
        "devuelve None en fallo. Sin señal de fallo, el endpoint marca "
        "isExpanded=True con la receta original y el frontend nunca reintenta."
    )
    # NO debe devolver la receta original como 'éxito' silencioso.
    assert "return meal_data.get(\"recipe\"" not in body and \
           "return meal_data.get('recipe'" not in body, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: expand_recipe_agent volvió a "
        "devolver `meal_data.get('recipe')` (eco silencioso de la original)."
    )
    assert "P1-RECIPE-EXPAND-FAILSIGNAL-AGENT" in body, "tooltip-anchor ausente"


def test_expand_agent_validates_nonempty_string_steps(ai_src: str):
    body = _fn_body(ai_src, "expand_recipe_agent")
    # Acepta solo lista; filtra pasos string no-blank.
    assert "isinstance(steps, list)" in body
    assert "s.strip()" in body, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: ya no se filtran pasos "
        "string no-blank antes de aceptar la expansión (gap schema #9)."
    )


def test_expand_agent_uses_model_knob(ai_src: str):
    assert "_recipe_expand_model_name" in ai_src, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: falta el helper "
        "_recipe_expand_model_name (knob del modelo)."
    )
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] default = constante DEEPSEEK_FLASH.
    assert re.search(
        r'_env_str\(\s*"MEALFIT_RECIPE_EXPAND_MODEL"\s*,\s*DEEPSEEK_FLASH',
        ai_src,
    ), "El knob debe leer MEALFIT_RECIPE_EXPAND_MODEL con default DEEPSEEK_FLASH."
    body = _fn_body(ai_src, "expand_recipe_agent")
    assert "model=_recipe_expand_model_name()" in body, (
        "expand_recipe_agent debe usar model=_recipe_expand_model_name()."
    )
    assert not re.search(r'model\s*=\s*"[a-z0-9.-]+"', body), (
        "expand_recipe_agent volvió a hardcodear el modelo."
    )


# ---------------------------------------------------------------------------
# #1/#3 — endpoint: fallo no marca/persiste/cobra; cobro tras éxito
# ---------------------------------------------------------------------------
def test_endpoint_handles_failure_without_charge_or_persist(plans_src: str):
    body = _expand_body(plans_src)
    assert "if not expanded_steps:" in body, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: el endpoint ya no gatea el "
        "caso fallo (`if not expanded_steps:`) — un None se persistiría/cobraría."
    )
    assert '"expansion_failed": True' in body
    assert '"success": False' in body, (
        "El path de fallo debe devolver success=False para que el frontend "
        "abra la receta original SIN marcar isExpanded (permite retry)."
    )


def test_endpoint_charges_quota_after_llm(plans_src: str):
    body = _expand_body(plans_src)
    llm_idx = body.find("expanded_steps = expand_recipe_agent(data)")
    # El log_api_usage de cobro (path LLM) debe venir DESPUÉS del LLM call.
    log_idx = body.rfind('log_api_usage(user_id, "llm_recipe_expand")')
    assert llm_idx > 0 and log_idx > 0
    assert log_idx > llm_idx, (
        "P1-RECIPE-EXPAND-FAILSIGNAL regresión: log_api_usage (cobro) ya no "
        "está DESPUÉS de expand_recipe_agent — un fallo cobraría un crédito "
        "del paywall sin entregar receta."
    )
    # Y el caso fallo retorna ANTES de ese cobro.
    fail_idx = body.find("if not expanded_steps:")
    assert 0 < fail_idx < log_idx


# ---------------------------------------------------------------------------
# #2 — dedup usa isinstance(..., list)
# ---------------------------------------------------------------------------
def test_dedup_uses_list_type_check(plans_src: str):
    body = _expand_body(plans_src)
    assert 'isinstance(existing_meal.get("recipe"), list)' in body, (
        "P2-RECIPE-DEDUP-LIST regresión: el dedup volvió a chequear "
        "`isinstance(..., str)` (siempre False → dedup muerto)."
    )
    assert 'isinstance(existing_meal.get("recipe"), str)' not in body
    assert "P2-RECIPE-DEDUP-LIST" in body, "tooltip-anchor ausente"


# ---------------------------------------------------------------------------
# #4 — disclaimer como elemento de lista (no stringificar)
# ---------------------------------------------------------------------------
def test_macro_disclaimer_appends_list_element():
    src = _ORCH.read_text(encoding="utf-8")
    assert "P2-RECIPE-DISCLAIMER-LIST" in src, "tooltip-anchor ausente"
    # Append como lista en AMBOS sitios (soft + strict balancing).
    assert len(re.findall(r"recipe_list\s*\+\s*\[disclaimer\]", src)) >= 2, (
        "P2-RECIPE-DISCLAIMER-LIST regresión: el disclaimer ya no se append "
        "como elemento de lista en los 2 sitios de macro-balancing."
    )
    # NO debe re-stringificar la receta concatenando el disclaimer a un str.
    assert "recipe_text + disclaimer" not in src, (
        "P2-RECIPE-DISCLAIMER-LIST regresión: reapareció la stringificación "
        "`recipe_text + disclaimer` (rompe el schema List[str] → fallback)."
    )


# ---------------------------------------------------------------------------
# #6 — edge recipe claves canónicas
# ---------------------------------------------------------------------------
def test_edge_recipe_canonical_keys():
    src = _CRON.read_text(encoding="utf-8")
    body = _fn_body(src, "_build_filtered_edge_recipe_day")
    assert "P2-EDGE-RECIPE-CANONICAL-KEYS" in body, "tooltip-anchor ausente"
    # Claves canónicas presentes.
    for key in ('"meal":', '"desc":', '"recipe":', '"cals":'):
        assert key in body, (
            f"P2-EDGE-RECIPE-CANONICAL-KEYS regresión: falta clave canónica "
            f"{key} en el edge-day → frontend renderiza en blanco."
        )
    # Claves NO-canónicas eliminadas.
    for bad in ('"type":', '"description":', '"instructions":'):
        assert bad not in body, (
            f"P2-EDGE-RECIPE-CANONICAL-KEYS regresión: reapareció la clave "
            f"no-canónica {bad} (el frontend no la mapea)."
        )


# ---------------------------------------------------------------------------
# #5 — swap resetea isExpanded (frontend + backend defensa)
# ---------------------------------------------------------------------------
def test_swap_resets_isexpanded_backend(plans_src: str):
    body = _fn_body(plans_src, "api_swap_meal_persist")
    assert 'new_meal["isExpanded"] = False' in body, (
        "P2-SWAP-RESET-ISEXPANDED regresión: el _swap_mutator ya no fuerza "
        "isExpanded=False → un plato swapeado heredaría la receta 'expandida'."
    )
    assert "P2-SWAP-RESET-ISEXPANDED" in body, "tooltip-anchor ausente"


def test_swap_resets_isexpanded_frontend():
    src = _ASSESS_CTX.read_text(encoding="utf-8")
    assert "isExpanded: false" in src, (
        "P2-SWAP-RESET-ISEXPANDED regresión: el override del swap en "
        "AssessmentContext ya no resetea isExpanded:false."
    )


# ---------------------------------------------------------------------------
# frontend — coerción defensiva recipe->array
# ---------------------------------------------------------------------------
def test_frontend_recipe_coercion_helper():
    src = _RECIPES_JSX.read_text(encoding="utf-8")
    assert "toRecipeSteps" in src, (
        "P2-RECIPE-DISCLAIMER-LIST regresión: falta el helper defensivo "
        "toRecipeSteps en Recipes.jsx (coerción recipe string→array)."
    )
    # El render y el PDF usan la coerción, no `.recipe.map` crudo en el render.
    assert "activeRecipeSteps.map(" in src
    assert "toRecipeSteps(meal.recipe)" in src
