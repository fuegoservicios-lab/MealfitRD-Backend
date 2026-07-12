"""[P3-SWAP-LLM-RETRIES-422 · 2026-05-23] Cuando el LLM del swap-meal
agota retries sin pasar validators, el comportamiento legacy era armar
un "Plato Fallback" con clean_ingredients[:4] que el frontend mostraba
al usuario como un plato real. Pero ese plato NO era coherente:

  - Título sintético: "Merienda con Cilantro y Aceite de oliva"
    (no es una merienda válida — el chef IA no inventó este plato).
  - Receta placeholder: "Mise en place / El toque de fuego / Montaje"
    — 3 pasos genéricos que el user NO puede cocinar.
  - Descripción genérica que NO menciona los ingredientes específicos.

Verificado en log productivo 2026-05-23 00:21-00:22: swap de "Batido
de Proteína con Piña Fresca" agotó 3 retries por "/pedazos de queso"
unauthorized, fallback engañoso entregado como éxito ("✅ COMPLETADO").

Fix estructural (opción B, recomendada al owner):

  1. ``agent.py::swap_meal`` cuando agota retries (NO strict-pantry
     vacío, que ya tiene su propio raise): raise
     ``ValueError("SWAP_LLM_RETRIES_EXHAUSTED: ...")`` en vez de armar
     el response engañoso.
  2. ``routers/plans.py::api_swap_meal`` mapea el error a HTTP 422 con
     ``detail.code='swap_llm_retries_exhausted'`` + copy amigable.
  3. ``AssessmentContext.jsx::handleSwapMeal`` catch del code → toast +
     ``return currentName`` (preserva plato original, NO setPlanData).
  4. Knob ``MEALFIT_SWAP_EMIT_FALLBACK_DISH=true`` revierte al
     comportamiento legacy (degradación graceful).

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_swap_llm_retries_422`` ↔ filename ``test_p3_swap_llm_retries_422.py``.
"""
import pathlib
import re

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
FRONTEND_ROOT = BACKEND_ROOT.parent / "frontend"

AGENT_PY = (BACKEND_ROOT / "agent.py").read_text(encoding="utf-8")
PLANS_PY = (BACKEND_ROOT / "routers" / "plans.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")
CONTEXT_JSX = (FRONTEND_ROOT / "src" / "context" / "AssessmentContext.jsx").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — agent.py raise instead of fallback dish (default behavior)
# ---------------------------------------------------------------------------

def test_agent_raises_swap_llm_retries_exhausted_by_default():
    """``swap_meal`` debe levantar ``ValueError("SWAP_LLM_RETRIES_EXHAUSTED: ...")``
    en el branch de fallback cuando el knob ``MEALFIT_SWAP_EMIT_FALLBACK_DISH``
    está en su default (false)."""
    assert "SWAP_LLM_RETRIES_EXHAUSTED" in AGENT_PY, (
        "Falta el raise `ValueError('SWAP_LLM_RETRIES_EXHAUSTED: ...')` "
        "en agent.py. Sin esto el backend sigue armando el fallback dish."
    )
    # El raise debe estar gateado por el knob (default false → raise)
    assert "MEALFIT_SWAP_EMIT_FALLBACK_DISH" in AGENT_PY, (
        "Falta el knob `MEALFIT_SWAP_EMIT_FALLBACK_DISH` — sin él el "
        "operador no puede revertir al comportamiento legacy sin redeploy."
    )


def test_knob_default_is_false_emits_422_path():
    """El knob ``MEALFIT_SWAP_EMIT_FALLBACK_DISH`` debe leerse con default
    ``"false"`` (raise → 422). Si el default fuera "true" el comportamiento
    legacy seguiría activo."""
    # Match sobre el read del env var con default explícito
    m = re.search(
        r'MEALFIT_SWAP_EMIT_FALLBACK_DISH[^\n]*?["\']false["\']',
        AGENT_PY,
    )
    assert m, (
        "Knob `MEALFIT_SWAP_EMIT_FALLBACK_DISH` debe leer con default "
        "'false' (raise → 422). Default 'true' mantiene el bug original."
    )


def test_raise_precedes_fallback_construction():
    """El raise debe estar ANTES del bloque que construye `fallback_ing`
    y `response = {...}`. Si está después, el code es alcanzable pero
    el raise inalcanzable por orden."""
    # Encuentra ambas posiciones
    raise_pos = AGENT_PY.find('raise ValueError(\n                "SWAP_LLM_RETRIES_EXHAUSTED')
    fallback_assignment = AGENT_PY.find("fallback_ing = clean_ingredients[:4]")
    assert raise_pos > 0, "No se encontró el raise SWAP_LLM_RETRIES_EXHAUSTED"
    assert fallback_assignment > 0, "No se encontró `fallback_ing = ...`"
    assert raise_pos < fallback_assignment, (
        "El raise debe estar ANTES del assignment a `fallback_ing` "
        "para que el fallback solo se ejecute con knob ON."
    )


# ---------------------------------------------------------------------------
# Section B — Router maps to HTTP 422 with canonical code
# ---------------------------------------------------------------------------

def test_router_maps_swap_llm_retries_to_422():
    """``routers/plans.py::api_swap_meal`` debe mapear el nuevo error a
    payload con code canónico ``swap_llm_retries_exhausted``.

    [P3-SWAP-SOFT-FAIL-200 · 2026-05-23] Tras la migración a soft-fail,
    el code aparece como ``error_code`` en el payload 200 (default) O
    como ``code`` en el detail del 422 legacy (gateado por knob
    MEALFIT_SWAP_HARD_FAIL_HTTP_422). Aceptamos ambos.
    """
    assert 'SWAP_LLM_RETRIES_EXHAUSTED' in PLANS_PY, (
        "Router debe matchear el prefijo 'SWAP_LLM_RETRIES_EXHAUSTED'."
    )
    # Soft-fail path (default) O hard-fail path (knob)
    soft_match = '"error_code": "swap_llm_retries_exhausted"' in PLANS_PY
    hard_match = '"code": "swap_llm_retries_exhausted"' in PLANS_PY
    assert soft_match or hard_match, (
        "Router debe emitir code canónico 'swap_llm_retries_exhausted' "
        "en al menos uno de los dos paths."
    )
    # Sanity: el path 422 hard-fail sigue disponible bajo el knob
    branch_match = re.search(
        r"SWAP_LLM_RETRIES_EXHAUSTED.*?status_code=422",
        PLANS_PY,
        re.DOTALL,
    )
    assert branch_match, (
        "El path 422 hard-fail debe seguir disponible bajo el knob "
        "MEALFIT_SWAP_HARD_FAIL_HTTP_422=true (rollback compat)."
    )


def test_router_message_mentions_plato_preservado():
    """El copy emitido debe comunicar que el plato original se mantiene
    intacto. Post-soft-fail aparece como ``error_message`` en el payload."""
    # Buscar el mensaje del payload soft-fail O del detail 422 legacy
    branch_match = re.search(
        r'"(?:error_message|message)":\s*\(\s*("[^"]*"\s*)+\)',
        PLANS_PY,
        re.DOTALL,
    )
    assert branch_match, (
        "No se encontró el copy del payload swap_llm_retries_exhausted."
    )
    # Buscar específicamente el body que tiene 'mantiene' / 'reintenta' / 'razón'
    keywords = ["mantiene", "reintenta", "razón"]
    matches = sum(1 for kw in keywords if kw in PLANS_PY.lower())
    assert matches >= 1, (
        f"El copy debe mencionar al menos uno de {keywords} para que el "
        f"usuario entienda qué pasó."
    )


# ---------------------------------------------------------------------------
# Section C — Frontend discrimina el code y preserva plato
# ---------------------------------------------------------------------------

def test_frontend_catches_swap_llm_retries_exhausted_code():
    """``AssessmentContext.jsx::handleSwapMeal`` debe tener un branch
    que matchee ``error.code === 'swap_llm_retries_exhausted'`` y muestre
    toast + preserve plato (return currentName)."""
    assert "swap_llm_retries_exhausted" in CONTEXT_JSX, (
        "Frontend debe matchear el code canónico nuevo. Sin el branch, "
        "el catch genérico cae a `getAlternativeMeal` (fallback local), "
        "mostrando otro plato falso."
    )


def test_frontend_branch_does_not_call_fallback_for_new_code():
    """El branch del nuevo code NO debe invocar ``getAlternativeMeal``
    — debe `return currentName` para preservar el plato original."""
    # Line-based extraction (regex de balanceo no es confiable con
    # objetos JSX inline tipo `{ description: error.detailMessage }`).
    m = re.search(
        r"if\s*\(\s*error\?\.\s*status\s*===\s*422\s*&&\s*error\?\.\s*code\s*===\s*'swap_llm_retries_exhausted'\s*\)\s*\{\s*\n((?:[^\n]*\n){1,8})\s*\}",
        CONTEXT_JSX,
    )
    assert m, (
        "No se encontró el branch del code 'swap_llm_retries_exhausted' "
        "en AssessmentContext.jsx."
    )
    branch_body = m.group(0)
    assert "getAlternativeMeal" not in branch_body, (
        "El branch NO debe degradar a fallback local — debe preservar el "
        "plato original (return currentName)."
    )
    assert "toast.error" in branch_body, (
        "El branch debe disparar toast.error para que el user sepa qué pasó."
    )
    # [P2-SWAP-TOAST-FIX · 2026-06-29] la rama devuelve `null` (preserva plato + suprime
    # el success-toast engañoso del caller); anchor actualizado 2026-07-12 (estaba stale
    # en el pool baseline desde el cambio).
    assert "return null" in branch_body, (
        "El branch debe `return currentName` para que el caller no "
        "actualice planData."
    )


# ---------------------------------------------------------------------------
# Section D — Knob de rollback respetado (legacy behavior)
# ---------------------------------------------------------------------------

def test_knob_true_preserves_legacy_fallback_dish_path():
    """Cuando ``MEALFIT_SWAP_EMIT_FALLBACK_DISH=true``, el código del
    fallback dish (response = {...} con name/desc/recipe) debe seguir
    presente y alcanzable. Es la red de seguridad si el 422 introduce
    UX problemática en prod."""
    # El branch del fallback dish (response = {...} con "name": f"{meal_type}...")
    # debe seguir existiendo en agent.py
    assert re.search(
        r'"name":\s*f"\{meal_type\}\s+con\s+\{_title_ings\}"',
        AGENT_PY,
    ), (
        "El fallback dish legacy (response con title `{meal_type} con "
        "{_title_ings}`) debe seguir presente como knob-protected path. "
        "Si lo eliminas, el knob `MEALFIT_SWAP_EMIT_FALLBACK_DISH=true` "
        "queda sin efecto."
    )


# ---------------------------------------------------------------------------
# Section E — Cross-link con strict-pantry-no-inventory (mismo patrón)
# ---------------------------------------------------------------------------

def test_both_422_codes_use_same_router_pattern():
    """``swap_strict_pantry_no_inventory`` y ``swap_llm_retries_exhausted``
    deben seguir el mismo patrón estructural en el handler ValueError.
    Tras P3-SWAP-SOFT-FAIL-200, el code canónico aparece en payload
    soft-fail (``error_code``) O detail legacy 422 (``code``)."""
    # Ambos prefixes deben aparecer en el handler de ValueError
    assert "SWAP_STRICT_PANTRY_NO_INVENTORY" in PLANS_PY
    assert "SWAP_LLM_RETRIES_EXHAUSTED" in PLANS_PY
    # Ambos codes canónicos deben aparecer (en al menos uno de los paths)
    for canonical_code in ["swap_strict_pantry_no_inventory", "swap_llm_retries_exhausted"]:
        soft = f'"error_code": "{canonical_code}"' in PLANS_PY
        hard = f'"code": "{canonical_code}"' in PLANS_PY
        assert soft or hard, (
            f"Code canónico {canonical_code!r} debe aparecer en al menos "
            f"un path (soft-fail con error_code O hard-fail con code)."
        )
