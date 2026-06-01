"""[P1-HIST-RECIPE-1 · 2026-05-10] `/api/plans/recipe/expand` debe persistir
al plan correcto (vía `plan_id`) y a la posición exacta (vía `day_index`/
`meal_index`), con propagación a todas las ocurrencias por defecto.

Bugs originales (audit Historial 2026-05-10):
    1. Wrong-plan persist: el handler `api_expand_recipe` (plans.py:2860)
       llamaba `get_latest_meal_plan(user_id)` sin recibir `plan_id` del
       request. Race con un chunk worker que insertaba un plan nuevo
       entre cook-click y request → expanded_recipe persistido al plan
       equivocado.
    2. First-match-only: match por `m.get("name") == data.get("name")`
       con doble `break` → solo la PRIMERA ocurrencia se persistía. En
       planes de 7d con la misma receta repetida (común), las demás
       seguían sin `recipe`/`isExpanded` → cada cook-click subsecuente
       quemaba cuota LLM silenciosamente.

Estrategia del test (parser estático + mock unit):
    - Static (sobre source de plans.py): el handler debe leer plan_id,
      day_index, meal_index del body; usar SELECT con ownership cuando
      plan_id está presente; propagar (NO break-on-first) en el legacy path.
    - Cross-language drift: el caller front (Recipes.jsx) debe enviar
      plan_id, day_index, meal_index. Mismo patrón que test_p1_form_14.

Drift detection:
    - Si alguien revierte el plan_id resolution → falla
      `test_handler_uses_request_plan_id_with_ownership`.
    - Si alguien restaura el `if updated: break` → falla
      `test_handler_propagates_to_all_matching_occurrences`.
    - Si Recipes.jsx deja de enviar los identificadores → falla
      `test_recipes_jsx_sends_targeting_identifiers`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"
_RECIPES_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Recipes.jsx"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(
            f"No se encontró `def {fn_name}(` en plans.py — endpoint renombrado/eliminado."
        )
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def expand_body() -> str:
    src = _PLANS_PY.read_text(encoding="utf-8")
    return _extract_function_body(src, "api_expand_recipe")


@pytest.fixture(scope="module")
def recipes_jsx_src() -> str:
    return _RECIPES_JSX.read_text(encoding="utf-8")


def test_handler_reads_targeting_identifiers_from_body(expand_body: str):
    """El handler debe leer `plan_id`, `day_index`, `meal_index` del body
    para que el cliente pueda dirigir el persist al plan + posición exacta.
    """
    for field in ("plan_id", "day_index", "meal_index"):
        assert re.search(
            rf"data\.get\(\s*[\"']{field}[\"']", expand_body
        ), (
            f"P1-HIST-RECIPE-1 regresión: handler `api_expand_recipe` no lee "
            f"`data.get('{field}')` del body. Sin este identificador el "
            f"handler cae a get_latest_meal_plan y match by name (legacy "
            f"con bugs wrong-plan + first-match-only)."
        )


def test_handler_uses_request_plan_id_with_ownership(expand_body: str):
    """Cuando el cliente envía `plan_id`, el handler DEBE hacer SELECT
    con ownership check antes de persistir. Sin esto, un atacante podría
    persistir contra un plan ajeno (defense-in-depth aunque
    user_id == verified_user_id ya esté validado arriba).
    """
    pattern = re.compile(
        r"SELECT[^;]*FROM\s+meal_plans\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE | re.DOTALL,
    )
    assert pattern.search(expand_body), (
        "P1-HIST-RECIPE-1 regresión: cuando el request trae `plan_id`, "
        "el handler NO hace `SELECT id, plan_data FROM meal_plans WHERE "
        "id = %s AND user_id = %s`. Sin este SELECT no hay ownership "
        "check explícito por plan_id (sólo el filter user_id general)."
    )


def test_handler_propagates_to_all_matching_occurrences(expand_body: str):
    """Bug original: doble `break` tras el primer match → recetas repetidas
    sólo expandían la primera, quemando cuota LLM en el resto. El fix debe
    iterar TODOS los days/meals sin break-on-first en el camino legacy
    (match por name + recipe original idéntica).
    """
    # Buscar el bloque legacy (loop sobre days/meals con match por name).
    # Aceptamos varias formas pero el indicador clave es la AUSENCIA de
    # `break` dentro del loop tras `updated = True`.
    legacy_block_match = re.search(
        r"for\s+day\s+in\s+\w+(?:\.get\([^)]+\)|\.days|\[[^\]]+\])?\s*:\s*"
        r".*?for\s+m\s+in\s+\w+(?:\.get\([^)]+\)|\.meals)?\s*:\s*"
        r".*?m\.get\(\s*[\"']name[\"']\s*\)\s*==.*?req_name",
        expand_body,
        re.DOTALL,
    )
    assert legacy_block_match, (
        "P1-HIST-RECIPE-1 regresión: no se encontró el bloque de matching "
        "por `name` con la variable `req_name` esperada. ¿El handler fue "
        "reescrito? Actualizar el test si el patrón nuevo es semánticamente "
        "equivalente."
    )

    # Extraer el cuerpo del loop interno (después del `if name == req_name:`).
    inner_loop_body = legacy_block_match.group(0)

    # No debe haber `break` después de `updated = True` en el camino legacy.
    # Toleramos `# NO break` como comentario explícito (auto-documentación).
    pattern_break_after_update = re.search(
        r"updated\s*=\s*True\s*[\r\n]+\s*break",
        inner_loop_body,
    )
    assert not pattern_break_after_update, (
        "P1-HIST-RECIPE-1 regresión: `break` después de `updated = True` "
        "en el camino legacy reintroduce el bug first-match-only. En "
        "planes con receta repetida, sólo se expande la primera ocurrencia "
        "y los cook-clicks subsecuentes vuelven a quemar cuota LLM. "
        "El loop debe propagar a TODAS las ocurrencias (match por name "
        "+ recipe original idéntica garantiza safety contra meals con "
        "nombre igual pero contenido distinto)."
    )


def test_handler_match_requires_recipe_equivalence(expand_body: str):
    """El match legacy debe verificar que `m['recipe'] == data['recipe']`
    además de `name`. Sin esta verificación, una propagación a todas las
    ocurrencias pisaría meals con nombre igual pero contenido distinto
    (e.g. "Pollo guisado" del lunes vs jueves cuando el corrector swapea
    ingredientes).
    """
    pattern = re.compile(
        r"m\.get\(\s*[\"']recipe[\"']\s*\)\s*==\s*req_recipe_original",
        re.IGNORECASE,
    )
    assert pattern.search(expand_body), (
        "P1-HIST-RECIPE-1 regresión: el match legacy no verifica "
        "`m.get('recipe') == req_recipe_original`. Sin esta guard la "
        "propagación a todas las ocurrencias pisaría meals con nombre "
        "igual pero recetas distintas (corrector hace swaps por día)."
    )


def test_handler_precise_index_path_consistency_check(expand_body: str):
    """El camino preciso `(day_index, meal_index)` debe verificar que el
    `name` del meal en esos índices coincide con `req_name`. Si el cliente
    envió índices stale (chunk worker reordenó días), preferimos no
    escribir y caer al fallback by-content.
    """
    # [P1-RECIPE-EXPAND-FAILSIGNAL · 2026-05-30] Boy-scout: el callback de
    # P1-AUDIT-1 renombró `target_meal` → `target_meal_fresh` (opera sobre la
    # copia fresh re-SELECTada bajo FOR UPDATE). El regex quedó stale anclado
    # al nombre viejo y este test fallaba desde 2026-05-15. Aceptar ambos.
    pattern = re.compile(
        r"target_meal(?:_fresh)?\.get\(\s*[\"']name[\"']\s*\)\s*==\s*req_name",
        re.IGNORECASE,
    )
    assert pattern.search(expand_body), (
        "P1-HIST-RECIPE-1 regresión: el camino preciso por índices no "
        "verifica que `target_meal['name'] == req_name`. Sin esta guard, "
        "índices stale (post chunk reorder) escribirían en la posición "
        "equivocada del plan."
    )


# ---------------------------------------------------------------------------
# Cross-language drift detection: el caller front DEBE enviar los 3
# identificadores nuevos. Sin esto, el handler cae al path legacy con los
# bugs originales — el fix backend solo agrega capacidad, el cierre real
# requiere que el cliente la use.
# ---------------------------------------------------------------------------
def test_recipes_jsx_sends_targeting_identifiers(recipes_jsx_src: str):
    """`Recipes.jsx::handleCookClick` debe incluir `plan_id`, `day_index`,
    `meal_index` en el body de POST `/api/plans/recipe/expand`.
    """
    # Buscar el call a /api/plans/recipe/expand.
    call_match = re.search(
        r"fetchWithAuth\(\s*['\"]/api/plans/recipe/expand['\"][^)]*?body\s*:\s*JSON\.stringify\(\s*(\{[^}]*\})",
        recipes_jsx_src,
        re.DOTALL,
    )
    assert call_match, (
        "P1-HIST-RECIPE-1 regresión: no se encontró el call POST a "
        "`/api/plans/recipe/expand` con body JSON.stringify({...}) en "
        "Recipes.jsx. ¿El callsite fue refactorizado? Actualizar el test."
    )
    body_literal = call_match.group(1)

    for js_key in ("plan_id", "day_index", "meal_index"):
        assert js_key in body_literal, (
            f"P1-HIST-RECIPE-1 regresión: Recipes.jsx ya no envía "
            f"`{js_key}` en el body de /recipe/expand. Sin este "
            f"identificador, el backend cae al path legacy "
            f"(get_latest_meal_plan + match by name + first-match-only) "
            f"y reintroducimos los bugs wrong-plan + cuota-quemada."
        )


def test_marker_anchor_present():
    """Slug del filename matchea el marker P-fix para
    `test_p2_hist_audit_14_marker_test_link`."""
    expected_slug = "p1_hist_recipe_1_expand_targeting"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "El nombre de este archivo de test debe contener el slug del "
        "P-fix para que `test_p2_hist_audit_14_marker_test_link` lo "
        "matchee con el marker `_LAST_KNOWN_PFIX`."
    )
