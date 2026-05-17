"""[P2-NEW-B · 2026-05-11] Test parser-based: el endpoint
`POST /api/plans/recalculate-shopping-list` DEBE aceptar `plan_id`
opcional en el body y, cuando se provea, hacer SELECT explícito con
ownership (`WHERE id = %s AND user_id = %s`) en lugar de
`get_latest_meal_plan_with_id`.

Motivación:
    Pre-P2-NEW-B, el handler resolvía el plan target con
    `get_latest_meal_plan_with_id(user_id)` SIEMPRE, sin importar
    qué plan tuviera el caller en mente. Bug bajo race:

      - Usuario hace swap en plan A (P0-NEW-A persiste atómico).
      - `_chunk_worker` crea plan B en paralelo (renewal o nuevo).
      - Frontend llama `/recalculate-shopping-list` esperando recalc
        sobre plan A.
      - Backend resuelve `latest_meal_plan` → plan B (más reciente).
      - Recalc opera sobre plan B; plan A queda sin lista actualizada.

    Mismo modo de fallo que P1-HIST-RECIPE-1 cerró sobre `/recipe/expand`
    (que también usaba `get_latest_meal_plan` antes del fix). Este test
    cierra el patrón también para `/recalculate-shopping-list`.

Cierre P2-NEW-B:
    1. Handler acepta `plan_id` opcional en el body.
    2. Si presente: SELECT explícito `WHERE id = %s AND user_id = %s`,
       404 si no resoluble (no leak de existencia cross-user).
    3. Si ausente: fallback a `get_latest_meal_plan_with_id` (back-compat
       con callers viejos).
    4. Frontend (Dashboard, Pantry, AssessmentContext) envía `plan_id`
       cuando lo conoce desde `planData.id`.

Tests (parser-based):
    1. Body del handler menciona `plan_id` como key del body.
    2. Body contiene SELECT `meal_plans WHERE id = %s AND user_id = %s`.
    3. Body propaga `HTTPException` (no la envuelve en 500 genérico).
    4. Los 3 frontend callsites envían `plan_id` cuando está disponible.
    5. Cross-link slug del marker.

Tooltip-anchor: P2-NEW-B-START | gap P2 audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLANS_PY = _REPO_ROOT / "backend" / "routers" / "plans.py"


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    if not m:
        raise AssertionError(f"Función `{fn_name}` no encontrada en plans.py")
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _PLANS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def recalc_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_recalculate_shopping_list")


# ---------------------------------------------------------------------------
# 1. Body acepta plan_id desde data
# ---------------------------------------------------------------------------
def test_handler_reads_plan_id_from_body(recalc_body: str):
    """El body del handler debe contener `data.get("plan_id")` (con
    validación isinstance str) para soportar el plan_id explícito.
    Sin esta lectura, el handler ignora silenciosamente el `plan_id`
    que el frontend envía → race lost-update sobre plan B.
    """
    pattern = re.compile(
        r"data\.get\s*\(\s*['\"]plan_id['\"]",
    )
    assert pattern.search(recalc_body), (
        "P2-NEW-B regresión: `api_recalculate_shopping_list` no lee "
        "`plan_id` del body. Sin esto, el handler siempre cae a "
        "`get_latest_meal_plan_with_id`, reabriendo el race con "
        "`_chunk_worker` creando plan B paralelo. Restaurar la lectura "
        "+ rama SELECT explícita con ownership."
    )


# ---------------------------------------------------------------------------
# 2. SELECT explícito con AND user_id
# ---------------------------------------------------------------------------
def test_handler_does_ownership_select_when_plan_id_present(recalc_body: str):
    """El handler debe ejecutar un SELECT sobre meal_plans con filtro
    `WHERE id = %s AND user_id = %s` cuando recibe `plan_id` explícito.
    Sin esto, un caller que envíe un plan_id ajeno podría leer
    plan_data cross-user (IDOR de lectura).
    """
    pattern = re.compile(
        r"SELECT\s+[^F]*FROM\s+meal_plans\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s",
        re.IGNORECASE | re.DOTALL,
    )
    assert pattern.search(recalc_body), (
        "P2-NEW-B regresión: el handler no ejecuta SELECT con filtro "
        "`AND user_id = %s` cuando se pasa `plan_id`. Sin esto, IDOR de "
        "lectura: cualquier user authenticated podría leer plan_data "
        "ajeno enviando un plan_id ajeno en el body. Restaurar el "
        "SELECT explícito (espejo de retry-chunk P0-HIST-IDOR-1)."
    )


# ---------------------------------------------------------------------------
# 3. HTTPException propaga (no se envuelve en 500 genérico)
# ---------------------------------------------------------------------------
def test_handler_propagates_httpexception(recalc_body: str):
    """El handler debe tener `except HTTPException: raise` ANTES del
    `except Exception` genérico. Sin esto, el 404 del ownership check
    se envuelve en 500 y el cliente pierde la señal del IDOR.
    """
    pattern = re.compile(
        r"except\s+HTTPException\s*:\s*\n\s*(?:#[^\n]*\n\s*)*raise",
        re.DOTALL,
    )
    assert pattern.search(recalc_body), (
        "P2-NEW-B regresión: el handler no tiene `except HTTPException: "
        "raise` antes del `except Exception` genérico. El 404 del "
        "ownership check se convierte en 500 → cliente y tests pierden "
        "la señal. Restaurar el except-and-raise (mismo patrón que "
        "/restock P0-NEW-1 y /retry-chunk)."
    )


# ---------------------------------------------------------------------------
# 4. Frontend callsites envían plan_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        "frontend/src/pages/Dashboard.jsx",
        "frontend/src/pages/Pantry.jsx",
        "frontend/src/context/AssessmentContext.jsx",
    ],
)
def test_frontend_callsite_sends_plan_id(rel_path: str):
    """Cada uno de los 3 frontends que llaman a
    `/api/plans/recalculate-shopping-list` debe incluir `plan_id` en el
    body. Sin esto, ese callsite cae al fallback latest (back-compat
    intencional) pero pierde la protección contra race con _chunk_worker.
    """
    f = _REPO_ROOT / rel_path
    src = f.read_text(encoding="utf-8")
    # Strip line + block JS comments para evitar falsos positivos.
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", src)
    no_comments = re.sub(r"//[^\n]*", "", no_block)

    # Encontrar cada llamada a recalculate-shopping-list y verificar que
    # el body adyacente contiene `plan_id:`. Heurística: capturar 1500
    # chars después del endpoint Y 1500 chars antes (para el caso
    # P3-RECALC-503-CLASSIFICATION · 2026-05-16 donde el body se declara
    # como `const recalcBody = JSON.stringify({...plan_id...})` ANTES de
    # la línea del `fetchWithAuth(...recalculate-shopping-list...)`).
    matches = list(re.finditer(r"recalculate-shopping-list", no_comments))
    assert matches, (
        f"P2-NEW-B sanity: {rel_path} no contiene `recalculate-shopping-list` "
        f"— refactor del callsite. Si la URL cambió, actualizar este test."
    )
    missing: list[int] = []
    for m in matches:
        start = max(0, m.start() - 1500)
        end = min(len(no_comments), m.start() + 1500)
        block = no_comments[start:end]
        if not re.search(r"plan_id\s*:\s*planData?\?", block) and "plan_id:" not in block and 'plan_id"' not in block:
            line_no = no_comments.count("\n", 0, m.start()) + 1
            missing.append(line_no)
    assert not missing, (
        f"P2-NEW-B regresión: {rel_path} invoca "
        f"`/recalculate-shopping-list` SIN incluir `plan_id` en el body "
        f"(líneas: {missing}). Sin esto, el backend cae al fallback "
        f"`get_latest_meal_plan_with_id`, reabriendo el race con "
        f"_chunk_worker. Añadir `plan_id: planData?.id` al body."
    )


# ---------------------------------------------------------------------------
# 5. Cross-link slug del marker
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p2_new_b"
    assert expected_slug in __file__.replace("\\", "/").lower()
