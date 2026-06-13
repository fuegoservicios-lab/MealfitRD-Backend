"""[P2-NEW-14 · 2026-05-11] Pre-PDF drift detection del plan en
`handleDownloadShoppingList` (Dashboard.jsx).

Bug original (re-audit 2026-05-11):
    Si chunk worker en background recalculó `aggregated_shopping_list*`
    mientras user estaba en Dashboard (P2-NEW-12 debounced recalc,
    swap-meal/persist, etc.), `planData` local está stale. El PDF se
    generaba con `getActiveShoppingList(planData, duration)` → lista
    vieja.

Fix:
    Espejo de P2-NEW-4 (Pantry recalc prefetch). Antes de armar
    `rawSourceIngredients`, SELECT estrecho del plan actual; si
    `_plan_modified_at` difiere, sync localStorage + setPlanData y usa
    `effectivePlanData` (fresh) para el PDF.

Estrategia del test (parser-based):
    1. Bloque marcado `[P2-NEW-14 · 2026-05-11]` existe en handler.
    2. SELECT del plan filtra por id + user_id (ownership).
    3. Comparación `_plan_modified_at` local vs latest.
    4. Sync localStorage + setPlanData + reasigna `effectivePlanData`.
    5. `rawSourceIngredients` usa `effectivePlanData`, no `planData`.
    6. Best-effort: try/catch alrededor del prefetch.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASH_FP = _REPO_ROOT / "frontend" / "src" / "pages" / "Dashboard.jsx"
# [P1-NEON-DB-MIGRATION · 2026-06-12] El SELECT estrecho del prefetch migró
# del cliente supabase-js al endpoint backend GET /api/plans-data/{plan_id}
# (routers/user_data.py) — el ownership ahora vive server-side (I2).
_USER_DATA_PY = _REPO_ROOT / "backend" / "routers" / "user_data.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _DASH_FP.read_text(encoding="utf-8")


def _extract_handler_block(src: str) -> str:
    start = src.find("const handleDownloadShoppingList")
    assert start > 0
    # Boundary: el siguiente `const handle*` o `}` top-level.
    after = src[start + 50:]
    next_match = re.search(r"\n    const handle[A-Z]", after)
    end = (start + 50 + next_match.start()) if next_match else (start + 50 + 4500)
    return src[start:end]


def test_p2_new_14_block_present(src: str):
    body = _extract_handler_block(src)
    assert "[P2-NEW-14 · 2026-05-11]" in body, (
        "P2-NEW-14 regresión: el bloque del prefetch drift ya no está en "
        "`handleDownloadShoppingList`. Sin él, PDF se genera con lista stale."
    )


def test_select_filters_by_id_and_user_id(src: str):
    """[P1-NEON-DB-MIGRATION · 2026-06-12] Re-anclado: el SELECT estrecho
    pasó de `.eq('id', planData.id).eq('user_id', session.user.id)`
    (supabase-js, ownership client-side) a `fetchWithAuth('/api/plans-data/
    {plan_id}')` con ownership server-side. La MISMA propiedad se verifica
    en dos mitades ejecutables:
      1. El handler PDF fetchea el plan POR SU id (no "el más reciente").
      2. El endpoint backend filtra `WHERE id = %s AND user_id = %s` (I2).
    """
    body = _extract_handler_block(src)
    p2_idx = body.find("[P2-NEW-14 · 2026-05-11]")
    block = body[p2_idx:p2_idx + 3500]
    # Mitad 1 (frontend, código ejecutable — no comments): el fetch usa el
    # plan_id local en la URL y va autenticado via fetchWithAuth.
    assert re.search(
        r"fetchWithAuth\(\s*`/api/plans-data/\$\{planData\.id\}`\s*\)",
        block,
    ), (
        "P2-NEW-14 regresión: el prefetch ya no fetchea "
        "`/api/plans-data/${planData.id}` via fetchWithAuth. Sin el id "
        "explícito, el prefetch traería otro plan (wrong-plan PDF); sin "
        "fetchWithAuth, el request va sin Bearer y el backend no puede "
        "resolver ownership."
    )

    # Mitad 2 (backend, SQL ejecutable): GET /plans-data/{plan_id} debe
    # filtrar por id Y user_id — equivalente server-side del chain
    # `.eq('id', ...).eq('user_id', ...)` que este test anclaba pre-Neon.
    user_data_src = _USER_DATA_PY.read_text(encoding="utf-8")
    m = re.search(r"def\s+api_get_plan_data\s*\(", user_data_src)
    assert m, (
        "P2-NEW-14/I2 regresión: `api_get_plan_data` no existe en "
        "routers/user_data.py. Si el endpoint se movió/renombró, "
        "actualizar este test ANTES de mergear."
    )
    nxt = re.search(r"\n(?:@router\.|def\s)", user_data_src[m.start() + 1:])
    fn_body = user_data_src[m.start(): m.start() + 1 + (nxt.start() if nxt else len(user_data_src))]
    assert re.search(r"WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s", fn_body), (
        "P2-NEW-14/I2 regresión: el SELECT de `api_get_plan_data` ya no "
        "filtra `WHERE id = %s AND user_id = %s`. El prefetch del PDF "
        "leería plan_data ajeno (IDOR) — mismo contrato que el chain "
        ".eq('id').eq('user_id') que este test anclaba pre-Neon."
    )


def test_compares_plan_modified_at(src: str):
    body = _extract_handler_block(src)
    p2_idx = body.find("[P2-NEW-14 · 2026-05-11]")
    block = body[p2_idx:p2_idx + 3500]
    assert "_plan_modified_at" in block, (
        "P2-NEW-14 regresión: el prefetch ya no compara `_plan_modified_at`. "
        "Sin esa señal semántica, no hay forma de detectar drift "
        "(updated_at físico cambia por trigger en muchos paths)."
    )
    assert re.search(
        r"latestModified\s+&&\s+latestModified\s*!==\s*localModified",
        block,
    ), (
        "P2-NEW-14 regresión: la condición de drift ya no es "
        "`latestModified && latestModified !== localModified`. Sin el "
        "primer truthy check, un local sin marker (legacy) dispararía "
        "sync espuria; sin el `!==`, no se detecta drift real."
    )


def test_syncs_localStorage_and_state(src: str):
    body = _extract_handler_block(src)
    p2_idx = body.find("[P2-NEW-14 · 2026-05-11]")
    block = body[p2_idx:p2_idx + 3500]
    assert "localStorage.setItem('mealfit_plan'" in block, (
        "P2-NEW-14 regresión: el sync ya no actualiza `localStorage`. "
        "Sin él, próximo refresh vuelve a leer stale."
    )
    assert "setPlanData(fresh)" in block, (
        "P2-NEW-14 regresión: el sync ya no llama `setPlanData(fresh)`. "
        "Sin él, otros componentes que consumen `planData` del context "
        "siguen con stale state."
    )


def test_effective_plan_data_used_downstream(src: str):
    body = _extract_handler_block(src)
    # `rawSourceIngredients` debe leer de `effectivePlanData`, no de planData.
    assert "getActiveShoppingList(effectivePlanData" in body, (
        "P2-NEW-14 regresión: `rawSourceIngredients` ya no usa "
        "`effectivePlanData`. Si revertiste a `planData`, el prefetch "
        "fue para nada — el PDF sigue construido sobre stale state."
    )


def test_prefetch_wrapped_in_try_catch(src: str):
    body = _extract_handler_block(src)
    p2_idx = body.find("[P2-NEW-14 · 2026-05-11]")
    block = body[p2_idx:p2_idx + 3500]
    # try/catch best-effort
    assert "try {" in block, (
        "P2-NEW-14 regresión: el prefetch ya no está envuelto en try. "
        "Un fallo de DB abortaría el PDF download — peor que stale data."
    )
    assert re.search(r"catch\s*\(\s*driftErr\s*\)", block), (
        "P2-NEW-14 regresión: el catch ya no captura `driftErr`. Sin "
        "captura amplia el endpoint puede crashear bajo network issues."
    )
