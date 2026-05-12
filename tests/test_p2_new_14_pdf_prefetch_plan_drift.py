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
    body = _extract_handler_block(src)
    p2_idx = body.find("[P2-NEW-14 · 2026-05-11]")
    block = body[p2_idx:p2_idx + 3500]
    # Patrón canónico: .eq('id', planData.id).eq('user_id', session.user.id)
    assert ".eq('id', planData.id)" in block, (
        "P2-NEW-14 regresión: el SELECT ya no filtra por `id`. "
        "Sin esto, el prefetch trae el plan más reciente del user — "
        "que podría NO ser el que se está exportando a PDF."
    )
    assert ".eq('user_id', session.user.id)" in block, (
        "P2-NEW-14 regresión: el SELECT ya no filtra por `user_id`. "
        "Defense-in-depth: aún con RLS Supabase, el filtro explícito "
        "previene casos edge (sesión recién expirada, etc.)."
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
