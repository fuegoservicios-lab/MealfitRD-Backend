"""[P2-NEW-13 · 2026-05-11] Multi-tab sync de `mealfit_plan` via storage event.

Bug original (re-audit 2026-05-11):
    Sin listener para `storage` event en AssessmentContext.jsx, dos tabs
    abiertas del mismo usuario divergían en planData state:
      - Tab A swap-meal → backend persiste vía jsonb_set → frontend Tab A
        actualiza localStorage `mealfit_plan` + `setPlanData(...)`.
      - Tab B no se entera. Sigue mostrando plan viejo en UI.
    Backend está safe (jsonb_set quirúrgico + AND user_id, no
    lost-update). Solo UX inconsistente entre tabs.

Fix:
    `useEffect` en `AssessmentProvider` con `addEventListener('storage')`.
    Filtra por `e.key === 'mealfit_plan'`. Soporta:
      - newValue=null → setPlanData(null) (logout en otra tab).
      - newValue válido → re-parse + setPlanData(fresh).
      - parse error → log warning + mantener estado actual.
    Cleanup via return de useEffect.

Estrategia del test (parser-based):
    1. Existe `useEffect` con listener de `storage` event.
    2. Filtra explícitamente por `e.key === 'mealfit_plan'`.
    3. Maneja newValue=null (logout sync) Y newValue parsed.
    4. try/catch alrededor del JSON.parse.
    5. Cleanup en return removeEventListener.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CTX_FP = _REPO_ROOT / "frontend" / "src" / "context" / "AssessmentContext.jsx"


@pytest.fixture(scope="module")
def src() -> str:
    return _CTX_FP.read_text(encoding="utf-8")


def test_storage_listener_block_exists(src: str):
    assert "[P2-NEW-13 · 2026-05-11]" in src, (
        "P2-NEW-13 regresión: el bloque del listener storage event ya no "
        "existe. Sin él, multi-tab divergence vuelve."
    )


def test_storage_event_filters_on_mealfit_plan_key(src: str):
    p2_idx = src.find("[P2-NEW-13 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3000]
    assert re.search(r"e\.key\s*!==\s*['\"]mealfit_plan['\"]", block), (
        "P2-NEW-13 regresión: el listener ya no filtra por "
        "`e.key === 'mealfit_plan'`. Sin filtro, reaccionaría a "
        "cambios de OTRAS keys (mealfit_dislikes, etc.) y pisaría "
        "el plan state con basura."
    )


def test_addEventListener_storage_present(src: str):
    p2_idx = src.find("[P2-NEW-13 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3000]
    assert re.search(
        r"window\.addEventListener\(\s*['\"]storage['\"]\s*,",
        block,
    ), (
        "P2-NEW-13 regresión: `window.addEventListener('storage', ...)` "
        "ya no se registra. Sin él, multi-tab divergence vuelve."
    )


def test_handles_null_newValue(src: str):
    p2_idx = src.find("[P2-NEW-13 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3000]
    # Patrón canónico: e.newValue === null → setPlanData(null).
    assert "e.newValue === null" in block or "newValue === null" in block, (
        "P2-NEW-13 regresión: el listener ya no maneja `newValue=null` "
        "(logout en otra tab). Sin esto, Tab B no se entera si Tab A "
        "logout y sigue mostrando plan viejo."
    )
    # Y debe llamar setPlanData(null) en ese caso.
    null_idx = block.find("newValue === null")
    if null_idx > 0:
        null_block = block[null_idx:null_idx + 400]
        assert "setPlanData(null)" in null_block, (
            "P2-NEW-13 regresión: cuando newValue=null, el listener ya "
            "no llama `setPlanData(null)`. Plan ajeno persiste en UI."
        )


def test_try_catch_around_json_parse(src: str):
    p2_idx = src.find("[P2-NEW-13 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3000]
    assert "try {" in block and "JSON.parse(e.newValue)" in block, (
        "P2-NEW-13 regresión: el JSON.parse ya no está envuelto en "
        "try. Una key corrupta en localStorage crashearía el listener."
    )
    assert "catch" in block, (
        "P2-NEW-13 regresión: el try no tiene catch. Sin él, "
        "una exception aquí rompe el listener para siempre."
    )


def test_cleanup_removes_listener(src: str):
    p2_idx = src.find("[P2-NEW-13 · 2026-05-11]")
    block = src[p2_idx:p2_idx + 3000]
    assert re.search(
        r"removeEventListener\(\s*['\"]storage['\"]",
        block,
    ), (
        "P2-NEW-13 regresión: el cleanup ya no remueve el listener. "
        "Sin cleanup, el listener persiste tras unmount del provider "
        "(memleak React)."
    )
