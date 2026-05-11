"""[P1-NEW-4 · 2026-05-11] Guard defensivo ownership en
`restorePlan(pastPlanData, expectedUserId)` y `restorePlanFromHistory(pastPlanRow)`
del frontend `AssessmentContext.jsx`.

Bug original (audit 2026-05-11):
    `restorePlan(oldPlan)` y `restorePlanFromHistory(pastPlanRow)`
    pisaban `setPlanData` sin verificar que el plan perteneciera
    al usuario actual. Backend tiene IDOR guards en mutaciones
    (`UPDATE meal_plans WHERE user_id = ...`), pero el state local
    se pintaba antes con plan ajeno si un deeplink/fetch
    interceptado lo inyectaba.

Fix:
    1. `restorePlan` acepta segundo parámetro opcional `expectedUserId`.
       Si presente y != session.user.id, aborta + toast.error.
    2. `restorePlanFromHistory` pre-check `pastPlanRow.user_id ===
       session.user.id` ANTES de pisar state local.

Estrategia del test (parser-based sobre AssessmentContext.jsx):
    1. restorePlan acepta `expectedUserId` como segundo argumento.
    2. Hay check `_currentUid !== expectedUserId` con `return`.
    3. restorePlanFromHistory hace pre-check `pastPlanRow.user_id`.
    4. Toast.error invocado en ambos paths cuando mismatch.
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


def test_restoreplan_accepts_expected_user_id(src: str):
    """`restorePlan` debe aceptar `expectedUserId` como segundo argumento."""
    pattern = re.compile(
        r"const\s+restorePlan\s*=\s*async\s*\(\s*pastPlanData\s*,\s*expectedUserId",
    )
    assert pattern.search(src), (
        "P1-NEW-4 regresión: `restorePlan` ya no acepta `expectedUserId` "
        "como segundo argumento. Sin él, los callers no pueden inyectar "
        "el guard de ownership."
    )


def test_restoreplan_aborts_on_mismatch(src: str):
    """El body debe contener la comparación + return tempranó."""
    # Buscar después de declarar restorePlan, antes de setPlanData.
    body_start = src.find("const restorePlan = async (pastPlanData")
    assert body_start > 0, "restorePlan declaración no encontrada."
    body_end = src.find("const restorePlanFromHistory", body_start)
    assert body_end > body_start, "fin del scope no encontrado"
    body = src[body_start:body_end]
    # Patrones: check `_currentUid !== expectedUserId` y return.
    assert re.search(r"_currentUid\s*!==\s*expectedUserId", body), (
        "P1-NEW-4 regresión: restorePlan no compara _currentUid con "
        "expectedUserId. Sin comparación, el guard no abortaría."
    )
    # Hay return temprano sin pisar setPlanData.
    assert "return;" in body or "return\n" in body, (
        "P1-NEW-4 regresión: restorePlan no tiene `return` temprano "
        "cuando hay mismatch."
    )


def test_restoreplan_from_history_precheck(src: str):
    """restorePlanFromHistory debe pre-checkear pastPlanRow.user_id
    ANTES de `setPlanData(pastPlanData)`."""
    func_start = src.find("const restorePlanFromHistory")
    assert func_start > 0
    set_idx = src.find("setPlanData(pastPlanData)", func_start)
    assert set_idx > 0, "setPlanData no encontrado en restorePlanFromHistory"
    pre_section = src[func_start:set_idx]
    # Pre-check: user_id mismatch + return
    assert re.search(r"pastPlanRow\.user_id", pre_section), (
        "P1-NEW-4 regresión: restorePlanFromHistory no lee "
        "`pastPlanRow.user_id` ANTES de pisar state."
    )
    assert "ownership_mismatch" in pre_section, (
        "P1-NEW-4 regresión: el guard de restorePlanFromHistory ya no "
        "marca error `ownership_mismatch`. Sin marker, el caller no "
        "puede diferenciar este rechazo de otros."
    )


def test_both_paths_show_toast_error_on_mismatch(src: str):
    """Ambos paths deben invocar `toast.error(...)` al detectar
    mismatch — sin toast, el usuario no entiende por qué falló."""
    # Solo necesitamos 2 menciones de toast.error en sections cercanas
    # al guard (P1-NEW-4).
    p1_new_4_block = ""
    idx = 0
    while True:
        idx = src.find("P1-NEW-4", idx)
        if idx < 0:
            break
        # Ventana ±300 chars alrededor.
        p1_new_4_block += src[max(0, idx-300):idx+700]
        idx += 1
    toast_errors = re.findall(r"toast\.error\s*\(", p1_new_4_block)
    assert len(toast_errors) >= 2, (
        f"P1-NEW-4 regresión: solo {len(toast_errors)} `toast.error()` "
        "en bloques P1-NEW-4 (esperado ≥2 — uno por path). Sin toast, "
        "el usuario no sabe que el restore fue rechazado."
    )
