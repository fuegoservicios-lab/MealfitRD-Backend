"""[P1-NEW-8 · 2026-05-11] `restorePlanFromHistory` debe abortar también
cuando `pastPlanRow.user_id` es falsy (null/undefined/"").

Bug original (audit 2026-05-11):
    El guard P1-NEW-4 gateaba con `if (pastPlanRow.user_id && _currentUid
    && ... && pastPlanRow.user_id !== _currentUid)`. El primer `&&`
    corto-circuitaba cuando `user_id` venía null (row legacy, data
    migration imperfecta, bug upstream). En ese caso el bloque saltaba
    al `setPlanData(pastPlanData)` directo y la UI pintaba un plan
    sin owner verificable durante 100-300ms hasta que el endpoint
    `/api/plans/restore` respondiera.

    Audit en producción HOY = 0 rows con `user_id IS NULL`, así que
    el fix es preventivo: cierra el hueco antes de que aparezca por
    migración o backend bug.

Fix:
    Reformular el guard como:
      `if (_currentUid && _currentUid !== 'guest' &&
           (!pastPlanRow.user_id || pastPlanRow.user_id !== _currentUid))`
    Así abortamos en ambos casos:
      a) `pastPlanRow.user_id` falsy → no hay owner verificable.
      b) `pastPlanRow.user_id` mismatch → guard P1-NEW-4 original.

Estrategia del test (parser-based sobre AssessmentContext.jsx):
    1. El predicate canónico está presente — `!pastPlanRow.user_id` Y
       `pastPlanRow.user_id !== _currentUid` dentro del mismo `if`.
    2. El predicate buggy P1-NEW-4 antiguo (`pastPlanRow.user_id &&
       _currentUid && _currentUid !== 'guest' && pastPlanRow.user_id
       !== _currentUid`) NO está presente — un revert accidental por
       merge conflict debe fallar el test.
    3. El marker `[P1-NEW-8]` aparece en el `console.warn` para que
       el operador pueda grep-ear en logs.
    4. El bloque comentario explica WHY (legacy null user_id).
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


def test_fix_predicate_present(src: str):
    """El predicate canónico debe estar dentro del scope de
    `restorePlanFromHistory`."""
    func_start = src.find("const restorePlanFromHistory")
    assert func_start > 0, "restorePlanFromHistory no encontrado"
    # Boundary: la primera ocurrencia de `setPlanData(pastPlanData)` —
    # el guard debe vivir antes del primer write a state local.
    set_idx = src.find("setPlanData(pastPlanData)", func_start)
    assert set_idx > func_start, "setPlanData no encontrado dentro de la función"
    pre = src[func_start:set_idx]

    assert "!pastPlanRow.user_id" in pre, (
        "P1-NEW-8 regresión: el guard ya no rechaza explícitamente el caso "
        "`pastPlanRow.user_id` falsy. Sin esa rama, una row legacy con "
        "user_id=null pintará plan en UI antes de que el endpoint backend "
        "responda."
    )
    assert re.search(r"pastPlanRow\.user_id\s*!==\s*_currentUid", pre), (
        "P1-NEW-8 regresión: el guard P1-NEW-4 original (mismatch entre "
        "row.user_id y current) ha desaparecido. Sin él, un row con "
        "user_id ajeno pintará plan en UI."
    )


def test_buggy_short_circuit_pattern_removed(src: str):
    """El patrón buggy (P1-NEW-4 versión inicial) NO debe estar presente.

    Patrón: `pastPlanRow.user_id && _currentUid && _currentUid !== 'guest'
    && pastPlanRow.user_id !== _currentUid` en ese orden literal.

    Si un merge conflict / refactor lo revierte, fallaremos aquí.
    """
    buggy = re.compile(
        r"pastPlanRow\.user_id\s*&&\s*"
        r"_currentUid\s*&&\s*"
        r"_currentUid\s*!==\s*['\"]guest['\"]\s*&&\s*"
        r"pastPlanRow\.user_id\s*!==\s*_currentUid",
        re.MULTILINE,
    )
    assert not buggy.search(src), (
        "P1-NEW-8 regresión: el patrón P1-NEW-4 buggy (short-circuit con "
        "`pastPlanRow.user_id && ...`) volvió. Si pastPlanRow.user_id es "
        "null, el guard se salta. Reescribir como "
        "`(!pastPlanRow.user_id || pastPlanRow.user_id !== _currentUid)`."
    )


def test_warn_marker_present(src: str):
    """El `console.warn` debe mencionar `[P1-NEW-8]` para grep en logs."""
    func_start = src.find("const restorePlanFromHistory")
    func_end = src.find("\n    };", func_start)
    body = src[func_start:func_end]
    assert "[P1-NEW-8]" in body, (
        "P1-NEW-8 regresión: el `console.warn` ya no menciona el marker. "
        "Sin él, SRE no puede grep-ear logs para auditar la frecuencia "
        "del path null-user_id."
    )


def test_comment_block_documents_why(src: str):
    """El bloque comentario en el guard debe explicar el WHY (legacy
    null user_id) — sin docs el próximo revisor puede volver al patrón
    buggy creyendo que es 'más legible'."""
    # Buscar en la zona del guard P1-NEW-8.
    block_start = src.find("[P1-NEW-8")
    assert block_start > 0, "Marker [P1-NEW-8 no encontrado en comentarios"
    # Ventana razonable.
    block = src[block_start:block_start + 1500]
    # Tokens que el comment DEBE mencionar (defensa contra reverter).
    for token in ("null", "legacy", "ownership"):
        assert token in block.lower(), (
            f"P1-NEW-8 regresión: el comment del guard ya no menciona "
            f"`{token}`. Sin context, el próximo revisor puede simplificar "
            f"de vuelta al patrón buggy con `pastPlanRow.user_id && ...`."
        )
