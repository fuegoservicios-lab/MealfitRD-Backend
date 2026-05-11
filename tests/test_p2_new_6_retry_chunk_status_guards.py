"""[P2-NEW-6 · 2026-05-10] Regression guard: los UPDATEs en `/retry-chunk`
llevan guards de status estrictos (no genéricos `IN (...)`).

Bug temido (audit 2026-05-10 — descartado tras verificación):
  El auditor proponía añadir `AND status IN ('failed', 'cancelled')` a
  cada UPDATE para evitar mutación parcial si el plan se borra entre el
  ownership check y los UPDATEs.

Verificación post-audit (routers/plans.py:4113-4135):
  Ambos UPDATEs YA tienen guards estrictos por status específico:
    - UPDATE 1 (reset chunk failed → pending): `AND status = 'failed'`.
    - UPDATE 2 (revivir cancelled → pending):  `AND status = 'cancelled'`.

  El guard estricto (`= 'failed'`) es MÁS tight que el `IN ('failed',
  'cancelled')` propuesto — cada UPDATE muta solo su status target.
  El IN combinaría ambas mutaciones en cada UPDATE causando doble reset.

  Adicionalmente, ambos llevan `AND meal_plan_id IN (SELECT id FROM
  meal_plans WHERE user_id = %s)` como defense-in-depth contra race
  ownership.

Este test bloquea regresión del contrato:
  Si alguien afloja el guard a `status IN (...)` o lo borra, este test
  falla — los dos UPDATEs deben mantener status guards estrictos.
"""
from __future__ import annotations

import re
from pathlib import Path


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PLANS_PY = _BACKEND_ROOT / "routers" / "plans.py"


def _read_retry_chunk_function() -> str:
    """Extrae el body de `api_retry_chunk` (líneas 4060-4170 aprox)."""
    src = _PLANS_PY.read_text(encoding="utf-8")
    start = re.search(
        r"@router\.post\(\"/\{plan_id\}/retry-chunk/\{chunk_id\}\"\)\s*\ndef api_retry_chunk",
        src,
    )
    assert start is not None, (
        "No encuentro el endpoint `/{plan_id}/retry-chunk/{chunk_id}` en "
        "routers/plans.py."
    )
    # Buscar hasta el siguiente `@router.` decorator (definición vecina).
    after = src[start.start():]
    next_route = re.search(r"\n@router\.", after[1:])
    end_offset = next_route.start() + 1 if next_route else len(after)
    return after[:end_offset]


def test_reset_failed_update_has_strict_status_failed_guard():
    """UPDATE 1 (reset chunk fallido → 'pending') debe llevar `AND status = 'failed'`
    exact match (no IN). Sin el guard, podríamos resetear cancelados accidentalmente
    o resetear un chunk que ya estaba 'completed'."""
    body = _read_retry_chunk_function()
    # Pattern: SET status = 'pending', ... WHERE id = %s AND ... AND status = 'failed' AND ...
    pattern = re.compile(
        r"UPDATE\s+plan_chunk_queue\s+"
        r"SET\s+status\s*=\s*'pending'[^;]*?"
        r"WHERE[^;]*?AND\s+status\s*=\s*'failed'",
        re.DOTALL,
    )
    assert pattern.search(body) is not None, (
        "El UPDATE de reset (failed → pending) perdió el guard estricto "
        "`AND status = 'failed'`. Sin él, un attacker o race podría "
        "resetear chunks completados o cancelados via este endpoint."
    )


def test_revive_cancelled_update_has_strict_status_cancelled_guard():
    """UPDATE 2 (revivir cancelled → 'pending') debe llevar
    `AND status = 'cancelled'` exact match. Sin él, podríamos resetear
    chunks failed/completed legítimos a pending."""
    body = _read_retry_chunk_function()
    pattern = re.compile(
        r"UPDATE\s+plan_chunk_queue\s+"
        r"SET\s+status\s*=\s*'pending'[^;]*?"
        r"WHERE[^;]*?AND\s+status\s*=\s*'cancelled'",
        re.DOTALL,
    )
    assert pattern.search(body) is not None, (
        "El UPDATE de revival (cancelled → pending) perdió el guard estricto "
        "`AND status = 'cancelled'`. Sin él, podríamos resetear chunks "
        "ya processing/completed/dead-letter por error."
    )


def test_both_updates_have_user_ownership_defense_in_depth():
    """Ambos UPDATEs deben mantener `meal_plan_id IN (SELECT id FROM meal_plans
    WHERE user_id = %s)` como defense-in-depth contra race del ownership check."""
    body = _read_retry_chunk_function()
    # Contar la presencia del subquery defensivo en el endpoint.
    pattern = re.compile(
        r"meal_plan_id\s+IN\s*\(\s*SELECT\s+id\s+FROM\s+meal_plans\s+WHERE\s+user_id\s*=\s*%s\s*\)",
        re.IGNORECASE,
    )
    matches = pattern.findall(body)
    assert len(matches) >= 2, (
        f"Esperaba >=2 subqueries `meal_plan_id IN (SELECT id FROM meal_plans "
        f"WHERE user_id = %s)` como defense-in-depth en los UPDATEs de "
        f"`/retry-chunk`. Encontrados: {len(matches)}. Sin él, un race "
        f"entre el ownership check y los UPDATEs permite mutación parcial "
        f"sobre planes ajenos."
    )


def test_ownership_check_uses_404_not_403():
    """Ownership check devuelve 404 (no 403) para no filtrar la existencia
    del plan ajeno. Patrón establecido P0-HIST-IDOR-1."""
    body = _read_retry_chunk_function()
    pattern = re.compile(
        r"if\s+not\s+owner\s*:\s*\n"
        r"\s*(?:#[^\n]*\n\s*){0,4}"
        r"\s*raise\s+HTTPException\(\s*status_code\s*=\s*404",
        re.DOTALL,
    )
    assert pattern.search(body) is not None, (
        "Ownership check no devuelve 404 cuando el plan no existe o no "
        "pertenece al usuario. 403 filtraría la existencia del plan ajeno."
    )
