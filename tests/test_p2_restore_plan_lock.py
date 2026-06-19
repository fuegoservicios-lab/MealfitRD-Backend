"""[P2-RESTORE-PLAN-LOCK · 2026-06-19] (audit fresco P2-14) api_restore_plan full-overwrite ↔ lock per-plan.

Bug (review del audit 2026-06-19): el test blanket P1-NEW-C detectaba full-overwrites de plan_data LÍNEA-
POR-LÍNEA → era CIEGO al overwrite MULTILÍNEA de api_restore_plan (`UPDATE meal_plans` / `SET plan_data =
%s::jsonb` partido en líneas físicas) → falso verde del contrato I7 para /restore. Además /restore sostenía
solo el lock per-USER (history_mutator), NO el per-PLAN 'general' que usan _chunk_worker T1/T2 y /shift →
ventana lost-update residual (la cancelación de chunks no es atómica respecto a un worker ya en su FOR UPDATE).

Fix: (a) /restore ahora toma `acquire_meal_plan_advisory_lock(purpose='general')` ANTES del overwrite; (b)
este test (statement-aware, scopeado a la función) ancla AMBAS mitades — que el overwrite multilínea EXISTE
y que el lock per-plan lo precede. Cierra el blind-spot sin reescribir el parser blanket (menor riesgo).
"""
from __future__ import annotations

import re
from pathlib import Path

_PLANS = (Path(__file__).resolve().parent.parent / "routers" / "plans.py").read_text(encoding="utf-8")


def _restore_fn_body() -> str:
    """Cuerpo de api_restore_plan: desde su `def` hasta el siguiente def a nivel módulo."""
    m = re.search(r"\ndef api_restore_plan\(", _PLANS)
    assert m, "api_restore_plan no encontrada en routers/plans.py"
    start = m.start()
    nxt = re.search(r"\n(?:async )?def [a-zA-Z_]", _PLANS[start + 1:])
    end = start + 1 + nxt.start() if nxt else len(_PLANS)
    return _PLANS[start:end]


def test_restore_has_multiline_full_overwrite():
    # Statement-aware: el overwrite existe partido en líneas (lo que el detector línea-por-línea no veía).
    body = _restore_fn_body()
    assert re.search(r"UPDATE meal_plans\s+SET plan_data = %s::jsonb", body), \
        "api_restore_plan debe tener el full-overwrite multilínea de plan_data (ancla del contrato I7)"


def test_restore_holds_per_plan_general_lock_before_overwrite():
    body = _restore_fn_body()
    assert "acquire_meal_plan_advisory_lock" in body and 'purpose="general"' in body, \
        "api_restore_plan debe tomar acquire_meal_plan_advisory_lock(purpose='general') — I7"
    # El lock debe ser sobre target_plan_id (el plan que se SOBRESCRIBE), no source (solo se lee). El call usa
    # el alias del import `acquire_meal_plan_advisory_lock as _restore_acquire_lock`.
    m_lock = re.search(r'_restore_acquire_lock\(\s*cur,\s*target_plan_id,\s*purpose="general"', body)
    assert m_lock is not None, "el lock 'general' debe ser sobre target_plan_id (no source)"
    m_upd = re.search(r"UPDATE meal_plans\s+SET plan_data = %s::jsonb", body)
    assert m_upd is not None
    assert 0 <= m_lock.start() < m_upd.start(), \
        "el lock per-plan 'general' debe adquirirse ANTES del UPDATE de plan_data (serializa el overwrite)"


def test_restore_still_holds_user_history_lock():
    # No regresión: el lock per-user original sigue presente (serializa restores concurrentes entre sí).
    body = _restore_fn_body()
    assert "acquire_user_history_advisory_lock" in body, \
        "api_restore_plan debe seguir tomando el lock per-user history_mutator (P1-HIST-AUDIT-7)"
