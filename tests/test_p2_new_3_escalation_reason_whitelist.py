"""[P2-NEW-3 · 2026-05-10] Validación canónica de `escalation_reason` en
`_escalate_unrecoverable_chunk` (cron_tasks.py:~8622).

Bug original (audit 2026-05-10):
  La función propagaba el string a:
    - plan_chunk_queue.dead_letter_reason (DB persist).
    - learning_metrics.escalation_reason (jsonb).
    - meal_plans.plan_data._user_action_required.reason.
    - push notification copy + deeplink URL.
  Sin validación. Un typo escapaba a producción polluyendo telemetría
  y rompiendo el copy del push (caía al else genérico).

Fix:
  Validar `escalation_reason ∈ ESCALATION_REASONS` (5 canónicos, definidos
  en constants.py). Si no: log error + early return SIN persistir.

Cobertura:
  1. Constant `ESCALATION_REASONS` existe con los 5 canónicos esperados.
  2. Cada call site de `_escalate_unrecoverable_chunk` en el codebase
     usa un reason del whitelist (drift detection parser-based).
  3. La función rechaza un reason inválido + loguea P2-NEW-3/INVALID-ESCALATION-REASON.
  4. La función acepta cada reason del whitelist.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"


# ---------------------------------------------------------------------------
# 1. Constant existe + tiene los canónicos
# ---------------------------------------------------------------------------
def test_escalation_reasons_constant_has_canonical_values():
    from constants import ESCALATION_REASONS
    assert isinstance(ESCALATION_REASONS, tuple)
    expected = {
        "recovery_exhausted",
        "unrecoverable_missing_anchor",
        "unrecoverable_corrupted_date",
        "unrecoverable_tz_unresolved",
        "missing_prior_lessons_unrecoverable",
    }
    actual = set(ESCALATION_REASONS)
    missing = expected - actual
    assert not missing, (
        f"ESCALATION_REASONS perdió los reasons canónicos: {sorted(missing)}. "
        f"Si se removió uno intencionalmente, verificar que ningún call site "
        f"lo siga emitiendo (grep `escalation_reason=` en cron_tasks.py)."
    )


# ---------------------------------------------------------------------------
# 2. Drift detection — call sites usan reasons del whitelist
# ---------------------------------------------------------------------------
def test_all_call_sites_use_whitelisted_reasons():
    """Parsea cron_tasks.py extrayendo `escalation_reason="..."` literal
    en call sites y verifica que cada uno está en ESCALATION_REASONS."""
    from constants import ESCALATION_REASONS
    src = _CRON_PY.read_text(encoding="utf-8")
    # Match keyword call: `escalation_reason="literal"` con o sin espacios.
    # Excluye `escalation_reason: str = "recovery_exhausted"` (signature
    # default) si fuera necesario — pero como ese default ESTÁ en la
    # whitelist, no requiere exclusión.
    matches = set(re.findall(
        r"escalation_reason\s*=\s*[\"']([a-z][a-z0-9_]+)[\"']",
        src,
    ))
    forbidden = matches - set(ESCALATION_REASONS)
    assert not forbidden, (
        f"Call sites de _escalate_unrecoverable_chunk usan reasons no "
        f"listados en ESCALATION_REASONS: {sorted(forbidden)}. "
        f"Añadir al whitelist si son legítimos + actualizar el bloque "
        f"if/elif del copy en cron_tasks.py:~8678+ con el nuevo case."
    )


# ---------------------------------------------------------------------------
# 3. Reason inválido es rechazado sin persistir
# ---------------------------------------------------------------------------
def test_invalid_reason_returns_early_without_persisting(monkeypatch, caplog):
    """`_escalate_unrecoverable_chunk` con escalation_reason no listado debe
    log error + early return SIN ejecutar el UPDATE."""
    import cron_tasks as ct
    import logging

    called = {"count": 0}

    def _stub_sql_write(*args, **kwargs):
        called["count"] += 1
        raise AssertionError(
            "execute_sql_write fue invocado con reason inválido — "
            "guard P2-NEW-3 falló."
        )

    monkeypatch.setattr(ct, "execute_sql_write", _stub_sql_write)
    caplog.set_level(logging.ERROR, logger="cron_tasks")

    # Reason inválido (typo de 'recovery_exhausted')
    ct._escalate_unrecoverable_chunk(
        task_id="task-xyz",
        user_id="u-1",
        plan_id="p-1",
        week_number=1,
        recovery_attempts=3,
        escalation_reason="recover_exhausted",  # typo
    )

    assert called["count"] == 0, "execute_sql_write no debe haberse invocado."
    assert any(
        "[P2-NEW-3/INVALID-ESCALATION-REASON]" in rec.message
        for rec in caplog.records
    ), (
        f"Falta log `[P2-NEW-3/INVALID-ESCALATION-REASON]`. "
        f"Logs: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# 4. Reasons válidos NO son rechazados (no falsos positivos)
# ---------------------------------------------------------------------------
def test_valid_reasons_pass_through_guard(monkeypatch):
    """Para cada reason en ESCALATION_REASONS, el guard P2-NEW-3 NO debe
    early-return — el flow debe llegar al UPDATE."""
    import cron_tasks as ct
    from constants import ESCALATION_REASONS

    update_calls = []

    def _stub_sql_write(sql, params=None):
        update_calls.append({"sql": sql, "params": params})
        return None

    # Stub también el push y el plan_data update para que el flow no
    # toque otros sistemas. Solo capturamos que el UPDATE de
    # dead_letter_reason se intentó.
    monkeypatch.setattr(ct, "execute_sql_write", _stub_sql_write)
    # No stub de send_push_notification: confiamos en que la función
    # lo maneja best-effort y un fallo no propaga.

    for reason in ESCALATION_REASONS:
        update_calls.clear()
        ct._escalate_unrecoverable_chunk(
            task_id=f"task-{reason}",
            user_id="u-1",
            plan_id="p-1",
            week_number=1,
            recovery_attempts=3,
            escalation_reason=reason,
        )
        # Al menos un UPDATE debe haberse intentado (el de
        # plan_chunk_queue dead_letter en líneas ~8654).
        update_sqls = [c for c in update_calls if "plan_chunk_queue" in c["sql"]]
        assert update_sqls, (
            f"Reason válido {reason!r} fue rechazado por el guard "
            f"P2-NEW-3 — el UPDATE de dead_letter no se intentó. "
            f"update_calls: {update_calls!r}"
        )
