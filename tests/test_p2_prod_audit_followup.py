"""[P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] Regression guards para los 10 P2 del
audit prod-readiness 2026-05-28 (8 code-fixes + 2 decisiones aceptadas).

Parser-based (lee el source como texto → corre con `pytest --noconftest`).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE_ROOT = _BACKEND_ROOT.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_PLANS = _BACKEND_ROOT / "routers" / "plans.py"
_CHAT = _BACKEND_ROOT / "routers" / "chat.py"
_SYSTEM = _BACKEND_ROOT / "routers" / "system.py"
_ORQ = _BACKEND_ROOT / "graph_orchestrator.py"
_CRON = _BACKEND_ROOT / "cron_tasks.py"
_DBPLANS = _BACKEND_ROOT / "db_plans.py"
_SA_MIGRATION = "p2_system_alerts_active_partial_idx_2026_05_28.sql"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P2-1: lock de los 3 ring-buffers del chunk worker
# ---------------------------------------------------------------------------
def test_p2_1_lock_helper_and_recorders():
    src = _read(_CRON)
    assert "def _get_fallback_buffers_lock(" in src, "P2-1: helper de lock ausente."
    assert "P2-FALLBACK-BUFFERS-LOCK" in src, "P2-1: tooltip-anchor ausente."
    # Ambos recorders adquieren el lock.
    assert src.count("with _get_fallback_buffers_lock():") >= 2, (
        "P2-1: los 2 recorders deben adquirir el lock + los readers."
    )


def test_p2_1_readers_snapshot_under_lock():
    src = _read(_SYSTEM)
    assert src.count("_get_fallback_buffers_lock") >= 2, (
        "P2-1: ambos readers (pantry + tz health) deben importar y usar el lock."
    )


# ---------------------------------------------------------------------------
# P2-2: leader-lock tripwire documentado
# ---------------------------------------------------------------------------
def test_p2_2_leader_lock_tripwire_documented():
    src = _read(_APP_PY)
    assert "P2-LEADER-LOCK-TRIPWIRE" in src, (
        "P2-2: la limitación aceptada del leader-lock debe estar documentada."
    )


# ---------------------------------------------------------------------------
# P2-3: cap duro del history_context
# ---------------------------------------------------------------------------
def test_p2_3_cap_helper_and_knob():
    src = _read(_ORQ)
    assert "def _cap_history_context(" in src, "P2-3: helper de cap ausente."
    assert "MEALFIT_HISTORY_CONTEXT_MAX_CHARS" in src, "P2-3: knob ausente."
    assert "P2-HISTORY-CONTEXT-CAP" in src, "P2-3: tooltip-anchor ausente."


def test_p2_3_cap_applied_to_all_fallbacks():
    src = _read(_ORQ)
    # 3 fallbacks (CB-open, short-result, exception) deben capear el contexto.
    assert src.count("_cap_history_context(history_context)") >= 3, (
        "P2-3: las 3 ramas de fallback del compressor deben aplicar el cap."
    )


# ---------------------------------------------------------------------------
# P2-4: tracking del fallback postprocess task
# ---------------------------------------------------------------------------
def test_p2_4_fallback_task_tracked():
    src = _read(_PLANS)
    assert "_BG_SSE_FALLBACK_TASKS" in src, "P2-4: set de strong-refs ausente."
    assert "P2-FALLBACK-TASK-TRACKED" in src, "P2-4: tooltip-anchor ausente."
    assert "_BG_SSE_FALLBACK_TASKS.discard" in src, (
        "P2-4: done-callback de discard ausente (evita leak del set)."
    )
    # Anti-regresión: ya no se crea el task con un lambda anónimo sin guardar ref.
    assert "lambda: asyncio.create_task(_fallback_postprocess())" not in src, (
        "P2-4 regresión: volvió el create_task sin strong-ref (GC mid-flight)."
    )


# ---------------------------------------------------------------------------
# P2-5: push vía bg_executor bounded
# ---------------------------------------------------------------------------
def test_p2_5_dispatch_push_uses_bg_executor():
    src = _read(_CRON)
    assert "P2-PUSH-VIA-BG-EXECUTOR" in src, "P2-5: tooltip-anchor ausente."
    body = src[src.index("def _dispatch_push_notification("):]
    body = body[: body.index("\n\n\ndef ")] if "\n\n\ndef " in body else body[:1500]
    assert "submit_bg_task(" in body, (
        "P2-5: _dispatch_push_notification debe rutear por submit_bg_task."
    )
    assert "from bg_executor import submit_bg_task" in body, (
        "P2-5: import de submit_bg_task ausente en el helper."
    )


# ---------------------------------------------------------------------------
# P2-6: contrato de pureza del mutator
# ---------------------------------------------------------------------------
def test_p2_6_mutator_purity_documented():
    src = _read(_DBPLANS)
    assert "P2-MUTATOR-PURITY" in src, "P2-6: tooltip-anchor ausente."
    # El contrato menciona el riesgo de re-entrada al pool.
    assert "re-entrada al pool" in src or "re-entre al pool" in src, (
        "P2-6: el contrato debe advertir contra re-entrada al pool."
    )


# ---------------------------------------------------------------------------
# P2-7: ownership check en /api/chat/feedback
# ---------------------------------------------------------------------------
def test_p2_7_feedback_ownership_check():
    src = _read(_CHAT)
    assert "P2-CHAT-FEEDBACK-OWNERSHIP" in src, "P2-7: tooltip-anchor ausente."
    body = src[src.index("async def api_chat_feedback("):]
    body = body[: body.index("\n@router")] if "\n@router" in body else body[:2000]
    assert "get_session_owner" in body, (
        "P2-7: el endpoint feedback no verifica ownership de la sesión."
    )
    assert "403" in body, "P2-7: no rechaza con 403 al owner distinto."


# ---------------------------------------------------------------------------
# P2-8: retención de llm_usage_events + api_usage
# ---------------------------------------------------------------------------
def test_p2_8_usage_retention_cron():
    src = _read(_CRON)
    assert "def _purge_old_usage_events(" in src, "P2-8: cron ausente."
    assert "P2-USAGE-EVENTS-RETENTION" in src, "P2-8: tooltip-anchor ausente."
    assert 'id="purge_old_usage_events"' in src, "P2-8: cron no registrado."
    assert "llm_usage_events" in src and "api_usage" in src, (
        "P2-8: el cron debe purgar ambas tablas."
    )


@pytest.mark.parametrize("knob", [
    "MEALFIT_USAGE_EVENTS_GC_ENABLED",
    "MEALFIT_USAGE_EVENTS_RETENTION_DAYS",
    "MEALFIT_USAGE_EVENTS_GC_MAX_ROWS",
    "MEALFIT_USAGE_EVENTS_GC_INTERVAL_HOURS",
])
def test_p2_8_knobs_present(knob):
    assert knob in _read(_CRON), f"P2-8: knob {knob} ausente."


# ---------------------------------------------------------------------------
# P2-9: índice parcial system_alerts
# ---------------------------------------------------------------------------
def test_p2_9_migration_ssot_dual_dir_identical():
    backend_mig = _BACKEND_ROOT / "migrations" / _SA_MIGRATION
    root_mig = _WORKSPACE_ROOT / "migrations" / _SA_MIGRATION
    assert backend_mig.exists(), f"P2-9: falta migración backend {backend_mig}."
    assert root_mig.exists(), f"P2-9: falta migración root {root_mig}."
    assert backend_mig.read_bytes() == root_mig.read_bytes(), (
        "P2-9: migración SSOT drift entre los dos dirs (P3-MIGRATIONS-SSOT)."
    )


def test_p2_9_migration_partial_index_idempotent():
    mig = (_BACKEND_ROOT / "migrations" / _SA_MIGRATION).read_text(encoding="utf-8")
    assert "CREATE INDEX IF NOT EXISTS idx_system_alerts_active" in mig, (
        "P2-9: la migración no crea el índice idempotente."
    )
    assert "WHERE resolved_at IS NULL" in mig, "P2-9: el índice no es parcial."
    assert "RAISE EXCEPTION" in mig, "P2-9: falta sanity check post-apply."


# ---------------------------------------------------------------------------
# P2-10: /api/plans/cancel — decisión aceptada documentada
# ---------------------------------------------------------------------------
def test_p2_10_cancel_no_auth_decision_documented():
    src = _read(_PLANS)
    assert "P2-CANCEL-NO-AUTH-ACCEPTED" in src, (
        "P2-10: la decisión aceptada (cancel sin auth) debe estar documentada."
    )


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------
def test_last_known_pfix_meets_bundle_floor():
    src = _read(_APP_PY)
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m is not None, "marker _LAST_KNOWN_PFIX no encontrado."
    marker = m.group(1)
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert date_m is not None, f"marker sin fecha ISO: {marker!r}."
    marker_date = datetime.strptime(date_m.group(1), "%Y-%m-%d").date()
    assert marker_date >= date(2026, 5, 28), (
        f"marker `{marker}` (fecha {marker_date}) por debajo del floor del "
        f"bundle P2-PROD-AUDIT-FOLLOWUP (2026-05-28)."
    )
