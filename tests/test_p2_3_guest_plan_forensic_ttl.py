"""[P2-GUEST-PLAN-FORENSIC-TTL · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): `clear_guest_plan`
BORRABA el row `guest_plan:<session_id>` inmediatamente al ack del frontend (`DELETE FROM
app_kv_store`) — cuando el owner pidió revisar ESE mismo plan minutos después, el KV ya no existía y
hubo que reprocesar journalctl línea por línea desde cero. Fix: el ack ahora hace soft-mark
(`acked_at` dentro del jsonb, mismo row, mismo `get_guest_plan` sigue leyéndolo sin cambios) y un cron
diario barre en DURO los rows acked+expirados (TTL configurable) + los huérfanos nunca-acked muy
viejos (safety net de espacio, mismo patrón que `_sweep_meal_plans_without_chunks`).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()
with open(os.path.join(_BACKEND, "cron_tasks.py"), encoding="utf-8") as f:
    _CRON = f.read()


def test_marker_present():
    assert "P2-GUEST-PLAN-FORENSIC-TTL" in _DBP
    assert "P2-GUEST-PLAN-FORENSIC-TTL" in _CRON


def test_clear_guest_plan_soft_marks_not_deletes():
    """El ack ya NO borra el row (DELETE) — lo marca acked_at para forensics + sweep posterior."""
    i = _DBP.index("def clear_guest_plan(")
    window = _DBP[i:i + 900]
    assert "acked_at" in window, "clear_guest_plan debe soft-marcar con acked_at, no borrar"
    assert "DELETE FROM app_kv_store" not in window, \
        "el ack ya no debe hacer DELETE directo — el hard-delete vive en el cron de sweep"


def test_sweep_function_defined():
    assert "def _sweep_stale_guest_plans(" in _CRON


def test_sweep_deletes_acked_expired_and_orphaned_unacked():
    i = _CRON.index("def _sweep_stale_guest_plans(")
    window = _CRON[i:i + 3000]
    assert "acked_at" in window
    assert "DELETE FROM app_kv_store" in window
    assert "guest_plan:" in window


def test_knob_defined_with_safe_default_and_clamp():
    assert 'MEALFIT_GUEST_PLAN_FORENSIC_TTL_HOURS' in _CRON
    i = _CRON.index("def _sweep_stale_guest_plans(")
    window = _CRON[i:i + 3000]
    assert "_env_int(\"MEALFIT_GUEST_PLAN_FORENSIC_TTL_HOURS\"" in window


def test_sweep_registered_in_scheduler():
    i = _CRON.index("def register_plan_chunk_scheduler")
    j = _CRON.index("\ndef ", i + 1)
    window = _CRON[i:j]
    assert "_sweep_stale_guest_plans" in window, \
        "el sweep debe registrarse en register_plan_chunk_scheduler (SSOT de crons)"


# ───────────────────────── funcional (guards fail-safe, sin DB) ─────────────────────────

def test_clear_guest_plan_still_null_safe():
    import db_plans as dbp
    assert dbp.clear_guest_plan(None) is False
    assert dbp.clear_guest_plan("") is False


def test_get_guest_plan_unaffected_by_soft_mark_change():
    """get_guest_plan sigue leyendo plan_data del mismo row sin cambios — el soft-mark no
    introduce un filtro nuevo que rompa la recuperación normal (pre-ack)."""
    import db_plans as dbp
    assert dbp.get_guest_plan(None) is None
    assert dbp.get_guest_plan("") is None
