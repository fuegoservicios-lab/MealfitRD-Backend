"""[P1-GUEST-PLAN-RECOVERY · 2026-07-09] Recuperación deep-search para GUESTS.

Los guests (no logueados) NO persisten en `meal_plans` (FK a user_profiles) → si cierran la pestaña
mid-generación el plan se descartaba (0 planes guest en DB confirmado). Fix: el plan de guest se guarda en
`app_kv_store` bajo `guest_plan:<session_id>` + el status en `pending_pipeline:<session_id>`, y
/pending-status + /guest-plan + ack aceptan `session_id`. Frontend pasa `mealfit_guest_session_id`.
Knob `MEALFIT_GUEST_PLAN_RECOVERY` (default True). Aislado del path autenticado (rama guest en el
done-callback, no toca `_postprocess_pipeline_result`).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8") as f:
    _PLANS = f.read()
with open(os.path.join(_BACKEND, "db_plans.py"), encoding="utf-8") as f:
    _DBP = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-GUEST-PLAN-RECOVERY" in _PLANS
    assert "P1-GUEST-PLAN-RECOVERY" in _DBP


def test_knob_default_on():
    assert 'GUEST_PLAN_RECOVERY_ENABLED = _env_bool("MEALFIT_GUEST_PLAN_RECOVERY", True)' in _PLANS


def test_db_helpers_defined():
    assert "def upsert_guest_plan(" in _DBP
    assert "def get_guest_plan(" in _DBP
    assert "def clear_guest_plan(" in _DBP


def test_pending_status_accepts_session_id_for_guests():
    i = _PLANS.index("async def api_pending_pipeline_status")
    window = _PLANS[i:i + 1600]
    assert "session_id: Optional[str] = Query(None)" in window
    assert "GUEST_PLAN_RECOVERY_ENABLED and session_id" in window


def test_guest_plan_endpoint_defined():
    assert '@router.get("/guest-plan")' in _PLANS
    assert "async def api_guest_plan(" in _PLANS


def test_guest_registered_at_start_and_stored_at_done():
    # registro 'generating' al arrancar
    assert "_guest_recovery_sid" in _PLANS
    assert "upsert_pending_pipeline as _upp_guest" in _PLANS
    # store del plan en el done-callback
    assert "upsert_guest_plan(_guest_recovery_sid" in _PLANS


def test_ack_clears_guest_kvs():
    i = _PLANS.index("async def api_pending_pipeline_ack")
    window = _PLANS[i:i + 1200]
    assert "clear_guest_plan" in window


# ───────────────────────── funcional (guards fail-safe, sin DB) ─────────────────────────

@pytest.fixture()
def dbp():
    import db_plans as _d
    return _d


def test_upsert_guest_plan_rejects_bad_input(dbp):
    # Todos estos retornan False ANTES de tocar la DB (guard clauses).
    assert dbp.upsert_guest_plan(None, {"days": [1]}) is False        # sin session
    assert dbp.upsert_guest_plan("sid", {}) is False                   # sin days
    assert dbp.upsert_guest_plan("sid", {"macros": {}}) is False       # sin days
    assert dbp.upsert_guest_plan("sid", "not-a-dict") is False         # no dict


def test_get_and_clear_guest_plan_null_safe(dbp):
    assert dbp.get_guest_plan(None) is None
    assert dbp.get_guest_plan("") is None
    assert dbp.clear_guest_plan(None) is False
    assert dbp.clear_guest_plan("") is False


def test_guest_plan_kv_key_format(dbp):
    assert dbp._guest_plan_kv_key("abc") == "guest_plan:abc"
