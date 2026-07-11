"""[P1-ADAPTIVE-RENEWAL · 2026-07-11] F2 — Renovación adaptativa (ciclo cerrado):

El motor "metabolismo evolutivo" (nutrition_calculator MEJORA 4: smoothing + velocidad +
anti-rebound + recomposición) existía desde siempre PERO el path SSE jamás recibía
`weight_history` (el cliente no lo envía y el strip P0-A2 lo vetaría — correcto): solo
los chunks lo inyectaban vía JIT. Resultado: las RENOVACIONES nunca calibraban calorías
con el progreso real.

1. Inyección SERVER-SIDE de weight_history (fuente: health_profile, jamás el request)
   en /analyze/stream para usuarios autenticados. Knob MEALFIT_ADAPTIVE_RENEWAL_INJECT.
2. POST /api/plans/renewal-checkin: peso (+ hambre/energía/adherencia opcionales) →
   append atómico a weight_history (dedupe por día) + _renewal_checkins (cap 12) +
   preview honesto de si el motor calibrará (≥2 registros, span ≥14 días).
3. Frontend: RenewalCheckinModal gatea el SSE en renovaciones autenticadas (Omitir
   nunca bloquea; recovery bypassa).

tooltip-anchor: P1-ADAPTIVE-RENEWAL
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_FRONT_PLAN = (_BACKEND.parent / "frontend" / "src" / "pages" / "Plan.jsx")


# ---------------------------------------------------------------------------
# 1. Inyección server-side en el SSE
# ---------------------------------------------------------------------------

def test_injection_after_untrusted_strip():
    i_strip = _PLANS.find('_strip_untrusted_internal_keys(pipeline_data, allow_set=None, log_prefix="ROUTER /analyze/stream")')
    i_inj = _PLANS.find('pipeline_data["weight_history"] = _wh_ar')
    assert i_strip > 0 and i_inj > i_strip, (
        "la inyección debe correr DESPUÉS del strip P0-A2 (server-side trusted; "
        "el weight_history del cliente sigue vetado)"
    )
    blk = _PLANS[i_inj - 1800: i_inj + 400]
    assert "SELECT health_profile FROM user_profiles" in blk, "fuente = health_profile, jamás el request"
    assert 'MEALFIT_ADAPTIVE_RENEWAL_INJECT' in blk, "knob de rollback"
    assert "if actual_user_id and" in blk, "guests sin perfil quedan fuera"
    assert "len(_wh_ar) >= 2" in blk, "sin 2+ registros el motor no activa — no inyectar ruido"


# ---------------------------------------------------------------------------
# 2. Endpoint de check-in
# ---------------------------------------------------------------------------

def test_checkin_endpoint_wiring():
    i = _PLANS.find('@router.post("/renewal-checkin")')
    assert i > 0, "el endpoint de check-in desapareció"
    blk = _PLANS[i: i + 4200]
    assert "Depends(_CHECKIN_LIMITER)" in blk, "RateLimiter, NO paywall (doctrina P1-AUDIT-3)"
    assert "update_user_health_profile_atomic" in blk, "append atómico (lost-update bajo concurrencia)"
    assert '_hp["_renewal_checkins"] = _ck[-12:]' in blk, "cap 12 ciclos"
    assert 'e.get("date") == _today' in blk, "dedupe por día (último pesaje del día gana)"
    assert 'raise HTTPException(status_code=403' in blk, "guests → 403 (necesita health_profile)"


def test_checkin_limiter_defined():
    assert "_CHECKIN_LIMITER = RateLimiter(max_calls=10, period_seconds=60)" in _PLANS


# ---------------------------------------------------------------------------
# 3. Preview del motor (funcional, puro)
# ---------------------------------------------------------------------------

def test_engine_preview_thresholds():
    from routers.plans import _renewal_engine_preview
    assert _renewal_engine_preview([]) == {"entries": 0, "days_span": 0, "engine_active": False}
    one = [{"date": "2026-07-01", "weight": 120}]
    assert _renewal_engine_preview(one)["engine_active"] is False
    short = [{"date": "2026-07-01", "weight": 120}, {"date": "2026-07-08", "weight": 119}]
    pv = _renewal_engine_preview(short)
    assert pv["entries"] == 2 and pv["days_span"] == 7 and pv["engine_active"] is False, (
        "span < 14 días → el motor real no dispara (fiabilidad); el preview debe espejarlo"
    )
    ok = [{"date": "2026-07-01", "weight": 120}, {"date": "2026-07-16", "weight": 118}]
    pv2 = _renewal_engine_preview(ok)
    assert pv2["engine_active"] is True and pv2["days_span"] == 15


def test_engine_preview_fail_safe():
    from routers.plans import _renewal_engine_preview
    assert _renewal_engine_preview([{"date": "no-es-fecha", "weight": 1}])["engine_active"] is False
    assert _renewal_engine_preview(None)["engine_active"] is False


# ---------------------------------------------------------------------------
# 4. Frontend: gate del SSE + modal (parser sobre Plan.jsx)
# ---------------------------------------------------------------------------

def test_frontend_checkin_gates_sse():
    src = _FRONT_PLAN.read_text(encoding="utf-8")
    assert "RenewalCheckinModal" in src
    i_gate = src.find("if (checkinPending) return;")
    i_process = src.find("processPlan();")
    assert 0 < i_gate < i_process, "el gate corre ANTES de disparar el SSE"
    assert "[loadingSensitive, checkinPending]" in src, "deps re-disparan al cerrar el check-in"
    assert "mealfit_plan_in_progress" in src.split("checkinPending")[1][:1200] or \
        "localStorage.getItem('mealfit_plan_in_progress')" in src, "recovery bypassa el modal"
    assert "!isGuest" in src, "guests no ven el check-in (endpoint requiere auth)"


def test_marker_anchored_in_source():
    assert _PLANS.count("P1-ADAPTIVE-RENEWAL") >= 3
