"""[P1-COACH-QUOTA-METER · 2026-09-02] La cuota mensual del coach se VE.

El backend ya separaba el chat de los planes (`verify_coach_quota`, P1-COACH-METER) pero
ninguna pantalla lo mostraba: el usuario descubría el tope con un 402 de texto. Ahora hay un
SSOT (`coach_quota_snapshot`) que usan el endpoint de lectura `GET /api/chat/quota` (exento
del paywall: read-only, cero LLM, RateLimiter 30/60s) y las cabeceras estructuradas del 402.

Tooltip-anchor: P1-COACH-QUOTA-METER | coach_quota_snapshot
"""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import auth


def test_snapshot_uses_the_same_arithmetic_as_the_gate():
    with patch.object(auth, "get_monthly_api_usage", return_value=12) as gm, \
         patch.object(auth, "get_user_profile", return_value={"plan_tier": "basic"}):
        snap = auth.coach_quota_snapshot("u1")
    gm.assert_called_once_with("u1", kind="coach")
    assert snap["used"] == 12 and snap["limit"] == auth._COACH_LIMITS["basic"]
    assert snap["remaining"] == auth._COACH_LIMITS["basic"] - 12
    assert snap["tier"] == "basic" and snap["period"] == "month"


def test_snapshot_resets_first_day_of_next_month_utc():
    with patch.object(auth, "get_monthly_api_usage", return_value=0), \
         patch.object(auth, "get_user_profile", return_value=None):
        snap = auth.coach_quota_snapshot("u1")
    reset = datetime.fromisoformat(snap["resets_at"])
    now = datetime.now(timezone.utc)
    assert reset.day == 1 and reset.tzinfo is not None and reset > now
    assert (reset.year, reset.month) == ((now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1))
    assert snap["tier"] == "gratis" and snap["limit"] == auth._COACH_LIMITS["gratis"]


def test_remaining_never_negative():
    with patch.object(auth, "get_monthly_api_usage", return_value=999), \
         patch.object(auth, "get_user_profile", return_value={"plan_tier": "gratis"}):
        assert auth.coach_quota_snapshot("u1")["remaining"] == 0


def test_402_carries_structured_headers():
    with patch.object(auth, "get_monthly_api_usage", return_value=auth._COACH_LIMITS["gratis"]), \
         patch.object(auth, "get_user_profile", return_value={"plan_tier": "gratis"}):
        with pytest.raises(HTTPException) as ei:
            auth.verify_coach_quota("u1")
    h = ei.value.headers
    assert ei.value.status_code == 402
    assert h["X-Coach-Quota-Limit"] == str(auth._COACH_LIMITS["gratis"]) and h["X-Coach-Quota-Used"] == h["X-Coach-Quota-Limit"]
    assert h["X-Coach-Quota-Resets-At"].endswith("+00:00")


def test_quota_endpoint_registered_read_only_and_rate_limited():
    from routers import chat
    routes = {r.path: r for r in chat.router.routes if hasattr(r, "path")}
    assert "/api/chat/quota" in routes and "GET" in routes["/api/chat/quota"].methods
    import inspect
    src = inspect.getsource(chat.api_chat_quota)
    assert "Depends(get_verified_user_id)" in src and "Depends(_COACH_QUOTA_LIMITER)" in src
    assert "verify_api_quota" not in src and "verify_coach_quota" not in src, "read-only: exento del paywall"


def test_quota_endpoint_returns_snapshot():
    from routers import chat
    with patch.object(chat, "coach_quota_snapshot", return_value={"used": 1, "limit": 60}) as cs:
        assert chat.api_chat_quota("u1", None) == {"used": 1, "limit": 60}
    cs.assert_called_once_with("u1")
    with pytest.raises(HTTPException):
        chat.api_chat_quota(None, None)
