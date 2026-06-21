"""[P1-GUEST-ADOPT-1 · 2026-06-21] El endpoint POST /api/plans/adopt-guest-plan
adopta el plan de un INVITADO (vive en localStorage) hacia su cuenta nueva / sin
plan. Spec del owner: la herencia del invitado SOLO aplica a cuentas nuevas o sin
plan; con una cuenta-CON-plan existente NO pasa nada distinto (el plan existente
manda). El endpoint codifica esa PRIORIDAD: 409 si la cuenta ya tiene plan → el
frontend descarta el de invitado. Reusa _save_plan_and_track_background (I1 plan_id
del INSERT, I2 user_id del JWT, I6 gateway backend).

Test unitario directo (sin TestClient/DB): monkeypatch de los nombres importados
en routers.plans. Tooltip-anchor: P1-GUEST-ADOPT-1
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

GOOD = {"plan_data": {"days": [{"meals": []}], "calories": 2000}}


def _fn():
    import routers.plans as rp
    return rp.api_adopt_guest_plan


def test_401_sin_auth():
    with pytest.raises(HTTPException) as ei:
        _fn()(GOOD, None)
    assert ei.value.status_code == 401


def test_400_sin_days():
    with pytest.raises(HTTPException) as ei:
        _fn()({"plan_data": {"calories": 2000}}, "u1")
    assert ei.value.status_code == 400


def test_400_plan_data_no_dict():
    with pytest.raises(HTTPException) as ei:
        _fn()({"plan_data": "no soy dict"}, "u1")
    assert ei.value.status_code == 400


def test_409_cuenta_con_plan_existente(monkeypatch):
    """PRIORIDAD: si la cuenta ya tiene plan → 409 (el frontend descarta el de invitado)."""
    import routers.plans as rp
    monkeypatch.setattr(rp, "get_latest_meal_plan_with_id", lambda uid: {"id": "existente"})
    with pytest.raises(HTTPException) as ei:
        rp.api_adopt_guest_plan(GOOD, "u1")
    assert ei.value.status_code == 409
    assert "already_has_plan" in str(ei.value.detail)


def test_adopta_cuando_sin_plan(monkeypatch):
    import routers.plans as rp
    monkeypatch.setattr(rp, "get_latest_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(
        rp, "_save_plan_and_track_background",
        lambda uid, pd, selected_techniques=None, return_id=False: "nuevo-plan-id",
    )
    out = rp.api_adopt_guest_plan(GOOD, "u1")
    assert out["success"] is True and out["adopted"] is True
    assert out["plan_id"] == "nuevo-plan-id"


def test_dedup_no_duplica(monkeypatch):
    """Si el save interno deduplica (return None), reporta adopted=False sin error."""
    import routers.plans as rp
    monkeypatch.setattr(rp, "get_latest_meal_plan_with_id", lambda uid: None)
    monkeypatch.setattr(
        rp, "_save_plan_and_track_background",
        lambda uid, pd, selected_techniques=None, return_id=False: None,
    )
    out = rp.api_adopt_guest_plan(GOOD, "u1")
    assert out["adopted"] is False
