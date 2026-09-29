"""[P1-PLAN-LOTE-772 · 2026-09-28] La cortesía y los créditos regalados cuentan en todo lo que lee el plan: el perfil,
el enrutado de modelos, las dos cuotas y el medidor del usuario. Lo pagado solo lo escriben billing.py y la
degradación del perfil."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

import auth
import db_core
import db_profiles
import regalos_cuenta as rc

_BACKEND = Path(__file__).resolve().parents[1]
UID = "33333333-3333-3333-3333-333333333333"
PLUS = [{"id": "g1", "kind": "plan", "plan": "plus", "amount": None, "ends_at": None,
         "created_at": datetime.now(timezone.utc)}]


def _fila(**extra):
    base = {"id": UID, "plan_tier": "gratis", "subscription_status": None, "subscription_end_date": None,
            "health_profile": {}}
    base.update(extra)
    return base


def test_el_perfil_trae_el_plan_efectivo_y_el_pagado(monkeypatch):
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)   # en CI no hay base: sin pool
    monkeypatch.setattr(db_profiles, "execute_sql_query", lambda *a, **k: _fila())
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: PLUS)
    p = db_profiles.get_user_profile(UID)
    assert p["plan_tier"] == "plus" and p["plan_tier_pagado"] == "gratis"
    assert p["cortesia"] == {"plan": "plus", "hasta": None}


def test_la_degradacion_escribe_solo_lo_pagado(monkeypatch):
    escrito = []
    monkeypatch.setattr(db_profiles, "connection_pool", object(), raising=False)
    monkeypatch.setattr(db_profiles, "execute_sql_query",
                        lambda *a, **k: _fila(plan_tier="basic", subscription_status="CANCELLED"))
    monkeypatch.setattr(db_profiles, "execute_sql_write", lambda q, p=None, **k: escrito.append(p) or True)
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: PLUS)
    p = db_profiles.get_user_profile(UID)
    assert escrito and escrito[0]["plan_tier"] == "gratis"                 # PayPal ya no cobra: lo pagado baja
    assert p["plan_tier"] == "plus" and p["plan_tier_pagado"] == "gratis"  # la cortesía sigue encima


def test_el_enrutado_de_modelos_sigue_al_plan_efectivo(monkeypatch):
    monkeypatch.setattr(db_profiles, "execute_sql_query", lambda *a, **k: {"plan_tier": "gratis"})
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: PLUS)
    assert db_profiles.get_user_plan_tier(UID) == "plus"
    monkeypatch.setattr(db_profiles, "execute_sql_query", lambda *a, **k: {"plan_tier": "admin"})
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: pytest.fail("admin no lee regalos"))
    assert db_profiles.get_user_plan_tier(UID) == "admin"


def test_el_contador_usa_la_misma_ventana_que_los_regalos(monkeypatch):
    capturado = []
    monkeypatch.setattr(db_core, "connection_pool", object())
    monkeypatch.setattr(db_profiles, "execute_sql_query",
                        lambda q, p=None, **k: capturado.append(p) or {"total": 3})
    assert db_profiles.get_monthly_api_usage(UID) == 3
    assert capturado[0][1] == rc.inicio_de_mes().isoformat()


def _perfil(extra_g=0, extra_c=0, tier="gratis"):
    return {"plan_tier": tier, "creditos_extra": {"generacion": extra_g, "coach": extra_c}}


def test_el_tope_de_planes_suma_el_regalo():
    lim = auth._TIER_LIMITS["gratis"]
    with patch.object(auth, "get_monthly_api_usage", return_value=lim + 4), \
         patch.object(auth, "get_user_profile", return_value=_perfil(5)):
        assert auth.verify_api_quota(UID) == UID
    with patch.object(auth, "get_monthly_api_usage", return_value=lim + 5), \
         patch.object(auth, "get_user_profile", return_value=_perfil(5)):
        with pytest.raises(HTTPException) as ei:
            auth.verify_api_quota(UID)
    assert ei.value.status_code == 402


def test_el_tope_del_coach_suma_su_regalo():
    lim = auth._COACH_LIMITS["gratis"]
    with patch.object(auth, "get_monthly_api_usage", return_value=lim + 10), \
         patch.object(auth, "get_user_profile", return_value=_perfil(0, 40)):
        assert auth.verify_coach_quota(UID) == UID
        snap = auth.coach_quota_snapshot(UID)
    assert snap["limit"] == lim + 40 and snap["bonus"] == 40 and snap["remaining"] == 30


def test_un_extra_raro_no_rompe_la_cuota():
    with patch.object(auth, "get_monthly_api_usage", return_value=0), \
         patch.object(auth, "get_user_profile", return_value={"plan_tier": "gratis", "creditos_extra": {"generacion": "x"}}):
        assert auth.verify_api_quota(UID) == UID


def test_resumen_para_el_medidor(monkeypatch):
    ahora = datetime.now(timezone.utc)
    fin = rc.inicio_de_mes(1)
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: [
        {"id": "a", "kind": "creditos_generacion", "amount": 20, "plan": None, "ends_at": fin, "created_at": ahora},
        {"id": "b", "kind": "creditos_coach", "amount": 50, "plan": None, "ends_at": fin,
         "created_at": ahora - timedelta(days=30)},
        {"id": "c", "kind": "plan", "amount": None, "plan": "basic", "ends_at": None, "created_at": ahora},
    ])
    r = rc.resumen_creditos({"id": UID, "plan_tier": "plus"})
    assert r["limit"] == auth._TIER_LIMITS["plus"] + 20 and r["bonus"] == 20 and r["bonus_hasta"] == fin.isoformat()
    # el de hace 30 días ya se avisó; la cortesía Básico no rige para quien disfruta Plus: no se anuncia
    assert [x["id"] for x in r["regalos_recientes"]] == ["a"]


def test_cortesia_igual_a_lo_pagado_no_se_anuncia_en_el_medidor(monkeypatch):
    # Review Focus 3: cortesía Plus y luego paga Plus por PayPal → la cortesía no está en efecto: no se anuncia.
    ahora = datetime.now(timezone.utc)
    plus = [{"id": "p", "kind": "plan", "amount": None, "plan": "plus", "ends_at": None, "created_at": ahora}]
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: plus)
    perfil = rc.superponer({"id": UID, "plan_tier": "plus"})
    assert perfil["cortesia"] is None
    assert rc.resumen_creditos(perfil)["regalos_recientes"] == []
    # control: con Básico pagado la misma cortesía SÍ está en efecto y se anuncia una vez
    perfil = rc.superponer({"id": UID, "plan_tier": "basic"})
    assert [x["id"] for x in rc.resumen_creditos(perfil)["regalos_recientes"]] == ["p"]


def test_resumen_de_admin_no_lee_regalos(monkeypatch):
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: pytest.fail("admin no lee regalos"))
    r = rc.resumen_creditos({"id": UID, "plan_tier": "admin"})
    assert r["bonus"] == 0 and r["regalos_recientes"] == []


def test_el_endpoint_de_creditos_devuelve_el_resumen():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    assert "resumen_creditos(get_user_profile(user_id))" in src and '"credits": credits_used' in src


def test_solo_billing_y_la_degradacion_escriben_plan_tier():
    patron = re.compile(r"UPDATE\s+(?:public\.)?user_profiles\b(?:(?!;).){0,200}?\bplan_tier\s*=", re.S)
    escritores = set()
    for p in list(_BACKEND.glob("*.py")) + list((_BACKEND / "routers").glob("*.py")):
        src = p.read_text(encoding="utf-8")
        if patron.search(src) or 'set_clauses.append("plan_tier' in src:
            escritores.add(p.relative_to(_BACKEND).as_posix())
    assert escritores <= {"routers/billing.py", "db_profiles.py"}, escritores
