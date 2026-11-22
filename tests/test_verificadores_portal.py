# -*- coding: utf-8 -*-
"""Tests del portal de verificadores Android (`/api/verificadores/*`) y campo `verificador_android` en admin."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_cuentas_lista as acl
import routers.verificadores as rv


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(rv.router)
    monkeypatch.setenv("MEALFIT_VERIFICADORES_ACCOUNTS", "equipo1:ClaveSegura123!,equipo2:OtraClave456")
    monkeypatch.setenv("MEALFIT_VERIFICADORES_SECRET", "secreto-super-largo-para-firmar-verificadores")
    return TestClient(app)


def _row(
    uid="11111111-1111-1111-1111-111111111111",
    email="tester1@gmail.com",
    nombre="Tester Uno",
    dias=15,
    comidas_ultima=None,
):
    now = datetime.now(timezone.utc)
    return {
        "user_id": uid,
        "email": email,
        "nombre": nombre,
        "alta": now - timedelta(days=20),
        "verificador_desde": (now - timedelta(days=10)).isoformat(),
        "verificador_hasta": (now + timedelta(days=20)).isoformat(),
        "origen": "android",
        "comidas_total": 12,
        "comidas_30d": 10,
        "comidas_ultima": comidas_ultima,
        "mensajes_coach": 4,
        "mensajes_ultima": None,
        "escaneos": 3,
        "escaneo_ultima": None,
        "peso_ultima": None,
        "agua_ultima": None,
        "dias_activos_30d": dias,
    }


def test_sin_configuracion_falla_cerrado(monkeypatch):
    app = FastAPI()
    app.include_router(rv.router)
    monkeypatch.delenv("MEALFIT_VERIFICADORES_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_VERIFICADORES_SECRET", raising=False)
    monkeypatch.delenv("MEALFIT_SESSION_SECRET", raising=False)
    c = TestClient(app)

    yo = c.get("/api/verificadores/yo")
    assert yo.status_code == 200
    assert yo.json() == {"autenticado": False, "usuario": None, "configurado": False}

    assert c.post("/api/verificadores/login", json={"usuario": "x", "clave": "y"}).status_code == 503
    assert c.get("/api/verificadores/cuentas").status_code == 503


def test_login_invalido_y_cuentas_bloqueado_sin_sesion(client):
    assert client.get("/api/verificadores/cuentas").status_code == 401
    bad = client.post("/api/verificadores/login", json={"usuario": "equipo1", "clave": "mal"})
    assert bad.status_code == 401
    assert rv.COOKIE_NAME not in client.cookies


def test_login_lista_filtros_privacidad_y_logout(client, monkeypatch):
    now = datetime.now(timezone.utc)
    filas = [
        _row(email="activo@gmail.com", nombre="Ana Activa", dias=15, comidas_ultima=now - timedelta(hours=2)),
        _row(
            uid="22222222-2222-2222-2222-222222222222",
            email="dormido@gmail.com",
            nombre="Luis Inactivo",
            dias=1,
            comidas_ultima=None,
        ),
    ]
    monkeypatch.setattr(rv, "execute_sql_query", lambda *a, **k: list(filas))

    ok = client.post("/api/verificadores/login", json={"usuario": "Equipo1", "clave": "ClaveSegura123!"})
    assert ok.status_code == 200
    assert ok.json() == {"ok": True, "usuario": "equipo1"}

    yo = client.get("/api/verificadores/yo")
    assert yo.status_code == 200
    assert yo.json()["autenticado"] is True

    resp = client.get("/api/verificadores/cuentas")
    assert resp.status_code == 200
    data = resp.json()
    assert data["resumen"] == {
        "total": 2,
        "meta_google_play": 12,
        "faltan": 10,
        "activos_7d": 1,
        "con_actividad": 1,
        "cumplen_14d": 1,
        "dias_meta": 14,
    }
    assert len(data["verificadores"]) == 2
    for v in data["verificadores"]:
        for secreto in ("gasto_ia_30d_usd", "plan_pagado", "plan_efectivo", "es_admin", "prueba", "alertas"):
            assert secreto not in v
            assert secreto not in v["actividad"]

    # Filtros conservan resumen real de todos los verificadores
    r_act = client.get("/api/verificadores/cuentas?estado=activos").json()
    assert len(r_act["verificadores"]) == 1
    assert r_act["verificadores"][0]["email"] == "activo@gmail.com"
    assert r_act["resumen"]["total"] == 2

    r_bus = client.get("/api/verificadores/cuentas?buscar=dormido").json()
    assert len(r_bus["verificadores"]) == 1
    assert r_bus["verificadores"][0]["email"] == "dormido@gmail.com"

    # Logout limpia sesion
    assert client.post("/api/verificadores/logout").status_code == 200
    assert client.get("/api/verificadores/cuentas").status_code == 401


def test_cookie_alterada_es_rechazada(client):
    ok = client.post("/api/verificadores/login", json={"usuario": "equipo2", "clave": "OtraClave456"})
    assert ok.status_code == 200
    client.cookies.set(rv.COOKIE_NAME, "payload_falso.firma_falsa")
    assert client.get("/api/verificadores/cuentas").status_code == 401
    assert client.get("/api/verificadores/yo").json()["autenticado"] is False


def test_admin_fila_expone_verificador_android(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", "11111111-1111-1111-1111-111111111111")
    si = acl._fila_cuenta({"user_id": "2", "probador_hasta": "2026-12-20T00:00:00Z"}, {}, set())
    no = acl._fila_cuenta({"user_id": "3", "probador_hasta": None}, {}, set())
    assert si["verificador_android"] is True
    assert no["verificador_android"] is False


def test_admin_ampliar_ficha_expone_verificador_android(monkeypatch):
    uid = "11111111-1111-1111-1111-111111111111"
    monkeypatch.delenv("MEALFIT_ADMIN_USER_IDS", raising=False)
    monkeypatch.setattr(acl.ajustes_cuenta, "ajustes_de", lambda _u: {"ajustes": [], "ajustes_dispositivo": {}})
    monkeypatch.setattr(acl, "prueba_de", lambda _u: None)

    monkeypatch.setattr(
        acl,
        "execute_sql_query",
        lambda q, *_a, **_k: (
            [{"user_id": uid, "probador_hasta": "2026-12-20T00:00:00Z"}]
            if "WITH base AS" in " ".join(q.split())
            else {}
        ),
    )
    f_si = acl.ampliar_ficha({"user_id": uid, "plan_pagado": "gratis"})
    assert f_si["verificador_android"] is True

    monkeypatch.setattr(
        acl,
        "execute_sql_query",
        lambda q, *_a, **_k: (
            [{"user_id": uid, "probador_hasta": None}]
            if "WITH base AS" in " ".join(q.split())
            else {}
        ),
    )
    f_no = acl.ampliar_ficha({"user_id": uid, "plan_pagado": "gratis"})
    assert f_no["verificador_android"] is False
