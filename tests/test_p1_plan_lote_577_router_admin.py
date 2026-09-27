# backend/tests/test_p1_plan_lote_577_router_admin.py
"""[P1-PLAN-LOTE-577 · 2026-09-27] Router del panel: 401 sin sesión, 404 a quien no está, métricas al dueño."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.admin as ra
from auth import get_verified_user_id

ADMIN = "11111111-1111-1111-1111-111111111111"
OTRO = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    monkeypatch.setattr(ra, "metricas", lambda dias: {"dias": dias, "bloques": []})

    def _hacer(uid):
        app = FastAPI()
        app.include_router(ra.router)
        app.dependency_overrides[get_verified_user_id] = lambda: uid
        return TestClient(app)
    return _hacer


def test_sin_sesion_401(cliente):
    assert cliente(None).get("/api/admin/metricas").status_code == 401


def test_fuera_de_la_lista_404(cliente):
    r = cliente(OTRO).get("/api/admin/metricas")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}


def test_el_dueno_ve_las_metricas(cliente):
    r = cliente(ADMIN).get("/api/admin/metricas?dias=30")
    assert r.status_code == 200 and r.json()["dias"] == 30


def test_dias_fuera_de_rango_422(cliente):
    assert cliente(ADMIN).get("/api/admin/metricas?dias=365").status_code == 422


def test_yo_anota_el_acceso_y_falla_cerrado(cliente, monkeypatch):
    anotadas = []
    monkeypatch.setattr(ra, "registrar_acceso", lambda uid, accion, *a, **k: anotadas.append((uid, accion)))
    assert cliente(ADMIN).get("/api/admin/yo").json() == {"ok": True}
    assert anotadas == [(ADMIN, "abrir_panel")]

    def _rota(*a, **k):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(ra, "registrar_acceso", _rota)
    assert cliente(ADMIN).get("/api/admin/yo").status_code == 503


def test_la_app_registra_el_router():
    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "from routers.admin import router as admin_router" in src and "app.include_router(admin_router)" in src
