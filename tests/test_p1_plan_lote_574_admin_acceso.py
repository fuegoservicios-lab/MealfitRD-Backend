# backend/tests/test_p1_plan_lote_574_admin_acceso.py
"""[P1-PLAN-LOTE-574 · 2026-09-27] Panel admin: quién entra (lista + interruptor) y el rastro fail-closed."""
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import admin_acceso
from auth import get_verified_user_id

ADMIN = "11111111-1111-1111-1111-111111111111"
RAIZ = Path(__file__).resolve().parents[2]


def _cliente(uid):
    app = FastAPI()

    @app.get("/x")
    def _x(admin: str = Depends(admin_acceso.require_admin)):
        return {"admin": admin}

    app.dependency_overrides[get_verified_user_id] = lambda: uid
    return TestClient(app)


@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", f" {ADMIN.upper()} , otro-id ")
    return monkeypatch


def test_la_lista_se_lee_limpia(entorno):
    assert admin_acceso.admin_ids() == frozenset({ADMIN, "otro-id"})
    assert admin_acceso.es_admin(ADMIN) and not admin_acceso.es_admin("33333333-3333-3333-3333-333333333333")
    assert not admin_acceso.es_admin(None)


def test_require_admin_401_404_200(entorno):
    assert _cliente(None).get("/x").status_code == 401
    assert _cliente("33333333-3333-3333-3333-333333333333").get("/x").status_code == 404
    r = _cliente(ADMIN).get("/x")
    assert r.status_code == 200 and r.json() == {"admin": ADMIN}


def test_con_el_panel_apagado_nadie_entra(entorno):
    entorno.setenv("MEALFIT_ADMIN_PANEL", "false")
    assert _cliente(ADMIN).get("/x").status_code == 404


def test_registrar_acceso_escribe_y_lanza_si_no_puede(monkeypatch):
    escritas = []
    monkeypatch.setattr(admin_acceso, "execute_sql_write", lambda q, p=None, **kw: escritas.append((q, p)))
    admin_acceso.registrar_acceso(ADMIN, "abrir_panel")
    q, p = escritas[0]
    assert "INSERT INTO public.admin_access_log" in q and p[0] == ADMIN and p[1] == "abrir_panel"

    def _rota(*a, **k):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(admin_acceso, "execute_sql_write", _rota)
    with pytest.raises(RuntimeError):
        admin_acceso.registrar_acceso(ADMIN, "abrir_panel")


def test_migracion_idempotente_en_los_dos_directorios():
    nombre = "p1_plan_lote_574_admin_access_log_2026_09_27.sql"
    a = (RAIZ / "migrations" / nombre).read_text(encoding="utf-8")
    assert a == (RAIZ / "backend" / "migrations" / nombre).read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS public.admin_access_log" in a and "RAISE EXCEPTION" in a and "auth." not in a
