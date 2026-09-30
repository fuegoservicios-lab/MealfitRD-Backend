"""[P1-PLAN-LOTE-835 · 2026-09-29] Lo que ve la PERSONA de su cuenta de prueba (spec §5 y §7): el perfil trae
`cuenta_de_prueba` solo con el interruptor maestro encendido, la app anota que enseñó el aviso y la persona SALE cuando
quiere — salir funciona SIEMPRE, también con el interruptor apagado, y nunca cobra cuota (cero IA: al llegar al tope la
persona tiene que poder salir). Todo con la base falsa del lote 830: nada de este fichero toca Neon.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import cuentas_prueba as cp
from auth import get_verified_user_id
from tests.test_p1_plan_lote_830 import ADMIN, MOTIVO, UID
from tests.test_p1_plan_lote_830 import _BD as _BDPrueba

_BACKEND = Path(__file__).resolve().parents[1]


def _ud():
    from routers import user_data
    return user_data


@pytest.fixture
def bd(monkeypatch):
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEALFIT_ADMIN_TEST_REQUIRE_NOTICE", raising=False)
    b = _BDPrueba()
    monkeypatch.setattr(cp, "execute_sql_query", b.query)
    monkeypatch.setattr(cp, "execute_sql_write", b.write)
    monkeypatch.setattr(cp, "registrar_acceso", b.anotar)
    _ud()._PRUEBA_LIMITER._hits.clear()
    yield b
    _ud()._PRUEBA_LIMITER._hits.clear()


def _encender(monkeypatch):
    monkeypatch.setenv("MEALFIT_ADMIN_TEST_ACCOUNTS", "true")


def _perfil(monkeypatch):
    import db
    perfil = {"id": UID, "email": "ana@correo.com", "health_profile": {}}
    monkeypatch.setattr(db, "get_user_profile", lambda uid: dict(perfil) if uid == UID else None)
    return perfil


def _rota(*a, **k):
    raise RuntimeError("sin DB")


# ═════════════════════════════════════════════ 1. el perfil
def test_con_el_interruptor_apagado_el_perfil_ni_trae_la_clave(bd, monkeypatch):
    _perfil(monkeypatch)
    monkeypatch.setattr(cp, "para_la_persona", lambda uid: pytest.fail("apagado no se mira la marca"))
    out = asyncio.run(_ud().api_get_profile(verified_user_id=UID))
    assert "cuenta_de_prueba" not in out["profile"]


def test_con_el_interruptor_el_perfil_trae_la_marca_de_la_persona(bd, monkeypatch):
    _encender(monkeypatch)
    _perfil(monkeypatch)
    ud = _ud()
    assert asyncio.run(ud.api_get_profile(verified_user_id=UID))["profile"]["cuenta_de_prueba"] is None
    cp.marcar(ADMIN, UID, MOTIVO)
    viva = asyncio.run(ud.api_get_profile(verified_user_id=UID))["profile"]["cuenta_de_prueba"]
    assert set(viva) == {"desde", "aviso_visto"} and viva["aviso_visto"] is False and viva["desde"]
    cp.aviso_visto(UID)
    assert asyncio.run(ud.api_get_profile(verified_user_id=UID))["profile"]["cuenta_de_prueba"]["aviso_visto"] is True


def test_si_la_marca_no_se_lee_el_perfil_carga_igual(bd, monkeypatch):
    _encender(monkeypatch)
    perfil = _perfil(monkeypatch)
    monkeypatch.setattr(cp, "execute_sql_query", _rota)
    out = asyncio.run(_ud().api_get_profile(verified_user_id=UID))["profile"]
    assert out["cuenta_de_prueba"] is None and out["email"] == perfil["email"]


# ═════════════════════════════════════════════ 2. el aviso visto
def test_el_aviso_visto_se_anota_en_la_marca_viva(bd, monkeypatch):
    _encender(monkeypatch)
    cp.marcar(ADMIN, UID, MOTIVO)
    assert asyncio.run(_ud().api_prueba_aviso_visto(verified_user_id=UID)) == {"ok": True}
    assert bd.filas[0]["aviso_visto_at"] is not None
    cuando = bd.filas[0]["aviso_visto_at"]
    assert asyncio.run(_ud().api_prueba_aviso_visto(verified_user_id=UID)) == {"ok": True}
    assert bd.filas[0]["aviso_visto_at"] == cuando, "solo la primera vez"


def test_con_el_interruptor_apagado_el_aviso_no_anota_nada(bd, monkeypatch):
    # La app solo enseña el aviso con el interruptor encendido: apagado, anotar «visto» abriría el contenido después
    # sin que la persona lo haya visto con el programa en marcha.
    cp.marcar(ADMIN, UID, MOTIVO)
    bd.vaciar()
    assert asyncio.run(_ud().api_prueba_aviso_visto(verified_user_id=UID)) == {"ok": True}
    assert bd.escrituras == [] and bd.filas[0]["aviso_visto_at"] is None


def test_si_el_aviso_no_se_guarda_503(bd, monkeypatch):
    _encender(monkeypatch)
    monkeypatch.setattr(cp, "execute_sql_write", _rota)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_ud().api_prueba_aviso_visto(verified_user_id=UID))
    assert ei.value.status_code == 503


# ═════════════════════════════════════════════ 3. salir
@pytest.mark.parametrize("encendido", [False, True])
def test_salir_funciona_siempre(bd, monkeypatch, encendido):
    if encendido:
        _encender(monkeypatch)
    cp.marcar(ADMIN, UID, MOTIVO)
    assert asyncio.run(_ud().api_prueba_salir(verified_user_id=UID)) == {"ok": True, "salio": True}
    fila = bd.filas[0]
    assert fila["quitada_at"] is not None and fila["quitada_por"] == UID and fila["quitada_por_la_persona"] is True
    assert asyncio.run(_ud().api_prueba_salir(verified_user_id=UID)) == {"ok": True, "salio": False}


def test_salir_sin_base_es_503_nunca_un_falso_salio(bd, monkeypatch):
    monkeypatch.setattr(cp, "execute_sql_write", _rota)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_ud().api_prueba_salir(verified_user_id=UID))
    assert ei.value.status_code == 503


@pytest.mark.parametrize("nombre", ["api_prueba_aviso_visto", "api_prueba_salir"])
def test_sin_sesion_401(bd, nombre):
    with pytest.raises(HTTPException) as ei:
        asyncio.run(getattr(_ud(), nombre)(verified_user_id=None))
    assert ei.value.status_code == 401


def test_las_rutas_de_la_persona_por_http(bd, monkeypatch):
    cp.marcar(ADMIN, UID, MOTIVO)
    app = FastAPI()
    app.include_router(_ud().router)
    app.dependency_overrides[get_verified_user_id] = lambda: UID
    c = TestClient(app)
    assert c.post("/api/profile/prueba/aviso-visto").json() == {"ok": True}
    assert c.post("/api/profile/prueba/salir").json() == {"ok": True, "salio": True}


# ═════════════════════════════════════════════ 4. limitador y cuota
def _src() -> str:
    return (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("ruta", ["/profile/prueba/aviso-visto", "/profile/prueba/salir"])
def test_las_rutas_de_la_persona_tienen_su_limitador_y_no_cobran_cuota(ruta):
    src = _src()
    i = src.index(f'@router.post("{ruta}")')
    firma = src[i:src.index("):", i)]
    assert "Depends(_PRUEBA_LIMITER)" in firma and "verify_api_quota" not in firma


def test_el_limitador_de_la_persona_tiene_su_par_propio():
    src = _src()
    m = re.search(r"_PRUEBA_LIMITER = RateLimiter\(max_calls=(\d+), period_seconds=(\d+)\)", src)
    assert m, "el limitador existe con sus números a la vista"
    assert re.fullmatch(r"_[A-Z_]+_LIMITER", "_PRUEBA_LIMITER"), "el guard de limitadores no admite dígitos"
    pares = []
    for f in [*_BACKEND.glob("*.py"), *(_BACKEND / "routers").glob("*.py")]:
        pares += re.findall(r"RateLimiter\(\s*(?:max_calls\s*=\s*)?(\d+)\s*,\s*(?:period_seconds\s*=\s*)?(\d+)\s*\)",
                            f.read_text(encoding="utf-8"))
    assert pares.count(m.groups()) == 1, "Redis comparte la ventana por par: salir no puede quedarse sin cupo por otro"


def test_el_limitador_corta_en_su_cupo(bd, monkeypatch):
    from starlette.requests import Request
    import rate_limiter
    monkeypatch.setattr(rate_limiter, "redis_client", None)
    lim = _ud()._PRUEBA_LIMITER
    peticion = Request({"type": "http", "client": ("1.2.3.4", 1), "headers": []})
    for _ in range(lim.max_calls):
        assert lim(peticion, UID) == UID
    with pytest.raises(HTTPException) as ei:
        lim(peticion, UID)
    assert ei.value.status_code == 429
