# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-847 · 2026-09-29] Al borrar la cuenta, su persona y sus eventos se borran en PostHog.

Auditoría App Store 2026-09-29, fila 3.3: la app identifica al usuario en PostHog por su id y el borrado de cuenta no
contactaba con PostHog. Sin red: `posthog_borrado._peticion` es el único punto de red y aquí se sustituye.

Qué se fija:
  * busca la persona por `distinct_id = user_id` y la borra con `delete_events=true`, con la clave personal;
  * solo borra personas cuyo `distinct_ids` contiene el id (si el filtro se ignorara, no se toca a nadie más);
  * sin `POSTHOG_PERSONAL_API_KEY` o `POSTHOG_PROJECT_ID` es inerte (una advertencia, ninguna llamada);
  * nunca lanza, tiene plazo total y la clave jamás llega al log;
  * `/api/account/delete` la llama tras la purga, con la cuenta ya cerrada, y sigue aunque PostHog falle.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest

import posthog_borrado

_BACKEND = Path(__file__).resolve().parent.parent
_UID = "11111111-2222-3333-4444-555555555555"
_PERSONA = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
_CLAVE = "phx_clave_personal_SECRETA_847"
_ENV = ("POSTHOG_PERSONAL_API_KEY", "POSTHOG_PROJECT_ID", "POSTHOG_HOST", "MEALFIT_POSTHOG_ACCOUNT_DELETE")


@pytest.fixture()
def sin_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(posthog_borrado, "_avisado_sin_configurar", False)
    return monkeypatch


@pytest.fixture()
def configurado(sin_env):
    sin_env.setenv("POSTHOG_PERSONAL_API_KEY", _CLAVE)
    sin_env.setenv("POSTHOG_PROJECT_ID", "12345")
    return sin_env


@pytest.fixture()
def red(monkeypatch):
    llamadas: list = []
    respuestas: dict = {}

    def _peticion(metodo, url, clave, params=None):
        llamadas.append((metodo, url, clave, dict(params or {})))
        r = respuestas.get(metodo)
        if isinstance(r, Exception):
            raise r
        if r is not None:
            return r
        if metodo == "GET":
            return 200, {"results": [{"id": _PERSONA, "distinct_ids": [_UID, "anon-1"]}]}
        return 204, None

    monkeypatch.setattr(posthog_borrado, "_peticion", _peticion)
    return llamadas, respuestas


def test_busca_por_distinct_id_y_borra_la_persona_con_sus_eventos(configurado, red):
    llamadas, _ = red
    r = posthog_borrado.borrar_de_posthog(_UID)
    assert r == {"borrado": True, "motivo": "ok", "personas": 1}
    (m1, u1, c1, p1), (m2, u2, c2, p2) = llamadas
    assert (m1, u1, p1) == ("GET", "https://us.posthog.com/api/projects/12345/persons/", {"distinct_id": _UID})
    assert (m2, u2) == ("DELETE", f"https://us.posthog.com/api/projects/12345/persons/{_PERSONA}/")
    assert p2 == {"delete_events": "true", "delete_recordings": "true"}
    assert c1 == c2 == _CLAVE


def test_el_host_se_puede_cambiar(configurado, red):
    configurado.setenv("POSTHOG_HOST", "https://eu.posthog.com/")
    posthog_borrado.borrar_de_posthog(_UID)
    assert red[0][0][1] == "https://eu.posthog.com/api/projects/12345/persons/"


def test_solo_borra_personas_que_llevan_ese_distinct_id(configurado, red):
    llamadas, respuestas = red
    respuestas["GET"] = (200, {"results": [
        {"id": "aaaaaaaa-0000-0000-0000-000000000001", "distinct_ids": ["otra-persona"]},
        {"id": "../../otra-ruta", "distinct_ids": [_UID]},
    ]})
    assert posthog_borrado.borrar_de_posthog(_UID) == {"borrado": True, "motivo": "sin_persona", "personas": 0}
    assert [m for m, *_ in llamadas] == ["GET"]


def test_sin_persona_en_posthog_no_borra_nada(configurado, red):
    llamadas, respuestas = red
    respuestas["GET"] = (200, {"results": []})
    assert posthog_borrado.borrar_de_posthog(_UID)["motivo"] == "sin_persona"
    assert len(llamadas) == 1


@pytest.mark.parametrize("respuesta,motivo", [
    ({"GET": (401, {"detail": "x"})}, "buscar_http_401"),
    ({"GET": (200, "no-json")}, "buscar_http_200"),
    ({"DELETE": (500, None)}, "borrar_http_500"),
    ({"GET": RuntimeError("timeout con https://us.posthog.com")}, "error"),
    ({"DELETE": RuntimeError("boom")}, "error"),
])
def test_un_fallo_de_posthog_no_lanza(configurado, red, respuesta, motivo):
    red[1].update(respuesta)
    r = posthog_borrado.borrar_de_posthog(_UID)
    assert r["borrado"] is False and r["motivo"] == motivo


def test_404_al_borrar_cuenta_como_hecho(configurado, red):
    red[1]["DELETE"] = (404, None)
    assert posthog_borrado.borrar_de_posthog(_UID)["borrado"] is True


@pytest.mark.parametrize("falta", ["POSTHOG_PERSONAL_API_KEY", "POSTHOG_PROJECT_ID"])
def test_sin_configurar_es_inerte_y_avisa_una_vez(configurado, red, caplog, falta):
    configurado.delenv(falta)
    with caplog.at_level(logging.WARNING, logger="posthog_borrado"):
        assert posthog_borrado.borrar_de_posthog(_UID) == {"borrado": False, "motivo": "sin_configurar"}
        assert posthog_borrado.borrar_de_posthog(_UID) == {"borrado": False, "motivo": "sin_configurar"}
    assert red[0] == []
    avisos = [r for r in caplog.records if "P1-PLAN-LOTE-847" in r.getMessage()]
    assert len(avisos) == 1


@pytest.mark.parametrize("proyecto,host", [("12/../34", ""), ("12345", "http://us.posthog.com")])
def test_configuracion_malformada_es_inerte(configurado, red, proyecto, host):
    configurado.setenv("POSTHOG_PROJECT_ID", proyecto)
    if host:
        configurado.setenv("POSTHOG_HOST", host)
    assert posthog_borrado.borrar_de_posthog(_UID)["motivo"] == "sin_configurar"
    assert red[0] == []


def test_el_interruptor_lo_apaga(configurado, red):
    configurado.setenv("MEALFIT_POSTHOG_ACCOUNT_DELETE", "false")
    assert posthog_borrado.borrar_de_posthog(_UID)["motivo"] == "apagado"
    assert red[0] == []


@pytest.mark.parametrize("uid", ["", None, "guest"])
def test_sin_usuario_no_llama(configurado, red, uid):
    assert posthog_borrado.borrar_de_posthog(uid)["motivo"] == "sin_usuario"
    assert red[0] == []


def test_la_clave_nunca_llega_al_log(configurado, red, caplog):
    red[1]["GET"] = RuntimeError(f"fallo con Authorization: Bearer {_CLAVE}")
    with caplog.at_level(logging.DEBUG):
        posthog_borrado.borrar_de_posthog(_UID)
        red[1]["GET"] = (500, None)
        posthog_borrado.borrar_de_posthog(_UID)
        red[1].clear()
        posthog_borrado.borrar_de_posthog(_UID)
    assert caplog.records
    assert all(_CLAVE not in r.getMessage() for r in caplog.records)
    assert all(_UID not in r.getMessage() for r in caplog.records)


def test_la_peticion_real_lleva_la_clave_en_la_cabecera_y_timeouts(monkeypatch):
    import httpx

    vistos: dict = {}

    class _Cliente:
        def __init__(self, timeout=None):
            vistos["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def request(self, metodo, url, params=None, headers=None):
            vistos.update(metodo=metodo, url=url, params=params, headers=headers)
            return httpx.Response(204, request=httpx.Request(metodo, url))

    monkeypatch.setattr(httpx, "Client", _Cliente)
    status, cuerpo = posthog_borrado._peticion("DELETE", "https://us.posthog.com/x/", _CLAVE, {"delete_events": "true"})
    assert status == 204 and cuerpo is None
    assert vistos["headers"] == {"Authorization": f"Bearer {_CLAVE}"}
    assert vistos["timeout"].connect == posthog_borrado._HTTP_CONNECT_TIMEOUT_S


def test_tiene_plazo_total_y_nunca_lanza(monkeypatch):
    monkeypatch.setattr(posthog_borrado, "borrar_de_posthog", lambda uid: time.sleep(1) or {"borrado": True})

    async def _medir():
        # Se mide DENTRO del bucle: `asyncio.run` espera al hilo al cerrar su executor; el servidor no.
        t0 = time.monotonic()
        r = await posthog_borrado.borrar_de_posthog_con_plazo(_UID, plazo_s=0.05)
        return r, time.monotonic() - t0

    r, espera = asyncio.run(_medir())
    assert r == {"borrado": False, "motivo": "plazo"}
    assert espera < 0.5
    assert 5 <= posthog_borrado.DELETE_DEADLINE_S <= 15

    def _explota(uid):
        raise RuntimeError("x")

    monkeypatch.setattr(posthog_borrado, "borrar_de_posthog", _explota)
    assert asyncio.run(posthog_borrado.borrar_de_posthog_con_plazo(_UID)) == {"borrado": False, "motivo": "error"}


# ─────────────────────────── el borrado de cuenta ───────────────────────────

def _cuerpo_borrado() -> str:
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    i = app.index("async def api_delete_my_account")
    return app[i:app.index("\n@app.", i)]


def test_el_borrado_llama_a_posthog_tras_la_purga_y_con_la_cuenta_cerrada():
    cuerpo = _cuerpo_borrado()
    posthog = cuerpo.index("await borrar_de_posthog_con_plazo(verified_user_id)")
    assert cuerpo.index("await revocar_con_plazo(verified_user_id)") < posthog
    assert cuerpo.index("asyncio.to_thread(delete_account_data") < posthog
    assert cuerpo.index("if not identity_deleted:") < posthog < cuerpo.index("clear_session_cookie(response)")
    assert "tooltip-anchor: P1-PLAN-LOTE-847-BORRAR-POSTHOG" in cuerpo


@pytest.fixture(scope="module")
def app_module():
    import app as _app

    return _app


def _preparar(app_module, monkeypatch, orden, identidad=True):
    import apple_tokens
    import db_profiles

    monkeypatch.setattr(app_module, "execute_sql_query", lambda *a, **k: None)   # sin suscripción PayPal
    monkeypatch.setattr(app_module, "execute_sql_write", lambda *a, **k: True)
    monkeypatch.setattr(apple_tokens, "revocar_de_usuario", lambda uid: orden.append("apple") or {"motivo": "ok"})

    def _purga(uid, include_profile=True):
        orden.append("purga")
        return {"user_id": uid, "deleted": {}, "anonymized": {}, "errors": [], "failed_steps": [],
                "profile_deleted": identidad, "identity_deleted": identidad, "storage_objects_removed": 0}

    monkeypatch.setattr(db_profiles, "delete_account_data", _purga)


@pytest.mark.parametrize("resultado", [
    {"borrado": True, "motivo": "ok"}, {"borrado": False, "motivo": "borrar_http_500"}, RuntimeError("boom"),
])
def test_el_borrado_llama_a_posthog_y_sigue_aunque_falle(app_module, monkeypatch, resultado):
    from starlette.responses import Response

    orden: list = []
    _preparar(app_module, monkeypatch, orden)

    def _posthog(uid):
        orden.append(("posthog", uid))
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    monkeypatch.setattr(posthog_borrado, "borrar_de_posthog", _posthog)
    out = asyncio.run(app_module.api_delete_my_account(
        response=Response(), data={"confirm": "ELIMINAR"}, verified_user_id=_UID))
    assert orden == ["apple", "purga", ("posthog", _UID)]
    assert out["success"] is True and out["identity_deleted"] is True


def test_si_la_cuenta_sigue_viva_no_se_toca_posthog(app_module, monkeypatch):
    from fastapi import HTTPException
    from starlette.responses import Response

    orden: list = []
    _preparar(app_module, monkeypatch, orden, identidad=False)
    monkeypatch.setattr(posthog_borrado, "borrar_de_posthog", lambda uid: orden.append("posthog"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(app_module.api_delete_my_account(
            response=Response(), data={"confirm": "ELIMINAR"}, verified_user_id=_UID))
    assert e.value.status_code == 503
    assert "posthog" not in orden
