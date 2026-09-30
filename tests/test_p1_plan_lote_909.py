"""[P1-PLAN-LOTE-909 · 2026-09-30] Diagnóstico anónimo del dictado y el modo voz (Android: «todavía tienen problemas»).

Sin red: la base es falsa. Se prueba qué se acepta, qué se guarda (nada de texto libre, sin user_id) y el endpoint.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import diagnostico_voz as dv

_UA_XIAOMI = ("Mozilla/5.0 (Linux; Android 16; 2312DRA50G Build/BP2A.250605.031.A3; wv) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Version/4.0 Chrome/153.0.8010.36 Mobile Safari/537.36")


def test_normaliza_con_el_codigo_original_y_el_telefono_del_user_agent():
    m = dv.normalizar({"donde": "dictado", "codigo": "language-not-supported", "crudo": "UNKNOWN_12",
                       "motor": "plugin_android", "idioma": "es-DO", "plataforma": "android",
                       "plugin_reconocimiento": True}, _UA_XIAOMI, "61a13831-2a70-4437-a084-0d3e09b653e4")
    assert m["crudo"] == "UNKNOWN_12" and m["codigo"] == "language-not-supported"
    assert m["android"] == "16" and m["modelo"] == "2312DRA50G"
    assert m["plugin_reconocimiento"] is True
    assert len(m["cuenta"]) == 12 and "61a13831" not in json.dumps(m), "la cuenta solo como hash corto"


def test_rechaza_lo_que_no_es_un_diagnostico_y_no_guarda_texto_libre():
    assert dv.normalizar({"donde": "otra_cosa"}) is None
    assert dv.normalizar("x") is None
    m = dv.normalizar({"donde": "modo_voz", "codigo": "me comí un mangú con salami", "motor": {"a": 1},
                       "plugin_sintesis": "true"})
    assert m == {"donde": "modo_voz"}, "un código con espacios (texto dicho) o tipos raros no entran"


def test_registrar_va_a_pipeline_metrics_sin_user_id(monkeypatch):
    import db_core
    visto = {}
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: visto.update(sql=sql, params=params))
    dv.registrar({"donde": "estado", "motor": "ninguno"})
    assert "INSERT INTO pipeline_metrics" in visto["sql"] and "VALUES (NULL, NULL" in visto["sql"]
    assert visto["params"][0] == "voz_diagnostico"
    monkeypatch.setattr(db_core, "execute_sql_write", lambda *a: (_ for _ in ()).throw(RuntimeError("caída")))
    dv.registrar({"donde": "estado"})   # no lanza


def test_endpoint_204_y_400(monkeypatch):
    from fastapi import HTTPException
    from routers import chat
    guardados = []
    monkeypatch.setattr(dv, "registrar", guardados.append)

    class _Req:
        headers = {"user-agent": _UA_XIAOMI}

    r = asyncio.run(chat.api_chat_voz_diagnostico(_Req(), {"donde": "estado", "motor": "plugin_android"}, None))
    assert r.status_code == 204 and guardados[0]["modelo"] == "2312DRA50G" and "cuenta" not in guardados[0]
    with pytest.raises(HTTPException) as e:
        asyncio.run(chat.api_chat_voz_diagnostico(_Req(), {"donde": "x"}, None))
    assert e.value.status_code == 400


def test_la_ruta_existe_sin_permiso_de_ia_ni_cuota():
    import app as app_module
    import auth
    import consentimientos
    from fastapi.routing import APIRoute
    ruta = next(r for r in app_module.app.routes if isinstance(r, APIRoute) and r.path == "/api/chat/diagnostico-voz")

    def _usa(dep, obj):
        return any(d.call is obj or _usa(d, obj) for d in dep.dependencies)

    assert not _usa(ruta.dependant, consentimientos.requiere_consentimiento_ia), "no llama a ninguna IA"
    assert not _usa(ruta.dependant, auth.verify_api_quota)
