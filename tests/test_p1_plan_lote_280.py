# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-280 · 2026-09-25] Push NATIVA con Firebase Cloud Messaging (Android).

La app nativa no recibía nada del servidor: los avisos que decide el backend (plan listo, coach proactivo, Nevera,
pausas) solo llegaban a la web/PWA por Web Push. El dueño creó el proyecto `bioboros-9e71d`; ahora
`utils_push.send_push_notification` envía también por FCM (`fcm_push.py`, API HTTP v1 con `google-auth`) a los tokens
de `device_push_tokens`. Los recordatorios de comida y agua van con `nativa=False`: en el teléfono ya son avisos
LOCALES y por FCM llegarían dos veces. Frontend: `src/__tests__/lote280.test.js`.

Tooltip-anchor: P1-PLAN-LOTE-280
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_UID = "61a13831-0000-4000-8000-000000000001"
_TOKEN = "t" * 40


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_el_mensaje_fcm_lleva_canal_etiqueta_y_la_marca_de_no_mirar():
    import fcm_push
    m = fcm_push.construir_mensaje(_TOKEN, "Tu plan está listo 🎉", "Toca para verlo.", "/dashboard", "plan-listo", True)
    msg = m["message"]
    assert msg["token"] == _TOKEN
    assert msg["notification"] == {"title": "Tu plan está listo 🎉", "body": "Toca para verlo."}
    assert msg["data"] == {"url": "/dashboard", "tag": "plan-listo", "solo_si_no_mira": "1"}
    assert msg["android"]["priority"] == "HIGH"
    assert msg["android"]["notification"]["channel_id"] == "bioboros-avisos"
    assert msg["android"]["notification"]["tag"] == "plan-listo"
    assert all(isinstance(v, str) for v in msg["data"].values())   # FCM exige data solo con strings


def test_sin_clave_no_hay_push_nativa(monkeypatch):
    import fcm_push
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_FILE", raising=False)
    assert fcm_push.fcm_configurado() is False
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_FILE", "C:/no/existe.json")
    assert fcm_push.fcm_configurado() is False


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


@pytest.fixture
def fcm(monkeypatch):
    import fcm_push
    import db_core
    import requests

    class _Cred:
        token = "ya29.falso"
        valid = True
    monkeypatch.setattr(fcm_push, "_credenciales", lambda: (_Cred(), "bioboros-9e71d"))
    estado = {"tokens": [_TOKEN, "m" * 40], "borrados": [], "posts": [], "respuestas": {}}
    monkeypatch.setattr(db_core, "execute_sql_query",
                        lambda sql, params, fetch_all=False, **k: [{"token": t} for t in estado["tokens"]])
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, params: estado["borrados"].append(params[0]))

    def _post(url, headers, data, timeout):
        body = json.loads(data)
        estado["posts"].append((url, headers["Authorization"], body))
        return estado["respuestas"].get(body["message"]["token"], _Resp(200))
    monkeypatch.setattr(requests, "post", _post)
    return fcm_push, estado


def test_envia_a_cada_telefono_y_borra_el_token_muerto(fcm):
    fcm_push, estado = fcm
    estado["respuestas"]["m" * 40] = _Resp(404, '{"error":{"status":"NOT_FOUND","details":[{"errorCode":"UNREGISTERED"}]}}')
    n = fcm_push.enviar_a_dispositivos(_UID, "Tu plan está listo 🎉", "Toca para verlo.", "/dashboard", "plan-listo", True)
    assert n == 1
    assert estado["posts"][0][0] == "https://fcm.googleapis.com/v1/projects/bioboros-9e71d/messages:send"
    assert estado["posts"][0][1] == "Bearer ya29.falso"
    assert estado["borrados"] == ["m" * 40]


def test_un_error_de_fcm_no_revienta(fcm, monkeypatch):
    fcm_push, estado = fcm
    import requests

    def revienta(*a, **k):
        raise requests.ConnectionError("sin red")
    monkeypatch.setattr(requests, "post", revienta)
    assert fcm_push.enviar_a_dispositivos(_UID, "t", "b") == 0


def test_send_push_notification_suma_la_nativa_y_respeta_nativa_false(monkeypatch):
    import utils_push
    import fcm_push
    enviados = []
    monkeypatch.setattr(utils_push, "_enviar_web_push", lambda *a, **k: False)
    monkeypatch.setattr(utils_push, "_traducidos", lambda uid, t, b: (t, b))
    monkeypatch.setattr(fcm_push, "fcm_configurado", lambda: True)
    monkeypatch.setattr(fcm_push, "enviar_a_dispositivos", lambda uid, t, b, **k: enviados.append((t, k)) or 1)
    assert utils_push.send_push_notification(_UID, "Título", "Cuerpo", tag="plan-listo", solo_si_no_mira=True) is True
    assert enviados == [("Título", {"url": "/dashboard", "tag": "plan-listo", "solo_si_no_mira": True})]
    enviados.clear()
    assert utils_push.send_push_notification(_UID, "¿Ya almorzaste?", "x", nativa=False) is False
    assert enviados == []


def test_los_recordatorios_de_comida_y_agua_no_van_por_fcm():
    pa = _src("proactive_agent.py")
    assert pa.count("nativa=False") == 2, "los dos avisos de comida (fijo y del coach) son locales en la app"
    hr = _src("hydration_reminders.py")
    assert "tag=ETIQUETA, nativa=False)" in hr, "el recordatorio de agua es local en la app"
    assert 'url="/dashboard/settings", tag=ETIQUETA)' in hr, "el aviso de «hidratación apagada» SÍ llega a la app"


def test_endpoint_y_migracion():
    rt = _src("routers/notifications.py")
    assert '@router.post("/device-token")' in rt and '@router.delete("/device-token")' in rt
    assert "Depends(_DEVICE_TOKEN_LIMITER)" in rt
    mig = _src("migrations/p1_plan_lote_280_device_push_tokens.sql")
    assert "CREATE TABLE IF NOT EXISTS public.device_push_tokens" in mig
    assert "ON DELETE CASCADE" in mig and "RAISE EXCEPTION" in mig
    fp = _src("fcm_push.py")
    assert "ON CONFLICT (token) DO UPDATE SET user_id = EXCLUDED.user_id" in fp


def test_la_clave_no_esta_en_el_repo():
    raiz = _BACKEND.parent
    for p in list(_BACKEND.rglob("*adminsdk*.json")) + list(raiz.glob("*adminsdk*.json")):
        pytest.fail(f"clave de servicio dentro del repo: {p}")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 280 and m.group(2) >= "2026-09-25"
