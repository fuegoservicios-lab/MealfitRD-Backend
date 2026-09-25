# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-300 · 2026-09-25] Push NATIVA en iOS: directo a Apple (APNs, HTTP/2 + JWT ES256), sin Firebase.

El dueño creó la clave APNs (Key ID LA6W6S739L, Team TKGU85LT22) y activó Push en el App ID `com.bioboros.app`. En el
VPS: `APNS_KEY_FILE` (fuera del repo), `APNS_KEY_ID`, `APNS_TEAM_ID`, `APNS_TOPIC`. El teléfono iOS registra su token
APNs en `device_push_tokens` con platform='ios' (lote 280); `fcm_push.enviar_a_dispositivos` reparte: android → FCM,
ios → `apns_push`. Frontend: `src/__tests__/lote300.test.js`. Tooltip-anchor: P1-PLAN-LOTE-300
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_UID = "61a13831-0000-4000-8000-000000000001"


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


def test_el_mensaje_apns_lleva_alerta_sonido_y_ruta():
    import apns_push
    h, body = apns_push.construir(("t" * 64), "Tu plan está listo 🎉", "Toca para verlo.", "/dashboard", "plan-listo",
                                  True, jwt_token="JWT", topic="com.bioboros.app")
    assert h["authorization"] == "bearer JWT" and h["apns-topic"] == "com.bioboros.app"
    assert h["apns-push-type"] == "alert" and h["apns-priority"] == "10" and h["apns-collapse-id"] == "plan-listo"
    b = json.loads(body)
    assert b["aps"]["alert"] == {"title": "Tu plan está listo 🎉", "body": "Toca para verlo."}
    assert b["aps"]["sound"] == "default" and b["url"] == "/dashboard" and b["solo_si_no_mira"] == "1"


def test_sin_configuracion_no_hay_apns(monkeypatch):
    import apns_push
    for k in ("APNS_KEY_FILE", "APNS_KEY_ID", "APNS_TEAM_ID"):
        monkeypatch.delenv(k, raising=False)
    assert apns_push.apns_configurado() is False


class _R:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


def test_reparte_por_plataforma_y_borra_el_token_muerto(monkeypatch):
    import fcm_push
    import apns_push
    import db_core
    borrados, fcm, apns = [], [], []
    monkeypatch.setattr(db_core, "execute_sql_query", lambda sql, p, fetch_all=False, **k: [
        {"token": "a" * 40, "platform": "android"}, {"token": "b" * 64, "platform": "ios"},
        {"token": "c" * 64, "platform": "ios"}])
    monkeypatch.setattr(db_core, "execute_sql_write", lambda sql, p: borrados.append(p[0]))
    monkeypatch.setattr(fcm_push, "_credenciales", lambda: (type("C", (), {"token": "x", "valid": True})(), "p"))
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: fcm.append(1) or _R(200))
    monkeypatch.setattr(apns_push, "apns_configurado", lambda: True)
    monkeypatch.setattr(apns_push, "_enviar_uno", lambda token, *a, **k: apns.append(token) or
                        (_R(200) if token.startswith("b") else _R(410, '{"reason":"Unregistered"}')))
    n = fcm_push.enviar_a_dispositivos(_UID, "t", "b", "/dashboard", "x", False)
    assert n == 2 and len(fcm) == 1 and apns == ["b" * 64, "c" * 64] and borrados == ["c" * 64]


def test_send_push_notification_llama_a_nativa_si_hay_apns_aunque_no_haya_fcm(monkeypatch):
    import utils_push
    import fcm_push
    import apns_push
    llamadas = []
    monkeypatch.setattr(utils_push, "_enviar_web_push", lambda *a, **k: False)
    monkeypatch.setattr(utils_push, "_traducidos", lambda uid, t, b: (t, b))
    monkeypatch.setattr(fcm_push, "fcm_configurado", lambda: False)
    monkeypatch.setattr(apns_push, "apns_configurado", lambda: True)
    monkeypatch.setattr(fcm_push, "enviar_a_dispositivos", lambda *a, **k: llamadas.append(1) or 1)
    assert utils_push.send_push_notification(_UID, "t", "b") is True and llamadas == [1]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 300


def test_la_clave_p8_no_esta_en_el_repo():
    for p in list(_BACKEND.rglob("*.p8")) + list(_BACKEND.parent.glob("*.p8")):
        pytest.fail(f"clave APNs dentro del repo: {p}")


# ── Detalles menores de los suplementos (pedido del dueño, 25-sep) ──────────────────────────────────────────────────

def test_M1_el_formulario_enciende_la_nevera_aunque_los_potes_ya_existan(monkeypatch):
    import suplementos
    import nevera_opcional
    from routers import user_data
    encendidas = []
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda uid, forzar=False: encendidas.append(forzar) or "encendida")
    monkeypatch.setattr(suplementos, "buscar", lambda uid, n: {"id": 1})     # todos ya estaban
    monkeypatch.setattr(suplementos, "guardar", lambda *a, **k: {"ok": True})
    r = user_data.guardar_suplementos_del_formulario(user_data.SuplementosFormulario(claves=["creatine"]), user_id="u")
    assert r == {"guardados": 0} and encendidas == [True]


def test_M4_invitado_recibe_un_mensaje_claro():
    import tools
    out = tools.guardar_suplemento.func("guest-session-123", "Creatina")
    assert "iniciar sesión" in out and "error interno" not in out


def test_M5_nombres_casi_iguales_resuelven_al_pote_correcto(monkeypatch):
    import suplementos
    import db_core
    filas = [{"id": 1, "ingredient_name": "Proteína Whey", "unit": "sup_scoop", "serving_unit": "scoop"},
             {"id": 2, "ingredient_name": "Proteína Vegana", "unit": "sup_scoop", "serving_unit": "scoop"}]
    monkeypatch.setattr(db_core, "execute_sql_query", lambda *a, **k: filas)
    assert suplementos.buscar("u", "proteína whey")["id"] == 1          # exacto (sin mayúsculas ni tildes)
    assert suplementos.buscar("u", "whey")["id"] == 1                   # único parecido
    assert suplementos.buscar("u", "proteína") is None                  # ambiguo: no adivina


def test_M5_guardar_actualiza_el_pote_existente_con_su_nombre(monkeypatch):
    import suplementos
    import nevera_opcional
    monkeypatch.setattr(nevera_opcional, "encender_por_uso", lambda uid, forzar=False: "activa")
    monkeypatch.setattr(suplementos, "buscar", lambda uid, n: {"id": 1, "ingredient_name": "Proteína Whey",
                                                                "serving_unit": "scoop"})
    escritos = []
    monkeypatch.setattr(suplementos, "_upsert", lambda *a: escritos.append(a))
    suplementos.guardar("u", "proteína whey", None, 20, "g", None, "estimado", "whey_protein")
    assert escritos[0][1] == "Proteína Whey" and escritos[0][4] == "scoop"
