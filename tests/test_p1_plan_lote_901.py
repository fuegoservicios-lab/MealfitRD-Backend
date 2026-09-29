"""[P1-PLAN-LOTE-901 · 2026-09-29] La voz del coach en streaming.

El dueño, tras el 688: «siento que todavía es muy lenta la latencia de respuesta del agente de voz». Medido en su turno
(«Activa la hidratación», 17:39 UTC): la primera frase SONÓ a los 5,2 s del mensaje, 2,1 s de ellos esperando el WAV
entero de `generateContent`. `streamGenerateContent?alt=sse` da el primer trozo a ~0,65 s (medido desde el VPS).
El endpoint espera ese primer trozo antes de responder: si Google falla, es 204 (voz del teléfono), nunca un 200 mudo.
"""
from __future__ import annotations

import asyncio
import base64
import json

import pytest

import coach_voz


def _evento(pcm: bytes = b"", uso: dict | None = None, fin: bool = False) -> str:
    cand = {"content": {"parts": [{"inlineData": {"mimeType": "audio/l16; rate=24000; channels=1",
                                                   "data": base64.b64encode(pcm).decode()}}]}} if pcm else {"finishReason": "STOP"}
    ev = {"candidates": [cand]}
    if uso:
        ev["usageMetadata"] = uso
    return "data: " + json.dumps(ev)


class _RespuestaFlujo:
    def __init__(self, lineas, status=200):
        self._lineas = lineas
        self.status_code = status
        self.cerrada = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lineas

    def close(self):
        self.cerrada = True


class _ClienteFlujo:
    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.peticiones = []

    def build_request(self, metodo, url, headers=None, json=None, timeout=None):
        self.peticiones.append({"metodo": metodo, "url": url, "headers": headers, "json": json})
        return object()

    def send(self, peticion, stream=False):
        assert stream is True, "sin stream=True httpx lee el cuerpo entero antes de devolver"
        return self.respuesta


@pytest.fixture
def uso_registrado(monkeypatch):
    visto = []
    monkeypatch.setattr(coach_voz, "registrar_uso", lambda voz, uid, extra=None: visto.append((voz, uid, extra)))
    monkeypatch.setattr(coach_voz, "_clave", lambda: "k")
    return visto


def _cliente_con(monkeypatch, lineas, status=200):
    r = _RespuestaFlujo(lineas, status)
    c = _ClienteFlujo(r)
    monkeypatch.setattr(coach_voz, "_cliente", lambda: c)
    return c, r


# ── 1. coach_voz.abrir_flujo ────────────────────────────────────────────────────────────────────────────────────

def test_abre_el_streaming_sse_y_devuelve_tras_el_primer_audio(monkeypatch, uso_registrado):
    c, r = _cliente_con(monkeypatch, [
        "",
        _evento(uso={"promptTokenCount": 5, "candidatesTokenCount": 0}),   # un evento sin audio no cuenta
        _evento(b"\x01\x00\x02\x00", {"promptTokenCount": 5, "candidatesTokenCount": 2}),
        _evento(b"\x03\x00", {"promptTokenCount": 5, "candidatesTokenCount": 4}),
        _evento(fin=True, uso={"promptTokenCount": 5, "candidatesTokenCount": 70}),
    ])
    flujo = coach_voz.abrir_flujo("Listo, anotado.", "es-DO", "u1")
    assert ":streamGenerateContent?alt=sse" in c.peticiones[0]["url"]
    assert c.peticiones[0]["json"]["generationConfig"]["speechConfig"]["languageCode"] == "es-US"
    assert flujo.frecuencia == 24000
    assert uso_registrado == [], "el gasto se registra al TERMINAR el flujo, no al abrirlo"
    assert b"".join(flujo) == b"\x01\x00\x02\x00\x03\x00"
    assert r.cerrada
    voz, uid, extra = uso_registrado[0]
    assert (voz.tokens_texto, voz.tokens_audio, uid) == (5, 70, "u1"), "vale el ÚLTIMO usageMetadata (acumulado)"
    assert extra["flujo"] is True and "primer_audio_s" in extra


def test_google_caido_levanta_y_cierra_la_conexion(monkeypatch, uso_registrado):
    _c, r = _cliente_con(monkeypatch, [], status=503)
    with pytest.raises(RuntimeError):
        coach_voz.abrir_flujo("Hola", "es-DO", "u1")
    assert r.cerrada and uso_registrado == []


def test_un_streaming_sin_audio_levanta(monkeypatch, uso_registrado):
    _c, r = _cliente_con(monkeypatch, [_evento(fin=True)])
    with pytest.raises(RuntimeError):
        coach_voz.abrir_flujo("Hola", "es-DO", "u1")
    assert r.cerrada


def test_texto_vacio_no_llama(monkeypatch, uso_registrado):
    c, _r = _cliente_con(monkeypatch, [])
    assert coach_voz.abrir_flujo("   ", "es-DO") is None
    assert c.peticiones == []


def test_cortar_el_flujo_a_medias_cierra_y_registra(monkeypatch, uso_registrado):
    _c, r = _cliente_con(monkeypatch, [_evento(b"\x01\x00"), _evento(b"\x02\x00"), _evento(b"\x03\x00")])
    flujo = coach_voz.abrir_flujo("Hola", "es-DO", "u1")
    it = iter(flujo)
    assert next(it) == b"\x01\x00"
    it.close()   # el cliente se fue
    assert r.cerrada and len(uso_registrado) == 1


# ── 2. El endpoint ──────────────────────────────────────────────────────────────────────────────────────────────

def _llamar(data, uid="u1"):
    from routers import chat
    return asyncio.run(chat.api_chat_voz_flujo(data, uid, None))


@pytest.fixture
def flujo_ok(monkeypatch):
    monkeypatch.delenv("MEALFIT_COACH_VOZ_NUBE", raising=False)
    monkeypatch.delenv("MEALFIT_COACH_VOZ_FLUJO", raising=False)
    monkeypatch.setattr(coach_voz, "hay_presupuesto", lambda: True)

    class _F:
        frecuencia = 24000
        primer_audio_ms = 640

        def __iter__(self):
            yield b"\x01\x00"

    monkeypatch.setattr(coach_voz, "abrir_flujo", lambda texto, locale, uid: _F())


def test_responde_el_pcm_en_streaming_sin_buffer_de_nginx(flujo_ok):
    from fastapi.responses import StreamingResponse
    r = _llamar({"texto": "Listo.", "locale": "es-DO"})
    assert isinstance(r, StreamingResponse) and r.status_code == 200
    assert r.headers["x-accel-buffering"] == "no"
    assert r.headers["x-voz-frecuencia"] == "24000"
    assert r.headers["cache-control"] == "no-store"


def test_streaming_apagado_por_knob_pide_el_wav(flujo_ok, monkeypatch):
    monkeypatch.setenv("MEALFIT_COACH_VOZ_FLUJO", "0")
    r = _llamar({"texto": "Hola"})
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "flujo_apagado"


def test_voz_apagada_y_sin_presupuesto_son_204(flujo_ok, monkeypatch):
    monkeypatch.setattr(coach_voz, "hay_presupuesto", lambda: False)
    assert _llamar({"texto": "Hola"}).headers["x-voz-motivo"] == "presupuesto"
    monkeypatch.setenv("MEALFIT_COACH_VOZ_NUBE", "0")
    assert _llamar({"texto": "Hola"}).headers["x-voz-motivo"] == "apagada"


def test_google_caido_es_204(flujo_ok, monkeypatch):
    def _boom(texto, locale, uid):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(coach_voz, "abrir_flujo", _boom)
    r = _llamar({"texto": "Hola"})
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "error"


def test_el_endpoint_esta_protegido_fuera_del_paywall_y_no_bloquea_el_worker():
    import inspect
    from routers import chat
    src = inspect.getsource(chat.api_chat_voz_flujo)
    assert "Depends(get_verified_user_id)" in src and "Depends(_VOZ_LIMITER)" in src
    assert "verify_api_quota" not in src and "verify_coach_quota" not in src
    assert "asyncio.to_thread(abrir_flujo" in src
    assert '@router.post("/voz/flujo")' in inspect.getsource(chat)
