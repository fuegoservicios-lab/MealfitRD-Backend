"""[P1-PLAN-LOTE-685 · 2026-09-28] La voz del coach en el modo voz: Gemini 3.8 Flash-Lite TTS (`coach_voz.py`) detrás
de `POST /api/chat/voz`, con la voz del teléfono como respaldo (204) y un tope diario de gasto.

Contrato medido contra la API real el 28-sep (VPS): `generateContent` (sin streaming) devuelve `inlineData` con
`mimeType: audio/wav` (cabecera RIFF) y `usageMetadata` con los tokens de texto y de AUDIO (25 por segundo). Una
instrucción de estilo delante del texto se LEE en voz alta (el audio se duplicaba): el texto va tal cual y el acento
por `languageCode`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct

import pytest

import coach_voz


def _wav(segundos: float = 1.0, frecuencia: int = 24000) -> bytes:
    return coach_voz._envolver_wav(b"\x00\x00" * int(segundos * frecuencia), frecuencia)


class _Respuesta:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Cliente:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status
        self.peticiones = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.peticiones.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Respuesta(self.payload, self.status)


def _payload_de(audio: bytes, mime="audio/wav", tokens_texto=12, tokens_audio=50):
    return {
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode()}}]}}],
        "usageMetadata": {"promptTokenCount": tokens_texto, "candidatesTokenCount": tokens_audio},
    }


@pytest.fixture
def cliente(monkeypatch):
    c = _Cliente(_payload_de(_wav(2.0)))
    monkeypatch.setattr(coach_voz, "_cliente", lambda: c)
    monkeypatch.setattr(coach_voz, "_clave", lambda: "clave-de-prueba")
    for k in ("MEALFIT_COACH_VOZ_NOMBRE", "MEALFIT_COACH_VOZ_MODELO", "MEALFIT_COACH_VOZ_MAX_CARACTERES"):
        monkeypatch.delenv(k, raising=False)
    return c


# ── 1. La petición a Google ──────────────────────────────────────────────────────────────────────────────────────

def test_la_peticion_lleva_el_texto_tal_cual_y_el_acento_por_language_code(cliente):
    voz = coach_voz.sintetizar("  Bien,   aquí atento. ", "es-DO")
    p = cliente.peticiones[0]
    assert p["url"].endswith("/models/gemini-3.8-flash-lite-tts:generateContent")
    assert p["headers"] == {"x-goog-api-key": "clave-de-prueba"}          # la key NUNCA en la URL
    assert p["json"]["contents"] == [{"parts": [{"text": "Bien, aquí atento."}]}]   # sin instrucción de estilo
    cfg = p["json"]["generationConfig"]
    assert cfg["responseModalities"] == ["AUDIO"]
    assert cfg["speechConfig"]["languageCode"] == "es-US"                 # es-DO → voz latinoamericana
    assert cfg["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Sulafat"
    assert voz.wav[:4] == b"RIFF" and voz.segundos == 2.0
    assert (voz.tokens_texto, voz.tokens_audio) == (12, 50)


@pytest.mark.parametrize("locale,idioma", [("en-US", "en-US"), ("pt-BR", "pt-BR"), ("fr-FR", "fr-FR"),
                                            ("it-IT", "it-IT"), ("es", "es-US"), ("xx-YY", "es-US")])
def test_idioma_de_voz(locale, idioma):
    assert coach_voz.idioma_de_voz(locale) == idioma


def test_el_texto_se_recorta_y_el_vacio_no_llama(cliente, monkeypatch):
    monkeypatch.setenv("MEALFIT_COACH_VOZ_MAX_CARACTERES", "20")
    coach_voz.sintetizar("x" * 100, "es-DO")
    assert cliente.peticiones[0]["json"]["contents"][0]["parts"][0]["text"] == "x" * 20
    assert coach_voz.sintetizar("   ", "es-DO") is None
    assert len(cliente.peticiones) == 1


def test_pcm_crudo_se_envuelve_en_wav(monkeypatch):
    pcm = b"\x01\x00" * 24000          # 1 s a 24 kHz
    c = _Cliente(_payload_de(pcm, mime="audio/L16;codec=pcm;rate=24000"))
    monkeypatch.setattr(coach_voz, "_cliente", lambda: c)
    monkeypatch.setattr(coach_voz, "_clave", lambda: "k")
    voz = coach_voz.sintetizar("Hola", "es-DO")
    assert voz.wav[:4] == b"RIFF" and voz.wav[8:12] == b"WAVE"
    assert struct.unpack("<I", voz.wav[24:28])[0] == 24000
    assert voz.wav[44:] == pcm and voz.segundos == 1.0


def test_google_caido_levanta(monkeypatch):
    c = _Cliente({"error": "boom"}, status=503)
    monkeypatch.setattr(coach_voz, "_cliente", lambda: c)
    monkeypatch.setattr(coach_voz, "_clave", lambda: "k")
    with pytest.raises(RuntimeError):
        coach_voz.sintetizar("Hola", "es-DO")


# ── 2. La key: la del escáner si ya va a Google; si no, la de Gemini del entorno (fail-loud) ────────────────────

def test_la_clave_de_vision_si_el_escaner_ya_va_a_google(monkeypatch):
    monkeypatch.setenv("VISION_API_KEY", "clave-vision")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    assert coach_voz._clave() == "clave-vision"


def test_sin_escaner_de_google_usa_la_de_gemini_y_si_no_hay_revienta(monkeypatch):
    import llm_provider
    monkeypatch.setenv("VISION_API_KEY", "clave-de-otro-proveedor")
    monkeypatch.setenv("MEALFIT_VISION_BASE_URL", "https://api.otro-proveedor.test/v1")
    monkeypatch.setattr(llm_provider, "_google_api_key", lambda: "clave-gemini")
    assert coach_voz._clave() == "clave-gemini"

    def _sin_clave():
        raise RuntimeError("modelo Gemini pedido sin la key de Gemini en el entorno")
    monkeypatch.setattr(llm_provider, "_google_api_key", _sin_clave)
    with pytest.raises(RuntimeError):
        coach_voz._clave()


# ── 3. El precio y el tope diario ───────────────────────────────────────────────────────────────────────────────

def test_el_precio_de_la_voz_es_el_suyo_y_no_el_del_escaner():
    from db_profiles import compute_llm_cost_micros
    # 47 tokens de texto × US$0,50/M + 295 de audio × US$6/M = 23,5 + 1.770 µUSD (turno medido el 28-sep)
    assert abs(compute_llm_cost_micros("gemini-3.8-flash-lite-tts", 47, 295) - 1793.5) <= 1
    assert compute_llm_cost_micros("gemini-3.8-flash-lite-tts", 0, 250) == 1500   # 10 s de audio = US$0,0015


@pytest.fixture
def gasto(monkeypatch):
    estado = {"micros_db": 0, "lecturas": 0}

    def _leer():
        estado["lecturas"] += 1
        return estado["micros_db"]
    monkeypatch.setattr(coach_voz, "_leer_gasto_de_hoy_micros", _leer)
    monkeypatch.setitem(coach_voz._GASTO, "dia", None)
    monkeypatch.setenv("MEALFIT_COACH_VOZ_PRESUPUESTO_DIARIO_USD", "0.01")
    return estado


def test_el_tope_diario_corta_con_lo_leido_y_con_lo_gastado_entre_lecturas(gasto):
    assert coach_voz.hay_presupuesto() is True
    coach_voz._sumar_gasto_local(6000)                  # 0,006 USD gastados en este proceso
    assert coach_voz.hay_presupuesto() is True
    coach_voz._sumar_gasto_local(5000)                  # 0,011 ≥ 0,01 → sin presupuesto, sin esperar a la DB
    assert coach_voz.hay_presupuesto() is False
    assert gasto["lecturas"] == 1, "la DB se relee cada 60 s, no en cada frase"


def test_un_dia_nuevo_empieza_de_cero(gasto, monkeypatch):
    gasto["micros_db"] = 20_000
    assert coach_voz.hay_presupuesto() is False
    monkeypatch.setattr(coach_voz, "_hoy_utc", lambda: "2099-01-01")
    gasto["micros_db"] = 0
    assert coach_voz.hay_presupuesto() is True


def test_el_uso_va_a_llm_usage_events_y_nunca_a_api_usage(monkeypatch):
    import db
    visto = {}
    monkeypatch.setattr(db, "log_llm_usage_event", lambda **k: visto.update(k))
    monkeypatch.setitem(coach_voz._GASTO, "dia", coach_voz._hoy_utc())
    monkeypatch.setitem(coach_voz._GASTO, "micros_locales", 0)
    voz = coach_voz.VozSintetizada(wav=b"", segundos=4.0, tokens_texto=10, tokens_audio=100, ms=1900)
    coach_voz.registrar_uso(voz, "u1")
    assert visto["node"] == "coach_voice_tts" and visto["model"] == "gemini-3.8-flash-lite-tts"
    assert (visto["input_tokens"], visto["output_tokens"], visto["user_id"]) == (10, 100, "u1")
    assert coach_voz._GASTO["micros_locales"] == 605     # 10 × 0,5 + 100 × 6
    src = open(coach_voz.__file__, encoding="utf-8").read()
    assert "log_api_usage" not in src, "la voz no escribe en api_usage (quemaría créditos del plan)"


# ── 4. El endpoint: 204 = «usa la voz del teléfono», nunca un error en rojo ─────────────────────────────────────

class _Tareas:
    def __init__(self):
        self.tareas = []

    def add_task(self, fn, *a, **k):
        self.tareas.append((fn, a, k))


def _llamar(data, uid="u1"):
    from routers import chat
    tareas = _Tareas()
    r = asyncio.run(chat.api_chat_voz(tareas, data, uid, None))
    return r, tareas


@pytest.fixture
def voz_ok(monkeypatch):
    monkeypatch.delenv("MEALFIT_COACH_VOZ_NUBE", raising=False)
    monkeypatch.setattr(coach_voz, "hay_presupuesto", lambda: True)
    audio = _wav(1.5)
    monkeypatch.setattr(coach_voz, "sintetizar",
                        lambda texto, locale: coach_voz.VozSintetizada(audio, 1.5, 8, 38, 1700))
    return audio


def test_devuelve_el_wav_y_registra_el_uso_despues(voz_ok):
    r, tareas = _llamar({"texto": "Bien, aquí atento.", "locale": "es-DO"})
    assert r.status_code == 200 and r.media_type == "audio/wav" and r.body == voz_ok
    assert r.headers["cache-control"] == "no-store"
    assert len(tareas.tareas) == 1 and tareas.tareas[0][0] is coach_voz.registrar_uso


def test_apagada_por_knob_es_204(voz_ok, monkeypatch):
    monkeypatch.setenv("MEALFIT_COACH_VOZ_NUBE", "0")
    r, tareas = _llamar({"texto": "Hola"})
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "apagada" and tareas.tareas == []


def test_sin_presupuesto_es_204(voz_ok, monkeypatch):
    monkeypatch.setattr(coach_voz, "hay_presupuesto", lambda: False)
    r, _ = _llamar({"texto": "Hola"})
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "presupuesto"


def test_google_caido_es_204_y_no_registra_gasto(voz_ok, monkeypatch):
    def _boom(texto, locale):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(coach_voz, "sintetizar", _boom)
    r, tareas = _llamar({"texto": "Hola"})
    assert r.status_code == 204 and r.headers["x-voz-motivo"] == "error" and tareas.tareas == []


def test_sin_texto_es_400(voz_ok):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        _llamar({"texto": "  "})
    assert e.value.status_code == 400


def test_el_endpoint_esta_protegido_y_fuera_del_paywall():
    import inspect
    from routers import chat
    src = inspect.getsource(chat.api_chat_voz)
    assert "Depends(get_verified_user_id)" in src and "Depends(_VOZ_LIMITER)" in src
    assert "verify_api_quota" not in src and "verify_coach_quota" not in src
    assert "asyncio.to_thread(sintetizar" in src, "la síntesis (~2 s) no puede bloquear el único worker"
