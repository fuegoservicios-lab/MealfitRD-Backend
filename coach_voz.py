# backend/coach_voz.py
"""[P1-PLAN-LOTE-684/685 · 2026-09-28] El modo voz del coach: que conteste rápido y con voz de verdad.

El dueño, probando el modo voz (lote 682) en su iPhone: «es muy lenta la respuesta del modo voz y la voz no es
nada realista». Medido en su turno («Cómo estás», 28-sep 18:07 UTC, journal + `llm_usage_events`):

  1,64 s  clasificar la respuesta a un nudge pendiente (LLM, solo analítica) → ahora en segundo plano (db_chat)
  1,75 s  router de RAG (LLM) para acabar en SKIP → la charla corta se decide aquí, sin LLM
  1,00 s  (en paralelo) clasificador de sentimiento → en modo voz no corre: el prompt de voz fija el tono
  1,72 s  una respuesta entera tirada por un falso positivo de P1-DIARY-CLAIM-VERIFY (agent.py)
  ~0,6 s  la respuesta buena, hasta su primera frase

y la voz era la del lector de accesibilidad del teléfono. Aquí vive:
  · `es_charla_corta`: la parte del router de RAG que no necesita un LLM;
  · `sintetizar`: la voz del coach con Gemini 3.8 Flash-Lite TTS (lote 685), con tope diario de gasto.

Gemini solo pone VOZ a un texto que el coach ya escribió: no razona, no aconseja, no ve el historial ni el
perfil. Por eso es la segunda costura permitida del blanket anti-Gemini (`test_p0_llm_provider_migration`),
marcada `[P1-PLAN-LOTE-685-VOZ]` en cada línea que nombra el modelo.

Medido contra la API real el 28-sep (VPS): sin streaming, ~2 s por petición aunque la frase sea cortísima, y
crece con el audio (3,2-3,5 s para 8-9 s de audio). Por eso el frontend manda la PRIMERA frase hasta la primera
coma y pide las siguientes en paralelo mientras suena la anterior. Una instrucción de estilo delante del texto
(«Lee con voz cálida…») DUPLICA el audio: la lee en voz alta. El acento va por `languageCode`, nunca por texto.
"""

from __future__ import annotations

import base64
import os
import re
import struct
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from knobs import _env_bool, _env_float, _env_int, _env_str

import logging

logger = logging.getLogger(__name__)


# ============================================================
# Charla corta: el router de RAG sin LLM
# ============================================================
# «Cómo estás», «qué tal», «todo bien, coach»… La lista de palabras sueltas del router (`rag_query_router`)
# no las veía y pagaban ~1,75 s de LLM para acabar en SKIP. Solo si TODO el mensaje es saludo/charla.

def _normalizar(texto: str) -> str:
    t = unicodedata.normalize("NFKD", str(texto or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_SALUDO = r"(?:hola|hey|ey|epa|buenas|buenos dias|buenas tardes|buenas noches|saludos|klk|que lo que)"
_CHARLA = (
    r"(?:como estas|como esta|como vas|como va|como andas|como te va|como sigues|como amaneciste|"
    r"que tal|que tal todo|que tal vas|que mas|todo bien|que hay|que haces|bien y tu|y tu|muy bien|"
    r"todo tranquilo|aqui|aqui estoy|aqui tranquilo)"
)
_VOCATIVO = r"(?:coach|bioboros|amigo|amiga|hermano|mano|brother)"
_RE_CHARLA_CORTA = re.compile(
    rf"^(?P<saludo>(?:{_SALUDO} ?)+)?(?P<charla>{_CHARLA})?(?: {_VOCATIVO})?$"
)


def es_charla_corta(prompt: str) -> bool:
    """True si el mensaje ENTERO es un saludo o charla corta (≤ 6 palabras): no hay nada que buscar en la memoria."""
    t = _normalizar(prompt)
    if not t or len(t.split()) > 6:
        return False
    m = _RE_CHARLA_CORTA.fullmatch(t)
    return bool(m and (m.group("saludo") or m.group("charla")))


# ============================================================
# La voz del coach: Gemini 3.8 Flash-Lite TTS
# ============================================================

_MODELO_TTS_DEFAULT = "gemini-3.8-flash-lite-tts"  # [P1-PLAN-LOTE-685-VOZ]
_URL_TTS = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
_NODO_USO = "coach_voice_tts"

# El acento: la voz del coach para quien lee en es-DO es la latinoamericana (no hay voz dominicana).
_IDIOMA_DE_VOZ = {
    "es-DO": "es-US",
    "es": "es-US",
    "en-US": "en-US",
    "pt-BR": "pt-BR",
    "fr-FR": "fr-FR",
    "it-IT": "it-IT",
}


def voz_en_la_nube_activa() -> bool:
    """Kill switch: `MEALFIT_COACH_VOZ_NUBE=0` devuelve a todos a la voz del teléfono, sin redeploy."""
    return _env_bool("MEALFIT_COACH_VOZ_NUBE", True)


def _modelo() -> str:
    return _env_str("MEALFIT_COACH_VOZ_MODELO", _MODELO_TTS_DEFAULT)


def _nombre_de_voz() -> str:
    # Sulafat («cálida» en el catálogo de Google). Probadas el 28-sep: Sulafat, Puck, Achird.
    return _env_str("MEALFIT_COACH_VOZ_NOMBRE", "Sulafat")


def _timeout_s() -> float:
    return _env_float("MEALFIT_COACH_VOZ_TIMEOUT_S", 8.0, validator=lambda v: 0.0 < v <= 30.0)


def _max_caracteres() -> int:
    return _env_int("MEALFIT_COACH_VOZ_MAX_CARACTERES", 420, validator=lambda v: 20 <= v <= 2000)


def presupuesto_diario_usd() -> float:
    """Tope de gasto de la voz por día (UTC), para TODOS los usuarios juntos. Al llegar, la app vuelve a la voz
    del teléfono hasta el día siguiente. Con ~US$0,002 por respuesta, US$0,50 son ~250 respuestas al día."""
    return _env_float("MEALFIT_COACH_VOZ_PRESUPUESTO_DIARIO_USD", 0.50, validator=lambda v: 0.0 <= v <= 100.0)


def idioma_de_voz(locale: str) -> str:
    loc = str(locale or "").strip()
    return _IDIOMA_DE_VOZ.get(loc) or _IDIOMA_DE_VOZ.get(loc.split("-")[0]) or "es-US"


def _clave() -> str:
    """La key de Gemini. En el VPS vive en `VISION_API_KEY` (el escáner ya va a Google con ella: misma cuenta, mismo
    saldo); si el escáner no apunta a Google, la de Gemini del entorno (`llm_provider._google_api_key`, fail-loud).
    NUNCA argumento del callsite."""
    base = os.environ.get("MEALFIT_VISION_BASE_URL", "")
    k = (os.environ.get("VISION_API_KEY") or "").strip()
    if k and "googleapis.com" in base:
        return k
    from llm_provider import _google_api_key
    return _google_api_key()


# ---------- gasto del día ----------
# Se lee de `llm_usage_events` (lo que ya se registró, también desde otros workers) cada 60 s y, entre lecturas,
# se suma en memoria lo que este proceso va gastando: un pico entre dos lecturas no se cuela del tope.
_GASTO = {"dia": None, "micros_db": 0, "micros_locales": 0, "leido_en": 0.0}
_GASTO_LOCK = threading.Lock()
_GASTO_RELECTURA_S = 60.0


def _hoy_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _leer_gasto_de_hoy_micros() -> int:
    from db import execute_sql_query
    fila = execute_sql_query(
        "SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros FROM llm_usage_events "
        "WHERE node = %s AND created_at >= date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'",
        (_NODO_USO,),
        fetch_one=True,
    )
    return int((fila or {}).get("micros") or 0)


def gasto_de_hoy_usd() -> float:
    ahora = time.monotonic()
    with _GASTO_LOCK:
        dia = _hoy_utc()
        if _GASTO["dia"] != dia:
            _GASTO.update(dia=dia, micros_db=0, micros_locales=0, leido_en=0.0)
        if ahora - _GASTO["leido_en"] >= _GASTO_RELECTURA_S:
            try:
                _GASTO["micros_db"] = _leer_gasto_de_hoy_micros()
                _GASTO["micros_locales"] = 0   # lo local ya está en la DB (o se perdió con su fila)
            except Exception as e:
                logger.warning(f"⚠️ [P1-PLAN-LOTE-685] no se pudo leer el gasto de la voz: {e}")
            _GASTO["leido_en"] = ahora
        return (_GASTO["micros_db"] + _GASTO["micros_locales"]) / 1_000_000


def _sumar_gasto_local(micros: int) -> None:
    with _GASTO_LOCK:
        if _GASTO["dia"] != _hoy_utc():
            _GASTO.update(dia=_hoy_utc(), micros_db=0, micros_locales=0, leido_en=0.0)
        _GASTO["micros_locales"] += max(0, int(micros or 0))


def hay_presupuesto() -> bool:
    return gasto_de_hoy_usd() < presupuesto_diario_usd()


# ---------- síntesis ----------

@dataclass
class VozSintetizada:
    wav: bytes
    segundos: float
    tokens_texto: int
    tokens_audio: int
    ms: int


_RE_ESPACIOS = re.compile(r"\s+")


def _texto_para_decir(texto: str) -> str:
    t = _RE_ESPACIOS.sub(" ", str(texto or "")).strip()
    return t[:_max_caracteres()]


def _envolver_wav(pcm: bytes, frecuencia: int = 24000, canales: int = 1, bits: int = 16) -> bytes:
    """PCM crudo (audio/L16) → WAV. Hoy el modelo ya devuelve WAV; esto cubre el formato de los TTS anteriores."""
    bloque = canales * bits // 8
    cabecera = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, canales, frecuencia,
        frecuencia * bloque, bloque, bits, b"data", len(pcm),
    )
    return cabecera + pcm


def _frecuencia_de(mime: str) -> int:
    m = re.search(r"rate=(\d+)", str(mime or ""))
    return int(m.group(1)) if m else 24000


def _segundos_de_wav(wav: bytes) -> float:
    try:
        canales, frecuencia = struct.unpack("<HI", wav[22:28])
        bits = struct.unpack("<H", wav[34:36])[0]
        datos = len(wav) - 44
        return round(datos / (frecuencia * canales * bits / 8), 2)
    except Exception:
        return 0.0


_CLIENTE = None
_CLIENTE_LOCK = threading.Lock()


def _cliente():
    """Un solo cliente HTTP por proceso: la conexión con Google queda abierta entre frases (sin TLS nuevo)."""
    global _CLIENTE
    with _CLIENTE_LOCK:
        if _CLIENTE is None:
            import httpx
            _CLIENTE = httpx.Client(timeout=_timeout_s())
        return _CLIENTE


def sintetizar(texto: str, locale: str = "es-DO") -> Optional[VozSintetizada]:
    """El texto → WAV con la voz del coach. None si no hay nada que decir. Levanta si Google falla: quien llama
    decide (el endpoint responde 204 y el teléfono pone su voz)."""
    frase = _texto_para_decir(texto)
    if not frase:
        return None
    cuerpo = {
        "contents": [{"parts": [{"text": frase}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "languageCode": idioma_de_voz(locale),
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": _nombre_de_voz()}},
            },
        },
    }
    t0 = time.monotonic()
    r = _cliente().post(
        _URL_TTS.format(modelo=_modelo()),
        headers={"x-goog-api-key": _clave()},
        json=cuerpo,
        timeout=_timeout_s(),
    )
    ms = int((time.monotonic() - t0) * 1000)
    r.raise_for_status()
    datos = r.json()
    parte = datos["candidates"][0]["content"]["parts"][0]["inlineData"]
    audio = base64.b64decode(parte["data"])
    wav = audio if audio[:4] == b"RIFF" else _envolver_wav(audio, _frecuencia_de(parte.get("mimeType")))
    uso = datos.get("usageMetadata") or {}
    return VozSintetizada(
        wav=wav,
        segundos=_segundos_de_wav(wav),
        tokens_texto=int(uso.get("promptTokenCount") or 0),
        tokens_audio=int(uso.get("candidatesTokenCount") or 0),
        ms=ms,
    )


def registrar_uso(voz: VozSintetizada, user_id: Optional[str], extra: Optional[dict] = None) -> None:
    """El gasto a `llm_usage_events` (NUNCA a `api_usage`: la voz no quema créditos del plan) y al tope del día."""
    from db import compute_llm_cost_micros, log_llm_usage_event
    modelo = _modelo()
    micros = compute_llm_cost_micros(modelo, voz.tokens_texto, voz.tokens_audio) or 0
    _sumar_gasto_local(micros)
    log_llm_usage_event(
        user_id=user_id,
        model=modelo,
        node=_NODO_USO,
        input_tokens=voz.tokens_texto,
        output_tokens=voz.tokens_audio,
        metadata={"duration_s": round(voz.ms / 1000, 3), "audio_s": voz.segundos, **(extra or {})},
    )


# ============================================================
# [P1-PLAN-LOTE-901 · 2026-09-29] La voz en streaming
# ============================================================
# El dueño, tras el 688: «siento que todavía es muy lenta la latencia de respuesta del agente de voz». Medido en su
# turno («Activa la hidratación», 29-sep 17:39 UTC): el texto completo a los 2,9 s y la primera frase SONANDO a los
# 5,2 s — 2,1 s de ellos esperando el WAV entero de `generateContent`. `streamGenerateContent?alt=sse` (medido el
# mismo día desde el VPS, la misma frase): el primer trozo de audio a los 0,64 s en vez de 2,94 s, y el resto más
# rápido que el tiempo real. Los trozos son PCM crudo (`audio/l16; rate=24000; channels=1`), no WAV.
#
# `abrir_flujo` espera el PRIMER trozo antes de devolver: si Google falla, falla aquí, y el endpoint aún puede
# contestar 204 (el teléfono pone su voz) en vez de una respuesta 200 cortada. El gasto se registra al terminar
# el flujo (el `usageMetadata` de Google es acumulado: vale el último).

_URL_TTS_FLUJO = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:streamGenerateContent?alt=sse"


def voz_en_flujo_activa() -> bool:
    """Kill switch del streaming: `MEALFIT_COACH_VOZ_FLUJO=0` devuelve la voz al WAV entero (`/api/chat/voz`)."""
    return _env_bool("MEALFIT_COACH_VOZ_FLUJO", True)


def _eventos_sse(respuesta):
    import json as _json
    for linea in respuesta.iter_lines():
        if not linea or not linea.startswith("data:"):
            continue
        try:
            yield _json.loads(linea[5:])
        except ValueError:
            continue


def _trozo_de_evento(evento: dict) -> tuple:
    """(pcm, mime, uso) de un evento del stream. Sin audio → b""."""
    uso = evento.get("usageMetadata") or None
    try:
        parte = (evento["candidates"][0].get("content") or {}).get("parts") or []
        datos = (parte[0].get("inlineData") or {}) if parte else {}
    except (KeyError, IndexError, AttributeError, TypeError):
        datos = {}
    crudo = base64.b64decode(datos["data"]) if datos.get("data") else b""
    return crudo, datos.get("mimeType"), uso


class FlujoDeVoz:
    """El audio de una frase MIENTRAS Google lo produce: PCM 16 bits little-endian, mono, a `frecuencia` Hz.
    Iterarlo entrega bytes; al acabar (o al cortarse) cierra la conexión y registra el gasto."""

    def __init__(self, respuesta, eventos, primero: bytes, frecuencia: int, uso: Optional[dict], t0: float,
                 primer_audio_ms: int, user_id: Optional[str]):
        self._respuesta = respuesta
        self._eventos = eventos
        self._primero = primero
        self.frecuencia = frecuencia
        self._uso = uso or {}
        self._t0 = t0
        self.primer_audio_ms = primer_audio_ms
        self._user_id = user_id
        self._bytes = 0
        self._cerrado = False

    def __iter__(self):
        try:
            self._bytes += len(self._primero)
            yield self._primero
            for evento in self._eventos:
                pcm, _mime, uso = _trozo_de_evento(evento)
                if uso:
                    self._uso = uso
                if pcm:
                    self._bytes += len(pcm)
                    yield pcm
        finally:
            self.cerrar()

    def cerrar(self) -> None:
        if self._cerrado:
            return
        self._cerrado = True
        try:
            self._respuesta.close()
        except Exception:
            pass
        voz = VozSintetizada(
            wav=b"",
            segundos=round(self._bytes / (2 * max(1, self.frecuencia)), 2),
            tokens_texto=int(self._uso.get("promptTokenCount") or 0),
            tokens_audio=int(self._uso.get("candidatesTokenCount") or 0),
            ms=int((time.monotonic() - self._t0) * 1000),
        )
        try:
            registrar_uso(voz, self._user_id, extra={"flujo": True, "primer_audio_s": round(self.primer_audio_ms / 1000, 3)})
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-901] no se pudo registrar el gasto de la voz en streaming: {e}")


def abrir_flujo(texto: str, locale: str = "es-DO", user_id: Optional[str] = None) -> Optional[FlujoDeVoz]:
    """Abre el streaming de la voz y espera su PRIMER trozo de audio. None si no hay nada que decir. Levanta si
    Google falla antes de dar audio: quien llama responde 204 y el teléfono pone su voz."""
    frase = _texto_para_decir(texto)
    if not frase:
        return None
    cuerpo = {
        "contents": [{"parts": [{"text": frase}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "languageCode": idioma_de_voz(locale),
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": _nombre_de_voz()}},
            },
        },
    }
    t0 = time.monotonic()
    cliente = _cliente()
    peticion = cliente.build_request(
        "POST",
        _URL_TTS_FLUJO.format(modelo=_modelo()),
        headers={"x-goog-api-key": _clave()},
        json=cuerpo,
        timeout=_timeout_s(),
    )
    respuesta = cliente.send(peticion, stream=True)
    try:
        respuesta.raise_for_status()
        eventos = _eventos_sse(respuesta)
        uso = None
        for evento in eventos:
            pcm, mime, uso_ev = _trozo_de_evento(evento)
            uso = uso_ev or uso
            if pcm[:4] == b"RIFF":   # por si algún día llega con cabecera: se toca igual, sin ella
                pcm = pcm[44:]
            if pcm:
                return FlujoDeVoz(respuesta, eventos, pcm, _frecuencia_de(mime), uso, t0,
                                  int((time.monotonic() - t0) * 1000), user_id)
        raise RuntimeError("el streaming de la voz terminó sin audio")
    except BaseException:
        respuesta.close()
        raise
