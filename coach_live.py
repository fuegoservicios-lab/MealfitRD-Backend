# backend/coach_live.py
"""[P1-PLAN-LOTE-905 · 2026-09-29] Modo voz con OpenAI GPT-Live-1: su voz y sus oídos, NUESTRO coach como cerebro.

El dueño, tras los lotes 900-904: «es muy tonto, lento y no entiende lo que le digo», y tras la investigación
«Voz del coach alternativas y costos»: «¿y si lo pruebo yo y te voy diciendo?». Con un tope DURO de US$0,60.

Cómo encaja (developers.openai.com, guías `live`, `live-delegation`, `voice-server-controls`, 29-sep-2026):
  · El teléfono abre la sesión por WebRTC contra OpenAI: GPT-Live-1 escucha y habla a la vez (full-duplex), se deja
    interrumpir y responde en ~1 s. El SDP pasa por NUESTRO servidor (la clave de OpenAI nunca sale de aquí).
  · Delegación al CLIENTE (`delegation.type = "client"`): cuando la persona pide algo, GPT-Live-1 emite
    `session.delegation.created` SIN el texto de la tarea; lo que dijo llega aparte en `session.input_transcript.delta`.
    Este servidor escucha la sesión por el canal lateral (`wss://api.openai.com/v1/live/sessions/{id}/attach`), junta
    lo que dijo desde la última delegación y corre un turno COMPLETO del coach de siempre (`/api/chat/stream`: guarda
    los mensajes, las herramientas con el usuario verificado, la memoria, el cobro). La respuesta vuelve con
    `session.commentary.append` y GPT-Live-1 la dice con sus palabras.
  · Tope: US$0,05/min de voz (facturado por segundo) + el coach de siempre. `session.usage.updated` da los segundos;
    al llegar al tope (o al máximo por sesión) este servidor cierra la sesión. Fila propia en `llm_usage_events`
    (node `coach_live_voice`), NUNCA en `api_usage`.

Solo para las cuentas de `MEALFIT_COACH_LIVE_USUARIOS` (vacío = apagado para todos).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from knobs import _env_float, _env_int, _env_str

logger = logging.getLogger(__name__)

MODELO = "gpt-live-1"
NODO_USO = "coach_live_voice"
USD_POR_MINUTO = 0.05          # developers.openai.com/api/docs/models/gpt-live-1 (29-sep-2026)
_URL_SESIONES = "https://api.openai.com/v1/live/sessions"
_URL_ATTACH = "wss://api.openai.com/v1/live/sessions/{id}/attach"


# ── knobs ──────────────────────────────────────────────────────────────────────────────────────────────────────

def usuarios_permitidos() -> set:
    crudo = _env_str("MEALFIT_COACH_LIVE_USUARIOS", "")
    return {u.strip() for u in crudo.split(",") if u.strip()}


def disponible_para(user_id: Optional[str]) -> bool:
    return bool(user_id) and str(user_id) in usuarios_permitidos()


def tope_usd() -> float:
    """Gasto TOTAL de la prueba (voz de GPT-Live-1, todas las sesiones, últimos 30 días)."""
    return _env_float("MEALFIT_COACH_LIVE_TOPE_USD", 0.60, validator=lambda v: 0.0 <= v <= 50.0)


def max_segundos_por_sesion() -> int:
    return _env_int("MEALFIT_COACH_LIVE_MAX_SEGUNDOS", 300, validator=lambda v: 30 <= v <= 3600)


def voz() -> str:
    return _env_str("MEALFIT_COACH_LIVE_VOZ", "marin")


# ── gasto ──────────────────────────────────────────────────────────────────────────────────────────────────────

def gastado_usd() -> float:
    from db import execute_sql_query
    fila = execute_sql_query(
        "SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros FROM llm_usage_events "
        "WHERE node = %s AND created_at >= now() - interval '30 days'",
        (NODO_USO,),
        fetch_one=True,
    )
    return int((fila or {}).get("micros") or 0) / 1_000_000


def costo_usd(segundos: float) -> float:
    return max(0.0, float(segundos or 0)) * USD_POR_MINUTO / 60.0


def registrar_uso(user_id: Optional[str], live_id: str, segundos: float, motivo: str) -> None:
    """Fila propia: el precio de GPT-Live-1 es por MINUTO, no por token (la tabla de `compute_llm_cost_micros`)."""
    from db import execute_sql_write
    micros = int(round(costo_usd(segundos) * 1_000_000))
    try:
        execute_sql_write(
            "INSERT INTO llm_usage_events (user_id, model, node, input_tokens, output_tokens, cost_usd_micros, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)",
            (user_id, MODELO, NODO_USO, 0, 0, micros,
             json.dumps({"live_id": live_id, "segundos": round(float(segundos or 0), 1), "motivo": motivo})),
        )
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-905] no se pudo registrar el gasto de la sesión {live_id}: {e}")


# ── instrucciones de la voz ────────────────────────────────────────────────────────────────────────────────────

def instrucciones(locale: str = "es-DO") -> str:
    idioma = {"en-US": "inglés", "pt-BR": "portugués de Brasil", "fr-FR": "francés", "it-IT": "italiano"}.get(
        locale, "español latinoamericano (el usuario es dominicano: entiende su forma de hablar y sus comidas)")
    return f"""Eres la VOZ de Bioboros, un coach de nutrición. Habla en {idioma}, cálido, natural y breve: una o dos frases.

Delegation policy (tu cerebro es el backend: tiene el diario, el plan, la Nevera, el agua y los ajustes de la app):
- DELEGA siempre que el usuario: cuente algo que comió o bebió; pregunte por su día, calorías, macros, agua, su plan,
  su Nevera o una receta; pida anotar, corregir o borrar algo; pida cambiar algo de la app o ir a una pantalla; o haga
  cualquier pregunta de nutrición o salud.
- Mientras esperas al backend, di algo muy corto y natural («déjame ver», «un segundo») y NO inventes el resultado.
- Cuando llegue el resultado, dilo con tus palabras SIN cambiar cifras, cantidades ni nombres de alimentos.
- Si al backend le falta un dato (por ejemplo cuántas lonjas de pan), pregúntaselo al usuario tal cual.
- Contesta tú mismo SOLO: saludos, «gracias», pedir que repita algo que no entendiste, o despedirte.
- Nunca des diagnósticos médicos ni dosis de medicamentos."""


# ── sesiones vivas ─────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class SesionLive:
    live_id: str
    user_id: str
    chat_session_id: str
    locale: str = "es-DO"
    local_date: Optional[str] = None
    tz_offset: Optional[int] = None
    gastado_antes_usd: float = 0.0
    creada: float = field(default_factory=time.time)
    segundos: float = 0.0
    cerrada: bool = False
    novedades: list = field(default_factory=list)   # [{n, ajustes_de_app, diario}] para el teléfono
    _oido: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)


SESIONES: dict = {}
_SESIONES_LOCK = threading.Lock()


def sesion_de(live_id: str, user_id: str) -> Optional[SesionLive]:
    with _SESIONES_LOCK:
        s = SESIONES.get(live_id)
    return s if s and s.user_id == str(user_id) else None


def _limpiar_viejas() -> None:
    limite = time.time() - 3600
    with _SESIONES_LOCK:
        for k in [k for k, s in SESIONES.items() if s.cerrada and s.creada < limite]:
            SESIONES.pop(k, None)


# ── crear la sesión (SDP del teléfono → OpenAI) ───────────────────────────────────────────────────────────────

class LiveNoDisponible(Exception):
    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo


def crear_sesion(user_id: str, sdp: str, chat_session_id: str, locale: str = "es-DO",
                 local_date: Optional[str] = None, tz_offset: Optional[int] = None) -> tuple:
    """(live_id, sdp de respuesta). Levanta `LiveNoDisponible` si no toca o no queda presupuesto."""
    if not disponible_para(user_id):
        raise LiveNoDisponible("no_habilitado")
    gastado = gastado_usd()
    if gastado >= tope_usd():
        raise LiveNoDisponible("presupuesto")
    clave = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not clave:
        raise LiveNoDisponible("sin_clave")
    import httpx
    cuerpo = {
        "session": {
            "model": MODELO,
            "instructions": instrucciones(locale),
            "audio": {"output": {"voice": voz()}},
            "delegation": {"type": "client"},
        },
        "transport": {"type": "webrtc", "sdp": sdp},
    }
    r = httpx.post(_URL_SESIONES, headers={"Authorization": f"Bearer {clave}", "Content-Type": "application/json"},
                   json=cuerpo, timeout=20.0)
    if r.status_code >= 400:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-905] OpenAI rechazó la sesión Live: HTTP {r.status_code} {r.text[:300]}")
        raise LiveNoDisponible(f"openai_{r.status_code}")
    datos = r.json()
    live_id = (datos.get("session") or {}).get("id")
    respuesta_sdp = (datos.get("transport") or {}).get("sdp")
    if not live_id or not respuesta_sdp:
        raise LiveNoDisponible("respuesta_incompleta")
    s = SesionLive(live_id=live_id, user_id=str(user_id), chat_session_id=chat_session_id, locale=locale,
                   local_date=local_date, tz_offset=tz_offset, gastado_antes_usd=gastado)
    _limpiar_viejas()
    with _SESIONES_LOCK:
        SESIONES[live_id] = s
    threading.Thread(target=_canal_lateral, args=(s, clave), name=f"coach-live-{live_id[:8]}", daemon=True).start()
    logger.info(f"🎙️ [P1-PLAN-LOTE-905] sesión Live {live_id} abierta (user={str(user_id)[:8]}, gastado {gastado:.3f} USD)")
    return live_id, respuesta_sdp


# ── el canal lateral: transcripciones, delegaciones, gasto ────────────────────────────────────────────────────

def _texto_del_evento(ev: dict) -> str:
    for k in ("delta", "text", "transcript"):
        v = ev.get(k)
        if isinstance(v, str):
            return v
    return ""


def _canal_lateral(s: SesionLive, clave: str) -> None:
    from websockets.sync.client import connect
    tope_sesion = max_segundos_por_sesion()
    tope_total = tope_usd()
    motivo = "desconocido"
    try:
        with connect(_URL_ATTACH.format(id=s.live_id), additional_headers={"Authorization": f"Bearer {clave}"},
                     open_timeout=15, close_timeout=5, max_size=None) as ws:
            vistos = 0
            while True:
                try:
                    crudo = ws.recv(timeout=1.0)
                except TimeoutError:
                    crudo = None
                # Vigía por reloj: aunque OpenAI no mande `usage`, la sesión no pasa del máximo.
                if not s.cerrada and time.time() - s.creada > tope_sesion + 20:
                    _cerrar(ws, s, "tope_de_tiempo")
                if crudo is None:
                    continue
                if isinstance(crudo, bytes):
                    continue
                try:
                    ev = json.loads(crudo)
                except ValueError:
                    continue
                tipo = ev.get("type", "")
                if vistos < 40 and "audio" not in tipo:
                    vistos += 1
                    logger.info(f"🎙️ [P1-PLAN-LOTE-905] {s.live_id[:12]} ← {tipo} {str(ev)[:220]}")
                if tipo == "session.input_transcript.delta":
                    with s._lock:
                        s._oido.append(_texto_del_evento(ev))
                elif tipo == "session.delegation.created":
                    deleg = ev.get("delegation") or {}
                    if deleg.get("target", "client") == "client" and deleg.get("id"):
                        with s._lock:
                            dicho = "".join(s._oido).strip()
                            s._oido.clear()
                        threading.Thread(target=_delegar, args=(ws, s, deleg["id"], dicho), daemon=True).start()
                elif tipo == "session.usage.updated":
                    s.segundos = float(((ev.get("usage") or {}).get("seconds")) or s.segundos)
                    if not s.cerrada and (s.segundos >= tope_sesion
                                          or s.gastado_antes_usd + costo_usd(s.segundos) >= tope_total):
                        _cerrar(ws, s, "tope")
                elif tipo == "session.closed":
                    s.segundos = float(((ev.get("usage") or {}).get("seconds")) or s.segundos)
                    motivo = str(ev.get("reason") or "closed")
                    break
                elif tipo == "error":
                    logger.warning(f"⚠️ [P1-PLAN-LOTE-905] {s.live_id[:12]} error: {str(ev)[:300]}")
    except Exception as e:
        motivo = f"canal_{type(e).__name__}"
        logger.warning(f"⚠️ [P1-PLAN-LOTE-905] canal lateral de {s.live_id} cayó: {type(e).__name__}: {str(e)[:200]}")
    finally:
        s.cerrada = True
        if not s.segundos:
            s.segundos = min(time.time() - s.creada, tope_sesion + 20)   # sin `usage`: el reloj, por lo alto
        registrar_uso(s.user_id, s.live_id, s.segundos, motivo)
        logger.info(f"🎙️ [P1-PLAN-LOTE-905] sesión {s.live_id} cerrada ({motivo}, {s.segundos:.0f} s, "
                    f"{costo_usd(s.segundos):.3f} USD)")


def _cerrar(ws, s: SesionLive, motivo: str) -> None:
    s.cerrada = True
    logger.info(f"🎙️ [P1-PLAN-LOTE-905] cerrando {s.live_id} por {motivo}")
    try:
        ws.send(json.dumps({"type": "session.close", "event_id": f"cierre_{uuid.uuid4().hex[:8]}"}))
    except Exception:
        pass


def _delegar(ws, s: SesionLive, delegation_id: str, dicho: str) -> None:
    """Un turno del coach de siempre con lo que dijo; su respuesta vuelve para que GPT-Live-1 la diga."""
    t0 = time.time()
    if not dicho:
        respuesta = "No alcancé a entender lo que dijo. Pídele que lo repita."
        cambios = {}
    else:
        try:
            respuesta, cambios = correr_turno_del_coach(s, dicho)
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-905] el coach falló en {s.live_id}: {type(e).__name__}: {str(e)[:200]}")
            respuesta, cambios = "Hubo un problema de mi lado y no pude hacerlo. Pídele que lo intente otra vez.", {}
    from agent import strip_ui_action_tags_for_persist
    texto = strip_ui_action_tags_for_persist(respuesta or "").strip()[:1800]
    try:
        ws.send(json.dumps({"type": "session.commentary.append", "event_id": f"coach_{uuid.uuid4().hex[:10]}",
                            "delegation_id": delegation_id, "content": texto or "Listo."}))
    except Exception as e:
        logger.warning(f"⚠️ [P1-PLAN-LOTE-905] no se pudo devolver la respuesta a {s.live_id}: {e}")
    with s._lock:
        s.novedades.append({"n": len(s.novedades) + 1, "oido": dicho, "respuesta": texto, **cambios})
    logger.info(f"🎙️ [P1-PLAN-LOTE-905] {s.live_id[:12]} delegación en {time.time() - t0:.1f} s: "
                f"«{dicho[:80]}» → «{texto[:80]}»")


def correr_turno_del_coach(s: SesionLive, dicho: str) -> tuple:
    """El MISMO turno que `POST /api/chat/stream` en modo voz (guardar, herramientas, memoria, cobro). Devuelve
    (texto de la respuesta, cambios para el teléfono: ajustes_de_app y si tocó el diario/Nevera)."""
    import asyncio
    from fastapi import BackgroundTasks
    from routers.chat import api_chat_stream
    tareas = BackgroundTasks()
    datos = {
        "session_id": s.chat_session_id,
        "prompt": dicho,
        "user_id": s.user_id,
        "is_call_mode": True,
        "local_date": s.local_date,
        "tz_offset": s.tz_offset,
    }
    resp = api_chat_stream(tareas, datos, s.user_id)
    final = {}

    async def _leer():
        async for trozo in resp.body_iterator:
            linea = trozo.decode() if isinstance(trozo, bytes) else str(trozo)
            for parte in linea.split("\n\n"):
                parte = parte.strip()
                if parte.startswith("data:"):
                    try:
                        ev = json.loads(parte[5:].strip())
                    except ValueError:
                        continue
                    if ev.get("type") == "done":
                        final.update(ev)
                    elif ev.get("type") == "error":
                        final.setdefault("error", ev)
        await tareas()

    asyncio.run(_leer())
    cambios = {}
    if final.get("ajustes_de_app"):
        cambios["ajustes_de_app"] = final["ajustes_de_app"]
    if final.get("pantry_modified_at"):
        cambios["nevera"] = True
    return str(final.get("response") or ""), cambios
