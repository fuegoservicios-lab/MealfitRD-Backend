# -*- coding: utf-8 -*-
"""[P1-DECISION-TRIAGE · 2026-11-11] Triaje con Gemini Decision 1 (`POST /v1beta/decisions`): probabilidades, no texto.

Tickets BB-074..BB-079. Doc: `docs/decision_triage.md`. tooltip-anchor: P1-DECISION-TRIAGE

Lo que esta capa PUEDE hacer (y nada más):
  · Prefiltro de foto (`/api/diary/upload`): si es casi seguro que la foto NO es comida, no se gasta la visión.
  · Freno de bebida ambigua (chat): frena UNA vez `log_consumed_meal` para que el coach pregunte el azúcar.
  · Telemetría (`pipeline_metrics`): intención y fusión SOLO se registran; jamás deciden.

Lo que NUNCA hace: escribir en la base por una probabilidad, decidir gramos, fusionar, editar, ni tocar el día
biológico (4:00) o el TTL de 30 min — esas reglas son reloj + estado y viven en código determinista.

Fail-open en todo: sin key, knob en `off`, timeout, HTTP != 200, JSON raro o usuario de la UE ⇒ `None` ⇒ la conducta
de siempre. Ninguna función pública lanza excepción.

Knobs: `MEALFIT_DECISION_TRIAGE_MODE` (off | shadow | guard; default off), `MEALFIT_DECISION_TRIAGE_MODEL`,
`MEALFIT_DECISION_TRIAGE_TIMEOUT_S` (0.4), `MEALFIT_DECISION_TRIAGE_CHAT_WAIT_S` (0.3),
`MEALFIT_DECISION_BEVERAGE_BLOCK` (0.80), `MEALFIT_DECISION_MEDIA_REJECT` (0.15).
"""
from __future__ import annotations

import base64
import contextvars
import json
import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

DECISIONS_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
_MODELO_POR_DEFECTO = "gemini-decision-1"  # [P1-DECISION-TRIAGE]
MODOS = {"off", "shadow", "guard"}

# El tipo sí/no del API se llama literalmente «noul» (doc oficial del 11-nov-2026). Constante para cambiarlo en un sitio.
TIPO_SI_NO = "noul"

Q_MEDIA = "is_valid_food_media"
Q_BEBIDA = "beverage_ambiguity"
Q_INTENCION = "routing_intent"
Q_FUSION = "ttl_fusion_valid"
INTENCIONES = ("append_active_meal", "correct_last_meal", "general_coaching", "new_meal_log", "off_topic")

FRENO_MARCA = "[P1-DECISION-TRIAGE-FRENO-BEBIDA]"
_TOPE_CONTEXTO_CHARS = 6000  # ~1.500 tokens
_TOOL_FRENABLE = "log_consumed_meal"


# ───────────────────────────── knobs ─────────────────────────────
def modo() -> str:
    try:
        from knobs import _env_str
        return _env_str("MEALFIT_DECISION_TRIAGE_MODE", "off", MODOS)
    except Exception:                                                          # noqa: BLE001
        return "off"


def _modelo() -> str:
    try:
        from knobs import _env_str
        return _env_str("MEALFIT_DECISION_TRIAGE_MODEL", _MODELO_POR_DEFECTO) or _MODELO_POR_DEFECTO
    except Exception:                                                          # noqa: BLE001
        return _MODELO_POR_DEFECTO


def _float_knob(nombre: str, defecto: float, lo: float, hi: float) -> float:
    try:
        from knobs import _env_float
        v = float(_env_float(nombre, defecto))
    except Exception:                                                          # noqa: BLE001
        v = defecto
    return max(lo, min(hi, v))


def timeout_s() -> float:
    return _float_knob("MEALFIT_DECISION_TRIAGE_TIMEOUT_S", 0.4, 0.05, 5.0)


def espera_chat_s() -> float:
    return _float_knob("MEALFIT_DECISION_TRIAGE_CHAT_WAIT_S", 0.3, 0.0, 5.0)


def umbral_bebida() -> float:
    return _float_knob("MEALFIT_DECISION_BEVERAGE_BLOCK", 0.80, 0.5, 0.99)


def umbral_media() -> float:
    return _float_knob("MEALFIT_DECISION_MEDIA_REJECT", 0.15, 0.01, 0.5)


# ───────────────────────────── contrato ─────────────────────────────
@dataclass
class TriageResult:
    answers: dict = field(default_factory=dict)
    latency_ms: int = 0
    input_tokens: int = 0

    def si(self, qid: str) -> Optional[float]:
        """Probabilidad de «sí» de una pregunta `noul` (None si falta)."""
        a = self.answers.get(qid)
        if not isinstance(a, dict):
            return None
        v = a.get(TIPO_SI_NO)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def eleccion(self, qid: str) -> Optional[str]:
        a = self.answers.get(qid)
        return a.get("choice") if isinstance(a, dict) and isinstance(a.get("choice"), str) else None

    def probabilidades(self, qid: str) -> dict:
        """Mapa opción→probabilidad. Por CLAVE: el API ordena las opciones alfabéticamente."""
        a = self.answers.get(qid)
        p = a.get("probabilities") if isinstance(a, dict) else None
        return dict(p) if isinstance(p, dict) else {}

    def resumen(self) -> dict:
        out = {}
        for qid in (Q_MEDIA, Q_BEBIDA, Q_FUSION):
            v = self.si(qid)
            if v is not None:
                out[qid] = round(v, 3)
        if self.eleccion(Q_INTENCION):
            out[Q_INTENCION] = self.eleccion(Q_INTENCION)
            out[Q_INTENCION + "_p"] = {k: round(float(v), 3) for k, v in self.probabilidades(Q_INTENCION).items()
                                       if isinstance(v, (int, float))}
        return out


def _preguntas(*, con_media: bool, con_texto: bool, con_comida_activa: bool) -> dict:
    q: dict = {}
    if con_media:
        q[Q_MEDIA] = {
            "type": TIPO_SI_NO,
            "instructions": "¿La imagen muestra comida, bebida, ingredientes o una nevera, con claridad suficiente "
                            "para estimar lo que hay?",
        }
    if con_texto:
        q[Q_BEBIDA] = {
            "type": TIPO_SI_NO,
            "instructions": "¿El usuario dice que tomó o va a registrar una bebida (jugo, refresco, café, batida, "
                            "alcohol, malta…) SIN decir si lleva azúcar añadida o si es regular, light o zero?",
            "criteria": {
                "true_description": "Hay una bebida y falta saber su azúcar o si es regular/zero.",
                "false_description": "No hay bebida, o ya está dicho el azúcar o el tipo (p. ej. «sin azúcar», "
                                     "«coca zero», «agua», «café negro»).",
            },
        }
        q[Q_INTENCION] = {
            "type": "choice",
            "instructions": "¿Qué quiere hacer el usuario con este mensaje?",
            "criteria": {
                "append_active_meal": "Añadir algo a la comida que acaba de registrar.",
                "correct_last_meal": "Corregir algo que ya registró.",
                "general_coaching": "Preguntar o conversar sobre nutrición o la app.",
                "new_meal_log": "Registrar una comida o bebida nueva.",
                "off_topic": "Algo ajeno a la nutrición.",
            },
        }
        if con_comida_activa:
            q[Q_FUSION] = {
                "type": TIPO_SI_NO,
                "instructions": "¿Lo que el usuario menciona ahora es un complemento de la comida activa (bebida, "
                                "postre o guarnición olvidada) y no una comida aparte?",
            }
    return q


def build_request(texto: Optional[str], *, comida_activa: Optional[str] = None,
                  media: Optional[dict] = None) -> Optional[dict]:
    """El cuerpo del POST. `media` = {"data": bytes, "mime_type": str}. Sin fechas, horas ni identificadores."""
    bloques: list = []
    t = (texto or "").strip()
    if t:
        bloques.append({"type": "text", "text": ("Mensaje del usuario: " + t)[:_TOPE_CONTEXTO_CHARS]})
    if t and comida_activa:
        bloques.append({"type": "text", "text": ("Comida activa: " + comida_activa)[:600]})
    if media and media.get("data"):
        bloques.append({"type": "image", "mime_type": media.get("mime_type") or "image/jpeg",
                        "data": base64.b64encode(media["data"]).decode("ascii"), "resolution": "low"})
    preguntas = _preguntas(con_media=bool(media and media.get("data")), con_texto=bool(t),
                           con_comida_activa=bool(t and comida_activa))
    if not bloques or not preguntas:
        return None
    return {"model": _modelo(), "input": bloques, "questions": preguntas}


def parse_response(data: Any, esperadas) -> Optional[dict]:
    """`answers` válidas, o None si falta alguna pregunta pedida o la forma no casa."""
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        return None
    answers = data["answers"]
    for qid in esperadas:
        a = answers.get(qid)
        if not isinstance(a, dict):
            return None
        tipo = a.get("type")
        if tipo == TIPO_SI_NO and not isinstance(a.get(TIPO_SI_NO), (int, float)):
            return None
        if tipo == "choice" and not isinstance(a.get("probabilities"), dict):
            return None
    return answers


# ───────────────────────────── cliente ─────────────────────────────
_CLIENTE = None


def _cliente():
    global _CLIENTE
    if _CLIENTE is None:
        import httpx
        _CLIENTE = httpx.Client()
    return _CLIENTE


def _excluido_por_region() -> bool:
    try:
        import region_ia
        return bool(region_ia.es_ue(region_ia.pais_actual()))
    except Exception:                                                          # noqa: BLE001
        return False


def _api_key() -> Optional[str]:
    try:
        from llm_provider import _google_api_key
        return _google_api_key()
    except Exception:                                                          # noqa: BLE001
        return None


def run_triage(texto: Optional[str] = None, *, comida_activa: Optional[str] = None, media: Optional[dict] = None,
               user_id: Optional[str] = None, timeout: Optional[float] = None) -> Optional[TriageResult]:
    """Una llamada al API. Síncrona (los nodos del chat lo son). None ante cualquier fallo."""
    try:
        if modo() == "off" or _excluido_por_region():
            return None
        if user_id and user_id != "guest":
            from consentimientos import permite_ia
            if not permite_ia(user_id, "triaje_de_decision"):   # [P1-PLAN-LOTE-843] además del 428 del endpoint
                return None
        body = build_request(texto, comida_activa=comida_activa, media=media)
        if body is None:
            return None
        key = _api_key()
        if not key:
            return None
        t0 = time.perf_counter()
        resp = _cliente().post(
            DECISIONS_URL, json=body, timeout=timeout if timeout is not None else timeout_s(),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
        latencia = int((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            logger.warning(f"⚠️ [P1-DECISION-TRIAGE] HTTP {resp.status_code}: {resp.text[:160]!r}")
            return None
        data = resp.json()
        answers = parse_response(data, body["questions"].keys())
        if answers is None:
            logger.warning("⚠️ [P1-DECISION-TRIAGE] respuesta sin las preguntas pedidas")
            return None
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        tokens = int(usage.get("total_input_tokens") or 0)
        try:
            from db import log_llm_usage_event
            log_llm_usage_event(user_id=user_id if user_id and user_id != "guest" else None, model=body["model"],
                                node="decision_triage", input_tokens=tokens, output_tokens=0,
                                metadata={"latency_ms": latencia})
        except Exception:                                                      # noqa: BLE001
            pass
        return TriageResult(answers=answers, latency_ms=latencia, input_tokens=tokens)
    except Exception as e:                                                     # noqa: BLE001
        logger.warning(f"⚠️ [P1-DECISION-TRIAGE] fail-open: {type(e).__name__}: {e}")
        return None


async def run_triage_async(*args, **kwargs) -> Optional[TriageResult]:
    import asyncio
    try:
        return await asyncio.to_thread(run_triage, *args, **kwargs)
    except Exception:                                                          # noqa: BLE001
        return None


def registrar_metrica(user_id: Optional[str], node: str, duracion_ms: float, metadata: dict) -> None:
    """Fila best-effort en `pipeline_metrics` (mismo patrón que `telemetria_escaner._insertar`)."""
    try:
        from db import execute_sql_write
        uid = user_id if user_id and user_id != "guest" else None
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, tokens_estimated, "
            "confidence, metadata) VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)",
            (uid, None, node, int(duracion_ms or 0), json.dumps(metadata, ensure_ascii=False)),
        )
    except Exception as e:                                                     # noqa: BLE001
        logger.debug(f"[P1-DECISION-TRIAGE] métrica {node} no-op: {e!r}")


# ───────────────────────────── prefiltro de foto ─────────────────────────────
def payload_no_comida() -> dict:
    """Misma forma que el «no es comida» de `vision_agent`: el frontend ya lo sabe pintar."""
    return {
        "photo_kind": "otro", "is_food": False, "items": [],
        "description": "La imagen no parece de comida. Prueba con otra foto del plato con buena luz.",
        "meal_name": "", "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0, "dudas": [],
        "triage_rejected": True,
    }


def foto_rechazada(res: Optional[TriageResult]) -> bool:
    """Solo en guard, y solo si es casi seguro que no es comida."""
    if res is None or modo() != "guard":
        return False
    p = res.si(Q_MEDIA)
    return p is not None and p <= umbral_media()


# ───────────────────────────── chat: un triaje por turno ─────────────────────────────
@dataclass
class _Turno:
    turn_id: str
    future: Future
    user_id: Optional[str] = None


_TURNO: contextvars.ContextVar = contextvars.ContextVar("mealfit_decision_triage_turno", default=None)
_POOL: Optional[ThreadPoolExecutor] = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="decision-triage")
    return _POOL


def _comida_activa_de(user_id: Optional[str]) -> Optional[str]:
    """Nombre e ingredientes de la comida principal de <30 min (lectura; reusa la consulta del TTL)."""
    if not user_id or user_id == "guest":
        return None
    try:
        from tools import _buscar_comida_principal_reciente_ttl
        fila = _buscar_comida_principal_reciente_ttl(user_id)
    except Exception:                                                          # noqa: BLE001
        return None
    if not isinstance(fila, dict):
        return None
    nombre = str(fila.get("meal_name") or "").strip()
    ings = fila.get("ingredients")
    if isinstance(ings, list):
        nombres = [str(i.get("name") if isinstance(i, dict) else i).strip() for i in ings][:12]
        if nombres:
            nombre = f"{nombre} ({', '.join(n for n in nombres if n)})"
    return nombre[:500] or None


def _triaje_del_turno(texto: str, user_id: Optional[str], turn_id: str) -> Optional[TriageResult]:
    t0 = time.perf_counter()
    activa = _comida_activa_de(user_id)
    res = run_triage(texto, comida_activa=activa, user_id=user_id)
    registrar_metrica(user_id, "decision_triage_shadow", (time.perf_counter() - t0) * 1000.0, {
        "turn_id": turn_id, "mode": modo(), "ok": res is not None, "con_comida_activa": bool(activa),
        "latency_ms": res.latency_ms if res else None, "answers": res.resumen() if res else {},
    })
    return res


def iniciar_turno(texto: Optional[str], user_id: Optional[str]) -> Optional[str]:
    """Lanza el triaje del mensaje en paralelo al agente. Devuelve el turn_id (None si no aplica)."""
    try:
        if modo() == "off" or not (texto or "").strip() or _excluido_por_region():
            _TURNO.set(None)
            return None
        turn_id = uuid.uuid4().hex[:12]
        ctx = contextvars.copy_context()
        fut = _pool().submit(ctx.run, _triaje_del_turno, texto, user_id, turn_id)
        _TURNO.set(_Turno(turn_id=turn_id, future=fut, user_id=user_id))
        return turn_id
    except Exception:                                                          # noqa: BLE001
        _TURNO.set(None)
        return None


def resultado_del_turno(espera: Optional[float] = None) -> tuple[Optional[str], Optional[TriageResult]]:
    turno = _TURNO.get()
    if turno is None:
        return None, None
    try:
        return turno.turn_id, turno.future.result(timeout=espera_chat_s() if espera is None else espera)
    except Exception:                                                          # noqa: BLE001
        return turno.turn_id, None


def _ya_se_freno(messages) -> bool:
    """Tope de un freno por comida: si un turno reciente ya frenó, el usuario ya contestó — no se vuelve a preguntar."""
    for m in list(messages or [])[-10:]:
        c = getattr(m, "content", None)
        if isinstance(c, str) and FRENO_MARCA in c:
            return True
    return False


def freno_de_bebida(tool_name: str, messages, *, turno_con_foto: bool = False,
                    repeticion: bool = False) -> Optional[str]:
    """Texto del ToolMessage que frena la escritura, o None (= ejecutar como siempre). Registra la decisión."""
    if tool_name != _TOOL_FRENABLE:
        return None
    try:
        m = modo()
        if m == "off":
            return None
        turn_id, res = resultado_del_turno()
        if res is None:
            return None
        p = res.si(Q_BEBIDA)
        frenar = (m == "guard" and p is not None and p >= umbral_bebida()
                  and not turno_con_foto and not repeticion and not _ya_se_freno(messages))
        turno = _TURNO.get()
        registrar_metrica(turno.user_id if turno else None, "decision_triage_tool", 0, {
            "turn_id": turn_id, "mode": m, "tool": tool_name, "beverage_ambiguity": p, "frenado": frenar,
        })
        if not frenar:
            return None
        logger.info(f"🥤 [P1-DECISION-TRIAGE] freno de bebida ambigua (p={p:.2f}) antes de {tool_name}")
        return (f"{FRENO_MARCA} No guardes todavía. Pregunta al usuario SOLO si la bebida lleva azúcar "
                "añadida o si es regular, light o zero, en una frase corta. Cuando conteste, regístrala con "
                "`log_consumed_meal`. (Nota interna: no le digas que hubo un freno.)")
    except Exception:                                                          # noqa: BLE001
        return None


__all__ = [
    "DECISIONS_URL", "TIPO_SI_NO", "FRENO_MARCA", "TriageResult", "build_request", "parse_response",
    "run_triage", "run_triage_async", "modo", "foto_rechazada", "payload_no_comida", "iniciar_turno",
    "resultado_del_turno", "freno_de_bebida", "registrar_metrica",
]
