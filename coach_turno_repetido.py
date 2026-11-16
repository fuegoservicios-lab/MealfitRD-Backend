# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1038 · A-23 · 2026-11-16] Mismo turno dos veces = una sola respuesta (y una sola escritura).

El cliente identifica cada mensaje con un `client_message_id` y lo REUTILIZA al reintentar (botón «Reintentar», corte de
red, iOS que suspende la vista web). El mensaje del usuario ya se guardaba una sola vez (`ON CONFLICT … DO NOTHING`),
pero el TURNO se ejecutaba entero otra vez: segunda llamada al modelo, segunda respuesta en el historial y —si el
modelo volvía a registrar la comida con otras cifras o pasado el minuto del antirrebote— segunda fila en el diario.
El caso real: el turno corre en su propio hilo (lote 760) y termina aunque el teléfono se vaya; el rescate del cliente
espera 26 s y, si no llegó, deja «Reintentar». El reintento llegaba con el primer turno aún vivo o recién terminado.

Ahora, cuando el router ve que ese `client_message_id` YA estaba guardado:

  1. espera a que termine el turno anterior de la sesión (los turnos van en serie, lote 1037);
  2. si el mensaje ya tiene su respuesta en el historial, la REPITE (`chunk` + `done` con `replayed: true`) sin llamar
     al modelo, sin ejecutar herramientas, sin guardar nada y sin cobrar;
  3. si no la tiene y el primer turno sigue vivo (tardó más que la espera), avisa con `turn_in_progress` en vez de
     lanzar un segundo turno en paralelo;
  4. si no la tiene y no hay turno vivo (el primero murió sin responder), el turno se ejecuta normalmente.

«Su respuesta» = el primer mensaje posterior al del usuario en esa sesión, si es del coach. Si entre medias hay otro
mensaje del usuario (dos dispositivos), no se repite nada y el turno corre como hasta hoy.

Knob `MEALFIT_CHAT_TURNO_REPETIDO` (1). Falla abierto: ante cualquier error de lectura, el turno se ejecuta.
tooltip-anchor: P1-PLAN-LOTE-1038-TURNO-REPETIDO
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_SQL_RESPUESTA = """
    SELECT n.role AS role, n.content AS content
      FROM public.agent_messages u
      JOIN LATERAL (
            SELECT m.role, m.content
              FROM public.agent_messages m
             WHERE m.session_id = u.session_id
               AND m.created_at > u.created_at
               AND m.content NOT LIKE '[SYSTEM_TITLE]%%'
             ORDER BY m.created_at ASC
             LIMIT 1
           ) n ON true
     WHERE u.session_id = %s AND u.client_message_id = %s AND u.role = 'user'
     LIMIT 1
"""

MENSAJE_EN_CURSO = "Tu mensaje anterior todavía se está respondiendo. Espera un momento."


def activo() -> bool:
    return (os.getenv("MEALFIT_CHAT_TURNO_REPETIDO", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def respuesta_ya_dada(session_id, client_message_id, execute_sql_query=None) -> Optional[str]:
    """La respuesta del coach a ESE mensaje si ya está en el historial; None si no la hay (o no se pudo leer)."""
    if not session_id or not client_message_id:
        return None
    try:
        if execute_sql_query is None:
            from db_core import execute_sql_query as _q
            execute_sql_query = _q
        fila = execute_sql_query(_SQL_RESPUESTA, (str(session_id), str(client_message_id)), fetch_one=True)
    except Exception as exc:  # noqa: BLE001 — sin lectura no hay repetición: el turno corre
        from fail_open import fail_open
        fail_open("chat_turno_repetido", exc)
        return None
    if not isinstance(fila, dict) or fila.get("role") != "model":
        return None
    texto = str(fila.get("content") or "")
    return texto if texto.strip() else None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def eventos_de_repeticion(texto: str, corr=None) -> list:
    """Los mismos dos eventos que un turno corto, marcados `replayed` (el router no los cobra ni los guarda)."""
    return [
        _sse({"type": "chunk", "text": texto, "replayed": True}),
        _sse({
            "type": "done", "corr": corr, "response": texto, "replayed": True,
            "updated_fields": {}, "new_plan": None, "coherence_warnings": [],
            "pantry_modified_at": None, "pantry_depleted_items": None,
            "ajustes_de_app": None, "diary_corrected_meal_ids": [],
        }),
    ]


def evento_en_curso() -> str:
    return _sse({"type": "error", "code": "turn_in_progress", "message": MENSAJE_EN_CURSO})


def es_repeticion(evento) -> bool:
    return isinstance(evento, dict) and evento.get("replayed") is True


def resolver(session_id, client_message_id, corr=None, turno_vivo=None, execute_sql_query=None) -> Optional[list]:
    """Para un mensaje que YA estaba guardado: los eventos a emitir en lugar del turno, o None si hay que ejecutarlo.

    `turno_vivo(session_id) -> bool` dice si este proceso sigue sirviendo un turno de la sesión (por defecto,
    `agent.is_turn_active`). Se llama DESPUÉS de esperar al turno anterior."""
    if not activo():
        return None
    texto = respuesta_ya_dada(session_id, client_message_id, execute_sql_query=execute_sql_query)
    if texto is not None:
        logger.info("[P1-PLAN-LOTE-1038] reintento del mismo mensaje en %s: se repite la respuesta ya dada", str(session_id)[:8])
        return eventos_de_repeticion(texto, corr)
    try:
        if turno_vivo is None:
            from agent import is_turn_active as turno_vivo
        vivo = bool(turno_vivo(session_id))
    except Exception:  # noqa: BLE001
        vivo = False
    if vivo:
        logger.info("[P1-PLAN-LOTE-1038] reintento del mismo mensaje en %s con el turno aún vivo", str(session_id)[:8])
        return [evento_en_curso()]
    return None


# ───────────────────────────────────────────────────────────── el corte, visto desde el teléfono
# Con el turno en su propio hilo (lote 760) el servidor ya NO ve los cortes: el stream se abandona y el turno termina
# igual, así que todos los turnos salen `ok`. Lo que le pasó al usuario solo lo sabe el cliente: el stream se cortó y
# su rescate (sondeo del historial) recuperó la respuesta (`rescued`), se rindió (`exhausted`) o el servidor dijo que
# el turno había muerto (`dead`). Una fila por episodio en `pipeline_metrics`, sin texto: resultado, intentos y espera.
NODO_CORTE = "chat_client_stream_cut"
RESULTADOS_DE_CORTE = ("rescued", "exhausted", "dead")


def registrar_corte_del_cliente(*, user_id, session_id: str, payload: dict) -> dict:
    """Guarda el episodio de corte que reporta el cliente. Nunca lanza."""
    from chat_phase_metrics import _clamp_ms, _db_safe_user_id, phase_metrics_enabled, sanitize_corr
    datos = payload if isinstance(payload, dict) else {}
    resultado = str(datos.get("outcome") or "")
    if resultado not in RESULTADOS_DE_CORTE:
        return {"recorded": False, "reason": "outcome"}
    intentos = datos.get("attempts")
    intentos = max(0, min(100, int(intentos))) if isinstance(intentos, (int, float)) and not isinstance(intentos, bool) else 0
    espera_ms = _clamp_ms(datos.get("waited_ms")) or 0
    meta = {
        "corr": sanitize_corr(datos.get("corr")),
        "outcome": resultado,
        "attempts": intentos,
        "waited_ms": espera_ms,
        "platform": str(datos.get("platform") or "").strip()[:32] or None,
    }
    if not phase_metrics_enabled():
        return {"recorded": False, "outcome": resultado}
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (_db_safe_user_id(user_id, session_id), session_id, NODO_CORTE, int(espera_ms),
             json.dumps(meta, ensure_ascii=False)),
        )
        return {"recorded": True, "outcome": resultado}
    except Exception as exc:  # noqa: BLE001
        from fail_open import fail_open
        fail_open("chat_client_stream_cut_emit", exc)
        return {"recorded": False, "outcome": resultado}
