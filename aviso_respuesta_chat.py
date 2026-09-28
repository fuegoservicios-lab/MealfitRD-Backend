# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-692 · 2026-09-28] «Bioboros te respondió» con la app cerrada.

El dueño: «cuando envío un mensaje al agente y él está cargando y yo me salgo de la app, me debería enviar una
notificación cuando termine su respuesta, y actualmente no lo hace».

Salir ya se podía: el turno sigue en el servidor y la respuesta se GUARDA en el `done` (medido: 0 turnos abortados por
el cliente en 7 días de producción; el `fetch` sobrevive al cambio de app lo que dura un turno). Lo que faltaba era
ENTERARSE, igual que con el plan (`aviso_plan_listo.py`, lote 228): el único aviso era la burbuja dentro del chat.

Se manda en CADA respuesta terminada, con `solo_si_no_mira`: quien la está viendo en pantalla no recibe nada (el service
worker la calla con una ventana visible; en la app nativa iOS no pinta pushes en primer plano —`presentationOptions: []`—
y el oyente de Android las descarta con esa marca). Así no hace falta que el servidor adivine si el usuario se fue.
El cuerpo es el principio de la respuesta, sin markdown: ya viene en el idioma del usuario. Knob
`MEALFIT_CHAT_REPLY_PUSH` (True).
tooltip-anchor: P1-PLAN-LOTE-692-AVISO-RESPUESTA-CHAT
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

TITULO = "Bioboros te respondió"
ETIQUETA = "chat-respuesta"
RUTA = "/dashboard/agent"
MAX_CUERPO = 140

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_UI_ACTION = re.compile(r"\[UI_ACTION:[^\]]*\]")
_ENLACE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MARCAS = re.compile(r"[*_`#>~|]+")
_VIÑETA = re.compile(r"(^|\n)\s*(?:[-•]|\d+\.)\s+")


def _activo() -> bool:
    from knobs import _env_bool
    return _env_bool("MEALFIT_CHAT_REPLY_PUSH", True)


def cuerpo_de(texto) -> str:
    """El principio de la respuesta, legible en la pantalla de bloqueo: sin markdown ni etiquetas, en una línea y
    cortado en palabra. Pura."""
    t = str(texto or "")
    t = _UI_ACTION.sub("", t)
    t = _ENLACE.sub(r"\1", t)
    t = _VIÑETA.sub(r"\1", t)
    t = _MARCAS.sub("", t)
    t = " ".join(t.split())
    if len(t) <= MAX_CUERPO:
        return t
    corte = t[:MAX_CUERPO].rsplit(" ", 1)[0].rstrip(",;:.—-")
    return (corte or t[:MAX_CUERPO]) + "…"


def avisar_respuesta(user_id, texto) -> bool:
    """Despacha la push de la respuesta en segundo plano. Best-effort: nunca lanza. Solo cuentas (UUID)."""
    try:
        if not _activo() or not isinstance(user_id, str) or not _UUID.match(user_id):
            return False
        cuerpo = cuerpo_de(texto)
        if not cuerpo:
            return False
        from utils_push import send_push_notification

        def _enviar():
            send_push_notification(user_id, TITULO, cuerpo, url=RUTA, tag=ETIQUETA, solo_si_no_mira=True)

        # En segundo plano: el `done` va en el camino del stream y `webpush` tiene su tope de 10 s por suscripción.
        from bg_executor import submit_bg_task
        submit_bg_task(_enviar, task_name="aviso_respuesta_chat")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-692] aviso de respuesta del chat no enviado: {e!r}")
        return False
