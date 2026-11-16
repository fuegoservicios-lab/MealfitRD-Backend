# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1041 · A-31 · 2026-11-16] Las fotos del chat se borran pasados 90 días desde que se enviaron.

Decisión del dueño (16-nov). La Política de Privacidad lo promete en §2 y §9 —en la app (LegalPages.jsx) y en
bioboros.com (bioboros-cinematic, content/privacy.html)—: los 90 días NO son knob; moverlos es cambiar la política.

Se borra la fila de `chat_attachments` (la foto). Se conservan el mensaje del chat (la burbuja pinta «Imagen no
disponible»: MessageBubble → ChatImage.onBroken) y las comidas del diario que salieron de ella (sus cifras viven en
`consumed_meals`; la copia del dispositivo para la ficha de la comida no la toca el servidor). Las no enviadas siguen
con su propia purga de 24 h (db_chat.purge_orphan_chat_attachments).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DIAS = 90            # la promesa publicada (Privacidad §2 y §9)
LOTE = 100
MAX_LOTES = 50       # 5.000 fotos por pasada; lo que quede lo recoge la del día siguiente

_SQL = f"""
    WITH viejas AS (
        SELECT id FROM public.chat_attachments
        WHERE created_at < now() - interval '{DIAS} days'
        ORDER BY created_at
        LIMIT %s
    )
    DELETE FROM public.chat_attachments a
    USING viejas WHERE a.id = viejas.id
    RETURNING a.id::text AS id
"""


def _borrar_lote(limite: int) -> int:
    from db_core import execute_sql_write
    return len(execute_sql_write(_SQL, (int(limite),), returning=True) or [])


def purgar_fotos_viejas(lote: int = LOTE, max_lotes: int = MAX_LOTES) -> int:
    """Cron diario. Un fallo de la base se RELANZA: la promesa es pública y el listener del scheduler la escala a
    `system_alerts` (mismo criterio que la purga de 24 h del lote 798). Devuelve cuántas fotos borró."""
    total = 0
    for _ in range(max(1, int(max_lotes))):
        try:
            n = _borrar_lote(max(1, int(lote)))
        except Exception as exc:
            logger.warning("[P1-PLAN-LOTE-1041] La purga de fotos del chat de más de %s días falló tras %s: %s",
                           DIAS, total, type(exc).__name__)
            raise
        total += n
        if n < lote:
            break
    logger.info("[P1-PLAN-LOTE-1041] Fotos del chat de más de %s días borradas: %s.", DIAS, total)
    return total


def registrar(scheduler, add_job, misfire_grace_s) -> None:
    """Lo llama `cron_tasks` (que está en su tope de líneas): cada 24 h, como las demás purgas de plazo."""
    if scheduler.get_job("purge_chat_attachments_90d"):
        return
    add_job(scheduler, purgar_fotos_viejas, "interval", hours=24, id="purge_chat_attachments_90d",
            max_instances=1, coalesce=True, replace_existing=True, misfire_grace_time=misfire_grace_s)
    logger.info("⏰ [P1-PLAN-LOTE-1041] Cron purge_chat_attachments_90d registrado cada 24 h.")
