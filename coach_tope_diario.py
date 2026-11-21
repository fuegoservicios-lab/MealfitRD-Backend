# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1045 · A-05 / D5 · 2026-11-16] El tope diario por usuario, cerrado como lo decidió el dueño.

D5 (16-nov): US$ 0,25 en 24 h corridas, las FOTOS CUENTAN (el escáner, `vision_scan`, hasta US$ 0,016 cada una, no
entraba: solo `node LIKE 'chat%'`), y quedan exentos el dueño/admin (ya lo estaban, lote 1016) y las cuentas de prueba.
El aviso dice la HORA real a la que se libera, en la hora local del usuario y en su idioma (antes: «en unas horas»).

La hora: con una ventana móvil de 24 h el gasto baja cuando los eventos más viejos salen de la ventana. Se recorren los
eventos de las últimas 24 h del más viejo al más nuevo restando su coste hasta quedar por debajo del tope; la hora es la
de ese evento + 24 h. Fail-open: si algo falla, el aviso sigue sin hora (el texto de siempre).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lo que cuenta contra el tope: el coach y el escáner de fotos.
FILTRO_NODOS_SQL = "(node LIKE 'chat%%' OR node = 'vision_scan')"
# [P1-PLAN-LOTE-1140] Lo que contestó el RESPALDO del coach (el principal falló) no cuenta contra los topes del usuario:
# la caída es nuestra, no suya. El gasto lo absorbe el dueño, acotado por `coach_respaldo_tope` (US$ 0,50/h).
SIN_RESPALDO_SQL = "COALESCE(metadata->>'respaldo', '') <> 'true'"


def es_cuenta_de_prueba(user_id: Optional[str]) -> bool:
    if not user_id:
        return False
    try:
        import cuentas_prueba
        return cuentas_prueba.marca_viva(str(user_id)) is not None
    except Exception as exc:  # noqa: BLE001
        logger.debug("[P1-PLAN-LOTE-1045] cuenta de prueba ilegible: %s", type(exc).__name__)
        return False


def libre_desde(user_id: str, limite_usd: float, *, execute_sql_query: Any = None) -> Optional[datetime]:
    """UTC a partir de la cual el gasto de 24 h corridas vuelve a estar por debajo del tope, o None."""
    try:
        if execute_sql_query is None:
            from db_core import execute_sql_query
        rows = execute_sql_query(
            "SELECT cost_usd_micros AS c, created_at AS t FROM public.llm_usage_events "
            f"WHERE user_id = %s AND {FILTRO_NODOS_SQL} AND {SIN_RESPALDO_SQL} AND created_at >= NOW() - INTERVAL '24 hours' "
            "ORDER BY created_at",
            (str(user_id),), fetch_all=True,
        ) or []
        total = sum(int((r.get("c") if isinstance(r, dict) else r[0]) or 0) for r in rows)
        tope = int(round(float(limite_usd) * 1_000_000))
        for r in rows:
            c = int((r.get("c") if isinstance(r, dict) else r[0]) or 0)
            t = r.get("t") if isinstance(r, dict) else r[1]
            total -= c
            if total < tope and isinstance(t, datetime):
                return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)) + timedelta(hours=24)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[P1-PLAN-LOTE-1045] hora de liberación del tope no calculada: %s", type(exc).__name__)
    return None


_FRASE = {
    "es": "Podrás volver a escribirle a partir de las {h}.",
    "en": "You'll be able to message the coach again from {h}.",
    "pt": "Você poderá falar com o coach de novo a partir das {h}.",
    "fr": "Vous pourrez réécrire au coach à partir de {h}.",
    "it": "Potrai scrivere di nuovo al coach dalle {h}.",
}


def frase_de_la_hora(libre_utc: Optional[datetime], tz_offset_min: Optional[int], idioma: str) -> str:
    """«Podrás volver a escribirle a partir de las 14:05.» en la hora local (tz_offset = convención JS: UTC − local)."""
    if not isinstance(libre_utc, datetime):
        return ""
    try:
        local = libre_utc - timedelta(minutes=int(tz_offset_min or 0))
    except (TypeError, ValueError):
        local = libre_utc
    return _FRASE.get(idioma, _FRASE["es"]).format(h=local.strftime("%H:%M"))
