# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-693 · 2026-09-28] El aviso de comida ya no interroga a quien SÍ come esa comida.

Auditoría del chat (28-sep): el aviso del almuerzo del dueño decía «Ya casi te toca almorzar… ¿Qué te está fallando con
el almuerzo: el tiempo, que no te gusta, o que comes fuera?» — ANTES de la hora del almuerzo y a alguien que registró el
almuerzo 7 de los 9 días avisados. Dos defectos:

  1. La señal. El tono «le falla esta comida» salía de la tasa de RESPUESTA al aviso (contestar en el chat o registrar
     algo en las 3 h siguientes), no de si esa comida se registró ese día. Medido en producción, por comida avisada:
     desayuno 45 % de «respuesta» y registrado 10/11 días; almuerzo 33 % y 7/9; cena 64 % y 10/11; merienda 25 % y 0/12.
     La única que de verdad se salta es la merienda.
  2. La contradicción. El lote 413 prohibió en el prompt los interrogatorios («¿qué está fallando?») y este tono los
     PEDÍA; ganaba el tono. Dos reglas sobre la misma frase: la que llega más tarde en el prompt manda.

Ahora el tono de «se la salta» se decide con los días en que esa comida NO aparece registrada (por su fecha local, con
la misma regla de «ya registrada» del cron: tipo o nombre), sobre días ya cerrados y con un mínimo de días. Y los dos
tonos que preguntaban por obstáculos se reescriben como invitación, sin interrogar.
tooltip-anchor: P1-PLAN-LOTE-693-TONO-DEL-AVISO
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

UMBRAL_SALTO = 0.70   # la salta al menos 7 de cada 10 días avisados
MIN_DIAS = 5

TONO_SE_LA_SALTA = (
    "Esta comida casi nunca la registra: la salta la mayoría de los días. No le preguntes qué está fallando ni por qué: "
    "dale UNA opción muy fácil y rápida para esa comida y, en una frase final opcional, dile que si algo se le complica "
    "(el tiempo, que no le apetece o que está fuera) te lo cuente y se la adaptas. Si no suele comer esa comida, "
    "puede apagar solo ese recordatorio en Configuración → Recordatorios de comida: díselo en media frase."
)
TONO_POCA_RESPUESTA = (
    "Últimamente casi no contesta los avisos. Sé muy breve y sin presión: una invitación concreta y ya. No le preguntes "
    "por obstáculos, estrés ni por qué no registra, y no asumas que se le olvidó."
)

# Días ya cerrados (hoy no cuenta: todavía puede registrarla) en los que se avisó esa comida, y cuántos de ellos no
# tienen un registro de esa comida por tipo o por nombre en su fecha LOCAL.
_SQL = (
    "SELECT COUNT(*) AS dias, COALESCE(SUM(CASE WHEN EXISTS ("
    "  SELECT 1 FROM consumed_meals c WHERE c.user_id::text = d.user_id"
    "  AND (lower(coalesce(c.meal_type, '')) LIKE '%%' || d.comida || '%%'"
    "       OR lower(coalesce(c.meal_name, '')) LIKE '%%' || d.comida || '%%')"
    "  AND (c.consumed_at - make_interval(mins => %s))::date = d.dia"
    ") THEN 0 ELSE 1 END), 0) AS saltados "
    "FROM (SELECT DISTINCT n.user_id::text AS user_id, lower(n.nudge_type) AS comida,"
    "      (n.sent_at - make_interval(mins => %s))::date AS dia"
    "      FROM nudge_outcomes n WHERE n.user_id = %s AND lower(n.nudge_type) = lower(%s)"
    "      AND (n.sent_at - make_interval(mins => %s))::date < (NOW() - make_interval(mins => %s))::date) d"
)


def tasa_de_salto(user_id: str, comida: str, tz_offset_min: int = 240) -> tuple[float, int]:
    """`(fracción de días avisados en que NO registró esa comida, días)`. Sin datos o con error: `(0.0, 0)`."""
    try:
        from db import execute_sql_query
        off = int(tz_offset_min)
        r = execute_sql_query(_SQL, (off, off, user_id, comida, off, off), fetch_one=True)
        dias = int((r or {}).get("dias") or 0)
        if dias <= 0:
            return 0.0, 0
        return float((r or {}).get("saltados") or 0) / dias, dias
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[P1-PLAN-LOTE-693] tasa de salto de {comida} ilegible para {user_id}: {e!r}")
        return 0.0, 0


def se_la_salta(tasa: float, dias: int) -> bool:
    return dias >= MIN_DIAS and tasa >= UMBRAL_SALTO
