# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1045 · A-22 / D6 · 2026-11-16] Tope de gasto del modelo de RESPALDO del coach: US$ 0,50 por hora.

Decisión del dueño (16-nov). Cuando el principal (`gpt-6-luna`) falla, el turno pasa al respaldo
(`MEALFIT_CHAT_AGENT_FALLBACK_MODEL`, hoy `gemini-4-hydrogen`). Una caída larga del principal podía costar sin límite:
el 13-nov, con Hydrogen como principal, una sola hora costó US$ 1,08. Ahora, si el respaldo ya gastó el tope en los
últimos 60 minutos, el turno NO usa el respaldo: falla con el error original del principal (el cliente pinta
«saturado») hasta que la ventana baje del tope, y queda una alerta en `system_alerts` (A-25 se la empuja al dueño).

El gasto sale de `llm_usage_events` (node `chat_call_model`, model = el del respaldo) con caché de 60 s por proceso.
Fail-open: si la base no responde se permite el respaldo (equivocarse hacia «cortar» deja al usuario sin coach por un
fallo de la telemetría).

Knob: `MEALFIT_CHAT_FALLBACK_MAX_USD_PER_HOUR` (0,50; 0 = sin tope).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from knobs import _env_float

logger = logging.getLogger(__name__)

NODE = "chat_call_model"
ALERT_KEY = "chat_fallback_hourly_cap"
_TTL_S = 60.0
_cache: dict = {}
_lock = threading.Lock()


def tope_usd() -> float:
    v = _env_float("MEALFIT_CHAT_FALLBACK_MAX_USD_PER_HOUR", 0.50)
    return max(0.0, float(v or 0.0))


def _gasto_sql(model: str) -> float:
    from db_core import execute_sql_query
    row = execute_sql_query(
        "SELECT COALESCE(SUM(cost_usd_micros), 0) AS micros FROM llm_usage_events "
        "WHERE created_at > now() - interval '60 minutes' AND node = %s AND model = %s",
        (NODE, str(model)), fetch_one=True,
    )
    micros = (row or {}).get("micros") if isinstance(row, dict) else (row[0] if row else 0)
    return float(micros or 0) / 1_000_000.0


def gasto_ultima_hora(model: str) -> Optional[float]:
    """US$ del respaldo en los últimos 60 min (caché 60 s), o None si no se pudo leer."""
    ahora = time.monotonic()
    with _lock:
        hit = _cache.get(model)
        if hit and ahora - hit[0] < _TTL_S:
            return hit[1]
    try:
        v = _gasto_sql(model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[P1-PLAN-LOTE-1045] gasto del respaldo ilegible (%s): se permite", type(exc).__name__)
        return None
    with _lock:
        _cache[model] = (ahora, v)
    return v


def _alertar(model: str, gasto: float, tope: float) -> None:
    try:
        from db_core import execute_sql_write
        from chat_observability_alerts import _upsert_alert
        _upsert_alert(
            execute_sql_write, alert_key=ALERT_KEY, alert_type="chat_fallback_hourly_cap", severity="critical",
            title="Coach: el respaldo llegó a su tope por hora",
            message=(f"El modelo de respaldo {model} gastó US$ {gasto:.2f} en la última hora (tope US$ {tope:.2f}). "
                     "Los turnos que necesitan respaldo fallan como «saturado» hasta que baje. Revisa el principal."),
            metadata={"model": model, "gasto_usd": round(gasto, 4), "tope_usd": tope},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[P1-PLAN-LOTE-1045] no se pudo escribir la alerta del tope: %s", type(exc).__name__)


def permitir(model: Optional[str]) -> bool:
    """¿Puede este turno usar el respaldo `model`? False solo con el tope alcanzado y leído de verdad."""
    tope = tope_usd()
    if not model or tope <= 0:
        return True
    gasto = gasto_ultima_hora(str(model))
    if gasto is None or gasto < tope:
        return True
    logger.error("🛑 [P1-PLAN-LOTE-1045] respaldo %r cortado: US$ %.2f en 60 min (tope %.2f)", model, gasto, tope)
    _alertar(str(model), gasto, tope)
    return False
