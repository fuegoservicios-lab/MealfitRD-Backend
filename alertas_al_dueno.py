# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1038 · A-25 · 2026-11-16] Las alertas que importan llegan al dueño sin que abra el panel.

Todas las alertas del sistema se escriben en `system_alerts` y solo se ven en /admin: una protección del coach
fallando en silencio, el tope de gasto o el límite del proveedor podían pasar un día entero sin que nadie los viera.
Medido el 16-nov: 36 tipos de alerta, ningún canal de aviso fuera del panel.

Qué avisa (push a los dispositivos de las cuentas de `MEALFIT_ADMIN_USER_IDS`):
  · cualquier alerta `critical` abierta en las últimas 24 h, de cualquier subsistema;
  · las del coach que piden mirar ya aunque sean `warning`: una protección fallando (`fail_open`), el guardarraíl de
    gasto (`cost_guardrail`), el límite del proveedor (`openai_ratelimit`) y las de salud del chat (`chat…`).

Qué NO hace: no avisa dos veces de la misma alerta en 24 h, no manda más de 3 push por pasada (el resto se resume en
la última: «y N más en el panel») y nunca rompe el cron que lo llama. El texto del push es el TÍTULO de la alerta, sin
identificadores de usuario. Corre dentro del vigilante de gasto (`_llm_daily_spend_alert_job`, cada 60 min), sin job
nuevo. Estado: UNA clave en `app_kv_store` (`alertas_al_dueno:avisadas`), podada a 7 días.

Knob `MEALFIT_ALERTAS_PUSH_AL_DUENO` (1). tooltip-anchor: P1-PLAN-LOTE-1038-ALERTAS-AL-DUENO
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

CLAVE_KV = "alertas_al_dueno:avisadas"
TIPOS_DEL_COACH = ("fail_open", "cost_guardrail", "openai_ratelimit")
MAX_POR_PASADA = 3
REPETIR_S = 24 * 3600
OLVIDAR_S = 7 * 24 * 3600
TITULO_PUSH = "Bioboros · alerta del sistema"

_SQL_PENDIENTES = """
    SELECT alert_key, alert_type, severity, title
      FROM system_alerts
     WHERE resolved_at IS NULL
       AND triggered_at >= NOW() - INTERVAL '24 hours'
       AND (severity = 'critical' OR alert_type = ANY(%s) OR alert_key LIKE 'chat%%')
     ORDER BY (severity = 'critical') DESC, triggered_at DESC
     LIMIT 50
"""
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def activo() -> bool:
    return (os.getenv("MEALFIT_ALERTAS_PUSH_AL_DUENO", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def texto_del_push(titulo: Any, mas: int = 0) -> str:
    """El título de la alerta en una línea, sin identificadores; con «y N más en el panel» si quedaron sin push."""
    limpio = " ".join(_UUID.sub("…", str(titulo or "Alerta sin título")).split())[:140]
    return f"{limpio} (y {mas} más en el panel)" if mas > 0 else limpio


def _leer_avisadas(execute_sql_query) -> dict:
    fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (CLAVE_KV,), fetch_one=True)
    valor = fila.get("value") if isinstance(fila, dict) else None
    if isinstance(valor, str):
        try:
            valor = json.loads(valor)
        except ValueError:
            valor = None
    return {str(k): float(v) for k, v in valor.items() if isinstance(v, (int, float))} if isinstance(valor, dict) else {}


def _guardar_avisadas(execute_sql_write, avisadas: dict, ahora: float) -> None:
    vivas = {k: v for k, v in avisadas.items() if ahora - v < OLVIDAR_S}
    execute_sql_write(
        """
        INSERT INTO app_kv_store (key, value, updated_at)
        VALUES (%s, %s::jsonb, NOW())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """,
        (CLAVE_KV, json.dumps(vivas)),
    )


def avisar_pendientes(
    execute_sql_query: Optional[Callable] = None,
    execute_sql_write: Optional[Callable] = None,
    enviar: Optional[Callable] = None,
    admins: Optional[Iterable[str]] = None,
    ahora: Optional[float] = None,
) -> dict:
    """Manda un push por cada alerta que importa y aún no se avisó. Devuelve `{enviadas, nuevas, destinatarios}`."""
    resumen = {"enviadas": 0, "nuevas": 0, "destinatarios": 0}
    if not activo():
        return resumen
    try:
        if admins is None:
            from admin_acceso import admin_ids
            admins = admin_ids()
        destinatarios = sorted({str(a).strip() for a in (admins or ()) if str(a).strip()})
        resumen["destinatarios"] = len(destinatarios)
        if not destinatarios:
            return resumen
        if execute_sql_query is None or execute_sql_write is None:
            from db_core import execute_sql_query as _q, execute_sql_write as _w
            execute_sql_query = execute_sql_query or _q
            execute_sql_write = execute_sql_write or _w
        if enviar is None:
            from utils_push import send_push_notification as enviar
        t = time.time() if ahora is None else float(ahora)

        filas = execute_sql_query(_SQL_PENDIENTES, (list(TIPOS_DEL_COACH),)) or []
        if not filas:
            return resumen
        avisadas = _leer_avisadas(execute_sql_query)
        nuevas = [f for f in filas if isinstance(f, dict) and f.get("alert_key")
                  and t - avisadas.get(str(f["alert_key"]), 0.0) >= REPETIR_S]
        resumen["nuevas"] = len(nuevas)
        if not nuevas:
            return resumen

        con_push = nuevas[:MAX_POR_PASADA]
        sin_push = len(nuevas) - len(con_push)
        for i, f in enumerate(con_push):
            cuerpo = texto_del_push(f.get("title"), sin_push if i == len(con_push) - 1 else 0)
            for uid in destinatarios:
                try:
                    if enviar(uid, TITULO_PUSH, cuerpo, url="/admin", tag=f"alerta:{f['alert_key']}"[:64]):
                        resumen["enviadas"] += 1
                except Exception as exc:  # noqa: BLE001 — un dispositivo que falla no corta el resto
                    logger.warning("[P1-PLAN-LOTE-1038] push de alerta no enviado: %s", type(exc).__name__)
        for f in nuevas:   # también las resumidas: ya se avisó de que existen
            avisadas[str(f["alert_key"])] = t
        _guardar_avisadas(execute_sql_write, avisadas, t)
        logger.info("[P1-PLAN-LOTE-1038] alertas al dueño: %d nueva(s), %d push aceptado(s)", len(nuevas), resumen["enviadas"])
    except Exception as exc:  # noqa: BLE001
        from fail_open import fail_open
        fail_open("alertas_al_dueno", exc)
    return resumen
