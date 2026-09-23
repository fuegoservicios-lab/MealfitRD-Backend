# -*- coding: utf-8 -*-
"""[P1-NEVERA-OPCIONAL · 2026-09-23] La Nevera opcional en modo contador — SSOT.

El encargo del dueño: «cuando el generador de planes esté desactivado, que la Nevera también tenga una opción en
Configuración para desactivarla: hay gente que solo quiere el contador y el agente». Y sobre el estado inicial:
«encendida como hoy, pero si en 48 horas no se usa, igual que con la hidratación, que se desactive sola».

LA REGLA (una sola, aquí): `nevera_activa = NOT (plan_mode = 'tracking' AND nevera_enabled IS FALSE)`.
  · En modo plan la Nevera SIEMPRE está activa: la lista de compras, la reposición y «Me lo comí» la necesitan. Por
    eso encender el generador la devuelve sin tocar el flag, y volver al contador respeta la última elección.
  · `nevera_enabled` es TRIESTADO: NULL = automático (encendida y elegible para el apagado automático) · TRUE =
    encendida por el usuario (NUNCA se apaga sola: fue su decisión) · FALSE = apagada (por él o por el sistema).

EL APAGADO AUTOMÁTICO (`apagar_neveras_sin_uso`, cron horario): solo modo contador, solo NULL, y solo si la Nevera
lleva `MEALFIT_NEVERA_AUTO_OFF_HOURS` (48) VACÍA — ninguna fila con cantidad y ningún movimiento en esa ventana.
«Vacía» y no «sin añadir nada»: quien la llenó la semana pasada y lleva dos días sin añadir la sigue usando (el coach
la lee y sus comidas descuentan). El reloj arranca en `nevera_reloj_desde` (la migración estampó su hora en las
cuentas existentes: 48 h de gracia desde el despliegue) o en el último cambio de modo, lo que sea más tarde.

APAGADA NO BORRA NADA: se oculta, el diario deja de descontar y el coach deja de verla. Encenderla la devuelve tal
cual. Kill switch `MEALFIT_NEVERA_SWITCH` = todo como antes (activa para todos, sin tarjeta, sin cron).
tooltip-anchor: nevera_activa_de, nevera_activa, estado_nevera, fijar_nevera, apagar_neveras_sin_uso,
MENSAJE_NEVERA_APAGADA, BLOQUE_PROMPT_NEVERA_APAGADA (test_p1_nevera_opcional.py)
"""
from __future__ import annotations

import logging
from typing import Optional

from db import execute_sql_query, execute_sql_write
from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

# Lo que devuelve una herramienta de Nevera del coach cuando la Nevera está apagada.
MENSAJE_NEVERA_APAGADA = (
    "La Nevera está DESACTIVADA por el usuario (Configuración → Capacidades). No la uses, no la menciones y no "
    "ofrezcas añadir alimentos a ella. Si el usuario pregunta por su Nevera, dile que puede encenderla allí."
)

# El bloque que el system prompt del coach recibe en lugar del inventario.
BLOQUE_PROMPT_NEVERA_APAGADA = (
    "\n\n🧊 NEVERA: DESACTIVADA por el usuario (Configuración → Capacidades). No la menciones, no preguntes qué hay "
    "en ella, no ofrezcas escanearla ni añadirle alimentos y no uses sus herramientas. Si pregunta por ella, dile que "
    "puede encenderla en Configuración → Capacidades."
)


def interruptor_disponible() -> bool:
    """Kill switch del feature completo."""
    return _env_bool("MEALFIT_NEVERA_SWITCH", True)


def _auto_apagado_encendido() -> bool:
    return _env_bool("MEALFIT_NEVERA_AUTO_OFF", True)


def horas_para_apagar() -> int:
    return _env_int("MEALFIT_NEVERA_AUTO_OFF_HOURS", 48, validator=lambda v: 24 <= v <= 336)


def nevera_activa_de(perfil: Optional[dict]) -> bool:
    """LA regla, pura. Fallo abierto: sin perfil, sin columnas o con el kill switch, la Nevera está activa."""
    if not interruptor_disponible() or not isinstance(perfil, dict):
        return True
    return not (perfil.get("plan_mode") == "tracking" and perfil.get("nevera_enabled") is False)


def nevera_activa(user_id: Optional[str]) -> bool:
    """La regla leyendo la fila. Fallo abierto (invitado, columna aún sin migrar, DB caída ⇒ activa): equivocarse
    hacia «activa» es la conducta de siempre; hacia «apagada», perder un descuento real."""
    if not user_id or user_id == "guest" or not interruptor_disponible():
        return True
    try:
        row = execute_sql_query(
            "SELECT plan_mode, nevera_enabled FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True,
        )
    except Exception as e:
        logger.warning(f"[P1-NEVERA-OPCIONAL] nevera_activa({user_id}) sin leer (queda activa): {e}")
        return True
    return nevera_activa_de(row or {})


def estado_nevera(user_id: str) -> dict:
    """Lo que pinta Configuración: la elección (NULL/TRUE/FALSE), si está activa, cuándo la apagó el sistema y si el
    interruptor existe (knob)."""
    row: dict = {}
    try:
        row = execute_sql_query(
            "SELECT plan_mode, nevera_enabled, nevera_auto_off_at FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True,
        ) or {}
    except Exception as e:
        logger.warning(f"[P1-NEVERA-OPCIONAL] estado_nevera({user_id}) sin leer: {e}")
    auto = row.get("nevera_auto_off_at")
    return {
        "enabled": row.get("nevera_enabled"),
        "activa": nevera_activa_de(row) if row else True,
        "auto_off_at": auto.isoformat() if hasattr(auto, "isoformat") else auto,
        "disponible": interruptor_disponible(),
    }


def fijar_nevera(user_id: str, enabled: bool) -> bool:
    """La elección EXPLÍCITA del usuario: TRUE o FALSE, nunca NULL (encenderla a mano la saca del apagado automático
    para siempre) y borra la marca del apagado automático (la nota ya no aplica). Filtra por id (I2)."""
    try:
        res = execute_sql_write(
            "UPDATE user_profiles SET nevera_enabled = %s, nevera_auto_off_at = NULL WHERE id = %s RETURNING id",
            (bool(enabled), user_id), returning=True,
        )
        return bool(res)
    except Exception as e:
        logger.error(f"❌ [P1-NEVERA-OPCIONAL] fijar_nevera({user_id}, {enabled}): {e}")
        return False


_SQL_APAGAR = """
UPDATE user_profiles p
   SET nevera_enabled = FALSE, nevera_auto_off_at = now()
 WHERE p.id IN (
        SELECT q.id FROM user_profiles q
         WHERE q.plan_mode = 'tracking'
           AND q.nevera_enabled IS NULL
           AND GREATEST(q.nevera_reloj_desde, COALESCE(q.plan_mode_changed_at, q.nevera_reloj_desde))
               < now() - (%s * interval '1 hour')
           AND NOT EXISTS (
                 SELECT 1 FROM user_inventory i
                  WHERE i.user_id = q.id
                    AND (i.quantity > 0 OR i.updated_at > now() - (%s * interval '1 hour')))
         LIMIT %s)
   AND p.nevera_enabled IS NULL
RETURNING p.id
"""


def apagar_neveras_sin_uso(limite: int = 500) -> list:
    """El apagado automático: un solo UPDATE por lotes. Devuelve los ids apagados. No-op con los knobs apagados."""
    if not interruptor_disponible() or not _auto_apagado_encendido():
        return []
    horas = horas_para_apagar()
    try:
        filas = execute_sql_write(_SQL_APAGAR, (horas, horas, int(limite)), returning=True) or []
    except Exception as e:
        # [P1-NEVERA-OPCIONAL · 2026-09-23] error, no warning: un apagado automático roto es silencioso por
        # naturaleza (nadie lo espera activamente) — debe ser RUIDOSO en los logs o nadie lo va a notar.
        logger.error(f"❌ [P1-NEVERA-OPCIONAL] apagado automático no corrió: {e}")
        return []
    ids = [str(f.get("id")) for f in filas if isinstance(f, dict) and f.get("id")]
    if ids:
        logger.info(f"[P1-NEVERA-OPCIONAL] Nevera apagada sola (vacía {horas} h en modo contador): {len(ids)} cuenta(s)")
    return ids
