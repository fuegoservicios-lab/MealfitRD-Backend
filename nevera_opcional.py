# -*- coding: utf-8 -*-
"""[P1-NEVERA-OPCIONAL · 2026-09-23] La Nevera opcional en modo contador — SSOT.

El encargo del dueño: «cuando el generador de planes esté desactivado, que la Nevera también tenga una opción en
Configuración para desactivarla: hay gente que solo quiere el contador y el agente». Y sobre el estado inicial:
«encendida como hoy, pero si en 48 horas no se usa, igual que con la hidratación, que se desactive sola».

LA REGLA (una sola, aquí): `nevera_activa = NOT (nevera_enabled IS FALSE AND (plan_mode = 'tracking' OR
MEALFIT_NEVERA_OFF_IN_PLAN_MODE))`.
  · [P1-PLAN-LOTE-217 · 2026-09-24] También en modo plan se puede apagar; antes era obligatoria y, vacía 48 h,
    congelaba el plan. Ver el bloque del final (apagado automático en modo plan y el freno por inactividad). Con el
    knob en False, en modo plan vuelve a estar SIEMPRE activa (la regla del 23-sep).
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

# [P1-NEVERA-OPCIONAL · ola final · 2026-09-23] Los dos textos son NEUTROS sobre quién la apagó: el apagado automático (48 h vacía)
# también la deja en FALSE, y «desactivada por el usuario» ponía al coach a afirmar algo que el usuario no hizo.
# Lo que la frase ORDENA no cambió.
# Lo que devuelve una herramienta de Nevera del coach cuando la Nevera está apagada.
MENSAJE_NEVERA_APAGADA = (
    "La Nevera está DESACTIVADA (Configuración → Capacidades). No la uses, no la menciones y no "
    "ofrezcas añadir alimentos a ella. Si el usuario pregunta por su Nevera, dile que puede encenderla allí."
)

# El bloque que el system prompt del coach recibe en lugar del inventario.
BLOQUE_PROMPT_NEVERA_APAGADA = (
    "\n\n🧊 NEVERA: DESACTIVADA (Configuración → Capacidades). No la menciones, no preguntes qué hay "
    "en ella, no ofrezcas escanearla ni añadirle alimentos y no uses sus herramientas. Si pregunta por ella, dile que "
    "puede encenderla en Configuración → Capacidades. [P1-PLAN-LOTE-290] Excepción: Si te PIDE guardar algo en ella (un "
    "alimento, o un suplemento o la foto de su pote), llama igual a modify_pantry_inventory o guardar_suplemento: la "
    "herramienta decide si la enciende sola o te dice que le preguntes."
)

# [P1-PLAN-LOTE-290 · 2026-09-25] Lo que devuelve una herramienta que quiso GUARDAR con la Nevera apagada a mano.
MENSAJE_NEVERA_PREGUNTAR = (
    "La Nevera de este usuario está APAGADA porque la apagó a mano. (Para el asistente: NO guardaste nada. Pregúntale "
    "en una frase si quiere que la enciendas para guardarlo; si dice que sí, repite esta herramienta con "
    "encender_nevera=true.)"
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
    if perfil.get("nevera_enabled") is not False:
        return True
    # [P1-PLAN-LOTE-217] apagada: en modo contador siempre; en modo plan, con el knob
    return not (perfil.get("plan_mode") == "tracking" or apagable_en_modo_plan())


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


def _perfil_nevera(user_id: str) -> dict:
    """plan_mode, nevera_enabled y nevera_auto_off_at del perfil ({} si no hay fila o falla la lectura)."""
    try:
        fila = execute_sql_query(
            "SELECT plan_mode, nevera_enabled, nevera_auto_off_at FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True,
        ) or {}
        return dict(fila) if isinstance(fila, dict) else {}
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-290] perfil de la Nevera de {user_id} sin leer: {e}")
        return {}


def encender_por_uso(user_id: str, *, forzar: bool = False) -> str:
    """[P1-PLAN-LOTE-290 · 2026-09-25] «La Nevera debe activarse solo cuando se necesite» (el dueño). Guardar algo en
    ella (coach, formulario) la enciende si la apagó el SISTEMA: vuelve a automático con 48 h nuevas, así que puede
    volver a apagarse si no se usa. Si la apagó el USUARIO, devuelve 'preguntar' sin escribir nada; solo `forzar=True`
    (su «sí» al coach, o guardar desde el formulario) la enciende, y entonces de forma definitiva (`fijar_nevera`).
    Devuelve 'activa' | 'encendida' | 'preguntar'. tooltip-anchor: P1-PLAN-LOTE-290-ENCENDER"""
    p = _perfil_nevera(user_id)
    if nevera_activa_de(p):
        return "activa"
    if p.get("nevera_auto_off_at") and not forzar:
        execute_sql_write(
            "UPDATE user_profiles SET nevera_enabled = NULL, nevera_auto_off_at = NULL, nevera_reloj_desde = now() "
            "WHERE id = %s",
            (user_id,),
        )
        return "encendida"
    if forzar:
        fijar_nevera(user_id, True)
        return "encendida"
    return "preguntar"


# [SUPLEMENTOS-OK: tener suplementos es usar la Nevera]
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
                    AND (i.quantity > 0 OR i.kind = 'supplement'
                         OR i.updated_at > now() - (%s * interval '1 hour')))
         LIMIT %s)
   AND p.nevera_enabled IS NULL
   AND p.plan_mode = 'tracking'
RETURNING p.id
"""
# [P1-NEVERA-OPCIONAL · ola final · 2026-09-23] El UPDATE externo repite sobre `p` lo que el usuario puede cambiar entre el SELECT
# interno y la escritura: su elección (`nevera_enabled IS NULL`) y el modo (`plan_mode = 'tracking'`). En READ
# COMMITTED, si otra transacción cambió la fila mientras tanto, Postgres re-evalúa las condiciones de `p` sobre la
# versión NUEVA; las del subselect `q` quedan con la foto del inicio de la sentencia.


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


# ─────────────────────────────────────────────── [P1-PLAN-LOTE-217 · 2026-09-24] La Nevera opcional TAMBIÉN en modo plan
# Hasta hoy, en modo plan la Nevera era obligatoria y, vacía 48 h, CONGELABA el plan (P1-PLAN-FREEZE). El caso real:
# c7b90ca3 creó su plan de 15 días el 17-sep, nunca abrió la Nevera y el plan quedó congelado el 19-sep con 12 días sin
# generar. El dueño (24-sep): «lo de la nevera opcional, si consideras que es lo mejor, hazlo».
#
#   · Apagada (por el usuario en Configuración, o por el sistema): el plan se genera SIN mirarla (sin validación estricta
#     ni pausas por Nevera), «Me lo comí» no descuenta, el coach no la usa y la lista de compras la pide entera.
#   · Automática (NULL) y vacía 48 h en modo plan: si el usuario sigue ACTIVO (`activo_reciente`), se apaga sola en vez
#     de congelar el plan — y si el plan ya estaba congelado por eso, se reanuda. Quien la ENCENDIÓ a mano conserva el
#     congelado: eligió cocinar con lo que tiene.
#   · El congelado era también el freno del gasto en cuentas abandonadas. Sin él, el freno es la inactividad: una cuenta
#     sin actividad en `MEALFIT_NEVERA_ACTIVE_DAYS` (14) no pierde la Nevera sola (se congela como siempre) ni recibe
#     rellenos con la Nevera apagada.
# Knob `MEALFIT_NEVERA_OFF_IN_PLAN_MODE` (True): en False, la regla vuelve a ser la del 23-sep (en modo plan, activa).


def apagable_en_modo_plan() -> bool:
    """tooltip-anchor: MEALFIT_NEVERA_OFF_IN_PLAN_MODE"""
    return _env_bool("MEALFIT_NEVERA_OFF_IN_PLAN_MODE", True)


def dias_actividad() -> int:
    """Ventana de «usuario activo». tooltip-anchor: MEALFIT_NEVERA_ACTIVE_DAYS"""
    return _env_int("MEALFIT_NEVERA_ACTIVE_DAYS", 14, validator=lambda v: 1 <= v <= 90)


# [SUPLEMENTOS-OK: tener suplementos es usar la Nevera]
_SQL_ULTIMA_ACTIVIDAD = """
SELECT GREATEST(
    (SELECT max(created_at) FROM agent_sessions WHERE user_id = %s),
    (SELECT max(created_at) FROM consumed_meals WHERE user_id = %s),
    (SELECT max(updated_at) FROM user_inventory WHERE user_id = %s),
    (SELECT max(created_at) FROM meal_plans WHERE user_id = %s)
) AS m
"""


def ultima_actividad(user_id: Optional[str]):
    """Lo último que el usuario HIZO en la app: chatear, registrar una comida, tocar su Nevera o crear un plan. None si
    nada, invitado o error. (No `api_usage`: también la escriben procesos en segundo plano.)"""
    if not user_id or user_id == "guest":
        return None
    try:
        row = execute_sql_query(_SQL_ULTIMA_ACTIVIDAD, (user_id, user_id, user_id, user_id), fetch_one=True) or {}
        return row.get("m")
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-217] ultima_actividad({user_id}) sin leer: {e}")
        return None


def activo_reciente(user_id: Optional[str], dias: Optional[int] = None) -> bool:
    """¿Hizo algo en la app en los últimos `dias` (knob)? Sin dato ⇒ False: sin evidencia de uso no se gasta ni se
    decide por él."""
    m = ultima_actividad(user_id)
    if m is None:
        return False
    from datetime import datetime, timedelta, timezone
    try:
        if getattr(m, "tzinfo", None) is None:
            m = m.replace(tzinfo=timezone.utc)
        return m >= datetime.now(timezone.utc) - timedelta(days=int(dias or dias_actividad()))
    except Exception:
        return False


def apagar_por_plan_vacio(user_id: str) -> bool:
    """El apagado automático en modo plan (lo decide el barrido del congelado, que ya mide las 48 h vacías con la gracia
    del plan). Solo si el usuario nunca eligió (NULL): encenderla a mano es definitivo. Filtra por id (I2)."""
    if not interruptor_disponible() or not _auto_apagado_encendido() or not apagable_en_modo_plan():
        return False
    try:
        res = execute_sql_write(
            "UPDATE user_profiles SET nevera_enabled = FALSE, nevera_auto_off_at = now() "
            "WHERE id = %s AND nevera_enabled IS NULL AND COALESCE(plan_mode, 'plan') <> 'tracking' RETURNING id",
            (user_id,), returning=True,
        )
        if res:
            logger.info(f"[P1-PLAN-LOTE-217] Nevera apagada sola en modo plan (vacía 48 h, usuario activo): {user_id}")
        return bool(res)
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-217] apagar_por_plan_vacio({user_id}): {e}")
        return False
