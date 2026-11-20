# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1115 · 2026-11-20] Quien deja de usar la app deja de gastar en planes; al volver, sigue solo.

Decisión del dueño (20-nov): «un sistema que pause los planes en determinado momento, y más para los gratuitos, para
ahorrar costos». Medido ese día: el cron `_background_refill_inactive_users` hacía justo lo contrario —cada 4 h buscaba
a quien llevaba ≥3 días sin abrir la app y le encolaba los días siguientes— y el bloque se generaba igual, sin que nadie
lo fuera a leer.

Cómo frena: un fragmento SQL CONSTANTE en el pickup de `process_plan_chunk_queue` (el único punto por el que pasa todo
el gasto LLM de bloques), igual que la pausa de P1-PLAN-MODE y el permiso de 843. NO cancela la cola: el bloque se queda
`pending` y, en cuanto el usuario vuelve (cualquier petición con sesión marca su presencia), el siguiente tick del
worker lo recoge. Reanudar no necesita código. El mismo fragmento va en la alerta de bloques «zombi», porque un bloque
que espera a su dueño no está atascado.

Inactivo = la última de: presencia (`presencia:<uid>`, la marca la autenticación, como mucho cada 30 min), mensaje
propio al coach, comida registrada o plan creado, más vieja que el umbral de su nivel (gratis 3 días, de pago 10).
Nunca frena el primer bloque de un plan (`initial_plan`: lo acaba de pedir) y, sin ningún dato, no frena (falla
abierto). Knobs: `MEALFIT_PAUSA_INACTIVIDAD` (off la apaga), `MEALFIT_PAUSA_INACTIVIDAD_DIAS_GRATIS`,
`MEALFIT_PAUSA_INACTIVIDAD_DIAS_PAGO`. tooltip-anchor: P1-PLAN-LOTE-1115-PAUSA
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

logger = logging.getLogger("mealfit")

PRESENCIA_PREFIJO = "presencia:"
_CADA_S = 30 * 60
_MARCAS: dict = {}
_MARCAS_MAX = 5000
_LOCK = threading.Lock()
_ALIAS_RX = re.compile(r"^[a-z][a-z0-9_]{0,15}$")


def activo() -> bool:
    return str(os.environ.get("MEALFIT_PAUSA_INACTIVIDAD", "on")).strip().lower() not in ("off", "0", "false", "no")


def _dias(nombre: str, defecto: int) -> int:
    try:
        v = int(os.environ.get(nombre, defecto))
    except (TypeError, ValueError):
        v = defecto
    return min(max(v, 1), 90)


def dias_gratis() -> int:
    return _dias("MEALFIT_PAUSA_INACTIVIDAD_DIAS_GRATIS", 3)


def dias_pago() -> int:
    return _dias("MEALFIT_PAUSA_INACTIVIDAD_DIAS_PAGO", 10)


def fragmento_sql(alias: str = "q1") -> str:
    """`AND …` para el WHERE de una consulta sobre `plan_chunk_queue <alias>`: deja fuera los bloques de quien lleva
    más del umbral de su nivel sin usar la app. '' con el knob en off. Sin entrada de usuario: alias validado, días
    enteros acotados."""
    if not activo():
        return ""
    if not _ALIAS_RX.match(alias or ""):
        raise ValueError("alias inválido")
    q = alias
    return f"""
                -- [P1-PLAN-LOTE-1115] quien no usa la app no gasta en bloques; al volver, el siguiente tick los recoge
                AND NOT (
                    COALESCE({q}.chunk_kind, '') <> 'initial_plan'
                    AND EXISTS (
                        SELECT 1 FROM user_profiles upi
                        WHERE upi.id = {q}.user_id
                          AND COALESCE(GREATEST(
                                (SELECT kvp.updated_at FROM app_kv_store kvp WHERE kvp.key = '{PRESENCIA_PREFIJO}' || {q}.user_id::text),
                                (SELECT max(cmi.created_at) FROM consumed_meals cmi WHERE cmi.user_id = {q}.user_id),
                                (SELECT max(ami.created_at) FROM agent_messages ami WHERE ami.user_id = {q}.user_id AND ami.role = 'user'),
                                (SELECT max(mpi.created_at) FROM meal_plans mpi WHERE mpi.user_id = {q}.user_id)
                              ), NOW())
                              < NOW() - make_interval(days => CASE WHEN COALESCE(upi.plan_tier, 'gratis') IN ('gratis', 'free', '')
                                                                   THEN {int(dias_gratis())} ELSE {int(dias_pago())} END)
                    )
                )"""


def _escribir(uid: str) -> None:
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, jsonb_build_object('at', NOW()), NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (PRESENCIA_PREFIJO + uid,),
        )
    except Exception as e:                                             # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1115] presencia sin anotar: {type(e).__name__}")


def marcar_presencia(uid) -> bool:
    """La app hizo una petición con sesión: anota la presencia (como mucho cada 30 min por usuario y proceso, en un
    hilo aparte: nunca frena la petición). True si se lanzó la escritura."""
    if not uid or uid == "guest" or not activo():
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") and not getattr(marcar_presencia, "_en_test", False):
        return False                       # la suite pega contra la base real: sin marcas de usuarios falsos

    ahora = time.monotonic()
    with _LOCK:
        if ahora - _MARCAS.get(uid, -1e9) < _CADA_S:
            return False
        if len(_MARCAS) >= _MARCAS_MAX:
            _MARCAS.clear()
        _MARCAS[uid] = ahora
    threading.Thread(target=_escribir, args=(str(uid),), daemon=True, name="presencia").start()
    return True
