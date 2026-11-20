# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1117 · 2026-11-20] Rango «probador»: margen holgado para quienes hacen la prueba cerrada de 14 días
de Google Play.

Pedido del dueño (20-nov): «sube el margen a Sherwin y a los 10 últimos que inicien sesión en Android… haran el
testeo de los 14 días para la Play Store, colócalos en un rango especial». Medido ese día: solo 5 cuentas habían
entrado alguna vez por Android (todas `gratis`), así que el rango arranca con esas 5 y se llena SOLO con las
siguientes que entren por Android hasta `CUPO` (Sherwin + 10), mientras la inscripción siga abierta.

No es un plan: `plan_tier` (lo de PayPal) y los regalos no se tocan, la página de planes no lo enseña. Solo cambia
los topes de `topes_por_nivel` (fila `probador`) y los cupos mensuales de coach y generación de `auth`, y caduca solo.
Un nivel de pago MAYOR (Plus, Max, admin) manda sobre el rango: ser probador nunca baja un tope.

Estado: una clave por cuenta en `app_kv_store` → `probador:<uid>` = {"hasta", "origen", "alta"}. Se borra con la
cuenta (prefijo en `db_profiles`). Knobs: `MEALFIT_RANGO_PROBADOR` (off apaga todo), `MEALFIT_PROBADOR_CUPO`,
`MEALFIT_PROBADOR_INSCRIPCION_HASTA` (fecha ISO), `MEALFIT_PROBADOR_DIAS` (duración de cada alta automática).
Fail-open hacia lo pagado: si el KV no se lee, la cuenta queda con su nivel normal. tooltip-anchor: P1-PLAN-LOTE-1117
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger("mealfit")

PREFIJO = "probador:"
NIVEL = "probador"
CUPO = 11
INSCRIPCION_HASTA = "2026-12-10"     # tras esto ya no entra nadie solo; los de dentro siguen hasta su `hasta`
DIAS = 30                            # 14 días de prueba de Google + revisión + margen
_ES_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_CACHE: dict = {}
_CACHE_S = 120.0
_LOCK = threading.Lock()


def activo() -> bool:
    return str(os.environ.get("MEALFIT_RANGO_PROBADOR", "on")).strip().lower() not in ("off", "0", "false", "no")


def _cupo() -> int:
    try:
        return max(0, int(os.environ.get("MEALFIT_PROBADOR_CUPO", CUPO)))
    except ValueError:
        return CUPO


def _dias() -> int:
    try:
        return max(1, int(os.environ.get("MEALFIT_PROBADOR_DIAS", DIAS)))
    except ValueError:
        return DIAS


def _inscripcion_abierta(ahora: Optional[datetime] = None) -> bool:
    ahora = ahora or datetime.now(timezone.utc)
    crudo = str(os.environ.get("MEALFIT_PROBADOR_INSCRIPCION_HASTA", INSCRIPCION_HASTA)).strip()
    try:
        tope = datetime.fromisoformat(crudo)
    except ValueError:
        return False
    if tope.tzinfo is None:
        tope = tope.replace(tzinfo=timezone.utc)
    return ahora < tope


def _vigente(valor, ahora: Optional[datetime] = None) -> bool:
    if not isinstance(valor, dict):
        return False
    try:
        hasta = datetime.fromisoformat(str(valor.get("hasta")))
    except ValueError:
        return False
    if hasta.tzinfo is None:
        hasta = hasta.replace(tzinfo=timezone.utc)
    return (ahora or datetime.now(timezone.utc)) < hasta


def es_probador(user_id) -> bool:
    uid = str(user_id or "")
    if not activo() or not _ES_UUID.match(uid) or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    ahora = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(uid)
        if hit and ahora - hit[0] < _CACHE_S:
            return hit[1]
    try:
        from db_core import execute_sql_query
        fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (PREFIJO + uid,), fetch_one=True)
        si = _vigente((fila or {}).get("value"))
    except Exception as e:                                             # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1117] rango ilegible: {type(e).__name__}")
        return False
    with _LOCK:
        if len(_CACHE) > 5000:
            _CACHE.clear()
        _CACHE[uid] = (ahora, si)
    return si


def inscribir_si_android(user_id, plataforma) -> bool:
    """La app nativa de Android informó su dispositivo: si queda cupo y la inscripción sigue abierta, entra al rango.
    Atómico (el INSERT cuenta los vigentes en la misma sentencia). Nunca lanza. True si la inscribió ahora."""
    uid = str(user_id or "")
    if (plataforma != "android" or not activo() or not _ES_UUID.match(uid) or not _inscripcion_abierta()
            or os.environ.get("PYTEST_CURRENT_TEST")):
        return False
    hasta = (datetime.now(timezone.utc) + timedelta(days=_dias())).isoformat(timespec="seconds")
    try:
        from db_core import execute_sql_write
        filas = execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) "
            "SELECT %s, jsonb_build_object('hasta', %s::text, 'origen', 'android_auto', 'alta', NOW()), NOW() "
            "WHERE (SELECT count(*) FROM app_kv_store WHERE key LIKE 'probador:%%' "
            "       AND (value->>'hasta')::timestamptz > NOW()) < %s "
            "ON CONFLICT (key) DO NOTHING RETURNING key",
            (PREFIJO + uid, hasta, _cupo()), returning=True)
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-1117] alta automática de probador fallida ({uid[:8]}): {e!r}")
        return False
    if filas:
        with _LOCK:
            _CACHE.pop(uid, None)
        logger.info(f"[P1-PLAN-LOTE-1117] probador inscrito por Android: {uid[:8]} hasta {hasta}")
        return True
    return False


# entre Plus y Max: cada tope de la fila `probador` es ≥ al de Plus y ≤ al de Max
_RANGO = {"gratis": 0, "basic": 1, "plus": 2, NIVEL: 3, "ultra": 4, "admin": 9}


def nivel_con_rango(user_id, nivel: str) -> str:
    """El nivel que manda para los topes: `probador` si la cuenta está en el rango y su nivel pagado es menor."""
    if _RANGO.get(nivel, 0) < _RANGO[NIVEL] and es_probador(user_id):
        return NIVEL
    return nivel
