# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1166 · 2026-11-22] Eliminar cuentas EN LOTE desde /admin → Cuentas.

El dueño, con la barra de selección del panel: «haz que esto funcione para seleccionar cuentas para editar, incluso
poder eliminar». Borrar es IRREVERSIBLE y el 20-nov se borraron por error 20 cuentas «Prueba» con gente activa (hubo
que restaurar un snapshot). Por eso el lote va en DOS pasos y con frenos que no dependen de la buena vista de nadie:

1. `revisar(ids)` (sin efectos): por cuenta, su correo, su última actividad (la MISMA que la lista: comida, mensaje al
   coach, escaneo, peso o agua), lo que hizo en 30 días y su bloqueo si lo tiene.
2. `eliminar(...)`:
   · tope `MAX_LOTE` (25) y confirmación escrita `ELIMINAR <n>` (el número de cuentas que se van);
   · NUNCA borra una cuenta admin ni una con suscripción de pago viva (cancelarla es otro flujo, con PayPal);
   · una cuenta con actividad en los últimos `DIAS_ACTIVA` días solo se borra si su id viene TAMBIÉN en
     `confirmar_activas` (la casilla que el admin marca a mano en el diálogo, cuenta por cuenta);
   · cada borrado deja su fila en `admin_access_log` ANTES de borrar (sin rastro no se borra) y usa el MISMO motor que
     «Eliminar mi cuenta» (`db_profiles.delete_account_data` + revocar Sign in with Apple).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Iterable

from admin_acceso import es_admin, registrar_acceso
from db_core import execute_sql_query

logger = logging.getLogger(__name__)

MAX_LOTE = 25
DIAS_ACTIVA = 14
_SUSCRIPCION_MUERTA = ("CANCELLED", "INACTIVE", "EXPIRED", "SUSPENDED", "APPROVAL_PENDING", "")


class ErrorEliminar(Exception):
    def __init__(self, status: int, detalle: str):
        super().__init__(detalle)
        self.status = status
        self.detalle = detalle


def _ids_validos(user_ids: Iterable[Any]) -> list[str]:
    vistos: list[str] = []
    for v in user_ids or []:
        try:
            u = str(uuid.UUID(str(v)))
        except (TypeError, ValueError):
            continue
        if u not in vistos:
            vistos.append(u)
    return vistos


def _filas(ids: list[str]) -> dict[str, dict]:
    import admin_cuentas_lista as acl
    sql = acl._sql_filas("p.id = ANY(%s::uuid[])") + (
        "SELECT filas.user_id, filas.email, filas.nombre, filas.ultima, filas.comidas_total, filas.comidas_30d, "
        "filas.mensajes_coach, filas.planes, filas.prueba_desde, "
        f"(filas.ultima IS NOT NULL AND filas.ultima >= now() - interval '{DIAS_ACTIVA} days') AS activa, "
        "pp.paypal_subscription_id, pp.subscription_status "
        "FROM filas JOIN public.user_profiles pp ON pp.id::text = filas.user_id")
    filas = execute_sql_query(sql, (ids,), fetch_all=True) or []
    return {str(f["user_id"]): f for f in filas}


def _bloqueo(uid: str, f: dict | None) -> str | None:
    if not f:
        return "no_existe"
    if es_admin(uid):
        return "es_admin"
    estado = str(f.get("subscription_status") or "").upper()
    if f.get("paypal_subscription_id") and estado not in _SUSCRIPCION_MUERTA:
        return "suscripcion_activa"
    return None


def revisar(user_ids: Iterable[Any]) -> list[dict]:
    """Paso 1, sin efectos: lo que hay que saber de cada cuenta antes de borrarla."""
    ids = _ids_validos(user_ids)
    if not ids:
        return []
    if len(ids) > MAX_LOTE:
        raise ErrorEliminar(422, "demasiadas")
    filas = _filas(ids)
    out = []
    for uid in ids:
        f = filas.get(uid)
        ultima = f.get("ultima") if f else None
        out.append({
            "user_id": uid,
            "email": (f or {}).get("email"),
            "nombre": (f or {}).get("nombre"),
            "ultima_actividad": ultima.isoformat() if hasattr(ultima, "isoformat") else ultima,
            "comidas_total": int((f or {}).get("comidas_total") or 0),
            "comidas_30d": int((f or {}).get("comidas_30d") or 0),
            "mensajes_coach": int((f or {}).get("mensajes_coach") or 0),
            "planes": int((f or {}).get("planes") or 0),
            "prueba": bool((f or {}).get("prueba_desde")),
            "activa": bool((f or {}).get("activa")),
            "bloqueo": _bloqueo(uid, f),
        })
    return out


def _borrar_una(uid: str) -> dict:
    from apple_tokens import revocar_con_plazo
    from db_profiles import delete_account_data
    try:
        asyncio.run(revocar_con_plazo(uid))
    except Exception as e:  # noqa: BLE001 — best-effort, igual que «Eliminar mi cuenta»
        logger.warning(f"[P1-PLAN-LOTE-1166] revocar Apple de {uid}: {type(e).__name__}")
    return delete_account_data(uid, True)


def eliminar(admin_id: str, user_ids: Iterable[Any], confirmacion: str, confirmar_activas: Iterable[Any] = ()) -> list[dict]:
    """Paso 2. Devuelve `{user_id, email, resultado}` con `eliminada | parcial | <bloqueo> | activa_sin_confirmar`."""
    ids = _ids_validos(user_ids)
    if not ids:
        raise ErrorEliminar(422, "sin_cuentas")
    if len(ids) > MAX_LOTE:
        raise ErrorEliminar(422, "demasiadas")
    if " ".join(str(confirmacion or "").split()).upper() != f"ELIMINAR {len(ids)}":
        raise ErrorEliminar(422, "confirmacion")
    activas_ok = set(_ids_validos(confirmar_activas))
    resultados = []
    for r in revisar(ids):
        uid = r["user_id"]
        if r["bloqueo"]:
            resultados.append({"user_id": uid, "email": r["email"], "resultado": r["bloqueo"]})
            continue
        if r["activa"] and uid not in activas_ok:
            resultados.append({"user_id": uid, "email": r["email"], "resultado": "activa_sin_confirmar"})
            continue
        # sin rastro no se borra: si no se puede anotar, el lote se corta aquí (lo ya borrado ya tiene su fila)
        try:
            registrar_acceso(admin_id, "eliminar_cuenta", uid, {
                "lote": len(ids), "activa": r["activa"], "comidas_total": r["comidas_total"], "prueba": r["prueba"]})
        except Exception as e:  # noqa: BLE001
            logger.error(f"[P1-PLAN-LOTE-1166] rastro no anotado para {uid}: {e!r}")
            raise ErrorEliminar(503, "sin_rastro") from e
        try:
            res = _borrar_una(uid)
            fallos = list(res.get("failed_steps") or [])
            resultado = "eliminada" if not fallos and not res.get("errors") else "parcial"
        except Exception as e:  # noqa: BLE001
            logger.error(f"[P1-PLAN-LOTE-1166] borrar {uid}: {e!r}")
            resultado = "parcial"
        if resultado != "eliminada":
            try:
                registrar_acceso(admin_id, "eliminar_cuenta_fallo", uid, {"lote": len(ids)})
            except Exception:  # noqa: BLE001
                pass
        resultados.append({"user_id": uid, "email": r["email"], "resultado": resultado})
    logger.info(f"[P1-PLAN-LOTE-1166] lote de borrado por {admin_id}: "
                f"{sum(1 for x in resultados if x['resultado'] == 'eliminada')}/{len(ids)} eliminadas")
    return resultados
