# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · 2026-09-29] El permiso para la IA de terceros: leerlo, darlo, retirarlo y el del invitado.

    GET  /api/consents            el estado de la cuenta (la misma forma que `profile.ai_consent` de GET /api/profile)
    POST /api/consents            conceder la IA (las DOS claves a true) y/o anotar la analítica
    POST /api/consents/withdraw   retirar la IA: bandera primero, después la pausa del generador (plan_mode)
    POST /api/consents/guest      el permiso del invitado, con sha256(session_id); en sus llamadas va la cabecera

Exentos de la cuota, como `PATCH /api/profile`: leer o decidir tu permiso no cuesta un crédito, y un 402 aquí dejaría
a alguien sin poder retirarlo. Anti-abuso con cupos propios. Lógica y SQL: `consentimientos.py` (SSOT); contrato para
el frontend: `docs/consentimiento_ia.md`. Los errores tienen cuerpo plano `{error_code, version, detail}`.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

import consentimientos as cs
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consents", tags=["consents"])

_CONSENTS_READ_LIMITER = RateLimiter(max_calls=30, period_seconds=60)
_CONSENTS_WRITE_LIMITER = RateLimiter(max_calls=10, period_seconds=60)


def _cuenta(verified_user_id: Optional[str]) -> str:
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Inicia sesión para ver o cambiar tu permiso.")
    return str(verified_user_id)


def _no_disponible(e: Exception, que: str) -> cs.ErrorDeConsentimiento:
    logger.error(f"❌ [P1-PLAN-LOTE-843] {que} falló: {type(e).__name__}: {e}")
    return cs.ErrorDeConsentimiento(503, "ai_consent_unavailable",
                                    "No pudimos guardar o leer tu permiso ahora mismo. Inténtalo en unos segundos.")


@router.get("")
def api_consents_estado(verified_user_id: Optional[str] = Depends(_CONSENTS_READ_LIMITER)):
    """El estado de la cuenta. Sin permiso: `vigente: false` y todo lo demás a null."""
    uid = _cuenta(verified_user_id)
    try:
        return cs.estado(uid)
    except Exception as e:  # noqa: BLE001
        raise _no_disponible(e, "leer el permiso")


@router.post("")
def api_consents_conceder(data: dict = Body(...), verified_user_id: Optional[str] = Depends(_CONSENTS_WRITE_LIMITER)):
    """`{version, ai_processing, ai_transfer_cn, analytics?, locale?, platform?, app_build?, text_sha256?}`."""
    uid = _cuenta(verified_user_id)
    peticion = cs.validar_peticion(data)
    try:
        return cs.registrar(uid, **peticion)
    except cs.PerfilInexistente:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    except Exception as e:  # noqa: BLE001
        raise _no_disponible(e, "anotar el permiso")


@router.post("/withdraw")
def api_consents_retirar(data: Optional[dict] = Body(None),
                         verified_user_id: Optional[str] = Depends(_CONSENTS_WRITE_LIMITER)):
    """Retira el permiso de IA. `{locale?, platform?, app_build?}` opcional (el registro dice desde dónde)."""
    uid = _cuenta(verified_user_id)
    datos = data if isinstance(data, dict) else {}
    contexto = cs.validar_contexto(datos)
    try:
        return cs.retirar(uid, **contexto)
    except cs.PerfilInexistente:
        raise HTTPException(status_code=404, detail="Perfil no encontrado.")
    except Exception as e:  # noqa: BLE001
        raise _no_disponible(e, "retirar el permiso")


@router.post("/guest")
def api_consents_invitado(data: dict = Body(...), _rl: Optional[str] = Depends(_CONSENTS_WRITE_LIMITER)):
    """El permiso del invitado: el mismo cuerpo que `POST /api/consents` más `session_id` (el suyo, el de
    `mealfit_guest_session_id`). Se guarda sha256(session_id), nunca el id."""
    peticion = cs.validar_peticion(data)
    if cs.hash_de_sesion((data or {}).get("session_id")) is None:
        raise cs.ErrorDeConsentimiento(422, "ai_consent_invalid_session", "Falta el identificador de la sesión.")
    try:
        return cs.registrar_invitado(data.get("session_id"), **peticion)
    except Exception as e:  # noqa: BLE001
        raise _no_disponible(e, "anotar el permiso del invitado")
