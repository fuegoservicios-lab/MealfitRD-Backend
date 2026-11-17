# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1059 · plan 1029/1035 · 2026-11-17] El juego de las semanas: endpoints.

- `GET  /api/progress/summary/{user_id}?tzOffset=`              resumen + eventos pendientes
- `GET  /api/progress/weeks/{user_id}?desde=&hasta=&tzOffset=`  semanas (tope 104)
- `POST /api/progress/events/seen/{user_id}`                     cuerpo `{claves: [...]}` ⇒ vistos

Mismo patrón que `routers/diary.py`: `get_verified_user_id`, 403 si no es su `user_id`, invitado ⇒ vacío. Nada de IA.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from auth import get_verified_user_id
from error_utils import safe_error_detail
from path_validators import assert_valid_uuid
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/progress", tags=["progress"])

_LEER_LIMITER = RateLimiter(max_calls=60, period_seconds=62)    # par propio (lote 945)
_VISTOS_LIMITER = RateLimiter(max_calls=20, period_seconds=62)


class Vistos(BaseModel):
    claves: List[str] = []


def _es_invitado(user_id: str, verified_user_id: Optional[str]) -> bool:
    assert_valid_uuid(user_id, allow_guest=True)
    if not user_id or user_id == "guest":
        return True
    if not verified_user_id or verified_user_id != user_id:
        raise HTTPException(status_code=403, detail="Prohibido.")
    return False


@router.get("/summary/{user_id}")
def api_resumen_progreso(
    user_id: str,
    tzOffset: Optional[int] = None,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_LEER_LIMITER),
):
    if _es_invitado(user_id, verified_user_id):
        return {}
    try:
        from progreso_semanas import resumen
        return resumen(user_id, tzOffset)
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [LOTE-1059] resumen de progreso falló: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/weeks/{user_id}")
def api_semanas(
    user_id: str,
    desde: Optional[int] = None,
    hasta: Optional[int] = None,
    tzOffset: Optional[int] = None,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_LEER_LIMITER),
):
    if _es_invitado(user_id, verified_user_id):
        return {"semanas": []}
    try:
        from progreso_semanas import semanas
        return {"semanas": semanas(user_id, tzOffset, desde, hasta)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [LOTE-1059] semanas de progreso fallaron: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/events/seen/{user_id}")
def api_eventos_vistos(
    user_id: str,
    body: Vistos = Body(default_factory=Vistos),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_VISTOS_LIMITER),
):
    if _es_invitado(user_id, verified_user_id):
        return {"ok": True, "marcados": 0}
    try:
        from progreso_semanas import marcar_vistos
        return {"ok": True, "marcados": marcar_vistos(user_id, (body.claves or [])[:50])}
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [LOTE-1059] marcar eventos vistos falló: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
