# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1032 · 2026-11-17] Informe para el médico: endpoints.

- `POST   /api/reports/health/{user_id}`        cuerpo `{dias: 14|30|90, con_redaccion: bool, tzOffset?, locale?}`
- `GET    /api/reports/health/{user_id}`        los 20 más recientes (sin contenido)
- `GET    /api/reports/health/{user_id}/{id}`   uno, con datos y redacción
- `DELETE /api/reports/health/{user_id}/{id}`   lo borra

Mismo patrón que `routers/diary.py`: `get_verified_user_id`, 403 si no es su `user_id`, invitado ⇒ 401 (un invitado no
tiene diario en el servidor). Tope: 3 por persona y día ⇒ 429. La redacción exige el permiso de IA de terceros
(`hay_permiso_ia`): sin él, el informe se crea igual, `data_only`, con `motivo: 'sin_permiso'`.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from auth import get_verified_user_id
from consentimientos import hay_permiso_ia
from error_utils import safe_error_detail
from path_validators import assert_valid_uuid
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])

_CREAR_LIMITER = RateLimiter(max_calls=4, period_seconds=61)   # par propio: Redis comparte la ventana por par (lote 945)
_LEER_LIMITER = RateLimiter(max_calls=40, period_seconds=61)

MENSAJE_TOPE = "Ya generaste 3 informes hoy. Podrás generar otro mañana."


class CrearInforme(BaseModel):
    dias: int = 30
    con_redaccion: bool = True
    tzOffset: Optional[int] = None
    locale: Optional[str] = None


def _propio(user_id: str, verified_user_id: Optional[str]) -> None:
    assert_valid_uuid(user_id, allow_guest=True)
    if not user_id or user_id == "guest" or not verified_user_id:
        raise HTTPException(status_code=401, detail="Inicia sesión para generar tu informe.")
    if verified_user_id != user_id:
        raise HTTPException(status_code=403, detail="Prohibido.")


@router.post("/health/{user_id}")
def api_crear_informe(
    user_id: str,
    body: CrearInforme = Body(default_factory=CrearInforme),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    permiso_ia: bool = Depends(hay_permiso_ia),
    _rl: None = Depends(_CREAR_LIMITER),
):
    _propio(user_id, verified_user_id)
    from informe_servicio import TopeDiario, crear_informe
    try:
        return crear_informe(user_id, dias=body.dias, tz_offset=body.tzOffset, locale=body.locale,
                             con_redaccion=bool(body.con_redaccion), con_permiso=permiso_ia is not False, origin="app")
    except TopeDiario:
        raise HTTPException(status_code=429, detail=MENSAJE_TOPE)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [LOTE-1032] crear informe falló: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/health/{user_id}")
def api_listar_informes(
    user_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_LEER_LIMITER),
):
    _propio(user_id, verified_user_id)
    from informe_servicio import listar
    try:
        return {"informes": listar(user_id)}
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [LOTE-1032] listar informes falló: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/health/{user_id}/{informe_id}")
def api_leer_informe(
    user_id: str,
    informe_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_LEER_LIMITER),
):
    _propio(user_id, verified_user_id)
    assert_valid_uuid(informe_id)
    from informe_servicio import leer
    inf = leer(user_id, informe_id)
    if not inf:
        raise HTTPException(status_code=404, detail="Ese informe no existe.")
    return inf


@router.delete("/health/{user_id}/{informe_id}")
def api_borrar_informe(
    user_id: str,
    informe_id: str,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_LEER_LIMITER),
):
    _propio(user_id, verified_user_id)
    assert_valid_uuid(informe_id)
    from informe_servicio import borrar
    if not borrar(user_id, informe_id):
        raise HTTPException(status_code=404, detail="Ese informe no existe.")
    return {"ok": True}
