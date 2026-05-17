"""User preferences router.

[LONG-TERM-MEMORY-TOGGLE · 2026-05-13] Endpoint mínimo para que usuarios
Básico+ activen/desactiven la memoria a largo plazo desde Settings.

Contrato del flag (ver migración add_long_term_memory_enabled_2026_05_13.sql):
- TRUE  = chat.py extrae nuevos hechos + consulta user_facts en cada turn.
- FALSE = no extrae ni consulta (datos previos en BD intactos, reversible).

Gate de tier: el endpoint NO bloquea por tier server-side intencionalmente
— un usuario gratis que toque el endpoint via DevTools puede setear el flag
pero no afecta nada (chat.py gateaa upstream por `is_plus`). La UI del toggle
solo aparece en Settings para usuarios Básico+; el server permanece neutral.

Auth: `get_verified_user_id` (sin `verify_api_quota` — no consume créditos).
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
import logging

from auth import get_verified_user_id
from db_profiles import (
    get_user_profile,
    update_long_term_memory_enabled,
    update_water_tracker_enabled,
    get_water_tracker_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/user/preferences",
    tags=["user-preferences"],
)


class MemoryPreferenceBody(BaseModel):
    long_term_memory_enabled: bool


@router.patch("/memory")
async def api_set_long_term_memory(
    body: MemoryPreferenceBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Actualiza el toggle de memoria a largo plazo del usuario autenticado.

    Echo del nuevo valor en la respuesta para que el frontend confirme
    sin necesidad de re-fetch del perfil completo.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    ok = update_long_term_memory_enabled(verified_user_id, body.long_term_memory_enabled)
    if not ok:
        logger.warning(
            f"[LONG-TERM-MEMORY-TOGGLE] update falló para user={verified_user_id} "
            f"enabled={body.long_term_memory_enabled} (sin fila afectada)"
        )
        raise HTTPException(status_code=500, detail="No se pudo actualizar la preferencia.")

    logger.info(
        f"[LONG-TERM-MEMORY-TOGGLE] user={verified_user_id} "
        f"long_term_memory_enabled={body.long_term_memory_enabled}"
    )
    return {"long_term_memory_enabled": body.long_term_memory_enabled}


@router.get("/memory")
async def api_get_long_term_memory(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el valor actual del flag para el usuario autenticado.

    El frontend lo lee al cargar Settings para reflejar el estado del toggle.
    Default TRUE si el perfil no existe o el campo es NULL (defensa contra
    perfiles legacy creados antes de la migración).
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    profile = get_user_profile(verified_user_id)
    enabled = True
    if profile and "long_term_memory_enabled" in profile:
        enabled = bool(profile.get("long_term_memory_enabled", True))

    return {"long_term_memory_enabled": enabled}


# [P3-WATER-TRACKER · 2026-05-16] Toggle del card de hidratacion del Dashboard.
# Sin gate de tier: disponible para todos los usuarios autenticados (free incluidos).
# Default TRUE: el card aparece a menos que el usuario explicitamente lo apague.


class WaterTrackerPreferenceBody(BaseModel):
    water_tracker_enabled: bool


@router.patch("/water-tracker")
async def api_set_water_tracker_enabled(
    body: WaterTrackerPreferenceBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Actualiza el toggle del water tracker del usuario autenticado."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    ok = update_water_tracker_enabled(verified_user_id, body.water_tracker_enabled)
    if not ok:
        logger.warning(
            f"[P3-WATER-TRACKER] update fallo para user={verified_user_id} "
            f"enabled={body.water_tracker_enabled} (sin fila afectada)"
        )
        raise HTTPException(status_code=500, detail="No se pudo actualizar la preferencia.")

    logger.info(
        f"[P3-WATER-TRACKER] user={verified_user_id} "
        f"water_tracker_enabled={body.water_tracker_enabled}"
    )
    return {"water_tracker_enabled": body.water_tracker_enabled}


@router.get("/water-tracker")
async def api_get_water_tracker_enabled(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el valor actual del flag. Default TRUE si perfil ausente
    o campo NULL (defensa contra perfiles legacy pre-migracion)."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    return {"water_tracker_enabled": get_water_tracker_enabled(verified_user_id)}
