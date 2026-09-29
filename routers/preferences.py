"""User preferences router.

[LONG-TERM-MEMORY-TOGGLE · 2026-05-13] Endpoint mínimo para que usuarios
Básico+ activen/desactiven la memoria a largo plazo desde Settings.

Contrato del flag (ver migración add_long_term_memory_enabled_2026_05_13.sql):
- TRUE  = chat.py extrae nuevos hechos + consulta user_facts en cada turn.
- FALSE = no extrae ni consulta (datos previos en BD intactos, reversible).
  [P1-PLAN-LOTE-717 · 2026-09-28] Cumplido de verdad en los dos caminos del coach (`agent.py`), la cola de pendientes y
  los textos libres de los paneles; ilegible ⇒ pausada. Lectura SSOT: `memoria_largo_plazo`.

Gate de tier: el endpoint NO bloquea por tier server-side intencionalmente
— un usuario gratis que toque el endpoint via DevTools puede setear el flag
pero no afecta nada (chat.py gateaa upstream por `is_plus`). La UI del toggle
solo aparece en Settings para usuarios Básico+; el server permanece neutral.

Auth: `get_verified_user_id` (sin `verify_api_quota` — no consume créditos).
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
import asyncio
import logging

from auth import get_verified_user_id
from db_profiles import (
    update_long_term_memory_enabled,
    update_water_tracker_enabled,
    get_water_tracker_enabled,
    update_ai_training_consent,
)

# [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] Los 4 handlers async de este router
# llamaban funciones DB síncronas (`get_user_profile`, `update_*`) sin envolver
# en `asyncio.to_thread`, bloqueando el event loop ~10-200ms por roundtrip
# a la DB. Mismo modo de fallo que P2-AUTH-ASYNC-SLEEP cerró para `auth.py`:
# bajo carga concurrente (≥50 req/s), throttling severo de TODOS los demás
# handlers async (chat stream, webhook PayPal, diary upload). Ahora cada call
# DB pasa por `await asyncio.to_thread(...)` — el event loop sirve otras
# requests mientras la DB responde. Tooltip-anchor: P1-ASYNC-SYNC-DB-BLOCKING.

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/user/preferences",
    tags=["user-preferences"],
)

# [P1-PLAN-LOTE-717 · 2026-09-28] Código ESTABLE del 503 de las lecturas de Configuración. Cuando la base no responde,
# los GET inventaban un valor y contestaban 200 (memoria → activada, entrenamiento → sin consentimiento, Nevera →
# activa): la pantalla pintaba un interruptor que no reflejaba nada y un toque «corregía» un valor que nunca se leyó.
# Ahora 503 con este `detail`, y la UI dice «no pudimos cargar — Reintentar». Sin sesión sigue siendo 401.
PREFERENCIA_NO_DISPONIBLE = "preference_unavailable"


def _no_disponible(que: str, user_id: str, error: Exception) -> HTTPException:
    logger.warning(f"[P1-PLAN-LOTE-717] preferencia '{que}' de {user_id} ilegible → 503: {error}")
    return HTTPException(status_code=503, detail=PREFERENCIA_NO_DISPONIBLE)


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

    ok = await asyncio.to_thread(
        update_long_term_memory_enabled, verified_user_id, body.long_term_memory_enabled
    )
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

    [P1-PLAN-LOTE-717 · 2026-09-28] Si la base no responde: 503 `preference_unavailable`, nunca un «activada»
    inventado (`get_user_profile` devolvía None tanto sin fila como con la base caída). Lectura SSOT:
    `memoria_largo_plazo.leer_memoria`.
    """
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    from memoria_largo_plazo import MemoriaIlegible, leer_memoria
    try:
        enabled = await asyncio.to_thread(leer_memoria, verified_user_id)
    except MemoriaIlegible as e:
        raise _no_disponible("memoria", verified_user_id, e)

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

    ok = await asyncio.to_thread(
        update_water_tracker_enabled, verified_user_id, body.water_tracker_enabled
    )
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
    # [P1-PLAN-LOTE-135] Encenderla pone a cero la cuenta de avisos ignorados: sin esto, quien la reactiva tras un
    # apagado automático la vería apagarse otra vez en el siguiente tick del cron.
    if body.water_tracker_enabled:
        import hydration_reminders
        await asyncio.to_thread(hydration_reminders.al_encender, verified_user_id)
    return {"water_tracker_enabled": body.water_tracker_enabled}


@router.get("/water-tracker")
async def api_get_water_tracker_enabled(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el valor actual del flag. Default TRUE si perfil ausente
    o campo NULL (defensa contra perfiles legacy pre-migracion)."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    enabled = await asyncio.to_thread(get_water_tracker_enabled, verified_user_id)
    return {"water_tracker_enabled": enabled}


# [P1-NEVERA-OPCIONAL · 2026-09-23] Interruptor de la Nevera (Configuración → Capacidades; en los dos modos desde
# P1-PLAN-LOTE-217).
# La regla y el apagado automático viven en nevera_opcional.py; aquí solo la elección explícita del usuario.
# Cero LLM ⇒ `get_verified_user_id`, nunca `verify_api_quota` (mismo criterio que water-tracker).


class NeveraPreferenceBody(BaseModel):
    enabled: bool


@router.get("/nevera")
async def api_get_nevera(verified_user_id: str = Depends(get_verified_user_id)):
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    import nevera_opcional
    try:
        return await asyncio.to_thread(nevera_opcional.estado_nevera, verified_user_id)
    except nevera_opcional.NeveraIlegible as e:   # [P1-PLAN-LOTE-717] 503, no una Nevera «activa» inventada
        raise _no_disponible("nevera", verified_user_id, e)


@router.patch("/nevera")
async def api_set_nevera(
    body: NeveraPreferenceBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    import nevera_opcional
    if not nevera_opcional.interruptor_disponible():
        raise HTTPException(status_code=409, detail="La opción de apagar la Nevera no está disponible.")
    ok = await asyncio.to_thread(nevera_opcional.fijar_nevera, verified_user_id, body.enabled)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo actualizar la preferencia.")
    logger.info(f"[P1-NEVERA-OPCIONAL] user={verified_user_id} nevera_enabled={body.enabled}")
    # [P1-PLAN-LOTE-217] apagarla en modo plan descongela el plan que esperaba a que se llenara
    if not body.enabled:
        try:
            from cron_tasks import try_unfreeze_plan_for_user
            await asyncio.to_thread(try_unfreeze_plan_for_user, verified_user_id)
        except Exception as _uf_e:
            logger.debug(f"[P1-PLAN-LOTE-217] descongelar tras apagar la Nevera: no-op ({_uf_e})")
    # [P1-PLAN-LOTE-717] La respuesta es el estado RELEÍDO (la UI avisa con él de lo que el servidor aplicó). Si la
    # relectura falla, 503: la elección ya se guardó y reintentar es idempotente; inventar `activa` haría mentir al aviso.
    try:
        return await asyncio.to_thread(nevera_opcional.estado_nevera, verified_user_id)
    except nevera_opcional.NeveraIlegible as e:
        raise _no_disponible("nevera", verified_user_id, e)


# [P2-AI-TRAINING-CONSENT · 2026-07-04] Consentimiento OPT-IN para uso futuro
# de datos en entrenamiento de modelos propios de MealfitRD (Configuración →
# Privacidad). DEFAULT FALSE fail-secure: perfil ausente / campo NULL / error
# = NO consiente. El corpus futuro DEBE filtrar por
# db_profiles.get_ai_training_consented_user_ids() (gate SSOT).
# Migración SSOT: migrations/p2_ai_training_consent_2026_07_04.sql.


class AiTrainingConsentBody(BaseModel):
    ai_training_consent: bool


@router.patch("/ai-training")
async def api_set_ai_training_consent(
    body: AiTrainingConsentBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Actualiza el consentimiento de training del usuario autenticado."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    ok = await asyncio.to_thread(
        update_ai_training_consent, verified_user_id, body.ai_training_consent
    )
    if not ok:
        logger.warning(
            f"[P2-AI-TRAINING-CONSENT] update falló para user={verified_user_id} "
            f"consent={body.ai_training_consent} (sin fila afectada)"
        )
        raise HTTPException(status_code=500, detail="No se pudo actualizar la preferencia.")

    logger.info(
        f"[P2-AI-TRAINING-CONSENT] user={verified_user_id} "
        f"ai_training_consent={body.ai_training_consent}"
    )
    return {"ai_training_consent": body.ai_training_consent}


@router.get("/ai-training")
async def api_get_ai_training_consent(
    verified_user_id: str = Depends(get_verified_user_id),
):
    """Devuelve el consentimiento actual. Default FALSE (opt-in fail-secure)
    si el perfil no existe o el campo es NULL (perfiles pre-migración).

    [P1-PLAN-LOTE-717 · 2026-09-28] FALSE es el valor por defecto de una fila sin decidir, no la respuesta a una base
    caída: si la lectura falla, 503 `preference_unavailable` (antes, 200 «no consientes» y la pantalla ofrecía
    «activar» sobre un valor que nunca se leyó). El corpus de entrenamiento sigue filtrando por
    `get_ai_training_consented_user_ids` (fail-secure), que esto no toca."""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")

    try:
        consent = await asyncio.to_thread(_leer_consentimiento_ia, verified_user_id)
    except Exception as e:
        raise _no_disponible("ai-training", verified_user_id, e)
    return {"ai_training_consent": consent}


def _leer_consentimiento_ia(user_id: str) -> bool:
    """[P1-PLAN-LOTE-717] Una columna, sin los efectos laterales de `get_user_profile` (que además devuelve None tanto
    sin fila como con la base caída). Lanza si la base falla; sin fila o NULL ⇒ False."""
    from db import execute_sql_query
    fila = execute_sql_query(
        "SELECT ai_training_consent FROM user_profiles WHERE id = %s",
        (user_id,), fetch_one=True,
    )
    return bool((fila or {}).get("ai_training_consent"))


# [P1-PLAN-LOTE-135 · 2026-09-20] La invitación «¿Quieres que la IA te arme el plan?» del contador: una vez por SEMANA
# y por USUARIO (el descarte vivía en el localStorage de cada dispositivo y volvía con cada binario o navegador nuevo).
# Cero LLM; `get_verified_user_id` como el resto del router. Motor y regla: plan_invite.py.


class PlanInviteBody(BaseModel):
    action: str


@router.get("/plan-invite")
async def api_get_plan_invite(
    verified_user_id: str = Depends(get_verified_user_id),
):
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    import plan_invite
    try:
        return await asyncio.to_thread(plan_invite.leer_invitacion, verified_user_id)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-135] invitación al plan de {verified_user_id} no leída: {e}")
        raise HTTPException(status_code=503, detail="No disponible.")


@router.patch("/plan-invite")
async def api_set_plan_invite(
    body: PlanInviteBody = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
):
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="No autenticado.")
    if body.action not in ("seen", "dismiss"):
        raise HTTPException(status_code=400, detail="`action` debe ser 'seen' o 'dismiss'.")
    import plan_invite
    try:
        return await asyncio.to_thread(plan_invite.anotar, verified_user_id, body.action)
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-135] invitación al plan de {verified_user_id} no anotada: {e}")
        raise HTTPException(status_code=503, detail="No disponible.")
