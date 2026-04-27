from fastapi import APIRouter, Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from typing import Optional
import logging
import traceback
import os
import threading
import asyncio
import json as _json
import time as _time

# Importaciones relativas del entorno
from auth import get_verified_user_id, verify_api_quota
from db import (
    supabase, get_user_likes, get_active_rejections, get_or_create_session, 
    save_message, update_user_health_profile, log_api_usage, get_latest_meal_plan,
    get_latest_meal_plan_with_id, update_meal_plan_data, insert_like
)
from memory_manager import build_memory_context, summarize_and_prune
from agent import analyze_preferences_agent, swap_meal
from graph_orchestrator import run_plan_pipeline, arun_plan_pipeline
from ai_helpers import expand_recipe_agent
from services import _save_plan_and_track_background, _process_swap_rejection_background, save_partial_plan_get_id
from db_inventory import restock_inventory, consume_inventory_items_completely

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/plans",
    tags=["plans"],
)

# ─── TEMPORARY DEBUG ENDPOINT (REMOVE AFTER DIAGNOSIS) ───
@router.get("/debug-scaling/{user_id}")
def debug_scaling(user_id: str):
    """Temporary: compare shopping list output for household sizes 1-6."""
    from shopping_calculator import get_shopping_list_delta
    from db_plans import get_latest_meal_plan_with_id as _get_plan
    
    plan_record = _get_plan(user_id)
    
    # Fallback: try to find ANY recent plan if user_id yields nothing
    if not plan_record:
        try:
            from db_core import execute_sql_query
            row = execute_sql_query(
                "SELECT id, user_id, plan_data FROM meal_plans ORDER BY created_at DESC LIMIT 1",
                fetch_one=True
            )
            if row:
                plan_record = row
                user_id = row.get("user_id", user_id)
            else:
                return {"error": f"No plans exist in database at all"}
        except Exception as e:
            return {"error": f"No plan found for {user_id} and fallback failed: {e}"}
    
    if not plan_record:
        return {"error": f"No plan found for {user_id}"}
    
    plan_data = plan_record["plan_data"]
    days = plan_data.get("days", [])
    num_days = len(days)
    
    KEYWORDS = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'aguacate', 'arroz', 'pollo', 'cebolla', 'tomate', 'melón', 'melon']
    
    comparison = {}
    for h in [1, 2, 3, 4, 5, 6]:
        scaled = get_shopping_list_delta(user_id, plan_data, is_new_plan=True, structured=True, multiplier=float(h))
        row = {}
        for item in scaled:
            name = item.get("name", "")
            if any(kw in name.lower() for kw in KEYWORDS):
                row[name] = {
                    "display_qty": item.get("display_qty"),
                    "market_qty": item.get("market_qty"),
                    "market_unit": item.get("market_unit"),
                }
        comparison[f"{h}_personas"] = row
    
    return {
        "found_user_id": user_id,
        "plan_id": plan_record.get("id"),
        "num_days_in_plan": num_days,
        "base_duration_scale": round(7.0 / max(1, num_days), 4),
        "comparison": comparison
    }
# ─── END TEMPORARY DEBUG ENDPOINT ───

from constants import PLAN_CHUNK_SIZE, split_with_absorb

def _user_has_profile(user_id: str) -> bool:
    """Devuelve True si user_id tiene fila en user_profiles. Auto-crea fila mínima si falta."""
    if not user_id or not supabase:
        return False
    try:
        res = supabase.table("user_profiles").select("id").eq("id", user_id).limit(1).execute()
        if res.data:
            return True
        # Usuario autenticado sin perfil → crear fila mínima para habilitar chunking y FK
        supabase.table("user_profiles").upsert({"id": user_id, "health_profile": {}}).execute()
        import logging as _log
        _log.getLogger(__name__).info(f"✅ [PROFILE] Fila mínima creada en user_profiles para {user_id}")
        return True
    except Exception as e:
        import logging as _log
        _log.getLogger(__name__).warning(f"⚠️ [PROFILE] No se pudo crear user_profiles para {user_id}: {e}")
        return False

def chunk_size_for_next_slot(days_since_creation: int, total_planned_days: int, base: int = 3):
    chunks = split_with_absorb(total_planned_days, base)
    consumed = 0
    for c in chunks:
        if days_since_creation < consumed + c:
            return c
        consumed += c
    return base

@router.post("/shift-plan")
def api_shift_plan(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """
    On-demand endpoint to trigger an atomic shift + rolling window generation.
    Idempotent: if the plan is already up-to-date, does nothing.
    """
    from db_core import execute_sql_write, execute_sql_query
    from datetime import datetime, timezone, timedelta
    import copy, random, json
    
    try:
        user_id = data.get("user_id")
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")

        # P0-2 FIX: Get latest plan using FOR UPDATE to prevent race conditions with chunk workers doing blind overwrites
        from db_core import connection_pool
        from psycopg.rows import dict_row
        
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "SELECT id, plan_data FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                        (user_id,)
                    )
                    plan_record = cursor.fetchone()
                    if not plan_record:
                        return {"success": False, "message": "No hay plan activo."}

                    plan_id = plan_record["id"]
                    plan_data = plan_record.get("plan_data", {})
                    days = plan_data.get("days", [])
                    total_planned_days = max(3, int(plan_data.get("total_days_requested", len(days))))
                    
                    if len(days) == 0:
                        return {"success": False, "message": "El plan está vacío."}

                    # Check if shift is needed
                    tz_offset = data.get('tzOffset', 0)
                    today = datetime.now(timezone.utc)
                    if tz_offset:
                        try:
                            today -= timedelta(minutes=int(tz_offset))
                        except (ValueError, TypeError):
                            pass

                    # Parse grocery_start_date to find actual day index
                    start_date_str = plan_data.get("grocery_start_date")
                    if not start_date_str:
                        return {"success": False, "message": "Falta fecha de inicio."}
                        
                    from constants import safe_fromisoformat
                    try:
                        start_dt = safe_fromisoformat(start_date_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        else:
                            start_dt = start_dt.astimezone(timezone.utc)
                        start_dt = start_dt - timedelta(minutes=int(tz_offset))
                        
                        # Remove time component
                        today_date = today.date()
                        start_date = start_dt.date()
                        days_since_creation = (today_date - start_date).days
                    except Exception as e:
                        return {"success": False, "message": f"Error parseando fecha: {e}"}

                    # Cuántos días del plan total quedan por vivir
                    days_remaining_in_plan = max(0, total_planned_days - days_since_creation)
                    
                    # Bloque dinámico P0-2: window_size depende de la posición en la distribución
                    window_size = chunk_size_for_next_slot(max(0, days_since_creation), total_planned_days, PLAN_CHUNK_SIZE)
                    
                    # La ventana necesita min(window_size, días restantes) días; si el plan expiró no necesita nada
                    window_needed = min(window_size, days_remaining_in_plan)

                    needs_shift = days_since_creation > 0
                    needs_fill_initial = len(days) < window_needed  # Para la guard de cortocircuito inicial

                    if not needs_shift and not needs_fill_initial:
                        return {"success": True, "message": "Plan ya est\u00e1 al d\u00eda y completo.", "plan_data": plan_data}

                    logger.info(f"\ud83d\udd04 [API SHIFT] Shifting {days_since_creation} días. Plan total={total_planned_days}, restantes={days_remaining_in_plan}")

                    shifted_data = copy.deepcopy(plan_data)
                    shifted_days = shifted_data.get('days', [])

                    # 1. Atomic Shift (in-memory, saved at the end within transaction)
                    if needs_shift:
                        shift_amount = min(days_since_creation, len(shifted_days))
                        shifted_days = shifted_days[shift_amount:]

                    # 2. Update day names AND renumber days 1..N (requerido para continuidad del chunk worker)
                    dias_es = ["Lunes", "Martes", "Mi\u00e9rcoles", "Jueves", "Viernes", "S\u00e1bado", "Domingo"]
                    for i, day_obj in enumerate(shifted_days):
                        target_date = today + timedelta(days=i)
                        day_obj['day_name'] = dias_es[target_date.weekday()]
                        day_obj['day'] = i + 1  # Renumerar desde 1 para mantener secuencia 1..N

                    # 3. Rolling window: si el plan no ha expirado y la ventana actual tiene menos de window_size días,
                    #    y no hay ya un chunk de IA en camino, encolar generación IA real (aprendizaje continuo).
                    modified = needs_shift
                    is_partial = plan_data.get('generation_status') in ('partial', 'generating_next')
                    needs_fill = len(shifted_days) < window_needed and days_remaining_in_plan > 0

                    # [P0-4 FIX] Antes: disable_rolling_refill_for_active_7d bloqueaba TODO refill
                    # mientras el plan de 7d siguiera vivo. Eso dejaba huérfanos los planes donde
                    # el encolado síncrono inicial (chunk 2 de 4d) falló: 3 días generados y 4 vacíos
                    # sin recuperación automática. Ahora solo bloqueamos cuando hay chunks vivos
                    # en queue (estado normal); si no hay chunks pendientes y existe gap real, se
                    # permite refill para recuperar el hueco.
                    disable_rolling_refill_for_active_7d = False
                    if total_planned_days == 7 and days_remaining_in_plan > 0:
                        cursor.execute(
                            "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                            "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                            (plan_id,)
                        )
                        chunks_in_flight = int(((cursor.fetchone() or {}).get('cnt') or 0))
                        has_orphan_gap = chunks_in_flight == 0 and len(shifted_days) < total_planned_days
                        disable_rolling_refill_for_active_7d = not has_orphan_gap

                    if disable_rolling_refill_for_active_7d and needs_fill:
                        logger.info(
                            f"[P1-1] Plan de 7 días {plan_id}: rolling refill bloqueado durante vida útil "
                            f"(restantes={days_remaining_in_plan}, visibles={len(shifted_days)})."
                        )
                    elif total_planned_days == 7 and days_remaining_in_plan > 0 and needs_fill and not is_partial:
                        logger.warning(
                            f"[P0-4] Plan de 7 días {plan_id}: detectado gap huérfano "
                            f"(visibles={len(shifted_days)}/{total_planned_days}, sin chunks vivos). "
                            f"Habilitando rolling refill de recuperación."
                        )
                    elif total_planned_days == 7 and days_remaining_in_plan == 0 and not is_partial:
                        # [P0-1] Plan semanal expirado: auto-renovar con señales de aprendizaje frescas.
                        # Genera una nueva semana [3,4] en el mismo plan_id, preservando historial.
                        try:
                            from cron_tasks import _enqueue_plan_chunk
                            cursor.execute(
                                "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                                "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                                (plan_id,)
                            )
                            if ((cursor.fetchone() or {}).get('cnt') or 0) == 0:
                                cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,))
                                profile_row = cursor.fetchone()
                                hp = (profile_row or {}).get("health_profile", {}) or {}
                                if hp:
                                    cursor.execute(
                                        "SELECT COALESCE(MAX(week_number), 0) AS max_week FROM plan_chunk_queue "
                                        "WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                        (plan_id,)
                                    )
                                    next_week = int((cursor.fetchone() or {}).get('max_week') or 0) + 1
                                    previous_meals = [
                                        m.get('name', '') for d in shifted_days
                                        for m in d.get('meals', []) if m.get('name')
                                    ]
                                    current_offset = 0
                                    # [P0-1 FIX] Plan renovado: start = today para que el gate
                                    # de adherencia previa calcule ventanas correctas.
                                    renewal_plan_start_iso = today.isoformat()
                                    for chunk_count in [3, 4]:
                                        snapshot = {
                                            "form_data": {
                                                **hp,
                                                "user_id": user_id,
                                                "totalDays": chunk_count,
                                                "_plan_start_date": renewal_plan_start_iso,
                                            },
                                            "taste_profile": "",
                                            "memory_context": "",
                                            "previous_meals": previous_meals,
                                            "totalDays": chunk_count,
                                            "_is_rolling_refill": True,
                                            "_is_weekly_renewal": True,
                                        }
                                        _enqueue_plan_chunk(
                                            user_id, plan_id, next_week, current_offset,
                                            chunk_count, snapshot, chunk_kind="rolling_refill",
                                        )
                                        logger.info(
                                            f"🔄 [P0-1 RENEWAL] Chunk semana {next_week} encolado "
                                            f"(offset={current_offset}, count={chunk_count}) plan {plan_id}"
                                        )
                                        next_week += 1
                                        current_offset += chunk_count
                                    shifted_days = []
                                    shifted_data['grocery_start_date'] = today.isoformat()
                                    shifted_data['generation_status'] = 'generating_next'
                                    modified = True
                                    logger.info(f"🔄 [P0-1 RENEWAL] Plan semanal {plan_id} renovado.")
                                else:
                                    logger.warning(f"⚠️ [P0-1 RENEWAL] Sin health_profile para user {user_id}.")
                        except Exception as e:
                            logger.error(f"❌ [P0-1 RENEWAL] Error renovando plan semanal: {e}")
                    elif is_partial and needs_fill and total_planned_days > 7:
                        # [VISUAL CONTINUITY] Plan mensual/quincenal en estado partial:
                        # los chunks futuros ya están encolados, pero el usuario está viendo
                        # un gap porque el día pasó y el siguiente chunk aún no fue disparado.
                        # No re-encolamos (causaría días duplicados); aceleramos el siguiente
                        # chunk pendiente para que execute_after = NOW() y el cron lo tome
                        # en su próxima corrida (≤ 1 minuto).
                        try:
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET execute_after = NOW(),
                                    updated_at = NOW()
                                WHERE id = (
                                    SELECT id FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s
                                      AND status IN ('pending', 'stale')
                                      AND execute_after > NOW()
                                    ORDER BY week_number ASC
                                    LIMIT 1
                                )
                                RETURNING id, week_number
                                """,
                                (plan_id,)
                            )
                            accelerated = cursor.fetchone()
                            if accelerated:
                                logger.info(
                                    f"⚡ [VISUAL CONTINUITY] Chunk {accelerated['week_number']} "
                                    f"(id={accelerated['id']}) acelerado a NOW() para plan {plan_id} "
                                    f"(gap visible: {len(shifted_days)}/{window_needed} días)"
                                )
                        except Exception as e:
                            logger.error(f"❌ [VISUAL CONTINUITY] Error acelerando chunk pendiente: {e}")
                    elif not is_partial and needs_fill:
                        try:
                            from cron_tasks import _enqueue_plan_chunk
                            cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,))
                            profile_row = cursor.fetchone()
                            hp = (profile_row or {}).get("health_profile", {}) or {}
                            if hp:
                                previous_meals = [
                                    m.get('name', '') for d in shifted_days
                                    for m in d.get('meals', []) if m.get('name')
                                ]
                                days_offset = len(shifted_days)

                                # [P0-5] Catch-up: enqueue ALL missing days as sequential chunks,
                                # not just the current window. This ensures that after a long absence
                                # (e.g. 13 days missing) the entire gap gets filled, not just 3 days.
                                total_missing = days_remaining_in_plan - days_offset
                                catchup_chunks = split_with_absorb(total_missing, PLAN_CHUNK_SIZE) if total_missing > 0 else []

                                # [P1-3] El siguiente week_number debe seguir al máximo chunk no-cancelado,
                                # no solo a los completed, para no desalinearse si hubo failed/stale intermedios.
                                cursor.execute(
                                    """
                                    SELECT COALESCE(MAX(week_number), 1) AS max_week
                                    FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s
                                      AND status <> 'cancelled'
                                    """,
                                    (plan_id,)
                                )
                                existing_chunks = cursor.fetchone()
                                next_week = int((existing_chunks or {}).get('max_week') or 1) + 1

                                current_offset = days_offset
                                chunks_enqueued = 0

                                # [P0-1 FIX] _plan_start_date vigente (post-shift) para que el
                                # gate de adherencia previa pueda calcular ventanas correctas.
                                catchup_plan_start_iso = (
                                    (start_dt + timedelta(days=days_since_creation)).isoformat()
                                    if needs_shift else start_date_str
                                )

                                for chunk_count in catchup_chunks:
                                    cursor.execute(
                                        """
                                        SELECT id, status, chunk_kind
                                        FROM plan_chunk_queue
                                        WHERE meal_plan_id = %s
                                          AND week_number = %s
                                          AND status IN ('pending', 'processing', 'stale', 'failed')
                                        ORDER BY updated_at DESC NULLS LAST, created_at DESC
                                        LIMIT 1
                                        """,
                                        (plan_id, next_week)
                                    )
                                    conflicting_chunk = cursor.fetchone()

                                    if conflicting_chunk:
                                        logger.info(
                                            f"[P1-2] Saltando catch-up rolling refill para plan {plan_id}: "
                                            f"ya existe chunk objetivo week={next_week} "
                                            f"(id={conflicting_chunk['id']}, status={conflicting_chunk['status']}, "
                                            f"kind={conflicting_chunk.get('chunk_kind', 'unknown')})."
                                        )
                                    else:
                                        snapshot = {
                                            "form_data": {
                                                **hp,
                                                "user_id": user_id,
                                                "totalDays": chunk_count,
                                                "_plan_start_date": catchup_plan_start_iso,
                                            },
                                            "taste_profile": "",
                                            "memory_context": "",
                                            "previous_meals": previous_meals,
                                            "totalDays": chunk_count,
                                            "_is_rolling_refill": True,
                                        }
                                        _enqueue_plan_chunk(
                                            user_id,
                                            plan_id,
                                            next_week,
                                            current_offset,
                                            chunk_count,
                                            snapshot,
                                            chunk_kind="rolling_refill",
                                        )
                                        chunks_enqueued += 1
                                        logger.info(
                                            f"🤖 [ROLLING WINDOW] Chunk IA encolado "
                                            f"(week={next_week}, offset={current_offset}, count={chunk_count}) "
                                            f"para plan {plan_id}"
                                        )

                                    next_week += 1
                                    current_offset += chunk_count

                                if chunks_enqueued > 0:
                                    shifted_data['generation_status'] = 'generating_next'
                                    logger.info(
                                        f"🤖 [P0-5 CATCHUP] {chunks_enqueued} chunk(s) encolados "
                                        f"(total_missing={total_missing}, sizes={catchup_chunks}) "
                                        f"para plan {plan_id}"
                                    )
                                    modified = True
                            else:
                                logger.warning(f"⚠️ [ROLLING WINDOW] Sin health_profile para user {user_id}.")
                        except Exception as e:
                            logger.error(f"❌ [ROLLING WINDOW] Error encolando chunk IA: {e}")

                    # 4. Save rolling window updates
                    if modified:
                        shifted_data['days'] = shifted_days
                        new_plan_start_iso = None
                        if needs_shift and start_date_str:
                            new_start = start_dt + timedelta(days=days_since_creation)
                            new_plan_start_iso = new_start.isoformat()
                            shifted_data['grocery_start_date'] = new_plan_start_iso
                            
                            # [P0-C] Accumulate shift days
                            current_accum = int(shifted_data.get("_shift_days_accumulated", 0))
                            shifted_data["_shift_days_accumulated"] = current_accum + days_since_creation

                        # [P0-2] Sello CAS: timestamp que el worker compara para detectar
                        # si el plan fue modificado externamente durante el LLM call.
                        shifted_data['_plan_modified_at'] = datetime.now(timezone.utc).isoformat()

                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                            (json.dumps(shifted_data, ensure_ascii=False), plan_id)
                        )

                        # [P0-5 FIX] Sincronizar _plan_start_date en los snapshots de los chunks
                        # vivos del plan: tras un shift, grocery_start_date avanza pero los chunks
                        # ya encolados conservaban el origen original, desfasando los cálculos de
                        # _check_chunk_learning_ready (ventana de adherencia) y la asignación de
                        # day_name en Smart Shuffle. Ahora todos los chunks no terminales heredan
                        # el nuevo origen del plan en la misma transacción.
                        if needs_shift and new_plan_start_iso:
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET pipeline_snapshot = jsonb_set(
                                        pipeline_snapshot,
                                        '{form_data,_plan_start_date}',
                                        %s::jsonb,
                                        true
                                    ),
                                    updated_at = NOW()
                                WHERE meal_plan_id = %s
                                  AND status IN ('pending', 'processing', 'stale', 'failed', 'pending_user_action')
                                """,
                                (json.dumps(new_plan_start_iso), plan_id)
                            )

                            # [P0-C] Shift execute_after for all pending future chunks
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET execute_after = execute_after + (%s || ' days')::interval,
                                    updated_at = NOW()
                                WHERE meal_plan_id = %s
                                  AND status IN ('pending', 'stale')
                                  AND execute_after > NOW()
                                """,
                                (days_since_creation, plan_id)
                            )

        return {"success": True, "message": "Plan actualizado a la fecha.", "plan_data": shifted_data}
    
    except Exception as e:
        logger.error(f"❌ [API SHIFT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
def api_analyze(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")

        history = []
        likes = []
        taste_profile = ""

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)
            
            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)

        # Detectar si es un plan de largo plazo que se beneficia del chunking
        total_days_requested = int(data.get("totalDays", 3))
        user_has_profile = actual_user_id and _user_has_profile(actual_user_id)
        use_chunking = bool(user_has_profile and total_days_requested > PLAN_CHUNK_SIZE)

        pipeline_data = dict(data)
        from datetime import datetime, timezone, timedelta

        # SIEMPRE recalcular _plan_start_date en el backend a midnight local de HOY.
        # Nunca confiar en lo que venga del frontend/localStorage: observamos valores
        # obsoletos (día anterior) que corrompían day_name.
        tz_offset_mins = int(data.get("tzOffset", 0))
        now_utc = datetime.now(timezone.utc)
        local_time = now_utc - timedelta(minutes=tz_offset_mins)
        local_midnight = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date_iso = (local_midnight + timedelta(minutes=tz_offset_mins)).isoformat()

        pipeline_data["_plan_start_date"] = start_date_iso
        data["_plan_start_date"] = start_date_iso

        if use_chunking:
            # Solo generar la Semana 1 ahora; las semanas 2-4 se generan en background
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE
        elif total_days_requested > PLAN_CHUNK_SIZE:
            # Usuario sin perfil o guest solicitó plan largo → capear a 3 días
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE

        # [P0 FIX GAP 1] Persistir update_reason global como señal de aprendizaje
        update_reason = data.get("update_reason")
        if actual_user_id and update_reason and update_reason != "dislike":
            def _persist_global_update_reason():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (actual_user_id, "full_plan", f"swap:{update_reason}")
                    )
                    logger.info(f"📝 [GLOBAL UPDATE LEARN] Razón persistida: reason=swap:{update_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [GLOBAL UPDATE LEARN] Error persistiendo update reason: {e}")
            background_tasks.add_task(_persist_global_update_reason)

        if actual_user_id:
            from cron_tasks import inject_learning_signals_from_profile
            inject_learning_signals_from_profile(actual_user_id, pipeline_data)

        memory_ctx = memory.get("full_context_str", "") if session_id else ""
        result = run_plan_pipeline(pipeline_data, history, taste_profile,
                                   memory_context=memory_ctx,
                                   background_tasks=background_tasks)

        if actual_user_id:
            hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
            if hp_data:
                update_user_health_profile(actual_user_id, hp_data)
                logger.info(f"💾 [SYNC] health_profile guardado para user {actual_user_id}")

        if session_id:
            goal = data.get('mainGoal', 'Desconocido')
            save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
            save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada generada analíticamente!")
            background_tasks.add_task(summarize_and_prune, session_id)

        if actual_user_id:
            log_api_usage(actual_user_id, "gemini_analyze")

        selected_techniques = result.pop("_selected_techniques", None)
        # Evitar filtraciones de estado interno al frontend
        result.pop("_profile_embedding", None)
        result.pop("_active_learning_signals", None)

        if use_chunking:
            # Guardar sincrónicamente para obtener el plan_id y encolar semanas restantes
            plan_id = save_partial_plan_get_id(actual_user_id, result, selected_techniques, total_days_requested)

            if plan_id:
                import math
                from cron_tasks import _enqueue_plan_chunk
                snapshot = {
                    "form_data": data,
                    "taste_profile": taste_profile,
                    "memory_context": memory_ctx,
                    "totalDays": total_days_requested,
                }
                chunks = split_with_absorb(total_days_requested, PLAN_CHUNK_SIZE)
                offset = chunks[0]
                for wk, count in enumerate(chunks[1:], start=2):
                    if count > 0:
                        try:
                            _enqueue_plan_chunk(actual_user_id, plan_id, wk, offset, count, snapshot, chunk_kind="initial_plan")
                        except Exception as chunk_err:
                            logger.warning(f"⚠️ [CHUNK] Error encolando chunk semana {wk} para {actual_user_id}: {chunk_err}")
                    offset += count
                logger.info(f"🚀 [CHUNK] Plan {plan_id} creado con semana 1. {len(chunks) - 1} chunks encolados en background.")

            # [GAP 6] Sembrar emergency_backup_plan con los 3 días recién generados (sin LLM).
            # Así Smart Shuffle tiene pool real si chunk 2+ falla antes de la rotación nocturna.
            if actual_user_id:
                from cron_tasks import _seed_emergency_backup_if_empty
                background_tasks.add_task(_seed_emergency_backup_if_empty, actual_user_id, result.get("days", []))

            # Anotar en la respuesta para que el frontend sepa que el plan está incompleto
            result["generation_status"] = "partial"
            result["total_days_requested"] = total_days_requested
            if plan_id:
                result["id"] = plan_id
        elif actual_user_id:
            background_tasks.add_task(_save_plan_and_track_background, actual_user_id, result, selected_techniques)
            # [GAP 6] También sembrar el backup para planes no-chunked (tier gratis).
            from cron_tasks import _seed_emergency_backup_if_empty
            background_tasks.add_task(_seed_emergency_backup_if_empty, actual_user_id, result.get("days", []))

        return result
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/stream")
async def api_analyze_stream(request: Request, background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """
    Streaming SSE endpoint para generación de planes con progreso en tiempo real.
    Emite eventos:
      - phase: cambio de fase del pipeline (skeleton, parallel_generation, assembly, review)
      - day_started: un worker paralelo inició la generación de un día
      - day_complete: un worker paralelo terminó un día
      - complete: plan final listo (contiene el plan JSON completo)
      - error: hubo un error
    """
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")

        history = []
        likes = []
        taste_profile = ""

        if session_id:
            get_or_create_session(session_id)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]

        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)
            
            # [GAP 10] → Movido a inject_learning_signals_from_profile (P0 fix)
            pass

        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)

        # [GAP 4 FIX: Detectar si es un plan de largo plazo que se beneficia del chunking]
        total_days_requested = int(data.get("totalDays", 3))
        user_has_profile = actual_user_id and _user_has_profile(actual_user_id)
        use_chunking = bool(user_has_profile and total_days_requested > PLAN_CHUNK_SIZE)

        pipeline_data = dict(data)
        from datetime import datetime, timezone, timedelta

        # SIEMPRE recalcular _plan_start_date en el backend a midnight local de HOY.
        # Nunca confiar en lo que venga del frontend/localStorage: observamos valores
        # obsoletos (día anterior) que corrompían day_name.
        tz_offset_mins = int(data.get("tzOffset", 0))
        now_utc = datetime.now(timezone.utc)
        local_time = now_utc - timedelta(minutes=tz_offset_mins)
        local_midnight = local_time.replace(hour=0, minute=0, second=0, microsecond=0)
        start_date_iso = (local_midnight + timedelta(minutes=tz_offset_mins)).isoformat()

        pipeline_data["_plan_start_date"] = start_date_iso
        data["_plan_start_date"] = start_date_iso

        if use_chunking:
            # Solo generar la Semana 1 ahora; las semanas 2-4 se generan en background
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE
        elif total_days_requested > PLAN_CHUNK_SIZE:
            # Usuario sin perfil o guest solicitó plan largo → capear a 3 días
            pipeline_data["_days_to_generate"] = PLAN_CHUNK_SIZE

        # [P0 FIX GAP 1] Persistir update_reason global como señal de aprendizaje
        update_reason = data.get("update_reason")
        if actual_user_id and update_reason and update_reason != "dislike":
            def _persist_global_update_reason_stream():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (actual_user_id, "full_plan", f"swap:{update_reason}")
                    )
                    logger.info(f"📝 [GLOBAL UPDATE LEARN (STREAM)] Razón persistida: reason=swap:{update_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [GLOBAL UPDATE LEARN (STREAM)] Error persistiendo update reason: {e}")
            background_tasks.add_task(_persist_global_update_reason_stream)

        # Inyectar TODAS las señales de aprendizaje continuo
        if actual_user_id:
            from cron_tasks import inject_learning_signals_from_profile
            inject_learning_signals_from_profile(actual_user_id, pipeline_data)

        # Cola async para comunicar progreso entre el thread del pipeline y el generador SSE
        progress_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_callback(event_data: dict):
            """Callback thread-safe que pone eventos en la cola async."""
            try:
                loop.call_soon_threadsafe(progress_queue.put_nowait, event_data)
            except Exception:
                pass

        # Ejecutar el pipeline en un thread separado para no bloquear el event loop
        pipeline_result = {"result": None, "error": None}

        async def run_pipeline():
            try:
                result = await arun_plan_pipeline(
                    pipeline_data, history, taste_profile,
                    memory_context=memory.get("full_context_str", "") if session_id else "",
                    progress_callback=progress_callback,
                    background_tasks=background_tasks
                )
                pipeline_result["result"] = result
            except Exception as e:
                pipeline_result["error"] = str(e)
                logger.error(f"❌ [SSE PIPELINE ERROR]: {e}")
                traceback.print_exc()
            finally:
                # Señal de fin para que el generador SSE cierre
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, {"event": "_done"})
                except Exception:
                    pass

        asyncio.create_task(run_pipeline())

        async def event_generator():
            """Generador SSE que consume la cola de progreso."""
            try:
                while True:
                    # Esperar eventos con timeout para detectar desconexión del cliente
                    try:
                        event_data = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Heartbeat para mantener la conexión viva
                        yield f"data: {_json.dumps({'event': 'heartbeat'})}\n\n"

                        # Verificar si el cliente cerró la conexión
                        if await request.is_disconnected():
                            logger.info("🔌 [SSE] Cliente desconectado, abortando stream.")
                            return
                        continue

                    # Señal de fin del pipeline
                    if event_data.get("event") == "_done":
                        # Enviar resultado final o error
                        if pipeline_result["error"]:
                            yield f"data: {_json.dumps({'event': 'error', 'data': {'message': pipeline_result['error']}})}\n\n"
                        elif pipeline_result["result"]:
                            result = pipeline_result["result"]

                            # Post-procesamiento idéntico al endpoint síncrono
                            if actual_user_id:
                                hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
                                if hp_data:
                                    try:
                                        update_user_health_profile(actual_user_id, hp_data)
                                    except Exception:
                                        pass

                            if session_id:
                                goal = data.get('mainGoal', 'Desconocido')
                                try:
                                    save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
                                    save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada!")
                                    # summarize_and_prune se ejecuta en background
                                    threading.Thread(target=summarize_and_prune, args=(session_id,), daemon=True).start()
                                except Exception:
                                    pass

                            if actual_user_id:
                                try:
                                    log_api_usage(actual_user_id, "gemini_analyze")
                                except Exception:
                                    pass

                            selected_techniques = result.pop("_selected_techniques", None)
                            # Evitar filtraciones de estado interno al frontend
                            result.pop("_profile_embedding", None)
                            result.pop("_active_learning_signals", None)
                            
                            # [GAP 4 FIX: Post-procesamiento y chunking background para streaming]
                            if use_chunking:
                                plan_id = save_partial_plan_get_id(actual_user_id, result, selected_techniques, total_days_requested)
                                if plan_id:
                                    import math
                                    from cron_tasks import _enqueue_plan_chunk
                                    week1_meals = [
                                        m["name"] for d in result.get("days", [])
                                        for m in d.get("meals", []) if m.get("name")
                                    ]
                                    snapshot = {
                                        "form_data": data,
                                        "taste_profile": taste_profile,
                                        "memory_context": memory.get("full_context_str", "") if session_id else "",
                                        "previous_meals": week1_meals,
                                        "totalDays": total_days_requested,
                                    }
                                    chunks = split_with_absorb(total_days_requested, PLAN_CHUNK_SIZE)
                                    offset = chunks[0]
                                    for wk, count in enumerate(chunks[1:], start=2):
                                        if count > 0:
                                            try:
                                                _enqueue_plan_chunk(actual_user_id, plan_id, wk, offset, count, snapshot, chunk_kind="initial_plan")
                                            except Exception as chunk_err:
                                                logger.warning(f"⚠️ [CHUNK SSE] Error encolando chunk semana {wk} para {actual_user_id}: {chunk_err}")
                                        offset += count
                                    logger.info(f"🚀 [CHUNK SSE] Plan {plan_id} creado con semana 1. {len(chunks) - 1} chunks encolados en background.")

                                # [GAP 6] Sembrar emergency_backup_plan con los 3 días recién generados.
                                if actual_user_id:
                                    from cron_tasks import _seed_emergency_backup_if_empty
                                    threading.Thread(
                                        target=_seed_emergency_backup_if_empty,
                                        args=(actual_user_id, result.get("days", [])),
                                        daemon=True
                                    ).start()

                                result["generation_status"] = "partial"
                                result["total_days_requested"] = total_days_requested
                                if plan_id:
                                    result["id"] = plan_id
                            elif actual_user_id:
                                threading.Thread(
                                    target=_save_plan_and_track_background,
                                    args=(actual_user_id, result, selected_techniques),
                                    daemon=True
                                ).start()
                                # [GAP 6] También para planes no-chunked (tier gratis).
                                from cron_tasks import _seed_emergency_backup_if_empty
                                threading.Thread(
                                    target=_seed_emergency_backup_if_empty,
                                    args=(actual_user_id, result.get("days", [])),
                                    daemon=True
                                ).start()

                            yield f"data: {_json.dumps({'event': 'complete', 'data': result}, ensure_ascii=False, default=str)}\n\n"
                        return

                    # Evento de progreso normal
                    yield f"data: {_json.dumps(event_data, ensure_ascii=False)}\n\n"

            except asyncio.CancelledError:
                logger.info("🔌 [SSE] Stream cancelado por el cliente.")
            except Exception as e:
                logger.error(f"❌ [SSE] Error en generador: {e}")
                yield f"data: {_json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze/stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recipe/expand")
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        if not data.get("recipe") or not data.get("name"):
            raise HTTPException(status_code=400, detail="Faltan datos de la receta para expandir.")
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_recipe_expand")
            
        expanded_steps = expand_recipe_agent(data)
        
        if user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            if current_plan and "days" in current_plan:
                updated = False
                for day in current_plan.get("days", []):
                    for m in day.get("meals", []):
                        if m.get("name") == data.get("name"):
                            m["recipe"] = expanded_steps
                            m["isExpanded"] = True
                            updated = True
                            break
                    if updated: break
                
                if updated:
                    plan_with_id = get_latest_meal_plan_with_id(user_id)
                    if plan_with_id and "id" in plan_with_id:
                        update_meal_plan_data(plan_with_id["id"], current_plan)

        return {"success": True, "expanded_recipe": expanded_steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recipe/expand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/swap-meal")
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        rejected_meal = data.get("rejected_meal")
        meal_type = data.get("meal_type", "")
        swap_reason = data.get("swap_reason", "variety")  # variety | time | dislike | similar | budget
        
        # Solo registrar rechazo cuando el usuario explícitamente dice "No me gusta"
        if rejected_meal and swap_reason == "dislike":
            logger.info(f"👎 [SWAP] Rechazo real registrado: '{rejected_meal}' (razón: {swap_reason})")
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)
        else:
            logger.info(f"🔄 [SWAP] Cambio sin rechazo: '{rejected_meal or 'N/A'}' (razón: {swap_reason})")

        # [P1 FIX] Persistir TODAS las swap reasons como señal de aprendizaje.
        # El cron las lee de abandoned_meal_reasons para detectar patrones
        # (ej: "usuario siempre cambia desayuno por falta de tiempo → simplificar").
        if user_id and user_id != "guest" and rejected_meal and swap_reason != "dislike":
            def _persist_swap_reason():
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (user_id, meal_type or "unknown", f"swap:{swap_reason}")
                    )
                    logger.info(f"📝 [SWAP LEARN] Razón persistida: meal_type={meal_type}, reason=swap:{swap_reason}")
                except Exception as e:
                    logger.warning(f"⚠️ [SWAP LEARN] Error persistiendo swap reason: {e}")
            background_tasks.add_task(_persist_swap_reason)
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_swap_meal")
            
            # --- HOT SIGNAL PATH (MEJORA 4) ---
            try:
                # Obtenemos likes y rechazos recientes para que el LLM no repita errores en el JIT Swap
                recent_likes = get_user_likes(user_id)
                recent_rejections = get_active_rejections(user_id=user_id, session_id=session_id)
                
                data["liked_meals"] = [like["meal_name"] for like in recent_likes] if recent_likes else []
                data["disliked_meals"] = [r["meal_name"] for r in recent_rejections] if recent_rejections else []
                logger.info(f"🔥 [HOT SIGNAL] Inyectando {len(data['liked_meals'])} likes y {len(data['disliked_meals'])} rechazos al JIT Swap.")
            except Exception as e:
                logger.warning(f"⚠️ [HOT SIGNAL] Error recuperando señales en tiempo real: {e}")
            
        result = swap_meal(data)
        return result
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/like")
def api_like(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        insert_like(data)
        return {"success": True, "message": "Tu like/dislike ha sido guardado exitosamente."}
    except Exception as e:
        return {"error": str(e)}

@router.post("/restock")
def api_restock(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        plan_id = data.get("plan_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión para usar la nevera virtual."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}

        # [P0-1] Validación de Idempotencia: Verificar si el plan ya fue registrado ANTES de insertar en DB
        real_plan_id = None
        plan_data = None
        if supabase:
            try:
                if plan_id:
                    plan_res = supabase.table("meal_plans").select("id, plan_data").eq("id", plan_id).execute()
                else:
                    plan_res = supabase.table("meal_plans").select("id, plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                
                if plan_res and plan_res.data and len(plan_res.data) > 0:
                    real_plan_id = plan_res.data[0].get("id")
                    plan_data = plan_res.data[0].get("plan_data", {})
                    
                    if plan_data.get("is_restocked") is True:
                        logger.warning(f"⚠️ [RESTOCK] El plan {real_plan_id} ya fue registrado previamente. Ignorando petición duplicada.")
                        return {"success": True, "message": "Las compras ya estaban registradas."}
            except Exception as check_err:
                logger.warning(f"⚠️ Error verificando estado is_restocked: {check_err}")

        # Validación MURO Omitida: Ahora confiamos en el Delta Shopping del frontend.
        # El frontend solo envía los ingredientes que no están en la Nevera.
        success = restock_inventory(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "restock_inventory")

            # Marcar el plan como "restocked" en BD para futuras peticiones
            if supabase and real_plan_id and plan_data is not None:
                try:
                    plan_data["is_restocked"] = True
                    supabase.table("meal_plans").update({"plan_data": plan_data}).eq("id", real_plan_id).execute()
                    logger.info(f"✅ [RESTOCK] plan_data 'is_restocked' guardado en DB para plan ID {real_plan_id}")
                except Exception as mark_err:
                    logger.warning(f"⚠️ No se pudo marcar plan como restocked: {mark_err}")

            return {"success": True, "message": "¡Despensa actualizada exitosamente!"}
        else:
            return {"success": False, "message": "Hubo un problema actualizando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/restock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inventory/consume")
def api_consume_inventory(data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        user_id = data.get("user_id")
        ingredients = data.get("ingredients")
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        if not ingredients or not isinstance(ingredients, list):
            return {"success": False, "message": "Lista de ingredientes inválida."}
            
        success = consume_inventory_items_completely(user_id, ingredients)
        
        if success:
            log_api_usage(user_id, "consume_inventory")
            return {"success": True, "message": "Inventario actualizado exitosamente."}
        else:
            return {"success": False, "message": "Hubo un problema vaciando algunos ingredientes."}
            
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/inventory/consume: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recalculate-shopping-list")
def api_recalculate_shopping_list(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """
    Recalcula la lista de compras escalando las recetas por el householdSize 
    y LUEGO deduciendo el inventario físico (is_new_plan=False).
    Este acercamiento garantiza exactitud matemática.
    """
    try:
        user_id = data.get("user_id")
        household_size = max(1, int(data.get("householdSize", 1) or 1))
        grocery_duration = data.get("groceryDuration", "weekly")
        is_new_plan_flag = data.get("is_new_plan", False)
        
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Debes iniciar sesión."}
            
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado.")
        
        plan_record = get_latest_meal_plan_with_id(user_id)
        if not plan_record:
            return {"success": False, "message": "No hay plan activo."}
        
        plan_id = plan_record["id"]
        plan_data = plan_record.get("plan_data", {})
        
        if not plan_data:
            return {"success": False, "message": "Datos de plan inválidos."}
            
        from shopping_calculator import get_shopping_list_delta
        from db_plans import update_meal_plan_data
        
        # Generar las 3 variantes escaladas dinámicamente según el householdSize
        # usando el delta matemático para evitar duplicados si hay inventario (Gap 3)
        scaled_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size))
        scaled_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size) * 2.0)
        scaled_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=is_new_plan_flag, structured=True, multiplier=float(household_size) * 4.0)
        
        # Debug: Log DETAILED per-item comparison to diagnose scaling
        if scaled_7:
            sample = [f"{it.get('display_string','?')}" for it in scaled_7[:3]]
            has_days = bool(plan_data.get("days"))
            len_days = len(plan_data.get("days", []))
            has_perfectDay = bool(plan_data.get("perfectDay"))
            logger.info(f"🔍 [RECALC DEBUG] ×{household_size} sample (7d): {sample} | has_days={has_days} len={len_days} has_perf={has_perfectDay}")
            
            # DEBUG GRANULAR: rastear proteínas y frutas específicas
            debug_keywords = ['pechuga', 'pavo', 'yogurt', 'lechosa', 'melón', 'melon', 'aguacate', 'arroz', 'pollo']
            for it in scaled_7:
                name_lower = it.get('name', '').lower()
                if any(kw in name_lower for kw in debug_keywords):
                    logger.info(f"  📊 [{household_size}p] {it.get('name')}: display_qty={it.get('display_qty')} | market_qty={it.get('market_qty')} {it.get('market_unit')} | display_string={it.get('display_string')}")
        
        # Seleccionar lista activa para el frontend legacy
        if grocery_duration == "biweekly":
            active_list = scaled_15
        elif grocery_duration == "monthly":
            active_list = scaled_30
        else:
            active_list = scaled_7
        
        # Actualizar plan en BD
        plan_data["aggregated_shopping_list"] = active_list
        plan_data["aggregated_shopping_list_weekly"] = scaled_7
        plan_data["aggregated_shopping_list_biweekly"] = scaled_15
        plan_data["aggregated_shopping_list_monthly"] = scaled_30
        
        # Solo limpiar `is_restocked` si los parámetros cambiaron realmente
        prev_hh = plan_data.get("calc_household_size")
        prev_dur = plan_data.get("calc_grocery_duration")
        has_changed = (prev_hh != household_size) or (prev_dur != grocery_duration)
        
        plan_data["calc_household_size"] = household_size
        plan_data["calc_grocery_duration"] = grocery_duration
        
        if has_changed and plan_data.get("is_restocked"):
            plan_data.pop("is_restocked", None)
            logger.info(f"🔄 [RECALC] is_restocked limpiado — cantidades cambiaron de {prev_hh}p/{prev_dur} a {household_size}p/{grocery_duration}, requiere re-registro")
        
        # DEBUG fingerprint: allows frontend to verify it received fresh data
        import time
        plan_data["_debug_recalc"] = {
            "household_size": household_size,
            "timestamp": time.time(),
            "weekly_items_count": len(scaled_7),
            "sample_item": scaled_7[0].get("display_string", "?") if scaled_7 else "empty"
        }
        
        update_meal_plan_data(plan_id, plan_data)
        
        logger.info(f"✅ [RECALC] Listas recalculadas exitosamente ×{household_size} personas")
        
        # Devolver el plan_data actualizado directamente para evitar race conditions
        # (el frontend no necesita re-fetch de Supabase)
        return {"success": True, "plan_data": plan_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recalculate-shopping-list: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{plan_id}/chunk-status")
def api_chunk_status(plan_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    from db_core import execute_sql_query
    try:
        res = execute_sql_query("SELECT user_id, plan_data FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
        if not res:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
            
        user_id = res["user_id"]
        plan_data = res["plan_data"]
        status = plan_data.get("generation_status", "complete")
        days_generated = len(plan_data.get("days", []))
        total_days = plan_data.get("total_days_requested", days_generated)
        
        # [GAP 2] Buscar chunks fallidos si hay problemas
        failed_chunks = []
        if status in ['failed', 'complete_partial']:
            chunks_res = execute_sql_query(
                "SELECT id, week_number, status, attempts FROM plan_chunk_queue WHERE meal_plan_id = %s AND status = 'failed' ORDER BY week_number ASC",
                (plan_id,)
            )
            if chunks_res:
                failed_chunks = chunks_res
                # Si hay chunks fallidos explícitos, forzamos el status general a failed
                status = "failed"
                
        # [GAP G FIX: Enriquecer payload con ETA y learning hint]
        eta_res = execute_sql_query(
            "SELECT execute_after FROM plan_chunk_queue WHERE meal_plan_id = %s AND status IN ('pending', 'processing') ORDER BY execute_after ASC LIMIT 1",
            (plan_id,), fetch_one=True
        )
        next_chunk_eta = eta_res["execute_after"].isoformat() if eta_res and eta_res.get("execute_after") else None
        
        last_learning_hint = "Analizando tus preferencias..."
        user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if user_res and user_res.get("health_profile"):
            hp = user_res["health_profile"]
            qh = hp.get("quality_history", [])
            if qh and len(qh) > 0:
                last_score = qh[-1].get("score", 0)
                last_learning_hint = f"Ajustando variedad (Quality Score: {last_score}/100)"

        # [GAP C] Exponer quality_warning y desglose de tiers
        quality_warning = bool(plan_data.get("quality_warning", False))
        quality_degraded_ratio = float(plan_data.get("quality_degraded_ratio", 0.0))
        tier_breakdown = execute_sql_query("""
            SELECT quality_tier, COUNT(*) AS cnt
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s AND status = 'completed' AND quality_tier IS NOT NULL
            GROUP BY quality_tier
        """, (plan_id,)) or []
        tier_summary = {r['quality_tier']: int(r['cnt']) for r in tier_breakdown}

        return {
            "status": status,
            "days_generated": days_generated,
            "total_days_requested": total_days,  # Added requested
            "total_days": total_days,  # Mantener por retrocompatibilidad
            "failed_chunks": failed_chunks,
            "next_chunk_eta": next_chunk_eta,
            "last_learning_hint": last_learning_hint,
            "quality_warning": quality_warning,
            "quality_degraded_ratio": quality_degraded_ratio,
            "tier_breakdown": tier_summary,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] en chunk-status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{plan_id}/retry-chunk/{chunk_id}")
def api_retry_chunk(plan_id: str, chunk_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    from db_core import execute_sql_write
    try:
        # Resetear el chunk fallido a 'pending'
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW(),
                updated_at = NOW()
            WHERE id = %s AND meal_plan_id = %s AND status = 'failed'
        """, (chunk_id, plan_id))

        # Revivir cualquier chunk que haya sido cancelado por culpa de este fallo
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                execute_after = NOW() + INTERVAL '1 minute',
                updated_at = NOW()
            WHERE meal_plan_id = %s AND status = 'cancelled'
        """, (plan_id,))

        # Volver a poner el plan en 'partial' para que el frontend retome el polling
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial"'),
                updated_at = NOW()
            WHERE id = %s
        """, (plan_id,))

        return {"success": True, "message": "Chunk reenviado a la cola"}

    except Exception as e:
        logger.error(f"❌ [ERROR] en retry-chunk: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# [GAP A] Endpoint de inspección de SLA — chunks atrasados
# ============================================================
def _verify_admin_token(authorization: Optional[str]):
    """Valida Bearer token contra CRON_SECRET. Si CRON_SECRET no está seteado,
    rechaza por defecto (no exponer admin endpoints en prod sin secreto).
    """
    cron_secret = os.environ.get("CRON_SECRET")
    if not cron_secret:
        raise HTTPException(status_code=503, detail="Admin endpoints disabled: CRON_SECRET not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    if token != cron_secret:
        raise HTTPException(status_code=403, detail="Invalid admin token")


@router.get("/admin/chunks/stuck")
def api_admin_chunks_stuck(
    request: Request,
    min_lag_hours: int = 1,
    limit: int = 100,
):
    """[GAP A] Inspección operacional: lista chunks atrasados (lag > min_lag_hours).

    Usar para diagnosticar por qué planes de 15-30 días no avanzan.
    Requiere Authorization: Bearer <CRON_SECRET>.
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        rows = execute_sql_query(
            """
            SELECT
                q.id,
                q.user_id,
                q.meal_plan_id,
                q.week_number,
                q.days_offset,
                q.days_count,
                q.status,
                q.attempts,
                q.quality_tier,
                q.escalated_at,
                q.execute_after,
                q.created_at,
                q.updated_at,
                EXTRACT(EPOCH FROM (NOW() - q.execute_after))::int AS lag_seconds,
                (mp.plan_data->>'total_days_requested')::int AS total_days_requested,
                jsonb_array_length(COALESCE(mp.plan_data->'days', '[]'::jsonb)) AS days_generated
            FROM plan_chunk_queue q
            LEFT JOIN meal_plans mp ON mp.id = q.meal_plan_id
            WHERE q.status IN ('pending', 'stale', 'processing', 'failed')
              AND q.execute_after < NOW() - make_interval(hours => %s)
            ORDER BY q.execute_after ASC
            LIMIT %s
            """,
            (int(min_lag_hours), int(limit)),
        ) or []

        # Resumen agregado
        by_status = {}
        for r in rows:
            s = r.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        max_lag_h = 0
        if rows:
            max_lag_h = round(max(int(r.get("lag_seconds") or 0) for r in rows) / 3600.0, 1)

        return {
            "count": len(rows),
            "max_lag_hours": max_lag_h,
            "by_status": by_status,
            "chunks": rows,
        }
    except Exception as e:
        logger.error(f"❌ [GAP A] Error en /admin/chunks/stuck: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plan_id}/regen-degraded")
def api_regen_degraded_chunks(plan_id: str, verified_user_id: Optional[str] = Depends(verify_api_quota)):
    """[GAP C] Regenera chunks completados en modo degradado (shuffle/edge/emergency)
    creando nuevos chunks pendientes que sobrescribirán los días afectados.

    Útil cuando el LLM volvió a estar disponible y el usuario quiere mejorar la calidad
    de un plan que se generó parcialmente con Smart Shuffle. Idempotente: si no hay
    chunks degradados completados, no hace nada.
    """
    from db_core import execute_sql_query, execute_sql_write
    import json
    try:
        # 1. Validar ownership del plan
        plan_row = execute_sql_query(
            "SELECT user_id, plan_data FROM meal_plans WHERE id = %s",
            (plan_id,), fetch_one=True
        )
        if not plan_row:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        if verified_user_id and str(plan_row["user_id"]) != str(verified_user_id):
            raise HTTPException(status_code=403, detail="No autorizado")

        # 2. Buscar chunks degradados completados que tengan snapshot recuperable
        degraded_chunks = execute_sql_query("""
            SELECT id, week_number, days_offset, days_count, pipeline_snapshot, user_id
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND status = 'completed'
              AND quality_tier IN ('shuffle', 'edge', 'emergency')
              AND pipeline_snapshot::text != '{}'
            ORDER BY week_number ASC
        """, (plan_id,)) or []

        if not degraded_chunks:
            return {
                "success": True,
                "regenerated": 0,
                "message": "No hay chunks degradados con snapshot recuperable. Si pasaron >48h, los snapshots ya fueron purgados."
            }

        # 3. Re-encolar como pending sin _degraded para que el worker use el LLM
        regenerated = 0
        for ch in degraded_chunks:
            snap = ch["pipeline_snapshot"]
            if isinstance(snap, str):
                snap = json.loads(snap)
            snap.pop("_degraded", None)

            execute_sql_write("""
                UPDATE plan_chunk_queue
                SET status = 'pending',
                    attempts = 0,
                    quality_tier = NULL,
                    pipeline_snapshot = %s::jsonb,
                    execute_after = NOW(),
                    escalated_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (json.dumps(snap, ensure_ascii=False), ch["id"]))
            regenerated += 1

        # 4. Marcar plan como partial para que el frontend retome polling
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial"'),
                updated_at = NOW()
            WHERE id = %s
        """, (plan_id,))

        return {
            "success": True,
            "regenerated": regenerated,
            "message": f"{regenerated} chunks degradados re-encolados. Procesarán en el próximo tick del worker."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [GAP C] Error en /regen-degraded: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/metrics")
def api_admin_metrics(
    request: Request,
    days: int = 7,
):
    """[GAP G] Agregados observacionales del pipeline de chunks.

    Responde:
      - totals por quality_tier
      - % degraded (shuffle+edge+emergency)
      - avg/p50/p95 duration_ms (usando percentile_cont)
      - learning quality: avg repeat_pct, count violations
      - top error messages (rate)
    """
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_query
    try:
        interval_str = f"{int(days)} days"

        tier_row = execute_sql_query(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE quality_tier = 'llm') AS llm,
                COUNT(*) FILTER (WHERE quality_tier = 'shuffle') AS shuffle,
                COUNT(*) FILTER (WHERE quality_tier = 'edge') AS edge,
                COUNT(*) FILTER (WHERE quality_tier = 'emergency') AS emergency,
                COUNT(*) FILTER (WHERE quality_tier = 'error') AS errors,
                COUNT(*) FILTER (WHERE was_degraded) AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        total = int(tier_row.get("total") or 0)
        degraded = int(tier_row.get("degraded") or 0)
        errors = int(tier_row.get("errors") or 0)
        degraded_pct = round((degraded / total) * 100.0, 2) if total else 0.0
        error_pct = round((errors / total) * 100.0, 2) if total else 0.0

        perf_row = execute_sql_query(
            """
            SELECT
                ROUND(AVG(duration_ms)::numeric, 0) AS avg_ms,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms)::int AS p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms)::int AS p95_ms,
                ROUND(AVG(lag_seconds)::numeric, 0) AS avg_lag_s,
                ROUND(AVG(retries)::numeric, 2) AS avg_retries
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND quality_tier != 'error'
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        learning_row = execute_sql_query(
            """
            SELECT
                ROUND(AVG(learning_repeat_pct)::numeric, 2) AS avg_repeat_pct,
                SUM(rejection_violations) AS rejection_violations_total,
                SUM(allergy_violations) AS allergy_violations_total,
                COUNT(*) FILTER (WHERE rejection_violations > 0) AS chunks_with_rej_violations,
                COUNT(*) FILTER (WHERE allergy_violations > 0) AS chunks_with_alg_violations
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND learning_repeat_pct IS NOT NULL
            """,
            (interval_str,), fetch_one=True,
        ) or {}

        top_errors = execute_sql_query(
            """
            SELECT LEFT(error_message, 200) AS error_prefix, COUNT(*) AS cnt
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - %s::interval
              AND error_message IS NOT NULL
            GROUP BY LEFT(error_message, 200)
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (interval_str,),
        ) or []

        return {
            "window_days": int(days),
            "total_chunks": total,
            "tiers": {
                "llm": int(tier_row.get("llm") or 0),
                "shuffle": int(tier_row.get("shuffle") or 0),
                "edge": int(tier_row.get("edge") or 0),
                "emergency": int(tier_row.get("emergency") or 0),
                "error": errors,
            },
            "degraded_pct": degraded_pct,
            "error_pct": error_pct,
            "perf": {
                "avg_ms": int(perf_row.get("avg_ms") or 0),
                "p50_ms": int(perf_row.get("p50_ms") or 0),
                "p95_ms": int(perf_row.get("p95_ms") or 0),
                "avg_lag_seconds": int(perf_row.get("avg_lag_s") or 0),
                "avg_retries": float(perf_row.get("avg_retries") or 0.0),
            },
            "learning": {
                "avg_repeat_pct": float(learning_row.get("avg_repeat_pct") or 0.0),
                "rejection_violations_total": int(learning_row.get("rejection_violations_total") or 0),
                "allergy_violations_total": int(learning_row.get("allergy_violations_total") or 0),
                "chunks_with_rejection_violations": int(learning_row.get("chunks_with_rej_violations") or 0),
                "chunks_with_allergy_violations": int(learning_row.get("chunks_with_alg_violations") or 0),
            },
            "top_errors": top_errors,
        }
    except Exception as e:
        logger.error(f"❌ [GAP G] Error en /admin/metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/chunks/{chunk_id}/escalate")
def api_admin_escalate_chunk(chunk_id: str, request: Request):
    """[GAP A] Forzar escalado/pickup inmediato de un chunk concreto."""
    _verify_admin_token(request.headers.get("authorization"))
    from db_core import execute_sql_write
    try:
        res = execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending',
                escalated_at = NOW(),
                execute_after = NOW(),
                attempts = 0,
                updated_at = NOW()
            WHERE id = %s AND status IN ('pending', 'stale', 'failed')
            RETURNING id, meal_plan_id, week_number
            """,
            (chunk_id,),
            returning=True,
        )
        if not res:
            raise HTTPException(status_code=404, detail="Chunk no encontrado o ya en processing")
        return {"success": True, "chunk": res[0]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [GAP A] Error en /admin/chunks/escalate: {e}")
        raise HTTPException(status_code=500, detail=str(e))
