import logging
import traceback
from datetime import datetime, timezone, timedelta
import json
import copy
import random
from db_core import execute_sql_query, execute_sql_write
from db_inventory import (
    deduct_consumed_meal_from_inventory,
    get_inventory_activity_since,
    get_raw_user_inventory,
    get_user_inventory_net,
    release_chunk_reservations,
    reserve_plan_ingredients,
)
from db import get_latest_meal_plan_with_id, get_user_likes, get_active_rejections, get_recent_plans
from db_facts import get_all_user_facts, get_consumed_meals_since
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from collections import Counter

from schemas import HealthProfileSchema

from constants import (
    CHUNK_MIN_FRESH_PANTRY_ITEMS,
    CHUNK_MAX_FAILURE_ATTEMPTS,
    CHUNK_LEARNING_MODE,
    CHUNK_LEARNING_READY_DELAY_HOURS,
    CHUNK_LEARNING_READY_MAX_DEFERRALS,
    CHUNK_LEARNING_READY_MIN_RATIO,
    CHUNK_PANTRY_EMPTY_MAX_REMINDERS,
    CHUNK_PANTRY_EMPTY_REMINDER_HOURS,
    CHUNK_PANTRY_EMPTY_TTL_HOURS,
    CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES,
    CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS,
    CHUNK_PANTRY_QUANTITY_MODE,
    CHUNK_PANTRY_SNAPSHOT_TTL_HOURS,
    CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS,
    CHUNK_RETRY_BASE_MINUTES,
    CHUNK_RETRY_CRITICAL_MINUTES,
    CHUNK_PROACTIVE_MARGIN_DAYS,
    CHUNK_SCHEDULER_INTERVAL_MINUTES,
    strip_accents,
    get_embedding,
    cosine_similarity,
)
from graph_orchestrator import run_plan_pipeline
from memory_manager import build_memory_context
from services import _save_plan_and_track_background
from agent import analyze_preferences_agent
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def register_plan_chunk_scheduler(scheduler) -> None:
    """Registra el polling del worker de chunks una sola vez en el scheduler global."""
    if not scheduler:
        return

    if not scheduler.get_job("process_plan_chunk_queue"):
        scheduler.add_job(
            process_plan_chunk_queue,
            "interval",
            minutes=CHUNK_SCHEDULER_INTERVAL_MINUTES,
            id="process_plan_chunk_queue",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [CHUNK SCHEDULER] Worker plan_chunk_queue registrado cada "
            f"{CHUNK_SCHEDULER_INTERVAL_MINUTES} min."
        )

    # [P0-C] Job separado: refresh proactivo de snapshots de despensa en chunks pending/stale
    # cuyo _pantry_captured_at supera la mitad del TTL. Evita que el worker llegue al
    # execute_after con un snapshot vencido si su live fetch puntual falla.
    if not scheduler.get_job("proactive_refresh_pantry_snapshots"):
        scheduler.add_job(
            _proactive_refresh_pending_pantry_snapshots,
            "interval",
            minutes=CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES,
            id="proactive_refresh_pantry_snapshots",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P0-C] Refresh proactivo de despensa registrado cada "
            f"{CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES} min "
            f"(horizonte={CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS}h, "
            f"max_users={CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS})."
        )

    # [P1-B] Job separado: recuperación de chunks failed en planes largos (15d+)
    if not scheduler.get_job("recover_failed_chunks_long_plans"):
        scheduler.add_job(
            _recover_failed_chunks_for_long_plans,
            "interval",
            minutes=15,
            id="recover_failed_chunks_long_plans",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info("⏰ [P1-B] Cron _recover_failed_chunks_for_long_plans registrado cada 15 min.")

    # [P0-D] Job nocturno: refrescar snapshots far-future (>48h)
    if not scheduler.get_job("nightly_refresh_far_future_snapshots"):
        scheduler.add_job(
            _nightly_refresh_all_pending_snapshots,
            CronTrigger(hour=3, minute=0, timezone=timezone.utc),
            id="nightly_refresh_far_future_snapshots",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info("⏰ [P0-D] Cron _nightly_refresh_all_pending_snapshots registrado a las 03:00 UTC.")

def _nightly_refresh_all_pending_snapshots() -> None:
    """[P0-D] Refresca diariamente snapshots de chunks que ejecutarán más allá del horizonte (48h).
    En planes largos, el chunk 10 (t+27d) no se refrescaba sino hasta 48h antes,
    quedando congelado con inventario de 3 semanas atrás si ocurría un fallo de red.
    """
    try:
        rows = execute_sql_query(
            """
            SELECT DISTINCT user_id::text, meal_plan_id::text
            FROM plan_chunk_queue
            WHERE status = 'pending'
              AND execute_after > NOW() + interval '48 hours'
            LIMIT 100
            """,
            fetch_all=True
        )
        
        if not rows:
            return
            
        refreshed_count = 0
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        
        for row in rows:
            uid = row.get("user_id")
            plan_id = row.get("meal_plan_id")
            if not uid or not plan_id:
                continue
                
            live_inventory = get_user_inventory_net(uid)
            if live_inventory is not None:
                try:
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET pipeline_snapshot = jsonb_set(
                                jsonb_set(
                                    pipeline_snapshot,
                                    '{form_data,userPantry}',
                                    %s::jsonb,
                                    true
                                ),
                                '{form_data,_pantry_captured_at}',
                                %s::jsonb,
                                true
                            ),
                            updated_at = NOW()
                        WHERE meal_plan_id = %s
                          AND status = 'pending'
                          AND execute_after > NOW() + interval '48 hours'
                        """,
                        (json.dumps(live_inventory, ensure_ascii=False), json.dumps(now_iso), plan_id)
                    )
                    refreshed_count += 1
                except Exception as inner_e:
                    logger.warning(f"[NIGHTLY REFRESH] Error actualizando plan {plan_id}: {inner_e}")
                    
        if refreshed_count > 0:
            logger.info(f"🌙 [P0-D] Nightly refresh completado: actualizados {refreshed_count} planes far-future.")
            
    except Exception as e:
        logger.error(f"❌ [NIGHTLY REFRESH] Error general: {e}")

def _proactive_refresh_pending_pantry_snapshots() -> None:
    """[P0-C] Refresca snapshots de inventario en chunks pending/stale antes de su execute_after.

    Solo targetea chunks cuyo _pantry_captured_at supera la mitad del TTL y que ejecutan
    dentro del horizonte (default 48h). Reusa _persist_fresh_pantry_to_chunks que ya
    propaga el live a todos los siblings vivos del mismo plan.
    """
    refresh_threshold_hours = CHUNK_PANTRY_SNAPSHOT_TTL_HOURS / 2.0
    horizon_hours = CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS
    max_users = CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS

    try:
        # Una fila por (user_id, meal_plan_id) con el chunk más urgente como ancla.
        # Filtramos por edad del snapshot dentro de SQL para no traer chunks ya frescos.
        candidates = execute_sql_query(
            """
            SELECT DISTINCT ON (user_id, meal_plan_id)
                id AS task_id,
                user_id::text AS user_id,
                meal_plan_id::text AS meal_plan_id,
                week_number,
                COALESCE(
                    (pipeline_snapshot->'form_data'->>'_pantry_captured_at')::timestamptz,
                    created_at
                ) AS captured_at
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND execute_after <= NOW() + make_interval(hours => %s)
              AND COALESCE(
                  (pipeline_snapshot->'form_data'->>'_pantry_captured_at')::timestamptz,
                  created_at
              ) < NOW() - make_interval(mins => %s)
            ORDER BY user_id, meal_plan_id, execute_after ASC
            LIMIT %s
            """,
            (horizon_hours, int(refresh_threshold_hours * 60), max_users * 4),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.warning(f"[P0-C/PROACTIVE] Error consultando candidatos: {e}")
        return

    if not candidates:
        return

    # Agrupar por user_id (una sola lectura live por usuario, sirve para todos sus planes).
    by_user: dict[str, list[dict]] = {}
    for row in candidates:
        by_user.setdefault(row["user_id"], []).append(row)
        if len(by_user) >= max_users:
            break

    refreshed_users = 0
    refreshed_plans = 0
    failed_users = 0

    for user_id, rows in by_user.items():
        try:
            live_inventory = get_user_inventory_net(user_id)
        except Exception as e:
            failed_users += 1
            logger.debug(f"[P0-C/PROACTIVE] Live fetch falló para {user_id}: {e}")
            continue

        if live_inventory is None:
            failed_users += 1
            continue

        refreshed_users += 1
        for row in rows:
            try:
                _persist_fresh_pantry_to_chunks(
                    row["task_id"], row["meal_plan_id"], list(live_inventory)
                )
                refreshed_plans += 1
            except Exception as e:
                logger.debug(
                    f"[P0-C/PROACTIVE] No se pudo propagar a plan {row['meal_plan_id']}: {e}"
                )

    logger.info(
        f"[P0-C/PROACTIVE] Refresh ejecutado: users_ok={refreshed_users} "
        f"users_failed={failed_users} plans_propagated={refreshed_plans} "
        f"(threshold={refresh_threshold_hours:.1f}h, horizon={horizon_hours}h)."
    )


def _refresh_chunk_pantry(
    user_id: str,
    form_data: dict,
    snapshot_form_data: dict | None = None,
    task_id: str | int | None = None,
    week_number: int | None = None,
) -> dict:
    """Refresh pantry from live inventory, falling back to the snapshot when needed.

    [P0-4] When using snapshot fallback, validates the snapshot age against
    CHUNK_PANTRY_SNAPSHOT_TTL_HOURS. If stale, attempts one extra live retry before
    accepting the stale data and marking source as 'stale_snapshot'.

    [P0-2] If live retry fails and snapshot age > TTL:
    - If age > CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS (24h): force generation in flexible mode.
    - Else: pause the chunk and request user action.
    """
    snapshot_form_data = snapshot_form_data or {}
    pantry_fallback = copy.deepcopy(snapshot_form_data.get("current_pantry_ingredients") or [])

    if not user_id or user_id == "guest":
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "guest"
        return form_data

    try:
        pantry_live = get_user_inventory_net(user_id)
        pantry_result = pantry_live if pantry_live is not None else pantry_fallback
        form_data["current_pantry_ingredients"] = pantry_result
        form_data["_fresh_pantry_source"] = "live"
        logger.debug(f"[P0-4/PANTRY] Inventario live OK para {user_id} ({len(pantry_result)} items).")
        return form_data
    except Exception as e:
        logger.warning(
            f"[P0-4/PANTRY] Error refrescando inventario vivo para chunk de {user_id}: {e}. "
            "Evaluando snapshot como fallback."
        )

    # [P0-4] El live falló. Antes de usar el snapshot, verificar su edad.
    captured_at_str = snapshot_form_data.get("_pantry_captured_at")
    snapshot_age_hours: float | None = None
    if captured_at_str:
        try:
            from constants import safe_fromisoformat
            captured_at = safe_fromisoformat(captured_at_str)
            if captured_at.tzinfo is None:
                captured_at = captured_at.replace(tzinfo=timezone.utc)
            snapshot_age_hours = (datetime.now(timezone.utc) - captured_at).total_seconds() / 3600
        except Exception:
            pass

    if snapshot_age_hours is not None and snapshot_age_hours > CHUNK_PANTRY_SNAPSHOT_TTL_HOURS:
        logger.warning(
            f"[P0-4/PANTRY] Snapshot de inventario para {user_id} tiene {snapshot_age_hours:.1f}h "
            f"(TTL={CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h). Intentando live retry adicional..."
        )
        try:
            pantry_live_retry = get_user_inventory_net(user_id)
            if pantry_live_retry is not None:
                form_data["current_pantry_ingredients"] = pantry_live_retry
                form_data["_fresh_pantry_source"] = "live"
                logger.info(
                    f"[P0-4/PANTRY] Live retry exitoso para {user_id} "
                    f"(snapshot tenía {snapshot_age_hours:.1f}h). {len(pantry_live_retry)} items."
                )
                return form_data
        except Exception as retry_e:
            logger.warning(f"[P0-4/PANTRY] Live retry también falló para {user_id}: {retry_e}.")

        # [P0-2] El live retry falló. Evaluar si pausamos o forzamos generación.
        if snapshot_age_hours > CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS:
            # Snapshot MUY viejo — forzar generación en modo flexible para no bloquear
            form_data["current_pantry_ingredients"] = pantry_fallback
            form_data["_fresh_pantry_source"] = "stale_snapshot_force"
            form_data["_pantry_flexible_mode"] = True
            form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
            logger.warning(
                f"[P0-2/FORCE] Generando con snapshot MUY viejo ({snapshot_age_hours:.1f}h > "
                f"{CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS}h) en modo flexible para no bloquear a {user_id}."
            )
            return form_data

        if task_id:
            # Snapshot vencido y live inaccesible — pausar chunk para que el usuario refresque
            logger.warning(
                f"[P0-2/STALE-PAUSE] Chunk {task_id} pausado para {user_id}: "
                f"snapshot {snapshot_age_hours:.1f}h > TTL {CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h, "
                f"live también falló."
            )
            _pause_chunk_for_stale_inventory(task_id, user_id, week_number or 0, snapshot_age_hours)
            form_data["_pantry_paused"] = True
            return form_data

        # Fallback si no hay task_id (no debería ocurrir en workers)
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "stale_snapshot"
        form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
    else:
        # Snapshot dentro del TTL (o sin timestamp) — aceptable
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "snapshot"
        if snapshot_age_hours is not None:
            logger.info(
                f"[P0-4/PANTRY] Usando snapshot ({snapshot_age_hours:.1f}h < TTL {CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h) "
                f"para {user_id}. {len(pantry_fallback)} items."
            )
        else:
            logger.info(f"[P0-4/PANTRY] Usando snapshot sin timestamp para {user_id}. {len(pantry_fallback)} items.")

    return form_data


def _calculate_inventory_drift(old_inv: list, new_inv: list) -> float:
    """Calcula el porcentaje de cambio (deriva) entre dos listas de inventario.
    
    [P1-D] Usa normalización de ingredientes para comparar bases reales. 
    Retorna un float entre 0.0 y 1.0 (drift percentage).
    """
    if not old_inv and not new_inv:
        return 0.0
    
    from constants import normalize_ingredient_for_tracking
    
    def _get_bases(inv_list):
        bases = set()
        for item in (inv_list or []):
            if not item: continue
            # Intentar extraer el nombre si es un string complejo (ej: "200g Pollo")
            try:
                from shopping_calculator import _parse_quantity
                _, _, name = _parse_quantity(item)
                norm = normalize_ingredient_for_tracking(name)
                if norm: bases.add(norm)
            except Exception:
                norm = normalize_ingredient_for_tracking(item)
                if norm: bases.add(norm)
        return bases

    old_bases = _get_bases(old_inv)
    new_bases = _get_bases(new_inv)
    
    if not old_bases and not new_bases:
        return 0.0
    if not old_bases: # Todo es nuevo
        return 1.0
        
    # Diferencia simétrica: items que están en uno pero no en otro
    diff = old_bases ^ new_bases
    # Referencia: tamaño del inventario original (mínimo 1 para evitar div/0)
    drift = len(diff) / max(len(old_bases), 1)
    return min(drift, 1.0)


def _get_dominant_technique(days: list) -> str:
    """Extrae la técnica más frecuente (moda) de una lista de días.
    
    [P1-E] Útil en Smart Shuffle para que 'last_technique' no se congele 
    y refleje el contenido real siendo servido.
    """
    if not days:
        return None
    
    from collections import Counter
    techs = []
    for d in days:
        if not isinstance(d, dict): continue
        # Priorizar _technique (interno) sobre technique (UI)
        t = d.get("_technique") or d.get("technique")
        if t:
            techs.append(t)
            
    if not techs:
        return None
        
    # Retornar la técnica más común
    return Counter(techs).most_common(1)[0][0]


def _filter_days_by_fresh_pantry(days: list, pantry_ingredients: list, min_match_ratio: float = 0.6) -> list:
    """Keep Smart Shuffle days whose base ingredients are mostly covered by the live pantry."""
    if not days:
        return []

    pantry_ingredients = pantry_ingredients or []
    if not pantry_ingredients:
        return list(days)

    from constants import normalize_ingredient_for_tracking

    pantry_bases = set()
    for item in pantry_ingredients:
        if not item:
            continue
        normalized = normalize_ingredient_for_tracking(item)
        if normalized:
            pantry_bases.add(normalized)
        pantry_bases.add(strip_accents(str(item).lower().strip()))

    if not pantry_bases:
        return list(days)

    filtered_days = []
    for day in days:
        if not isinstance(day, dict):
            continue

        unique_bases = []
        seen = set()
        for meal in day.get("meals", []) or []:
            if not isinstance(meal, dict):
                continue
            for raw_ing in meal.get("ingredients", []) or []:
                if not raw_ing:
                    continue
                base = normalize_ingredient_for_tracking(raw_ing) or strip_accents(str(raw_ing).lower().strip())
                if base and base not in seen:
                    seen.add(base)
                    unique_bases.append(base)

        if not unique_bases:
            filtered_days.append(day)
            continue

        matched = 0
        for base in unique_bases:
            if base in pantry_bases or any(pb and len(pb) > 2 and (pb in base or base in pb) for pb in pantry_bases):
                matched += 1

        coverage = matched / max(len(unique_bases), 1)
        if coverage >= min_match_ratio:
            filtered_days.append(day)

    return filtered_days


def _count_meaningful_pantry_items(pantry_ingredients: list) -> int:
    """Counts distinct, meaningful pantry bases to detect an empty or near-empty fridge."""
    pantry_ingredients = pantry_ingredients or []
    if not pantry_ingredients:
        return 0

    from constants import normalize_ingredient_for_tracking

    ignored_terms = {
        "", "agua", "sal", "pimienta", "aceite", "vinagre", "oregano", "cilantro",
        "canela", "sazon", "condimento",
    }

    normalized_items = set()
    for item in pantry_ingredients:
        if not item:
            continue
        normalized = normalize_ingredient_for_tracking(item) or strip_accents(str(item).lower().strip())
        if normalized and normalized not in ignored_terms and len(normalized) > 2:
            normalized_items.add(normalized)

    return len(normalized_items)


def _persist_fresh_pantry_to_chunks(
    task_id: str | int,
    meal_plan_id: str,
    fresh_inventory: list,
) -> None:
    """[P0-3] Propaga el inventario live recién leído al snapshot del chunk actual y de sus siblings.

    Sin esto, los siblings pending/stale conservan el snapshot capturado al crear el plan;
    si su live fetch falla, caen al fallback con datos de hace días. Aquí refrescamos esos
    snapshots para que el fallback sea, como mucho, tan viejo como el último chunk procesado
    con éxito y no la fecha de creación del plan.
    """
    if fresh_inventory is None:
        return
    captured_at = datetime.now(timezone.utc).isoformat()
    pantry_json = json.dumps(list(fresh_inventory), ensure_ascii=False)
    try:
        # Chunk actual: garantizado existir y estar en processing.
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET pipeline_snapshot = jsonb_set(
                    jsonb_set(
                        pipeline_snapshot,
                        '{form_data,current_pantry_ingredients}',
                        %s::jsonb,
                        true
                    ),
                    '{form_data,_pantry_captured_at}',
                    to_jsonb(%s::text),
                    true
                )
            WHERE id = %s
            """,
            (pantry_json, captured_at, task_id),
        )
        # Siblings vivos del mismo plan: pending y stale (no cancelled, completed, processing).
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET pipeline_snapshot = jsonb_set(
                    jsonb_set(
                        pipeline_snapshot,
                        '{form_data,current_pantry_ingredients}',
                        %s::jsonb,
                        true
                    ),
                    '{form_data,_pantry_captured_at}',
                    to_jsonb(%s::text),
                    true
                ),
                updated_at = updated_at
            WHERE meal_plan_id = %s
              AND id <> %s
              AND status IN ('pending', 'stale')
            """,
            (pantry_json, captured_at, str(meal_plan_id), task_id),
        )
        logger.info(
            f"[P0-3/PANTRY-PROP] Snapshot de inventario propagado a chunk {task_id} y siblings "
            f"del plan {meal_plan_id} ({len(fresh_inventory)} items, captured_at={captured_at})."
        )
    except Exception as prop_err:
        # Best-effort: si el UPDATE falla, el chunk actual igual usa el live en memoria.
        logger.warning(
            f"[P0-3/PANTRY-PROP] No se pudo propagar inventario fresco al snapshot "
            f"(plan {meal_plan_id}, chunk {task_id}): {prop_err}"
        )


def _pause_chunk_for_stale_inventory(
    task_id: str | int,
    user_id: str,
    week_number: int,
    snapshot_age_hours: float | None,
) -> None:
    """[P0-2] Pausa el chunk cuando el inventario en vivo no se pudo leer y el snapshot está vencido.

    Antes esto se aceptaba silenciosamente y se generaba contra datos viejos, rompiendo
    la promesa "los platos solo usan lo que hay en la nevera". Ahora congelamos el chunk
    hasta que el usuario refresque su nevera (mismo mecanismo de TTL/recordatorio que
    usa _pause_chunk_for_pantry_refresh, con un motivo distinto para telemetría).
    """
    pause_snapshot = execute_sql_query(
        "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
        (task_id,),
        fetch_one=True,
    )
    pause_snapshot = copy.deepcopy((pause_snapshot or {}).get("pipeline_snapshot") or {})
    if isinstance(pause_snapshot, str):
        pause_snapshot = json.loads(pause_snapshot)

    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
    # [P0-2] TTL específico para infraestructura caída (4h por defecto): el usuario no puede
    # accionar; conviene escalar rápido a flexible_mode si el live no se recupera.
    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
    pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS
    pause_snapshot["_pantry_pause_reason"] = "stale_snapshot"
    if snapshot_age_hours is not None:
        pause_snapshot["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)

    execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'pending_user_action',
            pipeline_snapshot = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
    )
    age_str = f" (snapshot {snapshot_age_hours:.1f}h)" if snapshot_age_hours is not None else ""
    logger.warning(
        f"[P0-2/STALE-PAUSE] Chunk {task_id} (Week {week_number}) pausado para {user_id}: "
        f"inventario live inaccesible y snapshot vencido{age_str}. "
        f"Recovery cron retentará live cada tick; escalará a flexible tras "
        f"{CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS}h."
    )
    # [P0-2] No enviamos push al usuario en stale_snapshot: la causa es server-side
    # (live fetch caído) y el usuario no puede accionar nada. El recovery cron resuelve solo.


def _pause_chunk_for_pantry_refresh(task_id: str | int, user_id: str, week_number: int, fresh_inventory: list, reason: str = None):
    """Pauses chunk generation when the live pantry is too empty to preserve the zero-waste promise."""
    pause_snapshot = execute_sql_query(
        "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
        (task_id,),
        fetch_one=True,
    )
    pause_snapshot = copy.deepcopy((pause_snapshot or {}).get("pipeline_snapshot") or {})
    if isinstance(pause_snapshot, str):
        pause_snapshot = json.loads(pause_snapshot)

    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
    if reason:
        pause_snapshot["_pantry_pause_reason"] = reason
    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PANTRY_EMPTY_TTL_HOURS
    pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS

    execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'pending_user_action',
            pipeline_snapshot = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
    )
    logger.warning(
        f"[P1-3/PANTRY] Chunk {week_number} pausado para {user_id}: inventario fresco insuficiente "
        f"({len(fresh_inventory or [])} items brutos)."
    )

    _dispatch_push_notification(
        user_id=user_id,
        title="Actualiza tu nevera para continuar",
        body="Tu próximo chunk quedó en pausa porque tu inventario está vacío o casi vacío. Actualiza 'Mi Nevera' y reintenta.",
        url="/dashboard",
    )


def _dispatch_push_notification(user_id: str, title: str, body: str, url: str = "/dashboard") -> None:
    try:
        import threading
        from utils_push import send_push_notification

        threading.Thread(
            target=send_push_notification,
            kwargs={
                "user_id": user_id,
                "title": title,
                "body": body,
                "url": url,
            },
            daemon=True,
        ).start()
    except Exception as push_err:
        logger.warning(f"[P1-3/PANTRY] No se pudo enviar push: {push_err}")


def _reconcile_chunk_reservations(user_id: str, chunk_id: str, days: list, max_retries: int = 3) -> None:
    """[P0-5] Reintenta las reservas faltantes de un chunk con reservation_status='partial'.

    Se llama de forma síncrona tras detectar reservas parciales. Itera hasta max_retries
    intentando reserve_plan_ingredients de nuevo. Si logra >= 50% de ingredientes parseables,
    marca reservation_status='ok'. Si no lo logra, lo deja en 'partial' y el bloqueo de 5 min
    en el pickup query protege al siguiente chunk de double-spend.
    """
    import time
    for _attempt in range(max_retries):
        try:
            reserved = reserve_plan_ingredients(user_id, chunk_id, days)
            from shopping_calculator import _parse_quantity
            _expected = 0
            for _d in days:
                for _m in (_d or {}).get('meals', []):
                    for _i in (_m.get('ingredients') or []):
                        if _i and len(str(_i).strip()) >= 3:
                            try:
                                _q, _u, _n = _parse_quantity(str(_i))
                                if _n and _q > 0:
                                    _expected += 1
                            except Exception:
                                pass
            _min_ok = max(1, int(_expected * 0.5))
            if reserved >= _min_ok:
                execute_sql_write(
                    "UPDATE plan_chunk_queue SET reservation_status = 'ok', updated_at = NOW() WHERE id = %s",
                    (chunk_id,)
                )
                logger.info(
                    f"[P0-5/RECONCILE] Reconciliación exitosa chunk {chunk_id}: "
                    f"{reserved}/{_expected} ingredientes reservados (intento {_attempt + 1})."
                )
                return
            logger.warning(
                f"[P0-5/RECONCILE] Intento {_attempt + 1}/{max_retries} parcial para chunk {chunk_id}: "
                f"{reserved}/{_expected}."
            )
        except Exception as e:
            logger.error(f"[P0-5/RECONCILE] Error en intento {_attempt + 1} chunk {chunk_id}: {e}")
        if _attempt < max_retries - 1:
            time.sleep(2)
    logger.error(
        f"[P0-5/RECONCILE] Reconciliación agotada para chunk {chunk_id} tras {max_retries} intentos. "
        f"reservation_status permanece 'partial'."
    )


def _should_pause_for_empty_pantry(
    fresh_inventory_source: str,
    fresh_inventory: list,
    snapshot: dict | None = None,
    form_data: dict | None = None,
) -> bool:
    snapshot = snapshot or {}
    form_data = form_data or {}
    flexible_mode = bool(snapshot.get("_pantry_flexible_mode") or form_data.get("_pantry_flexible_mode"))
    if flexible_mode:
        return False
    return (
        fresh_inventory_source == "live"
        and _count_meaningful_pantry_items(fresh_inventory) < CHUNK_MIN_FRESH_PANTRY_ITEMS
    )


def _recover_pantry_paused_chunks() -> None:
    """Revisa chunks en pending_user_action, recuerda al usuario y evita bloqueo indefinido."""
    try:
        paused_rows = execute_sql_query(
            """
            SELECT id, user_id, week_number, pipeline_snapshot,
                   EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS paused_seconds
            FROM plan_chunk_queue
            WHERE status = 'pending_user_action'
            ORDER BY updated_at ASC
            LIMIT 50
            """
        ) or []

        for row in paused_rows:
            snap = copy.deepcopy(row.get("pipeline_snapshot") or {})
            if isinstance(snap, str):
                snap = json.loads(snap)

            paused_seconds = int(row.get("paused_seconds") or 0)
            reminder_hours = int(snap.get("_pantry_pause_reminder_hours") or CHUNK_PANTRY_EMPTY_REMINDER_HOURS)
            ttl_hours = int(snap.get("_pantry_pause_ttl_hours") or CHUNK_PANTRY_EMPTY_TTL_HOURS)
            reminders_sent = int(snap.get("_pantry_pause_reminders") or 0)
            pause_reason = str(snap.get("_pantry_pause_reason") or "empty_pantry")
            user_id_str = str(row["user_id"])
            row_id = row["id"]
            week_num = row.get("week_number")

            # [P0-2] stale_snapshot: la causa es server-side (live fetch caído).
            # En cada tick (~15 min) intentamos un live-retry. Si pasa, refrescamos
            # los snapshots y desbloqueamos al instante; el usuario no recibe push.
            if pause_reason == "stale_snapshot":
                try:
                    fresh_inv = get_user_inventory_net(user_id_str)
                except Exception as live_e:
                    fresh_inv = None
                    logger.debug(f"[P0-2/STALE-RETRY] Live retry falló para {user_id_str}: {live_e}")

                if fresh_inv is not None:
                    # Live se recuperó. Persistimos al snapshot y re-encolamos sin escalar a flex.
                    try:
                        meal_plan_id_row = execute_sql_query(
                            "SELECT meal_plan_id FROM plan_chunk_queue WHERE id = %s",
                            (row_id,),
                            fetch_one=True,
                        )
                        mpid = meal_plan_id_row.get("meal_plan_id") if meal_plan_id_row else None
                        if mpid:
                            _persist_fresh_pantry_to_chunks(row_id, str(mpid), fresh_inv)
                    except Exception as prop_e:
                        logger.warning(f"[P0-2/STALE-RETRY] No se pudo propagar inv fresco: {prop_e}")

                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_pantry_pause_resolution"] = "live_recovered"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-2/STALE-RECOVERED] Chunk {week_num} reanudado: live fetch volvió "
                        f"tras {paused_seconds//60} min en stale_snapshot."
                    )
                    continue

                # Live sigue caído. Si superamos el TTL corto, escalamos a flexible.
                if paused_seconds >= ttl_hours * 3600:
                    degraded_snapshot = copy.deepcopy(snap)
                    degraded_snapshot["_degraded"] = True
                    degraded_snapshot["_pantry_flexible_mode"] = True
                    degraded_snapshot["_pantry_pause_resolution"] = "stale_snapshot_force_flex"
                    degraded_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            attempts = COALESCE(attempts, 0) + 1,
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.warning(
                        f"[P0-2/STALE-FORCE-FLEX] Chunk {week_num} expiró tras {ttl_hours}h en "
                        f"stale_snapshot sin recuperar live. Re-encolando en flexible_mode."
                    )
                # Antes del TTL: silencio. No spameamos push en pausa server-side.
                continue

            # empty_pantry / otros: mantenemos el comportamiento original (12h TTL + push).
            if paused_seconds >= ttl_hours * 3600:
                degraded_snapshot = copy.deepcopy(snap)
                degraded_snapshot["_degraded"] = True
                degraded_snapshot["_pantry_flexible_mode"] = True
                degraded_snapshot["_pantry_pause_resolution"] = "degraded_flexible_meal"
                degraded_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()

                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        attempts = COALESCE(attempts, 0) + 1,
                        execute_after = NOW(),
                        pipeline_snapshot = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                )
                logger.warning(
                    f"[P1-3/PANTRY] Chunk {week_num} expiró en pending_user_action. "
                    f"Re-encolando en modo flexible."
                )
                _dispatch_push_notification(
                    user_id=user_id_str,
                    title="Seguimos con un plato flexible",
                    body="Tu chunk seguía en pausa por nevera vacía. Lo reintentaremos con un plato flexible para no bloquear tu plan.",
                    url="/dashboard",
                )
                continue

            next_reminder_threshold = reminder_hours * 3600 * (reminders_sent + 1)
            if reminders_sent < CHUNK_PANTRY_EMPTY_MAX_REMINDERS and paused_seconds >= next_reminder_threshold:
                reminder_snapshot = copy.deepcopy(snap)
                reminder_snapshot["_pantry_pause_reminders"] = reminders_sent + 1
                reminder_snapshot["_pantry_pause_last_reminder_at"] = datetime.now(timezone.utc).isoformat()
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(reminder_snapshot, ensure_ascii=False), row_id),
                )
                _dispatch_push_notification(
                    user_id=user_id_str,
                    title="Tu chunk sigue esperando tu nevera",
                    body="Actualiza 'Mi Nevera' para continuar con el siguiente bloque del plan. Si no, usaremos una opción flexible más adelante.",
                    url="/dashboard",
                )
    except Exception as e:
        logger.warning(f"[P1-3/PANTRY] Error recuperando chunks pausados por pantry vacío: {e}")


def _recover_failed_chunks_for_long_plans() -> None:
    """
    [P0-4] Cron task que detecta chunks failed/dead_lettered en cualquier plan (7d+)
    con días remanentes y los re-encola con chunk_kind="catchup".
    Garantiza que el plan se complete aunque el usuario no abra la app.

    Antes el umbral era 15d+; planes de 7d quedaban sin recuperación y un fallo
    del chunk 2 dejaba al usuario con solo 3 de los 7 días generados.
    """
    try:
        # 1. Buscar chunks failed en cualquier plan (>= 7 días, mínimo soportado)
        # que aún no haya expirado: fecha_inicio + total_dias > NOW()
        failed_candidates = execute_sql_query("""
            SELECT
                q.id, q.user_id, q.meal_plan_id, q.week_number, q.days_offset, q.days_count, q.pipeline_snapshot,
                (p.plan_data->>'total_days_requested')::int as total_days,
                (p.plan_data->>'grocery_start_date')::text as start_date_iso
            FROM plan_chunk_queue q
            JOIN meal_plans p ON q.meal_plan_id = p.id
            WHERE q.status = 'failed'
              AND (p.plan_data->>'total_days_requested')::int >= 7
              AND (
                  (p.plan_data->>'grocery_start_date')::timestamptz +
                  ((p.plan_data->>'total_days_requested')::int * interval '1 day')
              ) > NOW()
            ORDER BY q.updated_at ASC
            LIMIT 20
        """, fetch_all=True) or []

        if not failed_candidates:
            return

        for row in failed_candidates:
            task_id = row['id']
            user_id = row['user_id']
            plan_id = row['meal_plan_id']
            week_number = row['week_number']
            days_offset = row['days_offset']
            days_count = row['days_count']
            snapshot = row['pipeline_snapshot']
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)

            logger.info(
                f"🔄 [P0-4/CATCHUP] Recuperando chunk failed (id={task_id}) "
                f"week={week_number} para plan {plan_id} (total_days={row.get('total_days')}). "
                f"Re-encolando como catchup..."
            )

            # Re-encolar usando la lógica centralizada que ya maneja el UPDATE de failed records.
            # _enqueue_plan_chunk reseteará attempts=0, status='pending' y pondrá execute_after inmediato.
            _enqueue_plan_chunk(
                user_id=str(user_id),
                meal_plan_id=str(plan_id),
                week_number=week_number,
                days_offset=days_offset,
                days_count=days_count,
                pipeline_snapshot=snapshot,
                chunk_kind="catchup"
            )

        # [P1-3] Detección y escalado proactivo de chunks atrasados (lag > 24h)
        # Se corre aquí para centralizar la recuperación y respetar el intervalo de 15 min.
        _detect_and_escalate_stuck_chunks()

    except Exception as e:
        logger.error(f"❌ [P1-B/CATCHUP] Error en cron de recuperación: {e}")


def _compute_chunk_retry_delay_minutes(next_attempt: int, is_critical: bool = False) -> int:
    """Backoff exponencial acotado para retries automáticos del worker."""
    if is_critical:
        return CHUNK_RETRY_CRITICAL_MINUTES
    exponent = max(0, next_attempt - 1)
    return min(CHUNK_RETRY_BASE_MINUTES * (2 ** exponent), 12 * 60)





def calculate_meal_success_scores(user_id: str, days_back: int = 14):
    """Calcula que platos del historial tuvieron mayor adherencia."""
    days_back_iso = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    
    plans = get_recent_plans(user_id, days=days_back)
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)
    
    def normalize(text):
        if not text: return ""
        return strip_accents(text.lower()).strip()
        
    consumed_names_list = [normalize(m.get('meal_name', '')) for m in consumed]
    consumed_names = set(consumed_names_list)
    
    consumed_counts = {}
    for name in consumed_names_list:
        if name:
            consumed_counts[name] = consumed_counts.get(name, 0) + 1
            
    frequent_meals = [m[0] for m in sorted(consumed_counts.items(), key=lambda x: x[1], reverse=True)]
    
    scores = {}
    for plan in plans:
        if not isinstance(plan, dict): continue
        for day in plan.get('days', []):
            if not isinstance(day, dict): continue
            for meal in day.get('meals', []):
                if not isinstance(meal, dict): continue
                name = normalize(meal.get('name', ''))
                if name:
                    scores[name] = {
                        'was_eaten': name in consumed_names,
                        'meal_type': meal.get('meal'),
                        'technique': meal.get('technique', ''),
                    }
                
    successful_techniques = [s['technique'] for s in scores.values() if s['was_eaten'] and s['technique']]
    abandoned_techniques = [s['technique'] for s in scores.values() if not s['was_eaten'] and s['technique']]
    
    return successful_techniques, abandoned_techniques, frequent_meals

def calculate_ingredient_fatigue(user_id: str, days_back: int = 14, tuning_metrics: dict = None):
    """
    Mejora 3 y GAP 5: Calcula la monotonia de ingredientes y categorias nutricionales 
    en los ultimos dias usando decaimiento temporal y NLP.
    Retorna ingredientes y categorias con alta frecuencia de aparicion (fatiga cruzada).
    """
    from constants import normalize_ingredient_for_tracking, get_nutritional_category
    from collections import defaultdict
    import ast
    
    now = datetime.now(timezone.utc)
    days_back_iso = (now - timedelta(days=days_back)).isoformat()
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)
    
    if not consumed:
        return {"score": 0.0, "fatigued_ingredients": []}
        
    ingredient_counts = defaultdict(float)
    category_counts = defaultdict(float)
    total_weight = 0.0
    
    for meal in consumed:
        # 1. Temporal Decay
        created_at_str = meal.get('created_at')
        days_ago = 0
        if created_at_str:
            try:
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(created_at_str)
                days_ago = max(0, (now - dt).days)
            except Exception:
                pass
                
        base_decay = tuning_metrics.get("fatigue_decay", 0.9) if tuning_metrics else 0.9
        decay_weight = base_decay ** days_ago
        total_weight += decay_weight
        
        # 2. Extraer ingredientes
        ingredients_raw = meal.get('ingredients')
        ing_list = []
        if isinstance(ingredients_raw, list):
            ing_list = ingredients_raw
        elif isinstance(ingredients_raw, str):
            try:
                ing_list = ast.literal_eval(ingredients_raw)
                if not isinstance(ing_list, list):
                    ing_list = [ingredients_raw]
            except Exception:
                ing_list = [i.strip() for i in ingredients_raw.split(',')]
                
        # 3. Normalizacion Avanzada (Constants.py NLP)
        for ing in ing_list:
            if not isinstance(ing, str) or not ing.strip():
                continue
            normalized = normalize_ingredient_for_tracking(ing)
            # Solo guardamos cosas que lograron normalizarse a bases reales (evitar ruido como 'agua')
            if normalized:
                ingredient_counts[normalized] += decay_weight
                category = get_nutritional_category(normalized)
                if category:
                    category_counts[category] += decay_weight
                
    fatigued_items = []
    # Umbrales: Ingrediente individual > 4.0 o > 35%. Categoria entera > 6.0 o > 45%.
    for ing, weight in ingredient_counts.items():
        if weight >= 4.0 or (total_weight > 0 and (weight / total_weight) > 0.35):
            fatigued_items.append(ing)
            
    for cat, weight in category_counts.items():
        if weight >= 6.0 or (total_weight > 0 and (weight / total_weight) > 0.45):
            fatigued_items.append(f"[CATEGORÃA] {cat}")
            
    # Set completo de ingredientes normalizados consumidos — usado para auto-tune cross-ciclo
    consumed_ingredient_set = set(ingredient_counts.keys())

    fatigue_score = min(1.0, len(fatigued_items) * 0.2)

    return {
        "score": round(fatigue_score, 2),
        "fatigued_ingredients": fatigued_items,
        "consumed_ingredient_set": consumed_ingredient_set,
    }


def calculate_day_of_week_adherence(user_id: str, days_back: int = 30):
    """
    Calcula un perfil de adherencia predictivo usando EMA (Exponential Moving Average)
    para cada dia de la semana. Detecta patrones emergentes de abandono.
    """
    days_back_iso = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)
    
    # Agrupar comidas por fecha especifica (YYYY-MM-DD)
    date_counts = {}
    for meal in consumed:
        created_at_str = meal.get('created_at')
        if created_at_str:
            try:
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(created_at_str)
                d_str = dt.date().isoformat()
                date_counts[d_str] = date_counts.get(d_str, 0) + 1
            except Exception:
                pass
                
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Si no hay datos, asumimos 100% de adherencia temporal
    if not date_counts:
        return {day_names[i]: 1.0 for i in range(7)}
        
    # Iterar explicitamente desde el primer dia registrado para incluir dias en 0
    first_tracking_date = min(date_counts.keys())
    first_date_dt = datetime.fromisoformat(first_tracking_date).date()
    now = datetime.now(timezone.utc).date()
    
    date_list = []
    current_d = first_date_dt
    while current_d <= now:
        date_list.append(current_d)
        current_d += timedelta(days=1)
        
    weekday_history = {i: [] for i in range(7)}
    for d in date_list:
        d_str = d.isoformat()
        count = date_counts.get(d_str, 0)
        weekday_history[d.weekday()].append(count)
        
    global_max_day_count = max(date_counts.values()) if date_counts else 3
    if global_max_day_count == 0:
        global_max_day_count = 3
        
    alpha = 0.4 # Factor EMA: 40% peso al evento mas reciente
    day_ema = {i: 0.0 for i in range(7)}
    
    for i in range(7):
        history = weekday_history[i] 
        if not history:
            day_ema[i] = 1.0 # No hay dias registrados para este weekday
            continue
            
        ema = None
        for count in history:
            ratio = min(count / global_max_day_count, 1.0)
            if ema is None:
                ema = ratio
            else:
                ema = (alpha * ratio) + ((1 - alpha) * ema)
                
        day_ema[i] = ema if ema is not None else 1.0
        
    return {day_names[k]: round(v, 2) for k, v in day_ema.items()}


def calculate_meal_level_adherence(user_id: str, plan_days: list, consumed_records: list, household_size: int = 1):
    """Calcula adherencia por tipo de comida (desayuno, almuerzo, cena, merienda)."""
    meal_type_stats = {}
    
    # Asumimos que el primer dia del plan representa la estructura diaria esperada
    days_to_check = [plan_days[0]] if plan_days else []
    
    for day in days_to_check:
        for meal in day.get('meals', []):
            mt = meal.get('meal', 'otro').lower()
            if mt not in meal_type_stats:
                meal_type_stats[mt] = {'planned_per_day': 0, 'eaten': 0}
            meal_type_stats[mt]['planned_per_day'] += 1
            
    # Contar comidas ingeridas
    unique_dates = set()
    for record in consumed_records:
        mt = record.get('meal_type', 'otro').lower()
        if mt in meal_type_stats:
            meal_type_stats[mt]['eaten'] += 1
        if 'created_at' in record:
            unique_dates.add(str(record['created_at'])[:10])
            
    days_passed = max(1, len(unique_dates))
            
    return {
        mt: min(1.0, round((s['eaten'] / household_size) / max(s['planned_per_day'] * days_passed, 1), 2))
        for mt, s in meal_type_stats.items()
    }


def calculate_plan_quality_score(user_id: str, plan_data: dict, consumed_records: list, household_size: int = 1) -> float:
    """
    Mejora 4: Evalua retrospectivamente la calidad del plan midiendo satisfaccion real y retencion.
    """
    total_meals = sum(len(d.get('meals', [])) for d in plan_data.get('days', []))
    eaten_meals = len(consumed_records) / household_size
    
    # 1. Adherencia bruta
    adherence = eaten_meals / max(total_meals, 1)
    
    # 2. Diversidad: ¿cuantos ingredientes distintos se comieron?
    eaten_ingredients = set()
    
    def normalize(text):
        if not text: return ""
        return strip_accents(text.lower()).strip()
        
    for r in consumed_records:
        ing_list = r.get('ingredients', [])
        if isinstance(ing_list, list):
            for ing in ing_list:
                if isinstance(ing, dict) and "name" in ing:
                    eaten_ingredients.add(normalize(ing["name"]))
                elif isinstance(ing, dict) and "display_string" in ing:
                    eaten_ingredients.add(normalize(ing["display_string"]))
                elif isinstance(ing, str):
                    eaten_ingredients.add(normalize(ing))
                    
    diversity_score = min(1.0, len(eaten_ingredients) / 5)
    
    # 3. Satisfaccion explicita (Likes vs Rejections)
    likes = get_user_likes(user_id)
    rejections = get_active_rejections(user_id)
    
    # Calculamos un ratio de satisfaccion neto normalizado
    # Partimos de un 0.5 (neutral). Likes suman, rechazos restan.
    net_satisfaction = len(likes) - len(rejections)
    satisfaction_score = max(0.0, min(1.0, 0.5 + (net_satisfaction / max(total_meals, 1) * 0.5)))
    rejection_penalty = len(rejections) * 0.05
    
    # 4. Retention Signal (¿Sigue activo el usuario?)
    from db_facts import get_consumed_meals_since
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    # Verificamos si ha registrado comidas en las ultimas 48 horas como seÃ±al de retencion
    recently_consumed = get_consumed_meals_since(user_id, since_iso_date=(now - timedelta(days=2)).isoformat())
    retention_score = 1.0 if recently_consumed else 0.3
    
    # Weighted score final
    quality = round(
        (adherence * 0.35 + 
         diversity_score * 0.20 + 
         satisfaction_score * 0.25 + 
         retention_score * 0.20) - rejection_penalty, 
        2
    )
    
    return max(0.0, min(1.0, quality))


def get_similar_user_patterns(user_id: str, health_profile: dict):
    """Para usuarios sin historial, busca que funciono para perfiles similares (Mejora 4)."""
    goal = health_profile.get('mainGoal')
    activity = health_profile.get('activityLevel')
    diet_types = health_profile.get('dietTypes', [])
    country = health_profile.get('country')
    
    if not goal or not activity:
        return []
        
    try:
        # 1. Recuperar rechazos activos para post-filtrado
        rejections = get_active_rejections(user_id)
        
        query = """
            SELECT meal_name, COUNT(*) as popularity
            FROM consumed_meals cm
            JOIN user_profiles up ON cm.user_id = up.id
            WHERE up.health_profile->>'mainGoal' = %s
            AND up.health_profile->>'activityLevel' = %s
            AND cm.created_at > NOW() - INTERVAL '30 days'
        """
        params = [goal, activity]
        
        # 2. Segmentacion de Dieta
        if diet_types and len(diet_types) > 0:
            import json
            diet = diet_types[0]
            query += " AND up.health_profile->'dietTypes' @> %s::jsonb"
            params.append(json.dumps([diet]))
            
        # 3. Segmentacion Cultural
        if country:
            query += " AND up.health_profile->>'country' = %s"
            params.append(country)
            
        query += " GROUP BY meal_name ORDER BY popularity DESC LIMIT 20"
        
        popular_meals = execute_sql_query(query, tuple(params), fetch_all=True)
        
        if not popular_meals:
            return []
            
        # 4. Post-filtro de Calidad (Excluir platos rechazados por el usuario)
        if rejections:
            def normalize(text):
                if not text: return ""
                return strip_accents(str(text).lower()).strip()
                
            normalized_rejections = [normalize(r.get('ingredient', '')) for r in rejections]
            
            filtered_meals = []
            for pm in popular_meals:
                meal_name_norm = normalize(pm.get('meal_name', ''))
                is_rejected = any(rej in meal_name_norm for rej in normalized_rejections if rej)
                
                if not is_rejected:
                    filtered_meals.append(pm)
            
            return filtered_meals[:10]
            
        return popular_meals[:10]
        
    except Exception as e:
        if "relation" not in str(e).lower():
            logger.warning(f" [COLD-START] Error en query similar users: {e}")
        return []
        

def _build_facts_memory_context(user_id: str) -> str:
    """
    Construye un string de contexto de memoria a partir de los hechos (facts)
    aprendidos por la IA sobre el usuario. Esto permite que la rotacion nocturna
    sepa cosas como "le gusta el pollo", "es intolerante a la lactosa", etc.
    """
    try:
        facts = get_all_user_facts(user_id)
        if not facts:
            return ""
        
        # Priorizar por categoria para que alergias/condiciones medicas aparezcan primero
        CATEGORY_ORDER = {
            "alergia": 0, "condicion_medica": 1, "rechazo": 2,
            "dieta": 3, "objetivo": 4, "preferencia": 5, "sintoma_temporal": 6
        }
        
        def sort_key(f):
            meta = f.get("metadata", {})
            cat = meta.get("category", "") if isinstance(meta, dict) else ""
            return CATEGORY_ORDER.get(cat, 7)
        
        facts_sorted = sorted(facts, key=sort_key)
        
        # Construir el contexto legible (maximo 15 facts para no saturar el prompt)
        fact_lines = []
        for f in facts_sorted[:15]:
            fact_text = f.get("fact", "")
            meta = f.get("metadata", {})
            cat = meta.get("category", "general") if isinstance(meta, dict) else "general"
            if fact_text:
                fact_lines.append(f"â€¢ [{cat.upper()}] {fact_text}")
        
        if not fact_lines:
            return ""
        
        return (
            "\n\n--- MEMORIA DEL CEREBRO IA (HECHOS APRENDIDOS SOBRE ESTE USUARIO) ---\n"
            "DEBES respetar OBLIGATORIAMENTE esta informacion al generar el plan:\n"
            + "\n".join(fact_lines)
            + "\n--------------------------------------------------------------------"
        )
    except Exception as e:
        logger.warning(f" [CRON] Error building facts memory context for {user_id}: {e}")
        return ""


def _persist_nightly_learning_signals(user_id: str, health_profile: dict, days: list, consumed_records: list):
    """Persiste señales de aprendizaje en health_profile durante planes largos de forma segura.
    
    [P1-5 FIX] Previene condiciones de carrera cuando múltiples chunks de diferentes
    planes activos del mismo usuario escriben simultáneamente. Utiliza un SELECT FOR UPDATE
    atómico para un read-modify-write estricto, dejando el LLM fuera del lock para evitar deadlocks.
    """
    from db_core import execute_sql_query, connection_pool
    import json
    from datetime import datetime, timezone, timedelta
    import psycopg
    from psycopg.rows import dict_row

    # 1. Ejecutar procesos pesados y propensos a red FUERA del lock
    run_retro = False
    last_retro_str = health_profile.get("last_llm_retrospective_date")
    if not last_retro_str:
        run_retro = True
    else:
        try:
            if last_retro_str.endswith('Z'):
                last_retro_str = last_retro_str[:-1] + '+00:00'
            last_retro_date = datetime.fromisoformat(last_retro_str)
            if last_retro_date.tzinfo is None:
                last_retro_date = last_retro_date.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_retro_date).days >= 7:
                run_retro = True
        except Exception as dt_err:
            logger.warning(f" [LLM-as-Judge] Error parseando last_retro_date: {dt_err}")
            run_retro = True

    llm_retro = None
    liked_flavor_profiles = None
    if run_retro:
        try:
            from ai_helpers import generate_llm_retrospective, extract_liked_flavor_profiles
            from db import get_user_likes, get_active_rejections
            
            recent_likes = get_user_likes(user_id)
            recent_rejections = get_active_rejections(user_id=user_id, session_id=None)
            
            llm_retro = generate_llm_retrospective(
                user_id=user_id, plan_data={'days': days},
                consumed_records=consumed_records, recent_likes=recent_likes,
                recent_rejections=recent_rejections
            )
            liked_flavor_profiles = extract_liked_flavor_profiles(recent_likes)
        except Exception as e:
            logger.error(f" [LLM-as-Judge] Error en el flujo offline: {e}")

    # 2. Bloque transaccional estricto (Read-Modify-Write atómico)
    if not connection_pool:
        logger.error(" [NIGHTLY LEARN] Error: connection_pool no está disponible.")
        return

    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    # Adquirir lock exclusivo de fila
                    cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s FOR UPDATE", (user_id,))
                    row = cursor.fetchone()
                    if not row:
                        return
                        
                    fresh_profile = row['health_profile'] or {}
                    
                    try:
                        validated_profile = HealthProfileSchema(**fresh_profile)
                    except Exception:
                        validated_profile = HealthProfileSchema()

                    # Adherencia
                    expected_meals_count = len(days[0].get('meals', [])) if days else 0
                    if expected_meals_count > 0:
                        current_cycle_adherence = calculate_meal_level_adherence(user_id, days, consumed_records)
                        is_weekend = (datetime.now(timezone.utc) - timedelta(hours=12)).weekday() >= 5
                        profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
                        
                        historical_adherence = getattr(validated_profile, profile_key, {})
                        alpha = 0.3
                        smoothed_adherence = {}
                        all_meal_types = set(list(current_cycle_adherence.keys()) + list(historical_adherence.keys()))
                        
                        for mt in all_meal_types:
                            current_val = current_cycle_adherence.get(mt, 1.0)
                            hist_val = historical_adherence.get(mt, 1.0)
                            smoothed_adherence[mt] = round((alpha * current_val) + ((1 - alpha) * hist_val), 2)
                            
                        avg_adherence = round(sum(smoothed_adherence.values()) / max(1, len(smoothed_adherence)), 2)
                        
                        adherence_history = fresh_profile.get('adherence_history_rotations', [])
                        if not isinstance(adherence_history, list): adherence_history = []
                        adherence_history.append(avg_adherence)
                        
                        fresh_profile[profile_key] = smoothed_adherence
                        fresh_profile['adherence_history_rotations'] = adherence_history[-5:]

                    # Quality Score
                    quality_score = calculate_plan_quality_score(user_id, {'days': days}, consumed_records)
                    quality_history = getattr(validated_profile, 'quality_history_rotations', [])
                    if not isinstance(quality_history, list): quality_history = []
                    quality_history.append(quality_score)
                    
                    fresh_profile['last_plan_quality'] = quality_score
                    fresh_profile['quality_history_rotations'] = quality_history[-5:]

                    # Snapshot de peso
                    cursor.execute("SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,))
                    weight_row = cursor.fetchone()
                    if weight_row:
                        fresh_profile['latest_weight_snapshot'] = {
                            "date": str(weight_row['date']), 
                            "weight": weight_row['weight'], 
                            "unit": weight_row.get('unit', 'lb')
                        }

                    # Retro LLM
                    if run_retro and (llm_retro or liked_flavor_profiles):
                        if llm_retro:
                            fresh_profile['llm_retrospective'] = llm_retro
                            fresh_profile['last_llm_retrospective_date'] = datetime.now(timezone.utc).isoformat()
                        if liked_flavor_profiles:
                            fresh_profile['liked_flavor_profiles'] = liked_flavor_profiles

                    # Actualización final unificada
                    cursor.execute(
                        "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s",
                        (json.dumps(fresh_profile), user_id)
                    )
                    logger.info(f" [NIGHTLY LEARN] Señales de aprendizaje guardadas atómicamente para {user_id}")
    except Exception as e:
        logger.error(f" [NIGHTLY LEARN] Error en la persistencia transaccional: {e}")



def _refill_emergency_backup_plan(user_id: str, pipeline_data: dict, taste_profile: str, memory_context: str):
    """Genera asincronamente un plan de respaldo si el usuario no tiene suficientes dias en cache."""
    logger.info(f" [CRON:REFILL] Checking emergency backup cache for user {user_id}...")
    try:
        from db_core import execute_sql_query, execute_sql_write
        user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if not user_res:
            return
            
        health_profile = user_res.get("health_profile", {})
        try:
            validated_profile = HealthProfileSchema(**health_profile)
        except Exception as e:
            logger.warning(f" [CRON:REFILL] health_profile malformado para {user_id}. Usando defaults: {e}")
            validated_profile = HealthProfileSchema()
            
        backup_plan = validated_profile.emergency_backup_plan
        
        if isinstance(backup_plan, list) and len(backup_plan) >= 3:
            logger.info(f" [CRON:REFILL] Cache full for user {user_id} ({len(backup_plan)} days). Skipping generation.")
            return
            
        logger.info(f" [CRON:REFILL] Cache low/empty for user {user_id}. Generating 3-day emergency backup...")
        
        # Override to ensure it's diverse and basic just in case
        pipeline_data["_is_emergency_generation"] = True
        
        emergency_memory = memory_context + "\n[INSTRUCCION CRITICA: Este es un plan de EMERGENCIA de respaldo. Asegurate de que las comidas sean seguras, sencillas de preparar, y muy amigables con las restricciones del usuario.]"
        
        succ_techs = pipeline_data.get('successful_techniques', [])
        freq_meals = pipeline_data.get('frequent_meals', [])
        
        if succ_techs or freq_meals:
            emergency_memory += "\n\n[SEMILLA DE ADHERENCIA: Utiliza OBLIGATORIAMENTE las siguientes tecnicas y platos recurrentes porque el usuario tiene 90%+ de probabilidad de adherirse a ellos.]"
            if succ_techs:
                emergency_memory += f"\n- Tecnicas de alta adherencia: {', '.join(succ_techs[:5])}"
            if freq_meals:
                emergency_memory += f"\n- Platos recurrentes: {', '.join(freq_meals[:5])}"
        
        from graph_orchestrator import run_plan_pipeline
        result = run_plan_pipeline(pipeline_data, [], taste_profile, emergency_memory, None, None)
        
        if 'error' not in result and 'days' in result and isinstance(result['days'], list) and len(result['days']) > 0:
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
                (json.dumps(result['days']), user_id)
            )
            logger.info(f"âœ… ðŸ›¡ï¸ [CRON:REFILL] Successfully generated and saved {len(result['days'])} backup days for user {user_id}.")
        else:
            logger.warning(f" [CRON:REFILL] Failed to generate emergency backup for {user_id}: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f" [CRON:REFILL] Exception during emergency backup generation for {user_id}: {e}")


# [GAP 6] Sembrado inmediato del emergency_backup_plan sin llamar al LLM.
# Llamado desde routers/plans.py justo despues de persistir el chunk 1, asi:
#   - Si chunk 2+ cae en Smart Shuffle antes de la primera rotacion nocturna,
#     el pool de respaldo no esta vacio.
#   - Idempotente: solo siembra si el backup actual tiene <3 dias.
def _seed_emergency_backup_if_empty(user_id: str, fresh_days):
    try:
        if not user_id or not isinstance(fresh_days, list) or len(fresh_days) == 0:
            return
        user_res = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True
        )
        if not user_res:
            return
        hp = user_res.get("health_profile") or {}
        current = hp.get("emergency_backup_plan") or []
        if isinstance(current, list) and len(current) >= 3:
            return  # Ya hay backup suficiente, no pisar

        import copy, json as _json
        seed = []
        for d in fresh_days[:3]:
            if not isinstance(d, dict) or not d.get("meals"):
                continue
            d_copy = copy.deepcopy(d)
            d_copy.pop("_is_degraded_shuffle", None)
            d_copy.pop("_mutated", None)
            seed.append(d_copy)
        if not seed:
            return

        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(COALESCE(health_profile, '{}'::jsonb), '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
            (_json.dumps(seed, ensure_ascii=False), user_id)
        )
        logger.info(f"[GAP6:SEED] emergency_backup_plan sembrado con {len(seed)} dias para user {user_id}.")
    except Exception as e:
        logger.warning(f"[GAP6:SEED] No se pudo sembrar emergency_backup_plan para {user_id}: {e}")


def _inject_advanced_learning_signals(user_id: str, pipeline_data: dict, health_profile: dict, days: list, consumed_records: list, days_since_last_chunk: int = 0):
    """Extrae las señales avanzadas de aprendizaje y las inyecta en pipeline_data."""
    from db_core import execute_sql_query, execute_sql_write
    from datetime import datetime, timezone, timedelta

    # [P1-4] La ventana de señales se basa en la distancia temporal real al chunk anterior
    # (days_since_last_chunk = days_offset del chunk actual), no en chunk_kind.
    # Ejemplo: chunk 2 de plan 30d tiene offset=3 → ventana de 4 días, igual que rolling refill.
    # Chunk 8 de plan 30d tiene offset=21 → ventana de 14 días (capped).
    # El lower bound (PLAN_CHUNK_SIZE+1) evita ventanas de 0-1 días en el primer chunk.
    from constants import PLAN_CHUNK_SIZE
    _days_elapsed = max(PLAN_CHUNK_SIZE + 1, int(days_since_last_chunk or 0))
    _fatigue_days_back = min(_days_elapsed, 14)
    _success_days_back = min(_days_elapsed, 14)
    _adherence_days_back = min(_days_elapsed, 30)
    logger.info(f"[P1-4] Ventana de aprendizaje por distancia temporal: days_elapsed={_days_elapsed} → fatigue/success={_fatigue_days_back}d adherencia={_adherence_days_back}d")

    # MEJORA 5: Ingredient Fatigue Detection
    fatigue_data = None
    try:
        tuning_metrics = health_profile.get("tuning_metrics", {})
        fatigue_data = calculate_ingredient_fatigue(user_id, days_back=_fatigue_days_back, tuning_metrics=tuning_metrics)
        if fatigue_data and fatigue_data.get('fatigued_ingredients'):
            pipeline_data['fatigued_ingredients'] = fatigue_data['fatigued_ingredients']
            logger.info(f"[CRON] Ingredient Fatigue detected for {user_id}: {fatigue_data['fatigued_ingredients']}")
    except Exception as e:
        logger.error(f"[CRON] Error calculating ingredient fatigue: {e}")

    # MEJORA 4: Auto-Tuning del fatigue_decay mediante tasa de falsos positivos cross-ciclo
    # Compara ingredientes "fatigados" en el ciclo anterior vs lo que el usuario comio en este ciclo.
    # Alto solapamiento = falso positivo = el decay era demasiado lento -> olvidar mas rapido.
    try:
        tuning_metrics = health_profile.get("tuning_metrics", {})
        fatigue_decay = tuning_metrics.get("fatigue_decay", 0.9)
        last_fatigued = health_profile.get("last_fatigued_ingredients", [])
        current_consumed = fatigue_data.get("consumed_ingredient_set", set()) if fatigue_data else set()

        if last_fatigued and current_consumed:
            fp_count = sum(1 for f in last_fatigued if f in current_consumed)
            fp_rate = round(fp_count / len(last_fatigued), 3)

            fp_history = tuning_metrics.get("fatigue_fp_history", [])
            fp_history.append(fp_rate)
            fp_history = fp_history[-3:]

            if len(fp_history) >= 2:
                mean_fp = sum(fp_history) / len(fp_history)
                if mean_fp > 0.6 and fatigue_decay > 0.70:
                    fatigue_decay = round(fatigue_decay - 0.03, 2)
                    logger.info(f"[FATIGUE-TUNE] Alta tasa FP ({mean_fp:.2f}): fatigue_decay -> {fatigue_decay} (olvidar mas rapido).")
                elif mean_fp < 0.2 and fatigue_decay < 0.98:
                    fatigue_decay = round(fatigue_decay + 0.02, 2)
                    logger.info(f"[FATIGUE-TUNE] Baja tasa FP ({mean_fp:.2f}): fatigue_decay -> {fatigue_decay} (memoria mas larga).")

            tuning_metrics["fatigue_fp_history"] = fp_history
            tuning_metrics["fatigue_decay"] = fatigue_decay
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb) WHERE id = %s",
                (json.dumps(tuning_metrics), user_id)
            )
            health_profile["tuning_metrics"] = tuning_metrics
            logger.info(f"[FATIGUE-TUNE] decay={fatigue_decay}, fp_rate={fp_rate}, history={fp_history}")

        # Guardar fatigued de ESTE ciclo (solo ingredientes individuales) para el proximo ciclo
        current_fatigued_pure = [
            f for f in (fatigue_data.get("fatigued_ingredients") or [])
            if fatigue_data and not f.startswith("[CATEGOR")
        ]
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_fatigued_ingredients}', %s::jsonb) WHERE id = %s",
            (json.dumps(current_fatigued_pure), user_id)
        )
        health_profile["last_fatigued_ingredients"] = current_fatigued_pure
    except Exception as e:
        logger.error(f"[FATIGUE-TUNE] Error en auto-tuning de fatigue_decay: {e}")

    # MEJORA 2: Feedback Loop Cerrado Granular (Self-Improving) con Decay
    try:
        household_size = max(1, int(health_profile.get('householdSize', 1)))
        expected_meals_count = len(days[0].get('meals', [])) if days else 0
        if not consumed_records or len(consumed_records) < 3:
            logger.info(f" [FEEDBACK LOOP] Insuficientes datos reales ({len(consumed_records) if consumed_records else 0} < 3). Omitiendo _meal_level_adherence para evitar falsos positivos EMA.")
        elif expected_meals_count > 0:
            # 1. Calcular adherencia granular del ciclo actual
            current_cycle_adherence = calculate_meal_level_adherence(user_id, days, consumed_records, household_size)
            
            # 2. Determinar si el ciclo evaluado fue fin de semana o dia de semana
            from datetime import datetime, timezone, timedelta
            # Restamos 12 horas para que si corre a las 3 AM, cuente como el dia anterior
            is_weekend = (datetime.now(timezone.utc) - timedelta(hours=12)).weekday() >= 5
            profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
            
            # 3. Obtener el historial de adherencia (por defecto 1.0 para todas las comidas)
            historical_adherence = health_profile.get(profile_key, {})
            if not isinstance(historical_adherence, dict):
                historical_adherence = {}
            
            # 4. Multi-Horizon EMA: corto plazo (responsivo) + largo plazo (estratégico)
            tuning_metrics = health_profile.get("tuning_metrics", {})

            # Alpha corto: auto-tunable (0.3–0.8) — ventana ~3–7 días, detecta caídas rápidas
            alpha_short = tuning_metrics.get("ema_alpha", 0.3)
            # Alpha largo: fijo 0.15 — ventana ~30 días, captura tendencias estacionales
            ALPHA_LONG = 0.15

            all_meal_types = set(list(current_cycle_adherence.keys()) + list(historical_adherence.keys()))

            # Historial del EMA largo (persiste en clave separada)
            long_profile_key = profile_key + '_long'
            hist_long = health_profile.get(long_profile_key, {})
            if not isinstance(hist_long, dict):
                hist_long = {}

            short_ema = {}
            long_ema = {}
            divergences = []
            for mt in all_meal_types:
                current_val = current_cycle_adherence.get(mt, 1.0)
                hist_short_val = historical_adherence.get(mt, 1.0)
                hist_long_val = hist_long.get(mt, 1.0)
                short_ema[mt] = round((alpha_short * current_val) + ((1 - alpha_short) * hist_short_val), 2)
                long_ema[mt] = round((ALPHA_LONG * current_val) + ((1 - ALPHA_LONG) * hist_long_val), 2)
                divergences.append(abs(current_val - hist_short_val))

            # AUTO-TUNE del alpha corto (sin tocar el largo que es fijo)
            if divergences:
                avg_div = sum(divergences) / len(divergences)
                div_history = tuning_metrics.get("ema_divergence_history", [])
                div_history.append(avg_div)
                div_history = div_history[-3:]
                if len(div_history) >= 3:
                    mean_div = sum(div_history) / 3
                    if mean_div > 0.4 and alpha_short < 0.8:
                        alpha_short = round(alpha_short + 0.05, 2)
                        logger.info(f" [AUTO-TUNE] Alta divergencia EMA ({mean_div:.2f}). Aumentando alpha_short a {alpha_short}.")
                    elif mean_div < 0.1 and alpha_short > 0.1:
                        alpha_short = round(alpha_short - 0.05, 2)
                        logger.info(f" [AUTO-TUNE] Baja divergencia EMA ({mean_div:.2f}). Reduciendo alpha_short a {alpha_short}.")
                tuning_metrics["ema_divergence_history"] = div_history
                tuning_metrics["ema_alpha"] = alpha_short
                execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb) WHERE id = %s", (json.dumps(tuning_metrics), user_id))

            # 5. Determinar hint contextual con ambos horizontes
            avg_short = sum(short_ema.values()) / len(short_ema) if short_ema else 1.0
            avg_long = sum(long_ema.values()) / len(long_ema) if long_ema else 1.0

            if avg_short < 0.3 and avg_long > 0.6:
                ema_hint = "temporary_dip"   # caída reciente, históricamente buen usuario → comfort food
            elif avg_short < 0.3 and avg_long < 0.4:
                ema_hint = "drastic_change"  # consistentemente bajo → intervención drástica
            elif avg_short > 0.6 and avg_long < 0.4:
                ema_hint = "improving"       # recuperación reciente → reforzar progreso
            else:
                ema_hint = "stable"

            logger.info(f" [DUAL-EMA] short={avg_short:.2f} long={avg_long:.2f} hint='{ema_hint}' ({profile_key})")

            # Inyectar ambas señales al pipeline
            pipeline_data['_meal_level_adherence'] = short_ema          # urgente: alarmas inmediatas
            pipeline_data['_meal_level_adherence_long'] = long_ema       # estratégico: tendencia
            pipeline_data['_adherence_ema_hint'] = ema_hint

            # Calcular adherencia general para el hint binario existente (compatibilidad)
            adherence_score = (len(consumed_records) / household_size) / expected_meals_count
            if adherence_score < 0.3:
                pipeline_data['_adherence_hint'] = 'low'
                logger.info(f" [FEEDBACK LOOP] Baja adherencia detectada (Score: {adherence_score:.2f}). Se pedirá simplificar el plan.")
            elif adherence_score > 0.8:
                pipeline_data['_adherence_hint'] = 'high'
                logger.info(f" [FEEDBACK LOOP] Alta adherencia detectada (Score: {adherence_score:.2f}). Se permitirá mayor variedad/creatividad.")

            # 6. Guardar ambos EMA en health_profile
            execute_sql_write(
                f"UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{{{profile_key}}}', %s::jsonb) WHERE id = %s",
                (json.dumps(short_ema), user_id)
            )
            execute_sql_write(
                f"UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{{{long_profile_key}}}', %s::jsonb) WHERE id = %s",
                (json.dumps(long_ema), user_id)
            )
    except Exception as e:
        logger.warning(f" [FEEDBACK LOOP] Error calculando adherencia: {e}")

    # GAP 2: Causal Loop - Extraer razones de abandono
    try:
        causal_reasons_raw = execute_sql_query(
            """SELECT meal_type, reason 
               FROM abandoned_meal_reasons 
               WHERE user_id = %s 
                 AND created_at >= NOW() - INTERVAL '14 days'
                 AND NOT (
                     reason IN ('swap:cravings', 'swap:weekend', 'swap:variety') 
                     AND created_at < NOW() - INTERVAL '48 hours'
                 )""",
            (user_id,)
        )
        if causal_reasons_raw:
            abandoned_reasons = {}
            for row in causal_reasons_raw:
                mt = row['meal_type']
                r = row['reason']
                if mt not in abandoned_reasons:
                    abandoned_reasons[mt] = []
                abandoned_reasons[mt].append(r)
            
            # Obtener la razon mas comun por comida
            most_common_reasons = {}
            for mt, reasons in abandoned_reasons.items():
                from collections import Counter
                most_common_reasons[mt] = Counter(reasons).most_common(1)[0][0]
            
            pipeline_data['_abandoned_reasons'] = most_common_reasons
            logger.info(f" [CAUSAL LOOP] Razones de abandono extraidas: {most_common_reasons}")
    except Exception as e:
        logger.warning(f" [CAUSAL LOOP] Error extrayendo razones de abandono: {e}")

    # GAP 4: Emotional State - Ajustar plan al estado animico reciente
    try:
        recent_sentiments = execute_sql_query(
            "SELECT response_sentiment FROM nudge_outcomes WHERE user_id = %s AND response_sentiment IS NOT NULL ORDER BY sent_at DESC LIMIT 3",
            (user_id,)
        )
        if recent_sentiments:
            sentiments = [row['response_sentiment'] for row in recent_sentiments]
            from collections import Counter
            dominant_sentiment = Counter(sentiments).most_common(1)[0][0]
            
            needs_comfort_sentiments = ['frustration', 'sadness', 'guilt', 'annoyed']
            ready_challenge_sentiments = ['motivation', 'positive', 'curiosity']
            
            if dominant_sentiment in needs_comfort_sentiments:
                pipeline_data['_emotional_state'] = 'needs_comfort'
                logger.info(f" [EMOTIONAL STATE] Usuario {user_id} necesita confort (Sentimiento: {dominant_sentiment})")
            elif dominant_sentiment in ready_challenge_sentiments:
                pipeline_data['_emotional_state'] = 'ready_for_challenge'
                logger.info(f"[EMOTIONAL STATE] Usuario {user_id} listo para reto (Sentimiento: {dominant_sentiment})")
    except Exception as e:
        logger.warning(f" [EMOTIONAL STATE] Error extrayendo sentimiento: {e}")

    # MEJORA 4: Auto-Evaluacion (Quality Score) y Consecuencias Adaptativas
    try:
        household_size = max(1, int(health_profile.get('householdSize', 1)))
        quality_score = calculate_plan_quality_score(user_id, {'days': days}, consumed_records, household_size)
        pipeline_data['_previous_plan_quality'] = quality_score
        logger.info(f" [SELF-EVALUATION] Calidad del Plan Anterior para {user_id}: {quality_score:.2f}")
        
        # --- MEJORA 2: ATTRIBUTION TRACKER (CLOSED-LOOP) ---
        try:
            from db_plans import get_latest_meal_plan
            prev_plan = get_latest_meal_plan(user_id)
            if prev_plan:
                # --- GAP 2: JUDGE FEEDBACK LOOP ---
                adv_winner = prev_plan.get("_adversarial_winner")
                if adv_winner:
                    import json
                    judge_calib = health_profile.get("judge_calibration", {"hits": 0, "total": 0, "score": 1.0})
                    # Si el quality_score > 0.6 o supera el promedio histórico, el juez tomó una buena decisión.
                    is_hit = 1 if quality_score > 0.6 else 0
                    
                    judge_calib["total"] += 1
                    judge_calib["hits"] += is_hit
                    judge_calib["score"] = round(judge_calib["hits"] / max(1, judge_calib["total"]), 2)
                    
                    health_profile["judge_calibration"] = judge_calib
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{judge_calibration}', %s::jsonb) WHERE id = %s",
                        (json.dumps(judge_calib), user_id)
                    )
                    logger.info(f" [JUDGE FEEDBACK] Calibración actualizada: {judge_calib['score']} (Hits: {judge_calib['hits']}/{judge_calib['total']})")

            if prev_plan and "_active_learning_signals" in prev_plan:
                signals_snapshot = prev_plan["_active_learning_signals"]
                if signals_snapshot:
                    attribution_tracker = health_profile.get("attribution_tracker", {})
                    
                    from datetime import datetime, timezone
                    import json
                    
                    from datetime import timedelta
                    for signal_key, signal_value in signals_snapshot.items():
                        tracker_key = f"{signal_key}:{signal_value}" if isinstance(signal_value, str) else str(signal_key)
                        stats = attribution_tracker.get(tracker_key, {"avg_score": 0.0, "count": 0})
                        
                        # --- GAP 1: Signal Decay Temporal ---
                        last_updated = stats.get("last_updated")
                        if last_updated:
                            try:
                                # Manejo seguro de Z y parseo de fecha
                                last_dt_str = str(last_updated)
                                if last_dt_str.endswith('Z'):
                                    last_dt_str = last_dt_str[:-1] + '+00:00'
                                last_dt = datetime.fromisoformat(last_dt_str)
                                if last_dt.tzinfo is None:
                                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                                
                                age_days = (datetime.now(timezone.utc) - last_dt).days
                                if age_days > 60:
                                    # Reset parcial: reducimos count a la mitad para dar peso a las nuevas observaciones
                                    # y olvidamos parcialmente el rendimiento histórico lejano.
                                    stats["count"] = max(1, stats["count"] // 2)
                            except Exception as e:
                                logger.warning(f" [ATTRIBUTION DECAY] Error parseando last_updated: {e}")
                        
                        new_count = stats["count"] + 1
                        new_avg = ((stats["avg_score"] * stats["count"]) + quality_score) / new_count
                        
                        attribution_tracker[tracker_key] = {
                            "avg_score": round(new_avg, 3),
                            "count": new_count,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{attribution_tracker}', %s::jsonb) WHERE id = %s",
                        (json.dumps(attribution_tracker), user_id)
                    )
                    health_profile["attribution_tracker"] = attribution_tracker  # Actualizar en memoria también
                    logger.info(f" [ATTRIBUTION] Quality Score {quality_score:.2f} atribuido a señales: {list(signals_snapshot.keys())}")
        except Exception as e:
            logger.warning(f" [ATTRIBUTION] Error procesando attribution tracker: {e}")
        
        # [A/B TESTING] Resolver experimento previo
        try:
            active_exp_id = health_profile.get("active_experiment_id")
            if active_exp_id:
                execute_sql_write("UPDATE learning_experiments SET outcome_quality_score = %s WHERE id = %s", (quality_score, active_exp_id))
                execute_sql_write("UPDATE user_profiles SET health_profile = health_profile - 'active_experiment_id' WHERE id = %s", (user_id,))
                logger.info(f" [A/B TESTING] Experimento {active_exp_id} resuelto con Quality Score: {quality_score:.2f}")
        except Exception as e:
            logger.warning(f" [A/B TESTING] Error resolviendo experimento: {e}")
        
        # MEJORA 3: Counterfactual Attribution — evaluar señales podadas en ciclos anteriores
        try:
            counterfactual_pending = health_profile.get("counterfactual_pending", {})
            if counterfactual_pending:
                attribution_tracker_cf = health_profile.get("attribution_tracker", {})
                reinstated = []
                confirmed_pruned = []

                for tracker_key, cf_data in list(counterfactual_pending.items()):
                    original_avg = cf_data.get("original_avg_score", 0.0)
                    pruned_at_str = cf_data.get("pruned_at", "")

                    # TTL: si fue podada hace >30 días, confirmar sin evaluar
                    try:
                        pruned_dt = datetime.fromisoformat(pruned_at_str.replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - pruned_dt).days
                    except Exception:
                        age_days = 999

                    if age_days > 30:
                        confirmed_pruned.append(tracker_key)
                        continue

                    # quality_score de ESTE ciclo = plan generado SIN la señal = contrafactual
                    delta = quality_score - original_avg
                    if delta < -0.15:
                        # El plan empeoró SIN la señal → era útil → re-instaurar
                        if tracker_key in attribution_tracker_cf:
                            # Subir avg_score por encima del umbral de poda (0.4) sin borrar historial
                            attribution_tracker_cf[tracker_key]["avg_score"] = max(
                                attribution_tracker_cf[tracker_key].get("avg_score", 0.0),
                                0.50
                            )
                            attribution_tracker_cf[tracker_key]["last_updated"] = datetime.now(timezone.utc).isoformat()
                        reinstated.append(f"{tracker_key} (delta={delta:+.2f}, orig={original_avg:.2f})")
                    else:
                        # El plan se mantuvo o mejoró → el pruning fue correcto
                        confirmed_pruned.append(tracker_key)

                # Limpiar entradas ya evaluadas
                new_pending = {k: v for k, v in counterfactual_pending.items()
                               if k not in reinstated and k not in confirmed_pruned}

                if reinstated:
                    logger.info(f" [COUNTERFACTUAL] Señales re-instauradas (el plan empeoró sin ellas): {', '.join(reinstated)}")
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{attribution_tracker}', %s::jsonb) WHERE id = %s",
                        (json.dumps(attribution_tracker_cf), user_id)
                    )
                    health_profile["attribution_tracker"] = attribution_tracker_cf
                if confirmed_pruned:
                    logger.info(f" [COUNTERFACTUAL] Podas confirmadas (plan no empeoró sin señal): {', '.join(confirmed_pruned)}")

                execute_sql_write(
                    "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{counterfactual_pending}', %s::jsonb) WHERE id = %s",
                    (json.dumps(new_pending), user_id)
                )
                health_profile["counterfactual_pending"] = new_pending
        except Exception as e:
            logger.warning(f" [COUNTERFACTUAL] Error en evaluación contrafactual: {e}")

        # [GAP 8] Diferenciar fuente de chunks (para evitar saturar con shifts diarios)
        quality_history_chunks = health_profile.get('quality_history_chunks', [])
        if not isinstance(quality_history_chunks, list):
            quality_history_chunks = []
            
        quality_history_chunks.append(quality_score)
        quality_history_chunks = quality_history_chunks[-5:] # Mantener los últimos 5 ciclos de chunks
        
        pipeline_data['quality_history_chunks'] = quality_history_chunks
        
        # Analizar tendencias para inyectar un hint al LLM
        if len(quality_history_chunks) >= 3:
            last_3 = quality_history_chunks[-3:]
            if all(score < 0.3 for score in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                
                # [A/B TESTING] Iniciar nuevo experimento Epsilon-Greedy
                import random
                try:
                    strategies = ["ethnic_rotation", "texture_swap", "protein_shock"]
                    chosen_strategy = random.choice(strategies)
                    
                    past_exps = execute_sql_query(
                        "SELECT strategy_applied, AVG(outcome_quality_score) as avg_score FROM learning_experiments WHERE user_id = %s AND outcome_quality_score IS NOT NULL GROUP BY strategy_applied ORDER BY avg_score DESC",
                        (user_id,), fetch_all=True
                    )
                    
                    # --- GAP 3: EPSILON DECAY PARA A/B TESTING ---
                    # En lugar de un 20% estático, reducimos la exploración conforme acumulamos experimentos.
                    total_exps_query = execute_sql_query("SELECT COUNT(*) as c FROM learning_experiments WHERE user_id = %s", (user_id,), fetch_all=True)
                    total_experiments = total_exps_query[0]["c"] if total_exps_query else 0
                    epsilon = max(0.05, 0.3 * (0.95 ** total_experiments))
                    
                    if past_exps and random.random() > epsilon:
                        best_strategy = past_exps[0]["strategy_applied"]
                        if past_exps[0].get("avg_score", 0) > 0.5:
                            chosen_strategy = best_strategy
                            
                    pipeline_data['_drastic_change_strategy'] = chosen_strategy
                    
                    res = execute_sql_query(
                        "INSERT INTO learning_experiments (user_id, strategy_applied) VALUES (%s, %s) RETURNING id",
                        (user_id, chosen_strategy), fetch_one=True
                    )
                    if res and res.get("id"):
                        exp_id = res["id"]
                        execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{active_experiment_id}', %s::jsonb) WHERE id = %s", (json.dumps(exp_id), user_id))
                        logger.info(f" [A/B TESTING] Iniciado experimento {exp_id} con estrategia: {chosen_strategy}")
                        
                except Exception as e:
                    logger.warning(f" [A/B TESTING] Error orquestando experimento: {e}")
                
                logger.warning(f" [FEEDBACK LOOP] Quality Score muy bajo por 3 ciclos consecutivos. Se activará un CAMBIO RADICAL (Estrategia: {pipeline_data.get('_drastic_change_strategy', 'default')}).")
            elif all(score > 0.8 for score in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
                logger.info(f" [FEEDBACK LOOP] Quality Score muy alto por 3 ciclos consecutivos. Se permitirá MAYOR COMPLEJIDAD.")
        
        # Detectar Plateau Silencioso (GAP 6 / 8)
        if len(quality_history_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(quality_history_chunks) / len(quality_history_chunks)
            variance = sum((q - mean_q)**2 for q in quality_history_chunks) / len(quality_history_chunks)
            if variance < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
                logger.warning(f" [FEEDBACK LOOP] Plateau Silencioso detectado. Quality score estancado en {mean_q:.2f}. Se forzará una ruptura de monotonía.")
                
        # Detectar Plateau de Adherencia (Mejora 2)
        adherence_history = health_profile.get('adherence_history_rotations', [])
        if len(adherence_history) >= 3 and not pipeline_data.get('_quality_hint'):
            last_3_adherence = adherence_history[-3:]
            # Check for consistent drop: e.g. 0.8 -> 0.7 -> 0.6
            is_dropping = all(last_3_adherence[i] < last_3_adherence[i-1] for i in range(1, 3))
            if is_dropping and last_3_adherence[-1] < 0.65:
                pipeline_data['_quality_hint'] = 'simplify_urgently'
                logger.warning(f" [FEEDBACK LOOP] Plateau de Adherencia detectado. Cayendo consistentemente a {last_3_adherence[-1]:.2f}. Se forzará a simplificar el plan.")
        
        # Guardamos el score y el historial en el health_profile
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_chunks}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_chunks), user_id)
        )
    except Exception as e:
        logger.warning(f" [SELF-EVALUATION] Error calculando Quality Score: {e}")


    # MEJORA 2: Historial de Peso para Metabolismo Evolutivo
    try:
        weight_log = execute_sql_query(
            "SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at",
            (user_id,), fetch_all=True
        )
        if weight_log:
            pipeline_data['weight_history'] = [
                {"date": str(w['date']), "weight": w['weight'], "unit": w.get('unit', 'lb')}
                for w in weight_log
            ]
            logger.info(f" [METABOLISMO EVOLUTIVO] {len(weight_log)} registros de peso cargados para el usuario {user_id}")
    except Exception as e:
        logger.error(f" [METABOLISMO EVOLUTIVO] Error cargando historial de peso: {e}")

    # MEJORA 3: Aprendizaje de Patrones de Ã‰xito y Temporalidad
    try:
        # [GAP 7] Ventana dinámica de aprendizaje basada en el offset del plan
        days_offset = int(pipeline_data.get('_days_offset', 0))
        dynamic_days_back = max(14, days_offset + 7)
        
        succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=dynamic_days_back)
        day_adherence = calculate_day_of_week_adherence(user_id, days_back=max(30, dynamic_days_back))
        
        pipeline_data['successful_techniques'] = list(set(succ_techs))
        pipeline_data['abandoned_techniques'] = list(set(aban_techs))
        pipeline_data['frequent_meals'] = freq_meals
        pipeline_data['day_of_week_adherence'] = day_adherence
        
        logger.info(f" [PATRONES DE Ã‰XITO] {len(succ_techs)} exitos, {len(aban_techs)} abandonos extraidos para {user_id}")
        logger.info(f" [TEMPORALIDAD] Perfil de dias: {day_adherence}")
    except Exception as e:
        logger.error(f" [PATRONES DE Ã‰XITO] Error calculando scores o temporalidad: {e}")

    # MEJORA 5 y 2: Sincronizacion Nudge <-> Rotacion con Efectividad Real
    try:
        nudge_data = execute_sql_query(
            "SELECT nudge_type, responded, meal_logged, response_sentiment FROM nudge_outcomes "
            "WHERE user_id = %s AND sent_at > NOW() - INTERVAL '7 days'",
            (user_id,), fetch_all=True
        )
        if nudge_data:
            ignored_meal_types = []
            frustrated_meal_types = []
            for n in nudge_data:
                if not n.get('responded') and not n.get('meal_logged'):
                    ignored_meal_types.append(n.get('nudge_type'))
                sentiment = n.get('response_sentiment')
                if sentiment in ['annoyed', 'frustration', 'sadness', 'guilt']:
                    frustrated_meal_types.append(n.get('nudge_type'))
                    
            if ignored_meal_types:
                pipeline_data['_ignored_meal_types'] = list(set(ignored_meal_types))
            if frustrated_meal_types:
                pipeline_data['_frustrated_meal_types'] = list(set(frustrated_meal_types))
                
        # Filtrar nudge_type por efectividad REAL
        effective_nudge_query = """
            SELECT nudge_type, 
            AVG(CASE WHEN meal_logged THEN 1.0 ELSE 0.0 END) as conversion_rate
            FROM nudge_outcomes 
            WHERE user_id = %s
            GROUP BY nudge_type
        """
        try:
            effective_nudges = execute_sql_query(effective_nudge_query, (user_id,), fetch_all=True)
            if effective_nudges:
                pipeline_data['_nudge_conversion_rates'] = {
                    n['nudge_type']: float(n['conversion_rate']) for n in effective_nudges if n['conversion_rate'] is not None
                }
                logger.info(f" [NUDGE SYNC] Tasas de conversion reales agregadas al contexto.")
        except Exception as eff_err:
            pass

        # MEJORA Gap F: Tonos de comunicacion exitosos
        try:
            successful_styles = execute_sql_query(
                "SELECT nudge_style, COUNT(*) as successes FROM nudge_outcomes WHERE user_id = %s AND nudge_style IS NOT NULL AND (meal_logged = true OR response_sentiment IN ('motivation', 'positive', 'happy', 'excited')) GROUP BY nudge_style ORDER BY successes DESC LIMIT 2",
                (user_id,), fetch_all=True
            )
            if successful_styles:
                styles_list = [row['nudge_style'] for row in successful_styles]
                pipeline_data['_successful_tone_strategies'] = styles_list
                logger.info(f"  [TONE SYNC] Estilos de comunicacion exitosos inyectados: {styles_list}")
        except Exception as style_err:
            pass
            
    except Exception as e:
        if "relation" not in str(e).lower() and "column" not in str(e).lower():
            logger.warning(f" [NUDGE SYNC] Error consultando nudge_outcomes: {e}")

    # MEJORA 7: Cold-Start Intelligence (Collaborative Filtering Ligero)
    try:
        if not consumed_records or len(consumed_records) < 3:
            popular_meals_data = get_similar_user_patterns(user_id, health_profile)
            if popular_meals_data:
                popular_names = [p['meal_name'] for p in popular_meals_data if p.get('meal_name')]
                pipeline_data['_cold_start_recommendations'] = popular_names
                logger.info(f" [COLD-START] Inyectando {len(popular_names)} platos populares de usuarios similares para {user_id}")
    except Exception as e:
        logger.warning(f" [COLD-START] Error procesando recomendaciones de inicio en frio: {e}")

    # MEJORA 8: Explicit Likes (❤️)
    try:
        likes_data = execute_sql_query(
            "SELECT meal_name FROM meal_likes WHERE user_id = %s",
            (user_id,), fetch_all=True
        )
        if likes_data:
            liked_meals = [row['meal_name'] for row in likes_data if row.get('meal_name')]
            if liked_meals:
                # Deduplicate and keep the most recent ones (assuming later inserts are at the end)
                liked_meals = list(dict.fromkeys(liked_meals))[-20:]
                pipeline_data['_liked_meals'] = liked_meals
                logger.info(f" [LIKES SYNC] Inyectando {len(liked_meals)} platos likeados al contexto para {user_id}.")
                
        liked_flavor_profiles = health_profile.get('liked_flavor_profiles', [])
        if liked_flavor_profiles:
            pipeline_data['_liked_flavor_profiles'] = liked_flavor_profiles
            logger.info(f" [LIKES SYNC] Inyectando {len(liked_flavor_profiles)} perfiles de sabor al contexto.")
    except Exception as e:
        logger.warning(f" [LIKES SYNC] Error consultando meal_likes: {e}")

    # MEJORA 2: Attribution Pruning (Closed-Loop)
    try:
        attribution_tracker = health_profile.get("attribution_tracker", {})
        if attribution_tracker:
            pruned_signals = []
            
            # Map the signals we care about pruning
            signals_to_check = {
                "quality_hint": "_quality_hint",
                "adherence_hint": "_adherence_hint",
                "emotional_state": "_emotional_state",
                "drastic_strategy": "_drastic_change_strategy",
                "cold_start": "_cold_start_recs"
            }
            
            for base_key, pipeline_key in signals_to_check.items():
                signal_value = pipeline_data.get(pipeline_key)
                if signal_value:
                    tracker_key = f"{base_key}:{signal_value}" if isinstance(signal_value, str) else str(base_key)
                    stats = attribution_tracker.get(tracker_key)
                    
                    if stats:
                        # PRUNING THRESHOLDS: attempted at least 2 times, avg score < 0.4
                        if stats.get("count", 0) >= 2 and stats.get("avg_score", 1.0) < 0.4:
                            # Prune it!
                            del pipeline_data[pipeline_key]
                            pruned_signals.append(f"{pipeline_key}={signal_value} (score: {stats.get('avg_score')})")
                            # MEJORA 3: Registrar en counterfactual_pending para medir impacto real en el siguiente ciclo
                            cf_pending = health_profile.get("counterfactual_pending", {})
                            cf_pending[tracker_key] = {
                                "pruned_at": datetime.now(timezone.utc).isoformat(),
                                "original_avg_score": round(stats.get("avg_score", 0.0), 3),
                                "pipeline_key": pipeline_key,
                                "signal_value": str(signal_value),
                            }
                            health_profile["counterfactual_pending"] = cf_pending
                            
            if pruned_signals:
                logger.warning(f" [ATTRIBUTION PRUNING] Señales podadas por bajo rendimiento histórico: {', '.join(pruned_signals)}")
                # Persistir counterfactual_pending actualizado (acumulado en el loop de pruning)
                execute_sql_write(
                    "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{counterfactual_pending}', %s::jsonb) WHERE id = %s",
                    (json.dumps(health_profile.get("counterfactual_pending", {})), user_id)
                )
    except Exception as e:
        logger.warning(f" [ATTRIBUTION PRUNING] Error procesando la poda: {e}")

    # P1: AUTO-ACTIVACIÓN AUTÓNOMA del Adversarial Self-Play (Cron Path)
    # Decide si activar el adversarial self-play basándose en la salud del pipeline.
    try:
        if not pipeline_data.get('_use_adversarial_play'):
            _auto_reasons = []

            # Condición 1: Quality Score bajo sostenido (< 0.5 por 2+ ciclos)
            qh = health_profile.get("quality_history_chunks", [])
            if isinstance(qh, list) and len(qh) >= 2 and all(s < 0.5 for s in qh[-2:]):
                _auto_reasons.append("quality_low_sustained")

            # Condición 2: Alta varianza en Attribution Tracker
            attr_tracker = health_profile.get("attribution_tracker", {})
            if len(attr_tracker) >= 3:
                scores = [v.get("avg_score", 0.5) for v in attr_tracker.values() if isinstance(v, dict)]
                if scores:
                    mean_s = sum(scores) / len(scores)
                    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                    if variance > 0.06:
                        _auto_reasons.append(f"attribution_high_variance ({variance:.3f})")

            # Condición 3: Historial de rechazos médicos frecuentes
            rejection_patterns = health_profile.get("rejection_patterns", [])
            if isinstance(rejection_patterns, list) and len(rejection_patterns) >= 5:
                _auto_reasons.append(f"frequent_rejections ({len(rejection_patterns)})")

            if _auto_reasons:
                pipeline_data['_use_adversarial_play'] = True
                logger.info(f" [ADVERSARIAL AUTO-ACTIVATE] Activado para {user_id}: {', '.join(_auto_reasons)}")
    except Exception as e:
        logger.warning(f" [ADVERSARIAL AUTO-ACTIVATE] Error evaluando condiciones: {e}")

    return pipeline_data


def inject_learning_signals_from_profile(user_id: str, pipeline_data: dict) -> dict:
    """Inyecta señales de aprendizaje para generaciones manuales (API path).

    Equivalente ligero de _inject_advanced_learning_signals (cron path).
    Lee señales persistidas del health_profile + queries ligeros en vivo.
    Solo escribe keys que NO estén ya presentes (no sobreescribe).
    """
    from db_core import execute_sql_query
    from datetime import datetime, timezone, timedelta

    try:
        profile_row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True
        )
        if not profile_row or not profile_row.get("health_profile"):
            return pipeline_data
        hp = profile_row["health_profile"]
    except Exception as e:
        logger.warning(f" [SIGNAL INJECT] Error reading health_profile for {user_id}: {e}")
        return pipeline_data

    # ── 1. Señales persistidas del health_profile ──

    # EMA Adherence (weekday/weekend)
    is_weekend = datetime.now(timezone.utc).weekday() >= 5
    profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
    meal_adherence = hp.get(profile_key, {})
    if meal_adherence and not pipeline_data.get('_meal_level_adherence'):
        pipeline_data['_meal_level_adherence'] = meal_adherence
        avg = sum(meal_adherence.values()) / max(len(meal_adherence), 1)
        if avg < 0.3:
            pipeline_data.setdefault('_adherence_hint', 'low')
        elif avg > 0.8:
            pipeline_data.setdefault('_adherence_hint', 'high')

    # Quality Score & adaptive hint
    quality_score = hp.get('last_plan_quality')
    if quality_score is not None:
        pipeline_data.setdefault('_previous_plan_quality', quality_score)

    qh_chunks = hp.get('quality_history_chunks', [])
    if isinstance(qh_chunks, list) and qh_chunks and not pipeline_data.get('_quality_hint'):
        if len(qh_chunks) >= 3:
            last_3 = qh_chunks[-3:]
            if all(s < 0.3 for s in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                import random
                pipeline_data['_drastic_change_strategy'] = random.choice(
                    ["ethnic_rotation", "texture_swap", "protein_shock"]
                )
            elif all(s > 0.8 for s in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
        if len(qh_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(qh_chunks) / len(qh_chunks)
            var = sum((q - mean_q) ** 2 for q in qh_chunks) / len(qh_chunks)
            if var < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
        # Adherence plateau check
        adh_hist = hp.get('adherence_history_rotations', [])
        if isinstance(adh_hist, list) and len(adh_hist) >= 3 and not pipeline_data.get('_quality_hint'):
            last_3a = adh_hist[-3:]
            is_dropping = all(last_3a[i] < last_3a[i - 1] for i in range(1, 3))
            if is_dropping and last_3a[-1] < 0.65:
                pipeline_data['_quality_hint'] = 'simplify_urgently'

    # LLM Retrospective
    if hp.get('llm_retrospective'):
        pipeline_data.setdefault('_llm_retrospective', hp['llm_retrospective'])

    # Liked Flavor Profiles
    if hp.get('liked_flavor_profiles'):
        pipeline_data.setdefault('_liked_flavor_profiles', hp['liked_flavor_profiles'])

    # Frequent meals (cold-start seed)
    if hp.get('previous_plan_frequent_meals'):
        pipeline_data.setdefault('frequent_meals', hp['previous_plan_frequent_meals'])

    # ── 2. Queries ligeros en vivo ──

    # Weight History
    try:
        if not pipeline_data.get('weight_history'):
            wl = execute_sql_query(
                "SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at",
                (user_id,), fetch_all=True
            )
            if wl:
                pipeline_data['weight_history'] = [
                    {"date": str(w['date']), "weight": w['weight'], "unit": w.get('unit', 'lb')} for w in wl
                ]
    except Exception:
        pass

    # Emotional State
    try:
        if not pipeline_data.get('_emotional_state'):
            rows = execute_sql_query(
                "SELECT response_sentiment FROM nudge_outcomes WHERE user_id = %s AND response_sentiment IS NOT NULL ORDER BY sent_at DESC LIMIT 3",
                (user_id,), fetch_all=True
            )
            if rows:
                from collections import Counter
                dom = Counter([r['response_sentiment'] for r in rows]).most_common(1)[0][0]
                if dom in ('frustration', 'sadness', 'guilt', 'annoyed'):
                    pipeline_data['_emotional_state'] = 'needs_comfort'
                elif dom in ('motivation', 'positive', 'curiosity'):
                    pipeline_data['_emotional_state'] = 'ready_for_challenge'
    except Exception:
        pass

    # Abandoned Reasons
    try:
        if not pipeline_data.get('_abandoned_reasons'):
            causal = execute_sql_query(
                """SELECT meal_type, reason 
                   FROM abandoned_meal_reasons 
                   WHERE user_id = %s 
                     AND created_at >= NOW() - INTERVAL '14 days'
                     AND NOT (
                         reason IN ('swap:cravings', 'swap:weekend', 'swap:variety') 
                         AND created_at < NOW() - INTERVAL '48 hours'
                     )""",
                (user_id,), fetch_all=True
            )
            if causal:
                from collections import Counter
                by_meal = {}
                for row in causal:
                    by_meal.setdefault(row['meal_type'], []).append(row['reason'])
                pipeline_data['_abandoned_reasons'] = {
                    mt: Counter(reasons).most_common(1)[0][0] for mt, reasons in by_meal.items()
                }
    except Exception:
        pass

    # Ingredient Fatigue
    try:
        if not pipeline_data.get('fatigued_ingredients'):
            fatigue = calculate_ingredient_fatigue(user_id, days_back=14, tuning_metrics=hp.get("tuning_metrics", {}))
            if fatigue and fatigue.get('fatigued_ingredients'):
                pipeline_data['fatigued_ingredients'] = fatigue['fatigued_ingredients']
    except Exception:
        pass

    # Successful/Abandoned Techniques + Day-of-Week Adherence
    # [P1-1 FIX] Usar ventana corta para rolling refills para evitar dilución EMA
    _is_rolling = pipeline_data.get('_is_rolling_refill', False)
    if _is_rolling:
        from constants import PLAN_CHUNK_SIZE
        _success_db = PLAN_CHUNK_SIZE + 1
        _adherence_db = PLAN_CHUNK_SIZE + 1
    else:
        _success_db = 14
        _adherence_db = 30
    try:
        if not pipeline_data.get('successful_techniques'):
            st, at, fm = calculate_meal_success_scores(user_id, days_back=_success_db)
            pipeline_data['successful_techniques'] = list(set(st))
            pipeline_data['abandoned_techniques'] = list(set(at))
            if fm:
                pipeline_data.setdefault('frequent_meals', fm)
        if not pipeline_data.get('day_of_week_adherence'):
            pipeline_data['day_of_week_adherence'] = calculate_day_of_week_adherence(user_id, days_back=_adherence_db)
    except Exception:
        pass

    # Nudge data (ignored, frustrated, conversion rates, tone strategies)
    try:
        nudge_rows = execute_sql_query(
            "SELECT nudge_type, nudge_style, responded, meal_logged, response_sentiment "
            "FROM nudge_outcomes WHERE user_id = %s AND sent_at > NOW() - INTERVAL '7 days'",
            (user_id,), fetch_all=True
        )
        if nudge_rows:
            if not pipeline_data.get('_ignored_meal_types'):
                ignored = list({n['nudge_type'] for n in nudge_rows if not n.get('responded') and not n.get('meal_logged')})
                if ignored:
                    pipeline_data['_ignored_meal_types'] = ignored
            if not pipeline_data.get('_frustrated_meal_types'):
                frust = list({n['nudge_type'] for n in nudge_rows if n.get('response_sentiment') in ('annoyed', 'frustration', 'sadness', 'guilt')})
                if frust:
                    pipeline_data['_frustrated_meal_types'] = frust
            if not pipeline_data.get('_nudge_conversion_rates'):
                from collections import defaultdict
                totals = defaultdict(lambda: [0, 0])
                for n in nudge_rows:
                    nt = n['nudge_type']
                    totals[nt][1] += 1
                    if n.get('meal_logged'):
                        totals[nt][0] += 1
                rates = {nt: round(v[0] / v[1], 2) for nt, v in totals.items() if v[1] > 0}
                if rates:
                    pipeline_data['_nudge_conversion_rates'] = rates
            if not pipeline_data.get('_successful_tone_strategies'):
                from collections import Counter
                success_styles = [n['nudge_style'] for n in nudge_rows
                                  if n.get('nudge_style') and (n.get('meal_logged') or n.get('response_sentiment') in ('motivation', 'positive', 'happy', 'excited'))]
                if success_styles:
                    pipeline_data['_successful_tone_strategies'] = [s for s, _ in Counter(success_styles).most_common(2)]
    except Exception:
        pass

    # Explicit Likes
    try:
        if not pipeline_data.get('_liked_meals'):
            likes_rows = execute_sql_query(
                "SELECT meal_name FROM meal_likes WHERE user_id = %s", (user_id,), fetch_all=True
            )
            if likes_rows:
                names = list(dict.fromkeys([r['meal_name'] for r in likes_rows if r.get('meal_name')]))[-20:]
                if names:
                    pipeline_data['_liked_meals'] = names
    except Exception:
        pass

    # Cold-Start Recommendations
    try:
        if not pipeline_data.get('_cold_start_recommendations'):
            from db_facts import get_consumed_meals_since
            recent = get_consumed_meals_since(user_id, since_iso_date=(datetime.now(timezone.utc) - timedelta(days=14)).isoformat())
            if not recent or len(recent) < 3:
                popular = get_similar_user_patterns(user_id, hp)
                if popular:
                    pipeline_data['_cold_start_recommendations'] = [p['meal_name'] for p in popular if p.get('meal_name')]
    except Exception:
        pass

    # P1: AUTO-ACTIVACIÓN AUTÓNOMA del Adversarial Self-Play (API Path)
    try:
        if not pipeline_data.get('_use_adversarial_play'):
            _auto_reasons = []

            qh = hp.get("quality_history_chunks", [])
            if isinstance(qh, list) and len(qh) >= 2 and all(s < 0.5 for s in qh[-2:]):
                _auto_reasons.append("quality_low_sustained")

            attr_tracker = hp.get("attribution_tracker", {})
            if len(attr_tracker) >= 3:
                scores = [v.get("avg_score", 0.5) for v in attr_tracker.values() if isinstance(v, dict)]
                if scores:
                    mean_s = sum(scores) / len(scores)
                    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                    if variance > 0.06:
                        _auto_reasons.append(f"attribution_high_variance ({variance:.3f})")

            rejection_patterns = hp.get("rejection_patterns", [])
            if isinstance(rejection_patterns, list) and len(rejection_patterns) >= 5:
                _auto_reasons.append(f"frequent_rejections ({len(rejection_patterns)})")

            if _auto_reasons:
                pipeline_data['_use_adversarial_play'] = True
                logger.info(f" [ADVERSARIAL AUTO-ACTIVATE] API path activado para {user_id}: {', '.join(_auto_reasons)}")
    except Exception as e:
        logger.warning(f" [ADVERSARIAL AUTO-ACTIVATE] Error en API path: {e}")

    injected_keys = [k for k in pipeline_data if k.startswith('_') or k in ('fatigued_ingredients', 'weight_history', 'successful_techniques', 'day_of_week_adherence', 'frequent_meals')]
    logger.info(f" [SIGNAL INJECT] {len(injected_keys)} señales inyectadas para generación manual: user={user_id}")
    return pipeline_data


# ============================================================
# BACKGROUND CHUNKING â€" Generacion de Semanas 2-4 en background
# ============================================================

def _compute_chunk_delay_days(
    days_offset: int,
    days_count: int,
    week_number: int,
    pipeline_snapshot: dict,
    chunk_kind: str | None = None,
    *,
    for_failed_retry: bool = False,
):
    """Calcula el delay del chunk según la política activa."""
    import math

    days_offset_int = max(0, int(days_offset))
    days_count_int = max(1, int(days_count))
    mode = (CHUNK_LEARNING_MODE or "strict").strip().lower()
    normalized_chunk_kind = (
        chunk_kind
        or ("rolling_refill" if (pipeline_snapshot or {}).get("_is_rolling_refill") else "initial_plan")
    ).strip().lower()

    if mode == "strict":
        proactive_margin_days = 0
        if normalized_chunk_kind == "initial_plan" and days_offset_int > 0:
            proactive_margin_days = CHUNK_PROACTIVE_MARGIN_DAYS
        if for_failed_retry:
            proactive_margin_days = max(proactive_margin_days, 1)
        delay_days = max(0, days_offset_int - proactive_margin_days)
    else:
        delay_days = max(0, days_offset_int - math.ceil(days_count_int / 2))

        try:
            total_days = int((pipeline_snapshot or {}).get("totalDays") or 0)
            if total_days >= 15:
                from constants import PLAN_CHUNK_SIZE
                total_weeks = math.ceil(total_days / PLAN_CHUNK_SIZE)
                if week_number >= total_weeks - 1:
                    delay_days = max(0, days_offset_int - 3)
                    logger.info(f" [GAP B] Chunk final {week_number}/{total_weeks} adelantado: delay={delay_days}d (mode={mode})")
        except Exception as e:
            logger.debug(f" [GAP B] No se pudo aplicar adelanto de chunk final: {e}")

    return min(delay_days, 180), mode, days_offset_int, days_count_int


def _compute_expected_preemption_seconds(days_offset: int, delay_days: int) -> int:
    """Cuánto del lag es esperado por adelantar intencionalmente el execute_after."""
    try:
        offset_days = max(0, int(days_offset))
        scheduled_days = max(0, int(delay_days))
        return max(0, offset_days - scheduled_days) * 86400
    except Exception:
        return 0


def _enqueue_plan_chunk(
    user_id: str,
    meal_plan_id: str,
    week_number: int,
    days_offset: int,
    days_count: int,
    pipeline_snapshot: dict,
    chunk_kind: str = None,
):
    """Inserta un job en plan_chunk_queue para generar un chunk en background."""
    import json
    # [P0-4] Estampar cuándo fue capturado el snapshot del inventario.
    # _refresh_chunk_pantry usa este timestamp para detectar snapshots vencidos
    # y forzar un live-retry antes de usar datos obsoletos.
    pipeline_snapshot = copy.deepcopy(pipeline_snapshot) if pipeline_snapshot else {}
    if isinstance(pipeline_snapshot.get("form_data"), dict):
        pipeline_snapshot["form_data"]["_pantry_captured_at"] = datetime.now(timezone.utc).isoformat()

    normalized_chunk_kind = chunk_kind or ("rolling_refill" if pipeline_snapshot.get("_is_rolling_refill") else "initial_plan")
    delay_days, chunk_mode, days_offset_int, days_count_int = _compute_chunk_delay_days(
        days_offset,
        days_count,
        week_number,
        pipeline_snapshot or {},
        normalized_chunk_kind,
    )
    expected_preemption_seconds = _compute_expected_preemption_seconds(days_offset_int, delay_days)

    # [GAP 3 FIX]: Calcular execute_after exacto usando la medianoche local en UTC
    start_date_iso = pipeline_snapshot.get("form_data", {}).get("_plan_start_date")
    if start_date_iso:
        from constants import safe_fromisoformat
        from datetime import timedelta
        try:
            start_dt = safe_fromisoformat(start_date_iso)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            # [VISUAL CONTINUITY] Disparar el chunk al final del día PREVIO al
            # primero que cubre, no al inicio del día objetivo. Así el usuario
            # nunca ve una ventana vacía cuando /shift-plan recorta el día que
            # pasó. Floor a NOW()+1m para catchup chunks (delay_days=0).
            execute_dt_target = start_dt + timedelta(days=delay_days, hours=-3)
            execute_dt_min = datetime.now(timezone.utc) + timedelta(minutes=1)
            execute_dt = max(execute_dt_target, execute_dt_min)
            
            # [GAP E] Idempotencia fuerte: si ya existe un chunk vivo (pending/processing/stale/failed)
            # para este (meal_plan_id, week_number), el UNIQUE parcial lo bloquea.
            # Usamos una guardia explícita para evitar la excepción y retornar silenciosamente.
            existing = execute_sql_query(
                """
                SELECT id, status FROM plan_chunk_queue
                WHERE meal_plan_id = %s AND week_number = %s
                  AND status IN ('pending', 'processing', 'stale')
                LIMIT 1
                """,
                (str(meal_plan_id), week_number), fetch_one=True
            )
            if existing:
                logger.info(f"[GAP E] Chunk {week_number} para plan {meal_plan_id} ya existe activo (id={existing['id']}, status={existing['status']}). Skip enqueue.")
                return

            failed_existing = execute_sql_query(
                """
                SELECT id FROM plan_chunk_queue
                WHERE meal_plan_id = %s AND week_number = %s
                  AND status = 'failed'
                LIMIT 1
                """,
                (str(meal_plan_id), week_number), fetch_one=True
            )
            if failed_existing:
                retry_delay_days, _, _, _ = _compute_chunk_delay_days(
                    days_offset_int,
                    days_count_int,
                    week_number,
                    pipeline_snapshot or {},
                    for_failed_retry=True,
                )
                expected_preemption_seconds = _compute_expected_preemption_seconds(days_offset_int, retry_delay_days)
                execute_dt_target = start_dt + timedelta(days=retry_delay_days, hours=-3)
                execute_dt_min = datetime.now(timezone.utc) + timedelta(minutes=1)
                execute_dt = max(execute_dt_target, execute_dt_min)
                logger.warning(
                    f"⚠️ [P0-1] Re-encolando chunk failed (id={failed_existing['id']}) "
                    f"para plan {meal_plan_id} week {week_number} en modo {chunk_mode} "
                    f"con margen de {days_offset_int - retry_delay_days}d"
                )
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        attempts = 0,
                        chunk_kind = %s,
                        pipeline_snapshot = %s::jsonb,
                        execute_after = %s::timestamptz,
                        expected_preemption_seconds = %s,
                        updated_at = NOW(),
                        days_offset = %s,
                        days_count = %s
                    WHERE id = %s
                    """,
                    (
                        normalized_chunk_kind,
                        json.dumps(pipeline_snapshot, ensure_ascii=False),
                        execute_dt.isoformat(),
                        expected_preemption_seconds,
                        days_offset_int,
                        days_count_int,
                        failed_existing['id']
                    )
                )
            else:
                execute_sql_write(
                    """
                    INSERT INTO plan_chunk_queue
                        (user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count, pipeline_snapshot, execute_after, expected_preemption_seconds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::timestamptz, %s)
                    ON CONFLICT (meal_plan_id, week_number)
                    WHERE status IN ('pending', 'processing', 'stale', 'failed')
                    DO NOTHING
                    """,
                    (
                        user_id, str(meal_plan_id), week_number, normalized_chunk_kind, days_offset_int, days_count_int,
                        json.dumps(pipeline_snapshot, ensure_ascii=False), execute_dt.isoformat(), expected_preemption_seconds
                    )
                )
            logger.info(
                f" [CHUNK] Chunk {week_number} encolado para plan {meal_plan_id} "
                f"(días {days_offset_int+1}–{days_offset_int+days_count_int}, kind={normalized_chunk_kind}, mode={chunk_mode}) "
                f"ejecutará a las {execute_dt.isoformat()}"
            )
            return
        except Exception as e:
            logger.warning(f" [CHUNK] Error en enqueue (intento con _plan_start_date), usando fallback NOW(): {e}")

    # Fallback si no hay _plan_start_date o hubo error parseando
    # [GAP E] Guardia de idempotencia
    existing = execute_sql_query(
        """
        SELECT id, status FROM plan_chunk_queue
        WHERE meal_plan_id = %s AND week_number = %s
          AND status IN ('pending', 'processing', 'stale')
        LIMIT 1
        """,
        (str(meal_plan_id), week_number), fetch_one=True
    )
    if existing:
        logger.info(f"[GAP E] Chunk {week_number} para plan {meal_plan_id} ya existe activo (id={existing['id']}, status={existing['status']}). Skip enqueue.")
        return

    failed_existing = execute_sql_query(
        """
        SELECT id FROM plan_chunk_queue
        WHERE meal_plan_id = %s AND week_number = %s
          AND status = 'failed'
        LIMIT 1
        """,
        (str(meal_plan_id), week_number), fetch_one=True
    )
    if failed_existing:
        # [P0-3 FIX] Antes solo logeaba warning y caía al INSERT con ON CONFLICT DO NOTHING,
        # dejando el chunk failed sin reintento (huecos permanentes en planes 15/30d).
        # Replicamos el UPDATE explícito del primer path para reactivar el chunk con snapshot
        # fresco, attempts=0 y delay recalculado para retry.
        retry_delay_days, _, _, _ = _compute_chunk_delay_days(
            days_offset_int,
            days_count_int,
            week_number,
            pipeline_snapshot or {},
            for_failed_retry=True,
        )
        retry_preemption_seconds = _compute_expected_preemption_seconds(days_offset_int, retry_delay_days)
        logger.warning(
            f"⚠️ [P0-3] Re-encolando chunk failed (id={failed_existing['id']}) "
            f"para plan {meal_plan_id} week {week_number} en modo {chunk_mode} "
            f"(fallback, delay={retry_delay_days}d)"
        )
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = 0,
                chunk_kind = %s,
                pipeline_snapshot = %s::jsonb,
                execute_after = NOW() + make_interval(days => %s),
                expected_preemption_seconds = %s,
                updated_at = NOW(),
                days_offset = %s,
                days_count = %s
            WHERE id = %s
            """,
            (
                normalized_chunk_kind,
                json.dumps(pipeline_snapshot, ensure_ascii=False),
                retry_delay_days,
                retry_preemption_seconds,
                days_offset_int,
                days_count_int,
                failed_existing['id'],
            )
        )
        logger.info(
            f" [CHUNK] Chunk {week_number} reactivado para plan {meal_plan_id} "
            f"(días {days_offset_int+1}–{days_offset_int+days_count_int}, kind={normalized_chunk_kind}, mode={chunk_mode}) "
            f"con delay de {retry_delay_days} días"
        )
        return

    execute_sql_write(
        """
        INSERT INTO plan_chunk_queue
            (user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count, pipeline_snapshot, execute_after, expected_preemption_seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, NOW() + make_interval(days => %s), %s)
        ON CONFLICT (meal_plan_id, week_number)
        WHERE status IN ('pending', 'processing', 'stale', 'failed')
        DO NOTHING
        """,
        (
            user_id, str(meal_plan_id), week_number, normalized_chunk_kind, days_offset_int, days_count_int,
            json.dumps(pipeline_snapshot, ensure_ascii=False), delay_days, expected_preemption_seconds
        )
    )
    logger.info(
        f" [CHUNK] Chunk {week_number} encolado para plan {meal_plan_id} "
        f"(días {days_offset_int+1}–{days_offset_int+days_count_int}, kind={normalized_chunk_kind}, mode={chunk_mode}) "
        f"con delay de {delay_days} días"
    )


def _process_pending_shopping_lists():
    """[GAP F FIX] Recalcula shopping lists asincronamente para planes que fallaron su generacion sincrona."""
    try:
        from shopping_calculator import get_shopping_list_delta
        import json
        
        # Buscar planes con status 'partial_no_shopping'
        plans = execute_sql_query("""
            SELECT id, user_id, plan_data 
            FROM meal_plans 
            WHERE plan_data->>'generation_status' = 'partial_no_shopping'
        """)
        
        if not plans:
            return
            
        logger.info(f" [GAP F] Procesando shopping lists pendientes para {len(plans)} planes...")
        
        for p in plans:
            meal_plan_id = p.get('id', 'unknown')
            try:
                user_id = p['user_id']
                plan_data = p['plan_data'] or {}
                
                # Fetch form_data for household and groceryDuration
                snap = execute_sql_query("SELECT pipeline_snapshot FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY created_at DESC LIMIT 1", (meal_plan_id,), fetch_one=True)
                if snap and snap.get('pipeline_snapshot'):
                    snapshot = snap['pipeline_snapshot']
                    if isinstance(snapshot, str): snapshot = json.loads(snapshot)
                    form_data = snapshot.get("form_data", {})
                else:
                    form_data = {}
                    
                household = form_data.get("householdSize", 1)
                
                aggr_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=1.0 * household)
                aggr_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=2.0 * household)
                aggr_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=4.0 * household)
                
                grocery_duration = form_data.get("groceryDuration", "weekly")
                if grocery_duration == "biweekly":
                    aggr_active = aggr_15
                elif grocery_duration == "monthly":
                    aggr_active = aggr_30
                else:
                    aggr_active = aggr_7
                    
                total_generated = plan_data.get('total_days_generated', 0)
                total_requested = plan_data.get('total_days_requested', 7)
                new_status = "complete" if total_generated >= int(total_requested) else "partial"
                
                execute_sql_write("""
                    UPDATE meal_plans 
                    SET plan_data = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    jsonb_set(plan_data, '{aggregated_shopping_list_weekly}', %s::jsonb),
                                    '{aggregated_shopping_list_biweekly}', %s::jsonb
                                ),
                                '{aggregated_shopping_list_monthly}', %s::jsonb
                            ),
                            '{aggregated_shopping_list}', %s::jsonb
                        ),
                        '{generation_status}', %s::jsonb
                    )
                    WHERE id = %s
                """, (
                    json.dumps(aggr_7, ensure_ascii=False),
                    json.dumps(aggr_15, ensure_ascii=False),
                    json.dumps(aggr_30, ensure_ascii=False),
                    json.dumps(aggr_active, ensure_ascii=False),
                    json.dumps(new_status),
                    meal_plan_id
                ))
                logger.info(f" [GAP F] Shopping list recuperada para plan {meal_plan_id}.")
            except Exception as e:
                logger.error(f" [GAP F] Error recuperando shopping list para plan {meal_plan_id}: {e}")
    except Exception as e:
        logger.error(f" [GAP F] Error general procesando pending shopping lists: {e}")

def _record_chunk_metric(
    chunk_id: str,
    meal_plan_id: str,
    user_id: str,
    week_number: int,
    days_count: int,
    duration_ms: int,
    quality_tier: str,
    was_degraded: bool,
    retries: int,
    lag_seconds: int,
    learning_metrics: dict = None,
    error_message: str = None,
    is_rolling_refill: bool = False,
):
    """[GAP G] Inserta una fila en plan_chunk_metrics para análisis histórico."""
    try:
        execute_sql_write("ALTER TABLE plan_chunk_metrics ADD COLUMN IF NOT EXISTS is_rolling_refill BOOLEAN DEFAULT false")
    except Exception:
        pass

    try:
        repeat_pct = None
        rej_viol = 0
        alg_viol = 0
        if learning_metrics:
            repeat_pct = learning_metrics.get("learning_repeat_pct")
            rej_viol = int(learning_metrics.get("rejection_violations") or 0)
            alg_viol = int(learning_metrics.get("allergy_violations") or 0)

        execute_sql_write(
            """
            INSERT INTO plan_chunk_metrics
                (chunk_id, meal_plan_id, user_id, week_number, days_count,
                 duration_ms, quality_tier, was_degraded, retries, lag_seconds,
                 learning_repeat_pct, rejection_violations, allergy_violations, error_message, is_rolling_refill)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk_id, meal_plan_id, user_id, week_number, days_count,
                duration_ms, quality_tier, was_degraded, retries, lag_seconds,
                repeat_pct, rej_viol, alg_viol, error_message, is_rolling_refill
            ),
        )
    except Exception as e:
        # No bloquear al worker por fallas de observabilidad
        logger.warning(f"[GAP G] Error insertando métrica de chunk: {e}")


def _ensure_quality_alert_schema():
    """Crea el esquema mínimo para alertas persistentes si aún no existe."""
    try:
        execute_sql_write(
            """
            CREATE TABLE IF NOT EXISTS system_alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                alert_key TEXT NOT NULL UNIQUE,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                affected_user_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE NULL
            )
            """
        )
        execute_sql_write(
            """
            ALTER TABLE user_profiles
            ADD COLUMN IF NOT EXISTS quality_alert_at TIMESTAMP WITH TIME ZONE
            """
        )
    except Exception as e:
        logger.warning(f"[GAP G] Error asegurando esquema de quality alerts: {e}")


def _persist_quality_degradation_alert(is_refill: bool, ratio: float, degraded: int, total: int):
    """Persiste una alerta deduplicada y marca perfiles afectados para mostrar banner en UI."""
    _ensure_quality_alert_schema()

    tipo = "refill" if is_refill else "initial"
    label = "Refill" if is_refill else "Inicial"
    alert_key = f"degraded_rate_high:{tipo}"

    try:
        degraded_rows = execute_sql_query(
            """
            SELECT DISTINCT user_id
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
              AND COALESCE(is_rolling_refill, false) = %s
              AND (was_degraded OR quality_tier IN ('shuffle', 'edge', 'emergency'))
              AND user_id IS NOT NULL
            """,
            (is_refill,),
            fetch_all=True,
        ) or []
        affected_user_ids = [str(row.get("user_id")) for row in degraded_rows if row.get("user_id")]

        metadata = {
            "window_hours": 24,
            "is_rolling_refill": is_refill,
            "ratio": round(ratio, 4),
            "degraded_chunks": degraded,
            "total_chunks": total,
            "threshold": 0.15,
            "affected_users": len(affected_user_ids),
        }
        message = (
            f"Degraded rate 24h ({label}) = {ratio:.1%} "
            f"({degraded}/{total} chunks). LLM posiblemente inestable."
        )

        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key)
            DO UPDATE SET
                severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                triggered_at = NOW(),
                resolved_at = NULL
            """,
            (
                alert_key,
                "degraded_rate_high",
                "critical",
                f"Calidad degradada alta ({label})",
                message,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(affected_user_ids),
            ),
        )

        if affected_user_ids:
            execute_sql_write(
                """
                UPDATE user_profiles
                SET quality_alert_at = NOW()
                WHERE id = ANY(%s::uuid[])
                """,
                (affected_user_ids,),
            )
    except Exception as e:
        logger.warning(f"[GAP G] Error persistiendo quality degradation alert: {e}")


def _alert_if_degraded_rate_high():
    """[GAP G] Escala degraded-rate alto a persistencia en DB y flag visible por la UI."""
    try:
        rows = execute_sql_query(
            """
            SELECT
                is_rolling_refill,
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE was_degraded OR quality_tier IN ('shuffle', 'edge', 'emergency')) AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY is_rolling_refill
            """,
            fetch_all=True,
        )
        for row in rows:
            total = int(row.get("total") or 0)
            degraded = int(row.get("degraded") or 0)
            is_refill = row.get("is_rolling_refill", False)
            if total >= 10:  # umbral mínimo de muestra
                ratio = degraded / total
                if ratio > 0.15:
                    tipo = "Refill" if is_refill else "Inicial"
                    logger.error(
                        f"[GAP G/ALERT] Degraded rate 24h ({tipo}) = {ratio:.1%} ({degraded}/{total} chunks). "
                        f"LLM possiblemente inestable. Investigar."
                    )
                    _persist_quality_degradation_alert(
                        is_refill=is_refill,
                        ratio=ratio,
                        degraded=degraded,
                        total=total,
                    )
    except Exception as e:
        logger.warning(f"[GAP G] Error en alert de degraded rate: {e}")


def _normalize_meal_name(text: str) -> str:
    if not text:
        return ""
    return strip_accents(str(text).lower()).strip()


def _calculate_chunk_consumption_ratio(previous_chunk_days: list, consumed_records: list) -> dict:
    """Calcula cuánto del chunk previo fue realmente consumido usando nombres de platos.

    [P0-3] Proxy implícito extendido a logging esparso:
    Si el usuario logea 0 comidas → se asume 100 % consumido (comportamiento original).
    Si logea MUY pocas (< 25 % del total planeado), también se usa el proxy porque
    ese nivel de logging no es señal representativa de baja adherencia.
    
    [P0-B] Matching flexible (exact, substring bidireccional, embedding top-3)
    para evitar castigar al usuario por variaciones menores al registrar manualmente.
    """
    _SPARSE_LOGGING_THRESHOLD = 0.25

    planned_pool = {}
    for day in previous_chunk_days or []:
        if not isinstance(day, dict):
            continue
        for meal in day.get("meals", []) or []:
            if not isinstance(meal, dict):
                continue
            meal_name = _normalize_meal_name(meal.get("name"))
            if meal_name and meal.get("status") not in ["swapped_out", "skipped", "rejected"]:
                planned_pool[meal_name] = planned_pool.get(meal_name, 0) + 1

    consumed_list = []
    for record in consumed_records or []:
        meal_name = _normalize_meal_name(record.get("meal_name") or record.get("name"))
        if meal_name:
            consumed_list.append(meal_name)

    planned_total = sum(planned_pool.values())
    explicit_logged = len(consumed_list)
    
    match_stats = {"exact": 0, "substring": 0, "embedding": 0, "unmatched": 0}
    explicit_matched = 0

    def _word_overlap(a: str, b: str) -> int:
        return len(set(a.split()) & set(b.split()))

    for c_name in consumed_list:
        # 1. Exact Match
        if planned_pool.get(c_name, 0) > 0:
            planned_pool[c_name] -= 1
            match_stats["exact"] += 1
            explicit_matched += 1
            continue

        # 2. Substring Match Bidireccional
        matched_p = None
        for p_name, count in planned_pool.items():
            if count > 0 and len(c_name) > 3 and len(p_name) > 3:
                if c_name in p_name or p_name in c_name:
                    matched_p = p_name
                    break
        
        if matched_p:
            planned_pool[matched_p] -= 1
            match_stats["substring"] += 1
            explicit_matched += 1
            continue

        # 3. Embedding Match Cosine >= 0.85 (Top 3 candidatos)
        available_planned = [p for p, count in planned_pool.items() if count > 0]
        if available_planned:
            candidates = sorted(available_planned, key=lambda p: _word_overlap(c_name, p), reverse=True)[:3]
            try:
                c_emb = get_embedding(c_name)
                best_score = -1.0
                best_cand = None
                
                for cand in candidates:
                    cand_emb = get_embedding(cand)
                    score = cosine_similarity(c_emb, cand_emb)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_score >= 0.85 and best_cand:
                    planned_pool[best_cand] -= 1
                    match_stats["embedding"] += 1
                    explicit_matched += 1
                    continue
            except Exception as e:
                logger.warning(f"[P0-B/MATCH] Error en embedding para '{c_name}': {e}")
                
        # Unmatched
        match_stats["unmatched"] += 1

    logger.info(f"[P0-B/MATCH] Auditoría de Adherencia: exact={match_stats['exact']} substr={match_stats['substring']} emb={match_stats['embedding']} unmatched={match_stats['unmatched']}")

    # [P0-3] Proxy implícito: sin logs O logging esparso (< 25 % de lo planeado).
    sparse_logging = (
        planned_total > 0
        and explicit_logged > 0
        and explicit_logged < max(2, planned_total * _SPARSE_LOGGING_THRESHOLD)
    )
    # [P0-1] Distinguir zero-log de sparse-log. Antes ambos colapsaban en used_implicit_proxy
    # con ratio=1.0; eso permitía que un chunk generara sin NINGUNA señal real de adherencia
    # (el caso más frágil del aprendizaje continuo). Ahora exponemos zero_log_proxy aparte
    # para que el caller pueda pausar / forzar variedad / notificar de forma diferenciada.
    zero_log_proxy = planned_total > 0 and explicit_logged == 0
    use_implicit_proxy = zero_log_proxy or sparse_logging
    matched = planned_total if use_implicit_proxy else explicit_matched
    ratio = matched / planned_total if planned_total else 1.0

    return {
        "ratio": round(ratio, 4),
        "matched_meals": matched,
        "planned_meals": planned_total,
        "explicit_matched_meals": explicit_matched,
        "explicit_logged_meals": explicit_logged,
        "used_implicit_proxy": use_implicit_proxy,
        "sparse_logging_proxy": sparse_logging,
        "zero_log_proxy": zero_log_proxy,
        "match_breakdown": match_stats,
    }


def _compute_prev_chunk_meal_breakdown(
    plan_days: list,
    prev_offset: int,
    prev_count: int,
    consumed_records: list,
    prev_chunk_number: int,
) -> dict | None:
    """[P0-5] Devuelve qué platos del chunk previo INMEDIATO consumió/saltó el usuario.

    A diferencia de `_meal_level_adherence` (EMA por TIPO de comida acumulada en todos
    los chunks), este breakdown es chunk-específico y por NOMBRE de plato — el insumo
    directo para que el LLM refuerce variantes de lo aceptado y evite repetir lo saltado.

    Devuelve None si no hay datos planeados (chunk_number<2, sin prior_days, etc.) para
    que el caller pueda omitir la inyección sin condicionales.
    """
    if not plan_days or prev_count <= 0 or prev_chunk_number < 1:
        return None

    prev_start_day = prev_offset + 1
    prev_end_day = prev_offset + prev_count

    planned_names: list[str] = []
    seen: set = set()
    for day in plan_days:
        if not isinstance(day, dict):
            continue
        day_num = int(day.get("day") or 0)
        if not (prev_start_day <= day_num <= prev_end_day):
            continue
        for meal in day.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            if meal.get("status") in ("swapped_out", "skipped", "rejected"):
                continue
            raw_name = meal.get("name")
            if not raw_name:
                continue
            norm = _normalize_meal_name(raw_name)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            planned_names.append(raw_name)

    if not planned_names:
        return None

    consumed_norm = set()
    for record in consumed_records or []:
        if not isinstance(record, dict):
            continue
        norm = _normalize_meal_name(record.get("meal_name") or record.get("name"))
        if norm:
            consumed_norm.add(norm)

    consumed: list[str] = []
    skipped: list[str] = []
    for raw_name in planned_names:
        if _normalize_meal_name(raw_name) in consumed_norm:
            consumed.append(raw_name)
        else:
            skipped.append(raw_name)

    if not consumed and not skipped:
        return None

    return {
        "chunk_number": int(prev_chunk_number),
        "prev_start_day": prev_start_day,
        "prev_end_day": prev_end_day,
        "consumed_meals": consumed[:8],
        "skipped_meals": skipped[:8],
        "consumed_count": len(consumed),
        "skipped_count": len(skipped),
    }


def _resolve_previous_chunk_window(meal_plan_id: str, week_number: int, days_offset: int, total_days_requested: int = None) -> tuple[int, int]:
    """Encuentra el offset/count del chunk anterior para gating por consumo real."""
    if week_number <= 2:
        return 0, max(0, int(days_offset))

    previous_chunk = execute_sql_query(
        """
        SELECT days_offset, days_count
        FROM plan_chunk_queue
        WHERE meal_plan_id = %s AND week_number = %s
        ORDER BY updated_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        (str(meal_plan_id), int(week_number) - 1),
        fetch_one=True
    )
    if previous_chunk:
        prev_offset = max(0, int(previous_chunk.get("days_offset") or 0))
        prev_count = max(1, int(previous_chunk.get("days_count") or 1))
        return prev_offset, prev_count

    # [P0-5] Fallback: recalcular desde split_with_absorb si sabemos el total.
    # Evita asumir PLAN_CHUNK_SIZE fijo para planes de 15/30 días con chunks de 3 o 4.
    if total_days_requested and int(total_days_requested) > 0:
        from constants import split_with_absorb as _split
        chunks = _split(int(total_days_requested))
        prev_idx = int(week_number) - 2  # 0-indexed del chunk anterior
        if 0 <= prev_idx < len(chunks):
            prev_offset = sum(chunks[:prev_idx])
            prev_count = chunks[prev_idx]
            return prev_offset, prev_count

    from constants import PLAN_CHUNK_SIZE
    prev_count = min(max(1, PLAN_CHUNK_SIZE), max(1, int(days_offset)))
    prev_offset = max(0, int(days_offset) - prev_count)
    return prev_offset, prev_count


def _check_chunk_learning_ready(user_id: str, meal_plan_id: str, week_number: int, days_offset: int, plan_data: dict, snapshot: dict) -> dict:
    """Verifica si el chunk previo fue suficientemente consumido para habilitar el siguiente."""
    if int(week_number) <= 1 or int(days_offset) <= 0:
        return {"ready": True, "reason": "first_chunk"}

    form_data = (snapshot or {}).get("form_data", {}) or {}
    plan_start_date_str = form_data.get("_plan_start_date")
    if not plan_start_date_str:
        return {"ready": True, "reason": "missing_plan_start_date"}

    total_days_requested = (snapshot or {}).get("totalDays") or form_data.get("totalDays")
    prev_offset, prev_count = _resolve_previous_chunk_window(meal_plan_id, week_number, days_offset, total_days_requested)
    prev_start_day = prev_offset + 1
    prev_end_day = prev_offset + prev_count
    previous_chunk_days = [
        d for d in (plan_data.get("days", []) or [])
        if isinstance(d, dict) and prev_start_day <= int(d.get("day") or 0) <= prev_end_day
    ]
    if not previous_chunk_days:
        return {"ready": True, "reason": "missing_previous_chunk_days"}

    from constants import safe_fromisoformat, CHUNK_PROACTIVE_MARGIN_DAYS as _proactive_margin
    plan_start_dt = safe_fromisoformat(plan_start_date_str)

    # [P0-1 FIX] Temporal gate: el chunk N+1 sólo puede aprender de N una vez el último día
    # de N haya transcurrido (margin=0, default). Antes la comparación era `>` lo que permitía
    # disparar el gate el MISMO día en que terminaba el chunk previo, antes de que el usuario
    # hubiera tenido cena/última comida — produciendo "aprendizaje" con un día incompleto.
    # Con `>=`, margin=0 bloquea cuando today<=prev_end y sólo pasa cuando today>prev_end.
    # Para usuarios que reporten gaps al despertar, subir CHUNK_PROACTIVE_MARGIN_DAYS a 1
    # restaura la posibilidad de disparar el chunk el mismo día que termina el previo.
    from datetime import datetime as _dt_p0b, timezone as _tz_p0b
    _today_utc = _dt_p0b.now(_tz_p0b.utc).date()
    _shift_days_accumulated = int(plan_data.get("_shift_days_accumulated", 0))
    _prev_end_date = (plan_start_dt + timedelta(days=prev_end_day - 1 + _shift_days_accumulated)).date()
    _days_until_prev_end = (_prev_end_date - _today_utc).days  # >=0 = último día aún no concluyó
    if _days_until_prev_end >= _proactive_margin:
        _prev_start_iso_log = (plan_start_dt + timedelta(days=prev_offset)).isoformat()
        return {
            "ready": False,
            "reason": "prior_chunk_not_elapsed",
            "ratio": None,
            "previous_chunk_start_day": prev_start_day,
            "previous_chunk_end_day": prev_end_day,
            "previous_chunk_start_iso": _prev_start_iso_log,
            "prev_end_date": _prev_end_date.isoformat(),
            "days_until_prev_end": _days_until_prev_end,
        }

    prev_start_iso = (plan_start_dt + timedelta(days=prev_offset)).isoformat()
    consumed_records = get_consumed_meals_since(user_id, prev_start_iso) or []
    ratio_info = _calculate_chunk_consumption_ratio(previous_chunk_days, consumed_records)
    ratio = ratio_info["ratio"]

    # [P0-1] zero_log_proxy=True significa que NO hubo logs reales del chunk previo.
    # El gate "técnicamente" pasa con ratio=1.0 por el proxy, pero no hay señal de aprendizaje.
    # Marcamos ready=False en este caso aunque el ratio sea 1.0, para que el worker pueda
    # decidir entre deferral/pausa/forzado con criterio explícito.
    is_zero_log = ratio_info.get("zero_log_proxy", False)
    ratio_ready = ratio >= CHUNK_LEARNING_READY_MIN_RATIO

    # [P0-D] Si no hay logs explícitos, mirar mutaciones del inventario como proxy de actividad.
    # Si el usuario tocó su nevera ≥ N veces desde el inicio del chunk previo (auto-deducción
    # al consumir, edición manual de cantidades, agregar items), asumimos que está siguiendo
    # el plan aunque no loguee. Mantenemos zero_log_proxy=True para que el worker sepa que
    # el aprendizaje será débil, pero ya no bloqueamos la generación silenciosamente.
    inventory_proxy_used = False
    inventory_mutations = 0
    _signal_too_weak = is_zero_log or ratio_info.get("sparse_logging_proxy")
    if _signal_too_weak:
        try:
            activity = get_inventory_activity_since(user_id, prev_start_iso)
            inventory_mutations = int(activity.get("mutations_count") or 0)
            if inventory_mutations >= CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS:
                inventory_proxy_used = True
        except Exception as e:
            logger.debug(f"[P0-NEW2] No se pudo medir actividad de inventario para {user_id}: {e}")

    return {
        "ready": (ratio_ready and not _signal_too_weak) or inventory_proxy_used,
        "ratio": ratio,
        "matched_meals": ratio_info["matched_meals"],
        "planned_meals": ratio_info["planned_meals"],
        "explicit_matched_meals": ratio_info.get("explicit_matched_meals", 0),
        "explicit_logged_meals": ratio_info.get("explicit_logged_meals", 0),
        "used_implicit_proxy": ratio_info.get("used_implicit_proxy", False),
        "sparse_logging_proxy": ratio_info.get("sparse_logging_proxy", False),
        "zero_log_proxy": is_zero_log,
        "inventory_proxy_used": inventory_proxy_used,
        "inventory_mutations": inventory_mutations,
        "previous_chunk_start_day": prev_start_day,
        "previous_chunk_end_day": prev_end_day,
        "previous_chunk_start_iso": prev_start_iso,
    }


def _calculate_learning_metrics(new_days: list, prior_meals: list, prior_days: list, rejected_names: list, allergy_keywords: list, fatigued_ingredients: list) -> dict:
    """[GAP F] Calcula métricas para validar que el aprendizaje continuo funciona.

    Devuelve:
      - learning_repeat_pct: % de prior_meals que reaparecen en new_days (idealmente 0-10%)
      - rejection_violations: # de nombres rechazados que reaparecen (debe ser 0)
      - allergy_violations: # de keywords de alergia en ingredientes de new_days (debe ser 0)
      - fatigued_violations: # de ingredientes fatigados que reaparecen (informativo)
      - sample_repeats: primeros 5 nombres que repitieron (para debug)
    """
    from constants import get_nutritional_category, normalize_ingredient_for_tracking

    def _norm(s: str) -> str:
        if not s:
            return ""
        try:
            return strip_accents(str(s).lower()).strip()
        except Exception:
            return str(s).lower().strip()

    def _normalize_fatigue_marker(value: str) -> tuple[str, str]:
        normalized = _norm(value)
        if not normalized:
            return "ingredient", ""
        if normalized.startswith("[") and "]" in normalized:
            prefix, remainder = normalized.split("]", 1)
            prefix_alpha = "".join(ch for ch in prefix if ch.isalpha())
            if prefix_alpha.startswith("categoria") or prefix_alpha.startswith("category"):
                return "category", remainder.strip()
        return "ingredient", normalized

    def _extract_bases_from_meal(meal: dict) -> list:
        bases = []
        if not isinstance(meal, dict):
            return bases
        for ing in meal.get("ingredients", []) or []:
            raw_ing = ""
            if isinstance(ing, dict):
                raw_ing = ing.get("name") or ing.get("display_string") or ""
            else:
                raw_ing = str(ing)
            base = normalize_ingredient_for_tracking(raw_ing)
            if base:
                bases.append(base)
        return bases

    prior_set = {_norm(m) for m in (prior_meals or []) if m}
    rejected_set = {_norm(m) for m in (rejected_names or []) if m}
    allergy_set = {_norm(k) for k in (allergy_keywords or []) if k and len(k) > 2}
    fatigued_ingredient_set = set()
    fatigued_category_set = set()
    for fatigue_marker in fatigued_ingredients or []:
        marker_kind, marker_value = _normalize_fatigue_marker(fatigue_marker)
        if not marker_value:
            continue
        if marker_kind == "category":
            fatigued_category_set.add(marker_value)
        else:
            fatigued_ingredient_set.add(marker_value)

    new_meal_names = []
    new_ingredients_blob = []
    new_ingredient_categories = set()
    prior_meal_bases = set()
    new_meal_bases = []
    repeated_base_samples = []
    for d in (new_days or []):
        if not isinstance(d, dict):
            continue
        for m in d.get("meals", []) or []:
            if not isinstance(m, dict):
                continue
            name = _norm(m.get("name", ""))
            if name:
                new_meal_names.append(name)
            meal_bases = _extract_bases_from_meal(m)
            if meal_bases:
                new_meal_bases.append(sorted(set(meal_bases)))
            for ing in m.get("ingredients", []) or []:
                raw_ing = ""
                if isinstance(ing, dict):
                    raw_ing = ing.get("name") or ing.get("display_string") or ""
                else:
                    raw_ing = str(ing)
                normalized_ing = _norm(raw_ing)
                if normalized_ing:
                    new_ingredients_blob.append(normalized_ing)
                normalized_base = normalize_ingredient_for_tracking(raw_ing)
                normalized_category = _norm(get_nutritional_category(normalized_base)) if normalized_base else ""
                if normalized_category:
                    new_ingredient_categories.add(normalized_category)

    for d in (prior_days or []):
        if not isinstance(d, dict):
            continue
        for m in d.get("meals", []) or []:
            prior_meal_bases.update(_extract_bases_from_meal(m))

    repeats = [n for n in new_meal_names if n in prior_set]
    rejection_hits = [n for n in new_meal_names if n in rejected_set]

    # [P1-8 FIX]: Comparar alergias sobre la lista estructurada de ingredientes para evitar falsos positivos
    allergy_hits = [k for k in allergy_set if any(k in ing for ing in new_ingredients_blob)]
    fatigued_hits = [k for k in fatigued_ingredient_set if any(k in ing for ing in new_ingredients_blob)]
    fatigued_category_hits = [k for k in fatigued_category_set if k in new_ingredient_categories]
    all_fatigued_hits = fatigued_hits + fatigued_category_hits

    repeated_base_meals = 0
    for meal_name, bases in zip(new_meal_names, new_meal_bases):
        repeated_bases = [b for b in bases if b in prior_meal_bases]
        if repeated_bases:
            repeated_base_meals += 1
            if len(repeated_base_samples) < 5:
                repeated_base_samples.append({"meal": meal_name, "bases": repeated_bases[:3]})

    total_new = len(new_meal_names)
    repeat_pct = round((len(repeats) / total_new) * 100.0, 2) if total_new else 0.0
    ingredient_base_repeat_pct = round((repeated_base_meals / total_new) * 100.0, 2) if total_new else 0.0

    return {
        "total_new_meals": total_new,
        "learning_repeat_pct": repeat_pct,
        "ingredient_base_repeat_pct": ingredient_base_repeat_pct,
        "rejection_violations": len(rejection_hits),
        "allergy_violations": len(allergy_hits),
        "fatigued_violations": len(all_fatigued_hits),
        "sample_repeats": repeats[:5],
        "sample_repeated_bases": repeated_base_samples,
        "sample_rejection_hits": rejection_hits[:5],
        "sample_allergy_hits": allergy_hits[:5],
        "prior_meals_count": len(prior_set),
        "prior_meal_bases_count": len(prior_meal_bases),
        "rejected_count": len(rejected_set),
        "allergy_keywords_count": len(allergy_set),
    }


def _build_filtered_edge_recipe_day(
    allergies: list | tuple | None,
    dislikes: list | tuple | None,
    diet: str = "",
    pantry_items: list | None = None,
) -> dict | None:
    """Construye un Edge Recipe usando solo ingredientes permitidos para el usuario.

    [P0-C FIX] Si se pasa pantry_items, intersecta cada categoría del catálogo con los
    ingredientes disponibles en la nevera. Si la intersección está vacía para una categoría,
    usa el catálogo completo filtrado como fallback para no bloquear la generación.
    """
    from constants import _get_fast_filtered_catalogs, normalize_ingredient_for_tracking

    allergies = tuple(a for a in (allergies or []) if isinstance(a, str) and a.strip())
    dislikes = tuple(d for d in (dislikes or []) if isinstance(d, str) and d.strip())
    filtered_proteins, filtered_carbs, filtered_veggies, _ = _get_fast_filtered_catalogs(
        allergies,
        dislikes,
        (diet or "").strip().lower(),
    )

    if not filtered_proteins or not filtered_carbs or not filtered_veggies:
        return None

    # [P0-C FIX] Preferir ingredientes disponibles en la nevera cuando se proporciona.
    if pantry_items:
        pantry_bases: set[str] = set()
        for item in pantry_items:
            if not item:
                continue
            normalized = normalize_ingredient_for_tracking(str(item))
            if normalized:
                pantry_bases.add(normalized)
            pantry_bases.add(strip_accents(str(item).lower().strip()))

        def _pantry_intersect(catalog: list) -> list:
            """Keep only catalog items whose normalized name matches a pantry base."""
            matched = [
                c for c in catalog
                if any(
                    (pb and len(pb) > 2 and (pb in strip_accents(c.lower()) or strip_accents(c.lower()) in pb))
                    for pb in pantry_bases
                )
            ]
            return matched if matched else catalog  # fallback: use full filtered catalog

        pantry_proteins = _pantry_intersect(filtered_proteins)
        pantry_carbs = _pantry_intersect(filtered_carbs)
        pantry_veggies = _pantry_intersect(filtered_veggies)

        _any_narrowed = (
            len(pantry_proteins) < len(filtered_proteins)
            or len(pantry_carbs) < len(filtered_carbs)
            or len(pantry_veggies) < len(filtered_veggies)
        )
        if not _any_narrowed:
            logger.debug("[P0-C] Edge Recipe: ninguna categoría intersectó con la nevera, usando catálogo completo.")
    else:
        pantry_proteins, pantry_carbs, pantry_veggies = filtered_proteins, filtered_carbs, filtered_veggies

    def _cap_ingredient(ingredient: str, default_g: int) -> str:
        if not pantry_items:
            return f"{default_g}g {ingredient}"
        from constants import normalize_ingredient_for_tracking
        norm = normalize_ingredient_for_tracking(ingredient)
        if not norm:
            return f"{default_g}g {ingredient}"
            
        total_g = 0.0
        try:
            from shopping_calculator import _parse_quantity
            from constants import _to_base_unit
            for p in pantry_items:
                p_norm = normalize_ingredient_for_tracking(str(p))
                if p_norm == norm:
                    q, u, _ = _parse_quantity(str(p))
                    bq, bu = _to_base_unit(q, u)
                    if bu == 'g' or bu == 'ml':
                        total_g += bq
        except Exception as e:
            logger.warning(f"[P0-NEW1] Error parsing quantity in Edge Recipe builder: {e}")

        if total_g > 0:
            capped_g = min(default_g, int(total_g))
            return f"{capped_g}g {ingredient}"
        return f"{default_g}g {ingredient}"

    breakfast_protein = _cap_ingredient(random.choice(pantry_proteins), 150)
    breakfast_carb = _cap_ingredient(random.choice(pantry_carbs), 100)
    lunch_protein = _cap_ingredient(random.choice(pantry_proteins), 200)
    lunch_carb = _cap_ingredient(random.choice(pantry_carbs), 150)
    lunch_veggie = _cap_ingredient(random.choice(pantry_veggies), 100)
    dinner_protein = _cap_ingredient(random.choice(pantry_proteins), 150)
    dinner_veggie = _cap_ingredient(random.choice(pantry_veggies), 150)

    return {
        "day": 0,
        "day_name": "",
        "meals": [
            {
                "name": f"Desayuno: {breakfast_protein} con {breakfast_carb}",
                "type": "Desayuno",
                "description": "Desayuno tradicional (Edge Recipe)",
                "ingredients": [breakfast_protein, breakfast_carb],
                "macros": {"calories": 400, "protein": 20, "carbs": 35, "fat": 15},
                "instructions": ["Preparar ingredientes según método tradicional"],
            },
            {
                "name": f"Almuerzo: {lunch_protein} con {lunch_carb} y {lunch_veggie}",
                "type": "Almuerzo",
                "description": "Almuerzo tradicional (Edge Recipe)",
                "ingredients": [lunch_protein, lunch_carb, lunch_veggie],
                "macros": {"calories": 500, "protein": 35, "carbs": 60, "fat": 10},
                "instructions": ["Cocinar a la plancha o al vapor"],
            },
            {
                "name": f"Cena: {dinner_protein} con {dinner_veggie}",
                "type": "Cena",
                "description": "Cena ligera (Edge Recipe)",
                "ingredients": [dinner_protein, dinner_veggie],
                "macros": {"calories": 350, "protein": 25, "carbs": 20, "fat": 15},
                "instructions": ["Saltear ingredientes juntos"],
            },
        ],
    }


def _detect_and_escalate_stuck_chunks():
    """[P1-3] Detecta chunks atrasados (lag > 24h sin pickup) y los escala.

    Acciones:
      - Loguea métricas [P1-3/STUCK-DETECT].
      - Si lag > 24h: marca escalated_at, baja execute_after a NOW() para que el
        worker los tome en el siguiente tick.
      - Idempotencia: si ya escalado en los últimos 30 min, skip.
      - Notificación: si lag > 4h (siempre cierto si lag > 24h), envía push suave
        para avisar que el plan está demorado (solo una vez por ventana de escalado).
      - Si lag > 72h y attempts >= 3: marca como 'failed' y envía push de rescate fallido.
    """
    try:
        # 1. Detectar stuck (>24h sin pickup) que aún NO fueron escalados o lo fueron hace > 30 min
        stuck_rows = execute_sql_query("""
            SELECT id, user_id, meal_plan_id, week_number, attempts,
                   EXTRACT(EPOCH FROM (NOW() - execute_after))::int AS lag_seconds,
                   GREATEST(
                       0,
                       EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                   )::int AS effective_lag_seconds,
                   escalated_at
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                  ) > 86400
              AND (escalated_at IS NULL OR escalated_at < NOW() - INTERVAL '30 minutes')
            ORDER BY execute_after ASC
            LIMIT 50
        """) or []

        if stuck_rows:
            logger.warning(f" [P1-3/STUCK-DETECT] {len(stuck_rows)} chunks stuck (effective_lag>24h) detectados. Escalando...")
            
            # Notificación push proactiva (deduplicada por usuario)
            notified_users = set()
            
            for r in stuck_rows:
                lag_s = r.get('effective_lag_seconds') or 0
                lag_h = round(lag_s / 3600.0, 1)
                effective_lag_h = round((r.get('effective_lag_seconds') or 0) / 3600.0, 1)
                
                _already_escalated = r.get('escalated_at') is not None
                _tag = "[RE-ESCALANDO]" if _already_escalated else "[ESCALANDO]"
                
                logger.warning(
                    f"   ↳ {_tag} chunk {r['id']} plan={r['meal_plan_id']} week={r['week_number']} "
                    f"lag={lag_h}h effective_lag={effective_lag_h}h attempts={r.get('attempts', 0)}"
                )

                # Push guard: solo emitir si lag > 4h y es la primera escalación o ha pasado tiempo suficiente
                # (la idempotencia de 30 min ya filtra el grueso del spam)
                if lag_s > 4 * 3600:
                    uid = str(r['user_id'])
                    if uid not in notified_users:
                        notified_users.add(uid)
                        try:
                            import threading
                            from utils_push import send_push_notification
                            threading.Thread(
                                target=send_push_notification,
                                kwargs={
                                    "user_id": uid,
                                    "title": "⚡ Optimizando tu plan",
                                    "body": "Estamos terminando de ajustar los últimos detalles de tu plan. Estará listo en breve.",
                                    "url": "/dashboard"
                                },
                                daemon=True
                            ).start()
                        except Exception as push_err:
                            logger.warning(f" [P1-3/STUCK-DETECT] Error enviando push proactivo a {uid}: {push_err}")

            execute_sql_write("""
                UPDATE plan_chunk_queue
                SET escalated_at = NOW(),
                    execute_after = NOW(),
                    updated_at = NOW()
                WHERE status IN ('pending', 'stale')
                  AND GREATEST(
                        0,
                        EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                      ) > 86400
                  AND (escalated_at IS NULL OR escalated_at < NOW() - INTERVAL '30 minutes')
            """)

        # 2. Detectar stuck terminal (>72h y ya intentado 3+ veces) → fail + notify
        terminal = execute_sql_query("""
            SELECT id, user_id, meal_plan_id, week_number
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND escalated_at < NOW() - INTERVAL '72 hours'
              AND COALESCE(attempts, 0) >= 3
            LIMIT 50
        """) or []

        if terminal:
            ids = [str(r['id']) for r in terminal]
            execute_sql_write(
                "UPDATE plan_chunk_queue SET status = 'failed', updated_at = NOW() WHERE id = ANY(%s::uuid[])",
                (ids,)
            )
            logger.error(f" [P1-3/STUCK-DETECT] {len(terminal)} chunks marcados como 'failed' tras 72h sin recuperación.")

            # Push de notificación por usuario afectado (deduplicado)
            notified_users = set()
            for r in terminal:
                uid = str(r['user_id'])
                if uid in notified_users:
                    continue
                notified_users.add(uid)
                try:
                    import threading
                    from utils_push import send_push_notification
                    threading.Thread(
                        target=send_push_notification,
                        kwargs={
                            "user_id": uid,
                            "title": "⚠️ Tu plan necesita atención",
                            "body": "No pudimos generar parte de tu plan a largo plazo. Entra a la app para regenerarlo.",
                            "url": "/dashboard"
                        },
                        daemon=True
                    ).start()
                except Exception as push_err:
                    logger.warning(f" [P1-3/STUCK-DETECT] Error enviando push de fallo terminal a {uid}: {push_err}")
    except Exception as e:
        logger.error(f" [P1-3/STUCK-DETECT] Error en _detect_and_escalate_stuck_chunks: {e}")


def process_plan_chunk_queue(target_plan_id=None):
    """Worker que genera las semanas 2-4 de planes de largo plazo. Corre cada minuto vía APScheduler."""
    import json

    _process_pending_shopping_lists()
    _recover_pantry_paused_chunks()

    # [GAP 3 FIX: Cleanup chunks huérfanos]
    # [P0-4] Liberar reservas antes de cancelar para evitar phantom reserved_quantity
    try:
        _orphan_chunks = execute_sql_query("""
            SELECT id, user_id FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale', 'processing')
            AND meal_plan_id::text NOT IN (SELECT id::text FROM meal_plans)
        """)
        for _oc in (_orphan_chunks or []):
            release_chunk_reservations(str(_oc["user_id"]), str(_oc["id"]))
        execute_sql_write("""
            UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW()
            WHERE status IN ('pending', 'stale', 'processing')
            AND meal_plan_id::text NOT IN (SELECT id::text FROM meal_plans)
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error limpiando chunks huérfanos: {e}")

    # [GAP 7 FIX: Garbage collection eager de pipeline_snapshots]
    # Libera memoria masiva (~10MB por chunk) inmediatamente luego de que un chunk termina o se cancela.
    # [GAP C] Preservar snapshots de chunks degradados por 48h para permitir /regen-degraded.
    try:
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET pipeline_snapshot = '{}'::jsonb, updated_at = NOW()
            WHERE pipeline_snapshot::text != '{}'
            AND (
                status = 'cancelled'
                OR (status = 'completed' AND quality_tier = 'llm')
                OR (status = 'completed' AND quality_tier IN ('shuffle', 'edge', 'emergency')
                    AND updated_at < NOW() - INTERVAL '48 hours')
            )
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error limpiando snapshots pesados: {e}")

    # [GAP 11 FIX: Purga definitiva de chunks cancelados > 48h]
    try:
        execute_sql_write("""
            DELETE FROM plan_chunk_queue
            WHERE status = 'cancelled'
            AND updated_at < NOW() - INTERVAL '48 hours'
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error purgando chunks cancelados: {e}")

    # Rescate de zombies: chunks procesando por más de 10 minutos
    try:
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET attempts = COALESCE(attempts, 0) + 1,
                status = CASE WHEN COALESCE(attempts, 0) + 1 >= 5 THEN 'failed' ELSE 'pending' END,
                execute_after = NOW() + make_interval(mins => 5),
                updated_at = NOW()
            WHERE status = 'processing'
            AND updated_at < NOW() - INTERVAL '10 minutes'
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error rescatando zombies: {e}")

    # [GAP B FIX: Serializar chunks por meal_plan_id y procesar en orden secuencial]
    # [GAP A] Capturamos lag_seconds_at_pickup en el mismo UPDATE para tener métrica de SLA.
    if target_plan_id:
        query = """
            UPDATE plan_chunk_queue
            SET status = 'processing',
                updated_at = NOW(),
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int,
                effective_lag_seconds_at_pickup = GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                )::int
            WHERE id IN (
                SELECT q1.id FROM plan_chunk_queue q1
                WHERE q1.status IN ('pending', 'stale')
                AND q1.meal_plan_id = %s
                AND q1.id = (
                    SELECT q2.id FROM plan_chunk_queue q2
                    WHERE q2.meal_plan_id = q1.meal_plan_id
                    AND q2.status IN ('pending', 'stale')
                    ORDER BY q2.week_number ASC
                    LIMIT 1
                )
                ORDER BY q1.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, effective_lag_seconds_at_pickup,
                      expected_preemption_seconds, escalated_at, attempts;
        """
        params = (target_plan_id,)
    else:
        query = """
            UPDATE plan_chunk_queue
            SET status = 'processing',
                updated_at = NOW(),
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int,
                effective_lag_seconds_at_pickup = GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                )::int
            WHERE id IN (
                SELECT q1.id FROM plan_chunk_queue q1
                WHERE q1.status IN ('pending', 'stale')
                AND q1.execute_after <= NOW()
                AND q1.meal_plan_id NOT IN (
                    SELECT meal_plan_id FROM plan_chunk_queue WHERE status = 'processing'
                )
                AND q1.meal_plan_id NOT IN (
                    SELECT meal_plan_id FROM plan_chunk_queue
                    WHERE reservation_status = 'partial'
                    AND updated_at > NOW() - INTERVAL '5 minutes'
                )
                AND q1.id = (
                    SELECT q2.id FROM plan_chunk_queue q2
                    WHERE q2.meal_plan_id = q1.meal_plan_id
                    AND q2.status IN ('pending', 'stale')
                    ORDER BY q2.week_number ASC
                    LIMIT 1
                )
                ORDER BY q1.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 3
            )
            RETURNING id, user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, effective_lag_seconds_at_pickup,
                      expected_preemption_seconds, escalated_at, attempts;
        """
        params = ()

    try:
        tasks = execute_sql_write(query, params, returning=True)
    except Exception as e:
        if "relation" not in str(e).lower():
            logger.error(f" [CHUNK] Error obteniendo tasks de plan_chunk_queue: {e}")
        return

    if not tasks:
        return

    # [GAP A] Resumen de SLA por batch: cuántos chunks tomados y con qué lag
    try:
        lags = [int(t.get("effective_lag_seconds_at_pickup") or 0) for t in tasks]
        if lags:
            max_lag_h = max(lags) / 3600.0
            avg_lag_h = (sum(lags) / len(lags)) / 3600.0
            escalated_count = sum(1 for t in tasks if t.get("escalated_at") is not None)
            if max_lag_h >= 1.0 or escalated_count > 0:
                logger.warning(
                    f"📊 [GAP A/SLA] Pickup batch: n={len(tasks)} avg_effective_lag={avg_lag_h:.1f}h "
                    f"max_effective_lag={max_lag_h:.1f}h escalated={escalated_count}"
                )
    except Exception:
        pass

    logger.info(f" [CHUNK] Procesando {len(tasks)} chunks de planes en background.")

    def _chunk_worker(task):
        task_id = task["id"]
        user_id = str(task["user_id"])
        meal_plan_id = str(task["meal_plan_id"])
        week_number = task["week_number"]
        days_offset = task["days_offset"]
        days_count = task["days_count"]
        lag_seconds = int(task.get("effective_lag_seconds_at_pickup") or 0)
        # [P0-6] CAS token: attempts al momento del pickup. El zombie rescue siempre lo incrementa,
        # lo que nos permite detectar si fuimos desplazados aun cuando el nuevo worker ya tomó el chunk.
        _pickup_attempts = int(task.get("attempts") or 0)
        # [GAP G] Métricas de observabilidad del chunk
        import time as _t
        chunk_start_ts = _t.time()
        # [GAP F] Defaults para que existan si no entramos al path LLM
        prior_meals = []
        rejected_meal_names = []
        _fatigued_ingredients = []
        _allergy_keywords = []
        learning_metrics = None

        snap = task["pipeline_snapshot"]
        if isinstance(snap, str):
            snap = json.loads(snap)
            
        chunk_kind = task.get("chunk_kind") or ("rolling_refill" if snap.get("_is_rolling_refill", False) else "initial_plan")
        is_rolling_refill = chunk_kind == "rolling_refill"
        form_data = copy.deepcopy(snap.get("form_data", {}))
        snapshot_form_data = snap.get("form_data", {}) or {}

        try:
            # [GAP 3 FIX: GUARD validar plan activo y no-fallido]
            active_plan = execute_sql_query(
                "SELECT id, plan_data->>'generation_status' as status FROM meal_plans WHERE id = %s",
                (meal_plan_id,), fetch_one=True
            )
            if not active_plan:
                logger.info(f" [CHUNK] Plan {meal_plan_id} no existe. Cancelando chunk {week_number}.")
                release_chunk_reservations(user_id, str(task_id))
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return
                
            if active_plan.get('status') == 'failed':
                logger.info(f" [CHUNK] Plan {meal_plan_id} esta fallido. Cancelando chunk {week_number}.")
                release_chunk_reservations(user_id, str(task_id))
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return

            plan_row_prior = execute_sql_query(
                "SELECT plan_data FROM meal_plans WHERE id = %s",
                (meal_plan_id,), fetch_one=True
            )
            prior_plan_data = plan_row_prior.get("plan_data", {}) if plan_row_prior else {}
            # [P0-2] Sello CAS: capturar el timestamp ANTES del LLM call para poder
            # compararlo dentro del bloque FOR UPDATE y detectar modificaciones externas.
            pre_read_modified_at = prior_plan_data.get('_plan_modified_at') if prior_plan_data else None
            learning_ready = _check_chunk_learning_ready(
                user_id=user_id,
                meal_plan_id=meal_plan_id,
                week_number=week_number,
                days_offset=days_offset,
                plan_data=prior_plan_data,
                snapshot=snap,
            )
            learning_ready_ratio = learning_ready.get("ratio")
            learning_ready_deferrals = int(snap.get("_learning_ready_deferrals") or 0)
            if learning_ready_ratio is not None:
                _proxy_flags = []
                if learning_ready.get("used_implicit_proxy"):
                    _proxy_flags.append("sparse_proxy" if learning_ready.get("sparse_logging_proxy") else "zero_log_proxy")
                if learning_ready.get("inventory_proxy_used"):
                    _proxy_flags.append(f"inv_proxy({learning_ready.get('inventory_mutations', 0)}m)")
                _proxy_tag = f" [{','.join(_proxy_flags)}]" if _proxy_flags else ""
                logger.info(
                    f"[CHUNK/LEARNING-READY] plan={meal_plan_id} chunk={week_number} "
                    f"ratio={learning_ready_ratio:.0%} matched={learning_ready.get('matched_meals', 0)}/"
                    f"{learning_ready.get('planned_meals', 0)} "
                    f"explicit_logged={learning_ready.get('explicit_logged_meals', 0)}{_proxy_tag} "
                    f"window=days {learning_ready.get('previous_chunk_start_day')}–{learning_ready.get('previous_chunk_end_day')} "
                    f"deferrals={learning_ready_deferrals}/{CHUNK_LEARNING_READY_MAX_DEFERRALS}"
                )

            # [P0-D] Si el inventory proxy aprobó el chunk pese a zero-log, marcamos para que
            # learning_metrics y _last_chunk_learning lo registren (telemetría + lección).
            if learning_ready.get("inventory_proxy_used"):
                form_data["_inventory_activity_proxy_used"] = True
                form_data["_inventory_activity_mutations"] = int(learning_ready.get("inventory_mutations") or 0)
            # [P0-1] El gate ahora distingue tres casos al fallar:
            #   (a) zero-log: el usuario no logueó NADA del chunk previo → no hay aprendizaje real.
            #   (b) sparse-log o ratio bajo: hay alguna señal pero por debajo del umbral.
            #   (c) prior_chunk_not_elapsed: el chunk previo aún no terminó en el calendario.
            # flexible_mode (heredado de pausa por nevera vacía o de un opt-in del usuario) bypasea el gate.
            _learning_flexible_mode = bool(snap.get("_pantry_flexible_mode") or snap.get("_learning_flexible_mode"))
            _is_zero_log = bool(learning_ready.get("zero_log_proxy"))
            _is_sparse_log = bool(learning_ready.get("sparse_logging_proxy"))
            _signal_too_weak = _is_zero_log or _is_sparse_log

            if not learning_ready.get("ready", True) and learning_ready_deferrals < CHUNK_LEARNING_READY_MAX_DEFERRALS and not _learning_flexible_mode:
                deferred_snapshot = copy.deepcopy(snap)
                deferred_snapshot["_learning_ready_deferrals"] = learning_ready_deferrals + 1
                deferred_snapshot["_last_learning_ready_ratio"] = learning_ready_ratio
                deferred_snapshot["_last_learning_zero_log"] = _is_zero_log
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        execute_after = NOW() + make_interval(hours => %s),
                        pipeline_snapshot = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        CHUNK_LEARNING_READY_DELAY_HOURS,
                        json.dumps(deferred_snapshot, ensure_ascii=False),
                        task_id,
                    )
                )
                logger.warning(
                    f"[CHUNK/LEARNING-READY] chunk {week_number} pospuesto {CHUNK_LEARNING_READY_DELAY_HOURS}h "
                    f"por baja adherencia real ({(learning_ready_ratio or 0):.0%} < {CHUNK_LEARNING_READY_MIN_RATIO:.0%}, "
                    f"zero_log={_is_zero_log}, sparse={_is_sparse_log}). "
                    f"Re-encolado {learning_ready_deferrals + 1}/{CHUNK_LEARNING_READY_MAX_DEFERRALS}."
                )
                # [P0-1] Push solo en el PRIMER deferral para no spamear. Mensaje cambia según motivo.
                if learning_ready_deferrals == 0:
                    if _is_zero_log:
                        _push_title = "Loguea tus comidas para tu próximo bloque"
                        _push_body = (
                            "No hemos visto qué comiste de tu plan actual. "
                            "Loguea tus comidas en el diario para que el siguiente bloque aprenda de ti."
                        )
                    elif _is_sparse_log:
                        _explicit = learning_ready.get('explicit_logged_meals', 0)
                        _planned = learning_ready.get('planned_meals', 0)
                        _push_title = "Loguea más comidas para que el plan aprenda"
                        _push_body = (
                            f"Solo registramos {_explicit} de {_planned} comidas — necesitamos al menos un 25 % para ajustar el siguiente bloque."
                        )
                    else:
                        _push_title = "Tu próximo bloque espera más feedback"
                        _push_body = (
                            "Loguea las comidas que hiciste estos días — el siguiente bloque del plan "
                            "se ajusta a partir de eso."
                        )
                    _dispatch_push_notification(
                        user_id=user_id,
                        title=_push_title,
                        body=_push_body,
                        url="/dashboard",
                    )
                return
            if not learning_ready.get("ready", True) and not _learning_flexible_mode:
                # [P0-1] Deferrals agotados. Si el motivo es zero-log, NO forzamos la generación
                # silenciosamente: pausamos en pending_user_action. El recovery ya existente
                # (_recover_pantry_paused_chunks) cae a flexible_mode tras CHUNK_PANTRY_EMPTY_TTL_HOURS,
                # con lo cual el chunk se acaba generando aunque el usuario nunca responda — pero
                # como camino explícito, no como bypass silencioso.
                if _is_zero_log:
                    pause_snapshot = copy.deepcopy(snap)
                    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
                    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
                    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PANTRY_EMPTY_TTL_HOURS
                    pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS
                    pause_snapshot["_pantry_pause_reason"] = "learning_zero_logs"
                    pause_snapshot["_last_learning_ready_ratio"] = learning_ready_ratio
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
                    )
                    logger.warning(
                        f"[P0-1/LEARNING-PAUSED] chunk {week_number} plan {meal_plan_id} pausado: "
                        f"zero-log tras {CHUNK_LEARNING_READY_MAX_DEFERRALS} deferrals. "
                        f"Esperando que el usuario loguee o que expire el TTL."
                    )
                    _dispatch_push_notification(
                        user_id=user_id,
                        title="Loguea tus comidas para continuar",
                        body=(
                            "Tu siguiente bloque está en pausa porque no tenemos registro de tus comidas. "
                            "Abre el diario y loguea lo que hayas comido para que aprenda de ti."
                        ),
                        url="/dashboard",
                    )
                    return

                # Sparse-log o ratio explícito bajo: generamos pero marcamos el chunk como
                # forzado para compensar con _force_variety y dejar trazado en plan_data.
                logger.warning(
                    f"[P0-1/LEARNING-FORCED] chunk {week_number} sigue bajo ({learning_ready_ratio:.0%}) "
                    f"tras {CHUNK_LEARNING_READY_MAX_DEFERRALS} deferrals. Generando con _force_variety=True."
                )
                form_data["_force_variety"] = True
                form_data["_learning_forced"] = True
                if learning_ready.get("sparse_logging_proxy"):
                    form_data["_sparse_logging_proxy"] = True
                form_data["_learning_forced_reason"] = (
                    "sparse_log" if learning_ready.get("sparse_logging_proxy") else "low_ratio"
                )
                snap["_learning_forced"] = True
            elif not learning_ready.get("ready", True) and _learning_flexible_mode:
                # flexible_mode bypassea el gate (usuario optó por seguir aunque sin aprendizaje).
                # Igualmente marcamos para que la lección y el prompt lo sepan.
                logger.warning(
                    f"[P0-1/LEARNING-FLEXIBLE] chunk {week_number} ratio={learning_ready_ratio:.0%}, "
                    f"flexible_mode activo. Generando con _force_variety=True como compensación."
                )
                form_data["_force_variety"] = True
                form_data["_learning_forced"] = True
                form_data["_learning_forced_reason"] = "flexible_mode_bypass"
                snap["_learning_forced"] = True

            form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
            if form_data.get("_pantry_paused"):
                return
            
            fresh_inventory = form_data.get("current_pantry_ingredients", [])
            fresh_inventory_source = form_data.get("_fresh_pantry_source")

            # [P0-3] Si tenemos lectura live, propagamos al snapshot del chunk actual y de los siblings vivos
            # para que su fallback futuro sea reciente (antes era la foto del momento de creación del plan).
            if fresh_inventory_source == "live":
                _persist_fresh_pantry_to_chunks(task_id, meal_plan_id, fresh_inventory)

            if _should_pause_for_empty_pantry(fresh_inventory_source, fresh_inventory, snap, form_data):
                _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, fresh_inventory)
                return

            is_degraded = snap.get("_degraded", False)
            result = {}
            
            if is_degraded:
                # [GAP 6 FIX: Probe LLM para auto-recovery]
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    import os
                    from datetime import datetime, timezone
                    
                    # Evitar flapping: revisar si hicimos downgrade hace menos de 10 minutos
                    user_res_flap = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                    hp_flap = user_res_flap.get("health_profile", {}) if user_res_flap else {}
                    last_downgrade = hp_flap.get('_last_downgrade_time')
                    can_probe = True
                    
                    if last_downgrade:
                        from constants import safe_fromisoformat
                        ld_dt = safe_fromisoformat(last_downgrade)
                        if (datetime.now(timezone.utc) - ld_dt).total_seconds() < 600:
                            can_probe = False
                            logger.info(f" [GAP 6] Downgrade reciente ({last_downgrade}), saltando probe para evitar flapping.")

                    if can_probe:
                        logger.info(f" [GAP 6] Iniciando Probe LLM para auto-recovery del chunk {week_number}...")
                        probe_llm = ChatGoogleGenerativeAI(
                            model="gemini-3.1-flash-lite-preview",
                            temperature=0.0,
                            google_api_key=os.environ.get("GEMINI_API_KEY"),
                            max_retries=0
                        )
                        import concurrent.futures as _cf
                        def _do_probe():
                            return probe_llm.invoke("ping")
                        
                        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                            ex.submit(_do_probe).result(timeout=10)
                            
                        logger.info(f" [GAP 6] LLM Probe exitoso. Sistema estabilizado, restaurando a modo AI.")
                        is_degraded = False
                        snap.pop('_degraded', None)
                        
                        # Actualizar en BD para el actual y todos los futuros chunks
                        execute_sql_write("UPDATE plan_chunk_queue SET pipeline_snapshot = pipeline_snapshot - '_degraded' WHERE meal_plan_id = %s", (meal_plan_id,))
                        
                        # Limpiar historial de downgrade
                        execute_sql_write("UPDATE user_profiles SET health_profile = health_profile - '_last_downgrade_time' WHERE id = %s", (user_id,))
                        
                except Exception as probe_e:
                    logger.warning(f" [CHUNK DEGRADED] Probe LLM falló o no pudo ejecutar ({probe_e}). Modo Smart Shuffle activo.")
            
            learning_metrics = None  # [P1-5] Inicializar para ambos paths (degraded/LLM)
            if is_degraded:
                logger.warning(f" [CHUNK DEGRADED] Generando chunk {week_number} en modo degraded (Smart Shuffle) para plan {meal_plan_id}...")
                form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
                if form_data.get("_pantry_paused"):
                    return
                prior_days = prior_plan_data.get("days", [])
                
                if not prior_days:
                    raise Exception("No prior days available for Smart Shuffle")
                
                
                new_days = []
                safe_pool = [d for d in prior_days if isinstance(d.get("meals"), list) and len(d["meals"]) > 0]
                if not safe_pool:
                    safe_pool = prior_days.copy()
                else:
                    safe_pool = safe_pool.copy()
                        
                # [GAP C FIX: Filtrar prior_days contra alergias y rechazos actuales]
                user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                health_profile = user_res.get("health_profile", {}) if user_res else {}
                current_allergies = health_profile.get("allergies", [])
                current_dislikes = health_profile.get("dislikes", [])
                current_diet = (
                    health_profile.get("dietType")
                    or ((health_profile.get("dietTypes") or [None])[0])
                    or ((snap.get("form_data", {}) or {}).get("dietType"))
                    or ""
                )
                
                if isinstance(current_allergies, str): current_allergies = [current_allergies] if current_allergies.strip() else []
                if isinstance(current_dislikes, str): current_dislikes = [current_dislikes] if current_dislikes.strip() else []

                blocklist = current_allergies + current_dislikes

                # [P1-6 FIX] Construir edge recipes con catálogos ya filtrados por alergias/dislikes/dieta.
                # [P0-C FIX] Pasar pantry para que solo elija ingredientes disponibles en la nevera.
                from constants import PLAN_CHUNK_SIZE as _PCS
                _fresh_pantry_for_edge = form_data.get("current_pantry_ingredients", [])
                if len(safe_pool) < _PCS:
                    for _ in range(_PCS - len(safe_pool)):
                        edge_day = _build_filtered_edge_recipe_day(
                            current_allergies,
                            current_dislikes,
                            current_diet,
                            pantry_items=_fresh_pantry_for_edge,
                        )
                        if not edge_day:
                            logger.warning(
                                f"[P1-6] No se pudo construir Edge Recipe seguro para {user_id} "
                                f"con restricciones actuales. Se omite expansión del pool."
                            )
                            break
                        edge_day["day_name"] = "Edge Recipe"
                        safe_pool.append(edge_day)
                
                if blocklist:
                    def _is_blocked(day):
                        for meal in day.get("meals", []):
                            txt = (meal.get("name", "") + " " + " ".join(meal.get("ingredients", []))).lower()
                            for alg in blocklist:
                                if alg.strip() and alg.strip().lower() in txt:
                                    return True
                        return False
                        
                    filtered_pool = [d for d in safe_pool if not _is_blocked(d)]
                    if filtered_pool:
                        safe_pool = filtered_pool
                    else:
                        logger.warning(f" [SMART SHUFFLE] Todos los días filtrados por restricciones {blocklist}. Pool vacío, forzando fallbacks.")
                        safe_pool = []

                # [P0-1] Filtrar por bases de ingredientes con fatiga aprendida.
                # Lee _last_chunk_learning, _recent_chunk_lessons y _lifetime_lessons_summary
                # del plan anterior para excluir días donde TODAS las comidas usan bases
                # que el sistema identificó como repetitivas. Si el pool quedaría vacío,
                # conserva el original (degradación sin bloqueo).
                _learned_bases_to_avoid: set = set()
                _p01_last = prior_plan_data.get("_last_chunk_learning") or {}
                _p01_recent = prior_plan_data.get("_recent_chunk_lessons") or []
                _p01_lifetime = prior_plan_data.get("_lifetime_lessons_summary") or {}

                for _p01_rb in (_p01_last.get("repeated_bases") or []):
                    if isinstance(_p01_rb, dict):
                        for _b in (_p01_rb.get("bases") or []):
                            if _b:
                                _learned_bases_to_avoid.add(str(_b).lower())
                    elif isinstance(_p01_rb, str) and _p01_rb:
                        _learned_bases_to_avoid.add(_p01_rb.lower())

                for _p01_lesson in _p01_recent:
                    if not isinstance(_p01_lesson, dict):
                        continue
                    for _p01_rb in (_p01_lesson.get("repeated_bases") or []):
                        if isinstance(_p01_rb, dict):
                            for _b in (_p01_rb.get("bases") or []):
                                if _b:
                                    _learned_bases_to_avoid.add(str(_b).lower())
                        elif isinstance(_p01_rb, str) and _p01_rb:
                            _learned_bases_to_avoid.add(_p01_rb.lower())

                for _b in (_p01_lifetime.get("top_repeated_bases") or []):
                    if _b:
                        _learned_bases_to_avoid.add(str(_b).lower())

                # Cap: evitar sobre-restringir el pool
                _learned_bases_to_avoid = set(list(_learned_bases_to_avoid)[:8])
                _shuffle_learning_applied = False

                if _learned_bases_to_avoid:
                    def _day_is_high_fatigue(day: dict) -> bool:
                        meals = day.get("meals", [])
                        if not meals:
                            return False
                        for meal in meals:
                            meal_text = (
                                meal.get("name", "") + " " +
                                " ".join(meal.get("ingredients", []))
                            ).lower()
                            if not any(base in meal_text for base in _learned_bases_to_avoid):
                                return False
                        return True

                    _low_fatigue_pool = [d for d in safe_pool if not _day_is_high_fatigue(d)]
                    if _low_fatigue_pool:
                        _excluded = len(safe_pool) - len(_low_fatigue_pool)
                        if _excluded > 0:
                            logger.info(
                                f"[P0-1/SHUFFLE-LEARNING] Chunk {week_number} plan {meal_plan_id}: "
                                f"excluidos {_excluded} días con bases fatigadas {_learned_bases_to_avoid}. "
                                f"Pool: {len(safe_pool)} → {len(_low_fatigue_pool)}."
                            )
                        safe_pool = _low_fatigue_pool
                        _shuffle_learning_applied = True
                    else:
                        logger.warning(
                            f"[P0-1/SHUFFLE-LEARNING] Chunk {week_number} plan {meal_plan_id}: "
                            f"todos los días tienen bases fatigadas {_learned_bases_to_avoid}. "
                            f"Conservando pool original (shuffle_with_learning=fallback)."
                        )

                pantry_filtered_pool = _filter_days_by_fresh_pantry(
                    safe_pool,
                    form_data.get("current_pantry_ingredients", []),
                )
                if pantry_filtered_pool:
                    if len(pantry_filtered_pool) < len(safe_pool):
                        logger.info(
                            f"[P1-2/PANTRY] Smart Shuffle filtró {len(safe_pool) - len(pantry_filtered_pool)} "
                            f"días por baja cobertura de inventario fresco."
                        )
                    safe_pool = pantry_filtered_pool
                elif safe_pool:
                    _current_pantry = form_data.get("current_pantry_ingredients", [])
                    from constants import CHUNK_MIN_FRESH_PANTRY_ITEMS
                    if _count_meaningful_pantry_items(_current_pantry) >= CHUNK_MIN_FRESH_PANTRY_ITEMS:
                        logger.warning(
                            f"[P0-2/PANTRY] Smart Shuffle sin cobertura para inventario fresco ({len(_current_pantry)} items). "
                            f"Pausando chunk en lugar de ignorar despensa."
                        )
                        _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, _current_pantry, reason="degraded_no_pantry_coverage")
                        return

                    logger.warning(
                        f"[P1-2/PANTRY] Ningún día del Smart Shuffle quedó mayoritariamente cubierto por "
                        f"el inventario fresco de {user_id}. Se conservan fallbacks previos."
                    )
                    
                # [GAP 4 FIX: Variedad en Smart Shuffle]
                backup_plan = execute_sql_query(
                    "SELECT health_profile->'emergency_backup_plan' as backup FROM user_profiles WHERE id = %s",
                    (user_id,), fetch_one=True
                )
                backup_days = backup_plan.get('backup', []) if backup_plan else []
                used_meal_names = set()
                
                last_chosen_hash = None
                fallback_failed = False

                for _shuffle_idx in range(days_count):
                    available_days = [d for d in safe_pool if str([m.get('name') for m in d.get('meals', [])]) != last_chosen_hash]
                    
                    if not available_days:
                        if backup_days:
                            available_days = [d for d in backup_days if str([m.get('name') for m in d.get('meals', [])]) != last_chosen_hash]

                    # [GAP 6] Ultimo recurso: si la restriccion de no-repetir-consecutivo no puede
                    # satisfacerse, inyectar una "Edge Recipe" del catalogo global antes de repetir.
                    is_emergency_repeat = False
                    is_edge_recipe = False
                    
                    if not available_days:
                        try:
                            edge_day = _build_filtered_edge_recipe_day(
                                current_allergies,
                                current_dislikes,
                                current_diet,
                                pantry_items=_fresh_pantry_for_edge,  # [P0-C FIX]
                            )
                            # Validar que no estemos repitiendo el hash por accidente
                            if edge_day and str([m.get('name') for m in edge_day.get('meals', [])]) != last_chosen_hash:
                                logger.info(f"[GAP6/CHUNK] Smart Shuffle sin variedad para {user_id}: inyectando Edge Recipe filtrado.")
                                available_days = [edge_day]
                                is_edge_recipe = True
                        except Exception as e:
                            logger.error(f"[GAP6] Error generando Edge Recipe: {e}")

                    if not available_days:
                        # Si Edge Recipe fallo, caer en el ultimo recurso de repeticion
                        repeat_pool = safe_pool or backup_days
                        if repeat_pool:
                            logger.warning(
                                f"[GAP6/CHUNK] Smart Shuffle sin variedad para {user_id}: "
                                f"permitiendo repeticion consecutiva como ultimo recurso."
                            )
                            available_days = repeat_pool
                            is_emergency_repeat = True
                        else:
                            logger.error(f"[CHUNK] Smart Shuffle fallo para {user_id}: pool vacio, no hay dias a repetir.")
                            fallback_failed = True
                            break
                    shuffled_day = copy.deepcopy(random.choice(available_days))
                    
                    # [P0-NEW1] Validate quantities against pantry for ALL day types in degraded mode
                    from constants import validate_ingredients_against_pantry, CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
                    _pantry_snap = form_data.get("current_pantry_ingredients", [])
                    _qty_validated = False
                    _shuffle_qty_attempts = 0
                    _max_shuffle_qty_attempts = 3
                    _fell_to_edge = is_edge_recipe
                    
                    while not _qty_validated and _shuffle_qty_attempts < _max_shuffle_qty_attempts:
                        _shuffled_ing = [
                            ing for m in shuffled_day.get('meals', [])
                            for ing in m.get('ingredients', []) if isinstance(ing, str) and ing.strip()
                        ]
                        _qty_check = validate_ingredients_against_pantry(
                            _shuffled_ing,
                            _pantry_snap,
                            strict_quantities=True,
                            tolerance=CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE,
                        )
                        if _qty_check is True:
                            _qty_validated = True
                            break
                        
                        _shuffle_qty_attempts += 1
                        logger.debug(f" [SHUFFLE-QTY] Candidato falló validación de cantidad: {_qty_check}")
                        
                        # Remove the failing candidate and try another
                        available_days = [d for d in available_days if d is not shuffled_day]
                        if not available_days:
                            break
                        shuffled_day = copy.deepcopy(random.choice(available_days))
                        _fell_to_edge = False # Reset flag if we pick a new day
                        is_emergency_repeat = False # Also reset this since it's a new day
                        is_edge_recipe = False

                    # Fallback to quantity-aware Edge Recipe if all 3 attempts fail
                    if not _qty_validated:
                        logger.warning(f"[P0-NEW1] Smart Shuffle falló {_shuffle_qty_attempts} intentos de cantidades, intentando Edge Recipe.")
                        try:
                            edge_day = _build_filtered_edge_recipe_day(
                                current_allergies,
                                current_dislikes,
                                current_diet,
                                pantry_items=_pantry_snap,
                            )
                            if edge_day:
                                _edge_ing = [
                                    ing for m in edge_day.get('meals', [])
                                    for ing in m.get('ingredients', []) if isinstance(ing, str) and ing.strip()
                                ]
                                _edge_qty_check = validate_ingredients_against_pantry(
                                    _edge_ing, _pantry_snap, strict_quantities=True, tolerance=CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
                                )
                                if _edge_qty_check is True:
                                    shuffled_day = edge_day
                                    is_edge_recipe = True
                                    _fell_to_edge = True
                                    _qty_validated = True
                                else:
                                    logger.warning(f"[P0-NEW1] Edge Recipe también falló cantidades: {_edge_qty_check}")
                        except Exception as e:
                            logger.error(f"[P0-NEW1] Error construyendo Edge Recipe con cantidades: {e}")

                    # If still unfeasible, pause the chunk
                    if not _qty_validated:
                        logger.warning(
                            f"[P0-NEW1/SHUFFLE-QTY] plan={meal_plan_id} chunk={week_number} "
                            f"candidate_attempts={_shuffle_qty_attempts} fallback=pause "
                            f"reason=degraded_quantities_unfeasible"
                        )
                        _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, _pantry_snap, reason="degraded_quantities_unfeasible")
                        return
                    
                    logger.info(
                        f"[P0-NEW1/SHUFFLE-QTY] plan={meal_plan_id} chunk={week_number} "
                        f"day_idx={_shuffle_idx} candidate_attempts={_shuffle_qty_attempts} "
                        f"fallback={'edge' if _fell_to_edge else 'none'}"
                    )

                    last_chosen_hash = str([m.get('name') for m in shuffled_day.get('meals', [])])
                    
                    meals = shuffled_day.get('meals', [])
                    for m_idx in range(len(meals)):
                        meal_name = meals[m_idx].get('name', '')
                        if meal_name in used_meal_names and backup_days:
                            # Swap con un meal del backup pool
                            backup_meals = [m for bd in backup_days for m in bd.get('meals', []) if m.get('name') not in used_meal_names]
                            if backup_meals:
                                meals[m_idx] = copy.deepcopy(random.choice(backup_meals))
                                shuffled_day['_mutated'] = True
                        used_meal_names.add(meals[m_idx].get('name', ''))
                    
                    shuffled_day['_is_degraded_shuffle'] = True
                    # [GAP C] Marcar el tier de calidad del día generado
                    # [P1-3 FIX] Añadir _degraded_fallback_level por día para
                    # distinguir "fallback funcionó" vs "fallback degradado".
                    if is_edge_recipe:
                        shuffled_day['_is_edge_recipe'] = True
                        shuffled_day['quality_tier'] = 'edge'
                        shuffled_day['_degraded_fallback_level'] = 'degraded'
                    elif is_emergency_repeat:
                        shuffled_day['_is_emergency_repeat'] = True
                        shuffled_day['quality_tier'] = 'emergency'
                        shuffled_day['_degraded_fallback_level'] = 'degraded'
                    else:
                        shuffled_day['quality_tier'] = 'shuffle'
                        shuffled_day['_degraded_fallback_level'] = 'functional'
                    # [GAP 3] Renumerar al dia absoluto que corresponde a este chunk.
                    # Sin esto, el dia shuffled conserva su 'day' original (ej. 1) y al mergear
                    # sobrescribiria el dia 1 del plan en vez de agregar dia 4/5/6.
                    shuffled_day['day'] = days_offset + _shuffle_idx + 1
                    # [GAP 3 FIX]: Calcular day_name exacto usando la fecha de inicio del plan
                    start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                    if start_date_str:
                        from constants import safe_fromisoformat
                        from datetime import timedelta
                        try:
                            dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                            start_dt = safe_fromisoformat(start_date_str)
                            target_date = start_dt + timedelta(days=shuffled_day['day'] - 1)
                            shuffled_day['day_name'] = dias_es[target_date.weekday()]
                        except Exception:
                            shuffled_day['day_name'] = ""
                    else:
                        shuffled_day['day_name'] = ""
                    new_days.append(shuffled_day)
                    
                if fallback_failed:
                    execute_sql_write("UPDATE plan_chunk_queue SET status = 'failed' WHERE id = %s", (task_id,))
                    return

                # [P0-A FIX] Calcular learning_metrics en Smart Shuffle para no romper la cadena
                # de aprendizaje continuo. Sin esto, _last_chunk_learning y _recent_chunk_lessons
                # nunca se actualizan cuando el chunk cae en modo degraded, rompiendo el contexto
                # de aprendizaje de todos los chunks LLM que vengan después.
                try:
                    prior_days_for_metrics = prior_plan_data.get("days", [])
                    prior_meals_for_metrics = [
                        m.get("name") for d in prior_days_for_metrics
                        for m in (d.get("meals") or [])
                        if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                    ]
                    rechazos_shuffle = get_active_rejections(user_id=user_id)
                    rejected_names_for_metrics = [r["meal_name"] for r in rechazos_shuffle] if rechazos_shuffle else []
                    learning_metrics = _calculate_learning_metrics(
                        new_days=new_days,
                        prior_meals=prior_meals_for_metrics,
                        prior_days=prior_days_for_metrics,
                        rejected_names=rejected_names_for_metrics,
                        allergy_keywords=[],
                        fatigued_ingredients=[],
                    )
                    # [P0-1] Anotar si el filtro de aprendizaje se aplicó efectivamente.
                    learning_metrics["shuffle_learning_applied"] = _shuffle_learning_applied
                    learning_metrics["shuffle_source"] = (
                        "shuffle_with_learning" if _shuffle_learning_applied else "shuffle_naive"
                    )
                    logger.info(
                        f"[P0-A/MEASURE] source={learning_metrics['shuffle_source']} "
                        f"plan={meal_plan_id} chunk={week_number} "
                        f"repeat_pct={learning_metrics.get('learning_repeat_pct', 'N/A')}% "
                        f"ingredient_base_repeat_pct={learning_metrics.get('ingredient_base_repeat_pct', 'N/A')}% "
                        f"rejection_violations={learning_metrics.get('rejection_violations', 'N/A')}"
                    )
                except Exception as _p0a_e:
                    logger.warning(f"[P0-A] Error calculando learning_metrics en Smart Shuffle: {_p0a_e}")
                    learning_metrics = None
            else:
                # [FIX CRÃTICO 3 â€" Anti-repeticion cross-semanas]:
                # NO usar snap["previous_meals"] (solo tiene semana 1). Releer TODOS los platos
                # ya generados del plan actual en DB. Esto cubre el caso donde los chunks
                # de semanas 2, 3, 4 se procesan en orden o se solapan.
                if not plan_row_prior:
                    raise Exception(f"Plan {meal_plan_id} no encontrado al leer contexto")
                prior_days = prior_plan_data.get("days", []) or []
                prior_meals = [
                    m.get("name") for d in prior_days
                    for m in (d.get("meals") or []) 
                    if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                ]
                last_chunk_learning = prior_plan_data.get("_last_chunk_learning", {})
                # [P0-4] Ventana rolling de lecciones: chunk N+1 hereda solo el último;
                # con _recent_chunk_lessons, chunk N+k hereda hasta 4 chunks anteriores.
                recent_chunk_lessons = prior_plan_data.get("_recent_chunk_lessons", [])
                if not isinstance(recent_chunk_lessons, list):
                    recent_chunk_lessons = []

                # Construir form_data para el pipeline con el offset correcto
                # form_data fue inicializado arriba
                form_data["_days_offset"] = days_offset
                form_data["_days_to_generate"] = days_count
                # Platos de TODOS los dias previos (no solo semana 1) para anti-repeticion real
                form_data["_chunk_prior_meals"] = prior_meals
                form_data["previous_meals"] = prior_meals

                # [P0-4] _force_variety si CUALQUIER lección de la ventana superó el umbral.
                _any_high_repeat = float(last_chunk_learning.get("ingredient_base_repeat_pct") or 0) > 60.0
                if not _any_high_repeat:
                    for _lesson in recent_chunk_lessons:
                        if float((_lesson or {}).get("ingredient_base_repeat_pct") or 0) > 60.0:
                            _any_high_repeat = True
                            break
                if _any_high_repeat:
                    form_data["_force_variety"] = True
                    logger.warning(
                        f"[P0-4/FORCE-VARIETY] Activando _force_variety para chunk {week_number} "
                        f"por ingredient_base_repeat_pct alta en ventana rolling "
                        f"(último={last_chunk_learning.get('ingredient_base_repeat_pct')}%)"
                    )

                # [P0-4 / P1-5 / P1-A] Propagar lecciones agregadas de la ventana rolling al LLM.
                # Agrega: repeated_bases (unión), violaciones (suma), repeat_pct (máx).
                # [P1-A] Ahora incluye también _lifetime_lessons_summary si existe.
                _all_lessons = ([last_chunk_learning] if last_chunk_learning else []) + recent_chunk_lessons
                _all_lessons_filtered = [l for l in _all_lessons if not l.get("metrics_unavailable")]
                _high_conf = [l for l in _all_lessons_filtered if not l.get("low_confidence")]
                if _high_conf:
                    _all_lessons_filtered = _high_conf
                if _all_lessons and not _all_lessons_filtered:
                    form_data["_learning_window_starved"] = True
                    form_data["_force_variety"] = True
                _all_lessons = _all_lessons_filtered
                
                lifetime_summary = prior_plan_data.get("_lifetime_lessons_summary", {})

                if _all_lessons or lifetime_summary:
                    _agg_repeated_bases: list = []
                    _agg_repeated_bases_seen: set = set()
                    _agg_repeated_meals: list = []
                    _agg_repeated_meals_seen: set = set()
                    _agg_rejected_meals: list = []
                    _agg_rejected_meals_seen: set = set()
                    _agg_allergy_hits: list = []
                    _agg_allergy_hits_seen: set = set()
                    _agg_rej_viol = 0
                    _agg_alg_viol = 0
                    _agg_max_repeat_pct = 0.0
                    _agg_max_base_repeat_pct = 0.0
                    _lesson_chunks: list = []

                    for _lesson in _all_lessons:
                        if not isinstance(_lesson, dict):
                            continue
                        _lesson_chunks.append(_lesson.get("chunk"))
                        _agg_rej_viol += int(_lesson.get("rejection_violations") or 0)
                        _agg_alg_viol += int(_lesson.get("allergy_violations") or 0)
                        _agg_max_repeat_pct = max(_agg_max_repeat_pct, float(_lesson.get("repeat_pct") or 0))
                        _agg_max_base_repeat_pct = max(_agg_max_base_repeat_pct, float(_lesson.get("ingredient_base_repeat_pct") or 0))
                        for _rb in (_lesson.get("repeated_bases") or []):
                            _key = str(_rb)
                            if _key not in _agg_repeated_bases_seen:
                                _agg_repeated_bases_seen.add(_key)
                                _agg_repeated_bases.append(_rb)
                        for _rm in (_lesson.get("repeated_meal_names") or []):
                            if _rm not in _agg_repeated_meals_seen:
                                _agg_repeated_meals_seen.add(_rm)
                                _agg_repeated_meals.append(_rm)
                        for _rj in (_lesson.get("rejected_meals_that_reappeared") or []):
                            if _rj not in _agg_rejected_meals_seen:
                                _agg_rejected_meals_seen.add(_rj)
                                _agg_rejected_meals.append(_rj)
                        for _ah in (_lesson.get("allergy_hits") or []):
                            if _ah not in _agg_allergy_hits_seen:
                                _agg_allergy_hits_seen.add(_ah)
                                _agg_allergy_hits.append(_ah)

                    # [P1-A] Inyectar datos del lifetime_summary si son más severos o adicionales
                    if lifetime_summary:
                        _agg_rej_viol = max(_agg_rej_viol, int(lifetime_summary.get("total_rejection_violations") or 0))
                        _agg_alg_viol = max(_agg_alg_viol, int(lifetime_summary.get("total_allergy_violations") or 0))
                        
                        # Agregar ingredientes rechazados históricos que no estén en la ventana actual
                        for _hist_rj in (lifetime_summary.get("top_rejection_hits") or []):
                            if _hist_rj not in _agg_rejected_meals_seen:
                                _agg_rejected_meals_seen.add(_hist_rj)
                                _agg_rejected_meals.append(_hist_rj)
                        
                        # Agregar bases históricas críticas
                        for _hist_rb in (lifetime_summary.get("top_repeated_bases") or []):
                            _key = str(_hist_rb)
                            if _key not in _agg_repeated_bases_seen:
                                _agg_repeated_bases_seen.add(_key)
                                _agg_repeated_bases.append(_hist_rb)

                    _has_actionable = (
                        _agg_repeated_bases
                        or _agg_rej_viol > 0
                        or _agg_alg_viol > 0
                        or lifetime_summary
                    )
                    if _has_actionable:
                        form_data["_chunk_lessons"] = {
                            "chunk_numbers": _lesson_chunks,
                            "ingredient_base_repeat_pct": _agg_max_base_repeat_pct,
                            "repeated_bases": _agg_repeated_bases[:12],  # Aumentado un poco para P1-A
                            "repeat_pct": _agg_max_repeat_pct,
                            "repeated_meal_names": _agg_repeated_meals[:10],
                            "rejection_violations": _agg_rej_viol,
                            "rejected_meals_that_reappeared": _agg_rejected_meals[:12],
                            "allergy_violations": _agg_alg_viol,
                            "allergy_hits": _agg_allergy_hits[:10],
                            "is_lifetime_aggregated": bool(lifetime_summary),
                            "_lifetime_window_days": lifetime_summary.get("_lifetime_window_days") if lifetime_summary else None
                        }

                        logger.info(
                            f"[P0-4] Lecciones agregadas chunks {_lesson_chunks} → chunk {week_number}: "
                            f"base_repeat_max={_agg_max_base_repeat_pct:.1f}% "
                            f"rej_viol={_agg_rej_viol} alg_viol={_agg_alg_viol} "
                            f"bases_únicas={len(_agg_repeated_bases)}"
                        )

                # [GAP 1 FIX] / [P1-2 FIX]: Propagar la ultima tecnica del chunk anterior
                # Leer desde la raiz (donde sobrevive a shifts) o fallback al _skeleton original
                last_tech = prior_plan_data.get("last_technique")
                if last_tech:
                    form_data["_last_technique"] = last_tech
                else:
                    prior_skeleton = prior_plan_data.get("_skeleton", {})
                    prior_techniques = prior_skeleton.get("_selected_techniques", [])
                    if prior_techniques:
                        form_data["_last_technique"] = prior_techniques[-1]
    
                # [APRENDIZAJE JIT]: Recalcular perfil y memoria con datos frescos
                logger.info(f" [CHUNK] Recalculando perfil y memoria Just-in-Time para usuario {user_id}...")
                session_id = form_data.get("session_id") or snap.get("form_data", {}).get("session_id")
                history = []

                likes_actualizados = get_user_likes(user_id)
                rechazos_nuevos = get_active_rejections(user_id=user_id)
                rejected_meal_names = [r["meal_name"] for r in rechazos_nuevos] if rechazos_nuevos else []

                if session_id:
                    try:
                        memory = build_memory_context(session_id)
                        history = memory.get("recent_messages", []) or []
                        memory_context = memory.get("full_context_str", "") or ""
                    except Exception as mem_e:
                        logger.warning(f"[P1-4] Error reconstruyendo memory_context desde session_id={session_id}: {mem_e}")
                        memory_context = _build_facts_memory_context(user_id)
                else:
                    memory_context = _build_facts_memory_context(user_id)

                # Re-evaluar gustos usando también la conversación reciente del plan actual
                taste_profile = analyze_preferences_agent(likes_actualizados, history, active_rejections=rejected_meal_names)

                # [GAP 1 - SEGURIDAD ALIMENTARIA]: Inyectar alergias aprendidas al snapshot
                # Esto asegura que el self_critique_node evalue estrictamente las alergias nuevas.
                from db_facts import get_user_facts_by_metadata
                alergias_facts = get_user_facts_by_metadata(user_id, 'category', 'alergia')
                if alergias_facts:
                    current_allergies = form_data.get('allergies', [])
                    if isinstance(current_allergies, str):
                        current_allergies = [current_allergies] if current_allergies.strip() else []
                    
                    for f in alergias_facts:
                        fact_text = f.get('fact', '')
                        if fact_text and fact_text not in current_allergies:
                            current_allergies.append(fact_text)
                    
                    form_data['allergies'] = current_allergies

                # [GAP 1 FIX]: Inyectar seÃ±ales avanzadas (Quality Score, EMA, etc) en el chunk worker
                from db_facts import get_consumed_meals_since
                from datetime import datetime, timezone, timedelta
                # [GAP 1 de 30 DÃAS FIX]: En vez de mirar solo 7 dias atras (lo que arruina el math para la semana 4),
                # miramos desde que inicio el plan actual para calcular la adherencia exacta sobre todos los 'prior_days'.
                # [GAP 3 FIX: _plan_start_date vs plan_start_date Inconsistencia]
                # Usar la clave correcta con underscore (que es como plans.py lo guarda)
                plan_start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                now_utc = datetime.now(timezone.utc)
                if plan_start_date_str:
                    # [P1-4] Ventana basada en distancia temporal real (days_offset) en vez de chunk_kind.
                    # chunk 2 de 30d (offset=3) → 4 días; chunk 8 de 30d (offset=21) → 14 días (capped).
                    from constants import PLAN_CHUNK_SIZE, safe_fromisoformat
                    try:
                        start_dt = safe_fromisoformat(plan_start_date_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        _consumed_window = min(max(PLAN_CHUNK_SIZE + 1, int(days_offset or 0)), 14)
                        window_start = now_utc - timedelta(days=_consumed_window)
                        since_time = max(start_dt, window_start).isoformat()
                    except Exception as e:
                        logger.warning(f"[P1-4] Error parseando plan_start_date: {e}")
                        since_time = plan_start_date_str
                else:
                    since_time = (now_utc - timedelta(days=max(7, days_offset))).isoformat()
                    
                chunk_consumed_records = get_consumed_meals_since(user_id, since_time)

                # [P0-5] Inyectar adherencia chunk-específica del previo INMEDIATO al prompt.
                # Diferencia con _meal_level_adherence (EMA por tipo, todo el historial):
                # esto lista NOMBRES de platos que el usuario consumió/saltó del chunk N-1.
                # El builder lo traduce a "refuerza variantes de X / evita repetir Y".
                if int(week_number) >= 2 and prior_days:
                    try:
                        _total_days_for_split = (
                            (snap.get("form_data", {}) or {}).get("totalDays")
                            or (snap.get("form_data", {}) or {}).get("total_days_requested")
                            or snap.get("totalDays")
                        )
                        _prev_offset, _prev_count = _resolve_previous_chunk_window(
                            meal_plan_id, int(week_number), int(days_offset or 0), _total_days_for_split
                        )
                        _breakdown = _compute_prev_chunk_meal_breakdown(
                            plan_days=prior_days,
                            prev_offset=_prev_offset,
                            prev_count=_prev_count,
                            consumed_records=chunk_consumed_records or [],
                            prev_chunk_number=int(week_number) - 1,
                        )
                        if _breakdown:
                            form_data["_prev_chunk_adherence"] = _breakdown
                            logger.info(
                                f"[P0-5/PREV-ADHERENCE] chunk={week_number} prev={_breakdown['chunk_number']} "
                                f"consumed={_breakdown['consumed_count']} skipped={_breakdown['skipped_count']} "
                                f"window=days {_breakdown['prev_start_day']}-{_breakdown['prev_end_day']}"
                            )
                    except Exception as _adh_e:
                        logger.warning(f"[P0-5/PREV-ADHERENCE] No se pudo calcular breakdown: {_adh_e}")

                user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                chunk_health_profile = user_res.get("health_profile", {}) if user_res else {}
                
                # [GAP 3 DE 30 DÃAS FIX - Descongelar form_data Snapshot]: 
                # Inyectar perfil en vivo para que los chunks asincronicos usen el objetivo (goal), peso, alergias y budget actualizados.
                # Se protegen variables internas de generacion como _days_offset.
                if chunk_health_profile:
                    # [GAP 5] Blindar ambas variantes de plan_start_date para que el merge
                    # de health_profile NO pise la fecha canonica guardada en form_data.
                    _protected_keys = {
                        '_plan_start_date', 'plan_start_date',
                        'generation_mode', 'session_id', 'user_id', 'total_days_requested',
                    }
                    for k, v in chunk_health_profile.items():
                        if not k.startswith('_') and k not in _protected_keys:
                            form_data[k] = v

                form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
                if form_data.get("_pantry_paused"):
                    return
                 
                form_data = _inject_advanced_learning_signals(user_id, form_data, chunk_health_profile, prior_days, chunk_consumed_records, days_since_last_chunk=days_offset)

                # [GAP F] Capturar triggers de aprendizaje para medir violaciones post-generación
                tuning_metrics = chunk_health_profile.get("tuning_metrics", {})
                _fatigue_data = calculate_ingredient_fatigue(user_id, tuning_metrics=tuning_metrics)
                _fatigued_ingredients = list(_fatigue_data.get('fatigued_ingredients', []) or [])
                _allergy_keywords = []
                try:
                    for f in (alergias_facts or []):
                        fact_text = f.get('fact', '').lower()
                        stopw = {"alergia", "alergico", "intolerante", "intolerancia", "condicion", "medica", "tiene", "sufre", "para"}
                        _allergy_keywords.extend([w for w in fact_text.split() if len(w) > 3 and w not in stopw])
                except Exception:
                    pass

                # Log estructurado: señales que entran al prompt del LLM
                logger.info(
                    f"[GAP F/SIGNALS] plan={meal_plan_id} chunk={week_number} "
                    f"prior_meals={len(prior_meals)} rejections={len(rejected_meal_names)} "
                    f"fatigued={len(_fatigued_ingredients)} allergy_kw={len(_allergy_keywords)}"
                )

                logger.info(f" [CHUNK] Generando chunk {week_number} para plan {meal_plan_id} "
                            f"(dias {days_offset+1}-{days_offset+days_count}, {len(prior_meals)} platos previos)...")
    
                # [P0-3] Retry loop: genera el chunk y valida que solo use ingredientes de la nevera.
                # Hasta PANTRY_MAX_RETRIES reintentos con feedback explícito al LLM.
                # Si agota reintentos, marca el chunk como failed_pantry_violation y sale.
                _PANTRY_MAX_RETRIES = 2
                _pantry_ok = False
                result = {}
                new_days = []
                for _pantry_attempt in range(_PANTRY_MAX_RETRIES + 1):
                    import concurrent.futures as _cf
                    _exec = _cf.ThreadPoolExecutor(max_workers=1)
                    _fut = _exec.submit(run_plan_pipeline, form_data, [], taste_profile, memory_context, None, None)
                    try:
                        result = _fut.result(timeout=90)
                    except _cf.TimeoutError:
                        raise Exception("Chunk pipeline timed out after 90s")
                    finally:
                        _exec.shutdown(wait=False, cancel_futures=True)

                    new_days = result.get("days", [])
                    if not new_days or "error" in result:
                        raise Exception(result.get("error", "No days generated"))

                    # [P0-3] Validación post-generación: todos los ingredientes deben estar en la nevera.
                    # strict_quantities=False: solo validamos existencia, no cantidades exactas.
                    _pantry_snapshot = form_data.get("current_pantry_ingredients", [])
                    if _pantry_snapshot:
                        _all_gen_ing = [
                            ing for d in new_days
                            for m in (d.get("meals") or [])
                            for ing in (m.get("ingredients") or [])
                            if isinstance(ing, str) and ing.strip()
                        ]
                        from constants import validate_ingredients_against_pantry as _vip
                        _val_result = _vip(_all_gen_ing, _pantry_snapshot, strict_quantities=False)
                        if _val_result is True:
                            # [P0-B] Existencia OK. Validamos cantidades según el modo configurado.
                            #   off      → aceptar tal cual.
                            #   advisory → loguear violaciones, anotarlas en form_data y aceptar.
                            #   hybrid   → reintentar con feedback; si retries se agotan, anotar y aceptar.
                            #   strict   → reintentar con feedback al LLM, fallar tras agotar retries.
                            from constants import CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
                            _qty_mode = (chunk_health_profile.get("_pantry_quantity_mode") or CHUNK_PANTRY_QUANTITY_MODE or "hybrid").lower()
                            _tolerance = CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
                            if _qty_mode == "strict":
                                _tolerance = 1.00
                            elif _qty_mode in ("off", "advisory"):
                                _tolerance = 1.10 # Legacy/Advisory (10% overflow)

                            if _qty_mode == "off":
                                # [P1-D] Race condition: User edits inventory during LLM generation
                                # Re-validate with a fresh live inventory check right before final merge.
                                if not form_data.get("_is_drift_retry"):
                                    try:
                                        fresh_live_inv = get_user_inventory_net(user_id)
                                        if fresh_live_inv is not None:
                                            old_inv = form_data.get("current_pantry_ingredients", [])
                                            drift_pct = _calculate_inventory_drift(old_inv, fresh_live_inv)
                                            if drift_pct > 0.20:
                                                logger.warning(
                                                    f"[P1-D/DRIFT] plan={meal_plan_id} chunk={week_number}: "
                                                    f"Inventario cambió {drift_pct*100:.1f}% durante generación. "
                                                    f"Activando 1 reintento rápido con datos frescos."
                                                )
                                                form_data["current_pantry_ingredients"] = fresh_live_inv
                                                form_data["_is_drift_retry"] = True
                                                form_data["_pantry_correction"] = (
                                                    "Tu generación anterior usó ingredientes que el usuario acaba de eliminar "
                                                    "o no incluyó nuevos disponibles. Por favor, regenera el plan usando este "
                                                    "inventario actualizado."
                                                )
                                                continue  # Re-generar con el nuevo inventario
                                    except Exception as drift_e:
                                        logger.warning(f"[P1-D] Error en validación de deriva: {drift_e}")

                                _pantry_ok = True
                                form_data.pop("_pantry_correction", None)
                                break

                            _qty_result = _vip(_all_gen_ing, _pantry_snapshot, strict_quantities=True, tolerance=_tolerance)
                            if _qty_result is True:
                                # [P1-D] Race condition check
                                if not form_data.get("_is_drift_retry"):
                                    try:
                                        fresh_live_inv = get_user_inventory_net(user_id)
                                        if fresh_live_inv is not None:
                                            old_inv = form_data.get("current_pantry_ingredients", [])
                                            drift_pct = _calculate_inventory_drift(old_inv, fresh_live_inv)
                                            if drift_pct > 0.20:
                                                logger.warning(
                                                    f"[P1-D/DRIFT] plan={meal_plan_id} chunk={week_number}: "
                                                    f"Inventario cambió {drift_pct*100:.1f}% durante generación. "
                                                    f"Activando 1 reintento rápido con datos frescos."
                                                )
                                                form_data["current_pantry_ingredients"] = fresh_live_inv
                                                form_data["_is_drift_retry"] = True
                                                form_data["_pantry_correction"] = "Inventario actualizado durante generación. Por favor, ajusta el plan."
                                                continue
                                    except Exception: pass

                                _pantry_ok = True
                                form_data.pop("_pantry_correction", None)
                                form_data.pop("_pantry_quantity_violations", None)
                                break

                            _qty_feedback = str(_qty_result)
                            # Solo nos interesa la línea de over-limit; las "inexistentes" ya pasaron
                            # la primera validación (matchearon vía similitud/fallback). Recortamos para logs.
                            _qty_summary = _qty_feedback.split("\n", 2)[0][:300]

                            if _qty_mode in ("strict", "hybrid") and _pantry_attempt < _PANTRY_MAX_RETRIES:
                                logger.warning(
                                    f"[P1-1/QTY-{_qty_mode.upper()}] Cantidad fuera de límite en plan={meal_plan_id} "
                                    f"chunk={week_number} intento={_pantry_attempt + 1}/{_PANTRY_MAX_RETRIES} "
                                    f"(tolerance={_tolerance}): {_qty_summary}"
                                )
                                form_data["_pantry_correction"] = _qty_feedback
                                continue  # retry con feedback de cantidades

                            if _qty_mode == "strict":
                                # [P1-1/STRICT] Reintentos agotados en modo estricto. Fallamos el chunk.
                                logger.error(
                                    f"[P1-1/QTY-STRICT] RECHAZO DEFINITIVO | plan={meal_plan_id} "
                                    f"chunk={week_number}: {_qty_summary}"
                                )
                                raise Exception(f"Violación de despensa estricta tras {_PANTRY_MAX_RETRIES} reintentos: {_qty_feedback}")

                            # Advisory / hybrid-retries-agotados: anotamos y seguimos.
                            _delta_pct_str = f"{(_tolerance - 1.0) * 100:.0f}%"
                            logger.warning(
                                f"[P1-1/QTY-{_qty_mode.upper()}] plan={meal_plan_id} chunk={week_number} "
                                f"acepta sobreuso de despensa tras agotar reintentos (modo={_qty_mode}, "
                                f"intento={_pantry_attempt + 1}, delta_pct={_delta_pct_str}): {_qty_summary}"
                            )
                            # [P1-D] Race condition check
                            if not form_data.get("_is_drift_retry"):
                                try:
                                    fresh_live_inv = get_user_inventory_net(user_id)
                                    if fresh_live_inv is not None:
                                        old_inv = form_data.get("current_pantry_ingredients", [])
                                        drift_pct = _calculate_inventory_drift(old_inv, fresh_live_inv)
                                        if drift_pct > 0.20:
                                            logger.warning(f"[P1-D/DRIFT] Deriva detectada ({drift_pct*100:.1f}%). Reintentando...")
                                            form_data["current_pantry_ingredients"] = fresh_live_inv
                                            form_data["_is_drift_retry"] = True
                                            form_data["_pantry_correction"] = "Inventario actualizado."
                                            continue
                                except Exception: pass

                            form_data["_pantry_quantity_violations"] = _qty_feedback
                            _pantry_ok = True
                            form_data.pop("_pantry_correction", None)
                            break
                        elif _pantry_attempt < _PANTRY_MAX_RETRIES:
                            _feedback_str = str(_val_result)
                            logger.warning(
                                f"[P0-3] Violación de despensa en plan={meal_plan_id} chunk={week_number} "
                                f"intento={_pantry_attempt + 1}/{_PANTRY_MAX_RETRIES}: "
                                f"{_feedback_str[:300]}"
                            )
                            form_data["_pantry_correction"] = _feedback_str
                        else:
                            logger.error(
                                f"[P0-3] Violación persistente tras {_PANTRY_MAX_RETRIES} reintentos "
                                f"en plan={meal_plan_id} chunk={week_number}. Marcando failed_pantry_violation."
                            )
                            execute_sql_write(
                                "UPDATE plan_chunk_queue SET status = 'failed', updated_at = NOW() "
                                "WHERE id = %s",
                                (task_id,)
                            )
                            return
                    else:
                        # [P1-D] Race condition check
                        if not form_data.get("_is_drift_retry"):
                            try:
                                fresh_live_inv = get_user_inventory_net(user_id)
                                if fresh_live_inv is not None:
                                    old_inv = form_data.get("current_pantry_ingredients", [])
                                    drift_pct = _calculate_inventory_drift(old_inv, fresh_live_inv)
                                    if drift_pct > 0.20:
                                        logger.warning(f"[P1-D/DRIFT] Deriva detectada ({drift_pct*100:.1f}%). Reintentando...")
                                        form_data["current_pantry_ingredients"] = fresh_live_inv
                                        form_data["_is_drift_retry"] = True
                                        form_data["_pantry_correction"] = "Inventario actualizado."
                                        continue
                            except Exception: pass

                        _pantry_ok = True
                        break

                # [GAP C] Marcar días generados por LLM con su tier de calidad
                for d in new_days:
                    if isinstance(d, dict) and 'quality_tier' not in d:
                        d['quality_tier'] = 'llm'

                # [GAP F] Métricas de aprendizaje continuo (solo en path LLM — Smart Shuffle no aplica)
                try:
                    learning_metrics = _calculate_learning_metrics(
                        new_days=new_days,
                        prior_meals=prior_meals,
                        prior_days=prior_days,
                        rejected_names=rejected_meal_names,
                        allergy_keywords=_allergy_keywords,
                        fatigued_ingredients=_fatigued_ingredients,
                    )
                    repeat_pct = learning_metrics["learning_repeat_pct"]
                    ingredient_base_repeat_pct = learning_metrics["ingredient_base_repeat_pct"]
                    rej_viol = learning_metrics["rejection_violations"]
                    alg_viol = learning_metrics["allergy_violations"]

                    # Log estructurado por chunk (observable con grep [GAP F/MEASURE])
                    logger.info(
                        f"[GAP F/MEASURE] plan={meal_plan_id} chunk={week_number} "
                        f"repeat_pct={repeat_pct}% ingredient_base_repeat_pct={ingredient_base_repeat_pct}% "
                        f"rejection_violations={rej_viol} allergy_violations={alg_viol} "
                        f"fatigued_hits={learning_metrics['fatigued_violations']}"
                    )

                    # Alertas: rechazos y alergias NUNCA deberían reaparecer
                    if rej_viol > 0:
                        logger.error(
                            f"[GAP F/VIOLATION] Chunk {week_number} plan {meal_plan_id}: "
                            f"{rej_viol} meals rechazados reaparecieron: {learning_metrics['sample_rejection_hits']}"
                        )
                    if alg_viol > 0:
                        logger.error(
                            f"[GAP F/VIOLATION] Chunk {week_number} plan {meal_plan_id}: "
                            f"{alg_viol} alergias violadas en ingredientes: {learning_metrics['sample_allergy_hits']}"
                        )
                    # Umbral de repetición: si >20%, warning (el LLM está copiando demasiado)
                    if repeat_pct > 20.0 and learning_metrics["prior_meals_count"] > 0:
                        logger.warning(
                            f"[GAP F/HIGH-REPEAT] Chunk {week_number} plan {meal_plan_id}: "
                            f"{repeat_pct}% meals repetidos de chunks previos: {learning_metrics['sample_repeats']}"
                        )
                    if ingredient_base_repeat_pct > 60.0 and learning_metrics["prior_meal_bases_count"] > 0:
                        logger.warning(
                            f"[P0-3/HIGH-BASE-REPEAT] Chunk {week_number} plan {meal_plan_id}: "
                            f"{ingredient_base_repeat_pct}% repite ingrediente base: {learning_metrics['sample_repeated_bases']}"
                        )

                    # [P0-B] Anotar violaciones de cantidad detectadas en advisory mode para que
                    # queden en plan_chunk_queue.learning_metrics y en _last_chunk_learning.
                    _qty_violations_str = form_data.get("_pantry_quantity_violations")
                    if _qty_violations_str:
                        learning_metrics["pantry_quantity_violations"] = 1
                        learning_metrics["sample_pantry_quantity_violations"] = _qty_violations_str[:500]
                    else:
                        learning_metrics.setdefault("pantry_quantity_violations", 0)

                    # [P0-D] Marcar si el chunk avanzó por inventory_proxy en lugar de logs reales.
                    # Permite filtrar en métricas: chunks con inventory_proxy_used=true son de
                    # "aprendizaje débil" (señal indirecta) y no deberían contar como adherencia plena.
                    if form_data.get("_inventory_activity_proxy_used"):
                        learning_metrics["inventory_activity_proxy_used"] = True
                        learning_metrics["inventory_activity_mutations"] = int(
                            form_data.get("_inventory_activity_mutations") or 0
                        )
                    else:
                        learning_metrics.setdefault("inventory_activity_proxy_used", False)
                        
                    # [P0-NEW2] Auditar sparse_logging_proxy
                    if form_data.get("_sparse_logging_proxy"):
                        learning_metrics["sparse_logging_proxy_used"] = True
                except Exception as lm_e:
                    logger.warning(f"[GAP F] Error calculando learning_metrics: {lm_e}")
                    learning_metrics = None

            # [FIX CRÃTICO 1 â€" Truncar si el pipeline devuelve mas dias de los pedidos]
            # Como medida de seguridad extra, si devuelve mas dias de los esperados, truncamos.
            if len(new_days) > days_count:
                new_days = new_days[:days_count]

            # [GAP 3 - VALIDACION PRE-MERGE: numeracion de dias del chunk]
            # Antes de mergear, verificar que los nuevos dias tienen EXACTAMENTE los
            # numeros absolutos esperados: {days_offset+1 .. days_offset+days_count}.
            # Si el pipeline (o Smart Shuffle) olvida setear 'day' o devuelve numeros incorrectos,
            # el merge sobrescribiria dias previos o dejaria huecos silenciosamente. Mejor fallar
            # aqui y que el outer catch re-encole el chunk con backoff.
            expected_day_nums = set(range(days_offset + 1, days_offset + days_count + 1))
            actual_day_nums = set()
            missing_day_field = 0
            for _nd in new_days:
                if not isinstance(_nd, dict):
                    continue
                _dn = _nd.get('day')
                if _dn is None:
                    missing_day_field += 1
                else:
                    actual_day_nums.add(_dn)

            if missing_day_field > 0 or actual_day_nums != expected_day_nums:
                missing = sorted(expected_day_nums - actual_day_nums)
                extra = sorted(actual_day_nums - expected_day_nums)
                
                raise Exception(
                    f"[GAP3] Chunk {week_number} numeracion invalida. "
                    f"Esperado {sorted(list(expected_day_nums))}, recibido {sorted(list(actual_day_nums))}. "
                    f"Faltan: {missing}, Extra: {extra}, sin_campo_day: {missing_day_field}"
                )

            # [P0-3 FIX] Guarda pre-merge: verificar que este chunk no fue cancelado
            # por una regeneración concurrente del usuario Y que el meal_plan_id sigue
            # siendo el plan activo. Si el usuario regeneró, services.py marca los chunks
            # como 'cancelled', pero hay una ventana donde el worker ya pasó el SELECT
            # inicial y está en medio de la generación LLM.
            try:
                _chunk_status_row = execute_sql_query(
                    "SELECT status FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                if _chunk_status_row and _chunk_status_row.get('status') == 'cancelled':
                    logger.warning(
                        f"[P0-3] Chunk {task_id} fue cancelado durante generación (plan regenerado). "
                        f"Abortando merge para evitar contaminación cruzada."
                    )
                    return  # Salir sin hacer merge ni marcar como completed

                _latest_plan_row = execute_sql_query(
                    "SELECT id FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                    (user_id,), fetch_one=True
                )
                if _latest_plan_row and _latest_plan_row.get('id') != meal_plan_id:
                    logger.warning(
                        f"[P0-3] Plan {meal_plan_id} ya no es el plan activo del usuario {user_id}. "
                        f"Plan activo: {_latest_plan_row.get('id')}. Abortando merge de chunk huérfano."
                    )
                    # Marcar como cancelled para que no se reintente
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW() WHERE id = %s",
                            (task_id,)
                        )
                    except Exception:
                        pass
                    return
            except Exception as _p03_err:
                logger.warning(f"[P0-3] Error en guarda pre-merge (continuando): {_p03_err}")

            # [GAP 2 - Race condition en merge atomico de chunks (P0)]
            # Usar FOR UPDATE real en la fila antes de mergear JSONB en Python para evitar 
            # solapamiento o perdida de dias cuando multiples chunks terminan simultaneamente.
            from db_core import connection_pool
            from psycopg.rows import dict_row

            update_result = None
            _stale_abort = False  # [P0-2] inicializado antes del bloque para ser accesible post-transacción
            _toctou_abort = False  # [P0-1] True si el abort fue por chunk cancelado (no re-encolar)
            if not connection_pool:
                raise Exception("db connection_pool is not available for atomic merge.")

            with connection_pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor(row_factory=dict_row) as cursor:
                        # 1. Bloquear la fila de forma exclusiva
                        cursor.execute("SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE", (meal_plan_id,))
                        row = cursor.fetchone()
                        if not row:
                            raise Exception(f"Meal plan {meal_plan_id} not found during atomic merge")

                        plan_data = row['plan_data'] or {}
                        existing_days = plan_data.get('days', [])
                        prior_count_raw = len(existing_days)  # Conteo bruto antes de dedup

                        # [P0-1/TOCTOU FIX] Verificar estado del chunk DENTRO de la transacción,
                        # con lock en plan_chunk_queue para eliminar la ventana TOCTOU entre
                        # el guard externo (líneas ~3606-3616) y la adquisición del FOR UPDATE.
                        cursor.execute(
                            "SELECT status, attempts FROM plan_chunk_queue WHERE id = %s FOR UPDATE",
                            (task_id,)
                        )
                        chunk_status_row = cursor.fetchone()
                        _current_status = chunk_status_row.get('status') if chunk_status_row else None
                        _current_attempts = int(chunk_status_row.get('attempts') or 0) if chunk_status_row else -1

                        if _current_status == 'cancelled':
                            logger.warning(
                                f"[P0-1/TOCTOU] Chunk {task_id} fue cancelado entre el guard externo "
                                f"y el FOR UPDATE. Abortando merge para evitar contaminación cruzada."
                            )
                            _stale_abort = True
                            _toctou_abort = True
                        elif _current_attempts != _pickup_attempts:
                            # [P0-6] El zombie rescue incrementó attempts → fuimos desplazados.
                            # El chunk ya está en cola (pending/processing) para otro worker.
                            # No re-encolar: _toctou_abort=True.
                            logger.warning(
                                f"[P0-6/ZOMBIE-DISPLACED] Chunk {task_id} desplazado por zombie rescue "
                                f"(attempts pickup={_pickup_attempts}, current={_current_attempts}). "
                                f"Abortando merge duplicado."
                            )
                            _stale_abort = True
                            _toctou_abort = True
                        elif _current_status != 'processing':
                            # [P0-6] Estado inesperado: no es processing ni cancelled.
                            # Puede ocurrir si el zombie reset a pending pero nadie lo recogió aún.
                            logger.warning(
                                f"[P0-6/UNEXPECTED-STATUS] Chunk {task_id} en estado inesperado "
                                f"durante merge: {_current_status!r}. Abortando."
                            )
                            _stale_abort = True
                            _toctou_abort = True

                        # [P0-2/CAS] Detectar si shift-plan modificó el plan durante el LLM call.
                        # Si el timestamp cambió Y estamos en modo degradado (Smart Shuffle usa
                        # prior_days del pre-read), los días del shuffle pueden ser obsoletos.
                        # Solución: abortar y re-encolar para que el próximo intento use datos frescos.
                        _locked_modified_at = plan_data.get('_plan_modified_at')
                        _locked_total_requested = int(plan_data.get('total_days_requested', 0))
                        _pre_read_total_requested = int(prior_plan_data.get('total_days_requested', 0)) if prior_plan_data else 0
                        _cas_stale = (
                            pre_read_modified_at is not None
                            and _locked_modified_at != pre_read_modified_at
                        )
                        if _cas_stale and not _stale_abort:
                            logger.warning(
                                f"[P0-2/CAS] Plan {meal_plan_id} chunk {week_number}: "
                                f"detectado cambio externo durante el LLM call "
                                f"(pre_read={pre_read_modified_at!r}, locked={_locked_modified_at!r}). "
                                f"is_degraded={is_degraded}."
                            )
                            if is_degraded:
                                # El Smart Shuffle usó prior_days obsoletos — re-encolar.
                                # La transacción no tiene escrituras así que hace rollback silencioso.
                                _stale_abort = True
                            elif _locked_total_requested != _pre_read_total_requested and _pre_read_total_requested > 0:
                                # total_days_requested cambió → regeneración completa del plan, no solo shift.
                                # Los días del LLM pueden ser válidos pero el plan ya no los espera.
                                # Abortar para que el cron re-evalúe si este chunk sigue siendo necesario.
                                logger.warning(
                                    f"[P0-1/CAS] Plan {meal_plan_id} chunk {week_number}: "
                                    f"total_days_requested cambió ({_pre_read_total_requested} → {_locked_total_requested}). "
                                    f"Posible regeneración completa. Abortando merge."
                                )
                                _stale_abort = True
                            else:
                                # En modo LLM los días son generados fresh por el modelo y
                                # total_requested no cambió (solo fue un shift). Continuar merge.
                                logger.info(
                                    f"[P0-2/CAS] Modo LLM: continuando merge (cambio fue shift, "
                                    f"total_requested={_locked_total_requested} sin cambio)."
                                )
                                _stale_abort = False
                        elif not _stale_abort:
                            _stale_abort = False

                        # 2. Mergear asegurando continuidad 1..N (P0-1 FIX)
                        # Debido a /shift-plan, el days_offset original del chunk puede estar obsoleto.
                        # Re-renumeramos todo a partir de 1 para empalmar perfectamente sin huecos ni colisiones.
                        # [GAP E] Detectar y loguear duplicados previos.
                        days_dict = {}
                        existing_day_nums_seen = set()
                        duplicates_in_existing = []

                        idx = 1
                        for d in existing_days:
                            if isinstance(d, dict):
                                original_day = d.get('day', idx)
                                if original_day in existing_day_nums_seen:
                                    duplicates_in_existing.append(original_day)
                                existing_day_nums_seen.add(original_day)
                                
                                d['day'] = idx
                                days_dict[idx] = d
                                idx += 1

                        if duplicates_in_existing:
                            logger.error(
                                f"[GAP E] Plan {meal_plan_id} tenía días duplicados en storage "
                                f"({duplicates_in_existing}). Deduplicando (última ocurrencia gana)."
                            )

                        # [P0-5 FIX] prior_count debe reflejar los días ÚNICOS post-deduplicación,
                        # no el len(existing_days) bruto. Si había duplicados, el conteo bruto
                        # es mayor que los días reales, causando falso-negativo en la validación.
                        prior_count = idx - 1  # idx se incrementó por cada día único procesado

                        sorted_new_days = sorted([d for d in new_days if isinstance(d, dict)], key=lambda x: x.get('day', 0))
                        
                        for d in sorted_new_days:
                            d['day'] = idx
                            days_dict[idx] = d
                            idx += 1

                        # Ordenar los dias para mantener coherencia en el array JSON
                        merged_days = [days_dict[k] for k in sorted(days_dict.keys())]

                        # [GAP 3 - VALIDACION POST-MERGE: continuidad 1..N]
                        # El set de days debe formar una secuencia contigua desde 1 hasta len(merged_days).
                        # Si hay hueco (ej. [1,2,3,5,6] sin dia 4) raise: la transaccion hace ROLLBACK
                        # automatico y el outer catch re-encola el chunk con backoff exponencial.
                        sorted_keys = sorted(days_dict.keys())
                        expected_keys = list(range(1, len(sorted_keys) + 1))
                        if sorted_keys != expected_keys:
                            gaps = sorted(set(expected_keys) - set(sorted_keys))
                            raise Exception(
                                f"[GAP3] Continuidad rota post-merge para plan {meal_plan_id} "
                                f"(chunk {week_number}). Keys={sorted_keys}, esperado={expected_keys}, "
                                f"huecos={gaps}. Abortando merge para preservar integridad del plan."
                            )

                        # 3. Recalcular contadores absolutos de forma segura
                        fallback_total = snap.get("totalDays", 7)
                        total_requested = int(plan_data.get('total_days_requested', fallback_total))

                        # [GAP 3] Limpieza de días huérfanos al regenerar en background
                        if len(merged_days) > total_requested:
                            logger.warning(f" [GAP 3] Recortando días huérfanos en chunk {week_number}. De {len(merged_days)} a {total_requested}")
                            merged_days = merged_days[:total_requested]

                        # [GAP E] Validación fuerte: solo aplica a planes con generación inicial multi-chunk.
                        # Rolling refills (_is_rolling_refill) reemplazan días expirados y tienen conteos distintos.
                        is_rolling_refill = snap.get('_is_rolling_refill', False)
                        if not is_rolling_refill:
                            try:
                                cursor.execute("""
                                    SELECT COALESCE(SUM(days_count), 0) AS days_from_chunks
                                    FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s AND status = 'completed'
                                """, (meal_plan_id,))
                                res_chunks = cursor.fetchone()
                                prior_days_from_chunks = int(res_chunks['days_from_chunks']) if res_chunks else 0
                                from constants import PLAN_CHUNK_SIZE as _PCS
                                expected_total = _PCS + prior_days_from_chunks + days_count
                                if expected_total > 0 and abs(len(merged_days) - expected_total) > days_count:
                                    logger.error(
                                        f"[GAP E] Plan {meal_plan_id} chunk {week_number}: len(merged_days)={len(merged_days)} "
                                        f"pero esperado ~{expected_total} (week1={_PCS} + prior_completed={prior_days_from_chunks} + this={days_count}). "
                                        f"Posible corrupción. Abortando merge para investigar."
                                    )
                                    raise Exception(
                                        f"[GAP E] Conteo inconsistente en merge: got {len(merged_days)}, expected ~{expected_total}"
                                    )
                            except Exception as _count_e:
                                if "[GAP E]" in str(_count_e):
                                    raise
                                logger.warning(f"[GAP E] Error validando conteo de chunks: {_count_e}")
                        else:
                            # [P0-5 FIX] Validación fuerte alternativa para rolling refills.
                            # prior_count ahora refleja días ÚNICOS post-dedup (no el bruto).
                            if prior_count != prior_count_raw:
                                logger.warning(
                                    f"[P0-5] prior_count ajustado por dedup: raw={prior_count_raw}, "
                                    f"post-dedup={prior_count} para plan {meal_plan_id}."
                                )
                            if len(merged_days) != prior_count + days_count:
                                logger.error(
                                    f"[GAP E/P0-5] Corrupción silenciosa prevenida en rolling refill para plan {meal_plan_id}. "
                                    f"prior_count(dedup)={prior_count}, prior_count_raw={prior_count_raw}, "
                                    f"days_count={days_count}, len(merged_days)={len(merged_days)}."
                                )
                                raise Exception(
                                    f"[GAP E] Conteo inconsistente en rolling refill merge: got {len(merged_days)}, expected {prior_count + days_count}"
                                )

                        # [P0-4 FIX] Idempotencia: verificar si este chunk ya fue mergeado previamente.
                        # Si el merge se commiteó pero la shopping list falló y el chunk fue re-encolado,
                        # los días ya están en plan_data. Re-mergearlos duplicaría silenciosamente.
                        merged_chunk_ids = plan_data.get('_merged_chunk_ids', [])
                        chunk_already_merged = str(task_id) in [str(x) for x in merged_chunk_ids]

                        if chunk_already_merged:
                            # Los días ya están en el plan — solo necesitamos reintentar la shopping list
                            logger.info(
                                f"[P0-4] Chunk {task_id} ya fue mergeado previamente en plan {meal_plan_id}. "
                                f"Saltando merge, solo recalculando shopping list."
                            )
                            new_total = len(existing_days)
                            new_status = plan_data.get('generation_status', 'partial')
                            merged_days = existing_days

                            # [P0-2] Backfill defensivo: si el chunk fue mergeado en una versión pre-P0-2
                            # (o si la transacción anterior se commit-eó pero la lección no se escribió por
                            # un bug), _last_chunk_learning puede no corresponder a ESTE chunk. Mientras
                            # estamos aún dentro del FOR UPDATE, reconstruimos la lección desde
                            # plan_chunk_queue.learning_metrics si existe, o un stub si no, y reescribimos
                            # plan_data atómicamente. Esto cierra la única ventana donde un chunk podía
                            # pasar a 'completed' sin dejar lección.
                            _persisted_lesson = plan_data.get('_last_chunk_learning') or {}
                            _persisted_chunk_id = _persisted_lesson.get('chunk') if isinstance(_persisted_lesson, dict) else None
                            if _persisted_chunk_id != week_number:
                                cursor.execute(
                                    "SELECT learning_metrics FROM plan_chunk_queue WHERE id = %s",
                                    (task_id,),
                                )
                                _row_lm = cursor.fetchone()
                                _queue_lm = (_row_lm or {}).get('learning_metrics') if _row_lm else None
                                if isinstance(_queue_lm, str):
                                    try:
                                        _queue_lm = json.loads(_queue_lm)
                                    except Exception:
                                        _queue_lm = None

                                from datetime import datetime as _dt_bf, timezone as _tz_bf
                                if _queue_lm:
                                    _backfill_lesson = {
                                        'repeat_pct': _queue_lm.get('learning_repeat_pct', 0),
                                        'ingredient_base_repeat_pct': _queue_lm.get('ingredient_base_repeat_pct', 0),
                                        'rejection_violations': _queue_lm.get('rejection_violations', 0),
                                        'allergy_violations': _queue_lm.get('allergy_violations', 0),
                                        'fatigued_violations': _queue_lm.get('fatigued_violations', 0),
                                        'repeated_bases': _queue_lm.get('sample_repeated_bases', []),
                                        'repeated_meal_names': _queue_lm.get('sample_repeats', []),
                                        'rejected_meals_that_reappeared': _queue_lm.get('sample_rejection_hits', []),
                                        'allergy_hits': _queue_lm.get('sample_allergy_hits', []),
                                        'chunk': week_number,
                                        'timestamp': _dt_bf.now(_tz_bf.utc).isoformat(),
                                        'metrics_unavailable': False,
                                        'low_confidence': bool(_queue_lm.get('inventory_activity_proxy_used') or _queue_lm.get('sparse_logging_proxy_used')),
                                        'backfilled': True,
                                    }
                                else:
                                    _backfill_lesson = {
                                        'repeat_pct': 0,
                                        'ingredient_base_repeat_pct': 0,
                                        'rejection_violations': 0,
                                        'allergy_violations': 0,
                                        'fatigued_violations': 0,
                                        'repeated_bases': [],
                                        'repeated_meal_names': [],
                                        'rejected_meals_that_reappeared': [],
                                        'allergy_hits': [],
                                        'chunk': week_number,
                                        'timestamp': _dt_bf.now(_tz_bf.utc).isoformat(),
                                        'metrics_unavailable': True,
                                        'low_confidence': True,
                                        'backfilled': True,
                                    }
                                logger.warning(
                                    f"[P0-2/BACKFILL] Chunk {task_id} (week {week_number}) ya mergeado pero "
                                    f"sin lección coherente en plan_data (encontrada chunk={_persisted_chunk_id}). "
                                    f"Reconstruyendo desde plan_chunk_queue.learning_metrics "
                                    f"(disponible={_queue_lm is not None})."
                                )
                                plan_data['_last_chunk_learning'] = _backfill_lesson
                                _recent_bf = plan_data.get('_recent_chunk_lessons', [])
                                if not isinstance(_recent_bf, list):
                                    _recent_bf = []
                                # No duplicar si ya hay una entrada para este week_number
                                _recent_bf = [l for l in _recent_bf if not (isinstance(l, dict) and l.get('chunk') == week_number)]
                                _recent_bf.append(_backfill_lesson)
                                
                                _total_req_bf = int(plan_data.get('total_days_requested', 7))
                                _win_size_bf = 8 if _total_req_bf >= 15 else 4
                                plan_data['_recent_chunk_lessons'] = _recent_bf[-_win_size_bf:]


                                # Re-escribir plan_data dentro del mismo FOR UPDATE
                                cursor.execute(
                                    "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                    (json.dumps(plan_data, ensure_ascii=False), meal_plan_id),
                                )
                                # Sellar también en la cola que la lección ya está garantizada
                                cursor.execute(
                                    """
                                    UPDATE plan_chunk_queue
                                    SET learning_persisted_at = COALESCE(learning_persisted_at, NOW())
                                    WHERE id = %s
                                    """,
                                    (task_id,),
                                )

                            update_result = [{
                                "new_total": new_total,
                                "new_status": new_status,
                                "full_plan_data": plan_data,
                                "_skip_merge_write": True,  # señal para no re-escribir plan_data
                            }]
                        else:
                            # Merge normal: primera vez que este chunk se integra

                            plan_data['days'] = merged_days
                            new_total = len(merged_days)
                            plan_data['total_days_generated'] = new_total

                            # [P1-2 FIX] / [P1-E FIX] Guardar última técnica del chunk generado para uso futuro.
                            # Fallback robusto: si _skeleton no tiene técnicas (modo degradado/emergency),
                            # intentar extraerla de los días generados o del form_data del pipeline.
                            chunk_skeleton = result.get("_skeleton", {})
                            chunk_techniques = chunk_skeleton.get("_selected_techniques", [])
                            if chunk_techniques:
                                plan_data['last_technique'] = chunk_techniques[-1]
                            else:
                                # [P1-E] Fallback 1: Derivar técnica dominante de los días finales (Smart Shuffle friendly)
                                _dominant_tech = _get_dominant_technique(new_days)
                                if _dominant_tech:
                                    plan_data['last_technique'] = _dominant_tech
                                    logger.info(f"[P1-E] last_technique (dominante) extraída de días finales: {_dominant_tech}")
                                else:
                                    # Fallback 2: buscar en el último día individualmente (legacy check)
                                    _fallback_tech = None
                                    for _nd in reversed(new_days):
                                        if isinstance(_nd, dict):
                                            _t = _nd.get('_technique') or _nd.get('technique')
                                            if _t:
                                                _fallback_tech = _t
                                                break
                                    if _fallback_tech:
                                        plan_data['last_technique'] = _fallback_tech
                                        logger.info(f"[P1-2] last_technique extraída de último día: {_fallback_tech}")
                                    elif form_data.get("_last_technique"):
                                        # Fallback 3: mantener la técnica del chunk anterior si este no produjo ninguna
                                        plan_data['last_technique'] = form_data["_last_technique"]
                                        logger.info(f"[P1-2] last_technique preservada del chunk anterior: {form_data['_last_technique']}")

                            # Rolling refills siempre marcan 'complete': la ventana de 3 días está llena.
                            # Planes normales usan total_requested para determinar si faltan más chunks.
                            if is_rolling_refill:
                                new_status = "complete"
                            else:
                                new_status = "complete" if new_total >= total_requested else "partial"
                            plan_data['generation_status'] = new_status

                            # [P0-4] Estampar idempotency marker ANTES del commit
                            if '_merged_chunk_ids' not in plan_data:
                                plan_data['_merged_chunk_ids'] = []
                            plan_data['_merged_chunk_ids'].append(str(task_id))

                            # [P1-5 / P0-4 / P0-2] Persistir métricas de aprendizaje en plan_data con muestras
                            # concretas para que el chunk siguiente pueda construir LECCIONES accionables.
                            # _last_chunk_learning: lección del chunk inmediatamente anterior (compat).
                            # _recent_chunk_lessons: ventana rolling de las últimas 4 lecciones (P0-4).
                            #
                            # [P0-2] Siempre escribimos una lección, incluso si _calculate_learning_metrics
                            # falló o el path es Smart Shuffle sin métricas. Si no hay datos, persistimos
                            # un stub con metrics_unavailable=True para que la ventana rolling no quede
                            # con huecos silenciosos (chunk N+1 necesita saber que chunk N ocurrió,
                            # aunque no se pudieron medir las violaciones).
                            from datetime import datetime, timezone
                            if learning_metrics:
                                _new_lesson = {
                                    'repeat_pct': learning_metrics.get('learning_repeat_pct', 0),
                                    'ingredient_base_repeat_pct': learning_metrics.get('ingredient_base_repeat_pct', 0),
                                    'rejection_violations': learning_metrics.get('rejection_violations', 0),
                                    'allergy_violations': learning_metrics.get('allergy_violations', 0),
                                    'fatigued_violations': learning_metrics.get('fatigued_violations', 0),
                                    'repeated_bases': learning_metrics.get('sample_repeated_bases', []),
                                    'repeated_meal_names': learning_metrics.get('sample_repeats', []),
                                    'rejected_meals_that_reappeared': learning_metrics.get('sample_rejection_hits', []),
                                    'allergy_hits': learning_metrics.get('sample_allergy_hits', []),
                                    'chunk': week_number,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'metrics_unavailable': False,
                                    'low_confidence': bool(learning_metrics.get('inventory_activity_proxy_used') or learning_metrics.get('sparse_logging_proxy_used')),
                                }
                            else:
                                _prior_days_for_stub = prior_plan_data.get("days", [])
                                _prior_meals_for_stub = [
                                    m.get("name") for d in _prior_days_for_stub
                                    for m in (d.get("meals") or [])
                                    if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                                ]
                                _new_lesson = None
                                if _prior_meals_for_stub and new_days:
                                    try:
                                        _stub_rechazos = get_active_rejections(user_id=user_id)
                                        _stub_rejected_names = [r["meal_name"] for r in _stub_rechazos] if _stub_rechazos else []
                                        _stub_metrics = _calculate_learning_metrics(
                                            new_days=new_days,
                                            prior_meals=_prior_meals_for_stub,
                                            prior_days=_prior_days_for_stub,
                                            rejected_names=_stub_rejected_names,
                                            allergy_keywords=current_allergies,
                                            fatigued_ingredients=list(_learned_bases_to_avoid)
                                        )
                                        _new_lesson = {
                                            'repeat_pct': _stub_metrics.get('learning_repeat_pct', 0),
                                            'ingredient_base_repeat_pct': _stub_metrics.get('ingredient_base_repeat_pct', 0),
                                            'rejection_violations': _stub_metrics.get('rejection_violations', 0),
                                            'allergy_violations': _stub_metrics.get('allergy_violations', 0),
                                            'fatigued_violations': _stub_metrics.get('fatigued_violations', 0),
                                            'repeated_bases': _stub_metrics.get('sample_repeated_bases', []),
                                            'repeated_meal_names': _stub_metrics.get('sample_repeats', []),
                                            'rejected_meals_that_reappeared': _stub_metrics.get('sample_rejection_hits', []),
                                            'allergy_hits': _stub_metrics.get('sample_allergy_hits', []),
                                            'chunk': week_number,
                                            'timestamp': datetime.now(timezone.utc).isoformat(),
                                            'metrics_unavailable': False,
                                            'low_confidence': True,
                                        }
                                        logger.info(f"[P0-3/STUB-LESSON] Chunk {week_number} plan {meal_plan_id}: "
                                                    f"Métricas reconstruidas a partir de prior_meals.")
                                    except Exception as e:
                                        logger.error(f"Error reconstruyendo stub metrics: {e}")
                                
                                if not _new_lesson:
                                    plan_data['_chunk_learning_stub_count'] = plan_data.get('_chunk_learning_stub_count', 0) + 1
                                    if plan_data['_chunk_learning_stub_count'] >= 2:
                                        logger.error(f"[P0-3/STUB-ALERT] El plan {meal_plan_id} acumula {plan_data['_chunk_learning_stub_count']} stubs puros. Posible problema sistémico.")
                                    _new_lesson = {
                                        'repeat_pct': 0,
                                        'ingredient_base_repeat_pct': 0,
                                        'rejection_violations': 0,
                                        'allergy_violations': 0,
                                        'fatigued_violations': 0,
                                        'repeated_bases': [],
                                        'repeated_meal_names': [],
                                        'rejected_meals_that_reappeared': [],
                                        'allergy_hits': [],
                                        'chunk': week_number,
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'metrics_unavailable': True,
                                        'chunk_learning_stub_count': plan_data['_chunk_learning_stub_count'],
                                        'low_confidence': True
                                    }
                                    logger.warning(
                                        f"[P0-3/STUB-LESSON] Chunk {week_number} plan {meal_plan_id}: "
                                        f"learning_metrics no disponible y no se pudo reconstruir. Persistiendo lección stub pura para "
                                        f"mantener la cadena de aprendizaje sin huecos."
                                    )
                            plan_data['_last_chunk_learning'] = _new_lesson
                            # [P0-4] Append a la ventana rolling (P1-A: 8 para planes largos, else 4).
                            _recent = plan_data.get('_recent_chunk_lessons', [])
                            if not isinstance(_recent, list):
                                _recent = []
                            _recent.append(_new_lesson)
                            
                            _total_req = int(plan_data.get('total_days_requested', 7))
                            _win_size = 8 if _total_req >= 15 else 4
                            plan_data['_recent_chunk_lessons'] = _recent[-_win_size:]

                            # [P1-5] Mantener historial para ventana de 60 días
                            from constants import LIFETIME_LESSONS_WINDOW_DAYS
                            from datetime import timedelta
                            
                            _history = plan_data.get('_lifetime_lessons_history', [])
                            if not isinstance(_history, list):
                                _history = []
                            _history.append(_new_lesson)
                            
                            _cutoff = (datetime.now(timezone.utc) - timedelta(days=LIFETIME_LESSONS_WINDOW_DAYS)).isoformat()
                            _history = [l for l in _history if isinstance(l, dict) and (l.get('timestamp') or "") >= _cutoff]
                            plan_data['_lifetime_lessons_history'] = _history

                            # [P1-A/P1-5] Recalcular _lifetime_lessons_summary desde el historial filtrado
                            _lifetime = {
                                "total_rejection_violations": sum(int(l.get("rejection_violations") or 0) for l in _history),
                                "total_allergy_violations": sum(int(l.get("allergy_violations") or 0) for l in _history),
                                "top_rejection_hits": [],
                                "top_repeated_bases": [],
                                "_lifetime_window_days": LIFETIME_LESSONS_WINDOW_DAYS
                            }
                            
                            _rej_set = set()
                            _base_set = set()
                            for _l in _history:
                                for _rj in (_l.get("rejected_meals_that_reappeared") or []):
                                    _rej_set.add(_rj)
                                for _rb_entry in (_l.get("repeated_bases") or []):
                                    if isinstance(_rb_entry, dict):
                                        for _b in (_rb_entry.get("bases") or []):
                                            _base_set.add(_b)
                            
                            _lifetime["top_rejection_hits"] = list(_rej_set)[:20]
                            _lifetime["top_repeated_bases"] = list(_base_set)[:20]
                            plan_data['_lifetime_lessons_summary'] = _lifetime


                            # [P0-2] Persistir learning_metrics en plan_chunk_queue DENTRO de la misma
                            # transacción que el plan_data. Antes esto se escribía después (línea ~4631)
                            # con un UPDATE separado: si el server crasheaba entre commits, plan_data
                            # tenía la lección pero la cola quedaba con learning_metrics=NULL, perdiendo
                            # la traza para auditoría/backfill. Ahora ambos van juntos en el mismo commit.
                            _lm_for_queue = json.dumps(learning_metrics, ensure_ascii=False) if learning_metrics else None
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET learning_metrics = %s::jsonb,
                                    learning_persisted_at = NOW()
                                WHERE id = %s
                                """,
                                (_lm_for_queue, task_id),
                            )
                            
                            # [P0-2] Si el CAS detectó datos obsoletos (degraded mode),
                            # NO escribir en BD. La transacción termina sin cambios,
                            # liberando el lock. El re-encole ocurre post-transacción.
                            if not _stale_abort:
                                # Sellar nuevo timestamp CAS antes de escribir.
                                from datetime import datetime as _dt, timezone as _tz
                                plan_data['_plan_modified_at'] = _dt.now(_tz.utc).isoformat()

                                # 4. Guardar los cambios
                                plan_data_json = json.dumps(plan_data, ensure_ascii=False)
                                cursor.execute(
                                    "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                    (plan_data_json, meal_plan_id)
                                )

                                update_result = [{
                                    "new_total": new_total,
                                    "new_status": new_status,
                                    "full_plan_data": plan_data
                                }]
            # [P0-1/TOCTOU] Chunk cancelado durante la ventana TOCTOU: no re-encolar, ya está cancelled.
            if _toctou_abort:
                logger.warning(
                    f"[P0-1/TOCTOU] Chunk {task_id} (plan {meal_plan_id}) abortado por cancelación "
                    f"detectada dentro del FOR UPDATE. No se re-encola."
                )
                return

            # [P0-2/CAS] Si se detectaron datos obsoletos (degraded o regeneración completa), re-encolar.
            if _stale_abort:
                logger.warning(
                    f"[P0-2/CAS] Re-encolando chunk {week_number} plan {meal_plan_id}: "
                    f"datos obsoletos detectados (degraded={is_degraded}). El próximo intento leerá datos frescos."
                )
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        attempts = COALESCE(attempts, 0) + 1,
                        execute_after = NOW() + make_interval(secs => 10),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (task_id,)
                )
                return

            if not update_result:
                raise Exception(f"Plan {meal_plan_id} no encontrado en UPDATE atomico")

            new_total = update_result[0].get("new_total", 0)
            new_status = update_result[0].get("new_status", "partial")
            full_plan_data = update_result[0].get("full_plan_data", {})

            # [GAP 2 FIX]: Recalcular lista de compras CON RETRY + ROLLBACK del merge si falla
            # Antes: solo logger.warning si fallaba -> plan quedaba con dias nuevos + shopping list vieja.
            # Ahora: retry 3x con backoff. Si falla -> marca chunk con flag para reintentar solo shopping.
            shopping_list_ok = False
            last_shop_error = None
            _SHOP_MAX_RETRIES = 3

            for _shop_attempt in range(1, _SHOP_MAX_RETRIES + 1):
                try:
                    from shopping_calculator import get_shopping_list_delta
                    household = form_data.get("householdSize", 1)
                    
                    # full_plan_data ya tiene el array de 'days' fusionado y actualizado
                    aggr_7 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=1.0 * household)
                    aggr_15 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=2.0 * household)
                    aggr_30 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=4.0 * household)
                    
                    grocery_duration = form_data.get("groceryDuration", "weekly")
                    if grocery_duration == "biweekly":
                        aggr_active = aggr_15
                    elif grocery_duration == "monthly":
                        aggr_active = aggr_30
                    else:
                        aggr_active = aggr_7
                        
                    execute_sql_write("""
                        UPDATE meal_plans 
                        SET plan_data = jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    jsonb_set(plan_data, '{aggregated_shopping_list_weekly}', %s::jsonb),
                                    '{aggregated_shopping_list_biweekly}', %s::jsonb
                                ),
                                '{aggregated_shopping_list_monthly}', %s::jsonb
                            ),
                            '{aggregated_shopping_list}', %s::jsonb
                        )
                        WHERE id = %s
                    """, (
                        json.dumps(aggr_7, ensure_ascii=False),
                        json.dumps(aggr_15, ensure_ascii=False),
                        json.dumps(aggr_30, ensure_ascii=False),
                        json.dumps(aggr_active, ensure_ascii=False),
                        meal_plan_id
                    ))
                    shopping_list_ok = True
                    logger.info(f"[CHUNK/GAP2] Shopping list consolidada recalculada para {new_total} dias (intento {_shop_attempt}).")
                    break  # Exito, salir del retry loop

                except Exception as shop_e:
                    last_shop_error = shop_e
                    if _shop_attempt < _SHOP_MAX_RETRIES:
                        backoff_secs = 2 ** _shop_attempt  # 2s, 4s
                        logger.warning(f"[CHUNK/GAP2] Shopping list intento {_shop_attempt}/{_SHOP_MAX_RETRIES} fallo: {shop_e}. "
                                       f"Reintentando en {backoff_secs}s...")
                        import time as _time
                        _time.sleep(backoff_secs)
                    else:
                        logger.error(f"[CHUNK/GAP2] Shopping list fallo {_SHOP_MAX_RETRIES} veces. Ultimo error: {shop_e}")

            # [P0-4 FIX] Si la shopping list falló, re-encolar chunk para reintentar.
            # Gracias al idempotency marker (_merged_chunk_ids), el reintento saltará el merge
            # y solo recalculará la shopping list, sin duplicar días.
            if not shopping_list_ok:
                logger.error(f" [CHUNK/P0-4] Shopping list falló para plan {meal_plan_id}. "
                             f"Re-encolando chunk (merge idempotente protege contra duplicación).")
                try:
                    shopping_retry_minutes = _compute_chunk_retry_delay_minutes(1)
                    execute_sql_write("""
                        UPDATE plan_chunk_queue 
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed'
                                ELSE 'pending'
                            END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW()
                                ELSE dead_lettered_at
                            END,
                            dead_letter_reason = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s
                                ELSE dead_letter_reason
                            END,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        shopping_retry_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        "shopping_list_retry_exhausted",
                        task_id,
                    ))
                except Exception as status_err:
                    logger.error(f" [CHUNK/P0-4] Error regresando chunk a pending: {status_err}")
                
                return  # Abortamos para no marcarlo como completed ni sobreescribir status

            try:
                from shopping_calculator import _parse_quantity
                _expected_ingredients = 0
                for _rd in new_days:
                    for _rm in (_rd or {}).get('meals', []):
                        for _ri in (_rm.get('ingredients') or []):
                            if _ri and len(str(_ri).strip()) >= 3:
                                try:
                                    _rq, _ru, _rn = _parse_quantity(str(_ri))
                                    if _rn and _rq > 0:
                                        _expected_ingredients += 1
                                except Exception:
                                    pass
                reserved_items = reserve_plan_ingredients(user_id, str(task_id), new_days)
                _expected_min = max(1, int(_expected_ingredients * 0.5))
                if reserved_items >= _expected_min:
                    logger.info(
                        f"[P1-2] Reservas de inventario aplicadas para chunk {week_number} "
                        f"plan {meal_plan_id}: {reserved_items}/{_expected_ingredients} ingredientes."
                    )
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET reservation_status = 'ok' WHERE id = %s",
                            (task_id,)
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(
                        f"[P0-5/PARTIAL] Reservas parciales chunk {week_number} plan {meal_plan_id}: "
                        f"{reserved_items}/{_expected_ingredients} (min={_expected_min}). "
                        f"Marcando reservation_status='partial'."
                    )
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET reservation_status = 'partial', updated_at = NOW() WHERE id = %s",
                            (task_id,)
                        )
                    except Exception as _partial_err:
                        logger.error(f"[P0-5] Error marcando partial: {_partial_err}")
                    # Agendar reconciliación
                    _reconcile_chunk_reservations(user_id, str(task_id), new_days)
            except Exception as reserve_err:
                logger.warning(f"[P0-5] Reservas fallidas para chunk {task_id}: {reserve_err}")
                try:
                    execute_sql_write(
                        "UPDATE plan_chunk_queue SET reservation_status = 'partial', updated_at = NOW() WHERE id = %s",
                        (task_id,)
                    )
                except Exception:
                    pass
                _reconcile_chunk_reservations(user_id, str(task_id), new_days)

            # [GAP C] Determinar tier dominante del chunk (peor tier de los días generados)
            chunk_tier = 'llm'
            try:
                tier_priority = {'emergency': 4, 'edge': 3, 'shuffle': 2, 'llm': 1}
                worst = 0
                for d in new_days:
                    if isinstance(d, dict):
                        t = d.get('quality_tier', 'llm')
                        if tier_priority.get(t, 0) > worst:
                            worst = tier_priority[t]
                            chunk_tier = t
            except Exception:
                pass

            # [GAP F] Persistir learning_metrics junto con el tier y status
            lm_json = json.dumps(learning_metrics, ensure_ascii=False) if learning_metrics else None
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET status = 'completed',
                    quality_tier = %s,
                    learning_metrics = %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (chunk_tier, lm_json, task_id,)
            )

            # [GAP C] Recalcular quality_warning del plan: si >30% de chunks completados son no-LLM, marcar.
            try:
                tier_stats = execute_sql_query("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'completed') AS completed_total,
                        COUNT(*) FILTER (WHERE status = 'completed' AND quality_tier IN ('shuffle', 'edge', 'emergency')) AS degraded
                    FROM plan_chunk_queue
                    WHERE meal_plan_id = %s
                """, (meal_plan_id,), fetch_one=True)

                if tier_stats and int(tier_stats.get('completed_total') or 0) > 0:
                    completed_total = int(tier_stats['completed_total'])
                    degraded = int(tier_stats.get('degraded') or 0)
                    degraded_ratio = degraded / completed_total
                    quality_warning = degraded_ratio > 0.30

                    execute_sql_write("""
                        UPDATE meal_plans
                        SET plan_data = jsonb_set(
                            jsonb_set(plan_data, '{quality_warning}', %s::jsonb),
                            '{quality_degraded_ratio}', %s::jsonb
                        )
                        WHERE id = %s
                    """, (
                        json.dumps(quality_warning),
                        json.dumps(round(degraded_ratio, 3)),
                        meal_plan_id,
                    ))
                    if quality_warning:
                        logger.warning(f"[GAP C] Plan {meal_plan_id} marcado con quality_warning=True ({degraded}/{completed_total} chunks degradados, {degraded_ratio:.0%}).")
            except Exception as q_err:
                logger.warning(f"[GAP C] Error calculando quality_warning para plan {meal_plan_id}: {q_err}")

            logger.info(f"[CHUNK] Chunk {week_number} completado para plan {meal_plan_id} "
                        f"(+{len(new_days)} dias, total={new_total}, status={new_status}, tier={chunk_tier})")

            # [GAP G] Registrar métrica de observabilidad
            try:
                duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                # Obtener retries actuales del chunk
                _attempts_row = execute_sql_query(
                    "SELECT attempts FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                _retries = int(_attempts_row.get("attempts") or 0) if _attempts_row else 0
                _record_chunk_metric(
                    chunk_id=task_id,
                    meal_plan_id=meal_plan_id,
                    user_id=user_id,
                    week_number=week_number,
                    days_count=days_count,
                    duration_ms=duration_ms,
                    quality_tier=chunk_tier,
                    was_degraded=chunk_tier != 'llm',
                    retries=_retries,
                    lag_seconds=lag_seconds,
                    learning_metrics=learning_metrics,
                    error_message=None,
                    is_rolling_refill=is_rolling_refill,
                )
                _alert_if_degraded_rate_high()
            except Exception as _mt_e:
                logger.warning(f"[GAP G] Error en registro métrica chunk exitoso: {_mt_e}")
                        
            # [GAP D FIX: Persistir señales de aprendizaje inter-chunk]
            # Esto permite que el chunk N+1 se beneficie de la adherencia recalculada por el chunk N,
            # manteniendo el aprendizaje verdaderamente continuo dentro del mismo día.
            try:
                user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                current_health_profile = user_res.get("health_profile", {}) if user_res else {}
                
                from db_facts import get_consumed_meals_since
                from datetime import datetime, timezone, timedelta
                
                plan_start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                if plan_start_date_str:
                    since_time = plan_start_date_str
                else:
                    since_time = (datetime.now(timezone.utc) - timedelta(days=max(7, days_offset))).isoformat()
                    
                chunk_consumed_records = get_consumed_meals_since(user_id, since_time)
                
                _persist_nightly_learning_signals(
                    user_id, 
                    current_health_profile, 
                    full_plan_data.get('days', []), 
                    chunk_consumed_records
                )
                logger.info(f" [CHUNK] Señales de aprendizaje persistidas tras chunk {week_number} para plan {meal_plan_id}")
            except Exception as persist_err:
                logger.warning(f" [CHUNK] Error persistiendo señales de aprendizaje en chunk {week_number}: {persist_err}")

        except Exception as e:
            import traceback; tb_str = traceback.format_exc(); logger.error(f" [CHUNK] Error procesando chunk {week_number} para plan {meal_plan_id}: {e}\n{tb_str}")
            # [GAP G] Registrar métrica de fallo (truncar mensaje a 1KB para no explotar tabla)
            try:
                duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                _attempts_row = execute_sql_query(
                    "SELECT attempts FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                _retries = int(_attempts_row.get("attempts") or 0) if _attempts_row else 0
                err_msg = str(e)[:1000]
                _record_chunk_metric(
                    chunk_id=task_id,
                    meal_plan_id=meal_plan_id,
                    user_id=user_id,
                    week_number=week_number,
                    days_count=days_count,
                    duration_ms=duration_ms,
                    quality_tier='error',
                    was_degraded=True,
                    retries=_retries,
                    lag_seconds=lag_seconds,
                    learning_metrics=learning_metrics,
                    error_message=err_msg,
                    is_rolling_refill=is_rolling_refill,
                )
            except Exception as _mt_e:
                logger.warning(f"[GAP G] Error en registro métrica chunk fallido: {_mt_e}")
            try:
                # [GAP B] Reintento agresivo (30 min fijo) si el chunk ya está atrasado >24h o fue escalado.
                # Si no, mantener backoff exponencial original (2^n * 2 - 1 min: 2, 8, 32, 128, 512).
                # Esto evita esperar horas en chunks críticos cuando el plan se está consumiendo.
                is_critical = lag_seconds > 86400 or task.get("escalated_at") is not None
                next_attempt = _retries + 1
                retry_delay_minutes = _compute_chunk_retry_delay_minutes(next_attempt, is_critical=is_critical)
                if is_critical:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW() ELSE dead_lettered_at END,
                            dead_letter_reason = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s ELSE dead_letter_reason END,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        retry_delay_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        str(e)[:240],
                        task_id,
                    ), returning=True)
                    logger.warning(f"[GAP B] Chunk {week_number} crítico (lag={lag_seconds//3600}h): retry en {retry_delay_minutes}min.")
                else:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW() ELSE dead_lettered_at END,
                            dead_letter_reason = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s ELSE dead_letter_reason END,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        retry_delay_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        str(e)[:240],
                        task_id,
                    ), returning=True)
                
                # [GAP 4 DE 30 DÃAS FIX / GAP 2 IMPLEMENTATION]: Manejo de Zombies y Fallbacks
                if res and res[0].get('status') == 'failed':
                    snap = task.get("pipeline_snapshot", {})
                    if isinstance(snap, str):
                        snap = json.loads(snap)
                    is_degraded = snap.get("_degraded", False)
                    
                    if is_degraded:
                        logger.error(f" [CHUNK ZOMBIE FATAL] Chunk {week_number} (Degraded Mode) fallo 5 veces. Abortando plan permanentemente.")
                        
                        # 1. Quitar el status 'partial' para liberar el frontend
                        execute_sql_write("""
                            UPDATE meal_plans 
                            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"failed"') 
                            WHERE id = %s
                        """, (meal_plan_id,))
                        
                        # 2. Cancelar los chunks futuros
                        execute_sql_write("""
                            UPDATE plan_chunk_queue 
                            SET status = 'cancelled', updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status = 'pending'
                        """, (meal_plan_id,))
                        
                        # 3. Notificar al usuario del fallo
                        try:
                            import threading
                            from utils_push import send_push_notification
                            threading.Thread(
                                target=send_push_notification,
                                kwargs={
                                    "user_id": user_id,
                                    "title": "âš ï¸ Error extendiendo tu plan",
                                    "body": "Hubo un problema generando tus proximas semanas. Tus dias actuales estan intactos. Intenta generar un nuevo plan pronto.",
                                    "url": "/dashboard"
                                }
                            ).start()
                        except Exception as push_err:
                            logger.warning(f" [CHUNK ZOMBIE] Fallo el push de error: {push_err}")
                    else:
                        logger.warning(f" [CHUNK ZOMBIE] Chunk {week_number} fallo 5 veces en modo IA. Activando Degraded Mode (Smart Shuffle).")
                        
                        # 1. Marcar el plan como 'complete_partial' (valido pero faltan dias)
                        execute_sql_write("""
                            UPDATE meal_plans 
                            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"complete_partial"') 
                            WHERE id = %s
                        """, (meal_plan_id,))
                        
                        # 2. Rescatar este chunk y los futuros en degraded mode
                        execute_sql_write("""
                            UPDATE plan_chunk_queue 
                            SET status = 'pending', 
                                attempts = 0,
                                pipeline_snapshot = jsonb_set(pipeline_snapshot, '{_degraded}', 'true'::jsonb),
                                updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status IN ('pending', 'failed')
                        """, (meal_plan_id,))
                        
                        # [GAP 6] Guardar timestamp del downgrade para evitar flapping
                        from datetime import datetime, timezone
                        execute_sql_write(
                            """
                            UPDATE user_profiles 
                            SET health_profile = jsonb_set(
                                COALESCE(health_profile, '{}'::jsonb), 
                                '{_last_downgrade_time}', 
                                %s::jsonb
                            ) WHERE id = %s
                            """,
                            (f'"{datetime.now(timezone.utc).isoformat()}"', str(user_id))
                        )
            except Exception as inner_e:
                logger.error(f" [CHUNK ZOMBIE] Error critico procesando fallback: {inner_e}")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(_chunk_worker, tasks)

def trigger_incremental_learning(user_id: str):
    """Hook asíncrono para [GAP 4] que calcula la adherencia y el Quality Score 
    inmediatamente después de que el usuario loguea una comida, en lugar de 
    esperar 18+ horas hasta el nightly rotation.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        from db_core import execute_sql_query
        from db_profiles import get_user_profile
        from db_facts import get_consumed_meals_since
        from datetime import datetime, timezone, timedelta
        
        # 1. Obtener el plan activo
        plan_res = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE user_id = %s AND status = 'active' ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )
        if not plan_res:
            return
            
        plan_data = plan_res[0].get('plan_data', {})
        days = plan_data.get('days', [])
        if not days:
            return
            
        # 2. Determinar start_date.
        plan_start_date_str = plan_data.get("_plan_start_date")
        if not plan_start_date_str:
            plan_start_date_str = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            
        # 3. Obtener comidas consumidas en el marco del plan
        consumed_records = get_consumed_meals_since(user_id, plan_start_date_str)
        
        # 4. Obtener perfil de salud actual
        profile = get_user_profile(user_id)
        if not profile:
            return
        health_profile = profile.get("health_profile", {})
        
        # 5. Modificar temporalmente el alpha para no decaer bruscamente la historia intradía
        # y delegar a la función principal de aprendizaje nocturno.
        # En una arquitectura estricta pasaríamos alpha como parámetro, pero para 
        # mantener la interfaz limpia aprovechamos la robustez del EMA suavizado actual.
        _persist_nightly_learning_signals(user_id, health_profile, days, consumed_records)
        
        logger.info(f"?? [GAP 4] Aprendizaje incremental persistido con éxito para {user_id}")
    except Exception as e:
        logger.error(f"?? [GAP 4] Error procesando aprendizaje incremental para {user_id}: {e}")


# ─── P0-2: ROLLING REFILL EN BACKGROUND (usuarios que no abren la app) ───────

def _background_shift_plan_for_user(user_id: str, tz_offset: int = 0) -> bool:
    """
    Replica la lógica de api_shift_plan() sin autenticación HTTP.
    Llamada por el cron P0-2 para usuarios inactivos.
    Retorna True si se encolaron chunks nuevos.
    """
    import copy
    import json as _json_bg
    from datetime import datetime, timezone, timedelta
    from constants import PLAN_CHUNK_SIZE, split_with_absorb, safe_fromisoformat
    from db_core import connection_pool
    from psycopg.rows import dict_row

    try:
        # chunk_size_for_next_slot se importa aquí para evitar import circular en módulo top-level
        from routers.plans import chunk_size_for_next_slot

        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "SELECT id, plan_data FROM meal_plans "
                        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                        (user_id,),
                    )
                    plan_record = cursor.fetchone()
                    if not plan_record:
                        return False

                    plan_id = plan_record["id"]
                    plan_data = plan_record.get("plan_data", {})
                    days = plan_data.get("days", [])
                    if not days:
                        return False

                    total_planned_days = max(3, int(plan_data.get("total_days_requested", len(days))))

                    today = datetime.now(timezone.utc)
                    if tz_offset:
                        today -= timedelta(minutes=int(tz_offset))

                    start_date_str = plan_data.get("grocery_start_date")
                    if not start_date_str:
                        return False

                    try:
                        start_dt = safe_fromisoformat(start_date_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        else:
                            start_dt = start_dt.astimezone(timezone.utc)
                        start_dt = start_dt - timedelta(minutes=int(tz_offset))
                        days_since_creation = (today.date() - start_dt.date()).days
                    except Exception as e:
                        logger.warning(f"[BG-REFILL] Error parseando fecha plan {plan_id}: {e}")
                        return False

                    days_remaining_in_plan = max(0, total_planned_days - days_since_creation)
                    is_expired_weekly = total_planned_days == 7 and days_remaining_in_plan == 0
                    # Solo ignorar planes expirados que no son semanales (los semanales se renuevan)
                    if days_remaining_in_plan == 0 and not is_expired_weekly:
                        return False

                    # [P0-4 FIX] Antes bloqueaba TODO refill mientras un plan de 7d siguiera vivo,
                    # incluso si su encolado síncrono inicial había fallado dejando huecos.
                    # Ahora solo bloqueamos si NO existe gap huérfano (plan completo en su día actual).
                    # La detección de gap se hace tras los guards de chunks-en-vuelo más abajo.
                    disable_rolling_refill_for_active_7d = (
                        total_planned_days == 7 and days_remaining_in_plan > 0
                    )

                    # Si ya hay chunks en camino, no duplicar
                    is_partial = plan_data.get("generation_status") in ("partial", "generating_next")
                    if is_partial:
                        return False

                    cursor.execute(
                        "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                        "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                        (plan_id,),
                    )
                    if ((cursor.fetchone() or {}).get("cnt") or 0) > 0:
                        return False

                    # [P0-4 FIX] Llegamos aquí con 0 chunks vivos. Si el plan de 7d aún tiene días
                    # por delante pero len(days) < total_planned_days, hay gap huérfano real:
                    # destrabar el refill para recuperarlo.
                    if (
                        disable_rolling_refill_for_active_7d
                        and len(days) < total_planned_days
                    ):
                        logger.warning(
                            f"[P0-4] Plan de 7 días {plan_id}: gap huérfano detectado "
                            f"(visibles={len(days)}/{total_planned_days}, sin chunks vivos). "
                            f"Habilitando rolling refill de recuperación."
                        )
                        disable_rolling_refill_for_active_7d = False

                    window_size = chunk_size_for_next_slot(
                        max(0, days_since_creation), total_planned_days, PLAN_CHUNK_SIZE
                    )
                    window_needed = min(window_size, days_remaining_in_plan)

                    needs_shift = days_since_creation > 0
                    needs_fill = len(days) < window_needed

                    if not needs_shift and not needs_fill and not is_expired_weekly:
                        return False

                    shifted_data = copy.deepcopy(plan_data)
                    shifted_days = shifted_data.get("days", [])

                    if needs_shift:
                        shift_amount = min(days_since_creation, len(shifted_days))
                        shifted_days = shifted_days[shift_amount:]

                    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                    for i, day_obj in enumerate(shifted_days):
                        target_date = today + timedelta(days=i)
                        day_obj["day_name"] = dias_es[target_date.weekday()]
                        day_obj["day"] = i + 1

                    modified = needs_shift
                    needs_fill_after_shift = (
                        len(shifted_days) < window_needed
                        and days_remaining_in_plan > 0
                        and not disable_rolling_refill_for_active_7d
                    )

                    chunks_enqueued = 0
                    if needs_fill_after_shift:
                        cursor.execute(
                            "SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,)
                        )
                        profile_row = cursor.fetchone()
                        hp = (profile_row or {}).get("health_profile", {}) or {}
                        if not hp:
                            logger.warning(f"[BG-REFILL] Sin health_profile para user {user_id}.")
                        else:
                            previous_meals = [
                                m.get("name", "")
                                for d in shifted_days
                                for m in d.get("meals", [])
                                if m.get("name")
                            ]
                            days_offset = len(shifted_days)
                            total_missing = days_remaining_in_plan - days_offset
                            catchup_chunks = (
                                split_with_absorb(total_missing, PLAN_CHUNK_SIZE)
                                if total_missing > 0
                                else []
                            )

                            # [P0-1 FIX] Propagar _plan_start_date vigente (post-shift) al snapshot
                            # para que _check_chunk_learning_ready pueda calcular ventanas de adherencia.
                            # Sin esto, el gate retornaba ready=True con reason=missing_plan_start_date
                            # y el aprendizaje continuo se desactivaba para todo rolling refill.
                            new_plan_start_iso = (
                                (start_dt + timedelta(days=days_since_creation)).isoformat()
                                if needs_shift else start_date_str
                            )

                            cursor.execute(
                                "SELECT COALESCE(MAX(week_number), 1) AS max_week "
                                "FROM plan_chunk_queue WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                (plan_id,),
                            )
                            next_week = int((cursor.fetchone() or {}).get("max_week") or 1) + 1
                            current_offset = days_offset

                            for chunk_count in catchup_chunks:
                                cursor.execute(
                                    "SELECT id FROM plan_chunk_queue "
                                    "WHERE meal_plan_id = %s AND week_number = %s "
                                    "AND status IN ('pending','processing','stale','failed') LIMIT 1",
                                    (plan_id, next_week),
                                )
                                if cursor.fetchone():
                                    next_week += 1
                                    current_offset += chunk_count
                                    continue

                                snapshot = {
                                    "form_data": {
                                        **hp,
                                        "user_id": user_id,
                                        "totalDays": chunk_count,
                                        "_plan_start_date": new_plan_start_iso,
                                    },
                                    "taste_profile": "",
                                    "memory_context": "",
                                    "previous_meals": previous_meals,
                                    "totalDays": chunk_count,
                                    "_is_rolling_refill": True,
                                    "_triggered_by": "background_cron_p0_2",
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
                                    f"🤖 [BG-REFILL] Chunk encolado user={user_id} "
                                    f"week={next_week} offset={current_offset} count={chunk_count}"
                                )
                                next_week += 1
                                current_offset += chunk_count

                            if chunks_enqueued > 0:
                                shifted_data["generation_status"] = "generating_next"
                                modified = True

                    # [P0-1] Plan semanal expirado: auto-renovar con señales de aprendizaje frescas
                    elif is_expired_weekly:
                        cursor.execute(
                            "SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,)
                        )
                        profile_row = cursor.fetchone()
                        hp = (profile_row or {}).get("health_profile", {}) or {}
                        if not hp:
                            logger.warning(f"[BG-REFILL P0-1] Sin health_profile para user {user_id}.")
                        else:
                            cursor.execute(
                                "SELECT COALESCE(MAX(week_number), 0) AS max_week FROM plan_chunk_queue "
                                "WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                (plan_id,),
                            )
                            next_week = int((cursor.fetchone() or {}).get("max_week") or 0) + 1
                            previous_meals = [
                                m.get("name", "")
                                for d in shifted_days
                                for m in d.get("meals", [])
                                if m.get("name")
                            ]
                            current_offset = 0
                            # [P0-1 FIX] El plan semanal se renueva con start = today.
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
                                    "_triggered_by": "background_cron_p0_1",
                                }
                                _enqueue_plan_chunk(
                                    user_id, plan_id, next_week, current_offset,
                                    chunk_count, snapshot, chunk_kind="rolling_refill",
                                )
                                chunks_enqueued += 1
                                logger.info(
                                    f"🔄 [BG-REFILL P0-1] Chunk renovación user={user_id} "
                                    f"week={next_week} offset={current_offset} count={chunk_count}"
                                )
                                next_week += 1
                                current_offset += chunk_count
                            shifted_days = []
                            shifted_data["grocery_start_date"] = today.isoformat()
                            shifted_data["generation_status"] = "generating_next"
                            modified = True
                            logger.info(f"🔄 [BG-REFILL P0-1] Plan semanal {plan_id} renovado para user {user_id}.")

                    if modified:
                        shifted_data["days"] = shifted_days
                        new_plan_start_iso = None
                        if needs_shift and not is_expired_weekly:
                            new_start = start_dt + timedelta(days=days_since_creation)
                            new_plan_start_iso = new_start.isoformat()
                            shifted_data["grocery_start_date"] = new_plan_start_iso
                        elif is_expired_weekly:
                            new_plan_start_iso = today.isoformat()
                        shifted_data["_plan_modified_at"] = datetime.now(timezone.utc).isoformat()
                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                            (_json_bg.dumps(shifted_data, ensure_ascii=False), plan_id),
                        )

                        # [P0-5 FIX] Sincronizar _plan_start_date en los snapshots de chunks
                        # vivos (ver justificación en routers/plans.py).
                        if new_plan_start_iso:
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
                                (_json_bg.dumps(new_plan_start_iso), plan_id),
                            )

                    return chunks_enqueued > 0

    except Exception as e:
        logger.error(f"❌ [BG-REFILL] Error procesando user {user_id}: {e}")
        return False


def trigger_background_rolling_refill() -> None:
    """
    [P0-2] Cron diario: dispara shift-plan para usuarios con planes activos que llevan
    >= 3 días sin abrir la app. Garantiza que los chunks de rolling refill se generen
    a tiempo aunque el usuario no haya entrado, preservando la promesa temporal del sistema.
    """
    try:
        from db_core import connection_pool
        if not connection_pool:
            return

        # Usuarios cuyo plan más reciente no ha sido actuado sobre en los últimos 3 días
        # (sin sesión activa) pero aún tienen días por vivir en su plan.
        query = """
            SELECT DISTINCT ON (mp.user_id) mp.user_id
            FROM meal_plans mp
            WHERE mp.user_id NOT IN (
                SELECT DISTINCT user_id
                FROM agent_sessions
                WHERE user_id IS NOT NULL
                  AND user_id::text != 'guest'
                  AND created_at >= NOW() - INTERVAL '3 days'
            )
            AND mp.id = (
                SELECT id FROM meal_plans mp2
                WHERE mp2.user_id = mp.user_id
                ORDER BY mp2.created_at DESC
                LIMIT 1
            )
            ORDER BY mp.user_id, mp.created_at DESC
        """
        inactive_users = execute_sql_query(query, fetch_all=True)

        if not inactive_users:
            logger.info("✅ [BG-REFILL] No hay usuarios inactivos con planes que refrescar.")
            return

        logger.info(f"⏰ [BG-REFILL] Procesando {len(inactive_users)} usuario(s) inactivos (P0-2)...")
        refilled = 0
        for row in inactive_users:
            uid = str(row.get("user_id") or "")
            if not uid:
                continue
            if _background_shift_plan_for_user(uid):
                refilled += 1

        logger.info(
            f"✅ [BG-REFILL] {refilled}/{len(inactive_users)} usuario(s) con chunks encolados."
        )
    except Exception as e:
        logger.error(f"❌ [BG-REFILL] Error en trigger_background_rolling_refill: {e}")
