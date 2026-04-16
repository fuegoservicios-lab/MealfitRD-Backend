import logging
import traceback
from datetime import datetime, timezone, timedelta
from db_core import execute_sql_query, execute_sql_write
from db_inventory import deduct_consumed_meal_from_inventory, get_raw_user_inventory
from db import get_latest_meal_plan_with_id, get_user_likes, get_active_rejections
from db_facts import get_all_user_facts
from graph_orchestrator import run_plan_pipeline
from services import _save_plan_and_track_background
from agent import analyze_preferences_agent

logger = logging.getLogger(__name__)


def _build_facts_memory_context(user_id: str) -> str:
    """
    Construye un string de contexto de memoria a partir de los hechos (facts)
    aprendidos por la IA sobre el usuario. Esto permite que la rotación nocturna
    sepa cosas como "le gusta el pollo", "es intolerante a la lactosa", etc.
    """
    try:
        facts = get_all_user_facts(user_id)
        if not facts:
            return ""
        
        # Priorizar por categoría para que alergias/condiciones médicas aparezcan primero
        CATEGORY_ORDER = {
            "alergia": 0, "condicion_medica": 1, "rechazo": 2,
            "dieta": 3, "objetivo": 4, "preferencia": 5, "sintoma_temporal": 6
        }
        
        def sort_key(f):
            meta = f.get("metadata", {})
            cat = meta.get("category", "") if isinstance(meta, dict) else ""
            return CATEGORY_ORDER.get(cat, 7)
        
        facts_sorted = sorted(facts, key=sort_key)
        
        # Construir el contexto legible (máximo 15 facts para no saturar el prompt)
        fact_lines = []
        for f in facts_sorted[:15]:
            fact_text = f.get("fact", "")
            meta = f.get("metadata", {})
            cat = meta.get("category", "general") if isinstance(meta, dict) else "general"
            if fact_text:
                fact_lines.append(f"• [{cat.upper()}] {fact_text}")
        
        if not fact_lines:
            return ""
        
        return (
            "\n\n--- MEMORIA DEL CEREBRO IA (HECHOS APRENDIDOS SOBRE ESTE USUARIO) ---\n"
            "DEBES respetar OBLIGATORIAMENTE esta información al generar el plan:\n"
            + "\n".join(fact_lines)
            + "\n--------------------------------------------------------------------"
        )
    except Exception as e:
        logger.warning(f"⚠️ [CRON] Error building facts memory context for {user_id}: {e}")
        return ""


def _process_user(user):
    user_id = str(user['id'])
    health_profile = user.get('health_profile', {})

    logger.info(f"🚀 [CRON] Starting rotation for user {user_id}")

    try:
        # === LOCK ANTI-DUPLICADO ===
        # Previene rotaciones dobles si APScheduler y Vercel Cron disparan simultáneamente.
        # Si el usuario ya fue rotado en las últimas 20 horas, se salta.
        last_rotated = health_profile.get('last_rotated_at')
        if last_rotated:
            try:
                last_dt_str = str(last_rotated)
                if last_dt_str.endswith('Z'):
                    last_dt_str = last_dt_str[:-1] + '+00:00'
                last_dt = datetime.fromisoformat(last_dt_str)
                hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                if hours_since < 20:
                    logger.info(f"🔒 [CRON] User {user_id} skipped: Already rotated {hours_since:.1f}h ago (lock window: 20h)")
                    return
            except (ValueError, TypeError) as parse_err:
                logger.warning(f"⚠️ [CRON] Could not parse last_rotated_at for {user_id}: {parse_err}. Proceeding.")

        # 2. Get latest plan
        plan_record = get_latest_meal_plan_with_id(user_id)
        if not plan_record:
            logger.info(f"⏭️ [CRON] User {user_id} skipped: No active plan found.")
            return

        plan_data = plan_record.get('plan_data', {})
        days = plan_data.get('days', [])
        if not days:
            logger.info(f"⏭️ [CRON] User {user_id} skipped: Plan has no days.")
            return

        # MEJORA 2: Usar consumed_meals reales en lugar de dar por hecho el Día 1
        from db_facts import get_consumed_meals_since
        from datetime import timedelta

        # Obtenemos Meals reales registrados desde la última rotación (o últimas 24h si es null)
        last_rotated = health_profile.get('last_rotated_at')
        if last_rotated:
            since_time = str(last_rotated)
        else:
            since_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

        consumed_records = get_consumed_meals_since(user_id, since_time)

        previous_meal_names = []
        ingredients_to_consume = []

        if consumed_records:
            for cm in consumed_records:
                if cm.get('meal_name'):
                    previous_meal_names.append(cm['meal_name'])

                ing_list = cm.get('ingredients', [])
                if isinstance(ing_list, list):
                    for ing in ing_list:
                        if isinstance(ing, dict) and "display_string" in ing:
                            ingredients_to_consume.append(ing["display_string"])
                        elif isinstance(ing, dict) and "name" in ing:
                            ingredients_to_consume.append(ing["name"])
                        else:
                            ingredients_to_consume.append(str(ing))
        else:
            # MEJORA 4: Modo Indulgente vs Estricto (El Caso Pizza)
            inventory_mode = health_profile.get('inventoryMode', 'indulgent')
            if inventory_mode == 'strict':
                # Fallback original: asumir Día 1 a ciegas
                logger.info(f"⚠️ [CRON] User {user_id} didn't log meals. Strict mode: falling back to day 1 estimation.")
                day_1 = days[0]
                today_meals = day_1.get('meals', [])
                for m in today_meals:
                    if m.get('name'):
                        previous_meal_names.append(m['name'])
                    ingredients_list = m.get('ingredients', [])
                    for ing in ingredients_list:
                        if isinstance(ing, dict) and "display_string" in ing:
                            ingredients_to_consume.append(ing["display_string"])
                        elif isinstance(ing, dict) and "name" in ing:
                            ingredients_to_consume.append(ing["name"])
                        else:
                            ingredients_to_consume.append(str(ing))
            else:
                # MODO INDULGENTE: No descontar NADA si no hubo registros.
                logger.info(f"🌿 [CRON] User {user_id} didn't log meals. Indulgent mode: skipping blind deduction. Preserving inventory.")
                # We intentionally leave ingredients_to_consume EMPTY.
                # However we feed previous_meal_names so the AI doesn't think they started fresh entirely.
                day_1 = days[0]
                today_meals = day_1.get('meals', [])
                for m in today_meals:
                    if m.get('name'):
                        previous_meal_names.append(m['name'])

        # 3. Consumir Inventario Físico (Resta Matemática)
        if ingredients_to_consume:
            logger.info(f"🛒 [CRON] Deducting {len(ingredients_to_consume)} consumed ingredients for user {user_id}")
            deduct_consumed_meal_from_inventory(user_id, ingredients_to_consume)

        # Obtener el inventario "vivo" con cantidades remanentes tras la resta
        raw_pantry = get_raw_user_inventory(user_id)
        current_pantry_ingredients = []
        for row in raw_pantry:
            if row.get('master_ingredient_id') and row.get('quantity', 0) > 0:
                current_pantry_ingredients.append(str(row['master_ingredient_id']))

        # 4. Construir perfil de gustos REAL del usuario (FIX CRÍTICO)
        # Antes se pasaba analyze_preferences_agent([], [], []) generando platos genéricos.
        # Ahora consultamos likes, rejections y facts para personalizar la rotación.
        likes = get_user_likes(user_id)
        active_rejections = get_active_rejections(user_id=user_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, [], active_rejections=rejected_meal_names)

        logger.info(f"🧠 [CRON] Taste profile built for {user_id}: "
                    f"{len(likes)} likes, {len(rejected_meal_names)} rejections")

        # Construir contexto de memoria (facts del Cerebro IA)
        memory_context = _build_facts_memory_context(user_id)
        if memory_context:
            logger.info(f"🧠 [CRON] Memory context loaded for {user_id} ({len(memory_context)} chars)")

        # 5. Trigger Plan Pipeline to Regenerate (Shift)
        # Reconstruct the "frontend request" data using their health profile + system variables
        pipeline_data = dict(health_profile)
        pipeline_data['user_id'] = user_id
        pipeline_data['session_id'] = user_id
        pipeline_data['previous_meals'] = previous_meal_names
        pipeline_data['current_pantry_ingredients'] = current_pantry_ingredients
        pipeline_data['_is_background_rotation'] = True

        logger.info(f"🧠 [CRON] Running AI Orchestrator for user {user_id}...")
        import time
        
        max_retries = 3
        result = {}
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"🔄 [CRON] AI Generation Attempt {attempt + 1}/{max_retries} for user {user_id}...")
                    
                # We run the orchestrator synchronously here because we are ALREADY in a background cron job thread!
                result = run_plan_pipeline(
                    data=pipeline_data, 
                    history=[], 
                    taste_profile=taste_profile, 
                    memory_context=memory_context, 
                    progress_callback=None
                )
    
                # MEJORA 7: Validar que el resultado de la IA tenga estructura útil antes de aceptarlo
                if 'error' not in result:
                    ai_days = result.get('days', [])
                    if not ai_days or not isinstance(ai_days, list):
                        result['error'] = "Malformed AI plan: missing or invalid 'days' array."
                    else:
                        # Verificar que al menos un día tenga platos reales (>= 2 meals)
                        has_valid_day = False
                        for d in ai_days:
                            meals_in_day = d.get('meals', [])
                            if isinstance(meals_in_day, list) and len(meals_in_day) >= 2:
                                has_valid_day = True
                                break
                        
                        if not has_valid_day:
                            result['error'] = "Malformed AI plan: no day has >= 2 meals (Empty structure)."
                
                if 'error' not in result:
                    if attempt > 0:
                        logger.info(f"✅ [CRON] AI Generation succeeded on attempt {attempt + 1} for user {user_id}!")
                    break  # Success! Exit retry loop
                else:
                    logger.warning(f"⚠️ [CRON] Attempt {attempt + 1} failed for user {user_id}: {result['error']}")
                    if attempt < max_retries - 1:
                        sleep_time = 3 ** attempt  # 1s, 3s
                        logger.info(f"⏳ [CRON] Waiting {sleep_time}s before retrying...")
                        time.sleep(sleep_time)
                        
            except Exception as ai_err:
                logger.error(f"❌ [CRON] Critical exception on attempt {attempt + 1} for user {user_id}: {ai_err}")
                result = {'error': str(ai_err)}
                if attempt < max_retries - 1:
                    sleep_time = 3 ** attempt
                    logger.info(f"⏳ [CRON] Waiting {sleep_time}s before retrying...")
                    time.sleep(sleep_time)

        if 'error' in result:
            logger.error(f"❌ [CRON] AI Orchestrator failed for user {user_id} after {max_retries} attempts: {result['error']}")
            # MEJORA 4 y 2: Mecanismo de Fallback Inteligente si la IA falla (Timeouts, API limits, etc.)
            if len(days) > 1:
                import copy
                import random
                logger.info(f"🔄 [CRON] Applying Smart Fallback shift for user {user_id} due to AI failure.")
                # Hacemos shift: removemos el Día 1, y todos los siguientes suben un nivel
                shifted_days = list(days)[1:]

                # MEJORA 2: Smart Fallback (Día Frankenstein / Remix de sobras)
                # Mezclamos las comidas de los días actuales por posición (desayuno, comida, etc.)
                # para que el nuevo día no sea idéntico al anterior.
                new_day = {
                    "day": int(shifted_days[-1].get("day", len(shifted_days))) + 1,
                    "meals": []
                }
                
                max_meals = max(len(d.get("meals", [])) for d in days)
                last_day_meals = [m.get("name") for m in shifted_days[-1].get("meals", [])]
                
                for i in range(max_meals):
                    candidates = []
                    for d in days:
                        meals_list = d.get("meals", [])
                        if i < len(meals_list):
                            candidates.append(meals_list[i])
                    
                    # Prevenir que el mismo plato se repita dos días seguidos
                    safe_candidates = [m for m in candidates if m.get("name") not in last_day_meals]
                    if not safe_candidates:
                        safe_candidates = candidates
                        
                    if safe_candidates:
                        chosen_meal = copy.deepcopy(random.choice(safe_candidates))
                        new_day["meals"].append(chosen_meal)
                
                shifted_days.append(new_day)

                # Construir un resultado "mock" que el resto del sistema pueda guardar
                result = {
                    "days": shifted_days,
                    "_fallback_used": "smart_shuffle"
                }
            else:
                logger.error(f"❌ [CRON] User {user_id} only had 1 day in plan. Cannot fallback shift. Skipping.")
                return

        # Preservar grocery_start_date del ciclo actual para no romper tracking de supermercado
        if previous_meal_names:
            result['grocery_start_date'] = plan_data.get('grocery_start_date', plan_record.get('created_at', datetime.now(timezone.utc).isoformat()))
            # MEJORA 6: Limitar historial a los últimos 30 días para no engordar el JSON
            nuevo_historial = plan_data.get('rotation_history', []) + [{
                "date": datetime.now(timezone.utc).isoformat(),
                "meals_consumed": previous_meal_names
            }]
            result['rotation_history'] = nuevo_historial[-30:]

        # 5. Guarda y Trackea
        selected_techniques = result.pop("_selected_techniques", None)
        _save_plan_and_track_background(user_id, result, selected_techniques)

        # === STAMP LOCK: Marcar que este usuario ya fue rotado hoy ===
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_rotated_at}', to_jsonb(%s::text)) WHERE id = %s",
                (now_iso, user_id)
            )
            logger.info(f"🔒 [CRON] Rotation lock stamped for user {user_id} at {now_iso}")

            # MEJORA 1: Enviar notificación post-rotación
            from utils_push import send_push_notification
            send_push_notification(
                user_id=user_id,
                title="¡Tus platos de hoy están listos! 🥗",
                body="Renovamos tu menú basándonos en tus últimos registros y gustos. Toca aquí para verlos.",
                url="/dashboard"
            )
        except Exception as lock_err:
            logger.warning(f"⚠️ [CRON] Failed to stamp rotation lock for {user_id}: {lock_err}")

        logger.info(f"✅ [CRON] Auto-Rotation successfully completed for user {user_id}")

    except Exception as e:
        logger.error(f"❌ [CRON] Exception while rotating user {user_id}: {e}")
        logger.error(traceback.format_exc())



def enqueue_nightly_rotations():
    logger.info("🕒 [CRON] Enqueuing Nightly Auto-Rotation for Premium Users...")
    query = '''
        SELECT id FROM user_profiles 
        WHERE (plan_tier IN ('basic', 'plus', 'ultra', 'admin') OR subscription_status = 'ACTIVE')
        AND (health_profile->>'autoRotateMeals')::boolean = true
    '''
    try:
        users = execute_sql_query(query, fetch_all=True)
    except Exception as e:
        logger.error(f"❌ [CRON] Error querying auto-rotation users: {e}")
        return
        
    if not users:
        return
        
    logger.info(f"🔄 [CRON] Found {len(users)} users. Enqueuing...")
    for u in users:
        user_id = u['id']
        try:
            execute_sql_write('''
                INSERT INTO nightly_rotation_queue (user_id, status)
                SELECT %s, 'pending'
                WHERE NOT EXISTS (
                    SELECT 1 FROM nightly_rotation_queue WHERE user_id = %s AND status = 'pending'
                )
            ''', (user_id, user_id))
        except Exception as e:
            logger.warning(f"⚠️ [CRON] Failed to enqueue user {user_id}: {e}")

def run_nightly_auto_rotation():
    # Backward compat
    enqueue_nightly_rotations()

def process_rotation_queue():
    logger.info("🕒 [CRON] Checking nightly rotation queue...")
    query = '''
        SELECT id, user_id FROM nightly_rotation_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT 5;
    '''
    try:
        pending_tasks = execute_sql_query(query, fetch_all=True)
    except Exception as e:
        logger.error(f"❌ [CRON] Error querying rotation queue: {e}")
        return

    if not pending_tasks:
        return
        
    logger.info(f"🔄 [CRON] Processing {len(pending_tasks)} users from the queue.")
    
    for t in pending_tasks:
        execute_sql_write("UPDATE nightly_rotation_queue SET status = 'processing' WHERE id = %s", (t['id'],))
        
    for task in pending_tasks:
        user_id = str(task['user_id'])
        user_query = "SELECT id, health_profile FROM user_profiles WHERE id = %s"
        try:
            u_data = execute_sql_query(user_query, (user_id,), fetch_all=False)
            if u_data:
                _process_user(u_data)
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'completed' WHERE id = %s", (task['id'],))
            else:
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed' WHERE id = %s", (task['id'],))
        except Exception as e:
            logger.error(f"❌ [CRON] Worker failed for user {user_id}: {e}")
            execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed' WHERE id = %s", (task['id'],))
