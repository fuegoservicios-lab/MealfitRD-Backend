import logging
import traceback
from datetime import datetime, timezone, timedelta
import json
import copy
import copy
import random
from db_core import execute_sql_query, execute_sql_write
from db_inventory import deduct_consumed_meal_from_inventory, get_raw_user_inventory, get_user_inventory
from db import get_latest_meal_plan_with_id, get_user_likes, get_active_rejections, get_recent_plans
from db_facts import get_all_user_facts, get_consumed_meals_since
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from schemas import HealthProfileSchema

from constants import strip_accents
from graph_orchestrator import run_plan_pipeline
from services import _save_plan_and_track_background
from agent import analyze_preferences_agent

logger = logging.getLogger(__name__)





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
    """Persiste señales de aprendizaje en health_profile durante planes largos.
    
    NO inyecta en pipeline_data (no hay pipeline). Solo actualiza la DB para que
    cuando un chunk_worker genere nuevos días, tenga datos frescos.
    """
    from db_core import execute_sql_query, execute_sql_write
    import json
    from datetime import datetime, timezone, timedelta

    # Validar el health_profile con Pydantic para evitar key errors y estructuras caóticas
    try:
        validated_profile = HealthProfileSchema(**health_profile)
    except Exception as e:
        logger.warning(f" [CRON] health_profile malformado para {user_id}. Usando defaults: {e}")
        validated_profile = HealthProfileSchema()

    # 1. Adherencia EMA
    try:
        expected_meals_count = len(days[0].get('meals', [])) if days else 0
        if expected_meals_count > 0:
            current_cycle_adherence = calculate_meal_level_adherence(user_id, days, consumed_records)
            is_weekend = (datetime.now(timezone.utc) - timedelta(hours=12)).weekday() >= 5
            profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
            
            # Usar datos validados (siempre será un dict)
            historical_adherence = getattr(validated_profile, profile_key, {})
            
            alpha = 0.3
            smoothed_adherence = {}
            all_meal_types = set(list(current_cycle_adherence.keys()) + list(historical_adherence.keys()))
            
            for mt in all_meal_types:
                current_val = current_cycle_adherence.get(mt, 1.0)
                hist_val = historical_adherence.get(mt, 1.0)
                smoothed_adherence[mt] = round((alpha * current_val) + ((1 - alpha) * hist_val), 2)
                
            # Calcular adherencia promedio general del ciclo
            avg_adherence = 1.0
            if smoothed_adherence:
                avg_adherence = round(sum(smoothed_adherence.values()) / len(smoothed_adherence), 2)
            
            adherence_history = health_profile.get('adherence_history_rotations', [])
            if not isinstance(adherence_history, list):
                adherence_history = []
            
            adherence_history.append(avg_adherence)
            adherence_history = adherence_history[-5:] # Mantener los últimos 5
            
            execute_sql_write(
                f"UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{{{profile_key}}}', %s::jsonb), '{{adherence_history_rotations}}', %s::jsonb) WHERE id = %s",
                (json.dumps(smoothed_adherence), json.dumps(adherence_history), user_id)
            )
            logger.info(f" [NIGHTLY LEARN] Adherencia guardada ({profile_key}): {smoothed_adherence} | Historial: {adherence_history}")
    except Exception as e:
        logger.warning(f" [NIGHTLY LEARN] Error guardando adherencia: {e}")

    # 2. Quality Score
    try:
        quality_score = calculate_plan_quality_score(user_id, {'days': days}, consumed_records)
        
        # [GAP 8] Diferenciar fuente de rotaciones
        quality_history_rotations = getattr(validated_profile, 'quality_history_rotations', [])
        if not isinstance(quality_history_rotations, list):
            quality_history_rotations = []
            
        quality_history_rotations.append(quality_score)
        quality_history_rotations = quality_history_rotations[-5:]
        
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_rotations}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_rotations), user_id)
        )
        logger.info(f" [NIGHTLY LEARN] Quality Score guardado: {quality_score:.2f}")
    except Exception as e:
        logger.warning(f" [NIGHTLY LEARN] Error guardando Quality Score: {e}")

    # 3. Latest weight snapshot
    try:
        weight_log = execute_sql_query(
            "SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (user_id,), fetch_one=True
        )
        if weight_log:
            latest_weight = {"date": str(weight_log['date']), "weight": weight_log['weight'], "unit": weight_log.get('unit', 'lb')}
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{latest_weight_snapshot}', %s::jsonb) WHERE id = %s",
                (json.dumps(latest_weight), user_id)
            )
            logger.info(f" [NIGHTLY LEARN] Ãšltimo peso guardado: {latest_weight}")
    except Exception as e:
        logger.error(f" [NIGHTLY LEARN] Error guardando historial de peso: {e}")

    # 4. [MEJORA 5] LLM-as-Judge Offline (Semanal)
    try:
        last_retro_str = health_profile.get("last_llm_retrospective_date")
        run_retro = False
        
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
                
        if run_retro:
            from ai_helpers import generate_llm_retrospective, extract_liked_flavor_profiles
            from db import get_user_likes, get_active_rejections
            
            recent_likes = get_user_likes(user_id)
            recent_rejections = get_active_rejections(user_id=user_id, session_id=None)
            
            llm_retro = generate_llm_retrospective(
                user_id=user_id,
                plan_data={'days': days},
                consumed_records=consumed_records,
                recent_likes=recent_likes,
                recent_rejections=recent_rejections
            )
            
            liked_flavor_profiles = extract_liked_flavor_profiles(recent_likes)
            
            if llm_retro or liked_flavor_profiles:
                # If we have one but not the other, preserve the old one (to be safe) or just write what we have
                execute_sql_write(
                    "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(jsonb_set(health_profile, '{llm_retrospective}', %s::jsonb), '{last_llm_retrospective_date}', %s::jsonb), '{liked_flavor_profiles}', %s::jsonb) WHERE id = %s",
                    (json.dumps(llm_retro), json.dumps(datetime.now(timezone.utc).isoformat()), json.dumps(liked_flavor_profiles), user_id)
                )
                logger.info(f" [LLM-as-Judge] Retrospectiva guardada exitosamente en DB.")
                
    except Exception as e:
        logger.error(f" [LLM-as-Judge] Error en el flujo offline: {e}")


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
        pipeline_data["_is_background_rotation"] = True
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


def _inject_advanced_learning_signals(user_id: str, pipeline_data: dict, health_profile: dict, days: list, consumed_records: list):
    """Extrae las seÃ±ales avanzadas de aprendizaje y las inyecta en pipeline_data."""
    from db_core import execute_sql_query, execute_sql_write
    from datetime import datetime, timezone, timedelta
    # MEJORA 5: Ingredient Fatigue Detection
    fatigue_data = None
    try:
        tuning_metrics = health_profile.get("tuning_metrics", {})
        fatigue_data = calculate_ingredient_fatigue(user_id, days_back=14, tuning_metrics=tuning_metrics)
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
    try:
        if not pipeline_data.get('successful_techniques'):
            st, at, fm = calculate_meal_success_scores(user_id, days_back=14)
            pipeline_data['successful_techniques'] = list(set(st))
            pipeline_data['abandoned_techniques'] = list(set(at))
            if fm:
                pipeline_data.setdefault('frequent_meals', fm)
        if not pipeline_data.get('day_of_week_adherence'):
            pipeline_data['day_of_week_adherence'] = calculate_day_of_week_adherence(user_id, days_back=30)
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


def _process_user(user):
    user_id = str(user['id'])
    health_profile = user.get('health_profile', {})

    logger.info(f" [CRON] Starting rotation for user {user_id}")

    try:
        # === LOCK ANTI-DUPLICADO ===
        # Previene rotaciones dobles si APScheduler y Vercel Cron disparan simultaneamente.
        # Si el usuario ya fue rotado en las ultimas 20 horas, se salta.
        last_rotated = health_profile.get('last_rotated_at')
        if last_rotated:
            try:
                last_dt_str = str(last_rotated)
                if last_dt_str.endswith('Z'):
                    last_dt_str = last_dt_str[:-1] + '+00:00'
                last_dt = datetime.fromisoformat(last_dt_str)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                if hours_since < 20:
                    logger.info(f" [CRON] User {user_id} skipped: Already rotated {hours_since:.1f}h ago (lock window: 20h)")
                    return
            except (ValueError, TypeError) as parse_err:
                logger.warning(f" [CRON] Could not parse last_rotated_at for {user_id}: {parse_err}. Proceeding.")

        # 2. Get latest plan
        plan_record = get_latest_meal_plan_with_id(user_id)
        if not plan_record:
            logger.info(f" [CRON] User {user_id} skipped: No active plan found.")
            return

        plan_data = plan_record.get('plan_data', {})
        days = plan_data.get('days', [])
        if not days:
            logger.info(f" [CRON] User {user_id} skipped: Plan has no days.")
            return

        # MEJORA 2: Usar consumed_meals reales en lugar de dar por hecho el Dia 1
        from db_facts import get_consumed_meals_since
        from datetime import timedelta

        # Obtenemos Meals reales registrados desde la ultima rotacion (o ultimas 24h si es null)
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
                # Fallback original: asumir Dia 1 a ciegas
                logger.info(f" [CRON] User {user_id} didn't log meals. Strict mode: falling back to day 1 estimation.")
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
                logger.info(f" [CRON] User {user_id} didn't log meals. Indulgent mode: skipping blind deduction. Preserving inventory.")
                # We intentionally leave ingredients_to_consume EMPTY.
                # However we feed previous_meal_names so the AI doesn't think they started fresh entirely.
                day_1 = days[0]
                today_meals = day_1.get('meals', [])
                for m in today_meals:
                    if m.get('name'):
                        previous_meal_names.append(m['name'])


        # [FIX CRÃTICO P0 / GAP 2 DE 30 DÃAS - Proteccion de Planes a Largo Plazo]
        # Evaluamos generation_status DESDE plan_data (no plan_record) y validamos total_days_requested.
        generation_status = plan_data.get('generation_status', 'complete')
        total_days_requested = int(plan_data.get('total_days_requested', 3))
        
        # [GAP 1 FIX: PLAN EXPIRADO]
        # Si el plan de largo plazo completo su generacion y ya estamos en su ultimo dia (o se agoto),
        # transicionamos al usuario de vuelta al modo de rotacion continua (3 dias).
        is_expired = False
        if total_days_requested > 3 and len(days) <= 1:
            if generation_status == 'complete':
                is_expired = True
                logger.info(f" [CRON] Long-term plan ({total_days_requested} days) expired for user {user_id}. Transitioning back to normal 3-day rotation.")
            else:
                # [GAP 5 FIX: Expiracion forzada para planes partial atascados]
                try:
                    from db_core import execute_sql_query
                    pending_chunks = execute_sql_query("""
                        SELECT COUNT(*) as cnt FROM plan_chunk_queue
                        WHERE meal_plan_id = %s AND status IN ('pending', 'processing')
                    """, (plan_record['id'],), fetch_one=True)
                    has_rescue = pending_chunks and int(pending_chunks.get('cnt', 0)) > 0

                    # [GAP B] Para planes largos (15+ dias) con chunks pendientes, NO expirar:
                    # darles ventana extra (buffer ~72h desde execute_after del chunk mas viejo)
                    # antes de cancelar. Mientras esperamos, escalamos los pickups.
                    if has_rescue and total_days_requested >= 15:
                        try:
                            buf = execute_sql_query("""
                                SELECT EXTRACT(EPOCH FROM (NOW() - MIN(execute_after)))::int AS oldest_lag
                                FROM plan_chunk_queue
                                WHERE meal_plan_id = %s AND status IN ('pending', 'processing')
                            """, (plan_record['id'],), fetch_one=True)
                            oldest_lag_h = (int(buf.get('oldest_lag') or 0) / 3600.0) if buf else 0
                            if oldest_lag_h < 72:
                                logger.warning(
                                    f"[GAP B] Plan {plan_record['id']} ({total_days_requested}d) en {len(days)} dia(s) "
                                    f"con chunks pendientes (oldest_lag={oldest_lag_h:.1f}h). Escalando pickups en vez de expirar."
                                )
                                from db_core import execute_sql_write
                                execute_sql_write("""
                                    UPDATE plan_chunk_queue
                                    SET escalated_at = COALESCE(escalated_at, NOW()),
                                        execute_after = NOW(),
                                        updated_at = NOW()
                                    WHERE meal_plan_id = %s AND status IN ('pending', 'stale')
                                """, (plan_record['id'],))
                                # Salir de _process_user sin expirar — el worker recogera los chunks pronto
                                return
                        except Exception as buf_e:
                            logger.warning(f"[GAP B] Error en buffer de expiracion: {buf_e}")

                    if not has_rescue:
                        is_expired = True
                        logger.warning(f" [CRON] Plan partial/stuck ({generation_status}) para {user_id} con {len(days)} dias y 0 chunks pendientes. Forzando expiracion.")
                        try:
                            from db_core import execute_sql_write
                            execute_sql_write("UPDATE meal_plans SET plan_data = jsonb_set(plan_data, '{generation_status}', '\"complete\"') WHERE id = %s", (plan_record['id'],))
                        except Exception:
                            pass
                except Exception as eval_e:
                    logger.warning(f" [CRON] Error evaluating forced expiration: {eval_e}")
        if is_expired:
            # [GAP 5 FIX] Cancelar chunks pendientes al expirar el plan
            try:
                from db_core import execute_sql_write
                res = execute_sql_write(
                    "UPDATE plan_chunk_queue SET status = 'cancelled' WHERE meal_plan_id = %s AND status IN ('pending', 'processing') RETURNING id",
                    (plan_record['id'],),
                    returning=True
                )
                if res:
                    logger.info(f" [GAP 5] Cancelados {len(res)} chunks para el plan expirado {plan_record['id']}")
            except Exception as e:
                logger.warning(f" [GAP 5] Error cancelando chunks del plan {plan_record['id']}: {e}")

            # --- [GAP 3 FIX: Notificacion Push de Expiracion] ---
            try:
                import threading
                from utils_push import send_push_notification
                threading.Thread(
                    target=send_push_notification,
                    kwargs={
                        "user_id": user_id,
                        "title": "Tu plan a largo plazo ha terminado",
                        "body": f"Tu plan de {total_days_requested} dias llego a su fin. Te regresamos a la rotacion de 3 dias para que no te quedes sin comida. ¡Entra a la app para generar otro!",
                        "url": "/dashboard"
                    }
                ).start()
            except Exception as push_e:
                logger.warning(f" [CRON] Error dispatching expiration push notification: {push_e}")

            total_days_requested = 3
            health_profile['totalDays'] = 3
            try:
                from db_core import execute_sql_write
                execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{totalDays}', '3'::jsonb) WHERE id = %s", (user_id,))
            except Exception as e:
                logger.error(f" [CRON] Error resetting totalDays for {user_id}: {e}")
                
            # [GAP 10] Guardar resumen de adherencia cruzada para el próximo plan
            try:
                from db_core import execute_sql_write
                succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=max(14, total_days_requested))
                if freq_meals:
                    import json
                    execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{previous_plan_frequent_meals}', %s::jsonb) WHERE id = %s", (json.dumps(freq_meals), user_id))
                    logger.info(f" [GAP 10] Resumen de comidas frecuentes guardado para seed del próximo plan de {user_id}")
            except Exception as gap10_e:
                logger.warning(f" [GAP 10] Error calculando resumen cruzado al expirar: {gap10_e}")
                
            # [GAP 4 IMPLEMENTATION]: Pre-cargar el emergency backup cache por si la regeneracion IA de hoy falla
            backup_plan = health_profile.get('emergency_backup_plan', [])
            if not isinstance(backup_plan, list) or len(backup_plan) == 0:
                if days and len(days) > 0:
                    import copy, json
                    seed_day = copy.deepcopy(days[0])
                    seed_day['_is_expiration_seed'] = True
                    try:
                        from db_core import execute_sql_write
                        execute_sql_write(
                            "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
                            (json.dumps([seed_day], ensure_ascii=False), user_id)
                        )
                        logger.info(f" [CRON] Seeded emergency backup with last day of expired plan for user {user_id}")
                    except Exception as e:
                        logger.error(f" [CRON] Error seeding emergency backup for user {user_id}: {e}")

            # Al no entrar en el elif de abajo ni hacer 'return', el script continuará 
            # con la regeneración diaria de 3 días de forma transparente (Pasos 3, 4 y 5).
        
        elif generation_status == 'partial' or total_days_requested >= 3:
            logger.info(f" [CRON] User {user_id} has {len(days)}/{total_days_requested} days (status: {generation_status}). Using atomic shift + rolling window.")
            
            now_iso = datetime.now(timezone.utc).isoformat()
            
            # [GAP 2 FIX: Fetch rejections para aprendizaje continuo JIT]
            active_rejections = get_active_rejections(user_id=user_id)
            rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
            
            # [FIX RACE CONDITION]: Shift atomico en PostgreSQL.
            # Operador '-' en jsonb borra el indice (0). Evita colisiones con el chunk_worker que hace append.
            if len(days) > 0:
                # [GAP 6 FIX: Guard pre-shift]
                if len(days) <= 2 and generation_status == 'partial':
                    try:
                        from db_core import execute_sql_query
                        pending_chunks = execute_sql_query("""
                            SELECT COUNT(*) as cnt FROM plan_chunk_queue 
                            WHERE meal_plan_id = %s AND status IN ('pending', 'processing')
                        """, (plan_record['id'],), fetch_one=True)
                        has_pending = pending_chunks and int(pending_chunks.get('cnt', 0)) > 0
                        if has_pending and len(days) <= 1:
                            logger.warning(f" [CRON] Postponing shift for {user_id}: only {len(days)} days left but chunks pending. Waiting for chunks.")
                            from db_core import execute_sql_write
                            execute_sql_write(
                                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_rotated_at}', to_jsonb(%s::text)) WHERE id = %s",
                                (now_iso, user_id)
                            )
                            return
                    except Exception as pg_e:
                        logger.warning(f" [CRON] Error in pre-shift guard: {pg_e}")

                try:
                    from db_core import execute_sql_write
                    updated_data = execute_sql_write(
                        """
                        WITH locked AS (
                            SELECT id FROM meal_plans WHERE id = %s FOR UPDATE
                        )
                        UPDATE meal_plans SET plan_data = jsonb_set(plan_data, '{days}', (plan_data->'days') - 0) 
                        WHERE id = (SELECT id FROM locked) RETURNING plan_data
                        """, 
                        (plan_record['id'],),
                        returning=True
                    )
                    
                    if updated_data:
                        shifted_plan_data = updated_data[0].get('plan_data', {})
                        shifted_days = shifted_plan_data.get('days', [])
                        
                        modified = False
                        
                        # --- [GAP 2 FIX: Recalcular day_names post-shift] ---
                        try:
                            from datetime import timedelta
                            today = datetime.now(timezone.utc)
                            tz_offset = health_profile.get('tzOffset', 0)
                            if tz_offset:
                                try:
                                    today -= timedelta(minutes=int(tz_offset))
                                except (ValueError, TypeError):
                                    pass

                            dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                            for i, day_obj in enumerate(shifted_days):
                                target_date = today + timedelta(days=i)
                                day_obj['day_name'] = dias_es[target_date.weekday()]
                            
                            logger.info(f" [CRON] Day names recalculated post-shift: {[d.get('day_name') for d in shifted_days[:3]]}")
                        except Exception as e:
                            logger.warning(f" [CRON] Error recalculating day names: {e}")

                        # --- [ROLLING WINDOW: Generar día(s) nuevo(s) para mantener plan_size constante] ---
                        try:
                            while len(shifted_days) < total_days_requested:
                                import copy, random
                                new_day_num = int(shifted_days[-1].get('day', len(shifted_days))) + 1 if shifted_days else 1
                                last_day_meal_names = [m.get('name', '') for m in shifted_days[-1].get('meals', [])] if shifted_days else []
                                
                                new_day = {'day': new_day_num, 'meals': []}
                                
                                # Smart Shuffle: mezclar comidas de los días existentes por posición
                                max_meals = max((len(d.get('meals', [])) for d in days), default=3)
                                for meal_idx in range(max_meals):
                                    candidates = []
                                    for d in days:  # Usar días ORIGINALES (pre-shift) como pool
                                        meals_list = d.get('meals', [])
                                        if meal_idx < len(meals_list):
                                            candidates.append(meals_list[meal_idx])
                                    
                                    # Filtrar: no repetir el último día, no usar rechazos
                                    safe = [m for m in candidates if m.get('name') not in last_day_meal_names and m.get('name') not in rejected_meal_names]
                                    if not safe:
                                        safe = [m for m in candidates if m.get('name') not in rejected_meal_names]
                                    if not safe:
                                        safe = candidates  # Fallback total
                                    
                                    if safe:
                                        new_day['meals'].append(copy.deepcopy(random.choice(safe)))
                                
                                # Asignar day_name correcto
                                try:
                                    new_day_date = today + timedelta(days=len(shifted_days))
                                    new_day['day_name'] = dias_es[new_day_date.weekday()]
                                except Exception:
                                    new_day['day_name'] = f'Día {new_day_num}'
                                
                                shifted_days.append(new_day)
                                modified = True
                                logger.info(f" [ROLLING] Generated new day via Smart Shuffle: {new_day.get('day_name')} (day #{new_day_num}) for {user_id}")
                        except Exception as roll_e:
                            logger.warning(f" [ROLLING] Error generating replacement day for {user_id}: {roll_e}")
                            
                        # --- [GAP 1 FIX: JIT Comprehensive Swap (Aprendizaje Continuo en Dias Pre-generados)] ---
                        tuning_metrics = health_profile.get("tuning_metrics", {})
                        fatigue_data = calculate_ingredient_fatigue(user_id, tuning_metrics=tuning_metrics)
                        fatigued_ingredients = [f.lower() for f in fatigue_data.get('fatigued_ingredients', []) if not f.startswith('[')]
                        
                        from db_facts import get_all_user_facts
                        facts = get_all_user_facts(user_id)
                        allergy_keywords = []
                        if facts:
                            for f in facts:
                                meta = f.get("metadata", {})
                                cat = meta.get("category", "") if isinstance(meta, dict) else ""
                                if cat in ["alergia", "condicion_medica"]:
                                    text = f.get("fact", "").lower()
                                    stopw = {"alergia", "alergico", "intolerante", "intolerancia", "condicion", "medica", "tiene", "sufre", "para"}
                                    words = [w for w in text.split() if len(w) > 3 and w not in stopw]
                                    allergy_keywords.extend(words)
                        
                        swap_triggers = [r.lower() for r in rejected_meal_names] + fatigued_ingredients + allergy_keywords
                        
                        # Verificar si hay platos lejanos que entraron al rango critico y necesitan swap
                        has_pending_swaps = any(m.get('_needs_swap') for d in shifted_days for m in d.get('meals', []))
                        
                        if swap_triggers or has_pending_swaps:
                            critical_triggers = set(allergy_keywords)
                            high_triggers = set(r.lower() for r in rejected_meal_names)
                            normal_triggers = set(fatigued_ingredients)
                            
                            MAX_CRITICAL_SWAPS = 999
                            MAX_HIGH_SWAPS = 5
                            MAX_NORMAL_SWAPS = 3
                            
                            critical_count, high_count, normal_count = 0, 0, 0
                            
                            for d_idx, day_obj in enumerate(shifted_days):
                                for m_idx, meal_obj in enumerate(day_obj.get('meals', [])):
                                    m_name = meal_obj.get('name', '').lower()
                                    m_ingredients = " ".join([str(i).lower() for i in meal_obj.get('ingredients', [])])
                                    
                                    matched_trigger = None
                                    if swap_triggers:
                                        for trig in swap_triggers:
                                            if trig in m_name or trig in m_ingredients:
                                                matched_trigger = trig
                                                break
                                                
                                    needs_swap = meal_obj.get('_needs_swap', False)
                                    swap_reason = meal_obj.get('_swap_reason', matched_trigger)
                                    
                                    if matched_trigger or needs_swap:
                                        severity = 'normal'
                                        if swap_reason in critical_triggers:
                                            severity = 'critical'
                                        elif swap_reason in high_triggers:
                                            severity = 'high'
                                            
                                        # GAP 2: High/Critical triggers y manual needs_swap se ejecutan en TODOS los días del plan
                                        in_critical_range = (d_idx <= 2) or (severity in ['critical', 'high']) or needs_swap
                                        
                                        can_swap = False
                                        if in_critical_range:
                                            if severity == 'critical' and critical_count < MAX_CRITICAL_SWAPS:
                                                can_swap = True
                                                critical_count += 1
                                            elif severity == 'high' and high_count < MAX_HIGH_SWAPS:
                                                can_swap = True
                                                high_count += 1
                                            elif severity == 'normal' and normal_count < MAX_NORMAL_SWAPS:
                                                can_swap = True
                                                normal_count += 1
                                                
                                        if can_swap:
                                            logger.info(f" [CRON] JIT Swap ({severity}): '{m_name}' matched trigger/flag '{swap_reason}'. Swapping in background...")
                                            try:
                                                from agent import swap_meal
                                                swap_form_data = dict(health_profile)
                                                swap_form_data['user_id'] = user_id
                                                swap_form_data['rejected_meal'] = meal_obj.get('name')
                                                swap_form_data['meal_type'] = meal_obj.get('meal', 'Comida principal')
                                                swap_form_data['target_calories'] = meal_obj.get('calories', 400)
                                                
                                                new_meal = swap_meal(swap_form_data)
                                                if new_meal and "name" in new_meal:
                                                    shifted_days[d_idx]['meals'][m_idx] = new_meal
                                                    modified = True
                                            except Exception as swap_e:
                                                logger.error(f"  [CRON] Error during proactive JIT swap for {user_id}: {swap_e}")
                                        else:
                                            # Deferred / limit reached
                                            if not needs_swap:
                                                shifted_days[d_idx]['meals'][m_idx]['_needs_swap'] = True
                                                shifted_days[d_idx]['meals'][m_idx]['_swap_reason'] = matched_trigger
                                                modified = True
                                                if in_critical_range:
                                                    logger.info(f" [CRON] JIT Swap limit reached ({severity}). Deferred meal '{m_name}'.")
                                                else:
                                                    logger.info(f" [CRON] Día {d_idx+1} meal '{m_name}' flagged for deferred swap (trigger: '{matched_trigger}')")
                        
                        # --- [GAP 2 FIX: Recalcular Macros y Lista de Compras] ---
                        total_cals = 0
                        total_p, total_c, total_f = 0, 0, 0
                        valid_days = len(shifted_days)
                        
                        if valid_days > 0:
                            for d in shifted_days:
                                for m in d.get('meals', []):
                                    total_cals += m.get('calories', 0)
                                    total_p += m.get('protein', 0)
                                    total_c += m.get('carbs', 0)
                                    total_f += m.get('fats', 0)
                            
                            avg_cals = int(total_cals / valid_days)
                            avg_p = int(total_p / valid_days)
                            avg_c = int(total_c / valid_days)
                            avg_f = int(total_f / valid_days)
                        else:
                            avg_cals, avg_p, avg_c, avg_f = 0, 0, 0, 0
                            
                        # --- [GAP 3: TDEE Drift Detection] ---
                        # Si el usuario cambió de peso significativamente, escalar las calorías de los días restantes
                        force_chunk_invalidation = False
                        current_tdee_target = None
                        current_tdee_val = None
                        try:
                            from nutrition_calculator import calculate_nutrition
                            
                            # Recalcular TDEE con datos actuales del health_profile
                            current_nutrition = calculate_nutrition(health_profile)
                            current_target = current_nutrition.get('total_daily_calories', 0) or current_nutrition.get('target_calories', 0)
                            current_tdee_target = current_target
                            current_tdee_val = current_nutrition.get('tdee')
                            
                            # Comparar con las calorías promedio de los días restantes
                            if current_target and avg_cals:
                                drift_pct = abs(current_target - avg_cals) / max(avg_cals, 1)
                                
                                if drift_pct > 0.08:  # >8% de diferencia
                                    force_chunk_invalidation = True
                                    scale_factor = current_target / max(avg_cals, 1)
                                    logger.info(f" [TDEE DRIFT] Detected {drift_pct*100:.1f}% calorie drift for {user_id}. Scaling {avg_cals}→{current_target} kcal (factor: {scale_factor:.2f})")
                                    
                                    for d in shifted_days:
                                        for m in d.get('meals', []):
                                            m['cals'] = int(m.get('cals', m.get('calories', 0)) * scale_factor)
                                            m['calories'] = m['cals']
                                            m['protein'] = int(m.get('protein', 0) * scale_factor)
                                            m['carbs'] = int(m.get('carbs', 0) * scale_factor)
                                            m['fats'] = int(m.get('fats', 0) * scale_factor)
                                    
                                    # Recalcular promedios con los nuevos valores
                                    avg_cals = current_target
                                    avg_p = int(sum(m.get('protein',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                                    avg_c = int(sum(m.get('carbs',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                                    avg_f = int(sum(m.get('fats',0) for d in shifted_days for m in d.get('meals',[])) / max(valid_days,1))
                        except Exception as tdee_e:
                            logger.warning(f" [TDEE DRIFT] Error checking TDEE drift: {tdee_e}")
                            
                        shifted_plan_data['calories'] = avg_cals
                        shifted_plan_data['macros'] = {
                            'protein': avg_p,
                            'carbs': avg_c,
                            'fats': avg_f
                        }
                        
                        try:
                            from shopping_calculator import get_shopping_list_delta
                            household = health_profile.get("householdSize", 1)
                            
                            aggr_7 = get_shopping_list_delta(user_id, shifted_plan_data, is_new_plan=False, structured=True, multiplier=1.0 * household)
                            aggr_15 = get_shopping_list_delta(user_id, shifted_plan_data, is_new_plan=False, structured=True, multiplier=2.0 * household)
                            aggr_30 = get_shopping_list_delta(user_id, shifted_plan_data, is_new_plan=False, structured=True, multiplier=4.0 * household)
                            
                            grocery_duration = health_profile.get("groceryDuration", "weekly")
                            if grocery_duration == "biweekly":
                                aggr_active = aggr_15
                            elif grocery_duration == "monthly":
                                aggr_active = aggr_30
                            else:
                                aggr_active = aggr_7
                                
                            shifted_plan_data['aggregated_shopping_list'] = aggr_active
                            shifted_plan_data['aggregated_shopping_list_weekly'] = aggr_7
                            shifted_plan_data['aggregated_shopping_list_biweekly'] = aggr_15
                            shifted_plan_data['aggregated_shopping_list_monthly'] = aggr_30
                            
                        except Exception as shop_e:
                            logger.warning(f" [CRON] Fallo el recalculo de la lista de compras en el shift: {shop_e}")

                        # --- [ROLLING WINDOW: Avanzar grocery_start_date 1 día] ---
                        start_date_str = shifted_plan_data.get('grocery_start_date')
                        if start_date_str:
                            try:
                                from constants import safe_fromisoformat
                                from datetime import timedelta, timezone
                                start_dt = safe_fromisoformat(start_date_str)
                                if start_dt.tzinfo is None:
                                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                                new_start = start_dt + timedelta(days=1)
                                shifted_plan_data['grocery_start_date'] = new_start.isoformat()
                            except Exception as dte:
                                logger.warning(f" [CRON] Error avanzando grocery_start_date: {dte}")

                        # Actualizar plan en DB (siempre lo hacemos para reflejar el recalculate, y si hubo swap)
                        import json
                        try:
                            execute_sql_write(
                                "UPDATE meal_plans SET plan_data = %s::jsonb, calories = %s, macros = %s::jsonb WHERE id = %s",
                                (
                                    json.dumps(shifted_plan_data, ensure_ascii=False),
                                    avg_cals,
                                    json.dumps(shifted_plan_data['macros']),
                                    plan_record['id']
                                )
                            )
                            logger.info(f" [CRON] Shift atomico (GAP 2): Calories, Macros y Shopping List recalculados para {user_id}")
                        except Exception as upd_e:
                            logger.error(f" [CRON] Error actualizando plan despues del shift para {user_id}: {upd_e}")
                            
                        # --- [GAP 5 IMPLEMENTATION: Chunk Invalidation] ---
                        try:
                            # Leer el snapshot del primer chunk pendiente
                            pending_chunk = execute_sql_query("""
                                SELECT id, pipeline_snapshot FROM plan_chunk_queue
                                WHERE meal_plan_id = %s AND status = 'pending'
                                ORDER BY week_number ASC LIMIT 1
                            """, (plan_record['id'],), fetch_one=True)
                            
                            if pending_chunk:
                                snap = pending_chunk.get('pipeline_snapshot', {})
                                if isinstance(snap, str):
                                    snap = json.loads(snap)
                                snap_form = snap.get('form_data', {})
                                
                                # Campos criticos que invalidan el snapshot
                                CRITICAL_FIELDS = ['mainGoal', 'activityLevel', 'dietTypes', 'allergies', 'weight', 'height']
                                
                                changed_fields = []
                                for field in CRITICAL_FIELDS:
                                    snap_val = snap_form.get(field)
                                    current_val = health_profile.get(field)
                                    if snap_val != current_val and current_val is not None:
                                        changed_fields.append(field)
                                        
                                if changed_fields or force_chunk_invalidation:
                                    logger.warning(f" [CHUNK INVALIDATION] Campos críticos cambiaron para {user_id}: {changed_fields} (Drift: {force_chunk_invalidation}). Actualizando snapshots de chunks pendientes.")
                                    
                                    # Actualizar el form_data de TODOS los chunks pendientes con el perfil actual
                                    updated_snap = dict(snap)
                                    updated_form = dict(snap_form)
                                    for field in CRITICAL_FIELDS:
                                        if health_profile.get(field) is not None:
                                            updated_form[field] = health_profile[field]
                                            
                                    # [GAP 4 FIX] Inyectar TDEE recalculado en los chunks pendientes
                                    if current_tdee_target:
                                        updated_form['target_calories'] = current_tdee_target
                                    if current_tdee_val:
                                        updated_form['tdee'] = current_tdee_val
                                        
                                    updated_snap['form_data'] = updated_form
                                    
                                    execute_sql_write("""
                                        UPDATE plan_chunk_queue
                                        SET pipeline_snapshot = %s::jsonb, updated_at = NOW()
                                        WHERE meal_plan_id = %s AND status = 'pending'
                                    """, (json.dumps(updated_snap, ensure_ascii=False), plan_record['id']))
                        except Exception as inv_err:
                            logger.warning(f" [CHUNK INVALIDATION] Error checking for profile changes: {inv_err}")
                            
                except Exception as e:
                    logger.error(f" [CRON] Error performing atomic shift for long-term plan {user_id}: {e}")
            
            # Lock de rotacion
            try:
                from db_core import execute_sql_write
                execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_rotated_at}', to_jsonb(%s::text)) WHERE id = %s", (now_iso, user_id))
            except Exception as e:
                logger.error(f" [CRON] Error stamping rotation lock for {user_id}: {e}")
                
            # Consumir Inventario Fisico (Resta Matematica) AHORA (ya que no llegaremos al paso 5)
            if ingredients_to_consume:
                logger.info(f" [CRON] Deducting {len(ingredients_to_consume)} consumed ingredients for long-term plan user {user_id}")
                try:
                    deduct_consumed_meal_from_inventory(user_id, ingredients_to_consume)
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "DELETE FROM failed_inventory_deductions WHERE id IN (SELECT id FROM failed_inventory_deductions WHERE user_id = %s AND ingredients = %s::jsonb ORDER BY created_at DESC LIMIT 1)",
                        (user_id, json.dumps(ingredients_to_consume))
                    )
                except Exception as deduct_err:
                    logger.error(f" [CRON] Error deducting live inventory for {user_id}: {deduct_err}")
            
            # Notificacion post-rotacion
            try:
                import threading
                from utils_push import send_push_notification
                threading.Thread(
                    target=send_push_notification,
                    kwargs={
                        "user_id": user_id,
                        "title": "¡Tu proximo dia esta listo! ðŸ—",
                        "body": "Hemos avanzado tu plan al siguiente dia. Toca aqui para ver tus comidas de hoy.",
                        "url": "/dashboard"
                    }
                ).start()
                
                # [GAP 6 FIX: Push advertencia dias bajos]
                if len(shifted_days) <= 2 and generation_status == 'partial':
                    threading.Thread(
                        target=send_push_notification,
                        kwargs={
                            "user_id": user_id,
                            "title": "â³ Tu plan esta por terminar",
                            "body": f"Te quedan {len(shifted_days)} dias planificados. Estamos generando mas, pero si tarda, considera crear un plan nuevo.",
                            "url": "/dashboard"
                        }
                    ).start()
            except Exception as e:
                logger.warning(f" [CRON] Error dispatching push notification: {e}")
                
            # [GAP 1 FIX]: Persistir seÃ±ales de aprendizaje para que los chunks futuros las usen
            try:
                _persist_nightly_learning_signals(user_id, health_profile, shifted_days, consumed_records)
            except Exception as learn_err:
                logger.warning(f" [CRON] Error persisting nightly learning signals: {learn_err}")
                
            return # Salimos tempranamente, no ejecutamos la IA
        # 3. Obtener el inventario (Previo a la deduccion fisica para asegurar idempotencia)
        # Retrasamos la deduccion hasta el paso 5 para prevenir doble descuento si la IA falla.
        # Retrasamos la deduccion hasta el paso 5 para prevenir doble descuento si la IA falla.
        current_pantry_ingredients = get_user_inventory(user_id)

        # 4. Construir perfil de gustos REAL del usuario (FIX CRÃTICO)
        # Antes se pasaba analyze_preferences_agent([], [], []) generando platos genericos.
        # Ahora consultamos likes, rejections y facts para personalizar la rotacion.
        likes = get_user_likes(user_id)
        active_rejections = get_active_rejections(user_id=user_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []

        taste_profile = analyze_preferences_agent(likes, [], active_rejections=rejected_meal_names)

        logger.info(f" [CRON] Taste profile built for {user_id}: "
                    f"{len(likes)} likes, {len(rejected_meal_names)} rejections")

        # Construir contexto de memoria (facts del Cerebro IA)
        memory_context = _build_facts_memory_context(user_id)
        if memory_context:
            logger.info(f" [CRON] Memory context loaded for {user_id} ({len(memory_context)} chars)")

        # 5. Trigger Plan Pipeline to Regenerate (Shift)
        # Reconstruct the "frontend request" data using their health profile + system variables
        pipeline_data = dict(health_profile)
        pipeline_data['user_id'] = user_id
        pipeline_data['session_id'] = user_id
        # Limitamos el historial de platos previos a los ultimos 15 para no saturar el token context de OpenAI
        pipeline_data['previous_meals'] = previous_meal_names[-15:] if previous_meal_names else []
        pipeline_data['current_pantry_ingredients'] = current_pantry_ingredients
        pipeline_data['_is_background_rotation'] = True

        # Inyectamos seÃ±ales avanzadas de aprendizaje (Adherencia, Fatiga, Quality Score, etc)
        pipeline_data = _inject_advanced_learning_signals(user_id, pipeline_data, health_profile, days, consumed_records)


        logger.info(f" [CRON] Running AI Orchestrator for user {user_id}...")
        import time
        
        max_retries = 3
        result = {}
        previous_ai_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f" [CRON] AI Generation Attempt {attempt + 1}/{max_retries} for user {user_id}...")
                    
                import concurrent.futures
                # We use a ThreadPoolExecutor with timeout to prevent Thread Stall if the AI hangs
                pipeline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = pipeline_executor.submit(
                    run_plan_pipeline,
                    pipeline_data, 
                    [], 
                    taste_profile, 
                    memory_context, 
                    None,
                    previous_ai_error
                )
                try:
                    result = future.result(timeout=45)
                except concurrent.futures.TimeoutError:
                    result = {'error': 'AI Orchestrator timed out after 45 seconds (Thread Stall prevented).'}
                finally:
                    pipeline_executor.shutdown(wait=False, cancel_futures=True)
    
                # MEJORA 7: Validar que el resultado de la IA tenga estructura util antes de aceptarlo
                if 'error' not in result:
                    ai_days = result.get('days', [])
                    if not ai_days or not isinstance(ai_days, list):
                        result['error'] = "Malformed AI plan: missing or invalid 'days' array."
                    else:
                        # Verificar que al menos un dia tenga platos reales (>= 2 meals)
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
                        logger.info(f" [CRON] AI Generation succeeded on attempt {attempt + 1} for user {user_id}!")
                    break  # Success! Exit retry loop
                else:
                    previous_ai_error = result['error']
                    logger.warning(f" [CRON] Attempt {attempt + 1} failed for user {user_id}: {result['error']}")
                    if attempt < max_retries - 1:
                        sleep_time = 3 ** attempt  # 1s, 3s
                        logger.info(f" [CRON] Waiting {sleep_time}s before retrying...")
                        time.sleep(sleep_time)
                        
            except Exception as ai_err:
                logger.error(f" [CRON] Critical exception on attempt {attempt + 1} for user {user_id}: {ai_err}")
                result = {'error': str(ai_err)}
                previous_ai_error = result['error']
                if attempt < max_retries - 1:
                    sleep_time = 3 ** attempt
                    logger.info(f" [CRON] Waiting {sleep_time}s before retrying...")
                    time.sleep(sleep_time)

        if 'error' in result:
            logger.error(f" [CRON] AI Orchestrator failed for user {user_id} after {max_retries} attempts: {result['error']}")
            # MEJORA 4 y 2: Mecanismo de Fallback Inteligente si la IA falla (Timeouts, API limits, etc.)
            import copy
            import random
            
            # Hacemos shift: removemos el Dia 1. Si solo hay 1 dia, el shift queda vacio inicialmente.
            shifted_days = list(days)[1:] if len(days) > 1 else []
            new_day_num = int(shifted_days[-1].get("day", len(shifted_days))) + 1 if shifted_days else int(days[0].get("day", 1)) + 1
            
            # --- MEJORA 2: Cache de Respaldo Proactivo (Backup Plan Cache) ---
            backup_plan = health_profile.get('emergency_backup_plan', [])
            
            if isinstance(backup_plan, list) and len(backup_plan) > 0:
                logger.info(f" [CRON] Applying EMERGENCY BACKUP CACHE for user {user_id} due to AI failure. Days left in cache: {len(backup_plan)}")
                backup_day = backup_plan.pop(0)
                backup_day["day"] = new_day_num
                shifted_days.append(backup_day)
                
                # Consumir un dia del cache actualizando la DB
                try:
                    from db_core import execute_sql_write
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
                        (json.dumps(backup_plan), user_id)
                    )
                except Exception as e:
                    logger.warning(f" [CRON] Failed to update emergency backup plan state for {user_id}: {e}")
                
                result = {
                    "days": shifted_days,
                    "_fallback_used": "emergency_cache"
                }
            else:
                logger.info(f" [CRON] Applying Smart Fallback shift for user {user_id} due to AI failure and empty backup cache.")

                # MEJORA Original: Smart Fallback (Dia Frankenstein / Remix de sobras)
                # Mezclamos las comidas de los dias actuales por posicion (desayuno, comida, etc.)
                new_day = {
                    "day": new_day_num,
                    "meals": []
                }
                
                max_meals = max(len(d.get("meals", [])) for d in days)
                last_day_meals = [m.get("name") for m in shifted_days[-1].get("meals", [])] if shifted_days else []
                
                for i in range(max_meals):
                    candidates = []
                    for d in days:
                        meals_list = d.get("meals", [])
                        if i < len(meals_list):
                            candidates.append(meals_list[i])
                    
                    # Prevenir que el mismo plato se repita dos dias seguidos Y filtrar rechazos activos
                    safe_candidates = [
                        m for m in candidates 
                        if m.get("name") not in last_day_meals 
                        and m.get("name") not in rejected_meal_names
                    ]
                    
                    # Si filtramos demasiado (ej. choca con last_day_meals), al menos intentar que NO sea un rechazo
                    if not safe_candidates:
                        safe_candidates = [m for m in candidates if m.get("name") not in rejected_meal_names]
                        
                    # MEJORA 4: Smart Fallback Forzado (Paradoja de los Rechazos)
                    # Si todos los candidatos chocan con alergias/rechazos, NUNCA asignar el plato original.
                    # En su lugar, inyectamos una "Receta Segura Estatica" desde un pool global que no contenga el rechazo.
                    if not safe_candidates:
                        is_breakfast = False
                        meal_label = "Comida principal"
                        if candidates and isinstance(candidates[0], dict) and candidates[0].get("meal"):
                            meal_label = candidates[0].get("meal")
                            if "desayuno" in meal_label.lower() or "breakfast" in meal_label.lower() or i == 0:
                                is_breakfast = True
                                
                        static_fallbacks_breakfast = [
                            {"name": "Avena cocida con frutas", "ingredients": ["Avena", "Agua", "Fruta"]},
                            {"name": "Tostadas de aguacate", "ingredients": ["Pan integral", "Aguacate"]},
                            {"name": "Huevos revueltos basicos", "ingredients": ["Huevos", "Aceite de oliva"]},
                        ]
                        
                        static_fallbacks_main = [
                            {"name": "Pollo a la plancha con arroz", "ingredients": ["Pechuga de pollo", "Arroz blanco"]},
                            {"name": "Pescado al horno con vegetales", "ingredients": ["Filete de pescado blanco", "Vegetales mixtos"]},
                            {"name": "Ensalada de garbanzos", "ingredients": ["Garbanzos", "Tomate", "Lechuga"]},
                            {"name": "Carne molida magra con papas", "ingredients": ["Carne molida magra", "Papa"]},
                        ]
                        
                        pool = static_fallbacks_breakfast if is_breakfast else static_fallbacks_main
                        
                        chosen_static = None
                        for fb in pool:
                            fb_text = (fb["name"] + " " + " ".join(fb["ingredients"])).lower()
                            conflict = False
                            for rej in rejected_meal_names:
                                if rej.lower() in fb_text:
                                    conflict = True
                                    break
                            if not conflict:
                                chosen_static = fb
                                break
                                
                        if not chosen_static:
                            chosen_static = pool[0] # Fallback del fallback
                            
                        safe_fallback_meal = {
                            "name": f"{chosen_static['name']} (Emergencia)",
                            "meal": meal_label,
                            "ingredients": chosen_static["ingredients"],
                            "instructions": "Preparar de forma basica y sencilla. Plato de emergencia generado por restricciones extremas.",
                            "calories": 400,
                            "protein": 25,
                            "carbs": 40,
                            "fats": 15
                        }
                        
                        logger.info(f" [CRON] Paradoja de rechazos para {user_id}. Inyectando receta estatica: {safe_fallback_meal['name']}")
                        safe_candidates = [safe_fallback_meal]
                        
                    if safe_candidates:
                        chosen_meal = copy.deepcopy(random.choice(safe_candidates))
                        new_day["meals"].append(chosen_meal)
                
                shifted_days.append(new_day)

                # Construir un resultado "mock" que el resto del sistema pueda guardar
                result = {
                    "days": shifted_days,
                    "_fallback_used": "smart_shuffle"
                }

        # Preservar grocery_start_date del ciclo actual para no romper tracking de supermercado
        if previous_meal_names:
            result['grocery_start_date'] = plan_data.get('grocery_start_date', plan_record.get('created_at', datetime.now(timezone.utc).isoformat()))
            # MEJORA 6: Limitar historial a los ultimos 30 dias para no engordar el JSON
            nuevo_historial = plan_data.get('rotation_history', []) + [{
                "date": datetime.now(timezone.utc).isoformat(),
                "meals_consumed": previous_meal_names
            }]
            result['rotation_history'] = nuevo_historial[-30:]

        # 5. Guarda y Trackea
        selected_techniques = result.pop("_selected_techniques", None)
        
        # === STAMP LOCK (TRANSACTIONAL) ===
        # Sellamos el candado en la misma transaccion que guarda el plan para asegurar
        # que no haya "Dual Writes" (si el plan se guarda, el lock se estampa si o si).
        now_iso = datetime.now(timezone.utc).isoformat()
        lock_query = "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_rotated_at}', to_jsonb(%s::text)) WHERE id = %s"
        lock_params = (now_iso, user_id)

        import json
        additional_queries = [(lock_query, lock_params)]
        
        if ingredients_to_consume:
            deduction_query = "INSERT INTO failed_inventory_deductions (user_id, ingredients) VALUES (%s, %s::jsonb)"
            additional_queries.append((deduction_query, (user_id, json.dumps(ingredients_to_consume))))

        # Si esto falla, la excepcion aborta la tarea, se hace rollback y el worker la reintenta.
        _save_plan_and_track_background(
            user_id, 
            result, 
            selected_techniques, 
            additional_db_queries=additional_queries
        )

        logger.info(f" [CRON] Rotation lock and inventory queue stamped transactionally for user {user_id} at {now_iso}")

        # === IDEMPOTENCIA: Consumir Inventario Fisico (Resta Matematica) AHORA ===
        # Solo descontamos si el plan y el lock se guardaron exitosamente.
        if ingredients_to_consume:
            logger.info(f" [CRON] Deducting {len(ingredients_to_consume)} consumed ingredients for user {user_id}")
            try:
                deduct_consumed_meal_from_inventory(user_id, ingredients_to_consume)
                # Si la deduccion en vivo tiene exito, eliminamos el registro de la cola de fallos
                execute_sql_write(
                    "DELETE FROM failed_inventory_deductions WHERE id IN (SELECT id FROM failed_inventory_deductions WHERE user_id = %s AND ingredients = %s::jsonb ORDER BY created_at DESC LIMIT 1)",
                    (user_id, json.dumps(ingredients_to_consume))
                )
                logger.info(f" [CRON] Live deduction succeeded. Removed from failed_inventory_deductions for {user_id}")
            except Exception as deduct_err:
                logger.error(f" [CRON] Error deducting live inventory for {user_id}: {deduct_err}. Leaving in queue.")

        # Enviar notificacion post-rotacion asincronamente (fire-and-forget)
        try:
            import threading
            from utils_push import send_push_notification
            threading.Thread(
                target=send_push_notification,
                kwargs={
                    "user_id": user_id,
                    "title": "¡Tus platos de hoy estan listos! ðŸ—",
                    "body": "Renovamos tu menu basandonos en tus ultimos registros y gustos. Toca aqui para verlos.",
                    "url": "/dashboard"
                },
                daemon=True
            ).start()
        except Exception as push_err:
            logger.warning(f"  [CRON] Failed to dispatch push notification thread for {user_id}: {push_err}")

        # === CACHÃ‰ DE RESPALDO (JIT) ===
        try:
            backup_plan = health_profile.get("emergency_backup_plan", [])
            if not isinstance(backup_plan, list) or len(backup_plan) < 3:
                import threading
                threading.Thread(
                    target=_refill_emergency_backup_plan,
                    args=(user_id, pipeline_data, taste_profile, memory_context),
                    daemon=True
                ).start()
        except Exception as refill_err:
            logger.warning(f"  [CRON] Failed to dispatch refill thread for {user_id}: {refill_err}")

        logger.info(f" [CRON] Auto-Rotation successfully completed for user {user_id}")

    except Exception as e:
        logger.error(f"  [CRON] Exception while rotating user {user_id}: {e}")
        logger.error(traceback.format_exc())
        # Re-lanzar la excepcion para que el Worker de la cola se entere y cuente el reintento
        raise



def enqueue_nightly_rotations():
    logger.info("ðŸ•’ [CRON] Enqueuing Nightly Auto-Rotation for Premium Users...")
    
    query = '''
        INSERT INTO nightly_rotation_queue (user_id, status)
        SELECT id, 'pending'
        FROM user_profiles 
        WHERE COALESCE((health_profile->>'autoRotateMeals')::boolean, true) = true
        AND extract(hour from (NOW() AT TIME ZONE COALESCE(health_profile->>'timezone', 'America/Santo_Domingo'))) BETWEEN 1 AND 4
        AND NOT EXISTS (
            SELECT 1 FROM nightly_rotation_queue nrq 
            WHERE nrq.user_id = user_profiles.id 
            AND (nrq.status IN ('pending', 'processing') OR nrq.updated_at > NOW() - INTERVAL '20 hours')
        )
        RETURNING user_id;
    '''
    try:
        inserted = execute_sql_write(query, returning=True)
        count = len(inserted) if inserted else 0
        if count > 0:
            logger.info(f" [CRON] Successfully enqueued {count} users for nightly rotation (Bulk Insert).")
        else:
            logger.info("â­ï¸ [CRON] No new users needed to be enqueued at this time.")
    except Exception as e:
        logger.error(f" [CRON] Error during bulk enqueuing of auto-rotation users: {e}")

def run_nightly_auto_rotation():
    # Backward compat
    enqueue_nightly_rotations()

def process_failed_deductions():
    """Background task para reintentar deducciones de inventario que fallaron temporalmente."""
    # 0. Crear tabla si no existe
    try:
        execute_sql_write("""
            CREATE TABLE IF NOT EXISTS failed_inventory_deductions (
                id SERIAL PRIMARY KEY,
                user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
                ingredients JSONB NOT NULL,
                attempts INT DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
    except Exception as e:
        logger.warning(f" [CRON] Error verificando/creando failed_inventory_deductions (quizas permisos): {e}")

    # Limpiar tareas muy viejas o que fallaron demasiadas veces (max 5 intentos)
    try:
        execute_sql_write("DELETE FROM failed_inventory_deductions WHERE attempts >= 5 OR created_at < NOW() - INTERVAL '7 days';")
    except Exception:
        pass
        
    query = '''
        SELECT id, user_id, ingredients FROM failed_inventory_deductions
        ORDER BY created_at ASC
        FOR UPDATE SKIP LOCKED
        LIMIT 10;
    '''
    try:
        pending_tasks = execute_sql_query(query, fetch_all=True)
    except Exception as e:
        # Si la tabla no existe aun, ignorar
        if "relation" not in str(e).lower():
            logger.error(f" [CRON] Error locking failed deductions: {e}")
        return

    if not pending_tasks:
        return
        
    logger.info(f" [CRON] Processing {len(pending_tasks)} failed inventory deductions.")
    
    for task in pending_tasks:
        task_id = task['id']
        user_id = str(task['user_id'])
        ingredients = task['ingredients']
        
        try:
            if isinstance(ingredients, str):
                import json
                ingredients = json.loads(ingredients)
                
            deduct_consumed_meal_from_inventory(user_id, ingredients)
            execute_sql_write("DELETE FROM failed_inventory_deductions WHERE id = %s", (task_id,))
            logger.info(f" [CRON] Successfully recovered failed deduction for {user_id}")
        except Exception as e:
            logger.error(f" [CRON] Failed to recover deduction {task_id} for {user_id}: {e}")
            execute_sql_write(
                "UPDATE failed_inventory_deductions SET attempts = COALESCE(attempts, 0) + 1, updated_at = NOW() WHERE id = %s", 
                (task_id,)
            )

def process_rotation_queue():
    logger.info("ðŸ•’ [CRON] Checking nightly rotation queue...")
    
    # Run recovery of failed inventory deductions
    process_failed_deductions()
    
    # 0. Asegurar que la columna 'attempts' existe (Migracion on-the-fly)
    try:
        execute_sql_write("ALTER TABLE nightly_rotation_queue ADD COLUMN IF NOT EXISTS attempts INT DEFAULT 0;")
    except Exception:
        pass
        
    # 0.5. Limpieza de Cola (Queue Bloat): Eliminar tareas viejas (completadas/fallidas de > 7 dias)
    try:
        execute_sql_write("""
            DELETE FROM nightly_rotation_queue 
            WHERE status IN ('completed', 'failed') AND updated_at < NOW() - INTERVAL '7 days';
        """)
    except Exception as e:
        logger.warning(f" [CRON] Error limpiando tareas antiguas de la cola: {e}")
    
    # 1. Rescate de Zombies: devolver a 'pending' o 'failed' tareas procesando por mas de 15 mins
    rescue_query = '''
        UPDATE nightly_rotation_queue 
        SET attempts = COALESCE(attempts, 0) + 1,
            status = CASE WHEN COALESCE(attempts, 0) + 1 >= 3 THEN 'failed' ELSE 'pending' END,
            updated_at = NOW() 
        WHERE status = 'processing' 
        AND updated_at < NOW() - INTERVAL '15 minutes'
    '''
    try:
        execute_sql_write(rescue_query)
    except Exception as e:
        logger.warning(f" [CRON] No se pudo rescatar tareas zombie: {e}")

    query = '''
        UPDATE nightly_rotation_queue
        SET status = 'processing', updated_at = NOW()
        WHERE id IN (
            SELECT id FROM nightly_rotation_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 5
        )
        RETURNING id, user_id;
    '''
    try:
        pending_tasks = execute_sql_write(query, returning=True)
    except Exception as e:
        logger.error(f" [CRON] Error locking rotation queue tasks: {e}")
        return

    if not pending_tasks:
        return
        
    logger.info(f" [CRON] Processing {len(pending_tasks)} users from the queue concurrently.")
        
    import concurrent.futures

    def _worker(task):
        user_id = str(task['user_id'])
        user_query = "SELECT id, health_profile FROM user_profiles WHERE id = %s"
        try:
            u_data = execute_sql_query(user_query, (user_id,), fetch_one=True)
            if u_data:
                _process_user(u_data)
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'completed', updated_at = NOW() WHERE id = %s", (task['id'],))
            else:
                # Irrecuperable, usuario no existe
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed', updated_at = NOW() WHERE id = %s", (task['id'],))
        except Exception as e:
            logger.error(f" [CRON] Worker failed for user {user_id}: {e}")
            try:
                # MEJORA 2: Reintentar tarea devolviendola a pending si tiene menos de 3 intentos
                update_result = execute_sql_write('''
                    UPDATE nightly_rotation_queue 
                    SET attempts = COALESCE(attempts, 0) + 1,
                        status = CASE WHEN COALESCE(attempts, 0) + 1 >= 3 THEN 'failed' ELSE 'pending' END,
                        updated_at = NOW() 
                    WHERE id = %s
                    RETURNING status
                ''', (task['id'],), returning=True)
                
                if update_result and update_result[0]['status'] == 'failed':
                    logger.error(f" [CRON] Task for user {user_id} permanently failed after 3 attempts. Sending contingency push.")
                    try:
                        import threading
                        from utils_push import send_push_notification
                        threading.Thread(
                            target=send_push_notification,
                            kwargs={
                                "user_id": user_id,
                                "title": "Problemas actualizando tu menu âš ï¸",
                                "body": "Tuvimos un inconveniente tecnico al rotar tus platos. Conservaremos tu menu actual mientras lo solucionamos.",
                                "url": "/dashboard"
                            },
                            daemon=True
                        ).start()
                    except Exception as push_err:
                        logger.warning(f" [CRON] Failed to dispatch fallback push notification thread for {user_id}: {push_err}")
            except Exception:
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed', updated_at = NOW() WHERE id = %s", (task['id'],))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(_worker, pending_tasks)


# ============================================================
# BACKGROUND CHUNKING â€" Generacion de Semanas 2-4 en background
# ============================================================

def _enqueue_plan_chunk(user_id: str, meal_plan_id: str, week_number: int, days_offset: int, days_count: int, pipeline_snapshot: dict):
    """Inserta un job en plan_chunk_queue para generar un chunk en background con delay just-in-time.

    Delay = max(0, days_offset - 1): el chunk se genera 1 día antes de necesitarse.
    Ejemplo: chunk que cubre días 4-6 (offset=3) → delay=1 día → se genera el día 2.

    [GAP B] Para chunks finales (week >= total_weeks - 1) en planes largos (15+ días),
    adelantamos el delay a (days_offset - 3) para tener margen ante fallos antes de que
    el plan se consuma vía rotación nocturna y entre en la lógica de expiración.
    Validamos days_offset como int para blindar el uso de make_interval contra cualquier input.
    """
    import json
    import math

    days_offset_int = max(0, int(days_offset))
    # [GAP 5] Delay = max(0, days_offset - 1): el chunk se genera 1 día antes para recabar más adherencia.
    delay_days = max(0, days_offset_int - 1)

    # [GAP B] Adelantar chunks finales en planes largos para reducir riesgo de expiración prematura.
    try:
        total_days = int(pipeline_snapshot.get("totalDays") or 0)
        if total_days >= 15:
            from constants import PLAN_CHUNK_SIZE
            total_weeks = math.ceil(total_days / PLAN_CHUNK_SIZE)
            # Los últimos 2 chunks tienen +2 días extra de margen
            if week_number >= total_weeks - 1:
                delay_days = max(0, days_offset_int - 3)
                logger.info(f" [GAP B] Chunk final {week_number}/{total_weeks} adelantado: delay={delay_days}d (was {days_offset_int - 1}d)")
    except Exception as e:
        logger.debug(f" [GAP B] No se pudo aplicar adelanto de chunk final: {e}")

    delay_days = min(delay_days, 180)  # hard cap: ~6 meses, evita fechas absurdas

    # [GAP 3 FIX]: Calcular execute_after exacto usando la medianoche local en UTC
    start_date_iso = pipeline_snapshot.get("form_data", {}).get("_plan_start_date")
    if start_date_iso:
        from constants import safe_fromisoformat
        from datetime import timedelta
        try:
            start_dt = safe_fromisoformat(start_date_iso)
            # Queremos ejecutar el chunk en la madrugada del día que corresponde
            execute_dt = start_dt + timedelta(days=delay_days, hours=1)
            
            # [GAP E] Idempotencia fuerte: si ya existe un chunk vivo (pending/processing/stale/failed)
            # para este (meal_plan_id, week_number), el UNIQUE parcial lo bloquea.
            # Usamos una guardia explícita para evitar la excepción y retornar silenciosamente.
            existing = execute_sql_query(
                """
                SELECT id, status FROM plan_chunk_queue
                WHERE meal_plan_id = %s AND week_number = %s
                  AND status IN ('pending', 'processing', 'stale', 'failed')
                LIMIT 1
                """,
                (str(meal_plan_id), week_number), fetch_one=True
            )
            if existing:
                logger.info(f"[GAP E] Chunk {week_number} para plan {meal_plan_id} ya existe (id={existing['id']}, status={existing['status']}). Skip enqueue.")
                return

            execute_sql_write(
                """
                INSERT INTO plan_chunk_queue
                    (user_id, meal_plan_id, week_number, days_offset, days_count, pipeline_snapshot, execute_after)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::timestamptz)
                """,
                (
                    user_id, str(meal_plan_id), week_number, days_offset_int, days_count,
                    json.dumps(pipeline_snapshot, ensure_ascii=False), execute_dt.isoformat()
                )
            )
            logger.info(f" [CHUNK] Chunk {week_number} encolado para plan {meal_plan_id} (días {days_offset_int+1}–{days_offset_int+days_count}) ejecutará a las {execute_dt.isoformat()}")
            return
        except Exception as e:
            logger.warning(f" [CHUNK] Error parseando _plan_start_date en enqueue, usando fallback NOW(): {e}")

    # Fallback si no hay _plan_start_date o hubo error parseando
    # [GAP E] Guardia de idempotencia
    existing = execute_sql_query(
        """
        SELECT id, status FROM plan_chunk_queue
        WHERE meal_plan_id = %s AND week_number = %s
          AND status IN ('pending', 'processing', 'stale', 'failed')
        LIMIT 1
        """,
        (str(meal_plan_id), week_number), fetch_one=True
    )
    if existing:
        logger.info(f"[GAP E] Chunk {week_number} para plan {meal_plan_id} ya existe (id={existing['id']}, status={existing['status']}). Skip enqueue.")
        return

    execute_sql_write(
        """
        INSERT INTO plan_chunk_queue
            (user_id, meal_plan_id, week_number, days_offset, days_count, pipeline_snapshot, execute_after)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb, NOW() + make_interval(days => %s))
        """,
        (
            user_id, str(meal_plan_id), week_number, days_offset_int, days_count,
            json.dumps(pipeline_snapshot, ensure_ascii=False), delay_days
        )
    )
    logger.info(f" [CHUNK] Chunk {week_number} encolado para plan {meal_plan_id} (días {days_offset_int+1}–{days_offset_int+days_count}) con delay de {delay_days} días")


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
):
    """[GAP G] Inserta una fila en plan_chunk_metrics para análisis histórico."""
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
                 learning_repeat_pct, rejection_violations, allergy_violations, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk_id, meal_plan_id, user_id, week_number, days_count,
                duration_ms, quality_tier, was_degraded, retries, lag_seconds,
                repeat_pct, rej_viol, alg_viol, error_message,
            ),
        )
    except Exception as e:
        # No bloquear al worker por fallas de observabilidad
        logger.warning(f"[GAP G] Error insertando métrica de chunk: {e}")


def _alert_if_degraded_rate_high():
    """[GAP G] Si el % de chunks Smart Shuffle (no-LLM) en las últimas 24h supera 15%, alertar en logs.
    Meramente observacional — pensado para scraping de logs / alertmanager.
    """
    try:
        row = execute_sql_query(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE was_degraded OR quality_tier IN ('shuffle', 'edge', 'emergency')) AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            """,
            fetch_one=True,
        )
        if not row:
            return
        total = int(row.get("total") or 0)
        degraded = int(row.get("degraded") or 0)
        if total >= 10:  # umbral mínimo de muestra
            ratio = degraded / total
            if ratio > 0.15:
                logger.error(
                    f"[GAP G/ALERT] Degraded rate 24h = {ratio:.1%} ({degraded}/{total} chunks). "
                    f"LLM possiblemente inestable. Investigar."
                )
    except Exception as e:
        logger.warning(f"[GAP G] Error en alert de degraded rate: {e}")


def _calculate_learning_metrics(new_days: list, prior_meals: list, rejected_names: list, allergy_keywords: list, fatigued_ingredients: list) -> dict:
    """[GAP F] Calcula métricas para validar que el aprendizaje continuo funciona.

    Devuelve:
      - learning_repeat_pct: % de prior_meals que reaparecen en new_days (idealmente 0-10%)
      - rejection_violations: # de nombres rechazados que reaparecen (debe ser 0)
      - allergy_violations: # de keywords de alergia en ingredientes de new_days (debe ser 0)
      - fatigued_violations: # de ingredientes fatigados que reaparecen (informativo)
      - sample_repeats: primeros 5 nombres que repitieron (para debug)
    """
    def _norm(s: str) -> str:
        if not s:
            return ""
        try:
            return strip_accents(str(s).lower()).strip()
        except Exception:
            return str(s).lower().strip()

    prior_set = {_norm(m) for m in (prior_meals or []) if m}
    rejected_set = {_norm(m) for m in (rejected_names or []) if m}
    allergy_set = {_norm(k) for k in (allergy_keywords or []) if k and len(k) > 2}
    fatigued_set = {_norm(f) for f in (fatigued_ingredients or []) if f and not f.startswith('[')}

    new_meal_names = []
    new_ingredients_blob = []
    for d in (new_days or []):
        if not isinstance(d, dict):
            continue
        for m in d.get("meals", []) or []:
            if not isinstance(m, dict):
                continue
            name = _norm(m.get("name", ""))
            if name:
                new_meal_names.append(name)
            for ing in m.get("ingredients", []) or []:
                new_ingredients_blob.append(_norm(str(ing)))

    repeats = [n for n in new_meal_names if n in prior_set]
    rejection_hits = [n for n in new_meal_names if n in rejected_set]

    ingredients_text = " ".join(new_ingredients_blob)
    allergy_hits = [k for k in allergy_set if k in ingredients_text]
    fatigued_hits = [k for k in fatigued_set if k in ingredients_text]

    total_new = len(new_meal_names)
    repeat_pct = round((len(repeats) / total_new) * 100.0, 2) if total_new else 0.0

    return {
        "total_new_meals": total_new,
        "learning_repeat_pct": repeat_pct,
        "rejection_violations": len(rejection_hits),
        "allergy_violations": len(allergy_hits),
        "fatigued_violations": len(fatigued_hits),
        "sample_repeats": repeats[:5],
        "sample_rejection_hits": rejection_hits[:5],
        "sample_allergy_hits": allergy_hits[:5],
        "prior_meals_count": len(prior_set),
        "rejected_count": len(rejected_set),
        "allergy_keywords_count": len(allergy_set),
    }


def _detect_and_escalate_stuck_chunks():
    """[GAP A] Detecta chunks atrasados (lag > 24h sin pickup) y los escala.

    Acciones:
      - Loguea métricas de chunks stuck.
      - Si lag > 24h: marca escalated_at, baja execute_after a NOW() para que el
        worker los tome en el siguiente tick, e incrementa attempts solo una vez
        para preservar la información de que ya fueron intervenidos.
      - Si lag > 72h y attempts >= 3: marca como 'failed' y envía push (rescate fallido).
    """
    try:
        # 1. Detectar stuck (>24h sin pickup) que aún NO fueron escalados
        stuck_rows = execute_sql_query("""
            SELECT id, meal_plan_id, week_number, attempts,
                   EXTRACT(EPOCH FROM (NOW() - execute_after))::int AS lag_seconds
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND execute_after < NOW() - INTERVAL '24 hours'
              AND escalated_at IS NULL
            ORDER BY execute_after ASC
            LIMIT 50
        """) or []

        if stuck_rows:
            logger.warning(f" [GAP A] {len(stuck_rows)} chunks stuck (lag>24h) detectados. Escalando...")
            for r in stuck_rows:
                lag_h = round((r.get('lag_seconds') or 0) / 3600.0, 1)
                logger.warning(f"   ↳ chunk {r['id']} plan={r['meal_plan_id']} week={r['week_number']} lag={lag_h}h attempts={r.get('attempts', 0)}")

            execute_sql_write("""
                UPDATE plan_chunk_queue
                SET escalated_at = NOW(),
                    execute_after = NOW(),
                    updated_at = NOW()
                WHERE status IN ('pending', 'stale')
                  AND execute_after < NOW() - INTERVAL '24 hours'
                  AND escalated_at IS NULL
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
            logger.error(f" [GAP A] {len(terminal)} chunks marcados como 'failed' tras 72h sin recuperación.")

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
                    logger.warning(f" [GAP A] Error enviando push de fallo terminal a {uid}: {push_err}")
    except Exception as e:
        logger.error(f" [GAP A] Error en _detect_and_escalate_stuck_chunks: {e}")


def process_plan_chunk_queue(target_plan_id=None):
    """Worker que genera las semanas 2-4 de planes de largo plazo. Corre cada minuto vía APScheduler."""
    import json

    _process_pending_shopping_lists()

    # [GAP A] Detección y escalado proactivo de chunks atrasados (lag > 24h)
    _detect_and_escalate_stuck_chunks()

    # [GAP 3 FIX: Cleanup chunks huérfanos]
    try:
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
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int
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
            RETURNING id, user_id, meal_plan_id, week_number, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, escalated_at;
        """
        params = (target_plan_id,)
    else:
        query = """
            UPDATE plan_chunk_queue
            SET status = 'processing',
                updated_at = NOW(),
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int
            WHERE id IN (
                SELECT q1.id FROM plan_chunk_queue q1
                WHERE q1.status IN ('pending', 'stale')
                AND q1.execute_after <= NOW()
                AND q1.meal_plan_id NOT IN (
                    SELECT meal_plan_id FROM plan_chunk_queue WHERE status = 'processing'
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
            RETURNING id, user_id, meal_plan_id, week_number, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, escalated_at;
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
        lags = [int(t.get("lag_seconds_at_pickup") or 0) for t in tasks]
        if lags:
            max_lag_h = max(lags) / 3600.0
            avg_lag_h = (sum(lags) / len(lags)) / 3600.0
            escalated_count = sum(1 for t in tasks if t.get("escalated_at") is not None)
            if max_lag_h >= 1.0 or escalated_count > 0:
                logger.warning(
                    f"📊 [GAP A/SLA] Pickup batch: n={len(tasks)} avg_lag={avg_lag_h:.1f}h "
                    f"max_lag={max_lag_h:.1f}h escalated={escalated_count}"
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
        lag_seconds = int(task.get("lag_seconds_at_pickup") or 0)
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
            
        form_data = copy.deepcopy(snap.get("form_data", {}))

        try:
            # [GAP 3 FIX: GUARD validar plan activo y no-fallido]
            active_plan = execute_sql_query(
                "SELECT id, plan_data->>'generation_status' as status FROM meal_plans WHERE id = %s",
                (meal_plan_id,), fetch_one=True
            )
            if not active_plan:
                logger.info(f" [CHUNK] Plan {meal_plan_id} no existe. Cancelando chunk {week_number}.")
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return
                
            if active_plan.get('status') == 'failed':
                logger.info(f" [CHUNK] Plan {meal_plan_id} esta fallido. Cancelando chunk {week_number}.")
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return

            is_degraded = snap.get("_degraded", False)
            
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
            
            if is_degraded:
                logger.warning(f" [CHUNK DEGRADED] Generando chunk {week_number} en modo degraded (Smart Shuffle) para plan {meal_plan_id}...")
                plan_row_prior = execute_sql_query(
                    "SELECT plan_data FROM meal_plans WHERE id = %s",
                    (meal_plan_id,), fetch_one=True
                )
                prior_plan_data = plan_row_prior.get("plan_data", {}) if plan_row_prior else {}
                prior_days = prior_plan_data.get("days", [])
                
                if not prior_days:
                    raise Exception("No prior days available for Smart Shuffle")
                
                
                new_days = []
                safe_pool = [d for d in prior_days if isinstance(d.get("meals"), list) and len(d["meals"]) > 0]
                if not safe_pool:
                    safe_pool = prior_days
                    
                # [GAP C FIX: Filtrar prior_days contra alergias y rechazos actuales]
                user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                health_profile = user_res.get("health_profile", {}) if user_res else {}
                current_allergies = health_profile.get("allergies", [])
                current_dislikes = health_profile.get("dislikes", [])
                
                if isinstance(current_allergies, str): current_allergies = [current_allergies] if current_allergies.strip() else []
                if isinstance(current_dislikes, str): current_dislikes = [current_dislikes] if current_dislikes.strip() else []
                
                blocklist = current_allergies + current_dislikes
                
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
                            from constants import DOMINICAN_PROTEINS, DOMINICAN_CARBS, DOMINICAN_VEGGIES_FATS
                            # Construir un dia sintetico (Edge Recipe) usando el catalogo global
                            edge_day = {
                                "day": 0,
                                "day_name": "",
                                "meals": [
                                    {
                                        "name": f"Desayuno: {random.choice(DOMINICAN_PROTEINS)} con {random.choice(DOMINICAN_CARBS)}",
                                        "type": "Desayuno",
                                        "description": "Desayuno tradicional (Edge Recipe)",
                                        "ingredients": ["1 porción proteína", "1 porción carbohidrato"],
                                        "macros": {"calories": 400, "protein": 20, "carbs": 35, "fat": 15},
                                        "instructions": ["Preparar ingredientes según método tradicional"]
                                    },
                                    {
                                        "name": f"Almuerzo: {random.choice(DOMINICAN_PROTEINS)} con {random.choice(DOMINICAN_CARBS)} y {random.choice(DOMINICAN_VEGGIES_FATS)}",
                                        "type": "Almuerzo",
                                        "description": "Almuerzo tradicional (Edge Recipe)",
                                        "ingredients": ["1 porción proteína", "1 porción carbohidrato", "1 porción vegetales"],
                                        "macros": {"calories": 500, "protein": 35, "carbs": 60, "fat": 10},
                                        "instructions": ["Cocinar a la plancha o al vapor"]
                                    },
                                    {
                                        "name": f"Cena: {random.choice(DOMINICAN_PROTEINS)} con {random.choice(DOMINICAN_VEGGIES_FATS)}",
                                        "type": "Cena",
                                        "description": "Cena ligera (Edge Recipe)",
                                        "ingredients": ["1 porción proteína", "1 porción vegetales"],
                                        "macros": {"calories": 350, "protein": 25, "carbs": 20, "fat": 15},
                                        "instructions": ["Saltear ingredientes juntos"]
                                    }
                                ]
                            }
                            # Validar que no estemos repitiendo el hash por accidente
                            if str([m.get('name') for m in edge_day.get('meals', [])]) != last_chosen_hash:
                                logger.info(f"[GAP6/CHUNK] Smart Shuffle sin variedad para {user_id}: inyectando Edge Recipe del catalogo global.")
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
                    if is_edge_recipe:
                        shuffled_day['_is_edge_recipe'] = True
                        shuffled_day['quality_tier'] = 'edge'
                    elif is_emergency_repeat:
                        shuffled_day['_is_emergency_repeat'] = True
                        shuffled_day['quality_tier'] = 'emergency'
                    else:
                        shuffled_day['quality_tier'] = 'shuffle'
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
            else:
                # [FIX CRÃTICO 3 â€" Anti-repeticion cross-semanas]:
                # NO usar snap["previous_meals"] (solo tiene semana 1). Releer TODOS los platos
                # ya generados del plan actual en DB. Esto cubre el caso donde los chunks
                # de semanas 2, 3, 4 se procesan en orden o se solapan.
                plan_row_prior = execute_sql_query(
                    "SELECT plan_data FROM meal_plans WHERE id = %s",
                    (meal_plan_id,), fetch_one=True
                )
                if not plan_row_prior:
                    raise Exception(f"Plan {meal_plan_id} no encontrado al leer contexto")
                prior_plan_data = plan_row_prior.get("plan_data") or {}
                prior_days = prior_plan_data.get("days", []) or []
                prior_meals = [
                    m.get("name") for d in prior_days
                    for m in (d.get("meals") or []) 
                    if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                ]
    
                # Construir form_data para el pipeline con el offset correcto
                # form_data fue inicializado arriba
                form_data["_days_offset"] = days_offset
                form_data["_days_to_generate"] = days_count
                # Platos de TODOS los dias previos (no solo semana 1) para anti-repeticion real
                form_data["_chunk_prior_meals"] = prior_meals
                form_data["previous_meals"] = prior_meals
                
                # [GAP 1 FIX]: Propagar la ultima tecnica del chunk anterior
                prior_skeleton = prior_plan_data.get("_skeleton", {})
                prior_techniques = prior_skeleton.get("_selected_techniques", [])
                if prior_techniques:
                    form_data["_last_technique"] = prior_techniques[-1]
    
                # [APRENDIZAJE JIT]: Recalcular perfil y memoria con datos frescos
                logger.info(f" [CHUNK] Recalculando perfil y memoria Just-in-Time para usuario {user_id}...")
                from db import get_user_likes, get_active_rejections
                from agent import analyze_preferences_agent
    
                likes_actualizados = get_user_likes(user_id)
                rechazos_nuevos = get_active_rejections(user_id=user_id)
                rejected_meal_names = [r["meal_name"] for r in rechazos_nuevos] if rechazos_nuevos else []
    
                # Re-evaluar gustos (aprende de la semana 1)
                taste_profile = analyze_preferences_agent(likes_actualizados, [], active_rejections=rejected_meal_names)
    
                # Re-leer memoria a largo plazo (alergias nuevas, etc)
                memory_context = _build_facts_memory_context(user_id)

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
                if plan_start_date_str:
                    since_time = plan_start_date_str
                else:
                    since_time = (datetime.now(timezone.utc) - timedelta(days=max(7, days_offset))).isoformat()
                    
                chunk_consumed_records = get_consumed_meals_since(user_id, since_time)
                
                # --- MEJORA 2: CHUNKS FRESCOS ---
                # Forzar un mini-persist de señales de aprendizaje antes de leer el perfil
                # Esto asegura que los likes/rechazos o comidas hechas hoy se tengan en cuenta
                try:
                    temp_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                    temp_hp = temp_res.get("health_profile", {}) if temp_res else {}
                    _persist_nightly_learning_signals(user_id, temp_hp, prior_days, chunk_consumed_records)
                    logger.info(f" [CHUNKS FRESCOS] Señales persistidas Just-In-Time para {user_id} antes del chunk.")
                except Exception as e:
                    logger.warning(f" [CHUNKS FRESCOS] Error en mini-persist: {e}")
                
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
                
                form_data = _inject_advanced_learning_signals(user_id, form_data, chunk_health_profile, prior_days, chunk_consumed_records)

                # [GAP F] Capturar triggers de aprendizaje para medir violaciones post-generación
                tuning_metrics = chunk_health_profile.get("tuning_metrics", {})
                _fatigue_data = calculate_ingredient_fatigue(user_id, tuning_metrics=tuning_metrics)
                _fatigued_ingredients = [f for f in _fatigue_data.get('fatigued_ingredients', []) if not str(f).startswith('[')]
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
    
                import concurrent.futures as _cf
                executor = _cf.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(run_plan_pipeline, form_data, [], taste_profile, memory_context, None, None)
                try:
                    result = future.result(timeout=90)
                except _cf.TimeoutError:
                    raise Exception("Chunk pipeline timed out after 90s")
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
    
                new_days = result.get("days", [])
                if not new_days or "error" in result:
                    raise Exception(result.get("error", "No days generated"))

                # [GAP C] Marcar días generados por LLM con su tier de calidad
                for d in new_days:
                    if isinstance(d, dict) and 'quality_tier' not in d:
                        d['quality_tier'] = 'llm'

                # [GAP F] Métricas de aprendizaje continuo (solo en path LLM — Smart Shuffle no aplica)
                try:
                    learning_metrics = _calculate_learning_metrics(
                        new_days=new_days,
                        prior_meals=prior_meals,
                        rejected_names=rejected_meal_names,
                        allergy_keywords=_allergy_keywords,
                        fatigued_ingredients=_fatigued_ingredients,
                    )
                    repeat_pct = learning_metrics["learning_repeat_pct"]
                    rej_viol = learning_metrics["rejection_violations"]
                    alg_viol = learning_metrics["allergy_violations"]

                    # Log estructurado por chunk (observable con grep [GAP F/MEASURE])
                    logger.info(
                        f"[GAP F/MEASURE] plan={meal_plan_id} chunk={week_number} "
                        f"repeat_pct={repeat_pct}% rejection_violations={rej_viol} allergy_violations={alg_viol} "
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

            # [GAP 2 - Race condition en merge atomico de chunks (P0)]
            # Usar FOR UPDATE real en la fila antes de mergear JSONB en Python para evitar 
            # solapamiento o perdida de dias cuando multiples chunks terminan simultaneamente.
            from db_core import connection_pool
            from psycopg.rows import dict_row

            update_result = None
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

                        # 2. Mergear sin duplicados (usamos el 'day' absoluto como llave)
                        # [GAP E] Detectar y loguear explícitamente:
                        #   - duplicados de day_num dentro de existing_days (corrupción previa)
                        #   - colisiones entre existing_days y new_days (overwrite accidental)
                        days_dict = {}
                        existing_day_nums_seen = set()
                        duplicates_in_existing = []
                        idx = 1
                        for d in existing_days:
                            if isinstance(d, dict):
                                day_num = d.get('day', idx)
                                if day_num in existing_day_nums_seen:
                                    duplicates_in_existing.append(day_num)
                                existing_day_nums_seen.add(day_num)
                                days_dict[day_num] = d
                            idx += 1

                        if duplicates_in_existing:
                            logger.error(
                                f"[GAP E] Plan {meal_plan_id} tenía días duplicados en storage "
                                f"({duplicates_in_existing}). Deduplicando (última ocurrencia gana)."
                            )

                        overwrites = []
                        for d in new_days:
                            if isinstance(d, dict):
                                day_num = d.get('day')
                                if day_num is not None:
                                    if day_num in days_dict:
                                        # Colisión: estamos sobrescribiendo un día ya generado.
                                        # Esto NO debería pasar tras GAP3 validación pre-merge,
                                        # pero si pasa, loguéalo como error.
                                        overwrites.append(day_num)
                                    days_dict[day_num] = d

                        if overwrites:
                            logger.error(
                                f"[GAP E] Chunk {week_number} del plan {meal_plan_id} sobrescribió "
                                f"días ya generados {overwrites}. Investigar: posible race o renumeración rota."
                            )

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

                        plan_data['days'] = merged_days
                        new_total = len(merged_days)
                        plan_data['total_days_generated'] = new_total

                        # Rolling refills siempre marcan 'complete': la ventana de 3 días está llena.
                        # Planes normales usan total_requested para determinar si faltan más chunks.
                        if is_rolling_refill:
                            new_status = "complete"
                        else:
                            new_status = "complete" if new_total >= total_requested else "partial"
                        plan_data['generation_status'] = new_status
                        
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
            if not update_result:
                raise Exception(f"Plan {meal_plan_id} no encontrado en UPDATE atomico")

            new_total = update_result[0].get("new_total", 0)
            new_status = update_result[0].get("new_status", "partial")
            full_plan_data = update_result[0].get("full_plan_data", {})

            # [GAP 2 FIX]: Recalcular lista de compras CON RETRY + ROLLBACK del merge si falla
            # Antes: solo logger.warning si fallaba -> plan quedaba con dias nuevos + shopping list vieja.
            # Ahora: retry 3x con backoff. Si falla -> rollback del merge + re-encolar chunk.
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

            # [GAP F FIX: Si la shopping list fallo, no descartar la valiosa generacion del LLM]
            # Se marca como partial_no_shopping para que el worker de recuperacion intente de nuevo.
            if not shopping_list_ok:
                logger.error(f" [CHUNK/GAP F] Shopping list falló para plan {meal_plan_id}. "
                             f"Marcando plan como 'partial_no_shopping' en lugar de revertir generación LLM.")
                try:
                    execute_sql_write("""
                        UPDATE meal_plans 
                        SET plan_data = jsonb_set(plan_data, '{generation_status}', '"partial_no_shopping"'), 
                            updated_at = NOW() 
                        WHERE id = %s
                    """, (meal_plan_id,))
                except Exception as status_err:
                    logger.error(f" [CHUNK/GAP F] Error seteando partial_no_shopping: {status_err}")
                
                # Dejamos que continue y marque el chunk como completado, 
                # porque los días en sí sí se generaron correctamente.

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
                )
            except Exception as _mt_e:
                logger.warning(f"[GAP G] Error en registro métrica chunk fallido: {_mt_e}")
            try:
                # [GAP B] Reintento agresivo (30 min fijo) si el chunk ya está atrasado >24h o fue escalado.
                # Si no, mantener backoff exponencial original (2^n * 2 - 1 min: 2, 8, 32, 128, 512).
                # Esto evita esperar horas en chunks críticos cuando el plan se está consumiendo.
                is_critical = lag_seconds > 86400 or task.get("escalated_at") is not None
                if is_critical:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= 5 THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + INTERVAL '30 minutes',
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (task_id,), returning=True)
                    logger.warning(f"[GAP B] Chunk {week_number} crítico (lag={lag_seconds//3600}h): retry en 30min en vez de backoff exponencial.")
                else:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= 5 THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + make_interval(mins => POWER(2, (COALESCE(attempts, 0) + 1) * 2 - 1)::int),
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (task_id,), returning=True)
                
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