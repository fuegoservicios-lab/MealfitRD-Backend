import os
import logging
import json
import re
import unicodedata
import time
from typing import Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from constants import normalize_ingredient_for_tracking, strip_accents, validate_ingredients_against_pantry
logger = logging.getLogger(__name__)

from db import (
    get_user_profile, update_user_health_profile, update_user_health_profile_atomic, delete_user_facts_by_metadata,
    get_user_likes, get_active_rejections, get_latest_meal_plan_with_id,
    update_meal_plan_data, search_deep_memory as db_search_deep_memory,
    log_consumed_meal as db_log_consumed_meal,
    save_new_meal_plan_robust, increment_ingredient_frequencies,
    get_latest_meal_plan
)
from schemas import MealModel
from prompts import PREFERENCES_AGENT_PROMPT, MODIFY_MEAL_PROMPT_TEMPLATE
from datetime import datetime
import threading
from graph_orchestrator import run_plan_pipeline

def analyze_preferences_agent(likes: list, history: list, active_rejections: Optional[list] = None):
    """
    Agente #1: Especialista en Preferencias y Perfiles de Gusto.
    Analiza el historial de Me Gusta y Rechazos ACTIVOS (últimos 7 días) 
    y devuelve un perfil conciso de los gustos del usuario.
    """
    logger.info("\n-------------------------------------------------------------")
    logger.info("🧠 [AGENTE DE PREFERENCIAS] Analizando Perfil de Gustos...")
    
    liked_meals = [f"{like.get('meal_name')} ({like.get('meal_type')})" for like in likes] if likes else []
    rejected_meals = active_rejections if active_rejections else []
    
    if rejected_meals:
        logger.info(f"🚫 Rechazos activos (últimos 7 días): {rejected_meals}")
    else:
        logger.info("➡️  No hay rechazos activos en los últimos 7 días.")
                
    if not liked_meals and not rejected_meals:
        logger.info("➡️  No hay datos suficientes para un perfil. Asumiendo gustos estándar.")
        logger.info("-------------------------------------------------------------")
        return ""
        
    prompt = PREFERENCES_AGENT_PROMPT.format(
        liked_meals=json.dumps(liked_meals), 
        rejected_meals=json.dumps(rejected_meals)
    )
    
    pref_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.3, # Baja temperatura para ser analítico
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    start_time = time.time()
    response = pref_llm.invoke(prompt)
    taste_profile = response.content
    
    end_time = time.time()
    logger.info(f"✅ [PERFIL LISTO] Resuelto en {round(end_time - start_time, 2)}s: {taste_profile}")
    logger.info("-------------------------------------------------------------\n")
    
    return f"\n\n--- PERFIL DE GUSTOS DEL USUARIO (OBLIGATORIO RESPETAR) ---\n{taste_profile}\n-----------------------------------------------------------"

# ============================================================
# TOOL: Actualizar Health Profile del usuario
# ============================================================

@tool
def update_form_field(user_id: str, field: str, new_value: str) -> str:
    """
    Actualiza el formulario en tiempo real en la UI del usuario.
    Campos válidos (field) y valores esperados (DEBES usar estos valores exactos en inglés para los selects):
    - 'weight': Ej: "129" (solo números)
    - 'height': Ej: "180" (en cm)
    - 'age': Ej: "30"
    - 'gender': "male" o "female"
    - 'dietType': "balanced", "vegetarian" o "vegan"
    - 'mainGoal': "lose_fat", "gain_muscle", "maintenance", o "performance"
    - 'activityLevel': "sedentary", "light", "moderate", "active", o "athlete"
    - 'budget': "low", "medium", "high", o "unlimited"
    - 'cookingTime': "none", "30min", "1hour", o "plenty"
    - 'allergies', 'medicalConditions', 'dislikes', 'struggles': Listas separadas por coma (Ej: "Lacteos, Gluten")
    """
    # [P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT — LEER ANTES DE MODIFICAR.
    # ────────────────────────────────────────────────────────────────────────
    # `user_id` viene de `tool_args` construido por la LLM. P0-AGENT-1 cerró
    # el IDOR vía prompt injection: `agent.py:execute_tools` force-overridea
    # `tool_args["user_id"] = state["user_id"]` (autenticado) ANTES de invocar
    # esta tool. El path normal (chat-agent → execute_tools → invoke) está
    # protegido. Para llamadores DIRECTOS (tests, scripts, futuros endpoints
    # HTTP que importen esta tool sin pasar por execute_tools):
    #
    #   1. NUNCA confiar en `user_id` pasado por la LLM sin validar contra el
    #      `verified_user_id` autenticado del request.
    #   2. Si añades una mutación SQL nueva acá, filtrar `AND user_id = %s`
    #      explícito (defense-in-depth). `update_user_health_profile_atomic`
    #      ya filtra; `delete_user_facts_by_metadata` filtra por user_id en
    #      su body. NO romper el contrato.
    #   3. Si extiendes para mutar tablas adicionales (`api_usage`,
    #      `consumed_meals`, etc.), aplicar el mismo filtro.
    #
    # Tooltip-anchor: P3-DOC-2-LIVE-TOOL-CONTRACT

    logger.debug(f"🔧 [TOOL EXECUTION] Actualizando form del usuario {user_id}: {field} -> {new_value}")
    
    # Auto-corrección de valores comunes al formato esperado por la UI
    new_value_lower = str(new_value).lower().strip()
    if field == 'dietType':
        if 'vegetariano' in new_value_lower or 'vegetariana' in new_value_lower: new_value = 'vegetarian'
        elif 'vegano' in new_value_lower or 'vegana' in new_value_lower: new_value = 'vegan'
        elif 'balanceado' in new_value_lower or 'balanceada' in new_value_lower: new_value = 'balanced'
    elif field == 'mainGoal':
        if 'perder' in new_value_lower or 'bajar' in new_value_lower or 'grasa' in new_value_lower or 'peso' in new_value_lower: new_value = 'lose_fat'
        elif 'ganar' in new_value_lower or 'musculo' in new_value_lower or 'masa' in new_value_lower: new_value = 'gain_muscle'
        elif 'mantener' in new_value_lower or 'mantenimiento' in new_value_lower: new_value = 'maintenance'
        elif 'rendimiento' in new_value_lower: new_value = 'performance'
    elif field == 'gender':
        if 'hombre' in new_value_lower or 'masculino' in new_value_lower: new_value = 'male'
        elif 'mujer' in new_value_lower or 'femenino' in new_value_lower: new_value = 'female'
        
    if field in ['weight', 'height', 'age']:
        extracted = re.search(r'\d+\.?\d*', str(new_value))
        if extracted:
            new_value = extracted.group()
            
    if user_id and user_id != "guest":
        # [P1-2] Mutator atómico. Antes era `get_user_profile + mutate +
        # update_user_health_profile` no atómico: si el chat-agent y el
        # wizard del frontend tocaban el mismo `user_id` en paralelo (raro
        # pero observado: usuario chatea con el agente mientras la app
        # autocompleta el form en background), el último UPDATE pisaba al
        # otro y se perdía silenciosamente la edición de un field. El
        # mutator solo escribe el field que estamos cambiando; los demás
        # quedan intactos bajo FOR UPDATE.
        if field in ['allergies', 'medicalConditions', 'dislikes', 'struggles']:
            _new_field_value = [item.strip() for item in str(new_value).split(",") if item.strip()]
        else:
            _new_field_value = new_value

        def _field_mutator(_hp):
            _hp[field] = _new_field_value
            return None

        update_user_health_profile_atomic(user_id, _field_mutator)

        # --- NUEVA LÓGICA DE LIMPIEZA DE VECTORES ---
        category_map = {
            'allergies': 'alergia',
            'medicalConditions': 'condicion_medica',
            'dislikes': 'rechazo',
            'dietType': 'dieta',
            'mainGoal': 'objetivo'
        }
        if field in category_map:
            cat = category_map[field]
            logger.info(f"🧹 [CLEANUP] Borrando vectores de categoría '{cat}' para evitar conflictos con el formulario.")
            delete_user_facts_by_metadata(user_id, {"category": cat})
        # ---------------------------------------------


    return f"¡Éxito! El campo '{field}' ha sido actualizado a '{new_value}'."

# ============================================================
# TOOL: Generar nuevo plan desde el Chat
# ============================================================

def execute_generate_new_plan(user_id: str, form_data: dict, instructions: str = "") -> str:
    logger.info(f"\n🚀 [TOOL] Generando plan nuevo desde el chat para user_id: {user_id}")
    if instructions:
        logger.info(f"📝 [TOOL] Instrucciones específicas del usuario: {instructions}")
    
    # 1. Validar y consolidar form_data (Priorizar el del frontend)
    actual_form_data = form_data.copy() if form_data else {}
    
    if not actual_form_data.get("age"):
        profile = get_user_profile(user_id)
        if profile:
            health_prof = profile.get("health_profile") or {}
            for k, v in health_prof.items():
                if k not in actual_form_data:
                    actual_form_data[k] = v
            
    # Extraer la despensa física activa de la BD para forzar Zero-Waste realista
    if user_id and user_id != "guest" and not actual_form_data.get("current_pantry_ingredients"):
        try:
            from db_inventory import get_user_inventory
            actual_form_data["current_pantry_ingredients"] = get_user_inventory(user_id)
        except Exception as e:
            logger.error(f"⚠️ Error intentando extraer despensa para zero-waste nuevo plan: {e}")
            
    if not actual_form_data or not actual_form_data.get("age"):
        return "ERROR: No se encontraron datos de salud. Completa el formulario de evaluación primero."
    
    actual_form_data["user_id"] = user_id
    
    # 2. Obtener preferencias de gustos
    likes = get_user_likes(user_id)
    active_rejections = get_active_rejections(user_id=user_id)
    rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
    taste_profile = analyze_preferences_agent(likes, [], active_rejections=rejected_meal_names)
    
    # Construir el contexto del usuario basado en sus peticiones del chat
    custom_memory_context = ""
    if instructions:
        custom_memory_context += f"\n\n--- INSTRUCCIÓN ESPECÍFICA DEL USUARIO PARA ESTE PLAN ---\nEL USUARIO PIDIÓ ESTO EXPLÍCITAMENTE, CUMPLE SU PETICIÓN A TODA COSTA:\n'{instructions}'\n------------------------------------------------------------\n"


    # 3. Ejecutar el pipeline multi-agente
    try:
        result = run_plan_pipeline(actual_form_data, [], taste_profile, memory_context=custom_memory_context)
        if result:
            # Intentar guardar en la tabla meal_plans (no crítico)
            try:
                calories = result.get("calories", 0)
                macros = result.get("macros", {})
                
                meal_names = []
                ingredients = []
                for d in result.get("days", []):
                    for m in d.get("meals", []):
                        if m.get("name"):
                            meal_names.append(m.get("name"))
                        if m.get("ingredients"):
                            ingredients.extend(m.get("ingredients"))
                            
                # Extraer _profile_embedding
                profile_embedding = result.pop("_profile_embedding", None)

                insert_data = {
                    "user_id": user_id,
                    "plan_data": result,
                    "calories": int(calories) if calories else 0,
                    "macros": macros,
                    "meal_names": meal_names,
                    "ingredients": ingredients
                }
                
                if profile_embedding:
                    insert_data["profile_embedding"] = profile_embedding
                
                # Nombre inteligente: primeras 2 comidas + calorías
                if len(meal_names) >= 2:
                    n1 = meal_names[0][:30] if len(meal_names[0]) > 30 else meal_names[0]
                    n2 = meal_names[1][:30] if len(meal_names[1]) > 30 else meal_names[1]
                    insert_data["name"] = f"{n1} · {n2} — {calories} kcal"
                elif meal_names:
                    insert_data["name"] = f"{meal_names[0][:30]} — {calories} kcal"
                else:
                    insert_data["name"] = f"Plan {calories} kcal — {datetime.now().strftime('%d/%m/%Y')}"
                
                salvo_ok = save_new_meal_plan_robust(insert_data)
                
                if salvo_ok:
                    logger.info("💾 Plan generado desde chat guardado en DB.")
                else:
                    logger.warning("⚠️ No se pudo guardar el plan generado desde chat.")
                
                # 📈 Frequency Tracking para plan generado por chat (siempre, independiente del guardado)
                try:
                    raw_ingredients = []
                    for d in result.get("days", []):
                        for m in d.get("meals", []):
                            raw_ingredients.extend(m.get("ingredients", []))
                    if raw_ingredients:
                        increment_ingredient_frequencies(user_id, raw_ingredients)
                        logger.info(f"📈 [FREQ TRACKING] Frecuencias de chat actualizadas para {user_id}")
                except Exception as freq_e:
                    logger.error(f"⚠️ [FREQ TRACKING ERROR] Error en tools: {freq_e}")

            except Exception as db_e:
                logger.error(f"⚠️ Aviso: No se pudo guardar el plan en Supabase (error {db_e}), pero el plan se devolverá al usuario.")
            
            return json.dumps(result)
        else:
            return "ERROR: El pipeline no pudo generar un plan. Intenta de nuevo."
    except Exception as e:
        logger.error(f"❌ Error generando plan desde chat: {e}")
        return f"ERROR: {str(e)}"

@tool
def generate_new_plan_from_chat(user_id: str, instructions: str = "") -> str:
    """
    Genera un plan alimenticio completamente nuevo ejecutando el pipeline multi-agente completo (LangGraph).
    Usa esta herramienta SOLO cuando el usuario pida explícitamente un plan nuevo desde el chat.
    Ejemplos: 'Hazme un plan nuevo', 'Genera mi rutina', 'Quiero un menú diferente', 'Cambia todo el plan'.
    El parámetro 'instructions' DEBE contener las peticiones específicas del usuario para el nuevo plan (ej: 'dame alimentos diferentes al plan anterior', 'quiero más atún', 'incluye mi receta favorita').
    NO la uses si el usuario solo pregunta sobre su plan actual o da información de salud.
    """
    return "DUMMY_CALL_ACTUALLY_INTERCEPTED"

@tool
def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0, ingredients: list[str] = None) -> str:
    """
    Registra una comida que el usuario afirma haber consumido realmente en su diario de consumo ("fuera del plan").
    Úsala SOLO cuando el usuario confirme que se ha comido lo que le analizaste o subió en la foto, o cuando explícitamente diga que comió algo.
    Incluye carbohidratos y grasas saludables si están disponibles.
    NUEVO IMPORTANTE: Si sabes o puedes inferir los ingredientes exactos (ej. ["2 huevos", "1 pan", "100g queso"]), envíalos en la lista 'ingredients' para un registro más detallado.
    """
    # [P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT — LEER ANTES DE MODIFICAR.
    # ────────────────────────────────────────────────────────────────────────
    # `user_id` viene de `tool_args` construido por la LLM. P0-AGENT-1 cerró
    # el IDOR: `agent.py:execute_tools` force-overridea `tool_args["user_id"]`
    # al `state["user_id"]` autenticado ANTES de invocar. Path normal seguro.
    # Para llamadores DIRECTOS (tests, scripts, endpoints HTTP futuros):
    #
    #   1. NUNCA confiar en `user_id` LLM-supplied sin validar contra el
    #      `verified_user_id` autenticado del request.
    #   2. `db_log_consumed_meal` y `deduct_consumed_meal_from_inventory`
    #      DEBEN seguir filtrando por `user_id` en sus respectivas SQLs
    #      (defense-in-depth). Si refactorizas estos helpers, NO eliminar
    #      el filtro `WHERE user_id = %s`.
    #   3. Si añades persistencia adicional (e.g., notificación push,
    #      analítica), aplicar mismo filtro.
    #
    # Tooltip-anchor: P3-DOC-2-LIVE-TOOL-CONTRACT

    logger.debug(f"🔧 [TOOL EXECUTION] Registrando comida consumida para user {user_id}: {meal_name} ({calories} kcal, {protein}g proteina, {carbs}g carbos, {healthy_fats}g grasas). Ingredientes a deducir: {ingredients}")
    # [P0.1] Marcar inventory_synced_at en la fila de consumed_meals si vamos a deducir
    # acto seguido — así la reconciliación al cierre del chunk no vuelve a descontar.
    has_ingredients = bool(ingredients)
    result = db_log_consumed_meal(
        user_id, meal_name, calories, protein, carbs, healthy_fats, ingredients,
        mark_inventory_synced=has_ingredients,
    )
    import db_inventory
    if has_ingredients:
        db_inventory.deduct_consumed_meal_from_inventory(user_id, ingredients)
        
    if result is not None:
        msg = f"¡Éxito! Se ha registrado el consumo de '{meal_name}' ({calories} kcal, {protein}g proteína, {carbs}g carbohidratos, {healthy_fats}g grasas saludables) en tu diario."
        return msg
    else:
        return "Hubo un error al intentar registrar la comida consumida. Por favor, intenta de nuevo."

# ============================================================
# TOOL: Modificar una comida individual del plan activo
# ============================================================

def execute_modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str, form_data: dict = None, allow_pantry_expansion: bool = False) -> str:
    """Ejecuta la modificación de una comida individual en el plan activo del usuario."""
    logger.debug(f"\n🔧 [TOOL] modify_single_meal: Día {day_number}, {meal_type}, cambios: '{changes}'")
    
    # 1. Obtener el plan actual con su ID
    plan_record = get_latest_meal_plan_with_id(user_id)
    if not plan_record:
        return "ERROR: No se encontró un plan activo. Genera un plan primero."
    
    plan_id = plan_record["id"]
    plan_data = plan_record["plan_data"]
    
    if not plan_data or not isinstance(plan_data, dict):
        return "ERROR: El plan guardado está corrupto o vacío."
    
    # 2. Localizar la comida específica
    days = plan_data.get("days", [])
    target_day = None
    target_meal = None
    target_meal_index = None
    
    for day in days:
        if day.get("day") == day_number:
            target_day = day
            break
    
    if not target_day:
        return f"ERROR: No se encontró el día {day_number} en el plan."
    
    meals = target_day.get("meals", [])
    for idx, meal in enumerate(meals):
        if meal.get("meal", "").lower().strip() == meal_type.lower().strip():
            target_meal = meal
            target_meal_index = idx
            break
    
    if target_meal is None:
        return f"ERROR: No se encontró '{meal_type}' en el día {day_number}. Comidas disponibles: {[m.get('meal') for m in meals]}"
    
    # 3. Regenerar solo esa comida con Gemini
    original_cals = target_meal.get("cals", 500)
    
    # Extraer ingredientes de la despensa física + lista de compras (Mejora 1: Virtual Pantry Expandido)
    clean_ingredients = []
    try:
        from db_inventory import get_user_inventory
        physical_inventory = get_user_inventory(user_id)
        if physical_inventory:
            clean_ingredients.extend(physical_inventory)
            
        # Añadir la lista de compras del plan actual (futuras compras)
        if plan_data and "aggregated_shopping_list" in plan_data:
            shopping_list = plan_data.get("aggregated_shopping_list")
            if shopping_list and isinstance(shopping_list, list):
                for item in shopping_list:
                    val = item.get("display_string", str(item)) if isinstance(item, dict) else str(item)
                    if val not in clean_ingredients:
                        clean_ingredients.append(val)
    except Exception as e:
        logger.error(f"⚠️ Error extrayendo Virtual Pantry en modify_meal: {e}")
    
    # Fallback al inventario del fronend si la base de datos no arrojó datos
    if not clean_ingredients and form_data:
        current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
        if current_pantry and isinstance(current_pantry, list):
            clean_ingredients = [item.strip() for item in current_pantry if item and isinstance(item, str) and len(item.strip()) > 2]
            
    context_extras = ""
    if clean_ingredients and not allow_pantry_expansion:
        context_extras = f"\n⚠️ REGLA DE RECICLAJE (ROTACIÓN DE DESPENSA): El usuario solicitó un cambio. DEBES utilizar OBLIGATORIAMENTE ingredientes que ya formen parte de su despensa actual. Ingredientes disponibles: {', '.join(clean_ingredients)}. Tienes permiso creativo para proponer un plato usando solo esta base, sin agregar ingredientes foráneos."
    elif allow_pantry_expansion:
        context_extras = f"\n💡 PERMISO DE EXPANSIÓN DE DESPENSA: El usuario ha autorizado explícitamente agregar ingredientes nuevos que no están en su despensa para este cambio (¡Va de compras!). Siéntete libre de proponer CUALQUIER ingrediente ideal para lograr la mejor comida."
        
    try:
        from shopping_calculator import get_master_ingredients
        master_list = get_master_ingredients()
        prices_context = "\n--- 💰 INTELIGENCIA DE PRECIOS (BUDGET-AWARE) ---\n"
        prices_context += "Costo promedio de los ingredientes (en RD$). Utilízalo si el usuario pide sustituciones más baratas u opciones económicas:\n"
        for m in master_list:
            price_lb = m.get("price_per_lb", 0)
            price_u = m.get("price_per_unit", 0)
            if price_lb: prices_context += f"- {m['name']}: RD${price_lb}/lb\n"
            elif price_u: prices_context += f"- {m['name']}: RD${price_u}/unidad\n"
        prices_context += "----------------------------------------------------------\n"
        context_extras += prices_context
    except Exception as e:
        logger.error(f"Error cargando precios en modify_meal: {e}")
    
    modify_prompt = MODIFY_MEAL_PROMPT_TEMPLATE.format(
        name=target_meal.get('name'),
        desc=target_meal.get('desc'),
        meal=target_meal.get('meal'),
        time=target_meal.get('time'),
        original_cals=original_cals,
        ingredients_json=json.dumps(target_meal.get('ingredients', [])),
        changes=changes,
        context_extras=context_extras
    )
    
    modify_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.1,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(MealModel)
    current_prompt = [modify_prompt]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"🔁 [MODIFY_MEAL RETRY] attempt={retry_state.attempt_number} | "
            f"reason=pantry_guardrail_rejection"
        )
    )
    def invoke_with_retry():
        res = modify_llm.invoke(current_prompt[0])
        
        # Validación post-generación
        if hasattr(res, "ingredients"):
            ingreds = getattr(res, "ingredients")
        elif isinstance(res, dict) and "ingredients" in res:
            ingreds = res["ingredients"]
        else:
            ingreds = []
            
        if clean_ingredients and not allow_pantry_expansion:
            val_result = validate_ingredients_against_pantry(ingreds, clean_ingredients)
            if val_result is not True:
                logger.warning(val_result)
                # Inyectar el feedback específico matematico al LLM para el próximo intento de @retry
                current_prompt[0] = modify_prompt + f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{val_result}\nPor favor revisa el inventario y ajusta la receta para que cumpla estrictamente."
                raise ValueError(val_result)
                
        return res
    
    try:
        try:
            new_meal_response = invoke_with_retry()
        except Exception as e:
            logger.error(f"❌ [TOOL] Fallaron los intentos: {e}. Aplicando Fallback de Seguridad Abortivo.")
            
            # En lugar de corromper la BD con ingredientes aleatorios, 
            # abortamos limpiamente la transacción e informamos al Agente principal.
            return ("FALLO POR INVENTARIO INSUFICIENTE: Después de varios intentos, "
                    "no fue posible hacer este cambio respetando de forma estricta los ingredientes de la despensa. "
                    "Informa al usuario amablemente que el cambio fue revertido porque carece de los ingredientes "
                    "adecuados en su inventario para lo que solicitó, y que el plato original se mantuvo intacto.")
        
        
        if hasattr(new_meal_response, "model_dump"):
            new_meal_data = new_meal_response.model_dump()
        elif isinstance(new_meal_response, dict):
            new_meal_data = new_meal_response
        else:
            return "ERROR: La IA generó una respuesta inválida al modificar la comida."
        
        # 4. Reemplazar la comida en el plan
        # Preservar el momento y la hora originales
        new_meal_data["meal"] = target_meal.get("meal")
        new_meal_data["time"] = target_meal.get("time")
        
        target_day["meals"][target_meal_index] = new_meal_data

        # [P1-AUDIT-1 · 2026-05-15] Inicialización a None para que el callback
        # `_apply_meal_modification` pueda preservar las keys del fresh si la
        # recomputación falla. Si el try-block recompute completa, sobrescribe;
        # si lanza excepción antes de asignarlas, el callback ve None y omite
        # la escritura de aggregated_shopping_list* (preserva fresh).
        aggr_7 = None
        aggr_15_hybrid = None
        aggr_30_hybrid = None
        aggr_list = None
        household = max(1, int(form_data.get("householdSize", 1) or 1) if form_data else 1)

        # 4.5 Recalcular la lista de compras consolidada para mantener coherencia
        try:
            grocery_duration = form_data.get("groceryDuration", "weekly") if form_data else "weekly"
            
            from shopping_calculator import get_shopping_list_delta, fetch_inventory_and_consumed_for_plan
            # [P1-AUDIT-1 · 2026-05-15] `household` ya inicializado arriba del
            # try-block para garantizar que el callback lo vea por closure
            # incluso si el try-block lanza antes de la recomputación.
            # [P1-5] Snapshot atómico de inventario + consumidos para las 3
            # multiplicidades. Sin esto, mutaciones concurrentes a
            # `user_inventory` entre las 3 llamadas producían deltas
            # inconsistentes según `groceryDuration`.
            _inv_s, _cons_s = fetch_inventory_and_consumed_for_plan(user_id, plan_data, False)
            aggr_7 = get_shopping_list_delta(
                user_id, plan_data, structured=True, multiplier=1.0 * household,
                inventory_override=_inv_s, consumed_override=_cons_s,
            )
            aggr_15 = get_shopping_list_delta(
                user_id, plan_data, structured=True, multiplier=2.0 * household,
                inventory_override=_inv_s, consumed_override=_cons_s,
            )
            aggr_30 = get_shopping_list_delta(
                user_id, plan_data, structured=True, multiplier=4.0 * household,
                inventory_override=_inv_s, consumed_override=_cons_s,
            )
            
            # [VISIÓN-C] Híbrido: staples=periodo, perishables=semanal.
            # [RIESGO-1] Cycle lock 7d para perecederos.
            try:
                from shopping_calculator import _build_hybrid_shopping_list as _build_hybrid
                _restocked_at = plan_data.get("restocked_at_iso") if plan_data.get("is_restocked") else None
                _restocked_items = plan_data.get("restocked_items") if isinstance(plan_data.get("restocked_items"), dict) else None
                aggr_15_hybrid = _build_hybrid(aggr_7, aggr_15, restocked_at_iso=_restocked_at, restocked_items=_restocked_items) if aggr_15 else aggr_15
                aggr_30_hybrid = _build_hybrid(aggr_7, aggr_30, restocked_at_iso=_restocked_at, restocked_items=_restocked_items) if aggr_30 else aggr_30
            except Exception as _hyb_e:
                logger.warning(f"[TOOL] _build_hybrid fallo: {_hyb_e}. Usando lista extrapolada.")
                aggr_15_hybrid = aggr_15
                aggr_30_hybrid = aggr_30

            if grocery_duration == "biweekly": aggr_list = aggr_15_hybrid
            elif grocery_duration == "monthly": aggr_list = aggr_30_hybrid
            else: aggr_list = aggr_7

            # [P1-AUDIT-1 · 2026-05-15] Asignaciones `plan_data["aggregated_
            # shopping_list*"] = ...` movidas al callback `_apply_meal_modification`
            # más abajo. La copia local `plan_data` ya NO se persiste tal cual
            # — `update_plan_data_atomic` re-SELECTea plan_data FRESH bajo
            # FOR UPDATE row lock y aplica las 4 keys + meals[idx]=new_meal_data
            # sobre la copia post-merge. Recompute de aggregated_shopping_list*
            # se queda FUERA del lock (mismo trade-off que recalc): si un swap
            # concurrente modificó OTRO día entre el SELECT inicial y el lock,
            # las listas reflejan la versión pre-swap-de-otro-día. Las
            # variables `aggr_7`, `aggr_15_hybrid`, `aggr_30_hybrid`, `aggr_list`
            # quedan accesibles via closure desde el callback.
            logger.info("✅ [TOOL] aggregated_shopping_list (7, 15, 30) recalculada post-modificación con Delta.")
        except Exception as e:
            logger.warning(f"⚠️ [TOOL] No se pudo recalcular aggregated_shopping_list Delta: {e}")
        
        # 5. Actualizar en Supabase atómicamente bajo FOR UPDATE row lock.
        #
        # [P1-AUDIT-1 · 2026-05-15] Migración del helper:
        # `update_meal_plan_data` → `update_plan_data_atomic`. Cierre del
        # follow-up natural documentado en P1-RECALC-LOSTUPDATE (2026-05-14):
        #
        # Pre-fix flow:
        #   t=0  `get_latest_meal_plan_with_id(user_id)` (línea ~352) lee
        #        plan_data sin lock.
        #   t=1  LLM genera new_meal_data (puede tomar 5-30s con retry hasta 3
        #        veces vía `invoke_with_retry`).
        #   t=2  Mutación local: `target_day["meals"][target_meal_index] =
        #        new_meal_data` + recompute aggregated_shopping_list (~100-500ms).
        #   t=3  acquire advisory lock + UPDATE full-overwrite via
        #        `update_meal_plan_data` (P1-NEXT-1).
        #
        # Ventana lost-update entre t=0 y t=3 (puede ser DECENAS DE SEGUNDOS
        # por la llamada LLM en t=1): si un endpoint hermano muta `plan_data`
        # quirúrgico entre nuestro SELECT y nuestro UPDATE, esa mutación se
        # pierde silenciosamente. Es la ventana más larga del sistema junto
        # con JIT week-2 (`proactive_agent`).
        #
        # Fix: `update_plan_data_atomic` re-SELECTea plan_data FRESH bajo
        # FOR UPDATE row lock y aplica el callback. Re-localizamos
        # `target_day` por `day_number` y el meal por `meal_type` dentro del
        # callback contra `plan_data_fresh.days`, así otras mutaciones del
        # mismo `plan_data` (otros días, otras keys top-level) sobreviven.
        #
        # Trade-off: aggregated_shopping_list* se computan FUERA del lock con
        # la copia local de `days` ya con new_meal_data aplicado. Si un swap
        # concurrente mutó OTRO day entre t=0 y el lock, las listas reflejan
        # la versión pre-swap-de-otro-día. Mismo trade-off documentado en
        # P1-RECALC-LOSTUPDATE (recompute dentro del lock extendería ~500ms
        # bajo lock, contendiendo con chunk_worker).
        #
        # Tooltip-anchor: P1-AUDIT-1-MODIFY-MEAL-START |
        # test_p1_audit_1_modify_single_meal_lostupdate
        from db_plans import update_plan_data_atomic

        # [P2-COHERENCE-1 · 2026-05-11] `_agent_divergences` capturadas via
        # closure desde el callback para incluir `_coherence_warnings` en el
        # return JSON. El agente las propaga al chat; el frontend las
        # renderea como toast cuando detecta el campo. Mode warn intencional
        # — el agente ya entregó respuesta al usuario; bloquear+retry es caro
        # en tokens.
        _agent_divergences: list = []

        def _apply_meal_modification(plan_data_fresh: dict):
            """Aplica la mutación del meal y las aggregated_shopping_list*
            sobre `plan_data_fresh` (copia fresh re-SELECTada bajo FOR UPDATE
            row lock). Re-localiza `target_day` por `day_number` y el meal
            por `meal_type` (case-insensitive). Si no se encuentran, aborta
            UPDATE retornando `False` (P0-2 contract).
            """
            if not isinstance(plan_data_fresh, dict):
                return False
            days_fresh = plan_data_fresh.get("days") or []
            if not isinstance(days_fresh, list):
                return False

            target_day_fresh = None
            for d in days_fresh:
                if isinstance(d, dict) and d.get("day") == day_number:
                    target_day_fresh = d
                    break
            if not target_day_fresh:
                logger.warning(
                    f"[P1-AUDIT-1/TOOL] day_number={day_number} no encontrado "
                    f"en plan_data_fresh.days tras lock. Plan mutado por "
                    f"hermano: aborta UPDATE."
                )
                return False

            meals_fresh = target_day_fresh.get("meals") or []
            if not isinstance(meals_fresh, list):
                return False

            # Re-localizar el meal por meal_type (case-insensitive). El
            # target_meal_index original puede estar stale si un hermano
            # reordenó meals dentro del día.
            target_idx_fresh = None
            for idx, m in enumerate(meals_fresh):
                if isinstance(m, dict) and m.get("meal", "").lower().strip() == meal_type.lower().strip():
                    target_idx_fresh = idx
                    break
            if target_idx_fresh is None:
                logger.warning(
                    f"[P1-AUDIT-1/TOOL] meal_type={meal_type!r} no encontrado "
                    f"en day {day_number} tras lock. Aborta UPDATE."
                )
                return False

            meals_fresh[target_idx_fresh] = new_meal_data

            # Aggregated lists (overwrite — el agent_tool es source-of-truth
            # de estas keys tras una modificación). Solo escribimos si la
            # recomputación FUERA del lock tuvo éxito; si falló (variables
            # quedaron en None por el except), preservamos las del fresh.
            if aggr_list is not None:
                plan_data_fresh["aggregated_shopping_list"] = aggr_list
            if aggr_7 is not None:
                plan_data_fresh["aggregated_shopping_list_weekly"] = aggr_7
            if aggr_15_hybrid is not None:
                plan_data_fresh["aggregated_shopping_list_biweekly"] = aggr_15_hybrid
            if aggr_30_hybrid is not None:
                plan_data_fresh["aggregated_shopping_list_monthly"] = aggr_30_hybrid

            # [P1-NEXT-2 · 2026-05-11] Coherence guard tras recompute por agent.
            # `action_taken="warn_only_agent_tool"` distingue origen post-mortem.
            # Mode warn intencional — el agente ya entregó respuesta al usuario.
            # Telemetría a `_shopping_coherence_block_history` (mutado in-place
            # por el helper sobre plan_data_fresh) permite post-mortem si el
            # agente produce drift recurrente.
            try:
                from shopping_calculator import run_shopping_coherence_guard_and_append_history as _coh_agent
                _divs, _ = _coh_agent(
                    plan_data_fresh,
                    multiplier=plan_data_fresh.get("calc_household_multiplier") or household,
                    mode_override="warn",
                    attempt=1,
                    action_taken="warn_only_agent_tool",
                    plan_id_hint=plan_id,
                )
                _agent_divergences.extend(_divs or [])
            except Exception as _coh_agent_e:
                logger.warning(f"[TOOL] coherence guard helper fallo (no aborta): {_coh_agent_e}")

            return plan_data_fresh

        # [P0-AGENT-1] user_id ya force-overrideado upstream por
        # `execute_tools` con `_trusted_uid` (defense-in-depth contra prompt
        # injection que emite user_id ajeno).
        # [P2-OPEN-1] plan_id resolved vía `get_latest_meal_plan_with_id(
        # user_id)` que filtra por user_id. Pasamos user_id al helper para
        # SELECT/UPDATE con `AND user_id = %s` defense-in-depth.
        merged_plan_data = update_plan_data_atomic(
            plan_id, _apply_meal_modification, user_id=user_id
        )
        if merged_plan_data:
            logger.info(f"[TOOL] Comida modificada exitosamente: '{new_meal_data.get('name')}'")
            # [P2-COHERENCE-1 · 2026-05-11] _coherence_warnings opcional —
            # presente solo si el guard reportó divergencias post-modificación.
            # El frontend (AgentPage) puede mostrar toast no-bloqueante.
            _resp = {"modified_meal": new_meal_data, "day": day_number, "meal_index": target_meal_index}
            try:
                if _agent_divergences:
                    from shopping_calculator import summarize_divergences_for_ui
                    _resp["_coherence_warnings"] = summarize_divergences_for_ui(_agent_divergences, max_items=5)
            except Exception as _sum_e:
                logger.warning(f"[TOOL/P2-COH-1] summarize_divergences_for_ui falló: {_sum_e}")
            return json.dumps(_resp)
        else:
            return "ERROR: Se generó la nueva comida pero no se pudo guardar en la base de datos."
    except Exception as e:
        logger.error(f"❌ [TOOL] Error modificando comida: {e}")
        return f"ERROR: {str(e)}"

@tool
def modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str, allow_pantry_expansion: bool = False) -> str:
    """
    Modifica UNA comida específica del plan activo del usuario. No genera un plan nuevo, solo cambia la comida indicada.
    Usa esta herramienta cuando el usuario pida un cambio puntual a una comida de su plan.
    IMPORTANTE: Opción A = day_number 1, Opción B = day_number 2, Opción C = day_number 3.

    Parámetros:
    - user_id: ID del usuario
    - day_number: número de la opción (1 para Opción A, 2 para Opción B, 3 para Opción C)
    - meal_type: momento exacto del día: 'Desayuno', 'Almuerzo', 'Merienda' o 'Cena'
    - changes: descripción en lenguaje natural del cambio solicitado por el usuario
    - allow_pantry_expansion: ponlo en True SOLO SI el usuario explícitamente autoriza comprar ingredientes nuevos, ir al súper, o salir de la regla de zero-waste para esta comida. Por defecto es False.

    [P3-NEW-7 · 2026-05-11] CONTRATO DE OWNERSHIP — LEER ANTES DE IMPLEMENTAR.
    ────────────────────────────────────────────────────────────────────────
    Esta tool es DUMMY hoy (retorna `"DUMMY_CALL_ACTUALLY_INTERCEPTED"`).
    El agente LLM la "llama", pero el handler real intercepta en una capa
    superior (`agent.py` `analyze_query` → router). La signature
    `user_id: str` viene del prompt del LLM, NO de un check autenticado.

    RIESGO: si un PR futuro implementa el cuerpo real (ej. delegar al
    endpoint `/swap-meal/persist` o ejecutar UPDATE inline), DEBE:

      1. Validar `user_id == verified_user_id` del contexto autenticado
         (NUNCA confiar en el `user_id` que el LLM le pasó — prompt
         injection puede inyectar un user_id ajeno).
      2. Filtrar `AND user_id = %s` en TODA mutación SQL sobre
         `meal_plans` o `user_inventory` (invariante I2 de CLAUDE.md).
      3. Aplicar `acquire_meal_plan_advisory_lock(purpose='general')` si
         hace full-overwrite de `plan_data` (invariante I7).
      4. NO bypassear `verify_api_quota` — los cambios cuestan tokens
         LLM downstream.

    Sin estos guards, la implementación abre IDOR + lost-update + quota
    burn simultáneamente. El test parser-based
    `test_p3_new_7_modify_single_meal_dummy_contract.py` ancla este
    contrato: detecta si la línea `return "DUMMY_CALL_..."` se reemplaza
    sin que el body nuevo contenga los tokens canónicos de ownership.

    Tooltip-anchor: P3-NEW-7-DUMMY-CONTRACT
    """
    return "DUMMY_CALL_ACTUALLY_INTERCEPTED"



# ============================================================
# TOOL: Buscar en Memoria Profunda (Cold Storage)
# ============================================================

@tool
def search_deep_memory(user_id: str, query: str) -> str:
    """
    Busca en la memoria profunda (archivo frío) los recuerdos históricos del usuario que ya fueron condensados y archivados.
    Usa esta herramienta SOLO cuando el usuario pregunte sobre su pasado lejano, experiencias anteriores con la dieta,
    o datos que no aparecen en la memoria reciente del chat.
    Ejemplos: '¿Recuerdas qué comía al principio?', '¿Qué me costó más en las primeras semanas?',
    '¿Cómo me sentía hace meses?', '¿Cuál era mi rutina anterior?'.
    
    Parámetros:
    - user_id: ID del usuario
    - query: palabra clave o frase corta para buscar en los archivos históricos (ej: 'adherencia', 'estrés', 'primera semana')
    """
    logger.info(f"🔍 [TOOL EXECUTION] Buscando en memoria profunda para user {user_id}: '{query}'")

    # [LONG-TERM-MEMORY-TOGGLE · 2026-05-13] Respetar el toggle del usuario.
    # Si está OFF, la tool NO consulta user_facts (lectura pausada). Los datos
    # quedan en BD, intactos, listos para ser usados de nuevo cuando reactive.
    # Defensive: si el perfil no tiene el campo (legacy), default TRUE.
    try:
        _profile = get_user_profile(user_id)
        if _profile and "long_term_memory_enabled" in _profile:
            if not bool(_profile.get("long_term_memory_enabled", True)):
                logger.info(f"[LONG-TERM-MEMORY-TOGGLE] search_deep_memory pausado por user toggle (user={user_id}).")
                return "La memoria a largo plazo está desactivada en tus ajustes. Actívala desde Settings para acceder a tus recuerdos históricos."
    except Exception:
        pass  # fail-open: si el lookup falla, comportamiento legacy

    results = db_search_deep_memory(user_id, query, limit=5)
    
    if not results:
        return "No se encontraron recuerdos históricos que coincidan con esa búsqueda. Es posible que aún no haya suficiente historial archivado."
    
    # Formatear los resultados para el agente
    formatted = []
    for idx, r in enumerate(results, 1):
        period = f"{r.get('messages_start', '?')} → {r.get('messages_end', '?')}"
        summary = r.get('summary', 'Sin contenido')
        formatted.append(f"📁 Recuerdo #{idx} (Período: {period}):\n{summary}")
    
    return "\n\n".join(formatted)

# ============================================================
# TOOL: Herramienta de Consulta Matemática del Carrito
# ============================================================

@tool
def check_shopping_list(user_id: str) -> str:
    """
    Herramienta de Consulta Matemática del Carrito (EL DELTA DE COMPRAS).
    Usa esta herramienta SIEMPRE que el usuario pregunte qué ingredientes o qué cantidades le FALTAN comprar
    o cuánto necesita de un ingrediente específico para su plan actual (ej: '¿cuántas libras de tomate debo comprar?', 'dame mi lista de compras').
    NUNCA SUMES INGREDIENTES MANUALMENTE MIRANDO EL PLAN, INVOCA ESTA HERRAMIENTA. Hará la suma matemática exacta EXTRAEYENDO tu despensa actual.
    
    Parámetros:
    - user_id: ID del usuario
    """
    logger.info(f"🛒 [TOOL EXECUTION] Calculando lista de compras matemática para user {user_id}")
            
    plan = get_latest_meal_plan(user_id)
    if not plan:
        return "El usuario no tiene un plan de comidas activo estructurado para calcular la lista de compras."
        
    try:
        from shopping_calculator import get_shopping_list_delta
        shop_list = get_shopping_list_delta(user_id, plan, categorize=True, structured=True)
        if not shop_list:
            return "¡Buenas noticias! El usuario tiene todos los ingredientes necesarios en su despensa física para el plan actual. No necesita comprar nada adicional."
        
        formatted_sections = []
        for category, items in shop_list.items():
            icon = "🛒"
            cat_upper = category.upper()
            if "PROTE" in cat_upper or "CARNE" in cat_upper: icon = "🥩"
            elif "FRUTA" in cat_upper or "VEGETAL" in cat_upper: icon = "🥗"
            elif "LÁCTEO" in cat_upper or "LACTEO" in cat_upper: icon = "🥛"
            elif "ESPECIA" in cat_upper: icon = "🧂"
            elif "ESTIMADO" in cat_upper or "💲" in cat_upper: icon = "💸"
            
            # Si el titulo ya trae icono como "💲", no duplicamos
            if icon == "💸":
                formatted_sections.append(f"**{cat_upper}**")
            else:
                formatted_sections.append(f"**{icon} {cat_upper}**")
            for item in items:
                val = item.get("display_string", str(item)) if isinstance(item, dict) else str(item)
                formatted_sections.append(f"- {val}")
            formatted_sections.append("")
            
        formatted_list = "\n".join(formatted_sections).strip()
        return f"RESULTADO MATEMÁTICO DE LA LISTA DE COMPRAS (SOLO LO QUE FALTA COMPRAR):\n{formatted_list}"
    except Exception as e:
        logger.error(f"❌ [TOOL] Error calculando lista de compras: {e}")
        return f"Error interno matemático al calcular la lista de ingredientes: {str(e)}"

@tool
def check_current_pantry(user_id: str) -> str:
    """
    Herramienta de Consulta de Inventario Físico / Despensa (Nevera Digital).
    Usa esta herramienta SIEMPRE que el usuario pregunte "qué me queda en la despensa",
    "qué me sobra", o "qué tengo en la nevera ahora mismo". 
    """
    logger.info(f"🛒 [TOOL EXECUTION] Consultando despensa física BD para user {user_id}")
            
    try:
        from db_inventory import get_user_inventory
        pantry = get_user_inventory(user_id)
        
        if not pantry:
            return "Al parecer el usuario no tiene inventario registrado en su despensa física actualmente."
            
        formatted_list = "\n".join([f"- {item}" for item in pantry])
        res_str = f"RESULTADO DEL INVENTARIO FÍSICO ACTUAL EN LA DESPENSA:\n{formatted_list}"
        return res_str
        
    except Exception as e:
        logger.error(f"❌ [TOOL] Error consultando despensa actual física: {e}")
        return f"Error consultando la base de datos de la despensa: {str(e)}"

@tool
def modify_pantry_inventory(user_id: str, items_to_add: list[str] = None, items_to_remove: list[str] = None) -> str:
    """
    Agrega o elimina ingredientes específicos de la despensa física del usuario (user_inventory).
    Usa esta herramienta cuando el usuario diga que compró algo extra (ej: 'añade 2 manzanas a mi nevera') 
    o que botó/gastó algo (ej: 'se me dañó el arroz, quítalo').
    Los ítems deben venir en formato string con cantidad, unidad e ingrediente (Ej: '2 unidades de Manzana', '500 g de Arroz').
    
    Parámetros:
    - items_to_add: Lista de strings con los ingredientes a sumar a la despensa.
    - items_to_remove: Lista de strings con los ingredientes a restar de la despensa.
    """
    # [P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT — LEER ANTES DE MODIFICAR.
    # ────────────────────────────────────────────────────────────────────────
    # `user_id` viene de `tool_args` construido por la LLM. P0-AGENT-1 cerró
    # el IDOR: `agent.py:execute_tools` force-overridea `tool_args["user_id"]`
    # al `state["user_id"]` autenticado ANTES de invocar. Path normal seguro.
    # Para llamadores DIRECTOS (tests, scripts, endpoints HTTP futuros):
    #
    #   1. NUNCA confiar en `user_id` LLM-supplied sin validar contra el
    #      `verified_user_id` autenticado del request.
    #   2. `add_or_update_inventory_item` y `deduct_consumed_meal_from_inventory`
    #      DEBEN filtrar `WHERE user_id = %s` (defense-in-depth). Tabla
    #      `user_inventory` no tiene RLS forzada para el role `postgres`
    #      (backend conecta como tal); el filtro app-level es la red de
    #      seguridad real.
    #   3. Si añades batch operations (bulk insert/update), conservar el
    #      filtro per-user en cada query.
    #
    # Tooltip-anchor: P3-DOC-2-LIVE-TOOL-CONTRACT

    logger.info(f"🛒 [TOOL EXECUTION] Modificando despensa física manual para user {user_id}")
    try:
        from db_inventory import add_or_update_inventory_item, deduct_consumed_meal_from_inventory
        from shopping_calculator import _parse_quantity
        
        added_count = 0
        removed_count = 0
        
        if items_to_add:
            for item in items_to_add:
                qty, unit, name = _parse_quantity(item)
                if name and qty > 0:
                    add_or_update_inventory_item(user_id, name, qty, unit)
                    added_count += 1
                    
        if items_to_remove:
            deduct_consumed_meal_from_inventory(user_id, items_to_remove)
            removed_count += len(items_to_remove)
            
        return f"¡Despensa actualizada! Se agregaron {added_count} ítems y se redujeron/eliminaron {removed_count} ítems físicamente en la nevera digital."
    except Exception as e:
        logger.error(f"❌ [TOOL] Error modificando despensa manualmente: {e}")
        return f"Error al modificar el inventario físico: {str(e)}"

# ============================================================
# [P3-WATER-TRACKER · 2026-05-16] TOOLS DE HIDRATACION
# ============================================================
# Conectan el card de Hidratacion del Dashboard con el chat agent.
# Antes: tracker aislado, el agente no sabia nada de vasos.
# Despues: el agente puede leer (`check_hydration_today`) y mutar
# (`log_water_glass`) el conteo diario.
#
# Security: ambas tools aceptan `user_id` como primer arg. P0-AGENT-1
# force-overridea ese arg al `_trusted_uid` autenticado en
# `agent.py:execute_tools` ANTES de invocar. Sin esa proteccion seria
# IDOR cross-user (mismo vector que las otras 9 tools).
# ============================================================

def _local_date_str_for_user() -> str:
    """Fecha LOCAL del servidor en formato YYYY-MM-DD. NOTA: el chat agent
    corre server-side; idealmente el cliente pasaria su fecha local, pero
    para v1 usamos la fecha del servidor (Easypanel/DO ambos en UTC-4 o
    similar). Si en el futuro el agente necesita la fecha exacta del
    cliente, parametrizar via state."""
    from datetime import datetime, timezone, timedelta
    # DO es UTC-4. Para no depender del TZ del servidor, calculamos
    # explicitamente la fecha en UTC-4 (Atlantic Standard Time).
    do_now = datetime.now(timezone.utc) - timedelta(hours=4)
    return do_now.date().isoformat()


@tool
def check_hydration_today(user_id: str) -> str:
    """Consulta cuantos vasos de agua ha registrado el usuario HOY y cual es
    su meta diaria personalizada (calculada segun su peso + actividad).

    Usa esta herramienta SIEMPRE que el usuario pregunte sobre su hidratacion
    actual: '¿cuanta agua llevo?', '¿cumpli mi meta?', '¿voy bien con el agua?',
    o cuando necesites contexto para decidir si recordarle tomar agua.

    Devuelve un string con: vasos consumidos hoy, meta diaria, porcentaje
    cumplido, y si la meta esta personalizada (peso del usuario) o es default.
    """
    logger.info(f"💧 [TOOL EXECUTION] check_hydration_today para user {user_id}")
    try:
        from db import supabase
        if not supabase:
            return "No puedo consultar la hidratacion ahora mismo, la base de datos no esta disponible."

        log_date = _local_date_str_for_user()

        # Lee el conteo del dia.
        res = (
            supabase.table("water_intake_log")
            .select("glasses, log_date")
            .eq("user_id", user_id)
            .eq("log_date", log_date)
            .limit(1)
            .execute()
        )
        glasses = 0
        if res and res.data and len(res.data) > 0:
            glasses = int(res.data[0].get("glasses") or 0)

        # Reusa la formula personalizada del endpoint /water-intake.
        # Import lazy para evitar circular (routers/plans.py importa de aqui en futuro).
        from routers.plans import _compute_water_goal
        meta = _compute_water_goal(user_id)
        goal = meta["goal"]
        weight_kg = meta.get("weight_kg")
        is_personalized = not meta.get("default", True) and weight_kg is not None

        pct = round((glasses / goal) * 100) if goal else 0
        remaining = max(0, goal - glasses)

        msg_parts = [f"Hidratacion HOY ({log_date}): {glasses} de {goal} vasos ({pct}%)."]
        if remaining > 0:
            msg_parts.append(f"Le faltan {remaining} para cumplir su meta.")
        else:
            msg_parts.append(f"Ya cumplio su meta del dia.")
        if is_personalized:
            w_str = str(int(weight_kg)) if float(weight_kg).is_integer() else f"{weight_kg:.1f}"
            msg_parts.append(f"Meta personalizada (basada en su peso de {w_str} kg).")
        else:
            msg_parts.append("Meta default (el usuario no tiene peso registrado o el sistema usa fallback).")
        return " ".join(msg_parts)
    except Exception as e:
        logger.error(f"❌ [TOOL] check_hydration_today error: {e}")
        return f"Error consultando hidratacion: {str(e)}"


@tool
def log_water_glass(user_id: str, count_delta: int = 1) -> str:
    """Suma o resta vasos de agua al registro de hidratacion del usuario para HOY.

    Usa esta herramienta cuando el usuario diga que se tomo agua o que se equivoco:
    - 'me tome un vaso' / 'marca uno mas' → count_delta=1 (default)
    - 'me tome 3 vasos seguidos' → count_delta=3
    - 'borra el ultimo / me equivoque' → count_delta=-1
    - 'resetea el dia' → primero check_hydration_today para saber el conteo
      actual N, luego log_water_glass(count_delta=-N).

    Para SETEAR un valor absoluto (ej: el usuario dice 'llevo 5 vasos'),
    primero usa `check_hydration_today` para saber el conteo actual, calcula
    el delta y pasa ese valor (5 - actual).

    El conteo total queda clamped a [0, 50] (cap defensivo de la tabla).
    Tras la mutacion, incluye SIEMPRE la etiqueta `[UI_ACTION: REFRESH_HYDRATION]`
    en tu respuesta para que el Dashboard recargue el card de Hidratacion.
    """
    logger.info(f"💧 [TOOL EXECUTION] log_water_glass user={user_id} delta={count_delta}")
    if not isinstance(count_delta, int) or isinstance(count_delta, bool):
        return "Error: el delta de vasos debe ser un entero (positivo para sumar, negativo para restar)."
    if count_delta == 0:
        return "El delta fue 0 — no se modifico nada."
    try:
        from db import supabase
        from datetime import datetime, timezone
        if not supabase:
            return "No puedo modificar la hidratacion ahora mismo, la base de datos no esta disponible."

        log_date = _local_date_str_for_user()

        # Lee el conteo actual.
        res = (
            supabase.table("water_intake_log")
            .select("glasses")
            .eq("user_id", user_id)
            .eq("log_date", log_date)
            .limit(1)
            .execute()
        )
        current = 0
        if res and res.data and len(res.data) > 0:
            current = int(res.data[0].get("glasses") or 0)

        new_count = max(0, min(50, current + count_delta))
        actual_delta = new_count - current
        if actual_delta == 0 and count_delta != 0:
            # Se intentó sumar/restar pero el clamp absorbió el cambio.
            if count_delta > 0:
                return f"El usuario ya esta en el maximo de 50 vasos para hoy (cap defensivo). No se sumo nada."
            else:
                return f"El usuario ya esta en 0 vasos para hoy. No se puede restar mas."

        payload = {
            "user_id": user_id,
            "log_date": log_date,
            "glasses": new_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (
            supabase.table("water_intake_log")
            .upsert(payload, on_conflict="user_id,log_date")
            .execute()
        )

        # Reusa la meta personalizada para el mensaje de confirmacion.
        try:
            from routers.plans import _compute_water_goal
            goal = _compute_water_goal(user_id).get("goal", 8)
        except Exception:
            goal = 8

        verb = "sumaron" if actual_delta > 0 else "restaron"
        reached = " ¡Cumplio su meta del dia!" if new_count >= goal and current < goal else ""
        return (
            f"Listo: se {verb} {abs(actual_delta)} vaso(s). "
            f"El usuario ahora lleva {new_count} de {goal} vasos hoy.{reached}"
        )
    except Exception as e:
        logger.error(f"❌ [TOOL] log_water_glass error: {e}")
        return f"Error registrando vaso de agua: {str(e)}"


@tool
def mark_shopping_list_purchased(user_id: str, excluded_items: list[str] = None, modified_items: list[str] = None) -> str:
    """
    Herramienta de Registro de Compras Automático y Parcial Inteligente.
    Usa esta herramienta cuando el usuario indique que FUE AL SUPERMERCADO.
    - excluded_items (Opcional): Lista de nombres de ingredientes que el usuario explícitamente NO compró o no encontró (ej: ["Aguacate", "Atún"]).
    - modified_items (Opcional): Lista de ingredientes que compró con una CANTIDAD diferente a la esperada, o ingredientes EXTRA (ej: ["3 lbs de Pollo", "2 paquetes de Galletas"]).
    """
    # [P3-DOC-2 · 2026-05-11] LIVE-TOOL CONTRACT — LEER ANTES DE MODIFICAR.
    # ────────────────────────────────────────────────────────────────────────
    # `user_id` viene de `tool_args` construido por la LLM. P0-AGENT-1 cerró
    # el IDOR: `agent.py:execute_tools` force-overridea `tool_args["user_id"]`
    # al `state["user_id"]` autenticado ANTES de invocar. Path normal seguro.
    # Para llamadores DIRECTOS (tests, scripts, endpoints HTTP futuros):
    #
    #   1. NUNCA confiar en `user_id` LLM-supplied sin validar contra el
    #      `verified_user_id` autenticado del request.
    #   2. `get_latest_meal_plan(user_id)` filtra por user_id en su query
    #      (db_plans.py). `restock_inventory(user_id, ...)` filtra al
    #      escribir `user_inventory`. Si refactorizas alguno, conservar
    #      el filtro `WHERE user_id = %s`.
    #   3. Esta tool LEE plan_data del usuario para construir `shop_list` —
    #      cualquier cambio que sustituya `get_latest_meal_plan` por un
    #      lookup global (e.g. `meal_plans WHERE id = %s` sin user_id)
    #      abre IDOR de lectura cross-user. NO hacerlo.
    #
    # Tooltip-anchor: P3-DOC-2-LIVE-TOOL-CONTRACT

    logger.info(f"🛒 [TOOL EXECUTION] Registrando compra completa/parcial para user {user_id}")
    
    try:
        from db_inventory import restock_inventory
        from shopping_calculator import get_shopping_list_delta, _parse_quantity
        from constants import strip_accents, normalize_ingredient_for_tracking
        
        plan = get_latest_meal_plan(user_id)
        if not plan:
            return "El usuario no tiene un plan activo para extraer la lista de compras."
            
        shop_list = get_shopping_list_delta(user_id, plan, structured=True)
        if not shop_list and not modified_items:
            return "La lista de compras Delta (ingredientes faltantes) está vacía, no hay nada nuevo que añadir a la despensa."
            
        # Normalizar bases a excluir
        excluded_bases = set()
        if excluded_items:
            for item in excluded_items:
                _, _, name = _parse_quantity(item)
                base = normalize_ingredient_for_tracking(name) or strip_accents(name.lower().strip())
                if base: excluded_bases.add(base)
                
        # Normalizar bases a modificar
        modified_bases = set()
        if modified_items:
            for item in modified_items:
                _, _, name = _parse_quantity(item)
                base = normalize_ingredient_for_tracking(name) or strip_accents(name.lower().strip())
                if base: modified_bases.add(base)
                
        # Filtrar shop_list original (excluyendo o si fue modificado con otra cantidad)
        final_shop_list = []
        for item in shop_list:
            val = item.get("display_string", str(item)) if isinstance(item, dict) else str(item)
            _, _, name = _parse_quantity(val)
            base = normalize_ingredient_for_tracking(name) or strip_accents(name.lower().strip())
            
            if base in excluded_bases or base in modified_bases:
                continue # Fue excluido o lo agregaremos con la nueva cantidad de modified_items
                
            final_shop_list.append(val)
            
        # Agregar los items modificados crudos
        if modified_items:
            final_shop_list.extend(modified_items)
            
        success = restock_inventory(user_id, final_shop_list)
        if success:
            msg = f"¡Felicidades! Se han agregado los {len(final_shop_list)} ingredientes a tu Nevera Virtual."
            if excluded_items:
                msg += f" (Se excluyeron {len(excluded_items)} ítems que indicaste)."
                msg += f"\n\n[ALERTA INTERNA PARA LA IA]: El usuario no pudo comprar: {', '.join(excluded_items)}. "
                msg += "Debes disparar INMEDIATAMENTE una recomendación proactiva en el chat preguntando si quiere que sustituyas los platos de esta semana que requerían esos ingredientes faltantes."
            if modified_items:
                msg += f" (Se modificaron/añadieron {len(modified_items)} ítems)."
            return msg
        else:
            return "Hubo un error al intentar agregar los ingredientes a la despensa."
            
    except Exception as e:
        logger.error(f"❌ [TOOL] Error en mark_shopping_list_purchased: {e}")
        return f"Error interno al realizar el registro de la compra: {str(e)}"

# Lista de tools disponibles para el agente
agent_tools = [update_form_field, generate_new_plan_from_chat, log_consumed_meal, modify_single_meal, search_deep_memory, check_shopping_list, check_current_pantry, modify_pantry_inventory, mark_shopping_list_purchased, check_hydration_today, log_water_glass]
