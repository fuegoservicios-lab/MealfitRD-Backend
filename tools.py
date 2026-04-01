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
logger = logging.getLogger(__name__)

from db import (
    get_user_profile, update_user_health_profile, delete_user_facts_by_metadata,
    get_user_likes, get_active_rejections, get_latest_meal_plan_with_id, 
    update_meal_plan_data, add_custom_shopping_items, search_deep_memory as db_search_deep_memory,
    log_consumed_meal as db_log_consumed_meal, deduct_inventory_items,
    save_new_meal_plan_robust, increment_ingredient_frequencies, get_custom_shopping_items,
    get_latest_meal_plan, deduplicate_shopping_items, delete_custom_shopping_items_batch,
    get_user_shopping_lock
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
        profile = get_user_profile(user_id)
        if profile:
            health_profile = profile.get("health_profile") or {}
            if field in ['allergies', 'medicalConditions', 'dislikes', 'struggles']:
                health_profile[field] = [item.strip() for item in str(new_value).split(",") if item.strip()]
            else:
                health_profile[field] = new_value
            update_user_health_profile(user_id, health_profile)
            
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
    actual_form_data = form_data or {}
    
    if not actual_form_data.get("age"):
        profile = get_user_profile(user_id)
        if profile:
            actual_form_data = profile.get("health_profile") or {}
            
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

    # --- CONSTRICCIÓN DE LISTA DE COMPRAS MOVIDA A GRAPH_ORCHESTRATOR ---
    # La lógica para forzar los ingredientes de la lista de compras del usuario
    # ahora se ejecuta centralizadamente dentro de run_plan_pipeline para cubrir
    # tanto este tool como la generación desde el dashboard.

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
                            
                insert_data = {
                    "user_id": user_id,
                    "plan_data": result,
                    "calories": int(calories) if calories else 0,
                    "macros": macros,
                    "meal_names": meal_names,
                    "ingredients": ingredients
                }
                
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
    NUEVO IMPORTANTE: Si sabes o puedes inferir los ingredientes exactos (ej. ["2 huevos", "1 pan", "100g queso"]), envíalos en la lista 'ingredients'. El sistema deducirá automáticamente estas cantidades de su despensa virtual/lista de compras actualizando su inventario físico en tiempo real.
    """
    logger.debug(f"🔧 [TOOL EXECUTION] Registrando comida consumida para user {user_id}: {meal_name} ({calories} kcal, {protein}g proteina, {carbs}g carbos, {healthy_fats}g grasas). Ingredientes a deducir: {ingredients}")
    result = db_log_consumed_meal(user_id, meal_name, calories, protein, carbs, healthy_fats)
    if result is not None:
        msg = f"¡Éxito! Se ha registrado el consumo de '{meal_name}' ({calories} kcal, {protein}g proteína, {carbs}g carbohidratos, {healthy_fats}g grasas saludables) en tu diario."
        if ingredients:
            deducted = deduct_inventory_items(user_id, ingredients)
            if deducted > 0:
                msg += f" Adicionalmente, se actualizaron y dedujeron {deducted} ingrediente(s) de tu despensa."
        return msg
    else:
        return "Hubo un error al intentar registrar la comida consumida. Por favor, intenta de nuevo."

# ============================================================
# TOOL: Modificar una comida individual del plan activo
# ============================================================

def execute_modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str) -> str:
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
    
    # --- CONSTRICCIÓN DE LISTA DE COMPRAS ---
    shopping_constraint = ""
    try:
        existing = get_custom_shopping_items(user_id)
        existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
        if existing_items:
            # Extract food items to constrain the AI
            excluded_cats = ["Suplementos", "Limpieza y Hogar", "Higiene Personal", "Otros"]
            ingredient_names = []
            for item in existing_items:
                if item.get("category") in excluded_cats:
                    continue
                
                # Solo consideramos ingredientes que ya están marcados como comprados (en despensa)
                if not item.get("is_checked", False):
                    continue
                
                name_val = item.get("display_name")
                if not name_val:
                    i_name = item.get("item_name")
                    if i_name:
                        try:
                            parsed = json.loads(i_name)
                            name_val = parsed.get("name") if isinstance(parsed, dict) else str(i_name)
                        except Exception:
                            name_val = str(i_name)
                if not name_val:
                    name_val = item.get("name")
                    
                if name_val:
                    ingredient_names.append(str(name_val).strip())
            if ingredient_names:
                shopping_constraint = f"\n\n⚠️ REGLA DE SUPERMERCADO ABSOLUTA E INQUEBRANTABLE: El usuario YA COMPRÓ su comida y no puede comprar nada más. TIENES ESTRICTAMENTE PROHIBIDO sugerir frutas, vegetales, carnes, lácteos, cereales o cualquier ingrediente que no esté en esta lista exacta: [{', '.join(ingredient_names)}]. Si la lista incluye tomate y cebolla, usa SOLO tomate y cebolla, no inventes lechuga ni aguacate. ESPECIAL ATENCIÓN: Si no ves pollo, pescado ni carnes en la lista, TIENES PROHIBIDO inventarlos; debes crear un plato vegetariano o basado en los granos/quesos que sí tenga la lista. Si creas una receta, la receta DEBE limitarse al 100% a lo que hay en esta lista o lo que sobró del plato original."
                logger.info("🛒 [CONSTRAINT] Aplicando restricción de lista de compras en regeneración de comida.")
    except Exception as check_e:
        logger.error(f"Error revisando lista de compras para restricción: {check_e}")
    # ----------------------------------------
    
    modify_prompt = MODIFY_MEAL_PROMPT_TEMPLATE.format(
        name=target_meal.get('name'),
        desc=target_meal.get('desc'),
        meal=target_meal.get('meal'),
        time=target_meal.get('time'),
        original_cals=original_cals,
        ingredients_json=json.dumps(target_meal.get('ingredients', [])),
        changes=changes,
        context_extras=shopping_constraint
    )
    
    modify_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.5,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(MealModel)
    
    try:
        new_meal_response = modify_llm.invoke(modify_prompt)
        
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
        
        # 5. Actualizar en Supabase
        update_result = update_meal_plan_data(plan_id, plan_data)
        if update_result is not None:
            logger.info(f"✅ [TOOL] Comida modificada exitosamente: '{new_meal_data.get('name')}'")
            return json.dumps({"modified_meal": new_meal_data, "day": day_number, "meal_index": target_meal_index})
        else:
            return "ERROR: Se generó la nueva comida pero no se pudo guardar en la base de datos."
    except Exception as e:
        logger.error(f"❌ [TOOL] Error modificando comida: {e}")
        return f"ERROR: {str(e)}"

@tool
def modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str) -> str:
    """
    Modifica UNA comida específica del plan activo del usuario. No genera un plan nuevo, solo cambia la comida indicada.
    Usa esta herramienta cuando el usuario pida un cambio puntual a una comida de su plan,
    como 'cámbiale el salami al mangú por huevos en la Opción A', 'quítale el arroz al almuerzo de la Opción B', 'ponle más proteína al desayuno'.
    IMPORTANTE: Opción A = day_number 1, Opción B = day_number 2, Opción C = day_number 3.
    
    Parámetros:
    - user_id: ID del usuario
    - day_number: número de la opción (1 para Opción A, 2 para Opción B, 3 para Opción C)
    - meal_type: momento exacto del día: 'Desayuno', 'Almuerzo', 'Merienda' o 'Cena'
    - changes: descripción en lenguaje natural del cambio solicitado por el usuario
    """
    return "DUMMY_CALL_ACTUALLY_INTERCEPTED"

# ============================================================
# TOOL: Añadir items a la lista de compras
# ============================================================

from constants import SHOPPING_CATEGORIES, parse_ingredient_qty, categorize_shopping_item
@tool
def add_to_shopping_list(user_id: str, items: list[dict], overwrite: bool = False, protected_meals: list[str] = None) -> str:
    """
    Añade uno o más ingredientes/items a la lista de compras personal del usuario o la sobreescribe completamente.
    Usa esta herramienta cuando el usuario diga que se quedó sin algo, necesita comprar algo, o pida añadir items.
    
    REGLA DE SOBREESCRITURA: Si el usuario te envía una lista COMPLETA o foto diciendo "esta es mi lista actual", DEBES establecer overwrite=True para REEMPLAZAR su despensa/lista.
    Si solo dice "añade X" o "me quedé sin Y", usa overwrite=False.
    
    REGLA DE INVENTARIO (is_checked):
    Si el usuario sube una FOTO DE SU DESPENSA/NEVERA o dice "tengo exactamente esto en mi cocina", DEBES incluir `"is_checked": true` en CADA ÍTEM de la lista `items` para registrar que ya los posee en su inventario (WMS).
    Si el usuario dice "añade esto para comprar" o "me falta pan", debes enviar `"is_checked": false` (o simplemente omitirlo).
    Puedes enviar ambos en la misma llamada si es necesario dependiendo del ingrediente.
    
    - items: Lista de objetos JSON. Cada objeto tiene 'name' (nombre), 'qty' (cantidad), 'category' y opcionalmente 'is_checked' (booleano). Ej: [{"name": "Avena", "qty": "1 paquete", "category": "Cereales", "is_checked": true}].
    - overwrite: booleano (True o False).
    - protected_meals: (Opcional) Array de strings con nombres de comidas protegidas si overwrite=True.
    """
    logger.debug(f"🔧 [TOOL EXECUTION] Añadiendo items a shopping list del usuario {user_id} (overwrite={overwrite}, protected_meals={protected_meals}): {items}")
    
    _re = re  # alias local para compatibilidad con el código existente
    MAX_ITEM_LENGTH = 100
    MAX_ITEMS_PER_CALL = 150
    
    # Soporte para legacy (si la IA envía un string roto o array encadenado)
    if isinstance(items, str):
        try:
            items = json.loads(items.replace("'", '"'))
        except Exception:
            items = [{"name": i.strip(), "qty": "", "category": ""} for i in items.split(",") if i.strip()]
    
    if not isinstance(items, list):
        items = [items]
        
    def _sanitize(text: str) -> str:
        """Elimina HTML/JS tags y limita longitud."""
        if not text: return ""
        clean = _re.sub(r'<[^>]+>', '', str(text)).strip()
        clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)  # control chars
        return clean[:MAX_ITEM_LENGTH]
    
    structured_items = []
    items_names = []
    
    was_truncated = len(items) > MAX_ITEMS_PER_CALL
    
    for item in items[:MAX_ITEMS_PER_CALL]:
        is_checked_val = False
        if isinstance(item, dict):
            raw_name = item.get("name", "")
            raw_qty = item.get("qty", "")
            raw_cat = item.get("category", "")
            check_raw = item.get("is_checked", False)
            is_checked_val = (str(check_raw).lower() == 'true') if isinstance(check_raw, str) else bool(check_raw)
        elif isinstance(item, str):
            raw_name = item
            raw_qty = ""
            raw_cat = ""
        else:
            continue
            
        name_clean = _sanitize(raw_name).capitalize()
        qty_clean = _sanitize(raw_qty)
        cat_clean = _sanitize(raw_cat).strip().title()
        
        if not name_clean: continue
        
        # Use global categorizer from constants
        cat, emoji = categorize_shopping_item(name_clean, suggested_category=cat_clean)
                
        structured_items.append({
            "category": cat,
            "emoji": emoji,
            "name": name_clean,
            "qty": qty_clean,
            "is_checked": is_checked_val
        })
        items_names.append(name_clean)
    
    if overwrite:
        try:
            plan_data = get_latest_meal_plan(user_id)
            if plan_data:
                protected_ingredients = set()
                
                if not protected_meals:
                    protected_list = ["almuerzo", "comida", "lunch", "comida principal"]
                else:
                    protected_list = [m.lower().strip() for m in protected_meals]
                
                # Soportar esquema nuevo (V2) donde plan_data tiene "days" (lista)
                if "days" in plan_data and isinstance(plan_data["days"], list):
                    for day_data in plan_data["days"]:
                        if isinstance(day_data, dict):
                            for meal_info in day_data.get("meals", []):
                                meal_type = meal_info.get("meal", "")
                                if isinstance(meal_info, dict) and meal_type and meal_type.lower() in protected_list:
                                    for ing in meal_info.get("ingredients", []):
                                        protected_ingredients.add(ing)
                else:
                    # Fallback para esquema viejo (V1)
                    for day_key, meals in plan_data.items():
                        if isinstance(meals, dict):
                            for meal_name, meal_info in meals.items():
                                if isinstance(meal_info, dict) and meal_name.lower() in protected_list:
                                    for ing in meal_info.get("ingredients", []):
                                        protected_ingredients.add(ing)
                
                existing_names = set(item["name"].lower() for item in structured_items)
                
                for ing in protected_ingredients:
                    num_float, unit, name_raw = parse_ingredient_qty(ing, to_metric=False)
                    num_str = ""
                    if num_float is not None:
                        num_str = str(int(num_float)) if num_float.is_integer() else str(num_float)
                    
                    
                    if num_str or unit:
                        qty = _sanitize(f"{num_str} {unit}".strip())
                        name = _sanitize(name_raw).capitalize()
                    else:
                        qty = ""
                        name = _sanitize(ing).capitalize()
                        
                    if name.lower() not in existing_names:
                        cat, emoji = categorize_shopping_item(name)
                        structured_items.append({
                            "category": cat,
                            "emoji": emoji,
                            "name": name,
                            "qty": qty,
                            "is_checked": False
                        })
                        items_names.append(name)
        except Exception as e:
            logger.error(f"Error extrayendo comidas protegidas de plan para shopping list (overwrite): {e}")

    if not structured_items:
        return "No se proporcionaron items válidos."
    
    lock = get_user_shopping_lock(user_id)
    with lock:
        result = add_custom_shopping_items(user_id, structured_items, source="chat", overwrite=overwrite)
        
    if result is not None:
        # Auto-deduplicar en segundo plano (Fire-and-forget) para no bloquear el Agent
        try:
            threading.Thread(target=deduplicate_shopping_items, args=(user_id,), daemon=True).start()
            
            # Lanzamos deduplicación semántica (LLM) también separada
            from services import async_semantic_deduplication
            async_semantic_deduplication(user_id)
        except Exception:
            pass  # No bloquear la respuesta si falla la dedup
        items_formatted = ", ".join(items_names)
        msg = f"¡Éxito! Se añadieron {len(structured_items)} item(s) a tu lista de compras: {items_formatted}."
        if was_truncated:
            msg += f"\n(Nota: La lista original excedía el límite, por lo que insertamos solo los primeros {MAX_ITEMS_PER_CALL} productos)"
        return msg
    else:
        return "Hubo un error al añadir los items a la lista de compras. Intenta de nuevo."

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
# TOOL: Eliminar de Lista de Compras
# ============================================================

@tool
def remove_from_shopping_list(user_id: str, item_names: list[str]) -> str:
    """
    Elimina uno o más ingredientes específicos de la lista de compras del usuario.
    Usa esta herramienta cuando el usuario pida explícitamente quitar o remover algo de su lista (ej: "quita la manzana").
    No uses overwrite=True en add_to_shopping_list si el usuario solo pidió quitar un par de items.
    
    Parámetros:
    - user_id: ID del usuario
    - item_names: Lista de strings con los nombres a borrar (ej: ["manzana", "leche descremada"]).
    """
    if not item_names or not isinstance(item_names, list):
        return "No se ha especificado qué elementos borrar."
        
    logger.info(f"🗑️ [TOOL] Intentando eliminar {len(item_names)} items de la lista de compras: {item_names}")
    
    try:
        # Obtener los items actuales para buscar coincidencias
        current = get_custom_shopping_items(user_id, limit=500)
        items_db = current.get("data", [])
        
        if not items_db:
            return "La lista de compras ya está vacía."
            
        # re, json y unicodedata ya importados a nivel de módulo
        _json = json  # alias local para compatibilidad

        def _norm(text: str) -> str:
            if not text: return ""
            nfkd = unicodedata.normalize('NFKD', str(text).lower().strip())
            return re.sub(r'[\s+]', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))
            
        targets = [_norm(t) for t in item_names if t]
        if not targets:
            return "Nombres inválidos."
            
        ids_to_del = set()
        removed_names = set()
        
        for row in items_db:
            name = row.get("display_name", "")
            if not name:
                raw = row.get("item_name", "")
                if raw.startswith("{"):
                    try:
                        name = _json.loads(raw).get("name", raw)
                    except:
                        name = raw
                else:
                    name = raw
            
            norm_name = _norm(name)
            
            # Matchear si el target es una palabra completa dentro del nombre en DB o viceversa
            for t in targets:
                # Usa \b con plurales opcionales para que "habichuela" borre "habichuelas"
                if re.search(r'\b' + re.escape(t) + r'(s|es)?\b', norm_name) or re.search(r'\b' + re.escape(norm_name) + r'(s|es)?\b', t):
                    ids_to_del.add(row["id"])
                    removed_names.add(name)
                    break
                    
        if not ids_to_del:
            return f"No encontré ninguno de estos ingredientes ({', '.join(item_names)}) en tu lista activa."
            
        # Ejecutar borrado usando el cliente de db.py (fuente centralizada) dentro de un lock
        lock = get_user_shopping_lock(user_id)
        with lock:
            delete_custom_shopping_items_batch(list(ids_to_del), user_id)
        
        items_formatted = ", ".join(removed_names)
        return f"¡Éxito! Eliminé {len(ids_to_del)} item(s) de tu lista: {items_formatted}."
        
    except Exception as e:
        logger.error(f"Error en remove_from_shopping_list: {e}")
        return f"Hubo un error al intentar borrar los items. Detalle: {e}"

# Lista de tools disponibles para el agente
agent_tools = [update_form_field, generate_new_plan_from_chat, log_consumed_meal, modify_single_meal, add_to_shopping_list, remove_from_shopping_list, search_deep_memory]
