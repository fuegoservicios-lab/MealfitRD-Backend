import os
import logging
import json
import time
from typing import Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
logger = logging.getLogger(__name__)

from db import (
    get_user_profile, update_user_health_profile, delete_user_facts_by_metadata,
    get_user_likes, get_active_rejections, get_latest_meal_plan_with_id, 
    update_meal_plan_data, add_custom_shopping_items, search_deep_memory as db_search_deep_memory,
    log_consumed_meal as db_log_consumed_meal
)
from schemas import MealModel
from prompts import PREFERENCES_AGENT_PROMPT, MODIFY_MEAL_PROMPT_TEMPLATE

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
        import re
        extracted = re.sub(r'[^\d.]', '', str(new_value))
        if extracted:
            new_value = extracted
            
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
                delete_user_facts_by_metadata(user_id, {"categoria": cat})
            # ---------------------------------------------
            
    return f"¡Éxito! El campo '{field}' ha sido actualizado a '{new_value}'."

# ============================================================
# TOOL: Generar nuevo plan desde el Chat
# ============================================================

def execute_generate_new_plan(user_id: str, form_data: dict, instructions: str = "") -> str:
    logger.info(f"\n🚀 [TOOL] Generando plan nuevo desde el chat para user_id: {user_id}")
    if instructions:
        logger.info(f"📝 [TOOL] Instrucciones específicas del usuario: {instructions}")
    
    from graph_orchestrator import run_plan_pipeline
    
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
                from supabase import create_client
                supabase_url = os.environ.get("SUPABASE_URL")
                
                # Intentamos usar la KEY de servicio si existe para saltar el RLS, o la anónima
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
                
                if supabase_url and supabase_key:
                    from datetime import datetime
                    sb = create_client(supabase_url, supabase_key)
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
                    
                    try:
                        sb.table("meal_plans").insert(insert_data).execute()
                    except Exception as try_db_e:
                        err_msg = str(try_db_e)
                        if "meal_names" in err_msg or "PGRST205" in err_msg or "Could not find" in err_msg:
                            logger.warning("⚠️ [DB] Faltan columnas en DB (meal_names). Guardando sin optimización.")
                            del insert_data["meal_names"]
                            del insert_data["ingredients"]
                            sb.table("meal_plans").insert(insert_data).execute()
                        else:
                            raise try_db_e
                    logger.info("💾 Plan generado desde chat guardado en DB.")
                    
                    # 📈 Frequency Tracking para plan generado por chat
                    try:
                        from db import increment_ingredient_frequencies
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
def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0) -> str:
    """
    Registra una comida que el usuario afirma haber consumido realmente en su diario de consumo ("fuera del plan").
    Úsala SOLO cuando el usuario confirme que se ha comido lo que le analizaste o subió en la foto, o cuando explícitamente diga que comió algo.
    Incluye carbohidratos y grasas saludables si están disponibles.
    """
    logger.debug(f"🔧 [TOOL EXECUTION] Registrando comida consumida para user {user_id}: {meal_name} ({calories} kcal, {protein}g proteina, {carbs}g carbos, {healthy_fats}g grasas)")
    result = db_log_consumed_meal(user_id, meal_name, calories, protein, carbs, healthy_fats)
    if result is not None:
        return f"¡Éxito! Se ha registrado el consumo de '{meal_name}' ({calories} kcal, {protein}g proteína, {carbs}g carbohidratos, {healthy_fats}g grasas saludables) en tu diario."
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
        from db import get_custom_shopping_items
        existing = get_custom_shopping_items(user_id)
        existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
        if existing_items:
            # Extract food items to constrain the AI
            excluded_cats = ["Suplementos", "Limpieza y Hogar", "Higiene Personal", "Otros"]
            ingredient_names = [item.get("name") for item in existing_items if item.get("category") not in excluded_cats]
            if ingredient_names:
                shopping_constraint = f"\n\n⚠️ RESTRICCIÓN DE INGREDIENTES (CRÍTICO): El usuario ya hizo sus compras del supermercado para este ciclo. DEBES crear la nueva receta utilizando EXCLUSIVAMENTE (o en su inmensa mayoría) los siguientes ingredientes que ya tiene disponibles: {', '.join(ingredient_names)}. NUNCA inventes ingredientes principales (ej: proteínas o víveres) que no estén en esta lista."
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

# Mapping local de keywords → (categoría, emoji) para categorización inteligente sin LLM
# ⚠️ Las keywords se emparejan con \b (word boundary), NO con substring.
#     Esto evita falsos positivos como "sal" matcheando "salmón".
#     Para frases multi-palabra ("pasta de tomate"), se prueban primero.
_CATEGORY_KEYWORDS = {
    ("Frutas y Verduras", "🥬"): [
        "lechuga", "tomate", "cebolla", "ajo", "pimiento", "zanahoria", "brocoli", "brócoli",
        "espinaca", "pepino", "aguacate", "limon", "limón", "naranja", "manzana", "banana",
        "platano", "plátano", "guineo", "mango", "piña", "papaya", "fresa", "uva", "melon",
        "sandia", "sandía", "cilantro", "perejil", "apio", "repollo", "coliflor", "batata",
        "yuca", "ñame", "tayota", "berro", "remolacha", "berenjena", "calabaza", "mazorca",
        "maiz", "maíz", "kiwi", "cereza", "arandano", "arándano", "mandarina", "guayaba",
        "chinola", "lechosa", "vaina",
        # Multi-palabra (se prueban primero por ser más específicos)
        "platano verde", "plátano verde", "platano maduro", "plátano maduro",
    ],
    ("Proteínas", "🥩"): [
        "pollo", "pechuga", "muslo", "carne", "cerdo", "chuleta", "costilla",
        "salmon", "salmón", "atun", "atún", "pescado", "camaron", "camarón", "camarones",
        "jamon", "jamón", "salami", "salchicha", "tocino", "bacon", "pavo", "cordero",
        "filete", "bistec", "molida", "longaniza", "chorizo", "tilapia", "sardina",
        "pulpo", "calamar", "langosta", "cangrejo",
        # Multi-palabra
        "carne de res", "carne molida",
    ],
    ("Lácteos", "🥛"): [
        "leche", "queso", "yogur", "yogurt", "mantequilla", "nata", "requesón",
        "mozzarella", "parmesano", "ricotta", "cheddar", "suero",
        # Multi-palabra (más específicos, se prueban primero)
        "queso crema", "crema de leche", "crema agria",
    ],
    ("Huevos", "🥚"): [
        "huevo", "huevos",
    ],
    ("Granos y Cereales", "🌾"): [
        "arroz", "avena", "quinoa", "trigo", "cebada", "lenteja", "lentejas",
        "frijol", "frijoles", "garbanzo", "guandule", "guandules",
        "habichuela", "habichuelas", "alubia",
        "pasta", "espagueti", "macarron", "fideos", "cereal", "granola", "harina",
        "tortilla", "arepa", "casabe",
        # Multi-palabra
        "pan integral", "pan de agua",
    ],
    ("Condimentos y Especias", "🧂"): [
        "pimienta", "oregano", "orégano", "comino", "curry", "canela", "paprika",
        "adobo", "sazon", "sazón", "vinagre", "salsa", "ketchup", "mostaza", "mayonesa",
        "soya", "sillao", "miel", "azucar", "azúcar", "sabora",
        # Multi-palabra
        "salsa de tomate", "sal de ajo", "sal marina", "sal rosada",
    ],
    ("Aceites y Grasas", "🫒"): [
        "aceite", "oliva", "girasol", "manteca", "spray", "ghee",
        # Multi-palabra (evita que "coco" matchee "agua de coco")
        "aceite de coco", "aceite vegetal", "aceite de girasol",
    ],
    ("Bebidas", "🥤"): [
        "agua", "jugo", "refresco", "soda", "cafe", "café", "cerveza",
        "vino", "whisky", "gaseosa", "energizante",
        # Multi-palabra
        "agua de coco",
    ],
    ("Snacks y Dulces", "🍪"): [
        "galleta", "chocolate", "dulce", "caramelo", "chicle", "chips", "palomita",
        "nuez", "almendra", "mani", "maní", "semilla", "barra", "peanut",
    ],
    ("Enlatados y Conservas", "🥫"): [
        # Solo multi-palabra para evitar que "lata" matchee "chocolate"
        "enlatado", "conserva",
        "pasta de tomate", "sardinas en lata", "atun en lata", "maiz en lata",
    ],
    ("Panadería", "🍞"): [
        "pan",
    ],
    ("Limpieza y Hogar", "🧹"): [
        "jabon", "jabón", "detergente", "cloro", "desinfectante", "servilleta",
        "esponja", "fabuloso", "suavizante",
        # Multi-palabra
        "papel toalla", "bolsa de basura",
    ],
    ("Higiene Personal", "🧴"): [
        "shampoo", "champú", "desodorante",
        "pasta dental", "crema dental", "cepillo dental",
    ],
}

def _categorize_item(item_name: str) -> tuple:
    """Categoriza un item por keywords locales con word-boundary matching.
    Retorna (categoría, emoji).
    
    Orden de matching (más específico primero):
    1. ALL multi-word keywords across ALL categories (substring match)
    2. ALL single-word keywords across ALL categories (\\b word boundary)
    Esto evita que 'tomate' (Frutas) gane a 'salsa de tomate' (Condimentos)."""
    import unicodedata, re
    if not item_name or not item_name.strip():
        return "Otros", "🛒"
    # Normalizar: minúsculas, sin acentos
    normalized = item_name.lower().strip()
    nfkd = unicodedata.normalize('NFKD', normalized)
    normalized = ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    def _norm_kw(kw):
        return ''.join(c for c in unicodedata.normalize('NFKD', kw.lower()) if not unicodedata.combining(c))
    
    # Fase 1: Multi-word keywords (más específicos, substring match OK)
    for (category, emoji), keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if ' ' not in kw:
                continue
            if _norm_kw(kw) in normalized:
                return category, emoji
    
    # Fase 2: Single-word keywords con word boundary
    for (category, emoji), keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if ' ' in kw:
                continue
            kw_norm = _norm_kw(kw)
            if re.search(r'\b' + re.escape(kw_norm) + r'\b', normalized):
                return category, emoji
    
    return "Otros", "🛒"

@tool
def add_to_shopping_list(user_id: str, items: str) -> str:
    """
    Añade uno o más ingredientes/items a la lista de compras personal del usuario.
    Usa esta herramienta cuando el usuario diga que se quedó sin algo, necesita comprar algo,
    o pida añadir items a su lista de compras.
    Ejemplos: 'Me quedé sin plátanos', 'Añade leche y huevos a mi lista', 'Necesito comprar arroz'.
    
    - items: string con items separados por coma, ej: 'plátanos, huevos, leche'
    """
    logger.debug(f"🔧 [TOOL EXECUTION] Añadiendo items a shopping list del usuario {user_id}: {items}")
    
    import re as _re
    MAX_ITEM_LENGTH = 100
    MAX_ITEMS_PER_CALL = 20
    
    def _sanitize(text: str) -> str:
        """Elimina HTML/JS tags y limita longitud."""
        clean = _re.sub(r'<[^>]+>', '', text).strip()
        clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)  # control chars
        return clean[:MAX_ITEM_LENGTH]
    
    items_list = [_sanitize(item) for item in items.split(",") if item.strip()]
    items_list = [i for i in items_list if i][:MAX_ITEMS_PER_CALL]
    
    if not items_list:
        return "No se proporcionaron items válidos para añadir."
    
    # Normalizar a JSON struct consistente con los items auto-generados (ShoppingItemModel)
    structured_items = []
    for item_name in items_list:
        cat, emoji = _categorize_item(item_name)
        structured_items.append({
            "category": cat,
            "emoji": emoji,
            "name": item_name.capitalize(),
            "qty": ""
        })
    
    result = add_custom_shopping_items(user_id, structured_items, source="chat")
    if result is not None:
        # Auto-deduplicar: si el usuario ya tenía "Leche" y añade "leche", se fusionan
        try:
            from db import deduplicate_shopping_items
            deduplicate_shopping_items(user_id)
        except Exception:
            pass  # No bloquear la respuesta si falla la dedup
        items_formatted = ", ".join(items_list)
        return f"¡Éxito! Se añadieron {len(items_list)} item(s) a tu lista de compras: {items_formatted}."
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

# Lista de tools disponibles para el agente
agent_tools = [update_form_field, generate_new_plan_from_chat, log_consumed_meal, modify_single_meal, add_to_shopping_list, search_deep_memory]
