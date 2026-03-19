# backend/agent.py

import os
import time
import json
import re
import unicodedata

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import random
from typing import List, Optional, Annotated, TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential

from nutrition_calculator import get_nutrition_targets
from db import get_user_profile, update_user_health_profile
from dotenv import load_dotenv

load_dotenv()

from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel, ShoppingListModel
from prompts import (
    ANALYZE_SYSTEM_PROMPT, DETERMINISTIC_VARIETY_PROMPT, SWAP_MEAL_PROMPT_TEMPLATE, 
    AUTO_SHOPPING_LIST_PROMPT, TITLE_GENERATION_PROMPT, RAG_ROUTER_PROMPT,
    CHAT_SYSTEM_PROMPT_BASE, CHAT_STREAM_SYSTEM_PROMPT_BASE
)
from tools import (
    update_form_field, generate_new_plan_from_chat,
    log_consumed_meal, modify_single_meal,
    add_to_shopping_list, search_deep_memory, agent_tools, analyze_preferences_agent
)

# Langchain Chat Model Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.2,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)


# ============================================================
# INVERSIÓN DE CONTROL DETERMINISTA (ANTI MODE-COLLAPSE)
# ============================================================
DOMINICAN_PROTEINS = [
    "Pollo", "Cerdo", "Res", "Pescado", "Atún", "Huevos", "Queso de Freír",
    "Salami Dominicano", "Camarones", "Chuleta", "Longaniza", "Berenjena",
    "Habichuelas Rojas", "Habichuelas Negras", "Gandules", "Lentejas", "Garbanzos", "Soya/Tofu"
]

DOMINICAN_CARBS = [
    "Plátano Verde", "Plátano Maduro", "Yuca", "Batata", "Arroz Blanco", 
    "Arroz Integral", "Avena", "Pan Integral", "Papas", "Guineítos Verdes", "Ñame", "Yautía"
]

def get_deterministic_variety_prompt(history_text: str, form_data: dict = None) -> str:
    """Implementa Inversión de Control Determinista para evitar Mode Collapse en el LLM."""
    print("🎲 [ANTI MODE-COLLAPSE] Calculando Matriz de Ingredientes (Round-Robin)...")
    history_lower = history_text.lower() if history_text else ""
    history_normalized = strip_accents(history_lower)
    
    # --- FILTRO DE RESTRICCIONES MÉDICAS Y DIETÉTICAS ---
    filtered_proteins = DOMINICAN_PROTEINS.copy()
    filtered_carbs = DOMINICAN_CARBS.copy()
    
    if form_data:
        allergies = [a.lower() for a in form_data.get("allergies", [])]
        dislikes = [d.lower() for d in form_data.get("dislikes", [])]
        diet = form_data.get("diet", form_data.get("dietType", "")).lower()
        
        restrictions = allergies + dislikes
        
        if diet in ["vegano", "vegan"]:
            restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "huevos", "queso", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco", "lácteo", "leche"])
        elif diet in ["vegetariano", "vegetarian"]:
            restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco"])
        elif diet in ["pescetariano", "pescatarian"]:
            restrictions.extend(["pollo", "cerdo", "res", "salami", "chuleta", "longaniza", "carne"])
            
        def is_allowed(item):
            item_normalized = strip_accents(item.lower())
            for r in restrictions:
                r_normalized = strip_accents(r.lower())
                # Comparación con word-boundary para evitar falsos positivos
                # (ej: restricción "res" NO debe bloquear "camarones")
                if re.search(r'\b' + re.escape(r_normalized) + r'\b', item_normalized):
                    return False
                if r_normalized in ["mariscos", "seafood", "marisco"] and any(
                    re.search(r'\b' + x + r'\b', item_normalized) 
                    for x in ["camaron", "camarones", "pescado", "atun"]
                ):
                    return False
                if r_normalized in ["carne", "carnes", "meat"] and any(
                    re.search(r'\b' + x + r'\b', item_normalized) 
                    for x in ["pollo", "cerdo", "res", "chuleta", "longaniza", "salami"]
                ):
                    return False
            return True
            
        filtered_proteins = [p for p in filtered_proteins if is_allowed(p)]
        filtered_carbs = [c for c in filtered_carbs if is_allowed(c)]
        

    # ----------------------------------------------------
    
    # 1. Analizar qué se ha usado (con Regex para evitar falsos positivos y atrapar sinónimos)
    used_proteins = set()
    used_carbs = set()
    
    # Mapeo de sinónimos comunes para ingredientes
    protein_synonyms = {
        "pollo": ["pollo", "pechuga", "muslo", "alitas", "chicharrón de pollo", "filete de pollo"],
        "cerdo": ["cerdo", "masita", "chicharrón de cerdo", "lomo", "pernil", "costilla"],
        "res": ["res", "carne molida", "bistec", "filete", "churrasco", "vaca", "picadillo", "carne de res"],
        "pescado": ["pescado", "dorado", "chillo", "mero", "salmón", "tilapia", "filete de pescado"],
        "atún": ["atún", "atun"],
        "huevos": ["huevos", "huevo", "tortilla", "revoltillo"],
        "queso de freír": ["queso de freír", "queso de freir", "queso frito", "queso de hoja"],
        "salami dominicano": ["salami dominicano", "salami", "salchichón"],
        "camarones": ["camarones", "camarón", "camaron"],
        "chuleta": ["chuleta", "chuletas", "chuleta frita", "chuleta al horno"],
        "longaniza": ["longaniza", "longanizas"],
        "berenjena": ["berenjena", "berenjenas", "berenjena rellena"],
        "habichuelas rojas": ["habichuelas rojas", "frijoles rojos", "habichuela roja"],
        "habichuelas negras": ["habichuelas negras", "frijoles negros", "habichuela negra"],
        "gandules": ["gandules", "guandules", "gandul", "guandul"],
        "lentejas": ["lentejas", "lenteja"],
        "garbanzos": ["garbanzos", "garbanzo"],
        "soya/tofu": ["soya", "tofu", "carne de soya"]
    }
    
    carb_synonyms = {
        "plátano verde": ["plátano verde", "platano verde", "mangú", "mangu", "tostones", "fritos verdes", "mangú de plátano", "mangu de platano"],
        "plátano maduro": ["plátano maduro", "platano maduro", "maduros", "plátano al caldero", "fritos maduros"],
        "yuca": ["yuca", "casabe", "arepitas de yuca", "puré de yuca"],
        "arroz blanco": ["arroz blanco", "arroz"],
        "arroz integral": ["arroz integral"],
        "avena": ["avena", "avena en hojuelas", "overnight oats"],
        "pan integral": ["pan integral", "pan", "tostada integral", "tostada"],
        "papas": ["papas", "papa", "puré de papas", "papa hervida"],
        "guineítos verdes": ["guineítos", "guineitos", "guineos verdes", "guineito verde", "guineitos verdes"],
        "ñame": ["ñame", "name", "ñame hervido"],
        "yautía": ["yautía", "yautia", "yautía hervida"],
        "batata": ["batata", "puré de batata", "batata hervida", "boniato"]
    }
    
    for p in filtered_proteins:
        syns = protein_synonyms.get(p.lower(), [p.lower()])
        for syn in syns:
            syn_normalized = strip_accents(syn.lower())
            if re.search(r'\b' + re.escape(syn_normalized) + r'\b', history_normalized):
                used_proteins.add(p)
                break
                
    for c in filtered_carbs:
        syns = carb_synonyms.get(c.lower(), [c.lower()])
        for syn in syns:
            syn_normalized = strip_accents(syn.lower())
            if re.search(r'\b' + re.escape(syn_normalized) + r'\b', history_normalized):
                used_carbs.add(c)
                break
                
    used_proteins = list(used_proteins)
    used_carbs = list(used_carbs)
    
    # 2. Filtrar catálogo (si quedan muy pocos, usamos todos revertiendo el filtro)
    available_proteins = [p for p in filtered_proteins if p not in used_proteins]
    if len(available_proteins) < 3:
        available_proteins = filtered_proteins.copy()
        
    available_carbs = [c for c in filtered_carbs if c not in used_carbs]
    if len(available_carbs) < 3:
        available_carbs = filtered_carbs.copy()
        
    # 3. Restricción Dura: Elegir 2 proteínas y 2 carbohidratos (reduciendo costo de supermercado)
    # Seleccionamos solo 2 diferentes para ahorrar dinero, y uno se repetirá en el 3er día.
    num_proteins_to_pick = min(2, len(available_proteins))
    num_carbs_to_pick = min(2, len(available_carbs))
    
    unique_proteins = random.sample(available_proteins, num_proteins_to_pick)
    unique_carbs = random.sample(available_carbs, num_carbs_to_pick)
    
    # Llenamos los 3 días asegurando que al menos aparezcan los 2 elegidos
    chosen_proteins = [unique_proteins[0], unique_proteins[1 if num_proteins_to_pick > 1 else 0], random.choice(unique_proteins)]
    chosen_carbs = [unique_carbs[0], unique_carbs[1 if num_carbs_to_pick > 1 else 0], random.choice(unique_carbs)]
    
    # Mezclamos para que el orden de los días sea dinámico
    random.shuffle(chosen_proteins)
    random.shuffle(chosen_carbs)
    
    blocked_text = ""
    if used_proteins or used_carbs:
        blocked_items = used_proteins + used_carbs
        blocked_text = f"🚫 BLOQUEO MATEMÁTICO: Quedan ESTRICTAMENTE PROHIBIDOS como base principal (porque el usuario ya comió mucho de esto): {', '.join(blocked_items[:6])}."
        
    prompt = DETERMINISTIC_VARIETY_PROMPT.format(
        protein_0=chosen_proteins[0], carb_0=chosen_carbs[0],
        protein_1=chosen_proteins[1], carb_1=chosen_carbs[1],
        protein_2=chosen_proteins[2], carb_2=chosen_carbs[2],
        blocked_text=blocked_text
    )
    print(f"✅ [ANTI MODE-COLLAPSE] Proteínas elegidas (optimizadas para costo): {chosen_proteins}")
    print(f"✅ [ANTI MODE-COLLAPSE] Carbohidratos elegidos (optimizados para costo): {chosen_carbs}")
    return prompt

def analyze_form(form_data: dict, history: Optional[list] = None, taste_profile: str = ""):
    print("\n-------------------------------------------------------------")
    print("⏳[INICIANDO] Generando Plan Alimenticio con Gemini 3.1 Pro (Python LangChain)...")
    goal = form_data.get("mainGoal", form_data.get("goal", "Desconocido"))
    print(f"➡️  Objetivo Principal: {goal}")
    print("-------------------------------------------------------------")
    
    start_time = time.time()
    
    # ========== AGENTE CALCULADOR (Pre-cálculo exacto en Python) ==========
    nutrition = get_nutrition_targets(form_data)
    
    nutrition_context = f"""
--- TARGETS NUTRICIONALES CALCULADOS (Fórmula Mifflin-St Jeor / Harris-Benedict Revisada) ---
⚠️ ESTOS NÚMEROS SON EXACTOS. NO LOS RECALCULES. ÚSALOS TAL CUAL.

• BMR (Tasa Metabólica Basal): {nutrition['bmr']} kcal
• TDEE (Gasto Energético Total Diario): {nutrition['tdee']} kcal  
• 🎯 CALORÍAS OBJETIVO DEL DÍA: {nutrition['target_calories']} kcal ({nutrition['goal_label']})
• Proteína: {nutrition['macros']['protein_g']}g ({nutrition['macros']['protein_str']})
• Carbohidratos: {nutrition['macros']['carbs_g']}g ({nutrition['macros']['carbs_str']})
• Grasas: {nutrition['macros']['fats_g']}g ({nutrition['macros']['fats_str']})

Detalles del cálculo: {nutrition['calculation_details']}

IMPORTANTE: El campo 'calories' del plan DEBE ser {nutrition['target_calories']}.
Los macros del plan DEBEN ser: protein='{nutrition['macros']['protein_str']}', carbs='{nutrition['macros']['carbs_str']}', fats='{nutrition['macros']['fats_str']}'.
La SUMA de las calorías de todas las comidas individuales DEBE ser cercana a {nutrition['target_calories']} kcal.
------------------------------------------------------------------------------------------
"""
    
    # Extraer comidas recientes del historial para dar contexto al modelo (evitar repetición)
    recent_meals_context = ""
    if history and len(history) > 0:
        recent_assistant_msgs = [msg["content"] for msg in history if msg["role"] == "model"]
        # Solo tomamos los últimos 3 planes para no saturar el prompt
        recent_assistant_msgs = recent_assistant_msgs[-3:]
        if recent_assistant_msgs:
            recent_meals_context = f"\n\n--- HISTORIAL RECIENTE DE COMIDAS (¡¡EVITAR REPETICIÓN A TODA COSTA!!) ---\nEl usuario ya ha estado comiendo lo siguiente en sus últimos planes. SU QUEJA PRINCIPAL ES LA MONOTONÍA. Es OBLIGATORIO que el nuevo plan sea COMPLETAMENTE DIFERENTE a esto. Inventa recetas nuevas, usa otros ingredientes, cambia la estructura. ¡Muestra tu máxima creatividad culinaria!\nHistorial reciente:\n{json.dumps(recent_assistant_msgs)}\n----------------------------------------------------------------------"
    
    # Manejo dinámico de skipLunch para forzar al LLM a respetar la exclusión del Almuerzo
    skip_lunch_instruction = ""
    is_skip_lunch_active = form_data.get("skipLunch", False)
    if is_skip_lunch_active:
        skip_lunch_instruction = "\n\n⚠️ INSTRUCCIÓN CRÍTICA Y OBLIGATORIA DE ESTRUCTURA ⚠️\nEl usuario indicó 'Almuerzo Familiar / Ya resuelto' (`skipLunch: true`). ESTÁ TOTAL Y ABSOLUTAMENTE PROHIBIDO incluir la comida 'Almuerzo' en este plan. Si incluyes un Almuerzo, el plan será rechazado. \nDebes generar EXACTAMENTE 3 comidas por el día entero: 'Desayuno', 'Merienda' y 'Cena'. \nIMPORTANTE PARA TU 'Estrategia' o 'PLAN DE ACCIÓN': EL ALMUERZO LO ELEGIRÁ EL USUARIO MANUALMENTE. Tu plan de acción debe felicitar y mencionar directamente que estás excluyendo el almuerzo y dejando espacio calórico porque **él/ella elegirá su almuerzo manualmente** (ya sea comiendo lo que cocinen en su casa o por su cuenta). NUNCA digas que estás concentrando las calorías en el desayuno/cena, sino que estás diseñando un plan de 3 comidas para darle libertad de que elija su almuerzo manualmente."

    # Intervención Determinista (Python)
    deterministic_variety_instruction = get_deterministic_variety_prompt(recent_meals_context, form_data)

    prompt_text = f"Analiza la siguiente información del usuario y genera un plan de comidas de 3 días ajustado a sus necesidades diarias.\n\nInformación del Usuario:\n{json.dumps(form_data, indent=2)}\n{nutrition_context}\n{taste_profile}\n{recent_meals_context}\n{deterministic_variety_instruction}\n{skip_lunch_instruction}\n\n{ANALYZE_SYSTEM_PROMPT}"
    
    # Enforce Pydantic Output schema
    structured_llm = llm.with_structured_output(PlanModel)
    
    # Retry mechanism for LLM invocation
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
    def invoke_with_validation():
        resp = structured_llm.invoke(prompt_text)
        res_dict = resp.model_dump() if hasattr(resp, "model_dump") else (resp if isinstance(resp, dict) else resp.dict())
        if is_skip_lunch_active:
            for d in res_dict.get("days", []):
                for m in d.get("meals", []):
                    if "almuerzo" in m.get("meal", "").lower():
                        print("⚠️ EL MODELO GENERÓ ALMUERZO IGNORANDO INSTRUCCIONES. REINTENTANDO...")
                        raise ValueError("El LLM ignoró la instrucción de omitir Almuerzo.")
        return res_dict
    
    response_dict = invoke_with_validation()
    
    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    print(f"✅ [COMPLETADO] El modelo LangChain finalizó en {duration_secs} segundos.")
    print("-------------------------------------------------------------\n")
    
    result: dict = response_dict
    
    # ========== POST-PROCESO: Forzar valores exactos del calculador ==========
    result["calories"] = nutrition["target_calories"]
    result["macros"] = {
        "protein": nutrition["macros"]["protein_str"],
        "carbs": nutrition["macros"]["carbs_str"],
        "fats": nutrition["macros"]["fats_str"],
    }
    result["main_goal"] = nutrition["goal_label"]
    
    print(f"🔒 [POST-PROCESO] Calorías forzadas a {nutrition['target_calories']} kcal (calculador Python)")
    
    return result

def swap_meal(form_data: dict):
    rejected_meal = form_data.get("rejected_meal", "")
    meal_type = form_data.get("meal_type", "Comida")
    target_calories = form_data.get("target_calories", 0)
    diet_type = form_data.get("diet_type", "balanced")
    
    allergies = form_data.get("allergies", [])
    dislikes = form_data.get("dislikes", [])
    liked_meals = form_data.get("liked_meals", [])
    disliked_meals = form_data.get("disliked_meals", [])
    
    context_extras = ""
    if allergies: context_extras += f"\n    - ALERGIAS (PROHIBIDO INCLUIR): {', '.join(allergies)}"
    if dislikes: context_extras += f"\n    - DISGUSTOS (PROHIBIDO INCLUIR): {', '.join(dislikes)}"
    
    # Ensure the temporarily rejected meal is added to disliked for this prompt
    all_disliked = set(disliked_meals)
    if rejected_meal:
        all_disliked.add(rejected_meal)
        
    if all_disliked: 
        context_extras += f"\n    - 🚫 EXCLUSIÓN ESTRICTA: ESTÁ TOTALMENTE PROHIBIDO generar cualquier plato o ingrediente principal de esta lista: {', '.join(list(all_disliked))}. NINGÚN PLATO NUEVO PUEDE LLAMARSE IGUAL NI PARECERSE."
        
    if liked_meals: context_extras += f"\n    - PLATOS FAVORITOS (PARA INSPIRACIÓN): {', '.join(liked_meals)}"

    # --- ANTI MODE-COLLAPSE PARA SWAPS ---
    # Sugerir una proteína diferente al LLM para evitar que siempre use Pollo+Arroz
    try:
        import random
        available_for_swap = [p for p in DOMINICAN_PROTEINS if p.lower() not in rejected_meal.lower()]
        # Filtrar por dieta
        if diet_type in ["vegano", "vegan"]:
            available_for_swap = [p for p in available_for_swap if p.lower() in ["habichuelas rojas", "habichuelas negras", "gandules", "lentejas", "garbanzos", "soya/tofu", "berenjena"]]
        elif diet_type in ["vegetariano", "vegetarian"]:
            available_for_swap = [p for p in available_for_swap if p.lower() not in ["pollo", "cerdo", "res", "pescado", "atún", "camarones", "chuleta", "longaniza", "salami dominicano"]]
        if available_for_swap:
            suggested_protein = random.choice(available_for_swap)
            context_extras += f"\n    - 💡 SUGERENCIA DE VARIEDAD: Para este swap, intenta basar el plato en **{suggested_protein}** como proteína principal (o en un ingrediente radicalmente diferente al rechazado)."
    except Exception:
        pass  # No bloquear el swap si falla la sugerencia

    print("\n-------------------------------------------------------------")
    print("⏳ [AGENTE DE SUSTITUCIÓN INTERPRETATIVO] Analizando rechazo...")
    print(f"➡️  Interpretando por qué rechazó: \"{rejected_meal}\" ({meal_type})")
    
    start_time = time.time()
    
    prompt_text = SWAP_MEAL_PROMPT_TEMPLATE.format(
        rejected_meal=rejected_meal,
        meal_type=meal_type,
        target_calories=target_calories,
        diet_type=diet_type,
        context_extras=context_extras
    )
    
    swap_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.8, # Temperatura más alta para buscar alternativas extremadamente creativas
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(MealModel)
    
    # Invocar LLM con reintentos automáticos (tenacity)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [SWAP] Reintento #{retry_state.attempt_number} tras error de formato...")
    )
    def invoke_with_retry():
        return swap_llm.invoke(prompt_text)
    
    response = invoke_with_retry()
    
    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    print(f"✅ [COMPLETADO] Nueva alternativa {meal_type} generada en {duration_secs} segundos.")
    print("-------------------------------------------------------------\n")
    if hasattr(response, "model_dump"):
        return getattr(response, "model_dump")()
    elif isinstance(response, dict):
        return response
    elif hasattr(response, "dict"):
        return getattr(response, "dict")()
    else:
        raise ValueError("El modelo de IA generó una respuesta inválida. Por favor, reintenta.")


def generate_auto_shopping_list(plan_data: dict) -> list:
    """Extrae ingredientes del plan, los consolida y categoriza por pasillo de supermercado."""
    print("\n-------------------------------------------------------------")
    print("🛒 [AUTO-SHOPPING LIST] Consolidando ingredientes del plan...")
    
    ingredients = []
    days = plan_data.get("days", [])
    for d in days:
        for m in d.get("meals", []):
            ing = m.get("ingredients", [])
            if ing:
                ingredients.extend(ing)
                
    if not ingredients:
        return []
        
    prompt = AUTO_SHOPPING_LIST_PROMPT.format(ingredients_json=json.dumps(ingredients))
    
    shopping_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.2,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(ShoppingListModel)
    
    try:
        response = shopping_llm.invoke(prompt)
        if hasattr(response, "items"):
            items = response.items
        elif isinstance(response, dict) and "items" in response:
            items = response["items"]
        else:
            items = []
            
        print(f"✅ Se consolidaron ingredientes en {len(items)} categorías.")
        print("-------------------------------------------------------------\n")
        return items
    except Exception as e:
        print(f"❌ Error generando auto shopping list: {e}")
        return []

# ============================================================
# ORQUESTACIÓN LANGGRAPH CHAT CON MEMORYSAVER
# ============================================================
class ChatState(MessagesState):
    user_id: str
    session_id: str
    form_data: dict
    current_plan: dict
    updated_fields: dict
    new_plan: dict
    sys_prompt: str

def call_model(state: ChatState):
    print(f"🧠 [LANGGRAPH NODE] call_model")
    messages = state["messages"]
    sys_prompt = state.get("sys_prompt", "")
    
    llm_messages = []
    if sys_prompt:
        llm_messages.append(SystemMessage(content=sys_prompt))
        
    for m in messages:
        if not isinstance(m, SystemMessage):
            llm_messages.append(m)
            
    chat_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.7,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    llm_with_tools = chat_llm.bind_tools(agent_tools)
    response = llm_with_tools.invoke(llm_messages)
    return {"messages": [response]}

def execute_tools(state: ChatState):
    import json
    messages = state["messages"]
    last_message = messages[-1]
    
    updated_fields = state.get("updated_fields", {})
    new_plan = state.get("new_plan", None)
    
    tool_messages = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            tool_result = ""
            print(f"🔧 [LANGGRAPH TOOL] Ejecutando {tool_name}")
            
            if tool_name == "update_form_field":
                field = tool_args.get("field")
                new_value = tool_args.get("new_value", "")
                
                # Sanitize numeric values for the frontend response too
                if field in ['weight', 'height', 'age']:
                    import re
                    extracted = re.sub(r'[^\d.]', '', str(new_value))
                    if extracted:
                        new_value = extracted
                        
                if field in ['allergies', 'medicalConditions', 'dislikes', 'struggles']:
                    updated_fields[field] = [item.strip() for item in (new_value if isinstance(new_value, str) else "").split(",") if item.strip()]
                else:
                    updated_fields[field] = new_value
                    
                # Re-inject the sanitized new_value into tool_args so the tool itself gets the clean version if it uses it directly
                # (Aunque ya limpiamos adentro del tool, es buena práctica pasarlo limpio)
                tool_args["new_value"] = new_value
                tool_result = update_form_field.invoke(tool_args)
                
            elif tool_name == "generate_new_plan_from_chat":
                user_instructions = tool_args.get("instructions", "")
                user_id = state.get("user_id")
                session_id = state.get("session_id")
                form_data = state.get("form_data", {})
                
                tool_result = execute_generate_new_plan(user_id if user_id and user_id != 'guest' else session_id, form_data, user_instructions)
                
                try:
                    parsed_plan = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                    if isinstance(parsed_plan, dict) and ("days" in parsed_plan or "meals" in parsed_plan):
                        new_plan = parsed_plan
                        tool_result = "El plan de comidas de 3 días fue generado exitosamente. Dile al usuario que lo revise en su dashboard."
                except Exception:
                    pass
                    
            elif tool_name == "modify_single_meal":
                user_id = state.get("user_id")
                session_id = state.get("session_id")
                
                tool_result = execute_modify_single_meal(
                    user_id=user_id if user_id and user_id != 'guest' else session_id,
                    day_number=tool_args.get("day_number", 1),
                    meal_type=tool_args.get("meal_type", "Desayuno"),
                    changes=tool_args.get("changes", "")
                )
                try:
                    parsed_mod = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                    if isinstance(parsed_mod, dict) and "modified_meal" in parsed_mod:
                        from db import get_latest_meal_plan
                        updated_plan = get_latest_meal_plan(user_id if user_id and user_id != 'guest' else session_id)
                        if updated_plan:
                            new_plan = updated_plan
                        tool_result = f"La comida fue modificada exitosamente. La nueva comida es: {parsed_mod['modified_meal'].get('name', 'Comida actualizada')}. Dile al usuario que su plan ya fue actualizado."
                except Exception:
                    pass
            else:
                for t in agent_tools:
                    if t.name == tool_name:
                        tool_result = t.invoke(tool_args)
                        break
                        
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
            
    return {"messages": tool_messages, "updated_fields": updated_fields, "new_plan": new_plan}

def route_tools(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"
    return END

# Removido el MemorySaver global estático
# chat_memory_saver = MemorySaver()
chat_builder = StateGraph(ChatState)
chat_builder.add_node("call_model", call_model)
chat_builder.add_node("execute_tools", execute_tools)
chat_builder.add_edge(START, "call_model")
chat_builder.add_conditional_edges("call_model", route_tools, ["execute_tools", END])
chat_builder.add_edge("execute_tools", "call_model")
# NOTA: chat_graph_app se compila dinámicamente usando el PostgresSaver en cada petición

# ============================================================
# CHAT CON AGENTE (Wrapper Principal)
# ============================================================

def generate_chat_title_background(session_id: str, first_message: str):
    try:
        from db import supabase
        if not supabase: return
        res = supabase.table("agent_messages").select("id").eq("session_id", session_id).like("content", "[SYSTEM_TITLE]%").execute()
        if res.data and len(res.data) > 0:
            return 
            
        title_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.4, google_api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = TITLE_GENERATION_PROMPT.format(first_message=first_message)
        response = title_llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        title = str(content).replace('"', '').replace("'", "").strip()
        
        supabase.table("agent_messages").insert({
            "session_id": session_id,
            "role": "model",
            "content": f"[SYSTEM_TITLE] {title}"
        }).execute()
        print(f"✅ Título generado para sesión {session_id}: {title}")
    except Exception as e:
        print(f"⚠️ Error generando título: {e}")

# ============================================================
# RAG QUERY ROUTING (Patrón HyDE)
# ============================================================
def rag_query_router(prompt: str) -> dict:
    """
    Decide si un mensaje del usuario amerita búsqueda RAG y, si sí,
    reescribe la query para que sea óptima para búsqueda vectorial.
    
    Retorna:
        {"skip": True} si el mensaje es casual y no necesita RAG.
        {"skip": False, "query": "..."} con la query reescrita para el embedding.
    """
    # Paso 1: Filtro rápido — mensajes cortos y claramente casuales
    casual_patterns = [
        'ok', 'okay', 'vale', 'sí', 'si', 'no', 'gracias', 'thanks',
        'hola', 'hello', 'hey', 'buenos días', 'buenas tardes', 'buenas noches',
        'perfecto', 'genial', 'entendido', 'claro', 'listo', 'dale',
        'jaja', 'jeje', 'lol', 'xd', 'bien', 'cool', 'nice',
        'de acuerdo', 'ya', 'ajá', 'aja', 'okey', 'bueno'
    ]
    
    clean = prompt.strip().lower().rstrip('!?.,')
    # Si es un mensaje muy corto O coincide con un patrón casual
    if len(clean) < 4 or clean in casual_patterns:
        print(f"⏭️ [RAG ROUTER] Mensaje casual detectado: '{prompt[:30]}' → Saltando RAG.")
        return {"skip": True}
    
    # Paso 2: Combos casuales ("ok gracias", "sí perfecto", etc.)
    words = clean.split()
    if len(words) <= 3 and all(w in casual_patterns for w in words):
        print(f"⏭️ [RAG ROUTER] Combo casual detectado: '{prompt[:30]}' → Saltando RAG.")
        return {"skip": True}
    
    # Paso 3: Para mensajes sustanciales, usar Flash-Lite para reescribir la query
    try:
        router_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.0,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        
        rewrite_prompt = RAG_ROUTER_PROMPT.format(prompt=prompt)
        
        response = router_llm.invoke(rewrite_prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        result = str(content).strip().strip('"').strip("'")
        
        if result.upper() == "SKIP":
            print(f"⏭️ [RAG ROUTER] Flash-Lite determinó que no necesita RAG: '{prompt[:30]}'")
            return {"skip": True}
        
        print(f"🎯 [RAG ROUTER] Query reescrita: '{prompt[:30]}...' → '{result}'")
        return {"skip": False, "query": result}
        
    except Exception as e:
        print(f"⚠️ [RAG ROUTER] Error en rewrite, usando prompt original: {e}")
        return {"skip": False, "query": prompt}

def chat_with_agent(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None):
    from memory_manager import build_memory_context
    
    # Obtener contexto de memoria inteligente (resúmenes + mensajes recientes)
    memory = build_memory_context(session_id)
    
    # === RAG INJECTION (con Query Routing inteligente) ===
    user_facts_text = ""
    visual_facts_text = ""
    
    if user_id:
        rag_decision = rag_query_router(prompt)
        
        if not rag_decision.get("skip"):
            try:
                from fact_extractor import get_embedding
                from db import search_user_facts, search_visual_diary
                
                optimized_query = rag_decision.get("query", prompt)
                
                # 1. Buscar hechos textuales con query optimizada
                print(f"🔍 [CHAT RAG] Buscando con query optimizada: '{optimized_query}'")
                query_emb = get_embedding(optimized_query)
                if query_emb:
                    facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                    if facts_data:
                        fact_list = [f"• {item['fact']}" for item in facts_data]
                        user_facts_text = "\n".join(fact_list)
                        print(f"🧠 [CHAT RAG] Hechos textuales recuperados: {len(facts_data)}")
                
                # 2. Buscar memoria visual
                from vision_agent import get_multimodal_embedding
                visual_query_emb = get_multimodal_embedding(optimized_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        visual_list = [f"• {item['description']}" for item in visual_data]
                        visual_facts_text = "\n".join(visual_list)
                        print(f"📸 [CHAT RAG VISUAL] Entradas visuales recuperadas: {len(visual_data)}")
            except Exception as e:
                print(f"⚠️ [CHAT RAG] Error recuperando memoria: {e}")
            
    rag_context = ""
    if user_facts_text or visual_facts_text:
        rag_context = "\n--- MEMORIA VECTORIAL (RAG) ---\nContexto recuperado de interacciones pasadas relevante a la pregunta actual:\n"
        if user_facts_text:
            rag_context += f"{user_facts_text}\n"
        if visual_facts_text:
            rag_context += f"Inventario Visual y Fotos:\n{visual_facts_text}\n"
        rag_context += "Úsalo para responder de forma súper personalizada.\n"
        rag_context += "⚠️ REGLA DE CONFLICTO: Si hay conflicto entre el historial reciente o los resúmenes y estos Hechos Permanentes, LOS HECHOS PERMANENTES SON LA LEY y tienen prioridad absoluta.\n"
        rag_context += "---------------------------------------------\n"

    system_prompt = """Eres el agente asistente de nutrición IA de MealfitRD. Tu objetivo principal es ayudar a los usuarios con dudas sobre su plan generado o sus objetivos de dieta. Trata de dar respuestas al grano, conversacionales y amigables.
IMPORTANTE: NUNCA saludes con 'Hola' ni repitas saludos introductorios. El usuario ya fue saludado al iniciar el chat. Ve directo al punto en cada respuesta.
REGLA CRUCIAL: El plan del usuario tiene 3 opciones distintas. Llámalas SIEMPRE "Opción A", "Opción B" y "Opción C". NUNCA te refieras a ellas como "Día 1", "Día 2" o "Día 3" en tu conversación con el usuario."""
    
    if rag_context:
        system_prompt += f"\n{rag_context}"
    
    # Determinar si es un usuario autenticado o invitado
    is_authenticated = user_id and user_id != session_id and user_id != "guest"
    
    system_prompt += f"""

TIENES HERRAMIENTAS DISPONIBLES:
- OBLIGATORIO: Usa `update_form_field` INMEDIATAMENTE y SIN EXCEPCIÓN cada vez que el usuario mencione un nuevo dato sobre sí mismo que deba actualizarse en su perfil (ej: "a partir de hoy soy vegano", "peso 80kg", "tengo diabetes", "soy intolerante a la lactosa", "no me gusta el tomate"). Si no usas esta herramienta para esos casos, la Interfaz Gráfica del usuario quedará desincronizada y el sistema fallará.
- Usa `generate_new_plan_from_chat` SOLO cuando el usuario pida explícitamente generar un plan nuevo (ej: 'hazme un plan', 'genera mi rutina', 'quiero un menú diferente'). Esta herramienta ejecuta el pipeline completo y genera un plan personalizado al instante.
- NO uses generate_new_plan_from_chat si el usuario solo da información de salud o pregunta sobre su plan actual.
- Usa `log_consumed_meal` para registrar en el diario cualquier comida que el usuario afirme haber comido. Si analizas una foto de una comida y el usuario confirma que se la comió, USA ESTA HERRAMIENTA usando los macros estimados (calorías, proteína, carbohidratos y grasas saludables), pasándolos todos a la herramienta.
- Usa `modify_single_meal` cuando el usuario pida un CAMBIO PUNTUAL a una comida específica de su plan (ej: 'cámbiale el salami al mangú por huevos en la Opción A', 'ponle más proteína al almuerzo', 'quítale el arroz a la cena de la Opción B'). Esta herramienta modifica SOLO esa comida, no regenera todo el plan. Debes identificar correctamente el day_number (1 para Opción A, 2 para Opción B, o 3 para Opción C) y el meal_type ('Desayuno', 'Almuerzo', 'Cena', 'Merienda') del plan activo del usuario. Si el usuario no especifica, asume 1 (Opción A).
- Usa `add_to_shopping_list` cuando el usuario diga que se quedó sin algo, necesita comprar algo o pida añadir items a su lista de compras (ej: 'me quedé sin plátanos', 'añade leche a mi lista', 'necesito comprar arroz y huevos'). Separa los items por coma.

El user_id del usuario actual es: {user_id}"""

    if current_plan:
        system_prompt += f"\n\nCONTEXTO CRÍTICO: El usuario actualmente tiene este plan de comidas activo:\n{json.dumps(current_plan)}\n\nUsa esta información para responder con exactitud preguntas sobre lo que le toca comer hoy o sugerir cambios basados en lo que ya tiene asignado (como desayuno, almuerzo o cena)."
        
        if form_data and form_data.get("skipLunch"):
            system_prompt += "\n⚠️ IMPORTANTE SOBRE EL ALMUERZO: El plan actual NO tiene 'Almuerzo' NO porque se haya omitido y redistribuido, sino porque el usuario eligió 'Almuerzo Familiar / Ya resuelto'. Esto significa que EL USUARIO SÍ VA A ALMORZAR en su casa libremente. NUNCA digas que 'omitimos el almuerzo y redistribuimos las calorías' porque eso es falso. Dile que en realidad tiene un 'Cupo Vacío' en el plan porque reservamos las calorías para que almorzara libremente en su casa, y aliéntalo a que te cuente qué comerá para anotarlo en su registro."
    
        system_prompt += f"\n\nCONTEXTO HISTÓRICO DEL USUARIO (resúmenes de conversaciones pasadas):\n{memory['summary_context']}"
        
    config = {"configurable": {"thread_id": session_id}}
    
    from db import connection_pool
    # Compilamos el grafo dinámicamente para usar la conexión compartida/pool en un entorno multi-worker
    if connection_pool:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver(connection_pool)
        chat_graph_app = chat_builder.compile(checkpointer=checkpointer)
    else:
        from langgraph.checkpoint.memory import MemorySaver
        print("⚠️ [LangGraph] No pool de PostgreSQL, usando MemorySaver en RAM.")
        checkpointer = MemorySaver()
        chat_graph_app = chat_builder.compile(checkpointer=checkpointer)
        
    existing_state = chat_graph_app.get_state(config)
    
    inputs = {
        "user_id": user_id or "guest",
        "session_id": session_id,
        "form_data": form_data or {},
        "current_plan": current_plan or {},
        "sys_prompt": system_prompt, # Sobre-escribe el prompt dinámicamente en cada ejecución
        "updated_fields": {},        # Reinicia los valores extraídos en cada ejecución
        "new_plan": None             # Reinicia el plan nuevo en cada ejecución
    }
    
    if not existing_state.values:
        print(f"🔄 [LANGGRAPH] Inicializando nuevo thread O restaurando tras reinicio para session_id: {session_id}")
        messages = []
        for msg in memory["recent_messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "model":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        inputs["messages"] = messages
    else:
        print(f"🔄 [LANGGRAPH] Thread existente detectado en Checkpointer. Inyectando solo el prompt actual.")
        inputs["messages"] = [HumanMessage(content=prompt)]
        
    print("\n-------------------------------------------------------------")
    print("⏳ [CHAT] LangGraph ejecutando pipeline...")
    start_time = time.time()
    
    final_state = chat_graph_app.invoke(inputs, config=config)
    
    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    print(f"✅ [COMPLETADO] LangGraph finalizó en {duration_secs} segundos.")
    print("-------------------------------------------------------------\n")
    
    final_messages = final_state["messages"]
    last_msg = final_messages[-1]
    content = last_msg.content
    
    if isinstance(content, list):
        content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        
    return str(content), final_state.get("updated_fields", {}), final_state.get("new_plan")

import asyncio
from typing import AsyncGenerator

async def achat_with_agent_stream(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None, local_date: Optional[str] = None, tz_offset: Optional[int] = None) -> AsyncGenerator[str, None]:
    """Versión asíncrona de chat_with_agent que emite eventos del modelo y herramientas mediante SSE (JSONlines)."""
    from memory_manager import build_memory_context
    memory = build_memory_context(session_id)
    
    # RAG INJECTION (con Query Routing inteligente)
    user_facts_text = ""
    visual_facts_text = ""
    if user_id:
        rag_decision = rag_query_router(prompt)
        
        if not rag_decision.get("skip"):
            try:
                from fact_extractor import get_embedding
                from db import search_user_facts, search_visual_diary
                
                optimized_query = rag_decision.get("query", prompt)
                
                query_emb = get_embedding(optimized_query)
                if query_emb:
                    facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                    if facts_data:
                        fact_list = [f"• {item['fact']}" for item in facts_data]
                        user_facts_text = "\n".join(fact_list)
                
                from vision_agent import get_multimodal_embedding
                visual_query_emb = get_multimodal_embedding(optimized_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        visual_list = [f"• {item['description']}" for item in visual_data]
                        visual_facts_text = "\n".join(visual_list)
            except Exception as e:
                print(f"⚠️ [CHAT RAG] Error en stream: {e}")
            
    rag_context = ""
    if user_facts_text or visual_facts_text:
        rag_context = "\n--- MEMORIA VECTORIAL (RAG) ---\n"
        if user_facts_text: rag_context += f"{user_facts_text}\n"
        if visual_facts_text: rag_context += f"Inventario Visual:\n{visual_facts_text}\n"
        rag_context += "Úsalo para responder de forma súper personalizada.\n⚠️ REGLA DE CONFLICTO: LOS HECHOS PERMANENTES SON LEY.\n---------------------------------------------\n"

    system_prompt = """Eres el agente asistente de nutrición IA de MealfitRD. Tu objetivo principal es ayudar a los usuarios con dudas sobre su plan generado o sus objetivos de dieta. Trata de dar respuestas al grano, conversacionales y amigables.
IMPORTANTE: NUNCA saludes con 'Hola' ni repitas saludos introductorios.
REGLA CRUCIAL: El plan del usuario tiene 3 opciones distintas. Llámalas SIEMPRE "Opción A", "Opción B" y "Opción C".

REGLAS DE FORMATO VISUAL (ESTRICTAS):
1. Usa **negritas** para resaltar nombres de alimentos, cantidades (ej. **350 kcal**, **35g de proteína**) y conceptos clave.
2. Usa viñetas (`-` o `•`) SIEMPRE para listar macros, ingredientes o pasos, haciéndolo súper visual y fácil de leer.
3. Aplica saltos de línea (párrafos cortos) para que el texto respire y no sea un bloque denso."""
    
    if rag_context: system_prompt += f"\n{rag_context}"
    
    system_prompt += f"""
TIENES HERRAMIENTAS DISPONIBLES:
- Usa `update_form_field` INMEDIATAMENTE al haber nuevos datos de perfil.
- Usa `generate_new_plan_from_chat` SOLO cuando el usuario pida explícitamente generar un plan nuevo.
- Usa `log_consumed_meal` para registrar en el diario cualquier comida consumida.
- Usa `modify_single_meal` para cambios puntuales.
- Usa `add_to_shopping_list` para añadir compras.
El user_id actual es: {user_id}"""

    if current_plan:
        system_prompt += f"\nCONTEXTO CRÍTICO: Plan activo:\n{json.dumps(current_plan)}\n"
        
        if form_data and form_data.get("skipLunch"):
            system_prompt += "⚠️ IMPORTANTE SOBRE EL ALMUERZO: El usuario escogió 'Almuerzo Familiar', EL USUARIO SÍ VA A ALMORZAR. NO digas que se omitió y redistribuyó. Dile que le dejaste un 'Cupo Vacío' y coméntale que te dicte qué almorzó para registrarlo.\n"
        system_prompt += f"\nHISTÓRICO:\n{memory['summary_context']}"
        
    if user_id and user_id != "guest":
        try:
            from db import get_consumed_meals_today
            consumed_today = get_consumed_meals_today(user_id, date_str=local_date, tz_offset_mins=tz_offset)
            if consumed_today:
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}. Revisa esto ANTES de preguntar si ya comió algo (por ejemplo, si ya tiene una cena registrada, no le preguntes si esa foto es su cena, asume que es un snack nocturno o pregúntale por qué repite). Si la foto o mensaje coincide con algo que ya está registrado, felicítalo o no lo registres de nuevo."
            else:
                system_prompt += "\n\nDIARIO DE HOY: El usuario no ha registrado ninguna comida el día de hoy todavía."
        except Exception as e:
            print(f"⚠️ Error inyectando contexto de diario: {e}")
            
    config = {"configurable": {"thread_id": session_id}}
    
    # Compilamos usando PostgresSaver sincrónico porque astream_events nativo asíncrono tiene problemas en Windows
    from db import connection_pool
    if connection_pool:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpointer = PostgresSaver(connection_pool)
        chat_graph_app = chat_builder.compile(checkpointer=checkpointer)
    else:
        from langgraph.checkpoint.memory import MemorySaver
        chat_graph_app = chat_builder.compile(checkpointer=MemorySaver())
        
    existing_state = chat_graph_app.get_state(config)
    
    inputs = {
        "user_id": user_id or "guest",
        "session_id": session_id,
        "form_data": form_data or {},
        "current_plan": current_plan or {},
        "sys_prompt": system_prompt,
        "updated_fields": {},
        "new_plan": None
    }
    
    if not existing_state.values:
        messages = []
        for msg in memory["recent_messages"]:
            if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "model": messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        inputs["messages"] = messages
    else:
        inputs["messages"] = [HumanMessage(content=prompt)]
        
    yield f"data: {json.dumps({'type': 'progress', 'message': 'Analizando tu mensaje...'})}\n\n"
    
    print(f"⏳ [CHAT STREAM] LangGraph iniciando astream nativo para {session_id}...")
    
    final_state_snapshot = None
    
    try:
        async for event in chat_graph_app.astream(inputs, config=config, stream_mode="messages"):
            # Identificar el contenido exacto del evento 'messages' (tupla mensaje, dict)
            if isinstance(event, tuple) and len(event) == 2:
                msg_chunk, metadata = event
                if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                    if not msg_chunk.tool_calls:
                        chunk_content = msg_chunk.content
                        if isinstance(chunk_content, list):
                            chunk_content = "".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in chunk_content])
                        if chunk_content: # Evitar chunks vacíos
                            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_content})}\n\n"
                    else:
                        for idx, tool_call in enumerate(msg_chunk.tool_calls):
                            if idx == 0:  # Mostrar el mensaje 1 sola vez por llamada múltiple
                                tool_name = tool_call.get("name", "")
                                if tool_name == "generate_new_plan_from_chat":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Calculando macros y diseñando plan...'})}\n\n"
                                elif tool_name == "modify_single_meal":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Modificando comida...'})}\n\n"
                                elif tool_name == "update_form_field":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Actualizando base de datos...'})}\n\n"
                                elif tool_name == "log_consumed_meal":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': 'Registrando progreso de hoy...'})}\n\n"
                                    
    except Exception as e:
        print(f"❌ [CHAT STREAM] Error en astream nativo: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return
        
    # Obtener el estado final actualizado
    try:
        final_state_snapshot = await chat_graph_app.aget_state(config)
    except Exception as e:
        print(f"⚠️ Error obteniendo aget_state tras stream: {e}")

    final_content = ""
    updated_fields = {}
    new_plan = None
    
    if final_state_snapshot and final_state_snapshot.values:
        updated_fields = final_state_snapshot.values.get("updated_fields", {})
        new_plan = final_state_snapshot.values.get("new_plan", None)
        final_messages = final_state_snapshot.values.get("messages", [])
        if final_messages:
            last_msg = final_messages[-1]
            extracted_content = last_msg.content
            if isinstance(extracted_content, list):
                extracted_content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in extracted_content])
            final_content = str(extracted_content)

    print("✅ [CHAT STREAM] Finalizado con éxito.")
    yield f"data: {json.dumps({'type': 'done', 'response': final_content, 'updated_fields': updated_fields, 'new_plan': new_plan})}\n\n"