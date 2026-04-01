# backend/agent.py

import os
import logging
import time
import json
import re
import unicodedata
logger = logging.getLogger(__name__)

from constants import strip_accents, parse_ingredient_qty
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



from db import get_user_profile, update_user_health_profile
import concurrent.futures
import traceback
from datetime import datetime, timezone
from cpu_tasks import _calcular_frecuencias_regex_cpu_bound
from schemas import SemanticDedupResult
from constants import categorize_shopping_item
from memory_manager import build_memory_context
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from langgraph.checkpoint.postgres import PostgresSaver
from db import get_user_ingredient_frequencies, get_custom_shopping_items, get_latest_meal_plan_with_id, get_session_messages, save_message, search_user_facts, search_visual_diary, connection_pool, get_consumed_meals_today
from dotenv import load_dotenv

load_dotenv()

from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel, ShoppingListModel
from prompts import (
    DETERMINISTIC_VARIETY_PROMPT, SWAP_MEAL_PROMPT_TEMPLATE, 
    AUTO_SHOPPING_LIST_PROMPT, TITLE_GENERATION_PROMPT, RAG_ROUTER_PROMPT,
    CHAT_SYSTEM_PROMPT_BASE, CHAT_STREAM_SYSTEM_PROMPT_BASE, SEMANTIC_DEDUP_PROMPT
)

from tools import (
    update_form_field, generate_new_plan_from_chat,
    log_consumed_meal, modify_single_meal,
    add_to_shopping_list, remove_from_shopping_list, search_deep_memory, agent_tools, analyze_preferences_agent,
    execute_generate_new_plan, execute_modify_single_meal
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
from constants import (
    PROTEIN_SYNONYMS as protein_synonyms, 
    CARB_SYNONYMS as carb_synonyms,
    VEGGIE_FAT_SYNONYMS as veggie_fat_synonyms,
    FRUIT_SYNONYMS as fruit_synonyms,
    _get_fast_filtered_catalogs
)
def swap_meal(form_data: dict):
    ingredient_names = []  # Inicializar para evitar NameError si el bloque de shopping constraint no se ejecuta
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

    # --- CONSTRICCIÓN DE LISTA DE COMPRAS ---
    user_id = form_data.get("user_id")
    if user_id and user_id != "guest":
        try:
            existing = get_custom_shopping_items(user_id)
            existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
            if existing_items:
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
                    context_extras += f"\n    - ⚠️ REGLA DE SUPERMERCADO ABSOLUTA E INQUEBRANTABLE: El usuario YA COMPRÓ su comida y no puede comprar nada más. TIENES ESTRICTAMENTE PROHIBIDO sugerir frutas, vegetales, carnes, lácteos, cereales o cualquier ingrediente que no esté en esta lista exacta: [{', '.join(ingredient_names)}]. Si la lista incluye tomate y cebolla, usa SOLO tomate y cebolla, no inventes lechuga ni aguacate. ESPECIAL ATENCIÓN: Si no ves pollo, pescado ni carnes en la lista, TIENES PROHIBIDO inventarlos; debes crear un plato vegetariano o basado en los granos/quesos que sí tenga la lista. La receta alternativa DEBE limitarse al 100% a lo que hay en esta lista."
                    logger.info("🛒 [CONSTRAINT] Restricción de lista de compras añadida en swap_meal (agent).")
        except Exception as check_e:
            logger.error(f"Error revisando lista de compras en swap_meal: {check_e}")
    # ----------------------------------------

    # --- ANTI MODE-COLLAPSE PARA SWAPS (Proteína + Carbohidrato + Vegetal) ---
    # Sugerir alternativas en las 3 dimensiones usando peso inverso por frecuencia
    try:
        
        # Usar el mismo filtro centralizado que el plan principal (DRY)
        swap_allergies = tuple([a.lower() for a in allergies]) if allergies else ()
        swap_dislikes = tuple([d.lower() for d in dislikes]) if dislikes else ()
        swap_diet = diet_type.lower() if diet_type else ""
        
        filtered_p, filtered_c, filtered_v, _ = _get_fast_filtered_catalogs(swap_allergies, swap_dislikes, swap_diet)
        
        # Excluir ingredientes del plato rechazado
        rejected_lower = rejected_meal.lower()
        available_proteins = [p for p in filtered_p if p.lower() not in rejected_lower]
        available_carbs = [c for c in filtered_c if c.lower() not in rejected_lower]
        available_veggies = [v for v in filtered_v if v.lower() not in rejected_lower]
        
        user_id = form_data.get("user_id")
        db_freq_map = {}
        if user_id and user_id != "guest":
            try:
                db_freq_map = get_user_ingredient_frequencies(user_id)
            except Exception as freq_e:
                logger.error(f"⚠️ [SWAP] Error consultando frecuencia, usando random simple: {freq_e}")
        
        def _pick_by_inverse_freq(available_items, synonyms_map):
            """Elige un ingrediente usando peso inverso por frecuencia."""
            if not available_items:
                return None
            if db_freq_map:
                freq = {}
                for item in available_items:
                    syns = synonyms_map.get(item.lower(), [item.lower()])
                    freq[item] = sum(db_freq_map.get(strip_accents(syn.lower()), 0) for syn in syns)
                # Peso inverso consistente con get_deterministic_variety_prompt(): 1/(freq+1)
                # Independiente del max del dataset → distribución estable y determinista.
                weights = [1.0 / (freq.get(item, 0) + 1) for item in available_items]
                return random.choices(available_items, weights=weights, k=1)[0]
            return random.choice(available_items)
        
        suggested_protein = _pick_by_inverse_freq(available_proteins, protein_synonyms)
        suggested_carb = _pick_by_inverse_freq(available_carbs, carb_synonyms)
        suggested_veggie = _pick_by_inverse_freq(available_veggies, veggie_fat_synonyms)
        
        suggestions = []
        if suggested_protein:
            suggestions.append(f"**{suggested_protein}** como proteína")
        if suggested_carb:
            suggestions.append(f"**{suggested_carb}** como carbohidrato")
        if suggested_veggie:
            suggestions.append(f"**{suggested_veggie}** como vegetal/grasa")
        
        if suggestions:
            if not ingredient_names:
                context_extras += f"\n    - 💡 SUGERENCIA DE VARIEDAD: Para este swap, intenta usar {', '.join(suggestions)} (o ingredientes radicalmente diferentes al rechazado)."
                logger.debug(f"🎲 [SWAP ANTI MODE-COLLAPSE] Sugerencias: {suggestions}")
            else:
                logger.debug("🛡️ [SWAP ANTI MODE-COLLAPSE] Desactivado debido a la Regla de Supermercado estricta.")
    except Exception:
        pass  # No bloquear el swap si falla la sugerencia

    logger.info("\n-------------------------------------------------------------")
    logger.info("⏳ [AGENTE DE SUSTITUCIÓN INTERPRETATIVO] Analizando rechazo...")
    logger.info(f"➡️  Interpretando por qué rechazó: \"{rejected_meal}\" ({meal_type})")
    
    start_time = time.time()
    
    prompt_text = SWAP_MEAL_PROMPT_TEMPLATE.format(
        rejected_meal=rejected_meal,
        meal_type=meal_type,
        target_calories=target_calories,
        diet_type=diet_type,
        context_extras=context_extras
    )
    
    temp = 0.2 if ingredient_names else 0.8
    swap_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=temp, 
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
    logger.info(f"✅ [COMPLETADO] Nueva alternativa {meal_type} generada en {duration_secs} segundos.")
    logger.info("-------------------------------------------------------------\n")
    if hasattr(response, "model_dump"):
        return getattr(response, "model_dump")()
    elif isinstance(response, dict):
        return response
    elif hasattr(response, "dict"):
        return getattr(response, "dict")()
    else:
        raise ValueError("El modelo de IA generó una respuesta inválida. Por favor, reintenta.")


def _pre_consolidate_ingredients_multiday(ingredients: list, base_days: int = 3) -> list:
    """Pre-consolida ingredientes sumando cantidades en Python, y genera las cantidades crudas para 7, 15 y 30 días."""
    
    def _normalize(text: str) -> str:
        if not text: return ""
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))


    groups = {}
    order = []
    
    for ing in ingredients:
        if not isinstance(ing, str) or not ing.strip():
            continue
        num, unit, name = parse_ingredient_qty(ing, to_metric=False)
        key = (_normalize(name), _normalize(unit))
        
        if key not in groups:
            groups[key] = {"total": num, "unit": unit, "name": name, "raw": ing, "can_sum": num is not None}
            order.append(key)
        else:
            entry = groups[key]
            if entry["can_sum"] and num is not None:
                entry["total"] = (entry["total"] or 0) + num
            else:
                entry["can_sum"] = False
    
    result = []
    
    def format_qty(qty):
        return str(int(qty)) if qty == int(qty) else f"{qty:.2f}"
            
    for key in order:
        entry = groups[key]
        if entry["can_sum"] and entry["total"] is not None:
            base_total = entry["total"]
            div = base_days if base_days > 0 else 1
            
            t7 = base_total * (7 / div)
            t15 = base_total * (15 / div)
            t30 = base_total * (30 / div)
            
            unit_part = f" {entry['unit']}" if entry["unit"] else ""
            
            result.append({
                "name": entry['name'],
                "raw_qty_7_days": f"{format_qty(t7)}{unit_part}".strip(),
                "raw_qty_15_days": f"{format_qty(t15)}{unit_part}".strip(),
                "raw_qty_30_days": f"{format_qty(t30)}{unit_part}".strip()
            })
        else:
            result.append({
                "name": entry["raw"],
                "raw_qty_7_days": "Al gusto / Variable",
                "raw_qty_15_days": "Al gusto / Variable",
                "raw_qty_30_days": "Al gusto / Variable"
            })
    
    return result

def generate_auto_shopping_list(plan_data: dict) -> list:
    """Extrae ingredientes del plan, pre-conslida las cantidades para 7, 15 y 30 días y categoriza por pasillo de supermercado."""
    logger.info("\n-------------------------------------------------------------")
    logger.debug("🛒 [AUTO-SHOPPING LIST] Consolidando ingredientes del plan para 7/15/30 días...")
    
    ingredients = []
    days = plan_data.get("days", [])
    base_plan_length = len(days)
    
    if base_plan_length == 0:
        return []
        
    for day_data in days:
        for m in day_data.get("meals", []):
            ing = m.get("ingredients", [])
            if ing:
                ingredients.extend(ing)
                
    if not ingredients:
        return []
    
    ingredients_json = _pre_consolidate_ingredients_multiday(ingredients, base_days=base_plan_length)
    logger.debug(f"🧮 [PRE-CONSOLIDACIÓN] {len(ingredients_json)} ingredientes únicos listos para LLM.")
        
    prompt = AUTO_SHOPPING_LIST_PROMPT.format(ingredients_json=json.dumps(ingredients_json, ensure_ascii=False))
    
    shopping_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.0,
        timeout=120,  # 120s por intento (Pro es un poco más lento pero más preciso)
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(ShoppingListModel)
    
    try:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
            before_sleep=lambda retry_state: print(f"⚠️  [SHOPPING] Reintento #{retry_state.attempt_number} tras error en auto-generación...")
        )
        def invoke_shopping_with_retry():
            return shopping_llm.invoke(prompt)
        
        response = invoke_shopping_with_retry()
        if hasattr(response, "items"):
            items = response.items
        elif isinstance(response, dict) and "items" in response:
            items = response["items"]
        else:
            items = []
            
        logger.info(f"✅ Se consolidaron ingredientes en {len(items)} categorías multiday.")
        logger.info("-------------------------------------------------------------\n")
        
        # Pydantic a Dict serializable si no lo es
        if items and not isinstance(items[0], dict):
            items = [item.model_dump() if hasattr(item, "model_dump") else getattr(item, "dict")() for item in items]
        
        # 🏷️ Las categorías ya están estrictamente normalizadas por el uso de Literal en ShoppingItemModel
        logger.debug(f"🏷️ [NORMALIZACIÓN ESTRICTA] Categorías finales: {set(item.get('category', '') for item in items)}")
        
        # 💊 Inyectar suplementos del plan como items de lista de compras
        seen_supps = set()
        for day_data in days:
            for supp in day_data.get("supplements") or []:
                supp_name = supp.get("name", "").strip()
                if supp_name and supp_name.lower() not in seen_supps:
                    seen_supps.add(supp_name.lower())
                    dose = supp.get("dose", "1 unidad")
                    items.append({
                        "category": "Suplementos",
                        "emoji": "💊",
                        "name": supp_name,
                        "qty_7": f"{dose} /día · 7 días",
                        "qty_15": f"{dose} /día · 15 días",
                        "qty_30": f"{dose} /día · 30 días",
                    })
        if seen_supps:
            logger.info(f"💊 [SUPLEMENTOS] {len(seen_supps)} suplementos añadidos a la lista de compras: {seen_supps}")
            
        return items
    except Exception as e:
        logger.error(f"❌ Error generando auto shopping list multiday (tras reintentos): {e}")
        # 🛡️ Fallback local: categorizar ingredientes sin LLM para no perder la lista
        logger.debug("🔄 [FALLBACK] Generando lista básica con categorización local (sin LLM)...")
        try:
            fallback_items = []
            for ing in ingredients_json:
                cat, emoji = categorize_shopping_item(ing["name"])
                fallback_items.append({
                    "category": cat,
                    "emoji": emoji,
                    "name": str(ing.get("name", "")).strip().capitalize(),
                    "qty_7": ing.get("raw_qty_7_days", ""),
                    "qty_15": ing.get("raw_qty_15_days", ""),
                    "qty_30": ing.get("raw_qty_30_days", "")
                })
            logger.debug(f"✅ [FALLBACK] {len(fallback_items)} ingredientes categorizados localmente.")
            return fallback_items
        except Exception as fallback_e:
            logger.error(f"❌ [FALLBACK] Error en categorización local: {fallback_e}")
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
    logger.info(f"🧠 [LANGGRAPH NODE] call_model")
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
            logger.debug(f"🔧 [LANGGRAPH TOOL] Ejecutando {tool_name}")
            
            if tool_name == "update_form_field":
                field = tool_args.get("field")
                new_value = tool_args.get("new_value", "")
                
                # Sanitize numeric values for the frontend response too
                if field in ['weight', 'height', 'age']:
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
                        updated_plan_record = get_latest_meal_plan_with_id(user_id if user_id and user_id != 'guest' else session_id)
                        if updated_plan_record and "plan_data" in updated_plan_record:
                            new_plan = updated_plan_record["plan_data"]
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

_generating_titles = set()

def generate_chat_title_background(user_id: str, session_id: str, first_message_text: str = None):
    """
    Se ejecuta en un thread separado. Llama a Gemini para generar el título
    y luego lo guarda en agent_messages con role='SYSTEM_TITLE'.
    """
    t0 = time.time()
    def dlog(msg):
        with open("title_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] [{session_id}] {time.time()-t0:.2f}s - {msg}\n")
    dlog("Thread started")
    if session_id in _generating_titles:
        dlog("Already generating, returning")
        return
    try:
        _generating_titles.add(session_id)
        
        # Check if a title already exists for this session
        res_data = get_session_messages(session_id)
        if res_data and any(str(m.get("content", "")).startswith("[SYSTEM_TITLE]") for m in res_data):
            dlog("Title exists, returning")
            return 
            
        first_message = ""
        if first_message_text:
            first_message = first_message_text
        else:
            first_message = "Consulta nueva"
            
        first_message = re.sub(r'\[\(Hora actual del usuario:.*?\)\]', '', first_message, flags=re.IGNORECASE|re.DOTALL)
        first_message = re.sub(r'\[Sistema:.*?\]', '', first_message, flags=re.IGNORECASE|re.DOTALL)
        first_message = re.sub(r'Instrucción:.*?$', '', first_message, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
        first_message = re.sub(r'\[IMAGE:.*?\]', '', first_message, flags=re.IGNORECASE|re.DOTALL)
        first_message = first_message.strip()
        if not first_message:
            first_message = "Interacción con imagen o sistema"
        
        dlog("Initializing LLM client")
        title_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.4, google_api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = TITLE_GENERATION_PROMPT.format(first_message=first_message)
        dlog("Calling LLM API")
        response = title_llm.invoke(prompt)
        dlog("LLM response received")
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        title = str(content).replace('"', '').replace("'", "").strip()
        
        # Strip prefijos indeseados si el LLM los generó
        lower_t = title.lower()
        if lower_t.startswith("título:"):
            title = title[7:].strip()
        elif lower_t.startswith("titulo:"):
            title = title[7:].strip()
        elif lower_t.startswith("title:"):
            title = title[6:].strip()
        
        dlog("Inserting SYSTEM_TITLE msg into DB")
        save_message(session_id, "model", f"[SYSTEM_TITLE] {title}")
        dlog("Insert successful. Finished.")
        logger.info(f"✅ Título generado para sesión {session_id}: {title}")
    except Exception as e:
        dlog(f"Exception caught: {e}")
        logger.error(f"⚠️ Error generando título: {e}")


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
        logger.info(f"⏭️ [RAG ROUTER] Mensaje casual detectado: '{prompt[:30]}' → Saltando RAG.")
        return {"skip": True}
    
    # Paso 2: Combos casuales ("ok gracias", "sí perfecto", etc.)
    words = clean.split()
    if len(words) <= 3 and all(w in casual_patterns for w in words):
        logger.info(f"⏭️ [RAG ROUTER] Combo casual detectado: '{prompt[:30]}' → Saltando RAG.")
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
            logger.info(f"⏭️ [RAG ROUTER] Flash-Lite determinó que no necesita RAG: '{prompt[:30]}'")
            return {"skip": True}
        
        logger.info(f"🎯 [RAG ROUTER] Query reescrita: '{prompt[:30]}...' → '{result}'")
        return {"skip": False, "query": result}
        
    except Exception as e:
        logger.error(f"⚠️ [RAG ROUTER] Error en rewrite, usando prompt original: {e}")
        return {"skip": False, "query": prompt}

def chat_with_agent(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None):
    
    # Obtener contexto de memoria inteligente (resúmenes + mensajes recientes)
    memory = build_memory_context(session_id)
    
    # === RAG INJECTION (con Query Routing inteligente) ===
    user_facts_text = ""
    visual_facts_text = ""
    
    if user_id:
        rag_decision = rag_query_router(prompt)
        
        if not rag_decision.get("skip"):
            try:
                
                optimized_query = rag_decision.get("query", prompt)
                
                # 1. Buscar hechos textuales con query optimizada
                logger.info(f"🔍 [CHAT RAG] Buscando con query optimizada: '{optimized_query}'")
                query_emb = get_embedding(optimized_query)
                if query_emb:
                    facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                    if facts_data:
                        fact_list = [f"• {item['fact']}" for item in facts_data]
                        user_facts_text = "\n".join(fact_list)
                        logger.info(f"🧠 [CHAT RAG] Hechos textuales recuperados: {len(facts_data)}")
                
                # 2. Buscar memoria visual
                visual_query_emb = get_multimodal_embedding(optimized_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        visual_list = [f"• {item['description']}" for item in visual_data]
                        visual_facts_text = "\n".join(visual_list)
                        logger.debug(f"📸 [CHAT RAG VISUAL] Entradas visuales recuperadas: {len(visual_data)}")
            except Exception as e:
                logger.error(f"⚠️ [CHAT RAG] Error recuperando memoria: {e}")
            
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
REGLA CRUCIAL: El plan del usuario tiene 3 opciones distintas. Llámalas SIEMPRE "Opción A", "Opción B" y "Opción C". NUNCA te refieras a ellas como "Día 1", "Día 2" o "Día 3" en tu conversación con el usuario.

REGLAS DE FORMATO VISUAL (ESTRICTAS):
1. Usa **negritas** para resaltar nombres de alimentos, cantidades (ej. **350 kcal**, **35g de proteína**) y conceptos clave.
2. Usa viñetas (`-` o `•`) SIEMPRE para listar macros, ingredientes o pasos, haciéndolo súper visual y fácil de leer.
3. Aplica saltos de línea (párrafos cortos) para que el texto respire y no sea un bloque denso."""

    now_chat = datetime.now()
    dias_chat = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses_chat = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    system_prompt += f"\n\n🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es {dias_chat[now_chat.weekday()]}, {now_chat.day} de {meses_chat[now_chat.month - 1]} de {now_chat.year}. La hora local es {now_chat.strftime('%I:%M %p')}."
    system_prompt += "\n🌟 REGLA DE CONTINUIDAD TEMPORAL PROACTIVA: Usa el día de la semana para dar sugerencias asombrosamente orgánicas, pero solo si la conversación se presta para ello. Por ejemplo:"
    system_prompt += "\n  - Si es Domingo o Lunes: Sugiere sutilmente hacer 'Meal Prep' (cocinar porciones extra) para ahorrar tiempo en la ajetreada semana laboral."
    system_prompt += "\n  - Si es Viernes o Sábado: Anímalo a disfrutar el fin de semana sin perder el control, o sugiérele ideas de comidas relajadas."
    system_prompt += "\nSé conversacional e intuitivo; no suenes como un robot leyendo el calendario, que se sienta natural."
    
    if rag_context:
        system_prompt += f"\n{rag_context}"
    
    # Determinar si es un usuario autenticado o invitado
    is_authenticated = user_id and user_id != session_id and user_id != "guest"
    
    system_prompt += f"""

TIENES HERRAMIENTAS DISPONIBLES:
- OBLIGATORIO: Usa `update_form_field` INMEDIATAMENTE y SIN EXCEPCIÓN cada vez que el usuario mencione un nuevo dato sobre sí mismo que deba actualizarse en su perfil (ej: "a partir de hoy soy vegano", "peso 80kg", "tengo diabetes", "soy intolerante a la lactosa", "no me gusta el tomate"). Si no usas esta herramienta para esos casos, la Interfaz Gráfica del usuario quedará desincronizada. ATENCIÓN: Lee atentamente los parámetros de esta herramienta, debes usar valores exactos en INGLÉS como 'lose_fat', 'vegetarian', 'male', etc. para que la UI los reconozca.
- Usa `generate_new_plan_from_chat` SOLO cuando el usuario pida explícitamente generar un plan nuevo (ej: 'hazme un plan', 'genera mi rutina', 'quiero un menú diferente'). Esta herramienta ejecuta el pipeline completo y genera un plan personalizado al instante.
- NO uses generate_new_plan_from_chat si el usuario solo da información de salud o pregunta sobre su plan actual.
- Usa `log_consumed_meal` para registrar en el diario cualquier comida que el usuario afirme haber comido. Si analizas una foto de una comida y el usuario confirma que se la comió, USA ESTA HERRAMIENTA usando los macros estimados (calorías, proteína, carbohidratos y grasas saludables), pasándolos todos a la herramienta.
- Usa `modify_single_meal` cuando el usuario pida un CAMBIO PUNTUAL a una comida específica de su plan (ej: 'cámbiale el salami al mangú por huevos en la Opción A', 'ponle más proteína al almuerzo', 'quítale el arroz a la cena de la Opción B'). Esta herramienta modifica SOLO esa comida, no regenera todo el plan. Debes identificar correctamente el day_number (1 para Opción A, 2 para Opción B, o 3 para Opción C) y el meal_type ('Desayuno', 'Almuerzo', 'Cena', 'Merienda') del plan activo del usuario. Si el usuario no especifica, asume 1 (Opción A).
- Usa `add_to_shopping_list` cuando el usuario diga que se quedó sin algo o pida añadir items. **ATENCIÓN:** Si el usuario te envía una foto o lista completa diciendo "esta es mi lista de compras actual", usa esta herramienta con `overwrite=True` para REEMPLAZAR todo. CRÍTICO: Extrae cada alimento y asígnale su CANTIDAD EXACTA unitaria (campo 'qty') y CATEGORÍA (campo 'category') (ej: name: "Avena", qty: "1 paquete", category: "Granos y Cereales"). El sistema protegerá automáticamente los ingredientes de su almuerzo.
- Usa `remove_from_shopping_list` cuando el usuario pida explícitamente quitar, remover o borrar uno o más productos de su lista de compras activa.

El user_id del usuario actual es: {user_id}"""

    if current_plan:
        system_prompt += f"\n\nCONTEXTO CRÍTICO: El usuario actualmente tiene este plan de comidas activo:\n{json.dumps(current_plan)}\n\nUsa esta información para responder con exactitud preguntas sobre lo que le toca comer hoy o sugerir cambios basados en lo que ya tiene asignado (como desayuno, almuerzo o cena)."
        
        if form_data and form_data.get("skipLunch"):
            system_prompt += "\n⚠️ IMPORTANTE SOBRE EL ALMUERZO: El plan actual NO tiene 'Almuerzo' NO porque se haya omitido y redistribuido, sino porque el usuario eligió 'Almuerzo Familiar / Ya resuelto'. Esto significa que EL USUARIO SÍ VA A ALMORZAR en su casa libremente. NUNCA digas que 'omitimos el almuerzo y redistribuimos las calorías' porque eso es falso. Dile que en realidad tiene un 'Cupo Vacío' en el plan porque reservamos las calorías para que almorzara libremente en su casa, y aliéntalo a que te cuente qué comerá para anotarlo en su registro."
        if form_data and form_data.get("includeSupplements"):
            selected_supps = form_data.get("selectedSupplements", [])
            if selected_supps:
                from constants import SUPPLEMENT_NAMES as SUPP_NAMES
                names = [SUPP_NAMES.get(s, s) for s in selected_supps]
                system_prompt += f"\n💊 SUPLEMENTOS SELECCIONADOS: El usuario toma o quiere incluir: {', '.join(names)}. Puedes referirte a ellos, dar consejos sobre timing y dosis, y responder preguntas sobre estos suplementos específicos."
            else:
                system_prompt += "\n💊 SUPLEMENTOS ACTIVOS: El usuario activó la opción de incluir suplementos en su plan. Su plan incluye recomendaciones de suplementos personalizados. Puedes referirte a ellos, dar consejos sobre timing y dosis, y responder preguntas sobre suplementación."

    if memory.get('summary_context'):
        system_prompt += f"\n\n<contexto_evolutivo_historico>\n{memory['summary_context']}\n</contexto_evolutivo_historico>"
    
    # Inyectar contexto del diario del día (paridad con stream)
    if user_id and user_id != "guest":
        try:
            consumed_today = get_consumed_meals_today(user_id)
            if consumed_today:
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}."
            else:
                system_prompt += "\n\nDIARIO DE HOY: El usuario no ha registrado ninguna comida el día de hoy todavía."
        except Exception as e:
            logger.error(f"⚠️ Error inyectando contexto de diario (non-stream): {e}")
        
    config = {"configurable": {"thread_id": session_id}}
    
    # Compilamos el grafo dinámicamente para usar la conexión compartida/pool en un entorno multi-worker
    if connection_pool:
        checkpointer = PostgresSaver(connection_pool)
        chat_graph_app = chat_builder.compile(checkpointer=checkpointer)
    else:
        logger.warning("⚠️ [LangGraph] No pool de PostgreSQL, usando MemorySaver en RAM.")
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
        logger.debug(f"🔄 [LANGGRAPH] Inicializando nuevo thread O restaurando tras reinicio para session_id: {session_id}")
        messages = []
        for msg in memory["recent_messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "model":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        inputs["messages"] = messages
    else:
        logger.debug(f"🔄 [LANGGRAPH] Thread existente detectado en Checkpointer. Inyectando solo el prompt actual.")
        inputs["messages"] = [HumanMessage(content=prompt)]
        
    logger.info("\n-------------------------------------------------------------")
    logger.info("⏳ [CHAT] LangGraph ejecutando pipeline...")
    start_time = time.time()
    
    final_state = chat_graph_app.invoke(inputs, config=config)
    
    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    logger.info(f"✅ [COMPLETADO] LangGraph finalizó en {duration_secs} segundos.")
    logger.info("-------------------------------------------------------------\n")
    
    final_messages = final_state["messages"]
    last_msg = final_messages[-1]
    content = last_msg.content
    
    if isinstance(content, list):
        content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])
        
    return str(content), final_state.get("updated_fields", {}), final_state.get("new_plan")

from typing import Generator

def chat_with_agent_stream(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None, local_date: Optional[str] = None, tz_offset: Optional[int] = None) -> Generator[str, None, None]:
    """Generador síncrono de chat que emite eventos del modelo y herramientas mediante SSE (JSONlines).
    FastAPI ejecuta esto en un threadpool externo, liberando el Event Loop para concurrencia real."""
    memory = build_memory_context(session_id)
    
    # RAG INJECTION (con Query Routing inteligente)
    user_facts_text = ""
    visual_facts_text = ""
    if user_id:
        rag_decision = rag_query_router(prompt)
        
        if not rag_decision.get("skip"):
            try:
                
                optimized_query = rag_decision.get("query", prompt)
                
                query_emb = get_embedding(optimized_query)
                if query_emb:
                    facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                    if facts_data:
                        fact_list = [f"• {item['fact']}" for item in facts_data]
                        user_facts_text = "\n".join(fact_list)
                
                visual_query_emb = get_multimodal_embedding(optimized_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        visual_list = [f"• {item['description']}" for item in visual_data]
                        visual_facts_text = "\n".join(visual_list)
            except Exception as e:
                logger.error(f"⚠️ [CHAT RAG] Error en stream: {e}")
            
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

    now_chat = datetime.now()
    dias_chat = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses_chat = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    system_prompt += f"\n\n🕒 CONTEXTO TEMPORAL ACTUAL: Hoy es {dias_chat[now_chat.weekday()]}, {now_chat.day} de {meses_chat[now_chat.month - 1]} de {now_chat.year}. La hora local es {now_chat.strftime('%I:%M %p')}."
    system_prompt += "\n🌟 REGLA DE CONTINUIDAD TEMPORAL PROACTIVA: Usa el día de la semana para dar sugerencias asombrosamente orgánicas, pero solo si la conversación se presta para ello. Por ejemplo:"
    system_prompt += "\n  - Si es Domingo o Lunes: Sugiere sutilmente hacer 'Meal Prep' (cocinar porciones extra) para ahorrar tiempo en la ajetreada semana laboral."
    system_prompt += "\n  - Si es Viernes o Sábado: Anímalo a disfrutar el fin de semana sin perder el control, o sugiérele ideas de comidas relajadas."
    system_prompt += "\nSé conversacional e intuitivo; no suenes como un robot leyendo el calendario, que se sienta natural."
    
    if rag_context: system_prompt += f"\n{rag_context}"
    
    system_prompt += f"""
TIENES HERRAMIENTAS DISPONIBLES:
- Usa `update_form_field` INMEDIATAMENTE al haber nuevos datos de perfil. IMPORTANTE: Revisa los valores permitidos, la UI usa nombres clave (ej: 'lose_fat', 'vegetarian', 'male').
- Usa `generate_new_plan_from_chat` SOLO cuando el usuario pida explícitamente generar un plan nuevo.
- Usa `log_consumed_meal` para registrar en el diario cualquier comida consumida.
- Usa `modify_single_meal` para cambios puntuales.
- Usa `add_to_shopping_list` para añadir compras, o con `overwrite=True` si envía toda su despensa actual. CRÍTICO: Extrae cada alimento, su 'qty' y su 'category' de cada elemento JSON (ej: name: 'Avena', qty: '1 paquete', category: 'Frutas y Vegetales'). El sistema salvará automáticamente los ingredientes del almuerzo.
- Usa `remove_from_shopping_list` para remover puntualmente items de su lista de compras.
El user_id actual es: {user_id}"""

    if current_plan:
        system_prompt += f"\nCONTEXTO CRÍTICO: Plan activo:\n{json.dumps(current_plan)}\n"
        
        if form_data and form_data.get("skipLunch"):
            system_prompt += "⚠️ IMPORTANTE SOBRE EL ALMUERZO: El usuario escogió 'Almuerzo Familiar', EL USUARIO SÍ VA A ALMORZAR. NO digas que se omitió y redistribuyó. Dile que le dejaste un 'Cupo Vacío' y coméntale que te dicte qué almorzó para registrarlo.\n"
        if form_data and form_data.get("includeSupplements"):
            selected_supps = form_data.get("selectedSupplements", [])
            if selected_supps:
                from constants import SUPPLEMENT_NAMES as SUPP_NAMES
                names = [SUPP_NAMES.get(s, s) for s in selected_supps]
                system_prompt += f"💊 SUPLEMENTOS SELECCIONADOS: El usuario toma o quiere incluir: {', '.join(names)}. Puedes referirte a ellos, dar consejos sobre timing y dosis, y responder preguntas sobre estos suplementos específicos.\n"
            else:
                system_prompt += "💊 SUPLEMENTOS ACTIVOS: El usuario activó la opción de incluir suplementos en su plan. Su plan incluye recomendaciones de suplementos personalizados. Puedes referirte a ellos, dar consejos sobre timing y dosis, y responder preguntas sobre suplementación.\n"

    if memory.get('summary_context'):
        system_prompt += f"\n\n<contexto_evolutivo_historico>\n{memory['summary_context']}\n</contexto_evolutivo_historico>"
        
    if user_id and user_id != "guest":
        try:
            consumed_today = get_consumed_meals_today(user_id, date_str=local_date, tz_offset_mins=tz_offset)
            if consumed_today:
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}. Revisa esto ANTES de preguntar si ya comió algo (por ejemplo, si ya tiene una cena registrada, no le preguntes si esa foto es su cena, asume que es un snack nocturno o pregúntale por qué repite). Si la foto o mensaje coincide con algo que ya está registrado, felicítalo o no lo registres de nuevo."
            else:
                system_prompt += "\n\nDIARIO DE HOY: El usuario no ha registrado ninguna comida el día de hoy todavía."
        except Exception as e:
            logger.error(f"⚠️ Error inyectando contexto de diario: {e}")
            
    config = {"configurable": {"thread_id": session_id}}
    
    # Compilamos usando PostgresSaver sincrónico porque astream_events nativo asíncrono tiene problemas en Windows
    if connection_pool:
        checkpointer = PostgresSaver(connection_pool)
        chat_graph_app = chat_builder.compile(checkpointer=checkpointer)
    else:
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
        
    def get_progress_msg(msg_type):
        opts = {
            "analizando": ["Procesando tu solicitud detalladamente...", "Evaluando tu perfil y macros...", "Alineando tu genética con el plan...", "Analizando tu objetivo con Inteligencia Nutricional...", "Revisando tus preferencias y contexto..."],
            "generando_plan": ["Armando la química perfecta de tus comidas...", "Diseñando un plan de alimentación premium...", "Calculando macros y esculpiendo tu dieta...", "Generando distribución óptima de nutrientes..."],
            "modificando_comida": ["Ajustando la receta a tus exigencias...", "Reemplazando ingredientes inteligentemente...", "Rediseñando esta comida sin perder tus macros...", "Aplicando cambios culinarios a tu plato..."],
            "actualizando_bd": ["Guardando tus preferencias en el sistema...", "Sincronizando perfil con tu base de datos...", "Actualizando tu historial clínico nutricional..."],
            "registrando_progreso": ["Inscribiendo tu ingesta en el registro diario...", "Contabilizando calorías y macros consumidos...", "Actualizando tu impacto metabólico del día..."]
        }
        return random.choice(opts.get(msg_type, ["Procesando..."]))

    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('analizando')})}\n\n"
    
    logger.info(f"⏳ [CHAT STREAM] LangGraph iniciando astream nativo para {session_id}...")
    
    final_state_snapshot = None
    
    try:
        for event in chat_graph_app.stream(inputs, config=config, stream_mode="messages"):
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
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('generando_plan')})}\n\n"
                                elif tool_name == "modify_single_meal":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('modificando_comida')})}\n\n"
                                elif tool_name == "update_form_field":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('actualizando_bd')})}\n\n"
                                elif tool_name == "log_consumed_meal":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('registrando_progreso')})}\n\n"
                                    
    except Exception as e:
        logger.error(f"❌ [CHAT STREAM] Error en astream nativo: {e}")
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return
        
    # Obtener el estado final actualizado
    try:
        final_state_snapshot = chat_graph_app.get_state(config)
    except Exception as e:
        logger.error(f"⚠️ Error obteniendo get_state tras stream: {e}")

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

    logger.info("✅ [CHAT STREAM] Finalizado con éxito.")
    yield f"data: {json.dumps({'type': 'done', 'response': final_content, 'updated_fields': updated_fields, 'new_plan': new_plan})}\n\n"