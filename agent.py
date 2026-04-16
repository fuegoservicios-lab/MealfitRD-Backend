# backend/agent.py

import os
import logging
import time
import json
import re
import unicodedata
logger = logging.getLogger(__name__)

from constants import strip_accents, CULINARY_KNOWLEDGE_BASE, validate_ingredients_against_pantry
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
from memory_manager import build_memory_context
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from langgraph.checkpoint.postgres import PostgresSaver
from db import get_user_ingredient_frequencies, get_latest_meal_plan_with_id, get_session_messages, save_message, search_user_facts, search_visual_diary, connection_pool, get_consumed_meals_today
from dotenv import load_dotenv

load_dotenv()

from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel
from prompts import (
    DETERMINISTIC_VARIETY_PROMPT, SWAP_MEAL_PROMPT_TEMPLATE, 
    CHAT_SYSTEM_PROMPT_BASE, CHAT_STREAM_SYSTEM_PROMPT_BASE,
    TITLE_GENERATION_PROMPT, RAG_ROUTER_PROMPT
)
from prompts.chat_agent import (
    CHAT_AGENT_INLINE_PROMPT,
    CHAT_VOICE_MODE_PROMPT,
    CHAT_STREAM_INLINE_PROMPT,
    build_temporal_context,
    build_circadian_context,
    build_temporal_proactive_context,
    build_tools_instructions,
    build_tools_instructions_stream,
    build_inventory_context,
)

from tools import (
    update_form_field, generate_new_plan_from_chat,
    log_consumed_meal, modify_single_meal,
    search_deep_memory, agent_tools, analyze_preferences_agent,
    execute_generate_new_plan, execute_modify_single_meal,
    check_current_pantry
)

# Langchain Chat Model Initialization
# Safety settings relajados: esta es una app de nutrición clínica donde los usuarios
# hablan sobre hábitos alimenticios y emociones — los filtros por defecto bloquean falsamente.
from google.genai.types import HarmCategory, HarmBlockThreshold

_safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.2,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    safety_settings=_safety_settings
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
    
    # --- REGLA CRÍTICA: ROTACIÓN CON INGREDIENTES EXISTENTES (ZERO-TRUST) ---
    clean_ingredients = []
    user_id = form_data.get("user_id")
    
    # Intento Primario: Extraer ingredientes directamente del plan activo en BD
    if user_id and user_id != "guest":
        try:
            from db_plans import get_latest_meal_plan_with_id
            plan_record = get_latest_meal_plan_with_id(user_id)
            if plan_record and "plan_data" in plan_record:
                from db_facts import get_consumed_meals_since
                from shopping_calculator import get_realtime_pantry
                
                plan_created_at = plan_record.get("created_at")
                consumed_ingredients = []
                if plan_created_at:
                    consumed_meals_list = get_consumed_meals_since(user_id, plan_created_at)
                    for cm in consumed_meals_list:
                        ings = cm.get("ingredients") or []
                        if isinstance(ings, list):
                            consumed_ingredients.extend(ings)
                
                clean_ingredients = get_realtime_pantry(plan_record["plan_data"], consumed_ingredients)
        except Exception as e:
            logger.error(f"⚠️ [SWAP_MEAL] Error extrayendo inventario desde BD: {e}")

    # Fallback: Usar lista enviada por el front si falló BD o es guest
    if not clean_ingredients:
        current_pantry_ingredients = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
        if current_pantry_ingredients and isinstance(current_pantry_ingredients, list) and len(current_pantry_ingredients) > 0:
            from shopping_calculator import aggregate_shopping_list
            clean_ingredients = aggregate_shopping_list([item.strip() for item in current_pantry_ingredients if item and isinstance(item, str) and len(item) > 2])
            
    if clean_ingredients:
        context_extras += f"\n    - ⚠️ REGLA DE RECICLAJE (ROTACIÓN DE DESPENSA): El usuario quiere cambiar este plato pero DEBES utilizar ingredientes que ya estén en su lista actual. Ingredientes disponibles: {', '.join(clean_ingredients)}. Tienes permiso creativo para proponer un plato usando solo esta base, sin agregar ingredientes foráneos."
    else:
        logger.warning(
            f"⚠️ [SWAP_MEAL] GUARDRAIL BYPASS — Sin despensa detectada | "
            f"user_id={user_id or 'guest'} | "
            f"bd_attempted={bool(user_id and user_id != 'guest')} | "
            f"frontend_list_size={len(form_data.get('current_pantry_ingredients', []))} | "
            f"mode=FREE_GENERATION"
        )


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
            context_extras += f"\n    - 💡 SUGERENCIA DE VARIEDAD: Para este swap, intenta usar {', '.join(suggestions)} (o ingredientes radicalmente diferentes al rechazado)."
            logger.debug(f"🎲 [SWAP ANTI MODE-COLLAPSE] Sugerencias: {suggestions}")
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
    
    temp = 0.3
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
        before_sleep=lambda retry_state: logger.warning(
            f"🔁 [SWAP RETRY] attempt={retry_state.attempt_number} | "
            f"reason=pantry_guardrail_rejection | meal_type={meal_type}"
        )
    )
    def invoke_with_retry():
        res = swap_llm.invoke(prompt_text)
        
        # Validación post-generación (guardrail determinista)
        if hasattr(res, "ingredients"):
            ingreds = getattr(res, "ingredients")
        elif isinstance(res, dict) and "ingredients" in res:
            ingreds = res["ingredients"]
        else:
            ingreds = []
            
        # Solo aplicamos restricción estricta si hay una despensa base limpia extraída
        if clean_ingredients:
            val_result = validate_ingredients_against_pantry(ingreds, clean_ingredients)
            if val_result is not True:
                logger.warning(val_result)
                raise ValueError(val_result)
                
        return res
    
    try:
        response = invoke_with_retry()
    except Exception as e:
        logger.error(f"❌ [SWAP_MEAL] Fallaron los intentos LLM y validador: {e}. Usando Plato Fallback.")
        fallback_ing = clean_ingredients[:4] if clean_ingredients else ["Pollo", "Arroz", "Aguacate"]
        response = {
            "name": f"Opción Segura: {' y '.join(fallback_ing[:2]).title()}",
            "desc": "Este plato fue autogenerado como medida de seguridad para garantizar una opción con ingredientes que ya posees.",
            "ingredients": fallback_ing,
            "recipe": [
                "Mise en place: Prepara de manera básica los ingredientes de la nevera.",
                "El Toque de Fuego: Cocina saludablemente a la plancha o al vapor.",
                "Montaje: Sirve porciones adecuadas según tu objetivo y disfruta."
            ],
            "cals": target_calories or 450,
            "protein": round((target_calories or 450) * 0.3 / 4),
            "carbs": round((target_calories or 450) * 0.4 / 4),
            "fats": round((target_calories or 450) * 0.3 / 9)
        }
        # Fake retries for the logging metric below
        if not hasattr(invoke_with_retry, 'retry'):
            invoke_with_retry.retry = type('obj', (object,), {'statistics': {'attempt_number': 3}})
    
    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    # Observabilidad: cuántos reintentos se usaron
    retries_used = invoke_with_retry.retry.statistics.get("attempt_number", 1) if hasattr(invoke_with_retry, 'retry') else 1
    logger.info(f"✅ [COMPLETADO] Nueva alternativa {meal_type} generada en {duration_secs}s | retries_used={retries_used}")
    logger.info("-------------------------------------------------------------\n")
    if hasattr(response, "model_dump"):
        return getattr(response, "model_dump")()
    elif isinstance(response, dict):
        return response
    elif hasattr(response, "dict"):
        return getattr(response, "dict")()
    else:
        raise ValueError("El modelo de IA generó una respuesta inválida. Por favor, reintenta.")







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
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        safety_settings=_safety_settings
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
                        tool_result = "El plan de comidas de 7 días fue generado exitosamente. Dile al usuario que lo revise en su dashboard."
                except Exception:
                    pass
                    
            elif tool_name == "modify_single_meal":
                user_id = state.get("user_id")
                session_id = state.get("session_id")
                form_data = state.get("form_data", {})
                
                tool_result = execute_modify_single_meal(
                    user_id=user_id if user_id and user_id != 'guest' else session_id,
                    day_number=tool_args.get("day_number", 1),
                    meal_type=tool_args.get("meal_type", "Desayuno"),
                    changes=tool_args.get("changes", ""),
                    form_data=form_data,
                    allow_pantry_expansion=tool_args.get("allow_pantry_expansion", False)
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
        # Garantizar que siempre se use el primer mensaje histórico real, no el prompt actual
        if res_data:
            for m in res_data:
                msg_role = str(m.get("role", "")).lower()
                if msg_role == "user" or msg_role == "human":
                    first_message = m.get("content", "")
                    break
                    
        if not first_message and first_message_text:
            first_message = first_message_text
        elif not first_message:
            first_message = "Consulta nueva"
            
        first_message = re.sub(r'\[?\(Hora actual del usuario:[^)]*\)\]?', '', first_message, flags=re.IGNORECASE|re.DOTALL)
        first_message = re.sub(r'\[Sistema:[^\]]*\]', '', first_message, flags=re.IGNORECASE)
        first_message = re.sub(r'Instrucción:.*?$', '', first_message, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
        first_message = re.sub(r'\[IMAGE:[^\]]*\]', '', first_message, flags=re.IGNORECASE)
        first_message = re.sub(r'Mensaje del usuario:\s*', '', first_message, flags=re.IGNORECASE|re.DOTALL)
        
        if '[El usuario subió una imagen.' in first_message:
            first_message = re.sub(r'\[El usuario subió una imagen\..+?\]', '', first_message, flags=re.DOTALL)
            
        first_message = first_message.strip()
        if not first_message:
            first_message = "El usuario acaba de subir una fotografía (probablemente de su comida o progreso físico) para ser analizada."
        
        dlog("Initializing LLM client")
        
        # Obtener títulos recientes para evitar repetirlos
        used_titles_str = ""
        try:
            from db import get_user_chat_sessions
            recent = get_user_chat_sessions(user_id)
            if recent:
                used = [str(s.get("title")) for s in recent[:15] if s.get("title") and str(s.get("title")) not in ["Nuevo chat", "Nuevo Chat"]]
                used_titles_str = ", ".join(list(set(used)))
        except Exception as e:
            logger.error(f"Error fetching recent titles for anti-duplication: {e}")
            
        title_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.7, google_api_key=os.environ.get("GEMINI_API_KEY"))
        prompt = TITLE_GENERATION_PROMPT.format(first_message=first_message, used_titles=used_titles_str)
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
            
        # Hard limit para evitar que rompa la UI
        if len(title) > 32:
            title = title[:32]
            # Truncar amablemente hasta el último espacio para no dejar palabras a medias
            if " " in title:
                title = title.rsplit(" ", 1)[0]
        
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

    system_prompt = CHAT_AGENT_INLINE_PROMPT

    system_prompt += build_temporal_context()

    schedule_type = form_data.get("scheduleType", "standard") if form_data else "standard"
    system_prompt += build_circadian_context(schedule_type)

    system_prompt += build_temporal_proactive_context()
    
    system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
    
    if rag_context:
        system_prompt += f"\n{rag_context}"
    
    # Determinar si es un usuario autenticado o invitado
    is_authenticated = user_id and user_id != session_id and user_id != "guest"
    
    system_prompt += build_tools_instructions(user_id)

    inventory_str = ""
    shopping_delta_str = ""
    
    if user_id and user_id != "guest":
        try:
            from db_inventory import get_user_inventory
            user_phys_inv = get_user_inventory(user_id)
            if user_phys_inv:
                inventory_str = ", ".join(user_phys_inv)
                
            from db_plans import get_latest_meal_plan_with_id
            plan_record = get_latest_meal_plan_with_id(user_id)
            if plan_record and "plan_data" in plan_record:
                from shopping_calculator import get_shopping_list_delta
                delta_list = get_shopping_list_delta(user_id, plan_record["plan_data"], is_new_plan=False)
                if delta_list:
                    shopping_delta_str = ", ".join(delta_list)
        except Exception as e:
            logger.error(f"⚠️ Error extrayendo inventario y delta para system_prompt: {e}")

    # Fallbacks
    if not inventory_str and form_data:
        current_pantry = form_data.get("current_pantry_ingredients", [])
        if current_pantry and isinstance(current_pantry, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_pantry = aggregate_shopping_list([item.strip() for item in current_pantry if isinstance(item, str) and len(item.strip()) > 2])
            inventory_str = ", ".join(cleaned_pantry)

    if not shopping_delta_str and form_data:
        current_shopping = form_data.get("current_shopping_list", [])
        if current_shopping and isinstance(current_shopping, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_shop = aggregate_shopping_list([item.strip() for item in current_shopping if isinstance(item, str) and len(item.strip()) > 2])
            shopping_delta_str = ", ".join(cleaned_shop)

    system_prompt += build_inventory_context(inventory_str, shopping_delta_str)

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
from sentiment_classifier import classify_sentiment

def chat_with_agent_stream(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None, local_date: Optional[str] = None, tz_offset: Optional[int] = None, is_call_mode: bool = False, plan_tier: str = "gratis") -> Generator[str, None, None]:
    """Generador síncrono de chat que emite eventos del modelo y herramientas mediante SSE (JSONlines).
    FastAPI ejecuta esto en un threadpool externo, liberando el Event Loop para concurrencia real."""
    memory = build_memory_context(session_id)
    
    # 🎭 ANÁLISIS DE SENTIMIENTO ADAPTATIVO (Solo Plus o superior)
    sentiment_result = {}
    if plan_tier in ["plus", "ultra", "admin"]:
        sentiment_result = classify_sentiment(prompt)
    
    # RAG INJECTION (con Query Routing inteligente)
    user_facts_text = ""
    visual_facts_text = ""
    if user_id and plan_tier in ["basic", "plus", "ultra", "admin"]:
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

    system_prompt = CHAT_STREAM_INLINE_PROMPT

    if is_call_mode:
        system_prompt = CHAT_VOICE_MODE_PROMPT

    system_prompt += build_temporal_context()
    
    schedule_type = form_data.get("scheduleType", "standard") if form_data else "standard"
    system_prompt += build_circadian_context(schedule_type)

    system_prompt += build_temporal_proactive_context()
    
    # 🎭 Inyectar personalidad adaptativa basada en el sentimiento detectado
    if sentiment_result.get("instruction"):
        system_prompt += f"\n\n{sentiment_result['instruction']}"
    
    system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
    
    if rag_context: system_prompt += f"\n{rag_context}"
    
    system_prompt += build_tools_instructions_stream(user_id)

    inventory_str = ""
    shopping_delta_str = ""
    
    if user_id and user_id != "guest":
        try:
            from db_inventory import get_user_inventory
            user_phys_inv = get_user_inventory(user_id)
            if user_phys_inv:
                inventory_str = ", ".join(user_phys_inv)
                
            from db_plans import get_latest_meal_plan_with_id
            plan_record = get_latest_meal_plan_with_id(user_id)
            if plan_record and "plan_data" in plan_record:
                from shopping_calculator import get_shopping_list_delta
                delta_list = get_shopping_list_delta(user_id, plan_record["plan_data"], is_new_plan=False)
                if delta_list:
                    shopping_delta_str = ", ".join(delta_list)
        except Exception as e:
            logger.error(f"⚠️ Error extrayendo inventario y delta para system_prompt: {e}")

    # Fallbacks
    if not inventory_str and form_data:
        current_pantry = form_data.get("current_pantry_ingredients", [])
        if current_pantry and isinstance(current_pantry, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_pantry = aggregate_shopping_list([item.strip() for item in current_pantry if isinstance(item, str) and len(item.strip()) > 2])
            inventory_str = ", ".join(cleaned_pantry)

    if not shopping_delta_str and form_data:
        current_shopping = form_data.get("current_shopping_list", [])
        if current_shopping and isinstance(current_shopping, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_shop = aggregate_shopping_list([item.strip() for item in current_shopping if isinstance(item, str) and len(item.strip()) > 2])
            shopping_delta_str = ", ".join(cleaned_shop)

    system_prompt += build_inventory_context(inventory_str, shopping_delta_str)

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
            "registrando_progreso": ["Inscribiendo tu ingesta en el registro diario...", "Contabilizando calorías y macros consumidos...", "Actualizando tu impacto metabólico del día..."],
            "calculando_compras": ["Calculando tu lista de compras matemáticamente...", "Sumando ingredientes de todas las opciones...", "Consolidando cantidades exactas para el súper..."],
            "buscando_memoria": ["Explorando tu historial profundo...", "Recuperando recuerdos de tus experiencias pasadas...", "Buscando en tu archivo de memoria a largo plazo..."]
        }
        return random.choice(opts.get(msg_type, ["Procesando..."]))

    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('analizando')})}\n\n"
    
    # Emitir el sentimiento detectado al frontend
    if sentiment_result.get("sentiment") != "neutral":
        yield f"data: {json.dumps({'type': 'sentiment', 'sentiment': sentiment_result.get('sentiment'), 'personality': sentiment_result.get('name'), 'emoji': sentiment_result.get('emoji')})}\n\n"
    
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
                                elif tool_name == "check_shopping_list":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('calculando_compras')})}\n\n"
                                elif tool_name == "search_deep_memory":
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('buscando_memoria')})}\n\n"
                                    
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