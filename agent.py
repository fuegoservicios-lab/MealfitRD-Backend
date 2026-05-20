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
from knobs import _env_str, _env_float  # [P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15] / [P0-CHAT-LLM-TIMEOUT · 2026-05-19] auto-registry
# [P1-CHAT-CB · 2026-05-19] Breaker per-modelo del graph_orchestrator. NO
# duplicamos la implementación — reusamos el singleton + knobs ya productivos
# (`MEALFIT_CB_FAILURE_THRESHOLD=3`, `MEALFIT_CB_RESET_TIMEOUT_S=30`). Import
# de un solo nivel: `graph_orchestrator` NO importa `agent` (verificado), no
# hay ciclo. Si en el futuro la dirección de import cambia, mover el helper
# a un módulo neutro.
from graph_orchestrator import _get_circuit_breaker
import concurrent.futures
import traceback
from datetime import datetime, timezone
from cpu_tasks import _calcular_frecuencias_regex_cpu_bound
from memory_manager import build_memory_context
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from langgraph.checkpoint.postgres import PostgresSaver
from db import get_user_ingredient_frequencies, get_latest_meal_plan_with_id, get_session_messages, save_message, search_user_facts, search_visual_diary, connection_pool, chat_checkpoint_pool, get_consumed_meals_today
from dotenv import load_dotenv

load_dotenv()

# [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad output server-side.
# El frontend renderiza chat content via LazyMarkdown + rehype-sanitize
# (P1-MARKDOWN-SANITIZE), que escapa tags peligrosos y event handlers en
# el árbol de DOM. Acá añadimos una segunda capa SERVER-SIDE: si
# rehype-sanitize falla por bug, regresión, dep maliciosa, o un caller
# futuro renderiza el contenido con `dangerouslySetInnerHTML`, las
# etiquetas más peligrosas siguen neutralizadas en el wire.
#
# Conservador: solo escapa tags que NUNCA deberían aparecer en respuestas
# legítimas del LLM (script/iframe/object/embed/style/base/link/meta/
# form/svg/math). NO usa `bleach` para evitar la dep y porque el LLM
# legítimamente emite tags como <details>, <sup>, <sub> que un bleach
# strict eliminaría rompiendo el formato markdown.
#
# También neutraliza event handlers `on*=...` y URIs `javascript:` —
# vectores XSS clásicos. Los reemplazos (`data-stripped-*`) son texto
# inocuo que NO ejecuta nada y deja un audit trail visible en el DOM si
# alguna vez ocurre — facilita diagnosticar prompt injection attempts
# en producción.
_DANGEROUS_HTML_TAG_RE = re.compile(
    r"<(?P<slash>/?)(?P<tag>script|iframe|object|embed|style|base|link|meta|form|svg|math)\b",
    re.IGNORECASE,
)
_ON_HANDLER_RE = re.compile(r"\bon([A-Za-z]+)\s*=", re.IGNORECASE)
_JS_URI_RE = re.compile(r"\b(href|src)\s*=\s*([\"']?)\s*javascript:", re.IGNORECASE)


def _sanitize_chat_output_for_wire(text):
    """Defensa-en-profundidad: neutraliza tags HTML peligrosas + event
    handlers en output del chat antes de enviarlo al wire SSE. NO toca
    markdown legítimo (headings, listas, blockquotes, code blocks).

    Retorna el input intacto si no es str (None, dict, etc) — los callers
    asumen que el helper es safe to wrap cualquier value.
    """
    if not text or not isinstance(text, str):
        return text
    text = _DANGEROUS_HTML_TAG_RE.sub(r"&lt;\g<slash>\g<tag>", text)
    text = _ON_HANDLER_RE.sub(r"data-stripped-on\1=", text)
    text = _JS_URI_RE.sub(r"\1=\2data-stripped:", text)
    return text


# [P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20] Helper para remover los tags
# silentes `[UI_ACTION: <NAME>]` ANTES de persistir el response del agente
# en `agent_messages.content`. Cubre REFRESH_PLAN, REFRESH_INVENTORY,
# REFRESH_HYDRATION (y cualquier futuro UI_ACTION declarado en el system
# prompt `prompts/chat_agent.py:126-130`).
#
# Por qué server-side:
#   El frontend (AgentPage.jsx) ya hace strip + dispatch durante el SSE
#   streaming Y en el evento `done`. Pero el backend persiste el
#   `response_text` RAW en `agent_messages.content`. Cuando el frontend
#   refetchea `GET /api/chat/history/<session_id>` (al recargar el chat
#   o navegar de vuelta), trae el contenido con tag → re-renderiza
#   visible. Bug reportado 2026-05-20: el user vio el tag desaparecer
#   durante el streaming y reaparecer al final/refetch.
#
# Patrón regex: `\[UI_ACTION:\s*[A-Z_]+\]` cubre todos los actions
# documentados sin tener que enumerarlos individualmente. case-insensitive
# por defensa (el LLM podría variar mayúsculas).
#
# Tooltip-anchor: P1-CHAT-UI-ACTION-INVENTORY.
_UI_ACTION_TAG_RE = re.compile(r"\[UI_ACTION:\s*[A-Z_]+\]", re.IGNORECASE)


def strip_ui_action_tags_for_persist(text):
    """[P1-CHAT-UI-ACTION-INVENTORY · 2026-05-20] Remueve tags silentes
    `[UI_ACTION: <NAME>]` del response del agente antes de persistirlo.
    Idempotente; safe-to-wrap cualquier value (None/dict pasan intactos)."""
    if not text or not isinstance(text, str):
        return text
    cleaned = _UI_ACTION_TAG_RE.sub("", text)
    # Collapse blank lines surplus que pueden quedar tras strip.
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


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

# [P2-AUDIT-1 · 2026-05-15] Knobs para overridear los modelos Gemini usados
# por las 5 callsites de `ChatGoogleGenerativeAI(...)` en este módulo:
#   - `llm` (módulo-level, swap/chat default)            → MEALFIT_CHAT_AGENT_MODEL
#   - `swap_llm` dentro de `swap_meal`                   → MEALFIT_CHAT_AGENT_SWAP_MODEL
#   - `chat_llm` dentro de `call_model` (LangGraph node) → MEALFIT_CHAT_AGENT_MODEL (reusa)
#   - `title_llm` dentro de `generate_session_title`     → MEALFIT_CHAT_TITLE_MODEL
#   - `router_llm` dentro de `rag_query_router`          → MEALFIT_CHAT_ROUTER_MODEL
#
# Por qué un knob (no hardcode): convención del repo `P3-PREVIEW-MODEL-KNOB
# · 2026-05-12` (CLAUDE.md). Los modelos `*-preview` de Google pueden
# deprecarse/retirarse sin aviso prolongado — incidente real 2026-05-11
# documentó CB rows stale por el modelo `gemini-3.1-pro-preview` 4.4 días
# seguidos. Sin knob, swap del modelo requiere redeploy. Knob permite swap
# inmediato sin redeploy: setear `MEALFIT_CHAT_AGENT_MODEL=gemini-3.1-flash`
# (stable, sin `-preview`) y reiniciar el worker.
#
# Defaults = current production models. Cambiar en env vars cuando Google
# publique notice de deprecation o cuando se quiera A/B test un modelo nuevo.
# Precedente en `proactive_agent.py:36` (`_proactive_model_name`).
#
# [P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15] Los 4 helpers leen via
# `_env_str(...)` (NO `os.environ.get`) para auto-registrarse en
# `_KNOBS_REGISTRY` (convención P3-NEW-D). Beneficio operacional: tras un
# `MEALFIT_CHAT_AGENT_MODEL=gemini-3.1-flash` en EasyPanel, el SRE puede
# verificar el cambio via `GET /api/system/admin/knobs` sin releer source.
# Test parser-based: `tests/test_p3_chat_model_knobs_registry.py`.
def _chat_agent_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_AGENT_MODEL",
        "gemini-3.5-flash",
    )

def _chat_agent_swap_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_AGENT_SWAP_MODEL",
        "gemini-3.5-flash",
    )

def _chat_title_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_TITLE_MODEL",
        "gemini-3.1-flash-lite",
    )

def _chat_router_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_ROUTER_MODEL",
        "gemini-3.1-flash-lite",
    )

# [P0-CHAT-LLM-TIMEOUT · 2026-05-19] Timeouts per-LLM-invoke y graph-total.
# Pre-fix: las 5 callsites de `ChatGoogleGenerativeAI(...)` se construían SIN
# `timeout=`. Resultado: si Gemini se colgaba (sobrecarga, red, quota silenciosa
# del provider), `*.invoke(...)` bloqueaba indefinidamente el worker thread del
# threadpool de FastAPI. Bajo concurrencia moderada → thread pool starvation.
# Es exactamente el modo de fallo que el resto del repo defiende con knobs
# `MEALFIT_CB_*` pero acá no se invocaba.
#
# Fix: el constructor de `ChatGoogleGenerativeAI` acepta `timeout=` (segundos)
# que propaga al gRPC `request_options.timeout`. Cualquier .invoke() que
# exceda raises (DeadlineExceeded/TimeoutError) — captura el catch de
# Exception del SSE generator (línea 1228-1235) o el del wrap concurrent.futures
# del `chat_graph_app.invoke` en `chat_with_agent` (non-streaming).
#
# Defaults eligen ventanas reales:
#   - LLM principal (chat/call_model): 15s. Conversaciones típicas <5s, p95 <10s.
#   - SWAP: 30s. Tiene retry tenacity 3x con wait_exponential(min=2,max=8) →
#     budget per-call más holgado para no abortar antes de retry.
#   - TITLE: 10s. Mensaje corto, una sola invocación.
#   - ROUTER (RAG decision): 8s. Flash-Lite, una sola invocación, sin retry.
#   - GRAPH TOTAL (non-streaming): 60s. Cubre call_model + execute_tools +
#     call_model (formateo de respuesta) con margen para tool roundtrips
#     legítimos (e.g. `generate_new_plan_from_chat` invoca pipeline completo).
#
# Knobs auto-registrados via `_env_float` (P3-NEW-D). Validator clamp (0, 120]
# para evitar timeouts patológicos por env var corrupta.
def _chat_agent_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_AGENT_LLM_TIMEOUT_S",
        15.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )

def _chat_swap_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_SWAP_LLM_TIMEOUT_S",
        30.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )

def _chat_title_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_TITLE_LLM_TIMEOUT_S",
        10.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )

def _chat_router_llm_timeout_s() -> float:
    # [P1-CHAT-EMPTY-RESPONSE · 2026-05-20] Default bumpeado 8.0 → 12.0.
    # Pre-fix: Gemini API rechaza con HTTP 400 INVALID_ARGUMENT
    # ("Manually set deadline 8s is too short. Minimum allowed deadline
    # is 10s.") porque 8s < 10s mínimo del API. El RAG router caía al
    # `except` cada vez y degradaba al prompt original sin reescribir —
    # feature silenciosamente rota desde el deploy del bundle P0-CHAT-LLM-TIMEOUT.
    # 12s = 10s mínimo + 2s margen para variabilidad del provider.
    # Validator extendido para enforce el floor a 10s incluso si el
    # operador setea el env var con valor inválido.
    return _env_float(
        "MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S",
        12.0,
        validator=lambda v: 10.0 <= v <= 120.0,
    )

def _chat_graph_total_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_GRAPH_TOTAL_TIMEOUT_S",
        60.0,
        validator=lambda v: 0.0 < v <= 300.0,
    )


# [P1-CHAT-STREAM-BUDGET · 2026-05-20] Total budget para el stream SSE
# (`chat_with_agent_stream`). Pre-fix: el wrapper non-stream tenía
# `_chat_graph_total_timeout_s` (60s) pero el stream NO. Caso de fallo:
# Gemini emite chunks ocasionales pero el turn total nunca termina por
# loops del agente (call_model → execute_tools → call_model bouncing),
# tool roundtrip que cuelga, o un stream genuinamente lento de plan-gen
# desde el chat. Sin tope total, un solo turn puede comer tokens y
# threadpool por minutos.
#
# Default 120s: el stream puede legítimamente exceder los 60s del
# non-stream porque tools como `generate_new_plan_from_chat` invocan el
# pipeline completo (puede tardar 30-60s solo). 120s da margen sin
# permitir runaway. Clamp (0, 600] — 10min absoluto.
#
# Defensa-en-profundidad sobre los per-LLM timeouts (15s) que cubren el
# caso "Gemini cuelga UNA invocación"; este cubre "agente entró en loop
# de N invocaciones legítimas pero el turn total no termina".
def _chat_stream_total_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_STREAM_TOTAL_TIMEOUT_S",
        120.0,
        validator=lambda v: 0.0 < v <= 600.0,
    )


# [P1-CHAT-STREAM-INACTIVITY · 2026-05-20] Inactivity timeout entre eventos
# emitidos por `chat_graph_app.stream(...)`. Si entre dos `next(stream_iter)`
# pasan más de N segundos sin que llegue ningún evento (chunk del LLM,
# tool_call, etc.), abortamos el stream. El per-LLM timeout (15s) ya cubre
# el caso "Gemini bloquea una invocación", pero NO cubre stalls en el
# middleware de LangGraph entre nodes ni cuelgues de checkpointer Postgres.
#
# Default 25s: holgura sobre el per-LLM timeout (15s) + buffer para
# checkpoint write y route_tools. Si baja de eso se vuelve flaky bajo
# carga normal. Clamp (0, 120].
#
# NOTA: implementado vía wall-clock check al tope del for-loop, NO via
# thread-watchdog (eso doblaría el thread count por request). Si Gemini
# emite UN chunk cada 26s seguidos, el check no dispara (porque hay
# actividad). Es válido — el caso problemático es "0 chunks por N
# segundos", no "chunks regulares pero lentos".
def _chat_stream_inactivity_timeout_s() -> float:
    return _env_float(
        "MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S",
        25.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )

llm = ChatGoogleGenerativeAI(
    model=_chat_agent_model_name(),
    temperature=0.2,
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    safety_settings=_safety_settings,
    timeout=_chat_agent_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
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
    
    swap_reason = form_data.get("swap_reason", "dislike")
    
    if swap_reason == 'variety':
        context_extras += "\n    - 💡 INTENCIÓN: El usuario NO rechaza este plato, solo quiere VARIEDAD. Sugiere combinaciones creativas, diferentes técnicas de cocción o perfiles de sabor novedosos pero accesibles."
    elif swap_reason == 'time':
        context_extras += "\n    - ⏱️ INTENCIÓN: El usuario NO TIENE TIEMPO HOY. Propón una receta extremadamente rápida (< 20 min), preferiblemente sin cocción extensa o usando ingredientes fáciles de armar."
    elif swap_reason == 'budget':
        context_extras += "\n    - 💰 INTENCIÓN: El usuario busca opciones ECONÓMICAS. Prioriza ingredientes de muy bajo costo y alto rendimiento, evitando proteínas premium."
    elif swap_reason == 'pantry_first':
        context_extras += "\n    - 📦 INTENCIÓN: El usuario quiere MAXIMIZAR SU INVENTARIO. Limítate estrictamente a usar ingredientes base comunes de despensa. Cero compras nuevas o ingredientes exóticos."
    elif swap_reason == 'cravings':
        context_extras += "\n    - 🤤 INTENCIÓN: El usuario tiene un ANTOJO. Propón algo indulgente, comfort food o una versión saludable de comida rápida, pero manteniendo los macros."
    elif swap_reason == 'weekend':
        context_extras += "\n    - 🎉 INTENCIÓN: FIN DE SEMANA ESPECIAL. El usuario quiere un plato más elaborado, festivo o premium. Ideal para disfrutar con tiempo."
    elif swap_reason == 'similar':
        context_extras += "\n    - 🍽️ INTENCIÓN: El usuario YA COMIÓ ALGO SIMILAR. Ofrece un perfil de sabor o técnica de cocción COMPLETAMENTE DISTINTA a la opción rechazada."

    
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
    except Exception as _swap_exc:
        # [P2-SILENT-DEGRADATION · 2026-05-13] El swap continúa sin sugerencia
        # anti mode-collapse (correctness preservada). Sin log, fallos
        # sistemáticos del helper de variedad pasan invisibles → cliente nota
        # "los swaps repiten siempre las mismas opciones" pero SRE no
        # correlaciona. Mantener fallback (no bloquear el swap).
        logger.debug(
            "[P2-SILENT-DEGRADATION] anti mode-collapse suggestion falló: "
            "%s: %s",
            type(_swap_exc).__name__,
            str(_swap_exc)[:160],
        )

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
        model=_chat_agent_swap_model_name(),
        temperature=temp,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        timeout=_chat_swap_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
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

# [P1-CHAT-CB · 2026-05-19] Excepción dedicada para "breaker abierto sobre
# el chat_llm". Se raise dentro del nodo `call_model` cuando
# `_get_circuit_breaker(model).can_proceed() == False` (failures >= threshold
# Y dentro de la ventana reset_timeout). LangGraph la propaga al caller de
# `chat_graph_app.invoke` / `.stream`; el router `/api/chat` la mapea a
# `HTTP 503 Service Unavailable` (semánticamente: upstream LLM saturado,
# reintentar en N segundos — donde N ≈ MEALFIT_CB_RESET_TIMEOUT_S).
#
# Defensa simétrica al P0-CHAT-LLM-TIMEOUT: timeout previene cuelgues
# individuales; el CB previene avalanchas tras múltiples fallos consecutivos
# (provider degradado, rate-limit del API key, modelo deprecado sin aviso).
# Resto del repo (pipeline de plan-gen) ya usa este CB — el chat era el
# único path productivo que invocaba Gemini sin breaker.
class LLMCircuitBreakerOpen(RuntimeError):
    """Raised by chat-agent LangGraph nodes when the LLM circuit breaker for
    the target model is open. Caller (router) should map to HTTP 503."""
    pass


# [P1-CHAT-LLM-429 · 2026-05-20] Excepción específica para rate-limit del
# provider (Gemini ResourceExhausted, HTTP 429). Pre-fix: cualquier fallo
# del invoke (timeout, 429, 5xx, parse error) contaba como `_cb.record_failure()`
# vía `except Exception` broad. Resultado: 3 bursts de 429 → CB abre 30s →
# usuarios legítimos ven 503 falso-positivo durante saturación temporal de
# Google. El CB está pensado para "provider degradado/down", no para
# "throttling natural del API key bajo carga concurrente".
#
# Defensa:
#   - Detección por type-name+message (Google api_core lo levanta como
#     `google.api_core.exceptions.ResourceExhausted` o como `ChatGoogleGenerativeAI`
#     wrapped error con "429" / "Resource has been exhausted" / "RATE_LIMIT" en
#     el mensaje).
#   - Cuando se detecta, NO `record_failure` (el CB queda intacto) — re-emit
#     como `LLMRateLimitedError` que el router mapea a HTTP 429 (no 503).
#     El cliente puede reintentar con Retry-After.
#   - Emit `pipeline_metrics` con `node='chat_llm_rate_limited'` para
#     telemetría: SRE puede graficar bursts de 429 sin contaminar el conteo
#     de circuit-breaker-failures.
#
# Tooltip-anchor: P1-CHAT-LLM-429.
class LLMRateLimitedError(RuntimeError):
    """Raised when the upstream LLM provider returns a rate-limit error
    (HTTP 429 / ResourceExhausted). Distinct from generic failures so the
    circuit breaker is NOT triggered. Caller (router) should map to HTTP 429."""
    pass


def _is_rate_limit_error(exc: BaseException) -> bool:
    """[P1-CHAT-LLM-429 · 2026-05-20] Heurística defensiva para detectar
    rate-limit del provider. Cubre 3 envoltorios:
      (a) `google.api_core.exceptions.ResourceExhausted` (raw gRPC).
      (b) `google.genai.errors.ClientError` con `code=429`.
      (c) `langchain_core.exceptions.OutputParserException` u otros wrappers
          que preservan el mensaje "429" / "Resource has been exhausted" /
          "RATE_LIMIT_EXCEEDED".

    NO usa isinstance contra la clase `ResourceExhausted` directo porque
    requeriría importar `google.api_core` a module-init (dep extra solo
    para clasificación). Match string es robusto contra cambios de
    wrappers de LangChain entre versiones.
    """
    try:
        _type_name = type(exc).__name__
        if _type_name in ("ResourceExhausted", "TooManyRequests", "RateLimitError"):
            return True
        _msg = str(exc).lower()
        if "resource has been exhausted" in _msg:
            return True
        if "rate_limit_exceeded" in _msg or "rate limit" in _msg:
            return True
        # HTTP code embebido en el mensaje del wrapper.
        if " 429 " in f" {_msg} " or "(429)" in _msg or "code: 429" in _msg or '"code":429' in _msg:
            return True
        # google.genai ClientError expone `.code` numérico.
        _code = getattr(exc, "code", None)
        if _code == 429:
            return True
        return False
    except Exception:
        return False


def _emit_chat_rate_limited_metric_best_effort(user_id, session_id, model_name):
    """[P1-CHAT-LLM-429 · 2026-05-20] Persiste un row en `pipeline_metrics`
    cuando detectamos 429 — separado del flujo del CB para que SRE pueda
    graficar bursts del provider sin que el CB se ensucie. Best-effort: cualquier
    fallo de DB no debe tumbar el response al caller."""
    try:
        from db_core import execute_sql_write
        import json as _json_rl
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, %s, %s, 0, 0, 0, 0, %s::jsonb)
            """,
            (
                user_id if user_id and user_id != "guest" else None,
                session_id,
                "chat_llm_rate_limited",
                _json_rl.dumps({"model": model_name, "provider": "gemini"}, ensure_ascii=False),
            ),
        )
    except Exception as _e_rl:
        try:
            logger.debug(f"[P1-CHAT-LLM-429] emit metric falló (best-effort): {_e_rl!r}")
        except Exception:
            pass


def _emit_chat_stream_total_duration_best_effort(user_id, session_id, model_name, duration_ms, outcome):
    """[P1-CHAT-STREAM-DURATION · 2026-05-20] Persiste `duration_ms` total del
    stream chat (graph-total wall-clock) en `pipeline_metrics` con
    `node='chat_stream_total_duration'`. Pre-fix: el chat-flow tenía duration
    per-LLM-invoke (P2-CHAT-TOKEN-TELEMETRY emite a `llm_usage_events`)
    pero NO graph-total — un turn con 3 invokes encadenados no era graphable
    como P99 latencia E2E. Outcome: 'ok'/'timeout'/'error'/'cancelled'."""
    try:
        from db_core import execute_sql_write
        import json as _json_dur
        execute_sql_write(
            """
            INSERT INTO pipeline_metrics
                (user_id, session_id, node, duration_ms, retries,
                 tokens_estimated, confidence, metadata)
            VALUES (%s, %s, %s, %s, 0, 0, 0, %s::jsonb)
            """,
            (
                user_id if user_id and user_id != "guest" else None,
                session_id,
                "chat_stream_total_duration",
                int(duration_ms),
                _json_dur.dumps({"model": model_name, "outcome": outcome}, ensure_ascii=False),
            ),
        )
    except Exception as _e_dur:
        try:
            logger.debug(f"[P1-CHAT-STREAM-DURATION] emit falló (best-effort): {_e_dur!r}")
        except Exception:
            pass


class ChatState(MessagesState):
    user_id: str
    session_id: str
    form_data: dict
    current_plan: dict
    updated_fields: dict
    new_plan: dict
    sys_prompt: str
    # [P2-AUDIT-NEW-1 · 2026-05-12] Acumulador de `_coherence_warnings`
    # extraídos de los tool_results JSON (hoy solo `modify_single_meal`,
    # P2-COHERENCE-1). Se propaga al evento SSE `done` para que el
    # frontend (AgentPage) emita toast no-bloqueante con `emitCoherenceToast`.
    # Default: lista vacía (no warnings).
    coherence_warnings: list

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
        model=_chat_agent_model_name(),
        temperature=0.7,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        safety_settings=_safety_settings,
        timeout=_chat_agent_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
    )
    llm_with_tools = chat_llm.bind_tools(agent_tools)

    # [P1-CHAT-CB · 2026-05-19] Gate per-modelo. Si el breaker está abierto
    # (failures >= MEALFIT_CB_FAILURE_THRESHOLD dentro de la ventana
    # MEALFIT_CB_RESET_TIMEOUT_S), fail-fast SIN invocar Gemini — el provider
    # ya está degradado y un nuevo intento solo agrava la condición + paga
    # latencia + tokens. Se reabre automáticamente cuando expira la ventana.
    # `record_success` / `record_failure` actualizan el estado para que el
    # resto del repo (pipeline de plan-gen, swap, etc.) vea el mismo breaker.
    #
    # NOTA: `_chat_agent_model_name()` se llama 2x (callsite del CGGA arriba +
    # aquí) en lugar de cachear en variable local — preserva el contrato del
    # test P2-AUDIT-1 (regex busca `model=_chat_*_model_name()` literal en
    # los args del CGGA). Costo trivial: el helper es un dict lookup en env
    # con UPSERT idempotente al registry.
    _cb_model = _chat_agent_model_name()
    _cb = _get_circuit_breaker(_cb_model)
    if not _cb.can_proceed():
        logger.warning(
            f"🛑 [P1-CHAT-CB] LLM circuit breaker abierto para model={_cb_model!r} "
            f"— fail-fast sin invocar Gemini. Reintentar tras "
            f"MEALFIT_CB_RESET_TIMEOUT_S segundos."
        )
        raise LLMCircuitBreakerOpen(
            f"chat LLM circuit breaker open for model={_cb_model}"
        )

    # [P2-CHAT-TOKEN-TELEMETRY · 2026-05-19] Instrumentación de tokens
    # del chat-agent. Pre-fix: el chat-flow NO se reportaba en
    # `llm_usage_events` porque `chat_llm` se construye directo
    # (`ChatGoogleGenerativeAI(...)`), sin pasar por el override
    # `ainvoke/astream` de `graph_orchestrator.py` que dispara
    # `_emit_llm_usage_event_best_effort` automáticamente. Resultado:
    # SRE veía costos de plan-gen (P1-COST-INSTRUMENTATION 2026-05-15) pero
    # 0 visibilidad de costos del agente conversacional. Bajo abuso (user
    # plus enviando 200 prompts/mes), el cron de alerting NO podía
    # detectar anomalías porque la fila no existía.
    #
    # Fix: medimos `duration_s` alrededor del `invoke` y emitimos el
    # evento post-success (NO en failure path — un timeout no consumió
    # tokens completos). El helper es best-effort: cualquier fallo de
    # parse/DB se silencia y NO afecta el response al caller. Reutiliza
    # el SSOT del repo (mismo helper que plan-gen, mismo schema
    # `llm_usage_events`, mismo cost calculation `compute_gemini_cost_micros`).
    # Tooltip-anchor: P2-CHAT-TOKEN-TELEMETRY.
    import time as _time_chat
    _chat_invoke_start = _time_chat.time()
    try:
        response = llm_with_tools.invoke(llm_messages)
    except Exception as _invoke_exc:
        # [P1-CHAT-LLM-429 · 2026-05-20] Diferenciar rate-limit del provider
        # de fallos genuinos. 429 NO debe contar como CB-failure: el provider
        # está vivo, solo está throttleando este API key (saturación temporal,
        # NO degradación). Si contábamos 429 como failure, 3 bursts en ventana
        # de 30s abrían el CB → 503 falso-positivo a usuarios legítimos.
        # Ahora: 429 → metric + re-raise como `LLMRateLimitedError` (mapea a
        # HTTP 429, router → Retry-After); resto → CB failure + re-raise.
        if _is_rate_limit_error(_invoke_exc):
            _emit_chat_rate_limited_metric_best_effort(
                state.get("user_id"), state.get("session_id"), _cb_model,
            )
            logger.warning(
                f"⚠️ [P1-CHAT-LLM-429] Gemini rate-limit detectado "
                f"model={_cb_model!r} exc_type={type(_invoke_exc).__name__} — "
                f"NO cuenta como CB failure."
            )
            raise LLMRateLimitedError(
                f"chat LLM rate limited for model={_cb_model}: {_invoke_exc!r}"
            ) from _invoke_exc
        # Resto: timeout, DeadlineExceeded, 5xx, parse error. El repo usa
        # broad-catch (graph_orchestrator.py:1423) — la excepción se
        # re-raises para que LangGraph la propague al caller.
        _cb.record_failure()
        raise

    _cb.record_success()

    # [P1-CHAT-EMPTY-RESPONSE · 2026-05-20] Detección de response vacío
    # post-invoke (Gemini safety filter). Modo de fallo observado en
    # prod: Gemini emite WARNING "produced an empty response" con
    # `Feedback: block_reason=PROHIBITED_CONTENT` y devuelve un
    # AIMessage(content='') SIN tool_calls. El graph rutea por `route_tools`
    # a END (no hay tool_calls), el SSE concluye con éxito, pero el
    # frontend renderiza un mensaje VACÍO del agente — UX confusa
    # (usuario asume bug del cliente). PROHIBITED_CONTENT es categoría
    # server-side de Google NO controlable desde safety_settings del
    # SDK (que solo cubre HATE/HARASSMENT/SEXUAL/DANGEROUS).
    #
    # Causa probable: system prompt del chat agent contiene frases
    # imperativas ("CERO COMPLACENCIA", "TIENES LA ORDEN", "JAMÁS lo
    # reprimas") que pueden activar el filtro de Google bajo combinación
    # con ciertos mensajes user. NO siempre dispara — el primer chat de
    # ese día funcionó OK con el mismo prompt + mensaje benigno.
    #
    # Fix: detectamos `(empty content) AND (no tool_calls)` post-invoke
    # y reemplazamos el AIMessage por uno con copy fallback explícito
    # que invita al user a reformular. Distingue del caso legítimo
    # "Gemini emitió tool_calls + content vacío" (response a tool
    # planeada): si hay tool_calls, NO sustituimos.
    #
    # Best-effort metric: emit `chat_llm_empty_response` para que SRE
    # grafique falsos positivos del filtro server-side y decida si
    # cambiar de modelo (gemini-3.5-pro es más permisivo que flash) o
    # suavizar el system prompt. Tooltip-anchor: P1-CHAT-EMPTY-RESPONSE.
    _resp_content_str = ""
    try:
        _resp_content_str = str(getattr(response, "content", "") or "").strip()
    except Exception:
        _resp_content_str = ""
    _resp_tool_calls = getattr(response, "tool_calls", None) or []
    if not _resp_content_str and not _resp_tool_calls:
        logger.warning(
            f"⚠️ [P1-CHAT-CHAT-EMPTY-RESPONSE] Gemini devolvió response vacío "
            f"sin tool_calls (probable PROHIBITED_CONTENT filter del provider). "
            f"model={_cb_model!r}. Sustituyendo por mensaje fallback."
        )
        try:
            from db_core import execute_sql_write
            import json as _json_empty
            execute_sql_write(
                """
                INSERT INTO pipeline_metrics
                    (user_id, session_id, node, duration_ms, retries,
                     tokens_estimated, confidence, metadata)
                VALUES (%s, %s, %s, 0, 0, 0, 0, %s::jsonb)
                """,
                (
                    state.get("user_id") if state.get("user_id") and state.get("user_id") != "guest" else None,
                    state.get("session_id"),
                    "chat_llm_empty_response",
                    _json_empty.dumps({"model": _cb_model, "provider": "gemini"}, ensure_ascii=False),
                ),
            )
        except Exception:
            pass
        _fallback_copy = (
            "No pude procesar esa solicitud por restricciones del modelo. "
            "¿Puedes reformularla con otras palabras? Si lo que querías era "
            "registrar una comida, intenta algo como: \"comí X gramos de Y "
            "para el almuerzo\"."
        )
        response = AIMessage(content=_fallback_copy)

    # [P2-CHAT-TOKEN-TELEMETRY · 2026-05-19] Best-effort post-invoke.
    # Importa lazy para evitar acoplamiento module-init con graph_orchestrator
    # (que ya importa este módulo en algunos paths). El helper acepta
    # `response` (AIMessage con `usage_metadata`) y `llm` para resolver
    # el model name. Cualquier fallo en el emit NO debe romper el chat.
    try:
        from graph_orchestrator import _emit_llm_usage_event_best_effort
        # [P3-CHAT-NODE-EXPLICIT · 2026-05-20] Pasamos `node='chat_call_model'`
        # explícito porque el chat-flow NO setea el ContextVar `_current_node_var`
        # que el helper consulta por default. Sin esto, todas las filas del
        # chat en `llm_usage_events` quedan con `node=NULL` y SRE no puede
        # filtrar costos chat vs plan-gen.
        _emit_llm_usage_event_best_effort(
            llm=chat_llm,
            result=response,
            duration_s=_time_chat.time() - _chat_invoke_start,
            node='chat_call_model',
        )
    except Exception:
        pass

    return {"messages": [response]}

def execute_tools(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    
    updated_fields = state.get("updated_fields", {})
    new_plan = state.get("new_plan", None)
    # [P2-AUDIT-NEW-1 · 2026-05-12] Acumulador de `_coherence_warnings` desde
    # tool_results. Preserva entries previos del state (rare con un solo
    # tool_call por turn, pero defensive si el LLM emite múltiples tool_calls
    # que retornan warnings — extiende la lista en lugar de pisarla).
    coherence_warnings = list(state.get("coherence_warnings") or [])

    tool_messages = []
    
    # [P0-AGENT-1 · 2026-05-11] Trusted user_id resolution UNA VEZ por
    # invocación del nodo. Lo usamos para force-override `tool_args["user_id"]`
    # en CADA tool_call antes de invocar la tool. Patrón espejo de los 2
    # branches que ya lo hacían inline (`generate_new_plan_from_chat`,
    # `modify_single_meal`) extendido a TODAS las tools.
    #
    # Razón: `tool_args` viene del LLM y antes confiábamos en que la LLM
    # reusara el `user_id` que el system prompt (`build_tools_instructions`)
    # le indicaba. Eso es prompt-trustable, NO enforced. Una entrada
    # adversaria del usuario o contenido inyectado vía recetas importadas /
    # transcripts de imágenes (vision_agent → chat-context) puede inducir a
    # la LLM a emitir tool_call con `user_id` ajeno → cross-user write/read
    # sobre `user_inventory`, `consumed_meals`, `user_facts`, `health_profile`.
    # Defensa simétrica a la sanitización P1-Q8/P0-A1 del pipeline de
    # generación, pero aplicada al chat-agent layer.
    _trusted_user_id = state.get("user_id")
    _trusted_session_id = state.get("session_id")
    _trusted_uid = (
        _trusted_user_id
        if _trusted_user_id and _trusted_user_id != "guest"
        else _trusted_session_id
    )

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # [P0-AGENT-1 · 2026-05-11] Force-override `user_id` en tool_args
            # ANTES de cualquier branch. Si la LLM pasó un `user_id` distinto
            # del autenticado (prompt injection o hallucinación), logueamos
            # WARN para telemetría y lo reescribimos al trusted. Cubre TODAS
            # las 9 tools de `agent_tools` (todas aceptan `user_id` en su
            # signature). NO confiamos en que cada branch nuevo del if/elif
            # se acuerde de hacer el override — se hace acá una sola vez.
            if isinstance(tool_args, dict):
                _llm_uid = tool_args.get("user_id")
                if _llm_uid and _trusted_uid and _llm_uid != _trusted_uid:
                    logger.warning(
                        f"🛡️ [P0-AGENT-1] tool={tool_name} llm_user_id={_llm_uid!r} "
                        f"!= trusted={_trusted_uid!r}. Override aplicado. Posible "
                        f"prompt injection — verificar último mensaje del usuario."
                    )
                tool_args["user_id"] = _trusted_uid

            tool_result = ""
            logger.debug(f"🔧 [LANGGRAPH TOOL] Ejecutando {tool_name}")

            # [P1-CHAT-TOOL-VALIDATE · 2026-05-20] Recuperación graceful si
            # el LLM emite `tool_args` con tipos inválidos para el schema
            # auto-generado de LangChain (`@tool` decorator usa Pydantic v2).
            # Pre-fix: `ValidationError` bubbleaba al graph y rompía el turn
            # entero — usuario veía HTTP 500 sin pista de qué pasó. Casos:
            #   - `log_consumed_meal(calories="muchas")` (str vs int).
            #   - `modify_pantry_inventory(items_to_add="leche")` (str vs list).
            #   - `log_water_glass(count_delta=1.5)` (float vs int).
            #
            # Fix: catch específico → tool_result inyecta mensaje claro al
            # LLM (que puede reintentar) + WARN para SRE. NO afecta el flujo
            # mainstream (validación ok → tool corre normal). Cubre TODO el
            # if/elif/else porque tanto el dispatch directo (`update_form_field.invoke(...)`)
            # como el genérico (`t.invoke(tool_args)`) pueden lanzar ValidationError.
            # Tooltip-anchor: P1-CHAT-TOOL-VALIDATE.
            try:
                _PydanticValidationError = __import__("pydantic", fromlist=["ValidationError"]).ValidationError
            except Exception:
                _PydanticValidationError = ValueError  # fallback inocuo

            try:
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
                    except Exception as _parse_exc:
                        # [P2-SILENT-DEGRADATION · 2026-05-13] JSON malformado /
                        # tool_result no parseable: el agente NO hidrata el plan
                        # en state pero conserva el raw tool_result (texto al LLM).
                        # Sin log, un cambio de schema del tool o tool_result vacío
                        # significa "el agente respondió pero el dashboard no
                        # refrescó" sin telemetría. Mantener fallback.
                        logger.debug(
                            "[P2-SILENT-DEGRADATION] generate_new_plan parse "
                            "falló: %s: %s",
                            type(_parse_exc).__name__,
                            str(_parse_exc)[:160],
                        )

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
                            # [P2-AUDIT-NEW-1 · 2026-05-12] Extraer
                            # `_coherence_warnings` ANTES de pisar `tool_result`
                            # con el friendly string. El tool `modify_single_meal`
                            # los inyecta cuando el guard P2-COHERENCE-1 detectó
                            # divergencia recetas↔lista post-modificación. Se
                            # propagan al state → SSE `done` → frontend toast.
                            _tool_warnings = parsed_mod.get("_coherence_warnings")
                            if isinstance(_tool_warnings, list) and _tool_warnings:
                                coherence_warnings.extend(_tool_warnings)
                            tool_result = f"La comida fue modificada exitosamente. La nueva comida es: {parsed_mod['modified_meal'].get('name', 'Comida actualizada')}. Dile al usuario que su plan ya fue actualizado."
                    except Exception as _mod_exc:
                        # [P2-SILENT-DEGRADATION · 2026-05-13] JSON malformado /
                        # `modified_meal` ausente: el agente NO hidrata el plan
                        # actualizado en state ni extrae warnings de coherencia.
                        # El plan en DB SÍ se modificó (modify_single_meal
                        # persiste antes de retornar) pero el frontend no
                        # refresca hasta el siguiente fetch. Sin log, fallos
                        # sistemáticos del parser quedan invisibles.
                        logger.debug(
                            "[P2-SILENT-DEGRADATION] modify_single_meal parse "
                            "falló: %s: %s",
                            type(_mod_exc).__name__,
                            str(_mod_exc)[:160],
                        )
                else:
                    for t in agent_tools:
                        if t.name == tool_name:
                            tool_result = t.invoke(tool_args)
                            break
            except _PydanticValidationError as _val_err:
                # [P1-CHAT-TOOL-VALIDATE · 2026-05-20] LLM emitió tool_args
                # con tipos inválidos. Inyectamos un tool_result legible al
                # LLM (que ve este string como "respuesta de la tool") para
                # que pueda recuperarse en el siguiente turn (reintentar con
                # tipos correctos o pedir aclaración al usuario). NO romper
                # el graph: el chat sigue funcional.
                _val_summary = str(_val_err)[:300]
                logger.warning(
                    f"⚠️ [P1-CHAT-TOOL-VALIDATE] ValidationError tool={tool_name} "
                    f"args_keys={list(tool_args.keys()) if isinstance(tool_args, dict) else '?'} "
                    f"summary={_val_summary!r}"
                )
                tool_result = (
                    f"[VALIDATION_ERROR] No pude ejecutar '{tool_name}' porque los "
                    f"argumentos enviados no cumplen el schema esperado. Detalle: "
                    f"{_val_summary}. Reintenta con los tipos correctos o pide "
                    f"aclaración al usuario antes de re-invocar la tool."
                )

            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
            
    return {
        "messages": tool_messages,
        "updated_fields": updated_fields,
        "new_plan": new_plan,
        # [P2-AUDIT-NEW-1 · 2026-05-12] Propagar warnings al state. El
        # stream wrapper (`chat_with_agent_stream`) lo lee de
        # `final_state_snapshot.values["coherence_warnings"]` y lo incluye
        # en el SSE event `done` para que el frontend emita toast.
        "coherence_warnings": coherence_warnings,
    }

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
            
        title_llm = ChatGoogleGenerativeAI(model=_chat_title_model_name(), temperature=0.7, google_api_key=os.environ.get("GEMINI_API_KEY"), timeout=_chat_title_llm_timeout_s())  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
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
            model=_chat_router_model_name(),
            temperature=0.0,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=_chat_router_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
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
                total_consumed = sum(m.get('calories', 0) for m in consumed_today)
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                
                target_calories = form_data.get("target_calories") if form_data else None
                if not target_calories and current_plan:
                    target_calories = current_plan.get("calories")
                
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}."
                
                if target_calories:
                    try:
                        target_cal_int = int(target_calories)
                        system_prompt += f" Total consumido: {total_consumed} kcal de un presupuesto de {target_cal_int} kcal."
                        
                        remaining = target_cal_int - total_consumed
                        if remaining < (target_cal_int * 0.35) and remaining > 0:
                            system_prompt += f"\n🚨 ALERTA DE MICRO-ADAPTACIÓN (MEJORA 6): Al usuario solo le quedan {remaining} kcal para el resto del día. TIENES LA OBLIGACIÓN PROACTIVA de hacerle notar este ajustado presupuesto con amabilidad de coach. Sugiérele usar tu herramienta 'modify_single_meal' para recalcular y reducir las porciones de sus próximas comidas de hoy para mantener su déficit."
                        elif remaining <= 0:
                            system_prompt += f"\n🚨 ALERTA CRÍTICA (MEJORA 6): El usuario ha superado su presupuesto calórico de hoy. Tiene un exceso de {abs(remaining)} kcal. Indícale esto con empatía y dale recomendaciones proactivas sobre cómo equilibrarse."
                    except ValueError:
                        pass
            else:
                system_prompt += "\n\nDIARIO DE HOY: El usuario no ha registrado ninguna comida el día de hoy todavía."
        except Exception as e:
            logger.error(f"⚠️ Error inyectando contexto de diario (non-stream): {e}")
        
    config = {"configurable": {"thread_id": session_id}}

    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Pool separado para PostgresSaver
    # (session mode, port 5432) evita "SSL bad length / EOF" cuando Supavisor
    # mata conexiones idle del Transaction Pooler durante el chat stream.
    # Fallback defensivo a `connection_pool` si el split pool no se creó.
    _checkpoint_pool = chat_checkpoint_pool or connection_pool
    if _checkpoint_pool:
        checkpointer = PostgresSaver(_checkpoint_pool)
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

    # [P0-CHAT-LLM-TIMEOUT · 2026-05-19] Total-graph timeout. Defensa-en-profundidad
    # sobre los per-LLM timeouts del constructor — cubre escenarios donde múltiples
    # invokes acumulan (call_model + execute_tools + call_model) o donde un tool
    # interno cuelga sin propagar timeout. Default 60s (knob
    # `MEALFIT_CHAT_GRAPH_TOTAL_TIMEOUT_S`). Si excede:
    #   - `concurrent.futures.TimeoutError` se propaga al caller (router).
    #   - El thread del executor sigue corriendo hasta que el LLM internal timeout
    #     lo abata — NO cancellable cooperativamente, pero el endpoint ya respondió.
    #   - El thread pool externo de FastAPI queda libre inmediatamente.
    _graph_timeout_s = _chat_graph_total_timeout_s()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="chat_graph_invoke"
    ) as _ex:
        _fut = _ex.submit(chat_graph_app.invoke, inputs, config=config)
        try:
            final_state = _fut.result(timeout=_graph_timeout_s)
        except concurrent.futures.TimeoutError as _to_exc:
            logger.error(
                f"⏱️ [P0-CHAT-LLM-TIMEOUT] chat_graph_app.invoke excedió "
                f"{_graph_timeout_s}s para session={session_id} user={user_id!r}. "
                f"Posible Gemini hang / network issue."
            )
            raise TimeoutError(
                f"chat_graph exceeded {_graph_timeout_s}s timeout"
            ) from _to_exc

    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    logger.info(f"✅ [COMPLETADO] LangGraph finalizó en {duration_secs} segundos.")
    logger.info("-------------------------------------------------------------\n")
    
    final_messages = final_state["messages"]
    last_msg = final_messages[-1]
    content = last_msg.content
    
    if isinstance(content, list):
        content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content])

    # [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad output non-stream.
    sanitized_content = _sanitize_chat_output_for_wire(str(content))
    return sanitized_content, final_state.get("updated_fields", {}), final_state.get("new_plan")

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
                total_consumed = sum(m.get('calories', 0) for m in consumed_today)
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                
                target_calories = form_data.get("target_calories") if form_data else None
                if not target_calories and current_plan:
                    target_calories = current_plan.get("calories")
                
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}. Revisa esto ANTES de preguntar si ya comió algo (por ejemplo, si ya tiene una cena registrada, no le preguntes si esa foto es su cena, asume que es un snack nocturno o pregúntale por qué repite). Si la foto o mensaje coincide con algo que ya está registrado, felicítalo o no lo registres de nuevo."
                
                if target_calories:
                    try:
                        target_cal_int = int(target_calories)
                        system_prompt += f" Total consumido: {total_consumed} kcal de un presupuesto de {target_cal_int} kcal."
                        
                        remaining = target_cal_int - total_consumed
                        if remaining < (target_cal_int * 0.35) and remaining > 0:
                            system_prompt += f"\n🚨 ALERTA DE MICRO-ADAPTACIÓN (MEJORA 6): Al usuario solo le quedan {remaining} kcal para el resto del día. TIENES LA OBLIGACIÓN PROACTIVA de hacerle notar este ajustado presupuesto con amabilidad de coach. Sugiérele usar tu herramienta 'modify_single_meal' para recalcular y reducir las porciones de sus próximas comidas de hoy para mantener su déficit."
                        elif remaining <= 0:
                            system_prompt += f"\n🚨 ALERTA CRÍTICA (MEJORA 6): El usuario ha superado su presupuesto calórico de hoy. Tiene un exceso de {abs(remaining)} kcal. Indícale esto con empatía de coach y dale recomendaciones proactivas sobre cómo equilibrarse en la cena o mañana."
                    except ValueError:
                        pass
            else:
                system_prompt += "\n\nDIARIO DE HOY: El usuario no ha registrado ninguna comida el día de hoy todavía."
        except Exception as e:
            logger.error(f"⚠️ Error inyectando contexto de diario: {e}")
            
    config = {"configurable": {"thread_id": session_id}}

    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Mismo split que el callsite del
    # non-stream chat. Sesión session-mode evita SSL EOF durante el SSE.
    # Compilamos usando PostgresSaver sincrónico porque astream_events nativo asíncrono tiene problemas en Windows
    _checkpoint_pool = chat_checkpoint_pool or connection_pool
    if _checkpoint_pool:
        checkpointer = PostgresSaver(_checkpoint_pool)
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

    # [P1-CHAT-CANCEL · 2026-05-19] Guardar referencia explícita al iterator
    # interno de LangGraph para poder cerrarlo si el cliente aborta el SSE
    # (tab-close, AbortController, network drop). FastAPI/Starlette dispara
    # `gen.close()` cuando el response stream se rompe → `GeneratorExit` se
    # inyecta en el yield activo de este generator. Antes del fix el
    # `except Exception` outer NO atrapaba GeneratorExit (hereda de
    # BaseException, no Exception) y el iterator de LangGraph seguía
    # invocando LLM/tools en threads internos hasta completar el turn →
    # costo LLM desperdiciado + posibles writes a BD que el user ya no
    # verá. Cerrar el iterator explícitamente propaga el cancel a los
    # workers y permite que el thread libere recursos.
    stream_iter = chat_graph_app.stream(inputs, config=config, stream_mode="messages")

    # [P1-CHAT-STREAM-BUDGET · 2026-05-20] Wall-clock total budget + inactivity
    # check entre eventos. Defensa-en-profundidad sobre los per-LLM timeouts (15s):
    #   - `_stream_started_at` (monotonic): tope total del turn entero. Si el
    #     stream entra en loop legítimo (call_model → execute_tools → call_model
    #     repetidos) y excede el budget, abortamos antes de gastar más tokens.
    #   - `_last_event_at`: detecta stalls "0 chunks por N segundos". Si Gemini
    #     emite chunks regulares pero todos lentos (3s cada uno), NO dispara —
    #     hay actividad. El caso problemático es silencio prolongado.
    # Outcome se reporta a `pipeline_metrics` en el finally (Fix #5 lite).
    import time as _t_stream
    _stream_started_at = _t_stream.monotonic()
    _last_event_at = _stream_started_at
    _stream_total_budget = _chat_stream_total_timeout_s()
    _stream_inactivity_budget = _chat_stream_inactivity_timeout_s()
    _stream_outcome = "ok"  # 'ok' / 'timeout_total' / 'timeout_inactivity' / 'error' / 'cancelled'

    try:
        for event in stream_iter:
            # [P1-CHAT-STREAM-BUDGET · 2026-05-20] Wall-clock checks al tope
            # del loop body — antes de procesar el evento. Si excedimos algún
            # budget, cerramos el iterator (cancela threads internos LangGraph)
            # y emitimos chunk SSE 'error' explicativo para que el frontend
            # muestre banner contextual antes de raise.
            _now = _t_stream.monotonic()
            _total_elapsed = _now - _stream_started_at
            _gap_since_last = _now - _last_event_at
            if _total_elapsed > _stream_total_budget:
                _stream_outcome = "timeout_total"
                logger.error(
                    f"⏱️ [P1-CHAT-STREAM-BUDGET] total budget excedido "
                    f"{_total_elapsed:.1f}s > {_stream_total_budget}s "
                    f"session={session_id} user={user_id!r}"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'El asistente excedió el tiempo máximo del turno. Intenta de nuevo en unos segundos.'})}\n\n"
                raise TimeoutError(
                    f"chat_with_agent_stream exceeded {_stream_total_budget}s total budget"
                )
            if _gap_since_last > _stream_inactivity_budget:
                _stream_outcome = "timeout_inactivity"
                logger.error(
                    f"⏱️ [P1-CHAT-STREAM-BUDGET] inactivity budget excedido "
                    f"{_gap_since_last:.1f}s > {_stream_inactivity_budget}s "
                    f"session={session_id} user={user_id!r}"
                )
                yield f"data: {json.dumps({'type': 'error', 'message': 'El asistente dejó de responder. Intenta de nuevo.'})}\n\n"
                raise TimeoutError(
                    f"chat_with_agent_stream inactivity {_gap_since_last:.1f}s > {_stream_inactivity_budget}s"
                )
            _last_event_at = _now
            # Identificar el contenido exacto del evento 'messages' (tupla mensaje, dict)
            if isinstance(event, tuple) and len(event) == 2:
                msg_chunk, metadata = event
                if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                    if not msg_chunk.tool_calls:
                        chunk_content = msg_chunk.content
                        if isinstance(chunk_content, list):
                            chunk_content = "".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in chunk_content])
                        if chunk_content: # Evitar chunks vacíos
                            # [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad
                            # del wire SSE chunk.
                            chunk_content = _sanitize_chat_output_for_wire(chunk_content)
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
                                    
    except GeneratorExit:
        # [P1-CHAT-CANCEL · 2026-05-19] Cliente cerró el SSE stream antes de
        # que terminemos (tab-close, AbortController.abort, network drop).
        # NO podemos suprimir GeneratorExit — Python lo requiere para
        # cleanup del generator — pero SÍ podemos cerrar el iterator de
        # LangGraph para que sus workers (ChatGoogleGenerativeAI invokes,
        # tool executors) reciban la señal de cancelado y dejen de
        # consumir tiempo LLM. NO emitir yields acá — la conexión está
        # cerrada y el write fallaría con BrokenPipeError. Log a `warning`
        # (NO error): es flujo legítimo del UX, no incidente. El cleanup
        # final (stream_iter.close()) vive en el `finally` block.
        _stream_outcome = "cancelled"
        logger.warning(
            f"[P1-CHAT-CANCEL] Cliente abortó SSE stream "
            f"session={session_id} user={user_id}"
        )
        raise
    except Exception as e:
        # [P3-TRACEBACK-PRINT-EXC · 2026-05-15] `logger.exception` emite
        # mensaje + stack como un solo log record que respeta `LOG_LEVEL`
        # y Sentry sampling. Reemplaza el legacy `logger.error + traceback.print_exc()`
        # que duplicaba la entrada + bypaseaba el sink configurado.
        # [P1-CHAT-STREAM-BUDGET · 2026-05-20] Si _stream_outcome ya fue
        # marcado por el budget-check (timeout_total/timeout_inactivity),
        # preservar — TimeoutError viene de allí. Si es "ok", marcar "error".
        if _stream_outcome == "ok":
            _stream_outcome = "error"
        logger.exception(f"❌ [CHAT STREAM] Error en astream nativo: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return
    finally:
        # [P1-CHAT-STREAM-FINALLY-CLOSE · 2026-05-19] Cleanup defensivo
        # del iterator de LangGraph en TODOS los exits (normal, exception,
        # GeneratorExit). Garbage collection eventualmente lo cerraría
        # (CPython refcount=0 dispara __del__), pero un `close()` explícito
        # libera de inmediato los recursos atados al iterator: threads
        # internos de astream, file descriptors del checkpointer Postgres
        # si la conexión está pinned. Bajo concurrencia alta con muchos
        # aborts (tab-close en mobile, mala red), confiar en GC produce
        # un leak slow-burn de descriptors. `close()` es idempotente
        # contra iterators ya cerrados; el try/except absorbe excepciones
        # raras (re-entrancy, iterator agotado) que no deben afectar el
        # exit del generator. Tooltip-anchor: P1-CHAT-STREAM-FINALLY-CLOSE.
        try:
            stream_iter.close()
        except Exception:
            pass
        # [P1-CHAT-STREAM-DURATION · 2026-05-20] Persist graph-total
        # wall-clock duration a `pipeline_metrics` con outcome. SRE puede
        # graficar P99 latencia E2E del chat-stream y desglosar por
        # outcome (ok/cancelled/timeout/error). Best-effort: el emit
        # silencia excepciones DB para no romper el cleanup del generator.
        try:
            _total_dur_ms = int((_t_stream.monotonic() - _stream_started_at) * 1000)
            _emit_chat_stream_total_duration_best_effort(
                user_id, session_id, _chat_agent_model_name(),
                _total_dur_ms, _stream_outcome,
            )
        except Exception:
            pass

    # Obtener el estado final actualizado
    try:
        final_state_snapshot = chat_graph_app.get_state(config)
    except Exception as e:
        logger.error(f"⚠️ Error obteniendo get_state tras stream: {e}")

    final_content = ""
    updated_fields = {}
    new_plan = None
    # [P2-AUDIT-NEW-1 · 2026-05-12] Coherence warnings acumulados por el
    # nodo `execute_tools` (P2-COHERENCE-1 emite desde `modify_single_meal`
    # cuando el guard detecta drift recetas↔lista post-modificación).
    # Default [] — sin warnings, frontend silencia el toast.
    coherence_warnings = []

    if final_state_snapshot and final_state_snapshot.values:
        updated_fields = final_state_snapshot.values.get("updated_fields", {})
        new_plan = final_state_snapshot.values.get("new_plan", None)
        coherence_warnings = final_state_snapshot.values.get("coherence_warnings") or []
        final_messages = final_state_snapshot.values.get("messages", [])
        if final_messages:
            last_msg = final_messages[-1]
            extracted_content = last_msg.content
            if isinstance(extracted_content, list):
                extracted_content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in extracted_content])
            final_content = str(extracted_content)

    logger.info("✅ [CHAT STREAM] Finalizado con éxito.")
    # [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad del payload `done`.
    # save_message en routers/chat.py persiste este `response` a DB — sanitizar
    # acá significa que la versión persistida también queda neutralizada.
    final_content = _sanitize_chat_output_for_wire(final_content)
    yield f"data: {json.dumps({'type': 'done', 'response': final_content, 'updated_fields': updated_fields, 'new_plan': new_plan, 'coherence_warnings': coherence_warnings})}\n\n"