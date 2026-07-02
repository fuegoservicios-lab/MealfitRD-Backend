# backend/agent.py

import os
import logging
import time
import json
import re
import unicodedata
logger = logging.getLogger(__name__)

from constants import strip_accents, CULINARY_KNOWLEDGE_BASE, validate_ingredients_against_pantry
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek con router por tier.
from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH, resolve_model_for_user
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
from knobs import _env_str, _env_float, _env_int, _env_bool  # [P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15] / [P0-CHAT-LLM-TIMEOUT · 2026-05-19] auto-registry
# [P1-CHAT-CB · 2026-05-19] Breaker per-modelo del graph_orchestrator. NO
# duplicamos la implementación — reusamos el singleton + knobs ya productivos
# (`MEALFIT_CB_FAILURE_THRESHOLD=3`, `MEALFIT_CB_RESET_TIMEOUT_S=30`). Import
# de un solo nivel: `graph_orchestrator` NO importa `agent` (verificado), no
# hay ciclo. Si en el futuro la dirección de import cambia, mover el helper
# a un módulo neutro.
from graph_orchestrator import _get_circuit_breaker, clinical_backstop_for_meal, UPDATE_CLINICAL_GUARD, renal_protein_trim_for_update, food_safety_backstop_for_meal, condition_substitution_backstop_for_meal, slot_coherence_backstop_for_meal, SLOT_APPROPRIATENESS_GATE_ENABLED, appetibility_fix_for_update, _meal_has_sweet_savory_clash, UPDATE_APPETIBILITY_GUARD
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


# [P2-GENCHUNK-SPEED · 2026-06-01] Claves derivadas/pesadas del `plan_data`
# que NO aportan nada al razonamiento del chat-agent y que hoy se serializan
# textualmente en el system prompt EN CADA TURNO (audit speed 2026-06-01).
# - Los 4 `aggregated_shopping_list*` son listas pre-agregadas que el agente
#   recomputa on-demand vía el tool `check_shopping_list`; la despensa +
#   delta pendiente ya se inyectan compactos vía `build_inventory_context`.
# - `_shopping_coherence_block*` es telemetría interna del guard de coherencia.
# - `_archived_days` es historial podado del shift rolling (crece sin techo
#   útil para el chat) — el Historial lo lee aparte, el chat no.
# - `calc_household_multiplier` es un escalar de cálculo de shopping.
# Mantener intactos: `calories`, `macros`, `name` y el `days[]` vivo completo
# (cada meal con name/description/meal_type/time/macros/ingredients/recipe) —
# el LLM los necesita para responder "qué como hoy" y para mapear el
# `day_number`/`meal_type` correcto en `modify_single_meal`. NO podar días
# week-2+ ni texto de recetas: degradaría la precisión del agente.
# tooltip-anchor: _CHAT_PLAN_PRUNE_KEYS (test_p2_genchunk_speed parsea esto)
_CHAT_PLAN_PRUNE_KEYS = (
    "aggregated_shopping_list",
    "aggregated_shopping_list_weekly",
    "aggregated_shopping_list_biweekly",
    "aggregated_shopping_list_monthly",
    "_shopping_coherence_block",
    "_shopping_coherence_block_history",
    "_archived_days",
    "calc_household_multiplier",
)


def _prune_plan_for_chat(plan):
    """[P2-GENCHUNK-SPEED · 2026-06-01] Devuelve una copia shallow de `plan`
    sin las claves derivadas/pesadas de `_CHAT_PLAN_PRUNE_KEYS`, para reducir
    los input-tokens del system prompt del chat sin perder contenido semántico
    que el agente razone. Defensivo: si `plan` no es dict, lo devuelve intacto.
    Proyección shallow (no deep-copy): solo excluimos claves top-level; los
    `days[]` y demás estructuras se referencian sin clonar (no se mutan)."""
    if not isinstance(plan, dict):
        return plan
    return {k: v for k, v in plan.items() if k not in _CHAT_PLAN_PRUNE_KEYS}


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
    build_user_identity_context,
)

from tools import (
    update_form_field, generate_new_plan_from_chat,
    log_consumed_meal, modify_single_meal,
    search_deep_memory, agent_tools, analyze_preferences_agent,
    execute_generate_new_plan, execute_modify_single_meal,
    check_current_pantry
)

# Langchain Chat Model Initialization
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] El bloque `_safety_settings`
# (HarmCategory/HarmBlockThreshold) fue eliminado: era exclusivo del SDK de
# Gemini. DeepSeek no expone content-filters configurables client-side, así
# que la decisión P3-CHAT-SAFETY-OFF (evitar false-positives en charlas de
# déficit/ayuno) queda satisfecha por defecto del provider.

# [P2-AUDIT-1 · 2026-05-15] Knobs para overridear los modelos LLM usados
# por las 5 callsites de `ChatDeepSeek(...)` en este módulo:
#   - `llm` (módulo-level, fallback default)             → MEALFIT_CHAT_AGENT_MODEL
#   - `swap_llm` dentro de `swap_meal`                   → MEALFIT_CHAT_AGENT_SWAP_MODEL
#   - `chat_llm` dentro de `call_model` (LangGraph node) → MEALFIT_CHAT_AGENT_MODEL (reusa)
#   - `title_llm` dentro de `generate_session_title`     → MEALFIT_CHAT_TITLE_MODEL
#   - `router_llm` dentro de `rag_query_router`          → MEALFIT_CHAT_ROUTER_MODEL
#
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] chat y swap son TIER-ROUTED: sin
# override explícito del knob, el modelo se resuelve por plan de pago via
# `llm_provider.resolve_model_for_user` (gratis/guest → deepseek-v4-flash,
# basic/plus/ultra → deepseek-v4-pro). El override del knob SIEMPRE gana
# (rollback / A-B test sin redeploy — convención P3-PREVIEW-MODEL-KNOB).
# title/router son tareas aux baratas → V4 Flash fijo para todos los tiers.
#
# CONSISTENCIA CB: los gates `_get_circuit_breaker(<model>)` DEBEN resolver
# el modelo con EXACTAMENTE el mismo `user_id` que el constructor del LLM —
# si difieren, el gate protege una key (`llm_circuit_breaker:<model>`)
# distinta de la que falla.
#
# [P3-CHAT-MODEL-KNOBS-REGISTRY · 2026-05-15] Los 4 helpers leen via
# `_env_str(...)` (NO `os.environ.get`) para auto-registrarse en
# `_KNOBS_REGISTRY` (convención P3-NEW-D). Beneficio operacional: tras un
# override en el VPS, el SRE puede verificar el cambio via
# `GET /api/system/admin/knobs` sin releer source.
# Test parser-based: `tests/test_p3_chat_model_knobs_registry.py`.
def _chat_agent_model_name(user_id: Optional[str] = None) -> str:
    override = _env_str("MEALFIT_CHAT_AGENT_MODEL", "")
    if override:
        return override
    return resolve_model_for_user(user_id)

def _chat_agent_swap_model_name(user_id: Optional[str] = None) -> str:
    override = _env_str("MEALFIT_CHAT_AGENT_SWAP_MODEL", "")
    if override:
        return override
    return resolve_model_for_user(user_id)

def _chat_title_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_TITLE_MODEL",
        DEEPSEEK_FLASH,
    )

def _chat_router_model_name() -> str:
    return _env_str(
        "MEALFIT_CHAT_ROUTER_MODEL",
        DEEPSEEK_FLASH,
    )

def _chat_title_max_output_tokens() -> int:
    """[P3-COST-TITLE-OUTPUT-CAP · 2026-06-01] Cap de output del generador de
    título de sesión. El prompt pide "2-4 palabras máximo" y el código YA
    trunca a 32 chars post-hoc (agent.py ~L2112) — si el LLM ignora la
    instrucción y emite una frase larga, esos tokens se generan (output
    facturado) y luego se DESCARTAN. Capar el output elimina ese desperdicio.
    Default 32 (holgado para 4 palabras es-DO; flash-lite no es thinking-capable
    → no hay reasoning de por medio). Knob MEALFIT_CHAT_TITLE_MAX_OUTPUT_TOKENS,
    clamp [8, 256]. Tooltip-anchor: P3-COST-TITLE-OUTPUT-CAP."""
    return _env_int(
        "MEALFIT_CHAT_TITLE_MAX_OUTPUT_TOKENS",
        32,
        validator=lambda v: 8 <= v <= 256,
    )

def _chat_prompt_static_prefix() -> bool:
    """[P2-CHAT-PROMPT-STATIC-PREFIX · 2026-06-01] Cuando True (default), el
    system prompt del chat se ensambla con los bloques ESTÁTICOS byte-estables
    (inline prompt + CULINARY_KNOWLEDGE_BASE + instrucciones de tools) al FRENTE
    y los VOLÁTILES (build_temporal_context con minuto, circadiano, proactivo,
    sentiment, RAG per-turn) al FINAL.

    Por qué importa para COSTO: el chat es el subsistema LLM de mayor frecuencia
    y su costo está dominado por el INPUT (~88% medido en prod: system prompt +
    historial). Gemini cachea implícitamente el PREFIJO byte-estable (cached
    input ~10x más barato), pero exige un mínimo de tokens. Pre-fix,
    build_temporal_context() (hora con MINUTO, cambia cada turno) iba en
    posición #2, dejando solo ~150 tok estáticos antes del primer byte volátil
    — por debajo del mínimo del cache → el prefijo casi nunca hiteaba. Mover los
    ~1300 tok estáticos al frente cruza el mínimo y habilita el descuento de
    cache en turnos 2..N de la sesión. Es PURO reorden (mismo texto, cero cambio
    semántico). Flip a False revierte al orden legacy sin redeploy. Tooltip-
    anchor: P2-CHAT-PROMPT-STATIC-PREFIX."""
    return _env_bool("MEALFIT_CHAT_PROMPT_STATIC_PREFIX", True)

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

# [P0-DEEPSEEK-MIGRATION] Singleton módulo-level: se construye a import-time
# (sin user en contexto) → resuelve al modelo FREE. Los paths per-request
# (call_model/swap_meal) construyen su LLM con tier del usuario real.
llm = ChatDeepSeek(
    model=_chat_agent_model_name(),
    temperature=0.2,
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
# [P3-SWAP-FALLBACK-TITLE-STRIP · 2026-05-23] Helper que extrae el nombre
# limpio de un display string tipo `'1 Cabeza (~500g) Brócoli'` → `'Brócoli'`.
# Necesario porque `get_realtime_pantry()` (shopping_calculator) retorna el
# output de `aggregate_and_deduct_shopping_list()`, que produce strings con
# formato display (cantidad + unidad + paréntesis + opcional 'de' + nombre).
# El P3-SWAP-FALLBACK-TITLE-COPY del día anterior solo cubría el caso DICT
# del empty-pantry-fallback, pero el path productivo dominante (realtime
# pantry NO vacío) emitía estos display strings al fallback title sin
# limpieza — verificado en caso real 2026-05-23 00:09 donde el title fue
# `"Cena con 1 Cabeza (~500g) Brócoli y 1 Mazo Cilantro"`.
#
# Estrategia:
#   1. Si el string NO empieza con dígito/fracción → ya es nombre limpio,
#      retorna as-is (idempotente para inputs ya limpios como "Pollo").
#   2. Si hay "<algo> de <NOMBRE>" → split en el PRIMER " de " y toma la
#      última parte ("1 botella (250ml) de Aceite de oliva" → "Aceite de oliva").
#   3. Si no hay "de" connector → strip el prefijo [num][unit][optional paren]
#      ("1 Cabeza (~500g) Brócoli" → "Brócoli").
def _extract_clean_name_from_display_string(s: str) -> str:
    import re as _re_extract
    if not isinstance(s, str):
        return ""
    cleaned = s.strip()
    if not cleaned:
        return ""
    # Si NO empieza con número/fracción, asumimos que ya es nombre limpio
    # (idempotente para inputs como "Pollo", "Lechuga", "Aceite de oliva").
    if not _re_extract.match(r"^[\d¼½¾⅓⅔.,]", cleaned):
        return cleaned
    # Split en el primer " de " (case-insensitive) si existe — los strings
    # del agg suelen tener formato "<qty> <unit> (<paren>) de <NAME>".
    parts = _re_extract.split(r"\s+de\s+", cleaned, maxsplit=1, flags=_re_extract.IGNORECASE)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    # Sin " de " — strip prefijo qty + unit + optional parenthetical.
    # Pattern: "1 Cabeza (~500g) Brócoli" → "Brócoli"
    cleaned2 = _re_extract.sub(
        r"^[\d¼½¾⅓⅔.,]+\s*"  # número o fracción
        r"[\wáéíóúñÁÉÍÓÚÑ]+\.?\s*"  # palabra-unidad (Cabeza, Mazo, lb, Ud., ...)
        r"(?:\([^)]*\)\s*)?",  # paréntesis opcional (~500g)
        "",
        cleaned,
    )
    return cleaned2.strip() or cleaned


def swap_meal(form_data: dict):
    rejected_meal = form_data.get("rejected_meal", "")
    meal_type = form_data.get("meal_type", "Comida")
    target_calories = form_data.get("target_calories", 0)
    diet_type = form_data.get("diet_type", "balanced")

    # [P1-SWAP-MACROS · 2026-05-22] Targets per-meal: si el cliente envía
    # target_protein/carbs/fats explícitos (pre-rejected meal's macros) los
    # usamos. Fallback: derivar desde target_calories vía MACRO_SPLITS del
    # goal (mismo cálculo que el plan principal en `calculate_macros`). Si
    # ni target_calories existe (legacy clients) → todos 0 = validador
    # skip-en-silencio per-key. Tooltip-anchor: P1-SWAP-MACROS-DERIVE.
    target_protein = form_data.get("target_protein") or 0
    target_carbs = form_data.get("target_carbs") or 0
    target_fats = form_data.get("target_fats") or 0
    if (not target_protein or not target_carbs or not target_fats) and target_calories:
        try:
            from nutrition_calculator import calculate_macros as _calc_macros
            _split = _calc_macros(int(target_calories), form_data.get("goal", "maintenance"))
            target_protein = target_protein or _split.get("protein_g", 0)
            target_carbs = target_carbs or _split.get("carbs_g", 0)
            target_fats = target_fats or _split.get("fats_g", 0)
        except Exception as _macro_e:
            logger.debug(f"[P1-SWAP-MACROS] No se derivaron targets desde calc_macros: {_macro_e}")

    # [P2-8-SWAP-SLOT-TARGET · 2026-06-23] (audit inteligencia P2-8) Validar contra el slot OBJETIVO
    # (derivado del objetivo diario del usuario) en vez del plato ACTUAL, que puede venir drifteado
    # (un desayuno a 12g cuando el slot pide 30g → todo swap se ancla a ~12g → el drift se vuelve
    # permanente). form_data trae biométricos hidratados server-side por el router (P2-12). REEMPLAZA
    # los target_* heredados; fallback a ellos si no hay daily targets / error.
    # [P1-VERIFIED-ONLY-DEFAULT-ON · 2026-07-02] Default OFF→ON en código: el knob corre ON en prod
    # vía .env desde 2026-06-27 (activación P1-SLOT-APPROPRIATENESS Fase 2-resto) y el anclaje al slot
    # es el contrato vigente (P2-CHATMOD-TARGET-ANCHOR ancla proteína al slot en chat-modify) — dejarlo
    # OFF-en-código era la regresión silenciosa ".env reseteado ⇒ target drifteado vuelve". El riesgo
    # pantry-strict citado para el A/B quedó mitigado por el skip explícito de regen-day (abajo).
    # Rollback sin redeploy: MEALFIT_SWAP_TARGET_FROM_SLOT=false. Knob MEALFIT_SWAP_TARGET_FROM_SLOT.
    _p28_uid = form_data.get("user_id")
    if (
        _p28_uid and _p28_uid != "guest"
        and os.environ.get("MEALFIT_SWAP_TARGET_FROM_SLOT", "true").strip().lower() in ("1", "true", "yes", "on")
        # [P2-REGEN-DAY-SLOT-OVERRIDE-SKIP · 2026-06-29] regenerate-day YA retargetea cada plato hacia el
        # objetivo del DÍA (P1-REGEN-DAY-RETARGET, contra el target REAL del plan) y pasa esos targets per-comida
        # en target_*. El slot-override re-deriva el target con `get_nutrition_targets(form_data)`, PERO el
        # meal_form de regen NO trae biométricos (weight/height/age) → cae a defaults (154lb/170/25 → ~2949 kcal,
        # vs el goal real ~2141) → sobre-asigna cada slot → el día sale fuera de banda (band_score 0.0). En regen,
        # los targets del retarget son AUTORITATIVOS → saltamos el override. El swap standalone (con biométricos
        # en el request) lo conserva. tooltip-anchor: P2-REGEN-DAY-SLOT-OVERRIDE-SKIP
        and not form_data.get("_skip_slot_target_override")
    ):
        try:
            from nutrition_calculator import get_nutrition_targets as _gnt8, allocate_macros_per_slot as _alloc8
            _nt8 = _gnt8(form_data)
            if _nt8 and _nt8.get("target_calories"):
                _m8 = _nt8.get("macros") or {}
                _daily8 = {
                    "kcal": _nt8.get("target_calories"), "protein": _m8.get("protein_g"),
                    "carbs": _m8.get("carbs_g"), "fats": _m8.get("fats_g"),
                }
                try:
                    _num8 = int(form_data.get("num_meals") or form_data.get("mealsPerDay") or 4)
                except (TypeError, ValueError):
                    _num8 = 4
                _slots8 = _alloc8(_daily8, _num8)
                _mt8 = strip_accents(str(meal_type or "").lower()).strip()
                _slot_key8 = next(
                    (k for k in _slots8 if k == _mt8 or k.split("_")[0] in _mt8 or _mt8 in k),
                    None,
                )
                if _slot_key8 is None and "merienda" in _mt8:
                    _slot_key8 = next((k for k in _slots8 if k.startswith("merienda")), None)
                _st8 = _slots8.get(_slot_key8) if _slot_key8 else None
                if _st8 and _st8.get("protein"):
                    target_calories = round(_st8["kcal"])
                    target_protein = round(_st8["protein"])
                    target_carbs = round(_st8["carbs"])
                    target_fats = round(_st8["fats"])
                    logger.info(
                        f"🎯 [P2-8-SWAP-SLOT-TARGET] target del slot '{_slot_key8}': "
                        f"{target_calories}kcal / {target_protein}g P (era plato actual)"
                    )
        except Exception as _p28_e:
            logger.debug(f"[P2-8-SWAP-SLOT-TARGET] slot target falló (no bloquea): {_p28_e}")

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

    # [P1-SWAP-SAME-DAY-VARIETY · 2026-06-27] Las otras comidas del MISMO día → el plato nuevo NO debe repetir
    # su proteína/alimento principal (el swap era ciego al día → metía soya/huevo cuando otra comida ya lo usaba).
    same_day_other_meals = form_data.get("same_day_other_meals") or []
    if same_day_other_meals:
        # [P1-SWAP-SAME-DAY-VARIETY · 2026-06-27] PREFERENCIA (no obligación): comer lo mismo el mismo día
        # fatiga, pero esto NO debe pelear con el guard de despensa (usar lo comprado) ni hacer fallar el swap.
        # Por eso es soft + "usando ingredientes disponibles": si la única opción viable repite, entrega plato
        # válido igual. (Si la generación es full-variety sin despensa, el gate determinista de S1 igual aplica.)
        context_extras += (
            f"\n    - 🔄 VARIEDAD DEL DÍA (preferencia fuerte): las OTRAS comidas de HOY son: "
            f"{', '.join(same_day_other_meals)}. Comer el mismo alimento dos veces el mismo día fatiga → "
            f"PREFIERE una proteína/alimento principal DISTINTO al de esas comidas, eligiendo entre los "
            f"ingredientes que YA tienes disponibles. Si NO hay otra proteína disponible en tu despensa, "
            f"prioriza un plato VÁLIDO y coherente aunque repita (no inventes ingredientes que no tengas)."
        )

    swap_reason = form_data.get("swap_reason", "dislike")
    
    if swap_reason == 'variety':
        context_extras += "\n    - 💡 INTENCIÓN: El usuario NO rechaza este plato, solo quiere VARIEDAD. Sugiere combinaciones creativas, diferentes técnicas de cocción o perfiles de sabor novedosos pero accesibles."
    elif swap_reason == 'time':
        context_extras += "\n    - ⏱️ INTENCIÓN: El usuario NO TIENE TIEMPO HOY. Propón una receta extremadamente rápida (< 20 min), preferiblemente sin cocción extensa o usando ingredientes fáciles de armar."
    # [P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Branches 'budget' y 'pantry_first'
    # eliminados del elif chain. Pre-fix 'budget' tenía un hint específico
    # ("📦 APROVECHAR SU NEVERA / LISTA DE COMPRAS") expuesto al user via
    # opción del modal — sugería que los demás reasons NO usaban la nevera,
    # cuando la nevera SIEMPRE es la fuente única (excepto antojos/weekend).
    # Decisión de producto: strict-pantry pasa a ser el DEFAULT para todos
    # los reasons base (variety/time/similar/dislike) y se inyecta un hint
    # genérico "RESPETA LA NEVERA" debajo del elif chain cuando swap_reason
    # ∉ {cravings, weekend}. Backend acepta 'budget'/'pantry_first' como
    # input por back-compat (legacy callers / clientes antiguos cached) —
    # entran al mismo path genérico via el guard `if swap_reason not in (...)`.
    elif swap_reason == 'cravings':
        context_extras += "\n    - 🤤 INTENCIÓN: El usuario tiene un ANTOJO. Propón algo indulgente, comfort food o una versión saludable de comida rápida, pero manteniendo los macros."
    elif swap_reason == 'weekend':
        context_extras += "\n    - 🎉 INTENCIÓN: FIN DE SEMANA ESPECIAL. El usuario quiere un plato más elaborado, festivo o premium. Ideal para disfrutar con tiempo."
    # [P2-SWAP-CONSISTENCY · 2026-05-22] Branch 'similar' eliminado: el helper
    # `_pick_by_inverse_freq` + el filtro `available_proteins/carbs/veggies =
    # [x for x in filtered if x.lower() not in rejected_lower]` (más abajo en
    # esta misma función) ya excluyen proteína/carb/veggie del meal rechazado
    # deterministically y sesgan hacia ingredientes con baja frecuencia
    # histórica — exactamente el efecto que el hint LLM duplicaba. El branch
    # era eco innecesario que solo confundía al LLM con instrucción
    # redundante. swap_reason='similar' sigue siendo válido en el modal y
    # llega aquí como reason pasivo (sin context_extras extra).

    # [P3-SWAP-PANTRY-DEFAULT · 2026-05-22] Hint genérico de pantry para
    # TODOS los reasons base (variety/time/similar/dislike + back-compat
    # budget/pantry_first). Antes solo 'budget' tenía este hint, pero al
    # convertir strict-pantry en default el LLM necesita la instrucción
    # explícita para no producir externos que el validator post-gen
    # rechazaría (retry overhead innecesario). cravings/weekend quedan
    # opt-out del hint (su tolerancia externa está reflejada en el prompt
    # via su propio context_extras + allow_external_count del validator).
    if swap_reason not in ("cravings", "weekend"):
        context_extras += "\n    - 📦 RESPETA LA NEVERA: Limítate a los ingredientes ya disponibles (la regla de reciclaje a continuación enumera la base exacta). Sin compras nuevas."


    # --- REGLA CRÍTICA: ROTACIÓN CON INGREDIENTES EXISTENTES (ZERO-TRUST) ---
    clean_ingredients = []
    # [P1-RENAL-UPDATE-ENFORCE · 2026-06-24] (re-audit P1-1) ¿El plan activo lleva cap renal KDIGO? Lo
    # leemos del plan persistido (se setea en S1). Si aplica, trimamos la proteína del plato nuevo al
    # techo del slot antes de devolverlo (un swap NO debe romper el cap iatrogénico). Default False.
    _renal_capped = False
    # [P1-UPDATE-CROSS-DAY-VARIETY · 2026-06-23] (audit inteligencia P1-5) Texto de las OTRAS comidas
    # del plan activo (acent-stripped) para sesgar la sugerencia anti mode-collapse hacia proteínas
    # NO presentes ya en el plan → un swap "para variar" no devuelve la misma proteína que domina el
    # resto del plan (señal que `get_user_ingredient_frequencies` pierde en un plan recién generado).
    _plan_meals_text_for_variety = ""
    user_id = form_data.get("user_id")

    # [P2-REGEN-DAY-PANTRY-OVERRIDE · 2026-06-24] (re-audit P2-5) Cuando el caller (loop de regenerate-day)
    # provee un ledger de pantry YA reservado (gramos restantes tras los platos del día ya aceptados), ESE
    # ledger es la fuente de verdad de la nevera — NO la nevera-virtual completa del plan
    # (`get_realtime_pantry`), que ignora la reserva inter-plato (D7) y deja que 2 platos del mismo día
    # reclamen el mismo ingrediente escaso. Honramos el override explícito. Default ON.
    _pantry_override = (
        bool(form_data.get("pantry_override"))
        and os.environ.get("MEALFIT_REGEN_DAY_PANTRY_OVERRIDE", "true").strip().lower() in ("1", "true", "yes", "on")
    )
    _override_lines = form_data.get("current_pantry_ingredients") if _pantry_override else None
    _has_override = bool(_override_lines and isinstance(_override_lines, list) and len(_override_lines) > 0)

    # Intento Primario: Extraer ingredientes directamente del plan activo en BD
    if user_id and user_id != "guest":
        try:
            from db_plans import get_latest_meal_plan_with_id
            plan_record = get_latest_meal_plan_with_id(user_id)
            if plan_record and "plan_data" in plan_record:
                from db_facts import get_consumed_meals_since
                from shopping_calculator import get_realtime_pantry, aggregate_shopping_list as _agg_pantry

                plan_created_at = plan_record.get("created_at")
                consumed_ingredients = []
                if plan_created_at:
                    consumed_meals_list = get_consumed_meals_since(user_id, plan_created_at)
                    for cm in consumed_meals_list:
                        ings = cm.get("ingredients") or []
                        if isinstance(ings, list):
                            consumed_ingredients.extend(ings)

                if _has_override:
                    # [P2-REGEN-DAY-PANTRY-OVERRIDE] El ledger reservado gana sobre la nevera-virtual del plan.
                    clean_ingredients = _agg_pantry([str(i).strip() for i in _override_lines if i and isinstance(i, str) and len(str(i)) > 2])
                else:
                    clean_ingredients = get_realtime_pantry(plan_record["plan_data"], consumed_ingredients)

                # [P1-RENAL-UPDATE-ENFORCE · 2026-06-24] Leer el flag del cap renal del plan persistido.
                try:
                    _renal_capped = bool(((plan_record.get("plan_data") or {}).get("renal_protein_cap") or {}).get("applied"))
                except Exception:
                    _renal_capped = False

                # [P1-UPDATE-CROSS-DAY-VARIETY] Capturar nombres de las OTRAS comidas del plan
                # (excluyendo la rechazada) para el sesgo de variedad cross-day más abajo.
                try:
                    _rej_low = strip_accents(str(rejected_meal or "").lower())
                    _names = []
                    for _d in (plan_record["plan_data"].get("days") or []):
                        for _m in (_d.get("meals") or []) if isinstance(_d, dict) else []:
                            if not isinstance(_m, dict):
                                continue
                            _nm = str(_m.get("name") or "")
                            if _nm and strip_accents(_nm.lower()) != _rej_low:
                                _names.append(_nm)
                    _plan_meals_text_for_variety = strip_accents(" ".join(_names).lower())
                except Exception:
                    _plan_meals_text_for_variety = ""
        except Exception as e:
            logger.error(f"⚠️ [SWAP_MEAL] Error extrayendo inventario desde BD: {e}")

    # Fallback: Usar lista enviada por el front si falló BD o es guest
    if not clean_ingredients:
        current_pantry_ingredients = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
        if current_pantry_ingredients and isinstance(current_pantry_ingredients, list) and len(current_pantry_ingredients) > 0:
            from shopping_calculator import aggregate_shopping_list
            clean_ingredients = aggregate_shopping_list([item.strip() for item in current_pantry_ingredients if item and isinstance(item, str) and len(item) > 2])

    # [P1-SWAP-EMPTY-PANTRY-FALLBACK · 2026-05-22] Si el realtime pantry quedó
    # vacío (todos los items del plan se consumieron) Y el frontend tampoco
    # envió `current_pantry_ingredients`, leer la `aggregated_shopping_list`
    # entera del plan_data como source-of-truth de ingredientes. Cierra el
    # requisito explícito del owner verificado audit 2026-05-22:
    # > "si la nevera está vacía debe tomar en cuenta la lista de compras
    # > pdf para crear los platos personalizados"
    # Pre-fix: `clean_ingredients` caía al hardcoded ["Pollo","Arroz",
    # "Aguacate"] (línea ~769) ignorando la lista del PDF que el user ya
    # comprometió como su nevera futura. Espejo del patrón ya implementado
    # en `tools.py::execute_modify_single_meal:570-576`. Knob
    # `MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST=false` desactiva
    # el fallback (vuelve al comportamiento legacy). Tooltip-anchor:
    # P1-SWAP-EMPTY-PANTRY-FALLBACK.
    if (
        not clean_ingredients
        and user_id
        and user_id != "guest"
        and os.environ.get(
            "MEALFIT_SWAP_EMPTY_PANTRY_FALLBACK_TO_SHOPPING_LIST",
            "true",
        ).lower() != "false"
    ):
        try:
            from db_plans import get_latest_meal_plan_with_id
            _fallback_plan_record = get_latest_meal_plan_with_id(user_id)
            if _fallback_plan_record and isinstance(
                _fallback_plan_record.get("plan_data"), dict
            ):
                _shopping_raw = (
                    _fallback_plan_record["plan_data"].get(
                        "aggregated_shopping_list"
                    )
                    or []
                )
                if isinstance(_shopping_raw, list) and _shopping_raw:
                    _shopping_fallback = []
                    for _item in _shopping_raw:
                        if isinstance(_item, dict):
                            # [P3-SWAP-FALLBACK-TITLE-COPY · 2026-05-22]
                            # Preferir `name` (limpio, ej "Lechuga") sobre
                            # `display_string` (formateado, ej "1 Cabeza
                            # (~400g) Lechuga"). El LLM no necesita las
                            # cantidades user-específicas para generar una
                            # receta nueva — feedearlas confundía y además
                            # el fallback title las exponía crudas con
                            # `.title()` mangling units ("(~400G)").
                            _val = (
                                _item.get("name")
                                or _item.get("display_string")
                                or ""
                            )
                        else:
                            _val = str(_item)
                        _val = _val.strip()
                        if _val and _val not in _shopping_fallback:
                            _shopping_fallback.append(_val)
                    if _shopping_fallback:
                        clean_ingredients = _shopping_fallback
                        logger.info(
                            f"📦 [P1-SWAP-EMPTY-PANTRY-FALLBACK] pantry vacío; "
                            f"usando aggregated_shopping_list del plan "
                            f"({len(clean_ingredients)} items) como nevera "
                            f"virtual."
                        )
        except Exception as _shop_fallback_exc:
            logger.debug(
                f"[P1-SWAP-EMPTY-PANTRY-FALLBACK] fallback falló (no "
                f"bloquea swap): {type(_shop_fallback_exc).__name__}: "
                f"{_shop_fallback_exc}"
            )

    if clean_ingredients:
        # [P5-SWAP-PORTION-DISCIPLINE · 2026-06-23] Antes el bloque listaba los ingredientes
        # pero NO daba disciplina de PORCIÓN → el LLM proponía cantidades grandes y el pantry
        # guard rechazaba por `over_limit` → reintentos (swap lento, ~26s/2 retries). Añadimos
        # la regla de "porciones moderadas de UN solo plato" para que acierte a la primera.
        context_extras += (
            f"\n    - ⚠️ REGLA DE RECICLAJE (ROTACIÓN DE DESPENSA): El usuario quiere cambiar este plato pero DEBES "
            f"utilizar ingredientes que ya estén en su despensa/lista actual. Ingredientes disponibles: "
            f"{', '.join(clean_ingredients)}. Tienes permiso creativo para proponer un plato usando solo esta base, "
            f"sin agregar ingredientes foráneos."
            f"\n    - 📏 CANTIDADES (CRÍTICO para no fallar): es UN SOLO plato para UNA comida. Usa porciones MODERADAS "
            f"y realistas por ingrediente (las normales de un plato individual: p.ej. ~100-150g de proteína, ~1 taza de "
            f"carbohidrato), NUNCA cantidades grandes ni 'toda la despensa'. El inventario es LIMITADO: si de un "
            f"ingrediente hay poco, úsalo en cantidad pequeña o no lo incluyas. Pedir más de lo que el usuario tiene "
            f"hará que el plato se rechace."
        )
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

        # [P1-UPDATE-CROSS-DAY-VARIETY · 2026-06-23] (audit inteligencia P1-5) Sesgar la sugerencia
        # hacia proteínas que NO aparecen ya en el resto del plan → un swap "para variar" aumenta la
        # variedad cross-day en vez de devolver la proteína dominante. Bias SUAVE: solo restringe si
        # quedan proteínas "frescas"; si todas ya están en el plan (o no hay texto del plan / guest),
        # mantiene available_proteins intacto (nunca deja sin candidatos). Knob.
        if (
            _plan_meals_text_for_variety
            and os.environ.get("MEALFIT_UPDATE_CROSS_DAY_VARIETY", "true").strip().lower() in ("1", "true", "yes", "on")
        ):
            _fresh_proteins = [
                p for p in available_proteins
                if strip_accents(p.lower()) not in _plan_meals_text_for_variety
            ]
            if _fresh_proteins:
                available_proteins = _fresh_proteins

        # [P2-9-GAINMUSCLE-MAINS · 2026-06-23] (audit inteligencia P2-9) gain_muscle: sesgar la
        # SUGERENCIA a proteínas de ALTA densidad (excluir mains de baja densidad: leguminosas /
        # ricotta-cottage-crema / yogurt regular) — paridad con el esqueleto de S1
        # (P3-GAINMUSCLE-PROTEIN-DENSITY, mismo set módulo-level). Antes el swap/regenerate-day podía
        # elegir Ricotta/Habichuelas como main → día bajo el piso de proteína. Graceful: si NO quedan de
        # alta densidad, conserva las disponibles. Knob compartido MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN.
        _swap_goal = (form_data.get("goal") or form_data.get("mainGoal") or "").strip().lower()
        if (
            _swap_goal == "gain_muscle"
            and os.environ.get("MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN", "true").strip().lower() in ("1", "true", "yes", "on")
        ):
            try:
                from ai_helpers import _LOW_DENSITY_AS_MAIN as _LDM
                _hd_proteins = [p for p in available_proteins if p.lower() not in _LDM]
                if _hd_proteins:
                    available_proteins = _hd_proteins
            except Exception as _gm_e:
                logger.debug(f"[P2-9-GAINMUSCLE-MAINS] filtro densidad falló (no bloquea): {_gm_e}")
        
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
    
    # [P1-UPDATE-SUPERPERS · 2026-06-23] (audit inteligencia P1-4) Inyectar súper-personalización
    # (gustos/cocina/religión/equipo/sabor/nivel) al prompt del swap — paridad con S1. Incluye la
    # exclusión DURA de religión (sin_cerdo/sin_alcohol/halal/kosher) que sin esto se reintroducía.
    if os.environ.get("MEALFIT_UPDATE_SUPERPERS", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from prompts.plan_generator import build_super_personalization_context
            _sp_block = build_super_personalization_context(form_data)
            if _sp_block:
                context_extras += "\n    " + _sp_block.strip()
        except Exception as _sp_e:
            logger.debug(f"[P1-UPDATE-SUPERPERS] super-pers context falló (no bloquea): {_sp_e}")
    # [P1-UPDATE-MICROS · 2026-06-23] (audit inteligencia P1-7) Inyectar directivas de condición
    # médica + fármaco-alimento (DM2/HTA/renal/anemia/embarazo/warfarina) al prompt del swap —
    # paridad con la directiva de S1. form_data trae medicalConditions/medications enriquecidos
    # server-side por el router.
    if os.environ.get("MEALFIT_UPDATE_CONDITION_DIRECTIVES", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from condition_rules import build_condition_prompt
            from medication_rules import build_medication_prompt
            _cond_block = build_condition_prompt(form_data)
            if _cond_block:
                context_extras += "\n    " + _cond_block.strip()
            _med_block = build_medication_prompt(form_data)
            if _med_block:
                context_extras += "\n    " + _med_block.strip()
        except Exception as _cond_e:
            logger.debug(f"[P1-UPDATE-MICROS] directivas condición/fármaco fallaron (no bloquea): {_cond_e}")

    # [P1-SLOT-APPROPRIATENESS · 2026-06-27] (audit G4) Inyecta las reglas de coherencia de HORARIO del
    # slot al prompt del swap (paridad con day_generator §9/§15 de S1). El usuario solo pidió "cámbialo"
    # → el sistema debe elegir un plato propio del horario. SSOT constants.build_meal_timing_rules.
    if SLOT_APPROPRIATENESS_GATE_ENABLED:
        try:
            from constants import build_meal_timing_rules as _bmtr
            _timing_block = _bmtr(meal_type)
            if _timing_block:
                context_extras += _timing_block
        except Exception as _tr_e:
            logger.debug(f"[P1-SLOT-APPROPRIATENESS] timing rules swap fallaron (no bloquea): {_tr_e}")

    # [P2-UPDATE-MICRO-STEER · 2026-06-27] (audit G2) Inyecta los pisos de micros (Mg/Fe/Ca/fibra/K) al prompt
    # del swap — el usuario SANO sin condición no los recibía (S1 sí; paridad de densidad nutricional). SOLO
    # cuando NO hay pantry detectada (usuario va de compras): con la Nevera-strict el pantry manda y añadir
    # presión de micros subiría fallos de convergencia. SSOT graph_orchestrator.build_update_micronutrient_directive.
    if not clean_ingredients:
        try:
            from graph_orchestrator import build_update_micronutrient_directive as _bmd
            _micro_block = _bmd(form_data)
            if _micro_block:
                context_extras += "\n    " + _micro_block.strip()
        except Exception as _msw_e:
            logger.debug(f"[P2-UPDATE-MICRO-STEER] micro steer swap falló (no bloquea): {_msw_e}")
    else:
        # [P2-PANTRY-MICRO-SOFT · 2026-06-29] (audit objetivo · P2-12) En pantry-strict NO inyectamos el steer
        # CUANTITATIVO (subiría fallos de convergencia con la Nevera) pero SÍ una preferencia SUAVE: que, entre
        # lo disponible, priorice ingredientes densos en micros. Cierra la asimetría "S1 siempre orienta micros;
        # el update pantry-strict no daba NINGUNA guía". tooltip-anchor: P2-PANTRY-MICRO-SOFT
        try:
            from graph_orchestrator import MICRONUTRIENT_STEER_ENABLED as _mse
            if _mse:
                context_extras += ("\n    - 🧪 DENSIDAD DE MICROS (preferencia suave, sin salir de la Nevera): "
                                   "entre los ingredientes disponibles, prioriza los más ricos en magnesio, hierro, "
                                   "calcio y fibra (hojas verdes, leguminosas, semillas, vegetales de color).")
        except Exception:
            pass

    # [P2-VERIFIED-ONLY-UPDATE · 2026-06-29] (audit objetivo · P2-6) Paridad de catálogo con S1: cuando el usuario
    # va de compras (no pantry-strict), el LLM del swap podía inventar un alimento/especia fuera del catálogo
    # verificado-por-precio → sobrevive en el TEXTO de la receta y se cae de la lista (incoherencia receta↔lista,
    # no costeable). Inyectamos el MISMO bloque "USA EXCLUSIVAMENTE" de S1 (gated por el mismo knob vía el helper;
    # string cacheado). SKIP en pantry-strict (los ingredientes ya son del catálogo). tooltip-anchor: P2-VERIFIED-ONLY-UPDATE
    if not clean_ingredients:
        try:
            from graph_orchestrator import _get_verified_catalog_instruction as _gvci
            _vc_block = _gvci(form_data)
            if _vc_block:
                context_extras += "\n    " + _vc_block.strip()
        except Exception as _vcsw_e:
            logger.debug(f"[P2-VERIFIED-ONLY-UPDATE] verified catalog swap falló (no bloquea): {_vcsw_e}")

    prompt_text = SWAP_MEAL_PROMPT_TEMPLATE.format(
        rejected_meal=rejected_meal,
        meal_type=meal_type,
        target_calories=target_calories,
        target_protein=int(round(float(target_protein or 0))),
        target_carbs=int(round(float(target_carbs or 0))),
        target_fats=int(round(float(target_fats or 0))),
        diet_type=diet_type,
        context_extras=context_extras
    )
    
    temp = 0.3
    # [P0-DEEPSEEK-MIGRATION] Tier-routing: el endpoint /swap-meal valida
    # ownership de `user_id` contra el JWT ANTES de llegar acá (api_swap_meal).
    _swap_uid = form_data.get("user_id")
    swap_llm = ChatDeepSeek(
        model=_chat_agent_swap_model_name(_swap_uid),
        temperature=temp,
        timeout=_chat_swap_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
    ).with_structured_output(MealModel)

    # [P1-CHAT-CB-EXTEND · 2026-05-20] CB gate per-modelo del swap_llm.
    # Espejo del gate en `call_model` (P1-CHAT-CB · 2026-05-19). Pre-fix:
    # si Gemini estaba degradado, los swaps seguían golpeando el provider
    # sin fail-fast — tenacity retry 3× AGRAVABA la condición (3 attempts
    # × N concurrent swaps). Ahora: si breaker abierto raise
    # `LLMCircuitBreakerOpen` ANTES del retry loop. Propaga al caller
    # (router → HTTP 503, semánticamente "upstream saturado, reintentar
    # tras MEALFIT_CB_RESET_TIMEOUT_S"). NO ejecutamos el fallback "Plato
    # Seguro" en este caso — el fallback es para "validador rechazó 3
    # attempts", no para "provider degradado". Mantener asimetría es
    # explícito y defendible: 503 le dice al user "el sistema sabe que
    # algo está mal", el plato fallback parecería decisión culinaria.
    # Tooltip-anchor: P1-CHAT-CB-EXTEND.
    _swap_cb_model = _chat_agent_swap_model_name(_swap_uid)
    _swap_cb = _get_circuit_breaker(_swap_cb_model)
    if not _swap_cb.can_proceed():
        logger.warning(
            f"🛑 [P1-CHAT-CB-EXTEND] swap_meal CB abierto para "
            f"model={_swap_cb_model!r} — fail-fast sin invocar Gemini. "
            f"Reintentar tras MEALFIT_CB_RESET_TIMEOUT_S segundos."
        )
        raise LLMCircuitBreakerOpen(
            f"swap_meal LLM circuit breaker open for model={_swap_cb_model}"
        )

    # [P3-SWAP-PANTRY-DEFAULT · 2026-05-22] strict-pantry pasa a ser el
    # DEFAULT del swap (decisión de producto 2026-05-22: el botón "Usar
    # solo lo que tengo" se eliminó del modal porque ese comportamiento
    # ES el contrato del swap-meal). Ahora solo `cravings`/`weekend`
    # (indulgencia explícita) opt-out — pueden traer 1-2 ingredientes
    # externos via `allow_external_count`. Resto (variety/time/similar/
    # dislike + back-compat budget/pantry_first) → strict.
    #
    # [P1-SWAP-STRICT-PANTRY · 2026-05-22] Original: el guard de pantry
    # se eleva de hint cosmético a hard constraint. El validador es el
    # mismo (validate_ingredients_against_pantry); el fallback abortivo
    # NO usa la lista hardcoded ["Pollo","Arroz","Aguacate"] sino que
    # SOLO arma el plato con `clean_ingredients[:4]`. Si no hay
    # clean_ingredients y strict_pantry está activo → el fallback raise
    # explícito (router lo mapea a 422). Tooltip-anchor: P1-SWAP-STRICT-PANTRY.
    # [P4-UPDATE-DISHES-STRICT-ALL · 2026-06-23] Requisito del owner: los botones de
    # actualizar platos cocinan con lo que hay en la NEVERA — TODOS los motivos deben ser
    # pantry-strict, incluidos cravings/weekend (antes exentos para permitir ingredientes
    # externos). `swap_meal` es EXCLUSIVO de esos botones (el chat usa
    # execute_modify_single_meal), así que el cambio no afecta otros surfaces. Dark-ship:
    # default OFF en código (preserva legacy + tests cravings/weekend) → ON en prod vía .env.
    _strict_all = os.environ.get("MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS", "false").strip().lower() in ("1", "true", "yes", "on")
    strict_pantry = True if _strict_all else (swap_reason not in ("cravings", "weekend"))

    # [P2-SWAP-CONSISTENCY · 2026-05-22] Tolerancia de ingredientes externos
    # cuando el user pidió un antojo / plato festivo: hard-pantry colisionaba
    # con "indulgente" / "premium" (modal opts "Tengo un antojo" / "Fin de
    # semana especial"). Permitimos hasta N "unauthorized" sin abortar; el
    # validador suma esto a su check estructural. Knob
    # `MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED` (default 2, clamp [0, 5]).
    # cravings/weekend: usa el knob. Resto: 0 (legacy strict). Tooltip-anchor:
    # P2-SWAP-CONSISTENCY-EXTERNAL.
    if swap_reason in ("cravings", "weekend") and not _strict_all:
        try:
            _external_tolerance = int(os.environ.get("MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED", "2"))
        except (TypeError, ValueError):
            _external_tolerance = 2
        _external_tolerance = max(0, min(5, _external_tolerance))
    else:
        # [P4-UPDATE-DISHES-STRICT-ALL] strict-all → cero ingredientes externos para TODOS.
        _external_tolerance = 0

    # [P1-SWAP-MACROS · 2026-05-22] Buffer mutable del prompt para inyectar
    # feedback del validador (pantry + macros) en attempts 2 y 3. Mismo
    # patrón que `execute_modify_single_meal` (tools.py:647).
    _current_prompt = [prompt_text]

    # [P1-SWAP-MACROS] Lazy import — el módulo nutrition_calculator es
    # liviano pero tiene side-effects de logging que preferimos contained.
    # [P2-SWAP-CONSISTENCY · 2026-05-22] añade prep_time validator (solo
    # consultado si swap_reason='time').
    try:
        from nutrition_calculator import (
            validate_meal_macros_against_targets as _validate_macros,
            _meal_macros_validate_enabled as _macros_validate_enabled,
            validate_meal_recipe_ingredients_coherence as _validate_recipe_coh,
            _swap_recipe_coherence_enabled as _recipe_coh_enabled,
            validate_meal_prep_time_against_target as _validate_prep_time,
            _swap_prep_time_validate_enabled as _prep_time_validate_enabled,
        )
    except Exception:
        _validate_macros = None
        _macros_validate_enabled = lambda: False  # noqa: E731 — fallback no-op
        _validate_recipe_coh = None
        _recipe_coh_enabled = lambda: False  # noqa: E731
        _validate_prep_time = None
        _prep_time_validate_enabled = lambda: False  # noqa: E731

    # [P2-UPDATE-MACRO-TRUTHUP · 2026-06-24] (re-audit P2-1) Truth-up de macros desde los strings de
    # ingredientes ANTES del band-validator (bloque dentro de invoke_with_retry). Cierra el inflado del
    # JSON por el LLM. db lazy compartida entre los reintentos (evita recargar el índice 3×). Knob
    # MEALFIT_UPDATE_MACRO_TRUTHUP default ON. tooltip-anchor: P2-UPDATE-MACRO-TRUTHUP
    _tu_db_holder = [None]
    _update_macro_truthup_enabled = lambda: os.environ.get(  # noqa: E731
        "MEALFIT_UPDATE_MACRO_TRUTHUP", "true").strip().lower() in ("1", "true", "yes", "on")

    # Invocar LLM con reintentos automáticos (tenacity)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"🔁 [SWAP RETRY] attempt={retry_state.attempt_number} | "
            f"reason=guardrail_rejection | meal_type={meal_type}"
        )
    )
    def invoke_with_retry():
        res = swap_llm.invoke(_current_prompt[0])

        # Validación post-generación (guardrail determinista)
        if hasattr(res, "ingredients"):
            ingreds = getattr(res, "ingredients")
        elif isinstance(res, dict) and "ingredients" in res:
            ingreds = res["ingredients"]
        else:
            ingreds = []

        # Solo aplicamos restricción estricta si hay una despensa base limpia extraída
        if clean_ingredients:
            # [P2-SWAP-CONSISTENCY · 2026-05-22] `_external_tolerance` calculado
            # arriba según swap_reason. Default 0 (legacy strict); cravings/weekend
            # permiten hasta MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED externos.
            val_result = validate_ingredients_against_pantry(
                ingreds,
                clean_ingredients,
                allow_external_count=_external_tolerance,
            )
            if val_result is not True:
                logger.warning(val_result)
                _current_prompt[0] = prompt_text + (
                    f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{val_result}"
                    f"\nPor favor revisa el inventario y ajusta la receta para que cumpla estrictamente."
                )
                raise ValueError(val_result)

        # [P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Mini-coherence check
        # per-meal sobre el output del LLM: si la receta menciona una
        # proteína canónica que NO está en `ingredients`, gateamos retry
        # (`cap_swallowed_modifier` a nivel meal-output). Cierra el gap
        # user-facing dejado abierto en el bundle inicial — sin este
        # check, un swap que entregue receta con "el pollo" cuando
        # ingredients=["pavo"] llegaba al shopping aggregator y se
        # propagaba al PDF del user. Knob
        # `MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE=false` desactiva.
        if _validate_recipe_coh is not None and _recipe_coh_enabled():
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                coh_passed, coh_divs, coh_summary = _validate_recipe_coh(meal_dump)
                if not coh_passed:
                    logger.warning(
                        f"⚠️ [P1-SWAP-RECIPE-COHERENCE] divergence detected | "
                        f"meal_type={meal_type} | divs={coh_divs}"
                    )
                    # [P3-SWAP-RETRY-COHERENCE-HINT · 2026-05-22] Append
                    # self-check directive al retry prompt. Pre-fix solo
                    # inyectaba el coh_summary; el LLM podía repetir la
                    # misma discrepancia (verificado: 3 intentos seguidos
                    # con el alias "dorado"). El self-check explícito sube
                    # la señal y obliga al LLM a verificar invariante antes
                    # de outputtear.
                    _current_prompt[0] = prompt_text + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{coh_summary}"
                        f"\n\n🔒 REGLA INVARIANTE: ANTES de devolver tu respuesta, recorre "
                        f"cada paso del array `recipe` y verifica que TODO alimento "
                        f"mencionado aparezca también (o un sinónimo razonable) en el "
                        f"array `ingredients`. Si encuentras una discrepancia, corrígela "
                        f"agregando el ingrediente faltante CON cantidad o reescribiendo "
                        f"el paso sin mencionarlo. NO devuelvas la respuesta hasta verificar."
                    )
                    raise ValueError(coh_summary)
            except ValueError:
                raise
            except Exception as _coh_exc:
                logger.warning(
                    f"[P1-SWAP-RECIPE-COHERENCE] validator helper falló (no aborta): "
                    f"{type(_coh_exc).__name__}: {_coh_exc}"
                )

        # [P2-SWAP-CONSISTENCY · 2026-05-22] Validador prep_time per-meal
        # cuando swap_reason='time' ("No tengo tiempo hoy"). Pre-fix: el
        # prompt inyectaba el hint "<20 min" pero NO había enforcement
        # post-gen → el LLM podía emitir receta de 40 min sin retry.
        # Cierra el gap "soft-only" detectado en el audit del modal
        # "¿Por qué quieres cambiar?". Solo se ejecuta para reason='time';
        # otros reasons skipean (la mayoría de meals legítimos sin tiempo
        # crítico toman >20 min y no queremos forzarles retries). Knob
        # `MEALFIT_SWAP_PREP_TIME_VALIDATE=false` desactiva sin redeploy.
        if (
            swap_reason == 'time'
            and _validate_prep_time is not None
            and _prep_time_validate_enabled()
        ):
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                pt_passed, pt_actual, pt_summary = _validate_prep_time(meal_dump)
                if not pt_passed:
                    logger.warning(
                        f"⚠️ [P2-SWAP-PREP-TIME] PREP_TIME drift | "
                        f"meal_type={meal_type} | actual={pt_actual} min"
                    )
                    _current_prompt[0] = prompt_text + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{pt_summary}"
                    )
                    raise ValueError(pt_summary)
            except ValueError:
                raise
            except Exception as _pt_exc:
                logger.warning(
                    f"[P2-SWAP-PREP-TIME] validator helper falló (no aborta): "
                    f"{type(_pt_exc).__name__}: {_pt_exc}"
                )

        # [P2-UPDATE-MACRO-TRUTHUP · 2026-06-24] (re-audit P2-1) Recompute del NÚMERO de macros desde los
        # strings FINALES de ingredientes ANTES del band-validator → cierra el inflado del JSON por el LLM
        # (emite protein:30 con ingredientes que rinden ~12g → pasaba la banda y persistía; Dashboard/PDF/
        # day_quality_warning operaban sobre cifra fantasma). Espejo del Guard 8z de S1
        # (graph_orchestrator._truth_up_meal_macros_from_strings). Solo NÚMEROS (NO strings → lista de
        # compras intacta). Fail-safe. Mutamos `res` para que el band-validator y la persistencia (_out)
        # lean la cifra real. tooltip-anchor: P2-UPDATE-MACRO-TRUTHUP
        if _update_macro_truthup_enabled():
            try:
                from graph_orchestrator import _truth_up_meal_macros_from_strings as _tu_fn
                if _tu_db_holder[0] is None:
                    from nutrition_db import IngredientNutritionDB as _TUDB
                    _tu_db_holder[0] = _TUDB()
                _tu_meal = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                if _tu_fn(_tu_meal, _tu_db_holder[0]):
                    for _tk in ("protein", "carbs", "fats", "cals", "macros"):
                        if _tk in _tu_meal:
                            if isinstance(res, dict):
                                res[_tk] = _tu_meal[_tk]
                            elif hasattr(res, _tk):
                                setattr(res, _tk, _tu_meal[_tk])
                    logger.info(f"🔎 [P2-UPDATE-MACRO-TRUTHUP] macros swap recomputadas desde strings | meal_type={meal_type}")
            except Exception as _tu_exc:
                logger.warning(
                    f"[P2-UPDATE-MACRO-TRUTHUP] truth-up swap falló (no aborta): "
                    f"{type(_tu_exc).__name__}: {_tu_exc}"
                )

        # [P1-SWAP-MACROS · 2026-05-22] Validación post-gen de macros vs
        # targets del slot. Pre-fix: prompt solo enviaba target_calories
        # como hint soft → drift arbitrario permitido (caso real: target
        # 350kcal/15g protein → LLM emitía 450kcal/8g protein sin queja,
        # macros semanales driftaban +28% kcal -47% protein).
        # Si la validación falla, inyectamos el summary al retry prompt
        # (mismo patrón que pantry validator) y forzamos retry tenacity.
        # Knob `MEALFIT_SWAP_MACROS_VALIDATE=false` desactiva si introduce
        # demasiados retries en prod.
        if _validate_macros is not None and _macros_validate_enabled():
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                passed, drifts, summary = _validate_macros(
                    meal_dump,
                    {
                        "cals": target_calories,
                        "protein": target_protein,
                        "carbs": target_carbs,
                        "fats": target_fats,
                    },
                )
                if not passed:
                    logger.warning(
                        f"⚠️ [P1-SWAP-MACROS] Drift detectado attempt-pending | "
                        f"meal_type={meal_type} | drifts={drifts}"
                    )
                    _current_prompt[0] = prompt_text + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{summary}"
                    )
                    raise ValueError(summary)
            except ValueError:
                raise
            except Exception as _macros_exc:
                # Best-effort: si el helper rompe (drift de schema, etc.)
                # NO bloqueamos el swap — el LLM ya entregó algo válido.
                logger.warning(
                    f"[P1-SWAP-MACROS] validator helper falló (no aborta): "
                    f"{type(_macros_exc).__name__}: {_macros_exc}"
                )

        # [P0-UPDATE-CLINICAL-GUARD · 2026-06-23] Backstop clínico determinista (alérgenos + dieta
        # hard veg*). El swap NO pasa por el grafo (ni reviewer médico ni capa clínica de S1) → sin
        # esto un alérgeno declarado o un producto veg*-prohibido podía persistirse. `allergies` y
        # `diet_type` ya vienen enriquecidos SERVER-SIDE desde health_profile por el router
        # (api_swap_meal / api_regenerate_day). Violación → feedback al retry prompt; si persiste
        # tras los 3 intentos, el caller cae al path fail-secure (preserva el plato original, NO
        # emite fallback que pudiera violar). FAIL-SECURE: error del backstop = violación.
        # Knob MEALFIT_UPDATE_CLINICAL_GUARD=false revierte. tooltip-anchor: P0-UPDATE-CLINICAL-GUARD
        if UPDATE_CLINICAL_GUARD:
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                _clin_viol = clinical_backstop_for_meal(
                    meal_dump, allergies=allergies, diet_type=diet_type, form_data=form_data
                )
            except Exception as _clin_exc:
                _clin_viol = [f"error backstop clínico: {type(_clin_exc).__name__}"]
            if _clin_viol:
                logger.warning(
                    f"🛡 [P0-UPDATE-CLINICAL-GUARD] swap viola seguridad clínica | "
                    f"meal_type={meal_type} | viol={_clin_viol}"
                )
                _current_prompt[0] = prompt_text + (
                    f"\n\n🛑 SEGURIDAD CLÍNICA (OBLIGATORIO, NO NEGOCIABLE): el plato anterior incluyó: "
                    f"{'; '.join(_clin_viol)}. Está TERMINANTEMENTE PROHIBIDO incluir esos alimentos "
                    f"(alergias / restricción de dieta del usuario). Regenera el plato SIN ellos ni sus "
                    f"derivados. Si el cambio solicitado exige un alimento prohibido, ignóralo y propón "
                    f"una alternativa segura."
                )
                raise ValueError("CLINICAL_VIOLATION: " + "; ".join(_clin_viol))

        # [P1-SLOT-APPROPRIATENESS · 2026-06-27] (audit G4) Backstop de coherencia de HORARIO en swap:
        # el usuario solo pidió "cámbialo" (no un plato específico) → NO debemos meter un plato fuera de
        # horario ("arroz de noche", arroz/locrio en desayuno, comida de desayuno en cena). Espejo del
        # backstop clínico pero CALIDAD: presiona retry vía feedback; si persiste tras los retries, el
        # except cae al fallback slot-genérico. NO levanta en strict_pantry-sin-inventario (evita un 422
        # por una cuestión de calidad). El ValueError NO cuenta como CB failure (P2-CB-GUARDRAIL-NOT-FAILURE).
        if SLOT_APPROPRIATENESS_GATE_ENABLED and not (strict_pantry and not clean_ingredients):
            try:
                _slot_dump = res.model_dump() if hasattr(res, "model_dump") else (res if isinstance(res, dict) else {})
                _slot_viol = slot_coherence_backstop_for_meal(_slot_dump, meal_type)
            except Exception:
                _slot_viol = []
            if _slot_viol:
                logger.warning(
                    f"🕒 [P1-SLOT-APPROPRIATENESS] swap fuera de horario | meal_type={meal_type} | viol={_slot_viol}"
                )
                _current_prompt[0] = prompt_text + (
                    f"\n\n🕒 COHERENCIA DE HORARIO (OBLIGATORIO): el plato anterior no encaja con el horario "
                    f"«{meal_type}»: {'; '.join(_slot_viol)}. Propón un plato que SÍ corresponda a ese momento "
                    f"del día para un dominicano — el arroz/locrio/pasta van en almuerzo/cena (NUNCA desayuno); "
                    f"la cena es ligera (evita 'arroz de noche' y comidas de desayuno). Mantén los macros objetivo."
                )
                raise ValueError("SLOT_INCOHERENCE: " + "; ".join(_slot_viol))

        # [P1-UPDATE-APPETIBILITY · 2026-06-27] (audit Fase 0) Pareo chocante fruta+salado en swap
        # (ej. "Arroz con Mango"): el usuario solo pidió "cámbialo" → presiona retry para un plato
        # coherente (espejo del backstop de slot: fail-open, no 422 en strict_pantry-sin-inventario,
        # el ValueError NO cuenta como CB failure). La proteína fantasma se corrige determinista en _out.
        if UPDATE_APPETIBILITY_GUARD and not (strict_pantry and not clean_ingredients):
            try:
                _appet_dump = res.model_dump() if hasattr(res, "model_dump") else (res if isinstance(res, dict) else {})
                _has_clash = _meal_has_sweet_savory_clash(_appet_dump)
            except Exception:
                _has_clash = False
            if _has_clash:
                logger.warning(
                    f"🍓 [P1-UPDATE-APPETIBILITY] swap con pareo fruta+salado | meal_type={meal_type} | "
                    f"name={str(_appet_dump.get('name'))[:48]!r}"
                )
                _current_prompt[0] = prompt_text + (
                    "\n\n🍓 COHERENCIA DE SABOR (OBLIGATORIO): el plato anterior combina fruta dulce dominante "
                    "(mango, piña, lechosa…) con una base salada (arroz, huevo revuelto, crucíferas). Eso choca. "
                    "La fruta dulce va con yogur/avena/nueces/queso fresco o sola — NUNCA con arroz, huevo salado "
                    "ni vegetales salados. Reemplaza la fruta por una guarnición salada coherente. Mantén los macros."
                )
                raise ValueError("SWEET_SAVORY_CLASH")

        # [P2-UPDATE-SAMEDAY-VARIETY · 2026-07-01] (audit slots GAP-4 / paridad GAP-4) La variedad same-day en
        # swap era SOLO prompt ("preferencia") → «cámbiame la cena» devolvía pechuga cuando el almuerzo YA era
        # pollo — exactamente la asimetría que P1-VARIETY-SAME-DAY-PROTEIN cerró en form-gen (gate). Backstop
        # determinista: si la proteína principal del plato nuevo coincide con la de otra comida de HOY → 1 retry
        # (marker en el prompt evita loops); en el reintento se entrega con log advisory (repetir es cosmético;
        # NUNCA fallback por esto). Word-boundary anti-'res'-en-'fresas'. Skip en strict_pantry (repetir puede
        # ser inevitable cocinando de la nevera). tooltip-anchor: P2-UPDATE-SAMEDAY-VARIETY
        if UPDATE_APPETIBILITY_GUARD and same_day_other_meals and not strict_pantry:
            try:
                import re as _re_sd
                from constants import strip_accents as _sa_sd
                _SD_PROT = {
                    "pollo": ("pollo", "pechuga", "muslo"), "cerdo": ("cerdo", "chuleta", "longaniza"),
                    "res": ("res", "bistec", "molida", "churrasco"), "pavo": ("pavo",),
                    "pescado": ("pescado", "tilapia", "salmon", "mero", "bacalao", "chillo", "merluza"),
                    "camarones": ("camaron", "camarones"), "atun": ("atun",),
                    "huevo": ("huevo", "huevos", "revoltillo"),
                }
                _sd_dump = res.model_dump() if hasattr(res, "model_dump") else (res if isinstance(res, dict) else {})
                _sd_name = _sa_sd(str(_sd_dump.get("name", "")).lower())
                _new_prot = next((c for c, syns in _SD_PROT.items()
                                  if any(_re_sd.search(r"\b" + s + r"\b", _sd_name) for s in syns)), None)
                if _new_prot:
                    _other_blob = _sa_sd(" ".join(str(x) for x in same_day_other_meals).lower())
                    _repeats = any(_re_sd.search(r"\b" + s + r"\b", _other_blob) for s in _SD_PROT[_new_prot])
                    if _repeats:
                        _SD_MARKER = "🔄 RETRY VARIEDAD DEL DÍA"
                        if _SD_MARKER not in str(_current_prompt[0]):
                            _current_prompt[0] = prompt_text + (
                                f"\n\n{_SD_MARKER} (OBLIGATORIO): el plato anterior repite la proteína "
                                f"«{_new_prot}» que OTRA comida de HOY ya usa ({', '.join(same_day_other_meals[:3])}). "
                                f"Propón un plato con una proteína principal DISTINTA. Mantén los macros objetivo."
                            )
                            raise ValueError(f"SAME_DAY_PROTEIN_REPEAT: {_new_prot}")
                        logger.info(f"🔄 [P2-UPDATE-SAMEDAY-VARIETY] swap repite '{_new_prot}' tras el retry — "
                                    f"entregado con advisory | meal_type={meal_type}")
            except ValueError:
                raise
            except Exception as _sd_e:
                logger.warning(f"[P2-UPDATE-SAMEDAY-VARIETY] backstop same-day falló (no bloquea): "
                               f"{type(_sd_e).__name__}: {_sd_e}")

        return res

    try:
        response = invoke_with_retry()
        # [P1-CHAT-CB-EXTEND · 2026-05-20] Marcar éxito en el CB tras
        # invoke + validación OK (mismo punto que `record_success` en
        # `call_model`). El reset_timeout window se renueva acá.
        _swap_cb.record_success()
    except Exception as e:
        # [P1-CHAT-CB-EXTEND · 2026-05-20] Discriminar antes de marcar
        # failure: rate-limit del provider (429/ResourceExhausted) NO
        # cuenta como CB failure — espejo del patrón en `call_model`
        # (P1-CHAT-LLM-429 · 2026-05-20). Si fueran las 3 attempts de
        # tenacity falladas por 429, propagamos como `LLMRateLimitedError`
        # (router → HTTP 429 con Retry-After) y NO ejecutamos el fallback
        # "Plato Seguro" — semánticamente distinto a "validador rechazó".
        # Resto de errores (timeout, 5xx, ValidationError del guardrail):
        # `record_failure` + mantener fallback existente como degradación
        # graceful (UX preservada).
        if _is_rate_limit_error(e):
            _emit_chat_rate_limited_metric_best_effort(
                form_data.get("user_id"),
                form_data.get("session_id"),
                _swap_cb_model,
            )
            logger.warning(
                f"⚠️ [P1-CHAT-LLM-429] swap_meal Gemini rate-limit "
                f"model={_swap_cb_model!r} — NO cuenta como CB failure."
            )
            raise LLMRateLimitedError(
                f"swap_meal LLM rate limited for model={_swap_cb_model}: {e!r}"
            ) from e
        # [P2-CB-GUARDRAIL-NOT-FAILURE · 2026-06-24] Un rechazo de GUARDRAIL/validador (coherencia
        # receta↔lista, macros, prep-time, clínico, pantry → ValueError) significa que el PROVEEDOR
        # respondió pero el output no pasó NUESTRA validación — NO es señal de salud del proveedor.
        # Contarlo como CB failure abría el breaker por un plato "difícil" y, al ser per-modelo
        # COMPARTIDO, tumbaba el regenerate-day/swaps de TODOS los usuarios (caso real 2026-06-24:
        # 'dorado' no listado en la receta agotó los 3 retries → breaker abierto → merienda/cena del
        # día ni se intentaron). Solo los errores REALES de transporte/proveedor (timeout/5xx/conexión)
        # cuentan; los validadores levantan ValueError, un fallo de proveedor NO. Knob
        # MEALFIT_SWAP_CB_COUNT_GUARDRAIL=true revierte al comportamiento anterior.
        # tooltip-anchor: P2-CB-GUARDRAIL-NOT-FAILURE
        _cb_count_guardrail = os.environ.get(
            "MEALFIT_SWAP_CB_COUNT_GUARDRAIL", "false").strip().lower() in ("1", "true", "yes", "on")
        if isinstance(e, ValueError) and not _cb_count_guardrail:
            logger.info(
                f"🎚 [P2-CB-GUARDRAIL-NOT-FAILURE] rechazo de guardrail NO cuenta como CB failure "
                f"(proveedor sano) | meal_type={meal_type}"
            )
        else:
            _swap_cb.record_failure()
        logger.error(f"❌ [SWAP_MEAL] Fallaron los intentos LLM y validador: {e}. Usando Plato Fallback.")
        # [P1-SWAP-STRICT-PANTRY · 2026-05-22] En modo strict (budget /
        # pantry_first) sin clean_ingredients, NO podemos construir un
        # fallback honesto: los hardcoded ["Pollo", "Arroz", "Aguacate"]
        # pueden NO estar en nevera y violarían la promesa que hizo el
        # modal al usuario ("Opciones económicas — Ingredientes de bajo
        # costo / Maximiza tu inventario"). Mejor levantar y dejar que
        # el router lo mapee a 422 con copy explícito al cliente.
        if strict_pantry and not clean_ingredients:
            logger.warning(
                f"⛔ [P1-SWAP-STRICT-PANTRY] swap_reason={swap_reason!r} sin "
                f"pantry detectada → 422 (no fallback honesto posible)."
            )
            raise ValueError(
                "SWAP_STRICT_PANTRY_NO_INVENTORY: el usuario eligió una razón "
                "que exige usar solo ingredientes de la nevera, pero no hay "
                "inventario detectado para construir el plato. Pide al usuario "
                "actualizar su nevera o cambiar a otra razón."
            )
        # [P3-SWAP-LLM-RETRIES-422 · 2026-05-23] Cuando el LLM agota retries
        # y NO es strict-pantry-vacío, el comportamiento legacy era armar
        # un "Plato Fallback" con clean_ingredients[:4] que el frontend
        # mostraba como un plato real al usuario. Resultado: receta
        # genérica de 3 pasos placeholder + título sin coherencia
        # ("Merienda con Cilantro y Aceite de oliva"), pegado al plan del
        # user como si fuera una alternativa válida. Verificado log
        # productivo 2026-05-23 00:21-00:22: 3 retries fallidos con
        # "/pedazos de queso" → fallback engañoso entregado como éxito.
        #
        # Default nuevo: raise ValueError → router 422 → frontend muestra
        # toast "El chef IA no pudo generar una alternativa" + PRESERVA
        # el plato original (mismo patrón que SWAP_STRICT_PANTRY_NO_INVENTORY).
        # Knob `MEALFIT_SWAP_EMIT_FALLBACK_DISH=true` revierte al legacy.
        _emit_fallback_dish = os.environ.get(
            "MEALFIT_SWAP_EMIT_FALLBACK_DISH", "false"
        ).lower() == "true"
        if not _emit_fallback_dish:
            logger.warning(
                f"⛔ [P3-SWAP-LLM-RETRIES-422] swap_reason={swap_reason!r} "
                f"meal_type={meal_type!r} agotó retries del LLM → 422 "
                f"(plato original preservado en el cliente)."
            )
            raise ValueError(
                "SWAP_LLM_RETRIES_EXHAUSTED: el chef IA no pudo generar una "
                "alternativa coherente tras varios intentos. Pide al usuario "
                "reintentar o elegir otra razón de cambio."
            )
        # Knob ON → mantenemos el fallback legacy (degradación graceful).
        # En strict CON pantry, la lista solo se construye desde clean_ingredients
        # (jamás cae al hardcoded). Sin pantry y NO-strict, el hardcoded
        # se acepta como degradación legacy.
        fallback_ing = clean_ingredients[:4] if clean_ingredients else ["Pollo", "Arroz", "Aguacate"]
        # [P1-SWAP-MACROS · 2026-05-22] Fallback ahora respeta los targets
        # de macros derivados arriba (si target_protein/carbs/fats son 0
        # los valores son los pesos por defecto del MACRO_SPLIT
        # "maintenance" — 25/45/30 proporcional).
        # [P3-SWAP-FALLBACK-TITLE-COPY · 2026-05-22 · revisado P3-SWAP-FALLBACK-TITLE-STRIP · 2026-05-23]
        # Title friendly sin `.title()` (mangla unidades: g→G) y sin prefijo
        # "Opción Segura" (jargon técnico). El revisión 2026-05-23 añade
        # `_extract_clean_name_from_display_string()` al pipeline porque
        # `clean_ingredients` puede contener display strings tipo
        # "1 Cabeza (~500g) Brócoli" cuando proviene de
        # `get_realtime_pantry()` (shopping_calculator.aggregate). El
        # extractor es idempotente para inputs ya limpios.
        _ing_title_tokens = []
        for _raw in fallback_ing[:2]:
            _clean = _extract_clean_name_from_display_string(str(_raw).strip())
            if _clean:
                _ing_title_tokens.append(_clean)
        _title_ings = " y ".join(_ing_title_tokens) if _ing_title_tokens else "ingredientes de tu nevera"
        response = {
            "name": f"{meal_type} con {_title_ings}",
            "desc": "Plato simple armado con ingredientes que tienes en casa. Ajusta la cocción a tu gusto.",
            "ingredients": fallback_ing,
            "recipe": [
                "Mise en place: Prepara de manera básica los ingredientes de la nevera.",
                "El Toque de Fuego: Cocina saludablemente a la plancha o al vapor.",
                "Montaje: Sirve porciones adecuadas según tu objetivo y disfruta."
            ],
            "cals": target_calories or 450,
            "protein": int(round(float(target_protein))) or round((target_calories or 450) * 0.3 / 4),
            "carbs": int(round(float(target_carbs))) or round((target_calories or 450) * 0.4 / 4),
            "fats": int(round(float(target_fats))) or round((target_calories or 450) * 0.3 / 9)
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
    # [P5-RESTOCK-PRESERVE · 2026-06-23] Señaliza si el plato se generó RESTRINGIDO a la
    # despensa (clean_ingredients no vacío → el LLM solo pudo usar lo de la Nevera y el pantry
    # guard lo validó). El frontend NO debe limpiar is_restocked para platos pantry-strict:
    # cocinan desde la Nevera, no introducen nada que el usuario deba comprar. Solo el
    # FREE_GENERATION (despensa vacía) deja pantry_constrained=False → ahí sí puede haber
    # ingredientes nuevos a comprar y limpiar is_restocked es correcto.
    _pantry_constrained = bool(clean_ingredients)
    if hasattr(response, "model_dump"):
        _out = getattr(response, "model_dump")()
    elif isinstance(response, dict):
        _out = response
    elif hasattr(response, "dict"):
        _out = getattr(response, "dict")()
    else:
        raise ValueError("El modelo de IA generó una respuesta inválida. Por favor, reintenta.")
    if isinstance(_out, dict):
        _out["pantry_constrained"] = _pantry_constrained

    # [P1-UPDATE-MACRO-REBALANCE · 2026-06-23] (audit inteligencia P1-2) Rebalanceador determinista de
    # macros hacia el target del slot — la MISMA maquinaria que en S1 lleva la proteína entregada de
    # ~85% del LLM crudo a ~98-103% (benchmark). swap/regenerate-day NO lo corrían (solo el gate ±15%,
    # que ACEPTA el drift sin re-escalar → la proteína se erosiona hacia el borde al cambiar varios
    # platos). regenerate-day lo hereda vía el loop de swap_meal (el ledger se decrementa con el meal YA
    # rebalanceado, sin desync). RIESGO PANTRY: escalar porciones puede exceder la Nevera → re-validamos
    # pantry y REVERTIMOS si rompe. [P1-OBJECTIVE-LEVERS-ON · 2026-06-29] Default flipped OFF→ON: es un
    # RE-ESCALADOR (no añade ingredientes; reverte si rompe pantry) = never-worse-than-current por construcción,
    # espejo del MEALFIT_REGEN_DAY_MACRO_REBALANCE que ya era ON → cierra la asimetría de banda en updates.
    # Rollback sin redeploy: MEALFIT_UPDATE_MACRO_REBALANCE=false. Fail-safe: error → deja el meal del LLM intacto.
    if (
        isinstance(_out, dict)
        and os.environ.get("MEALFIT_UPDATE_MACRO_REBALANCE", "true").strip().lower() in ("1", "true", "yes", "on")
        and (target_protein or target_carbs or target_fats)
    ):
        try:
            from graph_orchestrator import _rebalance_day_macros_to_target
            from nutrition_db import IngredientNutritionDB
            import copy as _copy
            _rb_db = IngredientNutritionDB()
            _snapshot = _copy.deepcopy(_out)
            _changed = _rebalance_day_macros_to_target(
                [_out], float(target_carbs or 0), float(target_fats or 0),
                _rb_db, target_protein=float(target_protein or 0),
            )
            if _changed and clean_ingredients:
                # Pantry-strict: el rebalance pudo escalar una porción por encima de la Nevera → re-validar.
                _reval = validate_ingredients_against_pantry(
                    _out.get("ingredients") or [], clean_ingredients, allow_external_count=_external_tolerance
                )
                if _reval is not True:
                    logger.info(f"🎚 [P1-UPDATE-MACRO-REBALANCE] rebalance rompió pantry → revertido | {_reval}")
                    _out.clear()
                    _out.update(_snapshot)
                    # [P1-PANTRY-DEGRADED-SIGNAL · 2026-07-01] (audit v3 macros GAP-2) señal ESTRUCTURADA
                    # (no solo log): el meal viaja con la marca → el persist atribuye el gap de banda a la
                    # Nevera (_quality_degraded_pantry_limited) y el frontend puede accionar "agrega ítems".
                    _out["_pantry_limited"] = True
                else:
                    logger.info(f"🎚 [P1-UPDATE-MACRO-REBALANCE] macros re-apuntadas al slot | meal_type={meal_type}")
        except Exception as _rb_e:
            logger.warning(f"[P1-UPDATE-MACRO-REBALANCE] rebalance falló (no bloquea): {type(_rb_e).__name__}: {_rb_e}")

    # [P2-SWAP-PROTEIN-CLOSER · 2026-06-24] (re-audit P2-2/P2-3) El gate de macros ACEPTA hasta -15% de
    # proteína sin re-escalar → swaps repetidos erosionan la proteína al borde inferior de la banda. Si el
    # plato pasó el gate pero quedó bajo el target del slot, rellena la proteína al ~target con proteína de
    # alta densidad allergen-safe (reusa el closer determinista de S1, espejo del piso de proteína). RIESGO
    # PANTRY: el closer AÑADE un ingrediente → re-validamos la Nevera y REVERTIMOS si rompe (never-worse-
    # than-current). Renal EXENTO (el trim renal manda — no subir proteína). [P1-OBJECTIVE-LEVERS-ON · 2026-06-29]
    # Default flipped OFF→ON: es el MISMO closer determinista de S1 (validado en benchmark: proteína entregada
    # 85%→98-103%); mueve la proteína HACIA el target del slot (no la aleja), con pantry-revert + skip renal →
    # cierra la erosión de proteína en swaps repetidos. Rollback sin redeploy: MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER=false.
    # regenerate-day (S2, P2-3) lo hereda vía el loop de swap_meal. Fail-safe: error → deja el meal del LLM.
    if (
        isinstance(_out, dict)
        and os.environ.get("MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER", "true").strip().lower() in ("1", "true", "yes", "on")
        and target_protein and float(target_protein or 0) > 0
        and not _renal_capped
    ):
        try:
            from graph_orchestrator import _close_protein_gap_for_meal, _safe_high_density_proteins
            from nutrition_db import IngredientNutritionDB
            import copy as _copy_cl
            if float(_out.get("protein") or 0) < float(target_protein):
                _cl_db = IngredientNutritionDB()
                _snap_cl = _copy_cl.deepcopy(_out)
                _cands = _safe_high_density_proteins(allergies, _cl_db)
                _added = _close_protein_gap_for_meal(_out, float(target_protein), _cl_db, _cands)
                if _added and clean_ingredients:
                    _reval_cl = validate_ingredients_against_pantry(
                        _out.get("ingredients") or [], clean_ingredients, allow_external_count=_external_tolerance
                    )
                    if _reval_cl is not True:
                        logger.info(f"🎚 [P2-SWAP-PROTEIN-CLOSER] closer rompió pantry → revertido | {_reval_cl}")
                        _out.clear()
                        _out.update(_snap_cl)
                        # [P1-PANTRY-DEGRADED-SIGNAL · 2026-07-01] espejo del revert del rebalance.
                        _out["_pantry_limited"] = True
                    else:
                        logger.info(f"🎚 [P2-SWAP-PROTEIN-CLOSER] proteína cerrada al target | meal_type={meal_type}")
                elif _added:
                    logger.info(f"🎚 [P2-SWAP-PROTEIN-CLOSER] proteína cerrada (sin pantry-strict) | meal_type={meal_type}")
        except Exception as _cl_e:
            logger.warning(f"[P2-SWAP-PROTEIN-CLOSER] closer falló (no bloquea): {type(_cl_e).__name__}: {_cl_e}")

    # [P2-PANTRY-VARIETY-ADVISORY · 2026-07-02] (audit v3 creatividad GAP-5) el guard same-day de variedad
    # se SALTA entero en pantry-strict (repetir puede ser inevitable cocinando de la nevera — correcto para
    # el RETRY), pero eso también silenciaba la SEÑAL. Advisory-only en pantry-strict: si la proteína del
    # plato entregado repite otra comida de hoy → flag `_same_day_protein_advisory` (telemetría/frontend,
    # jamás retry ni bloqueo). tooltip-anchor: P2-PANTRY-VARIETY-ADVISORY
    if isinstance(_out, dict) and strict_pantry and same_day_other_meals:
        try:
            import re as _re_pv
            from constants import strip_accents as _sa_pv
            _PV_PROT = ("pollo", "pechuga", "cerdo", "chuleta", "res", "bistec", "pavo", "pescado",
                        "tilapia", "salmon", "bacalao", "camaron", "atun", "huevo", "revoltillo")
            _pv_name = _sa_pv(str(_out.get("name", "")).lower())
            _pv_hit = next((t for t in _PV_PROT if _re_pv.search(r"\b" + t, _pv_name)), None)
            if _pv_hit:
                _pv_blob = _sa_pv(" ".join(str(x) for x in same_day_other_meals).lower())
                if _re_pv.search(r"\b" + _pv_hit, _pv_blob):
                    _out["_same_day_protein_advisory"] = True
                    logger.info(f"🔄 [P2-PANTRY-VARIETY-ADVISORY] proteína '{_pv_hit}' repetida hoy "
                                f"(pantry-strict → advisory, sin retry)")
        except Exception:
            pass

    # [P0-UPDATE-CLINICAL-GUARD · 2026-06-23] Guard FINAL defensa-en-profundidad: el path de
    # "Plato Fallback" (knob MEALFIT_SWAP_EMIT_FALLBACK_DISH=true) arma el plato desde
    # clean_ingredients[:4] sin pasar por el check del retry loop → podría contener un alérgeno
    # de la nevera. Escaneamos lo que se DEVUELVE; si viola, fail-secure raise → el router lo
    # mapea a soft-fail y el plato original (clínicamente validado en S1) se preserva.
    if UPDATE_CLINICAL_GUARD and isinstance(_out, dict):
        _final_viol = clinical_backstop_for_meal(_out, allergies=allergies, diet_type=diet_type, form_data=form_data)
        if _final_viol:
            logger.warning(
                f"🛡 [P0-UPDATE-CLINICAL-GUARD] plato final (fallback) viola seguridad clínica → "
                f"fail-secure | meal_type={meal_type} | viol={_final_viol}"
            )
            raise ValueError("CLINICAL_VIOLATION: " + "; ".join(_final_viol))

    # [P1-RENAL-UPDATE-ENFORCE · 2026-06-24] (re-audit P1-1) Si el plan lleva cap renal KDIGO, trima la
    # proteína del plato nuevo al techo del slot (`target_protein`, ya renal-aware porque el plan se capeó
    # en S1). El gate de macros ACEPTA hasta +15% de overshoot → en un paciente renal ese exceso compone
    # el techo iatrogénico. `renal_protein_trim_for_update` solo trima hacia abajo (best-effort, no bloquea).
    if isinstance(_out, dict) and _renal_capped and target_protein:
        try:
            renal_protein_trim_for_update([_out], float(target_protein or 0), renal_capped=True)
        except Exception as _renal_e:
            logger.warning(f"[P1-RENAL-UPDATE-ENFORCE] trim renal en swap falló (no bloquea): {type(_renal_e).__name__}: {_renal_e}")

    # [P2-FOOD-SAFETY-UPDATE · 2026-06-24] (re-audit P2-1) Re-aplica la mitigación determinista de seguridad
    # alimentaria (huevo crudo / pescado-marisco-carne crudos) — S1 la corre en el grafo pero el swap no.
    # Macro-preservante (solo añade nota a la receta), fail-open, idempotente, gateado por FOOD_SAFETY_GUARD.
    if isinstance(_out, dict):
        try:
            food_safety_backstop_for_meal(_out)
        except Exception as _fs_e:
            logger.warning(f"[P2-FOOD-SAFETY-UPDATE] food-safety en swap falló (no bloquea): {type(_fs_e).__name__}: {_fs_e}")

    # [P2-UPDATE-CONDITION-SUBST · 2026-06-26] (audit 3-flujos P2) Sustitución determinista por condición
    # médica (DM2 azúcar / HTA sodio / dislipidemia grasa sat.) — paridad con el Guard 3 de S1, que los
    # updates esquivaban (solo directiva-prompt advisory). Macro-preservante, idempotente, fail-open.
    # `form_data` trae medicalConditions enriquecidas server-side por _enrich_clinical_from_profile (aplica
    # a swap S3 y, por herencia del loop de swaps, a regenerate-day S2).
    if isinstance(_out, dict):
        try:
            condition_substitution_backstop_for_meal(_out, form_data)
        except Exception as _cs_e:
            logger.warning(f"[P2-UPDATE-CONDITION-SUBST] condition-subst en swap falló (no bloquea): {type(_cs_e).__name__}: {_cs_e}")

    # [P1-SWAP-PORTION-CAP · 2026-06-27] (paridad S1↔S3) Caps de porción DETERMINISTAS — DM2 (almidón alto-IG:
    # batata/yuca/papa/plátano maduro/casabe ≤cap_g) + bariátrica (queso ≤30g / yogurt ≤120g / fruta / aguacate /
    # frutos secos + volumen del pouch). S1 y regenerate-day (S2) ya los corren; el swap individual solo tenía
    # slot-target + prompt → el LLM no siempre obedece la directiva de porción (5 lonjas de queso en una cena
    # bariátrica colaban sin backstop). Solo RECORTAN (recuperan kcal escalando otros ingredientes → macro-safe);
    # como el recorte de lácteo baja proteína, RE-CERRAMOS el piso del slot con proteína animal densa NO-láctea
    # (espejo de FASE A; renal → skip KDIGO). Idempotente, fail-open. tooltip-anchor: P1-SWAP-PORTION-CAP
    if isinstance(_out, dict):
        try:
            from graph_orchestrator import (cap_dm2_high_gi_portions as _cap_dm2_s,
                                            cap_bariatric_portions as _cap_baria_s,
                                            _close_protein_gap_for_meal as _close_pc,
                                            _safe_high_density_proteins as _safe_pc)
            from nutrition_db import IngredientNutritionDB as _CapDB
            _cap_db = _CapDB()
            _wrap = [{"meals": [_out]}]
            _nd = _cap_dm2_s(_wrap, form_data, _cap_db)
            _nb = _cap_baria_s(_wrap, form_data, _cap_db)
            if (_nd or _nb) and target_protein and not _renal_capped:
                _cur_p = float(_out.get("protein") or 0)
                if _cur_p < 0.90 * float(target_protein):
                    _out["_protein_closed"] = False
                    _cands_pc = [c for c in _safe_pc(allergies, _cap_db, min_protein=18.0)
                                 if not any(_t in str(c[1]).lower()
                                            for _t in ("queso", "yogur", "leche", "ricotta", "cottage", "requeson"))]
                    if _cands_pc:
                        _close_pc(_out, float(target_protein), _cap_db, _cands_pc, max_add_g=90)
            if _nd or _nb:
                logger.info(f"🔒 [P1-SWAP-PORTION-CAP] plato de swap recortado: cap_dm2={_nd} "
                            f"cap_baria={_nb} | meal_type={meal_type}")
        except Exception as _pc_e:
            logger.warning(f"[P1-SWAP-PORTION-CAP] cap de porción en swap falló (no bloquea): {type(_pc_e).__name__}: {_pc_e}")

    # [P1-UPDATE-APPETIBILITY · 2026-06-27] (audit Fase 0) Honestidad de nombre (proteína fantasma) +
    # detección de clash sobre el plato FINAL (cubre también el path de fallback que esquiva el retry-loop).
    # namefix es determinista e idempotente; el clash en el plato final solo se loguea (advisory).
    if isinstance(_out, dict):
        try:
            _appet = appetibility_fix_for_update(_out)
            if _appet.get("name_fixed"):
                logger.info(f"🎭 [P1-UPDATE-APPETIBILITY] nombre de swap corregido (proteína fantasma) | meal_type={meal_type}")
            if _appet.get("sweet_savory_clash"):
                logger.warning(f"🍓 [P1-UPDATE-APPETIBILITY] plato final de swap mantiene pareo fruta+salado (advisory) | meal_type={meal_type}")
        except Exception as _ap_e:
            logger.warning(f"[P1-UPDATE-APPETIBILITY] appetibility fix en swap falló (no bloquea): {type(_ap_e).__name__}: {_ap_e}")

    # [P1-UPDATE-RECIPE-FINALIZE · 2026-06-29] (audit objetivo · paridad updates ↔ form-gen) Finalizadores de
    # coherencia de RECETA que assemble_plan_node corre en form-gen pero NINGÚN update corría: veg-fantasma en los
    # PASOS → ingredients[] (para que se compre + cuente macros), 'lonja de queso' → gramos, cap de hojas infladas.
    # Espejo per-meal del bundle de S1; idempotente, fail-open. regenerate-day lo hereda (es loop de swap_meal).
    # tooltip-anchor: P1-UPDATE-RECIPE-FINALIZE
    if isinstance(_out, dict):
        try:
            from graph_orchestrator import finalize_single_meal_recipe_coherence as _fin_rc
            # [P2-STEPVEG-PANTRY-GUARD · 2026-06-29] pantry-strict = el swap está armado desde la Nevera
            # (clean_ingredients no vacío) → el finalizer NO añade veg de catálogo (no se puede comprar más).
            # [P0-VEG-GUARD-ALLERGEN · 2026-07-01] allergies (enriquecidas server-side) → el veg-guard del
            # finalizer NO inyecta un alérgeno post-backstop (este bloque corre DESPUÉS del scan clínico).
            _nfix = _fin_rc(_out, pantry_strict=bool(clean_ingredients), allergies=allergies)
            if _nfix:
                logger.info(f"🍳 [P1-UPDATE-RECIPE-FINALIZE] {_nfix} fix(es) de coherencia de receta en plato de swap | meal_type={meal_type}")
        except Exception as _fin_e:
            logger.warning(f"[P1-UPDATE-RECIPE-FINALIZE] finalizador de receta en swap falló (no bloquea): {type(_fin_e).__name__}: {_fin_e}")
    # [P2-MACRO-UPD-3 · 2026-06-29] (re-audit objetivo · P2) Telemetría de banda per-comida (paridad del canal
    # degraded/alert con S1): loguea si el plato swapeado quedó materialmente fuera de la banda del target de
    # proteína del slot (drift >15%). No bloquea (el validador ±15% per-comida es el guard user-facing).
    if isinstance(_out, dict):
        try:
            _tp_b = float(target_protein or 0)
            if _tp_b > 0:
                _dp_b = abs(float(_out.get("protein") or 0) - _tp_b) / _tp_b
                if _dp_b > 0.15:
                    _out["_macro_band_low"] = True
                    logger.info(f"📊 [P2-MACRO-UPD-3] plato de swap fuera de banda de proteína "
                                f"(drift {_dp_b:.0%} vs target del slot) — telemetría | meal_type={meal_type}")
        except Exception:
            pass
    return _out







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
                _json_rl.dumps({"model": model_name, "provider": "deepseek"}, ensure_ascii=False),
            ),
        )
    except Exception as _e_rl:
        try:
            logger.debug(f"[P1-CHAT-LLM-429] emit metric falló (best-effort): {_e_rl!r}")
        except Exception:
            pass


# [P3-CHAT-OBSERVABILITY · 2026-05-20] Cooldown in-process del alert
# `chat_checkpoint_pool_split_missing`. Sin cooldown, cada request del
# chat (potencialmente miles/min bajo carga) emitiría un UPSERT al mismo
# row de `system_alerts` — contención inútil. Cooldown 1h = la alert
# vive como "abierta" (resolved_at IS NULL) mientras la condición exista;
# SRE la cierra manualmente tras reparar el pool. El lock garantiza
# atomicidad del check-and-set bajo workers concurrentes del mismo proceso.
# Tooltip-anchor: P3-CHAT-OBSERVABILITY.
import threading as _threading_obs
_POOL_SPLIT_ALERT_COOLDOWN_S = 3600.0
_pool_split_alert_last_ts = 0.0
_pool_split_alert_lock = _threading_obs.Lock()

# [P3-CHAT-OBSERVABILITY · 2026-05-20] TTL del lock cross-worker para
# `generate_chat_title_background`. Si un worker crashea sin cleanup,
# la fila en `app_kv_store` queda huérfana — el TTL permite que el
# siguiente claim la sobreescriba como stale. 5 min cubre el 99p del
# title generation (típicamente <10s) con margen amplio para casos
# patológicos (Gemini lento, retries, multi-stage).
_TITLE_LOCK_TTL_S = 300


def _try_claim_title_lock_cross_worker(session_id: str) -> bool:
    """[P3-CHAT-OBSERVABILITY · 2026-05-20] Atomic claim del lock
    cross-worker para `generate_chat_title_background`. Reemplaza el
    `_generating_titles = set()` in-memory que sufría race bajo
    gunicorn `-w N`: cada worker tenía su propio set → dedupe fallaba
    con probabilidad ~(N-1)/N → tokens LLM duplicados + N rows
    SYSTEM_TITLE concurrent que el último UPSERT pisaba.

    Returns:
        True  → este worker claimó el lock, debe proceder con la generación.
        False → otra worker ya está procesando (lock activo, NO stale).

    Estrategia: UPSERT con `WHERE existing.started_at < now - TTL`.
    RETURNING devuelve la fila solo si el INSERT/UPDATE ocurrió:
      - INSERT puro (fila nueva) → RETURNING emite ✓ claimed
      - UPDATE porque WHERE matched (stale) → RETURNING emite ✓ claimed
      - UPDATE skipped por WHERE False (lock activo) → RETURNING vacío ✗
    Postgres serializa ON CONFLICT DO UPDATE por-fila → race-free.

    Best-effort: si la DB no responde, retornamos True (fail-open) para
    NO bloquear title generation en outage del KV. Trade-off aceptable:
    title es cosmético, prefiero duplicarlo a perderlo.
    """
    try:
        from db_core import execute_sql_query
        import time as _t_claim
        _now_ts = _t_claim.time()
        _kv_key = f"title_gen_inflight:{session_id}"
        result = execute_sql_query(
            """
            INSERT INTO app_kv_store (key, value)
            VALUES (%s, jsonb_build_object('started_at', %s::float))
            ON CONFLICT (key) DO UPDATE SET
                value = jsonb_build_object('started_at', %s::float),
                updated_at = NOW()
            WHERE COALESCE((app_kv_store.value->>'started_at')::float, 0)
                  < %s::float
            RETURNING key
            """,
            (_kv_key, _now_ts, _now_ts, _now_ts - _TITLE_LOCK_TTL_S),
        )
        return bool(result)
    except Exception as _e_claim:
        logger.debug(
            f"[P3-CHAT-OBSERVABILITY] title lock claim falló "
            f"(fail-open) session={session_id}: {_e_claim!r}"
        )
        return True


def _emit_chat_rag_embedding_failed_metric_best_effort(user_id, session_id, source):
    """[P3-CHAT-OBSERVABILITY · 2026-05-20] Persiste a `pipeline_metrics`
    cuando el RAG embedding falla (catch broad en los 2 callsites de
    `chat_with_agent` / `chat_with_agent_stream`). Pre-fix: el chat
    seguía gracefully sin RAG pero SRE NO podía graficar "% de chats
    sin RAG" → regresión silenciosa del embedding service (Gemini
    embeddings API caída, parse error del input, OOM en pgvector)
    quedaba invisible hasta queja del user.

    `source` ∈ {'chat_with_agent', 'chat_with_agent_stream'} para
    diferenciar non-stream vs streaming en queries. Best-effort: cualquier
    fallo de DB se silencia y NO afecta el chat-flow."""
    try:
        from db_core import execute_sql_write
        import json as _json_rag
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
                "chat_rag_embedding_failed",
                _json_rag.dumps({"source": source}, ensure_ascii=False),
            ),
        )
    except Exception as _e_rag:
        try:
            logger.debug(f"[P3-CHAT-OBSERVABILITY] emit chat_rag_embedding_failed metric falló: {_e_rag!r}")
        except Exception:
            pass


def _emit_checkpoint_pool_split_missing_alert_best_effort():
    """[P3-CHAT-OBSERVABILITY · 2026-05-20] Emit `system_alerts` con
    `alert_key='chat_checkpoint_pool_split_missing'` cuando
    `chat_checkpoint_pool` no se creó al arranque y caemos al fallback
    `connection_pool` (transaction pooler). Esto reabre el modo de fallo
    que P1-CHECKPOINT-POOL-SPLIT · 2026-05-20 cerró (SSL bad length /
    EOF cuando Supavisor mata conexiones idle del Transaction Pooler
    durante el chat stream).

    Cooldown 1h in-process: bajo carga alta (1000 req/s), sin cooldown
    haríamos 1000 UPSERTs/s al mismo row de `system_alerts`. El UPSERT
    canonical (P2-NEW-3) mantiene la alert como "abierta" (resolved_at
    IS NULL) mientras la condición exista; SRE la cierra manualmente
    tras reparar el pool.

    Best-effort: cualquier fallo de DB se silencia."""
    global _pool_split_alert_last_ts
    import time as _t_alert
    _now_ts = _t_alert.time()
    with _pool_split_alert_lock:
        if _now_ts - _pool_split_alert_last_ts < _POOL_SPLIT_ALERT_COOLDOWN_S:
            return
        _pool_split_alert_last_ts = _now_ts
    try:
        from db_core import execute_sql_write
        import json as _json_alert
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'chat_checkpoint_pool_split_missing', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                message = EXCLUDED.message,
                resolved_at = NULL
            """,
            (
                "chat_checkpoint_pool_split_missing",
                "Chat usa fallback `connection_pool` (Transaction Pooler)",
                "El pool `chat_checkpoint_pool` (session-mode 5432) NO se creó al arranque — "
                "el chat compila PostgresSaver contra el Transaction Pooler. Esto reabre "
                "el modo de fallo SSL bad length / EOF (P1-CHECKPOINT-POOL-SPLIT · 2026-05-20). "
                "Revisar logs de `db.py` por errores en la creación del split pool.",
                _json_alert.dumps({"source": "agent.chat_with_agent_stream"}, ensure_ascii=False),
                _json_alert.dumps([]),
            ),
        )
    except Exception as _e_alert:
        try:
            logger.debug(f"[P3-CHAT-OBSERVABILITY] emit pool_split_missing alert falló: {_e_alert!r}")
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
    # [P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22] Timestamp epoch ms cuando
    # `execute_tools` ejecutó una tool que muta `user_inventory`
    # (`modify_pantry_inventory` o `log_consumed_meal` con `ingredients`).
    # Se propaga al SSE `done`; el frontend (Agent.jsx) escribe la key
    # localStorage `mealfit_pantry_dirty_at` para que Pantry.jsx invalide
    # su cache TTL=30s y re-fetcheé al próximo mount. Defensa en profundidad
    # sobre el canal Realtime (puede tener lag o estar cerrado si user
    # navega entre tabs/components durante la conversación).
    # Default: None — sin mutación de pantry, frontend silencia el flag.
    pantry_modified_at: float | None
    # [P3-AGENT-DEPLETE · 2026-05-22] Lista de items que el chat agent marcó
    # como AGOTADOS via `modify_pantry_inventory(items_to_deplete=[...])`.
    # Shape per item: {master_ingredient_id, ingredient_name, quantity,
    # unit, category, shelf_life_days, depleted_at}. Se propaga al SSE
    # `done`; AgentPage.jsx hace merge a `localStorage.mealfit_depleted_items`
    # para que Pantry.jsx muestre los items en la sección "Agotados" (que
    # también alimenta la lista de compras para re-stock).
    # Default: None — sin items agotados en el turn.
    pantry_depleted_items: list | None

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
            
    # [P0-DEEPSEEK-MIGRATION] Identidad para tier-routing (paid→pro). Guests
    # (session_id) resuelven a flash via fail-cheap del router.
    _model_uid = state.get("user_id") or state.get("session_id")
    chat_llm = ChatDeepSeek(
        model=_chat_agent_model_name(_model_uid),
        temperature=0.7,
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
    # NOTA: `_chat_agent_model_name(_model_uid)` se llama 2x (callsite del
    # constructor arriba + aquí) con el MISMO uid — el modelo del gate CB debe
    # coincidir con el del LLM construido (tier-routing P0-DEEPSEEK-MIGRATION).
    # Costo trivial: tier lookup cacheado con TTL en llm_provider.
    _cb_model = _chat_agent_model_name(_model_uid)
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
    # `llm_usage_events`, mismo cost calculation `compute_llm_cost_micros`).
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
                    _json_empty.dumps({"model": _cb_model, "provider": "deepseek"}, ensure_ascii=False),
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
    # [P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22] Preservar timestamp si
    # un turn previo ya lo seteó (caso edge: LLM emite múltiples tool_calls
    # incluyendo varias de inventory — quedamos con el más reciente).
    pantry_modified_at = state.get("pantry_modified_at")
    # [P3-AGENT-DEPLETE · 2026-05-22] Acumular items agotados de este turn
    # (LLM puede emitir múltiples tool_calls; concatenamos).
    pantry_depleted_items = list(state.get("pantry_depleted_items") or [])

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
                            # [P3-GENCHUNK-SPEED · 2026-06-01] `execute_modify_single_meal`
                            # ahora retorna el `plan_data` ya mergeado (fresh-post-lock,
                            # la misma data que la re-lectura traería). Usarlo directo
                            # evita un SELECT serial redundante justo tras la escritura.
                            # Fallback a `get_latest_meal_plan_with_id` solo si la key
                            # está ausente (back-compat / parser degradado).
                            _inband_plan = parsed_mod.get("plan_data")
                            if isinstance(_inband_plan, dict) and _inband_plan:
                                new_plan = _inband_plan
                            else:
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
                            # [P2-CHATMODIFY-BAND-WARN · 2026-07-01] (audit v2 paridad GAP-2, batch
                            # P2-AUDIT-V2-BATCH) Los flags de honestidad `_macro_band_low` (drift >15% vs el
                            # plato original, e.g. closer sin palanca en pantry-strict) y `_slot_advisory`
                            # (horario) SE PERSISTÍAN en el meal pero este branch los pisaba con "modificada
                            # exitosamente" — el mismo patrón silencioso que P2-SWAP-BAND-WARNING cerró para
                            # swap. Ahora: (a) toast no-bloqueante vía coherence_warnings (mismo canal SSE
                            # `done` que ya consume el frontend), (b) el coach recibe la instrucción de
                            # avisarlo en su respuesta. tooltip-anchor: P2-CHATMODIFY-BAND-WARN
                            _mod_meal_flags = parsed_mod.get("modified_meal") or {}
                            _band_warn_bits = []
                            if _mod_meal_flags.get("_macro_band_low"):
                                _band_warn_bits.append(
                                    "El plato nuevo quedó algo alejado de tu objetivo de macros "
                                    "(los ingredientes disponibles no alcanzaron el balance exacto)."
                                )
                            if _mod_meal_flags.get("_slot_advisory"):
                                _band_warn_bits.append(
                                    "El plato queda algo inusual para ese horario de comida."
                                )
                            if _band_warn_bits:
                                coherence_warnings.extend(_band_warn_bits)
                            tool_result = (
                                f"La comida fue modificada exitosamente. La nueva comida es: "
                                f"{parsed_mod['modified_meal'].get('name', 'Comida actualizada')}. "
                                f"Dile al usuario que su plan ya fue actualizado."
                                + ((" IMPORTANTE — avísale también, en tono honesto y breve: "
                                    + " ".join(_band_warn_bits)
                                    + " Puede volver a pedir el cambio con otras palabras si quiere afinarlo.")
                                   if _band_warn_bits else "")
                            )
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
                            # [P3-AGENT-DEPLETE · 2026-05-22] Si la tool inyectó
                            # marker `<<PANTRY_DEPLETED_JSON: [...]>>` en el
                            # tool_result, extraerlo + acumular al state +
                            # strip-earlo del str para que la LLM NO vea el
                            # JSON raw (sería ruido en su contexto).
                            if (
                                isinstance(tool_result, str)
                                and "<<PANTRY_DEPLETED_JSON:" in tool_result
                            ):
                                import json as _json_marker
                                import re as _re_marker
                                _marker_re = _re_marker.compile(
                                    r"<<PANTRY_DEPLETED_JSON:\s*(\[[^\]]*\]|\[.*?\])>>",
                                    _re_marker.DOTALL,
                                )
                                _m = _marker_re.search(tool_result)
                                if _m:
                                    try:
                                        _parsed = _json_marker.loads(_m.group(1))
                                        if isinstance(_parsed, list):
                                            pantry_depleted_items.extend(_parsed)
                                            logger.info(
                                                f"🪫 [P3-AGENT-DEPLETE] tool={tool_name} "
                                                f"marcó {len(_parsed)} item(s) como agotados "
                                                f"(user={_trusted_uid})"
                                            )
                                    except Exception as _parse_err:
                                        logger.warning(
                                            f"[P3-AGENT-DEPLETE] parse marker falló: {_parse_err!r}"
                                        )
                                    # Strip del marker del tool_result.
                                    tool_result = _marker_re.sub("", tool_result).strip()
                            # [P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22]
                            # Si la tool muta `user_inventory`, marcar el
                            # state con un timestamp epoch (ms). El SSE
                            # `done` lo propaga al frontend que lo escribe
                            # a `localStorage.mealfit_pantry_dirty_at`;
                            # Pantry.jsx lo lee al mount + storage event y
                            # invalida su cache TTL=30s. Defensa en
                            # profundidad sobre el canal Realtime (puede
                            # tener lag si user navega entre tabs durante
                            # la conversación). `log_consumed_meal` solo
                            # cuenta como mutación si trae `ingredients` —
                            # sin esa lista no toca pantry.
                            _mutates_pantry = (
                                tool_name == "modify_pantry_inventory"
                                or (
                                    tool_name == "log_consumed_meal"
                                    and isinstance(tool_args, dict)
                                    and bool(tool_args.get("ingredients"))
                                )
                            )
                            if _mutates_pantry:
                                import time as _time
                                pantry_modified_at = _time.time() * 1000.0
                                logger.info(
                                    f"🥚 [P3-PANTRY-INVALIDATE-FROM-CHAT] "
                                    f"tool={tool_name} marcó pantry_dirty "
                                    f"at={pantry_modified_at:.0f} (user={_trusted_uid})"
                                )
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
        # [P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22] Timestamp del último
        # modify_pantry_inventory / log_consumed_meal (con ingredients) en
        # este turn. None si no se mutó pantry. El stream wrapper lo emite
        # en el SSE `done` para que Agent.jsx setee la key localStorage.
        "pantry_modified_at": pantry_modified_at,
        # [P3-AGENT-DEPLETE · 2026-05-22] Items que el agente marcó como
        # agotados en este turn (de `modify_pantry_inventory(items_to_deplete)`).
        # SSE `done` lo emite; AgentPage.jsx merge a localStorage.
        "pantry_depleted_items": pantry_depleted_items if pantry_depleted_items else None,
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
    # [P2-CHAT-CLEANUP · 2026-05-20] Migrado `dlog()` (escribía a
    # `title_debug.log` en disco append-mode sin rotación) a `logger.debug`.
    # Pre-fix: cada thread background abría el file en cada log line — disk
    # I/O side-channel + crecimiento ilimitado en prod. Convención del repo
    # (P2-LOGGER-MIGRATION) prohíbe escritura directa a archivo desde código
    # productivo. Tooltip-anchor: P2-CHAT-CLEANUP.
    _t0 = time.monotonic()
    logger.debug(f"[chat_title bg] session={session_id} - Thread started")

    # [P3-CHAT-OBSERVABILITY · 2026-05-20] Dedupe híbrido: fast-path
    # in-memory (evita roundtrip DB cuando el mismo worker ya tiene el
    # lock) + cross-worker via `app_kv_store` (cierra race bajo
    # gunicorn `-w N`). Pre-fix: `_generating_titles = set()` por-proceso
    # → dedupe fallaba con probabilidad ~(N-1)/N en multi-worker → N
    # threads concurrent emitían N invokes Gemini + N rows SYSTEM_TITLE
    # de los que el último UPSERT pisaba (tokens duplicados sin valor).
    # Tooltip-anchor: P3-CHAT-OBSERVABILITY.
    if session_id in _generating_titles:
        logger.debug(f"[chat_title bg] session={session_id} - Already generating (in-process), returning")
        return
    if not _try_claim_title_lock_cross_worker(session_id):
        logger.debug(
            f"[chat_title bg] session={session_id} - claimed by another worker "
            f"(cross-process lock active), returning"
        )
        return
    try:
        _generating_titles.add(session_id)

        # Check if a title already exists for this session
        res_data = get_session_messages(session_id)
        if res_data and any(str(m.get("content", "")).startswith("[SYSTEM_TITLE]") for m in res_data):
            logger.debug(f"[chat_title bg] session={session_id} - Title exists, returning")
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
        
        logger.debug(f"[chat_title bg] session={session_id} - Initializing LLM client")

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
            
        # [P1-CHAT-CB-EXTEND · 2026-05-20] CB gate fire-and-forget. Si
        # breaker abierto, skip silente (NO raise — esto corre en thread
        # background y un raise solo se loguea sin afectar el chat-flow,
        # pero igual desperdicia el thread del executor). El user verá
        # "Nuevo chat" hasta que la próxima invocación legítima genere
        # el título. Trade-off aceptable: NO bloqueamos al user por un
        # title cosmético cuando el provider está degradado. Tooltip-
        # anchor: P1-CHAT-CB-EXTEND.
        _title_cb_model = _chat_title_model_name()
        _title_cb = _get_circuit_breaker(_title_cb_model)
        if not _title_cb.can_proceed():
            logger.info(
                f"[P1-CHAT-CB-EXTEND] title generation CB abierto "
                f"model={_title_cb_model!r} session={session_id} — skip "
                f"silente. Title quedará en 'Nuevo chat' hasta próximo turn."
            )
            return

        title_llm = ChatDeepSeek(model=_chat_title_model_name(), temperature=0.7, timeout=_chat_title_llm_timeout_s(), max_output_tokens=_chat_title_max_output_tokens())  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19] / [P3-COST-TITLE-OUTPUT-CAP · 2026-06-01]
        prompt = TITLE_GENERATION_PROMPT.format(first_message=first_message, used_titles=used_titles_str)
        logger.debug(f"[chat_title bg] session={session_id} - Calling LLM API")
        try:
            response = title_llm.invoke(prompt)
            # [P1-CHAT-CB-EXTEND · 2026-05-20] Marcar success post-invoke.
            _title_cb.record_success()
        except Exception as _title_invoke_exc:
            # [P1-CHAT-CB-EXTEND · 2026-05-20] Discriminar rate-limit del
            # provider (NO cuenta como CB failure) vs failure genuino.
            # Espejo del patrón en `call_model`. En ambos casos re-raise
            # al except outer que ya hace logger.error — preservar el
            # log path existente.
            if _is_rate_limit_error(_title_invoke_exc):
                _emit_chat_rate_limited_metric_best_effort(
                    user_id, session_id, _title_cb_model,
                )
                logger.warning(
                    f"⚠️ [P1-CHAT-LLM-429] title generation rate-limit "
                    f"model={_title_cb_model!r} session={session_id} — "
                    f"NO cuenta como CB failure."
                )
            else:
                _title_cb.record_failure()
            raise
        logger.debug(f"[chat_title bg] session={session_id} - LLM response received")
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
        
        logger.debug(f"[chat_title bg] session={session_id} - Inserting SYSTEM_TITLE msg into DB")
        save_message(session_id, "model", f"[SYSTEM_TITLE] {title}")
        _elapsed_s = time.monotonic() - _t0
        logger.info(f"✅ Título generado para sesión {session_id}: {title} (elapsed={_elapsed_s:.2f}s)")
    except Exception as e:
        logger.error(f"⚠️ Error generando título session={session_id}: {e}")
    finally:
        # [P3-CHAT-OBSERVABILITY · 2026-05-20] Cleanup del in-memory set.
        # Pre-fix: `_generating_titles.add(session_id)` se hacía en `try`
        # pero NUNCA se removía → set crecía indefinidamente con cada
        # generación (memory leak slow-burn). El cross-worker lock en
        # `app_kv_store` se auto-expira via TTL (5 min), pero el in-memory
        # set requería discard explícito. NO eliminamos el row del KV
        # acá — el TTL natural cierra la ventana sin INSERT extra y
        # mantiene defensa contra "mismo session_id re-claimed
        # inmediatamente tras success" (raro pero posible si el frontend
        # re-spawnea bg task).
        _generating_titles.discard(session_id)


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

    # [P1-CHAT-CB-EXTEND · 2026-05-20] CB gate hot-path del chat. El RAG
    # router se invoca síncrono en CADA turn del chat (línea 1351, 1596) —
    # si Gemini está degradado, cada chat paga `MEALFIT_CHAT_ROUTER_LLM_TIMEOUT_S`
    # (default 12s) antes del fallback. Con breaker abierto, retornamos
    # el fallback de inmediato (mismo behaviour que el except actual)
    # para preservar el hot-path. NO raise: el rag_query_router es
    # preprocessing y nunca debe abortar el chat upstream. La degradación
    # es graceful: el chat sigue funcionando sin RAG hasta que el provider
    # se recupere (cron `_sweep_stale_llm_circuit_breakers` cierra ventana).
    # Tooltip-anchor: P1-CHAT-CB-EXTEND.
    _router_cb_model = _chat_router_model_name()
    _router_cb = _get_circuit_breaker(_router_cb_model)
    if not _router_cb.can_proceed():
        logger.warning(
            f"🛑 [P1-CHAT-CB-EXTEND] rag_query_router CB abierto "
            f"model={_router_cb_model!r} — fallback prompt original sin "
            f"reescribir. Chat continúa sin RAG hasta que el provider "
            f"se recupere."
        )
        return {"skip": False, "query": prompt}

    try:
        router_llm = ChatDeepSeek(
            model=_chat_router_model_name(),
            temperature=0.0,
            timeout=_chat_router_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
        )

        rewrite_prompt = RAG_ROUTER_PROMPT.format(prompt=prompt)

        response = router_llm.invoke(rewrite_prompt)
        # [P1-CHAT-CB-EXTEND · 2026-05-20] Marcar success post-invoke.
        _router_cb.record_success()
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
        # [P1-CHAT-CB-EXTEND · 2026-05-20] Discriminar rate-limit del
        # provider antes de marcar failure. Espejo del patrón en
        # `call_model` (P1-CHAT-LLM-429). Para rate-limit, emit métrica
        # pero NO `record_failure` — el provider está vivo, solo throttling.
        # En ambos casos retornamos fallback (mismo behaviour que pre-fix);
        # `rag_query_router` NO debe abortar el chat upstream.
        if _is_rate_limit_error(e):
            _emit_chat_rate_limited_metric_best_effort(
                None, None, _router_cb_model,
            )
            logger.warning(
                f"⚠️ [P1-CHAT-LLM-429] rag_query_router rate-limit "
                f"model={_router_cb_model!r} — NO cuenta como CB failure."
            )
        else:
            _router_cb.record_failure()
        logger.error(f"⚠️ [RAG ROUTER] Error en rewrite, usando prompt original: {e}")
        return {"skip": False, "query": prompt}


# [P3-AGENT-HYDRATION-CONTEXT · 2026-05-27] Helper que retorna un bloque
# de system prompt con la hidratación actual del usuario (vasos consumidos
# hoy + meta diaria). Solo emite si el toggle `water_tracker_enabled` está
# activo (Settings → Personaliza tu panel → Hidratación). Si el toggle
# está apagado, retorna string vacío — el agente no debe saber nada del
# tracker para respetar la preferencia del usuario.
#
# El cómputo de la meta es una réplica simplificada de
# `routers/plans.py::_compute_water_goal` (no importable desde aquí por
# circular import: routers/plans.py ya importa de agent.py). Fórmula:
# 35 ml/kg + bonus por actividad, clamp a [6, 14] vasos de 250ml.
#
# Fail-secure: cualquier excepción → retorna "" (no inyecta nada). El
# agente puede usar la tool `check_hydration_today` si necesita el dato
# bajo demanda en lugar de en cada turno.
def _build_hydration_context(user_id: Optional[str], local_date_str: Optional[str] = None) -> str:
    if not user_id or user_id == "guest":
        return ""
    try:
        from db_profiles import get_water_tracker_enabled, get_water_intake_glasses_today
        if not get_water_tracker_enabled(user_id):
            return ""

        # [P3-HYDRATION-CTX-TZ · 2026-05-31] Preferir la fecha LOCAL del
        # cliente (la pasa el stream path). Si no llega (path non-stream
        # `/api/chat`), caer a la fecha LOCAL DOMINICANA (UTC-4) vía el
        # mismo helper que usan las tools `check_hydration_today` /
        # `log_water_glass` — NO a UTC. Pre-fix caía a UTC: para un usuario
        # de RD entre las 8 PM y medianoche (AST) la fecha UTC ya es
        # "mañana", así que el agente leía el bucket de mañana (0 vasos) y
        # podía regañar a un usuario que sí tomó agua hoy. Misma clase de
        # bug UTC-vs-AST que P1-PROACTIVE-TZ.
        if not local_date_str:
            from tools import _local_date_str_for_user
            local_date_str = _local_date_str_for_user()

        glasses = get_water_intake_glasses_today(user_id, local_date_str)

        # [P2-HYDRATION-GOAL-SSOT · 2026-05-31] Reusar la fórmula CANÓNICA
        # `_compute_water_goal` — la MISMA meta exacta que ve el card del
        # Dashboard y que reportan las tools check_hydration_today /
        # log_water_glass. Pre-fix reimplementaba la meta inline y divergía:
        # 250 ml/vaso (canónico = 240) + mapeo de actividad distinto
        # (active→+250 en vez de +500; very_active→+500 en vez de +750;
        # athlete/very_high→+0 en vez de +750; activityLevel ausente/null→+0
        # en vez del default moderate +250). Resultado observado: el agente
        # afirmaba una meta 1-2 vasos distinta a la del card para usuarios
        # reales (3/8 con activityLevel=null). Import lazy igual que
        # tools.check_hydration_today (cadena de carga routers.plans→agent→
        # tools; se resuelve en runtime, sin ciclo de import).
        try:
            from routers.plans import _compute_water_goal
            goal = int(_compute_water_goal(user_id).get("goal", 8) or 8)
        except Exception:
            goal = 8

        # Mensaje contextual según el estado actual
        if glasses >= goal:
            return (
                f"\n\n💧 HIDRATACIÓN HOY: El usuario ha consumido {glasses} de {goal} vasos "
                f"de agua hoy (meta alcanzada ✅). Si surge el tema de hidratación, puedes "
                f"reconocerlo. Toma esto en cuenta al hablar de energía o saciedad."
            )
        if glasses == 0:
            return (
                f"\n\n💧 HIDRATACIÓN HOY: El usuario aún no ha registrado ningún vaso de agua "
                f"hoy (meta diaria: {goal} vasos). Si la conversación lo permite (mañana, "
                f"comidas, energía), recuérdale amablemente la importancia de hidratarse."
            )
        pct = round((glasses / goal) * 100)
        return (
            f"\n\n💧 HIDRATACIÓN HOY: El usuario lleva {glasses} de {goal} vasos de agua "
            f"({pct}% de su meta diaria). Toma esto en cuenta al hablar de energía, "
            f"saciedad o digestión. Si lleva menos de la mitad y ya es tarde, sugiérele "
            f"acelerar el ritmo con amabilidad."
        )
    except Exception as e:
        logger.warning(f"⚠️ [AGENT-HYDRATION-CONTEXT] error: {e}")
        return ""


def chat_with_agent(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None):
    # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Wall-clock total para el path
    # non-stream del chat. Pre-fix: solo el stream emitía
    # `chat_stream_total_duration` (P1-CHAT-STREAM-DURATION), el
    # non-stream NO tenía métrica E2E — endpoint `/api/chat` (non-stream)
    # quedaba sin P99 graphable. Emit en `finally` del try/finally puntual
    # que envuelve `chat_graph_app.invoke` (más abajo) para cubrir todo
    # path: success / timeout / exception. Outcome se mapea: 'ok' /
    # 'timeout' / 'error'. Tooltip-anchor: P1-TOOLS-LLM-HARDENING.
    import time as _time_chat_total
    _chat_total_started_at = _time_chat_total.monotonic()
    _chat_total_outcome = "ok"

    # Obtener contexto de memoria inteligente (resúmenes + mensajes recientes)
    memory = build_memory_context(session_id, user_id)  # [P1-DREAMING-1] user_id → modelo del usuario
    
    # === RAG INJECTION (con Query Routing inteligente) ===
    user_facts_text = ""
    visual_facts_text = ""
    
    if user_id:
        rag_decision = rag_query_router(prompt)

        if not rag_decision.get("skip"):
            optimized_query = rag_decision.get("query", prompt)
            logger.info(f"🔍 [CHAT RAG] Buscando con query optimizada: '{optimized_query}'")

            # [P3-GENCHUNK-SPEED · 2026-06-01] Los dos round-trips de embedding
            # (texto vía gemini-embedding-001 vs multimodal vía
            # gemini-embedding-2) usan el MISMO `optimized_query`, golpean
            # modelos distintos y sus búsquedas vectoriales son independientes.
            # Antes corrían en serie (≈2× latencia de embedding en cache-miss).
            # Ahora corren concurrentes en un ThreadPoolExecutor (este path es
            # sync y corre en el threadpool de FastAPI). try/except POR-UNIDAD
            # preserva el aislamiento de fallos + el metric de observabilidad
            # P3-CHAT-OBSERVABILITY (ahora cada unidad falla independiente, lo
            # cual es estrictamente mejor que abortar la visual si la textual
            # falla). Idéntico al sibling stream.
            def _rag_text_unit():
                try:
                    query_emb = get_embedding(optimized_query)
                    if query_emb:
                        facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                        if facts_data:
                            logger.info(f"🧠 [CHAT RAG] Hechos textuales recuperados: {len(facts_data)}")
                            return "\n".join([f"• {item['fact']}" for item in facts_data])
                except Exception as e:
                    _emit_chat_rag_embedding_failed_metric_best_effort(user_id, session_id, "chat_with_agent")
                    logger.error(f"⚠️ [CHAT RAG] Error recuperando hechos textuales: {e}")
                return ""

            def _rag_visual_unit():
                try:
                    visual_query_emb = get_multimodal_embedding(optimized_query)
                    if visual_query_emb:
                        visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                        if visual_data:
                            logger.debug(f"📸 [CHAT RAG VISUAL] Entradas visuales recuperadas: {len(visual_data)}")
                            return "\n".join([f"• {item['description']}" for item in visual_data])
                except Exception as e:
                    _emit_chat_rag_embedding_failed_metric_best_effort(user_id, session_id, "chat_with_agent")
                    logger.error(f"⚠️ [CHAT RAG VISUAL] Error recuperando memoria visual: {e}")
                return ""

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _rag_ex:
                    _f_text = _rag_ex.submit(_rag_text_unit)
                    _f_visual = _rag_ex.submit(_rag_visual_unit)
                    user_facts_text = _f_text.result() or ""
                    visual_facts_text = _f_visual.result() or ""
            except Exception as e:
                logger.error(f"⚠️ [CHAT RAG] Error en ejecución concurrente de embeddings: {e}")

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

    schedule_type = form_data.get("scheduleType", "standard") if form_data else "standard"
    # Determinar si es un usuario autenticado o invitado
    is_authenticated = user_id and user_id != session_id and user_id != "guest"

    # [P2-CHAT-PROMPT-STATIC-PREFIX · 2026-06-01] Estáticos al frente, volátiles
    # al final → maximiza cache implícito de Gemini sobre el prefijo. Ver
    # `_chat_prompt_static_prefix`. Puro reorden; rama else = orden legacy.
    if _chat_prompt_static_prefix():
        system_prompt = CHAT_AGENT_INLINE_PROMPT
        system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
        system_prompt += build_tools_instructions(user_id)
        # --- bloques dinámicos (volátiles) al final ---
        system_prompt += build_temporal_context()
        system_prompt += build_circadian_context(schedule_type)
        system_prompt += build_temporal_proactive_context()
        if rag_context:
            system_prompt += f"\n{rag_context}"
    else:
        system_prompt = CHAT_AGENT_INLINE_PROMPT
        system_prompt += build_temporal_context()
        system_prompt += build_circadian_context(schedule_type)
        system_prompt += build_temporal_proactive_context()
        system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
        if rag_context:
            system_prompt += f"\n{rag_context}"
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

    # [P1-SUPERPERSONALIZATION-1 · 2026-06-19] Inyecta el bloque de súper
    # personalización (gustos/cultura/equipo/sabor/nivel/texto libre) también al
    # chat coach — reusa el mismo builder del generador de planes. Retorna "" si
    # el usuario no llenó el panel → no-op. Así el coach responde más preciso
    # (qué le ENCANTA, qué cocina prefiere, qué equipo tiene) sin tocar las
    # restricciones clínicas, que siguen viniendo de form_data estructurado.
    if form_data:
        try:
            from prompts.plan_generator import build_super_personalization_context
            system_prompt += build_super_personalization_context(form_data)
        except Exception as _sp_err:
            logger.warning(f"[P1-SUPERPERSONALIZATION-1] No se pudo inyectar súper personalización al chat: {_sp_err}")

    # [P3-CHAT-IDENTITY · 2026-06-20] Bloque de identidad + datos corporales
    # (nombre/sexo/edad/peso/altura/objetivo) → el coach SABE con quién habla y
    # personaliza (te saluda por tu nombre, adapta consejos por sexo/edad/objetivo).
    # Aditivo, NO clínico (no toca alergias/condiciones ni los macros del plan).
    # Best-effort: el nombre solo se carga para usuarios autenticados.
    try:
        _id_name = ""
        if user_id and user_id != session_id and user_id != "guest":
            _id_name = (get_user_profile(user_id) or {}).get("full_name") or ""
        system_prompt += build_user_identity_context(form_data or {}, _id_name)
    except Exception as _id_err:
        logger.warning(f"[P3-CHAT-IDENTITY] No se pudo inyectar identidad al chat: {_id_err}")

    if current_plan:
        # [P2-GENCHUNK-SPEED · 2026-06-01] Podar claves derivadas/pesadas antes
        # de serializar (shopping agregados, coherence telemetry, archived days).
        system_prompt += f"\n\nCONTEXTO CRÍTICO: El usuario actualmente tiene este plan de comidas activo:\n{json.dumps(_prune_plan_for_chat(current_plan))}\n\nUsa esta información para responder con exactitud preguntas sobre lo que le toca comer hoy o sugerir cambios basados en lo que ya tiene asignado (como desayuno, almuerzo o cena)."
        
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

        # [P3-AGENT-HYDRATION-CONTEXT · 2026-05-27] Inyectar hidratación
        # viva si el toggle está activo. Non-stream path no recibe
        # `local_date`, así que cae al UTC server-side dentro del helper.
        system_prompt += _build_hydration_context(user_id, local_date_str=None)

    config = {"configurable": {"thread_id": session_id}}

    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Pool separado para PostgresSaver
    # (session mode, port 5432) evita "SSL bad length / EOF" cuando Supavisor
    # mata conexiones idle del Transaction Pooler durante el chat stream.
    # Fallback defensivo a `connection_pool` si el split pool no se creó.
    _checkpoint_pool = chat_checkpoint_pool or connection_pool
    # [P3-CHAT-OBSERVABILITY · 2026-05-20] Alert si caímos al fallback del
    # transaction pooler — reabre el modo de fallo SSL bad length/EOF.
    # Cooldown 1h in-process previene contención bajo carga alta.
    if chat_checkpoint_pool is None and connection_pool is not None:
        _emit_checkpoint_pool_split_missing_alert_best_effort()
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
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chat_graph_invoke"
        ) as _ex:
            _fut = _ex.submit(chat_graph_app.invoke, inputs, config=config)
            try:
                final_state = _fut.result(timeout=_graph_timeout_s)
            except concurrent.futures.TimeoutError as _to_exc:
                _chat_total_outcome = "timeout"
                logger.error(
                    f"⏱️ [P0-CHAT-LLM-TIMEOUT] chat_graph_app.invoke excedió "
                    f"{_graph_timeout_s}s para session={session_id} user={user_id!r}. "
                    f"Posible Gemini hang / network issue."
                )
                raise TimeoutError(
                    f"chat_graph exceeded {_graph_timeout_s}s timeout"
                ) from _to_exc
    except Exception:
        # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Si _chat_total_outcome no
        # fue marcado por el branch específico de timeout, marca 'error'
        # genérico antes de re-raise. El emit en finally captura ambos.
        if _chat_total_outcome == "ok":
            _chat_total_outcome = "error"
        # Best-effort emit antes de re-raise (el caller no recibirá la
        # métrica si excepción rompe el flow).
        try:
            _total_dur_ms = int((_time_chat_total.monotonic() - _chat_total_started_at) * 1000)
            _emit_chat_stream_total_duration_best_effort(
                user_id, session_id, _chat_agent_model_name(),
                _total_dur_ms, _chat_total_outcome,
            )
        except Exception:
            pass
        raise

    end_time = time.time()
    duration_secs = round(float(end_time - start_time), 2)
    logger.info(f"✅ [COMPLETADO] LangGraph finalizó en {duration_secs} segundos.")
    logger.info("-------------------------------------------------------------\n")

    # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Emit total-duration del path
    # non-stream (outcome='ok'). Reusa el helper `_emit_chat_stream_total_duration_best_effort`
    # (SSOT) que ya emite con `node='chat_stream_total_duration'`. Queries
    # de SRE pueden diferenciar streams vs non-stream por `metadata.source`
    # — pero como hoy no lo necesitamos para alerting, el mismo node basta.
    try:
        _total_dur_ms = int((_time_chat_total.monotonic() - _chat_total_started_at) * 1000)
        _emit_chat_stream_total_duration_best_effort(
            user_id, session_id, _chat_agent_model_name(),
            _total_dur_ms, _chat_total_outcome,
        )
    except Exception:
        pass

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
    memory = build_memory_context(session_id, user_id)  # [P1-DREAMING-1] user_id → modelo del usuario
    
    # 🎭 ANÁLISIS DE SENTIMIENTO ADAPTATIVO (Solo Plus o superior)
    # [P3-GENCHUNK-SPEED · 2026-06-01] FASE 1 — `classify_sentiment` (gate
    # plus/ultra/admin) y `rag_query_router` (gate basic+) son LLM calls
    # independientes: ambas solo leen `prompt` y ninguna consume el output de
    # la otra. Antes corrían en serie antes del primer token (≈2 round-trips
    # Flash-Lite seriales sobre el critical path de TTFT). Ahora concurrentes
    # en un ThreadPoolExecutor (este path es un generador sync en el threadpool
    # de FastAPI). Ambas helpers tienen fallback seguro interno (neutral /
    # {skip:False,query:prompt}) y short-circuits propios (rag_router salta
    # mensajes casuales y tiene CB fast-path), así que la concurrencia NO añade
    # superficie de error ni cambia los inputs al LLM principal.
    sentiment_result = {}
    user_facts_text = ""
    visual_facts_text = ""
    _do_sentiment = plan_tier in ["plus", "ultra", "admin"]
    _do_rag = bool(user_id) and plan_tier in ["basic", "plus", "ultra", "admin"]
    rag_decision = None
    if _do_sentiment or _do_rag:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pre_ex:
            _f_sent = _pre_ex.submit(classify_sentiment, prompt) if _do_sentiment else None
            _f_rag = _pre_ex.submit(rag_query_router, prompt) if _do_rag else None
            if _f_sent is not None:
                try:
                    sentiment_result = _f_sent.result() or {}
                except Exception as _se:
                    logger.warning(f"⚠️ [CHAT SENTIMENT] fallo (neutral fallback): {_se}")
                    sentiment_result = {}
            if _f_rag is not None:
                try:
                    rag_decision = _f_rag.result()
                except Exception as _re2:
                    logger.warning(f"⚠️ [CHAT RAG ROUTER] fallo: {_re2}")
                    rag_decision = None

    # RAG INJECTION (con Query Routing inteligente) — FASE 2
    if _do_rag and rag_decision and not rag_decision.get("skip"):
        optimized_query = rag_decision.get("query", prompt)

        # [P3-GENCHUNK-SPEED · 2026-06-01] Los dos embeddings (texto vs
        # multimodal) sobre el MISMO `optimized_query` son independientes →
        # concurrentes. try/except por-unidad preserva el aislamiento de fallos
        # + el metric P3-CHAT-OBSERVABILITY. Espejo exacto del path non-stream.
        def _rag_text_unit():
            try:
                query_emb = get_embedding(optimized_query)
                if query_emb:
                    facts_data = search_user_facts(user_id, query_emb, threshold=0.5, limit=10)
                    if facts_data:
                        return "\n".join([f"• {item['fact']}" for item in facts_data])
            except Exception as e:
                _emit_chat_rag_embedding_failed_metric_best_effort(user_id, session_id, "chat_with_agent_stream")
                logger.error(f"⚠️ [CHAT RAG] Error texto (stream): {e}")
            return ""

        def _rag_visual_unit():
            try:
                visual_query_emb = get_multimodal_embedding(optimized_query)
                if visual_query_emb:
                    visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                    if visual_data:
                        return "\n".join([f"• {item['description']}" for item in visual_data])
            except Exception as e:
                _emit_chat_rag_embedding_failed_metric_best_effort(user_id, session_id, "chat_with_agent_stream")
                logger.error(f"⚠️ [CHAT RAG VISUAL] Error visual (stream): {e}")
            return ""

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _rag_ex:
                _f_text = _rag_ex.submit(_rag_text_unit)
                _f_visual = _rag_ex.submit(_rag_visual_unit)
                user_facts_text = _f_text.result() or ""
                visual_facts_text = _f_visual.result() or ""
        except Exception as e:
            logger.error(f"⚠️ [CHAT RAG] Error concurrente (stream): {e}")

    rag_context = ""
    if user_facts_text or visual_facts_text:
        rag_context = "\n--- MEMORIA VECTORIAL (RAG) ---\n"
        if user_facts_text: rag_context += f"{user_facts_text}\n"
        if visual_facts_text: rag_context += f"Inventario Visual:\n{visual_facts_text}\n"
        rag_context += "Úsalo para responder de forma súper personalizada.\n⚠️ REGLA DE CONFLICTO: LOS HECHOS PERMANENTES SON LEY.\n---------------------------------------------\n"

    schedule_type = form_data.get("scheduleType", "standard") if form_data else "standard"
    _base_inline = CHAT_VOICE_MODE_PROMPT if is_call_mode else CHAT_STREAM_INLINE_PROMPT

    # [P2-CHAT-PROMPT-STATIC-PREFIX · 2026-06-01] Estáticos al frente, volátiles
    # al final → maximiza cache implícito de Gemini. Ver `_chat_prompt_static_prefix`
    # (nota en chat_with_agent). Puro reorden; rama else = orden legacy.
    if _chat_prompt_static_prefix():
        system_prompt = _base_inline
        system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
        system_prompt += build_tools_instructions_stream(user_id)
        # --- bloques dinámicos (volátiles) al final ---
        system_prompt += build_temporal_context()
        system_prompt += build_circadian_context(schedule_type)
        system_prompt += build_temporal_proactive_context()
        # 🎭 Personalidad adaptativa basada en el sentimiento detectado (per-turn)
        if sentiment_result.get("instruction"):
            system_prompt += f"\n\n{sentiment_result['instruction']}"
        if rag_context:
            system_prompt += f"\n{rag_context}"
    else:
        system_prompt = _base_inline
        system_prompt += build_temporal_context()
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

    # [P1-SUPERPERSONALIZATION-1 · 2026-06-19] Inyecta el bloque de súper
    # personalización (gustos/cultura/equipo/sabor/nivel/texto libre) también al
    # chat coach — reusa el mismo builder del generador de planes. Retorna "" si
    # el usuario no llenó el panel → no-op. Así el coach responde más preciso
    # (qué le ENCANTA, qué cocina prefiere, qué equipo tiene) sin tocar las
    # restricciones clínicas, que siguen viniendo de form_data estructurado.
    if form_data:
        try:
            from prompts.plan_generator import build_super_personalization_context
            system_prompt += build_super_personalization_context(form_data)
        except Exception as _sp_err:
            logger.warning(f"[P1-SUPERPERSONALIZATION-1] No se pudo inyectar súper personalización al chat: {_sp_err}")

    # [P3-CHAT-IDENTITY · 2026-06-20] Identidad + datos corporales (paridad con el
    # path no-stream). El coach conoce nombre/sexo/edad/peso/altura/objetivo y
    # personaliza. Aditivo, no clínico. Nombre solo para autenticados (best-effort).
    try:
        _id_name = ""
        if user_id and user_id != session_id and user_id != "guest":
            _id_name = (get_user_profile(user_id) or {}).get("full_name") or ""
        system_prompt += build_user_identity_context(form_data or {}, _id_name)
    except Exception as _id_err:
        logger.warning(f"[P3-CHAT-IDENTITY] No se pudo inyectar identidad al chat: {_id_err}")

    if current_plan:
        # [P2-GENCHUNK-SPEED · 2026-06-01] Podar claves derivadas/pesadas (ver
        # _prune_plan_for_chat) antes de serializar — paridad con el path no-stream.
        system_prompt += f"\nCONTEXTO CRÍTICO: Plan activo:\n{json.dumps(_prune_plan_for_chat(current_plan))}\n"
        
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

        # [P3-AGENT-HYDRATION-CONTEXT · 2026-05-27] Inyectar hidratación
        # viva si el toggle está activo. El stream path SÍ recibe
        # `local_date` del cliente, que pasamos al helper para mayor
        # precisión en zonas horarias no-UTC.
        system_prompt += _build_hydration_context(user_id, local_date_str=local_date)

    config = {"configurable": {"thread_id": session_id}}

    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Mismo split que el callsite del
    # non-stream chat. Sesión session-mode evita SSL EOF durante el SSE.
    # Compilamos usando PostgresSaver sincrónico porque astream_events nativo asíncrono tiene problemas en Windows
    _checkpoint_pool = chat_checkpoint_pool or connection_pool
    # [P3-CHAT-OBSERVABILITY · 2026-05-20] Mismo alert que el callsite del
    # non-stream chat — el cooldown 1h dedupea bajo carga concurrente.
    if chat_checkpoint_pool is None and connection_pool is not None:
        _emit_checkpoint_pool_split_missing_alert_best_effort()
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
    _stream_outcome = "ok"  # 'ok' / 'timeout_total' / 'timeout_inactivity' / 'error' / 'cancelled' / 'checkpoint_lost'
    # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20] Contador de chunks AI ya
    # entregados al frontend. Si una excepción de checkpoint Postgres (SSL
    # bad length / EOF detected) ocurre DESPUÉS de haber streamado contenido,
    # la respuesta del LLM ya llegó al user — perder el checkpoint final NO
    # justifica el banner rojo. Ver `except Exception` abajo para la lógica
    # de degradación silenciosa. Tooltip-anchor: P1-CHAT-CHECKPOINT-DEGRADE.
    _chunks_yielded = 0

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
                            # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20]
                            _chunks_yielded += 1
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
        # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20] Degradación silenciosa
        # cuando el `PostgresSaver.put_writes` final muere por SSL bad length
        # / EOF detected POST-streaming. Modo de fallo: Supavisor mata la
        # conexión de `chat_checkpoint_pool` mientras LangGraph la mantiene
        # checkout durante el LLM call (~10-30s). El for loop ya emitió todo
        # el contenido al frontend; solo falla el `_checkpointer_put_after_previous`
        # interno de LangGraph. Yieldar 'error' al user es engañoso — ya
        # vio la respuesta completa, perder el checkpoint solo significa
        # que el próximo turn recargará history desde db_chat (no-op visible).
        #
        # Heurística defensiva: clasificamos como "checkpoint_lost" SOLO si
        # (a) la excepción menciona uno de los markers SSL del fallo + (b)
        # ya entregamos ≥1 chunk al frontend. Si chunks_yielded=0, el LLM
        # ni emitió tokens → el fallo es real (probablemente conn dead al
        # primer get_state) y SÍ debemos mostrar el banner.
        #
        # Pool recycling agresivo (db_core.py: min_size=0, max_idle=30s)
        # reduce frecuencia ~95%; el degrade silencioso cierra el residuo.
        # Tooltip-anchor: P1-CHAT-CHECKPOINT-DEGRADE.
        _err_str = str(e)
        _is_checkpoint_ssl_death = any(
            marker in _err_str
            for marker in (
                "SSL error: bad length",
                "EOF detected",
                "flush request failed",
                "connection is lost",
                "no connection to the server",
            )
        )
        if _is_checkpoint_ssl_death and _chunks_yielded > 0:
            _stream_outcome = "checkpoint_lost"
            logger.warning(
                f"⚠️ [P1-CHAT-CHECKPOINT-DEGRADE] Checkpoint conn died "
                f"post-stream (SSL/EOF), pero {_chunks_yielded} chunks ya "
                f"entregados al frontend → degradación silenciosa. "
                f"session={session_id} user={user_id} err={_err_str[:120]}"
            )
            return
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
    # [P3-PANTRY-INVALIDATE-FROM-CHAT · 2026-05-22] Timestamp epoch ms del
    # turn donde una tool mutó `user_inventory`. None = no se tocó pantry
    # en este stream; frontend silencia el flag de invalidación.
    pantry_modified_at = None
    # [P3-AGENT-DEPLETE · 2026-05-22] Items que el agente marcó como
    # agotados (vía `modify_pantry_inventory(items_to_deplete)`).
    # Default None — sin items agotados, frontend no toca localStorage.
    pantry_depleted_items = None

    if final_state_snapshot and final_state_snapshot.values:
        updated_fields = final_state_snapshot.values.get("updated_fields", {})
        new_plan = final_state_snapshot.values.get("new_plan", None)
        coherence_warnings = final_state_snapshot.values.get("coherence_warnings") or []
        pantry_modified_at = final_state_snapshot.values.get("pantry_modified_at")
        pantry_depleted_items = final_state_snapshot.values.get("pantry_depleted_items")
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
    yield f"data: {json.dumps({'type': 'done', 'response': final_content, 'updated_fields': updated_fields, 'new_plan': new_plan, 'coherence_warnings': coherence_warnings, 'pantry_modified_at': pantry_modified_at, 'pantry_depleted_items': pantry_depleted_items})}\n\n"