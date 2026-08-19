# backend/agent.py

import os
import logging
import time
import json
import re
import unicodedata
logger = logging.getLogger(__name__)

from constants import strip_accents, CULINARY_KNOWLEDGE_BASE, validate_ingredients_against_pantry, _to_base_unit
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek con router por tier.
from llm_provider import (ChatDeepSeek, DEEPSEEK_FLASH, GPT56_LUNA,
                          build_chat_llm, is_openai_model, resolve_model_for_user)
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
from datetime import date, datetime, timezone, timedelta
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
# ⚠️ NO insertar nada entre esta línea y la asignación de abajo: el test recortaba una ventana desde
# el ancla, así que un bloque intermedio empujaba la lista fuera del recorte (ya pasó).
_CHAT_PLAN_PRUNE_KEYS = (
    "aggregated_shopping_list",
    "aggregated_shopping_list_weekly",
    "aggregated_shopping_list_biweekly",
    "aggregated_shopping_list_monthly",
    "_shopping_coherence_block",
    "_shopping_coherence_block_history",
    "_archived_days",
    "calc_household_multiplier",
    # [P1-CULINARY-CONTRACT · post-review-final] Gemelas de `_shopping_coherence_block*`
    # arriba: telemetría interna del scan culinario determinista (capa 1) y del juez LLM
    # (capa 2) — violaciones/cobertura/history no aportan al razonamiento del chat-agent
    # y sin podarlas se serializaban al system prompt EN CADA turno (denylist no las tenía).
    "_culinary_contract_violations",
    "_culinary_contract_coverage",
    "_culinary_judge_history",
    # [P1-CHAT-CLINICAL-TOOL · 2026-07-12] Reportes internos de QA del pipeline
    # — el agente no los necesita para conversar y engordaban CADA turno del
    # chat (audit del plan vivo del owner: dish_quality_report/variety_report/
    # _review_issues_raw pueden ser KBs). Lo user-facing se queda: insights,
    # micronutrient_report/advice, budget_reconciliation, goal_eta,
    # _review_disclaimer, _quality_degraded*.
    "_review_issues_raw",
    "dish_quality_report",
    "variety_report",
    "_recipe_coherence_errors",
    "_recent_chunk_lessons",
    "_last_chunk_learning",
    "data_provenance",
    "resolution_coverage",
    "_transform_gate_advisory_final",
)

# [P2-SWAP-NUM-MEALS · 2026-07-29] (audit solver+seeder v4) El slot-target del swap se derivaba
# SIEMPRE con `num_meals = 4` (nadie aporta el campo) → en perfiles de 3/5/6 comidas el matcher
# devolvía la cuota del slot equivocado y el solver re-escalaba físicamente el plato nuevo a ese
# target. Con el knob, si el campo falta se deriva del perfil vía `decide_meals_per_day`; el literal
# 4 sigue siendo el último fallback, así que nunca es peor que antes.
# Rollback sin redeploy: MEALFIT_SWAP_NUM_MEALS_FROM_PLAN=false.
SWAP_NUM_MEALS_FROM_PLAN = _env_bool("MEALFIT_SWAP_NUM_MEALS_FROM_PLAN", True)

# [P2-SWAP-SLOT-KEY-MATCH · 2026-07-30] (audit solver+seeder v5) Calificativos de merienda →
# sufijo de la clave de slot. El orden importa: 'noche'/'nocturna' antes que nada, porque el
# matcher genérico de abajo los resolvía a `merienda_am` por ser la PRIMERA merienda del dict.
_MERIENDA_QUALIFIERS = (
    (("noct", "noche"), "merienda_noche"),
    (("pm", "tarde"), "merienda_pm"),
    (("am", "manana"), "merienda_am"),
)


def _num_meals_from_same_day(form_data) -> "int | None":
    """[P3-SWAP-NUMMEALS-SAMEDAY · 2026-07-30] (audit solver+seeder v5) Nº de comidas del día
    que se está editando, derivado de `same_day_other_meals` (las OTRAS comidas del día) + la
    que se swapea.

    Es el ground truth del plan vivo, frente a `decide_meals_per_day(form_data)` que re-deriva
    del PERFIL y puede diverger: el usuario pudo elegir el conteo a mano al generar, la regla de
    alto gasto depende de un kcal que el callsite no pasa, o el perfil cambió después del plan.
    `None` si no hay dato (el caller sigue con su cadena de fallbacks).
    tooltip-anchor: P3-SWAP-NUMMEALS-SAMEDAY"""
    try:
        _sdo = (form_data or {}).get("same_day_other_meals") or []
        if isinstance(_sdo, (list, tuple)) and _sdo:
            return len(_sdo) + 1
    except Exception as _exc:
        logger.debug(
            "[P2-SILENT-DEGRADATION] num_meals desde same_day_other_meals no derivable "
            "(se cae al perfil): %s: %s", type(_exc).__name__, str(_exc)[:160])
    return None


def _resolve_swap_slot_key(meal_type, slots: dict) -> "str | None":
    """[P2-SWAP-SLOT-KEY-MATCH · 2026-07-30] (audit solver+seeder v5) Resuelve el nombre de slot
    del plato que se swapea ('Merienda Nocturna') a la clave de `allocate_macros_per_slot`.

    El matcher era `k == _mt or k.split("_")[0] in _mt or _mt in k` sobre un dict ORDENADO
    (desayuno, merienda_am, almuerzo, merienda_pm, cena, merienda_noche). Para 'merienda
    nocturna' el primer match era `merienda_am` vía `'merienda' in 'merienda nocturna'` → cuota
    0.12 en vez de 0.08. En un bariátrico de 6 comidas y 1200 kcal eso son 144 kcal de target
    en vez de 96 (×1.5), y el solver determinista re-escala FÍSICAMENTE el plato a esa cuota,
    sobre-dimensionándolo para un pouch de 150-200 mL. El fix v4 (P2-SWAP-NUM-MEALS) redujo el
    error de ×1.875 a ×1.5 corrigiendo el CONTEO de comidas, pero no tocó el pareo de la clave —
    y su propio comentario usa este caso como motivación.

    En el split de 5 comidas el bug era inocuo por accidente (merienda_am y merienda_pm comparten
    0.10); solo `merienda_noche` del split de 6 diverge.
    tooltip-anchor: P2-SWAP-SLOT-KEY-MATCH"""
    _mt = strip_accents(str(meal_type or "").lower()).strip()
    if not _mt or not isinstance(slots, dict):
        return None
    if _mt in slots:
        return _mt
    # 1) Calificativo explícito de merienda: gana SIEMPRE al match genérico.
    if "merienda" in _mt:
        for _toks, _key in _MERIENDA_QUALIFIERS:
            if any(t in _mt for t in _toks) and _key in slots:
                return _key
    # 2) Match genérico (desayuno/almuerzo/cena y meriendas sin calificativo).
    _k = next((k for k in slots if k.split("_")[0] in _mt or _mt in k), None)
    if _k is not None:
        return _k
    # 3) Último recurso: cualquier merienda disponible.
    if "merienda" in _mt:
        return next((k for k in slots if k.startswith("merienda")), None)
    return None


def _plan_mode_for_chat(user_id):
    """El modo del usuario (`'plan'` | `'tracking'`), para el encuadre del chat.

    [P1-AGENT-WELCOME-TRACKING · 2026-08-14] Wrapper fino sobre
    `plan_mode.get_plan_mode` que existe por dos razones: (a) los tests lo
    monkeypatchean sin tocar la DB, y (b) el fail-open vive AQUÍ y no en cada
    caller — si la lectura falla, se asume 'plan', que es el comportamiento
    histórico y el de la inmensa mayoría de usuarios. Degradar el chat de todos
    por un fallo puntual de DB sería peor que el bug que esto arregla.
    """
    from plan_mode import get_plan_mode
    return str((get_plan_mode(user_id) or {}).get("plan_mode") or "plan")


def _plan_context_for_chat(user_id, current_plan):
    """[P1-AGENT-WELCOME-TRACKING · 2026-08-14] El bloque del plan para el system
    prompt, con el ENCUADRE que corresponde al modo del usuario.

    EL BUG QUE CIERRA. Los dos paths del chat (stream y no-stream) inyectaban el
    plan con el literal «tiene este plan de comidas activo» / «Plan activo» —
    incondicional. La pausa del modo contador conserva `plan_data` a propósito
    (es lo que permite «Reanudar»), así que el modelo recibía un plan PAUSADO
    jurado como vigente y contestaba «¿qué me toca hoy?» con él. La mitad visible
    (el saludo del frontend recitando la cena) la reportó el dueño con captura;
    esta es la misma mentira un piso más abajo.

    PAUSADO ≠ AMPUTADO: el plan viaja igual en ambos modos. El usuario puede
    preguntar qué tenía su plan, y el agente debe poder responder y ofrecer
    reanudar — la puerta de vuelta que P1-PLAN-MODE dejó abierta. Lo que cambia
    es el encuadre: en pausa, el modelo tiene PROHIBIDO presentarlo como lo que
    toca hoy.

    Es un helper y no dos f-strings inline por la lección de P1-CHAT-PAST-DAYS:
    «la divergencia entre ambos [paths] ya ha causado bugs antes».
    """
    plan_json = json.dumps(_prune_plan_for_chat(current_plan))
    try:
        en_pausa = _plan_mode_for_chat(user_id) == "tracking"
    except Exception as e:
        logger.warning(f"[P1-AGENT-WELCOME-TRACKING] plan_mode ilegible, asumo 'plan': {e}")
        en_pausa = False

    if en_pausa:
        return (
            "\n\nCONTEXTO CRÍTICO: El usuario puso su plan de comidas EN PAUSA — "
            "eligió usar la app solo como contador de macros y diario (modo "
            "seguimiento). El plan se conserva para cuando quiera reanudarlo:\n"
            f"{plan_json}\n\n"
            "NO presentes las comidas de este plan como lo que le toca comer hoy, "
            "ni lo recites al saludar, ni sugieras cambios sobre él como si "
            "estuviera vigente: hoy el usuario decide libremente qué come y tu "
            "papel es ayudarle a registrarlo y a cuadrar sus macros. Si ÉL "
            "pregunta por su plan pausado, respóndele con estos datos y "
            "recuérdale que puede reanudarlo desde su Historial (es gratis y "
            "retoma donde quedó)."
        )
    return (
        "\n\nCONTEXTO CRÍTICO: El usuario actualmente tiene este plan de comidas "
        f"activo:\n{plan_json}\n\n"
        "Usa esta información para responder con exactitud preguntas sobre lo que "
        "le toca comer hoy o sugerir cambios basados en lo que ya tiene asignado "
        "(como desayuno, almuerzo o cena)."
    )


def _plan_vigente_para_prompt(user_id, current_plan):
    """[P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] El plan que GOBIERNA el día, o
    `None` si el usuario lo tiene en pausa.

    POR QUÉ EXISTE ESTE DATO Y NO OTRO GATE POR CALL SITE. Esta misma mañana
    `_plan_context_for_chat` aprendió la pausa y la auditoría encontró que otros
    cuatro bloques del MISMO prompt la contradecían unas líneas más abajo:
    «HOY es el día N del menú», «DÍAS QUE FALTAN POR GENERARSE… ATRASADO», «hoy
    te quedan N comidas del plan», y el presupuesto de kcal del plan congelado.
    Gatear call sites es exactamente lo que produjo ese agujero: se arregla el
    que se ve y quedan los demás.

    Aquí el modo se resuelve UNA vez por turno y se deriva un DATO. Las secciones
    prescriptivas reciben `plan_vigente`; sus guardas de shape (que ya existían,
    `isinstance(current_plan, dict)`) las apagan solas. Los builders no aprenden
    nada de modos, y uno futuro que reciba este dato queda gateado sin que nadie
    se acuerde de gatearlo.

    Ojo con la distinción, que es todo el diseño:
      `current_plan`  el plan real. Lo sigue recibiendo `_plan_context_for_chat`,
                      que en pausa lo entrega con su encuadre (PAUSADO ≠
                      AMPUTADO: si el usuario pregunta por su plan hay que poder
                      responderle y ofrecerle reanudar).
      `plan_vigente`  el plan que manda HOY. `None` mientras esté en pausa.

    Fail-open al comportamiento histórico: si el modo no se puede leer, se asume
    'plan'. Degradar el chat de todos por un fallo de DB sería peor que el bug.
    """
    # [P2-CHAT-PLAN-TOOLS-PAUSE · 2026-08-15] Sin plan que gatear no hay nada que
    # preguntar: el resultado sería `None` en los dos modos. Este helper subió al
    # tope de ambas funciones de chat, así que corre en TODOS los turnos —
    # incluidos los de invitados y los de usuarios sin plan; sin este corte serían
    # otros tantos roundtrips a `user_profiles` que no cambian nada.
    if not current_plan or not user_id or user_id == "guest":
        return current_plan
    try:
        if _plan_mode_for_chat(user_id) == "tracking":
            return None
    except Exception as e:
        logger.warning(f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] plan_mode ilegible, asumo 'plan': {e}")
    return current_plan


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
    TITLE_GENERATION_PROMPT, RAG_ROUTER_PROMPT,
    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (FINAL-FIX F1c)] variante país-aware de
    # SWAP_MEAL_PROMPT_TEMPLATE (T2 pattern) — swap_meal() la usa en vez del template crudo.
    build_swap_meal_prompt_template
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
    build_clinical_guard_context,
    build_language_directive,
)
# [P1-CHAT-PAST-DAYS · 2026-07-27] Memoria de días pasados — doc:
# backend/docs/chat_past_days_memory.md
from chat_history_context import (
    build_past_diary_block,
    build_past_plan_days_block,
    build_pending_plan_days_lines,
    chat_history_days,
    rd_today,
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

# [P1-SWAP-LUNA · 2026-08-05] El swap (y por herencia `regenerate-day`, que es un bucle
# de swaps) pasa de `deepseek-v4-flash` a `gpt-5.6-luna` — el mismo modelo con el que ya
# NACE el plan en `day_generator`.
#
# La asimetría que cierra: el plan se generaba con luna y cada plato que el usuario
# sustituía después lo escribía flash. Es decir, cada actualización cambiaba un plato del
# modelo bueno por uno del modelo barato, dentro de un día cuyos macros ya estaban
# cuadrados.
#
# Medido contra la API real desde el VPS (2026-08-05, mismo prompt de swap, 3 corridas):
#
#   flash  temperature=0.3        8,1 s   → "Salmón glaseado…" LAS TRES VECES
#   luna   reasoning_effort=low   8,2 s   → mofongo tropical / mofongo con camarones /
#                                            pescado en escabeche criollo
#   luna   reasoning_effort=med  16,5 s   → 3 platos distintos
#
# Luna en `low` cuesta lo MISMO en espera que flash y ya rompe el colapso de flash hacia
# el salmón (que el dueño reportó como "¿por qué me apareció este salmón de repente?").
# `medium` dobla la espera; ver `_swap_reasoning_effort` para por qué cada superficie usa
# uno distinto.
#
# ⚠️ Fail-safe explícito: `build_chat_llm` LEVANTA si le pides un modelo OpenAI sin
# `OPENAI_API_KEY` en el entorno (contrato fail-loud de `_openai_api_key`). Un swap es
# user-facing: preferimos degradar al router por tier (flash) que devolverle un 500 al
# usuario porque falta una env var. Mismo criterio que la red post-fallo P1-NET-LUNA, que
# degrada a `deepseek-v4-pro` cuando la key no está.
_SWAP_MODEL_DEFAULT = GPT56_LUNA


def _chat_agent_swap_model_name(user_id: Optional[str] = None) -> str:
    override = _env_str("MEALFIT_CHAT_AGENT_SWAP_MODEL", "")
    if override:
        return override
    if is_openai_model(_SWAP_MODEL_DEFAULT) and not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "⚠ [P1-SWAP-LUNA] swap pedido en %r sin OPENAI_API_KEY → fail-safe al "
            "router por tier. El swap sigue funcionando, con el modelo anterior.",
            _SWAP_MODEL_DEFAULT,
        )
        return resolve_model_for_user(user_id)
    return _SWAP_MODEL_DEFAULT


# Valores que el API acepta de verdad para `reasoning_effort` en gpt-5.6-luna.
# ⚠️ Verificado CONTRA LA API, no contra la documentación (2026-08-05): `minimal` NO
# existe en este modelo — devuelve 400 "Supported values are: 'none', 'low', 'medium',
# 'high', and 'xhigh'". Un valor inválido aquí no degrada: rompe el swap entero.
_SWAP_EFFORT_VALID = ("none", "low", "medium", "high", "xhigh")

# Effort por SUPERFICIE, porque no cuestan lo mismo aunque sean el mismo motor:
#
#   · plato individual (`/swap-meal`) → UNA llamada. `medium` = ~16,5 s. Se tolera.
#   · día completo (`/regenerate-day`) → bucle EN SERIE de 4-5 swaps (no hay paralelismo;
#     `for meal in meals:` en routers/plans.py). Con `medium` el día pasa de ~35 s a
#     66-83 s. Más de un minuto de spinner es un problema de UX peor que el que este
#     cambio arregla, y en el A/B del day-gen la escalera de effort salió DECRECIENTE
#     (más effort no compró calidad proporcional). Por eso el día va en `low`, que mide
#     igual que flash.
#
# Decisión del dueño (2026-08-05) con estas cifras delante. Ambos son knobs: si el
# `band_score` por operación (P1-CHANGE-OUTCOME-TELEMETRY) muestra que `medium` sí gana
# precisión de macros, se sube el del día sin redeploy.
_SWAP_EFFORT_DEFAULTS = {"individual": "medium", "day": "low"}


def _swap_reasoning_effort(surface: str = "individual") -> str:
    """Effort de razonamiento para la superficie de swap dada.

    Cae al default de la superficie ante cualquier valor no reconocido — un knob mal
    escrito degrada a la configuración conocida, nunca a un 400 del provider.
    """
    key = surface if surface in _SWAP_EFFORT_DEFAULTS else "individual"
    default = _SWAP_EFFORT_DEFAULTS[key]
    val = (_env_str("MEALFIT_SWAP_EFFORT_%s" % key.upper(), default) or "").strip().lower()
    return val if val in _SWAP_EFFORT_VALID else default

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


def _chat_hold_pretool_text() -> bool:
    """[P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] Kill switch.

    Con `False` el stream vuelve al comportamiento anterior (todo el texto
    pre-tool se emite en vivo, deliberación incluida). Existe porque retener
    el texto cambia la sensación de la app: en un turno SIN herramienta la
    respuesta aparece de golpe al final en vez de irse escribiendo. Si eso
    resulta peor que el problema que arregla, se revierte sin redesplegar.
    """
    return _env_bool("MEALFIT_CHAT_HOLD_PRETOOL_TEXT", True)


def _chat_pretool_narration_max_chars() -> int:
    """Frontera entre narración corta (se emite) y deliberación (se descarta).

    300 chars: la narración legítima que P1-CHAT-NARRATION-KEPT preserva son
    frases como "Lo anoto y te digo cómo va" (~30-60 chars); la deliberación
    del incidente pasaba de 4.000. El umbral vive en medio de un hueco de dos
    órdenes de magnitud, así que no es un número peleado — moverlo entre 150 y
    1.000 no cambiaría ninguno de los dos casos.
    """
    return _env_int(
        "MEALFIT_CHAT_PRETOOL_NARRATION_MAX_CHARS",
        300,
        validator=lambda v: 0 <= v <= 20000,
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
    # [P2-CHAT-STREAM-TIMEOUT-TOOLS · 2026-07-12] Clamp 120→360: las tools
    # largas corren DENTRO de un solo nodo sin emitir eventos — con el retry
    # de expansión de despensa (P1-CHAT-MODIFY-EXPAND-FALLBACK) un
    # modify_single_meal puede callar ~2-4 min (dos generaciones LLM). Con el
    # clamp viejo el env no podía cubrirlo y el stream moría con el plato YA
    # persistido (vivo: "dio un error aunque actualizó el plato"). El default
    # 25s se conserva para conversación normal; el VPS sube la ventana por env.
    return _env_float(
        "MEALFIT_CHAT_STREAM_INACTIVITY_TIMEOUT_S",
        25.0,
        validator=lambda v: 0.0 < v <= 360.0,
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


# [P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] `get_realtime_pantry`/`aggregate_shopping_list`
# (shopping_calculator.py) cachean el techo de sus caps P6 en `_person_weeks =
# multiplier * num_days / 7.0` — sin `num_days`/`multiplier` reales, ese cómputo cae al
# fallback `_pw_days=3.0` + `multiplier=1.0` ⇒ `_person_weeks=1.0` SIEMPRE, sin importar
# household ni duración (semanal/quincenal/mensual) del plan real. Verificado ejecutando
# el agregador: un plan mensual de 2 personas veía el cap de atún caer de 9 latas
# (correcto) a 2 (semanal-para-1, el default) — la «nevera virtual» que ve el LLM del
# swap (path PRIMARIO) quedaba capada a 1 persona-semana en cualquier plan
# multi-semana/household>1.
#
# Mismo SSOT que `get_shopping_list_delta`/`routers/plans.py::scaled_30`: `num_days` =
# días REALMENTE generados (el ciclo base, típ. `PLAN_CHUNK_SIZE`=3) y `multiplier` =
# `household × cycle_qty_multiplier(duración) × 7/num_days` — el último factor deshace
# el promedio-a-7-días que ese mismo cómputo aplica upstream (comentario en `_pw_days`/
# `_person_weeks`, shopping_calculator.py:~9923), para que `_person_weeks` recupere
# exactamente `household × cycle_qty_multiplier(duración)` (la invariante que ancla
# `test_p1_person_weeks_cycle_aware.py`).
#
# `plan_data` sin `days` (guest sin plan, BD caída, dict incompleto) → `(None, 1.0)`,
# el fallback histórico exacto de `aggregate_and_deduct_shopping_list` — fail-open, nunca
# inventa un household/duración que no existe.
def _virtual_pantry_num_days_and_multiplier(plan_data) -> tuple:
    if not isinstance(plan_data, dict):
        return None, 1.0
    days = plan_data.get("days") or []
    num_days = len(days)
    if num_days < 1:
        return None, 1.0
    try:
        household = float(plan_data.get("calc_household_multiplier") or 1.0)
    except (TypeError, ValueError):
        household = 1.0
    if not household or household <= 0:
        household = 1.0
    duration_key = str(plan_data.get("calc_grocery_duration") or "").strip().lower()
    try:
        from shopping_calculator import cycle_qty_multiplier as _cycle_qty_mult_vp
        duration_factor = _cycle_qty_mult_vp(duration_key)
    except Exception:
        duration_factor = 1.0
    multiplier = household * duration_factor * (7.0 / num_days)
    return num_days, multiplier


# [P1-PANTRY-STRICT-CONSENT · 2026-08-02] "Nevera estricta + consentimiento" — decisión del
# owner: tras la compra inicial, swap/regen-day/fix-sodium-day cocinan SOLO de la Nevera
# FÍSICA real (`user_inventory`) por default; si el chef no encuentra alternativa ahí, el
# sistema PREGUNTA (nombre + cantidad + precio estimado) en vez de introducir el ingrediente
# en silencio. Caso real que lo motiva: un swap metió catibías de YUCA (75g de un día YA
# ARCHIVADO del plan, jamás registrada en `user_inventory`) sin preguntar — la lista de
# compras "renació" con 1 ítem y el botón "Ya compré la lista" reapareció SIN que el usuario
# hubiera dicho que sí a comprar nada. Causa raíz (ver reporte P1-PANTRY-STRICT-CONSENT):
# `clean_ingredients` (el universo que valida `validate_ingredients_against_pantry`) se
# construía con `get_realtime_pantry(plan_data)` — TODOS los ingredientes del plan
# (acumulativo, nunca expira, solo decrementa por consumo LOGGEADO en el diario) — no con
# la Nevera física. `regenerate-day` YA usa la Nevera real vía `pantry_override`
# (P2-REGEN-DAY-PANTRY-OVERRIDE, routers/plans.py) — este fix cierra la MISMA brecha para
# `/swap-meal` y `/fix-sodium-day`, que no seteaban ese override.
def _pantry_strict_updates_enabled() -> bool:
    """Knob `MEALFIT_PANTRY_STRICT_UPDATES` (default True). OFF ⇒ `swap_meal()` vuelve al
    comportamiento legacy exacto (universo = plan completo vía `get_realtime_pantry`,
    `allow_new_ingredients` ignorado, `swap_meal_with_consent` delega 1:1 a `swap_meal`)."""
    return os.environ.get("MEALFIT_PANTRY_STRICT_UPDATES", "true").strip().lower() in ("1", "true", "yes", "on")


def _swap_real_pantry_ledger_lines(user_id: str) -> list:
    """Universo autorizado = Nevera FÍSICA real (`user_inventory` filas con quantity>0,
    disponible = quantity - reserved_quantity), nombres canónicos vía
    `IngredientNutritionDB.lookup` (mismo matching alias/sinónimo que usa el resto del
    pipeline). Mirror INTENCIONAL de `routers/plans.py::_inventory_grams_ledger` +
    `_ledger_to_pantry_lines` (P2-REGEN-DAY-PANTRY-OVERRIDE, ya probado en producción vía
    regenerate-day) — no se importa desde ahí para evitar un ciclo `agent` ↔
    `routers.plans` (`routers/plans.py` ya hace `from agent import swap_meal, ...`).
    Fail-open: `[]` si falla el fetch (guest, DB caída, etc.) — el caller interpreta una
    lista vacía como "Nevera vacía", NUNCA cae de vuelta al plan (ese fallback es
    precisamente el leak que este fix cierra)."""
    if not user_id or user_id == "guest":
        return []
    try:
        from db import get_raw_user_inventory
        from nutrition_db import IngredientNutritionDB
        db = IngredientNutritionDB()
        ledger: dict = {}
        for row in get_raw_user_inventory(user_id) or []:
            try:
                info = db.lookup(row.get("ingredient_name") or "")
                if not info:
                    continue
                qty = row.get("available_quantity")
                if qty is None:
                    qty = row.get("quantity") or 0
                grams = db.to_grams(float(qty or 0), row.get("unit") or "", info)
                if grams and grams > 0:
                    ledger[info.name] = ledger.get(info.name, 0.0) + grams
            except Exception:
                continue
        return [f"{int(round(g))}g de {name}" for name, g in ledger.items() if g and g > 0]
    except Exception as e:
        logger.debug(f"[P1-PANTRY-STRICT-CONSENT] ledger real no-op: {type(e).__name__}: {e}")
        return []


def _pantry_singular_key(token: str) -> str:
    """[P1-SWAP-PANTRY-PLURAL · 2026-08-05] Clave singular/plural de UN token.

    Solo recorta la `s` final en tokens de mas de 3 letras: `res`, `mas`, `sal`
    quedan intactos. No es un stemmer: es lo minimo para que `huevos` y `huevo`
    compartan clave.
    """
    t = (token or "").strip()
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def _pantry_tokens(blob: str) -> set:
    """Tokens de la nevera, ya reducidos a su clave singular."""
    return {_pantry_singular_key(t) for t in re.split(r"[^a-z0-9]+", blob or "") if t}


def pantry_contains_food(blob: str, name: str) -> bool:
    """La nevera (blob de `clean_ingredients`) contiene este alimento?

    [P1-SWAP-PANTRY-PLURAL · 2026-08-05] El chequeo era `name in blob`, subcadena
    cruda. Con la nevera diciendo "Huevo" y el guard preguntando por "huevos", la
    subcadena falla porque el PLURAL es mas largo que el singular: `"huevos" in
    "huevo"` es False. Medido en produccion el 2026-08-05 contra la nevera real
    del dueno (45 items, "Huevo: 2 carton"): el reparador determinista de
    `P1-SWAP-COHERENCE-REPAIR` se declaraba "fuera de nevera" y el swap moria tras
    3 intentos IDENTICOS -> 422 -> plato original conservado. Era la causa
    dominante de los cambios que no devolvian nada.

    Emparejamos por TOKEN COMPLETO, no por subcadena: es lo que impide la familia
    de bugs de esta casa (`pollo` dentro de `repollo`, `sal` dentro de `salsa`,
    `res` dentro de `fresco`). Un alimento multi-palabra exige que TODOS sus
    tokens esten.

    La subcadena se conserva como primera via para no perder ninguna coincidencia
    que hoy funcione: esto solo ANADE emparejamientos, nunca quita.
    """
    if not blob or not name:
        return False
    if name in blob:
        return True
    toks = [_pantry_singular_key(t) for t in re.split(r"[^a-z0-9]+", name) if t]
    if not toks:
        return False
    disponibles = _pantry_tokens(blob)
    return all(t in disponibles for t in toks)


def _swap_macro_repair_enabled() -> bool:
    """[P1-SWAP-MACRO-REPAIR · 2026-08-09] Kill switch del repair determinista de porciones."""
    return os.environ.get("MEALFIT_SWAP_MACRO_REPAIR", "true").strip().lower() in (
        "1", "true", "yes", "on")


def _repair_swap_candidate_macros(meal_dump: dict, targets: dict, db):
    """[P1-SWAP-MACRO-REPAIR · 2026-08-09] Re-porciona DETERMINISTAMENTE el candidato del swap a
    los targets del slot antes de quemar un retry LLM. Medido (corr=78a438e0, run 31311796944):
    el swap le pedía al LLM aritmética de porciones multi-restricción y el LLM espiraleaba
    (carbs 110→151→269→332→344→422g contra target 146g) hasta SWAP_LLM_RETRIES_EXHAUSTED —
    8/14 swaps muertos en 73-117s. La generación NUNCA le pide esto al LLM: el motor re-porciona.
    Aquí igual: `_rebalance_day_macros_to_target` sobre [meal] (la MISMA maquinaria de prod,
    escala líneas macro-dominantes, re-cuantiza, deltas honestos, raw lockstep) + truth-up +
    step-sync, y re-valida. La IDENTIDAD del plato (ingredientes) no se toca — solo porciones.
    Muta `meal_dump`. Retorna (passed, drifts, summary) post-repair, o (False, None, None) si
    no hubo palanca (sin líneas dominantes movibles). Fail-safe: excepción → (False, None, None).
    tooltip-anchor: P1-SWAP-MACRO-REPAIR"""
    try:
        if not isinstance(meal_dump, dict) or not isinstance(meal_dump.get("ingredients"), list) \
                or not meal_dump.get("ingredients"):
            return (False, None, None)
        from graph_orchestrator import (
            _rebalance_day_macros_to_target as _mr_reb,
            _truth_up_meal_macros_from_strings as _mr_truthup,
            _sync_recipe_step_quantities as _mr_stepsync,
        )
        from nutrition_calculator import validate_meal_macros_against_targets as _mr_validate
        applied = _mr_reb(
            [meal_dump],
            float(targets.get("carbs") or 0),
            float(targets.get("fats") or 0),
            db,
            target_protein=float(targets.get("protein") or 0),
            tol=0.05,
        )
        if not applied:
            return (False, None, None)
        try:
            _mr_truthup(meal_dump, db)
        except Exception as _exc:
            logger.debug("[P1-SWAP-MACRO-REPAIR] truth-up post-repair no aplicado: %s: %s",
                         type(_exc).__name__, str(_exc)[:160])
        try:
            _mr_stepsync(meal_dump)
        except Exception as _exc:
            logger.debug("[P1-SWAP-MACRO-REPAIR] step-sync post-repair no aplicado: %s: %s",
                         type(_exc).__name__, str(_exc)[:160])
        return _mr_validate(meal_dump, targets)
    except Exception as _mr_exc:
        logger.warning(f"[P1-SWAP-MACRO-REPAIR] no-op: {type(_mr_exc).__name__}: {_mr_exc}")
        return (False, None, None)


# [P1-COUNTRY-SYSTEM-F1 · 2026-08-16] (T3) Feedback de retry del swap por país. Extraídos a
# funciones PURAS (país, ...) -> str — testeables sin invocar el pipeline LLM completo, y
# reusadas por las DOS ramas de `swap_meal` (guard de slot-horario, guard de raw-staple) desde
# la MISMA `_swap_country` derivada una sola vez.
def _swap_slot_feedback_suffix(country: str, meal_type: str, viol: list) -> str:
    """Sufijo de retry del backstop P1-SLOT-APPROPRIATENESS (swap fuera de horario). DO/país
    desconocido ⇒ texto actual EXACTO — ancla «para un dominicano» (byte-identidad). País BETA
    ⇒ el mismo texto con la nacionalidad reenmarcada a `name_es`; la REGLA en sí (arroz/locrio/
    pasta en almuerzo/cena, 'arroz de noche') queda intacta — es el espejo prompt de
    `SLOT_INAPPROPRIATE_FOODS`, territorio de F1-T4 (gates culturales por país), no de esta task.

    tooltip-anchor: _swap_slot_feedback_suffix (test_p1_country_system_f1.py)
    """
    from constants import canonicalize_country, COUNTRY_PROFILES
    canon = canonicalize_country(country)
    quien = (
        "para un dominicano" if canon == "DO"
        else f"en {COUNTRY_PROFILES.get(canon, {}).get('name_es', canon)}"
    )
    return (
        f"\n\n🕒 COHERENCIA DE HORARIO (OBLIGATORIO): el plato anterior no encaja con el horario "
        f"«{meal_type}»: {'; '.join(viol)}. Propón un plato que SÍ corresponda a ese momento "
        f"del día {quien} — el arroz/locrio/pasta van en almuerzo/cena (NUNCA desayuno); "
        f"la cena es ligera (evita 'arroz de noche' y comidas de desayuno). Mantén los macros objetivo."
    )


def _swap_raw_staple_feedback_suffix(country: str, marker: str, reason) -> str:
    """Sufijo de retry del backstop P2-AUDIT-V5-BATCH-RAW-STAPLE-SWAP (staple sin transformar).
    DO/país desconocido ⇒ texto actual EXACTO — ancla «una preparación dominicana REAL» (byte-
    identidad). País BETA ⇒ se preserva el REQUISITO (transformar el staple), solo se re-ancla
    la nacionalidad y las técnicas de ejemplo pasan a vocabulario genérico — mismo tratamiento
    que F1-T2 le dio a la regla 19 del day-gen (regla íntegra, ejemplos internacionales).

    tooltip-anchor: _swap_raw_staple_feedback_suffix (test_p1_country_system_f1.py)
    """
    from constants import canonicalize_country, COUNTRY_PROFILES
    canon = canonicalize_country(country)
    if canon == "DO":
        prep = "una preparación dominicana REAL"
        tecnicas = "guiso, locrio, revoltillo, arepitas, bollitos, al horno con majado"
    else:
        name_es = COUNTRY_PROFILES.get(canon, {}).get("name_es", canon)
        prep = f"una preparación real de la cocina de {name_es} o internacional"
        tecnicas = "guiso, salteado, revoltillo, panqueque/tortita, croqueta, al horno"
    return (
        f"\n\n{marker} (OBLIGATORIO): el plato anterior es un staple sin transformar "
        f"({str(reason)[:80]}). Conviértelo en {prep} — "
        f"{tecnicas} — manteniendo los "
        "macros objetivo y los mismos ingredientes base."
    )


def swap_meal(form_data: dict, surface: str = "individual"):
    """Sustituye una comida por otra que cumpla los targets del slot.

    Args:
        form_data: contexto del swap (comida rechazada, targets, dieta, user_id…).
        surface: quién pide el swap — ``"individual"`` (``/swap-meal``, UNA llamada) o
            ``"day"`` (``/regenerate-day``, bucle EN SERIE de 4-5). Solo decide el
            ``reasoning_effort`` cuando el modelo es de OpenAI; ver
            ``_swap_reasoning_effort``. El default deja intactos a los callers que no lo
            pasan, que son todos los individuales.
    """
    rejected_meal = form_data.get("rejected_meal", "")
    meal_type = form_data.get("meal_type", "Comida")
    target_calories = form_data.get("target_calories", 0)
    diet_type = form_data.get("diet_type", "balanced")
    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16] (T3) País del usuario — ÚNICA derivación por swap;
    # ambos guards de retry (slot-horario, raw-staple) la reutilizan vía closure de
    # `invoke_with_retry`. country_for_form_data es la ÚNICA puerta (T1); knob apagado ⇒ 'DO'.
    from constants import country_for_form_data
    _swap_country = country_for_form_data(form_data)

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
                # [P2-SWAP-NUM-MEALS · 2026-07-29] (audit solver+seeder v4) `_num8` caía SIEMPRE a 4:
                # ni el cliente ni el enriquecimiento server-side aportan `num_meals`/`mealsPerDay`.
                # Latente en la cohorte de 4 comidas, dispara al 100% en la clínica de 3/5/6.
                # Caso post-bariátrico: `decide_meals_per_day` devuelve 6 y la 'Merienda Nocturna'
                # vale 0.08 del día; con `_num8=4` el matcher hace match de 'merienda' dentro de
                # 'merienda nocturna' y devuelve la cuota de 0.15 → el plato nuevo se re-escala a
                # 300 kcal en vez de 160 (×1.875) dentro de un pouch de 150-200 mL.
                # Fallback: derivar del perfil antes del literal 4 — nunca es peor que hoy.
                _num8 = None
                try:
                    _num8 = int(form_data.get("num_meals") or form_data.get("mealsPerDay") or 0) or None
                except (TypeError, ValueError):
                    _num8 = None
                # [P3-SWAP-NUMMEALS-SAMEDAY · 2026-07-30] (audit solver+seeder v5) Antes de
                # derivar del PERFIL, mirar la estructura REAL del día que se está editando: el
                # mismo `form_data` ya la trae (el caller puebla `same_day_other_meals` para la
                # variedad intra-día). El perfil puede mentir sobre el plan vivo por tres vías:
                # (a) el usuario eligió num_meals explícito al generar y el plan se enforzó a ese
                # conteo, pero el form del swap no lo trae; (b) la regla de alto gasto depende del
                # kcal, que este callsite no pasa; (c) el perfil cambió DESPUÉS de generar el plan.
                # En los tres casos `allocate_macros_per_slot` repartiría con N distinto del real.
                if _num8 is None:
                    _num8 = _num_meals_from_same_day(form_data)
                    if _num8:
                        logger.info(f"🍽️ [P3-SWAP-NUMMEALS-SAMEDAY] num_meals={_num8} derivado del "
                                    f"DÍA que se edita (ground truth) en vez del perfil.")
                if _num8 is None and SWAP_NUM_MEALS_FROM_PLAN:
                    try:
                        from nutrition_calculator import decide_meals_per_day as _dmpd8
                        _num8 = int((_dmpd8(form_data) or {}).get("num_meals") or 0) or None
                        if _num8:
                            logger.info(f"🍽️ [P2-SWAP-NUM-MEALS] num_meals derivado del perfil: {_num8} "
                                        f"(el literal 4 daba la cuota del slot equivocado).")
                    except Exception:
                        _num8 = None
                _num8 = _num8 or 4
                _slots8 = _alloc8(_daily8, _num8)
                _slot_key8 = _resolve_swap_slot_key(meal_type, _slots8)
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
        # [P1-SAME-DAY-FORMULA-REPEAT · 2026-08-02] Si el plato nuevo reutiliza la MISMA base
        # (avena/arroz/yuca/plátano/pan) de otra comida de hoy, NO basta con cambiar la fruta o
        # guarnición — el swap debe entregar un FORMATO Y PERFIL distinto (cremoso/bowl ↔
        # horneado/arepitas/panqueques ↔ batido; dulce ↔ salado). Ejemplo prohibido: "Avena
        # Cremosa con mango" reemplazando un plato cuando la otra comida de hoy ya es "Bowl
        # Cremoso de Avena Tostada con granola y canela" — misma fórmula, solo cambió la fruta.
        context_extras += (
            "\n    - ⛔ NO CLONES LA FÓRMULA: si vas a reutilizar la MISMA base de carbohidrato "
            "(avena/arroz/yuca/plátano/pan) que otra comida de HOY, el plato nuevo debe cambiar de "
            "FORMATO (cremoso/bowl ↔ horneado/arepitas/panqueques ↔ batido) Y de PERFIL (dulce ↔ "
            "salado, caliente ↔ frío) — no alcanza con cambiar solo la fruta o el fruto seco."
        )

    # [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-F) variedad CROSS-DAY (preferencia suave): el swap era
    # ciego a los otros días — podía proponer el mismo plato que el usuario ya come ese slot 3 veces
    # esta semana. Soft (nunca pelea con la Nevera): mismo espíritu del same-day de arriba.
    _cross_day_names = form_data.get("cross_day_meal_names") or []
    if _cross_day_names:
        # [P1-SWAP-HISTORY-VARIETY · 2026-07-12] slice 8→12 (el helper ahora
        # trae historial de últimos 3 planes + diario, cap 12).
        context_extras += (
            f"\n    - 📅 VARIEDAD (preferencia FUERTE): este horario ya tuvo recientemente (plan actual, "
            f"planes anteriores y lo que el usuario registró comer): "
            f"{', '.join(str(n) for n in _cross_day_names[:12])}. Propón un plato CLARAMENTE DISTINTO a esos "
            f"(otra base, otra técnica), si los ingredientes disponibles lo permiten."
        )
    # [P2-AUDIT-V6-BATCH · 2026-07-03] (P2-F) inspiración de la biblioteca curada también en swap
    # (antes solo el day-gen de form-gen la recibía): 2 plantillas del slot, determinista, soft.
    try:
        from dish_library import build_swap_inspiration_context as _bsi_swap
        _insp_swap = _bsi_swap(
            form_data.get("meal_type") or "",
            seed=len(str(form_data.get("rejected_meal") or "")) + len(_cross_day_names) + 1,
            avoid_names=list(_cross_day_names) + [form_data.get("rejected_meal") or ""],
            # [P3-SWAP-INSPIRATION-DIET · 2026-07-31] (audit v6 · F26) La inspiración se ofrecía sin
            # mirar dieta ni alergias: proponía "Res guisada" a un vegetariano, el backstop lo
            # rechazaba y el swap quemaba reintentos. Los datos ya estaban en scope.
            diet_type=form_data.get("dietType") or form_data.get("diet"),
            allergies=form_data.get("allergies") or [],
        )
        if _insp_swap:
            context_extras += _insp_swap
    except Exception as _insp_e:
        logger.debug(f"[P2-AUDIT-V6-BATCH] (P2-F) inspiración swap no-op: {_insp_e}")

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

    # [P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] "Colocación consciente de sodio" — decisión del owner:
    # el sistema que prescribe la lista/nevera debe ELEGIR mejor el pareo del día en vez de prohibir
    # alimentos. Caso real: regenerar la cena de un día que YA llevaba ricotta armó "Berenjenas con
    # Camarones" → día en 2140/2000mg (7% sobre techo), banner "Menor" DESCUBIERTO tras el hecho. Antes
    # de generar la alternativa calculamos cuánto sodio le queda al DÍA; después de generarla, si el
    # candidato se pasa, UN reintento con directiva explícita — nunca un hard-gate (evidencia medida:
    # gatear el queso same-day quemó 3 reintentos/plan). Knob de rollback sin redeploy (default ON).
    _sodium_aware_on = os.environ.get(
        "MEALFIT_SODIUM_AWARE_SWAP", "true"
    ).strip().lower() in ("1", "true", "yes", "on")
    # Techo — MISMA fuente que el banner/panel (`micronutrients.dri_targets`, NO un literal nuevo).
    _sodium_ceiling_mg = None
    if _sodium_aware_on:
        try:
            from graph_orchestrator import _sodium_day_ceiling_mg_for_banner as _sod_ceil_fn
            _sodium_ceiling_mg = _sod_ceil_fn(form_data)
        except Exception as _sod_ceil_e:
            logger.debug(f"[P1-SODIUM-AWARE-PLACEMENT] techo no-op: {type(_sod_ceil_e).__name__}: {_sod_ceil_e}")
            _sodium_ceiling_mg = 2000.0
    # Sodio del RESTO del día (las OTRAS comidas, sin el plato a cambiar). None = sin contexto de día
    # (guest / sin plan / lookup falló) → el bloque de abajo skipea sodium-aware por completo (fail-open,
    # jamás un literal inventado). regen-day (routers/plans.py) pasa `sodium_resto_override_mg` EN VIVO
    # (ve los platos ya regenerados en ESTA request, que la BD todavía no tiene) — ese override GANA
    # sobre el fallback DB-based de abajo (mismo patrón de precedencia que P2-REGEN-DAY-PANTRY-OVERRIDE).
    _sodium_resto_mg = None
    if _sodium_aware_on:
        try:
            _sod_override = form_data.get("sodium_resto_override_mg")
            if _sod_override is not None:
                _sodium_resto_mg = float(_sod_override)
        except (TypeError, ValueError):
            _sodium_resto_mg = None

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

    # [P1-PANTRY-STRICT-CONSENT · 2026-08-02] Knob + señal de qué universo terminó
    # validando el guard — `_used_real_pantry_universe=True` desactiva los DOS fallbacks
    # legacy plan-derived de abajo (frontend/aggregated_shopping_list): una Nevera real
    # vacía debe levantar SWAP_STRICT_PANTRY_NO_INVENTORY honesto, no maquillarse con
    # ingredientes del plan (el leak exacto que este fix cierra — ver docstring arriba).
    _pantry_strict_updates_on = _pantry_strict_updates_enabled()
    _used_real_pantry_universe = False

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
                elif _pantry_strict_updates_on:
                    # [P1-PANTRY-STRICT-CONSENT] Universo = Nevera FÍSICA real (`user_inventory`),
                    # NO el plan. `/swap-meal` y `/fix-sodium-day` no setean `pantry_override` — sin
                    # esta rama caían aquí ("Intento Primario") a `get_realtime_pantry`, el mismo
                    # universo plan-derived que dejó pasar la yuca (75g de un día ya archivado, jamás
                    # en `user_inventory`) sin preguntar.
                    clean_ingredients = _swap_real_pantry_ledger_lines(user_id)
                    _used_real_pantry_universe = True
                else:
                    # [P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] Antes: sin num_days/multiplier,
                    # la nevera-virtual caía al techo default (person_weeks=1.0) sin importar
                    # household ni duración reales del plan. Derivamos ambos aquí.
                    _vp_num_days, _vp_multiplier = _virtual_pantry_num_days_and_multiplier(plan_record["plan_data"])
                    clean_ingredients = get_realtime_pantry(
                        plan_record["plan_data"], consumed_ingredients,
                        num_days=_vp_num_days, multiplier=_vp_multiplier,
                    )

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

                # [P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Fallback DB-based del sodio del resto del día
                # — cubre el swap standalone (`/swap-meal`), donde el plan en BD SÍ refleja el estado real
                # del día (no hay loop concurrente mutándolo antes de persistir, a diferencia de regen-day,
                # que manda su propio `sodium_resto_override_mg` en vivo y por eso este bloque se skipea
                # arriba cuando ya hay un override). Solo corre si aún no tenemos `_sodium_resto_mg`.
                if _sodium_aware_on and _sodium_resto_mg is None:
                    try:
                        from graph_orchestrator import _meal_sodium_mg as _sod_mm_swap
                        from nutrition_db import IngredientNutritionDB as _SodDBSwap
                        _sod_db_swap = _SodDBSwap()
                        _rej_low_sod = strip_accents(str(rejected_meal or "").lower())
                        for _d_sod in (plan_record["plan_data"].get("days") or []):
                            _meals_sod = (_d_sod.get("meals") or []) if isinstance(_d_sod, dict) else []
                            _names_sod = [str(_m.get("name", "")) for _m in _meals_sod if isinstance(_m, dict)]
                            if any(strip_accents(_n.lower()).strip() == _rej_low_sod for _n in _names_sod):
                                _sodium_resto_mg = sum(
                                    _sod_mm_swap(_m, _sod_db_swap) for _m in _meals_sod
                                    if isinstance(_m, dict)
                                    and strip_accents(str(_m.get("name", "")).lower()).strip() != _rej_low_sod
                                )
                                break
                    except Exception as _sod_ctx_e:
                        logger.debug(
                            f"[P1-SODIUM-AWARE-PLACEMENT] contexto de sodio del día no-op: "
                            f"{type(_sod_ctx_e).__name__}: {_sod_ctx_e}"
                        )
        except Exception as e:
            logger.error(f"⚠️ [SWAP_MEAL] Error extrayendo inventario desde BD: {e}")

    # [P1-PANTRY-STRICT-CONSENT · 2026-08-02] Consentimiento explícito: el caller
    # (`agent.py::swap_meal_with_consent` tras un discovery probe + el "sí" del usuario en el
    # modal) puede sumar nombres al universo autorizado ANTES de los fallbacks legacy — se
    # tratan como abundantes (9999g) para que el guard nunca los rechace por cantidad, solo
    # por existencia previa. Gateado por el knob: OFF ⇒ ignorado (comportamiento legacy).
    if _pantry_strict_updates_on:
        _consented_new = form_data.get("allow_new_ingredients")
        if isinstance(_consented_new, list) and _consented_new:
            _consented_lines = [
                f"9999 g de {str(_n).strip()}" for _n in _consented_new
                if isinstance(_n, str) and str(_n).strip()
            ]
            if _consented_lines:
                clean_ingredients = list(clean_ingredients or []) + _consented_lines
                logger.info(f"✅ [P1-PANTRY-STRICT-CONSENT] {len(_consented_lines)} ingrediente(s) consentido(s) sumado(s) al universo: {_consented_new}")

    # Fallback: Usar lista enviada por el front si falló BD o es guest
    # [P1-PANTRY-STRICT-CONSENT] `and not _used_real_pantry_universe`: si YA validamos contra
    # la Nevera real (autenticado + knob ON), una lista vacía es una Nevera vacía de verdad —
    # NO se maquilla con lo que el frontend mandó (que es plan-derived, mismo leak).
    if not clean_ingredients and not _used_real_pantry_universe:
        current_pantry_ingredients = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
        if current_pantry_ingredients and isinstance(current_pantry_ingredients, list) and len(current_pantry_ingredients) > 0:
            from shopping_calculator import aggregate_shopping_list
            # [P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] Este fallback NO tiene num_days/multiplier
            # reales en scope: a diferencia de `chat_with_agent`/`chat_with_agent_stream` (que
            # reciben `current_plan`), `swap_meal(form_data)` no recibe el plan completo — solo
            # `current_pantry_ingredients`, sin duración ni household adjuntos. Los defaults
            # (num_days=None→3.0, multiplier=1.0) preservan el comportamiento previo exacto;
            # documentar la ausencia de contexto es mejor que inventar un `household=1`
            # silencioso que finja una certeza que no existe.
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
        and not _used_real_pantry_universe  # [P1-PANTRY-STRICT-CONSENT] ídem: nevera real vacía != leer el PDF
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
        # [P1-SWAP-PANTRY-ROTATION · 2026-07-10] Variedad RICA desde la Nevera: con 40-60 items
        # disponibles el LLM gravitaba a los mismos 6-8 (avena/guineo/queso una y otra vez —
        # reporte del owner). Computa los items de la despensa que NADIE usa aún (ni este plato,
        # ni las otras comidas de hoy, ni el mismo slot de otros días) y pide priorizarlos. Soft
        # (los gates deterministas de arriba son la red dura). Fail-safe.
        try:
            from constants import strip_accents as _sa_rot
            _used_blob_rot = _sa_rot((
                str(rejected_meal or "") + " "
                + " ".join(str(b) for b in (form_data.get("same_day_other_meal_blobs") or [])) + " "
                + " ".join(str(n) for n in (_cross_day_names or []))
            ).lower())
            _unused_rot = []
            for _pi_rot in clean_ingredients:
                _nm_rot = _extract_clean_name_from_display_string(str(_pi_rot).strip())
                if not _nm_rot or len(_nm_rot) < 4:
                    continue
                _nm_low_rot = _sa_rot(_nm_rot.lower())
                if _nm_low_rot not in _used_blob_rot:
                    _unused_rot.append(_nm_rot)
                if len(_unused_rot) >= 10:
                    break
            if len(_unused_rot) >= 3:
                context_extras += (
                    f"\n    - 🌈 ROTACIÓN DE DESPENSA (variedad rica): estos alimentos de su nevera "
                    f"AÚN NO se usan en el plan: {', '.join(_unused_rot)}. PRIORIZA construir el plato "
                    f"alrededor de 2-3 de ellos (respetando macros y coherencia) en vez de repetir los "
                    f"ingredientes de siempre."
                )
        except Exception as _rot_e:
            logger.debug(f"[P1-SWAP-PANTRY-ROTATION] hint no-op: {type(_rot_e).__name__}: {_rot_e}")
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
        
        # [P1-COUNTRY-SYSTEM-F2 · 2026-08-17 (Task 9, j · T5-parked)] `_swap_country` ya derivado
        # arriba (T3, línea ~1071) por closure — reusado, no re-derivado. Antes este call site
        # SIEMPRE usaba el pool RD (default None de _get_fast_filtered_catalogs) sin importar el
        # país real del usuario beta.
        filtered_p, filtered_c, filtered_v, _ = _get_fast_filtered_catalogs(
            swap_allergies, swap_dislikes, swap_diet, country=_swap_country
        )
        
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
            # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T4)] reusa `_swap_country` (derivado UNA vez al
            # inicio de swap_meal, T3) — DO ⇒ camino byte-idéntico.
            _timing_block = _bmtr(meal_type, _swap_country)
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
        except Exception as _exc:
            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
            # pero sí dejar traza (antes: pass silencioso).
            logger.debug(
                "[P2-SILENT-DEGRADATION] guía nutricional extra no anexada al prompt: %s: %s",
                type(_exc).__name__, str(_exc)[:160])

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

    # [P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Directiva INFORMATIVA (no un hard-gate) del presupuesto
    # de sodio restante del día — SIEMPRE que haya techo Y contexto de día (`_sodium_resto_mg` no-None).
    # "Repartir, no prohibir": el chef decide con datos, el guard post-generación (más abajo, en
    # invoke_with_retry) es el backstop de UN solo reintento si igual se excede.
    _sodium_budget_mg = None
    if _sodium_aware_on and _sodium_ceiling_mg is not None and _sodium_resto_mg is not None:
        _sodium_budget_mg = max(0.0, _sodium_ceiling_mg - _sodium_resto_mg)
        context_extras += (
            f"\n    - 🧂 PRESUPUESTO DE SODIO: el resto del día ya suma ~{_sodium_resto_mg:.0f}mg de un "
            f"techo de {_sodium_ceiling_mg:.0f}mg/día. Presupuesto restante para ESTE plato: "
            f"~{_sodium_budget_mg:.0f}mg. Prefiere una proteína FRESCA no curada; evita combinar quesos "
            f"curados/embutidos/enlatados/camarones si eso va a excederlo. Si la despensa solo da eso, "
            f"úsalo con moderación — no es una prohibición, es una preferencia de colocación."
        )

    # [P1-STAPLE-FOODS · 2026-08-02] Directiva de básicos + modo universo-chico también en el swap
    # (no solo en el day-gen inicial) — un swap es tan capaz de violar "varía por técnica, no por
    # ingrediente" como la generación completa. `staple_foods` llega en `form_data` (payload del
    # cliente + hidratación server-side en `_enrich_clinical_from_profile`, routers/plans.py). Al
    # vivir en `context_extras` (parte de `prompt_text`, la BASE de todos los reintentos vía
    # `_current_prompt[0] = prompt_text + ...` más abajo) la directiva viaja automáticamente a
    # CADA reintento — cierra el pedido "el retry no quema candidatos imposibles" sin lógica extra.
    try:
        from graph_orchestrator import _small_universe_active as _sua_sw, _raw_staple_foods as _rsf_sw
        _staples_sw = _rsf_sw(form_data)[:8]
        if _staples_sw:
            context_extras += (
                f"\n    - 🥘 BÁSICOS DEL USUARIO (puedes proponerlos de nuevo — repetirlos no es un "
                f"fallo): {', '.join(_staples_sw)}. Si el básico ya aparece en OTRA comida de HOY "
                f"(ver 'VARIEDAD DEL DÍA' abajo), cocínalo con una técnica DISTINTA a esa aparición "
                f"(ej. huevo hervido vs huevo revuelto)."
            )
        if _sua_sw(form_data):
            context_extras += (
                f"\n    - 🔎 MODO UNIVERSO-CHICO: la Nevera disponible tiene pocos alimentos "
                f"distintos. Si te quedas sin opciones de ingrediente NUEVO, recombina lo disponible "
                f"con una TÉCNICA distinta (guisado, horneado, a la plancha, en tortitas, licuado) en "
                f"vez de forzar un ingrediente que no está en la despensa. Esto NO afloja macros, "
                f"reglas clínicas ni el cap de sodio — solo la exigencia de variedad por-ingrediente."
            )
    except Exception as _stp_ctx_e:
        logger.debug(f"[P1-STAPLE-FOODS] directiva de swap no aplicada (no bloquea): "
                     f"{type(_stp_ctx_e).__name__}: {_stp_ctx_e}")

    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (FINAL-FIX F1c)] reusa `_swap_country` (derivado UNA vez
    # al inicio de swap_meal, T3) — DO ⇒ SWAP_MEAL_PROMPT_TEMPLATE byte-idéntico.
    prompt_text = build_swap_meal_prompt_template(_swap_country).format(
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
    # [P2-SWAP-COST-INSTRUMENTATION · 2026-07-10] `include_raw=True` → el invoke retorna
    # {"raw": AIMessage, "parsed": MealModel|None, "parsing_error"}: el AIMessage crudo trae
    # `usage_metadata` real → las calls del swap dejan de ser INVISIBLES en `llm_usage_events`
    # (medido 2026-07-10: un regenerate-day quemó ~20 calls con CERO filas de costo; el
    # cost-by-node del admin no veía esta superficie). Se conserva la referencia al LLM base
    # (`_swap_base_llm`) porque el emit helper resuelve el modelo desde `.model`/`.model_name`.
    # Bonus: un fallo de PARSE ya no revienta como excepción de provider (contaba como CB
    # failure) — llega como parsed=None y se convierte en ValueError retryable (guardrail).
    # [P1-SWAP-LUNA · 2026-08-05] `build_chat_llm` en vez de `ChatDeepSeek` fijo.
    #
    # El hardcode era el bug latente: `MEALFIT_CHAT_AGENT_SWAP_MODEL` ya existía y parecía
    # bastar para mover el swap a otro modelo, pero solo cambiaba el NOMBRE — el cliente
    # seguía siendo el de DeepSeek. Ponerle un ID de OpenAI habría mandado cada swap al
    # base_url de DeepSeek con la key equivocada. Mismo defecto que P1-DAYGEN-LUNA-CANARY
    # ya había corregido en `_build_day_llm`, donde el comentario también decía "provider
    # correcto por prefijo" sin que estuviera implementado.
    #
    # ⚠️ La temperatura NO se le pasa a los modelos OpenAI: solo aceptan la de por defecto
    # y LangChain la descarta EN SILENCIO. Pasarla igualmente dejaría en el código una
    # garantía (`temp = 0.3`) que el runtime no cumple, que es peor que no tenerla. Aquí
    # 0.3 era un empujón hacia la variedad, no un contrato de determinismo — y el modelo
    # nuevo da más variedad que el viejo, no menos (flash devolvía salmón 3 de 3).
    # El modelo se resuelve UNA vez y se reusa: lo necesitan el detector de proveedor, el
    # constructor y el gate del circuit breaker de abajo. Antes cada uno llamaba al helper
    # por su cuenta y existía un test (P0-DEEPSEEK-MIGRATION `test_f_...`) solo para vigilar
    # que no divergieran en el `user_id`; con una variable única eso deja de ser posible.
    _swap_model_name = _chat_agent_swap_model_name(_swap_uid)
    _swap_effort_kwargs = (
        {"reasoning_effort": _swap_reasoning_effort(surface)}
        if is_openai_model(_swap_model_name)
        else {"temperature": temp}
    )
    _swap_base_llm = build_chat_llm(
        model=_swap_model_name,
        timeout=_chat_swap_llm_timeout_s(),  # [P0-CHAT-LLM-TIMEOUT · 2026-05-19]
        **_swap_effort_kwargs,
    )
    swap_llm = _swap_base_llm.with_structured_output(MealModel, include_raw=True)

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
    # [P1-SWAP-LUNA] Reusa la MISMA variable que el constructor: el gate y el cliente
    # no pueden divergir ni en modelo ni en user_id porque no hay dos resoluciones.
    _swap_cb_model = _swap_model_name
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
    # execute_modify_single_meal), así que el cambio no afecta otros surfaces.
    # [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-14) Default OFF→ON en código, mismo patrón
    # P1-VERIFIED-ONLY-DEFAULT-ON: el knob corre ON en prod vía .env desde 2026-06-23
    # (decisión D2 del owner "strict-pantry todos los motivos") y el dark-ship OFF-en-código
    # era la regresión silenciosa ".env reseteado ⇒ cravings/weekend vuelven a comprar".
    # Baseline legacy preservado en tests/conftest.py. Rollback sin redeploy:
    # MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS=false.
    _strict_all = os.environ.get("MEALFIT_UPDATE_DISHES_STRICT_ALL_REASONS", "true").strip().lower() in ("1", "true", "yes", "on")
    strict_pantry = True if _strict_all else (swap_reason not in ("cravings", "weekend"))

    # [P1-PANTRY-STRICT-CONSENT · 2026-08-02] Modo de DESCUBRIMIENTO: 1 probe interno que
    # `swap_meal_with_consent()` (abajo, tras `swap_meal` en este mismo módulo) dispara SOLO
    # cuando el intento nevera-strict normal ya falló y el usuario aún no consintió nada. Relaja
    # el guard para ESTA llamada (nunca expuesta al usuario ni persistida) — el candidato
    # resultante se diffea contra el universo autorizado para nombrar QUÉ falta. El flag NUNCA
    # lo manda el cliente en un request normal (el router no lo reenvía); solo lo setea el
    # wrapper interno.
    _pantry_discovery_mode = bool(form_data.get("_pantry_discovery_mode")) and _pantry_strict_updates_enabled()
    if _pantry_discovery_mode:
        strict_pantry = False

    # [P1-SWAP-EMPTY-PANTRY-WAIVER · 2026-08-09 · reubicado P0-SWAP-WAIVER-UNBOUND] Espejo de la
    # exención de GENERACIÓN («nevera vacía desactiva el modo estricto», decisión intencional
    # anclada): con universo VACÍO (guest, o Nevera real sin ítems) strict_pantry garantizaba el
    # fallo — el LLM recibía un universo imposible, agotaba retries y moría en
    # SWAP_STRICT_PANTRY_NO_INVENTORY. Con el waiver se genera del catálogo verificado con TODAS
    # las guardas clínicas activas. El raise honesto sigue vivo para strict con nevera NO vacía.
    # [P0-SWAP-WAIVER-UNBOUND · 2026-08-09] Este bloque vivió ~400 líneas ANTES de que
    # `strict_pantry` naciera (asignación de arriba): la asignación del waiver convirtió el nombre
    # en local para TODA la función → UnboundLocalError → **500 en el 100% de los swaps** desde el
    # deploy (medido: swap ok_pct=0.0, latency p50=0.2s en la corrida 31304538636). DEBE vivir
    # DESPUÉS de la asignación de strict_pantry y del override de discovery, ANTES del primer
    # read (prompt) y del raise honesto. Rollback sin redeploy: MEALFIT_SWAP_EMPTY_PANTRY_WAIVER=false.
    _swap_empty_waiver_on = os.environ.get(
        "MEALFIT_SWAP_EMPTY_PANTRY_WAIVER", "true").strip().lower() in ("1", "true", "yes", "on")
    if strict_pantry and not clean_ingredients and _swap_empty_waiver_on:
        strict_pantry = False
        logger.warning(
            "🧊 [P1-SWAP-EMPTY-PANTRY-WAIVER] Universo de nevera VACÍO (guest o Nevera sin "
            "ítems) → modo estricto desactivado para este swap; se genera del catálogo "
            "verificado con las guardas clínicas activas.")

    # [P2-SWAP-CONSISTENCY · 2026-05-22] Tolerancia de ingredientes externos
    # cuando el user pidió un antojo / plato festivo: hard-pantry colisionaba
    # con "indulgente" / "premium" (modal opts "Tengo un antojo" / "Fin de
    # semana especial"). Permitimos hasta N "unauthorized" sin abortar; el
    # validador suma esto a su check estructural. Knob
    # `MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED` (default 2, clamp [0, 5]).
    # cravings/weekend: usa el knob. Resto: 0 (legacy strict). Tooltip-anchor:
    # P2-SWAP-CONSISTENCY-EXTERNAL.
    if _pantry_discovery_mode:
        # [P1-PANTRY-STRICT-CONSENT] el probe DEBE poder proponer ingredientes fuera del
        # universo autorizado — es exactamente lo que queremos observar para nombrarlos.
        _external_tolerance = 999
    elif swap_reason in ("cravings", "weekend") and not _strict_all:
        try:
            _external_tolerance = int(os.environ.get("MEALFIT_SWAP_EXTERNAL_INGREDIENTS_ALLOWED", "2"))
        except (TypeError, ValueError):
            _external_tolerance = 2
        _external_tolerance = max(0, min(5, _external_tolerance))
    else:
        # [P4-UPDATE-DISHES-STRICT-ALL] strict-all → cero ingredientes externos para TODOS.
        _external_tolerance = 0

    # [P1-SWAP-CUMULATIVE-BAN · 2026-07-12] PROHIBICIÓN dura UPFRONT: los LABELS de proteína
    # ya usados hoy (derivados con el MISMO SSOT del gate — asimetría prompt↔gate = intentos
    # quemados) se enumeran como prohibidos en el prompt BASE cuando el swap NO es
    # pantry-strict (con despensa estricta manda la preferencia suave: no pelear con lo
    # comprado; el gate ya tiene su fallback anti-slot-imposible). Vivo (corr=382cf533,
    # swap 'menos tiempo'): la 'preferencia' con nombres de platos no bastó — 3 intentos
    # quemados proponiendo huevo/pollo ya usados (la MISMA tortilla dos veces).
    _banned_sd_labels = set()
    try:
        _sd_blobs_up = form_data.get("same_day_other_meal_blobs") or []
        if _sd_blobs_up:
            from graph_orchestrator import _protein_gate_labels_in_text as _pglt_up
            _used_up = set()
            for _b_up in _sd_blobs_up:
                _used_up |= _pglt_up(str(_b_up))
            if _used_up:
                # [P1-SWAP-CUMULATIVE-BAN · 2026-07-12 v2] el ban aplica TAMBIÉN en pantry-strict
                # (la mayoría de los swaps: todo reason salvo cravings/weekend es estricto — la v1
                # con `not strict_pantry` lo saltaba y el Chef propuso 3 tortillas seguidas,
                # corr=32edc94f). En estricto lleva cláusula de escape espejo del fallback
                # anti-imposible del gate. Para 'huevo' se nombran los PLATOS típicos: el modelo
                # tunelizado en 'rápido' ignoraba el label pero entiende "nada de tortillas".
                _ban_names = []
                for _lb_up in sorted(_used_up):
                    if _lb_up == "huevo":
                        _ban_names.append("huevo (nada de tortillas, revoltillos u omelets)")
                    else:
                        _ban_names.append(_lb_up)
                _escape_up = (
                    " Si NINGUNA otra proteína de la despensa disponible alcanza para el plato, "
                    "elige la menos repetida y hazlo notar en la descripción."
                    if strict_pantry else ""
                )
                prompt_text += (
                    f"\n\n🚫 PROTEÍNAS PROHIBIDAS HOY (ya usadas en otras comidas del día): "
                    f"{', '.join(_ban_names)}. NO las uses como proteína principal NI como "
                    f"ingrediente del plato nuevo — el validador RECHAZARÁ el plato si aparecen."
                    f"{_escape_up}"
                )
    except Exception as _ub_e:
        logger.debug(f"[P1-SWAP-CUMULATIVE-BAN] upfront ban no-op: {type(_ub_e).__name__}: {_ub_e}")

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

    # [P1-SODIUM-AWARE-PLACEMENT · 2026-08-02 · fix-review] SSOT del tope de intentos — el guard de
    # sodio (más abajo) necesita saber si está corriendo en el ÚLTIMO intento disponible del
    # presupuesto COMPARTIDO para NUNCA lanzar ahí (finding #2 del review adversarial: un candidato
    # que llega al intento 3 porque pantry/macros ya rechazaron 1-2 podía, pre-fix, hacer que el
    # `raise ValueError` de sodio se propagara vía `reraise=True` → SWAP_LLM_RETRIES_EXHAUSTED → 422,
    # contradiciendo "jamás falla el swap por sodio"). Constante única para que el decorator de abajo
    # y el guard NUNCA diverjan (antes el tope vivía como un literal `3` suelto en el decorator).
    _SWAP_MAX_LLM_ATTEMPTS = 3

    # Invocar LLM con reintentos automáticos (tenacity)
    @retry(
        stop=stop_after_attempt(_SWAP_MAX_LLM_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"🔁 [SWAP RETRY] attempt={retry_state.attempt_number} | "
            f"reason=guardrail_rejection | meal_type={meal_type}"
        )
    )
    def invoke_with_retry():
        _t0_inv = time.time()
        _res_env = swap_llm.invoke(_current_prompt[0])
        # [P2-SWAP-COST-INSTRUMENTATION · 2026-07-10] include_raw=True → envelope
        # {"raw", "parsed", "parsing_error"}. Emitimos usage del AIMessage crudo y
        # desempacamos el parsed para que el resto del flujo quede intacto.
        if isinstance(_res_env, dict) and ("parsed" in _res_env or "raw" in _res_env):
            try:
                from graph_orchestrator import _emit_llm_usage_event_best_effort as _emit_swap_usage
                _emit_swap_usage(llm=_swap_base_llm, result=_res_env.get("raw"),
                                 duration_s=time.time() - _t0_inv, node="swap_meal")
            except Exception as _emit_exc:
                logger.debug(f"[P2-SWAP-COST-INSTRUMENTATION] emit no-op: {type(_emit_exc).__name__}: {_emit_exc}")
            res = _res_env.get("parsed")
            if res is None:
                _perr = _res_env.get("parsing_error")
                raise ValueError(
                    f"SWAP_PARSE_ERROR: el LLM no devolvió el schema MealModel válido "
                    f"({type(_perr).__name__ if _perr is not None else 'parsed=None'}). Reintenta."
                )
        else:
            res = _res_env

        # Validación post-generación (guardrail determinista)
        if hasattr(res, "ingredients"):
            ingreds = getattr(res, "ingredients")
        elif isinstance(res, dict) and "ingredients" in res:
            ingreds = res["ingredients"]
        else:
            ingreds = []

        # [P1-SWAP-BASE-REPEAT-GATE · 2026-07-10] Mode collapse del swap: con reason
        # variety/similar/dislike el LLM devolvía el MISMO plato-base con variación cosmética
        # ("Panqueques de Avena y Guineo con Cottage" → "...con Yogur" → "...Maduro con Yogur",
        # 3 regens seguidos del owner con screenshots). El prompt solo veta el nombre EXACTO;
        # este gate veta el PLATO-BASE canónico (SSOT `_head_dish_base_token`, el mismo del gate
        # de variedad cross-día) → retry con directiva explícita de cambiar la BASE. Solo aplica
        # cuando el usuario pidió algo DISTINTO; 'time'/'cravings'/'weekend' pueden legítimamente
        # conservar la base. Knob MEALFIT_SWAP_BASE_REPEAT_GATE default ON. Fail-safe.
        # tooltip-anchor: P1-SWAP-BASE-REPEAT-GATE
        if (
            swap_reason in ("variety", "similar", "dislike")
            and os.environ.get("MEALFIT_SWAP_BASE_REPEAT_GATE", "true").strip().lower() in ("1", "true", "yes", "on")
        ):
            try:
                from graph_orchestrator import _head_dish_base_token as _hbt_br
                from constants import strip_accents as _sa_br
                _new_name_br = getattr(res, "name", None) if not isinstance(res, dict) else res.get("name")
                _cur_base_br = _hbt_br(_sa_br(str(rejected_meal or "").lower()))
                _new_base_br = _hbt_br(_sa_br(str(_new_name_br or "").lower()))
                # [P1-SWAP-CROSSDAY-BASE-GATE · 2026-07-10] el gate solo comparaba contra el plato
                # REEMPLAZADO → el swap podía proponer la base que otro día ya usa en el mismo slot
                # ("Avena Cremosa" en Día 1 Y Día 3, screenshot del owner). Bases de los otros días
                # (cross_day_meal_names, mismo slot) también vetadas para reasons de variedad.
                _cross_bases_br = set()
                for _cdn_br in (_cross_day_names or []):
                    _cb_br = _hbt_br(_sa_br(str(_cdn_br or "").lower()))
                    if _cb_br:
                        _cross_bases_br.add(_cb_br)
                _base_clash = _new_base_br and (
                    (_cur_base_br and _new_base_br == _cur_base_br) or _new_base_br in _cross_bases_br
                )
                if _base_clash:
                    _why_br = ("actual" if (_cur_base_br and _new_base_br == _cur_base_br) else "otros días del plan")
                    logger.warning(
                        f"🔁 [P1-SWAP-BASE-REPEAT-GATE] plato-base repetido ('{_new_base_br}', vs {_why_br}) | "
                        f"actual={str(rejected_meal)[:40]!r} propuesto={str(_new_name_br)[:40]!r} | meal_type={meal_type}"
                    )
                    _current_prompt[0] = prompt_text + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\nPropusiste un plato "
                        f"base '{_new_base_br}' que el usuario YA come ({_why_br}). CAMBIA LA BASE del "
                        f"plato por completo (otra preparación: revoltillo, avena cocida, tostadas, bowl, "
                        f"arepitas, guiso, al horno…), no solo los acompañantes."
                    )
                    raise ValueError(
                        f"SWAP_SAME_BASE: el plato propuesto repite el plato-base '{_new_base_br}' ({_why_br})."
                    )
            except ValueError:
                raise
            except Exception as _br_exc:
                logger.debug(f"[P1-SWAP-BASE-REPEAT-GATE] no-op: {type(_br_exc).__name__}: {_br_exc}")

        # [P1-SWAP-SAMEDAY-PROTEIN-GATE · 2026-07-10] Gate DETERMINISTA de proteína same-day:
        # el hint soft (P1-SWAP-SAME-DAY-VARIETY) no bastaba — el plan vivo del owner acumuló
        # 'huevo' en 2 comidas del Día 1 Y del Día 2 vía swaps (estado que el reviewer de
        # generación RECHAZARÍA). Mismo SSOT del detector oficial
        # (`_protein_gate_labels_in_text`: labels + aliases + word-boundary sobre
        # nombre+ingredientes). Aplica a TODO reason (repetir proteína el mismo día fatiga
        # siempre); el caller de regen-day ya trae el estado ACTUAL del día en
        # `same_day_other_meal_blobs` y su fallback sin-exclusiones lo retira para no dejar
        # slots imposibles. Knob MEALFIT_SWAP_SAMEDAY_PROTEIN_GATE default ON. Fail-safe.
        # tooltip-anchor: P1-SWAP-SAMEDAY-PROTEIN-GATE
        _sd_blobs_gate = form_data.get("same_day_other_meal_blobs") or []
        if (
            _sd_blobs_gate
            and os.environ.get("MEALFIT_SWAP_SAMEDAY_PROTEIN_GATE", "true").strip().lower() in ("1", "true", "yes", "on")
        ):
            try:
                from graph_orchestrator import _protein_gate_labels_in_text as _pglt_sd
                _cand_name_sd = getattr(res, "name", None) if not isinstance(res, dict) else res.get("name")
                _cand_blob_sd = str(_cand_name_sd or "") + " " + " ".join(str(i) for i in (ingreds or []))
                _cand_lbls_sd = _pglt_sd(_cand_blob_sd)
                _used_lbls_sd = set()
                for _b_sd in _sd_blobs_gate:
                    _used_lbls_sd |= _pglt_sd(str(_b_sd))
                _clash_sd = _cand_lbls_sd & _used_lbls_sd
                # [P1-STAPLE-FOODS · 2026-08-02] Decisión B del owner (espejo del gate del revisor
                # en graph_orchestrator.build_variety_report): un básico declarado por el usuario
                # (`form_data['staple_foods']`) puede repetirse el mismo día si la TÉCNICA del
                # candidato difiere de TODAS las apariciones previas de ese label. Solo se exime si
                # TODOS los labels en conflicto son básicos (una mezcla básico+no-básico sigue
                # rechazando por el no-básico) Y la técnica se pudo determinar en TODAS las
                # comidas involucradas — igual de conservador que el gate del revisor: cualquier
                # duda mantiene el rechazo. tooltip-anchor: P1-STAPLE-FOODS
                if _clash_sd:
                    try:
                        from graph_orchestrator import (
                            _user_staple_labels as _usl_sd,
                            _technique_signature_from_text as _tsft_sd,
                        )
                        from constants import strip_accents as _sa_stp
                        _user_staples_sd = _usl_sd(form_data)
                        if _user_staples_sd and _clash_sd <= _user_staples_sd:
                            _cand_recipe_sd = getattr(res, "recipe", None) if not isinstance(res, dict) else res.get("recipe")
                            _cand_tech_blob_sd = _cand_blob_sd
                            if isinstance(_cand_recipe_sd, list) and _cand_recipe_sd:
                                _cand_tech_blob_sd += " " + " ".join(str(s) for s in _cand_recipe_sd[:2])
                            _cand_sig_sd = _tsft_sd(_cand_tech_blob_sd, _sa_stp)
                            _all_distinct_sd = bool(_cand_sig_sd)
                            if _all_distinct_sd:
                                for _b_sd2 in _sd_blobs_gate:
                                    _other_sig_sd = _tsft_sd(str(_b_sd2), _sa_stp)
                                    if not _other_sig_sd or _other_sig_sd == _cand_sig_sd:
                                        _all_distinct_sd = False
                                        break
                            if _all_distinct_sd:
                                logger.info(
                                    f"🍳 [P1-STAPLE-FOODS] básico(s) {sorted(_clash_sd)} exentos del "
                                    f"gate same-day-protein en swap — técnica '{_cand_sig_sd}' distinta "
                                    f"de las demás comidas del día | meal_type={meal_type}"
                                )
                                _clash_sd = set()
                    except Exception as _stp_exc:
                        logger.debug(f"[P1-STAPLE-FOODS] exención no aplicada en swap (no bloquea): "
                                     f"{type(_stp_exc).__name__}: {_stp_exc}")
                if _clash_sd:
                    _clash_txt = ", ".join(sorted(_clash_sd))
                    logger.warning(
                        f"🍗 [P1-SWAP-SAMEDAY-PROTEIN-GATE] candidato repite proteína del día "
                        f"({_clash_txt}) | propuesto={str(_cand_name_sd)[:40]!r} | meal_type={meal_type}"
                    )
                    # [P1-SWAP-CUMULATIVE-BAN · 2026-07-12] el set de bans se ACUMULA entre
                    # intentos — antes la directiva era `prompt_base + clash ACTUAL` (reemplazo):
                    # el ban de huevo del intento 1 se perdía al banear pollo en el 2 y el 3
                    # volvía al huevo (corr=382cf533: la misma tortilla dos veces).
                    _banned_sd_labels |= set(_clash_sd)
                    _ban_txt = ", ".join(sorted(_banned_sd_labels))
                    _current_prompt[0] = prompt_text + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\nTu plato usa {_clash_txt}, "
                        f"pero OTRA comida de HOY ya lo lleva (revisa 'VARIEDAD DEL DÍA'). "
                        f"PROHIBIDO usar: {_ban_txt} — ni en el nombre ni en los ingredientes. "
                        f"Elige una proteína principal DIFERENTE de las disponibles en su despensa."
                    )
                    raise ValueError(
                        f"SWAP_SAMEDAY_PROTEIN: el plato propuesto repite '{_clash_txt}' ya usado "
                        f"en otra comida del mismo día."
                    )
            except ValueError:
                raise
            except Exception as _sd_exc:
                logger.debug(f"[P1-SWAP-SAMEDAY-PROTEIN-GATE] no-op: {type(_sd_exc).__name__}: {_sd_exc}")

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
                # [P1-SWAP-COHERENCE-REPAIR · 2026-07-10] Reparar ANTES de rechazar: el
                # regenerate-day del 2026-07-10 quemó 4 intentos completos por menciones
                # no-listadas ('dorado'/'pepino'/'tostada'/'arroz+guineítos+coliflor') que
                # son reparables determinísticamente añadiendo la línea faltante — el solver
                # pre-guardrail (abajo) re-escala después las porciones al target, así que
                # la qty inicial solo necesita ser plausible. PANTRY-SEGURO: en modo pantry
                # (clean_ingredients no vacío) solo se repara si el alimento ESTÁ en la
                # nevera (si no, se mantiene el reject → retry con hint, jamás inventamos
                # compra). Knob MEALFIT_SWAP_COHERENCE_REPAIR=false desactiva sin redeploy.
                if (not coh_passed and isinstance(coh_divs, dict) and coh_divs
                        and os.environ.get("MEALFIT_SWAP_COHERENCE_REPAIR", "true").strip().lower()
                        in ("1", "true", "yes", "on")):
                    try:
                        from constants import strip_accents as _coh_sa
                        _rep_qty_by_cat = {"proteína": 80, "proteina": 80, "carbohidrato": 60,
                                           "vegetal": 50, "fruta": 50}
                        _pantry_blob = _coh_sa(" ".join(str(x) for x in (clean_ingredients or [])).lower())
                        _rep_lines = []
                        _rep_ok = True
                        for _cf_food, _cf_info in coh_divs.items():
                            _alias = str((_cf_info or {}).get("mentioned_alias") or _cf_food).strip()
                            _alias_norm = _coh_sa(_alias.lower())
                            # [P1-SWAP-PANTRY-PLURAL · 2026-08-05] Por TOKEN, no por
                            # subcadena: "huevos" jamás casaba contra una nevera que
                            # dice "Huevo", y el swap moría en 3 intentos idénticos.
                            if clean_ingredients \
                                    and not pantry_contains_food(_pantry_blob, _alias_norm) \
                                    and not pantry_contains_food(_pantry_blob, _coh_sa(str(_cf_food).lower())):
                                _rep_ok = False  # fuera de nevera en modo pantry → no reparable
                                break
                            _cat = _coh_sa(str((_cf_info or {}).get("category") or "").lower())
                            _rep_lines.append(f"{_rep_qty_by_cat.get(_cat, 30)} g de {_alias}")
                        if _rep_ok and _rep_lines:
                            _cur_ings = list(meal_dump.get("ingredients") or [])
                            _cur_ings.extend(_rep_lines)
                            meal_dump["ingredients"] = _cur_ings
                            _re_passed, _re_divs, _re_summary = _validate_recipe_coh(meal_dump)
                            if _re_passed:
                                if isinstance(res, dict):
                                    res["ingredients"] = _cur_ings
                                elif hasattr(res, "ingredients"):
                                    setattr(res, "ingredients", _cur_ings)
                                if isinstance(meal_dump.get("ingredients_raw"), list):
                                    meal_dump["ingredients_raw"].extend(_rep_lines)
                                logger.info(
                                    f"🩹 [P1-SWAP-COHERENCE-REPAIR] {len(_rep_lines)} línea(s) "
                                    f"añadida(s) determinísticamente ({', '.join(_rep_lines)}) — "
                                    f"intento preservado (el solver re-escala) | meal_type={meal_type}"
                                )
                                coh_passed, coh_divs, coh_summary = _re_passed, _re_divs, _re_summary
                    except Exception as _rep_exc:
                        logger.warning(
                            f"[P1-SWAP-COHERENCE-REPAIR] repair falló (no aborta, cae al reject): "
                            f"{type(_rep_exc).__name__}: {_rep_exc}"
                        )
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

        # [P0-SWAP-DETERMINISTIC-RESCALE · 2026-07-10] Re-escala las PORCIONES del candidato
        # al target del slot ANTES del guardrail — el mismo solver per-ingrediente de la
        # generación (`_apply_macro_solver_to_meal`: factores acotados por línea → corrige
        # RATIO, no solo escala global). Root cause medido (regenerate-day 2026-07-10,
        # corr=8ce66cae): el guardrail pedía al LLM aritmética de ±15% por macro y la
        # rechazaba-y-re-pedía — 6+ candidatos murieron por drift (uno con kcal +1.2% pero
        # proteína +31%); la generación NUNCA le pide eso al LLM, lo resuelve el solver.
        # Tras el solver: re-truth-up (números honestos desde strings) + re-sync de las
        # menciones de cantidad en los pasos (evita "150g" en paso vs "180g" en línea).
        # El guardrail queda como red para lo irreparable. Fail-safe total.
        # Knob MEALFIT_SWAP_DETERMINISTIC_RESCALE=false desactiva sin redeploy.
        # tooltip-anchor: P0-SWAP-DETERMINISTIC-RESCALE
        if os.environ.get("MEALFIT_SWAP_DETERMINISTIC_RESCALE", "true").strip().lower() in ("1", "true", "yes", "on"):
            try:
                _rs_target = {
                    "kcal": float(target_calories or 0), "protein": float(target_protein or 0),
                    "carbs": float(target_carbs or 0), "fats": float(target_fats or 0),
                }
                _rs_meal = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                if _rs_target["kcal"] > 0 and isinstance(_rs_meal.get("ingredients"), list) and _rs_meal["ingredients"]:
                    from graph_orchestrator import (
                        _apply_macro_solver_to_meal as _rs_solver,
                        _truth_up_meal_macros_from_strings as _rs_truthup,
                        _sync_recipe_step_quantities as _rs_stepsync,
                    )
                    if _tu_db_holder[0] is None:
                        from nutrition_db import IngredientNutritionDB as _RSDB
                        _tu_db_holder[0] = _RSDB()
                    _rs_ok = _rs_solver(_rs_meal, _rs_target, _tu_db_holder[0])
                    if not _rs_ok:
                        # [P2-SWAP-SOLVER-FLAGS · 2026-07-30] (audit solver+seeder v5)
                        # `_solver_abstained_coverage` se escribe JUSTO ANTES del `return False`
                        # del solver, y el copy-back vivía dentro del `if` de éxito → era
                        # imposible de conservar por construcción. Sin esto, una comida swapeada
                        # donde el solver se abstiene por cobertura no deja rastro NUNCA, y la
                        # serie que mide esa abstención nace ciega a los swaps.
                        _rs_ab = _rs_meal.get("_solver_abstained_coverage")
                        if _rs_ab is not None:
                            if not isinstance(res, dict) and hasattr(res, "model_dump"):
                                res = res.model_dump()
                            if isinstance(res, dict):
                                res["_solver_abstained_coverage"] = _rs_ab
                    if _rs_ok:
                        try:
                            _rs_truthup(_rs_meal, _tu_db_holder[0])
                        except Exception as _exc:
                            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                            # pero sí dejar traza (antes: pass silencioso).
                            logger.debug(
                                "[P2-SILENT-DEGRADATION] truth-up post-reshuffle no aplicado (nombre puede exagerar): %s: %s",
                                type(_exc).__name__, str(_exc)[:160])
                        try:
                            _rs_stepsync(_rs_meal)
                        except Exception as _exc:
                            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                            # pero sí dejar traza (antes: pass silencioso).
                            logger.debug(
                                "[P2-SILENT-DEGRADATION] step-sync post-reshuffle no aplicado (pasos pueden desalinear): %s: %s",
                                type(_exc).__name__, str(_exc)[:160])
                        # [P2-SOLVER-CONVERGENCE-METRIC · 2026-07-29] (audit solver+seeder v4) El
                        # solver corre sobre `_rs_meal`, que es un `model_dump()` NUEVO: sus 6 flags
                        # de telemetría quedaban ahí y se perdían al copiar de vuelta solo 8 claves.
                        # Resultado: ninguna comida SWAPEADA aparecía jamás en la métrica per-run
                        # `solver_clamp`, y la serie de no-convergencia que se acaba de construir
                        # arrancaría sesgada con un agujero justo en las comidas swapeadas.
                        # [P2-SWAP-SOLVER-FLAGS · 2026-07-30] (audit solver+seeder v5) El copy-back
                        # era ESTRUCTURALMENTE inerte para su propósito: en el path normal `res` es
                        # un `MealModel` de pydantic (structured output) que NO declara ningún
                        # `_solver_*` ni `ingredients_raw`, así que `hasattr(res, "_solver_...")`
                        # era SIEMPRE False y las 6 claves se descartaban en silencio. Las 9
                        # declaradas sí copiaban, y por eso el bloque "funcionaba" a la vista.
                        # Convertimos a dict UNA vez: todo lo que viene después usa el idioma
                        # `res.model_dump() if hasattr(...) else (res if isinstance(res, dict))`,
                        # que trata un dict igual de bien — y de paso los copy-backs hermanos
                        # (protein-closer, fat-topup) dejan de tener la misma rama muerta.
                        if not isinstance(res, dict) and hasattr(res, "model_dump"):
                            res = res.model_dump()
                        # `_solver_failed_macros`/`_solver_infeasible`/`_solver_residuals` nacieron
                        # el MISMO día que este bloque y quedaron fuera de la lista.
                        # [P2-SOLVER-PIN-FROZEN · 2026-08-03] `_solver_frozen_lines` es la TERCERA
                        # tanda que nace después de esta lista. Esta NO se olvida: sin ella, toda
                        # comida SWAPEADA perdería el conteo y la serie nacería con el mismo agujero
                        # que P2-SOLVER-CONVERGENCE-METRIC documentó arriba. Es copy-BACK (preserva
                        # telemetría a través del `model_dump()` del structured output), NO un strip:
                        # las `_solver_*` no se ocultan a la LLM aquí, viven en `plan_data`.
                        for _rk in ("ingredients", "ingredients_raw", "recipe",
                                    "protein", "carbs", "fats", "cals", "macros",
                                    "_solver_clamp_saturated", "_solver_clamp_saturated_hi",
                                    "_solver_clamp_saturated_lo", "_solver_greedy_fallback",
                                    "_solver_not_converged", "_solver_raw_by_food",
                                    "_solver_failed_macros", "_solver_infeasible",
                                    "_solver_residuals", "_solver_frozen_lines"):
                            if _rk in _rs_meal and _rs_meal[_rk] is not None and isinstance(res, dict):
                                res[_rk] = _rs_meal[_rk]
                        logger.info(
                            f"🎯 [P0-SWAP-DETERMINISTIC-RESCALE] porciones re-escaladas al "
                            f"target del slot pre-guardrail | meal_type={meal_type}"
                        )
            except Exception as _rs_exc:
                logger.warning(
                    f"[P0-SWAP-DETERMINISTIC-RESCALE] no-op (no aborta): "
                    f"{type(_rs_exc).__name__}: {_rs_exc}"
                )

        # [P2-SWAP-PROTEIN-CLOSER · 2026-07-12] (pedido del owner: "que sea más preciso") Si tras
        # el solver el candidato sigue MATERIALMENTE bajo el objetivo de proteína del slot (<85%,
        # el mismo umbral del validador de abajo), corre el closer determinista de la GENERACIÓN
        # (`_close_protein_gap_for_meal`: scale-first → candidato allergen-safe día-aware con toda
        # la higiene: bolt cap 180g, sweet-guard, no-dup-cheese, wording SSOT del paso 💪) ANTES
        # de que el validador queme un retry LLM o entregue con el toast "menos preciso" (vivo:
        # moro de camarones 25g vs 38g). Solo cierra DÉFICIT (el exceso lo trata el solver/
        # validador). `day_used_proteins` desde los blobs del gate (SSOT) → jamás reintroduce un
        # repeat same-day. Fail-safe total. Knob MEALFIT_SWAP_PROTEIN_CLOSER=false sin redeploy.
        # tooltip-anchor: P2-SWAP-PROTEIN-CLOSER
        if os.environ.get("MEALFIT_SWAP_PROTEIN_CLOSER", "true").strip().lower() in ("1", "true", "yes", "on"):
            try:
                _pc_target = float(target_protein or 0)
                _pc_meal = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                _pc_cur = float(_pc_meal.get("protein") or 0)
                if (_pc_target > 0 and _pc_cur < _pc_target * 0.85
                        and isinstance(_pc_meal.get("ingredients"), list) and _pc_meal["ingredients"]):
                    from graph_orchestrator import (
                        _close_protein_gap_for_meal as _pc_closer,
                        _safe_high_density_proteins as _pc_pool,
                        _protein_gate_labels_in_text as _pc_labels,
                        _sync_recipe_step_quantities as _pc_stepsync,
                    )
                    if _tu_db_holder[0] is None:
                        from nutrition_db import IngredientNutritionDB as _PCDB
                        _tu_db_holder[0] = _PCDB()
                    _pc_allergies = form_data.get("allergies") or []
                    _pc_used = set()
                    for _pb in (form_data.get("same_day_other_meal_blobs") or []):
                        _pc_used |= _pc_labels(str(_pb))
                    _g_pc = _pc_closer(
                        _pc_meal, _pc_target, _tu_db_holder[0],
                        _pc_pool(_pc_allergies, _tu_db_holder[0]),
                        allergies=_pc_allergies, fill_pct=1.0,
                        slot_cal_target=float(target_calories or 0),
                        enforce_min_threshold=False,
                        day_used_proteins=_pc_used,
                    )
                    if _g_pc > 0:
                        try:
                            _pc_stepsync(_pc_meal)
                        except Exception as _exc:
                            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                            # pero sí dejar traza (antes: pass silencioso).
                            logger.debug(
                                "[P2-SILENT-DEGRADATION] step-sync post-cheapen no aplicado (pasos pueden desalinear): %s: %s",
                                type(_exc).__name__, str(_exc)[:160])
                        for _pk in ("ingredients", "ingredients_raw", "recipe", "name",
                                    "protein", "carbs", "fats", "cals", "macros"):
                            if _pk in _pc_meal and _pc_meal[_pk] is not None:
                                if isinstance(res, dict):
                                    res[_pk] = _pc_meal[_pk]
                                elif hasattr(res, _pk):
                                    setattr(res, _pk, _pc_meal[_pk])
                        logger.info(
                            f"💪 [P2-SWAP-PROTEIN-CLOSER] +{_g_pc}g de proteína determinista en el "
                            f"candidato ({int(_pc_cur)}g → {_pc_meal.get('protein')}g vs target "
                            f"{int(_pc_target)}g) | meal_type={meal_type}"
                        )
            except Exception as _pc_exc:
                logger.warning(
                    f"[P2-SWAP-PROTEIN-CLOSER] no-op (no aborta): {type(_pc_exc).__name__}: {_pc_exc}"
                )

        # [P2-SWAP-FATS-TRIM · 2026-07-12] Espejo del closer para el EXCESO de grasa: el
        # validador quemaba los 3 reintentos LLM por deltas minúsculos (vivo, regen v3 05:28Z:
        # moro con fats=8g vs target 5g → SWAP_LLM_RETRIES_EXHAUSTED → 2 slots conservados →
        # el día sin libertad para cuadrar → band 0.5). El recortador determinista de S1
        # (`_trim_day_fats_to_target`: shrink de fuentes de grasa, portadores de micros
        # protegidos) cierra el delta ANTES del validador. Solo actúa sobre exceso MATERIAL
        # (>115% del target — el mismo umbral del validador). Fail-safe total.
        # Knob MEALFIT_SWAP_FATS_TRIM (ON). tooltip-anchor: P2-SWAP-FATS-TRIM
        if os.environ.get("MEALFIT_SWAP_FATS_TRIM", "true").strip().lower() in ("1", "true", "yes", "on"):
            try:
                _ft_target = float(target_fats or 0)
                _ft_meal = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                _ft_cur = float(_ft_meal.get("fats") or 0)
                if (_ft_target > 0 and _ft_cur > _ft_target * 1.15
                        and isinstance(_ft_meal.get("ingredients"), list) and _ft_meal["ingredients"]):
                    from graph_orchestrator import (
                        _trim_day_fats_to_target as _ft_trim,
                        _truth_up_meal_macros_from_strings as _ft_truthup,
                        _sync_recipe_step_quantities as _ft_stepsync,
                    )
                    if _tu_db_holder[0] is None:
                        from nutrition_db import IngredientNutritionDB as _FTDB
                        _tu_db_holder[0] = _FTDB()
                    if _ft_trim([_ft_meal], _ft_target, _tu_db_holder[0]):
                        try:
                            _ft_truthup(_ft_meal, _tu_db_holder[0])
                        except Exception as _exc:
                            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                            # pero sí dejar traza (antes: pass silencioso).
                            logger.debug(
                                "[P2-SILENT-DEGRADATION] truth-up post-fit no aplicado (nombre puede exagerar): %s: %s",
                                type(_exc).__name__, str(_exc)[:160])
                        try:
                            _ft_stepsync(_ft_meal)
                        except Exception as _exc:
                            # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                            # pero sí dejar traza (antes: pass silencioso).
                            logger.debug(
                                "[P2-SILENT-DEGRADATION] step-sync post-fit no aplicado (pasos pueden desalinear): %s: %s",
                                type(_exc).__name__, str(_exc)[:160])
                        for _fk in ("ingredients", "ingredients_raw", "recipe",
                                    "protein", "carbs", "fats", "cals", "macros"):
                            if _fk in _ft_meal and _ft_meal[_fk] is not None:
                                if isinstance(res, dict):
                                    res[_fk] = _ft_meal[_fk]
                                elif hasattr(res, _fk):
                                    setattr(res, _fk, _ft_meal[_fk])
                        # [P2-SWAP-FATS-TRIM v2] el número post es el HONESTO (truth-up desde
                        # strings tras el trim) — puede SUBIR si el candidato sub-reportaba su
                        # grasa (vivo 05:47Z: reportado 8g, honesto 23g → el validador rechaza
                        # con el dato real; el log previo "recortada 8g → 23g" confundía).
                        logger.info(
                            f"🥑 [P2-SWAP-FATS-TRIM] trim aplicado | reportado_pre={_ft_cur:.0f}g "
                            f"→ honesto_post={_ft_meal.get('fats')}g (target {_ft_target:.0f}g) "
                            f"| meal_type={meal_type}"
                        )
            except Exception as _ft_exc:
                logger.warning(f"[P2-SWAP-FATS-TRIM] no-op (no aborta): {type(_ft_exc).__name__}: {_ft_exc}")

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
                    # [P1-SWAP-MACRO-REPAIR · 2026-08-09] ANTES de quemar un retry LLM:
                    # re-porcionado determinista del candidato (la identidad del plato la
                    # puso el LLM; las porciones las pone el motor — como en generación).
                    if _swap_macro_repair_enabled():
                        _mr_targets = {
                            "cals": target_calories,
                            "protein": target_protein,
                            "carbs": target_carbs,
                            "fats": target_fats,
                        }
                        if _tu_db_holder[0] is None:
                            from nutrition_db import IngredientNutritionDB as _MRDB
                            _tu_db_holder[0] = _MRDB()
                        _mr_passed, _mr_drifts, _mr_summary = _repair_swap_candidate_macros(
                            meal_dump, _mr_targets, _tu_db_holder[0])
                        if _mr_passed:
                            for _fk in ("ingredients", "ingredients_raw", "recipe",
                                        "protein", "carbs", "fats", "cals", "macros"):
                                if _fk in meal_dump and meal_dump[_fk] is not None:
                                    if isinstance(res, dict):
                                        res[_fk] = meal_dump[_fk]
                                    elif hasattr(res, _fk):
                                        setattr(res, _fk, meal_dump[_fk])
                            passed = True
                            logger.info(
                                f"🔧 [P1-SWAP-MACRO-REPAIR] candidato re-porcionado "
                                f"deterministamente a banda (drift original={drifts}) — "
                                f"retry LLM evitado | meal_type={meal_type}")
                    if not passed:
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
                # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T4 fix-round 1)] reusa `_swap_country`
                # (derivado UNA vez al inicio de swap_meal, T3) — DO ⇒ camino byte-idéntico.
                _slot_viol = slot_coherence_backstop_for_meal(_slot_dump, meal_type, _swap_country)
            except Exception:
                _slot_viol = []
            if _slot_viol:
                logger.warning(
                    f"🕒 [P1-SLOT-APPROPRIATENESS] swap fuera de horario | meal_type={meal_type} | viol={_slot_viol}"
                )
                _current_prompt[0] = prompt_text + _swap_slot_feedback_suffix(_swap_country, meal_type, _slot_viol)
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

        # [P2-UPDATE-DISHQUALITY-PRESSURE · 2026-07-02] (audit v4 paridad) El detector per-comida de
        # dish-quality (nombre placeholder / ingredientes 'Proteína magra al gusto' / receta hueca) corría
        # en updates solo como finalizer+telemetría — el swap TIENE retry-loop barato y no lo usaba: un
        # plato placeholder se entregaba sin presión de mejora. Espejo del backstop de slot/clash: feedback
        # al prompt + retry (fail-open, no 422 en strict_pantry-sin-inventario, el ValueError NO cuenta como
        # CB failure por P2-CB-GUARDRAIL-NOT-FAILURE). chat-modify queda advisory (el deseo del usuario
        # manda — mismo criterio que slot). Rollback: MEALFIT_SWAP_DISH_QUALITY_PRESSURE=false.
        # tooltip-anchor: P2-UPDATE-DISHQUALITY-PRESSURE
        if (os.environ.get("MEALFIT_SWAP_DISH_QUALITY_PRESSURE", "true").strip().lower() in ("1", "true", "yes", "on")
                and not (strict_pantry and not clean_ingredients)):
            try:
                from graph_orchestrator import _meal_dish_quality_issue as _mdqi_sw
                _dq_dump = res.model_dump() if hasattr(res, "model_dump") else (res if isinstance(res, dict) else {})
                _dq_low, _dq_reason = _mdqi_sw(_dq_dump)
            except Exception:
                _dq_low, _dq_reason = False, None
            if _dq_low:
                logger.warning(
                    f"🍽️ [P2-UPDATE-DISHQUALITY-PRESSURE] swap con plato placeholder/hueco "
                    f"({str(_dq_reason)[:80]}) | meal_type={meal_type}"
                )
                _current_prompt[0] = prompt_text + (
                    "\n\n🍽️ CALIDAD DE PLATO (OBLIGATORIO): el plato anterior parece un placeholder o viene "
                    f"incompleto ({str(_dq_reason)[:100]}). Entrega un plato REAL y cocinable: nombre específico "
                    "es-DO (no genérico), ingredientes concretos con cantidades, y receta con los 3 pilares "
                    "(Mise en place / El Toque de Fuego con tiempo / Montaje). Mantén los macros objetivo."
                )
                raise ValueError("DISH_QUALITY: " + str(_dq_reason)[:100])

        # [P2-AUDIT-V5-BATCH · 2026-07-02] (GAP-13) Presión anti-raw-staple en swap: form-gen subió
        # el detector a soft-gate ON (P2-RAW-STAPLE-PRESSURE) pero el swap solo corría dish-quality,
        # que POR DISEÑO no ve staples desnudos ("un 'Pollo a la Plancha' real puntuaba alto",
        # comentario del detector). En swap el output es UNA comida (ratio efectivo 1.0) y es la
        # superficie donde el usuario pidió explícitamente cambiar el plato — merece la presión de
        # creatividad. Single-retry vía marker (cosmético: en el reintento se entrega con log
        # advisory, NUNCA fallback); skip pantry-strict (transformar exige ingredientes que quizá
        # no hay); el ValueError NO cuenta como CB failure (P2-CB-GUARDRAIL-NOT-FAILURE).
        # Rollback sin redeploy: MEALFIT_SWAP_RAW_STAPLE_PRESSURE=false. chat-modify queda advisory
        # (deseo del usuario manda — mismo criterio que slot/dish-quality).
        # tooltip-anchor: P2-AUDIT-V5-BATCH-RAW-STAPLE-SWAP
        if (os.environ.get("MEALFIT_SWAP_RAW_STAPLE_PRESSURE", "true").strip().lower() in ("1", "true", "yes", "on")
                and not (strict_pantry and not clean_ingredients)):
            try:
                from graph_orchestrator import _meal_raw_staple_issue as _mrsi_sw
                _rs_dump = res.model_dump() if hasattr(res, "model_dump") else (res if isinstance(res, dict) else {})
                _rs_raw, _rs_reason = _mrsi_sw(_rs_dump)
            except Exception:
                _rs_raw, _rs_reason = False, None
            if _rs_raw:
                _RS_MARKER = "🍳 RETRY PLATO TRANSFORMADO"
                if _RS_MARKER not in str(_current_prompt[0]):
                    _current_prompt[0] = prompt_text + _swap_raw_staple_feedback_suffix(_swap_country, _RS_MARKER, _rs_reason)
                    raise ValueError("RAW_STAPLE: " + str(_rs_reason)[:100])
                logger.info(f"🍳 [P2-AUDIT-V5-BATCH] (GAP-13) swap sigue raw-staple tras el retry — "
                            f"entregado con advisory | meal_type={meal_type}")

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

        # [P1-SODIUM-AWARE-PLACEMENT · 2026-08-02] Backstop determinista: tras todos los
        # ajustes deterministas (solver/closer/trim) y guards de calidad de arriba, si candidato+resto-
        # del-día EXCEDE el techo → UN reintento con directiva explícita (mismo patrón single-retry-con-
        # marker de P1-SWAP-BASE-REPEAT-GATE / P2-UPDATE-SAMEDAY-VARIETY arriba). El reintento de sodio
        # comparte el presupuesto de `_SWAP_MAX_LLM_ATTEMPTS` intentos de tenacity con TODOS los demás
        # guards de esta función — NO tiene su propio @retry — por diseño: es exactamente el mismo
        # modelo ya vigente para pantry/coherencia/macros/clínico/slot/appetibility/dish-quality/raw-
        # staple/base-repeat/sameday-protein.
        #
        # [FINDING #2 · review adversarial 2026-08-02] Presupuesto COMPARTIDO significa que la primera
        # evaluación de sodio puede caer en el ÚLTIMO intento disponible (ej. pantry rechaza el 1 y el
        # 2, sodio recién corre en el 3 — patrón real visto en logs de producción). Pre-fix, el `raise`
        # ahí se propagaba vía `reraise=True` → `SWAP_LLM_RETRIES_EXHAUSTED` → 422 → EL SWAP ENTERO
        # FALLABA por un candidato que, sin este guard, se habría aceptado — contradicción directa con
        # "jamás falla el swap por sodio". Fix: el guard NUNCA lanza si `attempt_number >=
        # _SWAP_MAX_LLM_ATTEMPTS` (no queda a dónde retirarse) — en ese caso ACEPTA directamente
        # (accion=accept_final), igual que cuando el marker ya se usó. `attempt_number` se lee de
        # `invoke_with_retry.statistics` (verificado empíricamente: tenacity 9.x expone el contador
        # AHÍ, no en `invoke_with_retry.retry.statistics` — ese último es un dict compartido vacío;
        # confirmar con `f.statistics` no `f.retry.statistics` si se audita de nuevo). Fail-safe: si la
        # lectura falla, se ASUME que es el último intento (nunca arriesga el `raise`).
        #
        # [FINDING #1 · review adversarial 2026-08-02] Presupuesto IMPOSIBLE: si `_sodium_resto_mg` YA
        # excede el techo ANTES de sumar el candidato (día ya cargado, p.ej. resto=2200 > techo=2000),
        # NINGÚN candidato puede pasar — hasta uno con 0mg de sodio da resto > techo. Lanzar ahí quema
        # 1 de los 3 intentos compartidos con cero posibilidad matemática de éxito, justo en el
        # escenario donde el presupuesto restante es más escaso. Fix: skip TOTAL del post-check en ese
        # caso (la directiva pre-generación ya está clampeada a ~0mg, así que el LLM sigue informado).
        #
        # Si el reintento tampoco baja el sodio (o no hay reintento posible) → se ACEPTA el mejor
        # candidato (el aviso existente `_quality_degraded`/panel cubre el resto) — jamás se falla el
        # swap por sodio (repartir, no prohibir). Knob MEALFIT_SODIUM_AWARE_SWAP (default ON).
        # Telemetría: una línea por decisión, formato parseable
        # `attempt=N/M presupuesto=Xmg candidato_mg=Ymg decision=<retry|accept_final|skip_imposible|
        # accepted_after_retry|within_budget>`. tooltip-anchor: P1-SODIUM-AWARE-PLACEMENT
        if _sodium_aware_on and _sodium_ceiling_mg is not None and _sodium_resto_mg is not None:
            try:
                # [FINDING #1] Presupuesto imposible ANTES de sumar el candidato → skip total, no
                # consume ningún intento del presupuesto compartido.
                if _sodium_resto_mg >= _sodium_ceiling_mg:
                    logger.info(
                        f"🧂 [P1-SODIUM-AWARE-PLACEMENT] attempt=?/{_SWAP_MAX_LLM_ATTEMPTS} "
                        f"presupuesto=0mg candidato_mg=n/a decision=skip_imposible | "
                        f"resto≈{_sodium_resto_mg:.0f}mg ya >= techo {_sodium_ceiling_mg:.0f}mg — "
                        f"ningún candidato puede caber, post-check omitido (directiva pre-gen ya "
                        f"clampeada a ~0mg) | meal_type={meal_type}"
                    )
                else:
                    from graph_orchestrator import _meal_sodium_mg as _sod_meal_fn
                    if _tu_db_holder[0] is None:
                        from nutrition_db import IngredientNutritionDB as _SodMealDB
                        _tu_db_holder[0] = _SodMealDB()
                    _cand_dump_sod = res.model_dump() if hasattr(res, "model_dump") else (
                        res if isinstance(res, dict) else {}
                    )
                    _cand_sodium_mg = _sod_meal_fn(_cand_dump_sod, _tu_db_holder[0])
                    _day_total_sod = _sodium_resto_mg + _cand_sodium_mg
                    _sod_budget_left = max(0.0, _sodium_ceiling_mg - _sodium_resto_mg)

                    # [FINDING #2] Detectar si este es el ÚLTIMO intento del presupuesto compartido de
                    # tenacity — si lo es, el raise de sodio NUNCA es legal (no hay dónde aterrizar el
                    # retry). Fail-safe: cualquier fallo en la lectura ASUME que es el último (nunca
                    # arriesga el raise).
                    try:
                        _attempt_n = int((invoke_with_retry.statistics or {}).get("attempt_number") or _SWAP_MAX_LLM_ATTEMPTS)
                    except Exception:
                        _attempt_n = _SWAP_MAX_LLM_ATTEMPTS
                    _is_final_attempt = _attempt_n >= _SWAP_MAX_LLM_ATTEMPTS

                    if _day_total_sod > _sodium_ceiling_mg:
                        _SOD_MARKER = "🧂 RETRY PRESUPUESTO DE SODIO"
                        if _is_final_attempt:
                            logger.info(
                                f"🧂 [P1-SODIUM-AWARE-PLACEMENT] attempt={_attempt_n}/{_SWAP_MAX_LLM_ATTEMPTS} "
                                f"presupuesto={_sod_budget_left:.0f}mg candidato_mg={_cand_sodium_mg:.0f} "
                                f"decision=accept_final | excede el presupuesto pero es el ÚLTIMO intento "
                                f"disponible — ACEPTADO sin reintentar (jamás se falla el swap por sodio; "
                                f"el aviso existente cubre) | meal_type={meal_type}"
                            )
                        elif _SOD_MARKER not in str(_current_prompt[0]):
                            logger.warning(
                                f"🧂 [P1-SODIUM-AWARE-PLACEMENT] attempt={_attempt_n}/{_SWAP_MAX_LLM_ATTEMPTS} "
                                f"presupuesto={_sod_budget_left:.0f}mg candidato_mg={_cand_sodium_mg:.0f} "
                                f"decision=retry | candidato+resto={_day_total_sod:.0f}mg > techo "
                                f"{_sodium_ceiling_mg:.0f}mg | meal_type={meal_type}"
                            )
                            _current_prompt[0] = prompt_text + (
                                f"\n\n{_SOD_MARKER} (OBLIGATORIO): el plato anterior aporta "
                                f"~{_cand_sodium_mg:.0f}mg de sodio, pero el presupuesto restante del día "
                                f"era ~{_sod_budget_left:.0f}mg (resto del día ~{_sodium_resto_mg:.0f}mg de "
                                f"un techo de {_sodium_ceiling_mg:.0f}mg). Cambia la proteína principal a "
                                f"una FRESCA no curada (nada de quesos curados, embutidos, enlatados ni "
                                f"camarones) y reduce sal/salsas añadidas. Mantén los macros objetivo."
                            )
                            raise ValueError(
                                f"SODIUM_BUDGET_EXCEEDED: candidato {_cand_sodium_mg:.0f}mg + resto "
                                f"{_sodium_resto_mg:.0f}mg > techo {_sodium_ceiling_mg:.0f}mg"
                            )
                        else:
                            logger.info(
                                f"🧂 [P1-SODIUM-AWARE-PLACEMENT] attempt={_attempt_n}/{_SWAP_MAX_LLM_ATTEMPTS} "
                                f"presupuesto={_sod_budget_left:.0f}mg candidato_mg={_cand_sodium_mg:.0f} "
                                f"decision=accepted_after_retry | sigue sobre presupuesto tras el retry — "
                                f"aceptado (repartir, no prohibir; el aviso existente cubre) | "
                                f"meal_type={meal_type}"
                            )
                    else:
                        logger.info(
                            f"🧂 [P1-SODIUM-AWARE-PLACEMENT] attempt={_attempt_n}/{_SWAP_MAX_LLM_ATTEMPTS} "
                            f"presupuesto={_sod_budget_left:.0f}mg candidato_mg={_cand_sodium_mg:.0f} "
                            f"decision=within_budget | meal_type={meal_type}"
                        )
            except ValueError:
                raise
            except Exception as _sod_guard_e:
                logger.debug(
                    f"[P1-SODIUM-AWARE-PLACEMENT] guard no-op: {type(_sod_guard_e).__name__}: {_sod_guard_e}"
                )

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
        # [P2-SWAP-HONEST-LOG · 2026-07-10] "Usando Plato Fallback" se movió a la rama que SÍ lo
        # emite (knob legacy OFF por default desde P3-SWAP-LLM-RETRIES-422) — el log viejo anunciaba
        # un fallback que casi nunca se entrega y confundía el forensic (2026-07-10).
        logger.error(f"❌ [SWAP_MEAL] Fallaron los intentos LLM y validador: {e}")
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
        logger.warning(f"🍽 [SWAP_MEAL] Usando Plato Fallback (knob MEALFIT_SWAP_EMIT_FALLBACK_DISH=true) | meal_type={meal_type}")
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
            # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T4 fix-round 1)] reusa `_swap_country` (derivado
            # UNA vez al inicio de swap_meal, T3) — DO ⇒ camino byte-idéntico.
            _nfix = _fin_rc(_out, pantry_strict=bool(clean_ingredients), allergies=allergies, country=_swap_country)
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


# [P1-PANTRY-STRICT-CONSENT · 2026-08-02] Helpers del wrapper de consentimiento — SSOT del
# mensaje/precio para que `/swap-meal` y `/fix-sodium-day` (routers/plans.py) no dupliquen
# el copy es-DO ni la lógica de pricing.
def _price_missing_ingredients(raw_items: list) -> list:
    """Items CRUDOS `unauthorized` de `validate_ingredients_against_pantry(..., return_unauthorized=True)`
    → `[{name, qty_needed, unit, est_price_rd}]`, de-duplicados por nombre normalizado.
    `est_price_rd` es `None` cuando no hay match en el catálogo del Supermercado RD (fail-open,
    NUNCA se inventa un precio) o cuando la cantidad no resuelve a gramos."""
    from shopping_calculator import _parse_quantity, estimate_new_ingredient_price_rd
    out = []
    seen = set()
    for raw in raw_items or []:
        try:
            qty, unit, name = _parse_quantity(str(raw))
        except Exception:
            continue
        name = (name or str(raw)).strip()
        if not name:
            continue
        key = strip_accents(name.lower())
        if key in seen:
            continue
        seen.add(key)
        grams = None
        try:
            g, u = _to_base_unit(float(qty or 0), str(unit or ""))
            if u == "g" and g and g > 0:
                grams = g
        except Exception:
            grams = None
        price = estimate_new_ingredient_price_rd(name, grams) if grams else None
        out.append({
            "name": name,
            "qty_needed": qty,
            "unit": unit,
            "est_price_rd": price,
        })
    return out


def _build_consent_message(missing: list) -> str:
    """Copy es-DO honesto del prompt de consentimiento. Nombra hasta 3 ingredientes; si hay
    más, resume el resto con conteo (no lista 8 ítems en un toast)."""
    if not missing:
        return "El chef necesita ingredientes que no están en tu Nevera."
    parts = []
    for m in missing[:3]:
        try:
            qty_txt = f"{float(m.get('qty_needed') or 0):g} {m.get('unit') or ''}".strip()
        except (TypeError, ValueError):
            qty_txt = ""
        price = m.get("est_price_rd")
        price_txt = f" (~RD${price:.0f})" if isinstance(price, (int, float)) and price > 0 else ""
        parts.append(f"{m['name']} {qty_txt}{price_txt}".strip())
    joined = ", ".join(parts)
    if len(missing) > 3:
        joined += f" y {len(missing) - 3} más"
    return (
        f"El chef necesita {joined} — no está en tu Nevera. "
        "¿Lo añadimos a tu lista de compras y seguimos, o buscamos otra opción?"
    )


def swap_meal_with_consent(form_data: dict) -> dict:
    """[P1-PANTRY-STRICT-CONSENT · 2026-08-02] Envoltorio de "Nevera estricta +
    consentimiento" sobre `swap_meal()` — SSOT usado por `/swap-meal` y
    `/fix-sodium-day` (routers/plans.py).

    Contrato:
      - Knob OFF (`MEALFIT_PANTRY_STRICT_UPDATES=false`) ⇒ delega 1:1 a `swap_meal(form_data)`
        (comportamiento legacy exacto, cero discovery, cero consentimiento).
      - Knob ON, swap nevera-only exitoso ⇒ retorna el plato normal (idéntico a `swap_meal`).
      - Knob ON, swap nevera-only falla (`SWAP_STRICT_PANTRY_NO_INVENTORY` /
        `SWAP_LLM_RETRIES_EXHAUSTED`) Y el caller YA mandó `allow_new_ingredients` (consintió)
        ⇒ el fallo se propaga tal cual (soft-fail normal downstream) — NO se reintenta el
        discovery de nuevo (evita loop; el universo ya se amplió y aun así no alcanzó).
      - Knob ON, falla, SIN consentimiento previo ⇒ 1 probe de descubrimiento interno
        (`_pantry_discovery_mode=True`, nunca persistido/expuesto) para nombrar qué le
        falta al chef; si logra nombrar algo, retorna
        `{"needs_new_ingredients": True, "missing_ingredients": [...], "message": ...}`
        SIN levantar — el caller (router) responde 200 soft, no persiste nada, no cobra.
        Si el discovery TAMBIÉN falla o no revela nada accionable, se propaga el ValueError
        original (mismo soft-fail de siempre, cero regresión).
    """
    if not _pantry_strict_updates_enabled():
        return swap_meal(form_data)
    try:
        return swap_meal(form_data)
    except ValueError as ve:
        _msg = str(ve)
        if not (_msg.startswith("SWAP_STRICT_PANTRY_NO_INVENTORY") or _msg.startswith("SWAP_LLM_RETRIES_EXHAUSTED")):
            raise
        _consented = form_data.get("allow_new_ingredients")
        if isinstance(_consented, list) and _consented:
            raise
        _user_id = form_data.get("user_id")
        _universe = _swap_real_pantry_ledger_lines(_user_id) if _user_id and _user_id != "guest" else []
        _discovery_form = dict(form_data)
        _discovery_form["_pantry_discovery_mode"] = True
        try:
            _candidate = swap_meal(_discovery_form)
        except Exception as _disc_e:
            logger.debug(f"[P1-PANTRY-STRICT-CONSENT] discovery probe no-op: {type(_disc_e).__name__}: {_disc_e}")
            raise ve
        if not isinstance(_candidate, dict) or not _candidate.get("ingredients"):
            raise ve
        try:
            _res, _unauthorized = validate_ingredients_against_pantry(
                _candidate.get("ingredients") or [], _universe,
                strict_quantities=True, tolerance=1.30, allow_external_count=0,
                return_unauthorized=True,
            )
        except Exception as _val_e:
            logger.debug(f"[P1-PANTRY-STRICT-CONSENT] discovery diff no-op: {type(_val_e).__name__}: {_val_e}")
            raise ve
        if not _unauthorized:
            # El candidato del probe SÍ cabía en el universo real (p.ej. la Nevera se
            # restockeó entre el intento normal y este probe) — no hay nada honesto que
            # ofrecer como "falta"; preservamos el soft-fail original.
            raise ve
        missing = _price_missing_ingredients(_unauthorized)
        if not missing:
            raise ve
        # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16 (T7)] País beta sin precios nativos ⇒ anular
        # `est_price_rd` ANTES de que llegue a `_build_consent_message` (que ya omite el
        # "(~RD$...)" cuando el precio es None/falsy — cero cambio ahí) Y antes de que salga
        # en `missing_ingredients` del payload JSON (el mismo campo, no solo la prosa — la
        # lección del review final de F0: la prosa compuesta no es el único sitio donde
        # esconde un monto). `form_data` es el mismo SSOT que ya deriva `_swap_country`
        # arriba en `swap_meal` — country_for_form_data es la ÚNICA puerta (T1).
        from constants import country_for_form_data, COUNTRY_PROFILES
        _consent_cc = country_for_form_data(form_data)
        if not COUNTRY_PROFILES.get(_consent_cc, {}).get("has_native_prices", True):
            for _m in missing:
                _m["est_price_rd"] = None
        logger.info(
            f"🧊 [P1-PANTRY-STRICT-CONSENT] needs_new_ingredients user={_user_id!r}: "
            f"{[m['name'] for m in missing]}"
        )
        return {
            "needs_new_ingredients": True,
            "code": "needs_new_ingredients",
            "missing_ingredients": missing,
            "candidate_meal_name": _candidate.get("name"),
            "message": _build_consent_message(missing),
        }







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
            (_kv_key, _now_ts, _now_ts, _now_ts - _TITLE_LOCK_TTL_S), fetch_all=True
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
    # [P1-DIARY-CLAIM-VERIFY · 2026-07-31] Tope del reintento cuando el modelo
    # afirma haber registrado una comida sin llamar `log_consumed_meal`
    # (ver `route_tools` / `nudge_diary_tool`).
    # DEBE estar declarada aquí: LangGraph DESCARTA las claves que un nodo
    # devuelve y no existen en el schema del state — el flag se perdería en
    # silencio, `route_tools` lo leería siempre como False y el turno entraría
    # en bucle call_model→nudge→call_model quemando tokens hasta el timeout.
    # Default: ausente (falsy) — un turno normal nunca la escribe.
    diary_claim_retried: bool

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

                    _allow_exp = bool(tool_args.get("allow_pantry_expansion", False))
                    tool_result = execute_modify_single_meal(
                        user_id=user_id if user_id and user_id != 'guest' else session_id,
                        day_number=tool_args.get("day_number", 1),
                        meal_type=tool_args.get("meal_type", "Desayuno"),
                        changes=tool_args.get("changes", ""),
                        form_data=form_data,
                        allow_pantry_expansion=_allow_exp
                    )
                    # [P1-CHAT-MODIFY-EXPAND-FALLBACK · 2026-07-12] Paridad con el
                    # botón "Cambiar Plato": si el intento PANTRY-STRICT no
                    # convergió (sin `modified_meal`), reintenta UNA vez con
                    # expansión de despensa — equivale a que el usuario acepte
                    # comprar 1-2 ingredientes extra. Vivo: "actualiza el
                    # desayuno" + "algo variado" → el chat se rendía ("no cuajó
                    # sin salirse de lo que tienes") mientras el flujo del botón
                    # encuentra la vuelta. La transparencia viaja como warning
                    # (toast) + aviso en el ToolMessage para que el coach lo diga.
                    _expand_fallback_used = False
                    if not _allow_exp:
                        _first_failed = True
                        try:
                            _probe = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                            _first_failed = not (isinstance(_probe, dict) and "modified_meal" in _probe)
                        except Exception:
                            _first_failed = True
                        if _first_failed:
                            logger.info(
                                "🛒 [P1-CHAT-MODIFY-EXPAND-FALLBACK] strict no convergió → "
                                "retry automático con expansión de despensa"
                            )
                            _retry_result = execute_modify_single_meal(
                                user_id=user_id if user_id and user_id != 'guest' else session_id,
                                day_number=tool_args.get("day_number", 1),
                                meal_type=tool_args.get("meal_type", "Desayuno"),
                                changes=tool_args.get("changes", ""),
                                form_data=form_data,
                                allow_pantry_expansion=True
                            )
                            try:
                                _probe2 = json.loads(_retry_result) if isinstance(_retry_result, str) else _retry_result
                                if isinstance(_probe2, dict) and "modified_meal" in _probe2:
                                    tool_result = _retry_result
                                    _expand_fallback_used = True
                                    coherence_warnings.append(
                                        "Para lograr el cambio se usaron 1-2 ingredientes fuera de tu Nevera — se suman a tu lista de compras."
                                    )
                            except Exception as _exc:
                                # [P2-SILENT-DEGRADATION] best-effort: la falla no debe romper el flujo,
                                # pero sí dejar traza (antes: pass silencioso).
                                logger.debug(
                                    "[P2-SILENT-DEGRADATION] nota de ingredientes fuera de nevera no anexada al mensaje: %s: %s",
                                    type(_exc).__name__, str(_exc)[:160])
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
                                # [P1-CHAT-MODIFY-EXPAND-FALLBACK] Transparencia del retry:
                                + ((" NOTA: con solo lo de su Nevera no convergía, así que el plato "
                                    "usa 1-2 ingredientes nuevos que se suman a su lista de compras — "
                                    "díselo con naturalidad.")
                                   if _expand_fallback_used else "")
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

# ============================================================
# [P1-DIARY-CLAIM-VERIFY · 2026-07-31] "Cena registrada" sin haber registrado
# ============================================================
# Incidente (turno `corr=610fd9c8`): el usuario escribió "cene dos panes con
# queso" y el coach respondió «Cena registrada. Asumo 2 panes de molde…
# Quedó anotada como tu cena de hoy». El journal de ese turno es UNA sola línea
# `call_model` seguida de `Finalizado con éxito`: cero `execute_tools`, cero
# `🍽️ [DIARY]`, cero filas. El modelo NARRÓ el registro sin llamar la tool.
#
# `build_tools_instructions_stream` ya se lo prohíbe con todas las letras
# ("NUNCA digas 'lo registro' o 'anotado' si no llamaste la herramienta en ese
# turno"). Eso es una INSTRUCCIÓN, no un control: el tier `free` va a
# `deepseek-v4-flash` (llm_provider: todo lo que no sea basic/plus/ultra cae al
# barato por diseño fail-cheap) y flash se salta la llamada con bastante más
# frecuencia que pro. Un aviso en el prompt no es un guard.
#
# Coste de no verificarlo: el usuario cree que su comida está contada, el panel
# "Progreso en Tiempo Real" dice 0, y encima el diario queda mintiendo hacia
# atrás — el coach de mañana leerá "no registró nada" y le reprochará algo que
# sí hizo.
#
# La verificación es determinista y barata: si el texto final AFIRMA el registro
# y en ESTE turno no se llamó la tool, se reinyecta un system message y se
# vuelve a `call_model`. UNA sola vez (`diary_claim_retried`), porque un guard
# que puede reintentar sin tope es un bucle de facturación.
# Tooltip-anchor: P1-DIARY-CLAIM-VERIFY

_DIARY_WRITE_TOOLS = ("log_consumed_meal", "correct_consumed_meal")

# Afirmaciones de registro. Sin `\b` final en las raíces verbales a propósito:
# cubre "registrada/registrado/registré/anotada/anoté/apunté" sin enumerar cada
# flexión, que es justo la lista que caduca al primer sinónimo nuevo.
_RE_CLAIM_DIARY = re.compile(
    r"\b(?:regist[rn]?[aeéio]\w*|anot[aeéio]\w*|apunt[aeéio]\w*)\b",
    re.IGNORECASE,
)
# Una negación cerca ANTES del verbo lo convierte en lo contrario ("no pude
# registrarlo", "no quedó anotada"). Sin esto el guard dispararía justo cuando
# el modelo está siendo honesto — y lo castigaría por ello.
_RE_NEGACION = re.compile(r"\b(?:no|nunca|tampoco|sin)\b", re.IGNORECASE)


def _reply_claims_diary_write(text: str) -> bool:
    """True si el texto AFIRMA haber registrado algo en el diario."""
    if not text:
        return False
    for m in _RE_CLAIM_DIARY.finditer(text):
        previo = text[max(0, m.start() - 40):m.start()]
        if _RE_NEGACION.search(previo):
            continue  # "no pude registrarlo" — el modelo está siendo honesto
        return True
    return False


def _diary_tool_called_this_turn(messages: list) -> bool:
    """¿Se llamó alguna tool de diario desde el último mensaje del usuario?

    Se mira `AIMessage.tool_calls` y NO los `ToolMessage`: éstos se construyen
    con `tool_call_id` pero sin `name` (ver `execute_tools`), así que por sí
    solos no dicen QUÉ tool corrió.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return False  # llegamos al turno anterior sin encontrarla
        for tc in (getattr(msg, "tool_calls", None) or []):
            nombre = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if nombre in _DIARY_WRITE_TOOLS:
                return True
    return False


def nudge_diary_tool(state: ChatState):
    """Reinyecta la exigencia y devuelve el turno a `call_model`."""
    logger.warning(
        "🍽️ [P1-DIARY-CLAIM-VERIFY] El modelo afirmó haber registrado sin llamar "
        f"la tool (user={str(state.get('user_id'))[:8]}). Forzando reintento."
    )
    return {
        "messages": [SystemMessage(content=(
            "ALTO. Acabas de afirmar que registraste una comida en el diario, "
            "pero NO llamaste a `log_consumed_meal` en este turno, así que NO "
            "quedó nada guardado y el usuario vería 0 comidas en 'Progreso en "
            "Tiempo Real'.\n"
            "Llama AHORA a `log_consumed_meal` con los macros que estimaste y el "
            "`meal_type` correcto. Si el usuario dijo que fue de otro día, pasa "
            "`days_ago`.\n"
            "Si de verdad NO hay nada que registrar (el usuario no dijo que "
            "comiera algo), responde sin afirmar que registraste nada."
        ))],
        "diary_claim_retried": True,
    }


def route_tools(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "execute_tools"

    # [P1-DIARY-CLAIM-VERIFY] Antes de dar el turno por bueno: ¿el modelo dijo
    # que registró algo que nunca registró?
    if not state.get("diary_claim_retried"):
        contenido = getattr(last_message, "content", "") or ""
        if isinstance(contenido, list):  # algunos providers parten el content
            contenido = " ".join(str(p) for p in contenido)
        if _reply_claims_diary_write(contenido) and not _diary_tool_called_this_turn(messages):
            return "nudge_diary_tool"

    return END

# Removido el MemorySaver global estático
# chat_memory_saver = MemorySaver()
chat_builder = StateGraph(ChatState)
chat_builder.add_node("call_model", call_model)
chat_builder.add_node("execute_tools", execute_tools)
# [P1-DIARY-CLAIM-VERIFY · 2026-07-31] Ver `route_tools`.
chat_builder.add_node("nudge_diary_tool", nudge_diary_tool)
chat_builder.add_edge(START, "call_model")
chat_builder.add_conditional_edges(
    "call_model", route_tools, ["execute_tools", "nudge_diary_tool", END]
)
chat_builder.add_edge("execute_tools", "call_model")
chat_builder.add_edge("nudge_diary_tool", "call_model")
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
    # [P1-COUNTRY-SYSTEM-F2 · T3 · 2026-08-17] Decisión de alcance: este título NO gana
    # `build_language_directive` — es una etiqueta de navegación de 2-4 palabras (más cerca
    # del chrome del dashboard que de "prosa del coach"), no nombrada en el contrato de T3, y
    # requeriría un `get_user_profile` nuevo (esta función no lee perfil hoy). Sigue en
    # español para todo `locale`; follow-up propio si el dueño lo pide.
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
_WEEKDAYS_ES = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_CYCLE_DURATION_DAYS = {"weekly": 7, "semanal": 7, "biweekly": 15, "quincenal": 15, "monthly": 30, "mensual": 30}


def _build_plan_today_context(current_plan, local_date_str: Optional[str] = None) -> str:
    """[P1-CHAT-TODAY-CONTEXT · 2026-07-12] Ancla HOY al día del menú y al ciclo.
    Vivo: "actualiza el desayuno" → el agente preguntó "¿Opción A (Domingo) u
    Opción B (Lunes)?" siendo domingo — tenía la hora (build_temporal_context)
    pero NO el mapeo hoy→día-del-plan, ni la posición del ciclo (día k de 7/15/30,
    que al agotarse exige RENOVAR). Fail-open a "" ante cualquier shape rara."""
    try:
        if not isinstance(current_plan, dict):
            return ""
        days = current_plan.get("days") or []
        if not days:
            return ""
        from datetime import datetime, timezone, timedelta
        if local_date_str:
            today = datetime.strptime(str(local_date_str)[:10], "%Y-%m-%d").date()
        else:
            # Convención del repo: fecha local RD = UTC-4 explícito.
            today = (datetime.now(timezone.utc) - timedelta(hours=4)).date()
        wd = _WEEKDAYS_ES[today.weekday()]

        line = f"\n\n📅 HOY es {wd} {today.isoformat()}."

        # HOY → día del menú (match por day_name; soporta planes shifteados).
        idx = None
        for i, d in enumerate(days):
            if isinstance(d, dict) and str(d.get("day_name") or "").strip().lower() == wd:
                idx = i
                break
        if idx is not None:
            line += (
                f" En el menú del plan, HOY es el día {idx + 1} ('{days[idx].get('day_name')}'). "
                f"Si el usuario menciona una comida SIN especificar el día ('actualiza el desayuno', "
                f"'cámbiame la cena'), asume HOY → day_number={idx + 1} y NO le preguntes a cuál día "
                f"se refiere. Nunca digas 'Opción A/B/C': llama a los días por su nombre."
            )

        # Posición del ciclo (7/15/30 días) + recordatorio de renovación.
        start_raw = current_plan.get("cycle_start_date") or current_plan.get("grocery_start_date")
        dur_days = _CYCLE_DURATION_DAYS.get(str(current_plan.get("calc_grocery_duration") or "").strip().lower())
        if start_raw and dur_days:
            try:
                start_dt = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
                start_local = (start_dt - timedelta(hours=4)).date() if start_dt.tzinfo else start_dt.date()
                day_k = (today - start_local).days + 1
                if day_k >= 1:
                    # `rem` INCLUYE hoy — mismo cálculo que el chip del Dashboard
                    # ("30d mensual · 29d" en el día 2).
                    rem = dur_days - day_k + 1
                    if rem >= 1:
                        line += (
                            f" Ciclo del plan: día {day_k} de {dur_days} "
                            f"(quedan {rem} día(s) incluyendo hoy)."
                        )
                        if rem <= 3:
                            line += " El ciclo está por terminar: si viene al caso, recuérdale con suavidad que pronto deberá RENOVAR su plan."
                    else:
                        line += (
                            f" ⚠️ El ciclo del plan ({dur_days} días) YA TERMINÓ hace {day_k - dur_days} día(s): "
                            f"sugiérele renovar su plan para seguir con menú y lista frescos."
                        )
            except (ValueError, TypeError):
                pass
        return line
    except Exception:
        return ""


def _clamp_tz_offset_mins(value, default_mins: int = 240) -> int:
    """[P1-CHAT-PAST-DAYS · 2026-07-28] SSOT del clamp del `tz_offset` que manda
    el cliente, para los consumidores de este archivo.

    `getTimezoneOffset()` vive en [-840, 840] (UTC+14 .. UTC-14). Fuera de ahí
    el valor no es un huso: es un bug del cliente (un epoch, una unidad mal
    multiplicada). Sin clamp no revienta nada — simplemente vacía el bloque
    `DIARIO DE HOY` en silencio, porque `get_consumed_meals_today` construye una
    ventana [00:00, 23:59] desplazada a un siglo cualquiera.

    `None` → `default_mins`. La resolución es por `is not None` y NUNCA por
    truthiness: `0` es UTC, un offset legítimo que además es falsy.
    tooltip-anchor: P1-CHAT-PAST-DAYS-TZ-CLAMP
    """
    if value is None:
        return default_mins
    try:
        cand = int(value)
    except (TypeError, ValueError):
        return default_mins
    return cand if -840 <= cand <= 840 else default_mins


def _build_pending_days_lines_block(user_id: Optional[str], current_plan, today: date,
                                     plan_id: Optional[str] = None) -> str:
    """[P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] Envuelve `build_pending_plan_days_lines`
    con el COUNT barato de `plan_chunk_queue` que alimenta `compute_chunk_overdue`
    (mismo predicado SSOT que `/chunk-status` y el cron horario). El COUNT SOLO
    se paga si al ciclo VIGENTE del plan le quedan días por entregar
    (`plan_cycle_pending_days`, chat_history_context) — un plan ya completo (el
    caso común, la mayoría de mensajes de chat) sale por el guard de arriba sin
    tocar la DB. Fail-open a "".

    [Ronda 4] Ese guard medía `total_days_requested > len(_archived_days) + len(days)`,
    un conteo que se contamina entre ciclos cuando el plan renueva; ahora es el
    mismo helper de fechas que usan las otras dos superficies. El COUNT incluye
    `pending_user_action` (B2) y todo el bloque está gateado por
    `MEALFIT_UPCOMING_DAYS_UI` (B4).

    [Ronda 1 · fix ALTO 2] El COUNT filtra por `meal_plan_id = plan_id`, la
    columna canónica de `plan_chunk_queue` (FK NOT NULL; el consumidor SSOT
    `_chunk_overdue_alert_job` cuenta igual). Filtrar por `user_id` en su
    lugar suma chunks de CUALQUIER plan del usuario — tras `/restore` el plan
    viejo sigue cancelándose en segundo plano con chunks propios todavía
    `pending/processing/stale`, así que un COUNT por `user_id` puede dar
    `in_flight_count > 0` por un plan AJENO al que el usuario está viendo, y
    el índice diría "PENDIENTE" sobre un día que en realidad está ATRASADO
    (falso negativo). Si `plan_id` no está disponible en el callsite, este
    helper SALTA el COUNT — nunca cuenta por `user_id` como aproximación —
    y fuerza `in_flight_count=1` (⇒ `compute_chunk_overdue` siempre da
    `overdue=False`): sin certeza sobre la cola, el índice solo declara
    PENDIENTE, nunca ATRASADO. Contar de más es peor que no declarar nada.
    tooltip-anchor: P2-CHUNK-OVERDUE-SIGNAL-AGENT
    """
    try:
        if not isinstance(current_plan, dict) or not user_id or user_id == "guest":
            return ""
        # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] Un plan PAUSADO no tiene días
        # en camino: la pausa CANCELA la cola. Anunciar días «que se generan por
        # etapas», o declararlos «ATRASADOS», es hablarle al modelo de un trabajo
        # que nadie va a hacer — y «atrasado» invita a ofrecer soluciones a un
        # problema inexistente.
        #
        # Se pregunta al DATO (el propio plan) y no al modo de sesión: este bloque
        # llega anidado dentro de `_build_past_days_context`, y hacerle llegar el
        # modo exigiría enhebrar un parámetro por dos funciones y dos paths, que es
        # el hilo que se rompe en el próximo refactor. Preguntarle al plan cuesta
        # cero y sigue siendo correcto si mañana lo invoca un camino nuevo.
        if str(current_plan.get("generation_status") or "") == "paused_by_user":
            return ""
        # [Ronda 4 · B4] Mismo knob que `/chunk-status` y el cron horario: sin
        # esto el `COUNT` por turno del coach seguía vivo con la señal apagada.
        from chat_history_context import plan_cycle_pending_days, upcoming_days_signal_enabled
        if not upcoming_days_signal_enabled():
            return ""
        days = [d for d in (current_plan.get("days") or []) if isinstance(d, dict)]
        if not days:
            return ""
        # [Ronda 4 · B1] El short-circuit barato que evita pagar el COUNT era una
        # COPIA del guard de conteo (`len(archived) + len(days) >= total`), o sea
        # la misma 4ª instancia del defecto de ventana rolling: en un plan
        # RENOVADO se disparaba y el coach quedaba mudo aunque el predicado
        # dijera ATRASADO. Ahora usa el mismo helper SSOT que
        # `build_pending_plan_days_lines` — un solo criterio, tres superficies.
        if plan_cycle_pending_days(current_plan) <= 0:
            return ""
        if plan_id:
            from db import execute_sql_query
            # [Ronda 4 · B2] `pending_user_action` cuenta como "algo que lo va a
            # resolver" — la acción del usuario lo resuelve. Sin él, el coach
            # declaraba ATRASADO un día que solo espera consentimiento de nevera
            # y prometía un reintento automático que no existe.
            # ⚠️ [Ronda 5 · N-4 · DECISIÓN DIFERIDA] Sin término temporal: una pausa
            # encallada silencia el ATRASADO indefinidamente (el coach dirá
            # PENDIENTE para siempre). Umbral pendiente de producto — ver el
            # comentario de N-4 en `routers/plans.py::api_chunk_status`.
            cnt = execute_sql_query(
                "SELECT count(*)::int AS c FROM plan_chunk_queue "
                "WHERE meal_plan_id = %s AND status IN "
                "('pending', 'processing', 'stale', 'pending_user_action')",
                (plan_id,), fetch_one=True,
            ) or {}
            in_flight_count = int(cnt.get("c") or 0)
        else:
            in_flight_count = 1  # sin plan_id: no ATRASADO por certeza insuficiente (ver docstring)
        lines = build_pending_plan_days_lines(current_plan, today, in_flight_count)
        if not lines:
            return ""
        return (
            "\n\n🗓️ DÍAS QUE FALTAN POR GENERARSE — todavía NO existen, así que NUNCA "
            "inventes su menú (si preguntan, di que se generan por etapas):\n"
            + "\n".join(lines)
        )
    except Exception as e:
        logger.warning(f"[P2-CHUNK-OVERDUE-SIGNAL] pending days block fail-open: {e}")
        return ""


def _build_past_days_context(user_id: str, current_plan, local_date_str: Optional[str] = None,
                              tz_offset: Optional[int] = None, plan_id: Optional[str] = None) -> str:
    """[P1-CHAT-PAST-DAYS · 2026-07-27] Los dos bloques de días pasados:
    lo que el plan MANDABA + lo que el usuario REGISTRÓ. Fail-open a "".

    SSOT compartida entre el path stream y el no-stream — la divergencia entre
    ambos ya ha causado bugs de zona horaria antes.

    `tz_offset`: minutos que el cliente manda junto a `local_date` (stream
    path). Sin esto, `build_past_diary_block` toma `.date()` del UTC crudo de
    `consumed_at` y una comida de las 10:30pm RD se atribuye al día SIGUIENTE
    — y de paso el día real queda declarado 'SIN REGISTRO'. `tz_offset` es
    input de request body: se guarda la coerción a int, nunca se confía ciego.

    `plan_id`: [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] id de `meal_plans` del
    plan activo, cuando el callsite ya lo resolvió (`get_latest_meal_plan_with_id`
    para el shopping-delta, unas líneas antes en ambos paths). Se reenvía tal
    cual a `_build_pending_days_lines_block` — ver esa función para por qué
    filtrar el COUNT por `meal_plan_id` en vez de `user_id` importa.
    tooltip-anchor: P1-CHAT-PAST-DAYS-AGENT
    """
    try:
        days_back = chat_history_days()
        if days_back <= 0:
            return ""
        if local_date_str:
            try:
                today = datetime.strptime(str(local_date_str)[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                today = rd_today()
        else:
            today = rd_today()

        # `tz_offset is not None` (jamás truthiness: `0` es UTC, legítimo y falsy).
        # El clamp al rango de husos vive en `_clamp_tz_offset_mins` — SSOT
        # compartida con el callsite de `get_consumed_meals_today` del stream.
        tz_offset_mins = _clamp_tz_offset_mins(tz_offset) if tz_offset is not None else 240

        out = build_past_plan_days_block(current_plan, today, days_back=days_back)
        # [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] MISMO bloque, MISMA llamada (no
        # una 2ª pasada al LLM ni un bloque nuevo): días PENDIENTE/ATRASADO.
        out += _build_pending_days_lines_block(user_id, current_plan, today, plan_id=plan_id)

        try:
            from db_facts import get_consumed_meals_since
            since = (today - timedelta(days=days_back)).isoformat()
            rows = get_consumed_meals_since(user_id, since, include_ingredients=True) or []
        except Exception as e:
            logger.warning(f"[P1-CHAT-PAST-DAYS] no se pudo leer el diario multi-día: {e}")
            rows = []
        out += build_past_diary_block(rows, today, days_back=days_back, tz_offset_mins=tz_offset_mins)
        return out
    except Exception as e:
        logger.warning(f"[P1-CHAT-PAST-DAYS] contexto de días pasados fail-open: {e}")
        return ""


def _macro_totals_line(consumed_today: list, current_plan) -> str:
    """[P1-CHAT-MACRO-CONTEXT · 2026-07-12] Macros ACUMULADAS del día (proteína/
    carbos/grasas) con sus metas del plan — el DIARIO DE HOY solo llevaba kcal,
    así que el agente no podía razonar '33g de 125g de proteína' como la card
    'Progreso en Tiempo Real'. Fail-open a "" ante cualquier shape rara."""
    try:
        _p = sum(float(m.get("protein") or 0) for m in consumed_today if isinstance(m, dict))
        _c = sum(float(m.get("carbs") or 0) for m in consumed_today if isinstance(m, dict))
        _f = sum(float(m.get("healthy_fats") or 0) for m in consumed_today if isinstance(m, dict))
        _tm = current_plan.get("macros") if isinstance(current_plan, dict) else None
        if isinstance(_tm, dict) and _tm.get("protein"):
            return (
                f" Macros acumuladas hoy: {round(_p)}g proteína (meta {_tm.get('protein')}g), "
                f"{round(_c)}g carbohidratos (meta {_tm.get('carbs')}g), "
                f"{round(_f)}g grasas (meta {_tm.get('fats')}g)."
            )
        return (
            f" Macros acumuladas hoy: {round(_p)}g proteína, "
            f"{round(_c)}g carbohidratos, {round(_f)}g grasas."
        )
    except Exception:
        return ""


def _resolve_today_plan_day_index(current_plan, local_date_str: Optional[str] = None) -> Optional[int]:
    """[P1-TODAY-REMAINING · 2026-07-28] HOY → índice del día del plan (por
    `day_name`, es-DO), fail-open a `None`. Mismo mapeo que usa
    `_build_plan_today_context` para el header '📅 HOY es...' — extraído a su
    propia función para que `_build_today_remaining_context` (comidas
    restantes de hoy) NO reimplemente el weekday-match. `None` cuando no hay
    match (plan roto, o `days` con `day_name` que no cubre la semana)."""
    try:
        if not isinstance(current_plan, dict):
            return None
        days = current_plan.get("days") or []
        if not days:
            return None
        if local_date_str:
            today = datetime.strptime(str(local_date_str)[:10], "%Y-%m-%d").date()
        else:
            # Convención del repo: fecha local RD = UTC-4 explícito.
            today = (datetime.now(timezone.utc) - timedelta(hours=4)).date()
        wd = _WEEKDAYS_ES[today.weekday()]
        for i, d in enumerate(days):
            if isinstance(d, dict) and str(d.get("day_name") or "").strip().lower() == wd:
                return i
        return None
    except Exception:
        return None


def _build_today_remaining_context(current_plan, consumed_today: list, target_cal_int: int,
                                    total_consumed: float, local_date_str: Optional[str] = None) -> str:
    """[P1-TODAY-REMAINING · 2026-07-28] SSOT compartida entre el path stream
    (~:4462) y el non-stream (~:4090) para el bloque `DIARIO DE HOY` — antes
    los dos if/elif de tiers vivían duplicados y podían divergir por edición
    de uno solo (ya pasó: el texto de la ALERTA CRÍTICA difería palabra por
    palabra entre paths).

    Cierra dos gaps sobre el caso real del owner ("comí el desayuno y luego
    renové el plan — el desayuno de ayer no debería reaparecer"):

    (a) TIER FACTUAL nuevo, sin alarma, ARRIBA del gate de 35%. Un desayuno
        normal ronda 20-25% del presupuesto del día → `remaining` cae en
        ~75-80% del target → el gate viejo (`remaining < 0.35*target`) se
        queda MUDO justo en el caso más común. El tier nuevo cubre todo el
        rango [0.35*target, target) con una frase factual ("cuánto queda"),
        sin emoji de alarma ni urgencia — el 🚨 se reserva para cuando el
        margen es genuinamente ajustado (tiers existentes, sin cambios de
        umbral ni de intención, solo de dónde vive el código).

    (b) CUÁNTAS COMIDAS DEL PLAN quedan hoy, por nombre. El prompt ya tenía
        el plan completo (JSON) Y el diario de hoy por separado, pero nunca
        la resta explícita — el modelo tenía que inferirla sin ayuda.

    REGLA DE MATCH (misma que el frontend, `Dashboard.jsx`): un slot del plan
    de HOY cuenta como "ya comido" cuando una fila del diario de HOY trae un
    `meal_type` que canonicaliza (`canonical_slot_key`, constants.py) a la
    MISMA key que el slot (`meal`) del plan.

    REGLA DE AMBIGÜEDAD: si ≥2 slots del plan de HOY canonicalizan a la MISMA
    key (planes de 5-6 comidas con 2-3 meriendas) y el diario solo trae UNA
    fila de esa key, NO hay forma de saber CUÁL merienda fue — marcar la
    incorrecta es peor que no marcar ninguna. En ese caso NINGÚN meal de esa
    key se remueve de "restantes" (las kcal ya están reflejadas en
    `total_consumed`, que es la suma cruda del diario — la ambigüedad solo
    afecta la ATRIBUCIÓN por nombre, nunca el conteo de kcal).

    Todo kcal en este bloque es un ESTIMADO (buena parte del diario viene de
    una foto analizada por un modelo de visión) — el copy lo frasea así,
    nunca como medición exacta. Fail-open a "" ante cualquier shape rara.
    tooltip-anchor: P1-TODAY-REMAINING
    """
    try:
        out = ""
        remaining = target_cal_int - total_consumed

        # --- (a) tier de calorías restantes ---
        if remaining <= 0:
            out += (
                f"\n🚨 ALERTA CRÍTICA (MEJORA 6): El usuario ha superado su presupuesto "
                f"calórico de hoy. Tiene un exceso estimado de {abs(round(remaining))} kcal. "
                f"Indícale esto con empatía de coach y dale recomendaciones proactivas sobre "
                f"cómo equilibrarse en la cena o mañana."
            )
        elif remaining < (target_cal_int * 0.35):
            out += (
                f"\n🚨 ALERTA DE MICRO-ADAPTACIÓN (MEJORA 6): Al usuario le quedan solo "
                f"~{round(remaining)} kcal estimadas para el resto del día. TIENES LA "
                f"OBLIGACIÓN PROACTIVA de hacerle notar este ajustado presupuesto con "
                f"amabilidad de coach. Sugiérele usar tu herramienta 'modify_single_meal' "
                f"para recalcular y reducir las porciones de sus próximas comidas de hoy "
                f"para mantener su déficit."
            )
        else:
            # [P1-TODAY-REMAINING] Tier factual — sin alarma, sin 🚨. Cubre el
            # caso "ya desayunó" (~20-25% del día) que el gate de 35% dejaba
            # en silencio absoluto — exactamente el caso descrito por el owner.
            out += (
                f"\n📊 ESTADO DEL DÍA: El usuario lleva un estimado de {round(total_consumed)} "
                f"kcal de sus {target_cal_int} kcal del día — le quedan aproximadamente "
                f"{round(remaining)} kcal estimadas para el resto del día. No es una alerta, "
                f"solo el estado actual: tenlo presente si sugieres porciones o comidas nuevas."
            )

        # --- (b) comidas del plan que quedan hoy ---
        try:
            from constants import canonical_slot_key as _canon_slot
            day_idx = _resolve_today_plan_day_index(current_plan, local_date_str=local_date_str)
            days = current_plan.get("days") if isinstance(current_plan, dict) else None
            day = days[day_idx] if (day_idx is not None and isinstance(days, list) and 0 <= day_idx < len(days)) else None
            day_meals = day.get("meals") if isinstance(day, dict) else None
            if isinstance(day_meals, list) and day_meals:
                # Slots ya comidos hoy, canonicalizados. Set (no lista): loguear
                # la misma merienda 2 veces sigue siendo UN solo match de key.
                eaten_keys = set()
                for row in (consumed_today or []):
                    if not isinstance(row, dict):
                        continue
                    k = _canon_slot(row.get("meal_type"))
                    if k:
                        eaten_keys.add(k)

                # Agrupa los índices de los slots del plan de HOY por key canónica.
                groups: dict = {}
                for i, m in enumerate(day_meals):
                    k = _canon_slot(m.get("meal")) if isinstance(m, dict) else None
                    groups.setdefault(k, []).append(i)

                eaten_indices = set()
                for k in eaten_keys:
                    idxs = groups.get(k) or []
                    if len(idxs) == 1:
                        # Match inequívoco: exactamente un slot de hoy tiene esta key.
                        eaten_indices.add(idxs[0])
                    # len==0 (comida no planificada, ej. snack extra) o len>1
                    # (AMBIGUO — 2+ meriendas hoy) → no se marca nada. Ver
                    # docstring "REGLA DE AMBIGÜEDAD".

                remaining_meals = [
                    m for i, m in enumerate(day_meals)
                    if i not in eaten_indices and isinstance(m, dict) and m.get("meal")
                ]
                if remaining_meals:
                    names = ", ".join(str(m.get("meal")) for m in remaining_meals)
                    kcal_left = round(remaining) if remaining > 0 else 0
                    out += (
                        f"\n📋 Hoy te quedan {len(remaining_meals)} comida(s) del plan "
                        f"({names}) y ~{kcal_left} kcal estimadas."
                    )
                elif eaten_indices:
                    out += "\n📋 Hoy ya no te quedan más comidas del plan por registrar."
        except Exception as e:
            logger.warning(f"[P1-TODAY-REMAINING] no se pudo calcular comidas restantes de hoy: {e}")

        return out
    except Exception as e:
        logger.warning(f"[P1-TODAY-REMAINING] fail-open: {e}")
        return ""


def _build_pantry_context(user_id: Optional[str]) -> str:
    """[P1-CHAT-PANTRY-AWARE · 2026-07-12] Snapshot REAL de `user_inventory`
    al system prompt (bloque VOLÁTIL → va al final, no rompe el prefix-cache
    P2-CHAT-PROMPT-STATIC-PREFIX). Vivo: el agente confirmó "ya van 4 leches
    evaporadas" contando su memoria conversacional; la fila real decía 6 (el
    usuario también borra/edita desde la UI de la Nevera). ~400-600 tokens
    para una nevera de 60 items. Kill-switch: MEALFIT_CHAT_PANTRY_SNAPSHOT."""
    if not user_id or user_id == "guest":
        return ""
    try:
        from knobs import _env_bool as _pc_env_bool
        if not _pc_env_bool("MEALFIT_CHAT_PANTRY_SNAPSHOT", True):
            return ""
        from db import execute_sql_query
        rows = execute_sql_query(
            "SELECT ingredient_name, quantity::float8 AS quantity, unit, brand "
            "FROM user_inventory WHERE user_id = %s AND quantity > 0 "
            "ORDER BY ingredient_name LIMIT 120",
            (user_id,), fetch_all=True,
        ) or []
        if not rows:
            return (
                "\n\n🧊 NEVERA FÍSICA AHORA: vacía (0 items). Si el usuario habla "
                "de lo que tiene, invítalo a registrar su compra o escanear su nevera."
            )

        def _fmt_q(q):
            qf = float(q or 0)
            return str(int(qf)) if qf.is_integer() else f"{qf:g}"

        items = "; ".join(
            f"{r['ingredient_name']} {_fmt_q(r['quantity'])} {r['unit']}"
            + (f" ({r['brand']})" if r.get("brand") else "")
            for r in rows
        )
        return (
            f"\n\n🧊 NEVERA FÍSICA AHORA ({len(rows)} items — cantidades REALES de la "
            f"base de datos en este instante): {items}. Estos números son la VERDAD y "
            f"pueden diferir de lo que recuerdes de esta conversación (el usuario también "
            f"edita su Nevera desde la app). Cita SIEMPRE estas cantidades al hablar de lo "
            f"que tiene; tras modificar el inventario, el resultado de la herramienta trae "
            f"los totales nuevos — confirma con esos."
        )
    except Exception as e:
        logger.warning(f"⚠️ [P1-CHAT-PANTRY-AWARE] pantry context error: {e}")
        return ""


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


def _extract_ai_message_text(msg) -> str:
    """Extrae el texto de un AIMessage, normalizando `content` list (bloques
    multimodal/estructurados de algunos providers) a un string plano. Helper
    interno de `_build_final_content_from_messages` (P1-CHAT-NARRATION-KEPT).
    """
    content = msg.content
    if isinstance(content, list):
        content = "\n".join(
            [str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content]
        )
    return str(content) if content else ""


FILLER_STRIP_ENABLED = _env_bool("MEALFIT_CHAT_FILLER_STRIP", True)
_FILLER_MAX_CHARS = _env_int("MEALFIT_CHAT_FILLER_MAX_CHARS", 90, lambda v: 10 <= v <= 400)

# Patrones de ESPERA, no de contenido. Anclados al inicio del bloque y exigiendo que el bloque
# entero sea eso: un gerundio/anuncio suelto que termina en puntos suspensivos o dos puntos.
_FILLER_RX = re.compile(
    r"^\s*(?:"
    r"(?:un\s+)?(?:momento|segundo|instante)\b"          # "un momento…"
    r"|d[ae]me\s+un\s+(?:momento|segundo|instante)\b"     # "dame un segundo…"
    r"|(?:ya\s+)?(?:voy|vamos)\s+a\s+\w+"                 # "vamos a registrarlo…"
    r"|\w+ando\b|\w+iendo\b"                              # "registrando…", "calculando…"
    r"|(?:estimado|resultado|estimaci[oó]n)\s+aproximad[oa]"
    r")[\s\w,]*[.…:]*\s*$",
    re.IGNORECASE,
)
_TIENE_CIFRA_RX = re.compile(r"\d")


def _is_pure_filler(text: str) -> bool:
    """[P1-CHAT-FILLER-STRIP · 2026-07-30] True si un bloque es SOLO relleno de espera.

    Conservador a propósito (ver el bloque de `_build_final_content_from_messages`): exige las tres
    cosas a la vez — corto, SIN ninguna cifra, y que el bloque ENTERO encaje en el patrón de espera.
    Un bloque con kcal, gramos o una hora es contenido aunque suene a narración: "asumo pan de molde
    y 40 g de queso por sándwich" le dice al usuario de dónde salió el estimado y se queda.
    tooltip-anchor: P1-CHAT-FILLER-STRIP"""
    t = (text or "").strip()
    if not t or len(t) > _FILLER_MAX_CHARS:
        return False
    if _TIENE_CIFRA_RX.search(t):
        return False
    return bool(_FILLER_RX.match(t))


def _build_final_content_from_messages(messages: list) -> str:
    """[P1-CHAT-NARRATION-KEPT · 2026-07-28] Reconstruye el texto final del
    turno a partir de TODAS las AIMessage con contenido no vacío emitidas
    después del último HumanMessage — no solo `messages[-1]`.

    Por qué: cuando el modelo emite narración + tool_calls en el MISMO
    completion ("narrate-then-act": ej. "Lo anoto..." + tool_call), esa
    narración ya se streameó al usuario como chunks ordinarios (ver el loop
    `for event in stream_iter` en `chat_with_agent_stream`). El grafo corre
    la tool y vuelve a `call_model`, que produce un SEGUNDO AIMessage
    ("Listo, quedó anotado"). Tomar solo `final_messages[-1]` para el evento
    `done` — que además persiste vía `save_message` en `routers/chat.py` —
    descartaba la primera narración: el usuario la veía aparecer en vivo y
    luego "desaparecer" cuando el `done` la reemplazaba, y un reload del
    historial jamás la mostraba (pérdida de dato real, no solo visual).

    Reglas:
      - Solo el tramo DESPUÉS del último HumanMessage (turnos previos no
        contaminan el turno actual).
      - Solo AIMessage; ToolMessage/HumanMessage intermedios se ignoran.
      - Mensajes con contenido vacío se saltan (AIMessage(content='') que
        solo trae tool_calls, patrón normal del loop narrate-then-act).
      - De-duplicación verbatim: si una pasada posterior repite exactamente
        (tras strip) el texto de una pasada anterior, se omite — el patrón
        esperado es ADITIVO (narrar, actuar, confirmar resultado), no
        repetir la misma frase dos veces.
      - Unión legible con doble salto de línea entre partes.
    """
    if not messages:
        return ""

    last_human_idx = -1
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            last_human_idx = i
    tail = messages[last_human_idx + 1:] if last_human_idx >= 0 else messages

    seen_texts = set()
    parts = []
    for m in tail:
        # [P1-CHAT-MSG-DUCK-TYPE · 2026-07-30] isinstance + duck-type por `m.type == "ai"` (atributo
        # estable de los mensajes langchain). Solo-isinstance falla cuando el AIMessage del state
        # viene de un MODULO distinto al importado aqui (reload/stub en tests, versiones mixtas en
        # prod): dos clases identicas pero no-identicas => todos los mensajes se saltan y el turno
        # sale vacio (lo que P1-CHAT-NEVER-EMPTY degrada, pero mejor no llegar). Medido: 3 tests
        # verdes en aislamiento y rojos SOLO en la corrida completa, arbol congelado.
        if not (isinstance(m, AIMessage) or getattr(m, "type", None) == "ai"):
            continue
        text = _extract_ai_message_text(m)
        if not text:
            continue
        dedup_key = text.strip()
        if not dedup_key or dedup_key in seen_texts:
            continue
        seen_texts.add(dedup_key)
        parts.append(text)

    # [P1-CHAT-FILLER-STRIP · 2026-07-30] Quita los bloques que son PURO relleno de espera.
    #
    # Caso vivo del owner: su respuesta traía un bloque suelto que decía literalmente
    # "Registrando..." — el modelo narrando su propia llamada interna, que este helper preserva
    # (correctamente: P1-CHAT-NARRATION-KEPT existe porque descartar narración perdía contenido
    # real que el usuario ya había visto en vivo). La consecuencia no buscada es que cada preámbulo
    # de relleno queda GRABADO en la conversación para siempre.
    #
    # El prompt ya lo prohíbe (regla 3 del bloque de brevedad) y lleva dos días siendo ignorado
    # porque estaba escrito como lista negra de frases y el modelo la esquiva con sinónimos. Un
    # prompt es una petición; esto es la garantía.
    #
    # Deliberadamente CONSERVADOR — descartar de más es el bug que P1-CHAT-NARRATION-KEPT cerró:
    #   · nunca se toca el ÚLTIMO bloque (es la respuesta real del turno),
    #   · solo cae un bloque SIN cifras (si trae kcal/gramos/hora es contenido, no relleno),
    #   · solo si es corto y encaja en el patrón de espera (gerundio o anuncio + puntos suspensivos
    #     / dos puntos), nunca por longitud sola.
    # Si el filtro se comiera todo, se devuelve el original: preferimos ruido a una respuesta vacía.
    # [P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] Segunda mitad del guard del
    # stream. Retener la deliberación en el SSE no basta: este helper alimenta
    # el evento `done` Y el `save_message` de routers/chat.py, así que sin
    # filtrar también aquí el texto volvería al final del turno y quedaría
    # GRABADO en el historial — el bug se vería igual, solo que un segundo
    # más tarde. Dos mitades del mismo interruptor.
    #
    # Mismo criterio que en el stream (longitud, no prosa) y misma frontera.
    # `_is_pure_filler` no lo caza: la deliberación no es una frase de relleno
    # conocida, es razonamiento largo y variable — por eso hace falta este
    # segundo predicado en vez de alargar aquella lista.
    _delib_max = _chat_pretool_narration_max_chars()

    def _es_deliberacion(_p: str) -> bool:
        return _chat_hold_pretool_text() and len((_p or "").strip()) > _delib_max

    if FILLER_STRIP_ENABLED and len(parts) > 1:
        _keep = [p for i, p in enumerate(parts)
                 # El ÚLTIMO bloque nunca se toca: es la respuesta del turno.
                 # Sin esa excepción, un turno cuya única salida fuese larga
                 # quedaría vacío — el modo de fallo que P1-CHAT-NEVER-EMPTY
                 # degrada y que conviene no provocar.
                 if i == len(parts) - 1
                 or not (_is_pure_filler(p) or _es_deliberacion(p))]
        if _keep:
            _dropped = len(parts) - len(_keep)
            if _dropped:
                logger.info(f"🧹 [P1-CHAT-FILLER-STRIP] {_dropped} bloque(s) de relleno fuera del "
                            f"mensaje final del turno")
            parts = _keep

    out = "\n\n".join(parts)
    # [P1-CHAT-NEVER-EMPTY · 2026-07-30] Si la reconstrucción no sacó NADA, degradar al último
    # mensaje en vez de devolver "".
    #
    # Ninguno de los dos callsites tenía guarda de vacío: el path non-stream devuelve el string tal
    # cual y el stream emite `done` con `response: ""` — y `routers/chat.py::save_message` PERSISTE
    # esa cadena vacía. O sea el usuario ve una respuesta en blanco y el historial la guarda así,
    # sin ningún error en el log que lo explique.
    #
    # Basta con que `isinstance(m, AIMessage)` deje de matchear para llegar aquí: dos objetos
    # AIMessage de módulos distintos (versiones de langchain, un stub de import, un reload) son
    # clases DISTINTAS y el `isinstance` da False para todos los mensajes a la vez. El fallback
    # devuelve el comportamiento pre-P1-CHAT-NARRATION-KEPT (solo el último), que es peor que la
    # reconstrucción pero infinitamente mejor que el vacío — y lo dice en el log, que es lo que
    # faltaba para poder diagnosticarlo.
    if not out.strip() and messages:
        _ultimo = ""
        for m in reversed(tail or messages):
            _t = ""
            try:
                _t = _extract_ai_message_text(m) or ""
            except Exception:
                _t = ""
            if not _t:
                _t = str(getattr(m, "content", "") or "")
            if _t.strip():
                _ultimo = _t
                break
        if _ultimo.strip():
            logger.warning(
                "⚠️ [P1-CHAT-NEVER-EMPTY] la reconstrucción del turno salió vacía con "
                f"{len(messages)} mensaje(s) en el state — degradando al último. Si esto aparece, "
                f"el `isinstance(m, AIMessage)` no está matcheando (clases de módulos distintos).")
            return _ultimo

    return out


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

    # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14 · subido P2-CHAT-PLAN-TOOLS-PAUSE
    #  2026-08-15] El modo se resuelve UNA vez por turno y se deriva el DATO que
    #  apaga las secciones PRESCRIPTIVAS del prompt. Vive al TOPE de la funcion,
    #  donde solo depende de sus parametros: cada vez que un consumidor nuevo
    #  aparecia mas arriba habia que volver a moverlo, y una de esas veces se
    #  colo un NameError. Aqui ya no puede quedar por debajo de nadie.
    plan_vigente = _plan_vigente_para_prompt(user_id, current_plan)


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
        system_prompt += build_tools_instructions(user_id, plan_en_pausa=bool(current_plan) and plan_vigente is None)
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
        system_prompt += build_tools_instructions(user_id, plan_en_pausa=bool(current_plan) and plan_vigente is None)

    inventory_str = ""
    shopping_delta_str = ""
    # [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] Definido ANTES del try de abajo
    # (que ya resuelve `get_latest_meal_plan_with_id` para el shopping-delta)
    # para que `plan_record` esté siempre en scope más abajo, incluso si el
    # try revienta antes de la asignación o si `user_id` es guest.
    plan_record = None

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
    # [P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] `current_plan` (parámetro de esta función) SÍ
    # está en scope aquí — a diferencia de `swap_meal`, tanto el guest (que lo manda en el
    # body) como el autenticado (hidratado desde BD arriba) traen el plan completo. Derivamos
    # num_days/multiplier reales para que estos dos fallbacks no capen la nevera/lista a 1
    # persona-semana cuando el plan real es multi-semana/household>1.
    _vp_num_days, _vp_multiplier = _virtual_pantry_num_days_and_multiplier(current_plan)
    if not inventory_str and form_data:
        current_pantry = form_data.get("current_pantry_ingredients", [])
        if current_pantry and isinstance(current_pantry, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_pantry = aggregate_shopping_list(
                [item.strip() for item in current_pantry if isinstance(item, str) and len(item.strip()) > 2],
                num_days=_vp_num_days, multiplier=_vp_multiplier,
            )
            inventory_str = ", ".join(cleaned_pantry)

    if not shopping_delta_str and form_data:
        current_shopping = form_data.get("current_shopping_list", [])
        if current_shopping and isinstance(current_shopping, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_shop = aggregate_shopping_list(
                [item.strip() for item in current_shopping if isinstance(item, str) and len(item.strip()) > 2],
                num_days=_vp_num_days, multiplier=_vp_multiplier,
            )
            shopping_delta_str = ", ".join(cleaned_shop)

    # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] En pausa la lista de compras del
    # plan deja de ser una obligacion pendiente. El inventario NO cambia: la Nevera
    # funciona igual en modo contador.
    system_prompt += build_inventory_context(
        inventory_str, shopping_delta_str,
        plan_en_pausa=bool(current_plan) and plan_vigente is None,
    )

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
        # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] `locale` sale del MISMO perfil que ya
        # se lee para `full_name` — cero round-trips extra (mismo criterio de reuso que
        # `country_for_form_data`, pero locale vive en `user_profiles`, NO en `form_data`, así
        # que no hay un funnel existente que reusar salvo esta lectura). Guest/user_id==
        # session_id nunca entra al `if` ⇒ `_coach_locale` se queda en el default 'es-DO'
        # (Addendum §2: "Guests ⇒ es-DO always").
        _coach_locale = "es-DO"
        if user_id and user_id != session_id and user_id != "guest":
            _profile_for_prompt = get_user_profile(user_id) or {}
            _id_name = _profile_for_prompt.get("full_name") or ""
            _coach_locale = _profile_for_prompt.get("locale") or "es-DO"
        system_prompt += build_user_identity_context(form_data or {}, _id_name)
        # [P0-CHAT-CLINICAL-BLOCK · 2026-08-11] Va JUSTO DESPUÉS de la identidad y en
        # LOS DOS call sites. El de arriba declara en su docstring que es «NO clínico»
        # porque las alergias «viven en sus bloques estrictos» — cierto para el
        # generador de planes, falso para el chat, que no tenía ninguno. Hasta hoy el
        # coach solo se enteraba de una alergia por la inyección RAG (probabilística) o
        # yendo a buscarla él. Ver `build_clinical_guard_context`.
        system_prompt += build_clinical_guard_context(form_data or {})
        # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] Addendum §2: `locale` mueve la PROSA
        # del coach; comida/tool calls SIGUEN en español (frontera dura, ver
        # `build_language_directive`). es-DO/None/garbage ⇒ "" (byte-idéntico a hoy).
        system_prompt += build_language_directive(_coach_locale)
    except Exception as _id_err:
        logger.warning(f"[P3-CHAT-IDENTITY] No se pudo inyectar identidad al chat: {_id_err}")

    if current_plan:
        # [P2-GENCHUNK-SPEED · 2026-06-01] Podar claves derivadas/pesadas antes
        # de serializar (shopping agregados, coherence telemetry, archived days).
        # [P1-AGENT-WELCOME-TRACKING · 2026-08-14] Encuadre por modo (helper
        # compartido con el stream): en pausa, el plan viaja pero NO como lo de hoy.
        # Este bloque recibe `current_plan` A PROPÓSITO — es el único que debe ver
        # el plan real en pausa (PAUSADO ≠ AMPUTADO).
        system_prompt += _plan_context_for_chat(user_id, current_plan)
        
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
                # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] El respaldo sale del plan
                # VIGENTE, no del pausado: en modo contador esas eran las kcal de un plan
                # congelado presentadas como la meta de HOY, mientras el dashboard del
                # contador pintaba otras. El coach y su propia pantalla decian cifras
                # distintas. Sin plan vigente se usan las metas del modo seguimiento --
                # `get_nutrition_targets`, la MISMA funcion pura que sirve
                # /api/nutrition/targets, sin roundtrip HTTP.
                if not target_calories and plan_vigente:
                    target_calories = plan_vigente.get("calories")
                if not target_calories and form_data:
                    try:
                        from nutrition_calculator import get_nutrition_targets
                        target_calories = (get_nutrition_targets(form_data) or {}).get("target_calories")
                    except Exception as _tgt_e:
                        logger.warning(f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] metas del contador ilegibles: {_tgt_e}")
                
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}."
                # [P1-CHAT-MACRO-CONTEXT · 2026-07-12] Macros desglosadas del
                # día — las MISMAS que la card 'Progreso en Tiempo Real'.
                system_prompt += _macro_totals_line(consumed_today, current_plan)

                if target_calories:
                    try:
                        target_cal_int = int(target_calories)
                        system_prompt += f" Total consumido: {total_consumed} kcal de un presupuesto de {target_cal_int} kcal."
                        # [P1-TODAY-REMAINING · 2026-07-28] Tier factual/alertas +
                        # comidas del plan restantes hoy — SSOT compartida con el
                        # path stream (`_build_today_remaining_context`).
                        system_prompt += _build_today_remaining_context(
                            plan_vigente, consumed_today, target_cal_int, total_consumed,
                            local_date_str=None,
                        )
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
        # [P1-CHAT-PANTRY-AWARE · 2026-07-12] Snapshot real de la Nevera.
        system_prompt += _build_pantry_context(user_id)
        # [P1-CHAT-TODAY-CONTEXT · 2026-07-12] HOY → día del menú + ciclo.
        system_prompt += _build_plan_today_context(plan_vigente, local_date_str=None)
        # [P1-CHAT-PAST-DAYS · 2026-07-27] Paridad con el path stream. Este
        # path no recibe `tz_offset` del cliente: el helper cae a 240 (RD).
        # [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] `plan_record` ya se resolvió
        # arriba para el shopping-delta — reenviar su `id` evita un 2º roundtrip
        # y permite filtrar el COUNT de la cola por `meal_plan_id`.
        system_prompt += _build_past_days_context(
            user_id, current_plan, plan_id=(plan_record or {}).get("id"),
        )

    # [P1-COACH-LANGUAGE-RECENCY · 2026-08-18] REFUERZO de la directiva de idioma como
    # ÚLTIMO bloque del system prompt. La directiva de T3 (arriba, junto a la identidad)
    # quedaba enterrada bajo ~40 bloques españoles (plan JSON, culinary KB, tools, RAG...)
    # y el modelo la desobedecía: primer usuario real con locale='en-US' (2026-08-18
    # 23:14 UTC, session b9f147ca) recibió la respuesta en español con la directiva YA
    # en el prompt — la señal dominante (todo el prompt + «hola» del usuario en español)
    # ganó a una instrucción a mitad de contexto. Recency manda en adherencia: la misma
    # directiva, repetida al FINAL, es lo último que el modelo lee antes de responder.
    # es-DO/guest ⇒ "" (byte-idéntico). Best-effort: jamás rompe el chat.
    try:
        system_prompt += build_language_directive(_coach_locale)
    except Exception as _exc:
        # [P2-SILENT-DEGRADATION] El `pass` a secas dejaba al coach respondiendo en
        # el idioma equivocado SIN rastro: es justo el sintoma que este refuerzo
        # existe para corregir, asi que tragarse su fallo lo vuelve indepurable.
        # Sigue siendo best-effort —jamas rompe el chat—, pero ahora se entera.
        logger.debug(
            "[P2-SILENT-DEGRADATION] refuerzo de idioma del coach: %s: %s",
            type(_exc).__name__, str(_exc)[:160])

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
    # [P1-CHAT-NARRATION-KEPT · 2026-07-28] Antes solo `final_messages[-1]`,
    # que descartaba la narración de un pase narrate-then-act cuando el
    # modelo emitía content+tool_calls y el grafo volvía a `call_model` con
    # una segunda AIMessage. Ver `_build_final_content_from_messages`.
    content = _build_final_content_from_messages(final_messages)

    # [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad output non-stream.
    sanitized_content = _sanitize_chat_output_for_wire(str(content))
    return sanitized_content, final_state.get("updated_fields", {}), final_state.get("new_plan")

from typing import Generator
from sentiment_classifier import classify_sentiment

def chat_with_agent_stream(session_id: str, prompt: str, current_plan: Optional[dict] = None, user_id: Optional[str] = None, form_data: Optional[dict] = None, local_date: Optional[str] = None, tz_offset: Optional[int] = None, is_call_mode: bool = False, plan_tier: str = "gratis") -> Generator[str, None, None]:
    """Generador síncrono de chat que emite eventos del modelo y herramientas mediante SSE (JSONlines).
    FastAPI ejecuta esto en un threadpool externo, liberando el Event Loop para concurrencia real."""
    # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14 · subido P2-CHAT-PLAN-TOOLS-PAUSE
    #  2026-08-15] El modo se resuelve UNA vez por turno y se deriva el DATO que
    #  apaga las secciones PRESCRIPTIVAS del prompt. Vive al TOPE de la funcion,
    #  donde solo depende de sus parametros: cada vez que un consumidor nuevo
    #  aparecia mas arriba habia que volver a moverlo, y una de esas veces se
    #  colo un NameError. Aqui ya no puede quedar por debajo de nadie.
    plan_vigente = _plan_vigente_para_prompt(user_id, current_plan)

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
    # [P1-TIER-PARITY · 2026-07-12] Decisión de producto del owner: TODOS los
    # tiers (incluido gratis) tienen los MISMOS privilegios de features — la
    # única diferencia entre planes son las CANTIDADES de créditos
    # (auth._TIER_LIMITS). Pre-fix: sentimiento solo plus+ y RAG excluía a
    # gratis. Guests (sin cuenta) siguen fuera del RAG: no tienen user_facts.
    _do_sentiment = True
    _do_rag = bool(user_id) and user_id != "guest"
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
        system_prompt += build_tools_instructions_stream(user_id, plan_en_pausa=bool(current_plan) and plan_vigente is None)
        # --- bloques dinámicos (volátiles) al final ---
        system_prompt += build_temporal_context(local_date=local_date, tz_offset=tz_offset)
        system_prompt += build_circadian_context(schedule_type)
        system_prompt += build_temporal_proactive_context()
        # 🎭 Personalidad adaptativa basada en el sentimiento detectado (per-turn)
        if sentiment_result.get("instruction"):
            system_prompt += f"\n\n{sentiment_result['instruction']}"
        if rag_context:
            system_prompt += f"\n{rag_context}"
    else:
        system_prompt = _base_inline
        system_prompt += build_temporal_context(local_date=local_date, tz_offset=tz_offset)
        system_prompt += build_circadian_context(schedule_type)
        system_prompt += build_temporal_proactive_context()
        # 🎭 Inyectar personalidad adaptativa basada en el sentimiento detectado
        if sentiment_result.get("instruction"):
            system_prompt += f"\n\n{sentiment_result['instruction']}"
        system_prompt += f"\n{CULINARY_KNOWLEDGE_BASE}"
        if rag_context: system_prompt += f"\n{rag_context}"
        system_prompt += build_tools_instructions_stream(user_id, plan_en_pausa=bool(current_plan) and plan_vigente is None)

    inventory_str = ""
    shopping_delta_str = ""
    # [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] Definido ANTES del try de abajo
    # (que ya resuelve `get_latest_meal_plan_with_id` para el shopping-delta)
    # para que `plan_record` esté siempre en scope más abajo, incluso si el
    # try revienta antes de la asignación o si `user_id` es guest.
    plan_record = None

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
    # [P3-AGG-NUM-DAYS-PROPAGATE · 2026-08-04] `current_plan` (parámetro de esta función) SÍ
    # está en scope aquí — a diferencia de `swap_meal`, tanto el guest (que lo manda en el
    # body) como el autenticado (hidratado desde BD arriba) traen el plan completo. Derivamos
    # num_days/multiplier reales para que estos dos fallbacks no capen la nevera/lista a 1
    # persona-semana cuando el plan real es multi-semana/household>1.
    _vp_num_days, _vp_multiplier = _virtual_pantry_num_days_and_multiplier(current_plan)
    if not inventory_str and form_data:
        current_pantry = form_data.get("current_pantry_ingredients", [])
        if current_pantry and isinstance(current_pantry, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_pantry = aggregate_shopping_list(
                [item.strip() for item in current_pantry if isinstance(item, str) and len(item.strip()) > 2],
                num_days=_vp_num_days, multiplier=_vp_multiplier,
            )
            inventory_str = ", ".join(cleaned_pantry)

    if not shopping_delta_str and form_data:
        current_shopping = form_data.get("current_shopping_list", [])
        if current_shopping and isinstance(current_shopping, list):
            from shopping_calculator import aggregate_shopping_list
            cleaned_shop = aggregate_shopping_list(
                [item.strip() for item in current_shopping if isinstance(item, str) and len(item.strip()) > 2],
                num_days=_vp_num_days, multiplier=_vp_multiplier,
            )
            shopping_delta_str = ", ".join(cleaned_shop)

    # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] En pausa la lista de compras del
    # plan deja de ser una obligacion pendiente. El inventario NO cambia: la Nevera
    # funciona igual en modo contador.
    system_prompt += build_inventory_context(
        inventory_str, shopping_delta_str,
        plan_en_pausa=bool(current_plan) and plan_vigente is None,
    )

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
        # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] `locale` sale del MISMO perfil que ya
        # se lee para `full_name` — cero round-trips extra (mismo criterio de reuso que
        # `country_for_form_data`, pero locale vive en `user_profiles`, NO en `form_data`, así
        # que no hay un funnel existente que reusar salvo esta lectura). Guest/user_id==
        # session_id nunca entra al `if` ⇒ `_coach_locale` se queda en el default 'es-DO'
        # (Addendum §2: "Guests ⇒ es-DO always").
        _coach_locale = "es-DO"
        if user_id and user_id != session_id and user_id != "guest":
            _profile_for_prompt = get_user_profile(user_id) or {}
            _id_name = _profile_for_prompt.get("full_name") or ""
            _coach_locale = _profile_for_prompt.get("locale") or "es-DO"
        system_prompt += build_user_identity_context(form_data or {}, _id_name)
        # [P0-CHAT-CLINICAL-BLOCK · 2026-08-11] Va JUSTO DESPUÉS de la identidad y en
        # LOS DOS call sites. El de arriba declara en su docstring que es «NO clínico»
        # porque las alergias «viven en sus bloques estrictos» — cierto para el
        # generador de planes, falso para el chat, que no tenía ninguno. Hasta hoy el
        # coach solo se enteraba de una alergia por la inyección RAG (probabilística) o
        # yendo a buscarla él. Ver `build_clinical_guard_context`.
        system_prompt += build_clinical_guard_context(form_data or {})
        # [P1-COUNTRY-SYSTEM-F2 · Task 3 · 2026-08-17] Addendum §2: `locale` mueve la PROSA
        # del coach; comida/tool calls SIGUEN en español (frontera dura, ver
        # `build_language_directive`). es-DO/None/garbage ⇒ "" (byte-idéntico a hoy).
        system_prompt += build_language_directive(_coach_locale)
    except Exception as _id_err:
        logger.warning(f"[P3-CHAT-IDENTITY] No se pudo inyectar identidad al chat: {_id_err}")

    if current_plan:
        # [P2-GENCHUNK-SPEED · 2026-06-01] Podar claves derivadas/pesadas (ver
        # _prune_plan_for_chat) antes de serializar — paridad con el path no-stream.
        # [P1-AGENT-WELCOME-TRACKING · 2026-08-14] Mismo helper que el path
        # no-stream — la divergencia entre ambos ya costó bugs (P1-CHAT-PAST-DAYS).
        system_prompt += _plan_context_for_chat(user_id, current_plan)
        
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
            # [P1-CHAT-PAST-DAYS · 2026-07-28] Clamp al rango de husos, como los
            # otros dos consumidores del mismo `tz_offset`. Un valor basura no
            # revienta: vacía `DIARIO DE HOY` en silencio. `None` se PRESERVA —
            # `get_consumed_meals_today` exige `date_str` Y `tz_offset_mins` no
            # nulos para usar el día local; inventar 240 ahí cambiaría la
            # ventana de los clientes que no mandan offset.
            _tz_diario = _clamp_tz_offset_mins(tz_offset) if tz_offset is not None else None
            consumed_today = get_consumed_meals_today(user_id, date_str=local_date, tz_offset_mins=_tz_diario)
            if consumed_today:
                total_consumed = sum(m.get('calories', 0) for m in consumed_today)
                meals_text = ", ".join([f"{m.get('meal_name')} ({m.get('calories')} kcal)" for m in consumed_today])
                
                target_calories = form_data.get("target_calories") if form_data else None
                # [P1-CHAT-PAUSED-PROMPT-BLOCKS · 2026-08-14] El respaldo sale del plan
                # VIGENTE, no del pausado: en modo contador esas eran las kcal de un plan
                # congelado presentadas como la meta de HOY, mientras el dashboard del
                # contador pintaba otras. El coach y su propia pantalla decian cifras
                # distintas. Sin plan vigente se usan las metas del modo seguimiento --
                # `get_nutrition_targets`, la MISMA funcion pura que sirve
                # /api/nutrition/targets, sin roundtrip HTTP.
                if not target_calories and plan_vigente:
                    target_calories = plan_vigente.get("calories")
                if not target_calories and form_data:
                    try:
                        from nutrition_calculator import get_nutrition_targets
                        target_calories = (get_nutrition_targets(form_data) or {}).get("target_calories")
                    except Exception as _tgt_e:
                        logger.warning(f"[P1-CHAT-PAUSED-PROMPT-BLOCKS] metas del contador ilegibles: {_tgt_e}")
                
                system_prompt += f"\n\nDIARIO DE HOY: El usuario ya ha registrado consumir hoy las siguientes comidas: {meals_text}. [P1-CHAT-ACT-DONT-ASK] Úsalo para NO DUPLICAR, no para pedir permiso: si una foto o mensaje nuevo coincide con algo que ya está aquí, felicítalo o coméntalo sin volver a registrarlo de nuevo; si ya tiene una cena registrada y llega otra foto de noche, asume que es un snack nocturno (o pregúntale por qué repite) en vez de tratarla como si fuera la cena otra vez. Fuera de ese caso de duplicado, sigue la regla general: comida nueva en pasado = regístrala YA, sin preguntar."
                # [P1-CHAT-MACRO-CONTEXT · 2026-07-12] Macros desglosadas del
                # día — las MISMAS que la card 'Progreso en Tiempo Real'.
                system_prompt += _macro_totals_line(consumed_today, current_plan)
                
                if target_calories:
                    try:
                        target_cal_int = int(target_calories)
                        system_prompt += f" Total consumido: {total_consumed} kcal de un presupuesto de {target_cal_int} kcal."
                        # [P1-TODAY-REMAINING · 2026-07-28] Tier factual/alertas +
                        # comidas del plan restantes hoy — SSOT compartida con el
                        # path non-stream (`_build_today_remaining_context`).
                        system_prompt += _build_today_remaining_context(
                            plan_vigente, consumed_today, target_cal_int, total_consumed,
                            local_date_str=local_date,
                        )
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
        # [P1-CHAT-PANTRY-AWARE · 2026-07-12] Snapshot real de la Nevera.
        system_prompt += _build_pantry_context(user_id)
        # [P1-CHAT-TODAY-CONTEXT · 2026-07-12] HOY → día del menú + ciclo.
        system_prompt += _build_plan_today_context(plan_vigente, local_date_str=local_date)
        # [P1-CHAT-PAST-DAYS · 2026-07-27] Días que ya pasaron: plan prescrito
        # (índice barato) + diario real. Va DESPUÉS del DIARIO DE HOY y del
        # prefijo estático — ver docs/chat_past_days_memory.md §3 Pieza 2.
        # [P2-CHUNK-OVERDUE-SIGNAL · 2026-08-04] `plan_record` ya se resolvió
        # arriba para el shopping-delta — reenviar su `id` evita un 2º roundtrip
        # y permite filtrar el COUNT de la cola por `meal_plan_id`.
        system_prompt += _build_past_days_context(
            user_id, current_plan, local_date_str=local_date, tz_offset=tz_offset,
            plan_id=(plan_record or {}).get("id"),
        )

    # [P1-COACH-LANGUAGE-RECENCY · 2026-08-18] Mismo refuerzo final que el path
    # no-stream (la divergencia entre ambos ya costó bugs — P1-CHAT-PAST-DAYS,
    # P1-CHAT-PAUSED-PROMPT-BLOCKS): la directiva de idioma repetida como ÚLTIMO
    # bloque, porque a mitad de prompt el modelo la desobedeció con el primer
    # usuario real en-US. Ver el comentario gemelo en chat_with_agent.
    try:
        system_prompt += build_language_directive(_coach_locale)
    except Exception as _exc:
        # [P2-SILENT-DEGRADATION] El `pass` a secas dejaba al coach respondiendo en
        # el idioma equivocado SIN rastro: es justo el sintoma que este refuerzo
        # existe para corregir, asi que tragarse su fallo lo vuelve indepurable.
        # Sigue siendo best-effort —jamas rompe el chat—, pero ahora se entera.
        logger.debug(
            "[P2-SILENT-DEGRADATION] refuerzo de idioma del coach: %s: %s",
            type(_exc).__name__, str(_exc)[:160])

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

    # ================================================================
    # [P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] La deliberación en pantalla
    # ================================================================
    # Incidente: el usuario escribió "cene dos panes con queso" y antes de la
    # respuesta le aparecieron ~4.000 caracteres de deliberación en primera
    # persona — "Hmm, pero son las 10:23 AM", "Espera, según la regla 6-bis",
    # "Déjame pensar" — y, entre ellos, la frase que lo delata: «Déjame llamar
    # la herramienta primero (regla de cero texto antes de herramienta)». El
    # modelo CITA la regla que le prohíbe eso mientras la incumple.
    #
    # No es un leak de reasoning tokens: DeepSeek los manda en
    # `reasoning_content` (campo aparte que este loop no lee) y además el
    # thinking está desactivado desde P1-DEEPSEEK-THINKING-OFF. Es el modelo
    # escribiendo su deliberación como `content` normal.
    #
    # El guard que debería taparlo ya existe cinco líneas más abajo
    # (`if not msg_chunk.tool_calls`) pero **el orden del streaming lo
    # derrota**: los chunks de texto llegan ANTES que la tool_call, así que
    # cuando se evalúan todavía no hay `tool_calls` y pasan enteros.
    #
    # ⚠️ NO se descarta todo el texto pre-tool: eso desharía
    # P1-CHAT-NARRATION-KEPT (2026-07-28), que restauró a propósito la
    # narración corta ("Lo anoto...") porque antes aparecía y se desvanecía —
    # pérdida de dato real. Dos guardas sobre el mismo campo oscilan.
    #
    # El corte NO es "hay texto antes" sino CUÁNTO: una narración útil son
    # ~40 chars; la deliberación del incidente eran miles. Es una diferencia
    # de un orden de magnitud, no una heurística sobre prosa.
    _hold_pretool = _chat_hold_pretool_text()
    _pretool_max = _chat_pretool_narration_max_chars()
    _pretool_buf: list[str] = []
    _tool_seen = False

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
                            # [P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] Ver el
                            # bloque de arriba: el texto ANTERIOR a la primera
                            # tool_call se retiene hasta saber cuánto es.
                            if _hold_pretool and not _tool_seen:
                                _pretool_buf.append(chunk_content)
                            else:
                                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_content})}\n\n"
                                # [P1-CHAT-CHECKPOINT-DEGRADE · 2026-05-20]
                                _chunks_yielded += 1
                    else:
                        # [P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] Punto de
                        # decisión: llegó la tool_call, ya sabemos qué era el
                        # texto retenido.
                        if _hold_pretool and not _tool_seen:
                            _tool_seen = True
                            _retenido = "".join(_pretool_buf)
                            _pretool_buf.clear()
                            if len(_retenido.strip()) > _pretool_max:
                                logger.warning(
                                    f"🧠 [P1-CHAT-DELIBERATION-HIDDEN] {len(_retenido)} chars "
                                    f"de deliberación antes de la tool — descartados "
                                    f"(cap {_pretool_max}). Inicio: {_retenido[:90]!r}"
                                )
                            elif _retenido:
                                # Narración corta: se emite (P1-CHAT-NARRATION-KEPT).
                                yield f"data: {json.dumps({'type': 'chunk', 'text': _retenido})}\n\n"
                                _chunks_yielded += 1
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
                                else:
                                    # [P1-CHAT-NARRATION-KEPT-REVIEW-1 · 2026-07-28]
                                    # Fallback genérico para CUALQUIER tool_call sin
                                    # bucket de mensaje dedicado (ej. check_current_pantry,
                                    # modify_pantry_inventory, mark_shopping_list_purchased,
                                    # check_hydration_today, log_water_glass,
                                    # suggest_foods_for_nutrient, check_clinical_profile,
                                    # consultar_dia_del_plan, regenerate_full_day — y
                                    # cualquier tool futura añadida a `agent_tools`/
                                    # `_PLAN_MUTATION_TOOLS` sin branch explícito acá).
                                    #
                                    # Por qué: el frontend (AgentPage.jsx / ChatWidget.jsx)
                                    # usa el evento `progress` como ÚNICA señal de que una
                                    # nueva pasada narrate-then-act está por comenzar, para
                                    # insertar el mismo separador '\n\n' que
                                    # `_build_final_content_from_messages` usa al unir
                                    # AIMessage del turno (`"\n\n".join(parts)`, línea
                                    # ~4117). Sin un progress event para ESTA tool_call, el
                                    # frontend concatenaba las dos pasadas SIN separador —
                                    # el `done.response` (que SÍ trae '\n\n') nunca hacía
                                    # match contra lo ya mostrado en
                                    # `reconcileFinalChatText` (`final.startsWith(displayed)`
                                    # siempre falso para las 8+ tools sin branch dedicado),
                                    # forzando SIEMPRE la rama 'replace' — reflow visible al
                                    # final del turno. Ver hallazgo de review P1-CHAT-NARRATION-KEPT.
                                    yield f"data: {json.dumps({'type': 'progress', 'message': get_progress_msg('analizando')})}\n\n"

        # [P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] El turno terminó sin
        # ninguna tool_call ⇒ lo retenido NO era deliberación previa a una
        # herramienta: es la respuesta. Se emite entera. Sin este flush, un
        # turno conversacional normal (el caso más común del chat) saldría
        # VACÍO — el guard se comería justo lo que debe proteger.
        if _pretool_buf and not _tool_seen:
            _resto = "".join(_pretool_buf)
            _pretool_buf.clear()
            if _resto:
                yield f"data: {json.dumps({'type': 'chunk', 'text': _resto})}\n\n"
                _chunks_yielded += 1

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
            # [P1-CHAT-NARRATION-KEPT · 2026-07-28] Antes solo
            # `final_messages[-1]`: en un pase narrate-then-act (content +
            # tool_calls en el mismo completion) la narración ya se
            # streameó como chunks al usuario, pero el `done` la
            # reemplazaba por el segundo AIMessage post-tool — pérdida de
            # dato real, porque `routers/chat.py::save_message` persiste
            # justo este `response`. Ver `_build_final_content_from_messages`.
            final_content = _build_final_content_from_messages(final_messages)

    logger.info("✅ [CHAT STREAM] Finalizado con éxito.")
    # [P2-CHAT-SANITIZE · 2026-05-19] Defensa-en-profundidad del payload `done`.
    # save_message en routers/chat.py persiste este `response` a DB — sanitizar
    # acá significa que la versión persistida también queda neutralizada.
    final_content = _sanitize_chat_output_for_wire(final_content)
    yield f"data: {json.dumps({'type': 'done', 'response': final_content, 'updated_fields': updated_fields, 'new_plan': new_plan, 'coherence_warnings': coherence_warnings, 'pantry_modified_at': pantry_modified_at, 'pantry_depleted_items': pantry_depleted_items})}\n\n"