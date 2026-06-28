import os
import logging
import json
import re
import unicodedata
import time
from typing import Optional
from langchain_core.tools import tool
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Gemini → DeepSeek con router por tier.
from llm_provider import ChatDeepSeek, DEEPSEEK_FLASH, resolve_model_for_user
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
# [P1-TOOLS-LLM-HARDENING · 2026-05-20] Reuso del CB per-modelo del
# graph_orchestrator. Espejo del patrón de `agent.py` (P1-CHAT-CB-EXTEND).
# `run_plan_pipeline` ya se importa desde el mismo módulo — no añade ciclo.
from graph_orchestrator import run_plan_pipeline, _get_circuit_breaker, clinical_backstop_for_meal, UPDATE_CLINICAL_GUARD, renal_protein_trim_for_update, food_safety_backstop_for_meal, condition_substitution_backstop_for_meal, appetibility_fix_for_update
# [P1-TOOLS-LLM-HARDENING · 2026-05-20] Knobs auto-registrados para los 2
# callsites Gemini de este módulo (analyze_preferences_agent / execute_modify_single_meal).
# Pre-fix: ambos hardcodean `gemini-3.1-pro-preview` (viola P3-PREVIEW-MODEL-KNOB)
# + sin `timeout=` (puede colgar al worker indefinidamente) + sin CB gate
# (avalancha si Gemini está degradado). El P1-CHAT-CB-EXTEND ya cubrió los
# 4 callsites de `agent.py`; este fix cierra los 2 de `tools.py`.
from knobs import _env_str, _env_float


def _tools_pref_agent_model_name() -> str:
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] Default DeepSeek V4 Flash. El
    # preference analyzer hace clasificación simple (rechazos/gustos) —
    # tarea aux barata, mismo modelo para todos los tiers. Override:
    # `MEALFIT_TOOLS_PREF_AGENT_MODEL=deepseek-v4-pro` si la calidad degrada.
    return _env_str(
        "MEALFIT_TOOLS_PREF_AGENT_MODEL",
        DEEPSEEK_FLASH,
    )


def _tools_pref_agent_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_TOOLS_PREF_AGENT_LLM_TIMEOUT_S",
        15.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


def _tools_modify_meal_model_name(user_id: Optional[str] = None) -> str:
    # [P0-DEEPSEEK-MIGRATION · 2026-06-12] TIER-ROUTED: modificar una comida
    # del plan es surface user-facing de calidad — paid (basic/plus/ultra) →
    # deepseek-v4-pro, gratis/guest → deepseek-v4-flash. El override del knob
    # SIEMPRE gana (rollback sin redeploy).
    override = _env_str("MEALFIT_TOOLS_MODIFY_MEAL_MODEL", "")
    if override:
        return override
    return resolve_model_for_user(user_id)


def _tools_modify_meal_llm_timeout_s() -> float:
    # 30s default (espejo de `_chat_swap_llm_timeout_s` en agent.py): este
    # callsite corre dentro de tenacity retry 3× — budget per-call holgado
    # para no abortar antes del retry. El total wall-clock cap del chat
    # graph (60s non-stream / 120s stream) cubre el peor caso encadenado.
    return _env_float(
        "MEALFIT_TOOLS_MODIFY_MEAL_LLM_TIMEOUT_S",
        30.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


def _tools_get_chat_safety_helpers():
    """[P1-TOOLS-LLM-HARDENING · 2026-05-20] Lazy import de helpers
    definidos en `agent.py` (rate-limit detector, exceptions canónicas,
    metric emitter, llm-usage emitter). Importar al top-level crearía
    ciclo: `agent` ya importa de `tools`. Al diferir hasta runtime,
    `agent` ya está cargado completo. Cualquier import miss se silencia
    (fail-open con None/dummy) — los callers tienen fallbacks defensivos.
    Tooltip-anchor: P1-TOOLS-LLM-HARDENING-LAZY.
    """
    try:
        from agent import (
            _is_rate_limit_error,
            LLMCircuitBreakerOpen,
            LLMRateLimitedError,
            _emit_chat_rate_limited_metric_best_effort,
        )
    except Exception:
        # Defensivo: si el import falla en tests con mocks parciales,
        # devolvemos stubs que NO consideran nada rate-limit y no rompen.
        _is_rate_limit_error = lambda _exc: False
        class LLMCircuitBreakerOpen(RuntimeError):
            pass
        class LLMRateLimitedError(RuntimeError):
            pass
        _emit_chat_rate_limited_metric_best_effort = lambda *_a, **_kw: None
    try:
        from graph_orchestrator import _emit_llm_usage_event_best_effort
    except Exception:
        _emit_llm_usage_event_best_effort = lambda **_kw: None
    return (
        _is_rate_limit_error,
        LLMCircuitBreakerOpen,
        LLMRateLimitedError,
        _emit_chat_rate_limited_metric_best_effort,
        _emit_llm_usage_event_best_effort,
    )

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
    
    # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Knob model + timeout + CB gate +
    # rate-limit discrimination + token telemetry. Espejo exacto del patrón
    # de `agent.py::call_model` (P1-CHAT-CB-EXTEND + P1-CHAT-LLM-429 +
    # P2-CHAT-TOKEN-TELEMETRY). Cierra los 3 modos de fallo que el callsite
    # tenía abiertos:
    #   (a) hardcoded preview model → no rotable sin redeploy.
    #   (b) sin timeout → worker thread starvation si Gemini cuelga.
    #   (c) sin CB gate → avalancha bajo provider degradado.
    # Tooltip-anchor: P1-TOOLS-LLM-HARDENING.
    _pref_model = _tools_pref_agent_model_name()
    pref_llm = ChatDeepSeek(
        model=_pref_model,
        temperature=0.3, # Baja temperatura para ser analítico
        timeout=_tools_pref_agent_llm_timeout_s(),
    )

    (_is_rl_err, _CBOpen, _RLErr, _emit_rl_metric, _emit_usage_event) = _tools_get_chat_safety_helpers()
    _pref_cb = _get_circuit_breaker(_pref_model)
    if not _pref_cb.can_proceed():
        logger.warning(
            f"🛑 [P1-TOOLS-LLM-HARDENING] analyze_preferences_agent CB abierto "
            f"model={_pref_model!r} — fail-fast sin invocar Gemini. "
            f"Reintentar tras MEALFIT_CB_RESET_TIMEOUT_S segundos."
        )
        raise _CBOpen(
            f"analyze_preferences_agent LLM circuit breaker open for model={_pref_model}"
        )

    start_time = time.time()
    try:
        response = pref_llm.invoke(prompt)
        _pref_cb.record_success()
    except Exception as _pref_invoke_exc:
        if _is_rl_err(_pref_invoke_exc):
            _emit_rl_metric(None, None, _pref_model)
            logger.warning(
                f"⚠️ [P1-CHAT-LLM-429] analyze_preferences_agent rate-limit "
                f"model={_pref_model!r} — NO cuenta como CB failure."
            )
            raise _RLErr(
                f"analyze_preferences_agent LLM rate limited for model={_pref_model}: {_pref_invoke_exc!r}"
            ) from _pref_invoke_exc
        _pref_cb.record_failure()
        raise

    try:
        _emit_usage_event(
            llm=pref_llm,
            result=response,
            duration_s=time.time() - start_time,
            node='tool_analyze_preferences',
        )
    except Exception:
        pass

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
                    # [P3-PROD-AUDIT-2 · 2026-05-30] Emitir alerta de persist-fail
                    # (paridad con services._save_plan_and_track_background,
                    # P2-PLAN-PERSIST-FAILED). Este path chat-driven era un
                    # blind-spot de observabilidad: el fallo solo se logueaba.
                    try:
                        from services import _persist_plan_persist_failed_alert
                        _persist_plan_persist_failed_alert(
                            user_id, "chat_tool_save_failed: save_new_meal_plan_robust devolvió falsy"
                        )
                    except Exception as _al_e:
                        logger.debug(f"[P3-PROD-AUDIT-2] persist alert (else) falló: {_al_e}")
                
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
                logger.error(f"⚠️ Aviso: No se pudo guardar el plan en la DB (error {db_e}), pero el plan se devolverá al usuario.")
                # [P3-PROD-AUDIT-2 · 2026-05-30] idem rama else: alerta de persist-fail.
                try:
                    from services import _persist_plan_persist_failed_alert
                    _persist_plan_persist_failed_alert(user_id, f"chat_tool_save_failed: {type(db_e).__name__}")
                except Exception as _al_e:
                    logger.debug(f"[P3-PROD-AUDIT-2] persist alert (except) falló: {_al_e}")

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
    deduct_summary = None
    # [P2-CONSUMED-DEDUP-INVENTORY · 2026-05-30] NO deducir el inventario si el
    # log fue un dedup-skip (re-emisión del LLM dentro de la ventana de 60s):
    # db_log_consumed_meal retorna el sentinel "deduped" en ese caso. Sin este
    # gate, la 2ª emisión saltaba el INSERT (calorías OK) pero corría la
    # deducción de nuevo → la nevera se descontaba AL DOBLE del consumo real.
    if has_ingredients and result != "deduped":
        deduct_summary = db_inventory.deduct_consumed_meal_from_inventory(user_id, ingredients)

    if result is not None:
        msg = f"¡Éxito! Se ha registrado el consumo de '{meal_name}' ({calories} kcal, {protein}g proteína, {carbs}g carbohidratos, {healthy_fats}g grasas saludables) en tu diario."
        # [P1-AGENT-HINT · 2026-05-22] Si la deducción tuvo items que la
        # inferencia P1-PANTRY-INFER no pudo procesar, añadir hint visible
        # para la LLM en el ToolMessage. La LLM puede entonces pedir al
        # usuario cantidades aproximadas en su siguiente turno (red de
        # seguridad sobre #1: caso raro donde `_infer_typical_portion`
        # retorna None — name vacío post-parse — o donde
        # `add_or_update_inventory_item` falla por master mismatch).
        if isinstance(deduct_summary, dict):
            failed = deduct_summary.get("failed_to_deduct") or []
            if failed:
                # Mostrar primeros 5 items para no inflar el ToolMessage.
                preview = failed[:5]
                more = len(failed) - len(preview)
                items_str = ", ".join(f"'{x}'" for x in preview)
                if more > 0:
                    items_str += f" (+{more} más)"
                msg += (
                    f"\n\n⚠️ Aviso para el asistente: no pude descontar {len(failed)} "
                    f"ingrediente(s) de la nevera por falta de cantidad parseable: "
                    f"{items_str}. Si el usuario quiere precisión, pídele que confirme "
                    f"cuánto consumió (ej. '1 taza', '100g', '2 cdas') y vuelve a "
                    f"llamar log_consumed_meal con la cantidad incluida en el string "
                    f"del ingrediente."
                )
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
    # [P1-SWAP-MACROS · 2026-05-22] Targets per-meal: usamos los macros del
    # plato original como objetivo del modificado (preservación de presupuesto
    # nutricional del slot). El validador post-gen rechaza drift >15% por
    # macro (>22% en cals). Si el plato original no tiene esos campos
    # (planes muy legacy), los valores caen a 0 y el validador skip per-key.
    original_protein = target_meal.get("protein") or 0
    original_carbs = target_meal.get("carbs") or 0
    original_fats = target_meal.get("fats") or target_meal.get("fat") or 0

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
    
    # [P0-UPDATE-CLINICAL-GUARD · 2026-06-23] Cargar alergias + dieta SERVER-SIDE desde el
    # health_profile del user_id (que ya viene FORZADO al valor autenticado por el override de
    # execute_tools, P0-AGENT-1 — NUNCA del cliente). Pre-fix: MODIFY_MEAL_PROMPT_TEMPLATE no
    # tenía campo de alergia y su instrucción #1 ordenaba obedecer el cambio pedido → un alérgico
    # que pedía "cámbiale el pollo por camarones" obtenía camarones persistidos. Inyectamos un
    # bloque de seguridad como instrucción #0 (por encima del cambio) + el backstop determinista
    # corre antes del persist. tooltip-anchor: P0-UPDATE-CLINICAL-GUARD
    _clin_allergies = []
    _clin_diet = None
    _hp = {}
    try:
        from db import get_user_profile as _get_profile
        _hp = (_get_profile(user_id) or {}).get("health_profile") or {}
    except Exception as _hp_load_e:
        logger.warning(f"⚠️ [P0-UPDATE-CLINICAL-GUARD] no se cargó perfil (no bloquea): {_hp_load_e}")

    if UPDATE_CLINICAL_GUARD:
        try:
            _clin_allergies = _hp.get("allergies") or []
            _clin_diet = _hp.get("dietType") or _hp.get("diet_type")
            if form_data:  # union con lo que mande el caller (defensa-en-profundidad)
                _clin_allergies = list({*[str(a) for a in _clin_allergies], *[str(a) for a in (form_data.get("allergies") or [])]})
                _clin_diet = _clin_diet or form_data.get("dietType") or form_data.get("diet_type")
            _clin_bits = []
            if _clin_allergies:
                _clin_bits.append(f"ALERGIAS DECLARADAS (PROHIBIDO INCLUIR estos alimentos o sus derivados): {', '.join(str(a) for a in _clin_allergies)}")
            if _clin_diet and str(_clin_diet).strip().lower() not in ("balanced", "balanceada", "omnivoro", "omnívoro", "omnivora", ""):
                _clin_bits.append(f"DIETA: {_clin_diet} — respeta sus restricciones de alimentos de origen animal.")
            if _clin_bits:
                context_extras = (
                    "\n🛑 SEGURIDAD CLÍNICA (REGLA #0, POR ENCIMA DEL CAMBIO PEDIDO): "
                    + " ".join(_clin_bits)
                    + " Si el cambio solicitado exige un alimento prohibido, NO lo apliques: propón "
                      "una alternativa segura.\n"
                    + context_extras
                )
        except Exception as _clin_load_e:
            logger.warning(f"⚠️ [P0-UPDATE-CLINICAL-GUARD] no se procesó perfil clínico (no bloquea): {_clin_load_e}")

    # [P1-UPDATE-SUPERPERS · 2026-06-23] (audit inteligencia P1-4) Inyectar súper-personalización
    # (gustos/cocina/religión/equipo) al prompt del modify — paridad con S1 y con swap.
    if _hp and os.environ.get("MEALFIT_UPDATE_SUPERPERS", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from prompts.plan_generator import build_super_personalization_context
            _sp_block = build_super_personalization_context({"health_profile": _hp})
            if _sp_block:
                context_extras = "\n" + _sp_block.strip() + "\n" + context_extras
        except Exception as _sp_e:
            logger.debug(f"[P1-UPDATE-SUPERPERS] super-pers context (modify) falló: {_sp_e}")

    # [P1-UPDATE-MICROS · 2026-06-23] (audit inteligencia P1-7) Inyectar directivas de condición +
    # fármaco-alimento al prompt del modify — paridad con S1.
    if _hp and os.environ.get("MEALFIT_UPDATE_CONDITION_DIRECTIVES", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from condition_rules import build_condition_prompt
            from medication_rules import build_medication_prompt
            _cond_form = {
                "medicalConditions": _hp.get("medicalConditions") or _hp.get("medical_conditions") or [],
                "medications": _hp.get("medications") or [],
            }
            _cond_block = build_condition_prompt(_cond_form)
            if _cond_block:
                context_extras = "\n" + _cond_block.strip() + "\n" + context_extras
            _med_block = build_medication_prompt(_cond_form)
            if _med_block:
                context_extras = "\n" + _med_block.strip() + "\n" + context_extras
        except Exception as _cond_e:
            logger.debug(f"[P1-UPDATE-MICROS] directivas condición/fármaco (modify) fallaron: {_cond_e}")

    # [P2-CHATMODIFY-DISLIKES · 2026-06-24] (re-audit P2-6) Inyectar los DISGUSTOS como prohibición dura
    # (espejo de swap agent.py:548). chat-modify NUNCA los leía → un "cámbiame el almuerzo" podía
    # reintroducir un alimento marcado "no me gusta" (S1/swap sí lo respetan). UNION _hp+form_data
    # (defensa-en-profundidad, igual que allergies). El template ya honra la EXCEPCIÓN: si el usuario pide
    # explícitamente el alimento en `changes`, su deseo reciente gana. Knob MEALFIT_UPDATE_HYDRATE_DISLIKES.
    # tooltip-anchor: P2-CHATMODIFY-DISLIKES
    if os.environ.get("MEALFIT_UPDATE_HYDRATE_DISLIKES", "true").strip().lower() in ("1", "true", "yes", "on"):
        try:
            _cm_dislikes = list({
                *[str(d).strip() for d in (_hp.get("dislikes") or []) if str(d).strip()],
                *[str(d).strip() for d in ((form_data or {}).get("dislikes") or []) if str(d).strip()],
            })
            if _cm_dislikes:
                context_extras = (
                    "\n🚫 DISGUSTOS (PROHIBIDO INCLUIR estos alimentos ni como sustituto, salvo que el "
                    "usuario los pida EXPLÍCITAMENTE en el cambio): " + ", ".join(_cm_dislikes) + ".\n"
                    + context_extras
                )
        except Exception as _cm_dis_e:
            logger.debug(f"[P2-CHATMODIFY-DISLIKES] inyección de dislikes (modify) falló: {_cm_dis_e}")

    # [P2-CHATMODIFY-GAINMUSCLE-DENSITY · 2026-06-24] (re-audit P2-7) Para gain_muscle, prohibir como
    # PROTEÍNA PRINCIPAL las de baja densidad (leguminosas/ricotta/cottage/crema/yogurt regular) — ante un
    # cambio abierto el LLM (temp 0.1) podía devolver un main de baja densidad que swap/S1 filtran vía
    # _LOW_DENSITY_AS_MAIN → día bajo el piso de proteína. Honra la regla #1 del template (deseo explícito
    # gana). Knob compartido MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN. tooltip-anchor: P2-CHATMODIFY-GAINMUSCLE-DENSITY
    _cm_goal = (
        (_hp.get("goal") or _hp.get("mainGoal") or ((form_data or {}).get("goal") if form_data else "") or "")
    ).strip().lower()
    if (
        _cm_goal == "gain_muscle"
        and os.environ.get("MEALFIT_GAINMUSCLE_HIGH_DENSITY_PROTEIN", "true").strip().lower() in ("1", "true", "yes", "on")
    ):
        try:
            from ai_helpers import _LOW_DENSITY_AS_MAIN as _LDM_CM
            _ldm_names = ", ".join(sorted(str(x) for x in _LDM_CM))
            context_extras = (
                "\n💪 OBJETIVO GANAR MÚSCULO — PROTEÍNA PRINCIPAL: usa proteína animal de ALTA densidad "
                "(pollo, carne, pescado, huevos, atún). NO uses como BASE PRINCIPAL del plato proteínas de "
                "baja densidad (" + _ldm_names + ") salvo que el usuario lo pida explícitamente en el cambio.\n"
                + context_extras
            )
        except Exception as _cm_gm_e:
            logger.debug(f"[P2-CHATMODIFY-GAINMUSCLE-DENSITY] directiva densidad (modify) falló: {_cm_gm_e}")

    # [P1-SLOT-APPROPRIATENESS · 2026-06-27] (audit G4) Guía de coherencia de HORARIO al prompt de
    # chat-modify (ADVISORY, no backstop: si el usuario pide explícitamente algo fuera de horario su
    # deseo gana — el item 1 del template manda "Aplica EXACTAMENTE el cambio… DEBES incluirlo").
    # SSOT constants.build_meal_timing_rules. Paridad de PROMPT con swap S3 / day_generator S1.
    try:
        from constants import build_meal_timing_rules as _bmtr
        _timing_block = _bmtr(meal_type)
        if _timing_block:
            context_extras = _timing_block + "\n" + context_extras
    except Exception as _tr_e:
        logger.debug(f"[P1-SLOT-APPROPRIATENESS] timing rules chat-modify fallaron (no bloquea): {_tr_e}")

    # [P2-UPDATE-MICRO-STEER · 2026-06-27] (audit G2) Pisos de micros (Mg/Fe/Ca/fibra/K) al prompt de
    # chat-modify — paridad de densidad con S1/swap. SKIP en el path pantry-strict (reciclaje: clean_ingredients
    # sin allow_pantry_expansion) para no subir fallos de convergencia con la Nevera. SSOT del orquestador.
    if not (clean_ingredients and not allow_pantry_expansion):
        try:
            from graph_orchestrator import build_update_micronutrient_directive as _bmd
            _mform = dict(form_data or {})
            if not _mform.get("medicalConditions") and not _mform.get("medical_conditions"):
                _mform["medicalConditions"] = _hp.get("medicalConditions") or _hp.get("medical_conditions") or []
            for _k in ("gender", "age"):
                if not _mform.get(_k) and _hp.get(_k):
                    _mform[_k] = _hp.get(_k)
            _micro_block = _bmd(_mform)
            if _micro_block:
                context_extras = _micro_block + "\n" + context_extras
        except Exception as _msc_e:
            logger.debug(f"[P2-UPDATE-MICRO-STEER] micro steer chat-modify falló (no bloquea): {_msc_e}")

    modify_prompt = MODIFY_MEAL_PROMPT_TEMPLATE.format(
        name=target_meal.get('name'),
        desc=target_meal.get('desc'),
        meal=target_meal.get('meal'),
        time=target_meal.get('time'),
        original_cals=original_cals,
        original_protein=int(round(float(original_protein or 0))),
        original_carbs=int(round(float(original_carbs or 0))),
        original_fats=int(round(float(original_fats or 0))),
        ingredients_json=json.dumps(target_meal.get('ingredients', [])),
        changes=changes,
        context_extras=context_extras
    )
    
    # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Knob model + timeout + CB gate +
    # rate-limit discrimination + token telemetry. Espejo del patrón de
    # `agent.py::swap_meal` (P1-CHAT-CB-EXTEND): el CB gatea ANTES del retry
    # loop para que 3 attempts × N concurrentes no agraven la condición.
    # En 429 NO se cuenta como CB failure; en otros errores sí. Token
    # telemetry post-success rellena el blind-spot que tenía este callsite
    # en `llm_usage_events`. Tooltip-anchor: P1-TOOLS-LLM-HARDENING.
    # [P0-DEEPSEEK-MIGRATION] `user_id` viene FORZADO al valor autenticado por
    # el override de execute_tools (P0-AGENT-1) — seguro para tier-routing.
    _modify_model = _tools_modify_meal_model_name(user_id)
    modify_llm = ChatDeepSeek(
        model=_modify_model,
        temperature=0.1,
        timeout=_tools_modify_meal_llm_timeout_s(),
    ).with_structured_output(MealModel)
    # `_modify_llm_for_usage` queda como referencia para telemetría: el
    # objeto con `.model_name` (sin `.with_structured_output(...)`) es el que
    # `_emit_llm_usage_event_best_effort` puede leer para resolver el
    # model_name. Si fallara, el helper retorna sin emit (best-effort).
    _modify_llm_for_usage = ChatDeepSeek(
        model=_modify_model,
        temperature=0.1,
        timeout=_tools_modify_meal_llm_timeout_s(),
    )

    (_is_rl_err, _CBOpen, _RLErr, _emit_rl_metric, _emit_usage_event) = _tools_get_chat_safety_helpers()
    _modify_cb = _get_circuit_breaker(_modify_model)
    if not _modify_cb.can_proceed():
        logger.warning(
            f"🛑 [P1-TOOLS-LLM-HARDENING] execute_modify_single_meal CB abierto "
            f"model={_modify_model!r} — fail-fast sin invocar Gemini. "
            f"Reintentar tras MEALFIT_CB_RESET_TIMEOUT_S segundos."
        )
        raise _CBOpen(
            f"execute_modify_single_meal LLM circuit breaker open for model={_modify_model}"
        )

    current_prompt = [modify_prompt]
    _modify_invoke_start = time.time()

    # [P1-SWAP-MACROS · 2026-05-22] Lazy import del validador de macros
    # + recipe-coherence (mismo patrón que `agent.py::swap_meal`).
    try:
        from nutrition_calculator import (
            validate_meal_macros_against_targets as _validate_macros,
            _meal_macros_validate_enabled as _macros_validate_enabled,
            validate_meal_recipe_ingredients_coherence as _validate_recipe_coh,
            _swap_recipe_coherence_enabled as _recipe_coh_enabled,
        )
    except Exception:
        _validate_macros = None
        _macros_validate_enabled = lambda: False  # noqa: E731
        _validate_recipe_coh = None
        _recipe_coh_enabled = lambda: False  # noqa: E731

    # [P2-UPDATE-MACRO-TRUTHUP · 2026-06-24] (re-audit P2-1) Truth-up de macros desde strings de
    # ingredientes ANTES del band-validator (bloque dentro de invoke_with_retry). Cierra el inflado del
    # JSON por el LLM en chat-modify. db lazy compartida entre reintentos. Knob MEALFIT_UPDATE_MACRO_TRUTHUP
    # default ON. tooltip-anchor: P2-UPDATE-MACRO-TRUTHUP
    _tu_db_holder = [None]
    _update_macro_truthup_enabled = lambda: os.environ.get(  # noqa: E731
        "MEALFIT_UPDATE_MACRO_TRUTHUP", "true").strip().lower() in ("1", "true", "yes", "on")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"🔁 [MODIFY_MEAL RETRY] attempt={retry_state.attempt_number} | "
            f"reason=guardrail_rejection"
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

        # [P1-SWAP-RECIPE-COHERENCE · 2026-05-22] Mini-coherence check
        # per-meal sobre el output del LLM (mismo patrón que swap_meal).
        # Detecta cap_swallowed_modifier al nivel del meal-output: receta
        # menciona "el pollo" pero ingredients=["pavo"]. Gateamos retry
        # tenacity. Knob `MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE=false`
        # desactiva sin redeploy.
        if _validate_recipe_coh is not None and _recipe_coh_enabled():
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                coh_passed, coh_divs, coh_summary = _validate_recipe_coh(meal_dump)
                if not coh_passed:
                    logger.warning(
                        f"⚠️ [P1-SWAP-RECIPE-COHERENCE] divergence in modify_meal | "
                        f"meal_type={meal_type} | divs={coh_divs}"
                    )
                    # [P3-SWAP-RETRY-COHERENCE-HINT · 2026-05-22] Paridad
                    # con `agent.py::swap_meal`: append self-check directive
                    # al retry prompt para subir señal al LLM. Ver memoria
                    # del P-fix para razón (3 intentos consecutivos con
                    # mismo alias en log productivo 2026-05-22).
                    current_prompt[0] = modify_prompt + (
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

        # [P2-UPDATE-MACRO-TRUTHUP · 2026-06-24] (re-audit P2-1) Recompute del NÚMERO de macros desde los
        # strings FINALES de ingredientes ANTES del band-validator → cierra el inflado del JSON por el LLM
        # (chat-modify persiste el plan él mismo → la cifra fantasma quedaba commiteada). Espejo del Guard
        # 8z de S1. Solo NÚMEROS (NO strings → lista de compras intacta). Fail-safe. Mutamos `res` para que
        # el band-validator y el merge persistente lean la cifra real. tooltip-anchor: P2-UPDATE-MACRO-TRUTHUP
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
                    logger.info("🔎 [P2-UPDATE-MACRO-TRUTHUP] macros chat-modify recomputadas desde strings")
            except Exception as _tu_exc:
                logger.warning(
                    f"[P2-UPDATE-MACRO-TRUTHUP] truth-up chat-modify falló (no aborta): "
                    f"{type(_tu_exc).__name__}: {_tu_exc}"
                )

        # [P1-SWAP-MACROS · 2026-05-22] Validación post-gen de macros vs
        # el plato original. Mismo patrón que `agent.py::swap_meal`:
        # si drift > tolerancia → inyectamos summary al retry prompt y
        # forzamos tenacity retry. Knob `MEALFIT_SWAP_MACROS_VALIDATE=false`
        # desactiva sin redeploy.
        if _validate_macros is not None and _macros_validate_enabled():
            try:
                meal_dump = res.model_dump() if hasattr(res, "model_dump") else (
                    res if isinstance(res, dict) else {}
                )
                passed, drifts, summary = _validate_macros(
                    meal_dump,
                    {
                        "cals": original_cals,
                        "protein": original_protein,
                        "carbs": original_carbs,
                        "fats": original_fats,
                    },
                )
                if not passed:
                    logger.warning(
                        f"⚠️ [P1-SWAP-MACROS] Drift en modify_meal | "
                        f"meal_type={meal_type} | drifts={drifts}"
                    )
                    current_prompt[0] = modify_prompt + (
                        f"\n\n🛑 ATENCIÓN AL INTENTO FALLIDO ANTERIOR:\n{summary}"
                    )
                    raise ValueError(summary)
            except ValueError:
                raise
            except Exception as _macros_exc:
                logger.warning(
                    f"[P1-SWAP-MACROS] validator helper falló (no aborta): "
                    f"{type(_macros_exc).__name__}: {_macros_exc}"
                )

        return res

    try:
        try:
            new_meal_response = invoke_with_retry()
            # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Success path del retry
            # loop. Marca CB success + emit token telemetry post-éxito.
            # El helper acepta `result` del `.with_structured_output(MealModel)`
            # (sigue exponiendo `usage_metadata` proveniente del AIMessage
            # subyacente vía LangChain).
            _modify_cb.record_success()
            try:
                _emit_usage_event(
                    llm=_modify_llm_for_usage,
                    result=new_meal_response,
                    duration_s=time.time() - _modify_invoke_start,
                    node='tool_modify_single_meal',
                )
            except Exception:
                pass
        except Exception as e:
            # [P1-TOOLS-LLM-HARDENING · 2026-05-20] Discriminar 429 vs
            # failures genuinos ANTES de marcar CB failure. Si las 3
            # attempts del tenacity fallan por 429, NO contamos como CB
            # failure (el provider está vivo, solo throttling). Re-emit
            # como `LLMRateLimitedError` que el caller (chat router via
            # execute_tools → call_model exception path) puede mapear a
            # HTTP 429. Resto (timeout, 5xx, ValidationError del guardrail)
            # → CB failure + mantener fallback abortivo existente como
            # degradación graceful (correctness preservada).
            if _is_rl_err(e):
                _emit_rl_metric(user_id, None, _modify_model)
                logger.warning(
                    f"⚠️ [P1-CHAT-LLM-429] execute_modify_single_meal rate-limit "
                    f"model={_modify_model!r} — NO cuenta como CB failure."
                )
                raise _RLErr(
                    f"execute_modify_single_meal LLM rate limited for model={_modify_model}: {e!r}"
                ) from e
            # [P2-CB-GUARDRAIL-NOT-FAILURE · 2026-06-26] (espejo de agent.py::swap_meal) Un rechazo de
            # validador/guardrail en chat-modify (pantry L845 / recipe-coherence L878 / macro-drift L942
            # → ValueError) significa que el PROVEEDOR respondió pero el output no pasó NUESTRA validación
            # — NO es señal de salud del proveedor. El breaker `_modify_cb` es per-modelo COMPARTIDO
            # cross-worker; contar un guardrail como fallo lo abre por un plato "difícil" y deja
            # execute_modify_single_meal en fail-fast ~30s para TODO el tier con DeepSeek sano (mismo modo
            # de fallo que cerró el fix hermano el 2026-06-24, pero por la superficie del chat). Solo
            # timeout/5xx/conexión (no-ValueError) cuentan como CB failure. Knob
            # MEALFIT_MODIFY_CB_COUNT_GUARDRAIL=true revierte. tooltip-anchor: P2-CB-GUARDRAIL-NOT-FAILURE
            _cb_count_guardrail = os.environ.get(
                "MEALFIT_MODIFY_CB_COUNT_GUARDRAIL", "false").strip().lower() in ("1", "true", "yes", "on")
            if isinstance(e, ValueError) and not _cb_count_guardrail:
                logger.info(
                    f"🎚 [P2-CB-GUARDRAIL-NOT-FAILURE] rechazo de guardrail en chat-modify NO cuenta "
                    f"como CB failure (proveedor sano) | meal_type={meal_type}"
                )
            else:
                _modify_cb.record_failure()
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

        # [P0-UPDATE-CLINICAL-GUARD · 2026-06-23] Backstop determinista ANTES de persistir: el chat
        # modify NO pasa por el grafo (ni reviewer médico ni capa clínica). Si el plato generado
        # introduce un alérgeno declarado o un producto veg*-prohibido, NO lo persistimos —
        # fail-secure abortivo (mismo patrón que "FALLO POR INVENTARIO INSUFICIENTE"): el plato
        # original se mantiene intacto y el agente informa al usuario. tooltip-anchor: P0-UPDATE-CLINICAL-GUARD
        if UPDATE_CLINICAL_GUARD:
            try:
                # [P1-MERCURY-UPDATE-GUARD · 2026-06-24] El backstop necesita el perfil clínico para
                # detectar mercurio-embarazo. Construimos un form con `medicalConditions` del health_profile
                # server-side (no del cliente) para que `_is_pregnancy_or_lactation` lo vea aunque el chat
                # no lo haya enviado en `form_data`.
                _clin_form = dict(form_data or {})
                if not _clin_form.get("medicalConditions") and not _clin_form.get("medical_conditions"):
                    _clin_form["medicalConditions"] = _hp.get("medicalConditions") or _hp.get("medical_conditions") or []
                _clin_viol = clinical_backstop_for_meal(
                    new_meal_data, allergies=_clin_allergies, diet_type=_clin_diet, form_data=_clin_form
                )
            except Exception as _clin_exc:
                _clin_viol = [f"error backstop clínico: {type(_clin_exc).__name__}"]
            if _clin_viol:
                logger.warning(
                    f"🛡 [P0-UPDATE-CLINICAL-GUARD] modify_single_meal viola seguridad clínica → "
                    f"abort fail-secure | day={day_number} meal={meal_type} | viol={_clin_viol}"
                )
                return (
                    "FALLO POR SEGURIDAD CLÍNICA: el cambio solicitado introduciría un alimento que "
                    "el usuario NO puede consumir (" + "; ".join(_clin_viol) + "). El cambio fue "
                    "revertido y el plato original se mantuvo intacto. Informa al usuario amablemente "
                    "que no puedes incluir ese alimento por sus alergias o restricciones de dieta, y "
                    "ofrécele una alternativa segura."
                )

        # [P1-RENAL-UPDATE-ENFORCE · 2026-06-24] (re-audit P1-1) Si el plan lleva cap renal KDIGO, trima la
        # proteína del plato modificado al techo del slot (`original_protein`, ya renal-aware porque el plan
        # se capeó en S1) antes de persistir → un modify por chat no rompe el techo iatrogénico. No-op si el
        # plan no es renal; best-effort (no bloquea). Gateado internamente por RENAL_CAP_ENABLED.
        try:
            if bool((plan_data.get("renal_protein_cap") or {}).get("applied")) and original_protein:
                renal_protein_trim_for_update([new_meal_data], float(original_protein or 0), renal_capped=True)
        except Exception as _renal_e:
            logger.warning(f"[P1-RENAL-UPDATE-ENFORCE] trim renal en modify falló (no bloquea): {type(_renal_e).__name__}: {_renal_e}")

        # [P2-FOOD-SAFETY-UPDATE · 2026-06-24] (re-audit P2-1) Mitigación de seguridad alimentaria (huevo/
        # pescado-marisco crudos) — el chat-modify no pasa por el grafo. Macro-preservante, fail-open.
        try:
            food_safety_backstop_for_meal(new_meal_data)
        except Exception as _fs_e:
            logger.warning(f"[P2-FOOD-SAFETY-UPDATE] food-safety en modify falló (no bloquea): {type(_fs_e).__name__}: {_fs_e}")

        # [P2-UPDATE-CONDITION-SUBST · 2026-06-26] (audit 3-flujos P2) Sustitución determinista por condición
        # médica (DM2/HTA/dislipidemia) — paridad con el Guard 3 de S1. El chat no envía el wizard form →
        # enriquecemos medicalConditions desde el health_profile server-side (_hp), igual que el backstop
        # clínico de arriba. Macro-preservante, idempotente, fail-open (advisory, no aborta el modify).
        try:
            _cond_form = dict(form_data or {})
            if not _cond_form.get("medicalConditions") and not _cond_form.get("medical_conditions"):
                _cond_form["medicalConditions"] = _hp.get("medicalConditions") or _hp.get("medical_conditions") or []
            condition_substitution_backstop_for_meal(new_meal_data, _cond_form)
        except Exception as _cs_e:
            logger.warning(f"[P2-UPDATE-CONDITION-SUBST] condition-subst en modify falló (no bloquea): {type(_cs_e).__name__}: {_cs_e}")

        # [P1-SWAP-PORTION-CAP · 2026-06-27] (paridad S1↔S4 chat-modify) Caps de porción DETERMINISTAS — DM2
        # (almidón alto-IG) + bariátrica (queso ≤30g / yogurt ≤120g / fruta / aguacate / frutos secos + volumen del
        # pouch). El chat-modify, igual que swap_meal, tenía TODOS los backstops MENOS los caps de porción → el LLM
        # podía colar porciones de riesgo (5 lonjas de queso) al modificar un plato por chat. Solo RECORTAN (recuperan
        # kcal escalando otros ingredientes → macro-safe); como el recorte de lácteo baja proteína, RE-CERRAMOS el
        # piso del slot con proteína animal densa NO-láctea (espejo FASE A, max_add_g=90; renal → skip KDIGO).
        # Idempotente, fail-open. tooltip-anchor: P1-SWAP-PORTION-CAP
        try:
            from graph_orchestrator import (cap_dm2_high_gi_portions as _cap_dm2_m,
                                            cap_bariatric_portions as _cap_baria_m,
                                            _close_protein_gap_for_meal as _close_pc_m,
                                            _safe_high_density_proteins as _safe_pc_m)
            from nutrition_db import IngredientNutritionDB as _CapDBm
            _cap_form_m = dict(form_data or {})
            if not _cap_form_m.get("medicalConditions") and not _cap_form_m.get("medical_conditions"):
                _cap_form_m["medicalConditions"] = _hp.get("medicalConditions") or _hp.get("medical_conditions") or []
            _cap_db_m = _CapDBm()
            _wrap_m = [{"meals": [new_meal_data]}]
            _ndm = _cap_dm2_m(_wrap_m, _cap_form_m, _cap_db_m)
            _nbm = _cap_baria_m(_wrap_m, _cap_form_m, _cap_db_m)
            _is_renal_m = bool((plan_data.get("renal_protein_cap") or {}).get("applied"))
            if (_ndm or _nbm) and original_protein and not _is_renal_m:
                _cur_pm = float(new_meal_data.get("protein") or 0)
                if _cur_pm < 0.90 * float(original_protein):
                    new_meal_data["_protein_closed"] = False
                    _cands_pm = [c for c in _safe_pc_m(_clin_allergies, _cap_db_m, min_protein=18.0)
                                 if not any(_t in str(c[1]).lower()
                                            for _t in ("queso", "yogur", "leche", "ricotta", "cottage", "requeson"))]
                    if _cands_pm:
                        _close_pc_m(new_meal_data, float(original_protein), _cap_db_m, _cands_pm, max_add_g=90)
            if _ndm or _nbm:
                logger.info(f"🔒 [P1-SWAP-PORTION-CAP] plato de modify recortado: cap_dm2={_ndm} "
                            f"cap_baria={_nbm} | day={day_number} meal={meal_type}")
        except Exception as _pcm_e:
            logger.warning(f"[P1-SWAP-PORTION-CAP] cap de porción en modify falló (no bloquea): {type(_pcm_e).__name__}: {_pcm_e}")

        # [P1-UPDATE-APPETIBILITY · 2026-06-27] (audit Fase 0) Honestidad de nombre (proteína fantasma) +
        # detección de pareo fruta+salado — paridad con S1/swap. namefix determinista e idempotente; el clash
        # solo se loguea (advisory: el chat-modify no tiene retry barato y el cambio lo dirige el usuario).
        try:
            _appet = appetibility_fix_for_update(new_meal_data)
            if _appet.get("name_fixed"):
                logger.info(f"🎭 [P1-UPDATE-APPETIBILITY] nombre de modify corregido (proteína fantasma) | day={day_number} meal={meal_type}")
            if _appet.get("sweet_savory_clash"):
                logger.warning(f"🍓 [P1-UPDATE-APPETIBILITY] modify mantiene pareo fruta+salado (advisory) | day={day_number} meal={meal_type}")
        except Exception as _ap_e:
            logger.warning(f"[P1-UPDATE-APPETIBILITY] appetibility fix en modify falló (no bloquea): {type(_ap_e).__name__}: {_ap_e}")

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
            # [P3-CANONICAL-AGG-WEEKLY · 2026-05-18] is_new_plan=True garantiza
            # que agg_weekly almacene la lista canónica (full needs), no el
            # delta contra inventario. La deducción se hace at-render-time en
            # el frontend (Dashboard.buildDeltaShoppingList). Cierre del bug
            # "agotar+reponer rompe PDF" — mismo invariante que en
            # /recalculate-shopping-list (plans.py).
            aggr_7 = get_shopping_list_delta(
                user_id, plan_data, is_new_plan=True, structured=True, multiplier=1.0 * household,
                inventory_override=_inv_s, consumed_override=_cons_s,
            )
            aggr_15 = get_shopping_list_delta(
                user_id, plan_data, is_new_plan=True, structured=True, multiplier=2.0 * household,
                inventory_override=_inv_s, consumed_override=_cons_s,
            )
            aggr_30 = get_shopping_list_delta(
                user_id, plan_data, is_new_plan=True, structured=True, multiplier=4.0 * household,
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
        
        # 5. Actualizar en Neon atómicamente bajo FOR UPDATE row lock.
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

            # [P2-CHATMODIFY-MICROS-STALE · 2026-06-24] (re-audit P2-2) Recomputar el panel de micros sobre
            # el plan mutado → Dashboard/PDF no quedan stale. chat-modify PERSISTE el plan él mismo (este
            # callback) sin etapa downstream que refresque micros (a diferencia del swap, cuyo recalc
            # client-side al menos corre después). Espejo de regenerate-day P1-7. `_hp` ya cargado en el
            # closure (health_profile, ~tools.py:640). Best-effort. Knob compartido MEALFIT_UPDATE_RECOMPUTE_MICROS.
            # tooltip-anchor: P2-CHATMODIFY-MICROS-STALE
            if os.environ.get("MEALFIT_UPDATE_RECOMPUTE_MICROS", "true").strip().lower() in ("1", "true", "yes", "on"):
                try:
                    from graph_orchestrator import recompute_micronutrient_report_for_plan
                    _micro_form_cm = {
                        "gender": _hp.get("gender") or _hp.get("sex"),
                        "age": _hp.get("age"),
                        "medicalConditions": _hp.get("medicalConditions") or _hp.get("medical_conditions"),
                        "medications": _hp.get("medications"),
                    }
                    recompute_micronutrient_report_for_plan(plan_data_fresh, _micro_form_cm, db=None)
                except Exception as _cm_micro_e:
                    logger.debug(f"[P2-CHATMODIFY-MICROS-STALE] recompute (chat-modify) falló: {_cm_micro_e}")

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
            # [P3-GENCHUNK-SPEED · 2026-06-01] Incluir el `plan_data` ya mergeado
            # (autoridad fresh-post-lock que `update_plan_data_atomic` retorna)
            # para que `execute_tools` NO re-SELECTee el plan vía
            # `get_latest_meal_plan_with_id` justo después de escribirlo (round-trip
            # serial redundante en el critical path del chat). `execute_tools`
            # parsea esta key, hidrata `new_plan`, y LUEGO pisa `tool_result` con
            # un string amistoso → el blob NO llega al LLM (cero token bloat).
            _resp["plan_data"] = merged_plan_data
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

# [P1-SEARCH-DEEP-MEMORY-CACHE · 2026-05-19] TTL cache in-process para
# resultados de `search_deep_memory`. Pre-fix: cada invocación de la tool
# (a) hace `get_user_profile(user_id)` para validar el toggle LTM, (b)
# invoca `db_search_deep_memory(user_id, query, limit=5)` que ejecuta
# embedding + cosine search sobre `user_facts` + summary lookup en
# `conversation_summaries`.
#
# Multi-turn conversations donde el LLM invoca la tool repetidamente
# (e.g., "¿qué comía al principio?" → "¿y al mes siguiente?" → "¿cuándo
# bajé el carbo?") generan N queries casi idénticas en cuestión de
# segundos. Sin cache: N × (~150-400ms) latencia + N × cost de embedding
# + N × pgvector scan. p95 de la tool crece linealmente con N.
#
# Diseño:
#   - Cache key = (user_id, query_normalized). `query_normalized` =
#     `query.strip().lower()[:256]` — colapsa "Adherencia", "ADHERENCIA",
#     "adherencia " a la misma key. Cap a 256 chars defensivo (queries
#     legítimas son <50 chars; un cap >= 256 cubre 100% del uso real).
#   - TTL = 300s (5 min). Suficiente para conversación típica (~5-15 min
#     sesión activa) pero corto enough para que cambios en `user_facts`
#     se reflejen en la siguiente sesión.
#   - Maxsize = 1024 entries. Cada entry típicamente <5KB (5 summaries de
#     ~1KB c/u) → 5MB máximo. Eviction LRU implícita via cleanup en hit:
#     cuando un lookup encuentra entries expiradas, las descarta.
#   - In-process: cada worker tiene su propio cache. Esto es OK (igual
#     que `LLMCircuitBreaker._local_cache`); workers convergen rápido.
#   - NO uso `functools.lru_cache` porque NO maneja TTL y NO permite
#     invalidación selectiva (futuro: cron sweep / signal post-INSERT).
#
# Knob:
#   - `MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S` (default 300, clamp [0, 3600]).
#     Setear a 0 desactiva el cache (sin redeploy). Auto-registrado vía
#     `_env_int_safe_tools` (helper local, NO usamos `_env_int` del knobs
#     module aquí porque tools.py es importado por graph_orchestrator y
#     queremos evitar acoplamiento circular hasta haber audit completo
#     del import graph — mismo razonamiento que `rate_limiter.py`).
#
# Tooltip-anchor: P1-SEARCH-DEEP-MEMORY-CACHE.
_SEARCH_DEEP_MEMORY_CACHE: dict = {}
_SEARCH_DEEP_MEMORY_CACHE_MAX_ENTRIES = 1024
_SEARCH_DEEP_MEMORY_CACHE_TTL_S_DEFAULT = 300


def _env_int_safe_tools(name: str, default: int) -> int:
    """Lectura defensiva de env var entero. NO registra en `_KNOBS_REGISTRY`
    para evitar ciclo con `graph_orchestrator`. Misma razón que
    `rate_limiter._env_int_safe`. Tooltip-anchor: P1-SEARCH-DEEP-MEMORY-CACHE."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _search_deep_memory_cache_ttl_s() -> int:
    """[P1-SEARCH-DEEP-MEMORY-CACHE · 2026-05-19] TTL actual con clamp
    [0, 3600]. `0` desactiva el cache."""
    raw = _env_int_safe_tools(
        "MEALFIT_SEARCH_DEEP_MEMORY_CACHE_TTL_S",
        _SEARCH_DEEP_MEMORY_CACHE_TTL_S_DEFAULT,
    )
    if raw < 0:
        return 0
    if raw > 3600:
        return 3600
    return raw


def _search_deep_memory_cache_get(user_id: str, query_norm: str):
    """[P1-SEARCH-DEEP-MEMORY-CACHE · 2026-05-19] Lookup con TTL check.
    Retorna `(hit, value)`: `(True, str)` si hit fresco, `(False, None)`
    si miss o expired (también purga la entry expirada del dict)."""
    ttl = _search_deep_memory_cache_ttl_s()
    if ttl <= 0:
        return False, None
    key = (user_id, query_norm)
    entry = _SEARCH_DEEP_MEMORY_CACHE.get(key)
    if entry is None:
        return False, None
    cached_at, value = entry
    if (time.monotonic() - cached_at) >= ttl:
        # Expired — purge.
        _SEARCH_DEEP_MEMORY_CACHE.pop(key, None)
        return False, None
    return True, value


def _search_deep_memory_cache_set(user_id: str, query_norm: str, value: str) -> None:
    """[P1-SEARCH-DEEP-MEMORY-CACHE · 2026-05-19] Store con LRU defensivo:
    si el cache excede `_MAX_ENTRIES`, purga ~10% de las más viejas. NO
    es LRU estricto (no rastreamos uso) — es "MRU-keep" basado en
    `cached_at`, equivalente práctico para nuestro workload (uso
    transient en ventana corta)."""
    ttl = _search_deep_memory_cache_ttl_s()
    if ttl <= 0:
        return
    now = time.monotonic()
    if len(_SEARCH_DEEP_MEMORY_CACHE) >= _SEARCH_DEEP_MEMORY_CACHE_MAX_ENTRIES:
        # Eviction: drop ~10% más viejas. Sort by cached_at ascending.
        try:
            sorted_keys = sorted(
                _SEARCH_DEEP_MEMORY_CACHE.items(),
                key=lambda kv: kv[1][0],
            )
            to_drop = max(1, len(sorted_keys) // 10)
            for k, _ in sorted_keys[:to_drop]:
                _SEARCH_DEEP_MEMORY_CACHE.pop(k, None)
        except Exception:
            # Defensive: si el sort falla por race, no afectar la store.
            pass
    _SEARCH_DEEP_MEMORY_CACHE[(user_id, query_norm)] = (now, value)


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

    # [P1-SEARCH-DEEP-MEMORY-CACHE · 2026-05-19] Cache lookup. La key
    # normaliza la query (strip + lower + cap 256) para colapsar
    # variantes equivalentes ("Adherencia", "adherencia", etc) a la
    # misma entry. Cache miss → query DB y store; hit → retorna directo.
    _query_norm = (query or "").strip().lower()[:256]
    _hit, _cached = _search_deep_memory_cache_get(user_id, _query_norm)
    if _hit:
        logger.info(f"[P1-SEARCH-DEEP-MEMORY-CACHE] hit user={user_id} q='{_query_norm[:40]}'")
        return _cached

    results = db_search_deep_memory(user_id, query, limit=5)

    if not results:
        _empty = "No se encontraron recuerdos históricos que coincidan con esa búsqueda. Es posible que aún no haya suficiente historial archivado."
        # Cachear también el resultado vacío — evita re-scanear pgvector
        # cuando el LLM insiste con la misma query (caso común en
        # debug/exploración).
        _search_deep_memory_cache_set(user_id, _query_norm, _empty)
        return _empty

    # Formatear los resultados para el agente
    formatted = []
    for idx, r in enumerate(results, 1):
        period = f"{r.get('messages_start', '?')} → {r.get('messages_end', '?')}"
        summary = r.get('summary', 'Sin contenido')
        formatted.append(f"📁 Recuerdo #{idx} (Período: {period}):\n{summary}")

    _result = "\n\n".join(formatted)
    _search_deep_memory_cache_set(user_id, _query_norm, _result)
    return _result

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
def modify_pantry_inventory(user_id: str, items_to_add: list[str] = None, items_to_remove: list[str] = None, items_to_deplete: list[str] = None) -> str:
    """
    Modifica la despensa física del usuario (user_inventory). 3 paths:

    - `items_to_add`: SUMA al inventario. Usa cuando el usuario diga
      "compré X", "añade X a mi nevera". Formato: '2 unidades de Manzana',
      '500 g de Arroz'.

    - `items_to_deplete`: marca el ingrediente como AGOTADO (aparece en la
      sección "Agotados" de la app, listo para re-comprar). Usa cuando el
      usuario diga "se me acabó X", "ya no tengo X", "se terminó el X".
      Formato: nombre del ingrediente — la cantidad se infiere del inventario
      actual o se asume porción típica. Ejemplos: 'leche', 'arroz', 'queso'.
      NO usar para items que el usuario "botó" o "se dañaron" (usa
      items_to_remove para esos).

    - `items_to_remove`: BORRA definitivamente del inventario sin marcarlo
      como agotado. Usa solo cuando el usuario diga "bota X", "se me dañó
      el X", "elimina X de mi nevera". Es destrucción real, no consumo.

    Parámetros:
    - items_to_add: Lista de strings con cantidad+unidad+ingrediente a sumar.
    - items_to_remove: Lista de strings a eliminar definitivamente (dañado/botado).
    - items_to_deplete: Lista de strings (nombres) a marcar como agotados (se acabaron).
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
    #
    # [P3-AGENT-DEPLETE · 2026-05-22] `items_to_deplete` snapshots la cantidad
    # actual del inventario antes de eliminar la fila — el snapshot se inyecta
    # en el ToolMessage via marker `<<PANTRY_DEPLETED_JSON: [...]>>` que
    # `agent.py:execute_tools` extrae y propaga al state. AgentPage.jsx lo
    # escribe a `localStorage.mealfit_depleted_items` para que Pantry.jsx
    # lo muestre en la sección "Agotados". Tooltip-anchor: P3-AGENT-DEPLETE.

    logger.info(f"🛒 [TOOL EXECUTION] Modificando despensa física manual para user {user_id}")
    try:
        import json as _json
        import unicodedata as _ucd
        from datetime import datetime as _dt, timezone as _tz
        from db_inventory import (
            add_or_update_inventory_item,
            deduct_consumed_meal_from_inventory,
            get_raw_user_inventory,
        )
        from shopping_calculator import _parse_quantity

        added_count = 0
        removed_count = 0
        depleted_count = 0
        depleted_payload: list[dict] = []

        if items_to_add:
            for item in items_to_add:
                qty, unit, name = _parse_quantity(item)
                if name and qty > 0:
                    add_or_update_inventory_item(user_id, name, qty, unit)
                    added_count += 1

        if items_to_deplete:
            # [P3-AGENT-DEPLETE · 2026-05-22 · upgrade P3-DEPLETED-BD · 2026-05-22]
            # Snapshot + delete inventory row + INSERT a `user_depleted_items` BD
            # (cross-device sync). Pre-fix (P3-AGENT-DEPLETE) emitía marker JSON
            # inline para que AgentPage.jsx hiciera merge a localStorage — eso
            # limitaba el feature a un solo browser. Ahora la BD es la fuente
            # de verdad, el localStorage es solo cache local del frontend.
            from db_inventory import add_depleted_item as _bd_add_depleted
            current_rows = get_raw_user_inventory(user_id)

            def _strip_lower(s: str) -> str:
                nfd = _ucd.normalize("NFD", str(s or ""))
                return "".join(c for c in nfd if _ucd.category(c) != "Mn").lower().strip()

            row_by_name: dict[str, dict] = {}
            for r in current_rows:
                key = _strip_lower(r.get("ingredient_name", ""))
                if key:
                    row_by_name[key] = r

            now_iso = _dt.now(_tz.utc).isoformat()
            for item in items_to_deplete:
                try:
                    _q, _u, parsed_name = _parse_quantity(item)
                except Exception:
                    parsed_name = item
                lookup_key = _strip_lower(parsed_name) or _strip_lower(item)
                row = row_by_name.get(lookup_key)
                if not row:
                    for key, candidate_row in row_by_name.items():
                        if lookup_key and (lookup_key in key or key in lookup_key):
                            row = candidate_row
                            break
                if not row:
                    logger.info(
                        f"🪫 [P3-AGENT-DEPLETE] '{item}' no encontrado en pantry — "
                        f"skip (puede ya estar agotado o no haberse comprado)."
                    )
                    continue
                qty_snapshot = float(row.get("quantity") or 0)
                if qty_snapshot <= 0:
                    qty_snapshot = 1.0
                ingredient_name = row.get("ingredient_name")
                master_id = row.get("master_ingredient_id")

                # [P3-DEPLETED-BD] INSERT a user_depleted_items vía helper que
                # hace upsert idempotente por (user_id, master_id) o (user_id,
                # lower(name)). Si el item ya estaba agotado, actualiza la
                # qty + depleted_at — escenario edge legítimo.
                _bd_ok = _bd_add_depleted(
                    user_id,
                    ingredient_name=str(ingredient_name),
                    quantity=qty_snapshot,
                    unit=str(row.get("unit") or "unidad"),
                    master_ingredient_id=master_id,
                    category=None,
                    shelf_life_days=None,
                    depleted_at=now_iso,
                )
                if not _bd_ok:
                    logger.warning(
                        f"[P3-DEPLETED-BD] add_depleted_item falló para "
                        f"name={ingredient_name!r} — skip BD insert pero "
                        f"continuar con delete del inventory (best-effort)."
                    )

                depleted_payload.append({
                    "master_ingredient_id": master_id,
                    "ingredient_name": ingredient_name,
                    "quantity": qty_snapshot,
                    "unit": row.get("unit"),
                    "category": None,
                    "shelf_life_days": None,
                    "depleted_at": now_iso,
                })
                try:
                    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL
                    # directo (Neon). Preserva el filtro `AND user_id`
                    # (invariante I2).
                    from db import execute_sql_write as _sql_del
                    _sql_del(
                        "DELETE FROM public.user_inventory WHERE id = %s AND user_id = %s",
                        (row.get("id"), user_id),
                    )
                    depleted_count += 1
                except Exception as _del_err:
                    logger.warning(
                        f"[P3-AGENT-DEPLETE] DELETE row id={row.get('id')} falló: {_del_err}"
                    )

        if items_to_remove:
            deduct_consumed_meal_from_inventory(user_id, items_to_remove)
            removed_count += len(items_to_remove)

        # Construir mensaje human-friendly para la LLM.
        parts = []
        if added_count:
            parts.append(f"se agregaron {added_count} ítem(s)")
        if depleted_count:
            parts.append(f"se marcaron {depleted_count} como agotado(s)")
        if removed_count:
            parts.append(f"se eliminaron {removed_count} ítem(s)")
        if not parts:
            msg = "No se modificó la despensa (nada coincidió con lo solicitado)."
        else:
            msg = "¡Despensa actualizada! " + ", ".join(parts) + "."

        # [P3-AGENT-DEPLETE · 2026-05-22] Inyectar marker JSON inline al final
        # del tool_result. `agent.py:execute_tools` lo extrae con regex,
        # propaga al state field `pantry_depleted_items`, y lo strip-ea del
        # ToolMessage antes de pasarlo al siguiente call_model — la LLM NO
        # ve el JSON raw (sería ruido en su contexto).
        if depleted_payload:
            msg += f"\n\n<<PANTRY_DEPLETED_JSON: {_json.dumps(depleted_payload, ensure_ascii=False)}>>"

        return msg
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
        # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo (Neon).
        from db import connection_pool, execute_sql_query
        if not connection_pool:
            return "No puedo consultar la hidratacion ahora mismo, la base de datos no esta disponible."

        log_date = _local_date_str_for_user()

        # Lee el conteo del dia.
        _row = execute_sql_query(
            "SELECT glasses FROM public.water_intake_log "
            "WHERE user_id = %s AND log_date = %s LIMIT 1",
            (user_id, log_date),
            fetch_one=True,
        )
        # [P3-WATER-HALF-GLASS · 2026-06-24] numeric → float (medios vasos).
        glasses = float(_row.get("glasses") or 0) if _row else 0.0

        # Reusa la formula personalizada del endpoint /water-intake.
        # Import lazy para evitar circular (routers/plans.py importa de aqui en futuro).
        from routers.plans import _compute_water_goal
        meta = _compute_water_goal(user_id)
        goal = meta["goal"]
        weight_kg = meta.get("weight_kg")
        is_personalized = not meta.get("default", True) and weight_kg is not None

        pct = round((glasses / goal) * 100) if goal else 0
        remaining = max(0, goal - glasses)

        # `:g` formatea 4.0 → "4" y 4.5 → "4.5" (sin .0 colgante).
        msg_parts = [f"Hidratacion HOY ({log_date}): {glasses:g} de {goal} vasos ({pct}%)."]
        if remaining > 0:
            msg_parts.append(f"Le faltan {remaining:g} para cumplir su meta.")
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
def log_water_glass(user_id: str, count_delta: float = 1) -> str:
    """Suma o resta vasos de agua al registro de hidratacion del usuario para HOY.

    Acepta medios vasos (0.5) — un "sorbo" cuenta medio vaso.
    Usa esta herramienta cuando el usuario diga que se tomo agua o que se equivoco:
    - 'me tome un vaso' / 'marca uno mas' → count_delta=1 (default)
    - 'me tome un sorbo / medio vaso' → count_delta=0.5
    - 'me tome 3 vasos seguidos' → count_delta=3
    - 'borra el ultimo / me equivoque' → count_delta=-1
    - 'resetea el dia' → primero check_hydration_today para saber el conteo
      actual N, luego log_water_glass(count_delta=-N).

    Para SETEAR un valor absoluto (ej: el usuario dice 'llevo 5 vasos'),
    primero usa `check_hydration_today` para saber el conteo actual, calcula
    el delta y pasa ese valor (5 - actual).

    El delta debe ser multiplo de 0.5. El conteo total queda clamped a [0, 50]
    (cap defensivo de la tabla). Tras la mutacion, incluye SIEMPRE la etiqueta
    `[UI_ACTION: REFRESH_HYDRATION]` en tu respuesta para que el Dashboard
    recargue el card de Hidratacion.
    """
    logger.info(f"💧 [TOOL EXECUTION] log_water_glass user={user_id} delta={count_delta}")
    # [P3-WATER-HALF-GLASS · 2026-06-24] Acepta enteros y medios vasos (0.5).
    if isinstance(count_delta, bool) or not isinstance(count_delta, (int, float)):
        return "Error: el delta de vasos debe ser numerico (positivo para sumar, negativo para restar)."
    if (count_delta * 2) != int(count_delta * 2):
        return "Error: el delta debe ser multiplo de 0.5 (medio vaso)."
    if count_delta == 0:
        return "El delta fue 0 — no se modifico nada."
    count_delta = float(count_delta)
    try:
        from db import connection_pool, execute_sql_write
        if not connection_pool:
            return "No puedo modificar la hidratacion ahora mismo, la base de datos no esta disponible."

        log_date = _local_date_str_for_user()

        # [P3-WATER-ATOMIC-DELTA · 2026-05-30] Incremento ATÓMICO en UNA sola
        # sentencia. Pre-fix era read-modify-write (SELECT glasses → upsert
        # current+delta) que perdía la escritura concurrente del card de
        # Hidratación del Dashboard, el cual hace un SET ABSOLUTO sobre la misma
        # fila (user_id,log_date) vía POST /api/plans/water-intake. Si el set
        # absoluto aterrizaba entre el SELECT y el upsert de esta tool, el upsert
        # lo pisaba con `stale_current+delta` (lost-update). Ahora `glasses + %s`
        # se evalúa DENTRO del UPDATE bajo el row-lock del upsert (PK
        # user_id,log_date) → sin ventana. Clamp [0,50] en SQL (GREATEST/LEAST)
        # preserva el cap defensivo; RETURNING da el conteo autoritativo.
        # Tooltip-anchor: P3-WATER-ATOMIC-DELTA.
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Eliminado el fallback PostgREST
        # read-modify-write: los datos viven en Neon y el cliente legado ya
        # no apunta a la DB de datos (split-brain). Si el upsert atómico falla,
        # el except exterior devuelve el error al agente.
        _rows = execute_sql_write(
            """
            INSERT INTO water_intake_log (user_id, log_date, glasses, updated_at)
            VALUES (%s, %s, GREATEST(0, LEAST(50, %s)), now())
            ON CONFLICT (user_id, log_date)
            DO UPDATE SET glasses = GREATEST(0, LEAST(50, water_intake_log.glasses + %s)),
                          updated_at = now()
            RETURNING glasses
            """,
            (user_id, log_date, count_delta, count_delta),
            returning=True,
        )
        if not _rows:
            return "Error registrando vaso de agua: la base de datos no confirmo el conteo."
        # [P3-WATER-HALF-GLASS · 2026-06-24] numeric → float (medios vasos).
        new_count = float(_rows[0]["glasses"])

        # Reusa la meta personalizada para el mensaje de confirmacion.
        try:
            from routers.plans import _compute_water_goal
            goal = _compute_water_goal(user_id).get("goal", 8)
        except Exception:
            goal = 8

        verb = "sumaron" if count_delta > 0 else "restaron"
        boundary = ""
        if count_delta > 0 and new_count >= 50:
            boundary = " (tope defensivo de 50 vasos/dia)."
        elif count_delta < 0 and new_count <= 0:
            boundary = " (minimo de 0 vasos)."
        reached = " ¡Cumplio su meta del dia!" if new_count >= goal else ""
        # `:g` formatea 4.0 → "4" y 4.5 → "4.5" (sin .0 colgante).
        return (
            f"Listo: se {verb} {abs(count_delta):g} vaso(s). "
            f"El usuario ahora lleva {new_count:g} de {goal} vasos hoy.{reached}{boundary}"
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
            
        # [P0-RESTOCK-DEDUP-NAME · 2026-05-20] restock_inventory ahora retorna
        # (success, persisted_names). El agent tool solo necesita `success`.
        _restock_res = restock_inventory(user_id, final_shop_list)
        success = bool(_restock_res[0]) if isinstance(_restock_res, tuple) else bool(_restock_res)
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


# [P3-MICRO-FOOD-SUGGEST · 2026-06-15] Tabla nutriente (es/en) → columna del
# master_ingredients + label + unidad + `is_ceiling` (True = es un TECHO: sodio/
# azúcar/satfat/colesterol → sugerir alternativas BAJAS; False = es un PISO →
# sugerir alimentos RICOS). Las columnas coinciden con nutrition_db.py.
_MICRO_NUTRIENT_COLUMNS = {
    "fibra": ("fiber_g_per_100g", "fibra", "g", False),
    "fiber": ("fiber_g_per_100g", "fibra", "g", False),
    "hierro": ("iron_mg_per_100g", "hierro", "mg", False),
    "iron": ("iron_mg_per_100g", "hierro", "mg", False),
    "calcio": ("calcium_mg_per_100g", "calcio", "mg", False),
    "calcium": ("calcium_mg_per_100g", "calcio", "mg", False),
    "vitamina d": ("vitamin_d_mcg_per_100g", "vitamina D", "mcg", False),
    "vit d": ("vitamin_d_mcg_per_100g", "vitamina D", "mcg", False),
    "vitamin d": ("vitamin_d_mcg_per_100g", "vitamina D", "mcg", False),
    "vitamina b12": ("vitamin_b12_mcg_per_100g", "vitamina B12", "mcg", False),
    "vitamin b12": ("vitamin_b12_mcg_per_100g", "vitamina B12", "mcg", False),
    "b12": ("vitamin_b12_mcg_per_100g", "vitamina B12", "mcg", False),
    "potasio": ("potassium_mg_per_100g", "potasio", "mg", False),
    "potassium": ("potassium_mg_per_100g", "potasio", "mg", False),
    "magnesio": ("magnesium_mg_per_100g", "magnesio", "mg", False),
    "magnesium": ("magnesium_mg_per_100g", "magnesio", "mg", False),
    "proteina": ("protein_g_per_100g", "proteína", "g", False),
    "protein": ("protein_g_per_100g", "proteína", "g", False),
    "sodio": ("sodium_mg_per_100g", "sodio", "mg", True),
    "sodium": ("sodium_mg_per_100g", "sodio", "mg", True),
    "azucar": ("sugars_g_per_100g", "azúcar", "g", True),
    "azucares": ("sugars_g_per_100g", "azúcar", "g", True),
    "sugar": ("sugars_g_per_100g", "azúcar", "g", True),
    "grasa saturada": ("saturated_fat_g_per_100g", "grasa saturada", "g", True),
    "saturated fat": ("saturated_fat_g_per_100g", "grasa saturada", "g", True),
    "colesterol": ("cholesterol_mg_per_100g", "colesterol", "mg", True),
    "cholesterol": ("cholesterol_mg_per_100g", "colesterol", "mg", True),
}

# Tokens para filtrar el catálogo por tipo de dieta (best-effort; la LLM hace el
# filtro final con las restricciones completas del perfil en el system prompt).
_VEGETARIAN_EXCLUDE = [
    "carne", "pollo", "pavo", "pescado", "cerdo", "chuleta", "bacalao", "salami",
    "jamon", "tocino", "longaniza", "salchich", "atun", "sardina", "marisco",
    "camaron", "higado", "chicharron", "res ", "bistec", "costilla",
]
_VEGAN_EXCLUDE = _VEGETARIAN_EXCLUDE + [
    "huevo", "leche", "queso", "yogur", "mantequilla", "crema", "lacteo", "miel",
]

# [P3-MICRO-FOOD-SUGGEST] Condimentos/especias/hierbas secas: densísimos por 100g
# pero se consumen en pizcas → no son "fuentes" prácticas de un micronutriente.
# Sin esto, el top de fibra/hierro se llena de canela/orégano/albahaca y desplaza
# los alimentos reales (chía, habichuelas). Match accent-insensible vía strip_accents.
_CONDIMENT_EXCLUDE = [
    "canela", "oregano", "albahaca", "pimienta", "pimenton", "paprika", "comino",
    "laurel", "tomillo", "romero", "curcuma", "nuez moscada", "clavo de olor",
    "anis", "sazon", "cubito", "caldo en polvo", "vainilla", "esencia", "extracto",
    "especia", "condimento", "hierbas", "perejil seco", "cilantro seco",
    "ajo en polvo", "cebolla en polvo", "jengibre en polvo", "polvo de hornear",
    "bicarbonato",
]


def _resolve_micro_nutrient(nutrient: str):
    """Nombre del nutriente (es/en, con/ sin acentos) → (columna, label, unidad,
    is_ceiling) o None si no se reconoce."""
    key = strip_accents((nutrient or "").strip().lower())
    if key in _MICRO_NUTRIENT_COLUMNS:
        return _MICRO_NUTRIENT_COLUMNS[key]
    for k, v in _MICRO_NUTRIENT_COLUMNS.items():
        if k in key or key in k:
            return v
    return None


@tool
def suggest_foods_for_nutrient(user_id: str, nutrient: str, top_n: int = 6) -> str:
    """
    Sugiere alimentos del catálogo nutricional para mejorar un micronutriente del
    plan del usuario, filtrando por sus alergias, rechazos y tipo de dieta.
    Úsala cuando el usuario pregunte qué comer para subir (o reducir) un
    micronutriente: "¿qué como para más fibra?", "necesito más hierro", "cómo subo
    la vitamina D", "cómo bajo el sodio".

    Parámetros:
    - nutrient: nombre del nutriente (ej: "fibra", "hierro", "calcio", "vitamina D",
      "vitamina B12", "potasio", "magnesio", "proteína", "sodio", "azúcar",
      "grasa saturada", "colesterol").
    - top_n: cuántos alimentos sugerir (default 6).

    Devuelve una lista rankeada de alimentos del catálogo con su aporte por 100g.
    """
    logger.info(f"🥗 [TOOL EXECUTION] suggest_foods_for_nutrient nutrient='{nutrient}' user={user_id}")
    try:
        resolved = _resolve_micro_nutrient(nutrient)
        if not resolved:
            return (
                f"No reconozco el micronutriente '{nutrient}'. Nutrientes soportados: "
                "fibra, hierro, calcio, vitamina D, vitamina B12, potasio, magnesio, "
                "proteína, sodio, azúcar, grasa saturada, colesterol."
            )
        column, label, unit, is_ceiling = resolved

        # Restricciones del usuario (best-effort). Las listas pueden venir como
        # list o como string "Lacteos, Gluten".
        allergies, dislikes, diet_type = [], [], "balanced"
        try:
            profile = get_user_profile(user_id) or {}
            hp = profile.get("health_profile") or {}

            def _as_list(v):
                if isinstance(v, list):
                    return v
                if isinstance(v, str):
                    return [x.strip() for x in v.split(",") if x.strip()]
                return []

            allergies = [strip_accents(str(a).lower()) for a in _as_list(hp.get("allergies"))]
            dislikes = [strip_accents(str(d).lower()) for d in _as_list(hp.get("dislikes"))]
            diet_type = (hp.get("dietType") or "balanced").strip().lower()
        except Exception as _pe:
            logger.warning(f"⚠ [TOOL] suggest_foods_for_nutrient: perfil no disponible ({_pe})")

        exclude_tokens = [t for t in (allergies + dislikes) if t]
        if diet_type == "vegetarian":
            exclude_tokens += _VEGETARIAN_EXCLUDE
        elif diet_type == "vegan":
            exclude_tokens += _VEGAN_EXCLUDE

        from shopping_calculator import get_master_ingredients
        rows = get_master_ingredients() or []

        candidates = []
        for r in rows:
            name = r.get("name")
            if not name:
                continue
            try:
                val = float(r.get(column))
            except (TypeError, ValueError):
                continue
            # Piso → exigir aporte real (>0). Techo → incluir bajos (>=0).
            if not is_ceiling and val <= 0:
                continue
            if val < 0:
                continue
            name_norm = strip_accents(str(name).lower())
            if any(tok in name_norm for tok in exclude_tokens):
                continue
            # Condimentos/especias: densos por 100g pero no son fuentes prácticas.
            if any(tok in name_norm for tok in _CONDIMENT_EXCLUDE):
                continue
            candidates.append((name, val))

        if not candidates:
            return (
                f"No encontré alimentos en el catálogo {'bajos' if is_ceiling else 'ricos'} en "
                f"{label} compatibles con las restricciones del usuario. Sugiere fuentes "
                "generales conocidas con criterio, respetando sus alergias y dieta."
            )

        # Techo → los MÁS BAJOS (mejores swaps); piso → los MÁS RICOS.
        candidates.sort(key=lambda x: x[1], reverse=not is_ceiling)
        n = max(1, min(int(top_n or 6), 12))
        top = candidates[:n]

        verb = "más bajos en" if is_ceiling else "más ricos en"
        body = "\n".join(f"- {name}: {round(val, 1)}{unit} por cada 100g" for name, val in top)
        guidance = (
            f"Estos son alimentos del catálogo {verb} {label}, ya filtrados por las "
            "restricciones del usuario. Recomiéndale 2-3 opciones prácticas con cantidades "
            "realistas para integrarlas a su plan; NO listes todos crudos ni inventes valores."
        )
        return f"ALIMENTOS {verb.upper()} {label.upper()} (catálogo, por 100g):\n{body}\n\n{guidance}"

    except Exception as e:
        logger.error(f"❌ [TOOL] suggest_foods_for_nutrient error: {e}")
        return f"Error consultando el catálogo de alimentos: {str(e)}"


# Lista de tools disponibles para el agente
agent_tools = [update_form_field, generate_new_plan_from_chat, log_consumed_meal, modify_single_meal, search_deep_memory, check_shopping_list, check_current_pantry, modify_pantry_inventory, mark_shopping_list_purchased, check_hydration_today, log_water_glass, suggest_foods_for_nutrient]
