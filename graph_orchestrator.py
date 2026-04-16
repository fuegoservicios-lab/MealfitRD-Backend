# backend/graph_orchestrator.py
"""
Orquestación LangGraph: Flujo Map-Reduce multi-agente para generación de planes nutricionales.
Planificador → Generadores Paralelos (×3 días) → Ensamblador → Revisor Médico → (loop si falla)
"""

import os
import time
import json
from typing import TypedDict, Optional, Callable, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

import concurrent.futures
from datetime import datetime, timezone
import random
import re as _re
# NOTA: NO importar 'from agent import ...' a nivel de módulo → causa import circular
# (app → agent → tools → graph_orchestrator → agent). Se usa lazy import donde se necesite.
from cpu_tasks import _validar_repeticiones_cpu_bound, _normalize_meal_name
from constants import normalize_ingredient_for_tracking, strip_accents, TECHNIQUE_FAMILIES, ALL_TECHNIQUES, TECH_TO_FAMILY, SUPPLEMENT_NAMES
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from db import get_recent_techniques, get_recent_meals_from_plans, check_meal_plan_generated_today, search_user_facts, search_visual_diary, get_user_facts_by_metadata
from nutrition_calculator import get_nutrition_targets

logger = logging.getLogger(__name__)


# ============================================================
# ESTADO COMPARTIDO DEL GRAFO
# ============================================================
class PlanState(TypedDict):
    # Inputs (se setean al inicio)
    form_data: dict
    taste_profile: str
    nutrition: dict
    history_context: str
    
    # Plan generado (output del generador)
    plan_result: Optional[dict]
    
    # Esqueleto del planificador (fase map)
    plan_skeleton: Optional[dict]
    
    # Revisión médica
    review_passed: bool
    review_feedback: str
    
    # Control de flujo
    attempt: int
    user_facts: str
    
    # Callback de progreso para SSE streaming (opcional)
    progress_callback: Optional[Any]  # Callable o None
    



# ============================================================
# SCHEMAS (importados del módulo canónico schemas.py)
# ============================================================
from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel, PlanSkeletonModel, SingleDayPlanModel


# ============================================================
# PROMPTS (importados del paquete prompts/)
# ============================================================
from prompts.plan_generator import (
    GENERATOR_SYSTEM_PROMPT,
    build_nutrition_context,
    build_correction_context,
    build_rag_context,
    build_time_context,
    build_technique_injection,
    build_supplements_context,
    build_grocery_duration_context,
    build_pantry_context,
    build_prices_context,
)
from prompts.medical_reviewer import REVIEWER_SYSTEM_PROMPT
from prompts.planner import PLANNER_SYSTEM_PROMPT
from prompts.day_generator import DAY_GENERATOR_SYSTEM_PROMPT, build_day_assignment_context


# ============================================================
# HELPERS: Selección de técnicas y contextos compartidos
# ============================================================
def _emit_progress(state: PlanState, event: str, data: dict):
    """Emite un evento de progreso si hay callback registrado."""
    cb = state.get("progress_callback")
    if cb:
        try:
            cb({"event": event, "data": data})
        except Exception:
            pass


def _select_techniques(user_id: str | None) -> list:
    """Selecciona 7 técnicas de cocción diversificadas por familia con decaimiento temporal."""
    technique_freq = {}
    if user_id:
        try:
            recent_techs = get_recent_techniques(user_id, limit=6)
            now_utc = datetime.now(timezone.utc)
            decay_factor = 0.9
            for t, created_at_str in recent_techs:
                days_elapsed = 0
                if created_at_str:
                    try:
                        if created_at_str.endswith("Z"):
                            dt = datetime.fromisoformat(created_at_str[:-1]).replace(tzinfo=timezone.utc)
                        else:
                            dt = datetime.fromisoformat(created_at_str)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                        days_elapsed = max(0, (now_utc - dt).days)
                    except Exception:
                        pass
                decayed_weight = decay_factor ** days_elapsed
                technique_freq[t] = technique_freq.get(t, 0) + decayed_weight
            if technique_freq:
                print(f"🔍 [TÉCNICAS] Frecuencias con decaimiento temporal: { {k: round(v, 2) for k, v in technique_freq.items()} }")
        except Exception as e:
            print(f"⚠️ [TÉCNICAS] Error consultando DB, usando pesos uniformes: {e}")

    selected_techniques = []
    used_families = set()
    _pool_t = [(t, 1.0 / (technique_freq.get(t, 0) + 1)) for t in ALL_TECHNIQUES]

    while len(selected_techniques) < 3 and _pool_t:
        cross_family_pool = [(t, w) for t, w in _pool_t if TECH_TO_FAMILY.get(t) not in used_families]
        active_pool = cross_family_pool if cross_family_pool else _pool_t
        pick = random.choices([x[0] for x in active_pool], weights=[x[1] for x in active_pool], k=1)[0]
        selected_techniques.append(pick)
        used_families.add(TECH_TO_FAMILY.get(pick, ""))
        _pool_t = [(t, w) for t, w in _pool_t if t != pick]

    print(f"👨‍🍳 [TÉCNICAS] Seleccionadas (familias diversas): {[f'{t} ({TECH_TO_FAMILY.get(t)})' for t in selected_techniques]}")
    return selected_techniques


def _build_shared_context(state: PlanState) -> dict:
    """Construye todos los bloques de contexto compartidos entre nodos."""
    form_data = state["form_data"]
    nutrition = state["nutrition"]
    review_feedback = state.get("review_feedback", "")
    user_facts = state.get("user_facts", "")
    history_context = state.get("history_context", "")
    taste_profile = state.get("taste_profile", "")

    _uid = form_data.get("user_id") or form_data.get("session_id")
    if _uid == "guest": _uid = None

    from ai_helpers import get_deterministic_variety_prompt
    variety_prompt = get_deterministic_variety_prompt(history_context, form_data, user_id=_uid)

    return {
        "nutrition_context": build_nutrition_context(nutrition),
        "correction_context": build_correction_context(review_feedback),
        "rag_context": build_rag_context(user_facts),
        "time_context": build_time_context(),
        "variety_prompt": variety_prompt,
        "supplements_context": build_supplements_context(form_data),
        "grocery_duration_context": build_grocery_duration_context(form_data),
        "pantry_context": build_pantry_context(form_data),
        "prices_context": build_prices_context(),
        "taste_profile": taste_profile,
        "history_context": history_context,
        "user_id": _uid,
    }


# ============================================================
# NODO 1: PLANIFICADOR (Fase Map — esqueleto liviano)
# ============================================================
def plan_skeleton_node(state: PlanState) -> dict:
    """Genera un esqueleto liviano del plan: asignaciones de ingredientes y técnicas por día (~8s)."""
    attempt = state.get("attempt", 0) + 1
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    print(f"\n{'='*60}")
    print(f"📋 [PLANIFICADOR] Diseñando estructura del plan (intento #{attempt})...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "skeleton", "message": "Diseñando la estructura del plan..."})
    start_time = time.time()

    ctx = _build_shared_context(state)
    _uid = ctx["user_id"]

    # Seleccionar técnicas de cocción
    selected_techniques = _select_techniques(_uid)

    random_seed = random.randint(10000, 99999)

    techniques_str = "\n".join([f"• Día/Opción {i+1}: {selected_techniques[i] if len(selected_techniques) > i else 'Libre'}" for i in range(3)])

    prompt_text = (
        f"Analiza la siguiente información del usuario y diseña el ESQUELETO de un plan de 3 alternativas/días.\n"
        f"Semilla de generación aleatoria: {random_seed}\n\n"
        f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
        f"{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
        f"{ctx['rag_context']}\n{ctx['correction_context']}\n{ctx['history_context']}\n"
        f"{ctx['variety_prompt']}\n{ctx['pantry_context']}\n{ctx['prices_context']}\n\n"
        f"Técnicas de cocción asignadas (una por día):\n"
        f"{techniques_str}\n\n"
        f"{PLANNER_SYSTEM_PROMPT}"
    )

    is_re_roll = form_data.get("_is_same_day_reroll", False)
    base_temp = 0.95 if is_re_roll else 0.7

    planner_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=base_temp if attempt == 1 else (base_temp + 0.1),
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=0,
        timeout=60
    ).with_structured_output(PlanSkeletonModel)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [PLANIFICADOR] Reintento #{retry_state.attempt_number}...")
    )
    def invoke_planner():
        print(f"⏳ [PLANIFICADOR] Generando esqueleto del plan...")
        return planner_llm.invoke(prompt_text)

    response = invoke_planner()

    duration = round(time.time() - start_time, 2)
    print(f"✅ [PLANIFICADOR] Esqueleto generado en {duration}s")

    if hasattr(response, "model_dump"):
        skeleton = response.model_dump()
    elif isinstance(response, dict):
        skeleton = response
    else:
        skeleton = response.dict()

    # Guardar técnicas seleccionadas para persistencia
    skeleton["_selected_techniques"] = selected_techniques

    return {
        "plan_skeleton": skeleton,
        "attempt": attempt
    }


# ============================================================
# NODO 2: GENERADORES PARALELOS (Fase Reduce — 3 días simultáneos)
# ============================================================
def generate_days_parallel_node(state: PlanState) -> dict:
    """Genera los 7 días completos en PARALELO usando el esqueleto del planificador."""
    skeleton = state["plan_skeleton"]
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    print(f"\n{'='*60}")
    print(f"🚀 [GENERADORES PARALELOS] Lanzando 3 workers para generar las opciones...\n")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "parallel_generation", "message": "Generando las 3 opciones en paralelo..."})

    ctx = _build_shared_context(state)
    skeleton_days = skeleton.get("days", [])

    is_re_roll = form_data.get("_is_same_day_reroll", False)
    base_temp = 0.95 if is_re_roll else 0.7
    attempt = state.get("attempt", 1)

    def generate_single_day(skeleton_day: dict, day_num: int) -> dict:
        """Worker: genera un solo día completo."""
        _emit_progress(state, "day_started", {"day": day_num, "message": f"Generando Día {day_num}..."})
        day_start = time.time()

        assignment_context = build_day_assignment_context(skeleton_day, day_num)

        random_seed = random.randint(10000, 99999)

        prompt_text = (
            f"Genera las comidas completas para el DÍA {day_num} del plan.\n"
            f"Semilla de generación aleatoria: {random_seed}\n\n"
            f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
            f"{ctx['nutrition_context']}\n{ctx['time_context']}\n{ctx['taste_profile']}\n"
            f"{ctx['rag_context']}\n{ctx['correction_context']}\n"
            f"{ctx['supplements_context']}\n{ctx['grocery_duration_context']}\n"
            f"{ctx['pantry_context']}\n"
            f"{assignment_context}\n\n"
            f"{DAY_GENERATOR_SYSTEM_PROMPT}"
        )

        day_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            temperature=base_temp if attempt == 1 else (base_temp + 0.1),
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            max_retries=0,
            timeout=90
        ).with_structured_output(SingleDayPlanModel)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=8),
            reraise=True,
            before_sleep=lambda rs: print(f"⚠️  [DÍA {day_num}] Reintento #{rs.attempt_number}...")
        )
        def invoke_day():
            return day_llm.invoke(prompt_text)

        response = invoke_day()

        if hasattr(response, "model_dump"):
            day_result = response.model_dump()
        elif isinstance(response, dict):
            day_result = response
        else:
            day_result = response.dict()

        # Forzar day number correcto
        day_result["day"] = day_num

        day_duration = round(time.time() - day_start, 2)
        print(f"✅ [DÍA {day_num}] Generado en {day_duration}s")
        _emit_progress(state, "day_complete", {"day": day_num, "duration": day_duration})

        return day_result

    # Lanzar generaciones en paralelo
    parallel_start = time.time()
    generated_days = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for i, skel_day in enumerate(skeleton_days[:3]):
            day_num = i + 1
            futures[executor.submit(generate_single_day, skel_day, day_num)] = day_num

        for future in concurrent.futures.as_completed(futures):
            day_num = futures[future]
            try:
                result = future.result()
                generated_days.append(result)
            except Exception as e:
                print(f"❌ [DÍA {day_num}] Error fatal en generación paralela: {e}")
                # Crear un día placeholder mínimo para no romper el flujo
                generated_days.append({
                    "day": day_num,
                    "meals": [],
                    "supplements": None
                })

    # Ordenar por número de día
    generated_days.sort(key=lambda d: d.get("day", 0))

    parallel_duration = round(time.time() - parallel_start, 2)
    print(f"✅ [PARALELO] Los 3 días generados en {parallel_duration}s (en paralelo)")

    return {
        "plan_result": {
            "days": generated_days,
            "_skeleton": skeleton,
            "_parallel_duration": parallel_duration,
        }
    }


# ============================================================
# NODO 3: ENSAMBLADOR (combina días + shopping list + macros del calculador)
# ============================================================
def assemble_plan_node(state: PlanState) -> dict:
    """Ensambla el plan final combinando skeleton + días paralelos + datos del calculador."""
    partial = state["plan_result"]
    skeleton = partial.get("_skeleton", state.get("plan_skeleton", {}))
    form_data = state["form_data"]
    nutrition = state["nutrition"]

    print(f"\n{'='*60}")
    print(f"🔧 [ENSAMBLADOR] Combinando plan final...")
    print(f"{'='*60}")

    _emit_progress(state, "phase", {"phase": "assembly", "message": "Ensamblando tu plan final..."})

    result = {
        "main_goal": skeleton.get("main_goal", nutrition.get("goal_label", "")),
        "insights": skeleton.get("insights", []),
        "days": partial.get("days", []),
    }

    # Post-proceso: forzar valores exactos del calculador
    result["calories"] = nutrition.get("total_daily_calories", nutrition["target_calories"])
    active_macros = nutrition.get("total_daily_macros", nutrition["macros"])
    result["macros"] = {
        "protein": active_macros["protein_str"],
        "carbs": active_macros["carbs_str"],
        "fats": active_macros["fats_str"],
    }
    result["main_goal"] = nutrition["goal_label"]

    # Calcular shopping lists
    _uid = form_data.get("user_id") or form_data.get("session_id")
    if _uid == "guest": _uid = None

    from shopping_calculator import get_shopping_list_delta
    try:
        household = max(1, int(form_data.get("householdSize", 1) or 1))
        if household > 1:
            print(f"👨‍👩‍👧‍👦 [HOUSEHOLD] Escalando lista de compras ×{household} personas")

        aggr_list_7 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=1.0 * household) if _uid else []
        aggr_list_15 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=2.0 * household) if _uid else []
        aggr_list_30 = get_shopping_list_delta(_uid, result, is_new_plan=True, structured=True, multiplier=4.0 * household) if _uid else []

        grocery_duration = form_data.get("groceryDuration", "weekly")
        if grocery_duration == "biweekly":
            aggr_list = aggr_list_15
        elif grocery_duration == "monthly":
            aggr_list = aggr_list_30
        else:
            aggr_list = aggr_list_7

        result["aggregated_shopping_list"] = aggr_list
        result["aggregated_shopping_list_weekly"] = aggr_list_7
        result["aggregated_shopping_list_biweekly"] = aggr_list_15
        result["aggregated_shopping_list_monthly"] = aggr_list_30
    except Exception as e:
        import traceback
        print(f"⚠️ [SHOPPING MATH] Error agregando lista delta: {e}")
        traceback.print_exc()
        result["aggregated_shopping_list"] = []
        result["aggregated_shopping_list_weekly"] = []
        result["aggregated_shopping_list_biweekly"] = []
        result["aggregated_shopping_list_monthly"] = []

    # Guardar técnicas seleccionadas para persistencia en DB
    result["_selected_techniques"] = skeleton.get("_selected_techniques", [])

    print(f"✅ [ENSAMBLADOR] Plan final ensamblado")

    return {
        "plan_result": result
    }


# ============================================================
# NODO 2: AGENTE REVISOR MÉDICO
# ============================================================
def review_plan_node(state: PlanState) -> dict:
    """Revisa el plan generado para verificar seguridad médica."""
    plan = state["plan_result"]
    form_data = state["form_data"]
    taste_profile = state.get("taste_profile", "")
    attempt = state.get("attempt", 1)
    
    print(f"\n{'='*60}")
    print(f"🩺 [AGENTE REVISOR MÉDICO] Verificando plan (intento #{attempt})...")
    print(f"{'='*60}")
    
    _emit_progress(state, "phase", {"phase": "review", "message": "Verificación médica en curso..."})
    
    # Extraer restricciones del usuario
    allergies = form_data.get("allergies", [])
    # Combinar alergias del array con las escritas en texto libre (otherAllergies)
    other_allergies_text = form_data.get("otherAllergies", "")
    if other_allergies_text:
        # Separar por comas si el usuario escribió varias
        extra_allergies = [a.strip() for a in other_allergies_text.replace(",", ",").split(",") if a.strip()]
        allergies = list(allergies) + extra_allergies
    
    medical_conditions = form_data.get("medicalConditions", [])
    # Combinar condiciones del array con las escritas en texto libre (otherConditions)
    other_conditions_text = form_data.get("otherConditions", "")
    if other_conditions_text:
        extra_conditions = [c.strip() for c in other_conditions_text.replace(",", ",").split(",") if c.strip()]
        medical_conditions = list(medical_conditions) + extra_conditions
    diet_type = form_data.get("dietType", "balanced")
    dislikes = form_data.get("dislikes", [])
    
    # Si no hay restricciones, aprobar automáticamente
    if not allergies and not medical_conditions and diet_type == "balanced" and not dislikes and not taste_profile:
        print("✅ [REVISOR] Sin restricciones declaradas → Aprobado automáticamente.")
        return {
            "review_passed": True,
            "review_feedback": ""
        }
    
    start_time = time.time()
    
    # Extraer todos los ingredientes del plan para revisión
    all_ingredients = []
    all_meals_summary = []
    days = plan.get("days", [])
    if not days and plan.get("meals"):
        days = [{"day": 1, "meals": plan.get("meals")}] # Fallback seguro durante generación

    for day_obj in days:
        day_num = day_obj.get("day", "?")
        for meal in day_obj.get("meals", []):
            meal_name = meal.get("name", "Sin nombre")
            ingredients = meal.get("ingredients", [])
            all_ingredients.extend(ingredients)
            all_meals_summary.append(f"- Día {day_num} | {meal.get('meal', '?')}: {meal_name} → Ingredientes: {', '.join(ingredients)}")
    
    review_prompt = f"""
{REVIEWER_SYSTEM_PROMPT}

--- RESTRICCIONES DEL PACIENTE ---
Alergias declaradas: {json.dumps(allergies) if allergies else "Ninguna"}
Condiciones médicas: {json.dumps(medical_conditions) if medical_conditions else "Ninguna"}
Tipo de dieta: {diet_type}
Alimentos que no le gustan: {json.dumps(dislikes) if dislikes else "Ninguno"}

--- PERFIL DE GUSTOS (SI EXISTE) ---
{taste_profile if taste_profile else "Sin perfil de gustos disponible."}

--- PLAN A REVISAR ---
Calorías totales: {plan.get("calories")} kcal

Comidas e ingredientes:
{chr(10).join(all_meals_summary)}

--- TODOS LOS INGREDIENTES DEL PLAN ---
{json.dumps(all_ingredients)}

Responde ÚNICAMENTE con el JSON de revisión.
"""
    
    reviewer_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.1,  # Temperatura muy baja para ser preciso
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=0,
        timeout=60
    )
    
    # Invocar con reintentos automáticos
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [REVISOR] Reintento #{retry_state.attempt_number}...")
    )
    def invoke_with_retry():
        return reviewer_llm.invoke(review_prompt)
    
    response = invoke_with_retry()
    
    # Gemini puede devolver content como:
    # - string directo: "{ \"approved\": true ... }"
    # - lista de dicts: [{'type': 'text', 'text': '{ "approved": true ... }'}]
    # - lista de strings: ["{ \"approved\": true ... }"]
    raw_content = response.content
    if isinstance(raw_content, list):
        parts = []
        for part in raw_content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        review_text = " ".join(parts).strip()
    else:
        review_text = str(raw_content).strip()
    
    duration = round(time.time() - start_time, 2)
    
    # Parsear la respuesta JSON del revisor
    try:
        # Limpiar markdown si viene envuelto en ```json
        clean_text = review_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
        review_data = json.loads(clean_text)
        approved = review_data.get("approved", True)
        issues = review_data.get("issues", [])
        severity = review_data.get("severity", "none")
    except (json.JSONDecodeError, IndexError):
        # Si no puede parsear, rechazar por seguridad (fail-closed)
        print(f"⚠️  [REVISOR] No se pudo parsear respuesta, RECHAZANDO por defecto (Fail-Closed): {review_text[:200]}")
        approved = False
        issues = ["Error de formato en la revisión médica. Forzando regeneración por seguridad clínica."]
        severity = "critical"
    
    # ============================================================
    # VALIDACIÓN DETERMINISTA DE DESPENSA Y ANTI-REPETICIÓN (Post-LLM)
    # Verifica que el plan cumpla restricciones de inventario y no repita platos.
    # ============================================================
    if approved:
        # 1. Validación Estricta de Despensa (Pantry Guardrail)
        is_rotation = form_data.get("_is_rotation_reroll", False)
        if is_rotation:
            from constants import validate_ingredients_against_pantry
            current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
            
            clean_pantry = []
            if current_pantry and isinstance(current_pantry, list):
                clean_pantry = [item.strip() for item in current_pantry if item and isinstance(item, str) and len(item) > 2]
                
            if clean_pantry:
                val_result = validate_ingredients_against_pantry(all_ingredients, clean_pantry, strict_quantities=False)
                if val_result is not True:
                    approved = False
                    issues.append(val_result)  # val_result es el string de error generado por constants.py
                    severity = "high"
                    print(f"🚨 [PANTRY GUARD] Validación fallida en Revisor Médico.")
                else:
                    print(f"✅ [PANTRY GUARD] Todos los ingredientes cumplen con la despensa.")

        # 2. Validación Anti-Repetición
    if approved:
        try:
            user_id = form_data.get("user_id") or form_data.get("session_id")
            if user_id and user_id != "guest":

                recent_meal_names = get_recent_meals_from_plans(user_id, days=3)
                if recent_meal_names:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        future = executor.submit(
                            _validar_repeticiones_cpu_bound,
                            recent_meal_names,
                            days
                        )
                        repeated_meals = future.result()
                    
                    # Umbral: cero tolerancia, pero excluir nombres genéricos de desayuno
                    generic_ignores = ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa']
                    filtered_repeated = [rm for rm in repeated_meals if not any(g in rm for g in generic_ignores)]
                    
                    if len(filtered_repeated) > 0:
                        approved = False
                        issues.append(
                            f"REPETICIÓN DETECTADA: Los siguientes platos principales ya aparecieron en planes recientes y deben ser reemplazados por alternativas completamente diferentes: {', '.join(repeated_meals)}."
                        )
                        severity = "minor"
                        print(f"🔄 [ANTI-REPETICIÓN] {len(repeated_meals)} platos repetidos detectados: {repeated_meals}")
                    else:
                        print(f"✅ [ANTI-REPETICIÓN] Sin repeticiones detectadas contra {len(recent_meal_names)} platos recientes.")
            else:
                # Fallback para guests: validar contra el history_context in-memory
                history_ctx = state.get("history_context", "") if isinstance(state, dict) else ""
                if history_ctx and days:
                    # Extraer nombres de platos del history_context usando patrón común
                    # Los planes en history usan formato: "- NombrePlato" o "name: NombrePlato"
                    guest_recent = []
                    for line in history_ctx.split("\n"):
                        line = line.strip()
                        if line.startswith("- ") and len(line) > 5 and not line[2:].strip().startswith("["):
                            candidate = line[2:].strip()
                            # Filtrar líneas que parecen ingredientes (contienen cantidades)
                            if not _re.match(r'^\d', candidate) and len(candidate.split()) <= 8:
                                guest_recent.append(candidate)
                    if guest_recent:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            future = executor.submit(
                                _validar_repeticiones_cpu_bound,
                                guest_recent,
                                days
                            )
                            repeated_meals = future.result()
                        filtered_guest_repeated = [rm for rm in repeated_meals if not any(g in rm for g in ['huevosrevueltos', 'huevoshervidos', 'avenacocida', 'panezekiel', 'tostada', 'arepa'])]
                        if len(filtered_guest_repeated) > 0:
                            approved = False
                            issues.append(
                                f"REPETICIÓN DETECTADA (Guest): {', '.join(filtered_guest_repeated)}. Regenerar con variantes diferentes."
                            )
                            severity = "minor"
                            print(f"🔄 [ANTI-REPETICIÓN GUEST] {len(filtered_guest_repeated)} platos repetidos detectados")
        except Exception as e:
            print(f"⚠️ [ANTI-REPETICIÓN] Error en validación (no bloqueante): {e}")
    
    
    if approved:
        print(f"✅ [REVISOR MÉDICO] Plan APROBADO en {duration}s ✅")
        return {
            "review_passed": True,
            "review_feedback": ""
        }
    else:
        feedback = "\n".join([f"• {issue}" for issue in issues])
        print(f"❌ [REVISOR MÉDICO] Plan RECHAZADO en {duration}s (Severidad: {severity})")
        print(f"   Problemas encontrados:")
        for issue in issues:
            print(f"   ❌ {issue}")
        return {
            "review_passed": False,
            "review_feedback": feedback
        }


# ============================================================
# DECISIÓN CONDICIONAL: ¿Repetir o finalizar?
# ============================================================
def should_retry(state: PlanState) -> str:
    """Decide si regenerar el plan o enviarlo al usuario."""
    MAX_ATTEMPTS = 2
    
    if state.get("review_passed", False):
        print("✅ [ORQUESTADOR] Revisión aprobada → Enviando al usuario.")
        return "end"
    
    if state.get("attempt", 0) >= MAX_ATTEMPTS:
        if not state.get("review_passed", False):
            print(f"🚨 [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada → Tolerando y enviando mejor versión disponible.")
            return "end"
        print(f"⚠️  [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado → Enviando mejor versión disponible.")
        return "end"
    
    print("🔄 [ORQUESTADOR] Revisión fallida → Regenerando plan con correcciones...")
    return "retry"


# ============================================================
# CONSTRUCTOR DEL GRAFO
# ============================================================
def build_plan_graph() -> StateGraph:
    """Construye y compila el grafo de orquestación LangGraph Map-Reduce."""

    graph = StateGraph(PlanState)

    # Agregar nodos del pipeline Map-Reduce
    graph.add_node("plan_skeleton", plan_skeleton_node)
    graph.add_node("generate_days_parallel", generate_days_parallel_node)
    graph.add_node("assemble_plan", assemble_plan_node)
    graph.add_node("review_plan", review_plan_node)

    # Definir flujo: skeleton → parallel days → assemble → review
    graph.set_entry_point("plan_skeleton")
    graph.add_edge("plan_skeleton", "generate_days_parallel")
    graph.add_edge("generate_days_parallel", "assemble_plan")
    graph.add_edge("assemble_plan", "review_plan")

    # Edge condicional: revisor decide si regenerar o terminar
    graph.add_conditional_edges(
        "review_plan",
        should_retry,
        {
            "retry": "plan_skeleton",  # Vuelve al planificador en caso de rechazo
            "end": END
        }
    )

    return graph.compile()

# Module-level singleton: el grafo compilado es stateless y reutilizable entre requests.
_PLAN_GRAPH = build_plan_graph()


# ============================================================
# FUNCIÓN PÚBLICA: Ejecutar el pipeline completo
# ============================================================
def run_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "", progress_callback=None) -> dict:
    """
    Ejecuta el pipeline completo de generación de planes (Map-Reduce):
    Calculador → Planificador → Generadores Paralelos (×3) → Ensamblador → Revisor Médico → (loop si falla)
    
    Args:
        progress_callback: Función opcional que recibe dicts de progreso para streaming SSE.
    """
    print("\n" + "🔗" * 30)
    print("🔗 [LANGGRAPH] Iniciando Pipeline Multi-Agente")
    print("🔗" * 30)
    
    pipeline_start = time.time()
    
    # 0. Copia segura de form_data para no mutar origenes (evita guardar vars temporales en DB)
    actual_form_data = form_data.copy()
    
    # 1. Pre-calcular nutrición
    nutrition = get_nutrition_targets(actual_form_data)
    
    # 2. Preparar contexto del historial (memoria inteligente y platos recientes)
    history_context = ""
    if memory_context:
        history_context = memory_context + "\n"

    user_id = form_data.get("user_id") or form_data.get("session_id")
    if user_id == "guest":
        user_id = None
        
    # Nuevo motor anti-repetición robusto: Query directo a la base de datos
    if user_id:
        try:
            recent_meals = get_recent_meals_from_plans(user_id, days=5)
            if recent_meals:
                history_context += (
                    "\n\n--- HISTORIAL RECIENTE (PLATOS YA GENERADOS) ---\n"
                    "Estos platos fueron generados recientemente para el usuario:\n"
                    f"{json.dumps(recent_meals, ensure_ascii=False)}\n"
                    "🚨 REGLA DE ORO OBLIGATORIA: Puedes reutilizar los mismos INGREDIENTES de estos platos para optimizar tus ingredientes, PERO ESTÁ ESTRICTAMENTE PROHIBIDO repetir el mismo PLATO O PREPARACIÓN EXACTA.\n"
                    "Por ejemplo: Si dice 'Mangú de Plátano', NO uses Mangú, pero sí puedes usar el plátano para un Mofongo o Plátano Hervido.\n"
                    "Cambia la forma de cocinarlos y combínalos distinto. NO repitas el mismo nombre o concepto de plato en toda la semana (a menos que el usuario lo pida).\n"
                    "----------------------------------------------------------------------"
                )
        except Exception as e:
            print(f"⚠️ Error recuperando comidas recientes desde db: {e}")
            
        try:
            from db_facts import get_consumed_meals_since
            from datetime import datetime, timezone, timedelta
            
            since_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            consumed = get_consumed_meals_since(user_id, since_date)
            
            if consumed:
                consumed_names = [f"- {m.get('meal_name')} ({m.get('calories')} kcals)" for m in consumed]
                history_context += (
                    "\n\n--- FEEDBACK EVOLUTIVO (ÚLTIMOS 7 DÍAS) ---\n"
                    "El usuario hizo tracking de los siguientes platos en la ventana JIT anterior:\n"
                    f"{chr(10).join(consumed_names)}\n"
                    "INSTRUCCIÓN: Observa qué tipos de platos tuvieron éxito. "
                    "Incentiva opciones similares pero NO repitas las mismas recetas exactas.\n"
                    "----------------------------------------------------------------------\n"
                )
        except Exception as e:
            print(f"⚠️ Error recuperando feedback evolutivo de 7 días: {e}")
            
    # 2.15 --- REGENERACIÓN DEL MISMO DÍA (RECHAZO EXPLÍCITO) ---
    previous_meals = actual_form_data.get("previous_meals", [])
    if previous_meals and user_id:
        try:
            
            # Check if this is a Pantry Rotation vs a Full Rejected Plan Regeneration
            is_rotation = bool(actual_form_data.get("current_pantry_ingredients") or actual_form_data.get("current_shopping_list"))

            # Si el plan anterior se generó en el mismo día, interpretamos como RECHAZO o ROTACIÓN
            if check_meal_plan_generated_today(user_id):
                if is_rotation:
                    print("🔄 [ROTACIÓN DE PLATOS] Generando nuevas recetas ESTRICTAMENTE con la misma despensa.")
                    
                    current_pantry = actual_form_data.get("current_pantry_ingredients") or actual_form_data.get("current_shopping_list", [])
                    pantry_list_str = ", ".join(current_pantry) if current_pantry else "Ninguno detectado."
                    
                    history_context += (
                        f"\n\n🚨 INSTRUCCIÓN DE ROTACIÓN DE MENÚ 🚨\n"
                        f"El usuario solicitó 'Rotar Platos'. EVITA estas preparaciones anteriores:\n{', '.join(previous_meals)}\n"
                        f"DEBES inventar nuevas recetas pero OBLIGATORIAMENTE usando SOLO los ingredientes permitidos en la despensa base actual: {pantry_list_str}.\n"
                        f"ESTÁ ESTRICTAMENTE PROHIBIDO INVENTAR INGREDIENTES QUE NO ESTÉN EN ESTA LISTA EXACTA.\n"
                        f"----------------------------------------------------------------------\n"
                    )
                    actual_form_data["_is_rotation_reroll"] = True
                else:
                    print("🔄 [REGENERACIÓN] Usuario solicitó 'Generar Nueva Opción' el mismo día = RECHAZO del menú actual.")
                    
                    history_context += (
                        f"\n\n🚨 INSTRUCCIÓN DE VARIEDAD (RE-ROLL) 🚨\n"
                        f"El usuario quiere cambiar completamente las opciones de hoy:\n{', '.join(previous_meals)}\n"
                        f"REGLA CREATIVA: Inventa preparaciones inéditas. Cambia el método de cocción, la combinación o el corte para sorprender al usuario con algo nuevo.\n"
                        f"----------------------------------------------------------------------\n"
                    )
                    actual_form_data["_is_same_day_reroll"] = True
            else:
                print("🌅 [NUEVO DÍA] Generación para un nuevo día iniciada.")
        except Exception as e:
            print(f"⚠️ Error validando regeneración del mismo día: {e}")


    # 2.5 Buscar Hechos y Diario Visual en Memoria Vectorial (RAG multimodal)
    user_facts_text = ""
    visual_facts_text = ""
    facts_data_sorted = []
    visual_list = []
    if user_id:
        try:
            
            # 1. Recuperación estricta (Metadata JSONB) - ALERGIAS, CONDICIONES, RECHAZOS
            strict_facts_text = ""
            alergias = get_user_facts_by_metadata(user_id, 'category', 'alergia')
            if alergias:
                strict_facts_text += "🔴 ALERGIAS ESTRICTAS (PROHIBIDO USAR):\n" + "\n".join([f"  - {a['fact']}" for a in alergias]) + "\n"
                
            rechazos = get_user_facts_by_metadata(user_id, 'category', 'rechazo')
            if rechazos:
                strict_facts_text += "🔴 RECHAZOS (NO USAR):\n" + "\n".join([f"  - {r['fact']}" for r in rechazos]) + "\n"
                
            condiciones = get_user_facts_by_metadata(user_id, 'category', 'condicion_medica')
            if condiciones:
                strict_facts_text += "⚠️ CONDICIONES MÉDICAS (ADAPTAR PLAN):\n" + "\n".join([f"  - {c['fact']}" for c in condiciones]) + "\n"

            if strict_facts_text:
                user_facts_text += "=== REGLAS MÉDICAS Y DE GUSTO ABSOLUTAS (Extraídas de Base de Datos Estructurada) ===\n"
                user_facts_text += strict_facts_text + "=================================================================================\n\n"

            # 2. Buscar hechos textuales — QUERY DINÁMICA (Vectorial) para contexto general
            dynamic_parts = []
            if form_data.get("mainGoal"):
                dynamic_parts.append(f"Objetivo: {form_data['mainGoal']}")
            if form_data.get("allergies"):
                allergies = form_data["allergies"] if isinstance(form_data["allergies"], list) else [form_data["allergies"]]
                dynamic_parts.append(f"Alergias: {', '.join(allergies)}")
            if form_data.get("medicalConditions"):
                conditions = form_data["medicalConditions"] if isinstance(form_data["medicalConditions"], list) else [form_data["medicalConditions"]]
                dynamic_parts.append(f"Condiciones: {', '.join(conditions)}")
            if form_data.get("dietType"):
                dynamic_parts.append(f"Dieta: {form_data['dietType']}")
            if form_data.get("dislikes"):
                dislikes = form_data["dislikes"] if isinstance(form_data["dislikes"], list) else [form_data["dislikes"]]
                dynamic_parts.append(f"No le gusta: {', '.join(dislikes)}")
            if form_data.get("struggles"):
                struggles = form_data["struggles"] if isinstance(form_data["struggles"], list) else [form_data["struggles"]]
                dynamic_parts.append(f"Obstáculos: {', '.join(struggles)}")
            
            dynamic_query = ". ".join(dynamic_parts) if dynamic_parts else "Preferencias de comida, restricciones médicas, gustos y síntomas digestivos del usuario"
            print(f"🔍 [RAG] Query dinámica: {dynamic_query}")
            
            query_emb = get_embedding(dynamic_query)
            if query_emb:
                # Usar búsqueda híbrida pasando el texto
                facts_data = search_user_facts(user_id, query_emb, query_text=dynamic_query, threshold=0.5, limit=10)
                if facts_data:
                    # === PRIORIZACIÓN POR CATEGORÍA (Anti-Poda Bruta) ===
                    # Ordenar por peso de categoría para que el truncado solo corte lo irrelevante
                    CATEGORY_PRIORITY_WEIGHTS = {
                        "alergia": 0,            # Máxima prioridad (ya están en strict, pero por si acaso)
                        "condicion_medica": 1,    
                        "rechazo": 2,             
                        "dieta": 3,               
                        "objetivo": 4,            
                        "preferencia": 5,         
                        "sintoma_temporal": 6,    # Menor prioridad
                    }
                    
                    def get_fact_weight(fact_item):
                        meta = fact_item.get("metadata", {})
                        if isinstance(meta, dict):
                            cat = meta.get("category", "")
                            return CATEGORY_PRIORITY_WEIGHTS.get(cat, 7)
                        return 7  # Sin categoría → al final
                    
                    facts_data_sorted = sorted(facts_data, key=get_fact_weight)
                    print(f"🧠 [RAG] Hechos textuales recuperados: {len(facts_data)} (ordenados por prioridad de categoría)")
                    
            # Buscar memoria visual 
            visual_query_emb = get_multimodal_embedding(dynamic_query)
            if visual_query_emb:
                visual_data = search_visual_diary(user_id, visual_query_emb, threshold=0.5, limit=10)
                if visual_data:
                    visual_list = [f"• {item['description']}" for item in visual_data]
                    print(f"📸 [VISUAL RAG] Entradas visuales recuperadas: {len(visual_data)}")
                    
        except Exception as e:
            print(f"⚠️ [RAG] Error recuperando memoria: {e}")
            
    # === PRUNING FACT-BY-FACT (Basado en Tokens) ===
    # Estrategia: Los hechos estrictos (alergias, condiciones, rechazos) son NON-NEGOTIABLE.
    # Luego se agregan hechos generales y visuales uno por uno calculando ~4 caracteres por token.
    MAX_CONTEXT_TOKENS = 30000
    
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # 1. Strict facts son siempre incluidos (non-negotiable)
    full_rag_context = user_facts_text  # Ya contiene alergias, rechazos, condiciones
    
    # 2. Calcular presupuesto restante para hechos generales + visuales
    current_tokens = estimate_tokens(full_rag_context)
    remaining_tokens = MAX_CONTEXT_TOKENS - current_tokens
    
    if remaining_tokens > 0 and facts_data_sorted:
        header = "--- CONTEXTO GENERAL (Memoria Semántica, ordenado por prioridad) ---\n"
        general_section = header
        included_count = 0
        skipped_count = 0
        
        current_section_tokens = estimate_tokens(general_section)
        
        for item in facts_data_sorted:
            fact_line = f"• {item['fact']}\n"
            fact_tokens = estimate_tokens(fact_line)
            
            if current_section_tokens + fact_tokens <= remaining_tokens:
                general_section += fact_line
                current_section_tokens += fact_tokens
                included_count += 1
            else:
                skipped_count += 1
        
        if included_count > 0:
            full_rag_context += general_section
            remaining_tokens -= current_section_tokens
            if skipped_count > 0:
                print(f"✂️ [PRUNING] {skipped_count} hechos de baja prioridad descartados completos (límite tokens)")
    
    # 3. Agregar hechos visuales con el presupuesto restante
    if remaining_tokens > 25 and visual_list:  # ~100 chars
        visual_header = "\n--- INVENTARIO Y DIARIO VISUAL ---\nEl usuario subió fotos de estos alimentos:\n"
        visual_section = visual_header
        visual_section_tokens = estimate_tokens(visual_header)
        
        for vline in visual_list:
            candidate = vline + "\n"
            candidate_tokens = estimate_tokens(candidate)
            if visual_section_tokens + candidate_tokens <= remaining_tokens:
                visual_section += candidate
                visual_section_tokens += candidate_tokens
            else:
                break
        
        if len(visual_section) > len(visual_header):
            full_rag_context += visual_section
    
    if full_rag_context.strip():
        print(f"✅ [PRUNING] Contexto final: {estimate_tokens(full_rag_context)} tokens aprox (fact-by-fact, sin cortes)")
    # ====================================================
    
    # 3. Estado inicial del grafo
    initial_state: PlanState = {
        "form_data": actual_form_data,
        "taste_profile": taste_profile,
        "nutrition": nutrition,
        "history_context": history_context,
        "user_facts": full_rag_context,
        "plan_result": None,
        "plan_skeleton": None,
        "review_passed": False,
        "review_feedback": "",
        "attempt": 0,
        "progress_callback": progress_callback,
    }
    
    # 4. Ejecutar el grafo (singleton compilado a nivel de módulo)
    final_state = _PLAN_GRAPH.invoke(initial_state)
    
    pipeline_duration = round(time.time() - pipeline_start, 2)
    
    print(f"\n{'🔗' * 30}")
    print(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
    print(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
    print("🔗" * 30 + "\n")
    
    return final_state["plan_result"]
