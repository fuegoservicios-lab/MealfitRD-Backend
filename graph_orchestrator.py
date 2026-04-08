# backend/graph_orchestrator.py
"""
Orquestación LangGraph: Flujo cíclico multi-agente para generación de planes nutricionales.
Generador → Revisor Médico → (loop si falla, max 2 intentos)
"""

import os
import time
import json
from typing import TypedDict, Optional
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
from fact_extractor import get_embedding
from vision_agent import get_multimodal_embedding
from db import get_recent_techniques, get_recent_meals_from_plans, check_meal_plan_generated_today, get_custom_shopping_items, search_user_facts, search_visual_diary, get_user_facts_by_metadata
from nutrition_calculator import get_nutrition_targets
from constants import TECHNIQUE_FAMILIES, ALL_TECHNIQUES, TECH_TO_FAMILY, SUPPLEMENT_NAMES

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
    
    # Revisión médica
    review_passed: bool
    review_feedback: str
    
    # Control de flujo
    attempt: int
    user_facts: str


# ============================================================
# SCHEMAS (importados del módulo canónico schemas.py)
# ============================================================
from schemas import MacrosModel, MealModel, DailyPlanModel, PlanModel


# ============================================================
# PROMPTS (importados de prompts.py)
# ============================================================
from prompts import GENERATOR_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT


# ============================================================
# NODO 1: AGENTE GENERADOR
# ============================================================
def generate_plan_node(state: PlanState) -> dict:
    """Genera el plan alimenticio."""
    attempt = state.get("attempt", 0) + 1
    form_data = state["form_data"]
    nutrition = state["nutrition"]
    taste_profile = state.get("taste_profile", "")
    history_context = state.get("history_context", "")
    review_feedback = state.get("review_feedback", "")
    
    print(f"\n{'='*60}")
    print(f"🍽️  [AGENTE GENERADOR] Intento #{attempt}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    nutrition_context = f"""
--- TARGETS NUTRICIONALES CALCULADOS (Fórmula Mifflin-St Jeor) ---
⚠️ ESTOS NÚMEROS SON EXACTOS. NO LOS RECALCULES.

• BMR: {nutrition['bmr']} kcal
• TDEE: {nutrition['tdee']} kcal  
• 🎯 CALORÍAS OBJETIVO: {nutrition['target_calories']} kcal ({nutrition['goal_label']})
• Proteína: {nutrition['macros']['protein_g']}g | Carbos: {nutrition['macros']['carbs_g']}g | Grasas: {nutrition['macros']['fats_g']}g

IMPORTANTE: calories DEBE ser {nutrition['target_calories']}.
macros DEBEN ser: protein='{nutrition['macros']['protein_str']}', carbs='{nutrition['macros']['carbs_str']}', fats='{nutrition['macros']['fats_str']}'.
-------------------------------------------------------------------
"""
    
    # Si hay feedback del revisor, agregarlo como corrección urgente
    correction_context = ""
    if review_feedback:
        correction_context = f"""
⚠️⚠️⚠️ CORRECCIÓN URGENTE DEL REVISOR MÉDICO ⚠️⚠️⚠️
El plan anterior fue RECHAZADO por las siguientes razones:
{review_feedback}

DEBES corregir TODOS estos problemas en esta nueva versión.
Genera comidas COMPLETAMENTE DIFERENTES que NO tengan estos problemas.
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
"""
    
    # Inyectar hechos recuperados de RAG
    rag_context = ""
    user_facts = state.get("user_facts", "")
    if user_facts:
        rag_context = f"""
--- HECHOS PERMANENTES DEL USUARIO (MEMORIA VECTORIAL) ---
Estos son datos críticos que debes respetar.
{user_facts}
----------------------------------------------------------
"""

    # --- 🕒 INYECCIÓN DE CONTEXTO TEMPORAL DINÁMICO ---
    now_local = datetime.now()
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    dia_str = dias[now_local.weekday()]
    mes_str = meses_es[now_local.month - 1]
    
    # Estación simplificada adaptada al trópico/Hemisferio Norte
    estacion = "Verano"
    if now_local.month in [3, 4, 5]: estacion = "Primavera"
    elif now_local.month in [9, 10, 11]: estacion = "Otoño"
    elif now_local.month in [12, 1, 2]: estacion = "Invierno"

    time_context = (
        f"\n--- 📅 CONTEXTO TEMPORAL ACTUAL (OBLIGATORIO) ---\n"
        f"Hoy es {dia_str}, {now_local.day} de {mes_str} de {now_local.year}. Estación promedio local: {estacion} tropical.\n"
        f"Aplica las estrategias de Continuidad Temporal considerando que el usuario comenzará este plan en estos días.\n"
        f"--------------------------------------------------\n"
    )

    random_seed = random.randint(10000, 99999)
    
    # --- 🎲 INVERSIÓN DE CONTROL DETERMINISTA (ANTI MODE-COLLAPSE) ---
    # Python selecciona proteínas y carbos; el LLM solo "cocina" con ellos.
    # Extraer user_id para análisis de frecuencia basado en JSON de planes guardados
    _uid = form_data.get("user_id") or form_data.get("session_id")
    if _uid == "guest": _uid = None
    from ai_helpers import get_deterministic_variety_prompt  # No genera ciclo
    variety_prompt = get_deterministic_variety_prompt(history_context, form_data, user_id=_uid)
    print(f"🎲 [ORQUESTADOR] Inyectando variedad determinista en el generador.")
    
    # --- 🎲 INYECTOR DINÁMICO DE VARIEDAD CULINARIA ---
    # Clasificación por familia: garantiza que los 3 días usen perfiles de cocción diferentes.
    # Evita que las 3 técnicas sean "secas" (ej: Horneado + Airfryer + Parrilla).
    
    # Query estructurado contra la DB para contar frecuencia de cada técnica CON decaimiento temporal
    technique_freq = {}
    if _uid:
        try:
            recent_techs = get_recent_techniques(_uid, limit=6)
            # Construir mapa de frecuencia con decaimiento: 0.9^days_elapsed
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
                decayed_weight = decay_factor ** days_elapsed  # 1.0 hoy, 0.9 ayer, 0.81 antier...
                technique_freq[t] = technique_freq.get(t, 0) + decayed_weight
            if technique_freq:
                print(f"🔍 [TÉCNICAS] Frecuencias con decaimiento temporal: { {k: round(v, 2) for k, v in technique_freq.items()} }")
        except Exception as e:
            print(f"⚠️ [TÉCNICAS] Error consultando DB, usando pesos uniformes: {e}")
    
    # Selección ponderada por frecuencia inversa CON diversificación de familias.
    # Algoritmo: elegir 1 técnica de una familia diferente en cada iteración.
    # Esto garantiza que "Horneado + Guiso + Ceviche" (seca/húmeda/fresca) sea más
    # probable que "Horneado + Airfryer + Parrilla" (seca/seca/seca).
    selected_techniques = []
    used_families = set()
    _pool_t = [(t, 1.0 / (technique_freq.get(t, 0) + 1)) for t in ALL_TECHNIQUES]
    
    while len(selected_techniques) < 3 and _pool_t:
        # Fase 1: Preferir técnicas de familias NO usadas aún
        cross_family_pool = [(t, w) for t, w in _pool_t if TECH_TO_FAMILY.get(t) not in used_families]
        
        # Fase 2: Si todas las familias ya fueron usadas, tomar de cualquier familia restante
        active_pool = cross_family_pool if cross_family_pool else _pool_t
        
        pick = random.choices([x[0] for x in active_pool], weights=[x[1] for x in active_pool], k=1)[0]
        selected_techniques.append(pick)
        used_families.add(TECH_TO_FAMILY.get(pick, ""))
        _pool_t = [(t, w) for t, w in _pool_t if t != pick]
    
    print(f"👨‍🍳 [TÉCNICAS] Seleccionadas (familias diversas): {[f'{t} ({TECH_TO_FAMILY.get(t)})' for t in selected_techniques]}")
    
    technique_injection = (
        f"\n--- 👨🍳 INSTRUCCIÓN DINÁMICA DE VARIEDAD (OBLIGATORIA) ---\n"
        f"Para cumplir la regla de usar los MISMOS ingredientes del supermercado pero crear PLATOS DIFERENTES, "
        f"aplica obligatoriamente estas técnicas de cocción a las comidas principales (Almuerzo o Cena):\n"
        f"• Día 1 (Opción A): Aplica técnica '{selected_techniques[0]}'\n"
        f"• Día 2 (Opción B): Aplica técnica '{selected_techniques[1]}'\n"
        f"• Día 3 (Opción C): Aplica técnica '{selected_techniques[2]}'\n"
        f"Ajusta los gramos matemáticamente para cumplir las macros.\n"
        f"----------------------------------------------------------\n"
    )
    
    # --- SUPLEMENTOS (Condicional) ---
    supplements_context = ""
    if form_data.get("includeSupplements"):
        selected_supps = form_data.get("selectedSupplements", [])
        
        # Mapa de keys a nombres legibles para el prompt
        
        if selected_supps:
            # El usuario eligió suplementos específicos
            supp_names = [SUPPLEMENT_NAMES.get(s, s) for s in selected_supps]
            # Generar lista de suplementos NO seleccionados para refuerzo negativo
            all_supps = set(SUPPLEMENT_NAMES.keys())
            not_selected = all_supps - set(selected_supps)
            not_selected_names = [SUPPLEMENT_NAMES.get(s, s) for s in not_selected]
            
            supplements_context = (
                "\n--- 💊 SUPLEMENTOS SELECCIONADOS (OBLIGATORIO — LEE CON CUIDADO) ---\n"
                f"LISTA EXACTA de suplementos que DEBES incluir: {', '.join(supp_names)}\n"
                f"TOTAL: {len(supp_names)} suplemento(s). Ni más, ni menos.\n\n"
                "⚠️ PROHIBIDO incluir cualquier suplemento que NO esté en la lista de arriba.\n"
            )
            if not_selected_names:
                supplements_context += (
                    f"❌ NO INCLUIR (el usuario NO los seleccionó): {', '.join(not_selected_names)}\n"
                )
            supplements_context += (
                "\nPara CADA día del plan, agrega una sección 'supplements' con SOLO los suplementos listados arriba.\n"
                "Cada suplemento: 'name' (nombre exacto), 'dose' (dosis), 'timing' (momento del día), 'reason' (justificación).\n"
                "---------------------------------------------------\n"
            )
        else:
            # Toggle activado pero sin selección específica → recomendación libre
            supplements_context = (
                "\n--- 💊 SUPLEMENTOS PERSONALIZADOS (OBLIGATORIO) ---\n"
                "El usuario ACTIVÓ la opción de incluir suplementos en su plan pero NO seleccionó suplementos específicos.\n"
                "DEBES agregar para CADA día del plan una sección 'supplements' (lista de objetos) con suplementos personalizados.\n"
                "Cada suplemento debe tener: 'name' (nombre), 'dose' (dosis), 'timing' (momento del día), 'reason' (justificación breve).\n"
                "Adapta las recomendaciones al objetivo del usuario, su nivel de actividad y condiciones médicas.\n"
                "Ejemplos: Proteína Whey, Creatina Monohidrato, Omega-3, Vitamina D3, Multivitamínico, Magnesio, etc.\n"
                "---------------------------------------------------\n"
            )
    
    # --- DURACIÓN DE COMPRA (Condicional) ---
    grocery_duration = form_data.get("groceryDuration", "weekly")
    grocery_duration_context = ""
    DURATION_LABELS = {"weekly": "SEMANAL (7 días)", "biweekly": "QUINCENAL (15 días)", "monthly": "MENSUAL (30 días)"}
    DURATION_DAYS = {"weekly": 7, "biweekly": 15, "monthly": 30}
    if grocery_duration and grocery_duration != "weekly":
        label = DURATION_LABELS.get(grocery_duration, "SEMANAL (7 días)")
        days_num = DURATION_DAYS.get(grocery_duration, 7)
        grocery_duration_context = (
            f"\n--- 🛒 DURACIÓN DE COMPRA: {label} (OBLIGATORIO) ---\n"
            f"El usuario compra alimentos para {days_num} días en una sola ida al supermercado.\n"
            f"DEBES priorizar ingredientes que se conserven bien durante {days_num} días.\n"
        )
        if grocery_duration == "monthly":
            grocery_duration_context += (
                "Usa predominantemente: granos secos, arroz, avena, tubérculos (yuca, batata, plátano verde),\n"
                "proteínas congelables (pollo, carne, pescado empacado al vacío), leche en polvo o UHT, huevos.\n"
                "Para cualquier perecedero, incluye instrucciones de congelación en la receta.\n"
            )
        elif grocery_duration == "biweekly":
            grocery_duration_context += (
                "Equilibra entre frescos e ingredientes duraderos. Los vegetales de hoja y frutas muy maduras\n"
                "deben usarse en los primeros días del plan. Planifica congelación para proteínas frescas.\n"
            )
        grocery_duration_context += (
            f"RECUERDA: Los PLATOS varían cada día, pero los ALIMENTOS BASE se repiten durante los {days_num} días.\n"
            "---------------------------------------------------\n"
        )
    
    prompt_text = (
        f"Analiza la siguiente información del usuario y genera un plan de comidas de 3 días.\n"
        f"IMPORTANTE: Genera opciones creativas y diferentes a planes anteriores. Semilla de generación aleatoria: {random_seed}\n\n"
        f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
        f"{nutrition_context}\n{time_context}\n{taste_profile}\n{rag_context}\n{correction_context}\n{history_context}\n{variety_prompt}\n{technique_injection}\n{supplements_context}\n{grocery_duration_context}\n\n"
        f"{GENERATOR_SYSTEM_PROMPT}"
    )
    
    is_re_roll = form_data.get("_is_same_day_reroll", False)
    # Structured Output constraints crash/hang on Gemini if temperature > 1.0 due to generation mask mismatch.
    # We use 0.7 for standard, 0.95 for maximum variety same-day reroll.
    base_temp = 0.95 if is_re_roll else 0.7
    
    generator_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=base_temp if attempt == 1 else (base_temp + 0.1),  
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        max_retries=0,
        timeout=120
    ).with_structured_output(PlanModel)
    
    # Invocar LLM con reintentos automáticos (tenacity)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
        before_sleep=lambda retry_state: print(f"⚠️  [GENERADOR] Reintento #{retry_state.attempt_number} tras error de formato...")
    )
    def invoke_with_retry():
        print(f"⏳ [DEBUG] LLM invocado. Generando esquema JSON gigante, espera hasta 60s...")
        return generator_llm.invoke(prompt_text)
    
    response = invoke_with_retry()
    
    duration = round(time.time() - start_time, 2)
    print(f"✅ [GENERADOR] Plan creado en {duration}s")
    
    # Convertir a dict
    if hasattr(response, "model_dump"):
        result = response.model_dump()
    elif isinstance(response, dict):
        result = response
    else:
        result = response.dict()
    
    # Post-proceso: forzar valores exactos del calculador
    # Usamos total_daily_calories si existe para que el Dashboard muestre el objetivo completo del día.
    result["calories"] = nutrition.get("total_daily_calories", nutrition["target_calories"])
    
    active_macros = nutrition.get("total_daily_macros", nutrition["macros"])
    result["macros"] = {
        "protein": active_macros["protein_str"],
        "carbs": active_macros["carbs_str"],
        "fats": active_macros["fats_str"],
    }
    result["main_goal"] = nutrition["goal_label"]
    
    # Guardar técnicas seleccionadas para persistencia en DB (se extraen en app.py)
    result["_selected_techniques"] = selected_techniques
    
    return {
        "plan_result": result,
        "attempt": attempt
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
    # VALIDACIÓN DETERMINISTA ANTI-REPETICIÓN (Post-LLM)
    # Verifica que el plan NO repita platos recientes del usuario.
    # Esto es puro Python, sin costo de LLM adicional.
    # ============================================================
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
                    
                    # Umbral: cero tolerancia - si 1 o más comidas se repiten, rechazar
                    if len(repeated_meals) > 0:
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
                        if len(repeated_meals) > 0:
                            approved = False
                            issues.append(
                                f"REPETICIÓN DETECTADA (Guest): {', '.join(repeated_meals)}. Regenerar con variantes diferentes."
                            )
                            severity = "minor"
                            print(f"🔄 [ANTI-REPETICIÓN GUEST] {len(repeated_meals)} platos repetidos detectados")
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
            print(f"🚨 [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado y revisión NO aprobada → ABORTANDO por seguridad.")
            raise ValueError("Imposible generar un plan seguro tras varios intentos. Por favor, inténtalo de nuevo.")
        print(f"⚠️  [ORQUESTADOR] Máximo de {MAX_ATTEMPTS} intentos alcanzado → Enviando mejor versión disponible.")
        return "end"
    
    print("🔄 [ORQUESTADOR] Revisión fallida → Regenerando plan con correcciones...")
    return "retry"


# ============================================================
# CONSTRUCTOR DEL GRAFO
# ============================================================
def build_plan_graph() -> StateGraph:
    """Construye y compila el grafo de orquestación LangGraph."""
    
    graph = StateGraph(PlanState)
    
    # Agregar nodos
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("review_plan", review_plan_node)
    
    # Definir flujo
    graph.set_entry_point("generate_plan")
    graph.add_edge("generate_plan", "review_plan")
    
    # Edge condicional: revisor decide si repetir o terminar
    graph.add_conditional_edges(
        "review_plan",
        should_retry,
        {
            "retry": "generate_plan",
            "end": END
        }
    )
    
    return graph.compile()

# Module-level singleton: el grafo compilado es stateless y reutilizable entre requests.
_PLAN_GRAPH = build_plan_graph()


# ============================================================
# FUNCIÓN PÚBLICA: Ejecutar el pipeline completo
# ============================================================
def run_plan_pipeline(form_data: dict, history: list = None, taste_profile: str = "", memory_context: str = "") -> dict:
    """
    Ejecuta el pipeline completo de generación de planes:
    Calculador → Generador → Revisor Médico → (loop si falla)
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
                    "🚨 REGLA DE ORO OBLIGATORIA: Puedes reutilizar los mismos INGREDIENTES de estos platos para optimizar las compras del supermercado, PERO ESTÁ ESTRICTAMENTE PROHIBIDO repetir el mismo PLATO O PREPARACIÓN EXACTA.\n"
                    "Por ejemplo: Si dice 'Mangú de Plátano', NO uses Mangú, pero sí puedes usar el plátano para un Mofongo o Plátano Hervido.\n"
                    "Cambia la forma de cocinarlos y combínalos distinto. NO repitas el mismo nombre o concepto de plato en toda la semana (a menos que el usuario lo pida).\n"
                    "----------------------------------------------------------------------"
                )
        except Exception as e:
            print(f"⚠️ Error recuperando comidas recientes desde db: {e}")
            
    # 2.15 --- REGENERACIÓN DEL MISMO DÍA (RECHAZO EXPLÍCITO) ---
    previous_meals = actual_form_data.get("previous_meals", [])
    if previous_meals and user_id:
        try:
            
            # Si el plan anterior se generó en el mismo día, interpretamos como RECHAZO
            if check_meal_plan_generated_today(user_id):
                print("🔄 [REGENERACIÓN] Usuario solicitó 'Generar Nueva Opción' el mismo día = RECHAZO del menú actual.")
                
                # Inyectamos una regla simplificada y positiva para evitar que el LLM sufra parálisis de restricciones (504 Timeout)
                history_context += (
                    f"\n\n🚨 INSTRUCCIÓN DE VARIEDAD (RE-ROLL) 🚨\n"
                    f"El usuario quiere cambiar las siguientes opciones de hoy:\n{', '.join(previous_meals)}\n"
                    f"REGLA CREATIVA: Inventa preparaciones inéditas. Cambia el método de cocción, la combinación o el corte para sorprender al usuario con algo nuevo usando la misma lista de compras.\n"
                    f"----------------------------------------------------------------------\n"
                )
                # Añadimos una bandera secreta para subir la temperatura del LLM
                actual_form_data["_is_same_day_reroll"] = True
            else:
                print("🌅 [NUEVO DÍA] Generación para un nuevo día iniciada.")
        except Exception as e:
            print(f"⚠️ Error validando regeneración del mismo día: {e}")

    # 2.2 --- CONSTRICCIÓN DE LISTA DE COMPRAS ---
    if user_id:
        try:
            existing = get_custom_shopping_items(user_id)
            existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
            if existing_items:
                excluded_cats = ["Suplementos", "Limpieza y Hogar", "Higiene Personal", "Otros"]
                
                # Agrupamos los ingredientes por meal_slot para darle una instrucción detallada al modelo
                ingredients_by_slot = {
                    "Desayuno": [],
                    "Almuerzo": [],
                    "Merienda": [],
                    "Cena": [],
                }
                
                forced_proteins = []
                forced_carbs = []
                forced_veggies = []
                
                for item in existing_items:
                    cat = item.get("category", "")
                    
                    # El nombre ahora está prioritariamente en display_name (columnas estructuradas)
                    name = item.get("display_name") or ""
                    
                    # Fallback legacy: extraer de item_name si era JSON
                    if not name:
                        raw_name = item.get("item_name", "")
                        if raw_name.startswith("{"):
                            try:
                                parsed = json.loads(raw_name)
                                name = parsed.get("name", raw_name)
                            except Exception:
                                name = raw_name
                        else:
                            name = raw_name
                                
                    if not name or cat in excluded_cats:
                        continue
                    
                    meal_slot = item.get("meal_slot") or "Almuerzo"
                    if meal_slot not in ingredients_by_slot:
                        ingredients_by_slot[meal_slot] = []
                    ingredients_by_slot[meal_slot].append(name)
                    
                    cat_lower = cat.lower()
                    if any(k in cat_lower for k in ["carne", "pescado", "proteína", "protein", "huevo", "lácteo", "queso", "leche", "yogur"]):
                        forced_proteins.append(name)
                    elif any(k in cat_lower for k in ["despensa", "grano", "cereal", "arroz", "avena", "pan", "pasta", "vívere", "yuca", "plátano", "batata", "papa"]):
                        forced_carbs.append(name)
                    elif any(k in cat_lower for k in ["fruta", "verdura", "vegetal"]):
                        forced_veggies.append(name)
                
                total_ingredients = sum(len(lst) for lst in ingredients_by_slot.values())
                
                if total_ingredients > 0:
                    shopping_str = "\n".join([f"- {slot}: {', '.join(items) if items else 'Ninguno'}" for slot, items in ingredients_by_slot.items() if items])
                    
                    history_context += f"\n\n⚠️ REGLA DE SUPERMERCADO ABSOLUTA E INQUEBRANTABLE (CRÍTICO): El usuario ya tiene una lista de compras vigente y la ha categorizado por momento del día. DEBES crear TODO el nuevo plan utilizando EXCLUSIVAMENTE los siguientes ingredientes, respetando el momento para el que fueron comprados (Desayuno, Almuerzo, Cena, etc.). Los ingredientes asignados a un momento del día pueden usarse en cualquier comida si es necesario.\n\nInventario Disponible:\n{shopping_str}\n\nTIENES ESTRICTAMENTE PROHIBIDO sugerir o agregar ingredientes que no estén en este inventario.\n"
                    print(f"🛒 [CONSTRAINT] Aplicando restricción de lista de compras categorizada en graph_orchestrator.")
                    
                    if forced_proteins: actual_form_data["_force_base_proteins"] = forced_proteins
                    if forced_carbs: actual_form_data["_force_base_carbs"] = forced_carbs
                    if forced_veggies: actual_form_data["_force_base_veggies"] = forced_veggies
                    
        except Exception as check_e:
            print(f"⚠️ Error revisando lista de compras para restricción en orquestador: {check_e}")

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
        "review_passed": False,
        "review_feedback": "",
        "attempt": 0,
    }
    
    # 4. Ejecutar el grafo (singleton compilado a nivel de módulo)
    final_state = _PLAN_GRAPH.invoke(initial_state)
    
    pipeline_duration = round(time.time() - pipeline_start, 2)
    
    print(f"\n{'🔗' * 30}")
    print(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
    print(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
    print("🔗" * 30 + "\n")
    
    return final_state["plan_result"]
