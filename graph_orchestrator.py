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
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

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
    
    # Revisión médica
    review_passed: bool
    review_feedback: str
    
    # Control de flujo
    attempt: int
    user_facts: str


# ============================================================
# SCHEMAS (importados de agent.py para consistencia)
# ============================================================
class MacrosModel(BaseModel):
    protein: str = Field(description="Gramos de proteína totales, ej: '150g'")
    carbs: str = Field(description="Gramos de carbohidratos totales, ej: '200g'")
    fats: str = Field(description="Gramos de grasas totales, ej: '60g'")

class MealModel(BaseModel):
    meal: str = Field(description="Momento del día")
    time: str = Field(description="Hora sugerida")
    name: str = Field(description="Nombre del plato")
    desc: str = Field(description="Descripción del plato")
    prep_time: str = Field(description="Tiempo de preparación")
    cals: int = Field(description="Calorías del plato")
    macros: List[str] = Field(description="Lista rápida de macros")
    ingredients: List[str] = Field(description="Lista de ingredientes")
    recipe: List[str] = Field(description="Pasos de preparación con prefijos")

class DailyPlanModel(BaseModel):
    day: int = Field(description="Número de día (1, 2, o 3)")
    meals: List[MealModel] = Field(description="Lista de comidas estrictamente en orden cronológico: Desayuno, Almuerzo, Merienda, Cena")

class PlanModel(BaseModel):
    main_goal: str = Field(description="Objetivo principal")
    calories: int = Field(description="Total de calorías por día")
    macros: MacrosModel = Field(description="Macronutrientes por día")
    insights: List[str] = Field(description="3 frases: Diagnóstico, Estrategia, Tip del Chef")
    days: List[DailyPlanModel] = Field(description="Lista de 3 días planificados")


# ============================================================
# PROMPTS
# ============================================================
GENERATOR_SYSTEM_PROMPT = """
Eres un Nutricionista Clínico, Chef Profesional y la IA oficial de MealfitRD.
Tu misión es crear un plan alimenticio de EXACTAMENTE 3 DÍAS VARIADOS, altamente profesional y 100% adaptado a la biometría y preferencias del usuario.

REGLAS ESTRICTAS:
1. CALORÍAS Y MACROS PRE-CALCULADOS: Los cálculos de BMR, TDEE, calorías objetivo y macronutrientes ya fueron realizados por el Sistema Calculador. NO calcules estos números tú mismo. Usa EXACTAMENTE los valores provistos. La suma de calorías, proteínas, carbohidratos y grasas de todas las comidas de un día DEBE coincidir milimétricamente con el OBJETIVO DIARIO aportado. Distribuye las porciones con cuidado para lograr esta meta estricta.
2. EFICIENCIA DE SUPERMERCADO Y VARIEDAD: Diseña los 3 días utilizando listas de compras e ingredientes principales MUY SIMILARES (ej. usar Pollo, Huevos, Yuca, Avena a lo largo de los 3 días) para no gastar mucho en varias compras. Sin embargo, DEBES crear PREPARACIONES Y PLATOS COMPLETAMENTE DISTINTOS cada día usando esos mismos ingredientes combinados de forma diferente. NUNCA repitas la misma preparación exacta (Ej. Si la Opción A tiene Pollo Guisado, la Opción B debe tener Pollo a la Plancha o Desmenuzado).
3. INGREDIENTES DOMINICANOS: El menú DEBE usar alimentos típicos, accesibles y económicos de República Dominicana (Ej: Plátano, Yuca, Batata, Huevos, Salami, Queso de freír/hoja, Pollo guisado, Aguacate, Habichuelas, Arroz, Avena).
4. RECETAS PROFESIONALES: Los pasos de las recetas (`recipe`) DEBEN incluir obligatoriamente estos prefijos para la UI:
   - "Mise en place: [Instrucciones de preparación previa y cortes]"
   - "El Toque de Fuego: [Instrucciones de cocción en sartén, horno o airfryer]"
   - "Montaje: [Instrucciones de cómo servir para que luzca apetitoso]"
5. CUMPLE RESTRICCIONES ABSOLUTAMENTE: Si el usuario es vegetariano, tiene alergias (Ej. Lácteos), condiciones médicas (Ej. Diabetes T2) o indicó obstáculos (Ej: falta de tiempo, no sabe cocinar), el plan DEBE reflejar soluciones inmediatas a eso (comidas rápidas, sin azúcar, sin carne, etc).
6. ESTRUCTURA: Si el usuario indicó `skipLunch: true`, NO incluyas Almuerzo, distribuye las calorías en las demás comidas y asume que comerá la comida familiar.
7. VARIEDAD ESTRICTA: Revisa el historial de comidas anteriores provisto en el prompt (si lo hay) y NO REPITAS LOS MISMOS PLATOS NI NOMBRES EXACTOS DE LAS ÚLTIMAS 24-48 HORAS. Ofrécele opciones radicalmente diferentes en presentación y técnica de cocción, pero MANTENIENDO los mismos ingredientes base para ahorrar en el supermercado.
8. PROHIBICIÓN ABSOLUTA DE RECHAZOS: Lee detenidamente el Perfil de Gustos adjunto. Si el perfil dice que el usuario odia o rechazó un ingrediente (ej. plátano, avena), está TOTALMENTE PROHIBIDO incluirlo en este plan.
9. PESO EMOCIONAL (INTENSIDAD): Los hechos proporcionados en el contexto tienen un metadato de "intensidad" (1 a 5).
   - Intensidad 5: REGLA DE ORO. DEBES incluir este ingrediente/preferencia en el plan siempre que se ajuste a los macros.
   - Intensidad 4: Usa este ingrediente frecuentemente.
   - Intensidad 2: Usa con extrema moderación, o evítalo si es posible.
   - Intensidad 1: RECHAZO TOTAL. Trátalo igual que una prohibición o alergia.
"""

REVIEWER_SYSTEM_PROMPT = """
Eres el Agente Revisor Médico de MealfitRD. Tu ÚNICA misión es verificar que un plan alimenticio generado por la IA sea SEGURO para el paciente.

DEBES verificar estos puntos CRÍTICOS:

1. ALERGIAS: Revisa TODOS los ingredientes de TODAS las comidas. Si el paciente declaró alergia a un alimento (ej: "Lácteos", "Gluten", "Maní"), NINGÚN ingrediente debe contener ese alérgeno. Incluso derivados cuentan (ej: "queso" es lácteo, "pan" es gluten).

2. CONDICIONES MÉDICAS: 
   - Diabetes T2: No debe haber exceso de azúcares simples, harinas refinadas o miel
   - Hipertensión: Cuidado con salami, embutidos, exceso de sal
   - Enfermedades renales: Controlar exceso de proteína

3. DIETA DECLARADA:
   - Vegetariano: CERO carne, pollo, pescado, mariscos
   - Vegano: CERO productos animales (incluyendo huevos, lácteos, miel)
   - Sin gluten: CERO trigo, avena regular, cebada

4. RECHAZOS DEL PERFIL DE GUSTOS: Si el perfil dice que rechazó un ingrediente, NO debe aparecer.

Tu respuesta DEBE ser EXACTAMENTE en este formato JSON:
{
    "approved": true/false,
    "issues": ["Descripción del problema 1", "Descripción del problema 2"],
    "severity": "none" | "minor" | "critical"
}

Si approved es true, issues debe ser una lista vacía.
Si hay cualquier violación de alergias o condiciones médicas, severity DEBE ser "critical".
"""


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

    import random
    random_seed = random.randint(10000, 99999)
    
    # --- 🎲 INYECTOR DINÁMICO DE VARIEDAD CULINARIA ---
    cooking_techniques = [
        "Estilo Fusión Criolla", "Horneado Saludable", "Al Vapor con Finas Hierbas",
        "A la Plancha con Cítricos", "Guiso o Estofado Ligero", "Salteado tipo Wok",
        "Desmenuzado (Ropa Vieja)", "En Puré o Majado", "Estilo Ceviche o Fresco",
        "Asado a la Parrilla", "En Salsa a base de Vegetales Naturales", "En Airfryer Crujiente",
        "Croquetas o Tortitas al Horno", "Relleno (Ej. Canoas, Vegetales rellenos)"
    ]
    selected_techniques = random.sample(cooking_techniques, 3)
    
    technique_injection = (
        f"\n--- 👨🍳 INSTRUCCIÓN DINÁMICA DE VARIEDAD (OBLIGATORIA) ---\n"
        f"Para cumplir la regla de usar los MISMOS ingredientes del supermercado pero crear PLATOS DIFERENTES, "
        f"aplica obligatoriamente estas técnicas de cocción a las comidas principales:\n"
        f"• Día 1 (Opción A): Aplica técnica '{selected_techniques[0]}'\n"
        f"• Día 2 (Opción B): Aplica técnica '{selected_techniques[1]}'\n"
        f"• Día 3 (Opción C): Aplica técnica '{selected_techniques[2]}'\n"
        f"Ajusta los gramos matemáticamente para cumplir las macros.\n"
        f"----------------------------------------------------------\n"
    )
    
    prompt_text = (
        f"Analiza la siguiente información del usuario y genera un plan de comidas de 3 días.\n"
        f"IMPORTANTE: Genera opciones creativas y diferentes a planes anteriores. Semilla de generación aleatoria: {random_seed}\n\n"
        f"Información del Usuario:\n{json.dumps(form_data, indent=2)}\n"
        f"{nutrition_context}\n{taste_profile}\n{rag_context}\n{correction_context}\n{history_context}\n{technique_injection}\n\n"
        f"{GENERATOR_SYSTEM_PROMPT}"
    )
    
    generator_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=1.0 if attempt == 1 else 1.2,  # Máxima creatividad para evitar repeticiones
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
        google_api_key=os.environ.get("GEMINI_API_KEY")
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
    
    # 1. Pre-calcular nutrición
    nutrition = get_nutrition_targets(form_data)
    
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
            from db import get_recent_meals_from_plans
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
            
    # 2.5 Buscar Hechos y Diario Visual en Memoria Vectorial (RAG multimodal)
    user_facts_text = ""
    visual_facts_text = ""
    facts_data_sorted = []
    visual_list = []
    if user_id:
        try:
            from fact_extractor import get_embedding
            from db import search_user_facts, search_visual_diary, get_user_facts_by_metadata
            
            # 1. Recuperación estricta (Metadata JSONB) - ALERGIAS, CONDICIONES, RECHAZOS
            strict_facts_text = ""
            alergias = get_user_facts_by_metadata(user_id, 'categoria', 'alergia')
            if alergias:
                strict_facts_text += "🔴 ALERGIAS ESTRICTAS (PROHIBIDO USAR):\n" + "\n".join([f"  - {a['fact']}" for a in alergias]) + "\n"
                
            rechazos = get_user_facts_by_metadata(user_id, 'categoria', 'rechazo')
            if rechazos:
                strict_facts_text += "🔴 RECHAZOS (NO USAR):\n" + "\n".join([f"  - {r['fact']}" for r in rechazos]) + "\n"
                
            condiciones = get_user_facts_by_metadata(user_id, 'categoria', 'condicion_medica')
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
                            cat = meta.get("categoria", "")
                            return CATEGORY_PRIORITY_WEIGHTS.get(cat, 7)
                        return 7  # Sin categoría → al final
                    
                    facts_data_sorted = sorted(facts_data, key=get_fact_weight)
                    print(f"🧠 [RAG] Hechos textuales recuperados: {len(facts_data)} (ordenados por prioridad de categoría)")
                    
            # Buscar memoria visual 
            from vision_agent import get_multimodal_embedding
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
    MAX_CONTEXT_TOKENS = 1500000
    
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
        "form_data": form_data,
        "taste_profile": taste_profile,
        "nutrition": nutrition,
        "history_context": history_context,
        "user_facts": full_rag_context,
        "plan_result": None,
        "review_passed": False,
        "review_feedback": "",
        "attempt": 0,
    }
    
    # 4. Compilar y ejecutar el grafo
    graph = build_plan_graph()
    final_state = graph.invoke(initial_state)
    
    pipeline_duration = round(time.time() - pipeline_start, 2)
    
    print(f"\n{'🔗' * 30}")
    print(f"🔗 [LANGGRAPH] Pipeline completado en {pipeline_duration}s")
    print(f"🔗 Intentos: {final_state.get('attempt', 1)} | Aprobado: {final_state.get('review_passed', 'N/A')}")
    print("🔗" * 30 + "\n")
    
    return final_state["plan_result"]
