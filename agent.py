# backend/agent.py

import os
import time
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential

from nutrition_calculator import get_nutrition_targets
from db import get_user_profile, update_user_health_profile
from dotenv import load_dotenv

load_dotenv()

# Schema Definitions for Strict Structured Output (Sincronizado con React)
class MacrosModel(BaseModel):
    protein: str = Field(description="Gramos de proteína totales, ej: '150g'")
    carbs: str = Field(description="Gramos de carbohidratos totales, ej: '200g'")
    fats: str = Field(description="Gramos de grasas totales, ej: '60g'")

class MealModel(BaseModel):
    meal: str = Field(description="Momento del día, Ej: 'Desayuno', 'Almuerzo', 'Merienda', 'Cena'")
    time: str = Field(description="Hora sugerida, Ej: '8:00 AM'")
    name: str = Field(description="Nombre creativo y descriptivo del plato")
    desc: str = Field(description="Descripción apetitosa y profesional de la receta")
    prep_time: str = Field(description="Tiempo estimado de preparación, Ej: '15 min'")
    cals: int = Field(description="Calorías aproximadas de este plato")
    macros: List[str] = Field(description="Lista rápida de macros, Ej:['Alto en proteína', 'Bajo en carbohidratos']")
    ingredients: List[str] = Field(description="Lista de ingredientes con cantidades (texto simple), Ej:['1 plátano verde maduro', '2 huevos', '1/2 aguacate']")
    recipe: List[str] = Field(description="Pasos de preparación. DEBES usar los prefijos: 'Mise en place: ...', 'El Toque de Fuego: ...' y 'Montaje: ...'")

class DailyPlanModel(BaseModel):
    day: int = Field(description="Número de día (1, 2, o 3)")
    meals: List[MealModel] = Field(description="Lista de comidas en orden cronológico. MUY IMPORTANTE: Si el usuario omite el almuerzo, genera SOLO 3 comidas: Desayuno, Merienda, Cena.")

class PlanModel(BaseModel):
    main_goal: str = Field(description="El objetivo principal identificado. Ej: 'Pérdida de Peso (Déficit)'")
    calories: int = Field(description="Total de calorías estrictas planificadas sumando todas las comidas")
    macros: MacrosModel = Field(description="Distribución matemática de macronutrientes para el día")
    insights: List[str] = Field(description="Lista EXACTA de 3 frases: 1. Inicia con 'Diagnóstico: ', 2. Inicia con 'Estrategia: ', 3. Inicia con 'Tip del Chef: '")
    days: List[DailyPlanModel] = Field(description="Lista de 3 días con planes de comida variados")

# Langchain Chat Model Initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=0.2,
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

ANALYZE_SYSTEM_PROMPT = """
Eres un Nutricionista Clínico, Chef Profesional y la IA oficial de MealfitRD.
Tu misión es crear un plan alimenticio de EXACTAMENTE 3 DÍAS VARIADOS, altamente profesional y 100% adaptado a la biometría y preferencias del usuario.

REGLAS ESTRICTAS:
1. CALORÍAS Y MACROS PRE-CALCULADOS (REGLA INQUEBRANTABLE): Los cálculos de calorías objetivo y macronutrientes (proteína, carbohidratos, grasas) ya fueron realizados por el Sistema Calculador. Es OBLIGATORIO Y NO NEGOCIABLE que el menú que diseñes cumpla EXACTAMENTE con estos gramos de macros y calorías diarias. La inmensa variedad culinaria JAMÁS debe comprometer el cumplimiento estricto de las matemáticas nutricionales (macros). La suma de las calorías de cada comida en un día DEBE coincidir con el total provisto diario.
2. ESTRUCTURA DE 3 DÍAS (OPCIONES): Debes generar exactamente 3 opciones (`day: 1` para Opción A, `day: 2` para Opción B, `day: 3` para Opción C). Cada opción debe tener un ENFOQUE DE PROTEÍNAS DIFERENTE (Ej. Opción A: Pollo, Opción B: Pescado/Atún, Opción C: Huevos/Res) para evitar la fatiga alimenticia.
3. INGREDIENTES DOMINICANOS: El menú DEBE usar alimentos típicos, accesibles y económicos de República Dominicana (Ej: Plátano, Yuca, Batata, Huevos, Salami, Queso de freír/hoja, Pollo guisado, Aguacate, Habichuelas, Arroz, Avena).
4. RECETAS PROFESIONALES: Los pasos de las recetas (`recipe`) DEBEN incluir obligatoriamente estos prefijos para la UI:
   - "Mise en place: [Instrucciones de preparación previa y cortes]"
   - "El Toque de Fuego: [Instrucciones de cocción en sartén, horno o airfryer]"
   - "Montaje: [Instrucciones de cómo servir para que luzca apetitoso]"
5. CUMPLE RESTRICCIONES ABSOLUTAMENTE: Si el usuario es vegetariano, tiene alergias (Ej. Lácteos), condiciones médicas (Ej. Diabetes T2) o indicó obstáculos (Ej: falta de tiempo, no sabe cocinar), el plan DEBE reflejar soluciones inmediatas a eso (comidas rápidas, sin azúcar, sin carne, etc).
6. ESTRUCTURA: Si el usuario indicó `skipLunch: true`, NO incluyas Almuerzo, distribuye las calorías en las demás comidas y asume que comerá la comida familiar.
7. VARIEDAD EXTREMA Y CERO MONOTONÍA (DENTRO DE LOS MACROS): Revisa cuidadosamente el historial de comidas anteriores provisto en el prompt. ESTÁ ESTRICTAMENTE PROHIBIDO repetir platos, nombres o incluso los mismos ingredientes principales de los planes recientes. Tienes permiso para ser muy creativo, usar distintas fuentes de carbohidratos, diferentes cortes de carne, métodos de cocción variados (horno, guisos, plancha, asado) y combinaciones nuevas... PERO SIEMPRE respetando matemáticamente los macros asignados en la Regla 1. ¡Sorprende al usuario con opciones radicalmente diferentes que encajen perfecto en sus números!
8. PROHIBICIÓN ABSOLUTA DE RECHAZOS: Lee detenidamente el Perfil de Gustos adjunto. Si el perfil dice que el usuario odia o rechazó un ingrediente, está TOTALMENTE PROHIBIDO incluirlo en este plan. EXCEPCIÓN CRÍTICA: Si el usuario te pidió EXPLÍCITAMENTE ahora mismo (en sus instrucciones o en el chat reciente) que le incluyas un ingrediente específico (ej. "quiero avena"), DEBES priorizar esa petición reciente e ignorar por completo la prohibición histórica sobre ese ingrediente.
9. ALIMENTOS SALUDABLES OBLIGATORIOS Y PROHIBICIÓN DE COMIDA CHATARRA: ESTÁ ESTRICTAMENTE PROHIBIDO generar platos que sean comida rápida o chatarra (como Pizza regular, Hamburguesas comerciales, Hot Dogs, etc). MealfitRD es un sistema de nutrición saludable. Todas las recetas deben enfocarse en alimentos ricos en nutrientes. Si propones una opción estilo 'cheat meal', DEBE especificar claramente que es una versión saludable casera, baja en grasa y rica en nutrientes.
"""

def analyze_preferences_agent(likes: list, history: list, active_rejections: Optional[list] = None):
    """
    Agente #1: Especialista en Preferencias y Perfiles de Gusto.
    Analiza el historial de Me Gusta y Rechazos ACTIVOS (últimos 7 días) 
    y devuelve un perfil conciso de los gustos del usuario.
    
    Los rechazos expiran después de 7 días, permitiendo que la IA vuelva a sugerir esos alimentos.
    """
    print("\n-------------------------------------------------------------")
    print("🧠 [AGENTE DE PREFERENCIAS] Analizando Perfil de Gustos...")
    
    # 1. Mapear likes a un formato legible
    liked_meals = [f"{like.get('meal_name')} ({like.get('meal_type')})" for like in likes] if likes else []
    
    # 2. Usar rechazos activos (últimos 7 días) en lugar de permanentes
    rejected_meals = active_rejections if active_rejections else []
    
    if rejected_meals:
        print(f"🚫 Rechazos activos (últimos 7 días): {rejected_meals}")
    else:
        print("➡️  No hay rechazos activos en los últimos 7 días.")
                
    if not liked_meals and not rejected_meals:
        print("➡️  No hay datos suficientes para un perfil. Asumiendo gustos estándar.")
        print("-------------------------------------------------------------")
        return ""
        
    prompt = f"""
    Eres el Analista Psicológico de Gustos de MealfitRD. Tu trabajo es leer los "Me Gusta" y los "Rechazos TEMPORALES activos" de un paciente para extraer un perfil psicológico.
    
    IMPORTANTE: Los rechazos listados abajo son TEMPORALES (activos por 7 días). Después de ese período, estos alimentos podrán volver a sugerirse.
    
    Es CRÍTICO que extraigas los ingredientes base de las comidas rechazadas para prohibirlos TEMPORALMENTE. Por ejemplo, si el usuario rechazó "Mangú de Poder", debes deducir y ordenar explícitamente la prohibición temporal de "plátano verde" y "mangú".
    
    Comidas a las que el usuario le dio ME GUSTA (Sus favoritas):
    {json.dumps(liked_meals)}
    
    Comidas que el usuario RECHAZÓ RECIENTEMENTE (Exclusiones temporales activas):
    {json.dumps(rejected_meals)}
    
    Redacta el perfil de gustos AHORA. El formato DEBE ser directo y dictatorial para la IA que creará el plan: 
    "PERFIL: Al usuario le encanta [X].
    PROHIBICIONES TEMPORALES ACTIVAS: Está prohibido servirle [ingrediente principal del rechazo 1], [ingrediente principal del rechazo 2] porque los rechazó recientemente. Cero tolerancia con estos ingredientes en este plan."
    """
    
    pref_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.3, # Baja temperatura para ser analítico
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    start_time = time.time()
    response = pref_llm.invoke(prompt)
    taste_profile = response.content
    
    end_time = time.time()
    print(f"✅ [PERFIL LISTO] Resuelto en {round(end_time - start_time, 2)}s: {taste_profile}")
    print("-------------------------------------------------------------\n")
    
    return f"\n\n--- PERFIL DE GUSTOS DEL USUARIO (OBLIGATORIO RESPETAR) ---\n{taste_profile}\n-----------------------------------------------------------"

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
        skip_lunch_instruction = "\n\n⚠️ INSTRUCCIÓN CRÍTICA Y OBLIGATORIA DE ESTRUCTURA ⚠️\nEl usuario indicó 'Almuerzo Familiar / Ya resuelto' (`skipLunch: true`). ESTÁ TOTAL Y ABSOLUTAMENTE PROHIBIDO incluir la comida 'Almuerzo' en este plan. Si incluyes un Almuerzo, el plan será rechazado. \nDebes generar EXACTAMENTE 3 comidas por el día entero: 'Desayuno', 'Merienda' y 'Cena'. \nLas calorías objetivo proporcionadas ya tienen descontada la porción del almuerzo resuelto. Dives dividir todas esas calorías ÚNICAMENTE entre Desayuno, Merienda y Cena."

    prompt_text = f"Analiza la siguiente información del usuario y genera un plan de comidas de 3 días ajustado a sus necesidades diarias.\n\nInformación del Usuario:\n{json.dumps(form_data, indent=2)}\n{nutrition_context}\n{taste_profile}\n{recent_meals_context}\n{skip_lunch_instruction}\n\n{ANALYZE_SYSTEM_PROMPT}"
    
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

    print("\n-------------------------------------------------------------")
    print("⏳ [AGENTE DE SUSTITUCIÓN INTERPRETATIVO] Analizando rechazo...")
    print(f"➡️  Interpretando por qué rechazó: \"{rejected_meal}\" ({meal_type})")
    
    start_time = time.time()
    
    prompt_text = f"""
    Eres el Chef Analítico e Inteligencia Artificial de Intervención Rápida de MealfitRD.
    El usuario acaba de darle click a "Cambiar / No me gusta" para la siguiente comida: "{rejected_meal}" (Momento del día: {meal_type}).
    
    TAREA DEL AGENTE (INTERPRETACIÓN EN TIEMPO REAL):
    1. Interpreta silenciosamente POR QUÉ pudo haberlo rechazado. ¿Era muy pesado? ¿Ingredientes muy secos? ¿Quizás no le gustan esos ingredientes principales?
    2. Como respuesta a esa interpretación, diseña una alternativa RADICALMENTE OPUESTA en perfil de sabor y textura a la que acaba de rechazar, pero que mantenga las calorías cercanas a {target_calories} kcal.
    3. Asegura que la comida siga una dieta tipo '{diet_type}' y utilice gastronomía/ingredientes locales dominicanos.{context_extras}
    4. ⚠️ CRÍTICO: Bajo ninguna circunstancia puedes sugerir un plato que esté en la lista de exclusión o que tenga los mismos ingredientes principales de los platos rechazados.
    5. Devuelve estrictamente el esquema de comida solicitado, en español.
    6. Asegúrate de incluir los prefijos en la receta (Mise en place:, El Toque de Fuego:, Montaje:).
    """
    
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

# ============================================================
# TOOL: Actualizar Health Profile del usuario
# ============================================================

@tool
def update_form_field(user_id: str, field: str, new_value: str) -> str:
    """
    Actualiza el formulario en tiempo real en la UI del usuario.
    Campos válidos (field): 'weight', 'height', 'age', 'gender', 'weightUnit', 'dietType', 'mainGoal', 'allergies', 'medicalConditions', 'activityLevel', 'dislikes', 'struggles'.
    """
    print(f"🔧 [TOOL EXECUTION] Actualizando form del usuario {user_id}: {field} -> {new_value}")
    
    if field in ['weight', 'height', 'age']:
        import re
        extracted = re.sub(r'[^\d.]', '', str(new_value))
        if extracted:
            new_value = extracted
            
    if user_id and user_id != "guest":
        profile = get_user_profile(user_id)
        if profile:
            health_profile = profile.get("health_profile") or {}
            if field in ['allergies', 'medicalConditions', 'dislikes', 'struggles']:
                health_profile[field] = [item.strip() for item in new_value.split(",") if item.strip()]
            else:
                health_profile[field] = new_value
            update_user_health_profile(user_id, health_profile)
            
            # --- NUEVA LÓGICA DE LIMPIEZA DE VECTORES ---
            from db import delete_user_facts_by_metadata
            category_map = {
                'allergies': 'alergia',
                'medicalConditions': 'condicion_medica',
                'dislikes': 'rechazo',
                'dietType': 'dieta',
                'mainGoal': 'objetivo'
            }
            if field in category_map:
                cat = category_map[field]
                print(f"🧹 [CLEANUP] Borrando vectores de categoría '{cat}' para evitar conflictos con el formulario.")
                delete_user_facts_by_metadata(user_id, {"categoria": cat})
            # ---------------------------------------------
            
            
    return f"¡Éxito! El campo '{field}' ha sido actualizado a '{new_value}'."

# ============================================================
# TOOL: Generar nuevo plan desde el Chat
# ============================================================

def execute_generate_new_plan(user_id: str, form_data: dict, instructions: str = "") -> str:
    print(f"\n🚀 [TOOL] Generando plan nuevo desde el chat para user_id: {user_id}")
    if instructions:
        print(f"📝 [TOOL] Instrucciones específicas del usuario: {instructions}")
    
    from graph_orchestrator import run_plan_pipeline
    from db import get_user_likes, get_active_rejections, get_user_profile
    
    # 1. Validar y consolidar form_data (Priorizar el del frontend)
    actual_form_data = form_data or {}
    
    if not actual_form_data.get("age"):
        profile = get_user_profile(user_id)
        if profile:
            actual_form_data = profile.get("health_profile") or {}
            
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
        custom_memory_context = f"\n\n--- INSTRUCCIÓN ESPECÍFICA DEL USUARIO PARA ESTE PLAN ---\nEL USUARIO PIDIÓ ESTO EXPLÍCITAMENTE, CUMPLE SU PETICIÓN A TODA COSTA:\n'{instructions}'\n------------------------------------------------------------\n"

    # 3. Ejecutar el pipeline multi-agente
    try:
        result = run_plan_pipeline(actual_form_data, [], taste_profile, memory_context=custom_memory_context)
        if result:
            # Intentar guardar en la tabla meal_plans (no crítico)
            try:
                from supabase import create_client
                import os
                from datetime import datetime
                supabase_url = os.environ.get("SUPABASE_URL")
                
                # Intentamos usar la KEY de servicio si existe para saltar el RLS, o la anónima
                supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
                
                if supabase_url and supabase_key:
                    sb = create_client(supabase_url, supabase_key)
                    calories = result.get("calories", 0)
                    macros = result.get("macros", {})
                    sb.table("meal_plans").insert({
                        "user_id": user_id,
                        "plan_data": result,
                        "name": f"Plan del {datetime.now().strftime('%A %d de %B %Y')}",
                        "calories": int(calories) if calories else 0,
                        "macros": macros,
                    }).execute()
                    print("💾 Plan generado desde chat guardado en DB.")
            except Exception as db_e:
                print(f"⚠️ Aviso: No se pudo guardar el plan en Supabase (error {db_e}), pero el plan se devolverá al usuario.")
            
            return json.dumps(result)
        else:
            return "ERROR: El pipeline no pudo generar un plan. Intenta de nuevo."
    except Exception as e:
        print(f"❌ Error generando plan desde chat: {e}")
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
def log_consumed_meal(user_id: str, meal_name: str, calories: int, protein: int, carbs: int = 0, healthy_fats: int = 0) -> str:
    """
    Registra una comida que el usuario afirma haber consumido realmente en su diario de consumo ("fuera del plan").
    Úsala SOLO cuando el usuario confirme que se ha comido lo que le analizaste o subió en la foto, o cuando explícitamente diga que comió algo.
    Incluye carbohidratos y grasas saludables si están disponibles.
    """
    from db import log_consumed_meal as db_log_consumed_meal
    print(f"🔧 [TOOL EXECUTION] Registrando comida consumida para user {user_id}: {meal_name} ({calories} kcal, {protein}g proteina, {carbs}g carbos, {healthy_fats}g grasas)")
    result = db_log_consumed_meal(user_id, meal_name, calories, protein, carbs, healthy_fats)
    if result is not None:
        return f"¡Éxito! Se ha registrado el consumo de '{meal_name}' ({calories} kcal, {protein}g proteína, {carbs}g carbohidratos, {healthy_fats}g grasas saludables) en tu diario."
    else:
        return "Hubo un error al intentar registrar la comida consumida. Por favor, intenta de nuevo."

# ============================================================
# TOOL: Modificar una comida individual del plan activo
# ============================================================

def execute_modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str) -> str:
    """Ejecuta la modificación de una comida individual en el plan activo del usuario."""
    from db import get_latest_meal_plan_with_id, update_meal_plan_data
    
    print(f"\n🔧 [TOOL] modify_single_meal: Día {day_number}, {meal_type}, cambios: '{changes}'")
    
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
    
    modify_prompt = f"""Eres el Chef Profesional de MealfitRD. El usuario quiere MODIFICAR una comida específica de su plan.

COMIDA ORIGINAL:
- Nombre: {target_meal.get('name')}
- Descripción: {target_meal.get('desc')}
- Momento: {target_meal.get('meal')} ({target_meal.get('time')})
- Calorías: {original_cals}
- Ingredientes: {json.dumps(target_meal.get('ingredients', []))}

CAMBIO SOLICITADO POR EL USUARIO:
"{changes}"

INSTRUCCIONES:
1. Aplica EXACTAMENTE el cambio que pide el usuario (ej: si dice "cámbiale el salami por huevos", sustituye el salami por huevos en ingredientes y receta). EXCEPCIÓN CRÍTICA: Si el usuario pide explícitamente un ingrediente, DEBES incluirlo priorizando su deseo reciente, incluso si el algoritmo creía que no le gustaba históricamente.
2. Mantén las calorías lo más cercanas posible a {original_cals} kcal
3. Conserva el momento del día ({target_meal.get('meal')}) y la hora ({target_meal.get('time')})
4. Usa ingredientes dominicanos
5. Los pasos de la receta DEBEN usar los prefijos: 'Mise en place: ...', 'El Toque de Fuego: ...' y 'Montaje: ...'
6. Dale un nombre nuevo y creativo al plato modificado
"""
    
    modify_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        temperature=0.5,
        google_api_key=os.environ.get("GEMINI_API_KEY")
    ).with_structured_output(MealModel)
    
    try:
        new_meal_response = modify_llm.invoke(modify_prompt)
        
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
        
        target_day["meals"][target_meal_index] = new_meal_data
        
        # 5. Actualizar en Supabase
        update_result = update_meal_plan_data(plan_id, plan_data)
        if update_result is not None:
            print(f"✅ [TOOL] Comida modificada exitosamente: '{new_meal_data.get('name')}'")
            return json.dumps({"modified_meal": new_meal_data, "day": day_number, "meal_index": target_meal_index})
        else:
            return "ERROR: Se generó la nueva comida pero no se pudo guardar en la base de datos."
    except Exception as e:
        print(f"❌ [TOOL] Error modificando comida: {e}")
        return f"ERROR: {str(e)}"

@tool
def modify_single_meal(user_id: str, day_number: int, meal_type: str, changes: str) -> str:
    """
    Modifica UNA comida específica del plan activo del usuario. No genera un plan nuevo, solo cambia la comida indicada.
    Usa esta herramienta cuando el usuario pida un cambio puntual a una comida de su plan,
    como 'cámbiale el salami al mangú por huevos en la Opción A', 'quítale el arroz al almuerzo de la Opción B', 'ponle más proteína al desayuno'.
    IMPORTANTE: Opción A = day_number 1, Opción B = day_number 2, Opción C = day_number 3.
    
    Parámetros:
    - user_id: ID del usuario
    - day_number: número de la opción (1 para Opción A, 2 para Opción B, 3 para Opción C)
    - meal_type: momento exacto del día: 'Desayuno', 'Almuerzo', 'Merienda' o 'Cena'
    - changes: descripción en lenguaje natural del cambio solicitado por el usuario
    """
    return "DUMMY_CALL_ACTUALLY_INTERCEPTED"

# ============================================================
# TOOL: Añadir items a la lista de compras
# ============================================================

@tool
def add_to_shopping_list(user_id: str, items: str) -> str:
    """
    Añade uno o más ingredientes/items a la lista de compras personal del usuario.
    Usa esta herramienta cuando el usuario diga que se quedó sin algo, necesita comprar algo,
    o pida añadir items a su lista de compras.
    Ejemplos: 'Me quedé sin plátanos', 'Añade leche y huevos a mi lista', 'Necesito comprar arroz'.
    
    - items: string con items separados por coma, ej: 'plátanos, huevos, leche'
    """
    from db import add_custom_shopping_items
    
    print(f"🔧 [TOOL EXECUTION] Añadiendo items a shopping list del usuario {user_id}: {items}")
    
    items_list = [item.strip() for item in items.split(",") if item.strip()]
    
    if not items_list:
        return "No se proporcionaron items válidos para añadir."
    
    result = add_custom_shopping_items(user_id, items_list)
    if result is not None:
        items_formatted = ", ".join(items_list)
        return f"¡Éxito! Se añadieron {len(items_list)} item(s) a tu lista de compras: {items_formatted}."
    else:
        return "Hubo un error al añadir los items a la lista de compras. Intenta de nuevo."

# ============================================================
# TOOL: Buscar en Memoria Profunda (Cold Storage)
# ============================================================

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
    from db import search_deep_memory as db_search_deep_memory
    
    print(f"🔍 [TOOL EXECUTION] Buscando en memoria profunda para user {user_id}: '{query}'")
    
    results = db_search_deep_memory(user_id, query, limit=5)
    
    if not results:
        return "No se encontraron recuerdos históricos que coincidan con esa búsqueda. Es posible que aún no haya suficiente historial archivado."
    
    # Formatear los resultados para el agente
    formatted = []
    for idx, r in enumerate(results, 1):
        period = f"{r.get('messages_start', '?')} → {r.get('messages_end', '?')}"
        summary = r.get('summary', 'Sin contenido')
        formatted.append(f"📁 Recuerdo #{idx} (Período: {period}):\n{summary}")
    
    return "\n\n".join(formatted)

# Lista de tools disponibles para el agente
agent_tools = [update_form_field, generate_new_plan_from_chat, log_consumed_meal, modify_single_meal, add_to_shopping_list, search_deep_memory]

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
        prompt = f"Genera un título muy corto y atractivo (máximo 4 palabras) para este inicio de chat sobre nutrición/comida.\nMensaje del usuario: {first_message}\n\nSolo devuelve el título, sin comillas ni formato adicional."
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
        
        rewrite_prompt = f"""Eres un optimizador de búsqueda vectorial para una app de nutrición.
Dado el mensaje del usuario, genera UNA SOLA frase de búsqueda optimizada para encontrar hechos relevantes en una base de datos vectorial de salud/nutrición.

REGLAS:
- Si el mensaje menciona alimentos, dieta, salud, alergias, ejercicio, peso, objetivos → genera una query precisa.
- Si el mensaje es una pregunta sobre su plan de comidas → genera una query sobre preferencias alimenticias.
- Si el mensaje NO tiene nada que ver con nutrición/salud (ej: chit-chat, preguntas generales) → responde exactamente: SKIP
- La query debe ser en español, concisa (máx 15 palabras), sin explicaciones.

Mensaje del usuario: "{prompt}"

Query optimizada:"""
        
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
    
    print(f"⏳ [CHAT STREAM] LangGraph iniciando stream via thread para {session_id}...")
    
    final_state_snapshot = None
    
    import asyncio
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    def run_sync_stream():
        try:
            for event in chat_graph_app.stream(inputs, config=config, stream_mode="messages"):
                # Capturamos eventos de streaming de texto del modelo
                asyncio.run_coroutine_threadsafe(queue.put(event), loop)
            # Mandar señal de cierre
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            
    # Ejecutar en thread para no bloquear event loop
    task = asyncio.create_task(asyncio.to_thread(run_sync_stream))
    
    try:
        while True:
            event = await queue.get()
            
            if event is None:
                break
                
            if isinstance(event, Exception):
                raise event
                
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
        print(f"❌ [CHAT STREAM] Error en stream sync mode: {e}")
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return
    finally:
        await task
        
    # Obtener el estado final actualizado
    try:
        final_state_snapshot = await asyncio.to_thread(chat_graph_app.get_state, config)
    except Exception as e:
        print(f"⚠️ Error obteniendo get_state tras stream: {e}")

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