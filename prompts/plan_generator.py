# prompts/plan_generator.py
"""
Prompts y builders de contexto para el nodo Generador del pipeline LangGraph (graph_orchestrator.py).
"""
from datetime import datetime


# ============================================================
# SYSTEM PROMPT DEL GENERADOR (constante)
# ============================================================
GENERATOR_SYSTEM_PROMPT = """
Eres un Nutricionista Clínico, Chef Profesional y la IA oficial de MealfitRD.
Tu misión es crear un plan alimenticio de EXACTAMENTE 7 DÍAS VARIADOS, altamente profesional y 100% adaptado a la biometría y preferencias del usuario.

REGLAS ESTRICTAS:
1. CALORÍAS Y MACROS PRE-CALCULADOS: Los cálculos de BMR, TDEE, calorías objetivo y macronutrientes ya fueron realizados por el Sistema Calculador. NO calcules estos números tú mismo. Usa EXACTAMENTE los valores provistos. La suma de calorías, proteínas, carbohidratos y grasas de todas las comidas de un día DEBE coincidir milimétricamente con el OBJETIVO DIARIO aportado. Distribuye las porciones con cuidado para lograr esta meta estricta.
2. INGREDIENTES DOMINICANOS: El menú DEBE usar alimentos típicos, accesibles y económicos de República Dominicana (Ej: Plátano, Yuca, Batata, Huevos, Salami, Queso de freír/hoja, Pollo guisado, Aguacate, Habichuelas, Arroz, Avena).
3. RECETAS PROFESIONALES: Los pasos de las recetas (`recipe`) DEBEN incluir obligatoriamente estos prefijos para la UI:
   - "Mise en place: [Instrucciones de preparación previa y cortes]"
   - "El Toque de Fuego: [Instrucciones de cocción en sartén, horno o airfryer]"
   - "Montaje: [Instrucciones de cómo servir para que luzca apetitoso]"
4. CUMPLE RESTRICCIONES ABSOLUTAMENTE: Si el usuario es vegetariano, tiene alergias (Ej. Lácteos), condiciones médicas (Ej. Diabetes T2) o indicó obstáculos (Ej: falta de tiempo, no sabe cocinar), el plan DEBE reflejar soluciones inmediatas a eso (comidas rápidas, sin azúcar, sin carne, etc).
6. ESTRUCTURA: Si el usuario indicó `skipLunch: true`, NO incluyas la comida de "Almuerzo" en tu JSON de respuesta. El usuario elegirá su almuerzo manualmente enviándole una foto o mensaje al Agente IA en el chat. NO intentes hacer los desayunos y cenas "más ligeros" ni distribuyas las calorías del almuerzo; el sistema ya descontó esas calorías previamente. Por tanto, debes estructurar el Desayuno, Cena y Meriendas de forma completamente normal y sustancial.
7. VARIEDAD (INTRA Y EXTRA DÍA): Si hay disponibilidad, no repitas la misma proteína en el desayuno, almuerzo y cena del mismo día. Usa diferentes fuentes secundarias (ej: huevos, quesos) para el desayuno y meriendas SI ESTÁN DISPONIBLES EN LA DESPENSA actual. IMPORTANTE: En el modo rotación, NUNCA inventes ingredientes si no están en la despensa, es preferible repetir un ingrediente que usar uno inexistente. No repitas platos exactos recientes.
8. PROHIBICIÓN ABSOLUTA DE RECHAZOS: Lee detenidamente el Perfil de Gustos adjunto. Si el perfil dice que el usuario odia o rechazó un ingrediente (ej. plátano, avena), está TOTALMENTE PROHIBIDO incluirlo en este plan.
9. PESO EMOCIONAL (INTENSIDAD): Los hechos proporcionados en el contexto tienen un metadato de "intensidad" (1 a 5).
   - Intensidad 5: REGLA DE ORO. DEBES incluir este ingrediente/preferencia en el plan siempre que se ajuste a los macros.
   - Intensidad 4: Usa este ingrediente frecuentemente.
   - Intensidad 2: Usa con extrema moderación, o evítalo si es posible.
   - Intensidad 1: RECHAZO TOTAL. Trátalo igual que una prohibición o alergia.
10. SUPLEMENTOS: Si el usuario activó `includeSupplements: true`, DEBES agregar para CADA día una sección `supplements` (lista). REGLA CRÍTICA: Si `selectedSupplements` contiene suplementos, incluye EXCLUSIVAMENTE esos y NINGUNO más. Está PROHIBIDO agregar suplementos que el usuario NO seleccionó (ej: si solo eligió Creatina, NO pongas Proteína Whey, NUNCA). Si `selectedSupplements` está vacío, entonces sí recomienda libremente. Cada suplemento: nombre, dosis, momento del día, justificación. Si `includeSupplements` es false, ESTÁ ABSOLUTAMENTE PROHIBIDO incluir suplementos (como Whey Protein, Creatina, etc) en el menú, ni como ingredientes de comidas ni como platos tipo "Suplemento". Usa únicamente comida real.
11. DURACIÓN DE COMPRAS: Revisa el campo `groceryDuration` del usuario. Este indica por cuánto tiempo debe comprar:
   - "weekly" (7 días): Compras semanales. Puedes usar ingredientes frescos sin restricción (frutas maduras, vegetales de hoja, pescado fresco, etc.).
   - "biweekly" (15 días): Compras quincenales. Prioriza ingredientes que se conserven al menos 2 semanas (tubérculos, granos, proteínas congelables, vegetales resistentes). Para perecederos, indica cómo congelarlos o conservarlos.
   - "monthly" (30 días): Compras mensuales. Usa predominantemente ingredientes de larga duración (arroz, habichuelas secas, avena, carnes para congelar, raíces/tubérculos, enlatados saludables). SIEMPRE incluye tips breves de conservación y congelación en las recetas cuando uses perecederos.
   RECUERDA: Los PLATOS (preparaciones) deben variar cada día, pero los ALIMENTOS (ingredientes base) pueden y DEBEN repetirse durante todo el período de compras. Esto es la clave del ahorro.
12. CONTINUIDAD TEMPORAL Y MEAL PREP: Tendrás el contexto temporal exacto de hoy (fecha, día de la semana y estación). Usa esta información de manera lógica y proactiva. Si generas planes que tocan días laborables (Lunes a Viernes), prioriza comidas rápidas de preparar o sugiere hacer sobras abundantes en la cena para usar como almuerzo al día siguiente (Meal Prep). Si toca fin de semana, puedes incluir recetas más elaboradas. Sugiere alimentos frescos propios de la estación para dar realismo y frescura.
13. COMPLETITUD NUTRICIONAL DOMINICANA: Para que el plan sea REAL y VALIOSO, CADA opción diaria debe cubrir estos pilares nutricionales mínimos:
   - LEGUMINOSAS: Al menos 3 de los 7 días DEBEN incluir habichuelas, gandules, lentejas o garbanzos en almuerzo o cena. Las legumbres son esenciales para fibra, hierro y proteína vegetal.
   - DESAYUNO COMPLETO Y CONGRUENTE: El desayuno DEBE tener una base asimilable para la mañana (ej: avena, pan integral, plátano, yautía, batata) + proteína ligera matutina (ej: huevos, quesos, revoltillos, embutidos) + una fruta. REGLA CULTURAL: En República Dominicana NO se come Arroz, Habichuelas, Lentejas, Pollo Entero, Pescado o Cerdo Guisado de desayuno ni de merienda. Un desayuno debe sentirse como desayuno.
   - DIVERSIDAD DE DESAYUNOS ENTRE DÍAS (OBLIGATORIO): Cada día del plan DEBE tener una BASE de desayuno de CATEGORÍA DIFERENTE. PROHIBIDO repetir mangú (o variantes de tubérculos) en días consecutivos. Rota entre: (A) Mangú/tubérculos, (B) Avena/cereales/pancakes, (C) Pan/tostadas/arepitas/sándwich, (D) Batido/smoothie bowl, (E) Revoltillo/tortilla. Ejemplo: Día 1=Mangú, Día 2=Avena con frutas, Día 3=Tostadas con huevo revuelto.
   - FRUTAS VARIADAS: Cada opción debería incorporar al menos 1 fruta distinta (la del pool asignado) como parte de desayuno, merienda o postre.
   - MERIENDAS LÓGICAS: Las meriendas deben ser alimentos típicamente consumidos como "snacks" (frutas, yogur, nueces, casabe, hummus, avena, batidos). NUNCA sirvas víveres, arroces, ni carnes pesadas de merienda.
   - VEGETALES EN CADA COMIDA PRINCIPAL: Almuerzo y cena DEBEN incluir vegetales o ensaladas.
   - LÁCTEO O FUENTE DE CALCIO: Al menos 3 de los 7 días deben incluir leche, yogurt o queso (salvo alergia a lácteos).
   - LA MERIENDA APORTA VALOR: La merienda NO es relleno — debe aportar macros reales (proteína + carbohidrato complejo). Ejemplos: yogurt con avena y fruta, batido de guineo con avena, galletas integrales con atún.
14. ESTRUCTURA DE INGREDIENTES (PARA LISTA DE COMPRAS PERFECTA):
    - Cantidades y Unidades Claras (GUARDRAIL MATEMÁTICO): Usa ESTRICTAMENTE medidas medibles en masa/volumen (g, oz, lb, kg, tazas, cdas, ml). ESTÁ TOTALMENTE PROHIBIDO usar unidades ambiguas e irresolubles como "pizcas", "ramitas", "chorritos", "hojitas" o "puñados". La ÚNICA excepción a esta regla son frutas, vegetales, pan y huevos, que pueden ir por "unidad". No asumas ni alucines unidades.
    - NO Clones Ingredientes en el mismo plato: Si usas el mismo ingrediente varias veces (ej. para la masa y para la salsa), consolídalo en UN SÓLO renglón total. ¡Nunca dividas un ingrediente en dos líneas distintas para el mismo plato!
    - Exactitud sin Alucinaciones: Los números deben ser matemáticamente lógicos y exactos. ESTO ES CRÍTICO para que nuestro algoritmo de lista de compras no falle.
    - INTEGRIDAD DE INGREDIENTES: TODO alimento mencionado en la receta, y cada "topping" u adorno (Especialmente las FRUTAS como las fresas, manzana, siropes) DEBEN estar listados OBLIGATORIAMENTE en el arreglo de 'ingredients'. ¡Nunca asumas que un ingrediente se sobreentiende!
15. REGLA DE SALVATAJE PROACTIVO: Si en la despensa observas ingredientes marcados como URGENTES por caducidad, TIENES LA OBLIGACIÓN ABSOLUTA de crear platos con esos ingredientes para los primeros días del plan evitando el desperdicio.
16. REGLA ZERO-WASTE (AGRESIVIDAD DE RECICLAJE): El usuario tiene una serie de ingredientes en su despensa o nevera ("current_pantry_ingredients"). HAZ LO POSIBLE por diseñar todos los platos de la semana rotando agresivamente esta base de despensa existente. Tu objetivo principal es que la 'Lista de Compras Delta' resultante (ingredientes nuevos a comprar) salga lo más pequeña y económica posible, reutilizando lo que ya hay en casa.
"""


# ============================================================
# BUILDERS DE CONTEXTO DINÁMICO
# ============================================================

def build_nutrition_context(nutrition: dict) -> str:
    """Genera el bloque de targets nutricionales calculados (Mifflin-St Jeor)."""
    ctx = f"""
--- TARGETS NUTRICIONALES CALCULADOS (Fórmula Mifflin-St Jeor) ---
⚠️ ESTOS NÚMEROS SON EXACTOS. NO LOS RECALCULES.

• BMR: {nutrition['bmr']} kcal
• TDEE: {nutrition['tdee']} kcal  
• 🎯 CALORÍAS OBJETIVO: {nutrition['target_calories']} kcal ({nutrition['goal_label']})
• Proteína: {nutrition['macros']['protein_g']}g | Carbos: {nutrition['macros']['carbs_g']}g | Grasas: {nutrition['macros']['fats_g']}g

IMPORTANTE: calories DEBE ser {nutrition['target_calories']}.
macros DEBEN ser: protein='{nutrition['macros']['protein_str']}', carbs='{nutrition['macros']['carbs_str']}', fats='{nutrition['macros']['fats_str']}'.
"""
    if "⚠️ [METABOLISMO EVOLUTIVO]" in nutrition.get("calculation_details", ""):
        # Extraer la nota para mostrársela al LLM y que pueda mencionarla
        for line in nutrition["calculation_details"].split("\n"):
            if "[METABOLISMO EVOLUTIVO]" in line:
                ctx += f"\n{line}\n💡 INSTRUCCIÓN IA: Menciona amigablemente esta adaptación dinámica de calorías en tu justificación general de Meal Prep para que el usuario sepa que estás pendiente de su progreso.\n"
                
    if nutrition.get("kinematics"):
        k = nutrition["kinematics"]
        ctx += "\n--- CINÉTICA METABÓLICA (TENDENCIA REAL DEL USUARIO) ---\n"
        ctx += f"• Velocidad actual de peso: {k.get('velocity_current', 0.0):.2f} unidades/día\n"
        ctx += f"• Aceleración metabólica: {k.get('acceleration', 0.0):.3f} unidades/día²\n"
        ctx += f"• Tendencia de Grasa Corporal (estimada): {k.get('body_fat_trend', 0.0):.2f}%\n"
        
        if k.get('is_losing_decelerating'):
            ctx += "⚠️ ADVERTENCIA: El usuario está perdiendo peso CADA VEZ MÁS LENTO (desaceleración). Hay un riesgo inminente de estancamiento. Asegura que las recetas sean muy saciantes y estrictas con las calorías.\n"
        elif k.get('is_losing_accelerating'):
            ctx += "⚠️ ADVERTENCIA: El usuario está perdiendo peso CADA VEZ MÁS RÁPIDO (aceleración). Hay riesgo de pérdida muscular. Asegura suficiente proteína y no bajes más los carbohidratos.\n"
        elif k.get('is_gaining_decelerating'):
            ctx += "⚠️ ADVERTENCIA: El usuario está ganando peso CADA VEZ MÁS LENTO. Si su meta es ganar masa, necesita recetas más densas en calorías.\n"
        elif k.get('is_gaining_accelerating'):
            ctx += "⚠️ ADVERTENCIA: El usuario está ganando peso MUY RÁPIDO. Riesgo de acumular mucha grasa. Controla las grasas añadidas.\n"
            
        ctx += "💡 INSTRUCCIÓN IA: Usa esta información biométrica para ajustar tu tono. Si se está estancando, dale un mensaje de ánimo y refuerza la adherencia. Si va excelente, felicítalo por su ritmo de progreso.\n"

    ctx += "-------------------------------------------------------------------\n"
    return ctx


def build_correction_context(review_feedback: str) -> str:
    """Genera el bloque de corrección urgente cuando el revisor médico rechazó el plan."""
    if not review_feedback:
        return ""
    return f"""
⚠️⚠️⚠️ CORRECCIÓN URGENTE DEL REVISOR MÉDICO ⚠️⚠️⚠️
El plan anterior fue RECHAZADO por las siguientes razones:
{review_feedback}

DEBES corregir TODOS estos problemas en esta nueva versión.
Genera comidas COMPLETAMENTE DIFERENTES que NO tengan estos problemas.
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
"""


def build_rag_context(user_facts: str) -> str:
    """Genera el bloque de hechos permanentes del usuario (memoria vectorial)."""
    if not user_facts:
        return ""
    return f"""
--- HECHOS PERMANENTES DEL USUARIO (MEMORIA VECTORIAL) ---
Estos son datos críticos que debes respetar.
{user_facts}
----------------------------------------------------------
"""


def build_time_context() -> str:
    """Genera el bloque de contexto temporal dinámico (fecha, día, clima caribeño y cultura)."""
    now_local = datetime.now()
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    dia_str = dias[now_local.weekday()]
    mes_str = meses_es[now_local.month - 1]

    # 1. Clima Dominicano (Temporada de lluvia vs seca)
    temporada = "Seca"
    if now_local.month in [5, 6, 7, 8, 9, 10, 11]:
        temporada = "De Lluvia/Huracanes"
        
    clima_hint = ""
    if now_local.month in [6, 7, 8, 9]:
        clima_hint = "- Clima: Hace MUCHO calor en el Caribe. Prioriza comidas más frescas, bowls, ensaladas y opciones hidratantes."
    elif temporada == "De Lluvia/Huracanes" and now_local.month in [10, 11]:
        clima_hint = "- Clima: Época de lluvias frecuentes. Integrar algún caldo o sopa (ej. sancocho ligero) puede ser muy reconfortante."
        
    # 2. Eventos Culturales y Feriados
    cultura_hint = ""
    if now_local.month == 12:
        cultura_hint = "- Cultura: Época de Navidad. Los usuarios tendrán cenas pesadas fuera de casa. Asegura que el plan sugerido aquí sea digestivo, anti-inflamatorio y ligero."
    elif now_local.month == 1:
        cultura_hint = "- Cultura: Inicio de año post-Navidad. Incluye opciones limpias, altas en fibra y ligeras para ayudar al 'reset' metabólico."
    elif now_local.month in [3, 4]:
        cultura_hint = "- Cultura: Cuaresma/Semana Santa. Sugiere inteligentemente proteínas como bacalao, tilapia, chillo, atún o berenjenas, limitando un poco la carne roja."

    hints = ""
    if clima_hint: hints += f"{clima_hint}\n"
    if cultura_hint: hints += f"{cultura_hint}\n"
    
    # 3. Día laboral vs fin de semana -> complejidad de recetas
    is_weekend = now_local.weekday() >= 5
    if is_weekend:
        hints += "- 🗓️ Es FIN DE SEMANA. El usuario tiene más tiempo. Puedes sugerir recetas un poco más elaboradas y meal prep dominical.\n"
    else:
        hints += "- 🗓️ Es DÍA LABORAL. Prioriza comidas rápidas (<15 min) o batch-cooking de la noche anterior.\n"

    return (
        f"\n--- 📅 CONTEXTO ESTACIONAL Y CULTURAL (OBLIGATORIO) ---\n"
        f"Hoy es {dia_str}, {now_local.day} de {mes_str} de {now_local.year}. Contexto en República Dominicana:\n"
        f"- Temporada Caribeña: {temporada}.\n"
        f"{hints}"
        f"INSTRUCCIÓN: Adapta sutilmente la propuesta a este contexto para que se sienta hiper-personalizado al entorno del usuario.\n"
        f"----------------------------------------------------------\n"
    )


def build_technique_injection(selected_techniques: list) -> str:
    """Genera el bloque de inyección de variedad culinaria (técnicas de cocción)."""
    if not selected_techniques or len(selected_techniques) < 3:
        return ""
    return (
        f"\n--- 👨🍳 INSTRUCCIÓN DINÁMICA DE VARIEDAD (OBLIGATORIA) ---\n"
        f"Para cumplir la regla de usar los MISMOS ingredientes del supermercado pero crear PLATOS DIFERENTES, "
        f"aplica obligatoriamente estas técnicas de cocción a las comidas principales (Almuerzo o Cena):\n"
        f"• Día 1 (Opción A): Aplica técnica '{selected_techniques[0]}'\n"
        f"• Día 2 (Opción B): Aplica técnica '{selected_techniques[1]}'\n"
        f"• Día 3 (Opción C): Aplica técnica '{selected_techniques[2]}'\n"
        f"Ajusta los gramos matemáticamente para cumplir las macros.\n"
        f"----------------------------------------------------------\n"
    )


def build_supplements_context(form_data: dict) -> str:
    """Genera el bloque de suplementos condicionado a la selección del usuario."""
    if not form_data.get("includeSupplements"):
        return ""

    from constants import SUPPLEMENT_NAMES

    selected_supps = form_data.get("selectedSupplements", [])
    if selected_supps:
        supp_names = [SUPPLEMENT_NAMES.get(s, s) for s in selected_supps]
        all_supps = set(SUPPLEMENT_NAMES.keys())
        not_selected = all_supps - set(selected_supps)
        not_selected_names = [SUPPLEMENT_NAMES.get(s, s) for s in not_selected]

        ctx = (
            "\n--- 💊 SUPLEMENTOS SELECCIONADOS (OBLIGATORIO — LEE CON CUIDADO) ---\n"
            f"LISTA EXACTA de suplementos que DEBES incluir: {', '.join(supp_names)}\n"
            f"TOTAL: {len(supp_names)} suplemento(s). Ni más, ni menos.\n\n"
            "⚠️ PROHIBIDO incluir cualquier suplemento que NO esté en la lista de arriba.\n"
        )
        if not_selected_names:
            ctx += f"❌ NO INCLUIR (el usuario NO los seleccionó): {', '.join(not_selected_names)}\n"
        ctx += (
            "\nPara CADA día del plan, agrega una sección 'supplements' con SOLO los suplementos listados arriba.\n"
            "Cada suplemento: 'name' (nombre exacto), 'dose' (dosis), 'timing' (momento del día), 'reason' (justificación).\n"
            "---------------------------------------------------\n"
        )
        return ctx
    else:
        return (
            "\n--- 💊 SUPLEMENTOS PERSONALIZADOS (OBLIGATORIO) ---\n"
            "El usuario ACTIVÓ la opción de incluir suplementos en su plan pero NO seleccionó suplementos específicos.\n"
            "DEBES agregar para CADA día del plan una sección 'supplements' (lista de objetos) con suplementos personalizados.\n"
            "Cada suplemento debe tener: 'name' (nombre), 'dose' (dosis), 'timing' (momento del día), 'reason' (justificación breve).\n"
            "Adapta las recomendaciones al objetivo del usuario, su nivel de actividad y condiciones médicas.\n"
            "Ejemplos: Proteína Whey, Creatina Monohidrato, Omega-3, Vitamina D3, Multivitamínico, Magnesio, etc.\n"
            "---------------------------------------------------\n"
        )


def build_grocery_duration_context(form_data: dict) -> str:
    """Genera el bloque de duración de compra (semanal/quincenal/mensual)."""
    grocery_duration = form_data.get("groceryDuration", "weekly")
    if not grocery_duration or grocery_duration == "weekly":
        return ""

    DURATION_LABELS = {"weekly": "SEMANAL (7 días)", "biweekly": "QUINCENAL (15 días)", "monthly": "MENSUAL (30 días)"}
    DURATION_DAYS = {"weekly": 7, "biweekly": 15, "monthly": 30}

    label = DURATION_LABELS.get(grocery_duration, "SEMANAL (7 días)")
    days_num = DURATION_DAYS.get(grocery_duration, 7)
    ctx = (
        f"\n--- 🛒 DURACIÓN DE COMPRA: {label} (OBLIGATORIO) ---\n"
        f"El usuario compra alimentos para {days_num} días en una sola ida al supermercado.\n"
        f"DEBES priorizar ingredientes que se conserven bien durante {days_num} días.\n"
    )
    if grocery_duration == "monthly":
        ctx += (
            "Usa predominantemente: granos secos, arroz, avena, tubérculos (yuca, batata, plátano verde),\n"
            "proteínas congelables (pollo, carne, pescado empacado al vacío), leche en polvo o UHT, huevos.\n"
            "Para cualquier perecedero, incluye instrucciones de congelación en la receta.\n"
        )
    elif grocery_duration == "biweekly":
        ctx += (
            "Equilibra entre frescos e ingredientes duraderos. Los vegetales de hoja y frutas muy maduras\n"
            "deben usarse en los primeros días del plan. Planifica congelación para proteínas frescas.\n"
        )
    ctx += (
        f"RECUERDA: Los PLATOS varían cada día, pero los ALIMENTOS BASE se repiten durante los {days_num} días.\n"
        "---------------------------------------------------\n"
    )
    return ctx


def build_skeleton_quality_context(quality_score: float, meal_adherence: dict) -> str:
    """Genera el bloque de métricas cuantitativas para el planificador (GAP 2)."""
    if quality_score is None:
        return ""
    
    ctx = f"\n--- 📊 MÉTRICAS CUANTITATIVAS DEL PLAN ANTERIOR ---\n"
    ctx += f"Quality Score: {quality_score:.2f} (0.0 a 1.0)\n"
    
    if meal_adherence:
        ctx += "Adherencia por comida:\n"
        for meal, rate in meal_adherence.items():
            try:
                rate_float = float(rate)
                ctx += f" - {meal.capitalize()}: {rate_float * 100:.0f}%\n"
            except (ValueError, TypeError):
                ctx += f" - {meal.capitalize()}: {rate}\n"
            
        # Encontrar la comida con peor adherencia
        try:
            valid_adherence = {k: float(v) for k, v in meal_adherence.items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())}
            if valid_adherence:
                worst_meal = min(valid_adherence.items(), key=lambda x: x[1])
                if worst_meal[1] < 0.5:
                    ctx += f"\n🚨 FALLO CRÍTICO: El {worst_meal[0].capitalize()} fue el punto de fallo principal ({worst_meal[1] * 100:.0f}%).\n"
                    ctx += f"INSTRUCCIÓN: Asigna al {worst_meal[0].capitalize()} los ingredientes MÁS simples y la técnica MÁS rápida.\n"
        except Exception:
            pass
            
    ctx += "---------------------------------------------------\n"
    return ctx


def build_adherence_context(adherence_hint: str, meal_level_adherence: dict = None, ignored_meal_types: list = None, abandoned_reasons: dict = None, emotional_state: str = None, successful_tone_strategies: list = None, nudge_conversion_rates: dict = None, frustrated_meal_types: list = None) -> str:
    """Genera el bloque de feedback loop según la adherencia del usuario general y por tipo de comida."""
    ctx = ""
    if adherence_hint == 'low':
        ctx += """
--- 📉 FEEDBACK DE ADHERENCIA: BAJA ---
El usuario no ha estado comiendo la mayoría de las comidas recomendadas recientemente.
ACCIÓN REQUERIDA (OBLIGATORIA): 
- Simplifica drásticamente las recetas y técnicas de preparación.
- Prioriza comidas de preparación MUY rápida o "comfort food" saludable.
- Usa ingredientes altamente palatables y conocidos para mejorar la adherencia.
- Evita recetas complejas de muchos pasos que desmotiven al usuario.
---------------------------------------
"""
    elif adherence_hint == 'high':
        ctx += """
--- 📈 FEEDBACK DE ADHERENCIA: ALTA ---
El usuario está siguiendo el plan de manera excelente y constante.
ACCIÓN REQUERIDA:
- Puedes introducir mayor variedad culinaria y técnicas creativas.
- Tienes libertad para incluir recetas un poco más elaboradas.
- Sigue retando positivamente su paladar manteniendo el buen balance.
---------------------------------------
"""

    if emotional_state == 'needs_comfort':
        ctx += """
--- ❤️ ESTADO EMOCIONAL: NECESITA CONFORT (OBLIGATORIO) ---
El usuario ha manifestado frustración, culpa o agobio recientemente.
ACCIÓN REQUERIDA (OBLIGATORIA):
- Diseña 'comfort food' saludable. Evita ensaladas frías o comidas aburridas.
- Prioriza texturas cremosas, platos calientes o versiones saludables de comidas reconfortantes (ej. pastas saludables, guisos, bowls calientes).
- La comida debe sentirse como un abrazo, no como una dieta estricta.
-----------------------------------------------------------
"""
    elif emotional_state == 'ready_for_challenge':
        ctx += """
--- 🔥 ESTADO EMOCIONAL: MOTIVADO / LISTO PARA RETOS ---
El usuario está altamente motivado y positivo.
ACCIÓN REQUERIDA:
- Es un excelente momento para introducir recetas más desafiantes o nuevos perfiles de sabor.
- Diseña comidas enfocadas en máximo rendimiento, variedad exótica o técnicas culinarias que requieran un poco más de atención.
--------------------------------------------------------
"""

    if meal_level_adherence:
        # Detectar comidas que el usuario sistemáticamente se salta (< 40% de adherencia)
        skipped_meals = [meal for meal, rate in meal_level_adherence.items() if rate < 0.4]
        if skipped_meals:
            skipped_str = ", ".join([m.capitalize() for m in skipped_meals])
            ctx += f"""
--- 🎯 FEEDBACK GRANULAR DE ABANDONO ---
El usuario SIEMPRE o CASI SIEMPRE se salta o abandona estas comidas: {skipped_str}.
ACCIÓN REQUERIDA (OBLIGATORIA):
- Haz que los platos de {skipped_str} sean ULTRA-SIMPLES, de preparación "cero esfuerzo" (ej. fruta, yogurt, batido, pan tostado).
- Si es posible, minimiza la cantidad de comida en {skipped_str} y redistribuye los macros al resto de comidas del día.
----------------------------------------
"""

    if ignored_meal_types:
        ignored_str = ", ".join([m.capitalize() for m in ignored_meal_types])
        ctx += f"""
--- 🔔 FEEDBACK DE RECORDATORIOS (NUDGES) IGNORADOS ---
El usuario ha ignorado sistemáticamente los recordatorios de estas comidas en los últimos días: {ignored_str}.
ACCIÓN REQUERIDA (OBLIGATORIA):
- Diseña estas comidas ({ignored_str}) como zero-effort (cero esfuerzo), que sean instantáneas de preparar o grab-and-go.
- Alternativamente, si ves que es imposible hacerlas zero-effort, omítelas o redúcelas a lo mínimo y redistribuye sus macros al resto del día.
-------------------------------------------------------
"""
            
    if abandoned_reasons:
        ctx += "\n--- 🧠 DIAGNÓSTICO CAUSAL DE ABANDONO (OBLIGATORIO) ---\n"
        ctx += "El usuario ha explicado explícitamente POR QUÉ abandonó ciertas comidas en su ciclo anterior:\n"
        for meal_type, reason in abandoned_reasons.items():
            if reason == 'no_time':
                ctx += f"- {meal_type.capitalize()}: Falta de tiempo. OBLIGATORIO: Genera recetas para {meal_type} que tomen MENOS de 10 minutos reales. Prioriza meal-prep o grab-and-go.\n"
            elif reason == 'no_ingredients':
                ctx += f"- {meal_type.capitalize()}: Faltaron ingredientes. OBLIGATORIO: Usa exclusivamente ingredientes muy comunes o que estén marcados como disponibles en despensa.\n"
            elif reason == 'not_hungry':
                ctx += f"- {meal_type.capitalize()}: Falta de hambre. OBLIGATORIO: Reduce el volumen de comida para {meal_type} usando ingredientes de alta densidad calórica y bajo volumen (nueces, aceite de oliva, batidos líquidos).\n"
            elif reason == 'didnt_like':
                ctx += f"- {meal_type.capitalize()}: No le gustó la comida. OBLIGATORIO: Apégate estrictamente a sus 'likes' y evita innovar demasiado en esta comida.\n"
            elif reason == 'ate_out':
                ctx += f"- {meal_type.capitalize()}: Comió fuera. OBLIGATORIO: Sugiere una comida rápida y sabrosa que compita con la comida de calle ('fake-away').\n"
        ctx += "-------------------------------------------------------\n"

    if successful_tone_strategies:
        ctx += "\n--- 🗣️ TONO DE COMUNICACIÓN COMPROBADO (OBLIGATORIO) ---\n"
        ctx += "Estos tonos/mensajes fueron exitosos cuando el usuario (u otros similares) estuvo frustrado:\n"
        for strat in successful_tone_strategies:
            ctx += f"- {strat}\n"
        ctx += "ACCIÓN REQUERIDA: Inyecta este estilo empático/motivacional en tus explicaciones de recetas y en el campo 'explanation' del plan.\n"
        ctx += "--------------------------------------------------------\n"

    if nudge_conversion_rates:
        low_conversion = [m for m, rate in nudge_conversion_rates.items() if rate < 0.2]
        if low_conversion:
            low_conv_str = ", ".join([m.capitalize() for m in low_conversion])
            ctx += f"""
--- ⚠️ CONVERSIÓN DE NUDGES CRÍTICA ---
El usuario ignora casi todos los recordatorios para: {low_conv_str} (Conversión muy baja).
ACCIÓN REQUERIDA: Minimiza o haz completamente opcionales estas comidas. Si las incluyes, deben ser de cero esfuerzo.
---------------------------------------
"""

    if frustrated_meal_types:
        frustrated_str = ", ".join([m.capitalize() for m in frustrated_meal_types])
        ctx += f"""
--- 😤 TIPOS DE COMIDA FRUSTRANTES ---
El usuario ha expresado frustración repetida o sentimientos negativos asociados a estas comidas: {frustrated_str}.
ACCIÓN REQUERIDA: Aplica una estrategia de 'Comfort Food' extrema para {frustrated_str}. Evita platos aburridos o restricciones excesivas en estos horarios específicos.
--------------------------------------
"""

    return ctx


def build_success_patterns_context(successful_techniques: list, abandoned_techniques: list, cold_start_recs: list = None) -> str:
    """Genera el bloque de patrones de éxito basado en el historial real de consumo."""
    if not successful_techniques and not abandoned_techniques and not cold_start_recs:
        return ""
        
    ctx = "\n--- 🎯 PATRONES DE ÉXITO Y ABANDONO (DATOS REALES) ---\n"
    
    if cold_start_recs:
        rec_str = ", ".join(set(cold_start_recs))
        ctx += f"❄️ COLD-START (USUARIO NUEVO): Perfiles similares al de este usuario han tenido MUCHO ÉXITO con los siguientes platos:\n"
        ctx += f"   💡 {rec_str}\n"
        ctx += "   INSTRUCCIÓN: Inspírate en estos platos o inclúyelos directamente en el plan para garantizar una alta adherencia inicial.\n\n"
    
    if successful_techniques:
        succ_str = ", ".join(set(successful_techniques))
        ctx += f"✅ TÉCNICAS EXITOSAS (El usuario SÍ comió estas preparaciones): {succ_str}\n"
        ctx += "   INSTRUCCIÓN: Fomenta el uso de estas técnicas, le gustan y le funcionan.\n"
        
    if abandoned_techniques:
        aban_str = ", ".join(set(abandoned_techniques))
        ctx += f"❌ TÉCNICAS ABANDONADAS (El usuario IGNORÓ estas preparaciones): {aban_str}\n"
        ctx += "   INSTRUCCIÓN: EVITA usar estas técnicas en este ciclo. Causaron fricción o no le gustaron.\n"
        
    ctx += "-------------------------------------------------------\n"
    return ctx


def build_fatigue_context(fatigued_ingredients: list) -> str:
    """Genera el bloque de contexto para la fatiga de ingredientes (Gap A)."""
    if not fatigued_ingredients:
        return ""
    
    ing_str = ", ".join(fatigued_ingredients)
    ctx = "\n--- ⚠️ FATIGA DE INGREDIENTES DETECTADA (OBLIGATORIO) ---\n"
    ctx += f"El usuario ha estado consumiendo excesivamente los siguientes ingredientes/categorías: {ing_str}.\n"
    ctx += "INSTRUCCIÓN CRÍTICA: Debes REDUCIR DRÁSTICAMENTE o EVITAR COMPLETAMENTE el uso de estos ingredientes en este nuevo plan para prevenir el aburrimiento y abandono.\n"
    ctx += "Busca fuentes alternativas de proteínas/carbohidratos.\n"
    ctx += "----------------------------------------------------------\n"
    return ctx


def build_quality_hint_context(quality_hint: str, drastic_strategy: str = None) -> str:
    """Genera el bloque de contexto para el hint de calidad del plan (Gap B) con soporte de A/B Testing."""
    if not quality_hint:
        return ""
    
    ctx = "\n--- 🎯 DIRECTIVA DE CALIDAD ADAPTATIVA (OBLIGATORIA) ---\n"
    if quality_hint == "drastic_change":
        ctx += "El usuario ha rechazado la mayoría de los planes recientes (Quality Score bajo sostenido).\n"
        if drastic_strategy == "ethnic_rotation":
            ctx += "INSTRUCCIÓN CRÍTICA (ESTRATEGIA ÉTNICA): Cambia radicalmente el perfil étnico de los sabores. Si usabas sabores criollos, usa orientales, mediterráneos, mexicanos o internacionales.\n"
        elif drastic_strategy == "texture_swap":
            ctx += "INSTRUCCIÓN CRÍTICA (ESTRATEGIA DE TEXTURAS): Cambia radicalmente las texturas de las comidas. Si había muchos guisos o purés, usa preparaciones crujientes, horneadas, secas o frescas.\n"
        elif drastic_strategy == "protein_shock":
            ctx += "INSTRUCCIÓN CRÍTICA (ESTRATEGIA DE PROTEÍNAS): Usa fuentes de proteína completamente diferentes a las usuales (ej. más mariscos, opciones plant-based, cortes distintos, o preparaciones de huevos muy diferentes).\n"
        else:
            ctx += "INSTRUCCIÓN CRÍTICA: Haz un CAMBIO DRÁSTICO. Cambia completamente los perfiles de sabor, usa fuentes de proteína completamente diferentes a las usuales, y diseña un menú totalmente inesperado y novedoso para recuperar su interés.\n"
    elif quality_hint == "increase_complexity":
        ctx += "El usuario domina sus recetas actuales y se está aburriendo de lo simple (Quality Score indica pérdida de interés).\n"
        ctx += "INSTRUCCIÓN CRÍTICA: INCREMENTA LA COMPLEJIDAD. Introduce técnicas culinarias nuevas, platos de autor, o recetas que requieran un poco más de tiempo de preparación para mantener el engagement.\n"
    elif quality_hint == "break_plateau":
        ctx += "El usuario está estancado (Plateau detectado).\n"
        ctx += "INSTRUCCIÓN CRÍTICA: ROMPE EL ESTANCAMIENTO. Ajusta drásticamente la densidad de nutrientes (volumetrics) o sugiere recetas diametralmente opuestas a su rutina para darle un shock positivo a su dieta.\n"
    elif quality_hint == "simplify_urgently":
        ctx += "El usuario está perdiendo adherencia de forma continua (Plateau de Adherencia detectado).\n"
        ctx += "INSTRUCCIÓN CRÍTICA: SIMPLIFICA URGENTEMENTE. Reduce el número de ingredientes por receta, usa técnicas de cocción muy rápidas (ensaladas, a la plancha, airfryer), y ofrece platos extremadamente fáciles y seguros para recuperar su motivación.\n"
    else:
        return ""
        
    ctx += "--------------------------------------------------------\n"
    return ctx


def build_weight_history_context(weight_history: list) -> str:
    """Genera el bloque de contexto motivacional basado en el historial de peso (Gap C)."""
    if not weight_history or len(weight_history) < 2:
        return ""
    
    clean_history = []
    for w in weight_history:
        if isinstance(w, dict):
            val = w.get("weight") or w.get("value") or w.get("peso")
            if val is not None:
                clean_history.append(str(val))
        else:
            clean_history.append(str(w))
            
    if not clean_history:
        return ""

    history_str = " -> ".join(clean_history)
    ctx = "\n--- ⚖️ TENDENCIA DE PESO Y MOTIVACIÓN ---\n"
    ctx += f"Historial reciente de peso del usuario: {history_str}.\n"
    ctx += "INSTRUCCIÓN CRÍTICA: Utiliza esta tendencia para incluir un mensaje corto y altamente motivacional en la explicación del plan. Si hay progreso hacia su objetivo, felicítalo. Si está estancado o con un revés, ofrécele ánimos empáticos.\n"
    ctx += "---------------------------------------\n"
    return ctx


def build_liked_meals_context(liked_meals: list) -> str:
    """Genera el bloque de contexto explícito de los platos que el usuario ha marcado con Like (Gap G)."""
    if not liked_meals:
        return ""
    
    liked_fmt = ", ".join(liked_meals[:10])
    ctx = "\n--- ❤️ PLATOS CON 'ME GUSTA' EXPLÍCITO (OBLIGATORIO) ---\n"
    ctx += f"El usuario le ha dado 'Like' (❤️) a estos platos recientemente: {liked_fmt}.\n"
    ctx += "INSTRUCCIÓN CRÍTICA: Esta es una señal de extrema importancia. Prioriza incluir variaciones directas de estos sabores, ingredientes y técnicas en el plan, ya que garantizan alta satisfacción.\n"
    ctx += "--------------------------------------------------------\n"
    return ctx


def build_temporal_adherence_context(day_of_week_adherence: dict) -> str:
    """Genera el bloque de perfilamiento conductual por día de la semana."""
    if not day_of_week_adherence:
        return ""
    
    # Encontrar los días con menor adherencia (<= 0.6)
    sorted_days = sorted(day_of_week_adherence.items(), key=lambda x: x[1])
    low_adherence_days = [d[0] for d in sorted_days if d[1] <= 0.6]
    
    if not low_adherence_days:
        return ""
        
    ctx = "\n--- 📆 PERFIL CONDUCTUAL POR DÍA DE LA SEMANA ---\n"
    ctx += f"El usuario suele tener DIFICULTAD o ABANDONA la dieta los siguientes días: {', '.join(low_adherence_days)}.\n"
    ctx += "INSTRUCCIÓN CRÍTICA: Cuando generes platos para esos días específicos, MUTA tu estrategia:\n"
    ctx += "- NO pongas recetas complejas ni de muchos pasos.\n"
    ctx += "- Diseña 'Comfort Food' Saludable (Ej: Hamburguesas fit, pizza con base de avena/coliflor, wraps rápidos).\n"
    ctx += "- Haz que el tiempo de preparación sea inferior a 10 minutos.\n"
    ctx += "---------------------------------------------------\n"
    return ctx


def build_pantry_context(form_data: dict) -> str:
    """Genera el bloque de reciclaje de despensa (Zero-Waste predictivo)."""
    current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
    if not current_pantry or not isinstance(current_pantry, list):
        return ""

    clean_pantry = [item.strip() for item in current_pantry if item and isinstance(item, str) and len(item.strip()) > 2]
    if not clean_pantry:
        return ""

    if form_data.get("_is_rotation_reroll", False):
        return ""

    PERISHABLE_KEYWORDS = [
        "aguacate", "pescado", "pollo", "carne", "res", "cerdo", "tomate", "lechuga", "espinaca",
        "brocoli", "brócoli", "guineo", "platano", "plátano", "banano", "manzana", "fresa",
        "vegetal", "cebolla", "cilantro", "verdura", "marisco", "camaron", "camarón", "queso",
        "leche", "yogurt", "huevo", "zanahoria", "pimiento", "aji", "ají", "berenjena",
        "calabacín", "zucchini"
    ]

    perishables = []
    stables = []
    predictions = []
    
    for item in clean_pantry:
        item_lower = item.lower()
        if "[⚠️ PREDICCIÓN:" in item:
            predictions.append(item)
        elif "[⚠️ URGENTE:" in item or "[⚠️ ATENCIÓN:" in item or "caducado" in item_lower or "caduca en" in item_lower:
            perishables.append(item)
        elif any(key in item_lower for key in PERISHABLE_KEYWORDS):
            perishables.append(item)
        else:
            stables.append(item)

    ctx = "\n--- ♻️ PRIORIDAD DE RECICLAJE DE DESPENSA E INVENTORY INTELLIGENCE ---\n"
    ctx += "El usuario ya tiene los siguientes ingredientes en su despensa:\n\n"

    if perishables:
        import json
        ctx += f"⚠️ INGREDIENTES URGENTES (a punto de dañarse - PRIORIDAD MÁXIMA PARA SALVARLOS ECONÓMICAMENTE):\n"
        ctx += f"{json.dumps(perishables, ensure_ascii=False)}\n\n"

    if predictions:
        import json
        ctx += f"📉 PREDICCIÓN DE AGOTAMIENTO (INVENTORY INTELLIGENCE):\n"
        ctx += f"{json.dumps(predictions, ensure_ascii=False)}\n"
        ctx += f"INSTRUCCIÓN: El sistema ha calculado que estos ingredientes se agotarán muy pronto basado en la tasa de consumo.\n"
        ctx += f"Diseña los primeros días usándolos, y para los últimos días de la semana diseña OPORTUNIDADES CON PROTEÍNAS/INGREDIENTES ALTERNATIVOS para cuando ya no haya.\n\n"

    if stables:
        import json
        ctx += f"✅ INGREDIENTES ESTABLES (Larga duración - Usar como complemento):\n"
        ctx += f"{json.dumps(stables, ensure_ascii=False)}\n\n"

    if form_data.get("_is_background_rotation"):
        ctx += "🛑 REGLA DE RECICLAJE ESTRICTA ('ACTUALIZAR PLATOS'): El usuario solicitó cambiar su menú PERO sin tener que ir al supermercado de nuevo.\n"
        ctx += "Tu ÚNICO POOL de ingredientes permitidos es ESTRICTAMENTE la lista dictada arriba.\n"
        ctx += "ESTÁ ESTRICTAMENTE PROHIBIDO inventar ingredientes (proteínas, vegetales, carbohidratos principales) que no estén en la lista anterior.\n"
        ctx += "Diseña los platos como si estuvieras en una cocina solo con esos ingredientes.\n"
        ctx += "Sólo puedes añadir ingredientes básicos no mencionados si son absolutos pilares estructurales (sal, pimienta, aceite, ajo).\n"
    else:
        ctx += "ESTRATEGIA ZERO-WASTE: Es OBLIGATORIO que bases este nuevo plan en agotar estos ingredientes sobrantes ANTES de pedirle que compre productos nuevos. Sé creativo para transformarlos en platos totalmente nuevos.\n"
        
    ctx += "----------------------------------------------------------\n"
    return ctx


def build_unified_behavioral_profile(user_facts: str, fatigued_ingredients: list, liked_meals: list, liked_flavor_profiles: list, cold_start_recs: list, allergies_list: list, llm_retrospective: str = "") -> str:
    """
    Consolida la memoria del usuario resolviendo conflictos entre preferencias, fatiga, alergias y recomendaciones genéricas.
    Prioridad: Lecciones (LLM) > Alergias > Fatiga > Preferencias (Facts/Liked) > Cold Start.
    """
    ctx = "\n--- 🧠 PERFIL DE COMPORTAMIENTO UNIFICADO Y RESOLUCIÓN DE CONFLICTOS ---\n"
    
    # 0. Lecciones Cualitativas (Prioridad Máxima)
    if llm_retrospective:
        ctx += f"🧠 LECCIONES DE APRENDIZAJE CONTINUO (MÁXIMA PRIORIDAD): \n{llm_retrospective}\nAPLICA ESTAS REGLAS OBLIGATORIAMENTE EN TUS DECISIONES.\n\n"
        
    # 1. Alergias (Prioridad Absoluta)
    alergias_str = ", ".join(allergies_list) if allergies_list else "Ninguna"
    ctx += f"🛑 RESTRICCIONES MÉDICAS Y ALERGIAS (PRIORIDAD 1): {alergias_str}. NUNCA INCLUYAS ESTOS INGREDIENTES, INCLUSO SI LE GUSTAN.\n\n"
    
    # 2. Fatiga
    fatiga_str = ", ".join(fatigued_ingredients) if fatigued_ingredients else "Ninguna"
    ctx += f"⚠️ INGREDIENTES FATIGADOS (PRIORIDAD 2): {fatiga_str}. El usuario está aburrido de estos. REDUCE DRÁSTICAMENTE su uso o evítalos temporalmente (máximo 1 vez a la semana), incluso si son sus favoritos.\n\n"
    
    # 3. Preferencias a largo plazo (Liked Meals & Facts)
    ctx += "❤️ PREFERENCIAS ESTABLECIDAS (PRIORIDAD 3):\n"
    if liked_flavor_profiles:
        ctx += f"- Perfiles de Sabor Favoritos (Basado en Likes): {', '.join(liked_flavor_profiles)}.\n"
    if liked_meals:
        ctx += f"- Comidas que le gustan explícitamente: {', '.join(liked_meals)}.\n"
    if user_facts:
        ctx += f"- Hechos permanentes sobre sus gustos:\n{user_facts}\n"
    if not liked_meals and not user_facts:
        ctx += "- No hay suficientes preferencias establecidas aún.\n"
    ctx += "Usa estas preferencias SOLO SI NO ENTRAN EN CONFLICTO con la fatiga o alergias.\n\n"
    
    # 4. Cold Start
    if cold_start_recs:
        ctx += f"🌱 SUGERENCIAS DE DESCUBRIMIENTO (PRIORIDAD 4): {', '.join(cold_start_recs)}.\n"
        ctx += "Usa estas sugerencias genéricas si el usuario tiene un perfil nuevo, respetando siempre las reglas de fatiga y alergias.\n\n"
        
    ctx += "-----------------------------------------------------------------------\n"
    return ctx


def build_prices_context() -> str:
    """Genera el bloque de inteligencia de precios (Budget-Aware)."""
    try:
        from shopping_calculator import get_master_ingredients
        master_list = get_master_ingredients()
        ctx = "\n--- 💰 INTELIGENCIA DE PRECIOS (BUDGET-AWARE) ---\n"
        ctx += "A continuación se muestra el costo promedio de los ingredientes (en RD$). Utiliza esta información para optimizar el presupuesto del plan si el usuario pide algo económico o para evitar ingredientes excesivamente costosos:\n"
        for m in master_list:
            price_lb = m.get("price_per_lb", 0)
            price_u = m.get("price_per_unit", 0)
            if price_lb:
                ctx += f"- {m['name']}: RD${price_lb}/lb\n"
            elif price_u:
                ctx += f"- {m['name']}: RD${price_u}/unidad\n"
        ctx += "----------------------------------------------------------\n"
        return ctx
    except Exception:
        return ""
