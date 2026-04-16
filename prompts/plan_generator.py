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
    return f"""
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
    """Genera el bloque de contexto temporal dinámico (fecha, día, estación)."""
    now_local = datetime.now()
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    meses_es = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    dia_str = dias[now_local.weekday()]
    mes_str = meses_es[now_local.month - 1]

    estacion = "Verano"
    if now_local.month in [3, 4, 5]:
        estacion = "Primavera"
    elif now_local.month in [9, 10, 11]:
        estacion = "Otoño"
    elif now_local.month in [12, 1, 2]:
        estacion = "Invierno"

    return (
        f"\n--- 📅 CONTEXTO TEMPORAL ACTUAL (OBLIGATORIO) ---\n"
        f"Hoy es {dia_str}, {now_local.day} de {mes_str} de {now_local.year}. Estación promedio local: {estacion} tropical.\n"
        f"Aplica las estrategias de Continuidad Temporal considerando que el usuario comenzará este plan en estos días.\n"
        f"--------------------------------------------------\n"
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
    for item in clean_pantry:
        item_lower = item.lower()
        if any(key in item_lower for key in PERISHABLE_KEYWORDS):
            perishables.append(item)
        else:
            stables.append(item)

    ctx = "\n--- ♻️ PRIORIDAD DE RECICLAJE DE DESPENSA (ZERO-WASTE PREDICTIVO) ---\n"
    ctx += "El usuario ya tiene los siguientes ingredientes en su despensa:\n\n"

    if perishables:
        ctx += f"⚠️ INGREDIENTES ALERTA NARANJA (PERECEDEROS - DEBES USARLOS OBLIGATORIAMENTE Y PRIORIZARLOS EN LOS PRIMEROS DÍAS):\n"
        ctx += f"{', '.join(perishables)}\n\n"

    if stables:
        ctx += f"✅ INGREDIENTES ESTABLES (NO PERECEDEROS - USAR COMO COMPLEMENTO):\n"
        ctx += f"{', '.join(stables)}\n\n"

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
