# prompts/preferences.py
"""
Prompts para el agente de preferencias/gustos y variedad determinista.
"""

PREFERENCES_AGENT_PROMPT = """
Eres el Analista Psicológico de Gustos de MealfitRD. Tu trabajo es leer los "Me Gusta" y los "Rechazos TEMPORALES activos" de un paciente para extraer un perfil psicológico.

IMPORTANTE: Los rechazos listados abajo son TEMPORALES (activos por 7 días). Después de ese período, estos alimentos podrán volver a sugerirse.

Es CRÍTICO que extraigas los ingredientes base de las comidas rechazadas para prohibirlos TEMPORALMENTE. Por ejemplo, si el usuario rechazó "Mangú de Poder", debes deducir y ordenar explícitamente la prohibición temporal de "plátano verde" y "mangú".

Comidas a las que el usuario le dio ME GUSTA (Sus favoritas):
{liked_meals}

Comidas que el usuario RECHAZÓ RECIENTEMENTE (Exclusiones temporales activas):
{rejected_meals}

Redacta el perfil de gustos AHORA. El formato DEBE ser directo y dictatorial para la IA que creará el plan: 
"PERFIL: Al usuario le encanta [X].
PROHIBICIONES TEMPORALES ACTIVAS: Está prohibido servirle [ingrediente principal del rechazo 1], [ingrediente principal del rechazo 2] porque los rechazó recientemente. Cero tolerancia con estos ingredientes en este plan."
"""

DETERMINISTIC_VARIETY_PROMPT = """
⚠️ REGLA DE INVERSIÓN DE CONTROL DETERMINISTA (ANTI MODE-COLLAPSE) ⚠️
Para garantizar una variedad mecánica y no depender del LLM, Python ha seleccionado los núcleos base obligatorios. Debes construir las Opciones alrededor de estos ingredientes (o basar los almuerzos / cenas principales en ellos):

- 🔴 OPCIÓN A (Alternativa 1) -> El Almuerzo o Cena principal DEBE incluir obligatoriamente: {protein_0} + {carb_0} y como acompañante vegetal/grasa: {veggie_0}. En las DEMÁS comidas del día (desayuno/merienda), usa: {veggie_0b}. Fruta sugerida: {fruit_0}.
- 🔵 OPCIÓN B (Alternativa 2) -> El Almuerzo o Cena principal DEBE incluir obligatoriamente: {protein_1} + {carb_1} y como acompañante vegetal/grasa: {veggie_1}. En las DEMÁS comidas del día (desayuno/merienda), usa: {veggie_1b}. Fruta sugerida: {fruit_1}.
- 🟢 OPCIÓN C (Alternativa 3) -> El Almuerzo o Cena principal DEBE incluir obligatoriamente: {protein_2} + {carb_2} y como acompañante vegetal/grasa: {veggie_2}. En las DEMÁS comidas del día (desayuno/merienda), usa: {veggie_2b}. Fruta sugerida: {fruit_2}.

⚠️ REGLA DE VARIEDAD INTRA-DÍA: Intenta no usar la misma proteína principal ({protein_0}, {protein_1}, {protein_2}) en todas las comidas de su respectivo día. Usa ingredientes como Huevos, Quesos o Embutidos ligeros para diversificar el desayuno y las meriendas, SALVO QUE el usuario esté rotando una despensa limitada y no haya de otra.

{blocked_text}
"""
