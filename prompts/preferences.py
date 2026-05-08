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

⛔ REGLA DE PROTEÍNA EXCLUSIVA POR DÍA (CRÍTICA — el day_generator la enforced):
La proteína asignada a CADA día (Opción A→{protein_0}, B→{protein_1}, C→{protein_2}) es la ÚNICA carne/leguminosa principal permitida ese día. NO sustituyas ni complementes con otra carne distinta:
   - Si la Opción A dice "{protein_0}", el día A NO puede tener cerdo, pollo, res ni pescado salvo que esa sea la proteína {protein_0}.
   - El `protein_pool` que pases en el skeleton al day_generator es enforced: el sistema rechazará cualquier carne distinta que el LLM intente meter como "complemento".
   - Para diversificar desayuno/merienda usa SIEMPRE estas opciones livianas (no cuentan como otra carne):
     • Huevos enteros / claras de huevo
     • Queso fresco / ricotta / queso de hoja
     • Yogurt griego natural
     • Frutos secos (almendras, nueces, maní)
     • Mantequilla de maní o de almendras

⚠️ REGLA DE VARIEDAD INTRA-DÍA: NO uses la misma proteína principal ({protein_0}/{protein_1}/{protein_2}) en TODAS las comidas de su día. La proteína principal va en almuerzo y/o cena; desayuno y merienda usan las opciones livianas listadas arriba. Si lo violas, el self-critique te forzará un retry costoso (~120s).

{blocked_text}
"""
