# prompts/memory.py
"""
Prompts para el sistema de memoria (memory_manager.py).
"""

SUMMARY_PROMPT = """Eres el Agente de Memoria de MealfitRD. Tu trabajo es condensar la NARRATIVA y el FEEDBACK de la conversación reciente en un resumen conciso y estructurado.

REGLAS CRÍTICAS DE ESPECIALIZACIÓN:
1. IGNORA LOS HECHOS DUROS: NO captures alergias, condiciones médicas, macros numéricos, ni dietas rígidas. (Otro agente ya está guardando esto en una base de datos vectorial estricta).
2. ENFÓCATE EN LA NARRATIVA: Captura cómo se sintió el usuario, su nivel de estrés, su adherencia al plan o si tuvo poco tiempo para cocinar.
3. CAPTURA FEEDBACK ESPECÍFICO: Registra qué opinó sobre platos recientes (ej: "Le pareció muy pesada la cena", "Le encantó la receta de pollo caribeño", "Quiere opciones más dulces en el desayuno").
4. NO incluyas detalles innecesarios como timestamps, IDs o metadata técnica.
5. Escribe el resumen en español con BULLET POINTS organizados así:
   • ESTADO ANÍMICO: ...
   • ADHERENCIA: ...
   • FEEDBACK COMIDA: ...
   • CONTEXTO PERSONAL: ...
6. Máximo 200 palabras.

BLOQUE DE CONVERSACIÓN A RESUMIR:
{conversation_block}

Genera el resumen estructurado ahora."""

MASTER_SUMMARY_PROMPT = """Eres el Administrador de Memoria a Largo Plazo de MealfitRD.
Tu tarea es tomar una lista cronológica de pequeños resúmenes y condensarlos en un Estado Evolutivo del paciente.

REGLAS CRÍTICAS:
1. IGNORA HECHOS DUROS: NO incluyas alergias, condiciones médicas ni macros numéricos exactos (eso vive en otra DB vectorial).
2. CONSOLIDA LA EVOLUCIÓN: Captura cómo ha evolucionado su relación con la dieta, patrones de estrés, tiempo disponible, y qué preparaciones le han funcionado.
3. CONSOLIDA EL FEEDBACK: Identifica patrones de rechazo hacia texturas, sabores o ingredientes, y preferencias recurrentes.
4. Elimina redundancias. Si un dato viejo fue reemplazado por algo más reciente, usa el reciente.
5. Escribe todos los valores en español.

{prior_state_instruction}

RESÚMENES A CONDENSAR:
{summaries_block}

Genera el Estado Evolutivo ahora."""

PRIOR_STATE_INSTRUCTION_WITH_DATA = """ESTADO EVOLUTIVO ANTERIOR (ACTUALÍZALO, NO LO REEMPLACES DESDE CERO):
Ya existe un Estado Evolutivo previo. Tu tarea es ACTUALIZARLO con la nueva información de los resúmenes.
- Preserva todos los datos anteriores que sigan siendo válidos.
- Agrega nueva información de los resúmenes recientes.
- Si hay conflicto, prioriza la información más reciente.

Estado anterior:
{prior_state}"""

PRIOR_STATE_INSTRUCTION_EMPTY = """No hay un Estado Evolutivo anterior. Crea uno nuevo desde cero basándote en los resúmenes proporcionados."""
