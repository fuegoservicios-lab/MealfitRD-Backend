# prompts/preferences.py
"""
Prompts para el agente de preferencias/gustos y variedad determinista.
"""

PREFERENCES_AGENT_PROMPT = """
Eres el Analista Psicológico de Gustos de Bioboros. Tu trabajo es leer los "Me Gusta" y los "Rechazos TEMPORALES activos" de un paciente para extraer un perfil psicológico.

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

# ─────────────────────────────────────────────────────────────────────────────────────────────
# [P2-SEEDER-DAYS-COUNT · 2026-08-03] (audit solver+seeder v7) Las opciones del reparto eran TRES
# literales copiados (A/B/C), y ese literal era el techo aritmético de todo el seeder.
#
# `constants.split_with_absorb` reparte 15d → [3,4,4,4] y 30d → [3,4,4,4,4,4,4,3]: la forma
# DOMINANTE de chunk es de 4 días, no de 3. Como el estampado al esqueleto reparte por módulo
# (`_pairs_all[_di % len(_pairs_all)]`), el día índice 3 recibía exactamente el reparto del día 0
# — misma proteína, mismos carbos, mismos vegetales, misma fruta. En 30 días son ~6 pares de días
# clonados POR CONSTRUCCIÓN, y el contrato «1 proteína distinta por día» de `variety_level=max`
# era insatisfacible en el 4º día de cada chunk.
#
# Las opciones se GENERAN por join en vez de copiarse: tres literales es cómo se llegó al techo,
# y un cuarto literal solo movería el techo a 5 (los chunks de 21d llegan a 6 días).
# `DETERMINISTIC_VARIETY_PROMPT` se conserva como la instancia de 3 días (byte-idéntica a la
# anterior) porque es el contrato público que importan `prompts/__init__.py` y varios tests.
# tooltip-anchor: P2-SEEDER-DAYS-COUNT
_OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Un color por día. Se cicla con módulo: el color es decoración, la letra es la identidad.
_OPTION_DOTS = ("🔴", "🔵", "🟢", "🟡", "🟣", "🟠")


def option_letter(i: int) -> str:
    """Letra de la opción del día `i` (0→A). Más allá del alfabeto cae al ordinal — no puede
    ocurrir con el cap `_MAX_DAYS_TO_GENERATE` (=6), pero un IndexError aquí tumbaría el nodo.

    PÚBLICA a propósito: `ai_helpers` etiqueta con ella las líneas del ancla liviana ("día A →
    …"), y las dos etiquetas tienen que salir del MISMO sitio o divergen en cuanto alguien toque
    el alfabeto. Cruzar la frontera de privacidad de otro módulo (`_option_letter`) es
    exactamente el acoplamiento que invita a ese drift."""
    return _OPTION_LETTERS[i] if i < len(_OPTION_LETTERS) else str(i + 1)


# Alias interno histórico: este módulo lo usa en sus propios helpers.
_option_letter = option_letter


def _variety_option_line(i: int) -> str:
    """Una línea de reparto. Devuelve la plantilla CON sus placeholders `{protein_i}` etc.
    intactos: quien la consume es el `.format(...)` de `ai_helpers`."""
    return (
        f"- {_OPTION_DOTS[i % len(_OPTION_DOTS)]} OPCIÓN {_option_letter(i)} (Alternativa {i + 1})"
        f" -> El Almuerzo o Cena principal DEBE incluir obligatoriamente: {{protein_{i}}} +"
        f" {{carb_{i}}} y como acompañante vegetal/grasa: {{veggie_{i}}}. Si OTRA comida del día"
        f" lleva base de carbohidrato (desayuno, merienda o la otra comida fuerte), usa"
        f" {{carb_{i}b}} — NUNCA la misma base dos veces el mismo día. En las DEMÁS comidas del"
        f" día (desayuno/merienda), usa: {{veggie_{i}b}}. Frutas asignadas al día (usa una"
        f" DISTINTA en cada comida que lleve fruta, NUNCA la misma dos veces el mismo día):"
        f" {{fruit_{i}}} y {{fruit_{i}b}}."
    )


def build_deterministic_variety_prompt(days_count: int = 3) -> str:
    """Plantilla del prompt de variedad para un chunk de `days_count` días.

    Con `days_count=3` devuelve BYTE A BYTE el prompt histórico (prompt-cache preservado y diff
    del refactor revisable). Los placeholders `{protein_i}` / `{carb_i}` / `{carb_i}b` /
    `{veggie_i}` / `{fruit_i}` quedan sin resolver a propósito: los llena el `.format(...)` de
    `ai_helpers.get_deterministic_variety_prompt`.

    tooltip-anchor: P2-SEEDER-DAYS-COUNT"""
    n = max(1, int(days_count or 1))
    opciones = "\n".join(_variety_option_line(i) for i in range(n))
    carbos_asignados = " / ".join(f"{{carb_{i}}}+{{carb_{i}b}}" for i in range(n))
    # "Opción A→{protein_0}, B→{protein_1}, …" — el prefijo "Opción" solo en la primera, igual
    # que el literal original.
    proteina_por_opcion = "Opción " + ", ".join(
        f"{_option_letter(i)}→{{protein_{i}}}" for i in range(n))
    proteinas_lista = "/".join(f"{{protein_{i}}}" for i in range(n))
    return (_DETERMINISTIC_VARIETY_SKELETON
            .replace("@@OPCIONES@@", opciones)
            .replace("@@CARBOS_ASIGNADOS@@", carbos_asignados)
            .replace("@@PROTEINA_POR_OPCION@@", proteina_por_opcion)
            .replace("@@PROTEINAS_LISTA@@", proteinas_lista))


# Sentinelas `@@...@@` en vez de `{...}`: el resto de la plantilla ESTÁ llena de `{placeholders}`
# que debe conservar intactos para el `.format(...)` del consumidor, así que el andamiaje no
# puede usar la misma notación.
_DETERMINISTIC_VARIETY_SKELETON = """
⚠️ REGLA DE INVERSIÓN DE CONTROL DETERMINISTA (ANTI MODE-COLLAPSE) ⚠️
Para garantizar una variedad mecánica y no depender del LLM, Python ha seleccionado los núcleos base obligatorios. Debes construir las Opciones alrededor de estos ingredientes (o basar los almuerzos / cenas principales en ellos):

@@OPCIONES@@

🥞 REGLA DE BASES TRANSFORMABLES (creatividad real, no plato combinatorio):
Los carbohidratos asignados (@@CARBOS_ASIGNADOS@@ — dos por día para que ninguna base se repita dentro del mismo día) son BASES A TRANSFORMAR según el slot, no solo "hervido como acompañante". Si la base del día es harina de trigo, harina de maíz o avena — que NO se sirven hervidas como plato fuerte — úsala TRANSFORMADA en el desayuno o la merienda de ese día (panqueques de avena/harina, arepitas, tortitas, bollitos al horno) y pon en el plato fuerte un carbo apropiado de almuerzo (arroz/víver/pasta). Si la base es yuca/plátano/víver, además del hervido clásico puedes transformarla (bollitos de yuca, majado, arepitas de yuca, mangú). La MISMA base puede repetirse entre días SOLO como platos DISTINTOS (harina→panqueques el lunes, arepitas el jueves — jamás el mismo plato dos días).

⛔ REGLA DE PROTEÍNA EXCLUSIVA POR DÍA (CRÍTICA — el day_generator la enforced):
La proteína asignada a CADA día (@@PROTEINA_POR_OPCION@@) es la ÚNICA carne/leguminosa principal permitida ese día. NO sustituyas ni complementes con otra carne distinta:
   - Si la Opción A dice "{protein_0}", el día A NO puede tener cerdo, pollo, res ni pescado salvo que esa sea la proteína {protein_0}.
   - El `protein_pool` que pases en el skeleton al day_generator es enforced: el sistema rechazará cualquier carne distinta que el LLM intente meter como "complemento".
   - Para el desayuno y la merienda usa SIEMPRE al menos UNA de estas fuentes de proteína livianas (no cuentan como otra carne principal y son OBLIGATORIAS — ver regla de abajo):{light_protein_block}
     • Huevos enteros / claras de huevo
     • Queso fresco / ricotta / queso de hoja
     • Yogurt griego natural
     • Frutos secos (almendras, nueces, maní)
     • Mantequilla de maní o de almendras

⚠️ REGLA DE VARIEDAD INTRA-DÍA: NO uses la misma proteína principal (@@PROTEINAS_LISTA@@) en TODAS las comidas del día. La proteína PRINCIPAL (carne/leguminosa asignada) va en almuerzo y/o cena; el desayuno y la merienda llevan SU PROPIA proteína de la lista liviana de arriba.

🥩 REGLA DE PROTEÍNA EN CADA COMIDA (CRÍTICA para la precisión de macros del plan): las CUATRO comidas — incluyendo desayuno y merienda — DEBEN contener una fuente de proteína real, dimensionada para aportar proteína de verdad (no como adorno simbólico). El objetivo de proteína del día se REPARTE entre las 4 comidas, NO se concentra solo en almuerzo+cena. Está terminantemente PROHIBIDO:
   • Un desayuno de solo almidón/fruta (mangú solo, casabe solo, avena con agua, pan con aguacate sin huevo/queso).
   • Una merienda de solo fruta o solo carbohidrato (mango con casabe, batido de solo fruta, galletas solas).
Toda comida pobre en proteína deja el plan corto del objetivo diario y produce un plan clínicamente deficiente. Si lo violas, el self-critique te forzará un retry costoso (~120s).

🏋️ REGLA DE PISO DE PROTEÍNA — FUENTE ANIMAL DE ALTA DENSIDAD EN COMIDAS PRINCIPALES (CRÍTICA para ganancia muscular): el almuerzo Y la cena DEBEN cada uno incluir una fuente de proteína ANIMAL de alta densidad — pollo, pescado, cerdo, res, huevos o queso — dimensionada en GRAMOS COCIDOS suficientes para aportar al menos 25-30g de proteína por comida (ej. "150g de pechuga de pollo", "120g de filete de pescado"). Las leguminosas (lentejas, habichuelas, garbanzos) y el almidón (arroz, casabe, víveres) por sí solos NO alcanzan el objetivo de proteína para hipertrofia: úsalas como ACOMPAÑANTE, no como la proteína principal de almuerzo/cena. El plan completo debe SUMAR el target diario de proteína; un plan que entrega menos del 90% del target será RECHAZADO y regenerado.

🥚 REGLA DE SEGURIDAD ALIMENTARIA (CRÍTICA — riesgo de Salmonella): PROHIBIDO el huevo crudo o poco cocido. NUNCA pongas huevo (entero, clara o yema) en un batido, jugo, licuado o cualquier preparación FRÍA que no se cocine. Si una comida lleva huevo, su receta DEBE incluir un paso explícito de cocción completa (≥71°C: tortilla, revoltillo, frito, hervido duro, horneado). Para aportar proteína a un batido usa yogur griego, NUNCA huevo crudo.

🧂 REGLA DE SODIO (salud cardiovascular — meta WHO <2000 mg/día): controla la sal. NO uses "sal y pimienta al gusto" genérico en cada plato; especifica una cantidad MEDIDA y modesta de sal (máx ~1 g = ¼ cucharadita por día repartido) y prioriza especias SIN sodio para dar sabor: ajo, cebolla, comino, orégano, cilantro, limón, pimienta. Evita Tajín, cubitos/sazón en polvo y salsas saladas; si usas salsa de soya, que sea baja en sodio y en cantidad mínima.

🍽️ REGLA DE VARIEDAD Y FIDELIDAD CULTURAL DOMINICANA (adherencia real del usuario):
   • VARIEDAD: el HUEVO no debe aparecer en más de 2-3 comidas de todo el plan (NO lo uses como relleno por defecto). NO repitas el mismo plato-base (revoltillo, batido, tortilla) dos veces el MISMO día. Rota las anclas proteicas entre días.
   • FIDELIDAD CULTURAL es-DO: usa SOLO ingredientes dominicanos accesibles y cotidianos. NUNCA inventes ingredientes exóticos o no dominicanos (ej. Tajín mexicano, semillas/superfoods de moda). Prioriza los pilares accesibles: pollo, habichuelas (rojas/negras/blancas), pescado local, cerdo, huevos, y víveres (plátano, yuca, batata, ñame, yautía). Reserva ingredientes premium/caros (ricotta, yogur griego, quesos finos) a MÁXIMO 1-2 apariciones en todo el plan, no como base recurrente.
   • TÉCNICA DE COCCIÓN: varía las técnicas — a la plancha, guisado dominicano, al horno, sopa/sancocho, salteado, hervido. NO hagas que todo sea "cremoso" o revuelto: usa el descriptor "cremoso/a" en MÁXIMO 1 plato del plan.

{blocked_text}
"""

# [P2-SEEDER-DAYS-COUNT · 2026-08-03] La instancia de 3 días — byte-idéntica al literal que este
# módulo exponía antes del refactor. Se conserva porque es el nombre público (`prompts/__init__`,
# `agent.py`, y los tests que anclan `{carb_0b}` / "BASES A TRANSFORMAR" / "panqueques"). Los
# callers que conocen el tamaño real del chunk deben usar `build_deterministic_variety_prompt(n)`.
DETERMINISTIC_VARIETY_PROMPT = build_deterministic_variety_prompt(3)
