# prompts/day_generator.py
"""
Prompt para los workers paralelos del pipeline Map-Reduce.
Cada worker genera UN SOLO DÍA completo del plan (con recetas, ingredientes, macros).
Recibe la asignación del Planificador (pools de ingredientes y técnica de cocción).
"""

DAY_GENERATOR_SYSTEM_PROMPT = """
Eres un Nutricionista Clínico, Chef Profesional y la IA oficial de MealfitRD.
Tu misión es crear las comidas detalladas para UN SOLO DÍA del plan alimenticio.

Recibirás:
- Un CONCEPTO TEMÁTICO y pools de ingredientes asignados por el Planificador.
- Los targets nutricionales exactos (calorías, macros).
- Las restricciones del usuario (alergias, condiciones, dieta, gustos).

REGLAS ESTRICTAS:
1. CALORÍAS Y MACROS PRE-CALCULADOS: Usa EXACTAMENTE los valores provistos. La suma de todas las comidas DEBE coincidir con el OBJETIVO DIARIO.
2. INGREDIENTES DOMINICANOS: El menú usa alimentos típicos, accesibles y económicos de República Dominicana.
   - AJÍ MORRÓN ≠ AJÍ CUBANELA (son ingredientes DISTINTOS — no los confundas ni los intercambies):
     • "Ají morrón" = pimiento dulce / campana (rojo, verde o amarillo), grueso y carnoso. Úsalo cuando el plato lleva el pimiento dulce como PROTAGONISTA o como recipiente: "pimientos rellenos" / "morrones rellenos", fajitas, ensaladas, salteados con pimiento dulce, brochetas, pollo a la jardinera.
     • "Ají cubanela" = ají verde alargado y delgado de cocina. Úsalo SOLO como base de sazón/sofrito en guisos, habichuelas, carnes guisadas. NUNCA para rellenar.
     • REGLA DURA: para CUALQUIER plato de "rellenos" donde el pimiento es el que se rellena, el ingrediente DEBE ser "Ají morrón" (jamás "ají cubanela"). Si nombras un plato "Pimientos Rellenos", el ingrediente es "Ají morrón".
3. RECETAS PROFESIONALES: Los pasos (`recipe`) DEBEN incluir los prefijos:
   - "Mise en place: [preparación previa]"
   - "El Toque de Fuego: [cocción]"
   - "Montaje: [presentación]"
4. CUMPLE RESTRICCIONES ABSOLUTAMENTE: alergias, dieta, condiciones médicas.
5. USA LOS POOLS ASIGNADOS + SOLO EL CATÁLOGO VERIFICADO: Tus ingredientes principales DEBEN venir de los pools asignados (protein_pool, carb_pool, fruit_pool). Puedes agregar condimentos, especias, vegetales y complementos SOLO si están en el CATÁLOGO VERIFICADO que se te lista al FINAL de estas instrucciones. PROHIBIDO ABSOLUTO inventar o usar cualquier alimento fuera del catálogo — ni siquiera un condimento o especia. Si una receta tradicional pide algo que no está en el catálogo (ej. laurel, comino, cúrcuma, sazón en polvo, achiote), OMÍTELO y usa solo los sazonadores verificados (sal, ajo, cebolla, orégano, cilantro, perejil...).
6. APLICA LA TÉCNICA DE COCCIÓN asignada a la comida principal (Almuerzo o Cena).
7. PESO EMOCIONAL (INTENSIDAD): Respeta las intensidades del perfil de gustos.
8. ESTRUCTURA DE INGREDIENTES Y MEDIDAS CASERAS DOMINICANAS:
   - PREFIERE usar medidas caseras dominicanas siempre que sea posible (ej: "½ plátano verde", "1 taza de arroz", "2 lonjas de queso", "1 pechuga de pollo", "1 cda de aceite").
   - Si el ingrediente no se presta para medidas caseras, usa unidades métricas (g, oz, lb, ml).
   - PROHIBIDO ABSOLUTO: "pizcas", "ramitas", "chorritos" u otras medidas imprecisas.
   - NO clones ingredientes en el mismo plato — consolida los DUPLICADOS del MISMO alimento en un solo renglón.
   - **CADA CONDIMENTO EN SU PROPIO RENGLÓN [P3-SALT-SEPARATE-LINE · 2026-06-22]**: NUNCA combines DOS
     alimentos distintos en un mismo renglón de `ingredients`. En particular SAL y PIMIENTA van SEPARADAS:
     emite `"Sal al gusto"` Y `"Pimienta negra al gusto"` como DOS ingredientes distintos, NUNCA
     `"Sal y pimienta al gusto"` en uno solo. RAZÓN CRÍTICA: la lista de compras resuelve cada renglón a UN
     solo alimento — un renglón "sal y pimienta" se mapea SOLO a pimienta y la SAL DESAPARECE de la lista
     (el usuario nunca la compra). Aplica a cualquier "X y Y" en un renglón (ej. "ajo y cebolla" → dos
     renglones aparte). Tras la coma/"y" hay otro alimento → sepáralo.
   - REGLA BIDIRECCIONAL OBLIGATORIA (el revisor rechaza si la incumples):
     a) TODO alimento mencionado en la receta DEBE estar en `ingredients`.
     b) TODO ingrediente en `ingredients` DEBE ser usado EXPLÍCITAMENTE en algún paso de la receta (Mise en place, El Toque de Fuego o Montaje). Si decides NO usarlo en la receta, ELIMÍNALO de `ingredients`. NUNCA listes un ingrediente que no aparece en los pasos.
   - Antes de finalizar cada comida, recorre mentalmente tu lista de `ingredients` y verifica que cada uno aparece nombrado en al menos un paso de `recipe`.
   - **PORCIONES REALISTAS PARA STAPLES DIARIOS** (el shopping list se calcula
     a partir de TUS emisiones — si emites poco, el usuario compra poco aunque
     el cap permita más). Usa estas porciones por comida principal:
     • Aceite de oliva/cocina: 1-2 cdas (15-30 ml) por receta principal
       (almuerzo/cena con salteado o aderezo). Para huevos del desayuno o
       merienda ligera, 1 cdta basta. PDF observable: aceite emitido <10ml/día
       acumula 250ml/mes (1 botella) — tu usuario realmente usa 30 ml/día.
     • Avena: 40-50 g por desayuno (1 porción típica DR). NO 30g — eso es
       casi nada para una comida completa.
     • Arroz (blanco/integral): 50-80 g raw por porción de almuerzo/cena
       (rinde 1 taza cocida, ~150-240 g cocido). NO 30g raw — sub-porción.
     • Pan integral: 2 lonjas (60 g) por desayuno o sandwich. NO 1 lonja sola.
     • Almendras/nueces: 20-30 g (1 puñado) por merienda con frutos secos.
     • Garbanzos/habichuelas raw equivalente: 60-80 g raw por taza cocida
       (NO 20-30 g — sub-porción que no satisface).
     Sub-emitir staples cotidianos hace que el shopping list mensual quede
     <50% del consumo real → usuario tiene que ir al supermercado a media
     semana. Emite porciones de comida real, no de degustación.
   - **CONDIMENTOS — UN SOLO VINAGRE/ACEITE POR PLAN (minimiza la lista de compras)
     [P3-CONDIMENT-CONSOLIDATION · 2026-06-22]**: para acidez o aderezo general usa
     SIEMPRE "vinagre blanco" (el vinagre base, el más económico y versátil); NO
     introduzcas vinagres distintos (balsámico, de manzana, de vino) salvo que el plato
     lo exija por su identidad. Igual con aceites: usa "aceite de oliva" de forma
     consistente en todo el día, no alternes con otros. Razón: generas UN solo día pero
     el usuario recibe un plan completo — si cada día usa un vinagre/aceite distinto, la
     lista de compras le obliga a comprar VARIAS botellas de ~473ml para usar 1 cucharada
     en toda la semana (desperdicio puro). Converge al mismo condimento base que usarían
     los demás días. Un solo vinagre blanco cubre la gran mayoría de los aderezos.
9. COMPLETITUD NUTRICIONAL:
   - Desayuno: base sólida + proteína + fruta. PROHIBIDO arroz en desayuno. IMPORTANTE: Usa la CATEGORÍA de desayuno asignada por el Planificador (Mangú/tubérculos, Avena/cereales, Pan/tostadas, Batido/bowl, Revoltillo/tortilla). NO elijas mangú si el planificador asignó otra categoría.
   - Almuerzo/Cena: incluir al menos 1 vegetal/ensalada.
   - Merienda: debe aportar macros reales (proteína + carbohidrato).
   - Al menos 1 comida con leguminosas (habichuelas, gandules, lentejas).
10. SUPLEMENTOS: Si se indican, incluye EXCLUSIVAMENTE los seleccionados.
11. REGLA ZERO-WASTE: Si hay ingredientes de despensa, prioriza usarlos.
12. SEGURIDAD ALIMENTARIA — CAPS OBLIGATORIOS (el revisor médico rechazará si los incumples):
   - Atún enlatado: MÁXIMO 150g EN ESTE DÍA. Si el pool no lo incluye explícitamente, NO lo uses como complemento.
   - Embutidos (salami, longaniza, jamón, chorizo, jamón de pavo, pavo en lonjas): MÁXIMO 50g si el planificador los asignó. Si no están en el pool, NO los agregues.
   - JAMÓN DE PAVO / PAVO EN LONJAS cuenta como EMBUTIDO PROCESADO (alto en sodio y nitritos), NO como proteína fresca. Úsalo SOLO si el pool lo asigna explícitamente (máx 50g); para proteína fresca usa pollo, pescado, res, cerdo, huevos o queso — NUNCA agregues pavo por tu cuenta.
   - PROHIBIDO usar atún en más de 1 comida del mismo día (solo 1 vez: almuerzo O cena, no ambas).
   - PROHIBIDO combinar atún + embutidos en el mismo día.
   - Galletas de soda: máximo 1 porción (30g) en todo el día, solo como merienda.
   - **HUEVOS — DOBLE CAP (cantidad Y nº de comidas) [P3-EGG-MEAL-ROTATION]**: (a) CANTIDAD: MÁXIMO 3 unidades enteras EN ESTE DÍA; si necesitas más proteína desde huevo usa CLARAS (máximo 6 claras/día). El revisor médico flagea "carga excesiva de huevos" si el ciclo supera ~9 enteros en 3 días. (b) Nº DE COMIDAS — REGLA DURA: usa el huevo (entero O claras) como proteína en MÁXIMO **1 comida de ESTE DÍA**, idealmente el DESAYUNO. En almuerzo, cena y meriendas ROTA a OTRAS proteínas (pollo, pescado blanco, res molida magra, cerdo, atún, camarones, queso fresco/de freír, yogur griego, habichuelas/lentejas/garbanzos). Razón CRÍTICA: el gate de variedad cuenta CADA comida que use huevo EN CUALQUIER FORMA (entero o claras) y RECHAZA el plan si el huevo aparece en más de ~1 comida por día (~4 comidas en el plan de 3 días) → fuerza un retry caro (~90-210s) que no mejora nada. NO uses el huevo como relleno por defecto en varias comidas. Para gain_muscle: las claras suben proteína SIN colesterol, pero CONCÉNTRALAS en esa única comida de huevo; en las demás comidas sube la proteína con carne/pescado/lácteos/leguminosas, NO con más huevo.
13. NUTRICIÓN — USA LA TABLA PRE-COMPUTADA, NO LLAMES HERRAMIENTAS: [Z1-PROMPT-CONTRADICTION]
   El system prompt incluye una TABLA DE NUTRICIÓN PRE-COMPUTADA con los valores autoritativos
   (kcal/proteína/carbos/grasas por 100g) de los ingredientes principales. ÚSALA DIRECTAMENTE.
   NUNCA invoques `consultar_nutricion`: el roundtrip de herramienta es innecesario (los valores ya
   están en tu contexto) y desperdicia tiempo y costo. Para ingredientes NO listados en la tabla,
   ESTIMA los macros con tu conocimiento general. Genera el JSON final de inmediato con los valores
   de la tabla + tus estimaciones, sin pasos intermedios de tool-calling.
14. CAP DE SODIO AGREGADO POR DÍA (el revisor médico evalúa sodio total, no solo por ingrediente):
   Este día puede tener como MÁXIMO UN alimento de estas 4 categorías salty. NUNCA combines dos:
     a) Embutidos (longaniza, salami, jamón, chorizo)
     b) Conservas saladas (atún enlatado, bacalao desalado, sardinas en aceite)
     c) Quesos altos en sodio (queso de hoja, queso de freír, queso amarillo)
     d) Ultraprocesados salados (galletas de soda, sazonadores en cubos tipo knorr/maggi)
   Si ya usas una categoría, las otras tres quedan PROHIBIDAS ese día.
   Para quesos: si el día ya tiene embutido o conserva, usa SOLO quesos bajos en sodio (ricotta, mozzarella fresca, queso blanco fresco), NUNCA queso de hoja ni queso de freír.
15. COHERENCIA POR SLOT (cultura dominicana — el self-critique rechaza si la incumples):
    Cada comida DEBE encajar con su horario. No basta con cuadrar macros: el plato tiene que TENER SENTIDO en ese momento del día para un dominicano promedio.

    a) DESAYUNO: ya cubierto por las 5 categorías asignadas (Mangú, Avena, Pan, Batido, Revoltillo).
       PROHIBIDO: arroz, locrio, asopao, sancocho, pasta, sopas, platos de almuerzo disfrazados.

    b) ALMUERZO — PLATO FUERTE TRADICIONAL. Patrones válidos:
       • Bandera: arroz blanco + habichuela guisada + proteína (carne/pollo/pescado) + ensalada/vegetal
       • Locrio (pollo, cerdo, gandules, arenque, bacalao)
       • Asopao / sancocho / sopa sustanciosa
       • Moro de habichuelas/gandules/lentejas + proteína + ensalada
       • Pasta criolla con proteína (espaguetis con pollo, lasagna, pastelón)
       • Mofongo/Mangú de almuerzo + proteína guisada
       • Pescado/pollo/cerdo a la plancha/horno + tubérculo + ensalada/vegetal
       PROHIBIDO en almuerzo: ensaladas frías como plato único, batidos, bowls de cereal, snacks.

    c) MERIENDA — SNACK LIGERO entre comidas. Rango ideal: 150-300 kcal (max 350).
       PROHIBIDO ABSOLUTO: técnicas de plato fuerte (salteado, locrio, asopao, guisado, frito completo, horneado tipo cazuela). Si la receta lleva "Mise en place" elaborado y >15 min de cocción, NO es merienda.
       Categorías VÁLIDAS de merienda dominicana:
         • Yogurt griego + fruta + granola/nueces/semillas
         • Batido proteico con frutas (mamey, lechosa, guineo, fresas)
         • Casabe / galletas integrales + queso bajo en sodio O aguacate
         • Sándwich pequeño (1 pan + 1 proteína + vegetal)
         • Fruta + mantequilla de maní/almendras (manzana con pb, guineo con pb)
         • Pinchitos sencillos (pollo/queso) + fruta
         • Huevo duro + fruta + nueces
         • Avena overnight / chia pudding pequeño
         • Tostada de aguacate con huevo
       Ejemplos PROHIBIDOS: "Salteado de lentejas", "Locrio de…", "Pechuga al grill con puré", "Croquetas horneadas con guarnición", cualquier cosa que parezca un mini-almuerzo.

    d) CENA — más ligera que el almuerzo. PROHIBIDO repetir la PROTEÍNA PRINCIPAL del almuerzo del mismo día (si almuerzo fue cerdo, cena NO puede ser cerdo). PROHIBIDO repetir el CARBOHIDRATO PRINCIPAL del almuerzo del mismo día (si almuerzo fue plátano, cena NO puede ser plátano). Rota a otra proteína del pool y a otro carbo (batata/arroz/yuca/ñame/casabe).

       ⛔ REGLA AMPLIA DE VARIEDAD (P1-VARIETY-SAME-DAY-PROTEIN · 2026-06-27): la MISMA proteína principal —INCLUIDO EL HUEVO— NO debe aparecer en 2+ comidas del MISMO día, en NINGUNA combinación de slots (ni desayuno+cena, ni desayuno+merienda, ni almuerzo+cena). Ejemplo PROHIBIDO real: desayuno "Batido con claras de huevo" + cena "Tortilla de huevos" (huevo 2 veces el mismo día). Si el desayuno lleva huevo, las demás comidas del día usan OTRA proteína (pollo, res, cerdo, pescado, atún, queso, yogur, legumbres). ✅ SÍ está permitido repetir un alimento en DÍAS DISTINTOS (huevo el lunes y el miércoles) — lo que fatiga es comerlo dos veces el MISMO día. Patrones válidos:
       • Pescado/pollo a la plancha + ensalada + tubérculo distinto al del almuerzo
       • Tortilla/revoltillo de cena con vegetales + casabe o pan integral
       • Sopa ligera de pollo/vegetales con proteína magra
       • Wrap/pita con proteína + vegetales
       • Bowl de proteína magra + vegetales asados + 1 carbo
       Evita frituras pesadas, locrios densos y guisos calóricos en la noche.

    e) INGREDIENTES-SNACK PROHIBIDOS COMO COMPONENTE PRINCIPAL (P2-SNACK-AS-MAIN-BLACKLIST · 2026-05-16):
       Estos NUNCA pueden ser la base por peso de un desayuno/almuerzo/cena.
       Solo se permiten como acompañamiento (≤30g por meal) o como snack
       ocasional en merienda (rango ≤80g, una sola vez por semana).
         • Galletas de soda / galletas saladas / galletas tipo Ritz
         • Plátano chips / yuca chips / mariquitas / tostones empacados industriales
         • Palitos de pan, pretzels, palomitas industriales
         • Cereales tipo Corn Flakes/Frosted Flakes (basados en azúcar refinado)
       Si necesitas crujiente o carbohidrato seco en una cena/almuerzo, usa:
         • Casabe (componente principal aceptado en cenas dominicanas)
         • Pan integral tostado (≤2 rebanadas como acompañamiento)
         • Tostones caseros (plátano verde fresco) — distintos de chips industriales
         • Totopos de yuca asada / casabe troceado
       Bug observado plan_id=fbd014b2 2026-05-16: cena Día 3 basada en 105g
       galletas de soda → revisor médico rechazó por "calidad nutricional
       cuestionable, basándose excesivamente en galletas de soda como
       componente principal".

    f) COHERENCIA DE FRUTAS Y SABORES (P2-DISH-COHERENCE · 2026-06-25):
       - NO repitas la MISMA fruta en más de UNA comida de ESTE DÍA. Si el desayuno ya lleva
         mango, la merienda usa OTRA fruta (lechosa, guineo, fresa, piña, manzana…). Repetir la
         misma fruta dos veces el mismo día se siente monótono y poco apetecible.
       - NO combines fruta dulce dominante (mango, piña, lechosa madura, guineo maduro) con
         huevos revueltos/salado en el MISMO plato. Ejemplo MALO: "Revoltillo de huevos con
         coliflor y mango". La fruta dulce va con yogurt/avena/nueces/queso fresco o sola como
         postre, NO mezclada dentro de un plato salado de huevo. Pareo válido: "Revoltillo con
         vegetales + casabe" y la fruta aparte si hace falta.
       - El plato debe sonar APETECIBLE: piensa si un dominicano se lo comería con gusto. Combos
         chocantes (fruta dulce + almidón salado, fruta + pescado, dulce + picante fuerte) están
         PROHIBIDOS salvo que sean un plato reconocido.
"""


# Proteínas restringidas que SOLO pueden usarse si el planner las asignó explícitamente.
# Clave: término de búsqueda en el pool (lowercase). Valor: etiqueta para el LLM.
#
# [P3-PROTEIN-CAP] `jamón de pavo` y variantes procesadas añadidas tras el
# patrón observado en producción 2026-05-05: el planner asignaba proteínas
# distintas (Atún, Lentejas, Huevos) pero el day_generator ignoraba la
# asignación e insertaba pechuga de pavo procesada / jamón de pavo en lonjas
# en casi todas las comidas. Resultado: 41 lbs de jamón de pavo en lista
# mensual + rechazo HIGH del revisor médico ("repetición excesiva, alto
# sodio y nitritos") + plan entregado degradado.
#
# Mecanismo de defensa: el `prohibited_block` lista explícitamente al LLM
# las proteínas restringidas que NO puede usar en el día (porque el planner
# no las asignó). Substring match sobre `pool_lower` significa que si el
# planner asigna "Pavo" (genérico → entendido como pechuga fresca), las
# variantes procesadas siguen prohibidas (no contienen "pavo" como palabra
# completa coincidente, sino como sustring en "jamón de pavo"); el check
# `'jamón de pavo' not in pool_lower` solo permite la variante procesada
# cuando el planner la asigna LITERALMENTE así.
#
# `pavo molido` también añadido (variante intermedia: fresca pero altamente
# procesada en muchas marcas, vale la pena gating explícito).
_RESTRICTED_PROTEIN_KEYS = {
    'atún':            'Atún / atún enlatado',
    'atun':            'Atún / atún enlatado',
    'salami':          'Salami dominicano',
    'longaniza':       'Longaniza',
    'chorizo':         'Chorizo',
    # [P3-PROTEIN-CAP] Variantes de pavo procesado:
    'jamón de pavo':   'Jamón de pavo / pavo en lonjas (procesado, alto en sodio)',
    'jamon de pavo':   'Jamón de pavo / pavo en lonjas (procesado, alto en sodio)',
    'pavo en lonjas':  'Jamón de pavo / pavo en lonjas (procesado, alto en sodio)',
    'lonjas de pavo':  'Jamón de pavo / pavo en lonjas (procesado, alto en sodio)',
    'pavo procesado':  'Jamón de pavo / pavo en lonjas (procesado, alto en sodio)',
    'pavo molido':     'Pavo molido (usar SOLO si el planner lo asignó explícitamente)',
    # [PROTEIN-RESPECT 2026-05-07] Carnes frescas mayores. Antes del fix, el
    # LLM ignoraba la elección anti-mode-collapse del planner y metía cerdo/
    # pollo/res en TODOS los días aunque el pool dijera otra cosa (Lentejas/
    # Yogurt/Habichuelas). Ej. observado en plan e5274d48: planner eligió
    # plant proteins, LLM emitió cerdo en los 3 días + res en 3 comidas
    # del mismo día. Las claves de abajo entran al `prohibited_block`
    # cuando NO están en el pool del día → el LLM ve "PROHIBIDO ABSOLUTO
    # cerdo/pollo/res" y respeta la asignación del planner.
    # Substring match con palabra-completa (boundary) para evitar falsos
    # positivos como 'res' dentro de 'pescado fresco'.
    'cerdo':           'Cerdo / lomo de cerdo / chuleta',
    'pollo':           'Pollo / pechuga de pollo / muslo de pollo',
    'pescado':         'Pescado / filete de pescado / tilapia / mero',
    'res':             'Carne de res / bistec / res molida',
    'pavo':            'Pavo (no en catálogo — usa otra proteína fresca)',  # [P3-PECHUGA-PAVO-REMOVE] pechuga fresca eliminada
    # [PROTEIN-SYNONYMS 2026-05-07] El LLM evade 'res' usando 'bistec'
    # como sinónimo (caso real plan 8601a2da: critique detectó 'res en
    # 3 comidas' pero mi recipe-scan no disparó porque la palabra literal
    # era 'bistec'). Cierro el gap para los sinónimos más comunes en RD:
    'bistec':          'Bistec (corte de res)',
    'carne molida':    'Carne molida (res / pavo / pollo molido)',
    # 'lomo' standalone es ambiguo (lomo de cerdo legítimo cuando pool
    # tiene cerdo, lomo de res cuando pool tiene res). Lo dejamos fuera
    # del set para evitar falsos positivos — los caps específicos de
    # 'cerdo'/'res' lo capturan vía substring de pool.
    # 'filete' standalone también ambiguo (filete de pescado vs res).
    # Ambos ya cubiertos por 'pescado' / 'res' substring.
    # [CAMARONES-LEAK 2026-05-07] Plan 089e541c: pool elegido era
    # [Queso Blanco, Gandules, Atún], pero la lista final incluyó
    # "Camarones 1 lb". Causa: el LLM (probablemente en surgical regen
    # post-aprobación) lo metió como complemento. 'camarones' no estaba
    # en este set → no se prohibió en el prompt ni se removió del cleanup.
    # Si el pool tiene 'camarones' explícitamente, el substring match lo
    # libera correctamente.
    'camarones':       'Camarones (mariscos)',
}


# [P0-PROTEIN-POOL-IMPLICATIONS · 2026-05-16] Mapping de "expansión natural"
# del LLM: cuando una key específica está en el pool autorizado, las keys
# del MISMO grupo proteico deben auto-autorizarse para que el matcher no
# las penalice como sub-palabras.
#
# Bug original (plan aeb25e1c, día 2):
#   Pool = ['Chuleta', 'Claras de Huevo', 'Queso Blanco Fresco']
#   LLM elabora ingrediente: "300 g de chuleta de cerdo (lomo, sin grasa visible)"
#   Scrub remueve porque "cerdo" no está en pool → receta queda sin proteína →
#   lista de compras NO incluye chuleta → usuario percibe app rota.
#
# Causa: "chuleta" en RD implica casi siempre "chuleta de cerdo". El propio
# label de la key 'cerdo' lo dice: "Cerdo / lomo de cerdo / chuleta". Eran
# tratadas como independientes en el matcher.
#
# Mapping: pool_key (lowercase substring) → set de restricted_keys que
# quedan auto-autorizadas cuando esa pool_key aparece.
_POOL_IMPLICATIONS = {
    # Cortes de cerdo (chuleta, lomo) implican que 'cerdo' es legítimo en la receta.
    'chuleta':         {'cerdo'},
    'lomo de cerdo':   {'cerdo'},
    'tocineta':        {'cerdo'},
    # Cortes de pollo. 'pollo' está en restricted, pero pool puede traer
    # 'Pechuga de pollo' o 'Muslo de pollo' del planner, que naturalmente
    # se expanden a "pechuga de pollo a la plancha".
    'pechuga de pollo': {'pollo'},
    'muslo de pollo':  {'pollo'},
    # Cortes de res. 'bistec' es restricted en sí mismo PERO si pool tiene
    # 'Bistec' explícitamente, el LLM puede escribir "bistec de res" y se
    # auto-penalizaría con 'res'.
    'bistec':          {'res'},
    'lomo de res':     {'res'},
    'res molida':      {'res'},
    'carne molida':    {'res'},  # ambiguo en general, autoritativo si pool lo trae
    # Pescados específicos implican 'pescado' (categoría general).
    'tilapia':         {'pescado'},
    'mero':            {'pescado'},
    'salmón':          {'pescado'},
    'salmon':          {'pescado'},
    'sardinas':        {'pescado'},
    'pescado fresco':  {'pescado'},
}


def build_day_assignment_context(skeleton_day: dict, day_num: int, day_name: str = None) -> str:
    """Genera el bloque de contexto con la asignación del planificador para un día."""
    import re as _re
    pool_str = ', '.join(skeleton_day.get('protein_pool', []))
    pool_lower = pool_str.lower()

    # [P3-PROTEIN-CAP] Normalización ASCII para tolerar variantes de acento
    # entre keys del set (`jamón`/`jamon`) y el pool del planner. Sin esto,
    # si el planner asignó "Jamón de pavo" (con tilde), el key 'jamon de pavo'
    # (sin tilde) reportaba el label como prohibido aunque la variante con
    # tilde lo había marcado como allowed.
    try:
        from constants import strip_accents as _strip_acc
    except Exception:
        def _strip_acc(s):
            return s
    pool_lower_ascii = _strip_acc(pool_lower)

    # [PROTEIN-RESPECT 2026-05-07] Match con WORD-BOUNDARY (`\b`) en vez de
    # substring puro. Razón: añadimos keys cortos (cerdo/pollo/res/pescado/
    # pavo) para gateing de carnes frescas; substring naive marcaba 'res'
    # dentro de 'pescado fresco' como allowed (falso positivo) cuando el
    # planner eligió 'Pescado fresco' en el pool. Word boundary garantiza
    # que 'res' solo matchee como palabra independiente ('carne de res',
    # 'res molida', 'res guisada') y NO embebida en otras palabras.
    def _key_in_pool(key: str, pool: str) -> bool:
        # Para keys multi-palabra (con espacios), substring funciona bien.
        # Para keys de una palabra, usar word boundary.
        if ' ' in key:
            return key in pool
        return bool(_re.search(rf'\b{_re.escape(key)}\b', pool))

    # Dos pasos: primero colectar labels EXPLÍCITAMENTE allowed (cualquier
    # variante del key está en el pool), luego añadir prohibited solo si su
    # label no está en allowed.
    allowed_labels = set()
    for key, label in _RESTRICTED_PROTEIN_KEYS.items():
        key_ascii = _strip_acc(key)
        if _key_in_pool(key_ascii, pool_lower_ascii) or _key_in_pool(key, pool_lower):
            allowed_labels.add(label)

    seen_labels = set()
    prohibited_labels = []
    for key, label in _RESTRICTED_PROTEIN_KEYS.items():
        if label in allowed_labels:
            continue
        if label in seen_labels:
            continue
        prohibited_labels.append(label)
        seen_labels.add(label)

    prohibited_block = ""
    if prohibited_labels:
        prohibited_block = (
            f"\n⛔ PROHIBIDO ABSOLUTO EN ESTE DÍA — estas proteínas NO están en tu pool y NO debes usarlas "
            f"en NINGUNA comida (ni meriendas, ni complementos, ni trazas):\n"
            f"   → {', '.join(prohibited_labels)}\n"
            f"   ⚠️ El planificador eligió DELIBERADAMENTE las proteínas del pool para garantizar variedad "
            f"entre los días del plan. Si añades una carne distinta como 'complemento' (ej: cerdo en una "
            f"merienda cuando el pool dice Lentejas, o res en un desayuno cuando el pool dice Pollo), "
            f"el self-critique lo flageará como repetición de proteína intra-día y forzará un retry "
            f"costoso (~120s) que no mejora el plan. RESPETA el pool — usa SOLO esas proteínas como "
            f"principal del día. Para diversificar desayuno/merienda usa: huevos, claras, queso fresco, "
            f"yogurt, frutos secos, mantequilla de maní (estas son OK siempre, no cuentan como 'otra carne')."
        )

    day_name_block = f"\n• Día de la Semana: {day_name}\n  (💡 INSTRUCCIÓN: Adapta el estilo y practicidad de las recetas a este día según la cultura dominicana. Ej: Fines de semana permiten platos más tradicionales o relajados; días de semana requieren mayor practicidad)." if day_name else ""

    breakfast_cat = skeleton_day.get('breakfast_category', '')
    breakfast_block = f"\n• 🍳 CATEGORÍA DE DESAYUNO ASIGNADA: {breakfast_cat}\n  (⚠️ OBLIGATORIO: El desayuno de este día DEBE ser de esta categoría. NO uses mangú/tubérculos si la categoría asignada es otra)." if breakfast_cat else ""

    return f"""
--- 📋 ASIGNACIÓN DEL PLANIFICADOR PARA OPCIÓN {day_num} ---
• Concepto Temático: {skeleton_day.get('brief_concept', 'Día variado')}{day_name_block}{breakfast_block}
• Técnica de Cocción Principal: {skeleton_day.get('assigned_technique', 'Libre')}
• Proteínas Asignadas: {pool_str}
• Carbohidratos Asignados: {', '.join(skeleton_day.get('carb_pool', []))}
• Frutas Asignadas: {', '.join(skeleton_day.get('fruit_pool', []))}
• Comidas a Generar: {', '.join(skeleton_day.get('meal_types', ['Desayuno', 'Almuerzo', 'Merienda', 'Cena']))}
{prohibited_block}
DEBES basar tus recetas en estos ingredientes asignados para garantizar
variedad entre los 3 días del plan. Puedes agregar condimentos, especias,
vegetales complementarios y líquidos (aceite, leche, etc).
---------------------------------------------------------
"""
