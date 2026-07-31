# prompts/day_generator.py
"""
Prompt para los workers paralelos del pipeline Map-Reduce.
Cada worker genera UN SOLO DÍA completo del plan (con recetas, ingredientes, macros).
Recibe la asignación del Planificador (pools de ingredientes y técnica de cocción).
"""

DAY_GENERATOR_SYSTEM_PROMPT = """
Eres un Nutricionista Clínico, Chef Profesional y la IA oficial de Bioboros.
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
2.5. TRANSFORMA LOS STAPLES EN PLATOS CRIOLLOS APETECIBLES [P2-CREATIVITY-TRANSFORM · 2026-06-29]: NO sirvas el
   staple "crudo/simple" por defecto (ni "proteína a la plancha + arroz blanco" en cada comida). Conviértelo en una
   preparación criolla apetecible, manteniendo CADA componente desglosado en `ingredients` (para que la lista de
   compras lo costee). Ejemplos por staple: harina → panqueques / bollos / arepas / tortillas / empanadas al horno;
   avena → panqueques de avena / overnight oats / avena cremosa; yuca → bollos de yuca / arepitas / casabe / yuca al
   mojo; plátano → mofongo / mangú / tostones; maíz → arepitas / chacá; huevo → tortilla / revoltillo. Aplica
   ESPECIALMENTE a MERIENDA y CENA (no solo al desayuno). La creatividad es en la PREPARACIÓN, NUNCA en inventar
   alimentos fuera del catálogo verificado (regla 5 manda). Mantén la coherencia receta↔ingredientes (regla 8).
   APETECIBILIDAD [P1-DISH-PALATABILITY · 2026-06-30]: la combinación debe ser apetecible para el paladar dominicano,
   NO un disparate. La avena/staples dulces van en preparación DULCE (panqueques/overnight/cremosa), NUNCA en un
   "salteado salado" raro (avena con guisantes y soya = disparate). NO pegues una proteína que no encaje con el plato
   (sardinas/atún en lata dentro de un revoltillo de huevo; marisco en un plato dulce). Si la comida es ligera y ya
   tiene proteína coherente (huevo, queso), NO le añadas una 2ª proteína incongruente solo para subir gramos.
3. RECETAS PROFESIONALES (LISTAS PARA COCINAR) [P2-RECIPE-STEP-CONTRACT · 2026-06-29]: Los pasos (`recipe`) DEBEN incluir los 3 prefijos EN ORDEN, y cada paso debe ser SUSTANTIVO (PROHIBIDO "Cocinar"/"Servir"/"Mezclar todo" a secas):
   - "Mise en place: [corte, lavado y MEDIDA de cada ingrediente]"
   - "El Toque de Fuego: [cocción con AL MENOS un TIEMPO concreto en minutos Y/O una temperatura o nivel de fuego — ej. 'fuego medio 8-10 min', 'horno 180°C 20 min'; nombra el ingrediente principal en su paso]"
   - "Montaje: [presentación]"
   - EXCEPCIÓN SIN COCCIÓN [P1-RECIPE-CONTRACT-NOCOOK · 2026-07-05]: si el plato NO lleva cocción alguna (fruta fresca con yogurt/cottage, ensalada fría con todo listo, overnight oats, parfait), OMITE "El Toque de Fuego" y emite solo "Mise en place" y "Montaje" — JAMÁS inventes un tiempo de fuego falso para un plato frío. Si CUALQUIER ingrediente requiere fuego (tostar, hervir un huevo, dorar), el plato SÍ lleva los 3 prefijos completos.
   - TÉCNICA CORRECTA POR ALIMENTO [P1-CASABE-NO-BOIL · 2026-07-30]: el CASABE es una torta seca de yuca YA COCIDA — se sirve tal cual, se tuesta o se calienta en sartén/horno 1-2 min; JAMÁS se hierve, se cocina en agua ni "se deja reposar tapado" como si fuera arroz (un plan real instruyó "Cocina Casabe en 1½ tazas de agua con sal, tapa y hierve 15 minutos" — eso arruina el plato). Lo mismo aplica a pan, tostadas, galletas y tortillas ya horneadas: NUNCA les apliques la plantilla de cocción de granos (proporción agua:grano, hervir, reposar). Esa plantilla es SOLO para arroz, bulgur, quinoa, avena y granos crudos.
4. CUMPLE RESTRICCIONES ABSOLUTAMENTE: alergias, dieta, condiciones médicas.
5. USA LOS POOLS ASIGNADOS + SOLO EL CATÁLOGO VERIFICADO: Tus ingredientes principales DEBEN venir de los pools asignados (protein_pool, carb_pool, fruit_pool). Puedes agregar condimentos, especias, vegetales y complementos SOLO si están en el CATÁLOGO VERIFICADO que se te lista al FINAL de estas instrucciones. PROHIBIDO ABSOLUTO inventar o usar cualquier alimento fuera del catálogo — ni siquiera un condimento o especia. Si una receta tradicional pide algo que no está en el catálogo (ej. achiote, sazón en polvo, clavo dulce, pimienta de olor, SALSA DE SOYA, salsa inglesa/Worcestershire, salsa de pescado, teriyaki, BBQ, mostaza, miel si no está listada), OMÍTELO y usa solo los sazonadores verificados. OFENSORES MEDIDOS EN PRODUCCIÓN [P1-OFFCATALOG-TOP-OFFENDERS · 2026-07-30] — estos 4 se usan una y otra vez y NINGUNO está en el catálogo, así que el usuario cocina una receta que los pide y su lista de compras no los trae: ROMERO (usa orégano o tomillo del catálogo), MENTA (usa cilantro, o omite la hoja decorativa), VINO DE JEREZ / cualquier vino o licor de cocina (usa vinagre blanco del catálogo o omítelo), TORTILLAS DE MAÍZ (usa tortilla de trigo o integral, que SÍ están). SAZONA CON SABOR [P1-SPICES-CATALOG-SYNC · 2026-07-01]: comino, cúrcuma, laurel, tomillo, curry y cebolla en polvo SÍ son verificados (además de sal, ajo, cebolla, orégano, cilantro, perejil) — úsalos activamente para dar sabor criollo real a guisos, locrios y habichuelas cuando aparezcan en el catálogo listado. CRÍTICO [P1-RECIPE-OFFCATALOG-CONDIMENT · 2026-06-30]: si MENCIONAS en los PASOS un condimento fuera del catálogo, el sistema lo BORRA de la lista de compras pero NO de tu texto → el usuario lee "añade salsa de soya" pero nunca la compró (receta ROTA). Por eso: NUNCA nombres en los pasos un alimento que no esté en `ingredients` y en el catálogo.
6. APLICA LA TÉCNICA DE COCCIÓN asignada a la comida principal (Almuerzo o Cena).
7. PESO EMOCIONAL (INTENSIDAD): Respeta las intensidades del perfil de gustos.
8. ESTRUCTURA DE INGREDIENTES Y MEDIDAS CASERAS DOMINICANAS:
   - PREFIERE usar medidas caseras dominicanas siempre que sea posible (ej: "½ plátano verde", "1 taza de arroz", "2 lonjas de queso", "1 pechuga de pollo", "1 cda de aceite").
   - Si el ingrediente no se presta para medidas caseras, usa unidades métricas (g, oz, lb, ml).
   - PROHIBIDO ABSOLUTO: "pizcas", "ramitas", "chorritos" u otras medidas imprecisas.
   - **UNIDADES ENTERAS EN DISCRETOS [P3-HUMAN-WHOLE-DISCRETE · 2026-06-28]**: los alimentos que se cuentan por pieza
     INDIVISIBLE van en ENTEROS, NUNCA en fracciones raras: "2 huevos"/"3 huevos" (NUNCA "2.5 huevos" ni "0.5 huevo"),
     "1 rebanada de pan" (NUNCA "0.5 rebanada"), "1 tostada", "1 galleta". Lo que SÍ se parte (½ aguacate, ½ batata,
     ½ plátano, ½ cebolla, ½ tomate, ½ guineo) puede ir en mitades. Hojas (lechuga/espinaca): tazas redondas (1, 2, 3),
     nunca tercios ("3.33 tazas"). Para ajustar proteína/calorías cambia los GRAMOS de la proteína/almidón, no partas un huevo.
   - **PROTEÍNA INTEGRADA, NO DE RELLENO [P3-PROTEIN-INTEGRATED · 2026-06-28]**: la proteína principal debe ser
     COHERENTE con el plato e INTEGRADA en la preparación, no pegada como extra de última hora. PROHIBIDO meter
     proteína SALADA (camarón, chivo, pescado, carne) en un plato DULCE (yogurt con fruta, avena, batido): ahí la
     proteína va de yogurt griego, queso fresco/ricotta o frutos secos. Emite porciones de proteína REALES
     (≥40-60g de carne/pescado por comida principal), nunca "10g de camarones" de adorno.
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
   - HUEVOS: ENTEROS PRIMERO [P1-WHOLE-EGGS-FIRST · 2026-07-30]: un plato de SOLO claras obliga al usuario a separar huevos y quedarse con las yemas en la mano (desperdicio real en cocina dominicana). Usa huevos ENTEROS por defecto; recurre a claras SOLO para el excedente de proteína que la grasa del día ya no permite como huevo entero, y en ese caso PREFIERE la mezcla ("2 huevos + 2 claras") antes que claras puras ("4 claras"). Claras puras solo si la banda de grasa de la comida no admite NI UNA yema.
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
         • Fruta + mantequilla de maní/almendras (manzana con pb, guineo con pb) — SOLO FRUTA.
           ⛔ [P1-SLOT-MERIENDA-CRUDITES · 2026-07-26] NO generalices esto a VEGETALES: apio relleno
           de mantequilla de maní, bastones de zanahoria con crema, brócoli al vapor con dip de
           yogurt y demás crudités son merienda de dieta AMERICANA, no dominicana. Aquí un vegetal
           crudo NUNCA es el vehículo de una crema o un dip. El gate determinista los rechaza.
         • Pinchitos sencillos (pollo/queso) + fruta
         • Huevo duro + fruta + nueces
         • Avena overnight / chia pudding pequeño
         • Tostada de aguacate con huevo
       Ejemplos PROHIBIDOS: "Salteado de lentejas", "Locrio de…", "Pechuga al grill con puré", "Croquetas horneadas con guarnición", cualquier cosa que parezca un mini-almuerzo.

    d) CENA — más ligera que el almuerzo. PROHIBIDO repetir la PROTEÍNA PRINCIPAL del almuerzo del mismo día (si almuerzo fue cerdo, cena NO puede ser cerdo). PROHIBIDO repetir el CARBOHIDRATO PRINCIPAL del almuerzo del mismo día (si almuerzo fue plátano, cena NO puede ser plátano). PROHIBIDO el "ARROZ DE NOCHE": NADA de arroz blanco/integral, locrio, moro, asopao NI platos cuya BASE sea arroz aunque el nombre no diga "arroz" (chofán/arroz frito, paella, risotto, congrí, mamposteao) en la cena (no se acostumbra en la cena dominicana y el gate lo rechaza). Rota a otro carbo de cena: batata, yuca, ñame, casabe o pan integral (NUNCA arroz).

       ⛔ REGLA AMPLIA DE VARIEDAD (P1-VARIETY-SAME-DAY-PROTEIN · 2026-06-27): la MISMA proteína principal —INCLUIDO EL HUEVO— NO debe aparecer en 2+ comidas del MISMO día, en NINGUNA combinación de slots (ni desayuno+cena, ni desayuno+merienda, ni almuerzo+cena). Ejemplo PROHIBIDO real: desayuno "Batido con claras de huevo" + cena "Tortilla de huevos" (huevo 2 veces el mismo día). Si el desayuno lleva huevo, las demás comidas del día usan OTRA proteína (pollo, res, cerdo, pescado, atún, queso, yogur, legumbres). ✅ SÍ está permitido repetir un alimento en DÍAS DISTINTOS (huevo el lunes y el miércoles) — lo que fatiga es comerlo dos veces el MISMO día. Patrones válidos:
       • Pescado/pollo a la plancha + ensalada + tubérculo distinto al del almuerzo
       • Tortilla/revoltillo de cena con vegetales + casabe o pan integral
       • Sopa ligera de pollo/vegetales con proteína magra
       • Wrap/pita con proteína + vegetales
       • Bowl de proteína magra + vegetales asados + 1 carbo
       Evita frituras pesadas, locrios densos y guisos calóricos en la noche.

    d-bis) ⛔ NO DUPLIQUES EL CARBOHIDRATO [P1-NIGHT-RICE-MIN-G · 2026-07-26]: si el plato YA tiene su
       base de carbohidrato (tortilla/wrap, pan, casabe, ñame, yuca, batata, plátano, avena, papa, pasta,
       o la fruta de una merienda), NO le añadas además una porción de arroz para cuadrar los carbos.
       Casos reales rechazados: "Wrap Fresco de Atún" con 2 tortillas integrales + 35 g de arroz;
       "Tortilla de Ñame Rallado" + 25 g de arroz; "Pera Asada con Mantequilla de Maní" + 20 g de arroz.
       Nadie come eso. Si faltan carbos, AGRANDA la base que el plato ya tiene — no apiles una segunda.
       (Es la regla simétrica de "no pegar una segunda proteína principal a un plato que ya la tiene.)

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

# [P2-SLOT-SSOT-PROMPT · 2026-07-02] (audit v3 slots GAP-F) Bloque de reglas de horario DERIVADO del SSOT
# del validador (constants.SLOT_INAPPROPRIATE_FOODS + SLOT_POSITIVE_HINT vía build_meal_timing_rules) y
# APPENDEADO al system prompt a IMPORT-time. Antes el §15 hardcodeaba su propia copia de las reglas: el
# prompt prohibía MÁS de lo que el validador enforzaba (causa raíz de GAPs B/C/D) y endurecer el validador
# NO propagaba al prompt de form-gen (los prompts de UPDATE sí derivan del SSOT). Import-time = string
# estático por deploy → el prompt-cache del SystemMessage queda INTACTO (P1-PROMPT-CACHE). La prosa
# creativa del §15 se mantiene; este bloque añade el contrato EXACTO que el gate rechaza. Fail-safe.
# tooltip-anchor: P2-SLOT-SSOT-PROMPT
try:
    from constants import build_meal_timing_rules as _bmtr_ssot
    _SLOT_SSOT_RULES_BLOCK = "\n".join(
        _b for _b in (_bmtr_ssot(_s) for _s in ("Desayuno", "Almuerzo", "Cena", "Merienda")) if _b
    ).strip()
except Exception:
    _SLOT_SSOT_RULES_BLOCK = ""
if _SLOT_SSOT_RULES_BLOCK:
    DAY_GENERATOR_SYSTEM_PROMPT = DAY_GENERATOR_SYSTEM_PROMPT + (
        "\n16. CONTRATO EXACTO DEL VALIDADOR DE HORARIO (derivado del SSOT — el gate rechaza "
        "exactamente esto, sin excepciones):\n    " + _SLOT_SSOT_RULES_BLOCK + "\n"
    )

# [P1-PRECISION-LEVERS · 2026-07-04] (lever 1) Presupuesto CUANTITATIVO de sodio en el system
# prompt. Evidencia en vivo 2026-07-04: un intento salió con 4,261 mg/día (el gate de sodio lo
# rechazó → retry completo pagado) porque el prompt solo decía "modera la sal" sin números ni
# reglas de conteo. String ESTÁTICO a import-time → prompt-cache del SystemMessage intacto
# (P1-PROMPT-CACHE). El gate MEALFIT_SODIUM_EXCESS_GATE queda como backstop.
DAY_GENERATOR_SYSTEM_PROMPT = DAY_GENERATOR_SYSTEM_PROMPT + (
    "\n17. PRESUPUESTO DE SODIO (el panel lo MIDE y el validador RECHAZA el exceso flagrante):\n"
    "    - Techo diario: ≤2000 mg de sodio (OMS). El techo aplica a CADA DÍA por separado — un día\n"
    "      salado NO se compensa con otro día bajo (el usuario come días, no promedios).\n"
    "    - Máximo UN ítem enlatado en TODO el día (atún, sardinas, maíz o habichuela de lata): cada\n"
    "      lata carga 400-700 mg. Dos enlatados en un día casi garantizan pasarse.\n"
    "    - Queso (blanco/de freír) y embutidos son cargas grandes de sodio: si un plato los lleva,\n"
    "      NO añadas sal en los pasos de ese plato y no los combines con enlatado en el mismo día.\n"
    "    - Sazona con ajo, cebolla, orégano, limón, cilantro y ají — NO con cubitos ni sazón completo\n"
    "      (un cubito ≈ 1000 mg de sodio: revienta el presupuesto él solo).\n"
    "    - Los pasos de receta solo dicen 'sal' UNA vez por plato como máximo ('pizca de sal').\n"
    # [P1-SALT-LINE-AUTOFIX · 2026-07-05] medido en vivo: '1 cdta de sal' como INGREDIENTE
    # = 2,358 mg de sodio — una sola línea revienta el techo del día completo.
    "    - Si listas sal en `ingredients`, escribe SIEMPRE 'Sal al gusto' — JAMÁS cantidades\n"
    "      ('1 cdta de sal' = 2,358 mg de sodio: más que el techo del día entero en una línea).\n"
)

# [P1-DAYGEN-FATS-BUDGET · 2026-07-05] (espejo del §17 de sodio) Presupuesto CUANTITATIVO de
# GRASAS per-día. Medido en vivo (corridas 2026-07-04/05): días con grasas 141-166% del target
# saturan el clamp del solver [0.3, 3.5] y el del reconcile [0.4, 1.8] — con el clamp saturado,
# NINGÚN corrector determinista aguas abajo alcanza la banda [0.90, 1.12] → banner
# low_band_macro:fats en el plan entregado. El target numérico del día viaja en el tramo
# dinámico (nutrition_context); esta sección enseña las REGLAS DE CONTEO que el LLM no conocía.
# String ESTÁTICO a import-time → prompt-cache del SystemMessage intacto (P1-PROMPT-CACHE).
DAY_GENERATOR_SYSTEM_PROMPT = DAY_GENERATOR_SYSTEM_PROMPT + (
    "\n18. PRESUPUESTO DE GRASAS (el motor MIDE la banda ±10% POR DÍA; un exceso profundo NO es corregible):\n"
    "    - Tu target de grasas del día viene en los macros objetivo del contexto. Respétalo POR DÍA —\n"
    "      un día grasoso NO se compensa con otro día magro (el usuario come días, no promedios).\n"
    "    - Máximo DOS fuentes grasas DENSAS en TODO el día entre: aguacate, aceite (si pasa de 1 cda),\n"
    "      mantequilla de maní, frutos secos/semillas (>15g), queso de freír/amarillo, coco/leche de coco.\n"
    "      La tercera fuente densa casi garantiza reventar la banda del día.\n"
    "    - Aceite de cocinar: máximo 1 cda (15 ml) por plato — y CUENTA como grasa del día, no es 'gratis'.\n"
    "    - Cuenta las grasas OCULTAS de la proteína: salmón, res molida 80/20, muslo/pollo con piel,\n"
    "      huevo (~5g c/u), leche entera. Si el plato lleva proteína grasa, NO le añadas además\n"
    "      aguacate NI frutos secos NI queso — elige UNA grasa protagonista por plato.\n"
    "    - Si el día va cargado de grasa, prefiere métodos magros: plancha/horno/vapor sin aceite extra\n"
    "      y proteínas magras (pechuga, pescado blanco, claras, yogurt descremado).\n"
)

# [P1-DAYGEN-TRANSFORM-NUDGE · 2026-07-09] (forense plan f19d55a6: intento 1 rechazado HIGH por el gate
# TRANSFORM_SOFT_GATE — transform_meals=0 → "El plan no incluye NINGUNA preparación transformada" → retry
# COMPLETO pagado). El prompt PEDÍA creatividad pero no exigía un MÍNIMO de preparaciones transformadas; el
# LLM emitía "proteína a la plancha + carbo hervido + vegetal suelto" (staples servidos) y el gate reintentaba.
# Esta sección enseña la REGLA DE CONTEO (≥1 preparación transformada real) que cierra el gate desde el
# intento 1. Espejo del §17/§18. String ESTÁTICO a import-time → prompt-cache del SystemMessage intacto
# (P1-PROMPT-CACHE). tooltip-anchor: P1-DAYGEN-TRANSFORM-NUDGE
DAY_GENERATOR_SYSTEM_PROMPT = DAY_GENERATOR_SYSTEM_PROMPT + (
    "\n19. PREPARACIONES TRANSFORMADAS (el validador RECHAZA un plan de puros staples servidos):\n"
    "    - Un plato 'transformado' es una PREPARACIÓN dominicana real donde los ingredientes se integran:\n"
    "      guisos, locrios (almuerzo), panqueques/arepitas con las harinas, bollitos/buñuelos de yuca o\n"
    "      víveres, revoltillos, tortitas/croquetas al horno, mangú, ensaladas COMPUESTAS. NO cuenta:\n"
    "      proteína a la plancha + carbo hervido + vegetal crudo suelto servidos por separado (eso es un\n"
    "      'staple servido' y el validador lo rechaza si el día NO trae ninguna preparación transformada).\n"
    "    - Incluye AL MENOS una preparación transformada por día — idealmente que la comida principal lo sea.\n"
    "      Un día entero de puros staples servidos se rechaza y se regenera (pierde tiempo y calidad).\n"
    "    - Transformar es la TÉCNICA (cómo se cocina y se presenta), NO cambia los macros: mantén las mismas\n"
    "      cantidades de proteína/carbohidrato/grasa del plato.\n"
)


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


def build_slot_targets_block(daily_targets: dict, meal_types: list) -> str:
    """[P3-DAYGEN-SLOT-TARGETS · 2026-07-29] (audit solver+seeder v4) Una línea por slot con su cuota
    de kcal/P/C/F, derivada del SSOT `allocate_macros_per_slot` (el mismo que el swap ya consume).

    Por qué: el day-gen recibía kcal/P/C/F del DÍA y la orden «la suma DEBE coincidir con el objetivo
    diario», pero NUNCA el reparto por slot contra el que el solver lo va a medir. Componía cada
    plato ciego al reparto y luego el solver lo forzaba re-escalando — o no podía, porque faltaba el
    PORTADOR del macro. Caso medido: merienda de 0.15 sobre 68 g de grasa = 10.2 g; el LLM compone
    'Yogurt Griego con Guineo y Avena Tostada' (~1.7 g de grasa) y el máximo alcanzable escalando
    todo al tope es 6.0 g. **13 de las 16 infactibilidades por-coordenada medidas eran de grasa**, de
    ahí la regla explícita del portador.

    Nace OFF: toca el prompt del nodo MÁS CARO del pipeline (62.6% del costo), así que exige canario.
    Devuelve "" si no hay datos suficientes (fail-open: el prompt queda como hoy).
    tooltip-anchor: P3-DAYGEN-SLOT-TARGETS"""
    try:
        from nutrition_calculator import allocate_macros_per_slot
        _mt = [m for m in (meal_types or []) if m]
        if not _mt or not isinstance(daily_targets, dict):
            return ""
        _slots = allocate_macros_per_slot(daily_targets, len(_mt)) or {}
        if not _slots:
            return ""
        # [P3-SLOT-TARGETS-BY-NAME · 2026-07-31] (audit solver+seeder v6 · F24) Antes esto pareaba
        # `_slots.items()` con `_mt[_i]` por ÍNDICE y descartaba `_k`, que es justo la clave que sabe
        # de qué slot es la cuota. Si el esqueleto llega con los meal_types en orden no-canónico
        # (Desayuno→Merienda→Almuerzo→Cena, p.ej. con el decisor clínico de nº de comidas caído), el
        # prompt le dice al LLM "Merienda ≈ 35% del día · Almuerzo ≈ 15%" y el day-gen compone una
        # merienda de plato fuerte y un almuerzo raquítico.
        # Se parea por NOMBRE con el resolvedor que ya existe. `canonical_slot_key` devuelve
        # 'merienda' mientras el asignador produce 'merienda_am'/'merienda_pm' cuando hay 5+ comidas:
        # esa familia se resuelve por prefijo y EN ORDEN de aparición, consumiendo cada clave una
        # sola vez. Lo que no resuelve cae al índice, que es el comportamiento previo (fail-open: sin
        # cuota es peor que con una cuota aproximada). tooltip-anchor: P3-SLOT-TARGETS-BY-NAME
        try:
            from constants import canonical_slot_key as _csk
        except Exception:
            _csk = lambda _x: None  # noqa: E731

        _libres = list(_slots.keys())

        def _slot_para(_meal_type, _idx):
            _canon = _csk(_meal_type)
            if _canon:
                if _canon in _libres:
                    _libres.remove(_canon)
                    return _canon
                for _cand in _libres:
                    if _cand.startswith(_canon):      # merienda → merienda_am, luego merienda_pm
                        _libres.remove(_cand)
                        return _cand
            _todas = list(_slots.keys())
            return _todas[_idx] if _idx < len(_todas) else None

        _rows, _needs_fat = [], []
        for _i, _mt_name in enumerate(_mt):
            _k = _slot_para(_mt_name, _i)
            _v = _slots.get(_k) if _k else None
            if not isinstance(_v, dict):
                continue
            _rows.append(f"    · {_mt_name} ≈ {round(_v.get('kcal') or 0)} kcal · "
                         f"{round(_v.get('protein') or 0)} g P · {round(_v.get('carbs') or 0)} g C · "
                         f"{round(_v.get('fats') or 0)} g G")
            if float(_v.get("fats") or 0) >= 5.0:
                _needs_fat.append(_mt_name)
        if not _rows:
            return ""
        _fat_rule = ""
        if _needs_fat:
            _fat_rule = (
                f"\n  ⚠️ PORTADOR DE GRASA OBLIGATORIO en: {', '.join(_needs_fat)} — su cuota supera "
                f"los 5 g y la grasa NO se puede fabricar re-escalando lo que no la tiene. Incluye una "
                f"fuente real (aceite, aguacate, frutos secos, mantequilla de maní, queso). Un yogurt "
                f"con fruta y avena NO llega ni escalándolo al máximo.")
        return ("\n• 🎯 CUOTA POR COMIDA (el motor mide cada plato contra ESTO, no solo el total del día):\n"
                + "\n".join(_rows) + _fat_rule)
    except Exception:
        return ""


def build_day_assignment_context(skeleton_day: dict, day_num: int, day_name: str = None,
                                 daily_targets: dict = None) -> str:
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

    # [P1-PRECISION-LEVERS · 2026-07-04] (lever 2) Anti-repetición ENTRE DÍAS: los días se generan
    # en PARALELO (asyncio.gather) y no se ven entre sí — el "salteado ×3" / "revoltillo ×3" solo lo
    # frenaba el retry del gate de variedad (tokens pagados). El dispatch inyecta en cada
    # skeleton_day un brief DETERMINISTA de los otros días (`_other_days_brief`: técnica asignada +
    # categoría de desayuno) y aquí lo convertimos en instrucción negativa explícita. Determinista
    # por skeleton → prompt-cache per-día estable. Fail-open → ''.
    cross_day_block = ""
    try:
        _others = []
        for _od in (skeleton_day.get("_other_days_brief") or []):
            _t = str((_od or {}).get("technique") or "").strip()
            _b = str((_od or {}).get("breakfast") or "").strip()
            if _t or _b:
                _others.append((_t or "libre") + (f" (desayuno: {_b})" if _b else ""))
        if _others:
            cross_day_block = (
                f"\n• 🔀 ANTI-REPETICIÓN ENTRE DÍAS — los OTROS días de este plan ya usan: "
                f"{'; '.join(_others)}.\n"
                f"  ⚠️ Tu día debe DISTINGUIRSE: tu técnica asignada es la identidad de tu plato fuerte. "
                f"NO produzcas el mismo plato-base que esos días van a producir (si otro día es "
                f"'salteado', el tuyo NO es otro salteado; si otro desayuno es revoltillo/huevo, el "
                f"tuyo usa una base distinta DE TU categoría asignada). El validador rechaza el mismo "
                f"plato-base en 3+ días — y ese retry es evitable respetando esta línea."
            )
    except Exception:
        cross_day_block = ""

    # [P1-NEXT-LEVEL-BATCH · 2026-07-02] (LIBRARY) Inspiración curada por slot (elige-y-adapta
    # sobre un espacio verificado — creatividad por recombinación, prioriza transformadas).
    # Determinista por (día, pool) → prompt-cache preservado. Fail-open → ''.
    dish_library_block = ""
    try:
        from dish_library import build_dish_library_context
        dish_library_block = build_dish_library_context(skeleton_day, day_num) or ""
    except Exception:
        dish_library_block = ""

    # [P1-DAYGEN-DINNER-IDENTITY · 2026-07-09] (opción A del análisis de calidad) Nudge determinista para
    # que la CENA tenga IDENTIDAD PROPIA de plato fuerte y NO defaultee a tortilla/revoltillo de huevo. La
    # técnica asignada se aplica a "la comida principal" (§6) y el LLM tendía a ponerla en el almuerzo y
    # dejar la cena "libre" → tortilla de huevo las 3 noches (forense plan ae7ab047: 3 tortillas de cena →
    # rechazo cross-day-dish tolerado en el intento final). Additive + knob-gated + prompt-cache-safe
    # (string estático). El gate cross-day sigue siendo el enforcement; esto reduce que el LLM caiga ahí.
    import os as _os_dg
    _dinner_nudge_on = _os_dg.environ.get(
        "MEALFIT_DAYGEN_DINNER_IDENTITY", "true"
    ).strip().lower() not in ("false", "0", "off", "no")
    dinner_identity_block = ""
    if _dinner_nudge_on:
        dinner_identity_block = (
            "\n• ⚠️ IDENTIDAD DE LA CENA: la Cena debe ser una preparación REAL con identidad propia "
            "(guiso/estofado, al horno, a la plancha, salteado, en su técnica asignada) usando la proteína "
            "asignada. NO uses tortilla/revoltillo/omelette de huevo como cena por defecto — el huevo va en "
            "el desayuno, pero la cena necesita su propio plato fuerte. Varía la preparación de la cena "
            "respecto a las técnicas de los OTROS días indicadas arriba (no repitas la misma forma 3 noches)."
        )

    # [P1-DAYGEN-PROTEIN-DIVERSITY · 2026-07-09] Nudge additive + knob-gated para NO sobrecargar el día de
    # queso como proteína principal. Forense plan 55b659c5 (gain_muscle, renovación en vivo): 8/12 comidas
    # usaban queso (freír/cottage/crema/blanco) → sodio día 3 2410mg > techo 2000mg = ÚNICA causa del
    # _quality_degraded (micro_worst_day_ceiling) + monotonía + proteína menos magra. El queso de freír es
    # MUY salado; cottage moderado. Instruimos: queso como proteína PRINCIPAL en ≤1 comida/día, resto con
    # proteína animal magra variada. Prompt-cache-safe (string estático). NO es enforcement — el panel de
    # micros + gates de variedad siguen siendo el enforcement; esto reduce que el LLM caiga en el default.
    _protein_diversity_on = _os_dg.environ.get(
        "MEALFIT_DAYGEN_PROTEIN_DIVERSITY", "true"
    ).strip().lower() not in ("false", "0", "off", "no")
    # [P1-CARB-BASE-NO-REPEAT · 2026-07-27] El sembrador reparte DOS bases por día
    # (`ai_helpers._rotate_pairs`), pero esa asignación solo llegaba al prompt del ESQUELETO
    # (`prompts/preferences.py`). Quien escribe los ingredientes es ESTE generador, y aquí el
    # `carb_pool` se listaba pelado, sin regla alguna. Medido en el plan vivo 08114452: almuerzo
    # y cena del día 2 llevaron papa las dos veces (225 g + 74 g).
    #
    # ⚠️ Solo se emite con ≥2 bases en el pool. Pedir "no repitas" con una sola base sería una
    # restricción insatisfacible, que es exactamente como el gate de fruta acabó forzando el 67%
    # de reintentos (P1-FRUIT-SEEDER-GATE-CONTRACT · 2026-07-26).
    # Se sanea UNA vez y se usa para las dos cosas (mostrar el pool y decidir la regla). El
    # `', '.join(carb_pool)` original reventaba con un None dentro de la lista — hoy lo cubre el
    # esquema Pydantic (`List[str]`), pero este prompt es camino crítico: si lanza, NO se genera
    # ningún día. Un test de basura lo destapó al añadir la regla.
    _carbs_asignados = [str(c).strip() for c in (skeleton_day.get('carb_pool') or [])
                        if c is not None and str(c).strip()]
    carb_no_repeat_block = ""
    if len(_carbs_asignados) >= 2:
        carb_no_repeat_block = (
            f"\n• ⛔ NO REPITAS LA BASE: el Almuerzo y la Cena deben llevar bases DISTINTAS entre sí — "
            f"una con '{_carbs_asignados[0]}' y la otra con '{_carbs_asignados[1]}'. Usar la misma base "
            f"en las dos comidas fuertes del día es un fallo: produce jornadas con papa en almuerzo Y "
            f"cena. Si el desayuno o la merienda ya llevan una de las dos, tanto mejor variar."
        )

    protein_diversity_block = ""
    if _protein_diversity_on:
        protein_diversity_block = (
            "\n• ⚠️ DIVERSIDAD DE PROTEÍNA: el queso (de freír, cottage, crema, blanco) es ALTO EN SODIO — "
            "úsalo como proteína PRINCIPAL en máximo 1 comida del día, NO en varias. Para el resto de las "
            "comidas prioriza proteína animal magra y variada (pollo, pescado, res, cerdo, calamar, huevo, "
            "hígado, atún) o legumbres. Evita que 2+ comidas del mismo día dependan del queso para su "
            "proteína: aporta menos variedad y dispara el sodio del día. "
            # [P1-DAYGEN-PROTEIN-DIVERSITY-LEAN · 2026-07-09] (forense plan f19d55a6 intento 2: día con
            # grasas 174% del target → rechazo de banda; la grasa venía EMBEBIDA en la proteína, que el
            # trim determinista NO puede recortar). Complementa el §18: la diversidad NO debe pelearse con
            # el presupuesto de grasa — prefiere cortes MAGROS por defecto.
            "Al elegir la proteína PREFIERE cortes MAGROS (pechuga/muslo sin piel, pescado blanco, lomo, "
            "claras, atún en agua, pavo) sobre los grasos (salmón, res 80/20, muslo con piel, hígado): la "
            "grasa embebida en un corte graso NO se puede recortar después y revienta el presupuesto de "
            "grasa del día (§18), forzando un rechazo de banda. Usa un corte graso solo si es LA grasa "
            "protagonista del plato y no le añades otra fuente de grasa."
        )

    # [P2-VEGGIE-CHANNEL-DAYGEN · 2026-07-30] (audit solver+seeder v5) El seeder reparte
    # vegetales/grasas por día (2 distintos, pool ya filtrado por alergias/dislikes/dieta) y hasta
    # ahora esa decisión moría en el prompt del ESQUELETO: el day-gen —quien escribe los
    # ingredientes reales— generaba ciego a ella y caía en su default (la misma ensalada
    # verde/aguacate los 3 días). Replica el patrón de `carb_no_repeat_block`, con el mismo guard
    # de ≥2 ítems para no crear una restricción insatisfacible.
    _veggie_block = ""
    _vp_dg = [str(v).strip() for v in (skeleton_day.get("veggie_pool") or []) if str(v).strip()]
    if len(_vp_dg) >= 2:
        _veggie_block = (
            f"\n• Vegetales/Grasas Asignados: {', '.join(_vp_dg)} "
            f"(usa AMBOS en el día, en comidas distintas — son la variedad vegetal de esta opción)")

    # [P3-DAYGEN-SLOT-TARGETS · 2026-07-29] OFF por default: nace vacío ⇒ prompt byte-idéntico.
    _slot_targets_block = ""
    try:
        import os as _os_dg
        if str(_os_dg.environ.get("MEALFIT_DAYGEN_SLOT_TARGETS_IN_PROMPT", "false")
               ).strip().lower() in ("1", "true", "yes", "on") and daily_targets:
            _slot_targets_block = build_slot_targets_block(
                daily_targets, skeleton_day.get("meal_types") or [])
    except Exception:
        _slot_targets_block = ""

    return f"""
--- 📋 ASIGNACIÓN DEL PLANIFICADOR PARA OPCIÓN {day_num} ---
• Concepto Temático: {skeleton_day.get('brief_concept', 'Día variado')}{day_name_block}{breakfast_block}{cross_day_block}
• Técnica de Cocción Principal: {skeleton_day.get('assigned_technique', 'Libre')}
• Proteínas Asignadas: {pool_str}
• Carbohidratos Asignados: {', '.join(_carbs_asignados)}{carb_no_repeat_block}
• Frutas Asignadas: {', '.join(skeleton_day.get('fruit_pool', []))}{_veggie_block}
• Comidas a Generar: {', '.join(skeleton_day.get('meal_types', ['Desayuno', 'Almuerzo', 'Merienda', 'Cena']))}{_slot_targets_block}{dinner_identity_block}{protein_diversity_block}
{dish_library_block}{prohibited_block}
DEBES basar tus recetas en estos ingredientes asignados para garantizar
variedad entre los 3 días del plan. Puedes agregar condimentos, especias,
vegetales complementarios y líquidos (aceite, leche, etc).
---------------------------------------------------------
"""
