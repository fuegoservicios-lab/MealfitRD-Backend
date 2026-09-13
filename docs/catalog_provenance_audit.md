# Auditoría de procedencia del catálogo nutricional

**Fecha de la medición: 2026-08-19** · `master_ingredients`, 347 filas, Neon prod.

## Qué se midió y por qué

`fdc_id` es la única prueba de dónde salió cada valor nutricional. La pregunta que abrió
esta auditoría fue si esa prueba se sostiene. No se sostiene en 47 filas.

Dos hallazgos estructurales antes del detalle:

1. **47 de 347 filas comparten `fdc_id` con otra fila** (20 grupos). Una fila de USDA
   haciendo de sustituto de varios alimentos.
2. ~~**`fdc 330137` devuelve HTTP 404.** El id ya no existe en USDA.~~ **CORREGIDO el
   mismo día:** el 404 es real pero la conclusión era falsa. `330137` es de tipo
   `Foundation`, y el endpoint de *detalle* de USDA no sirve ese tipo — el *buscador* sí
   lo conoce, con la descripción y los macros correctos. Un barrido posterior de los
   **288 `fdc_id` del catálogo dio CERO ids muertos**. La lección que queda no es sobre
   los datos sino sobre la sonda: *un 404 dice que tu petición falló, no que la cosa no
   exista* — y los 7 falsos positivos eran exactamente los 7 registros `Foundation`.

## Las dos clases (no confundirlas: el arreglo es distinto)

| Clase | Grupos | Filas | Qué pasa | Arreglo |
|---|---|---|---|---|
| **COPIADO** | 13 | 32 | Los macros son idénticos: una fila de USDA es el valor literal de N alimentos | Hace falta una **fuente nueva por alimento** |
| **DIFERENCIADO** | 7 | 15 | Los valores se ajustaron a mano; lo único compartido es la etiqueta | Solo hay que **corregir o vaciar el `fdc_id`** |

> Cuidado con el veredicto automático: la primera clasificación marcó los 7 embutidos como
> «diferenciados» porque **una** fila tenía 13.63 y las otras 13.6 — puro redondeo. Son
> idénticos. El umbral correcto es una tolerancia relativa (2%), no la igualdad exacta.

## COPIADO — 13 grupos

| `fdc_id` | Fila USDA real | Alimentos | Ámbito |
|---|---|---|---|
| 173859 | *Sausage, pork, chorizo, raw* · 296 kcal | Chistorra, Chorizo español, Chorizo mexicano, Chorizo santarrosano, Chorizo verde, Longaniza puertorriqueña, **Sobrasada** | beta |
| 169396 | *Peppers, ancho, dried* · 281 kcal | Chile ancho, Chile guajillo, Chile mulato | beta |
| 167750 | 41 kcal | Tuna de nopal, Xoconostle | beta |
| 167761 | 66 kcal | Borojó, Guanábana | beta |
| 168277 | 393 kcal | Panceta ibérica, Tocineta | beta |
| 168282 | 195 kcal | Jamón ibérico, Jamón serrano | beta |
| 170851 | 151 kcal | Queso ricotta, Requesón | beta |
| 170932 | 318 kcal | Chile chipotle, Chile de árbol | beta |
| 171631 | 290 kcal | Butifarra, Salchicha italiana | beta |
| 171714 | 103 kcal | Chontaduro, Panapén | beta |
| 173443 | 136 kcal | Crema mexicana, Suero costeño | beta |
| 173944 | *Bananas, raw* · 89 kcal | Guineo, Guineo verde | **DO** |
| 330137 | *Yogurt, Greek, plain, nonfat* (`Foundation`) | Yogurt, Yogurt griego sin azúcar | **DO** |

No todos son igual de graves. **`Queso ricotta` / `Requesón` son de hecho el mismo
alimento con dos nombres** — ahí el problema no es el dato sino que existan dos filas en
vez de una fila con alias. Lo mismo con `Habichuelas blancas` / `Judías blancas` (en la
otra clase). Los chiles secos se usan en cantidades pequeñas y son parientes cercanos.

Los que sí eran materialmente falsos, medidos contra BEDCA:

| Alimento | kcal en catálogo | kcal real | |
|---|---|---|---|
| Sobrasada | 296 | **595** | el catálogo contaba la mitad |
| Lomo embuchado | 110 | **321** | ~3× · además proteína 20.3 → 34.0 |
| Chistorra | 296 | **512** | |
| Jamón serrano | 195 | **318** | grasa 8.32 → 22.6 |
| Cecina | 153 | **242** | |
| Panceta ibérica | 393 | **465** | |
| Chorizo español | 296 | **322** | proteína 13.6 → **27.0**, el doble |

## DIFERENCIADO — 7 grupos

`169108` (Chinola/Curuba/Granadilla), `169998` (Champús/Maíz dulce), `170591` (Nueces
mixtas/Piñones), `171320` (Canela/Especias para arroz con dulce), `174220`
(Mejillones/Vieira), `175179` (**Camarones/Tilapia**), `175202` (Habichuelas
blancas/Judías blancas).

Aquí el dato está bien y sobra el tag. `Camarones` y `Tilapia` son el ejemplo limpio: 85
vs 96 kcal, colesterol 161 vs 50 — alguien los ajustó a mano y solo quedó el `fdc_id`
mintiendo.

**El veredicto por grupo esconde copias dentro del grupo.** En `169108`, Chinola sí está
diferenciada (108.6 kcal) pero Curuba y Granadilla siguen idénticas entre ellas (97). En
`169998` y `171320` las kcal difieren y proteína/carbos/grasa son idénticas. La
clasificación es por grupo; leerla como «estas 15 filas están bien» sería un error.

## Lo que esta auditoría NO ve

Solo caza `fdc_id` **compartidos**. Un `fdc_id` único que apunte al alimento equivocado
es invisible para ella — y existe: `Lomo embuchado` tenía el suyo propio, apuntando a
lomo de cerdo **crudo** (110 kcal) en vez de al curado (321). Lo destapó comparar contra
BEDCA, no el barrido de duplicados.

Un barrido que sí lo cazaría tendría que comparar la descripción de la fila USDA contra
el nombre del alimento. Queda pendiente.

## Estado — CERRADO el 2026-08-19

Las cuatro migraciones del catálogo y la de procedencia están **aplicadas en producción**.

| Qué | Antes | Ahora |
|---|---|---|
| Filas sin `prep_methods` | 141 | **0** (+ CHECK que lo impide) |
| `fdc_id` compartidos | 20 grupos / 47 filas | **2 grupos / 4 filas** |
| Filas con procedencia declarada como proxy | 0 | **19** (`nutrition_source_ref`) |
| Proteína de `Yogurt` | 10,3 g (perfil griego) | **3,47 g** |
| Valores distintos de kcal en el cluster de embutidos | 1 | **4** |

**La regla que se aplicó** (`P1-PROVENANCE-TRUTHFUL`): un `fdc_id` es una afirmación,
así que solo lo conserva la fila cuya identidad Y valores coinciden con la fila real de
USDA — descripción consultada a la API, grupo por grupo. Las 19 restantes pasan a
`fdc_id = NULL` + `nutrition_source = 'manual'` + `nutrition_source_ref = 'usda:<id>
(proxy: <descripción>)'`. La traza se conserva; lo que desaparece es la afirmación falsa.

### Cerrado del todo: **0 `fdc_id` compartidos**

Con una `USDA_API_KEY` propia (1.000 req/hora frente a las 30 de `DEMO_KEY`) se
consultaron los 3 grupos que habían quedado sin verificar — y **DOS de las tres
conjeturas razonadas estaban INVERTIDAS**:

| `fdc_id` | USDA dice | Se suponía | Es realmente |
|---|---|---|---|
| 174220 | *Mollusks, **scallop**, raw* | Mejillones | **Vieira** — cuadra exacto (69 / 12,1 / 0,49 / 3,18); Mejillones diverge 25% porque tiene los suyos |
| 175202 | *Beans, **white**, mature seeds, raw* | Habichuelas blancas | **Judías blancas** — clava 333 = 333,0; Habichuelas está en 342,4 |
| 173443 | *Sour cream, light* | (ninguna) | **Crema mexicana** — recupera el reclamo; Suero costeño lo copiaba |

**Esa tabla es la justificación de haberlos dejado abiertos.** Si la ronda 1 los hubiera
cerrado «con criterio» habría escrito dos afirmaciones falsas — exactamente lo que este
P-fix corrige. Declarar la ignorancia salió más barato que razonarla.

El desempate es **coincidencia exacta**, no parecido: Habichuelas blancas queda a 2,8% del
valor de USDA y aun así pierde el reclamo frente a la que coincide al decimal.

Estado final: 20 filas con procedencia declarada (15 `proxy:` + 5 `id previo; valores
propios`), 3 dueños recuperados, **cero ids compartidos**, cero sentinels.

### Lo que sigue abierto — y ya no es de procedencia

- **LATINFOODS — hecho a medias, con cifras.** [P1-LATINFOODS-TCAC · 2026-08-19] Cinco
  alimentos andinos pasaron a la **Tabla de Composición de Alimentos Colombianos**
  (TCAC 2015, ICBF): Chontaduro **103 → 332 kcal** (vivía sobre *Breadfruit*: el
  chontaduro tiene 25,7 g de grasa, el panapén 0,23), Curuba 97 → 35, Borojó 66 → 134,
  Chinola 108,6 → 59, Suero costeño 136 → 83 kcal **y proteína 3,5 → 11,0** (vivía sobre
  *Sour cream*: es suero fermentado, no crema — el error era de categoría, no de
  magnitud). Extraído del PDF (no hay API ni Excel) con **Atwater como guard de
  parseo**: cada fila se acepta solo si 4P+4C+9G cuadra con las kcal dentro del 5%; las
  cinco cruzan bajo 1,5%. Solo macros — la tabla proximal no trae fibra ni minerales.
  **Quedan 9 filas sobre un proxy**: los chiles secos mexicanos (chipotle, guajillo,
  mulato), Xoconostle, los embutidos latinos (chorizo santarrosano y verde, longaniza
  puertorriqueña), Guineo verde y Requesón. Los mexicanos necesitan SMAE/INSP; los
  embutidos, una tabla que los tenga.
- **Sinónimos que son dos filas**: `Requesón`/`Queso ricotta`, `Judías blancas`/
  `Habichuelas blancas`. **Decisión tomada: NO fusionar.** El catálogo se resuelve por
  cadena, no por id: borrar una fila rompería cualquier plan, `user_inventory` o
  `supermarket_products.master_food_name` que la referencie por nombre.
- **Un `fdc_id` ÚNICO mal apuntado sigue siendo invisible** para esta auditoría. Existió
  (`Lomo embuchado`: apuntaba a lomo crudo, 110 vs 321 kcal) y lo destapó comparar contra
  BEDCA, no el barrido de duplicados. Un barrido que lo cazara compararía la descripción
  de la fila USDA contra el nombre del alimento — ahora es barato con clave propia.

## Barrido descripción-USDA ↔ nombre: el id ÚNICO mal apuntado (`P1-PLAN-LOTE-34` · 2026-09-13)

**Medido primero.** Esta auditoría dejó escrito que un `fdc_id` único mal apuntado era invisible a un barrido de
duplicados. `scripts/catalog_fdc_sweep.py` mira las dos señales fila a fila sobre las **288** filas con `fdc_id` (todas;
USDA respondió 287): la identidad (qué fracción de los tokens de `name_en`/`name` aparece en la descripción del id) y los
valores (desvío máximo de kcal/proteína/grasa/carbohidrato, con tolerancia absoluta de 12 kcal / 1,5 g / 1,5 g / 3 g
para no confundir convenciones de energía con otro alimento). Veredicto automático: 214 `coincide`, 19 `mal_apuntado`
(las dos señales en contra), 54 `revisar` (una), 1 `sin_respuesta`. Artefacto con los umbrales, el método y la revisión
humana de cada fila marcada: `scripts/data/catalog_fdc_sweep_2026_09_13.json`. Solo lectura (base `read_only`; la clave
de USDA por cabecera, nunca en la URL; las respuestas en caché para re-clasificar sin red).

**Lo que encontró, revisado a mano uno a uno** (y cada corrección verificada contra USDA antes de escribirla):

| Veredicto | Filas | Ejemplos |
|---|---|---|
| id corregido: apuntaba a OTRO alimento y los valores eran los buenos | 22 | Tamarindo → «Purslane, raw» (verdolaga); Hígado de res → T-bone asado; Leche de almendras → un experimento de genética de TOMATES; Kéfir → leche de avena; Mero → salmón rosado en lata; Cangrejo → mahimahi; Salmón → el de GRANJA (208 kcal) con valores del salvaje (142); Pavo molido → cocido con valores crudos |
| proxy no declarado → declarado (`fdc_id` NULL, `manual`, `usda:<id> (proxy: …)`) | 11 | Percebes con los valores del cangrejo azul; Guascas con los del diente de león; Huitlacoche con los del champiñón crimini; Leche de cabra en polvo con los de leche de VACA entera en polvo; Mapuey con los del ñame genérico (el id es de la fila Ñame) |
| valores sin la fuente que decían tener → «valores propios» con la traza | 7 | Flor de Jamaica (id del cáliz fresco, valores de flor seca); Yogur de coco (id de un maíz dulce con mantequilla); Guisantes secos (id correcto, valores de una edición anterior de SR); Frijoles pintos (sinónimo de Judías pintas, que tiene el id) |
| kcal por Atwater general con el id correcto → anotado | 17 | Canela 349,5 frente a 247; Vainilla 51 frente a 288 (el alcohol no está en 4·P+4·C+9·G); Vinagre 3,7 frente a 21 (el ácido acético tampoco) |
| glosa inglesa de otro alimento | 2 | Níspero «Loquat» → «Sapodilla» (en RD el níspero es el zapote, y el id ya lo era); Cebollín «Chives» → «Scallions» |
| correcto (nombre en español, uso dominicano o representante declarado) | 23 | Gambas/shrimp, Tuna de nopal/prickly pear, Auyama/pumpkin, Filete de pescado blanco/tilapia; Plátano maduro: Foundation publica dos energías y el catálogo usa la general |

**Lo que NO era, medido.** Los backfills de micronutrientes leyeron USDA POR `fdc_id`, así que la sospecha era que las
filas mal apuntadas llevaran los micros del alimento equivocado (Mapuey con la vitamina K de la col rizada). Comparado
micro a micro: **0 de 19** filas los heredó — los ids se torcieron después, o los valores entraron por otra vía. No hay
nada que re-traer, y la migración no toca ningún valor nutricional.

**Qué se escribió.** `migrations/p1_plan_lote_34_catalogo_fdc_2026_09_13.sql` (espejo en la raíz): 59 UPDATE que sólo
tocan `fdc_id`, `nutrition_source`, `nutrition_source_ref` y `name_en`, cada uno por `name` exacto, con tres sanity:
las 22 correcciones escritas, **cero `fdc_id` compartidos** (dos filas no pueden ser el mismo alimento de USDA — por eso
Mapuey y Frijoles pintos no toman el id de Ñame y Judías pintas: los declaran), y ninguna fila proxy o propia conserva un
id que la contradiga.

**Lo que sigue abierto, dicho.** Las 9 filas en proxy de antes siguen en proxy, ahora sabiendo por qué: USDA no tiene
chipotle, guajillo ni mulato secos, ni xoconostle, ni los tres embutidos latinos, ni guineo verde, ni requesón (Requesón
es sinónimo declarado de Queso ricotta; Yogurt de cabra sigue sobre su proxy de etiqueta). Salir del proxy exige tablas
nacionales (SMAE/INSP para México, INCAP para Centroamérica, TCAC 2018 para Colombia) que se publican como PDF: bajarlas
es una decisión del dueño. La vara del barrido no ve un id mal apuntado cuyos valores **y** nombre casen por casualidad
con otro alimento; tampoco juzga si el valor de USDA es el adecuado para el mercado dominicano — sólo si la afirmación
«esta fila es ese id» es verdad.

## Cómo re-ejecutarla

La clasificación no necesita red: sale de comparar macros dentro de cada grupo de
`fdc_id`. Solo la columna «Fila USDA real» exige llamar a la API — y con `DEMO_KEY`
(30/hora) se agota rápido, así que conviene tener `USDA_API_KEY` propia en el entorno.
