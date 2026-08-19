# Auditoría de procedencia del catálogo nutricional

**Fecha de la medición: 2026-08-19** · `master_ingredients`, 347 filas, Neon prod.

## Qué se midió y por qué

`fdc_id` es la única prueba de dónde salió cada valor nutricional. La pregunta que abrió
esta auditoría fue si esa prueba se sostiene. No se sostiene en 47 filas.

Dos hallazgos estructurales antes del detalle:

1. **47 de 347 filas comparten `fdc_id` con otra fila** (20 grupos). Una fila de USDA
   haciendo de sustituto de varios alimentos.
2. **`fdc 330137` devuelve HTTP 404.** El id ya no existe en USDA. Nada re-valida los
   `fdc_id`, así que la procedencia se pudre en silencio y nadie se entera.

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
| 330137 | **HTTP 404** | Yogurt, Yogurt griego sin azúcar | **DO** |

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

### Lo que sigue abierto, y por qué

- **2 grupos sin verificar**: `174220` (Mejillones/Vieira) y `175202` (Habichuelas
  blancas/Judías blancas). La API no devolvió su descripción por el límite de `DEMO_KEY`
  (30 req/hora) y decidir sin ella sería adivinar. Se cierran con una `USDA_API_KEY`
  propia en el entorno. El test ancla el número: si aparece un tercer grupo compartido,
  falla.
- **Sinónimos que son dos filas**: `Requesón`/`Queso ricotta`, `Judías blancas`/
  `Habichuelas blancas`. **Decisión tomada: NO fusionar.** El catálogo se resuelve por
  cadena, no por id, así que borrar una fila rompería cualquier plan, `user_inventory` o
  `supermarket_products.master_food_name` que la referencie por nombre. Fusionarlas es
  una migración de datos con su propio alcance, no el efecto colateral de una limpieza
  de procedencia.
- **LATINFOODS / SMAE** para los andinos y mexicanos que siguen sobre un proxy (Curuba,
  Chontaduro, Borojó, Xoconostle, Champús, Suero costeño, Crema mexicana, chiles secos).
  Ahora al menos **están etiquetados como proxy**, que era la mitad del problema. FAO no
  publica API: son PDF/Excel, así que es curación manual con su propio presupuesto.

## Cómo re-ejecutarla

La clasificación no necesita red: sale de comparar macros dentro de cada grupo de
`fdc_id`. Solo la columna «Fila USDA real» exige llamar a la API — y con `DEMO_KEY`
(30/hora) se agota rápido, así que conviene tener `USDA_API_KEY` propia en el entorno.
