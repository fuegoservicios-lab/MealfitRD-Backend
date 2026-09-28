# Envases y catálogo por país en la lista de compras (lotes 790 y 791)

[P1-PLAN-LOTE-790 · P1-PLAN-LOTE-791 · 2026-09-28 · revisión ronda 1] Auditoría G58/G13 del 28-sep, medida
con replay determinista del agregador sobre una copia de `master_ingredients` (sin DB ni IA).

## Qué problema había

1. **El rótulo del envase hablaba dominicano en todos los países.** `_sku_size_label` convertía el peso del
   envase a libras y onzas («1 paquete (¼ lb)») y truncaba bajo 1 g («1 sobre (0g)» de azafrán).
2. **Las filas envasadas de los países beta no tenían envase.** De 130 filas sin precio, las 64 que se
   venden en envase salían a peso («1 lb de Azafrán»); Dátiles y Cúrcuma (DO) tenían el mismo hueco.
3. **Fuga de catálogo entre países.** El agregador conserva un alimento de catálogo-país de CUALQUIERA de
   los seis: un plan español llevó «Tortilla de maíz» (sólo MX) a su lista como si fuera de su mercado.

## Qué hace el código (lote 790, `envase_pais.py`)

| Pieza | Qué hace |
|---|---|
| `etiqueta_envase` | Rótulo del envase en el sistema del país de la lista: ES/MX/CO en g/kg/ml/L con coma; DO/US/PR como siempre; bajo 1 g con decimales. La etiqueta de un producto REAL (`market_packages[].label`) no se toca. Palanca: `MEALFIT_UNIT_SYSTEM_BY_COUNTRY` (la misma de la proyección métrica). |
| `con_pais_del_plan` / `lista_de_pais` | El país viaja por contexto desde el sello del plan (`constants.country_for_plan`) en `get_shopping_list_delta` y `get_realtime_pantry`. Sin plan (chat, swap) el contexto es `None` y todo es como antes. |
| `sellar_catalogo_de_otro_pais` | El alimento de catálogo de OTRO país se QUEDA en la lista y lleva `catalogo_de_otro_pais=[<países>]`. Puerto Rico y Estados Unidos cuentan como el mismo mercado. DO no se sella. Knob `MEALFIT_COUNTRY_CATALOG_FOREIGN_FLAG` (default on). El WARN `[P1-PLAN-LOTE-790]` sale una vez por (país, alimentos) y hora; lo repetido va a DEBUG. |
| `tope_de_comida` | En una fila SIN precio que se vende en paquete, bolsa o funda (comida, no especiero), el tope de condimentos (P1-SHOPLIST-SANITY-CAP) no baja de los envases que cubren la demanda que la receta dio EN GRAMOS: 4 × «60 g de Chile guajillo» = 3 paquetes de 85 g, no 1. La demanda por CONTEO sin peso por unidad se sigue capando: es la misma inflación que el tope corta en el orégano. El especiero no cambia. Knob `MEALFIT_CONDIMENT_CAP_FOOD_GRAMS_FLOOR` (default on). |

**Por qué sólo filas sin precio (revisión ronda 2):** en la tabla viva hay UNA fila dominicana Despensa en
paquete de ≤ 120 g, «Nueces mixtas» (paquete de 100 g, RD$95). Con el suelo para todas, 7 × «30 g de
Nueces mixtas» pasaba de «1 paquete» (RD$95) a «3 paquetes» (RD$285) y la mensual de RD$285 a RD$855. Era
arreglar una compra corta anterior al lote, pero cambiaba la lista DO sin declararlo. Con precio manda el
tope de siempre, y la lista dominicana queda idéntica (`fila_sin_precio`). **La compra corta de las nueces
sigue abierta** (210 g de receta en 100 g comprados, sin nota): pide su propio lote, como el pan.

**Por qué el ítem ajeno se queda y no se quita:** la receta lo pide (quitarlo deja la lista incompleta sin
aviso), el espejo del guard de coherencia no sabe de países (quitarlo en un lado abre una divergencia) y el
bloque de un país no es lo que se vende allí, sólo sus altas SIN PRECIO. La fuga nace en el generador: su
catálogo ya pregunta por país y el modelo escribió la tortilla igual.

**Datos del registro por país tocados en la revisión:** «trucha» pasa también a ES y «chile en polvo»
también a MX (mismo patrón que «duraznos», P1-PLAN-LOTE-624). Se venden allí; sellarlos era un falso
positivo. Efecto: el catálogo del generador los ofrece a ES y MX.

## Qué hace la migración (lote 791)

`migrations/p1_plan_lote_791_envases_beta_2026_09_28.sql` (sin aplicar; la copia en `migrations/` del repo
raíz la sube el integrador):

- Llena `market_container`, `container_weight_g` y `available_sizes_g` de las 64 filas beta envasadas; el
  bloque beta sólo toca filas con precio 0 y sin envase. No escribe precio ni `market_packages`.
- Columna nueva `container_source` con la procedencia de cada fila (Open Food Facts, ODbL).
- **Regla del tamaño**, una para las 66: el tamaño real más cercano a la mediana, salvo cuando la arrastran
  formatos que no son el de referencia (granel, hostelería, tarrinas de 32-64 oz) y la moda minorista es
  otra: entonces la moda (Adobo 227 g, Sofrito 340 g en frasco, Anchoas, Sazonador…).
- **Peso por unidad de los seis chiles secos** (bloque 2b, revisión ronda 2): «3 chiles anchos» × 7 salía
  «1 paquete» de 85 g para ~357 g de receta. Con `density_g_per_unit` el conteo se vuelve gramos y sale
  «4 paquetes (85 g c/u)». Regla: la porción «1 pepper» de USDA SR Legacy cuando existe para ese chile
  (ancho 17 g, pasilla 7 g); si no, la mediana de los conteos minoristas publicados (guajillo 9 g, mulato
  11 g con n=1, chipotle morita 3 g, de árbol 1 g). Sólo filas sin precio y sin peso previo; la
  procedencia se añade a `container_source`. Efecto declarado: la nutrición de «N chiles X» también
  resuelve gramos.
- CHECK `master_ingredients_envase_requires_weight`: una fila con unidad de envase exige
  `container_weight_g > 0` (con `COALESCE`, a prueba de NULL).
- **Excepción DO declarada:** el bloque 2 llena el envase de Dátiles (340 g) y Cúrcuma (57 g), filas DO con
  precio, con datos de EE. UU. Espera el OK del dueño con captura de su súper antes de aplicar.

## Pendiente del dueño

1. **El sello no lo pinta nadie todavía**: ni el frontend ni una métrica. O se pinta en la lista, o se acepta
   por escrito «quedarse y sellar» como la resolución de «restringir la fuga».
2. **Dátiles y Cúrcuma**: OK con captura antes de aplicar la 791 (ver arriba).
3. **Atribución pública de Open Food Facts**: pie del PDF o página legal (`docs/data_provenance_licenses.md`).
4. **Las etiquetas lb/oz de productos reales** (`market_packages`) siguen igual en ES/MX: es la regla vigente.
5. **Chile mulato**: su peso por unidad (11 g) sale de una sola fuente; el ancho, el mismo poblano seco,
   pesa 17 g en USDA. Confirmar con captura si se quiere afinar.

## Abierto, anterior al lote (cada uno cambia listas DO: su propio lote)

- **Compra corta de las Nueces mixtas** (arriba) y del **pan**: «2 unidades de Pan de agua» × 7 sale
  «1 unidad» para 840 g de demanda.
- **El tope borra el tamaño** cuando actúa («3 sobres de Azafrán», «3 frascos de Orégano»).
- **La coma decimal no se lee**: «0,2 g de Azafrán» → 150 g; «1,5 tazas de Arroz blanco» → 907 g (con
  punto, 277,5 g). Es del parser de cantidades, no de este lote.
- **Cúrcuma, dos medidas de la misma demanda**: con el frasco de 57 g, «1 cdta de Cúrcuma» × 30 sale con
  «alcanza ~24 de 30 días — recompra» (compra contra la necesidad ANTES de P6-SPICE-CAP, 70 g) mientras
  `base_qty` dice 28 g (DESPUÉS del tope). Sólo aparece si el dueño aprueba el frasco de 57 g.

## Validación de la revisión ronda 2 (replay, sin DB en el cálculo ni IA)

426 planes guardados (380 DO, 30 sin país, 5 ES, 11 MX) × ciclos de 7, 15 y 30 días, con la foto de
`master_ingredients` del 28-sep (SELECT de solo lectura) y la 791 emulada sobre ella:

- **Sólo código, contra la base anterior al lote 790:** 0 listas DO y 0 sin país distintas. ES y MX, sólo
  el rótulo del lote 790. La versión anterior de esta rama sí cambiaba 2 listas DO semanales de nueces
  (rd15, rd8: 105 y 140 g → «2 paquetes», RD$190); ahora vuelven a «1 paquete» (RD$95), como antes.
- **Peso por unidad de los chiles (datos de la 2b):** sólo cambian 6 listas MX, todas de Chile chipotle
  pedido por conteo (2,1 y 1,2 chipotles ≈ 6 y 3,5 g): «2 paquetes» / «3 paquetes (85 g c/u)» en 15 y 30
  días pasan a «1 paquete (85 g)».
- **Base anterior al lote contra la rama con la 791:** en DO sólo cambian Dátiles, Cúrcuma y Tortilla de
  maíz. En los 13 planes vivos (todos DO), sólo Dátiles y Cúrcuma.

## Tests

`tests/test_p1_plan_lote_790.py` (rótulos contra un oráculo copiado de la función anterior, contexto de
país, sello, PR↔US, WARN deduplicado, catálogo del generador) y `tests/test_p1_plan_lote_791.py` (forma e
idempotencia de la migración, cobertura exacta, CHECK, tabla LITERAL de rótulos de las 66 filas, chiles
fuera del tope de condimentos, la lista DO de nueces idéntica con y sin el suelo, el chile contado que se
compra por sus gramos, licencia).
