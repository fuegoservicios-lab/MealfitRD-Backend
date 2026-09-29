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

## Lote 852: la lista beta sin productos del súper de RD

[P1-PLAN-LOTE-852 · 2026-09-29] Validación beta G24 (6 planes reales con P1-PLAN-LOTE-815). Código en
`lista_sin_super_rd.py`; tres líneas en `shopping_calculator.py`.

**Qué había.** Entre 14 y 20 alimentos de cada lista beta llevaban `brand_product_id` de un producto de
`supermarket_products` con su envase («1 funda (1 Lb) de Sal», «Selecto 1 Lb», «Queso ricotta · Sosua» en
CO: el saneador de marcas de P1-BETA-PRICE-LEAKS no ve la marca tras paréntesis anidados), y entre 17 y
22 `market_pkg_price_rd`. `get_shopping_list_delta` pedía las marcas default y las preferencias del súper
para toda lista, sin mirar el país que ya viajaba por contexto (lote 790).

**Qué hace.** Con el país de la lista distinto de RD (palanca `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS`):
no se consultan marcas default ni preferencias, y al final del agregador se quitan `brand_product_id` y
`market_pkg_price_rd` (este último también lo ponía el `market_packages` del propio catálogo, que es del
Supermercado Nacional: `price_source='nacional_tienda'`). El precio RD se sigue usando DENTRO del agregador
para elegir el tamaño, como hasta hoy. En una lista métrica (ES/MX/CO; palanca
`MEALFIT_BETA_METRIC_PACKAGE_LABELS`) la talla imperial del envase del catálogo elegido se reescribe en
g/kg/ml después de elegirlo («35.2 oz» → «1 kg», «1 lb seco» → «454 g seco», «Amarilla 3 Lb» → «Amarilla
1,4 kg»): la cuenta no cambia. **Esto toca la decisión pendiente n.º 4 de arriba** («las etiquetas lb/oz de
productos reales siguen igual en ES/MX»): en un país beta ese producto no está en su estante. Si el dueño
prefiere la regla de 790, basta apagar esa palanca. La palanca de tallas depende de la del súper: apagar
`MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS` devuelve la lista anterior byte a byte, tallas incluidas. RD,
planes sin sello y superficies sin plan (chat, swap): idénticos.

**Replay (sin IA, DB sólo SELECT):** las listas de 7, 15 y 30 días de los 6 planes G24 re-agregadas con el
código anterior y con el nuevo. RD idéntica byte a byte en los tres ciclos. Semanal, antes → después:

| País | brand_product_id | market_pkg_price_rd | ítems con lb/oz/«funda» | ítems que cambian |
|---|---|---|---|---|
| ES | 16 → 0 | 17 → 0 | 11 → 0 | 16 de 38 |
| MX | 14 → 0 | 17 → 0 | 12 → 1 (Manzana: «funda», el envase del catálogo) | 16 de 38 |
| CO | 18 → 0 | 19 → 0 | 14 → 0 | 18 de 38 |
| US | 18 → 0 | 20 → 0 | 22 → 21 (su sistema) | 18 de 39 |
| PR | 20 → 0 | 22 → 0 | 25 → 21 (su sistema) | 19 de 46 |

Ningún alimento entra ni sale de ninguna lista. Cambian cuentas donde el envase del súper y el del catálogo
tienen otro tamaño (PR Atún 2 → 3 latas; US Espinacas 1×450 g → 2×150 g; CO mensual Lentejas 2×500 g →
2×800 g seco, porque el catálogo sólo tiene esa funda).

**Ronda 1 de revisión (tres cambios):**

1. *La palanca de tallas actuaba con la del súper apagada.* El rollback documentado no devolvía la
   conducta anterior: reescribía rótulos de productos reales del súper («Selecto 1 Lb» → «Selecto 454 g»;
   «1 Lb (454 gr) · Sosua» → «454 g (454 gr) · Sosua»), justo lo que P1-UNIT-SYSTEM-BY-COUNTRY prohíbe.
   Ahora `_talla_metrica_activa` exige también la del súper, y la talla jamás toca un ítem con
   `brand_product_id` (corre antes de quitar ese campo, para que la defensa lo vea).
2. *El número sale de `package_grams`, no sólo de la onza del rótulo.* La onza es ambigua (peso 28,35 g,
   fluida 29,57 ml) y el tamaño del envase ya está medido en `package_grams` (lo que guarda la Nevera).
   Si una talla del rótulo lo describe (±3 %), se pinta desde esos gramos y el factor que cuadra decide g
   o ml; si no cuadra ninguno («1 lb seco» = 1135 g cocida), el número del rótulo. A ≤0,5 % de un kilo o
   litro entero, el entero. No se usa `envase_pais._etiqueta_metrica_envase` para la CLASE porque decide
   por el envase («botella» ⇒ ml, que era el error de la mostaza); sí sus formateadores. De los 109
   rótulos imperiales del catálogo (SELECT), cambian 8: Ajo en polvo «89 ml» → «85 g», Mostaza «237 ml» →
   «227 g», Salsa de soya «283 g» → «295 ml», Harina de maíz precocida «998 g» → «1 kg», Lentejas «439 g»
   → «440 g», Vinagre blanco y de manzana «454 g» → «473 ml», y Vainilla «148 ml» → «142 g» (el catálogo
   guarda 141,75 = 5 × 28,35: si es extracto líquido, el dato a corregir es su `package_grams`).
3. *Un envase que ES una libra* (Chicharrón, Pernil, Tocineta, Gallina criolla: `unit='libra'`, «1 lb»)
   ya no repite la talla: «5 kg (454 g c/u) de Chicharrón» → «5 kg de Chicharrón». `sku_size_label`
   conserva «454 g».

Replay de la ronda (mismas 18 listas): con la palanca apagada, los 6 países idénticos byte a byte a la
base; RD idéntica; respecto al commit anterior cambian 9 ítems, todos de rótulo: Salsa de soya en ES
(«283 g» → «295 ml», 3 ciclos) y Harina de maíz precocida en MX y CO («998 g» → «1 kg», 6 ítems).

**Lo que este lote NO cierra (con datos):**

1. **«Garbanzos (Secos)» cuando la receta es de lata (CO).** Sin el súper sale «1 paquete (454 g seco)»:
   la receta dice «garbanzos cocidos (de lata, enjuagados y escurridos)» y `envase_legumbre.linea_lista`
   (lote 285) quita lo que va entre paréntesis ANTES de buscar la forma (`envase_legumbre.py:144`,
   `re.sub(r"\([^)]*\)", " ", t)`): `_LINEA_LISTA_RX` casaría con «(de lata», pero ya no lo ve. Así que no
   pide el envase listo y gana el seco por precio. Arreglarlo cambia listas RD que escriben igual: su
   propio lote.
2. **Requesón en México: hueco del catálogo, no fuga del súper.** Sólo lo reclama el bloque de ES
   (`_COUNTRY_CATALOG_UNPRICED_BY_COUNTRY`); el catálogo MX del generador no lo ofrece (245 nombres), el
   registro de platos MX no tiene ninguno con requesón y `data/country_gaps/` sólo lo registra en ES
   (y como caída en RD). El modelo
   lo escribió igual («Gorditas… con requesón») y la lista lo conserva sellado `catalogo_de_otro_pais=['ES']`
   (lote 790). El revisor dice que es muy mexicano: si el dueño lo confirma, añadir `requeson` al bloque MX
   (el mismo patrón que «trucha» → ES y «chile en polvo» → MX en la revisión de 790). No se hace aquí para
   no inventar el dato.
3. **«Orégano» en ES resuelve a «Orégano dominicano».** Es la ÚNICA fila de orégano del catálogo y su alias
   «orégano» la alcanza desde cualquier país. Lo que ve el usuario ya es neutro («Orégano»,
   P3-OREGANO-DISPLAY-NAME) y, tras este lote, sin marca ni precio: «1 sobre (45 g) de Orégano». Lo que queda
   es de DATO: los micronutrientes son los de la fila dominicana. Cerrarlo pide una fila de orégano común
   (Origanum vulgare) con su fuente: tarea de catálogo del dueño, no de resolución de nombres.
4. **«funda» como palabra de envase** en ES/MX/CO cuando el catálogo la declara (`market_container`): 1 ítem
   en G24 (Manzana, MX). Es vocabulario de la vista: va con la capa de nombres por país.
5. **El saneador de marcas con paréntesis anidados** (`_strip_prices_for_beta_pricing_mode`) sigue dejando
   pasar «· Sosua» si se apaga `MEALFIT_BETA_NO_DO_SUPERMARKET_PRODUCTS`: con la palanca encendida no hay
   marca que sanear.

Test: `tests/test_p1_plan_lote_852.py` (listas de los cinco países sin producto ni precio del súper, sin
siquiera consultarlo; RD idéntica y con su súper; palancas, con la del súper apagada idéntica a la base;
superficies sin país; talla métrica con la misma cuenta que RD, desde `package_grams`; la talla no toca un
producto del súper; el envase por libra no repite la talla; conversión token a token).
