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
| `envase_de_comida` | El tope de condimentos (P1-SHOPLIST-SANITY-CAP) no capa una fila que se vende en paquete, bolsa o funda: es comida, no especiero. Knob `MEALFIT_CONDIMENT_CAP_FOOD_PACKAGE_EXEMPT` (default on). |

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

## Tests

`tests/test_p1_plan_lote_790.py` (rótulos contra un oráculo copiado de la función anterior, contexto de
país, sello, PR↔US, WARN deduplicado, catálogo del generador) y `tests/test_p1_plan_lote_791.py` (forma e
idempotencia de la migración, cobertura exacta, CHECK, tabla LITERAL de rótulos de las 66 filas, chiles
fuera del tope de condimentos, licencia).
