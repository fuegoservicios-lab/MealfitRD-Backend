# Roadmap 2.7 · Fase F1 — el embudo de selección

**[P0-ARQ27-F1 · 2026-09-06]** Seis gaps del mismo sitio: lo que el motor OFRECE como candidato y lo
que puede llamar íntegro. Todo medido sobre `b4684a0` y el catálogo vivo de 347 filas.

## El resumen en una tabla

| Gap | Qué estaba roto | Antes → después |
|---|---|---|
| ARQ27-P0-01 | `template_candidates` no recibía la dieta | **1.109 de 1.646 candidatos incompatibles (67,4 %) → 0** |
| ARQ27-P1-01 | `legumbre` es etiqueta de CLASE y la familia se resolvía por etiqueta | Lentejas 0→19, Garbanzos 0→15, Habichuelas 0→27, Guandules 0→2 |
| ARQ27-P0-02 | `status=ok` ignoraba `declared_unresolved` y `no_grams` | DO 131 ok → **127 ok / 4 partial** |
| ARQ27-P0-03 | `_f(None)` = 0,0: un dato ausente se sumaba como cero | 7 plantillas con `phosphorus_high: None` en vez de `False` |
| ARQ27-P1-07 | `compile_from_form` nunca pasaba `known_ingredients` | `market_check_skipped` (57/57) → `market_check_applied`; **26 plantillas DO** dejan de ofrecerse en ES/US |
| ARQ27-P1-02 | el embudo medía con una tabla propia sobre `protein` | paridad exacta embudo ↔ selector en 6 países × 3 dietas × 4 franjas |

## Las cinco decisiones que no son obvias

### 1. La dieta se comprueba sobre los CONSTITUYENTES, no sobre la etiqueta

`protein` dice qué protagoniza el plato, jamás qué contiene: una plantilla `none` puede llevar
lácteos y una `mixta` puede llevar jamón. Por eso el filtro recorre `constituents` y pregunta a
`_diet_pool_item_banned` — el guard SSOT que ya decide esto en el skeleton y en el day-gen (mismo
matcher por palabra completa, mismos `_DIET_*_TERMS`, misma excusa plant-adjacent «carne de soya»).

**No es una cuarta tabla.** La lección de `P1-DIET-CANON-SSOT` es que las tablas de dieta paralelas
derivan y acaban sirviendo pollo a vegetarianas; aquí hay un adaptador, memoizado por `content_hash`.

Si el guard no se puede importar, `template_candidates` devuelve `[]`. Quedarse sin bloque de
registro en el prompt es la conducta previa a F6; ofrecer carne a un vegano no lo es.

### 2. El puente de familia se abre SOLO para las etiquetas genéricas

De las diez etiquetas `protein` que existen en los seis snapshots, nueve nombran el alimento
(`pollo`, `huevo`, `pescado`, `queso`, `res`, `atun`, `cerdo`, `camarones`, `pavo`) y **`legumbre` es
la única que nombra una CLASE** — dice «esta receta se apoya en una legumbre», no cuál. Como el
allocator programa la familia por su nombre de alimento, `family_matches('Lentejas', 'legumbre')` era
`False` y las 64 plantillas de legumbre quedaban inalcanzables.

`family_matches_template` mira dentro de la plantilla **solo** cuando la etiqueta es genérica. Esto es
load-bearing: resolver siempre por constituyentes haría que «caldo de pollo» dentro de un guiso de
res colara la receta como familia `pollo`. Con las etiquetas específicas resolviendo por etiqueta,
ese falso positivo no existe — y una receta de garbanzos sigue sin satisfacer un ancla de lentejas.

`test_legumbre_es_la_unica_etiqueta_generica` acusa cualquier etiqueta nueva: si mañana el compilador
emite `cereal`, su familia quedaría inalcanzable en silencio igual que le pasó a `legumbre`.

### 3. Ausente ≠ vacío, en los dos lados

- **Nutrientes.** Un valor `NULL` del catálogo se anota en `nutrition_unknown` y la señal de riesgo
  que dependa de él sale `None`, no `False`. Medido: 5 de 347 filas sin `phosphorus_mg` (Hoja santa,
  Chontaduro, Champús, Borojó, Achiote — el lote beta de CO/MX) que sostienen 7 constituyentes ya
  compilados. Los otros nueve nutrientes están completos hoy; el contrato es lo que impide que la
  próxima fila con un hueco vuelva a certificarse como cero. **A un perfil renal se le presentaba
  «Arroz con pollo colombiano» como fósforo bajo cuando la verdad era que nadie lo había medido.**
- **Mercado.** Si el catálogo no se puede leer, `catalog_capability` devuelve `None` y el compilador
  conserva `market_check_skipped`. Devolver `[]` diría «este país no vende nada» y borraría todas las
  anclas del usuario.

Un faltante bloquea **solo a quien exige ese dato**: `required_nutrients` resuelve las condiciones con
`condition_rules.detect_active_rules` — `renal` exige fósforo y potasio, `hta` sodio, y un perfil sin
condiciones no exige nada.

### 4. Cultura ≠ mercado, también en el selector (I16)

La biblioteca de platos es de la COCINA; el catálogo es del MERCADO, y no tienen por qué coincidir.
Medido: **26 plantillas dominicanas piden Casabe u Orégano dominicano**, que ni ES ni US llevan en
catálogo. Sin el filtro, a una cocina dominicana comprando en Estados Unidos se le ofrecían igual
(DO/almuerzo: 52 candidatos en DO, 40 en US/ES).

`template_candidates(..., market_country=)` lo cierra. Desconocido no recorta nada.

### 5. Una plantilla incompleta no puede llamarse íntegra

`declared_unresolved` y `no_grams` bloquean el estado `ok` igual que `not_in_catalog`. Las cuatro
plantillas DO que figuraban íntegras teniendo exclusiones dentro:

| Plantilla | Qué le falta |
|---|---|
| Batida de zapote ligera | el zapote |
| Chillo al horno con vegetales | el chillo (se compone con filete de pescado blanco) |
| Mangú con salami de pavo | el salami (se compone con jamón de pavo) |
| Frutas picadas con limón y menta | la menta |

Las dos del medio son **sustituciones declaradas**: el título promete un alimento que la receta
compilada no tiene. Un plato sustituido necesita identidad propia, no un rótulo heredado.

La única salida es que la fuente marque el constituyente `optional: true` — una excepción curada y
visible, que sigue apareciendo en `excluded[]` para que el revisor la vea.

## Hallazgos que la implementación destapó

- **La 14ª de la clase «dos ortografías del mismo alimento».** El catálogo escribe `Gandules` (fila
  canónica, con «guandules» como alias) y `_FAMILY_TOKENS` solo tenía la forma con u. La familia
  `Guandules` —que el pool VEGANO programa— no alcanzaba ni una plantilla, ni las dos de
  `protein=legumbre` que la llevan dentro. `_FAMILY_REPRESENTATIVE` apuntaba también al alias, así que
  la inyección con Nevera vacía metía un nombre que el catálogo no usa.
- **`Tofu` está en el pool vegano y ninguna de las 690 plantillas lo lleva.** Un quinto del pool
  programa una familia que el registro no puede servir. Es ARQ27-P1-03 (fase F2), anclado como
  `xfail(strict=True)`: cuando se curen recetas con tofu el caso pasará a XPASS y romperá la suite,
  obligando a borrar la excepción en vez de dejarla caducada.
- **`MEALFIT_COUNTRY_SYSTEM` está en `False` en `conftest`** y sin él `country_for_form_data` colapsa
  los seis países a DO. Un test que no declara el flag que necesita mide otro producto. Es
  exactamente lo que ARQ27-P1-06 llama «batería con flags efectivos».
- **`_relax` guarda el motivo en `reason_code`, no en `reason`.** Leer el campo equivocado ya fabricó
  un gap inexistente en este mismo roadmap (el recuento de `relaxations[]`).
- **La caché de módulo sobrevive al `monkeypatch`.** Un snapshot de tres filas construido con un
  catálogo falso se leía en el test siguiente como si fuera el vivo. De ahí `reset_cache()`.

## Coste

`catalog_capability` cuesta ~2 s la primera vez de un proceso, y **eso es el `import` de
`graph_orchestrator`**, que el backend ya paga al arrancar. Snapshots posteriores: 0,001 s. El
veredicto por (alimento, mercado) se memoiza; el barrido de 6 países × 4 franjas × 2 dietas × 2
mercados tarda 0,4 s en caliente.

Resolver «¿este alimento existe en el mercado?» con `dish_registry.resolve_constituent` sobre el
índice del catálogo —el mismo SSOT que usa el compilador— en vez de un barrido lineal de `_matches`
quitó 4,5 s de la primera generación.

## Ficheros y tests

| Fichero | Qué aporta |
|---|---|
| [`dish_registry.py`](../dish_registry.py) | `template_candidates(diet=, require_known_nutrients=, market_country=)`, `_template_violates_diet`, `_BLOCKING_EXCLUSIONS`, `_RISK_SOURCES`, `nutrition_unknown` |
| [`horizon.py`](../horizon.py) | `family_matches_template`, `_GENERIC_PROTEIN_TAGS`, `required_nutrients`, cableado de los dos call sites |
| [`catalog_capability.py`](../catalog_capability.py) | `CatalogCapabilitySnapshot` del país de compra; `is_available`, `template_buyable_in`, `reset_cache` |
| [`plan_policy.py`](../plan_policy.py) | `compile_from_form` pasa `known_ingredients`; nota `market_check_applied` |
| [`scripts/coverage_funnel.py`](../scripts/coverage_funnel.py) | guarda real, etapas de mercado y datos, IDs y motivos por caída, cruces sin soja/sin gluten |

Tests: `test_p0_arq27_f1_selector_diet.py` (P0-01 + P1-01) · `test_p0_arq27_f1_integridad_compilada.py`
(P0-02 + P0-03) · `test_p0_arq27_f1_mercado.py` (P1-07) · `test_p0_arq27_f1_embudo.py` (P1-02).

```bash
python -m pytest backend/tests/test_p0_arq27_f1_*.py -q
python backend/scripts/coverage_funnel.py            # el embudo, con la guarda real
```
