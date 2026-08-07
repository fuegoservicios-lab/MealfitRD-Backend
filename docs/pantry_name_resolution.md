# Resolución de filas de la Nevera

[P1-PANTRY-NAME-RESOLUTION · 2026-08-07]

Doc canónica de **cómo se decide qué fila de `user_inventory` corresponde a un
nombre de alimento suelto**. Es la base sobre la que se apoya "la nevera solo
baja por lo que el usuario registra": sin resolución fiable, conectar más
superficies de consumo solo multiplica los descuentos fantasma.

---

## El incidente

Reproducido contra la nevera real del dueño (43 items; la fila dice `Huevo`):

```
coach registra "2 huevos"
  → _parse_quantity            → name = "Huevos"
  → WHERE ingredient_name = 'Huevos'        → 0 filas
  → quantity = -2
      · no entra al INSERT       (exige quantity >= 0.01)
      · no entra al guard de unidad incompatible (exige existing_rows no vacío)
  → return True                             ← MIENTE
```

Consecuencias encadenadas, todas silenciosas:

| Capa | Qué debió pasar | Qué pasaba |
|---|---|---|
| `user_inventory` | 3 huevos → 1 | queda en 3 |
| `failed_inventory_deductions` | fila para reintento | nada |
| cron de backlog | alerta | nada que alertar |
| `ToolMessage` al coach | "no pude descontar" | "¡Éxito!" |
| Usuario | ve la nevera bajar | ve 3 huevos para siempre |

Un descuento que falla es diagnosticable. Uno que **falla reportando éxito** no:
no deja rastro en ninguna de las tres capas de observabilidad que el repo ya
tenía montadas para exactamente este problema.

Familia conocida: `P1-SWAP-PANTRY-PLURAL` (2026-08-05) cerró el MISMO plural en
el reparador de coherencia (`"huevos" in "huevo"` es `False` porque el plural es
más largo) y dejó abierto el lado del inventario.

---

## La escalera

SSOT del criterio: [`constants.py`](../constants.py) →
`canonical_pantry_key` + `pantry_names_match`
(tooltip-anchor `P1-PANTRY-NAME-RESOLUTION-SSOT`).

SSOT del acceso a datos: [`db_inventory.py`](../db_inventory.py) →
`find_pantry_rows_for_name`
(tooltip-anchor `P1-PANTRY-NAME-RESOLUTION-RESOLVER`).

| # | Peldaño | Qué tolera | Coste |
|---|---|---|---|
| 1 | `exact` | nada — igualdad de string | 1 query indexada |
| 2 | `canonical` | case, acentos, espacios, cantidad al inicio, singular/plural | 1 query por usuario (~45 filas), comparación en memoria |
| — | `none` | — | no hay fila: el caller decide qué reportar |

El peldaño 2 solo corre si el 1 falla. No se puede indexar sin una columna
generada, y eso es DDL: iría a `migrations/`, no a un fix de comportamiento.

### Equivalencia singular/plural

`_pantry_token_variants` devuelve un **conjunto** de formas singulares
plausibles por token, no una sola. El español no permite decidir sin lexicón si
`-nes` viene de `-n` o de `-ne`:

- `limones` → limón (quitar `-es`)
- `carnes` → carne (quitar `-s`)

Emitir ambas y exigir intersección acierta en los dos casos. Forzar una sola
regla inventa un fallo en el otro. Las formas de más no crean falsos positivos
porque el match sigue siendo **token a token y con el mismo número de tokens**.

Tokens de menos de 4 letras no se singularizan nunca: es la disciplina de
`P1-SWAP-PANTRY-PLURAL` — `res`, `sal`, `ajo`, `mas` son palabras completas.
(`reses` y `sales` sí resuelven a `res`/`sal`: es la palabra LARGA la que se
singulariza, nunca la corta.)

---

## Lo que el matcher NO hace, y por qué

> ⚠️ **No conviertas esto en un cuarto consumidor de `GLOBAL_REVERSE_MAP`.**

`normalize_ingredient_for_tracking` colapsa **sinónimos**: `pechuga`→`pollo`,
`muslo`→`pollo`, `lomo`→`cerdo`. Eso es correcto para tracking de frecuencia y
para el guard de coherencia recetas↔lista, y catastrófico acá:

- comerte una pechuga descontaría de la fila `Muslo de pollo`
- la Nevera fusionaría dos alimentos que el usuario compró por separado, con
  precio y unidad distintos

La identidad de una fila física es **ortogonal** al parentesco nutricional.
`test_p1_pantry_name_resolution.py::test_does_not_reuse_the_synonym_normalizer`
falla si alguien "simplifica" delegando en el mapa de sinónimos.

Tampoco matchea:

| No matchea | Por qué |
|---|---|
| `Arroz integral` ↔ `Arroz` | compras distintas, precio y unidad distintos |
| `Leche de coco` ↔ `Leche` | alimentos distintos |
| `Pollo` ↔ `Repollo`, `Sal` ↔ `Salsa`, `Res` ↔ `Fresco` | los casos trampa de `P1-SWAP-PANTRY-PLURAL`: jamás por subcadena |
| `Guineo` ↔ `Plátano` | en RD son alimentos distintos (`P2-VISION-GUINEO-PLATANO`) |

**Regla de diseño:** ante la duda, NO matchear. Un no-match degrada a "no está
en tu nevera" — visible y seguro. Un match de más descuenta del alimento
equivocado — silencioso y corrupto.

---

## Call sites

Los cuatro sitios que resolvían filas tenían su propia copia del `WHERE
ingredient_name = %s` exacto, así que el mismo plural rompía los cuatro por
separado.

| Función | Qué se perdía pre-fix |
|---|---|
| `add_or_update_inventory_item` | el descuento del consumo (el incidente) |
| `_consume_reserved_inventory` | la reserva del plan quedaba colgada tras consumir |
| `_apply_reservation_delta` | la reserva no se creaba → `get_user_inventory_net` reportaba más disponible del real |
| `deduct_consumed_meal_from_inventory` | clasificaba la ausencia como éxito |

Cuando el match es canónico, **la ortografía de la nevera gana**: el resto de la
función (lookup en `master_ingredients`, refresh de `brand`, INSERT de fallback)
opera sobre el nombre que el usuario tiene, no sobre el que emitió la LLM. Sin
eso, el INSERT de fallback crearía justo la fila duplicada que se quiere evitar.

`reserve_plan_ingredients` pasa el lote completo del batch fetch en vez del dict
`rows_by_name` indexado por nombre exacto — ese índice heredaba el mismo punto
ciego que el SELECT que evitaba.

---

## `not_in_pantry`: la cuarta categoría

`deduct_consumed_meal_from_inventory` devuelve:

```python
{"succeeded": [...], "inferred": [...], "failed_to_deduct": [...], "not_in_pantry": [...]}
```

`not_in_pantry` es nueva. Pre-fix la ausencia y la deducción real caían **ambas**
en `succeeded`, así que el resumen decía "descontados 4 items" con la nevera
intacta.

**No se enruta a `failed_inventory_deductions`** a propósito: esa cola es de
fallos REINTENTABLES y su cron reintenta hasta dead-letter. Un item que no existe
en la nevera no mejora reintentándolo — solo gasta ticks y ensucia la alerta de
backlog (`failed_inventory_deductions_backlog`).

`tools.log_consumed_meal` lo expone en el `ToolMessage` para que el coach diga
"no los tenías registrados" en vez de afirmar que descontó.

### Presencia: un solo snapshot por comida

La clasificación usa **un** fetch de la nevera para todos los ingredientes de la
comida. Resolver item por item habría sumado 1-2 SELECT por ingrediente encima de
los que ya hacen `_consume_reserved_inventory` y `add_or_update_inventory_item`
(comida de 5 ingredientes: 30 queries en vez de 10).

El snapshot se usa **solo** para el sí/no de presencia. La aritmética de
cantidades sigue leyendo filas frescas dentro de `add_or_update_inventory_item`:
reutilizar cantidades del snapshot reabriría la ventana de lost-update que `P0-4`
cerró con la RPC `apply_inventory_delta`.

Si el snapshot falla, **no** se clasifica nada como ausente (eso sería la mentira
que este fix elimina): degrada a resolución per-item.

---

## Knob

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PANTRY_CANONICAL_MATCH` | `True` | A `False` desactiva el peldaño 2 y vuelve al comportamiento exact-only, sin redeploy. El peldaño 1 y la clasificación `not_in_pantry` siguen activos. |

---

## Cómo verificar

```bash
pytest backend/tests/test_p1_pantry_name_resolution.py -v
```

48 casos: el matcher (plurales sí / sinónimos y subcadenas no / simetría), la
escalera del resolver, la clasificación `not_in_pantry`, el knob de rollback, y
anclajes parser-based que hacen fallar el test si algún call site vuelve a
resolver por igualdad exacta o si el `ToolMessage` deja de reportar las ausencias.

---

## Superficies de consumo y su precisión

[P1-EAT-PLAN-MEAL · 2026-08-07] Los tres caminos por los que un consumo puede
llegar a la Nevera, ordenados por cuánto tienen que adivinar:

| Surface | Cómo obtiene los ingredientes | ¿Descuenta? |
|---|---|---|
| **"Me lo comí"** (`POST /api/diary/consumed-from-plan`) | los lee del plato del plan, que ya los trae con cantidades | Sí — **sin adivinanza** |
| **Foto** (`/upload` → `/consumed`) | el vision agent los emite como `items` estructurados y **el usuario los confirma** en el modal | Sí, confirmados |
| Chat (`tools.log_consumed_meal`) | la LLM tiene que acertarlos desde texto libre; `_infer_typical_portion` rellena las cantidades que falten | Sí, con estimación |

El botón "Me lo comí" (Dashboard, cards de HOY) es el camino preciso: el
cliente manda **coordenadas** (`plan_id` + `day_index` + `meal_index`), nunca
contenido, y el backend relee `plan_data` filtrando `AND user_id = %s` — misma
doctrina que I-Billing-1. Un cliente que pudiera declarar `ingredients`
descontaría de la Nevera lo que quisiera. Test:
[`test_p1_eat_plan_meal.py`](../tests/test_p1_eat_plan_meal.py).

Relación con el matcher por slot: `P1-TODAY-REMAINING` **deriva** "ya comiste"
comparando `meal_type` del diario contra el slot del plan, y se declara ambiguo
cuando hay ≥2 slots iguales (2-3 meriendas). El botón no compite con él:
convierte la inferencia en una **declaración** del usuario sobre un plato
concreto — justo el dato que al heurístico le falta. El registro se guarda por
`meal_type`, así que el matcher lo ve y atenúa la card como siempre.

---

## Lo que este P-fix NO resuelve

- ~~**La foto no descuenta.**~~ Cerrado por `P1-PHOTO-DEDUCTS` +
  `P1-VISION-PLATO-ITEMS` (2026-08-07): el prompt de visión ahora pide los
  componentes del plato **estructurados** (antes decía literalmente "deja items
  vacio" para `photo_kind='plato'`, y `_coerce_meal_scan` los descartaba con un
  `"items": []` hardcodeado), y `ConsumedMealRequest` acepta `ingredients`
  confirmados por el usuario. Test
  [`test_p1_photo_deducts.py`](../tests/test_p1_photo_deducts.py).

  > **Por qué aquí sí puede el cliente mandar ingredientes y en
  > `consumed-from-plan` no**: allí describen un plato del plan, que el backend
  > puede releer y verificar — aceptarlos del cliente sería dejarle declarar el
  > contenido de un dato que el servidor ya posee. Aquí describen lo que el
  > usuario declara haber comido fuera del plan, igual que si lo escribiera en
  > el chat; no hay fuente server-side contra la cual verificarlos. La
  > confirmación humana en el modal es la autorización.
- ~~**"Deshacer registro" no devuelve la comida a la nevera.**~~ Cerrado por
  `P1-CONSUMPTION-LEDGER` (2026-08-07) — ver "Descuentos reversibles" abajo.
- **Deriva por no registrar.** Lo que el usuario come sin registrar nunca sale
  de la nevera. Mitigación propuesta sin romper la regla "solo acciones del
  usuario reducen": reconciliación periódica que PREGUNTA (usa el
  `shelf_life` de `_infer_shelf_life_days`), no descuento automático.
- **Cantidades inferidas siguen aplicándose sin marcar.**
  `_infer_typical_portion` adivina 50 g / 1 unidad y lo aplica; el usuario no
  puede distinguir números reales de adivinados.

---

## Descuentos reversibles (el ledger)

[P1-CONSUMPTION-LEDGER · 2026-08-07]

`DELETE /api/diary/consumed/{meal_id}` borraba la fila del diario y dejaba la
Nevera descontada:

```
registra "2 huevos"  → diario +1 fila, Nevera 3 → 1
deshace el registro  → diario −1 fila, Nevera SIGUE EN 1
```

La asimetría es visible para el usuario y erosiona la confianza más rápido que
cualquier error de estimación.

### Por qué hacía falta una tabla

Para devolver hay que saber **qué** se descontó, y eso se perdía al aplicar el
delta. El string original (`"2 huevos"`) no basta:

- `P1-PANTRY-NAME-RESOLUTION` pudo mapearlo a la fila `Huevo` (otra ortografía).
- `P1-PANTRY-INFER` pudo **inventar** la cantidad cuando el parse no la extrajo.

Re-parsear el string al revertir repetiría ambas decisiones y podría llegar a
otra respuesta — devolviendo una cantidad distinta de la que se quitó. El ledger
guarda el nombre **ya resuelto** y la cantidad **ya aplicada**: revertir es leer
y sumar, no volver a interpretar.

### Qué es reversible

| `outcome` | ¿Movió la Nevera? | ¿Reversible? |
|---|---|---|
| `deducted` | sí | **sí** |
| `inferred` | sí (cantidad inferida) | **sí** |
| `not_in_pantry` | no | no — devolverlo **crearía** comida inexistente |
| `failed` | no | no |

### Decisiones de diseño

- **Sin FK a `consumed_meals`.** La fila es borrable por el usuario: un
  `CASCADE` borraría el registro de una devolución que **sí** ocurrió, y un
  `RESTRICT` impediría el propio `DELETE` que este ledger existe para soportar.
  La integridad que importa es el rastro, no la referencia.
- **`reverted_at` en vez de borrar la fila.** Hace el revert idempotente (un
  segundo `DELETE` no vuelve a sumar) y conserva la historia. Un ledger que se
  borra a sí mismo no es un ledger.
- **`CHECK (quantity > 0)`.** Los eventos registran magnitud; el signo lo pone
  la operación. Un evento negativo, al revertirse, restaría más.
- **Se marca `reverted_at` ANTES de sumar.** Si el proceso muere a mitad, el
  modo de fallo es "no devolví todo" (la Nevera queda baja — visible, el
  usuario lo corrige) en vez de "devolví dos veces" (queda alta, nadie lo nota,
  y el plan compra de menos). Entre dos fallos parciales, gana el detectable.
- **El revert corre ANTES del `DELETE`.** Si se borrara primero y el revert
  fallara, la comida se pierde sin rastro visible. Al revés, el usuario ve la
  comida aún en su diario y reintenta; el revert es idempotente, así que el
  segundo intento no duplica.
- **Persistir el ledger es best-effort.** Si falla, el descuento ya ocurrió y
  negarlo costaría el registro calórico entero. Se pierde la capacidad de
  deshacer *ese* registro, y eso se declara en el log.

### Productores

Los cuatro atan el evento a su fila de `consumed_meals` — sin `consumed_meal_id`
el evento es huérfano y el revert no lo encuentra:

| `source` | Call site |
|---|---|
| `plan_meal` | `POST /api/diary/consumed-from-plan` |
| `photo` | `POST /api/diary/consumed` |
| `chat` | `tools.log_consumed_meal` |
| `chunk_reconcile` | `sync_inventory_after_chunk_completion` |

`source` no es decorativo: cuando una Nevera no cuadra, la primera pregunta es
"¿qué la movió?", y las superficies tienen fiabilidades muy distintas.

### Cómo verificar

```bash
pytest backend/tests/test_p1_consumption_ledger.py -v   # 20 casos
```

Migración: [`migrations/p1_consumption_ledger_2026_08_07.sql`](../migrations/p1_consumption_ledger_2026_08_07.sql)
(idempotente; ⚠️ recordar la copia de workspace-root por `P3-MIGRATIONS-SSOT`).
