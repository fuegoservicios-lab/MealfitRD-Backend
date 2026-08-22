# La lista de compras y la ventana de días que encoge

`[P0-SHOPPING-CYCLE-DAYS · 2026-08-22]` — doc canónica. Ancla: [`test_p0_shopping_cycle_days.py`](../tests/test_p0_shopping_cycle_days.py).

## El incidente

Plan real `2245eb45` (30 días, `country=US`, `_pricing_mode=beta_no_prices`).

El generador entregó 3 días de menú y una lista de compras de **48 alimentos** —
Pechuga de pollo, Cebolla, Habichuelas negras, Pan integral familiar, Champiñones,
Comino, Miel, Soya texturizada, Casabe, Batata, Brócoli… El shift rodante fue podando
los días ya vividos hacia `_archived_days` hasta dejar `days == []`. Cualquier
recálculo posterior reconstruyó la lista desde esa ventana erosionada y la
**sobrescribió**.

Quedaron **25 alimentos**, que son *exactamente* el conjunto canónico del último día
superviviente. Medido reconstruyendo con el agregador real sobre cada subconjunto:

| Fuente de días | Alimentos |
|---|---|
| día 1 (18-ago) solo | 21 |
| día 2 (19-ago) solo | 22 |
| **día 3 (20-ago) solo** | **25 — coincidencia byte a byte con lo publicado** |
| días 1+2 | 36 |
| días 2+3 | 40 |
| **los 3 (lo que el generador produjo)** | **48** |
| `days=[]` (estado persistido antes del fix) | **0** |

El usuario pulsó «ya compré la lista» y su nevera nació como espejo fiel de la lista
mutilada: 25 filas con **una sola proteína** (Huevo), sin cebolla y sin almidón básico.
El chunk siguiente intentó cocinar 3 días nuevos contra esa nevera, falló el gate de
despensa post-merge y quedó en `pending_user_action` — el usuario se quedó sin menú.

## Lo que NO fue

**El generador de planes no tuvo la culpa.** Produjo 48 alimentos; la mediana de la
flota (33 planes reales) es 46 alimentos y 4 proteínas. El plan del incidente fue el
único a 56% de cobertura receta↔lista con 1 proteína; el resto va de 86% a 98%.

También quedaron descartados con medición, y conviene no volver a perseguirlos:

- **El catálogo del país.** `master_ingredients` tiene 347 filas, las 347 con precio, y
  Cebolla / Pechuga de pollo / Arroz blanco / Habichuelas / Comino / Lentejas existen
  todas. Al generador se le ofrecen 249 nombres para `country=US`, 40 de categoría
  Proteínas.
- **El drop VERIFIED-ONLY.** Los 24 alimentos perdidos son todos
  `_is_verified_for_shopping=True`.
- **Un filtro por precio o por `beta_no_prices`.** `_strip_prices_for_beta_pricing_mode`
  sólo anula importes; ninguna rama elimina filas.
- **El concepto de «staple».** El único que excluye algo es `_should_ignore_shopping`
  (agua/hielo) — se comió «Agua fría», y nada más. Eso explica el 49→48.
- **La resta del delta contra la nevera.** La nevera estaba **vacía**: las 25 filas
  comparten `created_at` con el restock, y no había ninguna antes.

## El mecanismo

1. `get_shopping_list_delta` construía desde `plan_result["days"]` — la ventana viva.
2. El shift (`api_shift_plan` y su gemelo cron) poda `days` → `_archived_days` y **no**
   reconstruye ni marca ninguna de las 4 `aggregated_shopping_list*`.
3. `/recalculate-shopping-list` —que dispara el frontend al tocar la Nevera, el
   Dashboard o una preferencia de marca— sobrescribe las 4 listas con lo que salga de
   la ventana vigente. Sin comparación contra la lista anterior.

### Por qué el coherence guard no lo vio

`expected_sum_from_recipes` (lado ESPERADO) leía el **mismo** `plan_data["days"]`
encogido que el lado COMPRADO. Los dos lados se recortan a la vez y la divergencia se
cancela. La telemetría del plan lo confirma: pasó de **31 divergencias / 18 de
presencia** el 20-ago 04:32 a **6 / 3** y ahí se quedó durante 15 recálculos.

**Mutilar la lista MEJORÓ la métrica del guard.** Ése es el modo de fallo a recordar:
un verificador que toma su referencia del mismo sitio que el sujeto verificado no
verifica nada.

Corroboración por el otro lado: restaurando `days` desde `_archived_days` sobre el
`plan_data` real, el guard de HEAD devuelve 23 divergencias `cap_swallowed_modifier`
(Pollo, Cebolla, Pan integral, Habichuelas, Comino, Casabe, Soya texturizada…) y
`_has_severe_divergence()` da `True`. El guard siempre supo detectarlo; le habían
vaciado el lado de las recetas.

## El contrato

`shopping_source_days(plan_data)` ([`shopping_calculator.py`](../shopping_calculator.py))
es el **único** sitio que decide desde qué días se agrega la lista, y lo usan los dos
lados —`get_shopping_list_delta` y `expected_sum_from_recipes`— precisamente para que no
puedan volver a divergir. El test
`TestListaDeCompras::test_builder_y_guard_comparten_la_fuente` ancla esa simetría.

Reglas del helper:

- Une `_archived_days + days`, en ese orden (cronológico).
- **Acota al ciclo vivo**: descarta archivados con `date` anterior a `cycle_start_date`.
  `_archived_days` nunca se vacía, ni al renovar
  ([`chat_history_context.py:204`](../chat_history_context.py)) — sin este filtro un plan
  renovado arrastraría los alimentos de la temporada anterior.
- **Techo** en `total_days_requested` (default 30), quedándose con los **más recientes**.
- Días sin `date`: se conservan. Fail-open — perder menú es peor que arrastrarlo.
- Entrada corrupta → `[]`.

### Agregar más días no infla la compra

Es la objeción obvia y es infundada. El total es

```
Σ(ingredientes) × (7 / num_days) × cycle_qty_multiplier
  = promedio_por_día × días_del_ciclo
```

invariante en `num_days`. Con más días el promedio es un **mejor estimador**, no mayor.

## Knob

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_SHOPPING_SOURCE_INCLUDES_ARCHIVED` | `True` | `False` restaura la conducta previa (sólo `days`). Rollback sin redeploy. |

## Verificación contra el plan real

Con el fix, reconstruir la lista de `2245eb45` da **48 alimentos** en vez de 25, y las
proteínas pasan de `[Huevo]` a `[Habichuelas negras, Huevo, Pechuga de pollo, Queso
ricotta, Soya texturizada]`. Los 23 recuperados son exactamente los que el chunk pausado
reclamaba como «COMPLETAMENTE INEXISTENTES».

## Lo que este fix NO cierra

- **`get_realtime_pantry`** (`shopping_calculator.py`) sigue leyendo
  `plan_result["days"]` a pelo. Es otra superficie (despensa virtual del chat/swap) con
  la misma ceguera; se dejó fuera a propósito para no ampliar el radio del fix.
- **El shift sigue sin reconstruir la lista** al podar. El fix hace que el recálculo
  posterior ya no la encoja, pero la lista sigue describiendo el ciclo hasta que algo la
  recalcule.
- **Los planes ya erosionados en producción** no se reparan solos: necesitan un
  recálculo (o un arreglo de datos) para volver a la lista completa.
