# Canario de entrega (E4 · ARQ27-P1-06) — latencia, coste, reintentos y fallos por plan; swap y último chunk verificados

[P1-PLAN-LOTE-15 · 2026-09-12] Cierra la parte de ARQ27-P1-06 que `scripts/delivery_battery.py` declaraba abierta, con su
frase: «la batería NO mide latencia ni coste por plan entregado —eso es el canary y necesita generaciones reales— ni
ejercita swap y último chunk end-to-end». Las generaciones reales YA existen: son los planes de producción. Este lote los
lee, plan a plan, sin escribir una fila.

## Lo que faltaba de verdad (medido el 2026-09-12, antes de tocar código)

- **El coste LLM no se podía atribuir a ningún plan de la cola.** `llm_usage_events` (30 días): `day_generator` **0 de 116** filas con `plan_id` y 0 con `corr`; `planner` 0/47; `reviewer` 0/126; `swap_meal` 0/117 con `plan_id` **y 0/117 con
  `user_id`**. Sólo `plan_display_i18n` (8/8) atribuía. El «canje `corr` → `plan_id`» de `P1-COST-ATTRIBUTION` (07-31)
  nació para el SSE, donde el id del plan llegaba DESPUÉS de generar; en la cola (el 100 % de los planes desde el 09-04) el
  placeholder ya tiene id ANTES, el worker corre en un hilo sin correlación de request y el INSERT que dispara el canje no
  ocurre. *Una atribución que depende de que el id nazca después deja de atribuir el día en que el id nace antes.*
- **Swap y último chunk no dejaban evidencia verificable**: 0 nodos `*swap*` en `pipeline_metrics` en 45 días, 0 entradas
  `post_swap_revalidation` en la history de los planes vivos, y `/swap-meal/persist` no estampa la comida sustituida.

## Qué cambia

- `graph_orchestrator.plan_id_var` (ContextVar) + `set_llm_attribution(user_id, plan_id)` / `reset_llm_attribution(toks)`.
  El emisor `_emit_llm_usage_event_best_effort` lee `plan_id_var` y estampa `plan_id` en la fila. Lo fijan:
  - el **worker de chunks** (`cron_tasks._chunk_worker`): conoce usuario y plan al reclamar; reset en el `finally` junto a
    la limpieza del contexto thread-local (el thread del pool se reutiliza);
  - **`/swap-meal`**: sólo si el plan del body es SUYO (`_owned_plan_id_for_attribution`, un SELECT por `id AND user_id`;
    un id ajeno o inválido → la fila queda sin plan, como antes). Contexto por request: no hay que deshacerlo;
  - **`/regenerate-day`**: después del SELECT de propiedad que ya hacía el handler.
  El canje por `corr` sigue vivo para el SSE legacy. Default `None` → conducta anterior. Las filas anteriores al despliegue
  quedan sin atribuir: el canario las muestra como **«sin atribuir», no como US$ 0**.

## Cómo se lee

```bash
python scripts/canary_plan_delivery.py --days 30 [--json]      # por plan y por cohorte (país · dieta · clínica)
python scripts/verify_swap_last_chunk.py --days 30 [--plan <id>] [--json]
```

`canary_plan_delivery.py` (sólo SELECTs, conexión `read_only`): por plan → entrega (`generation_status` + días vivos y
archivados), validez (entregado y sin `_quality_degraded`, sin `_review_failed_but_delivered`, sin chunk muerto), latencia
del bloque 1 (`plan_chunk_queue`, `initial`, created → completed), reintentos (chunks completados con `attempts ≥ 1`),
muertos, coste y llamadas LLM atribuidas, swaps atribuidos, alertas con `metadata.plan_id`, estado de la proyección de
compras (`classify_projection_jobs`, puro). Publica tasas **con su denominador** por cohorte y la fila global al final.

`verify_swap_last_chunk.py`: recetas ↔ lista con el **MISMO guard de producción** (`run_shopping_coherence_guard`, con
`_emit_coherence_guard_metric` sustituido por un no-op en el proceso — el guard no escribe nada), recetas completas
(ingredientes y receta en cada comida viva), horizonte (pedidos vs generados, cola cerrada) y proyección, sobre el
`plan_data` persistido — es decir, DESPUÉS de todos los swaps y del último chunk. Knobs de producción
(`prod_profile.perfil_aplicado`). Sin comidas vivas el veredicto es «no concluyente», nunca «pasa».

## Medición del 2026-09-12 (6 planes de 45 días, 3 usuarios)

| cohorte | n | entregados | válidos | bloque 1 p50 / p95 | con reintento | dead | coste atribuido | proyección |
|---|---|---|---|---|---|---|---|---|
| DO · balanced | 5 | 5 | 5 | 292 s / 365 s | 0 | 0 | 0 / 5 (sin atribuir) | 5 `stale` |
| DO · balanced · clínica | 1 | 1 | 0 (`quality_degraded` + `review_failed_but_delivered`) | 217 s | 1 | 0 | 1 (sólo i18n: US$ 0,016) | 1 `stale` |

Verificación con el guard, 6/6 planes: **0 divergencias severas** (sólo `recipe_unquantified`, 2-4 por plan), **0 comidas sin
ingredientes o sin receta**, cola cerrada en 5/6 (el sexto está `partial` con 7 chunks abiertos: en curso, no un fallo).
Incluye el plan `3957a669` del dueño con **48 revisiones** (swaps, regeneraciones, i18n): coherente tal cual está.

## Hallazgos que deja la medición (no arreglados aquí, dichos)

1. **La proyección de compras sale `stale` en 6 de 6 planes.** `classify_projection_jobs` compara la `plan_revision` del
   último job `done` con `meal_plans.revision`, y la revisión sube con CUALQUIER escritura de `plan_data` (i18n `_display`,
   history del guard, sellos) — mientras la reproyección se salta a propósito cuando la lista no cambió
   (`list_fingerprint`). El estado que ve el usuario dice «desactualizada» cuando la lista es la misma. La comparación
   honesta es por huella de la lista, no por revisión. Decisión de producto/UI: dueño.
2. Los 5 planes de un usuario del 09-09 tienen 7 chunks cancelados cada uno: regeneró cinco veces en un día y cada plan
   nuevo cancela al anterior. No es fallo de entrega; el canario lo enseña como `cancelled`, no como `dead`.
3. El plan del dueño se entregó con `review_passed=False` (`plan_quality_degraded`, «déficit de proteína día 5, 76 de
   105 g», 3 intentos) — es la alerta I5 funcionando; el canario lo cuenta como entregado y NO válido.
4. `plan_generation_runs.completed_at` es NULL en los 6 runs: `mark_run_completed` corre al cerrar el bloque 1 en
   `run_initial_chunk`, pero los runs vivos no lo tienen. Sin efecto en el usuario (el estado del run se deriva de la cola);
   a mirar si se quiere latencia por run en vez de por chunk.

## Lo que este canario NO afirma

- Cero hallazgos en 6 planes no demuestran una garantía universal: cada tasa lleva su N.
- El coste por plan sólo existe desde este despliegue; comparar con la ventana anterior es comparar con «sin dato».
- «Coherente tal cual está» no dice que cada swap intermedio lo fuera: dice que el estado que el usuario tiene HOY lo es.
  Con `swap_meal` atribuido por plan, la siguiente lectura puede acotar la verificación a los planes con swaps recientes.
