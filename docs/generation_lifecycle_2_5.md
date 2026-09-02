# Lifecycle único de generación (roadmap 2.5, Fase 1)

[P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Primera fase del roadmap «Núcleo único»
([`docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md`](../../docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md) §5 y §12).
Motor SSOT: [`generation_lifecycle.py`](../generation_lifecycle.py). Entradas compartidas:
[`generation_inputs.py`](../generation_inputs.py). Endpoints: [`routers/plans_generation.py`](../routers/plans_generation.py).
Migración: [`migrations/arq25_f1_lifecycle_expand_2026_09_02.sql`](../migrations/arq25_f1_lifecycle_expand_2026_09_02.sql).
Test ancla: [`tests/test_p1_arq25_f1_lifecycle.py`](../tests/test_p1_arq25_f1_lifecycle.py).

## Qué cambia (detrás de `MEALFIT_INITIAL_VIA_QUEUE`, default OFF)

| Antes (SSE legacy, sigue vivo) | Ahora (cola) |
|---|---|
| `POST /api/plans/analyze/stream` lanza `asyncio.create_task(run_pipeline())` sin shield; el plan se INSERTA al final; dos `sleep(3)` evitan la doble persistencia; KV `pending_pipeline:*` es el único handle | `POST /api/plans/generation-runs` crea el run (I9), un **placeholder** en `meal_plans` (`generation_status='generating'`, `days=[]`) y un chunk `chunk_kind='initial'` en `plan_chunk_queue`; despierta al worker |
| Progreso = el propio SSE | `GET .../{run_id}/events` tailea DB + KV `run_progress:<run_id>`; `GET .../{run_id}` es el snapshot durable |
| Cancelar = cortar la conexión | `POST .../{run_id}/cancel` (cooperativo: el chunk que corre lo lee antes de persistir) |

El worker (`_chunk_worker`) ve `chunk_kind='initial'` y delega en `run_initial_chunk`, que corre
el MISMO pipeline y el MISMO postprocess que el SSE (`_postprocess_pipeline_result` con
`existing_plan_id`): el placeholder se RELLENA (`db_plans.fill_placeholder_meal_plan_atomic`) en vez
de insertar otro plan; los chunks 2..N se encolan igual que hoy.

## Por qué placeholder y no `meal_plan_id NULL`

El pickup del worker filtra `q1.meal_plan_id NOT IN (SELECT meal_plan_id … WHERE status='processing')`.
Un `NULL` en esa subconsulta hace el predicado NULL para **toda** la cola: nadie más se procesaría
mientras un chunk 0 corre. Con placeholder, el chunk 0 tiene plan real desde el nacimiento (I1) y
el pickup no cambia una línea. Coste aceptado: el placeholder es visible para los lectores
(`generation_status='generating'`, `days=[]`) mientras dura la generación.

## Invariantes

- **I9** `plan_generation_runs(user_id, idempotency_key)` UNIQUE. Replay ⇒ el run existente;
  mismo key con otro cuerpo ⇒ `409 idempotency_key_conflict`. La clave no caduca.
- **I10** el token de fencing es **`attempts`** (`FENCING_TOKEN_COLUMN`): el commit es
  `WHERE id = %s AND attempts = %s AND status = 'processing'`; 0 filas ⇒ el worker viejo
  descarta. No se añadió `lease_token`.
- **I11** `run_plan_pipeline` corre fuera de toda transacción; el fill del placeholder hace el
  finalize CPU-bound ANTES de abrir la transacción (mismo patrón que `save_new_meal_plan_atomic`).
- **I12** `meal_plans.revision` sube por **trigger** (`BEFORE UPDATE OF plan_data`,
  `IS DISTINCT FROM`), no por convención: 55 sitios escriben `plan_data` hoy.
- **I19** el worker es la única autoridad; el test ancla que `create_task(run_pipeline())`
  sigue apareciendo UNA vez (legacy) y no crece.
- **H1** el chunk 0 pasa por el pickup con `__PLAN_MODE_GATE__` (modo seguimiento no genera).
- **H5** `derive_availability` no devuelve `PLAN_READY` con `days=[]`.
- **H2/H3** intactas: el chunk 0 usa `days_offset=0` y el rebase/waivers no lo distinguen.

## Estados derivados

`derive_run_status` (PENDING/RUNNING/WAITING_RETRY/WAITING_USER/PAUSED/FAILED/CANCELLED/COMPLETED) y
`derive_availability` (NONE/PREVIEW_READY/PLAN_READY) se calculan desde `plan_chunk_queue` +
`meal_plans` + `plan_generation_runs`. **No se almacenan.** `PREVIEW_READY` = `PLAN_CHUNK_SIZE` (3)
salvo `MEALFIT_PREVIEW_READY_DAYS` (decisión #1 del roadmap).

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_INITIAL_VIA_QUEUE` | `False` | Interruptor de la fase. OFF ⇒ `POST /generation-runs` 404; SSE legacy intacto. Rollback para runs nuevos; los chunk 0 en vuelo se drenan. |
| `MEALFIT_INITIAL_CHUNK_MAX_ATTEMPTS` | `2` | Reintentos del chunk 0 (pipeline/postprocess) antes de dead-letter + placeholder→`failed`. |
| `MEALFIT_PREVIEW_READY_DAYS` | `PLAN_CHUNK_SIZE` | Umbral de `PREVIEW_READY`. |
| `MEALFIT_RUN_PROGRESS_THROTTLE_S` | `1.0` | Mínimo entre escrituras de progreso al KV. |
| `MEALFIT_RUN_EVENTS_POLL_MS` / `MEALFIT_RUN_EVENTS_MAX_S` | `2000` / `1500` | Cadencia y tope del SSE tail. |
| `MEALFIT_SHUTDOWN_DRAIN_S` | `90` | Drain cooperativo en SIGTERM: ticks nuevos no reclaman; se espera al tick en vuelo. |

## Guests

Siguen por `/analyze/stream` (400 `guest_not_supported_use_stream` en el endpoint nuevo). La cola
está keyed por `user_id` UUID. Decisión #3 del roadmap, opción recomendada.

## Deploy

1. `python scripts/apply_migration.py migrations/arq25_f1_lifecycle_expand_2026_09_02.sql --apply`
   (ANTES del binario: `attach_plan_to_run` y el fill leen `revision`/`run_id`).
2. Desplegar con el knob OFF. Verificar `/health/version` → `P1-ARQ25-F1-LIFECYCLE`.
3. Canary: `MEALFIT_INITIAL_VIA_QUEUE=true` + `VITE_INITIAL_VIA_QUEUE=true` sólo para la cuenta
   del dueño (o global con usuarios de test). Gate de la fase: 10 planes consecutivos sin
   duplicado ni CAS stale, ≥2 con kill del proceso mid-LLM recuperados por el zombie rescue,
   7 días sin alerta nueva.

## Deuda declarada (Fase 9)

- Los bloques de inyección server-side (`weight_history`/check-ins, «desde mi Nevera») están
  duplicados entre el SSE y `generation_inputs.py`; al retirar el SSE queda una copia.
- `plan_jobs` nace vacía; la Fase 5 conecta `display_i18n`, `shopping_commercial`, `dish_media`.
- El placeholder es visible en Historial/Dashboard como `generating`; si molesta antes de la
  Fase 4, filtrar `generation_status='generating' AND days=[]` en `/history-list`.
