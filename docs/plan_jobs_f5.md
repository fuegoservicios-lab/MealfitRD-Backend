# `plan_jobs` en producción — Fase 5 del roadmap 2.5 (rebanada 1)

[P1-ARQ25-F5-PLAN-JOBS · 2026-09-04] Motor SSOT: [`backend/plan_jobs.py`](../plan_jobs.py). Test ancla:
[`tests/test_p1_arq25_f5_plan_jobs.py`](../tests/test_p1_arq25_f5_plan_jobs.py). Roadmap: §5.1 («`plan_jobs`
(nueva) — el outbox»), §5.3 («Protocolo de worker»), §5.7 («consumidores en orden de llegada»).

## Qué es

Una cola genérica de **proyecciones asíncronas** del plan (read models que no deben bloquear la entrega:
traducciones, compras comerciales, imágenes). La tabla existe desde la Fase 1
(`migrations/arq25_f1_lifecycle_expand_2026_09_02.sql`) y la Fase 3 ya encola `shopping_projection`. Esta
rebanada pone el **worker** y el primer consumidor, **`display_i18n`**.

## Estados

```text
pending → processing → done
             ├──→ failed  (reintentable: execute_after = NOW() + backoff(attempts))
             ├──→ dead    (attempts ≥ MEALFIT_PLAN_JOBS_MAX_ATTEMPTS; dead_lettered_at)
             └──→ stale   (meal_plans.revision cambió antes de escribir → se re-encola para la vigente)
```

## Protocolo (el mismo que `_chunk_worker`)

| Paso | Función | Garantía |
|---|---|---|
| Claim | `claim_plan_jobs` (`CLAIM_SQL`) | `FOR UPDATE SKIP LOCKED` sobre `status IN ('pending','failed') AND execute_after <= NOW() AND job_type = ANY(<con consumidor>)`; `attempts += 1`, `claimed_by`, `heartbeat_at`. **`attempts` es el token de fencing.** Los tipos sin consumidor quedan `pending` intactos. |
| Consumo | `CONSUMERS[job_type](job)` | Fuera de la DB (I11). Devuelve `(status, error_code, result)`; nunca lanza (cinturón en el tick). |
| Revisión | `_consume_display_i18n` | Compara `meal_plans.revision` con `plan_jobs.plan_revision` (I13). Distinta ⇒ `stale` + `enqueue_plan_job` para la revisión vigente (dedup nuevo). |
| Commit | `finish_plan_job` (`FINISH_SQL`) | `WHERE id AND claimed_by AND attempts AND status='processing'`. 0 filas ⇒ `fencing_rejected` (métrica). `failed` con intentos agotados ⇒ `dead`. |
| Reclaim | `reclaim_stale_processing` (`RECLAIM_SQL`) | `processing` sin heartbeat desde `MEALFIT_PLAN_JOBS_HEARTBEAT_STALE_S` ⇒ `failed` (o `dead`). Cubre el deploy que mató al worker a mitad. |
| Wake | `wake_plan_jobs_worker` | `enqueue_plan_job` adelanta el próximo tick a AHORA (paridad con `wake_chunk_worker`). |

Semántica **at-least-once**: el consumidor es idempotente. `enrich_plan_display` ya lo era (lock KV,
`jsonb_set` por comida, ownership `AND user_id`).

## Consumidor `display_i18n`

- **Disparador**: `plan_display_i18n.schedule_plan_display_enrichment` llama a `maybe_enqueue_display_i18n`
  ANTES de abrir el hilo legacy. Con la cola viva (`MEALFIT_PLAN_JOBS_ENABLED=1` y
  `MEALFIT_PLAN_JOBS_DISPLAY_I18N=1`) y usuario con UUID, encola y vuelve; si la fila ya existía en
  `pending/processing/failed`, tampoco abre hilo (eso cierra los **ecos** que motivaron la fase). Guests
  (`session_id`, sin FK a `user_profiles`) y knob apagado ⇒ hilo legacy, sin cambios.
- **Dedup**: `display_i18n:<plan_id>:<revision>:<locale>:<all|d1,d2>`.
- **Veredicto** (`verdict_for_display_result`): `skipped ∈ {no_meals, no_days, knob_off, locale, not_found}` ⇒
  `done` (nada que reintentar); `{circuit_breaker_open, dedupe_inprocess, dedupe_locked, exception,
  partial_loss}` ⇒ `failed` con backoff (el lote perdido de `partial_loss` se recupera en el siguiente
  intento: lo ya escrito no se toca).

## Consumidor `shopping_projection` (rebanada 2 · P1-ARQ25-F5-SHOPPING-PROJECTION)

- **Quién encola**: la Fase 3 (`horizon.enqueue_shopping_projection_job`, solo bajo `enforce`), con las ventanas de la
  política (`main` + `fresh_topup` cada N días), `policy_hash`, `total_days`, `freezer_mode`. Dedup
  `shopping_projection:<plan>:<revisión>:<hash12>`.
- **Qué hace**: `build_shopping_projection` corre el MISMO agregador del recálculo (`get_shopping_list_delta`, Nevera
  y consumidos descontados, `calc_household_multiplier` × multiplicador de ciclo para la principal; solo hogar para los
  top-ups, filtrados a perecederos). Determinista, cero LLM.
- **Dónde vive el read model**: en `plan_jobs.payload.result.projection` del job `done` (no en `plan_data`: escribir ahí
  bumpea `revision` y haría stale a la propia proyección). `stale` ⇒ re-encola para la revisión vigente con el mismo hash.
- **Quién lo lee**: `GET /api/plans/{plan_id}/projections` (exento de cuota, `_PROJECTIONS_LIMITER` 30/60 s, `AND user_id`)
  → `{status: none|pending|ready|failed|stale, revision, projection?}` vía `classify_projection_jobs` (puro). La lista
  síncrona (`aggregated_shopping_list_*`) no cambia.

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PLAN_JOBS_ENABLED` | `False` | Interruptor maestro: worker (no-op) y encolado (cae a legacy). |
| `MEALFIT_PLAN_JOBS_DISPLAY_I18N` | `True` | Consumidor de traducciones. Apagado ⇒ el disparador vuelve al hilo. |
| `MEALFIT_PLAN_JOBS_BATCH` | `10` | Jobs por tick, clamp [1, 100]. |
| `MEALFIT_PLAN_JOBS_MAX_ATTEMPTS` | `5` | Intentos antes de `dead`, clamp [1, 20]. |
| `MEALFIT_PLAN_JOBS_BACKOFF_BASE_S` | `60` | Base del backoff exponencial (tope 6 h). |
| `MEALFIT_PLAN_JOBS_HEARTBEAT_STALE_S` | `600` | Edad del heartbeat para el reclaim, clamp [60, 21600]. |
| `MEALFIT_PLAN_JOBS_WORKER_INTERVAL_S` | `60` | Intervalo del cron `process_plan_jobs`, clamp [15, 600]. |

Los knobs se leen al arrancar el proceso (`.env` vía dotenv): cambiarlos requiere restart.

## Observabilidad

- `pipeline_metrics.node = 'plan_jobs'`: una fila por job terminado (`metadata.status ∈ done|failed|dead|stale|fencing_rejected`,
  `job_type`, `attempts`, `plan_revision`, `error_code`; `duration_ms`).
- Logs: `[ARQ25-F5] plan_jobs tick: {...}` por tick con trabajo; `job DEAD` en `error`.
- Gate de la fase (roadmap): lag p95 < 2 min en canary y cero `dead` sin alerta. SQL:

```sql
SELECT status, count(*), percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (processed_at - created_at))) AS p95_s
FROM plan_jobs WHERE job_type = 'display_i18n' AND created_at > NOW() - INTERVAL '7 days' GROUP BY 1;
SELECT id, plan_id, attempts, error_code, dead_lettered_at FROM plan_jobs WHERE status = 'dead' ORDER BY dead_lettered_at DESC LIMIT 20;
```

## Runbook

- **Encender**: `MEALFIT_PLAN_JOBS_ENABLED=1` en `/opt/mealfit/backend/.env` + restart. Verificar en el journal
  `Worker plan_jobs registrado` y, tras un disparador, `plan_jobs tick`.
- **Rollback**: `MEALFIT_PLAN_JOBS_ENABLED=0` + restart. Las filas quedan `pending` (inertes); los disparadores
  vuelven al hilo legacy.
- **Un `dead`**: leer `error_code`/`error_redacted`; para reintentar, `UPDATE plan_jobs SET status='failed',
  attempts=0, execute_after=NOW() WHERE id=...` (el worker lo recoge en el siguiente tick).

## Pendiente de la Fase 5 (siguientes rebanadas)

1. ~~Consumidor `shopping_projection`~~ (rebanada 2, 2026-09-04). Falta: marcas/retailer por presentación del
   `supermarket_products` como capa aparte (`shopping_commercial`), si el producto la pide.
2. ~~Reproyección encolada en el commit~~ (2026-09-05, `P1-ARQ25-F5-REPROJECTION`): `enqueue_shopping_reprojection`
   en recálculo (cubre Nevera/restock/consumo), swap, regeneración de día y relleno de bloques (T1); la huella
   de la lista (`list_fingerprint`) evita re-proyectar cuando la lista no cambió (el recálculo corre en cada visita).
3. ~~Estados UI~~ (2026-09-05, `P1-ARQ25-F5-UI-STATES`: `ShoppingProjectionStatus` en el Dashboard).
4. ~~Extracción de `shopping/projection/`~~ (2026-09-05, `P3-SHOPPING-PROJECTION-PKG`): `shopping/projection/{read_model,reprojection,status}.py`;
   `plan_jobs.py` queda como outbox puro y re-exporta. Packaging/pricing/shelf-life siguen en `shopping_calculator.py` (→ Fase 9); su
   tamaño y el de los otros god files queda CONGELADO por `test_p3_shopping_projection_pkg.py`.
5. ~~`system_alerts` para `dead`~~ (2026-09-05, `plan_jobs_dead:<job_type>`, modelo Manual).
