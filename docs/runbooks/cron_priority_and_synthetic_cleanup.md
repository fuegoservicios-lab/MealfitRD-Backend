# Runbook: Cron priority map + Plan Sintético cleanup SOP

> Origen: P3-LIVE-10 (mapa de prioridad de los 30+ crons registrados en
> `register_plan_chunk_scheduler` con clasificación CRITICAL/HIGH/MEDIUM/
> LOW/HYGIENE para decisiones bajo saturación del scheduler) + P3-LIVE-12
> (SOP paso-a-paso para detectar y limpiar planes test-fixture acumulados
> en prod, complementario al cron diario `_sweep_synthetic_test_plans`
> P1-LIVE-3).

Este runbook complementa `system_alerts_sops.md`. Su propósito:

1. **P3-LIVE-10**: dar al SRE/operador una tabla de prioridades cuando
   hay que decidir qué cron sacrificar en saturación del scheduler.
2. **P3-LIVE-12**: SOP manual de detección y limpieza de
   `Plan Sintético% — Test%`, complementario al cron diario
   `_sweep_synthetic_test_plans` (P1-LIVE-3) para casos one-shot urgentes.

---

## P3-LIVE-10 · Mapa de prioridad de crons

Los 30+ jobs registrados en `register_plan_chunk_scheduler`
(`backend/cron_tasks.py`) se clasifican en 5 niveles. Cuando el thread
pool está saturado y APScheduler skipea jobs, esta tabla informa la
decisión: ¿toleramos el skip o escalamos workers/grace?

### CRITICAL — load-bearing (sistema se rompe si missed sostenido)

| job_id | Frecuencia | Por qué es CRITICAL |
|---|---|---|
| `process_plan_chunk_queue` | `CHUNK_SCHEDULER_INTERVAL_MINUTES` | Worker principal: si missed, chunks no se procesan → usuarios sin plan |
| `recover_failed_chunks_long_plans` | 15min | Re-encola chunks failed con backoff; sin él chunks failed quedan zombies |
| `recover_future_scheduled_pending_chunks` | 60min | P1-NEW-D: escala chunks con execute_after fuera del plan window; sin él zombies de meses |
| `recover_orphan_chunk_reservations` | `CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES` | Libera reservas de inventario huérfanas; sin él pantry queda bloqueada |
| `cleanup_orphan_chunks` | `CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES` | Cancela chunks con meal_plan_id inexistente (FK orphans) |
| `finalize_zombie_partial_plans` | `CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES` | Convierte planes `partial` con chunks completos en `complete` |

**Acción si CRITICAL skip sostenido**: escalar `MEALFIT_SCHEDULER_MAX_WORKERS`
o subir `MEALFIT_SCHEDULER_MISFIRE_GRACE_S`. NO desactivar el knob.

### HIGH — autoheal/observability core

| job_id | Frecuencia | Por qué es HIGH |
|---|---|---|
| `resolve_stale_scheduler_alerts` | `MEALFIT_SCHED_SWEEP_INT` (default 60min) | 5 sweeps: standard TTL, one-off UUID, cascade post-stab, cascade hard-cap, missed hard-cap |
| `alert_scheduler_cascade_missed` | `MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN` (default 30min) | Detector P0-2 + autoheal P0-NEW-2 (Sentry + sweep) |
| `sweep_stale_llm_circuit_breakers` | `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` (default 60min) | Cierra CB rows is_open=true sin recurrencia (P2-NEW-D) |
| `resolve_stale_plan_quality_alerts` | `MEALFIT_PLAN_QUALITY_AUTO_RESOLVE_INTERVAL_MIN` (default 60min) | P2-NEW-10: auto-cierra `plan_quality_degraded:*` por semántica plan-posterior-limpio + `post_swap_critical_divergence_*` por edad ≥24h |
| `alert_deploy_lag_marker_stale` | `MEALFIT_DEPLOY_LAG_INTERVAL_MIN` | Detecta `_LAST_KNOWN_PFIX` stale o drift vs `expected_last_known_pfix` KV |

**Acción si HIGH skip sostenido**: el hard-floor asyncio (P0-LIVE-1) cubre
los 2 primeros cada 300s. El tercero también — verificar tick
`_hardfloor_autoheal_tick` en pipeline_metrics.

### MEDIUM — alerts user-impacting

| job_id | Frecuencia | Función |
|---|---|---|
| `alert_chunks_paused_indefinitely` | `CHUNK_INDEFINITE_PAUSE_INTERVAL_MINUTES` | Detecta chunks paused >X tiempo, alerta SRE |
| `alert_new_dead_lettered_chunks` | `_P12_DL_INT` | Detecta chunks dead-letter recientes |
| `alert_chunk_lag_excessive` | `_P0G_LAG_INT` | Lag entre execute_after y pickup > umbral |
| `alert_chunk_dual_processing` | `_P0G_DUAL_INT` | Detecta race: dos workers procesando el mismo chunk |
| `alert_chunk_pantry_snapshots_stale` | `_P1D_INT` | Pantry snapshots stale en chunks pending (P1-LIVE-1: requiere `_pantry_captured_at IS NOT NULL`) |
| `nudge_chronic_zero_log_users` | `CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES` | Push a usuarios que no loggean comidas |
| `nudge_users_with_unresolved_tz` | `_P1B_TZ_INT` | Push a usuarios con tz_unresolved |
| `detect_chronic_deferrals` | `_P12_INT` | Detecta usuarios con N deferrals consecutivos |
| `proactive_refresh_pantry_snapshots` | `CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES` | Refresca pantry snapshots para chunks próximos |
| `alert_atomic_pool_fallback` | `_P26_INT` | Alerta si update_user_health_profile_atomic cae a non-atomic |
| `alert_chunks_stuck_in_tz_unresolved` | `_P0A_TZ_INT` | Detecta chunks bloqueados por tz_unresolved |
| `alert_coherence_watchdog_silent` | `_COH_LIVENESS_INT` | P0-3 liveness: detecta cron de coherencia silenciado |
| `reactivate_shopping_list_after_perishable_cycle` | `_reactivate_interval_min` | Refresca shopping list tras ciclo de perecederos |

**Acción si MEDIUM skip sostenido**: tolerable 1-2h pero re-emisión cae a
tiempos no accionables.

### LOW — telemetry only

| job_id | Frecuencia | Función |
|---|---|---|
| `aggregate_coherence_block_history_metrics` | `MEALFIT_COHERENCE_METRICS_INTERVAL_MIN` (default 60min) | P3-B: agrega `_shopping_coherence_block_history` a pipeline_metrics |
| `alert_high_synthesized_lesson_ratio` | `_P0A_INT` | Alerta si ratio synth:real > umbral |
| `alert_failed_inventory_deductions_backlog` | `MEALFIT_FAILED_DEDUCTIONS_ALERT_INTERVAL_MIN` (default 60min) | Backlog del log de deducciones fallidas |
| `shopping_coherence_alert` | Diario 04:00 UTC | P2-LIVE-9: re-evaluation 24h con tick observable |
| `nightly_refresh_long_plan_snapshots` | Diario 00:00 UTC | Refresh batch nocturno |
| `background_rolling_refill` | `_BG_RR_HOURS` | Rolling refill para planes activos |

**Acción si LOW skip**: aceptable >24h.

### HYGIENE — housekeeping

| job_id | Frecuencia | Función |
|---|---|---|
| `sweep_meal_plans_without_chunks` | Dom 03:30 UTC | P2-NEXT-3: marca abandoned planes sin chunks vivos >7d |
| `sweep_synthetic_test_plans` | Diario 03:45 UTC | P1-LIVE-3: cancela + abandona `Plan Sintético% — Test%` >24h |
| `sweep_stale_emit_locks_kv` | `MEALFIT_PLAN_QUALITY_EMIT_LOCK_SWEEP_MIN` (default 360min) | P2-NEW-16: DELETE `app_kv_store` rows `plan_quality_emit_lock:%` con `updated_at < NOW() - 24h` |
| `gc_orphan_chunk_telemetry` | `_ORPHAN_GC_INT` (default 6h) | GC rows huérfanas chunk_lesson_telemetry/chunk_deferrals (FK SET NULL) |
| `gc_orphan_conversation_summaries` | `_SUMMARIES_GC_INT` (default 12h) | P2-NEW-6: GC conversation_summaries huérfanas |
| `flush_pending_deferrals` | `_P16_INT` | Persiste deferrals pendientes |
| `flush_pending_lesson_telemetry` | `_P010_INT` | Persiste lesson telemetry pendiente |
| `sync_chunk_queue_tz_offsets` | `_TZ_SYNC_INT` | Sincroniza tz_offsets en plan_chunk_queue |

**Acción si HYGIENE skip**: aceptable indefinido (datos correctos, solo
crece backlog).

---

## P3-LIVE-12 · SOP `Plan Sintético` cleanup manual

Complementario al cron diario `_sweep_synthetic_test_plans` (P1-LIVE-3).
Usa este SOP cuando:

- El cron está MISSED por saturación del scheduler.
- QA inyectó >50 fixtures y necesitas limpieza inmediata.
- Necesitas limpiar fixtures con pattern distinto al estricto
  (`Plan Sintético% — Test%`).

### Paso 1 — detectar fixtures vivos

```sql
SELECT
  id,
  user_id,
  plan_data->>'name' AS name,
  plan_data->>'generation_status' AS status,
  created_at,
  (SELECT COUNT(*) FROM plan_chunk_queue q
   WHERE q.meal_plan_id = mp.id
     AND q.status IN ('pending', 'processing', 'stale')) AS alive_chunks
FROM meal_plans mp
WHERE plan_data->>'name' ILIKE 'Plan Sintético%'
   OR plan_data->>'name' ILIKE '%fixture%'
   OR plan_data->>'name' ILIKE '%test plan%'
ORDER BY created_at DESC;
```

Revisa la columna `alive_chunks`: si >0, son chunks que el worker
procesará. Cancelar primero.

### Paso 2 — backup defensivo a meal_plans_audit

```sql
INSERT INTO meal_plans_audit (meal_plan_id, user_id, action, plan_data_before, created_at)
SELECT id, user_id, 'manual_synthetic_cleanup_p3_live_12', plan_data, NOW()
FROM meal_plans
WHERE plan_data->>'name' ILIKE 'Plan Sintético%'  -- ajustar pattern según paso 1
  AND plan_data->>'generation_status' IS DISTINCT FROM 'abandoned';
```

### Paso 3 — cancelar chunks vivos

```sql
UPDATE plan_chunk_queue
SET status = 'cancelled', updated_at = NOW()
WHERE meal_plan_id IN (
  SELECT id FROM meal_plans
  WHERE plan_data->>'name' ILIKE 'Plan Sintético%'
)
  AND status IN ('pending', 'processing', 'stale')
  AND dead_lettered_at IS NULL
RETURNING id, meal_plan_id;
```

### Paso 4 — marcar planes abandoned via jsonb merge

```sql
UPDATE meal_plans
SET plan_data = plan_data || jsonb_build_object(
        'generation_status', 'abandoned',
        '_abandoned_at', NOW()::text,
        '_abandoned_reason', 'synthetic_fixture_manual_sop'
    ),
    updated_at = NOW()
WHERE plan_data->>'name' ILIKE 'Plan Sintético%'
  AND plan_data->>'generation_status' IS DISTINCT FROM 'abandoned'
RETURNING id, plan_data->>'name' AS name;
```

### Paso 5 — verificar

```sql
SELECT
  COUNT(*) FILTER (WHERE plan_data->>'generation_status' = 'abandoned') AS abandoned,
  COUNT(*) FILTER (WHERE plan_data->>'generation_status' IS DISTINCT FROM 'abandoned') AS still_alive
FROM meal_plans
WHERE plan_data->>'name' ILIKE 'Plan Sintético%';
```

`still_alive` debe ser 0. Si > 0, revisar el filtro pattern del UPDATE en
paso 4 (puede que algunos pasen el filtro de SELECT pero no del UPDATE
por race con escritor concurrente — rara, retry el UPDATE).

### Paso 6 — post-mortem si recurrente

Si la limpieza se ejecuta >3 veces por semana:

1. Identificar el inyector (probablemente test runner del equipo).
2. Considerar añadir validación server-side en `/analyze` que rechace
   nombres con sufijo `— Test` excepto para users en lista de QA
   documentada.
3. O escalar el cron diario a horario
   (`MEALFIT_SWEEP_SYNTHETIC_PLANS_AGE_HOURS=2`).
