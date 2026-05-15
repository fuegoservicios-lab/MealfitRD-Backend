-- [P1-B · 2026-05-12] Autovacuum tuning para 7 tablas chronic-UPDATE.
--
-- Auditoría comprehensiva 2026-05-12 (pg_stat_user_tables snapshot):
--   app_kv_store        : 4 live  / 40 dead (90.9% dead_pct)
--   plan_chunk_queue    : 6 live  / 49 dead (89.1%)
--   meal_plans          : 3 live  / 17 dead (85.0%)
--   agent_sessions      : 3 live  / 14 dead (82.4%)
--   plan_chunk_metrics  : 4 live  / 9  dead (69.2%)
--   user_profiles       : 4 live  / 7  dead (63.6%)
--   system_alerts       : 33 live / 21 dead (38.9%)
--
-- Por qué los defaults globales son insuficientes:
--   autovacuum_vacuum_threshold=50 + scale_factor=0.2 significa que
--   autovacuum solo se dispara cuando `dead > 50 + 0.2 * live`. Para
--   tablas pequeñas y UPDATE-heavy (CB heartbeats cada 60s sobre
--   app_kv_store, status transitions sobre plan_chunk_queue,
--   alert_key upserts sobre system_alerts), las filas vivas son <10
--   pero las muertas se acumulan a ritmo de cron. El threshold global
--   de 50 hace que autovacuum espere demasiado y la ratio dead/live
--   se dispare a >80% sin que VACUUM corra.
--
--   Bajando a threshold=25 + scale_factor=0.05, autovacuum corre cada
--   ~25-30 UPDATEs sobre estas tablas, manteniendo el bloat bajo
--   control sin penalizar throughput (estas tablas son pequeñas, el
--   VACUUM termina en <50ms cada vez).
--
-- Por qué NO incluir pipeline_metrics:
--   pipeline_metrics es INSERT-heavy (poco UPDATE). 706 live / 137 dead
--   (16% — saludable). Su autovacuum solo necesita `analyze` agresivo
--   para que el planner tenga estadísticas frescas — se tuneará por
--   separado si las queries del cron _aggregate_coherence_block_history
--   muestran scan plans lentos en producción.
--
-- Idempotente: ALTER TABLE ... SET (storage_parameter = value) es UPSERT.
-- Re-aplicar la migración es no-op.
--
-- Rollback (si bloquea producción): reemplazar valores explícitos por
-- `RESET (autovacuum_vacuum_scale_factor, autovacuum_vacuum_threshold,
-- autovacuum_analyze_scale_factor, autovacuum_analyze_threshold)`.

ALTER TABLE public.app_kv_store SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.plan_chunk_queue SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.meal_plans SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.agent_sessions SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.plan_chunk_metrics SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.user_profiles SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

ALTER TABLE public.system_alerts SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

COMMENT ON TABLE public.app_kv_store IS
'[P1-B · 2026-05-12] Autovacuum tuneado (scale_factor=0.05, threshold=25). '
'CB heartbeats + KV upserts cada 60s generan dead rows rápido sobre <5 '
'filas vivas; defaults globales esperan 50 dead = dead_pct ~90%.';

COMMENT ON TABLE public.plan_chunk_queue IS
'[P1-B · 2026-05-12] Autovacuum tuneado (scale_factor=0.05, threshold=25). '
'Status transitions (pending→processing→completed/failed/cancelled) y FK '
'CASCADE deletes generan churn alto sobre <50 filas vivas en estado típico.';

COMMENT ON TABLE public.system_alerts IS
'[P1-B · 2026-05-12] Autovacuum tuneado (scale_factor=0.05, threshold=25). '
'Upsert por alert_key + resolved_at PATCH (auto-resolve sweeps) genera '
'UPDATE constante. Sin tuning, dead rows acumulaban con cada autoheal tick.';
