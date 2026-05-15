-- [P2-PERF-2 · 2026-05-12] Autovacuum tuning para tablas LangGraph checkpoint*.
--
-- Audit production-readiness 2026-05-11 (MCP snapshot):
--   checkpoints        : 61 live  / 0 dead  | last_autovacuum 2026-03-23 (50d)
--   checkpoint_blobs   : 148 live / 0 dead  | last_autovacuum 2026-03-23 (50d)
--   checkpoint_writes  : 290 live / 0 dead  | last_autovacuum 2026-03-23 (50d)
--
-- Estas 3 tablas son escritas por el persistence layer del LangGraph
-- (`AsyncPostgresSaver` / `PostgresSaver`) cada vez que el grafo persiste
-- un checkpoint: ~1 row en `checkpoints` + N rows en `checkpoint_blobs`
-- + M rows en `checkpoint_writes` por turno de chat agent / pipeline run.
--
-- Por qué autovacuum corrió hace 50 días:
--   Defaults globales (scale_factor=0.2 + threshold=50) requieren que
--   `dead > 50 + 0.2 * live` para disparar. Con `live=61` (checkpoints)
--   el umbral es `50 + 12 = 62` dead rows. Pero la app DELETE/UPDATEa
--   checkpoints raramente (LangGraph mantiene historia, no purga
--   agresiva); el flujo dominante es INSERT con UPDATE ocasional.
--   El `last_autovacuum 2026-03-23` cuadra con esa cadencia: el último
--   ciclo fue tras un GC anterior.
--
-- Problema potencial bajo carga:
--   Sin tuning, cuando el chat agent escala (más turnos de chat /
--   pipeline reruns), `checkpoint_writes` crecerá rápido y los UPDATEs
--   de status (LangGraph reordena/marca checkpoints) acumularán dead
--   rows. La query plan del `psycopg_pool` del checkpointer se degrada
--   si las estadísticas están stale 50+ días — el planner asume
--   distribuciones que no reflejan la realidad.
--
-- Por qué NO usar exactamente los mismos parámetros que P1-B:
--   P1-B (7 tablas hot) puso threshold=25 + scale_factor=0.05 porque
--   esas son UPDATE-heavy puras sobre <50 filas. Las checkpoint_*
--   están dominadas por INSERT (no UPDATE/DELETE). Threshold=25 sería
--   agresivo de más para INSERT (autovacuum corre tras cada lote
--   pequeño; thrash). Compromiso: threshold=50 (default) +
--   scale_factor=0.1 (mitad del default 0.2) — autovacuum corre
--   ~2x más frecuente que default, sin saturar I/O en startup-bursts.
--
-- Idempotente: ALTER TABLE ... SET (storage_parameter = value) es UPSERT.
-- Re-aplicar la migración es no-op (pg_class.reloptions ya tiene los
-- valores).
--
-- Rollback (si penaliza throughput de pipeline runs):
--   ALTER TABLE public.checkpoints RESET (
--     autovacuum_vacuum_scale_factor,
--     autovacuum_vacuum_threshold,
--     autovacuum_analyze_scale_factor,
--     autovacuum_analyze_threshold);
--   (repetir para checkpoint_blobs y checkpoint_writes).
--
-- Verificación post-deploy:
--   SELECT s.relname, c.reloptions, s.last_autovacuum, s.last_analyze
--   FROM pg_stat_user_tables s JOIN pg_class c ON c.oid = s.relid
--   WHERE s.relname IN ('checkpoints','checkpoint_blobs','checkpoint_writes');
--
-- NOTA: `checkpoint_migrations` queda con defaults (10 rows estáticos,
-- no merece tuning).

ALTER TABLE public.checkpoints SET (
  autovacuum_vacuum_scale_factor = 0.1,
  autovacuum_vacuum_threshold = 50,
  autovacuum_analyze_scale_factor = 0.1,
  autovacuum_analyze_threshold = 50
);

ALTER TABLE public.checkpoint_blobs SET (
  autovacuum_vacuum_scale_factor = 0.1,
  autovacuum_vacuum_threshold = 50,
  autovacuum_analyze_scale_factor = 0.1,
  autovacuum_analyze_threshold = 50
);

ALTER TABLE public.checkpoint_writes SET (
  autovacuum_vacuum_scale_factor = 0.1,
  autovacuum_vacuum_threshold = 50,
  autovacuum_analyze_scale_factor = 0.1,
  autovacuum_analyze_threshold = 50
);

COMMENT ON TABLE public.checkpoints IS
'[P2-PERF-2 · 2026-05-12] Autovacuum tuneado (scale_factor=0.1, threshold=50). '
'Tabla LangGraph persistence. INSERT-heavy con UPDATE ocasional; defaults '
'globales esperaban 50+0.2*live dead rows — autovacuum corría 1×/50d.';

COMMENT ON TABLE public.checkpoint_blobs IS
'[P2-PERF-2 · 2026-05-12] Autovacuum tuneado (scale_factor=0.1, threshold=50). '
'Storage de payloads serializados de checkpoints. Crece linealmente con '
'turnos de chat agent / pipeline runs.';

COMMENT ON TABLE public.checkpoint_writes IS
'[P2-PERF-2 · 2026-05-12] Autovacuum tuneado (scale_factor=0.1, threshold=50). '
'Writes individuales por step del grafo; el más grande de los 3.';
