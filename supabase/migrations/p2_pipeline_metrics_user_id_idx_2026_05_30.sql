-- [P1-PROD-AUDIT-2 · 2026-05-30] Índice B-tree parcial sobre
-- `pipeline_metrics (user_id, created_at DESC)` para el hot-path de generación.
--
-- ─────────────────────────────────────────────────────────────────────────
-- POR QUÉ (verificado con EXPLAIN ANALYZE en vivo durante el audit 2026-05-30)
-- ─────────────────────────────────────────────────────────────────────────
--
-- `graph_orchestrator.py` (preflight meta-learning, rama `if not auto_adjusted:`)
-- ejecuta en CADA generación de plan, para usuarios sin `pipeline_score_history`
-- reciente:
--     SELECT node, duration_ms, confidence, metadata
--     FROM pipeline_metrics WHERE user_id = %s
--     ORDER BY created_at DESC LIMIT 20;
--
-- EXPLAIN ANALYZE en vivo: **Seq Scan, 29.330 filas removidas por filtro,
-- 713 ms** — degradación real OCURRIENDO en el hot-path user-facing, no
-- hipotética. Los únicos índices previos eran `node_created (node, created_at)`
-- (no sirve a un filtro puro por user_id) y `created_brin` (range temporal).
-- La retención BRIN acota a ~50k filas steady-state (~1.2 s), no elimina el scan.
--
-- `pipeline_metrics.user_id` es type **TEXT** (no uuid) — el índice es sobre la
-- columna text. La mayoría de filas de telemetría tienen `user_id = NULL`
-- (heartbeats, rate_limiter_cleanup, etc.); el índice PARCIAL `WHERE user_id IS
-- NOT NULL` cubre solo las filas per-usuario (las que la query filtra) → índice
-- pequeño + el advisor `unused_index` no lo flageará espuriamente (cubre un
-- `WHERE user_id` directo, no una FK).
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
-- `CREATE INDEX IF NOT EXISTS` nativo idempotente. COMMENT siempre overwrite.
-- Sanity check post-apply verifica existencia.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_user_id_created
    ON public.pipeline_metrics (user_id, created_at DESC)
    WHERE user_id IS NOT NULL;

COMMENT ON INDEX public.idx_pipeline_metrics_user_id_created IS
    'P1-PROD-AUDIT-2 2026-05-30: B-tree parcial (user_id, created_at DESC) WHERE '
    'user_id IS NOT NULL. Sirve el SELECT del preflight meta-learning '
    '(graph_orchestrator.py, rama not auto_adjusted) que corría como Seq Scan de '
    '713ms / 29k filas en CADA generación. Partial porque la mayoría de filas son '
    'telemetría con user_id NULL.';

-- Sanity check post-apply.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'pipeline_metrics'
          AND indexname = 'idx_pipeline_metrics_user_id_created'
    ) THEN
        RAISE EXCEPTION 'P1-PROD-AUDIT-2 sanity: idx_pipeline_metrics_user_id_created NO se creó';
    END IF;
END;
$$;

COMMIT;
