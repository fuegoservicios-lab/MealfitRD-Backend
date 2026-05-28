-- [P1-PROD-AUDIT-BUNDLE · 2026-05-28] Índice BRIN sobre `pipeline_metrics.created_at`
-- para soportar el cron de retención age-based `_purge_old_pipeline_metrics`
-- (tooltip-anchor P1-PIPELINE-METRICS-RETENTION en backend/cron_tasks.py).
--
-- ─────────────────────────────────────────────────────────────────────────
-- POR QUÉ BRIN (no B-tree)
-- ─────────────────────────────────────────────────────────────────────────
--
-- `pipeline_metrics` es append-only: ~40 callsites INSERT, ~1.5–1.9k filas/día
-- (cada tick de cron emite un heartbeat). Audit production-readiness 2026-05-28:
-- la tabla tenía 25.467 filas / 14 MB y crecía sin retención (~550k filas/año).
-- El único índice previo era `idx_pipeline_metrics_node_created (node, created_at
-- DESC)`, que NO sirve bien a un DELETE puro por `created_at < cutoff`.
--
-- Como las filas se insertan en orden temporal monótono (append-only), los
-- bloques físicos quedan naturalmente correlacionados con `created_at` → BRIN
-- es el índice ideal: ~kilobytes en disco (vs MB de un B-tree) y resuelve el
-- range-scan `created_at < NOW() - interval` del cron de retención sin write
-- amplification en cada INSERT.
--
-- El advisor `unused_index` puede reportarlo hasta que el cron de retención
-- ejecute su primer DELETE — mismo modo de fallo que los índices aceptados en
-- CLAUDE.md "Advisors aceptados → Performance".
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `CREATE INDEX IF NOT EXISTS` es nativo idempotente. COMMENT ON INDEX es
-- siempre overwrite. Sanity check post-apply verifica existencia.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_pipeline_metrics_created_brin
    ON public.pipeline_metrics USING brin (created_at);

COMMENT ON INDEX public.idx_pipeline_metrics_created_brin IS
    'P1-PROD-AUDIT-BUNDLE 2026-05-28: BRIN sobre created_at para el cron de '
    'retención age-based _purge_old_pipeline_metrics (knob '
    'MEALFIT_PIPELINE_METRICS_RETENTION_DAYS, default 30). Tabla append-only '
    'monótona → BRIN tiny y eficiente para el range-scan created_at < cutoff. '
    'Puede aparecer como unused_index hasta el primer DELETE de retención '
    '(advisor ciego, lección P2-PERF-1: NO dropear basado solo en el advisor).';

-- Sanity check post-apply.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'pipeline_metrics'
          AND indexname = 'idx_pipeline_metrics_created_brin'
    ) THEN
        RAISE EXCEPTION 'P1-PROD-AUDIT-BUNDLE sanity: '
            'idx_pipeline_metrics_created_brin NO se creó';
    END IF;
END;
$$;

COMMIT;
