-- [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] Índice parcial para la consulta caliente
-- "alertas actualmente activas" sobre `system_alerts`.
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTOR
-- ─────────────────────────────────────────────────────────────────────────
--
-- Múltiples crons consultan `WHERE resolved_at IS NULL` en CADA tick del
-- scheduler (listener de cascada, sweeps de alerts huérfanas, subqueries de
-- "alerta activa") — ver backend/cron_tasks.py. Hoy la tabla es pequeña
-- (~67 filas) → seq-scan barato, pero el patrón corre con altísima frecuencia
-- y la tabla crece con cada incidente. Un índice parcial sobre las filas
-- ACTIVAS (resolved_at IS NULL) mantiene la consulta O(activas) en lugar de
-- O(total) a medida que el histórico de alertas resueltas se acumula.
--
-- Partial WHERE resolved_at IS NULL: solo indexa las filas vivas (las
-- resueltas son la mayoría a largo plazo y no participan de la consulta hot).
-- Ordenado por triggered_at para servir también los ORDER BY triggered_at.
--
-- El advisor `unused_index` puede reportarlo hasta que el planner lo elija;
-- mismo modo de fallo que los índices aceptados en CLAUDE.md.
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `CREATE INDEX IF NOT EXISTS` nativo idempotente. COMMENT siempre overwrite.
-- Sanity check post-apply verifica existencia.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_system_alerts_active
    ON public.system_alerts (triggered_at)
    WHERE resolved_at IS NULL;

COMMENT ON INDEX public.idx_system_alerts_active IS
    'P2-PROD-AUDIT-FOLLOWUP 2026-05-28: índice parcial para la consulta hot '
    '"alertas activas" (WHERE resolved_at IS NULL) que corre en cada tick del '
    'scheduler. Mantiene el costo O(activas) en lugar de O(total) cuando el '
    'histórico de alertas resueltas se acumula. Puede reportarse como '
    'unused_index hasta que el planner lo elija (advisor ciego, lección P2-PERF-1).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'system_alerts'
          AND indexname = 'idx_system_alerts_active'
    ) THEN
        RAISE EXCEPTION 'P2-PROD-AUDIT-FOLLOWUP sanity: '
            'idx_system_alerts_active NO se creó';
    END IF;
END;
$$;

COMMIT;
