-- [P2-HIST-AUDIT-3 · 2026-05-09] Índice compuesto
-- `(user_id, created_at DESC)` en `meal_plans`.
--
-- Motivación (audit historial 2026-05-08):
--   El listado del Historial (`api_plans_history_list`) y el SELECT
--   target del restore (`api_restore_plan`) hacen `ORDER BY
--   GREATEST(created_at, COALESCE((plan_data->>'_plan_modified_at')
--   ::timestamptz, created_at)) DESC, created_at DESC`. EXPLAIN
--   reveló:
--     Limit
--       -> Sort  (Sort Key: GREATEST(...) DESC, created_at DESC)
--         -> Index Scan using idx_meal_plans_user_id
--   El `Index Scan` resuelve el filtro `user_id = ?`, pero el sort
--   queda en memoria (no aprovecha el índice para reducir el costo
--   ordenamiento).
--
-- Diseño elegido (índice compuesto simple `(user_id, created_at DESC)`):
--   - Los rows ya vienen pre-ordenados por created_at DESC dentro
--     del bucket del user. El sort por GREATEST(...) reordena solo
--     cuando un plan tiene `_plan_modified_at` posterior a su
--     created_at — minoría en cualquier dataset realista. Postgres
--     usa insertion sort para los pocos elementos out-of-order.
--   - Queries simples `WHERE user_id=? ORDER BY created_at DESC`
--     (Dashboard "recientes", paths legacy) ahora usan el índice
--     directamente sin Sort step.
--   - Compatibilidad: el índice existente `idx_meal_plans_user_id`
--     se preserva (drop separado si en futuro el planner siempre
--     elige el compuesto — el simple no aporta nada que el compuesto
--     no cubra). Mantenerlo no rompe nada y permite rollback fácil.
--
-- Trade-off documentado:
--   El sort por GREATEST(...) NO se elimina con este índice (la
--   expresión involucra `(plan_data->>'_plan_modified_at')::timestamptz`
--   que no es IMMUTABLE — un expression index requeriría una función
--   wrapper IMMUTABLE, que solo es seguro si TODOS los call sites
--   sellan ISO con offset explícito). Dejamos puerta abierta para
--   esa optimización (opción B) cuando el dataset alcance volúmenes
--   donde el sort en memoria sea bottleneck (>10K planes per user).
--
-- Patrón consistente con `idx_chunk_lesson_telemetry_user_created_at`
-- y `chunk_user_locks_heartbeat_at_idx` que ya existen como índices
-- compuestos en otras tablas chunk_*.
--
-- Idempotencia: `CREATE INDEX IF NOT EXISTS` — re-aplicar la
-- migración no rompe.

CREATE INDEX IF NOT EXISTS idx_meal_plans_user_created_at
ON public.meal_plans
USING btree (user_id, created_at DESC);

-- COMMENT documental para diagnóstico SQL futuro.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_class
        WHERE relname = 'idx_meal_plans_user_created_at'
    ) THEN
        EXECUTE 'COMMENT ON INDEX public.idx_meal_plans_user_created_at IS ''[P2-HIST-AUDIT-3] Índice compuesto user_id + created_at DESC para acelerar listado del Historial y SELECT target del restore. Pre-ordena rows dentro del bucket del user para reducir el costo del sort GREATEST(...) en api_plans_history_list y api_restore_plan.''';
    END IF;
END $$;
