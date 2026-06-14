-- [P2-USER-DEPLETED-ITEMS-FK-IDX · 2026-05-23] Covering index para el FK
-- `user_depleted_items_master_ingredient_id_fkey` (→ master_ingredients(id)).
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTOR CERRADO
-- ─────────────────────────────────────────────────────────────────────────
--
-- Supabase Performance Advisor (audit production-readiness 2026-05-23):
--   "Table `public.user_depleted_items` has a foreign key
--    `user_depleted_items_master_ingredient_id_fkey` without a covering
--    index. This can lead to suboptimal query performance."
--
-- Tabla creada en P3-DEPLETED-BD · 2026-05-22 con 2 unique partial indexes
-- para dedupe (master_id NOT NULL vs lower(name) cuando NULL), pero ningún
-- índice solo sobre `master_ingredient_id` para el FK. Modo de fallo
-- preventivo:
--
--   Cada DELETE/UPDATE sobre `master_ingredients` (que tiene FK
--   ON DELETE SET NULL hacia esta tabla, ver
--   `p3_user_depleted_items_2026_05_22.sql`) hace un seq-scan de
--   `user_depleted_items` para localizar las filas dependientes. Con la
--   tabla creciendo (docenas de items × cada user activo), el seq-scan se
--   vuelve dominante en latencia de operaciones sobre `master_ingredients`.
--
-- ─────────────────────────────────────────────────────────────────────────
-- CICLO DEL ADVISOR (pre-fix vs post-fix)
-- ─────────────────────────────────────────────────────────────────────────
--
-- Pre-migración: advisor reporta `unindexed_foreign_keys` (vector arriba).
-- Post-migración: el índice existe pero el advisor lo reporta como
-- `unused_index` porque PostgreSQL aún no observa scans (tabla joven,
-- ningún DELETE sobre master_ingredients todavía). Este es exactamente
-- el mismo modo de fallo del advisor `unused_index` que afecta a los
-- otros 5 índices documentados en CLAUDE.md "Advisors aceptados →
-- Performance": el advisor NO observa uso interno por FK. Por eso, una
-- vez aplicado, este índice TAMBIÉN se documenta en esa tabla — su lugar
-- correcto post-apply es bajo el rationale "cubre FK, advisor ciego al
-- uso interno". El COMMENT abajo refleja esta dualidad.
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `CREATE INDEX IF NOT EXISTS` es nativo idempotente. COMMENT ON INDEX
-- es siempre overwrite. Sanity check post-apply verifica existencia.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_user_depleted_items_master_ingredient_id
    ON public.user_depleted_items (master_ingredient_id)
    WHERE master_ingredient_id IS NOT NULL;

COMMENT ON INDEX public.idx_user_depleted_items_master_ingredient_id IS
    'P2-USER-DEPLETED-ITEMS-FK-IDX 2026-05-23: covering index para FK '
    'user_depleted_items_master_ingredient_id_fkey (ON DELETE SET NULL '
    'desde master_ingredients). Partial WHERE master_ingredient_id IS NOT NULL '
    'porque las filas con master_id NULL (items manuales sin canonicalizar) '
    'no participan del CASCADE. Pre-fix cerraba advisor unindexed_foreign_keys; '
    'post-fix aparece como unused_index (advisor ciego al uso interno por FK) '
    '— mismo modo de fallo que los 5 unused_index aceptados en CLAUDE.md '
    'sección "Advisors aceptados → Performance". Lección P2-PERF-1: '
    'NO dropear basado solo en el advisor.';

-- Sanity check post-apply.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'user_depleted_items'
          AND indexname = 'idx_user_depleted_items_master_ingredient_id'
    ) THEN
        RAISE EXCEPTION 'P2-USER-DEPLETED-ITEMS-FK-IDX sanity: '
            'idx_user_depleted_items_master_ingredient_id NO se creó';
    END IF;
END;
$$;

COMMIT;
