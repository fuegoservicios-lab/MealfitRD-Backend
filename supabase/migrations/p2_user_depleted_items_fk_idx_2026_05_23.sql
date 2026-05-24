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
-- POR QUÉ NO VA EN LA TABLA DE "ADVISORS ACEPTADOS"
-- ─────────────────────────────────────────────────────────────────────────
--
-- Los otros 5 `unused_index` listados como aceptados en CLAUDE.md cubren
-- el caso INVERSO: índices existentes que el advisor cataloga como
-- "unused" porque no observa uso interno por FK CASCADE. Acá es lo
-- contrario: el advisor reporta `unindexed_foreign_keys` — el índice
-- legítimamente NO existe y debe crearse, no aceptarse.
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
    'no participan del CASCADE. Cierra advisor unindexed_foreign_keys del '
    'audit production-readiness 2026-05-23. NO confundir con los 5 '
    'unused_index aceptados en CLAUDE.md — esos cubren el caso inverso.';

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
