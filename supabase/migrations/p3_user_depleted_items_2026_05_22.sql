-- [P3-DEPLETED-BD · 2026-05-22] Tabla `user_depleted_items` para persistir
-- el estado de items "agotados" del Pantry en BD (cross-device sync).
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTOR CERRADO
-- ─────────────────────────────────────────────────────────────────────────
--
-- Pre-fix: el state de `depletedItems` vivía solo en
-- `localStorage.mealfit_depleted_items` (Pantry.jsx::_persistDepleted).
-- Limitación verificada 2026-05-22 tras P3-AGENT-DEPLETE:
--
--   - User agota desde mobile → no aparece en desktop.
--   - Tab privado/incognito pierde el state.
--   - Limpieza de localStorage borra los agotados.
--   - El backend (chat agent) podía marcar items como agotados (P3-AGENT-DEPLETE)
--     pero dependía de que el cliente abra la app para hacer el merge.
--
-- Fix: tabla `user_depleted_items` con RLS owner-only + endpoints REST +
-- realtime channel. Migration one-shot desde localStorage en frontend al
-- primer mount post-deploy preserva el estado pre-existente.
--
-- ─────────────────────────────────────────────────────────────────────────
-- DISEÑO
-- ─────────────────────────────────────────────────────────────────────────
--
-- - `user_id` NOT NULL — los agotados solo aplican a usuarios autenticados
--   (sesiones guest no usan pantry persistente).
-- - FK CASCADE a `auth.users(id)` — patrón canónico del repo
--   (`user_inventory`, `consumed_meals`, `user_facts`).
-- - `master_ingredient_id` opcional con FK SET NULL — si el master se elimina
--   (raro pero posible), el agotado queda como entry "huérfano" identificable
--   por `ingredient_name` que el user todavía puede restockear manualmente.
-- - Dedupe: unique partial index sobre `(user_id, master_ingredient_id)` cuando
--   master_ingredient_id IS NOT NULL + unique sobre `(user_id, lower(trim(ingredient_name)))`
--   cuando NULL. El frontend puede tener entries de ambos tipos durante la
--   migración del localStorage (algunos items pre-existentes no tienen master).
-- - RLS policies wrap `auth.uid()` en `(select auth.uid())` para que postgres
--   compute el InitPlan una vez por query (patrón P1-RLS-INITPLAN del 2026-05-20).
-- - 2 índices: el unique partial ya cubre lookup por master + name; añadir
--   `idx_user_depleted_items_user_depleted_at_desc` para el listado del UI
--   ordenado por más reciente.
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- Toda la migration es idempotente vía `IF NOT EXISTS` (CREATE TABLE / INDEX /
-- ADD COLUMN), `DROP POLICY IF EXISTS` antes de CREATE POLICY, y `DO $$ ...
-- RAISE EXCEPTION` sanity al final. Patrón P3-MIGRATION-IDEMPOTENCE-DOC.

BEGIN;

CREATE TABLE IF NOT EXISTS public.user_depleted_items (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    master_ingredient_id UUID NULL REFERENCES public.master_ingredients(id) ON DELETE SET NULL,
    ingredient_name TEXT NOT NULL,
    quantity        NUMERIC NOT NULL DEFAULT 1,
    unit            TEXT NOT NULL DEFAULT 'unidad',
    category        TEXT NULL,
    shelf_life_days INTEGER NULL,
    depleted_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.user_depleted_items IS
    'P3-DEPLETED-BD · 2026-05-22. Items que el usuario marcó como agotados '
    '(desde Pantry UI o chat agent vía modify_pantry_inventory[items_to_deplete]). '
    'Sustituye el localStorage.mealfit_depleted_items pre-existente que no '
    'sincronizaba cross-device. RLS owner-only via auth.uid(). El restock '
    'desde la lista de compras DELETE-ea la fila aquí + re-inserta en user_inventory.';

-- Dedupe cuando hay master: 1 row max por (user, master).
CREATE UNIQUE INDEX IF NOT EXISTS uniq_user_depleted_items_user_master
    ON public.user_depleted_items (user_id, master_ingredient_id)
    WHERE master_ingredient_id IS NOT NULL;

-- Dedupe cuando NO hay master: 1 row max por (user, lower(trim(name))).
CREATE UNIQUE INDEX IF NOT EXISTS uniq_user_depleted_items_user_name_nomaster
    ON public.user_depleted_items (user_id, (lower(trim(ingredient_name))))
    WHERE master_ingredient_id IS NULL;

-- Sirve el listado del UI: items del user ordenados por más reciente.
CREATE INDEX IF NOT EXISTS idx_user_depleted_items_user_depleted_at_desc
    ON public.user_depleted_items (user_id, depleted_at DESC);

-- RLS owner-only.
ALTER TABLE public.user_depleted_items ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "user_depleted_items_select_own" ON public.user_depleted_items;
CREATE POLICY "user_depleted_items_select_own" ON public.user_depleted_items
    FOR SELECT TO authenticated
    USING ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS "user_depleted_items_insert_own" ON public.user_depleted_items;
CREATE POLICY "user_depleted_items_insert_own" ON public.user_depleted_items
    FOR INSERT TO authenticated
    WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS "user_depleted_items_update_own" ON public.user_depleted_items;
CREATE POLICY "user_depleted_items_update_own" ON public.user_depleted_items
    FOR UPDATE TO authenticated
    USING ((select auth.uid()) = user_id)
    WITH CHECK ((select auth.uid()) = user_id);

DROP POLICY IF EXISTS "user_depleted_items_delete_own" ON public.user_depleted_items;
CREATE POLICY "user_depleted_items_delete_own" ON public.user_depleted_items
    FOR DELETE TO authenticated
    USING ((select auth.uid()) = user_id);

-- Sanity post-migration: tabla existe + RLS enabled + 4 policies + 3 índices.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'user_depleted_items'
    ) THEN
        RAISE EXCEPTION 'P3-DEPLETED-BD sanity: tabla user_depleted_items NO se creó';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        WHERE c.relname = 'user_depleted_items' AND c.relrowsecurity = true
    ) THEN
        RAISE EXCEPTION 'P3-DEPLETED-BD sanity: RLS no enabled en user_depleted_items';
    END IF;
    IF (
        SELECT COUNT(*) FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'user_depleted_items'
    ) < 4 THEN
        RAISE EXCEPTION 'P3-DEPLETED-BD sanity: <4 policies en user_depleted_items';
    END IF;
END;
$$;

COMMIT;
