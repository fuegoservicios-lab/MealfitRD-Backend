-- [P2-NEW-1 · 2026-05-10] Lockear `search_path` de la trigger function
-- `set_meal_plans_updated_at` para resolver el advisor de Supabase
-- `function_search_path_mutable` (WARN, post-audit 2026-05-10).
--
-- Causa raíz:
--   La migración original P0-2 (`p0_2_meal_plans_updated_at.sql`) creó la
--   función sin `SET search_path = ''`. Para una trigger function que solo
--   llama `NOW()` y asigna `NEW.updated_at`, el riesgo práctico es bajo
--   (no resuelve nombres de tablas/funciones desde el search_path), pero
--   el advisor lo flag como WARN y la convención de Supabase es lockear
--   el search_path en TODAS las funciones (defense-in-depth contra futuros
--   refactors que añadan referencias schema-qualified-implícitas).
--
-- Por qué `SET search_path = ''` (cadena vacía) en vez de `= 'public'`:
--   - `''` fuerza que cualquier referencia a tabla/función dentro del
--     body requiera schema qualifier explícito (`public.<obj>`).
--   - `'public'` permite resolver `<obj>` → `public.<obj>` pero queda
--     vulnerable a attacks por temp tables maliciosas (un usuario con
--     `CREATE` en un schema interceptable como `pg_temp` puede shadowear
--     un nombre).
--   - Para esta trigger function no hay diferencia funcional (solo usa
--     `NOW()` built-in que no toca search_path), así que aplicamos el
--     candado más estricto.
--
-- Idempotente: `CREATE OR REPLACE FUNCTION` sin alterar signature; el
-- trigger `trg_meal_plans_set_updated_at` referencia por nombre, así que
-- sigue válido sin DROP/RECREATE.
--
-- Tooltip-anchor: P2-NEW-1-START | test_p2_new_1_set_meal_plans_search_path_locked

BEGIN;

CREATE OR REPLACE FUNCTION public.set_meal_plans_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

COMMENT ON FUNCTION public.set_meal_plans_updated_at() IS
    '[P0-2 · 2026-05-10] Auto-actualiza meal_plans.updated_at en cada UPDATE. '
    'Trigger BEFORE UPDATE — no requiere que el caller incluya updated_at = NOW(). '
    'Defense-in-depth contra olvidos en mutators futuros. '
    '[P2-NEW-1 · 2026-05-10] search_path lockeado a '''' (cierra advisor '
    'function_search_path_mutable; el body solo usa NOW() built-in así que '
    'el lock es defensa preventiva sin cambio funcional).';

COMMIT;

-- Notificar PostgREST para que recargue el schema cache. Aunque cambio
-- function-only sin ALTER de columnas, mantenemos el patrón de P0-2 para
-- consistencia (no daña si no hay cambios visibles para PostgREST).
NOTIFY pgrst, 'reload schema';
