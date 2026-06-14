-- [P0-4 · 2026-05-10] RPC atómica `apply_inventory_delta` para deducciones
-- backend (consumed_meals → user_inventory) sin lost-update race.
--
-- Causa raíz (audit 2026-05-10):
--   [`backend/db_inventory.py:1085-1146`] `add_or_update_inventory_item` hacía
--   SELECT-MODIFY-WRITE en app-layer:
--     1. SELECT id, quantity, unit FROM user_inventory WHERE user_id+name.
--     2. for row: convert_amount(...); new_qty = row.quantity + converted_qty.
--     3. UPDATE SET quantity = new_qty (o DELETE si < 0.01).
--
--   Bajo concurrencia (2 chunks generando simultáneamente que deducen el
--   mismo ingrediente; o usuario loggeando 2× la misma comida porque el
--   frontend permite duplicado), dos threads pueden:
--     T1: SELECT qty=100
--     T2: SELECT qty=100  (antes que T1 commitee)
--     T1: UPDATE SET qty=90  (delta -10)
--     T2: UPDATE SET qty=85  (delta -15, computado desde 100, NO desde 90)
--   Final qty=85, real qty esperada=75. Se perdieron 10 unidades.
--
--   Tests existentes (`test_pantry_auto_sync_between_chunks.py`) mockean
--   y NO ejercen DB real → la race quedaba invisible. `failed_inventory_deductions`
--   tabla existe pero nadie escribe a ella (gap P0-5 separado).
--
-- Por qué una RPC nueva y no extender `increment_inventory_quantity`:
--   La RPC existente usa `auth.uid()` directamente — funciona desde frontend
--   autenticado pero NO desde backend (service_role JWT no setea claim). El
--   backend deduce vía `supabase.table().update()` directo, fuera del path
--   atomic. La nueva RPC toma `p_user_id` explícito + verifica ownership
--   (`WHERE id = p_row_id AND user_id = p_user_id`) en el SELECT — defense-
--   in-depth: aunque la RPC sea SECURITY DEFINER, no permite cross-user
--   sin que el caller adivine ID + user_id simultáneamente.
--
-- Garantías post-P0-4:
--   - `SELECT … FOR UPDATE` lockea la fila → concurrent calls serializan.
--   - DELETE atómico cuando qty resultante < 0.01 (mismo threshold legacy).
--   - Retorna jsonb con prev_qty/new_qty/status para observabilidad (no
--     fail-silent: el caller sabe si la operación se aplicó o cayó por
--     ownership/missing).
--   - GRANT solo a `service_role` — los callsites son crons + endpoints
--     backend, no frontend. La RPC `increment_inventory_quantity` sigue
--     siendo el path autenticado para usuarios.

BEGIN;

CREATE OR REPLACE FUNCTION public.apply_inventory_delta(
    p_user_id uuid,
    p_row_id bigint,
    p_delta numeric,
    p_mutation_type text DEFAULT 'consumption',
    p_master_id uuid DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
    v_row record;
    v_new_qty numeric;
BEGIN
    -- 1) Lock la fila objetivo. El `FOR UPDATE` serializa concurrent calls
    --    sobre la misma fila — segunda llamada espera al commit del primero.
    --    `user_id = p_user_id` es ownership check defense-in-depth (la RPC
    --    es SECURITY DEFINER, así que sin esto un caller que conozca
    --    `id` ajeno podría mutar fila de otro usuario).
    SELECT id, quantity, unit, source, user_id
      INTO v_row
      FROM public.user_inventory
     WHERE id = p_row_id
       AND user_id = p_user_id
       FOR UPDATE;

    IF NOT FOUND THEN
        -- No fail loud — el caller puede tener varios row candidatos
        -- (multiple units del mismo ingrediente) y reintentar otro.
        -- `failed_inventory_deductions` (P0-5) recibe el log si todos fallan.
        RETURN jsonb_build_object(
            'status', 'not_found',
            'row_id', p_row_id,
            'reason', 'row_missing_or_ownership_mismatch'
        );
    END IF;

    -- 2) Computar new_qty. Misma precisión (4 decimales) que el path legacy
    --    en `db_inventory.py:1118`. Sin GREATEST(0,...) porque el caller ya
    --    debe haber validado que delta no excede qty (P0-4 trabaja sobre
    --    delta arbitrario; el clamp-a-cero del path frontend
    --    `increment_inventory_quantity` es semántica distinta — UI permite
    --    spam de clicks).
    v_new_qty := ROUND(v_row.quantity + p_delta, 4);

    -- 3) DELETE si cae por debajo del threshold (mismo comportamiento legacy).
    IF v_new_qty < 0.01 THEN
        DELETE FROM public.user_inventory WHERE id = p_row_id;
        RETURN jsonb_build_object(
            'status', 'deleted',
            'row_id', p_row_id,
            'prev_qty', v_row.quantity,
            'delta', p_delta
        );
    END IF;

    -- 4) UPDATE preservando `source` (first-writer-wins per P0.2) y
    --    actualizando master_id solo si el caller lo provee.
    UPDATE public.user_inventory
       SET quantity = v_new_qty,
           master_ingredient_id = COALESCE(p_master_id, master_ingredient_id),
           last_mutation_type = p_mutation_type
     WHERE id = p_row_id;

    RETURN jsonb_build_object(
        'status', 'updated',
        'row_id', p_row_id,
        'prev_qty', v_row.quantity,
        'new_qty', v_new_qty,
        'delta', p_delta
    );
END;
$$;

COMMENT ON FUNCTION public.apply_inventory_delta(uuid, bigint, numeric, text, uuid) IS
    '[P0-4 · 2026-05-10] Aplicación atómica de delta sobre user_inventory '
    '(lock FOR UPDATE + UPDATE/DELETE en misma TX). SECURITY DEFINER + '
    'ownership check explícito porque backend service_role no setea '
    'auth.uid(). Usar increment_inventory_quantity (auth.uid() interno) '
    'para path frontend autenticado.';

-- 5) Lockdown de permisos: solo backend (service_role). Frontend usa la
--    RPC existente; PUBLIC/anon no deben poder llamar nunca.
REVOKE ALL ON FUNCTION public.apply_inventory_delta(uuid, bigint, numeric, text, uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.apply_inventory_delta(uuid, bigint, numeric, text, uuid) FROM anon;
REVOKE ALL ON FUNCTION public.apply_inventory_delta(uuid, bigint, numeric, text, uuid) FROM authenticated;
GRANT EXECUTE ON FUNCTION public.apply_inventory_delta(uuid, bigint, numeric, text, uuid) TO service_role;

COMMIT;
