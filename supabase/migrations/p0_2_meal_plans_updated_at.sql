-- [P0-2 · 2026-05-10] Agrega columna `updated_at` a `meal_plans` con trigger
-- BEFORE UPDATE que la auto-mantiene.
--
-- Causa raíz (audit 2026-05-10):
--   `meal_plans` se diseñó originalmente sin `updated_at`. Múltiples sites
--   asumen su existencia, todos rotos en silencio o ruidosamente:
--
--   1. [`backend/routers/plans.py:4141`] `retry-chunk` ejecuta
--      `UPDATE meal_plans SET plan_data = jsonb_set(...), updated_at = NOW()`
--      → 500 al usuario cada retry (excepción capturada por el except
--      superior, devuelve `safe_error_detail`).
--
--   2. Cron `_aggregate_coherence_block_history_metrics` históricamente
--      pegaba `?updated_at=gte=<cutoff>` vía PostgREST → 400 horario en
--      logs API desde el merge de P3-B (2026-05-08). Cierre P0-OBS-1
--      cambió el filtro a `created_at` como workaround, pero la
--      intención original (capturar regeneraciones de planes viejos) se
--      perdió.
--
--   3. Cualquier query futura que ordene/filtre por "última modificación"
--      cae al jsonb `plan_data->>_plan_modified_at` (string sort,
--      no-indexable) — toda la suite Historial (P1-HIST-4) lo hace
--      client-side post-fetch porque no había alternativa indexada.
--
-- Por qué columna física + trigger (vs. estandarizar a `_plan_modified_at`
-- jsonb):
--   - 5+ callsites ya asumen `updated_at` (retry-chunk, cron P3-B histórico,
--     queries PostgREST exploratorias).
--   - Columna física permite índice B-tree para window queries (`gte`/orden
--     por modificación) — el jsonb path requiere expression index sobre
--     `(plan_data->>'_plan_modified_at')::timestamptz` que es más frágil.
--   - Trigger BEFORE UPDATE garantiza consistencia sin requerir que cada
--     callsite recuerde el `SET updated_at = NOW()` — defense-in-depth
--     contra futuros olvidos.
--
-- Compatible con `plan_data._plan_modified_at` existente: ambos coexisten
-- (jsonb sigue siendo SSOT para semántica "última edición de contenido del
-- plan"; columna física es timestamp de cualquier UPDATE). Si en el futuro
-- se decide unificar, este es el lado más fácil de drop.

BEGIN;

-- 1) Agregar la columna con default seguro.
ALTER TABLE public.meal_plans
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- 2) Backfill defensivo: las filas existentes obtienen `updated_at = created_at`
--    (lo más cercano a "última modificación conocida"; el default NOW() ya
--    cubre filas nuevas pero podría dar timestamps engañosos a filas viejas
--    si la migración corre tras el deploy).
UPDATE public.meal_plans
SET updated_at = created_at
WHERE updated_at IS NULL OR updated_at = NOW();
-- Nota: el segundo predicado captura filas que recibieron `NOW()` como
-- default literal en el mismo segundo de la migración. No es perfecto bajo
-- carga concurrente, pero el plan en producción tiene ~decenas de filas y la
-- ventana de error es <1s.

-- 3) NOT NULL ahora que todas las filas tienen valor.
ALTER TABLE public.meal_plans
    ALTER COLUMN updated_at SET NOT NULL;

-- 4) Trigger function: mantiene `updated_at` sin requerir que cada UPDATE
--    explícitamente lo setee. SECURITY INVOKER por default (correcto:
--    corre con privilegios del caller).
CREATE OR REPLACE FUNCTION public.set_meal_plans_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

COMMENT ON FUNCTION public.set_meal_plans_updated_at() IS
    '[P0-2 · 2026-05-10] Auto-actualiza meal_plans.updated_at en cada UPDATE. '
    'Trigger BEFORE UPDATE — no requiere que el caller incluya updated_at = NOW(). '
    'Defense-in-depth contra olvidos en mutators futuros.';

-- 5) Trigger BEFORE UPDATE — corre por fila, no por statement, para cubrir
--    UPDATE-multi-row (ej. backfills batch).
DROP TRIGGER IF EXISTS trg_meal_plans_set_updated_at ON public.meal_plans;
CREATE TRIGGER trg_meal_plans_set_updated_at
    BEFORE UPDATE ON public.meal_plans
    FOR EACH ROW
    EXECUTE FUNCTION public.set_meal_plans_updated_at();

-- 6) Índice para window queries (cron P3-B podrá volver a usar updated_at
--    en lugar de created_at, capturando regeneraciones de planes viejos).
--    No CONCURRENTLY porque estamos dentro de transacción + tabla chica
--    en producción.
CREATE INDEX IF NOT EXISTS idx_meal_plans_user_updated_at
    ON public.meal_plans (user_id, updated_at DESC);

COMMENT ON INDEX public.idx_meal_plans_user_updated_at IS
    '[P0-2 · 2026-05-10] Cubre window queries del cron P3-B '
    '(_aggregate_coherence_block_history_metrics) y futuro ordenamiento del '
    'Historial por última modificación física. Reemplaza el patrón '
    'client-side de ordenar por plan_data->>_plan_modified_at.';

COMMENT ON COLUMN public.meal_plans.updated_at IS
    '[P0-2 · 2026-05-10] Timestamp del último UPDATE físico. Mantenido por '
    'trigger trg_meal_plans_set_updated_at. Distinto de '
    'plan_data->>_plan_modified_at (SSOT semántico para "última edición de '
    'contenido del plan" — settable explícitamente por mutators que sí '
    'cambian el plan vs. mutators de metadata interna como '
    '_anchor_recovery_attempts).';

COMMIT;

-- 7) Notificar PostgREST para refrescar su schema cache. Sin esto, las
--    queries `?updated_at=gte=...` siguen devolviendo 400 hasta el próximo
--    restart natural del worker PostgREST (~minutos).
NOTIFY pgrst, 'reload schema';
