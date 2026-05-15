-- [P0-3] Backfill _plan_start_date y grocery_start_date para planes legacy.
--
-- Antes: planes creados antes del helper `_ensure_grocery_start_date`
-- (db_plans.py:192-217) o cuando el pipeline LLM olvidaba persistir
-- `_plan_start_date` quedaban con `plan_data->>'_plan_start_date' IS NULL`.
-- La cascada de fallback en `_resolve_chunk_start_anchor` (cron_tasks.py)
-- intenta:
--   1. snapshot.form_data._plan_start_date  (path normal — ausente)
--   2. user_profiles.health_profile.grocery_start_date  (puede faltar)
--   3. meal_plans.plan_data.grocery_start_date  (puede faltar)
--   4. meal_plans.created_at o hardcoded "8am UTC"  (último recurso)
--
-- Para usuarios en TZ no-UTC, el step 4 (hardcoded) desalineaba hasta 24h: un
-- chunk programado para "mañana 8am UTC" disparaba en realidad a las 4am
-- locales (UTC-4) o a las 8pm anteriores (UTC+12).
--
-- Esta migración rellena ambos campos a nivel de fila desde `meal_plans.created_at`
-- en UTC — equivalente al step 4 pero estabilizado: el chunk worker ahora siempre
-- encuentra un anchor en el step 1 o 3 sin caer al hardcoded final.
--
-- Idempotente: el WHERE filtra solo rows con NULL, así que re-ejecutar es noop.
-- Limitada a `created_at > NOW() - INTERVAL '60 days'` para no tocar planes
-- antiguos que ya no se usan (no merece el bloat de índice).
--
-- [P0.2] Esta migración es la fuente única de verdad para el backfill.
-- Antes existía además un helper Python (`_backfill_plan_anchors_oneshot` en
-- cron_tasks.py) que replicaba estos UPDATEs al startup del scheduler,
-- gobernado por `BACKFILL_PLAN_ANCHORS_DONE`. Ese path fue eliminado:
-- dependía de una variable de entorno frágil de configurar en deploy y
-- duplicaba la lógica con drift de comentarios. Las migraciones SQL son
-- ahora el único path; aplicar vía `supabase db push` o pegando este
-- archivo en el SQL editor de Supabase.

-- 1. Backfill `_plan_start_date` (formato ISO completo con TZ, igual al
--    pipeline `routers/plans.py` que produce `start_date_iso` para planes
--    nuevos).
UPDATE meal_plans
   SET plan_data = jsonb_set(
       COALESCE(plan_data, '{}'::jsonb),
       '{_plan_start_date}',
       to_jsonb(to_char(created_at AT TIME ZONE 'UTC',
                        'YYYY-MM-DD"T"HH24:MI:SS"+00:00"')),
       true
   )
 WHERE (plan_data->>'_plan_start_date') IS NULL
   AND created_at IS NOT NULL
   AND created_at > NOW() - INTERVAL '60 days';

-- 2. Backfill `grocery_start_date` (formato date-only `YYYY-MM-DD`, igual al
--    helper `db_plans._ensure_grocery_start_date` que se aplica a planes nuevos).
UPDATE meal_plans
   SET plan_data = jsonb_set(
       COALESCE(plan_data, '{}'::jsonb),
       '{grocery_start_date}',
       to_jsonb(to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD')),
       true
   )
 WHERE (plan_data->>'grocery_start_date') IS NULL
   AND created_at IS NOT NULL
   AND created_at > NOW() - INTERVAL '60 days';

-- Verificación post-backfill (no falla si quedan NULLs fuera de la ventana 60d).
-- Útil para que el operador confirme manualmente:
--
--   SELECT COUNT(*) AS missing_psd
--     FROM meal_plans
--    WHERE (plan_data->>'_plan_start_date') IS NULL
--      AND created_at > NOW() - INTERVAL '60 days';
--   -- → debe devolver 0 tras la migración.
--
-- NOTA sobre CHECK constraint: el plan original P0-3 sugería agregar
--   ALTER TABLE meal_plans
--     ADD CONSTRAINT meal_plans_plan_data_anchor_required
--     CHECK ((plan_data->>'_plan_start_date') IS NOT NULL);
-- Lo dejamos OPCIONAL: si un INSERT futuro vía path no-canónico (admin SQL,
-- import de datos, fixture de test) olvida el campo, el constraint rompería el
-- INSERT en tiempo de prod. Aplicar después de un ciclo de telemetría sin
-- nuevos NULLs reportados.
