-- [P1-HIST-NEW-7 · 2026-05-09] Recrear `idx_chunk_lesson_telemetry_plan_week`.
--
-- Causa raíz:
--   La migración `p1_3_chunk_lesson_telemetry_table.sql` (20260503024130)
--   declara este índice cubriendo la FK `chunk_lesson_telemetry.meal_plan_id`,
--   pero el advisor de Supabase reporta `unindexed_foreign_keys` para esa FK
--   porque el índice está AUSENTE en producción. Verificación:
--     SELECT indexname FROM pg_indexes WHERE tablename='chunk_lesson_telemetry';
--   Solo devuelve `_pkey`, `_event_created_at`, `_user_created_at`. No hay
--   `DROP INDEX … plan_week` en migrations/, así que la causa más
--   probable es que cuando p1_3 corrió, la tabla ya existía (creada por el
--   runtime DDL antiguo de app.py:lifespan, anterior a P2-NEW-E que lo
--   consolidó al SSOT migrations) y un drop manual posterior se aplicó
--   directo en producción sin SOT en source.
--
-- Justificación:
--   Lo usa el endpoint `/{plan_id}/lifetime-lessons` (plans.py:5010+) y el
--   cron `_record_chunk_lesson_telemetry` (cron_tasks.py:9478+) cuando
--   filtran por `meal_plan_id`. Sin el índice, seq-scan de toda la tabla;
--   tolerable hoy (telemetría liviana) pero crece linealmente con cada chunk
--   completado en producción. Cubrir la FK también acelera el ON DELETE
--   SET NULL aplicado por p0_hist_3_telemetry_orphan_fk.sql.
--
-- Idempotente con IF NOT EXISTS — re-run safe.

CREATE INDEX IF NOT EXISTS idx_chunk_lesson_telemetry_plan_week
    ON public.chunk_lesson_telemetry (meal_plan_id, week_number);

COMMENT ON INDEX public.idx_chunk_lesson_telemetry_plan_week IS
    '[P1-HIST-NEW-7 · 2026-05-09] Cubre FK chunk_lesson_telemetry_meal_plan_id_fkey + sirve query de /lifetime-lessons (plans.py:5010+) filtrando por (meal_plan_id, week_number).';
