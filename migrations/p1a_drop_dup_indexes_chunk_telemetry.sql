-- P1-A (2026-05-07): drop duplicate indexes en chunk_lesson_telemetry
-- Advisor performance reportó 2 pares idénticos:
--   idx_chunk_lesson_telemetry_event_recent ≡ idx_chunk_lesson_telemetry_event_created_at
--   idx_chunk_lesson_telemetry_user_recent  ≡ idx_chunk_lesson_telemetry_user_created_at
-- Ambos USING btree con mismas columnas y orden. Preservamos los `_created_at`
-- (nombre canónico). Idempotente con IF EXISTS — re-run safe.
-- Aplicada a producción (mpoodlmnzaeuuazsazbj) vía MCP el 2026-05-07.

DROP INDEX IF EXISTS public.idx_chunk_lesson_telemetry_event_recent;
DROP INDEX IF EXISTS public.idx_chunk_lesson_telemetry_user_recent;
