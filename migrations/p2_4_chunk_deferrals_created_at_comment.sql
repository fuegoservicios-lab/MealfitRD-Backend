-- [P2-4 · 2026-05-10] COMMENT operacional sobre `chunk_deferrals.created_at`.
--
-- Causa raíz (audit 2026-05-10):
--   La migración `20260503231237_rename_chunk_deferrals_deferred_at_to_created_at`
--   renombró la columna pero NO documentó la semántica resultante. El INSERT
--   actual en `cron_tasks.py:11904` usa
--   `VALUES (..., COALESCE(%s::timestamptz, NOW()))`
--   donde `%s` es `rec.get("buffered_at")` — el timestamp in-memory del momento
--   en que el defer event se buffereó. Si el buffer record carece de
--   `buffered_at` (legacy records pre-fix), el COALESCE cae a NOW() → la
--   columna recibe TIME DE FLUSH en lugar de TIME DE BUFFER.
--
--   Resultado bajo carga normal: `created_at` = buffer time (correcto, dado
--   que los buffer writes actuales SIEMPRE setean `buffered_at` —
--   ver cron_tasks.py:12372 y 12279). Resultado bajo data drift (legacy
--   records sin `buffered_at` en disco): `created_at` = flush time, sesgando
--   analytics que asumen "tiempo del evento original".
--
--   GC (`_gc_orphan_chunk_telemetry`, cron_tasks.py:~1566) filtra por
--   `created_at < NOW() - interval` asumiendo que es "edad del defer event".
--   Para records legacy con fallback NOW(), la edad reportada es 0 — el GC
--   los preserva más tiempo del semánticamente correcto. Trade-off
--   aceptable (mejor preservar que purgar prematuramente), pero el operador
--   debe saber.
--
-- Por qué COMMENT y no rename de la columna a `buffered_at`:
--   Otra rename rompería los call sites de SELECT + el cron de GC + el
--   test `test_p1_3_chunk_deferrals_telemetry.py`. El COMMENT documenta
--   la semántica sin tocar el contrato API. Si en el futuro se decide
--   migrar a un timestamp de evento estricto, la decisión y los pasos
--   quedan trazados.

BEGIN;

COMMENT ON COLUMN public.chunk_deferrals.created_at IS
    '[P2-4 · 2026-05-10] Timestamp del evento original de defer (cuando el '
    'temporal_gate rechazó el chunk in-memory), NO necesariamente el time '
    'del INSERT. Persistido vía `COALESCE(rec.buffered_at, NOW())` en '
    '`cron_tasks._flush_pending_deferrals`. Bajo flujo normal el valor '
    'refleja el momento del defer (los buffer writes setean `buffered_at` '
    'en `_record_chunk_deferral` y `_record_chunk_lesson_telemetry`). El '
    'fallback NOW() solo aplica para records legacy en disco que '
    'predataban el fix — caso degradado documentado, no error.';

COMMIT;
