-- [P2-PROD-AUDIT-BUNDLE · 2026-05-27] Autovacuum tuning para `user_facts`.
--
-- Observación del audit prod-readiness 2026-05-27 (pg_stat_user_tables):
--   user_facts : 6 live / 20 dead (333.3% dead_pct)
--
-- Patrón de churn: DELETE side-effect en cada `search_user_facts*` /
-- `get_user_facts_by_metadata` ejecuta `delete_expired_temporal_facts(user_id)`
-- antes de cada read. Sumado al cleanup cron sobre `sintoma_temporal` (TTL 48h)
-- y los UPSERT de embeddings vectoriales, esta tabla acumula dead rows rápido
-- aunque el cardinal de live rows sea bajo.
--
-- Defaults globales (`autovacuum_vacuum_threshold=50 + scale_factor=0.2`)
-- requieren `dead > 50 + 0.2 * live` para disparar. Con 6 live → necesita
-- 51 dead rows para que autovacuum corra; el snapshot de hoy muestra 20
-- dead y subiendo lentamente.
--
-- Tuning P1-B (scale_factor=0.05, threshold=25) ya se aplicó a 7 tablas
-- chronic-UPDATE incluyendo `user_profiles`. Esta migración extiende el
-- mismo patrón a `user_facts` (la única tabla user-scoped del audit P1-B
-- que quedó fuera). Coherente con el playbook.
--
-- Por qué NO cambiar el knob del debounce P2-1 en lugar de tunear vacuum:
--   El debounce reduce DELETEs en N+1, pero el cron diario sobre
--   `sintoma_temporal` con TTL 48h sigue generando churn legítimo. El
--   debounce es per-user dentro del mismo proceso; el vacuum tuning ataca
--   el lado del bloat. Son complementarios, no alternativos.
--
-- Idempotente: `ALTER TABLE ... SET (...)` es UPSERT — re-aplicar la
-- migración no causa errores ni cambios extra.
--
-- Rollback (si bloquea producción): `ALTER TABLE public.user_facts RESET
-- (autovacuum_vacuum_scale_factor, autovacuum_vacuum_threshold,
-- autovacuum_analyze_scale_factor, autovacuum_analyze_threshold);`

ALTER TABLE public.user_facts SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_vacuum_threshold = 25,
  autovacuum_analyze_scale_factor = 0.05,
  autovacuum_analyze_threshold = 25
);

COMMENT ON TABLE public.user_facts IS
'[P2-PROD-AUDIT-BUNDLE · 2026-05-27] Autovacuum tuneado (scale_factor=0.05, '
'threshold=25). DELETEs frecuentes de `sintoma_temporal` (TTL 48h) + side-effect '
'cleanup en search paths (P2-1 debounce mitiga parcialmente) generan churn '
'sostenido sobre <50 filas vivas en estado típico. Defaults globales '
'esperaban 50+0.2*live = 51+ dead rows para autovacuum, dead_pct llegaba a '
'333% antes del primer VACUUM.';
