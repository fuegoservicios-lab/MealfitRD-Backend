-- [P2-NEW-G · 2026-05-08] Segunda ronda de consolidación de runtime DDL al SSOT.
-- P2-NEW-E cubrió 7 tablas + ALTERs sobre user_profiles/user_inventory. Esta
-- migración cierra dos columnas residuales:
--
--   1. plan_chunk_metrics.is_rolling_refill BOOLEAN
--      - Antes: cron_tasks.py::_record_chunk_metric ejecutaba
--        `ALTER TABLE plan_chunk_metrics ADD COLUMN IF NOT EXISTS is_rolling_refill ...`
--        en cada inserción de métrica (línea 10518). Idempotente pero violaba SSOT
--        (el mismo bug estructural que P1-NEW-A para índices).
--      - Ahora: la columna vive aquí; el bloque runtime queda removido.
--
--   2. plan_chunk_metrics.pantry_snapshot_age_hours NUMERIC
--      - Drift implícito: la columna se INSERTaba (cron_tasks.py:10546-10553) pero
--        nunca aparecía en ninguna migración. Solo existía en producción gracias a
--        un ALTER manual aplicado vía SQL editor en su día (P0-3). Este archivo la
--        formaliza al SSOT para que cualquier branch/dev DB nuevo la reciba.
--
-- Migración idempotente: NOOP sobre el schema actual de producción.
-- ============================================================================

ALTER TABLE plan_chunk_metrics
    ADD COLUMN IF NOT EXISTS is_rolling_refill BOOLEAN DEFAULT FALSE;

ALTER TABLE plan_chunk_metrics
    ADD COLUMN IF NOT EXISTS pantry_snapshot_age_hours NUMERIC;

COMMENT ON COLUMN plan_chunk_metrics.is_rolling_refill IS
    'TRUE si la métrica corresponde a un chunk de rolling_refill (no plan inicial). '
    'Permite separar tasas de degradación por tipo de chunk en el cron '
    '_alert_chunk_quality_degradation.';

COMMENT ON COLUMN plan_chunk_metrics.pantry_snapshot_age_hours IS
    '[P0-3] Edad del snapshot de pantry al momento del pickup, en horas. '
    'Permite detectar planes 30d ejecutando con snapshots de 20+ días donde el '
    'LLM puede generar platos con ingredientes ya consumidos.';
