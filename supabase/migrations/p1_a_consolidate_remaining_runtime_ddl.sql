-- ============================================================================
-- [P1-A · 2026-05-08] Consolidación residual de DDL standalone al SSOT.
-- ----------------------------------------------------------------------------
-- Tras P2-NEW-E (push_subscriptions, system_alerts, pipeline_metrics,
-- plan_chunk_queue, plan_chunk_metrics, chunk_deferrals,
-- chunk_lesson_telemetry, ALTERs user_profiles/user_inventory) y
-- P2-NEW-G (consolidación 2ª ronda + stub _ensure_quality_alert_schema),
-- aún quedaban 2 scripts standalone con DDL ejecutable fuera del SSOT:
--
--   - `backend/add_price_cols.py`       → master_ingredients.price_per_lb
--                                         master_ingredients.price_per_unit
--                                         + UPDATEs hardcoded de pricing
--                                         (RD$ por libra/unidad para 6 items)
--
--   - `backend/migrate_subscriptions.py` → user_profiles.paypal_subscription_id
--                                          user_profiles.paypal_plan_id
--                                          user_profiles.subscription_status
--
-- Verificado vía MCP execute_sql que las 5 columnas YA existen en producción
-- (alguien corrió los scripts a mano hace tiempo). Esta migración formaliza
-- el schema en el SSOT para que un greenfield deploy las cree sin depender
-- de los scripts standalone que se renombran a `_deprecated_*.py.bak` en
-- el mismo P-fix. Idempotente: NOOP sobre el schema actual de producción.
--
-- Los UPDATE de pricing del script original NO se replican aquí: pricing
-- semilla es data-fix, no schema-fix. Si un environment greenfield necesita
-- precios iniciales, se ejecuta como seed manual.
-- ============================================================================

-- ============================================================
-- 1. master_ingredients: columnas de pricing
-- ============================================================
ALTER TABLE master_ingredients
    ADD COLUMN IF NOT EXISTS price_per_lb NUMERIC DEFAULT 0;

ALTER TABLE master_ingredients
    ADD COLUMN IF NOT EXISTS price_per_unit NUMERIC DEFAULT 0;

COMMENT ON COLUMN master_ingredients.price_per_lb IS
    'Precio por libra (RD$). Origen histórico: backend/add_price_cols.py (ahora deprecated). Consolidado a SSOT por P1-A 2026-05-08.';

COMMENT ON COLUMN master_ingredients.price_per_unit IS
    'Precio por unidad (RD$). Origen histórico: backend/add_price_cols.py (ahora deprecated). Consolidado a SSOT por P1-A 2026-05-08.';

-- ============================================================
-- 2. user_profiles: columnas de subscripción PayPal
-- ============================================================
ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS paypal_subscription_id TEXT;

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS paypal_plan_id TEXT;

ALTER TABLE user_profiles
    ADD COLUMN IF NOT EXISTS subscription_status TEXT;

COMMENT ON COLUMN user_profiles.paypal_subscription_id IS
    'ID de subscripción activa de PayPal. Origen histórico: backend/migrate_subscriptions.py (ahora deprecated). Consolidado a SSOT por P1-A 2026-05-08.';

COMMENT ON COLUMN user_profiles.paypal_plan_id IS
    'ID del plan PayPal (Pro mensual/anual). Origen histórico: backend/migrate_subscriptions.py (ahora deprecated). Consolidado a SSOT por P1-A 2026-05-08.';

COMMENT ON COLUMN user_profiles.subscription_status IS
    'Estado: active|cancelled|past_due|trial. Origen histórico: backend/migrate_subscriptions.py (ahora deprecated). Consolidado a SSOT por P1-A 2026-05-08.';
