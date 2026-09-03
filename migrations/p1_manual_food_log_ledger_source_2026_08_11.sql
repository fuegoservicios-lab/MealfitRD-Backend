-- [P1-MANUAL-FOOD-LOG · 2026-08-11] `source='manual'` en el ledger de la Nevera.
-- (copia IDÉNTICA en migrations/ del workspace-root — P3-MIGRATIONS-SSOT, mismo commit)
--
-- POR QUÉ EXISTE. El registro manual de comida (componedor del diario) descuenta de la
-- Nevera y anota cada descuento en `inventory_consumption_events` con su origen. Sin
-- esta fila en el CHECK, el INSERT del ledger viola la constraint, db_inventory se lo
-- traga con un logger.warning (el DESCUENTO sí se aplica) y «Deshacer registro» deja de
-- poder devolver esa comida a la Nevera. En silencio. Para siempre. Es la asimetría
-- exacta que el ledger nació para cerrar (P1-CONSUMPTION-LEDGER), reintroducida por una
-- fila que falta en un CHECK.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): DROP IF EXISTS antes de ADD, sanity con
-- RAISE EXCEPTION. Mismo patrón que p1_pantry_reconciliation_2026_08_07.sql, que es la
-- versión anterior de ESTA MISMA constraint — se re-declara completa, no se parchea.
BEGIN;

ALTER TABLE public.inventory_consumption_events
    DROP CONSTRAINT IF EXISTS inventory_consumption_events_source_chk;

ALTER TABLE public.inventory_consumption_events
    ADD CONSTRAINT inventory_consumption_events_source_chk
    CHECK (source IN ('chat', 'photo', 'plan_meal', 'chunk_reconcile',
                      'agent_tool', 'reconciliation', 'manual', 'unknown'));

DO $$
DECLARE v_ok BOOLEAN;
BEGIN
    SELECT pg_get_constraintdef(oid) LIKE '%manual%' INTO v_ok
      FROM pg_constraint WHERE conname = 'inventory_consumption_events_source_chk';
    IF NOT COALESCE(v_ok, FALSE) THEN
        RAISE EXCEPTION 'P1-MANUAL-FOOD-LOG sanity: el CHECK de source no admite manual';
    END IF;
END $$;

COMMIT;
