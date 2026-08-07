-- [P1-PANTRY-RECONCILIATION · 2026-08-07] Extiende el ledger de consumo para
-- absorber la reconciliación periódica de la Nevera.
--
-- ─────────────────────────────────────────────────────────────────────────
-- QUÉ CIERRA
-- ─────────────────────────────────────────────────────────────────────────
--
-- La Nevera solo baja por lo que el usuario registra (esa es la regla del
-- producto, y es la correcta). Su consecuencia inevitable: lo que come sin
-- registrar NUNCA sale. A las 2-3 semanas la Nevera sobre-reporta, la lista de
-- compras sub-compra, y el usuario deja de creerle.
--
-- El arreglo NO es descontar automático — eso rompe la regla y devuelve el
-- sistema al problema que P1-PANTRY-NAME-RESOLUTION cerró (mover la Nevera por
-- algo que el usuario no puede auditar). El arreglo es PREGUNTAR: "estos items
-- no se han movido en N días, ¿los usaste, se dañaron, o siguen ahí?". La
-- reducción sigue exigiendo una acción humana; el sistema solo la hace barata.
--
-- ─────────────────────────────────────────────────────────────────────────
-- POR QUÉ EXTENDER EL LEDGER Y NO CREAR OTRA TABLA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `inventory_consumption_events` ya existe para contestar "¿qué movió mi
-- Nevera?". Una salida por reconciliación es exactamente eso. Una tabla
-- paralela obligaría a unir dos fuentes para responder la misma pregunta —
-- y es justo el patrón de duplicación que en este repo termina drifteando.
--
-- Dos valores nuevos, ambos en CHECKs existentes:
--
--   source  += 'reconciliation'
--   outcome += 'spoiled'
--
-- `spoiled` merece ser su propio outcome y no colapsarse con `deducted`: la
-- comida que se daña es información de COMPRA (comprar menos perecedero, o en
-- envase más chico), no de consumo. Colapsarlas hace imposible medir el
-- desperdicio, que es medio motivo por el que alguien lleva una nevera digital.
--
-- ─────────────────────────────────────────────────────────────────────────
-- REVERSIBILIDAD
-- ─────────────────────────────────────────────────────────────────────────
--
-- Los eventos de reconciliación nacen con `consumed_meal_id IS NULL`, y
-- `revert_consumption_events` busca POR `consumed_meal_id` — así que quedan
-- naturalmente fuera del "Deshacer registro" sin necesidad de un caso especial.
-- Es correcto: no hay ningún registro de diario que deshacer, y devolver
-- comida que el usuario declaró dañada la resucitaría.
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `DROP CONSTRAINT IF EXISTS` antes de `ADD CONSTRAINT` (patrón
-- P3-MIGRATION-IDEMPOTENCE-DOC: un CHECK no admite `IF NOT EXISTS`, así que se
-- recrea). Correr esto dos veces deja el mismo estado.

BEGIN;

ALTER TABLE public.inventory_consumption_events
    DROP CONSTRAINT IF EXISTS inventory_consumption_events_outcome_chk;

ALTER TABLE public.inventory_consumption_events
    ADD CONSTRAINT inventory_consumption_events_outcome_chk
    CHECK (outcome IN ('deducted', 'inferred', 'not_in_pantry', 'failed', 'spoiled'));

ALTER TABLE public.inventory_consumption_events
    DROP CONSTRAINT IF EXISTS inventory_consumption_events_source_chk;

ALTER TABLE public.inventory_consumption_events
    ADD CONSTRAINT inventory_consumption_events_source_chk
    CHECK (source IN ('chat', 'photo', 'plan_meal', 'chunk_reconcile',
                      'agent_tool', 'reconciliation', 'unknown'));

COMMENT ON COLUMN public.inventory_consumption_events.outcome IS
    'deducted = bajó la Nevera. inferred = bajó, con cantidad inferida por '
    'P1-PANTRY-INFER. spoiled = el usuario declaró que se dañó (P1-PANTRY-'
    'RECONCILIATION) — sale de la Nevera igual que deducted pero es señal de '
    'COMPRA, no de consumo, y colapsarlas haría imposible medir desperdicio. '
    'not_in_pantry = el usuario no tenía ese item registrado (NO movió nada). '
    'failed = error al aplicar. Revertir not_in_pantry/failed crearía comida '
    'inexistente.';

-- Sanity: los dos valores nuevos deben ser aceptados y uno inventado rechazado.
DO $$
DECLARE
    v_ok BOOLEAN;
BEGIN
    SELECT pg_get_constraintdef(oid) LIKE '%spoiled%' INTO v_ok
      FROM pg_constraint
     WHERE conname = 'inventory_consumption_events_outcome_chk';
    IF NOT COALESCE(v_ok, FALSE) THEN
        RAISE EXCEPTION 'P1-PANTRY-RECONCILIATION sanity: outcome CHECK no admite spoiled';
    END IF;

    SELECT pg_get_constraintdef(oid) LIKE '%reconciliation%' INTO v_ok
      FROM pg_constraint
     WHERE conname = 'inventory_consumption_events_source_chk';
    IF NOT COALESCE(v_ok, FALSE) THEN
        RAISE EXCEPTION 'P1-PANTRY-RECONCILIATION sanity: source CHECK no admite reconciliation';
    END IF;
END $$;

COMMIT;
