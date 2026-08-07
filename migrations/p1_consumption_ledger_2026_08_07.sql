-- [P1-CONSUMPTION-LEDGER · 2026-08-07] Tabla `inventory_consumption_events`:
-- registro reversible de cada descuento de la Nevera.
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTOR CERRADO
-- ─────────────────────────────────────────────────────────────────────────
--
-- `DELETE /api/diary/consumed/{meal_id}` (P1-DIARY-EDITABLE) borra la fila del
-- diario pero NO devuelve la comida a la Nevera. La asimetría es visible para
-- el usuario y erosiona la confianza más rápido que cualquier error de
-- estimación:
--
--   registra "2 huevos"  → diario +1 fila, Nevera 3 → 1
--   deshace el registro  → diario -1 fila, Nevera SIGUE EN 1
--
-- Y no se podía arreglar sin esta tabla: para devolver hay que saber QUÉ se
-- descontó, y eso se perdía en el momento en que `add_or_update_inventory_item`
-- aplicaba el delta. El string original ("2 huevos") no basta — la resolución
-- de nombre (P1-PANTRY-NAME-RESOLUTION) puede haberlo mapeado a la fila
-- "Huevo", y la inferencia de porción (P1-PANTRY-INFER) puede haber inventado
-- la cantidad. Re-parsear el string al revertir repetiría ambas decisiones y
-- podría llegar a otra respuesta.
--
-- Fix: cada descuento deja un evento con el nombre YA RESUELTO y la cantidad
-- YA APLICADA. Revertir es leer el evento y sumar — no volver a interpretar.
--
-- ─────────────────────────────────────────────────────────────────────────
-- DISEÑO
-- ─────────────────────────────────────────────────────────────────────────
--
-- - `consumed_meal_id` SIN foreign key, a propósito. La fila de
--   `consumed_meals` es borrable por el usuario; un FK CASCADE borraría el
--   registro de una devolución que SÍ ocurrió, y un FK RESTRICT impediría el
--   propio DELETE que este ledger existe para soportar. La integridad que
--   importa aquí es el rastro, no la referencia.
--
-- - `outcome` distingue lo que MOVIÓ la Nevera de lo que no. Es la columna
--   crítica del revert: `not_in_pantry` y `failed` nunca descontaron nada, así
--   que devolverlos CREARÍA comida que el usuario nunca tuvo. Solo
--   `deducted`/`inferred` son reversibles.
--
-- - `reverted_at` en vez de borrar la fila: hace el revert idempotente (un
--   segundo DELETE no vuelve a sumar) y conserva la historia de que hubo un
--   descuento y una devolución. Un ledger que se borra a sí mismo no es un
--   ledger.
--
-- - `quantity > 0` por CHECK: los eventos registran MAGNITUD; el signo lo pone
--   la operación (descuento resta, revert suma). Permitir negativos abriría la
--   puerta a un evento que al revertirse restara más.
--
-- - `source` no es decorativo: cuando una nevera no cuadra, la primera
--   pregunta es "¿qué la movió?", y las superficies tienen fiabilidades muy
--   distintas (el plan trae cantidades exactas, la foto las estima, el chat
--   las adivina desde texto libre).
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- `IF NOT EXISTS` en CREATE TABLE/INDEX, `DROP POLICY IF EXISTS` antes de
-- CREATE POLICY, y `DO $$ ... RAISE EXCEPTION` de sanity al final.
-- Patrón P3-MIGRATION-IDEMPOTENCE-DOC.

BEGIN;

CREATE TABLE IF NOT EXISTS public.inventory_consumption_events (
    id                BIGSERIAL PRIMARY KEY,
    -- `public.user_profiles`, NO `auth.users`: el schema `auth` es de Supabase y
    -- no existe en Neon (ver el bloque "SIN RLS" abajo — mismo error, misma
    -- causa). Es la convención de las tablas user-scoped nacidas ya en Neon,
    -- p.ej. `user_taste_events` (2026-07-02).
    user_id           UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    -- Sin FK a consumed_meals — ver DISEÑO.
    consumed_meal_id  UUID NULL,
    source            TEXT NOT NULL,
    -- Nombre YA RESUELTO contra la Nevera (la ortografía de la fila real),
    -- no el string que emitió la LLM.
    ingredient_name   TEXT NOT NULL,
    quantity          NUMERIC NOT NULL CHECK (quantity > 0),
    unit              TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    reverted_at       TIMESTAMP WITH TIME ZONE NULL,
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT inventory_consumption_events_outcome_chk
        CHECK (outcome IN ('deducted', 'inferred', 'not_in_pantry', 'failed')),
    CONSTRAINT inventory_consumption_events_source_chk
        CHECK (source IN ('chat', 'photo', 'plan_meal', 'chunk_reconcile', 'agent_tool', 'unknown'))
);

COMMENT ON TABLE public.inventory_consumption_events IS
    'P1-CONSUMPTION-LEDGER · 2026-08-07. Un evento por ingrediente por descuento '
    'de la Nevera, con el nombre YA RESUELTO y la cantidad YA APLICADA — para '
    'que "Deshacer registro" pueda devolver la comida sin re-interpretar el '
    'string original (cuya resolución de nombre e inferencia de porción podrían '
    'dar otra respuesta la segunda vez). Solo outcome deducted/inferred movieron '
    'la Nevera y por tanto son reversibles.';

COMMENT ON COLUMN public.inventory_consumption_events.outcome IS
    'deducted = bajó la Nevera. inferred = bajó, con cantidad inferida por '
    'P1-PANTRY-INFER. not_in_pantry = el usuario no tenía ese item registrado '
    '(NO movió nada). failed = error al aplicar. Revertir not_in_pantry/failed '
    'crearía comida inexistente.';

COMMENT ON COLUMN public.inventory_consumption_events.reverted_at IS
    'NOT NULL = ya se devolvió a la Nevera. Hace el revert idempotente: un '
    'segundo DELETE del mismo meal no vuelve a sumar.';

-- Lookup del revert: eventos vivos de un meal concreto. Parcial porque los ya
-- revertidos nunca se vuelven a leer por este camino.
CREATE INDEX IF NOT EXISTS idx_ice_consumed_meal_pending
    ON public.inventory_consumption_events (consumed_meal_id)
    WHERE consumed_meal_id IS NOT NULL AND reverted_at IS NULL;

-- Auditoría "¿qué movió mi Nevera?": eventos del user, más recientes primero.
CREATE INDEX IF NOT EXISTS idx_ice_user_created_at_desc
    ON public.inventory_consumption_events (user_id, created_at DESC);

-- SIN RLS, y no es un olvido.
--
-- La primera versión de esta migración traía `ENABLE ROW LEVEL SECURITY` + una
-- policy `USING ((select auth.uid()) = user_id)`, copiada de las migraciones
-- viejas del repo. Reventó al aplicarse:
--
--     psycopg.errors.InvalidSchemaName: schema "auth" does not exist
--
-- `auth.uid()` y el rol `authenticated` son construcciones de **Supabase**, que
-- este repo eliminó por completo en P1-NEON-DB-MIGRATION (2026-06-12). En Neon
-- no existe ese schema, así que la policy no protegía nada: hacía la migración
-- IMPOSIBLE de aplicar, y como todo va dentro de BEGIN/COMMIT, tumbaba la tabla
-- entera con ella.
--
-- Y aunque el schema existiera, la policy sobraría. RLS defiende contra un
-- cliente que abre conexión a la base; en esta arquitectura no hay ninguno:
-- PostgREST está prohibido (CLAUDE.md), el backend habla por `execute_sql_*` y
-- el frontend va por endpoints de `routers/user_data.py`. La frontera de
-- seguridad de este ledger es el `AND user_id = %s` de la invariante I2, igual
-- que en las otras tablas user-scoped nacidas ya en Neon — `user_taste_events`
-- (2026-07-02) y `ai_training_consent` (2026-07-04) tampoco llevan RLS.
--
-- Si algún día se expone lectura directa desde el cliente, ESE es el momento de
-- añadir RLS, con el mecanismo de identidad que exista entonces.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'inventory_consumption_events'
    ) THEN
        RAISE EXCEPTION 'P1-CONSUMPTION-LEDGER sanity: tabla NO se creó';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = 'idx_ice_consumed_meal_pending'
    ) THEN
        RAISE EXCEPTION 'P1-CONSUMPTION-LEDGER sanity: falta el índice del revert';
    END IF;
END $$;

COMMIT;
