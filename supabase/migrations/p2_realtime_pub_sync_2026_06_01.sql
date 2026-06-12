-- [P2-REALTIME-PUB-SYNC · 2026-06-01] Sincroniza la publicación
-- `supabase_realtime` con las suscripciones REALES del frontend. Auditoría de
-- velocidad (Supabase MCP, 2026-06-01): el lector WAL de Realtime era ~86% del
-- tiempo de ejecución de la DB y descartaba el 99.94% de los eventos
-- procesados (3.8M llamadas → 2.455 filas entregadas). Causa raíz: la
-- publicación estaba desalineada del consumo real en dos puntos.
--
-- ─────────────────────────────────────────────────────────────────────────
-- (1) AÑADE `meal_plans` — suscripción muerta (latente)
-- ─────────────────────────────────────────────────────────────────────────
-- El canal `meal-plan-chunk-updates`
-- (frontend/src/context/AssessmentContext.jsx:1170, `table: 'meal_plans'`,
-- `event: 'UPDATE'`, `filter: user_id=eq.<uid>`) se suscribía exitosamente
-- pero `meal_plans` NO estaba en la publicación → `postgres_changes` NUNCA
-- entregaba eventos. Las semanas generadas en background (chunking) solo
-- aparecían vía el fallback PESADO: canal `user_profiles` →
-- `refreshProfileAndPlan` = refetch REST completo de perfil + plan, + polling
-- (visibilitychange/focus/history-status-summary). Publicar activa el
-- push-merge LIGERO de líneas 1187-1215 (solo fusiona `days` en el state
-- local). meal_plans usa REPLICA IDENTITY DEFAULT (PK) — suficiente para
-- filtrar UPDATE por `user_id` (la fila NEW lleva todas las columnas).
-- Bug/patrón idéntico a P3-DEPLETED-BD-REALTIME-FIX (2026-05-22, que añadió
-- user_depleted_items por la misma razón).
--
-- ─────────────────────────────────────────────────────────────────────────
-- (2) QUITA `custom_shopping_items` — publicada sin consumidor + FULL
-- ─────────────────────────────────────────────────────────────────────────
-- `custom_shopping_items` estaba en la publicación CON `REPLICA IDENTITY
-- FULL` (la única tabla FULL del schema) pero NINGÚN cliente la consume vía
-- Realtime — cero `.channel`/`postgres_changes` en el frontend; solo se usa
-- por REST en el backend (`get_custom_shopping_items`, etc.). El lector WAL
-- procesaba todos sus cambios (~7.5k/ventana) para cero consumidores, y FULL
-- además logueaba la fila vieja COMPLETA en cada UPDATE/DELETE. Despublicar +
-- REPLICA IDENTITY DEFAULT elimina ese desperdicio. Si un futuro feature se
-- suscribe, re-añadir con el mismo patrón DO $$ (ver punto 1).
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA  (P3-MIGRATION-IDEMPOTENCE-DOC)
-- ─────────────────────────────────────────────────────────────────────────
-- ALTER PUBLICATION ADD/DROP TABLE NO tiene IF [NOT] EXISTS nativo → wrap en
-- DO $$ con check pre-apply (NOT EXISTS / EXISTS). REPLICA IDENTITY DEFAULT
-- es naturalmente idempotente (no-op si ya es default). Sanity DO $$ al final.

BEGIN;

-- (1) Publicar meal_plans (ADD solo si falta).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'meal_plans'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.meal_plans;
        RAISE NOTICE 'P2-REALTIME-PUB-SYNC: meal_plans añadida a supabase_realtime';
    ELSE
        RAISE NOTICE 'P2-REALTIME-PUB-SYNC: meal_plans YA estaba en supabase_realtime (idempotente noop)';
    END IF;
END;
$$;

-- (2a) Despublicar custom_shopping_items (DROP solo si presente).
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'custom_shopping_items'
    ) THEN
        ALTER PUBLICATION supabase_realtime DROP TABLE public.custom_shopping_items;
        RAISE NOTICE 'P2-REALTIME-PUB-SYNC: custom_shopping_items removida de supabase_realtime';
    ELSE
        RAISE NOTICE 'P2-REALTIME-PUB-SYNC: custom_shopping_items YA no estaba en supabase_realtime (idempotente noop)';
    END IF;
END;
$$;

-- (2b) custom_shopping_items: REPLICA IDENTITY FULL → DEFAULT (naturalmente
-- idempotente). Sin consumidor Realtime, FULL solo engorda el WAL.
ALTER TABLE public.custom_shopping_items REPLICA IDENTITY DEFAULT;

-- ─────────────────────────────────────────────────────────────────────────
-- SANITY CHECKS post-apply (fail-loud — P3-MIGRATION-IDEMPOTENCE-DOC)
-- ─────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
    -- meal_plans DEBE estar publicada (de lo contrario el push-merge sigue muerto).
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime' AND schemaname = 'public' AND tablename = 'meal_plans'
    ) THEN
        RAISE EXCEPTION 'P2-REALTIME-PUB-SYNC sanity: meal_plans NO está en supabase_realtime publication';
    END IF;

    -- custom_shopping_items NO debe estar publicada.
    IF EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime' AND schemaname = 'public' AND tablename = 'custom_shopping_items'
    ) THEN
        RAISE EXCEPTION 'P2-REALTIME-PUB-SYNC sanity: custom_shopping_items SIGUE en supabase_realtime publication';
    END IF;

    -- custom_shopping_items REPLICA IDENTITY debe ser DEFAULT ('d').
    IF (SELECT relreplident FROM pg_class WHERE oid = 'public.custom_shopping_items'::regclass) <> 'd' THEN
        RAISE EXCEPTION 'P2-REALTIME-PUB-SYNC sanity: custom_shopping_items REPLICA IDENTITY no es DEFAULT';
    END IF;
END;
$$;

COMMIT;
