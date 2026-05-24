-- [P3-DEPLETED-BD-REALTIME-FIX · 2026-05-22] Añade `user_depleted_items` a
-- la publication `supabase_realtime` para que el frontend reciba eventos
-- INSERT/UPDATE/DELETE en vivo. Sin esta entrada, la suscripción en
-- Pantry.jsx (`supabase.channel('user_depleted_items_${uid}')`) se establece
-- exitosamente pero NUNCA recibe eventos — solo el fetch inicial popula
-- el state, perdiendo la promesa de cross-device sync del bundle P3-DEPLETED-BD.
--
-- ─────────────────────────────────────────────────────────────────────────
-- VECTOR CERRADO
-- ─────────────────────────────────────────────────────────────────────────
--
-- Bug verificado 2026-05-22 05:52: el user dijo "se me acabo la lechosa"
-- al chat agent. Backend hizo todo correctamente:
--   - INSERT a user_depleted_items: ok (verificado via MCP, id=1).
--   - DELETE de user_inventory: ok (verificado, lechosa ausente).
--   - SSE emite pantry_modified_at + pantry_depleted_items: ok.
--
-- PERO el frontend mostraba lechosa todavía en sección Frutas activas.
-- Causa: `user_depleted_items` faltaba en `supabase_realtime` publication
-- → el channel del frontend no recibía el INSERT event → state nunca
-- se actualizaba con la nueva entrada.
--
-- Fix: ALTER PUBLICATION + sanity check post-apply.
--
-- ─────────────────────────────────────────────────────────────────────────
-- IDEMPOTENCIA
-- ─────────────────────────────────────────────────────────────────────────
--
-- ALTER PUBLICATION ADD TABLE NO tiene IF NOT EXISTS nativo. Wrap en DO $$
-- con check pre-apply (NO EXISTS) para idempotencia segura.

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'user_depleted_items'
    ) THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.user_depleted_items;
        RAISE NOTICE 'P3-DEPLETED-BD-REALTIME-FIX: user_depleted_items añadida a supabase_realtime';
    ELSE
        RAISE NOTICE 'P3-DEPLETED-BD-REALTIME-FIX: user_depleted_items YA estaba en supabase_realtime (idempotente noop)';
    END IF;
END;
$$;

-- Sanity check post-apply.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_publication_tables
        WHERE pubname = 'supabase_realtime'
          AND schemaname = 'public'
          AND tablename = 'user_depleted_items'
    ) THEN
        RAISE EXCEPTION 'P3-DEPLETED-BD-REALTIME-FIX sanity: user_depleted_items NO está en supabase_realtime publication';
    END IF;
END;
$$;

COMMIT;
