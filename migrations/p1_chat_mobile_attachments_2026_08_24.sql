-- P1-CHAT-MOBILE-ATTACHMENTS · 2026-08-24
-- Adjuntos privados y durables para el chat. El binario debe desplegarse DESPUÉS
-- de esta migración; no contiene backfill destructivo y mantiene [IMAGE:] legacy.

BEGIN;

ALTER TABLE public.agent_messages
    ADD COLUMN IF NOT EXISTS attachments jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS client_message_id uuid;

CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_messages_session_client_message
    ON public.agent_messages (session_id, client_message_id)
    WHERE client_message_id IS NOT NULL;

ALTER TABLE public.agent_messages
    DROP CONSTRAINT IF EXISTS agent_messages_attachments_array;
ALTER TABLE public.agent_messages
    ADD CONSTRAINT agent_messages_attachments_array
    CHECK (jsonb_typeof(attachments) = 'array' AND jsonb_array_length(attachments) <= 4);

CREATE TABLE IF NOT EXISTS public.chat_attachments (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES public.agent_sessions(id) ON DELETE CASCADE,
    message_id uuid REFERENCES public.agent_messages(id) ON DELETE CASCADE,
    -- [P1-NEON-DB-MIGRATION] `auth.users` es el esquema de SUPABASE, eliminado por
    -- completo el 2026-06-12: en Neon ese esquema NO EXISTE y la migración abortaba con
    -- InvalidSchemaName. La tabla de usuarios viva es `public.user_profiles`; el cascade
    -- por cuenta que este fichero promete se cumple igual desde ella.
    user_id uuid NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    content bytea NOT NULL,
    content_type text NOT NULL,
    byte_size integer NOT NULL,
    original_name text,
    width integer,
    height integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    claimed_at timestamptz,
    CONSTRAINT chat_attachments_content_type
        CHECK (content_type IN ('image/jpeg', 'image/png', 'image/webp', 'image/heic')),
    CONSTRAINT chat_attachments_byte_size
        CHECK (byte_size > 0 AND byte_size <= 20971520)
);

CREATE INDEX IF NOT EXISTS idx_chat_attachments_session_created
    ON public.chat_attachments (session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_attachments_user_created
    ON public.chat_attachments (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_attachments_unclaimed
    ON public.chat_attachments (created_at)
    WHERE message_id IS NULL;

-- [P1-NEON-DB-MIGRATION · corregido 2026-08-24] SIN RLS, y conviene decir por qué en vez
-- de que parezca un olvido: `auth.uid()` es de Supabase y en Neon no existe, así que esta
-- política habría sido inaplicable — y una política que no se puede evaluar no protege nada,
-- sólo lo aparenta. En Neon el backend se conecta con credenciales de servicio (RLS no le
-- aplica) y el aislamiento por usuario lo da el filtro `AND user_id = %s` de cada consulta,
-- que es la invariante I2 del repo y está cubierta por sus tests parser-based.
-- El acceso al binario va además firmado por HMAC temporal (build_chat_attachment_url).

COMMENT ON TABLE public.chat_attachments IS
'P1-CHAT-MOBILE-ATTACHMENTS: media privada del chat, servida solo con URL HMAC temporal o sesión propietaria; cascadea con sesión/mensaje/cuenta.';
COMMENT ON COLUMN public.agent_messages.attachments IS
'Array ordenado (máx. 4) de metadata de chat_attachments; [IMAGE:] se conserva solo para historial legacy.';

COMMIT;
