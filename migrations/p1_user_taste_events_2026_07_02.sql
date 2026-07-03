-- [P1-NEXT-LEVEL-BATCH · 2026-07-02] (TASTE) Señales de gusto APRENDIDAS del uso.
-- El motor personalizaba solo con lo DECLARADO en el formulario; esta tabla captura lo
-- que el usuario HACE (swap-away con cambio de proteína, reemplazo por chat con negación
-- explícita) y `taste_model.py` la agrega con umbral (≥2.0) + ventana (90d) para inyectar
-- una preferencia SUAVE al planner/day-gen. Fail-open: el código tolera la tabla ausente.
-- Idempotente (IF NOT EXISTS + DO $$ sanity). SSOT: copiar a AMBOS dirs de migrations.

CREATE TABLE IF NOT EXISTS public.user_taste_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    token TEXT NOT NULL,
    signal TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Cubre el FK CASCADE (lección P2-5: el advisor unused_index NO observa uso por FK)
-- y la query de agregación (user_id + ventana temporal).
CREATE INDEX IF NOT EXISTS idx_user_taste_events_user_created
    ON public.user_taste_events (user_id, created_at DESC);

COMMENT ON TABLE public.user_taste_events IS
    '[P1-NEXT-LEVEL-BATCH] Señales de gusto aprendidas del uso (swap-away/chat-replace). '
    'Agregadas por backend/taste_model.py (umbral+ventana) → preferencia suave al planner. '
    'Append-only; sin RLS (acceso solo backend service).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'user_taste_events'
    ) THEN
        RAISE EXCEPTION 'sanity: user_taste_events no existe tras la migración';
    END IF;
END $$;
