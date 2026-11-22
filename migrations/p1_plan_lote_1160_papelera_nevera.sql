-- [P1-PLAN-LOTE-1160 · 2026-11-21] Papelera de la Nevera: lo que el usuario borra se puede recuperar durante 30 días.
-- Caso vivo del dueño (21-nov): «Borrar todos» vació la Nevera Y los tres potes de suplementos con su etiqueta (DELETE
-- duro), le pidió al coach recuperarlos y no había de dónde: se rescataron desde una instantánea de Neon.
-- Cada borrado que pide el usuario (uno, «Borrar todos», el coach) archiva aquí la fila ENTERA (`fila` = to_jsonb de
-- user_inventory: etiqueta, marca, tipo, unidad) en la misma sentencia del DELETE. Restaurar = reinsertar la fila.
-- Lo escribe y lo lee solo `backend/papelera_nevera.py`. Se purga a los 30 días al archivar. Idempotente.
CREATE TABLE IF NOT EXISTS public.user_inventory_papelera (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    original_id BIGINT,
    fila JSONB NOT NULL,
    via TEXT NOT NULL DEFAULT 'uno',
    borrado_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.user_inventory_papelera DROP CONSTRAINT IF EXISTS user_inventory_papelera_via_chk;
ALTER TABLE public.user_inventory_papelera ADD CONSTRAINT user_inventory_papelera_via_chk
    CHECK (via IN ('uno', 'todos', 'coach'));

ALTER TABLE public.user_inventory_papelera DROP CONSTRAINT IF EXISTS user_inventory_papelera_fila_chk;
ALTER TABLE public.user_inventory_papelera ADD CONSTRAINT user_inventory_papelera_fila_chk
    CHECK (jsonb_typeof(fila) = 'object' AND fila ? 'ingredient_name');

CREATE INDEX IF NOT EXISTS user_inventory_papelera_user_idx
    ON public.user_inventory_papelera (user_id, borrado_at DESC);

REVOKE ALL ON public.user_inventory_papelera FROM PUBLIC;

COMMENT ON TABLE public.user_inventory_papelera IS
    'P1-PLAN-LOTE-1160: filas borradas de user_inventory por el usuario (30 días), para «Deshacer» y para que el coach las recupere. Solo backend/papelera_nevera.py.';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'user_inventory_papelera') THEN
        RAISE EXCEPTION 'p1_plan_lote_1160: user_inventory_papelera no existe tras la migracion';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'user_inventory_papelera_via_chk') THEN
        RAISE EXCEPTION 'p1_plan_lote_1160: falta user_inventory_papelera_via_chk';
    END IF;
END $$;
