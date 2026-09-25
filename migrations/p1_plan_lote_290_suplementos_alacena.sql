-- [P1-PLAN-LOTE-290 · 2026-09-25] Suplementos en la Nevera → Alacena. SSOT: backend/suplementos.py.
-- kind           'food' (ingrediente: lo usan el generador, las puertas, la compra) | 'supplement' (fuera de todo eso).
-- serving_label  etiqueta por porción {serving_g,kcal,protein_g,carbs_g,fats_g}; NULL = sin etiqueta todavía.
-- serving_unit   scoop | capsula | porcion | g ; quantity = porciones restantes.
-- label_source   foto | marca | estimado (de dónde salió la etiqueta). Idempotente.
ALTER TABLE public.user_inventory ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'food';
ALTER TABLE public.user_inventory ADD COLUMN IF NOT EXISTS serving_label JSONB;
ALTER TABLE public.user_inventory ADD COLUMN IF NOT EXISTS serving_unit TEXT;
ALTER TABLE public.user_inventory ADD COLUMN IF NOT EXISTS label_source TEXT;
ALTER TABLE public.user_inventory DROP CONSTRAINT IF EXISTS user_inventory_kind_check;
ALTER TABLE public.user_inventory ADD CONSTRAINT user_inventory_kind_check CHECK (kind IN ('food', 'supplement'));
CREATE INDEX IF NOT EXISTS user_inventory_user_kind_idx ON public.user_inventory (user_id, kind);
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'user_inventory' AND column_name = 'kind') THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-290] falta user_inventory.kind tras la migración';
    END IF;
END $$;
