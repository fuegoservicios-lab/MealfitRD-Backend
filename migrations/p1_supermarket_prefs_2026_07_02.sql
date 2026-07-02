-- [P1-SUPERMARKET-PREFS · 2026-07-02] Marca preferida por usuario (fase 2 de la
-- conexión lista de compras ↔ Supermercado RD).
--
-- Una fila por (usuario, alimento normalizado): el producto del súper que el
-- usuario eligió como su marca/presentación preferida para ese alimento.
-- `food_key` es el nombre del alimento NORMALIZADO (minúsculas, sin acentos,
-- espacios colapsados — misma normalización que `_norm_food` en
-- routers/supermarket.py y `foodKeyOf` en el frontend) para que la preferencia
-- sobreviva drifts de escritura y regeneraciones de plan.
--
-- Acceso: SOLO backend (Neon, sin RLS — convención post-migración). Los
-- endpoints filtran `WHERE user_id = %s` (invariante I2). ON DELETE CASCADE
-- en ambas FKs: borrar usuario o producto limpia sus preferencias.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): IF NOT EXISTS en CREATE + DO $$
-- sanity al final.

CREATE TABLE IF NOT EXISTS public.user_brand_preferences (
    user_id     uuid        NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    food_key    text        NOT NULL CHECK (length(food_key) BETWEEN 1 AND 120),
    product_id  uuid        NOT NULL REFERENCES public.supermarket_products(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, food_key)
);

-- Cubre la FK a supermarket_products (lección P2-PERF-1: sin índice, borrar un
-- producto del catálogo haría seq-scan sobre las preferencias). El advisor
-- `unused_index` puede reportarlo con 0 scans — es load-bearing por FK.
CREATE INDEX IF NOT EXISTS idx_user_brand_preferences_product
    ON public.user_brand_preferences (product_id);

COMMENT ON TABLE public.user_brand_preferences IS
    '[P1-SUPERMARKET-PREFS] Marca/presentación del súper preferida por usuario y alimento (food_key normalizado). Solo backend; endpoints en routers/supermarket.py filtran user_id (I2).';
COMMENT ON INDEX public.idx_user_brand_preferences_product IS
    '[P1-SUPERMARKET-PREFS] Cubre FK a supermarket_products (ON DELETE CASCADE) — mantener aunque el advisor unused_index reporte 0 scans (lección P2-PERF-1).';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'user_brand_preferences'
    ) THEN
        RAISE EXCEPTION '[P1-SUPERMARKET-PREFS] sanity: user_brand_preferences no existe tras la migración';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = 'idx_user_brand_preferences_product'
    ) THEN
        RAISE EXCEPTION '[P1-SUPERMARKET-PREFS] sanity: falta idx_user_brand_preferences_product';
    END IF;
END $$;
