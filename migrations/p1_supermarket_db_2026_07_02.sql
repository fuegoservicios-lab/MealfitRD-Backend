-- [P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial.
--
-- Tabla editable desde el landing (/supermercado, gate admin CRON_SECRET) que modela el
-- "supermercado dominicano" de MealfitRD: cada fila es UNA presentación comprable de un
-- alimento verificado (alimento + marca opcional + presentación + porción + duración +
-- precio RD$). Seed inicial: dataset de +200 alimentos verificados del owner
-- (backend/scripts/seed_supermarket_2026_07_02.py).
--
-- Diseño:
--  * `brand` NULL = producto genérico verificado (sin marca). Las variantes de marca del
--    mismo alimento son filas adicionales con el mismo `food_name` y `brand` distinto —
--    así el cliente podrá elegir "la marca de su yogurt" cuando se conecte a la lista
--    de compras.
--  * `master_food_name` = link SUAVE (por nombre) al catálogo `master_ingredients` para
--    la futura conexión con el sistema de lista de compras. Se llena/cura vía admin UI;
--    NO es FK dura para que el supermercado pueda contener productos aún sin mapear.
--  * Escrituras SOLO vía backend (routers/supermarket.py, gate `_verify_admin_token`).
--    El frontend jamás escribe directo (invariante I6 análoga).
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): IF NOT EXISTS + DO $$ sanity.

CREATE TABLE IF NOT EXISTS public.supermarket_products (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    food_name        text NOT NULL,
    brand            text,
    presentation     text,
    portion_label    text,
    duration_label   text,
    price_rd         numeric(10,2),
    notes            text,
    category         text,
    master_food_name text,
    is_verified      boolean NOT NULL DEFAULT true,
    active           boolean NOT NULL DEFAULT true,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT supermarket_products_price_nonneg
        CHECK (price_rd IS NULL OR price_rd >= 0),
    CONSTRAINT supermarket_products_food_name_nonempty
        CHECK (length(btrim(food_name)) > 0)
);

COMMENT ON TABLE public.supermarket_products IS
    '[P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial: presentaciones comprables de alimentos verificados (+variantes de marca). Editable solo vía backend /api/supermarket (admin CRON_SECRET). Futuro: conexión a lista de compras vía master_food_name.';

COMMENT ON COLUMN public.supermarket_products.brand IS
    'NULL = genérico verificado. Variantes de marca = filas adicionales con mismo food_name.';

COMMENT ON COLUMN public.supermarket_products.master_food_name IS
    'Link suave (por nombre) a master_ingredients.name para la futura integración con la lista de compras. NO es FK dura.';

-- Unicidad de variante (case-insensitive): un alimento + marca + presentación solo una vez.
-- También hace idempotente el seed (ON CONFLICT DO NOTHING).
CREATE UNIQUE INDEX IF NOT EXISTS uq_supermarket_products_variant
    ON public.supermarket_products (
        lower(food_name),
        lower(coalesce(brand, '')),
        lower(coalesce(presentation, ''))
    );

-- Lookups del landing: búsqueda por alimento y filtro por categoría (solo activos).
CREATE INDEX IF NOT EXISTS idx_supermarket_products_food
    ON public.supermarket_products (lower(food_name));

CREATE INDEX IF NOT EXISTS idx_supermarket_products_category_active
    ON public.supermarket_products (category)
    WHERE active;

-- Sanity: la tabla y el unique index deben existir tras la migración.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'supermarket_products'
    ) THEN
        RAISE EXCEPTION 'P1-SUPERMARKET-DB: tabla supermarket_products no fue creada';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'supermarket_products'
          AND indexname = 'uq_supermarket_products_variant'
    ) THEN
        RAISE EXCEPTION 'P1-SUPERMARKET-DB: unique index uq_supermarket_products_variant no fue creado';
    END IF;
END $$;
