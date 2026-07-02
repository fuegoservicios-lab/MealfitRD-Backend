-- [P1-SUPERMARKET-DB · 2026-07-02 · fase 2 catálogo] Media + specs del Supermercado RD.
--
-- El rediseño estilo catálogo (tarjeta de producto → modal de detalle con imagen y
-- especificaciones, patrón La Sirena) necesita dos campos que el schema inicial no tenía:
--   * `image_url`   — URL de la imagen del producto (se cura desde la admin UI; NULL ⇒
--                     la página renderiza un placeholder por categoría).
--   * `description` — especificaciones legibles del SKU ("Leche entera UHT 3.1% grasa"),
--                     mostradas en la tarjeta de detalle.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): ADD COLUMN IF NOT EXISTS + DO $$ sanity.

ALTER TABLE public.supermarket_products
    ADD COLUMN IF NOT EXISTS image_url text,
    ADD COLUMN IF NOT EXISTS description text;

COMMENT ON COLUMN public.supermarket_products.image_url IS
    '[P1-SUPERMARKET-DB fase 2] URL de imagen del producto (admin UI). NULL ⇒ placeholder por categoría en el frontend.';

COMMENT ON COLUMN public.supermarket_products.description IS
    '[P1-SUPERMARKET-DB fase 2] Especificaciones legibles del SKU para el modal de detalle (ej. "Leche entera UHT 3.1% grasa").';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'supermarket_products'
          AND column_name = 'image_url'
    ) THEN
        RAISE EXCEPTION 'P1-SUPERMARKET-DB fase 2: columna image_url no fue creada';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'supermarket_products'
          AND column_name = 'description'
    ) THEN
        RAISE EXCEPTION 'P1-SUPERMARKET-DB fase 2: columna description no fue creada';
    END IF;
END $$;
