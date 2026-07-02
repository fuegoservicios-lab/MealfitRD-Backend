-- [P2-BRANDPREF-SIZE-COLUMN · 2026-07-02] (audit v4 presupuesto↔lista, batch P2-OBJECTIVE-V4-BATCH)
-- Tamaño EXPLÍCITO del envase en gramos (líquidos ≈ ml, densidad ~1) para `supermarket_products`.
--
-- Por qué: el overlay de costeo de marca preferida (P1-SUPERMARKET-COSTING) parsea el tamaño del
-- texto libre `presentation`; la "L" suelta es AMBIGUA en el catálogo (el PDF del owner usa "L"
-- para LIBRA en produce Y para LITRO en leche) → el parser hace fail-open y esos productos PIERDEN
-- el overlay (costeo estándar silencioso). Con `size_grams` poblado (editable en el admin de
-- /supermercado), el costeo lo usa DIRECTO y el parser de `presentation` queda como fallback.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): ADD COLUMN IF NOT EXISTS + sanity check.
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): migrations/ (workspace-root) Y backend/migrations/.

ALTER TABLE public.supermarket_products
    ADD COLUMN IF NOT EXISTS size_grams double precision;

COMMENT ON COLUMN public.supermarket_products.size_grams IS
    '[P2-BRANDPREF-SIZE-COLUMN] Tamaño explícito del envase en gramos (líquidos ≈ ml). '
    'Fuente autoritativa del overlay de costeo P1-SUPERMARKET-COSTING; NULL → se parsea '
    '`presentation` (la "L" suelta es ambigua libra/litro → fail-open a costeo estándar).';

-- Sanity: la columna existe y es double precision.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'supermarket_products'
          AND column_name = 'size_grams'
          AND data_type = 'double precision'
    ) THEN
        RAISE EXCEPTION 'p2_supermarket_size_grams: columna size_grams ausente o de tipo incorrecto';
    END IF;
END $$;
