-- [P1-PKG-DURATION-PRICING · 2026-06-22] Precio por TAMAÑO de envase (no precio plano).
-- Problema de producción: la lista de compras ya ELIGE bien el envase por duración
-- (`_find_best_sku` con `available_sizes_g`: 2lb para 7 días, 10lb para 30 días), pero el
-- COSTO se calculaba con un único `price_per_lb`/`price_per_unit` → ignoraba el descuento
-- por volumen real. Ej. arroz blanco plan 30 días = 10 lb × 55 = RD$550, cuando el paquete
-- real de 10 lb cuesta RD$327 (datos verificados in-store por el owner). Sobrecobro ~68%.
--
-- Fix de datos: nueva columna `market_packages` = lista de envases reales con su tamaño en
-- gramos Y su precio RD$, p.ej. arroz blanco:
--   [{"grams":907,"price":165,"label":"2 lb"},
--    {"grams":2268,"price":235,"label":"5 lb"},
--    {"grams":4536,"price":327,"label":"10 lb"}]
-- El calculador usa los `grams` para seleccionar el SKU (igual que hoy) y el `price` del
-- tamaño elegido para costear (count × price). NULLABLE → degradación grácil: sin
-- `market_packages` el item sigue el path actual (price_per_lb/price_per_unit plano).
--
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS market_packages jsonb;

COMMENT ON COLUMN public.master_ingredients.market_packages IS
    '[P1-PKG-DURATION-PRICING] Envases reales con tamaño+precio: [{"grams":N,"price":RD$,"label":"..."}]. El SKU resolver elige por grams (duracion-aware via base_duration_scale) y costea con el price del tamaño elegido (descuento por volumen). NULL → fallback a price_per_lb/price_per_unit plano.';

-- Sanity: idempotencia + presencia de la columna nueva.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'market_packages'
    ) THEN
        RAISE EXCEPTION 'P1-PKG-DURATION-PRICING: columna market_packages no fue creada';
    END IF;
END $$;
