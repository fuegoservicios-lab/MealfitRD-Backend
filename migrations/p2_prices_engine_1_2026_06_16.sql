-- [P2-PRICES-ENGINE-1 · 2026-06-16] Cimiento del costeo real + motor de inflación.
--
-- CONTEXTO: el costeo de la lista de compras YA existe — `shopping_calculator.py`
-- lee `master_ingredients.price_per_lb`/`price_per_unit` y computa `estimated_cost_rd`
-- por ítem (+ `total_estimated_cost`). Pero esas columnas están en 0 (sin poblar) y
-- no tienen proveniencia ni mecanismo de actualización. Esta migración añade:
--
--   1. Columnas BASE + proveniencia en master_ingredients. La separación
--      base↔vivo es la clave del motor de inflación: el cron reescala
--      `price_per_lb = price_per_lb_base × (índice_actual / índice_base)` SIN
--      compoundear (la base nunca cambia). El calculador sigue leyendo
--      `price_per_lb`/`price_per_unit` exactamente como hoy → cero cambio de
--      comportamiento hasta que se pueblen precios base.
--
--   2. Tabla `price_inflation_index`: serie del subíndice IPC "Alimentos y
--      bebidas no alcohólicas" del Banco Central de RD (BCRD), 1 fila/mes.
--      El motor compara la base (período de captura) contra el último período
--      ingerido para proyectar el precio actual sin re-encuestar el supermercado.
--
-- HONESTIDAD: un precio proyectado por índice es ESTIMADO, no exacto. La columna
-- `price_confidence` ('high'/'medium'/'low') permite a la UI etiquetar perecederos
-- (volátiles, baja confianza) distinto de staples (estables). Re-anclaje periódico
-- (re-captura de la base) corrige el drift acumulado.
--
-- Todo NULLABLE → degradación grácil: sin base price el item simplemente no
-- contribuye al costo (como hoy). Idempotente (re-ejecutable). SSOT dual-dir
-- (P3-MIGRATIONS-SSOT): este archivo vive en migrations/ Y backend/migrations/.

-- ── 1. Columnas base + proveniencia en master_ingredients ──────────────────
ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS price_per_lb_base    numeric,
    ADD COLUMN IF NOT EXISTS price_per_unit_base  numeric,
    ADD COLUMN IF NOT EXISTS price_base_period    text,       -- YYYY-MM del índice base
    ADD COLUMN IF NOT EXISTS price_source         text,       -- 'nacional_online' / 'sirena_online' / 'manual' / 'crowdsource' / ...
    ADD COLUMN IF NOT EXISTS price_captured_at    date,       -- fecha de captura del precio base
    ADD COLUMN IF NOT EXISTS price_confidence     text,       -- 'high' / 'medium' / 'low'
    ADD COLUMN IF NOT EXISTS price_adjusted_at    timestamptz; -- última reescala por el cron de inflación

-- Enum de confianza (idempotente: DROP antes de ADD). NULL permitido.
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_price_confidence_check;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_price_confidence_check
    CHECK (price_confidence IS NULL OR price_confidence IN ('high', 'medium', 'low'));

-- Período base bien formado (YYYY-MM) o NULL.
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_price_base_period_check;
ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_price_base_period_check
    CHECK (price_base_period IS NULL OR price_base_period ~ '^[0-9]{4}-[0-9]{2}$');

COMMENT ON COLUMN public.master_ingredients.price_per_lb_base IS
    '[P2-PRICES-ENGINE-1] Precio base RD$/libra antes de ajuste por inflación. El cron escribe price_per_lb (vivo) = base × (índice_actual / índice_base). Separar base de vivo evita compoundear el ajuste.';
COMMENT ON COLUMN public.master_ingredients.price_base_period IS
    '[P2-PRICES-ENGINE-1] Período YYYY-MM del subíndice IPC contra el que se capturó el precio base (FK lógica a price_inflation_index.period).';
COMMENT ON COLUMN public.master_ingredients.price_confidence IS
    '[P2-PRICES-ENGINE-1] Confianza del precio: high (staple estable), medium, low (perecedero volátil). La UI lo usa para etiquetar "estimado" con distinto énfasis.';

-- ── 2. Serie del índice de inflación de alimentos (BCRD) ───────────────────
CREATE TABLE IF NOT EXISTS public.price_inflation_index (
    period       text PRIMARY KEY,                          -- YYYY-MM
    food_cpi     numeric NOT NULL,                          -- subíndice IPC alimentos (base index del BCRD)
    source       text NOT NULL DEFAULT 'bcrd',
    note         text,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT price_inflation_index_period_format_check
        CHECK (period ~ '^[0-9]{4}-[0-9]{2}$'),
    CONSTRAINT price_inflation_index_food_cpi_positive_check
        CHECK (food_cpi > 0)
);

COMMENT ON TABLE public.price_inflation_index IS
    '[P2-PRICES-ENGINE-1] Serie mensual del subíndice IPC "Alimentos y bebidas no alcohólicas" del Banco Central de RD. 1 fila/mes. El motor de precios proyecta precio_actual = precio_base × (food_cpi_actual / food_cpi_base) para no re-encuestar el supermercado cada vez. Server-side, append/upsert por período.';

ALTER TABLE public.price_inflation_index ENABLE ROW LEVEL SECURITY;

-- ── 3. Sanity: idempotencia + presencia de columnas/tabla clave ────────────
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'master_ingredients'
          AND column_name = 'price_per_lb_base'
    ) THEN
        RAISE EXCEPTION 'P2-PRICES-ENGINE-1: columna price_per_lb_base no fue creada';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'price_inflation_index'
    ) THEN
        RAISE EXCEPTION 'P2-PRICES-ENGINE-1: tabla price_inflation_index no fue creada';
    END IF;
END $$;
