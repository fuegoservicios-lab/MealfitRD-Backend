-- [P2-NEVERA-BRANDS · 2026-07-06] Marca comprada visible en la Nevera.
-- Pedido del owner: "en la nevera también debería decir las marcas". Al hacer
-- restock ("Ya compré la lista"), cada ítem llega con el brand_product_id del
-- producto que la lista usó (default más barato o preferencia manual) → el
-- backend resuelve la marca y la persiste aquí. Ítems añadidos a mano quedan
-- con brand NULL (sin chip). Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC).

ALTER TABLE public.user_inventory
    ADD COLUMN IF NOT EXISTS brand text;

COMMENT ON COLUMN public.user_inventory.brand IS
    '[P2-NEVERA-BRANDS 2026-07-06] Marca del producto comprado (resuelta desde '
    'supermarket_products via brand_product_id en /restock). NULL = añadido a '
    'mano o pre-feature. Última compra gana (COALESCE en el upsert).';

-- Sanity: la columna existe.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'user_inventory'
          AND column_name = 'brand'
    ) THEN
        RAISE EXCEPTION 'p2_nevera_brands: columna user_inventory.brand ausente tras ALTER';
    END IF;
END $$;
