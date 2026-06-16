-- [P1-RESOLVER-COVERAGE · 2026-06-16 · follow-up] Densidad volumétrica del Huevo.
--
-- Gap medido en comidas entregables: "375 ml de claras de huevo" / "1.5 tazas de claras de huevo"
-- (~5 ocurrencias) NO resolvían porque Huevo tenía density_g_per_unit=50 (por huevo) pero
-- density_g_per_cup=NULL → to_grams no puede convertir ml/taza a gramos para el huevo líquido.
--
-- USDA: 1 taza (240 ml) de huevo líquido (entero o claras) ≈ 243 g (densidad ~1.01 g/ml). Aditivo:
-- la forma por-unidad ("2 huevos" → density_g_per_unit=50) NO se afecta — to_grams usa cup-density solo
-- para unidades de volumen (taza/ml) y unit-density solo para "unidad". Cero riesgo en líneas ya resueltas.
--
-- Idempotente: solo escribe si density_g_per_cup está vacío. Sync: migrations/ + backend/migrations/.

UPDATE public.master_ingredients
SET density_g_per_cup = 243
WHERE name = 'Huevo' AND density_g_per_cup IS NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients
                   WHERE name = 'Huevo' AND density_g_per_cup = 243) THEN
        RAISE EXCEPTION '[P1-EGG-DENSITY] sanity: Huevo.density_g_per_cup no quedó en 243';
    END IF;
END $$;
