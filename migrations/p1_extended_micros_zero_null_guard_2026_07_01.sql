-- [P1-EXTENDED-MICROS-GUARD · 2026-07-01] (audit objetivo v2 · P1-2)
-- Cierra el gap de completitud de las 8 columnas EXTENDIDAS del panel de micros
-- (zinc/folato/vitA/vitC/vitE/vitK/selenio/omega3) en master_ingredients.
--
-- CONTEXTO: las 10 columnas BASE tienen DO-block cero-NULL desde
-- p1_micronutrient_catalog_backfill_2026_06_24.sql, pero las 8 EXTENDIDAS se poblaron
-- solo `WHERE fdc_id IS NOT NULL` sin verificación de completitud (riesgo documentado en
-- scripts/check_extended_micro_coverage.py). Audit 2026-07-01 en vivo: quedaba EXACTAMENTE
-- 1 fila con NULLs — 'Gandules' (vitE/vitK, USDA fdc 172436 no reporta esos 2 nutrientes) —
-- que el script de cobertura ocultaba por redondeo (201/202 → "100%"). Sin valor, el
-- aporte del ingrediente se descarta → cobertura por-nutriente baja → `estimado_bajo` →
-- el micro-closer SALTA ese micro (graph_orchestrator `_close_micro_gaps_for_plan`).
-- Gandules es staple DD (moro de gandules) → el hueco tocaba planes reales.
--
-- (1) Backfill de las 2 celdas con proxy leguminosa CONSERVADOR-BAJO (no sobreafirmar):
--     vitE 0.2 mg/100g (≈ habichuela roja cruda USDA 0.21), vitK 5 mcg/100g (≈ lenteja
--     cruda USDA ~5; habichuela roja 19 — se toma el bajo). COALESCE = idempotente.
-- (2) DO-block cero-NULL espejo del base: NINGUNA fila puede quedar con extendido NULL.
--     Si esta migración falla en el futuro, un alimento nuevo entró sin panel extendido —
--     completar el backfill (backfill_extended_micros.py para fdc_id; valor manual USDA
--     si no tiene), NO relajar el guard.

BEGIN;

UPDATE public.master_ingredients
SET vitamin_e_mg_per_100g = COALESCE(vitamin_e_mg_per_100g, 0.2),
    vitamin_k_mcg_per_100g = COALESCE(vitamin_k_mcg_per_100g, 5)
WHERE name = 'Gandules';

-- Sanity: cero-NULL en las 8 columnas extendidas (espejo del DO-block de las 10 BASE).
DO $$
DECLARE n_null integer;
BEGIN
  SELECT count(*) INTO n_null FROM public.master_ingredients
  WHERE zinc_mg_per_100g IS NULL OR folate_mcg_dfe_per_100g IS NULL
     OR vitamin_a_mcg_rae_per_100g IS NULL OR vitamin_c_mg_per_100g IS NULL
     OR vitamin_e_mg_per_100g IS NULL OR vitamin_k_mcg_per_100g IS NULL
     OR selenium_mcg_per_100g IS NULL OR omega3_ala_g_per_100g IS NULL;
  IF n_null > 0 THEN
    RAISE EXCEPTION 'P1-EXTENDED-MICROS-GUARD: % filas con micro EXTENDIDO en NULL — completar backfill (backfill_extended_micros.py / valor manual USDA), NO relajar el guard', n_null;
  END IF;
END $$;

COMMIT;
