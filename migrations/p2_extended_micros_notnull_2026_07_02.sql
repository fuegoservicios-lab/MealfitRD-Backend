-- [P2-EXTENDED-MICROS-CONSTRAINT · 2026-07-02] (audit v3 micros GAP-6)
-- Las 8 columnas EXTENDIDAS del panel de micros estaban protegidas solo por:
--   (a) el DO-block cero-NULL de la migración p1_extended_micros_zero_null_guard (corre UNA vez), y
--   (b) el script manual scripts/check_extended_micro_coverage.py (disciplina de operador).
-- Un lote futuro de catálogo (add_foods_batch*.py) cuyo JSON omita una columna reintroduce un NULL
-- silencioso → cobertura por-nutriente <0.6 → `estimado_bajo` → el micro-closer SALTA ese micro sin
-- que nadie lo vea (hasta que la alerta P2-MICRO-KPI-ALERT dispare días después).
--
-- Constraint a nivel DB: NOT NULL en las 8 columnas → el INSERT del lote FALLA FUERTE en el momento
-- del alta (fail-loud > drift silencioso). Si esta constraint rechaza un INSERT en runtime, el bug
-- está en el script de alta (JSON incompleto) — completar el panel extendido del alimento
-- (backfill_extended_micros.py para fdc_id; valor manual USDA si no tiene), NO relajar la constraint.
--
-- Idempotente: SET NOT NULL sobre columna ya NOT NULL es no-op en Postgres; el DO-block pre-check
-- da mensaje accionable si algún NULL residual bloquearía el ALTER.

BEGIN;

-- Pre-check accionable (mismo espíritu del guard 2026-07-01): si hay NULLs, el ALTER de abajo
-- fallaría con un mensaje críptico — este RAISE dice exactamente qué backfillear.
DO $$
DECLARE n_null integer;
BEGIN
  SELECT count(*) INTO n_null FROM public.master_ingredients
  WHERE zinc_mg_per_100g IS NULL OR folate_mcg_dfe_per_100g IS NULL
     OR vitamin_a_mcg_rae_per_100g IS NULL OR vitamin_c_mg_per_100g IS NULL
     OR vitamin_e_mg_per_100g IS NULL OR vitamin_k_mcg_per_100g IS NULL
     OR selenium_mcg_per_100g IS NULL OR omega3_ala_g_per_100g IS NULL;
  IF n_null > 0 THEN
    RAISE EXCEPTION 'P2-EXTENDED-MICROS-CONSTRAINT: % filas con micro EXTENDIDO en NULL — backfillear (backfill_extended_micros.py / USDA manual) ANTES de aplicar NOT NULL', n_null;
  END IF;
END $$;

ALTER TABLE public.master_ingredients ALTER COLUMN zinc_mg_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN folate_mcg_dfe_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN vitamin_a_mcg_rae_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN vitamin_c_mg_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN vitamin_e_mg_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN vitamin_k_mcg_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN selenium_mcg_per_100g SET NOT NULL;
ALTER TABLE public.master_ingredients ALTER COLUMN omega3_ala_g_per_100g SET NOT NULL;

COMMENT ON COLUMN public.master_ingredients.vitamin_k_mcg_per_100g IS
  'NOT NULL desde P2-EXTENDED-MICROS-CONSTRAINT (2026-07-02): el panel extendido es obligatorio al alta; NULL silencioso degradaba el micro-closer a estimado_bajo. Si tu INSERT falla: completa el panel, no relajes la constraint.';

COMMIT;
