-- [P1-PLAN-LOTE-2 · 2026-09-11 · G2] `user_profiles.plan_tier`: el esquema decía 'free' y el código 'gratis'.
--
-- La columna nació con DEFAULT 'free'::text (medido en producción el 2026-09-11: 2 de 2 perfiles vivos
-- llevaban 'free') y TODO el backend habla en 'gratis' (`_TIER_LIMITS`, `get_user_tier`, `PAID_TIERS`,
-- `get_user_plan_tier`: «tier crudo (gratis/basic/plus/ultra)»). Un perfil nuevo tenía un tier que ninguna
-- tabla conocía y caía al defecto por `.get(tier, _TIER_LIMITS["gratis"])`: funcionaba por accidente. La
-- auditoría de PayPal del 2026-08-22 lo anotó (§2) y quedó abierto.
--
-- Qué hace, idempotente (P3-MIGRATION-IDEMPOTENCE-DOC):
--   1. Sanity: si hay un tier fuera del conjunto conocido (más 'free', que ESTA migración normaliza), aborta
--      hablando — un valor desconocido es un bug aguas arriba, no algo que una CHECK deba tapar.
--   2. UPDATE 'free' → 'gratis' (2 filas hoy).
--   3. DEFAULT 'gratis'.
--   4. CHECK `user_profiles_plan_tier_canonical`: gratis | basic | plus | ultra | admin. NULL no pasa: la
--      columna siempre tuvo defecto y el código no contempla ausencia.
-- El código (`llm_provider.get_user_tier`) acepta 'free' como alias por si algún escritor viejo lo emite.
--
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.

DO $$
DECLARE
  bad_count int;
BEGIN
  SELECT COUNT(*) INTO bad_count
  FROM public.user_profiles
  WHERE plan_tier IS NULL
     OR plan_tier NOT IN ('gratis', 'basic', 'plus', 'ultra', 'admin', 'free');
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      'P1-PLAN-LOTE-2: % filas de user_profiles tienen plan_tier fuera del conjunto conocido. '
      'Consulta: SELECT id, plan_tier FROM public.user_profiles WHERE plan_tier IS NULL OR plan_tier '
      'NOT IN (''gratis'',''basic'',''plus'',''ultra'',''admin'',''free'');',
      bad_count;
  END IF;
END $$;

UPDATE public.user_profiles SET plan_tier = 'gratis' WHERE plan_tier = 'free';

ALTER TABLE public.user_profiles ALTER COLUMN plan_tier SET DEFAULT 'gratis';

ALTER TABLE public.user_profiles DROP CONSTRAINT IF EXISTS user_profiles_plan_tier_canonical;
ALTER TABLE public.user_profiles
  ADD CONSTRAINT user_profiles_plan_tier_canonical
  CHECK (plan_tier IN ('gratis', 'basic', 'plus', 'ultra', 'admin'));
