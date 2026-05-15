-- [P1-DEFINER-LOCKDOWN · 2026-05-12] Lockear las 3 functions SECURITY DEFINER
-- nuevas (`handle_new_user`, `get_monthly_plan_count`, `log_unknown_ingredient_rpc`)
-- al pattern P3-NEW-2 + emitir REVOKE EXECUTE explícito como contrato SSOT.
--
-- Causa raíz (audit production-readiness 2026-05-12):
--   El audit detectó 3 functions SECURITY DEFINER en `public` schema que NO
--   estaban listadas en la tabla "Functions ya bajo el pattern" de CLAUDE.md
--   (sección "Pattern: SET search_path = ''"):
--
--     1. handle_new_user()                              — trigger de auth.users
--     2. get_monthly_plan_count(user_uuid uuid)         — quota check legacy
--     3. log_unknown_ingredient_rpc(p_user_id, ...)     — invocada desde
--                                                          backend/db_plans.py:1196
--
--   Inspección de pg_proc (snapshot pre-migración):
--     - Las 3 usan `SET search_path TO 'public'` (NO `''` como pide P3-NEW-2).
--     - Los GRANTs ya eran seguros por default: solo `service_role` tenía
--       EXECUTE. Ni `anon`, ni `authenticated`, ni `public` podían invocarlas
--       via PostgREST `/rest/v1/rpc/*`. Por eso el advisor de Supabase
--       `authenticated_security_definer_function_executable` NO las flageaba
--       (solo aparece `increment_inventory_quantity`, que SÍ está expuesta a
--       authenticated por decisión documentada P2-4).
--
-- Riesgo IDOR existente (pre-migración):
--   Teórico, no explotable en producción. Requiere que un futuro operador
--   ejecute `GRANT EXECUTE ... TO authenticated` por error sin notar que las
--   3 functions aceptan `user_id`/`p_user_id` arbitrario sin validar contra
--   `auth.uid()`. Esta migración cierra esa puerta de manera SSOT auditable:
--   un GRANT futuro a authenticated tendría que ser deliberadamente añadido
--   (no caer "por accidente"), y en ese caso un test parser-based puede
--   verificar el contrato.
--
-- Por qué `SET search_path = ''` (cadena vacía) en vez de `'public'`:
--   Patrón canónico P3-NEW-2 (CLAUDE.md). Forza schema qualifier explícito
--   en cualquier referencia. Defense-in-depth contra shadowing por temp
--   tables. Las 3 functions ya tienen sus referencias internas qualified
--   (`public.meal_plans`, `public.unknown_ingredients`, `public.user_profiles`),
--   así que el flip es safe sin cambio funcional.
--
-- Por qué REVOKE EXECUTE FROM PUBLIC, anon, authenticated explícito:
--   Los GRANTs implícitos (default) son fácilmente sobre-escribibles. Un
--   `GRANT ALL ON FUNCTION ... TO PUBLIC` ejecutado por error en otra
--   migración o desde el dashboard pisaría el lockdown silenciosamente.
--   El REVOKE explícito en SSOT hace que el contrato esté declarado:
--     - cualquier futuro GRANT debería revertir este REVOKE primero
--       (visibilidad alta en code review),
--     - el test parser-based puede grepar la migración y exigir presencia
--       del REVOKE.
--   `service_role` mantiene EXECUTE — es el único caller legítimo.
--
-- Idempotente:
--   - `CREATE OR REPLACE FUNCTION` sin alterar signature.
--   - `REVOKE` retorna sin error si el grant no existe.
--   - `COMMENT ON FUNCTION` reemplaza el comentario previo.
--   - Re-aplicar la migración es no-op funcional.
--
-- Cross-link:
--   - Test: backend/tests/test_p1_definer_functions_lockdown.py
--   - CLAUDE.md sección "Pattern: SET search_path = ''" tabla "Functions ya
--     bajo el pattern" — añadir 3 filas + sección "Anti-patrones de DEFINER".
--
-- Tooltip-anchor: P1-DEFINER-LOCKDOWN-START | test_p1_definer_functions_lockdown

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────
-- 1) handle_new_user — trigger sobre auth.users
-- ─────────────────────────────────────────────────────────────────────────
-- Body usa `public.user_profiles` qualified + `new.<col>` (campos del
-- trigger row, no resueltos por search_path). Flip a '' es safe.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.user_profiles (id, email, full_name, created_at)
  VALUES (
    new.id,
    new.email,
    new.raw_user_meta_data ->> 'full_name',
    NOW()
  );
  RETURN new;
END;
$$;

REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM anon;
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM authenticated;
GRANT  EXECUTE ON FUNCTION public.handle_new_user() TO service_role;

COMMENT ON FUNCTION public.handle_new_user() IS
'[P1-DEFINER-LOCKDOWN . 2026-05-12] Trigger SECURITY DEFINER sobre auth.users que crea fila correspondiente en public.user_profiles. Usa new.id/new.email/new.raw_user_meta_data del trigger row (no input externo). search_path locked to '''' (P3-NEW-2 pattern). EXECUTE solo concedido a service_role; PUBLIC/anon/authenticated revoked explicitamente para que un futuro GRANT accidental sea visible en code review.';

-- ─────────────────────────────────────────────────────────────────────────
-- 2) get_monthly_plan_count — quota check legacy
-- ─────────────────────────────────────────────────────────────────────────
-- NOTA: actualmente CERO callsites en backend o frontend (función huérfana,
-- probablemente legacy de un quota check sustituido por verify_api_quota).
-- Mantenida porque eliminarla requiere DDL y puede haber consumers no-grep'eables
-- (cron interno, dashboard SRE, otra app). Locked + revoke + COMMENT marca
-- la decisión: "huérfana pero segura".
CREATE OR REPLACE FUNCTION public.get_monthly_plan_count(user_uuid uuid)
RETURNS integer
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT count(*)::integer
  FROM public.meal_plans
  WHERE user_id = user_uuid
    AND created_at >= date_trunc('month', NOW());
$$;

REVOKE EXECUTE ON FUNCTION public.get_monthly_plan_count(uuid) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.get_monthly_plan_count(uuid) FROM anon;
REVOKE EXECUTE ON FUNCTION public.get_monthly_plan_count(uuid) FROM authenticated;
GRANT  EXECUTE ON FUNCTION public.get_monthly_plan_count(uuid) TO service_role;

COMMENT ON FUNCTION public.get_monthly_plan_count(uuid) IS
'[P1-DEFINER-LOCKDOWN . 2026-05-12] Quota legacy: cuenta meal_plans del mes actual para un user_uuid. CERO callsites en backend/frontend (verificado audit 2026-05-12) - huerfana pero mantenida como API estable para cron SRE / dashboard. NO valida user_uuid = auth.uid() internamente porque solo service_role puede invocarla (REVOKE EXECUTE FROM authenticated, anon, public). search_path locked to '''' (P3-NEW-2 pattern).';

-- ─────────────────────────────────────────────────────────────────────────
-- 3) log_unknown_ingredient_rpc — invocada desde db_plans.py:1196
-- ─────────────────────────────────────────────────────────────────────────
-- Backend invoca con SERVICE_ROLE pasando user_id ya validado por
-- get_verified_user_id. Cero callers fuera del backend.
CREATE OR REPLACE FUNCTION public.log_unknown_ingredient_rpc(
  p_user_id uuid,
  p_ingredient text,
  p_raw_text text
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.unknown_ingredients (user_id, ingredient, raw_text, occurrences, last_seen)
  VALUES (p_user_id, p_ingredient, p_raw_text, 1, NOW())
  ON CONFLICT (user_id, ingredient)
  DO UPDATE SET
    occurrences = public.unknown_ingredients.occurrences + 1,
    last_seen   = NOW();
END;
$$;

REVOKE EXECUTE ON FUNCTION public.log_unknown_ingredient_rpc(uuid, text, text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.log_unknown_ingredient_rpc(uuid, text, text) FROM anon;
REVOKE EXECUTE ON FUNCTION public.log_unknown_ingredient_rpc(uuid, text, text) FROM authenticated;
GRANT  EXECUTE ON FUNCTION public.log_unknown_ingredient_rpc(uuid, text, text) TO service_role;

COMMENT ON FUNCTION public.log_unknown_ingredient_rpc(uuid, text, text) IS
'[P1-DEFINER-LOCKDOWN . 2026-05-12] Loguea ingredientes que el LLM produjo pero el sistema de sinónimos no reconocio. Llamada exclusivamente desde backend/db_plans.py:1196 con SERVICE_ROLE; user_id ya validado upstream por get_verified_user_id. NO valida p_user_id = auth.uid() internamente porque solo service_role puede invocarla (REVOKE EXECUTE FROM authenticated, anon, public). search_path locked to '''' (P3-NEW-2 pattern). ON CONFLICT requiere unique index (user_id, ingredient) en public.unknown_ingredients.';

COMMIT;

-- Notificar PostgREST para que recargue el schema cache (CREATE OR REPLACE
-- function changes pueden afectar el cache de signatures aunque no añadimos
-- ni quitamos columns).
NOTIFY pgrst, 'reload schema';
