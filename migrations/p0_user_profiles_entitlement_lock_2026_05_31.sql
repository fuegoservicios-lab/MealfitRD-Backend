-- [P0-TIER-RLS-LOCK · 2026-05-31] Trigger BEFORE UPDATE que impide que los
-- roles cliente de PostgREST (`authenticated`/`anon`) modifiquen las columnas
-- de entitlement/billing de `public.user_profiles`.
--
-- Root cause (audit frontend speed+security 2026-05-31): la policy RLS
-- "Usuarios editan su propio perfil" permite UPDATE de CUALQUIER columna de la
-- propia fila (USING/WITH CHECK = auth.uid()=id, SIN restricción de columna), y
-- `authenticated` tiene UPDATE a nivel de tabla. Combinado, un usuario logueado
-- podía ejecutar desde la consola del browser:
--     supabase.from('user_profiles').update({plan_tier:'ultra'}).eq('id', miId)
-- y auto-otorgarse tier ilimitado (o 'admin'), evadiendo TODO el billing
-- server-side: `auth.verify_api_quota` deriva el límite de créditos de
-- `plan_tier` (backend/auth.py:158-167) y `agent.py:2578` gatea features premium
-- por `plan_tier in [plus,ultra,admin]`. El backend deriva el tier de PayPal con
-- service_role (routers/billing.py:385-491) — pero el hueco client-side lo
-- defiende I-Billing-1 y este trigger lo cierra a nivel DB (defensa-en-profundidad
-- simétrica de la invariante I6 para meal_plans).
--
-- Por qué un trigger y NO `REVOKE UPDATE (col)`: `authenticated` tiene UPDATE a
-- nivel de TABLA, que en Postgres supersede un revoke de columna (el revoke de
-- columna solo aplica si NO hay grant de tabla). Revocar el UPDATE de tabla +
-- re-grant por columna sería frágil (hay que enumerar cada columna editable y un
-- olvido rompe edición de perfil). El trigger es preciso: solo dispara cuando una
-- columna de billing CAMBIA y el rol es cliente.
--
-- Roles exentos: el backend escribe estas columnas como `postgres` (psycopg
-- directo) o `service_role` (supabase-py) — ninguno está en ('authenticated',
-- 'anon'), así que pasan sin tocar. Verificado por dry-run (BEGIN/ROLLBACK) el
-- 2026-05-31: cliente bloqueado en plan_tier, cliente OK en full_name, backend OK
-- en plan_tier. Verificado además que ningún path legítimo del frontend escribe
-- estas columnas (`updateUserProfile` solo recibe {full_name}/{health_profile}).
--
-- Columnas protegidas: plan_tier, paypal_subscription_id, paypal_plan_id,
-- subscription_status, subscription_end_date.
--
-- SECURITY INVOKER a propósito: el trigger debe ver el rol REAL del statement
-- (`current_user` = 'authenticated' para requests PostgREST). SECURITY DEFINER
-- haría `current_user` = owner y el guard nunca dispararía.
--
-- Idempotente: CREATE OR REPLACE FUNCTION + DROP TRIGGER IF EXISTS + CREATE.

CREATE OR REPLACE FUNCTION public.guard_user_profiles_entitlement_columns()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $body$
BEGIN
  -- Solo roles cliente PostgREST. Backend (postgres/service_role) pasa.
  IF current_user IN ('authenticated', 'anon') THEN
    IF NEW.plan_tier              IS DISTINCT FROM OLD.plan_tier
       OR NEW.paypal_subscription_id IS DISTINCT FROM OLD.paypal_subscription_id
       OR NEW.paypal_plan_id         IS DISTINCT FROM OLD.paypal_plan_id
       OR NEW.subscription_status    IS DISTINCT FROM OLD.subscription_status
       OR NEW.subscription_end_date  IS DISTINCT FROM OLD.subscription_end_date THEN
      RAISE EXCEPTION
        'P0-TIER-RLS-LOCK: las columnas de entitlement/billing de user_profiles '
        '(plan_tier, paypal_*, subscription_*) no son modificables por el cliente. '
        'El tier se deriva server-side de PayPal.'
        USING ERRCODE = '42501'; -- insufficient_privilege
    END IF;
  END IF;
  RETURN NEW;
END;
$body$;

DROP TRIGGER IF EXISTS trg_guard_user_profiles_entitlement ON public.user_profiles;

CREATE TRIGGER trg_guard_user_profiles_entitlement
  BEFORE UPDATE ON public.user_profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.guard_user_profiles_entitlement_columns();

COMMENT ON FUNCTION public.guard_user_profiles_entitlement_columns() IS
'[P0-TIER-RLS-LOCK · 2026-05-31] Bloquea que authenticated/anon cambien '
'plan_tier y columnas de suscripción PayPal en user_profiles. Backend '
'(postgres/service_role) exento. Cierra escalación de tier client-side que '
'evadía el billing server-side. SECURITY INVOKER a propósito (lee current_user '
'real). Migración SSOT p0_user_profiles_entitlement_lock_2026_05_31.sql.';

-- Sanity check: el trigger quedó adjunto a user_profiles.
DO $$
DECLARE
  v_count int;
BEGIN
  SELECT COUNT(*) INTO v_count
  FROM pg_trigger t
  JOIN pg_class c ON c.oid = t.tgrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public'
    AND c.relname = 'user_profiles'
    AND t.tgname = 'trg_guard_user_profiles_entitlement'
    AND NOT t.tgisinternal;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'P0-TIER-RLS-LOCK: el trigger no quedó adjunto (count=%).', v_count;
  END IF;
END $$;
