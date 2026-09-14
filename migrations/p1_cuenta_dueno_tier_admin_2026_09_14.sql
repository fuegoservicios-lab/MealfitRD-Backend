-- [P1-CUENTA-DUENO-ADMIN · 2026-09-14] La cuenta del dueño (la de las pruebas RD) pasa al tier interno `admin`:
-- 999.999 créditos/mes (`auth._TIER_LIMITS`) en vez de los 10 de `gratis`, agotados con las 10 generaciones de las
-- pruebas del 9 y del 14 de septiembre (medidor 0/10). Pedido por el dueño en el chat el 14-sep.
--
-- `admin` NO está en `llm_provider.PAID_TIERS` (basic, plus, ultra): la generación sigue igual que en `gratis`
-- (día generado con Luna low, sin self-play adversarial), así que las pruebas siguen midiendo lo mismo. El historial
-- de `api_usage` no se toca.
--
-- Revertir: UPDATE public.user_profiles SET plan_tier = 'gratis', updated_at = now()
--           WHERE id = '4da5c079-f0ba-47af-87d1-3ec732d187d9';
--
-- Idempotente. El trigger P0-TIER-RLS-LOCK sólo bloquea a los roles cliente (authenticated/anon).

UPDATE public.user_profiles
   SET plan_tier = 'admin', updated_at = now()
 WHERE id = '4da5c079-f0ba-47af-87d1-3ec732d187d9'
   AND plan_tier IS DISTINCT FROM 'admin';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.user_profiles
                  WHERE id = '4da5c079-f0ba-47af-87d1-3ec732d187d9' AND plan_tier = 'admin') THEN
    RAISE EXCEPTION 'P1-CUENTA-DUENO-ADMIN: la cuenta del dueño no quedó en el tier admin';
  END IF;
END $$;
