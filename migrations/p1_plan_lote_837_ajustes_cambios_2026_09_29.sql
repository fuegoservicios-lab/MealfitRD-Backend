-- [P1-PLAN-LOTE-837 · 2026-09-29] Los ajustes de cada cuenta (spec 2026-09-29-admin-cuentas-actividad-pruebas-design
-- §13.2 y §13.3): el historial de cambios de los ajustes y los ajustes que viven solo en el teléfono.
--
-- 1. `public.ajustes_cambios`: una fila por ajuste que CAMBIA en `user_profiles` (antes, después, quién y cuándo). La
--    llena el trigger `trg_ajustes_cambios`, así que cubre a TODOS los escritores (Configuración, el coach, el sistema y
--    los que vengan) sin tocarlos. Vigila SOLO los ajustes del registro (`backend/ajustes_cuenta.py`): nunca el perfil de
--    salud. Se exporta con la cuenta, se va con ella (ON DELETE CASCADE) y se purga a los
--    MEALFIT_AJUSTES_CAMBIOS_RETENTION_DAYS (730) días (cron diario `purge_ajustes_cambios`).
-- 2. El ORIGEN (`app` | `coach` | `sistema`): quien escribe lo fija EN LA MISMA SENTENCIA,
--        UPDATE … FROM (SELECT set_config('mealfit.origen_ajuste', '<origen>', true)) AS _origen WHERE …
--    (`ajustes_cuenta.sql_con_origen`). `true` = local a la transacción: el backend va en autocommit, así que la
--    transacción es la propia sentencia; el trigger AFTER corre al final de esa sentencia, dentro de ella, y lo ve; en la
--    siguiente ya no existe. Sin marca ⇒ `app` (la persona). La Nevera que se apaga pasando `nevera_auto_off_at` de NULL
--    a un valor en el mismo UPDATE es del sistema aunque nadie lo marque.
-- 3. El trigger es `AFTER UPDATE OF <columnas vigiladas>, health_profile` y su `WHEN` compara SOLO las columnas y claves
--    vigiladas: un UPDATE de `fact_locked_at` (el lock del extractor de hechos, muy frecuente) ni lo considera. Registrar
--    NUNCA tumba el UPDATE de la persona: un fallo se queda en un WARNING.
-- 4. `user_profiles.ajustes_dispositivo`: los ajustes que solo existen en el teléfono (tema, permiso de notificaciones…),
--    por plataforma (`{"web": {…, "at"}, "ios": …}`). Lo escribe solo `ajustes_cuenta.guardar_dispositivo`; el trigger
--    no lo mira.
-- Idempotente. La aplica el controlador en el despliegue (no las tareas).

-- Las columnas vigiladas tienen que existir: el trigger las nombra. Si falta una, falta aplicar antes su migración
-- (p1_plan_mode, p1_nevera_opcional, p1_i18n_dashboard_locale, p1_plan_lote_843_user_consents…).
DO $$
DECLARE
    v_faltan text;
BEGIN
    SELECT string_agg(c, ', ') INTO v_faltan
      FROM unnest(ARRAY['plan_mode', 'logging_preference', 'long_term_memory_enabled', 'water_tracker_enabled',
                        'nevera_enabled', 'nevera_auto_off_at', 'locale', 'analytics_consent', 'ai_training_consent',
                        'ai_consent_version', 'ai_consent_revoked_at', 'health_profile']) AS c
     WHERE NOT EXISTS (SELECT 1 FROM information_schema.columns ic
                        WHERE ic.table_schema = 'public' AND ic.table_name = 'user_profiles'
                          AND ic.column_name::text = c);
    IF v_faltan IS NOT NULL THEN
        RAISE EXCEPTION 'p1_plan_lote_837: faltan columnas en user_profiles (%): aplica antes sus migraciones', v_faltan;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS public.ajustes_cambios (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    clave TEXT NOT NULL,
    antes JSONB,
    despues JSONB,
    origen TEXT NOT NULL DEFAULT 'app',
    at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- La ficha y el historial de UNA cuenta (y el borrado en cascada de la FK).
CREATE INDEX IF NOT EXISTS ajustes_cambios_user_at_idx ON public.ajustes_cambios (user_id, at DESC);
-- El resumen del periodo y la purga por plazo.
CREATE INDEX IF NOT EXISTS ajustes_cambios_at_idx ON public.ajustes_cambios (at DESC);

REVOKE ALL ON public.ajustes_cambios FROM PUBLIC;

COMMENT ON TABLE public.ajustes_cambios IS
    'P1-PLAN-LOTE-837: un cambio por fila de los ajustes vigilados de user_profiles (antes, despues, origen app|coach|sistema). Lo llena el trigger trg_ajustes_cambios; lo leen y lo purgan backend/ajustes_cuenta.py.';

-- Los ajustes que viven solo en el teléfono, por plataforma. Siempre un objeto (el lector y el jsonb_set lo asumen).
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS ajustes_dispositivo JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.user_profiles DROP CONSTRAINT IF EXISTS user_profiles_ajustes_dispositivo_objeto_chk;
ALTER TABLE public.user_profiles ADD CONSTRAINT user_profiles_ajustes_dispositivo_objeto_chk
    CHECK (jsonb_typeof(ajustes_dispositivo) = 'object');

COMMENT ON COLUMN public.user_profiles.ajustes_dispositivo IS
    'P1-PLAN-LOTE-837: ajustes que solo existen en el dispositivo (tema, permiso de notificaciones...), por plataforma {"web": {..., "at"}, "ios": ..., "android": ...}. Lo escribe solo backend/ajustes_cuenta.guardar_dispositivo (lista cerrada).';

-- Una fila por clave vigilada que cambió. SECURITY INVOKER: escribe con los permisos de quien hace el UPDATE (el backend).
-- `search_path` vacío: todo nombre va calificado (`public.`) y nadie puede suplantarlo con una tabla temporal.
CREATE OR REPLACE FUNCTION public.registrar_cambio_ajustes()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $body$
DECLARE
    v_origen text := COALESCE(NULLIF(current_setting('mealfit.origen_ajuste', true), ''), 'app');
    v_viejo jsonb := to_jsonb(OLD);
    v_nuevo jsonb := to_jsonb(NEW);
    v_clave text;
    v_antes jsonb;
    v_despues jsonb;
BEGIN
    IF v_origen NOT IN ('app', 'coach', 'sistema') THEN
        v_origen := 'app';
    END IF;
    BEGIN
        FOREACH v_clave IN ARRAY ARRAY['plan_mode', 'logging_preference', 'long_term_memory_enabled',
                                       'water_tracker_enabled', 'nevera_enabled', 'locale', 'analytics_consent',
                                       'ai_training_consent', 'ai_consent_version', 'ai_consent_revoked_at']::text[]
        LOOP
            v_antes := NULLIF(v_viejo -> v_clave, 'null'::jsonb);
            v_despues := NULLIF(v_nuevo -> v_clave, 'null'::jsonb);
            IF v_antes IS DISTINCT FROM v_despues THEN
                INSERT INTO public.ajustes_cambios (user_id, clave, antes, despues, origen)
                VALUES (NEW.id, v_clave, v_antes, v_despues,
                        CASE WHEN v_clave = 'nevera_enabled' AND NEW.nevera_enabled IS FALSE
                                  AND OLD.nevera_auto_off_at IS NULL AND NEW.nevera_auto_off_at IS NOT NULL
                             THEN 'sistema' ELSE v_origen END);
            END IF;
        END LOOP;
        FOREACH v_clave IN ARRAY ARRAY['avisos_comida', 'avisos_agua', 'avisos_por_comida', 'country',
                                       'groceryDuration', 'budget', 'budgetCurrency', 'weightUnit']::text[]
        LOOP
            v_antes := NULLIF(OLD.health_profile -> v_clave, 'null'::jsonb);
            v_despues := NULLIF(NEW.health_profile -> v_clave, 'null'::jsonb);
            IF v_antes IS DISTINCT FROM v_despues THEN
                INSERT INTO public.ajustes_cambios (user_id, clave, antes, despues, origen)
                VALUES (NEW.id, v_clave, v_antes, v_despues, v_origen);
            END IF;
        END LOOP;
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'p1_plan_lote_837: cambio de ajustes sin registrar (cuenta %): % (%)', NEW.id, SQLERRM, SQLSTATE;
    END;
    RETURN NULL;
END;
$body$;

COMMENT ON FUNCTION public.registrar_cambio_ajustes() IS
    'P1-PLAN-LOTE-837: trigger AFTER UPDATE de user_profiles. Anota en ajustes_cambios una fila por ajuste vigilado que cambio; origen = mealfit.origen_ajuste (lo fija quien escribe, en la misma sentencia) o app. Un fallo no tumba el UPDATE (WARNING).';

DROP TRIGGER IF EXISTS trg_ajustes_cambios ON public.user_profiles;

CREATE TRIGGER trg_ajustes_cambios
    AFTER UPDATE OF plan_mode, logging_preference, long_term_memory_enabled, water_tracker_enabled, nevera_enabled,
        locale, analytics_consent, ai_training_consent, ai_consent_version, ai_consent_revoked_at, health_profile
    ON public.user_profiles
    FOR EACH ROW
    WHEN (OLD.plan_mode IS DISTINCT FROM NEW.plan_mode
          OR OLD.logging_preference IS DISTINCT FROM NEW.logging_preference
          OR OLD.long_term_memory_enabled IS DISTINCT FROM NEW.long_term_memory_enabled
          OR OLD.water_tracker_enabled IS DISTINCT FROM NEW.water_tracker_enabled
          OR OLD.nevera_enabled IS DISTINCT FROM NEW.nevera_enabled
          OR OLD.locale IS DISTINCT FROM NEW.locale
          OR OLD.analytics_consent IS DISTINCT FROM NEW.analytics_consent
          OR OLD.ai_training_consent IS DISTINCT FROM NEW.ai_training_consent
          OR OLD.ai_consent_version IS DISTINCT FROM NEW.ai_consent_version
          OR OLD.ai_consent_revoked_at IS DISTINCT FROM NEW.ai_consent_revoked_at
          OR (OLD.health_profile -> 'avisos_comida') IS DISTINCT FROM (NEW.health_profile -> 'avisos_comida')
          OR (OLD.health_profile -> 'avisos_agua') IS DISTINCT FROM (NEW.health_profile -> 'avisos_agua')
          OR (OLD.health_profile -> 'avisos_por_comida') IS DISTINCT FROM (NEW.health_profile -> 'avisos_por_comida')
          OR (OLD.health_profile -> 'country') IS DISTINCT FROM (NEW.health_profile -> 'country')
          OR (OLD.health_profile -> 'groceryDuration') IS DISTINCT FROM (NEW.health_profile -> 'groceryDuration')
          OR (OLD.health_profile -> 'budget') IS DISTINCT FROM (NEW.health_profile -> 'budget')
          OR (OLD.health_profile -> 'budgetCurrency') IS DISTINCT FROM (NEW.health_profile -> 'budgetCurrency')
          OR (OLD.health_profile -> 'weightUnit') IS DISTINCT FROM (NEW.health_profile -> 'weightUnit'))
    EXECUTE FUNCTION public.registrar_cambio_ajustes();

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'ajustes_cambios') THEN
        RAISE EXCEPTION 'p1_plan_lote_837: ajustes_cambios no existe tras la migracion';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_class r ON r.oid = c.confrelid
        WHERE t.relname = 'ajustes_cambios' AND r.relname = 'user_profiles'
          AND c.contype = 'f' AND c.confdeltype = 'c'
    ) THEN
        RAISE EXCEPTION 'p1_plan_lote_837: falta el FK en cascada a user_profiles';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'public' AND table_name = 'user_profiles'
                     AND column_name = 'ajustes_dispositivo' AND is_nullable = 'NO') THEN
        RAISE EXCEPTION 'p1_plan_lote_837: falta user_profiles.ajustes_dispositivo NOT NULL';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        CROSS JOIN LATERAL unnest(p.proconfig) AS cfg
        WHERE n.nspname = 'public' AND p.proname = 'registrar_cambio_ajustes'
          AND cfg LIKE 'search_path=%' AND position('public' IN cfg) = 0
    ) THEN
        RAISE EXCEPTION 'p1_plan_lote_837: registrar_cambio_ajustes sin SET search_path vacio';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'user_profiles'
          AND t.tgname = 'trg_ajustes_cambios' AND NOT t.tgisinternal
    ) THEN
        RAISE EXCEPTION 'p1_plan_lote_837: el trigger trg_ajustes_cambios no quedo adjunto';
    END IF;
END $$;
