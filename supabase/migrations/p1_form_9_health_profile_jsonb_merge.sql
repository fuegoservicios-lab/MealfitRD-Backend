-- [P1-FORM-9] RPC para JSONB merge de `user_profiles.health_profile`.
--
-- ANTES: el frontend llamaba directamente
--     supabase.from('user_profiles').update({health_profile: {...}}).eq('id', uid)
-- que REEMPLAZA la columna entera. Si el cliente enviaba un dict parcial
-- (porque la hidratación cifrada de `mealfit_form_secure` aún no había
-- terminado y `formData` tenía `medicalConditions=[]`/`allergies=[]`), el
-- update borraba los datos médicos previos persistidos en DB.
--
-- El frontend ya tiene defensa de primera línea vía
-- `buildHealthProfilePayload` (`secureFormStorage.js`), que detecta el race
-- y aborta. Esta RPC es DEFENSE-IN-DEPTH a nivel DB: aunque un cliente legacy
-- o una integración externa envíe un dict parcial, la columna se MERGEA con
-- el `||` operator de jsonb en lugar de reemplazarse — solo las keys
-- explícitas en el patch sobrescriben; el resto del JSON se preserva.
--
-- Contrato:
--   - `target_user_id`: UUID del usuario cuyo perfil se actualiza. Solo se
--     permite que coincida con `auth.uid()` (mismo usuario que el JWT).
--     Esto reemplaza la guarda de RLS que el `update()` original tenía
--     implícitamente vía la policy de la tabla.
--   - `patch`: jsonb con las claves a mergear. Si una key existe en el JSON
--     actual y en el patch, el patch gana (semántica de `||`). Si una key
--     existe SOLO en el JSON actual, se PRESERVA.
--
-- Ejemplo:
--   Estado actual: `{"allergies": ["Mani"], "medicalConditions": ["Diabetes"]}`
--   Patch:        `{"householdSize": 4}`
--   Resultado:    `{"allergies": ["Mani"], "medicalConditions": ["Diabetes"],
--                   "householdSize": 4}`
--
--   Mismo escenario con `update()` tradicional habría producido:
--                  `{"householdSize": 4}` — datos médicos PERDIDOS.
--
-- Idempotente: re-ejecutar la migración recrea la función con el mismo body.
-- `CREATE OR REPLACE` permite ajustar el body sin DROP.
--
-- Seguridad: SECURITY INVOKER (default) — la función corre con los permisos
-- del CALLER, así que las policies RLS de `user_profiles` se aplican. El
-- guard adicional `WHERE id = auth.uid()` añade defensa en profundidad.

CREATE OR REPLACE FUNCTION public.update_health_profile_merge(patch jsonb)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    -- Validación: solo permitimos a un usuario actualizar su propio perfil.
    -- `auth.uid()` retorna el UUID del JWT del request; si no hay JWT
    -- (anon key sin auth), retorna NULL → el WHERE no matchea ninguna fila
    -- y la operación es no-op (segura).
    IF auth.uid() IS NULL THEN
        RAISE EXCEPTION 'No authenticated user — cannot update health_profile.'
            USING ERRCODE = '28000'; -- invalid_authorization_specification
    END IF;

    -- Validación de tipo del patch: debe ser un objeto JSON, no array ni
    -- escalar. `jsonb_typeof` retorna 'object'/'array'/'string'/etc.
    IF jsonb_typeof(patch) IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'patch must be a JSON object, got: %', jsonb_typeof(patch)
            USING ERRCODE = '22023'; -- invalid_parameter_value
    END IF;

    -- JSONB merge: `||` mergea top-level keys (NO recursivo). Si una key
    -- existe en ambos lados, el operando derecho gana. Para preservar la
    -- semántica del `update()` original (donde el cliente reemplaza el array
    -- entero, no entradas individuales), el merge top-level es exactamente
    -- el comportamiento deseado.
    --
    -- COALESCE protege contra rows con `health_profile IS NULL` (rarísimo,
    -- pero posible si la fila se creó con un INSERT explícito sin defaults).
    UPDATE public.user_profiles
    SET health_profile = COALESCE(health_profile, '{}'::jsonb) || patch
    WHERE id = auth.uid();

    -- No raise si no se actualiza fila — significa que la fila no existe
    -- (usuario nuevo sin profile aún) y el caller debería haber llamado a
    -- INSERT antes. Mantenemos silencio para no romper flujos parciales.
END;
$$;

-- [P1-FORM-9] Hardening de grants — defensa en capas:
--   - REVOKE FROM PUBLIC: Postgres concede EXECUTE a PUBLIC por default al
--     crear funciones; sin esta revocación el GRANT TO authenticated era
--     redundante y anon API key podía invocar la RPC.
--   - REVOKE FROM anon: Supabase también concede explícitamente EXECUTE a
--     anon/authenticated/service_role para toda función en `public`. Como
--     esta RPC requiere JWT, anon NO debe poder invocarla — la función
--     raise si auth.uid() IS NULL, pero gatear en el GRANT es mejor:
--     PostgREST corta con 403 antes de entrar a plpgsql.
--   - GRANT TO authenticated: usuarios con JWT válido (path normal del
--     frontend tras login) — ÚNICO caller esperado de la RPC.
--   - service_role mantiene EXECUTE (default Supabase) — cron jobs / admin
--     operations pueden usar la RPC con la service_role key.
REVOKE EXECUTE ON FUNCTION public.update_health_profile_merge(jsonb) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.update_health_profile_merge(jsonb) FROM anon;
GRANT  EXECUTE ON FUNCTION public.update_health_profile_merge(jsonb) TO authenticated;

-- Comentario de documentación accesible desde `psql \df+`.
COMMENT ON FUNCTION public.update_health_profile_merge(jsonb) IS
    '[P1-FORM-9] Merge de health_profile JSONB del usuario autenticado. '
    'Reemplaza el patrón inseguro `update({health_profile: {...}})` que '
    'borraba datos médicos previos cuando el cliente enviaba un dict '
    'parcial durante la race de hidratación cifrada post-login. '
    'Ver buildHealthProfilePayload en frontend/src/config/secureFormStorage.js.';
