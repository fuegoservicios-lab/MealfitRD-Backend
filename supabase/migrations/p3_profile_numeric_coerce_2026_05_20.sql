-- [P3-PROFILE-NUMERIC-COERCE · 2026-05-20] Normaliza `health_profile`
-- numeric fields (weight, height, age, bodyFat) de JSON string a JSON
-- number en filas legacy.
--
-- Source del drift: el wizard `InteractiveQuestions` (frontend) usa
-- `e.target.value` directo en `updateData('weight', ...)` que es string;
-- el spread persiste `formData` tal cual al jsonb. Pre-fix (P3-PROFILE-
-- NUMERIC-COERCE en buildHealthProfilePayload, frontend) las escrituras
-- nuevas ya son numbers — esta migración limpia las filas previas.
--
-- Comportamiento:
--   - Solo toca filas donde el field es `jsonb_typeof = 'string'` Y el
--     contenido matchea `^-?\d+(\.\d+)?$` (entero o decimal estricto, sin
--     `NaN`/`Infinity`/sufijos/scientific notation/coma decimal). El
--     filtro por regex es defensivo: si algún row legacy tiene "70,5"
--     (coma decimal pre-P1-9), el cast a numeric fallaría — preferimos
--     dejarlo y dejar que P1-9 lo normalice on next write.
--   - Idempotente: vuelve a correr y no toca filas ya numéricas (typeof
--     no es 'string' → falta del WHERE).
--   - Por field separado: si un row tiene weight numeric pero height
--     string, solo height se normaliza.
--   - jsonb_set quirúrgico (no full overwrite), preserva el resto del
--     health_profile.
--
-- Idempotencia + safety: el filtro `jsonb_typeof = 'string'` + regex
-- garantiza que ejecuciones repetidas o filas válidas no se tocan.

DO $$
BEGIN
    -- weight
    UPDATE public.user_profiles
    SET health_profile = jsonb_set(
        health_profile,
        '{weight}',
        to_jsonb((health_profile->>'weight')::numeric)
    )
    WHERE jsonb_typeof(health_profile->'weight') = 'string'
      AND (health_profile->>'weight') ~ '^-?[0-9]+(\.[0-9]+)?$';

    -- height
    UPDATE public.user_profiles
    SET health_profile = jsonb_set(
        health_profile,
        '{height}',
        to_jsonb((health_profile->>'height')::numeric)
    )
    WHERE jsonb_typeof(health_profile->'height') = 'string'
      AND (health_profile->>'height') ~ '^-?[0-9]+(\.[0-9]+)?$';

    -- age
    UPDATE public.user_profiles
    SET health_profile = jsonb_set(
        health_profile,
        '{age}',
        to_jsonb((health_profile->>'age')::numeric)
    )
    WHERE jsonb_typeof(health_profile->'age') = 'string'
      AND (health_profile->>'age') ~ '^-?[0-9]+(\.[0-9]+)?$';

    -- bodyFat
    UPDATE public.user_profiles
    SET health_profile = jsonb_set(
        health_profile,
        '{bodyFat}',
        to_jsonb((health_profile->>'bodyFat')::numeric)
    )
    WHERE jsonb_typeof(health_profile->'bodyFat') = 'string'
      AND (health_profile->>'bodyFat') ~ '^-?[0-9]+(\.[0-9]+)?$';

    -- Sanity check post-update: en filas válidas (no-null health_profile
    -- con field presente y string que matchea numeric pattern), el typeof
    -- debe ser 'number'. Si la migración corrió pero algo quedó string
    -- legítimo (e.g. valor con coma decimal), el RAISE NOTICE expone
    -- el conteo restante para visibility.
    DECLARE
        v_remaining_strings INTEGER;
    BEGIN
        SELECT COUNT(*) INTO v_remaining_strings
        FROM public.user_profiles
        WHERE (
            jsonb_typeof(health_profile->'weight') = 'string'
            AND (health_profile->>'weight') ~ '^-?[0-9]+(\.[0-9]+)?$'
        ) OR (
            jsonb_typeof(health_profile->'height') = 'string'
            AND (health_profile->>'height') ~ '^-?[0-9]+(\.[0-9]+)?$'
        ) OR (
            jsonb_typeof(health_profile->'age') = 'string'
            AND (health_profile->>'age') ~ '^-?[0-9]+(\.[0-9]+)?$'
        ) OR (
            jsonb_typeof(health_profile->'bodyFat') = 'string'
            AND (health_profile->>'bodyFat') ~ '^-?[0-9]+(\.[0-9]+)?$'
        );

        IF v_remaining_strings > 0 THEN
            RAISE EXCEPTION '[P3-PROFILE-NUMERIC-COERCE] % filas con campos numéricos en string tras la migración — UPDATE no aplicó.', v_remaining_strings;
        END IF;
    END;
END $$;
