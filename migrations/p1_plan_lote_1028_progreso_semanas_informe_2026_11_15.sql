-- [P1-PLAN-LOTE-1028 · 2026-11-15] Progreso por semanas numeradas («Semana 1, 2, 3… hasta el infinito») e informe para el
-- médico. Plan: docs/superpowers/plans/2026-11-15-progreso-semanas-coach-informe-medico.md (§3, «Base de datos»).
--
-- Tres tablas, todas de la persona (FK a user_profiles ON DELETE CASCADE: se van con la cuenta):
--
-- 1. `public.user_journey` — el DÍA 1 de cada persona: el primer día (biológico, en su huso) con un registro en el diario.
--    Hasta hoy ese día solo vivía en el teléfono (localStorage `mealfit_semana_inicio_{uid}`) y se deducía de los últimos
--    365 días: en otro dispositivo, o pasado un año, la cuenta de semanas cambiaba. Aquí queda fijo. `week1_start` es el
--    DOMINGO de esa semana (columna generada): las semanas van de domingo a sábado, como las filas del calendario.
--    Regla de escritura (backend): solo se MUEVE HACIA ATRÁS (`LEAST(actual, nuevo)`); borrar la primera comida no
--    reinicia el viaje.
-- 2. `public.user_week_summaries` — una fila por semana numerada: días registrados, en meta, bajo y sobre, estrellas
--    (0–3: 3, 5 y 7 días en meta), medias de kcal y proteína. La escribe el backend (`progreso_semanas.py`): al cerrarse
--    la semana y cada vez que cambia el diario de uno de sus días. La leen el coach (un SELECT, sin recomputar un año),
--    la app (semanas de hace más de 365 días) y el informe. NO se rellena aquí: «día en meta» depende de la meta vigente
--    de CADA día (`diary_targets.resolve_targets_by_day`) y de la banda por objetivo, que viven en Python.
-- 3. `public.user_health_reports` — los informes para el médico que la persona genera: el periodo, los DATOS calculados
--    por el servidor (`data`, la única fuente de las cifras) y el texto redactado por el modelo (`narrative`, que no
--    puede introducir cifras nuevas). Es dato de salud: se exporta con la cuenta y se purga por plazo (cron del backend).
--    No es un diagnóstico: es un resumen de lo registrado para llevar a consulta.
--
-- Idempotente. Aplicada a Neon por el controlador el 2026-11-15 (apply_migration.py --apply), ANTES del backend que la usa.

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'user_profiles') THEN
        RAISE EXCEPTION 'p1_plan_lote_1028: falta public.user_profiles';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'consumed_meals') THEN
        RAISE EXCEPTION 'p1_plan_lote_1028: falta public.consumed_meals';
    END IF;
END $$;

-- ───────────────────────────────────────────────────────────── 1. el día 1
CREATE TABLE IF NOT EXISTS public.user_journey (
    user_id UUID PRIMARY KEY REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    first_log_date DATE NOT NULL,
    -- el domingo de la semana del primer registro: el día 1 de la Semana 1
    week1_start DATE GENERATED ALWAYS AS (first_log_date - EXTRACT(DOW FROM first_log_date)::int) STORED,
    -- el huso con el que se cortó `first_log_date` (minutos, convención getTimezoneOffset de JS: RD = 240)
    tz_offset_min INTEGER NOT NULL DEFAULT 240 CHECK (tz_offset_min BETWEEN -840 AND 840),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

REVOKE ALL ON public.user_journey FROM PUBLIC;

COMMENT ON TABLE public.user_journey IS
    'P1-PLAN-LOTE-1028: el dia 1 de cada persona (primer dia biologico con registro en el diario). week1_start = el domingo de esa semana; las semanas numeradas van de domingo a sabado. Solo se mueve hacia atras (LEAST). La escribe backend/progreso_semanas.py.';

-- Relleno inicial: el primer día registrado de quien ya tiene diario. El día se corta como en /api/diary/consumed-range:
-- huso del perfil (tz_offset_minutes; 240 = RD si falta o no es un número) + 4 h de «día biológico» (rollover 04:00).
INSERT INTO public.user_journey (user_id, first_log_date, tz_offset_min)
SELECT m.user_id,
       MIN((m.consumed_at - make_interval(mins => t.off_min + 240))::date),
       t.off_min
  FROM public.consumed_meals m
  JOIN public.user_profiles p ON p.id = m.user_id
  CROSS JOIN LATERAL (
      SELECT CASE
                 WHEN (p.health_profile->>'tz_offset_minutes') ~ '^-?[0-9]{1,3}$'
                      AND (p.health_profile->>'tz_offset_minutes')::int BETWEEN -840 AND 840
                 THEN (p.health_profile->>'tz_offset_minutes')::int
                 ELSE 240
             END AS off_min
  ) t
 WHERE m.consumed_at IS NOT NULL
 GROUP BY m.user_id, t.off_min
ON CONFLICT (user_id) DO NOTHING;

-- ───────────────────────────────────────────────────────────── 2. una fila por semana numerada
CREATE TABLE IF NOT EXISTS public.user_week_summaries (
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    week_number INTEGER NOT NULL CHECK (week_number >= 1),
    week_start DATE NOT NULL,                       -- domingo
    days_logged SMALLINT NOT NULL DEFAULT 0 CHECK (days_logged BETWEEN 0 AND 7),
    days_on_goal SMALLINT NOT NULL DEFAULT 0 CHECK (days_on_goal BETWEEN 0 AND 7),
    days_under SMALLINT NOT NULL DEFAULT 0 CHECK (days_under BETWEEN 0 AND 7),
    days_over SMALLINT NOT NULL DEFAULT 0 CHECK (days_over BETWEEN 0 AND 7),
    stars SMALLINT NOT NULL DEFAULT 0 CHECK (stars BETWEEN 0 AND 3),
    kcal_avg INTEGER,                               -- media de los días CON registro (NULL si no hubo)
    protein_avg INTEGER,
    closed BOOLEAN NOT NULL DEFAULT false,          -- true cuando la semana ya terminó (su sábado quedó atrás)
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, week_number),
    CHECK (days_on_goal + days_under + days_over = days_logged)
);

REVOKE ALL ON public.user_week_summaries FROM PUBLIC;

COMMENT ON TABLE public.user_week_summaries IS
    'P1-PLAN-LOTE-1028: resumen por semana numerada (domingo a sabado): dias registrados, en meta, bajo, sobre, estrellas 0-3 (3/5/7 dias en meta) y medias. La escribe backend/progreso_semanas.py al cerrar la semana y al cambiar el diario de uno de sus dias; la leen el coach, la app y el informe.';

-- ───────────────────────────────────────────────────────────── 3. informes para el médico
CREATE TABLE IF NOT EXISTS public.user_health_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    locale TEXT NOT NULL DEFAULT 'es-DO',
    status TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'data_only', 'failed')),
    data JSONB NOT NULL,                            -- las cifras, calculadas por el servidor (única fuente de números)
    narrative TEXT,                                 -- la redacción del modelo; NULL en 'data_only' / 'failed'
    narrative_model TEXT,
    cost_usd_micros INTEGER NOT NULL DEFAULT 0 CHECK (cost_usd_micros >= 0),
    origin TEXT NOT NULL DEFAULT 'app' CHECK (origin IN ('app', 'coach')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (period_end >= period_start)
);

-- La lista de informes de UNA persona (y el tope de informes por día).
CREATE INDEX IF NOT EXISTS user_health_reports_user_created_idx
    ON public.user_health_reports (user_id, created_at DESC);
-- La purga por plazo.
CREATE INDEX IF NOT EXISTS user_health_reports_created_idx
    ON public.user_health_reports (created_at);

REVOKE ALL ON public.user_health_reports FROM PUBLIC;

COMMENT ON TABLE public.user_health_reports IS
    'P1-PLAN-LOTE-1028: informes para el medico generados por la persona. data = cifras calculadas por el servidor; narrative = redaccion del modelo (sin cifras nuevas). Dato de salud: se exporta con la cuenta y se purga por plazo. No es un diagnostico.';

-- Sanity: las tres tablas existen y el relleno no dejó a nadie con un día 1 en el futuro.
DO $$
DECLARE
    v_futuros integer;
BEGIN
    IF to_regclass('public.user_journey') IS NULL OR to_regclass('public.user_week_summaries') IS NULL
       OR to_regclass('public.user_health_reports') IS NULL THEN
        RAISE EXCEPTION 'p1_plan_lote_1028: no se crearon las tres tablas';
    END IF;
    SELECT count(*) INTO v_futuros FROM public.user_journey WHERE first_log_date > (now() + interval '1 day')::date;
    IF v_futuros > 0 THEN
        RAISE EXCEPTION 'p1_plan_lote_1028: % fila(s) de user_journey con el dia 1 en el futuro', v_futuros;
    END IF;
END $$;

COMMIT;
