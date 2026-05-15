-- [P0-HIST-3 · 2026-05-09] FK con ON DELETE SET NULL en
-- chunk_lesson_telemetry.meal_plan_id y chunk_deferrals.meal_plan_id.
--
-- Motivación (audit historial 2026-05-08):
--   Estas dos tablas tienen columna `meal_plan_id uuid` pero CERO FK
--   contra `meal_plans(id)`. Cuando el usuario borra un plan desde el
--   Historial, los rows correspondientes quedan huérfanos (con un id
--   que ya no apunta a ninguna fila) y la tabla crece monótona. No
--   es un bug funcional inmediato (las tablas son append-only y los
--   crons agregadores no rompen con NULL), pero genera deuda técnica:
--     - Costo de scan en `_aggregate_coherence_block_history_metrics`
--       y `_per_user_synthesis_ratio_exceeded` crece con orphans.
--     - Análisis post-mortem (queries por meal_plan_id) devuelve
--       resultados confusos (rows con meal_plan_id de un plan
--       eliminado no son recuperables).
--
-- Decisión: SET NULL (no CASCADE).
--   `chunk_lesson_telemetry` es append-only y representa el aprendizaje
--   continuo del sistema. Eliminar las lecciones porque su plan fue
--   borrado destruiría telemetría histórica que sigue siendo válida
--   (las lecciones ya fueron consumidas por chunks subsecuentes y
--   contribuyeron al perfil del usuario). SET NULL preserva la
--   telemetría agregable mientras desreferencia el plan eliminado.
--   Lo mismo para `chunk_deferrals`: la razón del defer es válida
--   independiente del plan que la disparó.
--
-- Idempotencia: misma estrategia que p2_alpha_plan_chunk_queue_fk_cascade
-- (DO block + lookup en information_schema). Si la FK ya existe con la
-- delete_rule correcta, RAISE NOTICE y skip; si existe con otra rule,
-- DROP+ADD; si no existe, ADD directo.
--
-- Pre-condición verificada al diseño: ambas tablas tenían 0 rows con
-- `meal_plan_id NOT IN (SELECT id FROM meal_plans)` al aplicar (no hay
-- orphans previos que requieran cleanup antes de la FK).

DO $$
DECLARE
    fk_record RECORD;
BEGIN
    -- ============================================================
    -- 1. chunk_lesson_telemetry.meal_plan_id → meal_plans(id) ON DELETE SET NULL
    -- ============================================================
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'chunk_lesson_telemetry'
    ) THEN
        fk_record := NULL;
        SELECT tc.constraint_name, rc.delete_rule
          INTO fk_record
          FROM information_schema.table_constraints tc
          JOIN information_schema.referential_constraints rc
               ON tc.constraint_name = rc.constraint_name
          JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
         WHERE tc.table_schema = 'public'
           AND tc.table_name = 'chunk_lesson_telemetry'
           AND tc.constraint_type = 'FOREIGN KEY'
           AND kcu.column_name = 'meal_plan_id'
         LIMIT 1;

        IF fk_record.constraint_name IS NULL THEN
            -- Asegurar columna nullable (SET NULL requiere nullable).
            BEGIN
                ALTER TABLE public.chunk_lesson_telemetry
                    ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL; -- ya era nullable
            END;
            ALTER TABLE public.chunk_lesson_telemetry
                ADD CONSTRAINT chunk_lesson_telemetry_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P0-HIST-3] Creada FK chunk_lesson_telemetry.meal_plan_id → meal_plans.id ON DELETE SET NULL.';
        ELSIF fk_record.delete_rule <> 'SET NULL' THEN
            EXECUTE format(
                'ALTER TABLE public.chunk_lesson_telemetry DROP CONSTRAINT %I',
                fk_record.constraint_name
            );
            BEGIN
                ALTER TABLE public.chunk_lesson_telemetry
                    ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
            ALTER TABLE public.chunk_lesson_telemetry
                ADD CONSTRAINT chunk_lesson_telemetry_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P0-HIST-3] Reemplazada FK chunk_lesson_telemetry.meal_plan_id (delete_rule=% → SET NULL).', fk_record.delete_rule;
        ELSE
            RAISE NOTICE '[P0-HIST-3] FK chunk_lesson_telemetry.meal_plan_id ya tiene ON DELETE SET NULL. Skip.';
        END IF;
    END IF;

    -- ============================================================
    -- 2. chunk_deferrals.meal_plan_id → meal_plans(id) ON DELETE SET NULL
    -- ============================================================
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'chunk_deferrals'
    ) THEN
        fk_record := NULL;
        SELECT tc.constraint_name, rc.delete_rule
          INTO fk_record
          FROM information_schema.table_constraints tc
          JOIN information_schema.referential_constraints rc
               ON tc.constraint_name = rc.constraint_name
          JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
         WHERE tc.table_schema = 'public'
           AND tc.table_name = 'chunk_deferrals'
           AND tc.constraint_type = 'FOREIGN KEY'
           AND kcu.column_name = 'meal_plan_id'
         LIMIT 1;

        IF fk_record.constraint_name IS NULL THEN
            BEGIN
                ALTER TABLE public.chunk_deferrals
                    ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
            ALTER TABLE public.chunk_deferrals
                ADD CONSTRAINT chunk_deferrals_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P0-HIST-3] Creada FK chunk_deferrals.meal_plan_id → meal_plans.id ON DELETE SET NULL.';
        ELSIF fk_record.delete_rule <> 'SET NULL' THEN
            EXECUTE format(
                'ALTER TABLE public.chunk_deferrals DROP CONSTRAINT %I',
                fk_record.constraint_name
            );
            BEGIN
                ALTER TABLE public.chunk_deferrals
                    ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
            ALTER TABLE public.chunk_deferrals
                ADD CONSTRAINT chunk_deferrals_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P0-HIST-3] Reemplazada FK chunk_deferrals.meal_plan_id (delete_rule=% → SET NULL).', fk_record.delete_rule;
        ELSE
            RAISE NOTICE '[P0-HIST-3] FK chunk_deferrals.meal_plan_id ya tiene ON DELETE SET NULL. Skip.';
        END IF;
    END IF;
END $$;

-- Annotaciones para drift detection.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chunk_lesson_telemetry_meal_plan_id_fkey'
    ) THEN
        EXECUTE 'COMMENT ON CONSTRAINT chunk_lesson_telemetry_meal_plan_id_fkey ON public.chunk_lesson_telemetry IS ''[P0-HIST-3] ON DELETE SET NULL: preservar lecciones agregables tras eliminación del plan asociado.''';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chunk_deferrals_meal_plan_id_fkey'
    ) THEN
        EXECUTE 'COMMENT ON CONSTRAINT chunk_deferrals_meal_plan_id_fkey ON public.chunk_deferrals IS ''[P0-HIST-3] ON DELETE SET NULL: preservar razones de deferral aunque el plan haya sido eliminado.''';
    END IF;
END $$;
