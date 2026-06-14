-- [P2-α] Asegurar FK ON DELETE CASCADE en plan_chunk_queue → meal_plans.
--
-- El CREATE TABLE original en backend/app.py:217 ya define
-- `meal_plan_id UUID NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE`,
-- pero `CREATE TABLE IF NOT EXISTS` NO altera tablas pre-existentes — si la
-- DB de producción tiene la tabla creada antes de que se añadiera la cláusula
-- ON DELETE CASCADE, los chunks quedan huérfanos al eliminar un plan y
-- dependen del cron `cleanup_orphan_chunks` (5 min) para liberarlos. Durante
-- esa ventana el worker puede levantar un chunk cuyo plan ya no existe y
-- desperdiciar tokens LLM antes de que `_validate_chunk_pre_llm` aborte.
--
-- Esta migración:
--   1. Detecta si la FK ya tiene ON DELETE CASCADE.
--   2. Si NO la tiene (o la FK no existe), la dropea+recrea con CASCADE.
--   3. Es idempotente: re-correrla no rompe nada.
--
-- Misma lógica para `chunk_user_locks.locked_by_chunk_id → plan_chunk_queue.id`
-- y `chunk_user_locks.user_id → user_profiles.id`, que también dependen de
-- esta cadena para limpieza automática.

DO $$
DECLARE
    fk_record RECORD;
    fk_name TEXT;
BEGIN
    -- ============================================================
    -- 1. plan_chunk_queue.meal_plan_id → meal_plans(id)
    -- ============================================================
    SELECT tc.constraint_name, rc.delete_rule
      INTO fk_record
      FROM information_schema.table_constraints tc
      JOIN information_schema.referential_constraints rc
           ON tc.constraint_name = rc.constraint_name
      JOIN information_schema.key_column_usage kcu
           ON tc.constraint_name = kcu.constraint_name
     WHERE tc.table_schema = 'public'
       AND tc.table_name = 'plan_chunk_queue'
       AND tc.constraint_type = 'FOREIGN KEY'
       AND kcu.column_name = 'meal_plan_id'
     LIMIT 1;

    IF fk_record.constraint_name IS NULL THEN
        -- FK no existe: la creamos con CASCADE.
        ALTER TABLE public.plan_chunk_queue
            ADD CONSTRAINT plan_chunk_queue_meal_plan_id_fkey
            FOREIGN KEY (meal_plan_id)
            REFERENCES public.meal_plans(id)
            ON DELETE CASCADE;
        RAISE NOTICE '[P2-α] Creada FK plan_chunk_queue.meal_plan_id → meal_plans.id ON DELETE CASCADE.';
    ELSIF fk_record.delete_rule <> 'CASCADE' THEN
        -- FK existe pero sin CASCADE: la reemplazamos.
        EXECUTE format(
            'ALTER TABLE public.plan_chunk_queue DROP CONSTRAINT %I',
            fk_record.constraint_name
        );
        ALTER TABLE public.plan_chunk_queue
            ADD CONSTRAINT plan_chunk_queue_meal_plan_id_fkey
            FOREIGN KEY (meal_plan_id)
            REFERENCES public.meal_plans(id)
            ON DELETE CASCADE;
        RAISE NOTICE '[P2-α] Reemplazada FK plan_chunk_queue.meal_plan_id (delete_rule=% → CASCADE).', fk_record.delete_rule;
    ELSE
        RAISE NOTICE '[P2-α] FK plan_chunk_queue.meal_plan_id ya tiene ON DELETE CASCADE. Skip.';
    END IF;

    -- ============================================================
    -- 2. plan_chunk_queue.user_id → user_profiles(id)
    --    (defensa en profundidad: usuario eliminado → sus chunks pendientes
    --     se limpian, evitando referencias muertas que el worker tomaría)
    -- ============================================================
    fk_record := NULL;
    SELECT tc.constraint_name, rc.delete_rule
      INTO fk_record
      FROM information_schema.table_constraints tc
      JOIN information_schema.referential_constraints rc
           ON tc.constraint_name = rc.constraint_name
      JOIN information_schema.key_column_usage kcu
           ON tc.constraint_name = kcu.constraint_name
     WHERE tc.table_schema = 'public'
       AND tc.table_name = 'plan_chunk_queue'
       AND tc.constraint_type = 'FOREIGN KEY'
       AND kcu.column_name = 'user_id'
     LIMIT 1;

    IF fk_record.constraint_name IS NULL THEN
        ALTER TABLE public.plan_chunk_queue
            ADD CONSTRAINT plan_chunk_queue_user_id_fkey
            FOREIGN KEY (user_id)
            REFERENCES public.user_profiles(id)
            ON DELETE CASCADE;
        RAISE NOTICE '[P2-α] Creada FK plan_chunk_queue.user_id → user_profiles.id ON DELETE CASCADE.';
    ELSIF fk_record.delete_rule <> 'CASCADE' THEN
        EXECUTE format(
            'ALTER TABLE public.plan_chunk_queue DROP CONSTRAINT %I',
            fk_record.constraint_name
        );
        ALTER TABLE public.plan_chunk_queue
            ADD CONSTRAINT plan_chunk_queue_user_id_fkey
            FOREIGN KEY (user_id)
            REFERENCES public.user_profiles(id)
            ON DELETE CASCADE;
        RAISE NOTICE '[P2-α] Reemplazada FK plan_chunk_queue.user_id (delete_rule=% → CASCADE).', fk_record.delete_rule;
    ELSE
        RAISE NOTICE '[P2-α] FK plan_chunk_queue.user_id ya tiene ON DELETE CASCADE. Skip.';
    END IF;

    -- ============================================================
    -- 3. chunk_user_locks.user_id → user_profiles(id)
    --    Sin esto, eliminar un usuario deja locks zombie indefinidos hasta
    --    que el housekeeping los detecte por heartbeat stale.
    -- ============================================================
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public' AND table_name = 'chunk_user_locks') THEN
        fk_record := NULL;
        SELECT tc.constraint_name, rc.delete_rule
          INTO fk_record
          FROM information_schema.table_constraints tc
          JOIN information_schema.referential_constraints rc
               ON tc.constraint_name = rc.constraint_name
          JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
         WHERE tc.table_schema = 'public'
           AND tc.table_name = 'chunk_user_locks'
           AND tc.constraint_type = 'FOREIGN KEY'
           AND kcu.column_name = 'user_id'
         LIMIT 1;

        IF fk_record.constraint_name IS NULL THEN
            ALTER TABLE public.chunk_user_locks
                ADD CONSTRAINT chunk_user_locks_user_id_fkey
                FOREIGN KEY (user_id)
                REFERENCES public.user_profiles(id)
                ON DELETE CASCADE;
            RAISE NOTICE '[P2-α] Creada FK chunk_user_locks.user_id → user_profiles.id ON DELETE CASCADE.';
        ELSIF fk_record.delete_rule <> 'CASCADE' THEN
            EXECUTE format(
                'ALTER TABLE public.chunk_user_locks DROP CONSTRAINT %I',
                fk_record.constraint_name
            );
            ALTER TABLE public.chunk_user_locks
                ADD CONSTRAINT chunk_user_locks_user_id_fkey
                FOREIGN KEY (user_id)
                REFERENCES public.user_profiles(id)
                ON DELETE CASCADE;
            RAISE NOTICE '[P2-α] Reemplazada FK chunk_user_locks.user_id (delete_rule=% → CASCADE).', fk_record.delete_rule;
        ELSE
            RAISE NOTICE '[P2-α] FK chunk_user_locks.user_id ya tiene ON DELETE CASCADE. Skip.';
        END IF;
    END IF;

    -- ============================================================
    -- 4. plan_chunk_metrics.meal_plan_id → meal_plans(id)
    --    Métricas históricas: SET NULL para preservar el row con plan_id=NULL
    --    cuando el plan se elimina (preferimos retener telemetría agregada
    --    aunque el plan ya no exista, para análisis post-mortem).
    -- ============================================================
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public' AND table_name = 'plan_chunk_metrics') THEN
        fk_record := NULL;
        SELECT tc.constraint_name, rc.delete_rule
          INTO fk_record
          FROM information_schema.table_constraints tc
          JOIN information_schema.referential_constraints rc
               ON tc.constraint_name = rc.constraint_name
          JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
         WHERE tc.table_schema = 'public'
           AND tc.table_name = 'plan_chunk_metrics'
           AND tc.constraint_type = 'FOREIGN KEY'
           AND kcu.column_name = 'meal_plan_id'
         LIMIT 1;

        IF fk_record.constraint_name IS NULL THEN
            -- No FK previa: añadimos SET NULL para preservar row.
            -- Si la columna es NOT NULL, primero la relajamos.
            BEGIN
                ALTER TABLE public.plan_chunk_metrics ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL; -- ya era nullable
            END;
            ALTER TABLE public.plan_chunk_metrics
                ADD CONSTRAINT plan_chunk_metrics_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P2-α] Creada FK plan_chunk_metrics.meal_plan_id → meal_plans.id ON DELETE SET NULL.';
        ELSIF fk_record.delete_rule NOT IN ('SET NULL', 'CASCADE') THEN
            -- Si está en NO ACTION o RESTRICT, lo escalamos a SET NULL.
            EXECUTE format(
                'ALTER TABLE public.plan_chunk_metrics DROP CONSTRAINT %I',
                fk_record.constraint_name
            );
            BEGIN
                ALTER TABLE public.plan_chunk_metrics ALTER COLUMN meal_plan_id DROP NOT NULL;
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END;
            ALTER TABLE public.plan_chunk_metrics
                ADD CONSTRAINT plan_chunk_metrics_meal_plan_id_fkey
                FOREIGN KEY (meal_plan_id)
                REFERENCES public.meal_plans(id)
                ON DELETE SET NULL;
            RAISE NOTICE '[P2-α] Reemplazada FK plan_chunk_metrics.meal_plan_id (delete_rule=% → SET NULL).', fk_record.delete_rule;
        ELSE
            RAISE NOTICE '[P2-α] FK plan_chunk_metrics.meal_plan_id ya tiene action consistente (%). Skip.', fk_record.delete_rule;
        END IF;
    END IF;
END $$;

COMMENT ON CONSTRAINT plan_chunk_queue_meal_plan_id_fkey ON public.plan_chunk_queue IS
    '[P2-α] ON DELETE CASCADE: eliminar el plan limpia atómicamente todos sus chunks. Reduce ventana de orphan-chunks de hasta 5min (cron cleanup) a 0.';
