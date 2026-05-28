-- [P2-MEAL-PLANS-GENSTATUS-IDX · 2026-05-28] Índice funcional sobre
-- plan_data->>'generation_status'.
--
-- Contexto: ~10 callsites en cron_tasks.py filtran meal_plans por
-- `plan_data->>'generation_status'` (IN ('partial','generating_next'), ='complete',
-- ='failed', ='partial_no_shopping', IN ('abandoned','cancelled'), ...) — sweeps de
-- recovery, finalize de zombies, reactivación de shopping. pg_stat_statements
-- (prod 2026-05-28) muestra un lookup `meal_plans WHERE plan_data->>$1 = $2` con
-- ~70k calls haciendo seq-scan (mean 2.3ms HOY con ~1 fila viva, pero crece
-- linealmente con #planes → degradación a escala 1k-10k usuarios).
--
-- Un índice funcional B-tree sobre la expresión sirve esos filtros a escala. NO
-- incluye user_id: los filtros de status son cross-user (corren en crons).
--
-- Idempotente: CREATE INDEX IF NOT EXISTS. NO se usa CONCURRENTLY porque las
-- migraciones del proyecto corren en transacción; a la escala actual (~decenas de
-- filas) el lock ACCESS EXCLUSIVE es instantáneo. Si se aplica con la tabla ya
-- grande, correr manualmente la variante CONCURRENTLY fuera de transacción.
CREATE INDEX IF NOT EXISTS idx_meal_plans_generation_status
    ON public.meal_plans ((plan_data->>'generation_status'));

-- Sanity: falla en voz alta si el índice no quedó creado.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public' AND indexname = 'idx_meal_plans_generation_status'
  ) THEN
    RAISE EXCEPTION '[P2-MEAL-PLANS-GENSTATUS-IDX] índice no creado';
  END IF;
END $$;
