-- [P1-DB-STMT-TIMEOUT-ROLE · 2026-05-28] statement_timeout +
-- idle_in_transaction_session_timeout a nivel del ROL `service_role`.
--
-- Contexto: el backend conecta como service_role (BYPASSRLS). El código aplica
-- timeouts a nivel SESIÓN (configure_sync_conn / configure_async_conn,
-- P1-DB-STMT-TIMEOUT) pero (a) ese binario aún no está desplegado y (b) cualquier
-- conexión que no pase por configure_*_conn quedaría sin cota. Esta migración
-- añade la defensa a nivel ROL como respaldo independiente del deploy.
--
-- Evidencia prod 2026-05-28 (pg_stat_statements):
--   SELECT id, plan_data FROM meal_plans WHERE user_id=$1 ... FOR UPDATE
--   -> mean 4.3s, max 119s sobre 445 calls. Una query atascada retiene un slot
--   del pool (76 conns, ajustado al cap del plan) hasta que termina o el cliente
--   muere -> riesgo de agotamiento de pool.
--
-- Valores generosos (60s statement / 30s idle-in-txn): ninguna query OLTP
-- legítima del backend tarda 60s (la generación de plan es LLM, no SQL); el
-- objetivo es cortar SÓLO cuelgues anómalos. El más restrictivo gana por sesión,
-- así que coexiste con el session-level de 30s del código sin conflicto.
--
-- Idempotente: ALTER ROLE ... SET reescribe el setting (no acumula).
-- Reversión:
--   ALTER ROLE service_role RESET statement_timeout;
--   ALTER ROLE service_role RESET idle_in_transaction_session_timeout;

ALTER ROLE service_role SET statement_timeout = '60s';
ALTER ROLE service_role SET idle_in_transaction_session_timeout = '30s';

-- Sanity: la migración falla en voz alta si el setting no quedó aplicado.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_roles
    WHERE rolname = 'service_role'
      AND rolconfig @> ARRAY['statement_timeout=60s']
  ) THEN
    RAISE EXCEPTION '[P1-DB-STMT-TIMEOUT-ROLE] statement_timeout no quedó aplicado en service_role';
  END IF;
END $$;
