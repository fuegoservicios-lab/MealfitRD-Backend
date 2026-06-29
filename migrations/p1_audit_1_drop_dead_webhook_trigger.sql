-- [P1-AUDIT-1 · 2026-05-12] Drop trigger + función con URL y secret hardcoded.
--
-- Contexto: el audit production-readiness del 2026-05-12 encontró que la
-- función SECURITY DEFINER `public.trigger_process_pending_facts_webhook`,
-- invocada por el trigger AFTER INSERT `process_facts_on_insert` sobre
-- `public.pending_facts_queue`, tenía:
--
--   webhook_url    := 'https://mealfit-rd.vercel.app/api/webhooks/process-pending-facts';
--   webhook_secret := 'mealfit_secure_webhook_secret_2026';
--
-- ambos hardcoded dentro del cuerpo de la función. Problemas:
--
--   1. Secret en cleartext dentro de pg_proc — visible a cualquier rol con
--      `SELECT` sobre `pg_catalog.pg_proc`. Zero rotabilidad sin DDL.
--   2. URL apunta a un dominio `vercel.app` que pre-data la migración a
--      el VPS Oracle (per `backend/.env.example`). Si el host quedó
--      stale, cada INSERT disparaba un pg_net request a un endpoint muerto.
--   3. Dependencia obligatoria de la extensión `pg_net` para algo que el
--      backend puede polear directamente.
--
-- Reemplazo: cron `drain_pending_facts_queue` (cron_tasks.py) registrado en
-- `register_plan_chunk_scheduler` corre cada 1 min, hace
-- `SELECT DISTINCT user_id FROM pending_facts_queue` y delega a
-- `process_pending_queue_sync(user_id)` — el mismo procesador que el webhook
-- usaba, así que la semántica de procesamiento no cambia. Knob
-- `MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN` clamp [1, 30].
--
-- La latencia pasa de <1s (trigger) a 30-60s (cron). Aceptable: la cola se
-- llena solo en el overflow-path (`fact_extractor.py:439`, cuando la
-- extracción inline falla) y no es time-critical — el siguiente request del
-- usuario verá los facts ya extraídos.
--
-- La TABLA `pending_facts_queue` se preserva (el backend INSERTea aún via
-- `db_facts.py::enqueue_pending_fact`) y el endpoint
-- `/api/webhooks/process-pending-facts` (app.py:1096) también se preserva
-- por si en el futuro queremos restaurar el trigger con URL parametrizada
-- en `app_kv_store` (no para esta migración).
--
-- Idempotente: `DROP ... IF EXISTS` re-aplicar es no-op.
--
-- Rollback (NO recomendado — reintroduciría el secret hardcoded):
--
--   CREATE OR REPLACE FUNCTION public.trigger_process_pending_facts_webhook()
--   RETURNS trigger
--   LANGUAGE plpgsql
--   SECURITY DEFINER
--   SET search_path = public, pg_catalog
--   AS $$
--   DECLARE ...
--   BEGIN
--     perform net.http_post(
--       url := 'https://<host>/api/webhooks/process-pending-facts',
--       body := payload::jsonb,
--       headers := jsonb_build_object(
--         'Authorization', 'Bearer <secret-from-kv>',
--         'X-Webhook-Secret', '<secret-from-kv>'
--       )
--     );
--     return NEW;
--   END;
--   $$;
--   CREATE TRIGGER process_facts_on_insert AFTER INSERT
--     ON public.pending_facts_queue FOR EACH ROW
--     EXECUTE FUNCTION public.trigger_process_pending_facts_webhook();
--
-- Si el rollback se ejecuta, leer URL+secret desde `app_kv_store` en lugar
-- de hardcodear — el patrón está documentado en CLAUDE.md sección
-- "Pattern: SET search_path = '' en functions Postgres".

DROP TRIGGER IF EXISTS process_facts_on_insert ON public.pending_facts_queue;

DROP FUNCTION IF EXISTS public.trigger_process_pending_facts_webhook();

COMMENT ON TABLE public.pending_facts_queue IS
'[P1-AUDIT-1 · 2026-05-12] Drained por cron `drain_pending_facts_queue` '
'(cron_tasks.py) cada 1 min (knob MEALFIT_PENDING_FACTS_DRAIN_INTERVAL_MIN). '
'Antes: trigger AFTER INSERT process_facts_on_insert + función '
'trigger_process_pending_facts_webhook con URL Vercel y secret hardcoded en '
'pg_proc — eliminados en esta migración. Backend INSERTea via '
'db_facts.py::enqueue_pending_fact (overflow path fact_extractor:439); el '
'cron lee DISTINCT user_id y llama process_pending_queue_sync(user_id) — '
'mismo procesador que el webhook usaba.';
