-- [P1-AUDIT-HIST-3 · 2026-05-09] CHECK constraint sobre
-- `plan_chunk_queue.status` — whitelist canónica de los 7 valores
-- válidos del state machine de chunks.
--
-- Bug original (audit Historial 2026-05-09):
--   `plan_chunk_queue.status` (varchar) NO tenía CHECK constraint.
--   La DB de producción MealFitRD tenía un row con `status='complete'`
--   (sin `d`) coexistiendo con el canónico `'completed'`. Todas las
--   queries del código filtran por `status='completed'` (con `d`):
--     - `routers/plans.py:3349` (tier_breakdown del Dashboard).
--     - `routers/plans.py:4249` (history-status-summary, P0-AUDIT-HIST-2).
--     - `routers/plans.py:5262` (regen-degraded chunks).
--     - `cron_tasks.py:3560 / 4599` (synthesis source filter).
--   Resultado: el row con `'complete'` quedaba SILENCIOSAMENTE
--   excluido de TODAS esas queries → tier perdido en analytics, chunks
--   no contados en el agregador del Historial, etc. Una bomba de tiempo:
--   no hay error visible, solo undercounts.
--
-- Diseño (transacción única, atómica):
--   1. Normalizar valores no-canónicos:
--      - `'complete'` → `'completed'` (drift conocido, mapeo directo).
--      - Cualquier otro valor inválido → `'cancelled'` (estado terminal
--        seguro: no es elegido por ningún cron de pickup, no dispara
--        retry, preserva la fila para audit). Anota `dead_letter_reason`
--        con el valor original entre paréntesis para post-mortem.
--   2. DROP CONSTRAINT IF EXISTS para idempotencia (re-ejecutar
--      la migración no falla).
--   3. ADD CONSTRAINT con los 7 valores canónicos.
--
-- Estados canónicos (SSOT — fuente: grep `SET status = '...'` en
-- backend/cron_tasks.py, backend/routers/plans.py, backend/db_plans.py,
-- backend/services.py):
--   1. `pending`           — fila creada, lista para pickup.
--   2. `processing`        — un worker la tomó (lock activo).
--   3. `stale`             — heartbeat expiró; sweep cron la
--                            reactivará a `pending` o la dead-letter.
--   4. `failed`             — falló; el recovery cron decide retry o
--                            escalar a `pending_user_action`.
--   5. `pending_user_action` — pausa que requiere acción del usuario
--                              (pantry/tz/missing-lessons/dead-letter).
--   6. `completed`         — chunk generó días exitosamente.
--   7. `cancelled`         — abortado (delete plan, restore overwrite).
--
-- Trade-offs:
--   - Si en futuro el state machine añade un estado (e.g.
--     `preempted_quota_exceeded`), debe AÑADIRSE en una nueva
--     migración. Sin esto, el INSERT/UPDATE rechazará con
--     "violates check constraint" — error LOUD, lo opuesto al silent
--     undercount actual.
--   - El DROP previo al ADD permite re-evaluar el set sin downtime
--     (DROP+ADD en la misma transacción no requiere lock pesado en
--     una tabla de tamaño moderado como plan_chunk_queue).
--
-- Patrón consistente con migraciones existentes
-- (p2_alpha_plan_chunk_queue_fk_cascade.sql, p0_hist_3_telemetry_orphan_fk.sql).
--
-- Idempotencia: el bloque entero re-ejecutable. Los UPDATE
-- de normalización son no-ops si ya se aplicó antes (no hay rows
-- inválidos). El DROP+ADD se mantiene equivalente.

BEGIN;

-- 1) Normalizar `'complete'` → `'completed'` (drift conocido confirmado
--    en producción MealFitRD).
UPDATE plan_chunk_queue
SET status = 'completed',
    updated_at = NOW()
WHERE status = 'complete';

-- 2) Defensa: cualquier otro valor inválido → `'cancelled'` (terminal).
--    Preservamos el valor original en `dead_letter_reason` para que
--    un post-mortem pueda diagnosticar de dónde salió el drift.
UPDATE plan_chunk_queue
SET status = 'cancelled',
    dead_letter_reason = COALESCE(
        dead_letter_reason,
        'p1_audit_hist_3_drift_normalize_from_' || status
    ),
    dead_lettered_at = COALESCE(dead_lettered_at, NOW()),
    updated_at = NOW()
WHERE status NOT IN (
    'pending', 'processing', 'stale', 'failed',
    'pending_user_action', 'completed', 'cancelled'
);

-- 3) DROP previo idempotente.
ALTER TABLE public.plan_chunk_queue
    DROP CONSTRAINT IF EXISTS plan_chunk_queue_status_check;

-- 4) ADD CONSTRAINT canónico.
ALTER TABLE public.plan_chunk_queue
    ADD CONSTRAINT plan_chunk_queue_status_check
    CHECK (status IN (
        'pending', 'processing', 'stale', 'failed',
        'pending_user_action', 'completed', 'cancelled'
    ));

COMMENT ON CONSTRAINT plan_chunk_queue_status_check ON public.plan_chunk_queue IS
'[P1-AUDIT-HIST-3 · 2026-05-09] Whitelist canónica de los 7 estados del chunk state machine. Cierra drift confirmado en producción donde la DB permitía valores no-canónicos (e.g. ''complete'' sin ''d'') que las queries del código filtraban silenciosamente, causando undercounts en tier_breakdown / history-status-summary. Si se añade un estado nuevo al state machine, actualizar este CHECK en una nueva migración.';

COMMIT;
