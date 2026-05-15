-- [P1-NEW-5 · 2026-05-11] Consolida ON DELETE CASCADE para FK user_id
-- en tablas que tenían FK pero SIN CASCADE (gap del audit 2026-05-11).
--
-- Bug original:
--   `api_usage.user_id` y `meal_rejections.user_id` apuntan a
--   `auth.users(id)` pero sin ON DELETE CASCADE. Si un usuario es
--   eliminado de `auth.users` (manual SRE action, GDPR delete, etc.):
--     - Las rows en estas tablas quedan con `user_id` apuntando a
--       un UUID inexistente.
--     - El FK ya NO previene corrupción (no cascade) pero tampoco
--       deja que el delete progrese si la versión usa RESTRICT.
--     - Mantenimiento + dashboards ven rows "huérfanas" sin owner.
--
-- Audit 2026-05-11 detectó las únicas 2 tablas con `user_id → auth.users`
-- sin CASCADE de las 15 tablas user-scoped con FK declarado.
--
-- Notas:
--   - Idempotente: DROP IF EXISTS + ADD CONSTRAINT permite re-correr.
--   - Seguro: las rows existentes ya pasan el FK original (mismo
--     constraint, solo cambia comportamiento ON DELETE), no requiere
--     NOT VALID + VALIDATE.
--   - NO incluye tablas SIN FK (agent_sessions, weight_log, etc.) —
--     esas requieren decisión per-tabla documentada en CLAUDE.md.

-- ------------------------------------------------------------
-- api_usage: telemetría de uso por usuario (paywall counters)
-- ------------------------------------------------------------
ALTER TABLE public.api_usage
    DROP CONSTRAINT IF EXISTS api_usage_user_id_fkey;

ALTER TABLE public.api_usage
    ADD CONSTRAINT api_usage_user_id_fkey
        FOREIGN KEY (user_id)
        REFERENCES auth.users(id)
        ON DELETE CASCADE;

COMMENT ON CONSTRAINT api_usage_user_id_fkey ON public.api_usage IS
    'P1-NEW-5 · 2026-05-11: ON DELETE CASCADE para que las filas de '
    'telemetría se eliminen automáticamente al borrar el usuario en '
    'auth.users (GDPR / SRE delete). Pre-fix, las rows quedaban '
    'huérfanas con user_id apuntando a UUID inexistente.';


-- ------------------------------------------------------------
-- meal_rejections: rechazos de menús por user (para feedback loop)
-- ------------------------------------------------------------
ALTER TABLE public.meal_rejections
    DROP CONSTRAINT IF EXISTS meal_rejections_user_id_fkey;

ALTER TABLE public.meal_rejections
    ADD CONSTRAINT meal_rejections_user_id_fkey
        FOREIGN KEY (user_id)
        REFERENCES auth.users(id)
        ON DELETE CASCADE;

COMMENT ON CONSTRAINT meal_rejections_user_id_fkey ON public.meal_rejections IS
    'P1-NEW-5 · 2026-05-11: ON DELETE CASCADE para que los rechazos de '
    'menús se eliminen automáticamente al borrar el usuario. Pre-fix, '
    'las rows quedaban huérfanas y podrían sesgar análisis agregados '
    '(rejections sin user real conectaban contra IDs inexistentes).';
