-- [LONG-TERM-MEMORY-TOGGLE · 2026-05-13]
-- Añade flag para que usuarios Básico+ puedan apagar/encender la extracción
-- y consulta de memoria a largo plazo (user_facts) desde Settings del Dashboard.
--
-- Semántica del flag:
--   TRUE  (default) = chat.py ejecuta async_extract_and_save_facts en cada turn
--                     + busca user_facts relevantes al construir contexto.
--   FALSE          = chat.py NO extrae nuevos hechos NI consulta los previos
--                     (datos existentes en user_facts quedan en BD, no se borran).
--
-- Para tier='gratis' el flag se ignora — la memoria larga ya está gateada
-- upstream por `is_plus = plan_tier in ["basic","plus","admin","ultra"]`
-- en chat.py. El toggle en Settings tampoco se muestra a usuarios gratis.
--
-- IF NOT EXISTS para idempotencia (re-aplicar la migración no debe romper).
ALTER TABLE public.user_profiles
ADD COLUMN IF NOT EXISTS long_term_memory_enabled BOOLEAN NOT NULL DEFAULT TRUE;

COMMENT ON COLUMN public.user_profiles.long_term_memory_enabled IS
'Toggle visible en Settings para usuarios Básico+. FALSE = chat.py no extrae ni consulta user_facts (datos en BD intactos). TRUE = comportamiento legacy. Tier gratis ignora el flag — memoria larga gateada upstream por is_plus check.';
