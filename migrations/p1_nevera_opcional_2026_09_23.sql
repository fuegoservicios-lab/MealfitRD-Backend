-- [P1-NEVERA-OPCIONAL · 2026-09-23] La Nevera opcional en modo contador. SSOT: backend/nevera_opcional.py.
--
-- nevera_enabled       NULL = automático (encendida; elegible para el apagado automático tras 48 h vacía)
--                      TRUE = encendida por el usuario (nunca se apaga sola) · FALSE = apagada (usuario o sistema).
-- nevera_auto_off_at   cuándo la apagó el sistema (nota única en la app); se limpia en cualquier cambio del usuario.
-- nevera_reloj_desde   desde cuándo cuentan las 48 h. ADD COLUMN con DEFAULT now() estampa en las filas existentes
--                      la hora de ESTA migración: las cuentas de hoy tienen sus 48 h desde el despliegue.
-- La regla (`nevera_activa = NOT (plan_mode='tracking' AND nevera_enabled IS FALSE)`) NO vive en SQL: vive en
-- nevera_opcional.nevera_activa_de. Idempotente.

ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS nevera_enabled BOOLEAN;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS nevera_auto_off_at TIMESTAMPTZ;
ALTER TABLE public.user_profiles ADD COLUMN IF NOT EXISTS nevera_reloj_desde TIMESTAMPTZ NOT NULL DEFAULT now();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_schema = 'public' AND table_name = 'user_profiles' AND column_name = 'nevera_reloj_desde'
    ) THEN
        RAISE EXCEPTION '[P1-NEVERA-OPCIONAL] falta user_profiles.nevera_reloj_desde tras la migración';
    END IF;
END $$;
