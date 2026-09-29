-- [P1-PLAN-LOTE-830 · 2026-09-29] Cuentas de prueba (spec 2026-09-29-admin-cuentas-actividad-pruebas-design §3.1): las
-- cuentas que el dueño marca desde /admin → Cuentas y cuyo contenido (formulario, comidas, planes, conversaciones)
-- puede ver el equipo. La propia persona ve la marca (aviso en la app) y puede salir cuando quiera; el contenido solo se
-- abre mientras la marca está viva Y la persona ya vio el aviso (`aviso_visto_at`, §13.4).
--
-- Una marca se QUITA, nunca se borra: el historial (quién la puso y por qué, quién la quitó y si salió ella) es la propia
-- tabla. Se va con la cuenta (ON DELETE CASCADE); el rastro de lo que hizo el personal vive además en admin_access_log,
-- que no tiene FK. `marcada_por` / `quitada_por` son ids del PERSONAL (o de la persona, si salió ella): no salen en la
-- exportación de la cuenta. Lo escribe y lo lee solo backend/cuentas_prueba.py. Idempotente.
CREATE TABLE IF NOT EXISTS public.cuentas_de_prueba (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.user_profiles(id) ON DELETE CASCADE,
    marcada_por UUID NOT NULL,
    marcada_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    motivo TEXT NOT NULL,
    aviso_visto_at TIMESTAMPTZ,
    quitada_at TIMESTAMPTZ,
    quitada_por UUID,
    quitada_por_la_persona BOOLEAN NOT NULL DEFAULT false,
    motivo_quitar TEXT
);

ALTER TABLE public.cuentas_de_prueba DROP CONSTRAINT IF EXISTS cuentas_de_prueba_motivo_chk;
ALTER TABLE public.cuentas_de_prueba ADD CONSTRAINT cuentas_de_prueba_motivo_chk
    CHECK (char_length(motivo) BETWEEN 3 AND 300);

-- El motivo de quitarla (solo lo pone un admin; cuando sale la persona queda NULL). El IS NULL va explícito: un CHECK que
-- evalúa a NULL PASA.
ALTER TABLE public.cuentas_de_prueba DROP CONSTRAINT IF EXISTS cuentas_de_prueba_motivo_quitar_chk;
ALTER TABLE public.cuentas_de_prueba ADD CONSTRAINT cuentas_de_prueba_motivo_quitar_chk
    CHECK (motivo_quitar IS NULL OR char_length(motivo_quitar) BETWEEN 3 AND 300);

-- Toda marca quitada dice quién la quitó: el admin, o la propia persona (quitada_por = user_id).
ALTER TABLE public.cuentas_de_prueba DROP CONSTRAINT IF EXISTS cuentas_de_prueba_quitada_chk;
ALTER TABLE public.cuentas_de_prueba ADD CONSTRAINT cuentas_de_prueba_quitada_chk
    CHECK (quitada_at IS NULL OR quitada_por IS NOT NULL);

-- Una sola marca viva por cuenta: el índice corta la carrera de dos admins marcando a la vez.
CREATE UNIQUE INDEX IF NOT EXISTS cuentas_de_prueba_una_viva_idx
    ON public.cuentas_de_prueba (user_id) WHERE quitada_at IS NULL;

-- El historial de una cuenta y el borrado en cascada de la FK, sin recorrer la tabla entera.
CREATE INDEX IF NOT EXISTS cuentas_de_prueba_user_idx
    ON public.cuentas_de_prueba (user_id, marcada_at DESC);

REVOKE ALL ON public.cuentas_de_prueba FROM PUBLIC;

COMMENT ON TABLE public.cuentas_de_prueba IS
    'P1-PLAN-LOTE-830: marcas de cuenta de prueba (el equipo ve el contenido de la cuenta; la persona lo sabe y puede salir). Una marca se quita, nunca se borra. Lo escribe y lo lee solo backend/cuentas_prueba.py.';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'public' AND table_name = 'cuentas_de_prueba') THEN
        RAISE EXCEPTION 'p1_plan_lote_830: cuentas_de_prueba no existe tras la migracion';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes
                   WHERE schemaname = 'public' AND indexname = 'cuentas_de_prueba_una_viva_idx') THEN
        RAISE EXCEPTION 'p1_plan_lote_830: falta el indice unico de la marca viva';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_class r ON r.oid = c.confrelid
        WHERE t.relname = 'cuentas_de_prueba' AND r.relname = 'user_profiles'
          AND c.contype = 'f' AND c.confdeltype = 'c'
    ) THEN
        RAISE EXCEPTION 'p1_plan_lote_830: falta el FK en cascada a user_profiles';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'cuentas_de_prueba_quitada_chk') THEN
        RAISE EXCEPTION 'p1_plan_lote_830: falta el CHECK de quien quita la marca';
    END IF;
END $$;
