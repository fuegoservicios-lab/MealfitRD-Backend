-- [P2-SYNTHETIC-CORPUS · 2026-07-04] Corpus de entrenamiento de modelos propios.
--
-- Rol de cada `source`:
--   - 'synthetic': planes generados por el motor con perfiles sintéticos
--     diversos (scripts/generate_synthetic_corpus.py). Sirven para PROBAR el
--     pipeline del corpus y como set de EVALUACIÓN. NO son la dieta principal
--     de entrenamiento (entrenar solo con salidas del LLM = destilación
--     circular: se heredan sus sesgos/errores).
--   - 'user': datos reales ANONIMIZADOS de usuarios que consintieron
--     (P2-AI-TRAINING-CONSENT). El ETL futuro DEBE partir de
--     db_profiles.get_ai_training_consented_user_ids() (gate SSOT) y
--     actualizar las páginas legales ANTES de escribir la primera fila.

CREATE TABLE IF NOT EXISTS public.ai_training_corpus (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    user_id uuid NULL,
    profile jsonb NOT NULL,
    plan jsonb NOT NULL,
    quality jsonb NOT NULL DEFAULT '{}'::jsonb,
    engine_version text,
    model_id text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Idempotente: DROP IF EXISTS antes de ADD (patrón P3-MIGRATION-IDEMPOTENCE-DOC).
ALTER TABLE public.ai_training_corpus
    DROP CONSTRAINT IF EXISTS ai_training_corpus_source_valid;
ALTER TABLE public.ai_training_corpus
    ADD CONSTRAINT ai_training_corpus_source_valid
    CHECK (source IN ('synthetic', 'user'));

-- Coherencia source↔user_id: sintético NUNCA lleva user_id (no hay usuario que
-- proteger); 'user' SIEMPRE lo lleva (trazabilidad de consentimiento/borrado).
ALTER TABLE public.ai_training_corpus
    DROP CONSTRAINT IF EXISTS ai_training_corpus_source_user_coherence;
ALTER TABLE public.ai_training_corpus
    ADD CONSTRAINT ai_training_corpus_source_user_coherence
    CHECK (
        (source = 'synthetic' AND user_id IS NULL)
        OR (source = 'user' AND user_id IS NOT NULL)
    );

CREATE INDEX IF NOT EXISTS idx_ai_training_corpus_source
    ON public.ai_training_corpus (source);

COMMENT ON TABLE public.ai_training_corpus IS
    '[P2-SYNTHETIC-CORPUS 2026-07-04] Corpus de training de modelos propios. '
    'source=synthetic: motor + perfiles sintéticos (test/eval del pipeline). '
    'source=user: SOLO datos anonimizados de usuarios con ai_training_consent '
    '(gate SSOT get_ai_training_consented_user_ids) y legales actualizadas.';

-- Sanity check idempotente.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'ai_training_corpus'
    ) THEN
        RAISE EXCEPTION 'p2_synthetic_corpus: la tabla no existe tras el CREATE';
    END IF;
END $$;
