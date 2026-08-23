-- [P2-I18N-MIGRACIONES-SIN-LIBRO · 2026-08-23] El libro de migraciones.
--
-- Nada registraba qué migraciones se habían aplicado a Neon. `scripts/apply_migration.py`
-- ejecuta el fichero y no deja rastro; cada «¿está aplicada?» era una auditoría a mano
-- contra `information_schema`. Medido el 2026-08-23 con esa auditoría: 110 ficheros, y uno
-- —`p3_country_db_check_2026_08_22.sql`, el CHECK de país— SIN APLICAR en producción sin que
-- nada lo dijera. La migración de i18n del 21-ago (`locale` nullable) también se aplicó a
-- mano y se anotó en una doc, que es donde las cosas se pierden.
--
-- Una fila por fichero aplicado. `checksum` es el sha256 del contenido en el momento de
-- aplicarlo: si el fichero cambia después (lo hacen, este repo los edita para añadir
-- sanity checks), `--status` lo señala como «aplicada con otro contenido» en vez de
-- mentir «al día».
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): IF NOT EXISTS + DO $$ sanity.

CREATE TABLE IF NOT EXISTS public.schema_migrations (
    name        text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now(),
    applied_by  text,
    note        text
);

COMMENT ON TABLE public.schema_migrations IS
    '[P2-I18N-MIGRACIONES-SIN-LIBRO] Una fila por fichero de migrations/ aplicado. '
    'Lo escribe scripts/apply_migration.py (--apply registra; --record anota sin ejecutar). '
    'checksum = sha256 del fichero al aplicarlo.';

-- Sanity: la tabla existe con las columnas que el runner espera.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'schema_migrations'
          AND column_name IN ('name', 'checksum', 'applied_at')
        GROUP BY table_name HAVING COUNT(*) = 3
    ) THEN
        RAISE EXCEPTION 'P2-I18N-MIGRACIONES-SIN-LIBRO: schema_migrations sin las 3 columnas que usa el runner';
    END IF;
END $$;
