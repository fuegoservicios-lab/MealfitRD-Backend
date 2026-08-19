-- [P1-PLAN-DISPLAY-I18N · Task 5 · 2026-08-19] Columna `name_en` en
-- `master_ingredients`: gloss en inglés para la etiqueta de la lista de
-- compras bilingüe ("Black beans (Habichuelas rojas)") — fase 1b de
-- docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md, regla de
-- oro: el usuario cocina en su idioma pero COMPRA en español, la lista
-- NUNCA sale en inglés puro.
--
-- DISPLAY-ONLY, restricción dura: `name_en` NUNCA entra a `normalize_name`,
-- aliases, matchers, ni `pantry_names_match` — la identidad de una fila del
-- catálogo sigue resolviendo EXCLUSIVAMENTE por `name` (español canónico).
-- Misma clase de bug que P1-PANTRY-NAME-RESOLUTION advierte: un campo de
-- display que se cuela a un matcher rompe la resolución de identidad, a
-- veces en silencio. Guard grep-proof en
-- backend/tests/test_p1_plan_display_i18n.py (sección "catálogo").
--
-- Población: script one-shot `backend/scripts/fill_catalog_name_en.py`
-- (`--dry-run` default, `--commit` explícito, UNA llamada LLM flash batch)
-- — deliberadamente FUERA de esta migración: es trabajo de LLM, no DDL, y
-- el dueño audita el dry-run antes de correr `--commit`.
--
-- Idempotente: `ADD COLUMN IF NOT EXISTS` — re-aplicar es no-op.
--
-- SSOT de migrations (P3-MIGRATIONS-SSOT): este archivo vive IDÉNTICO en
-- `migrations/` (workspace-root) y `backend/migrations/`.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS name_en TEXT;

COMMENT ON COLUMN public.master_ingredients.name_en IS
    '[P1-PLAN-DISPLAY-I18N · Task 5 · 2026-08-19] Gloss en inglés del '
    'nombre del alimento, SOLO para la etiqueta bilingüe de la lista de '
    'compras del PDF ("English gloss (Nombre canónico español)"). '
    'DISPLAY-ONLY: nunca entra a normalize_name/aliases/matchers/'
    'pantry_names_match — la identidad del catálogo sigue resolviendo '
    'EXCLUSIVAMENTE por name (español canónico). NULL = sin traducir '
    'todavía (fallback: el frontend muestra solo el nombre español). '
    'Poblado por backend/scripts/fill_catalog_name_en.py (one-shot, LLM '
    'batch, --dry-run default, --commit explícito).';

-- ── Sanity: la columna existe y es del tipo esperado ────────────────────
DO $$
DECLARE _tipo text;
BEGIN
    SELECT data_type INTO _tipo
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'master_ingredients'
      AND column_name = 'name_en';
    IF _tipo IS NULL THEN
        RAISE EXCEPTION '[P1-PLAN-DISPLAY-I18N] columna name_en no se creó en master_ingredients';
    END IF;
    IF _tipo <> 'text' THEN
        RAISE EXCEPTION '[P1-PLAN-DISPLAY-I18N] name_en tiene tipo inesperado: %', _tipo;
    END IF;
END $$;
