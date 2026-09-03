-- [P1-I18N-DASHBOARD · 2026-08-15] `user_profiles.locale`: el idioma de la
-- interfaz, siguiendo al USUARIO y no al dispositivo.
--
-- Por qué en la DB y no solo en localStorage (como el tema, `mealfit_theme`):
-- el tema es una preferencia de pantalla y se acepta que cada dispositivo tenga
-- la suya. El idioma no — quien elige francés en el móvil y abre el portátil en
-- español lee eso como un fallo del producto, no como una preferencia local.
-- `localStorage` sigue existiendo, pero como CACHÉ anti-parpadeo: fija
-- `<html lang>` antes del primer paint mientras el perfil viaja por la red.
--
-- Por qué un CHECK y no solo validación en el endpoint:
-- el whitelist de `PATCH /api/profile` valida las CLAVES, no los VALORES
-- (`_PROFILE_SCALAR_WHITELIST` existe para frenar columnas de entitlement, y
-- `full_name` es texto libre, así que nunca necesitó mirar el valor). Sin este
-- CHECK, cualquier escritor futuro —un script de soporte, un backfill, un
-- endpoint nuevo que reutilice el whitelist— puede dejar 'jp' o 'français' en
-- la columna, y el frontend degradaría a español sin que nadie se entere.
-- Mismo criterio que I8: la invariante vive en la DB, no solo en el camino que
-- hoy resulta que la escribe.
--
-- SSOT de la lista: `frontend/src/i18n/locales.js`. Este CHECK, el
-- `_LOCALE_VALUES` del backend y el boot de `index.html` son sus espejos;
-- `test_p1_i18n_dashboard.py` falla si divergen.
--
-- Idempotente: ADD COLUMN IF NOT EXISTS + DROP CONSTRAINT IF EXISTS antes de
-- ADD (P3-MIGRATION-IDEMPOTENCE-DOC).

-- 1. La columna. DEFAULT 'es-DO' porque el producto nace en español dominicano:
--    toda fila existente queda explícitamente en el idioma que ya estaba viendo.
ALTER TABLE public.user_profiles
  ADD COLUMN IF NOT EXISTS locale text NOT NULL DEFAULT 'es-DO';

-- 2. Sanity ANTES del CHECK: si alguna fila trae un valor fuera de la lista, la
--    migración se detiene y lo dice. Con la columna recién creada esto es
--    imposible; en una re-aplicación sobre datos vivos, no.
DO $$
DECLARE
  bad_count int;
BEGIN
  SELECT COUNT(*) INTO bad_count
  FROM public.user_profiles
  WHERE locale IS NOT NULL
    AND locale NOT IN ('es-DO', 'en-US', 'pt-BR', 'fr-FR', 'it-IT');
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      'P1-I18N-DASHBOARD: % filas de user_profiles tienen un locale fuera de la '
      'lista soportada. Normalizarlas a ''es-DO'' (o añadir el idioma a '
      'frontend/src/i18n/locales.js, al CHECK de esta migración y a _LOCALE_VALUES '
      'del backend) ANTES de aplicar la constraint.',
      bad_count;
  END IF;
END $$;

-- 3. El CHECK.
ALTER TABLE public.user_profiles
  DROP CONSTRAINT IF EXISTS user_profiles_locale_supported;

ALTER TABLE public.user_profiles
  ADD CONSTRAINT user_profiles_locale_supported
  CHECK (locale IN ('es-DO', 'en-US', 'pt-BR', 'fr-FR', 'it-IT'));

COMMENT ON COLUMN public.user_profiles.locale IS
'[P1-I18N-DASHBOARD · 2026-08-15] Idioma de la INTERFAZ del dashboard. NO afecta '
'al contenido generado (planes, recetas, coach) ni a los nombres del catálogo de '
'alimentos, que siguen en es-DO. SSOT de la lista: frontend/src/i18n/locales.js.';

COMMENT ON CONSTRAINT user_profiles_locale_supported ON public.user_profiles IS
'[P1-I18N-DASHBOARD · 2026-08-15] Solo los 5 idiomas soportados. Al añadir uno, '
'tocar los cuatro espejos: locales.js, este CHECK, _LOCALE_VALUES (backend) y el '
'boot de index.html.';
