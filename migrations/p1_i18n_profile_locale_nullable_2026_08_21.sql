-- [P1-I18N-PROFILE-DEFAULT-PISA · 2026-08-21] `user_profiles.locale` pasa a admitir
-- NULL, y NULL significa «este usuario todavía no ha elegido idioma».
--
-- ═══════════════════════════════════════════════════════════════════════════
-- POR QUÉ
-- ═══════════════════════════════════════════════════════════════════════════
--
-- La columna nació como `text NOT NULL DEFAULT 'es-DO'`, y eso hacía imposible que el
-- perfil dijera «no he elegido»: «español» y «todavía nada» eran el MISMO valor.
--
-- La traza que eso rompía, verificada paso a paso:
--
--   1. Un visitante nuevo anglófono → `P1-AUTO-LOCALE` lo detecta a `en-US`. Correcto.
--   2. Se registra. La fila nace con el DEFAULT `'es-DO'`.
--   3. `fetchProfile` llama a `syncLocaleFromProfile('es-DO')`. Como es un locale
--      soportado y distinto del activo, el motor entra, carga el catálogo español y
--      **escribe `'es-DO'` en `localStorage`**.
--   4. La autodetección no vuelve a dispararse JAMÁS: `getStoredLocale()` encuentra un
--      valor soportado y sale antes de consultarla.
--
-- O sea: la feature desplegada el 2026-08-20 funcionaba sólo para visitantes anónimos, y
-- el primer login la apagaba de forma permanente. Y no «fallaba»: hacía exactamente lo
-- que el esquema le permitía distinguir.
--
-- Es la misma lección que `P1-COUNTRY-RENEW-OVERWRITE` dejó con el país: **un default
-- sembrado es indistinguible de una elección**.
--
-- ═══════════════════════════════════════════════════════════════════════════
-- POR QUÉ `NULL` Y NO UNA COLUMNA HERMANA `locale_explicit`
-- ═══════════════════════════════════════════════════════════════════════════
--
-- La alternativa era `locale_explicit boolean DEFAULT false` que `Settings.jsx` pone a
-- `true`. No se elige porque añade un estado más que mantener sincronizado con el
-- primero —dos columnas que pueden contradecirse— y porque `NULL` ya degrada correcto
-- sin tocar el motor: `syncLocaleFromProfile` empieza con
-- `if (!isSupportedLocale(profileLocale)) return false;`, así que un `null` sale por ahí
-- y el idioma DETECTADO se conserva.
--
-- ═══════════════════════════════════════════════════════════════════════════
-- LOS CUATRO LECTORES DEL BACKEND, verificados uno a uno ANTES de tocar el DEFAULT
-- ═══════════════════════════════════════════════════════════════════════════
--
--   · `agent.py` (×2)        `.get("locale") or "es-DO"`
--   · `proactive_agent.py`   `.get("locale") or "es-DO"`
--   · `cron_tasks.py`        `if _p1_i18n_locale and _p1_i18n_locale != "es-DO"`
--                            (NULL es falsy → no se despacha traducción, que es lo
--                             correcto para quien no ha elegido idioma)
--   · `routers/user_data.py` sólo valida en la ESCRITURA
--
-- Ninguno rompe con NULL. `test_p1_i18n_profile_default_pisa.py` los ancla para que
-- sigan así: quitar un `or "es-DO"` convertiría el NULL en un `None` viajando hasta un
-- prompt o hasta un `import()`.
--
-- LAS FILAS EXISTENTES NO SE TOCAN. Quien ya tenga `'es-DO'` lo conserva: puede que lo
-- eligiera de verdad, y no hay forma de saberlo a posteriori. Esto arregla a los
-- usuarios NUEVOS, que es donde el defecto duele; convertir en NULL a los actuales
-- sería inventar una intención que nadie expresó.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC) y presente en los DOS directorios
-- (P3-MIGRATIONS-SSOT).

-- 1. Sanity ANTES de tocar nada: si hay valores fuera de la lista, parar y decirlo.
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
      'P1-I18N-PROFILE-DEFAULT-PISA: % filas de user_profiles tienen un locale fuera '
      'de la lista soportada. Normalizarlas ANTES de re-aplicar la constraint.',
      bad_count;
  END IF;
END $$;

-- 2. El DEFAULT y el NOT NULL. Los dos, y en este orden: quitar sólo el DEFAULT dejaría
--    la columna imposible de poner a NULL, o sea «no elegido» seguiría sin ser
--    representable — el arreglo a medias que parece hecho.
ALTER TABLE public.user_profiles
  ALTER COLUMN locale DROP DEFAULT;

ALTER TABLE public.user_profiles
  ALTER COLUMN locale DROP NOT NULL;

-- 3. El CHECK se vuelve a declarar admitiendo NULL EXPLÍCITAMENTE.
--
--    En Postgres un CHECK que evalúa a NULL deja pasar la fila, así que `locale IN (…)`
--    con `locale = NULL` ya funcionaría por accidente. Se escribe el `IS NULL` a
--    propósito: la diferencia entre «funciona» y «alguien decidió que funcione» es lo
--    único que impide que el siguiente lector lo tome por un descuido y lo «arregle».
ALTER TABLE public.user_profiles
  DROP CONSTRAINT IF EXISTS user_profiles_locale_supported;

ALTER TABLE public.user_profiles
  ADD CONSTRAINT user_profiles_locale_supported
  CHECK (locale IS NULL OR locale IN ('es-DO', 'en-US', 'pt-BR', 'fr-FR', 'it-IT'));

-- [P3-I18N-COMMENT-DB-ALCANCE-STALE · 2026-08-22] El texto de este COMMENT se corrige
-- ANTES de aplicarlo por primera vez. La versión anterior decía «NO afecta al contenido
-- generado», y eso lleva siendo falso desde el 2026-08-17 (prosa del coach) y el
-- 2026-08-19 (capa `_display` del plan). Aplicar la migración tal y como estaba escrita
-- habría metido en producción un comentario que ya sabíamos falso — que es exactamente
-- el daño que `P2-I18N-DOC-ALCANCE-MIENTE` documentó: una doc canónica equivocada no
-- confunde sólo a las personas, y ésta la lee quien abre el esquema.
COMMENT ON COLUMN public.user_profiles.locale IS
'[P1-I18N-DASHBOARD · 2026-08-15 · NULL desde P1-I18N-PROFILE-DEFAULT-PISA 2026-08-21 · '
'alcance corregido P3-I18N-COMMENT-DB-ALCANCE-STALE 2026-08-22] Idioma del usuario. '
'NULL = NO ha elegido todavía, y entonces manda la autodetección del navegador '
'(P1-AUTO-LOCALE). Antes la columna era NOT NULL DEFAULT ''es-DO'' y eso hacía '
'indistinguible «español» de «todavía nada»: el primer login escribía es-DO en '
'localStorage y apagaba la autodetección para siempre. GOBIERNA: la interfaz del '
'dashboard, la PROSA del coach (chat + notificaciones proactivas, '
'prompts.chat_agent.build_language_directive) y la capa `_display` que traduce plan, '
'recetas e insights (backend/plan_display_i18n.py, knob MEALFIT_PLAN_DISPLAY_I18N). '
'NO GOBIERNA, JAMÁS: los nombres de alimentos y platos del catálogo, que siguen en '
'español canónico porque son el SSOT de pantry_names_match, del guard de coherencia '
'recetas↔lista y del backstop clínico de alergias. SSOT de la lista de idiomas: '
'frontend/src/i18n/locales.js.';

COMMENT ON CONSTRAINT user_profiles_locale_supported ON public.user_profiles IS
'[P1-I18N-DASHBOARD · 2026-08-15 · NULL desde 2026-08-21] Sólo los 5 idiomas '
'soportados, o NULL (= sin elegir). Al añadir uno, tocar los espejos: locales.js, este '
'CHECK, _LOCALE_VALUES (backend) y el boot de index.html — la lista completa de los 12, '
'con qué se ve si falta cada uno, en la §6 de backend/docs/i18n_dashboard.md.';
