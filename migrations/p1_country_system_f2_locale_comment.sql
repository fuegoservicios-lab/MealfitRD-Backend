-- [P1-COUNTRY-SYSTEM-F2 · T3 · 2026-08-17] Actualiza el COMMENT de
-- `user_profiles.locale` — NO toca la columna, el DEFAULT ni el CHECK (siguen
-- vivos en `p1_i18n_dashboard_locale_2026_08_15.sql`, que NO se edita: ya
-- aplicada en producción, y una migración ya aplicada no se reabre).
--
-- Por qué:
--   El comment original (P1-I18N-DASHBOARD) decía que la columna "NO afecta
--   al contenido generado (planes, recetas, coach)". El Addendum del dueño
--   §2 al sistema de países ("Idioma ≠ país, extendido al AGENTE") hizo que
--   la PROSA del coach (chat + notificaciones proactivas) SÍ siga `locale`
--   desde P1-COUNTRY-SYSTEM-F2 Task 3 — es el pedido en vivo del dueño, NO
--   parte del sistema de países en oscuro, así que quedó activo desde su
--   propio deploy, sin esperar al flip de `MEALFIT_COUNTRY_SYSTEM`. El
--   comment viejo pasó a describir la columna incorrectamente en ese
--   instante; este archivo lo corrige sin reabrir la migración original.
--
-- Idempotente: `COMMENT ON COLUMN` sobre-escribe el comentario existente sin
-- error en cualquier número de re-aplicaciones — no requiere IF NOT EXISTS.
--
-- SSOT de migrations (P3-MIGRATIONS-SSOT): este archivo vive IDÉNTICO en
-- `migrations/` (workspace-root) y `backend/migrations/`.

COMMENT ON COLUMN public.user_profiles.locale IS
'[P1-I18N-DASHBOARD · 2026-08-15 · actualizado P1-COUNTRY-SYSTEM-F2 T3 · 2026-08-17] '
'Idioma de la INTERFAZ del dashboard Y de la PROSA del coach (chat + notificaciones '
'proactivas — ver prompts.chat_agent.build_language_directive en el backend). NO '
'afecta el plan, las recetas, la lista de compras ni los nombres del catálogo de '
'alimentos, que SIEMPRE quedan en español canónico (SSOT de pantry_names_match, el '
'guard de coherencia recetas↔lista y el backstop de alergias). SSOT de la lista de '
'idiomas: frontend/src/i18n/locales.js.';
