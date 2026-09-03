-- [P3-COUNTRY-DB-CHECK · 2026-08-22] `health_profile.country`: el único de los
-- tres ajustes de Configuración sin defensa en la base de datos.
--
-- ⚠️ ESTA MIGRACIÓN NO se ha aplicado a producción. La aplica el dueño en su
-- ventana de deploy, tras leer el bloque «Antes de aplicar» del final.
--
-- EL CONTRASTE QUE LA MOTIVA. `locale` nació el mismo mes, en la misma tabla, y
-- tiene las cuatro cosas: columna propia, NOT NULL, DEFAULT y CHECK; además su
-- endpoint valida el VALOR con un 400. El sistema de países entero —seis
-- países, flip ejecutado el 2026-08-18— no tenía ni una sola migración, y
-- `country` vive dentro del JSONB `health_profile`, que `PATCH /api/profile`
-- mergea con `||` crudo sin mirar qué hay dentro.
--
-- POR QUÉ UN CHECK Y NO SÓLO LA VALIDACIÓN DEL ENDPOINT. Es literalmente el
-- argumento que escribió la migración de `locale`, y sigue siendo cierto: la
-- invariante vive en la DB, no sólo en el camino que HOY resulta que la
-- escribe. Hoy el único escritor es el selector de Configuración, que manda
-- códigos ISO. Mañana es un script de soporte, un backfill, o el
-- `update_form_field` del chat — que ya fue el tercer setter de país sin
-- jerarquía (P2-TOOLS-FIELD-WHITELIST) y que el system prompt ordena llamar
-- «OBLIGATORIO y SIN EXCEPCIÓN» ante cualquier dato personal nuevo. Un «me mudé
-- a España» escribiría country='España'.
--
-- Y el fallo es SILENCIOSO para quien lo sufre: `canonicalize_country` cae a
-- 'DO' ante cualquier valor no canónico, así que ese usuario recibiría planes
-- dominicanos indefinidamente sin nada en pantalla que se lo dijera.
-- P2-COUNTRY-FAILSAFE-LOUD le puso un logger.warning a ese momento, pero un log
-- avisa al operador DESPUÉS y no impide la escritura.
--
-- POR QUÉ NO SE PROMUEVE A COLUMNA. Sería el arreglo «completo» y es otro
-- proyecto: hay lectores del JSONB por todo el backend y el frontend. El patrón
-- «CHECK sobre expresión jsonb» ya es el de la invariante I8
-- (meal_plans_complete_requires_days), que lleva meses sosteniéndose.
--
-- POR QUÉ ADMITE NULL. Medido en producción hoy (SELECT read-only, 16 perfiles):
--   health_profile->>'country'   NULL: 15   'DO': 1
--   jsonb_typeof(->'country')    string: 1
-- La AUSENCIA es el caso normal —15 de 16— y es legítima: el fail-safe por
-- ausencia está declarado y no se toca. Lo que este CHECK cierra es el otro
-- caso, el string que no canoniza. Un CHECK que no admitiera NULL rechazaría a
-- casi toda la base instalada.
--
-- SSOT de la lista: `constants.COUNTRY_PROFILES`. Este CHECK es su espejo;
-- `test_p3_country_db_check.py` falla si divergen.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): DROP CONSTRAINT IF EXISTS antes
-- del ADD, y sanity DO $$ RAISE EXCEPTION antes de imponer.

-- 1. Sanity ANTES del CHECK: si alguna fila trae un país fuera de la lista, la
--    migración se detiene y dice cuántas. Hoy sale limpio; en una re-aplicación
--    sobre datos futuros no tiene por qué. Este repo ya vio un dry-run abortar
--    contra 12 filas preexistentes y CORRECTAS: mejor que aborte hablando.
DO $$
DECLARE
  bad_count int;
BEGIN
  SELECT COUNT(*) INTO bad_count
  FROM public.user_profiles
  WHERE health_profile->>'country' IS NOT NULL
    AND health_profile->>'country' NOT IN ('DO', 'ES', 'US', 'MX', 'PR', 'CO');
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      'P3-COUNTRY-DB-CHECK: % filas de user_profiles tienen health_profile.country '
      'fuera de la lista canónica. Normalizarlas (o añadir el país a '
      'constants.COUNTRY_PROFILES, al CHECK de esta migración y al selector del '
      'frontend) ANTES de aplicar la constraint. Consulta para verlas: '
      'SELECT id, health_profile->>''country'' FROM public.user_profiles '
      'WHERE health_profile->>''country'' IS NOT NULL AND health_profile->>''country'' '
      'NOT IN (''DO'',''ES'',''US'',''MX'',''PR'',''CO'');',
      bad_count;
  END IF;
END $$;

-- 2. El CHECK. NULL pasa (ausencia legítima, 15 de 16 filas vivas). Un valor
--    presente tiene que ser uno de los seis códigos ISO del selector.
ALTER TABLE public.user_profiles
  DROP CONSTRAINT IF EXISTS user_profiles_country_supported;

ALTER TABLE public.user_profiles
  ADD CONSTRAINT user_profiles_country_supported
  CHECK (
    health_profile->>'country' IS NULL
    OR health_profile->>'country' IN ('DO', 'ES', 'US', 'MX', 'PR', 'CO')
  );

-- 3. Índice parcial sobre el país, que hasta hoy tampoco existía.
--    Parcial (WHERE ... IS NOT NULL) porque 15 de 16 filas no tienen país: un
--    índice completo indexaría sobre todo NULLs. Sirve a las consultas
--    operativas por país (cuántos usuarios beta hay, y de dónde) que hoy hacen
--    seq scan sobre el JSONB.
CREATE INDEX IF NOT EXISTS idx_user_profiles_country
  ON public.user_profiles ((health_profile->>'country'))
  WHERE health_profile->>'country' IS NOT NULL;

-- ── Antes de aplicar ────────────────────────────────────────────────────────
--
-- 1. Contar lo que hay (read-only, seguro en cualquier momento):
--      SELECT health_profile->>'country' AS pais, count(*)
--      FROM public.user_profiles GROUP BY 1 ORDER BY 2 DESC;
--    Medido el 2026-08-22: 15 NULL + 1 'DO'. Si sale sólo eso, aplica limpia.
--
-- 2. Si aparece algún valor fuera de la lista, el bloque DO $$ abortará la
--    transacción entera y no dejará nada a medias. Normaliza esas filas primero
--    (o borra la clave: la ausencia es válida y cae a 'DO' por el fail-safe, que
--    es exactamente lo que ese usuario ya está recibiendo hoy).
--
-- 3. Tras aplicar, el PATCH del endpoint sigue siendo la primera línea: el
--    CHECK es la red de abajo, no el sustituto. Un 400 le dice al usuario qué
--    pasó; una violación de constraint le da un 500.
