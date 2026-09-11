-- [P1-PLAN-LOTE-2 · 2026-09-11 · F7] Fósforo de dos filas que lo tenían en NULL, con su fuente.
--
-- Cinco filas del catálogo no traen `phosphorus_mg_per_100g` (Achiote, Borojó, Champús, Chontaduro, Hoja
-- santa; todas sin `fdc_id`: USDA no las tiene). Un plan renal EXIGE conocer el fósforo
-- (`horizon.required_nutrients` ⇒ `require_known_nutrients`), así que las plantillas que las usan quedan
-- fuera para esos usuarios — que es el comportamiento correcto mientras el dato falte. Aquí entran las DOS
-- con un valor publicado y citable; las otras tres siguen en NULL a sabiendas (la TCAC 2018 del ICBF está
-- escaneada sin texto y no se pudo leer el valor; adivinar sería peor que excluir).
--
--   · Borojó (Alibertia patinoi), pulpa: 160 mg P / 100 g de porción comestible — tabla de composición
--     colombiana citada en Revista IAlimentos («Borojó, todo lo que debe saber») y en la ficha de El Tiempo
--     (93 kcal, 1,1 g proteína, 24,7 g carbohidratos, 25 mg Ca, 1,5 mg Fe).
--   · Hoja santa (Piper auritum), hoja fresca: 38,0 mg P / 100 g — «Quelites usados en alimentación
--     avícola» (Oaxaca; ResearchGate 371361497), tabla de composición de quelites.
--
-- Procedencia en las columnas que ya existen para eso: `nutrition_source='manual'` con
-- `nutrition_source_ref` y `nutrition_source_date` (mismo patrón que las 44 filas manuales del 2026-08-17).
-- Idempotente: sólo escribe si el fósforo sigue en NULL.
--
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.

UPDATE public.master_ingredients
   SET phosphorus_mg_per_100g = 160.0,
       nutrition_source = COALESCE(nutrition_source, 'manual'),
       nutrition_source_ref = COALESCE(nutrition_source_ref, '') ||
           CASE WHEN COALESCE(nutrition_source_ref, '') = '' THEN '' ELSE ' | ' END ||
           'P: 160 mg/100 g, tabla colombiana vía Revista IAlimentos / El Tiempo (2026-09-11)',
       nutrition_source_date = CURRENT_DATE
 WHERE name = 'Borojó' AND phosphorus_mg_per_100g IS NULL;

UPDATE public.master_ingredients
   SET phosphorus_mg_per_100g = 38.0,
       nutrition_source = COALESCE(nutrition_source, 'manual'),
       nutrition_source_ref = COALESCE(nutrition_source_ref, '') ||
           CASE WHEN COALESCE(nutrition_source_ref, '') = '' THEN '' ELSE ' | ' END ||
           'P: 38 mg/100 g, «Quelites usados en alimentación avícola» (ResearchGate 371361497) (2026-09-11)',
       nutrition_source_date = CURRENT_DATE
 WHERE name = 'Hoja santa' AND phosphorus_mg_per_100g IS NULL;
