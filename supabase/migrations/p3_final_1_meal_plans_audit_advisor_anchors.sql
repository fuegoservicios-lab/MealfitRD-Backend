-- [P3-FINAL-1 · 2026-05-11] Anclar advisors INFO/WARN intencionales
-- de `meal_plans_audit` (introducida por P2-NEW-5 · 2026-05-11) en
-- COMMENTs SSOT, paralelo al patrón P2-PERF-1.
--
-- Causa raíz:
--   Tras crear la tabla `meal_plans_audit` (backup defensivo append-only
--   referenciado por SOP P3-AUDIT-6), el linter de Supabase emite 4
--   advisors INFO/WARN que NO están auto-suprimidos:
--
--     1. rls_enabled_no_policy → public.meal_plans_audit
--     2. unused_index → idx_meal_plans_audit_meal_plan_id
--     3. unused_index → idx_meal_plans_audit_user_id
--     4. unused_index → idx_meal_plans_audit_action_created
--
--   Los 4 son INTENCIONALES y la justificación está en la migración
--   `p2_new_5_meal_plans_audit_table.sql`, pero un futuro operador
--   leyendo el output del linter no tiene visibilidad de esa migración
--   sin búsqueda manual. Al añadir COMMENTs en pg_class metadata,
--   `\d+ <objeto>` o `obj_description(<oid>, 'pg_class')` devuelven
--   la razón en línea — mismo patrón que P2-PERF-1 para los otros 3
--   índices KEEP del repo.
--
-- Sin cambio de comportamiento: solo añade COMMENT (metadata catálogo);
-- la tabla y los índices físicos ya existen desde P2-NEW-5. Idempotente:
-- `COMMENT ON ...` reemplaza el comentario previo si ya estaba aplicado.
--
-- Cross-link: la sección "Advisors aceptados" de CLAUDE.md mantiene la
-- tabla canónica. Si Supabase añade supresión nativa de advisors en el
-- dashboard, este archivo se conserva como referencia histórica.

-- 1) RLS sin policies (deliberado — solo service_role accede via SOP)
COMMENT ON TABLE public.meal_plans_audit IS
    '[P2-NEW-5 . 2026-05-11 / P3-FINAL-1 . 2026-05-11] Backup defensivo append-only de plan_data pre-mutacion correctiva. Referenciado por SOP P3-AUDIT-6 en CLAUDE.md. Solo INSERT (write-once log). RLS ENABLED + FORCE sin policies es DELIBERADO: solo service_role escribe/lee (operadores SRE acceden via dashboard server-side). Advisor rls_enabled_no_policy aceptado.';

-- 2/3/4) Indices serving SOP queries — naturalmente "unused" en traffic normal
COMMENT ON INDEX public.idx_meal_plans_audit_meal_plan_id IS
    '[P3-FINAL-1 . 2026-05-11] KEEP. Sirve lookup principal del SOP P3-AUDIT-6 (recuperar ultimo estado pre-fix de un plan especifico): SELECT plan_data_before FROM meal_plans_audit WHERE meal_plan_id = ? ORDER BY created_at DESC. Tabla operacional rara → advisor unused_index reporta 0 scans pero el indice es load-bearing en incidente. Leccion P2-5/P2-PERF-1.';

COMMENT ON INDEX public.idx_meal_plans_audit_user_id IS
    '[P3-FINAL-1 . 2026-05-11] KEEP. Sirve queries forensics post-incidente filtrando por user_id (auditoria cross-plan de un mismo usuario). Partial index (user_id IS NOT NULL) por eficiencia: el plan puede haber sido borrado dejando user_id null. Tabla operacional rara → advisor unused_index reporta 0 scans pero load-bearing en SOP. Leccion P2-5/P2-PERF-1.';

COMMENT ON INDEX public.idx_meal_plans_audit_action_created IS
    '[P3-FINAL-1 . 2026-05-11] KEEP. Sirve analytics agregadas del SOP P3-AUDIT-6 paso 7 (post-mortem si los incidentes se repiten >3 por semana sobre el mismo field_name): SELECT action, COUNT(*) FROM meal_plans_audit WHERE created_at > NOW() - INTERVAL ? GROUP BY action. Tabla operacional rara → advisor unused_index reporta 0 scans. Leccion P2-5/P2-PERF-1.';
