# Advisors aceptados (no actuar)

> Movido desde `CLAUDE.md` el 2026-07-26 por [P3-CLAUDEMD-CAP]. El propio
> `test_p3_claudemd_cap` nombraba esta tabla como la siguiente candidata a salir: es la segunda
> seccion mas grande y la que menos se consulta por turno.
>
> **Contexto importante:** estos advisors los emitia el linter de **Supabase**, que ya no corre
> — el proyecto migro a Neon por completo (P1-NEON-DB-MIGRATION, 2026-06-12). Se conserva porque
> el RAZONAMIENTO sigue vigente: por que un indice aparentemente sin uso es load-bearing (cubre
> una FK, o sirve un SOP de incidente), y por que una funcion es `SECURITY DEFINER` a proposito.
> Si alguna vez se corre un linter equivalente sobre Neon, esta es la lista de "ya decidido".
>
> Los anclajes en migraciones los enforza
> [`test_p2_whitelist_advisors_anchors_alive.py`](../tests/test_p2_whitelist_advisors_anchors_alive.py),
> que mantiene su PROPIA copia de los nombres canonicos (no parsea este archivo).

Esta sección documenta los advisors de Supabase que han sido auditados y declarados intencionales. Si vuelven a aparecer en el linter (security/performance), **no actuar**: la decisión está tomada y la razón está fija. Si quieres cambiarlas, primero lee la memoria correspondiente para entender el contexto.

### Security

| Advisor | Estado | Razón | Memoria de cierre |
|---|---|---|---|
| `authenticated_security_definer_function_executable` (`increment_inventory_quantity`) | **WARN intencional** | Frontend usa `RPC` directo para incrementos atómicos en pantry. Switching a `SECURITY INVOKER` rompería la operación bajo concurrencia. La función internamente fuerza `WHERE user_id = auth.uid()` (7 tests de regresión). | [`project_p2_4_increment_inventory_decision_2026_05_07.md`](~/.claude/projects/.../memory/project_p2_4_increment_inventory_decision_2026_05_07.md) |
| `auth_leaked_password_protection` (Disabled) | **WARN intencional** | Toggle nativo de Supabase requiere plan Pro. Implementado en frontend vía HIBP k-anonymity (Register + Reset). Knob `VITE_LEAKED_PASSWORD_CHECK`. | [`project_p2_3_leaked_password_self_implemented_2026_05_07.md`](~/.claude/projects/.../memory/project_p2_3_leaked_password_self_implemented_2026_05_07.md) |
| `rls_enabled_no_policy` (`meal_plans_audit`) | **INFO intencional** | Tabla operacional append-only (SOP P3-AUDIT-6, backup defensivo pre-mutación). RLS ENABLED + FORCE sin policies bloquea PostgREST por completo: solo `service_role` escribe/lee (SRE via dashboard server-side). No hay clientes externos. | [`project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md`](~/.claude/projects/.../memory/project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md) |

### Performance

| Advisor | Estado | Razón | Memoria de cierre |
|---|---|---|---|
| `unused_index` (`idx_chunk_lesson_telemetry_plan_week`) | **INFO intencional** | Cubre FK `chunk_lesson_telemetry_meal_plan_id_fkey` (ON DELETE SET NULL) + sirve query de `/lifetime-lessons` filtrando por `(meal_plan_id, week_number)`. Advisor `unused_index` NO observa uso interno por FK. | [`project_p1_hist_new_7_chunk_lesson_telemetry_plan_week_idx.md`](~/.claude/projects/.../memory/project_p1_hist_new_7_chunk_lesson_telemetry_plan_week_idx.md) |
| `unused_index` (`idx_failed_inventory_deductions_user_id`) | **INFO intencional** | Cubre FK a `auth.users(id) ON DELETE CASCADE`. Sin el índice, eliminar un usuario auth haría seq-scan masivo. Lección P2-5: el advisor `unused_index` NO observa uso interno por FK. | [`project_p2_perf_1_consolidate_unused_index_comments_2026_05_10.md`](~/.claude/projects/.../memory/project_p2_perf_1_consolidate_unused_index_comments_2026_05_10.md) |
| `unused_index` (`idx_nightly_rotation_queue_user_id`) | **INFO intencional** | Cubre FK a `user_profiles(id) ON DELETE CASCADE`. Misma lección P2-5. | [`project_p2_perf_1_consolidate_unused_index_comments_2026_05_10.md`](~/.claude/projects/.../memory/project_p2_perf_1_consolidate_unused_index_comments_2026_05_10.md) |
| `unused_index` (`idx_meal_plans_audit_meal_plan_id`) | **INFO intencional** | Sirve lookup principal del SOP P3-AUDIT-6 (`SELECT plan_data_before WHERE meal_plan_id = ? ORDER BY created_at DESC`). Tabla operacional rara: advisor reporta 0 scans pero el índice es load-bearing en incidente. Misma lección P2-PERF-1. | [`project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md`](~/.claude/projects/.../memory/project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md) |
| `unused_index` (`idx_meal_plans_audit_user_id`) | **INFO intencional** | Sirve queries forensics post-incidente filtrando por `user_id` (auditoría cross-plan de un usuario). Partial index `WHERE user_id IS NOT NULL` por eficiencia. Tabla operacional rara: misma lección P2-PERF-1. | [`project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md`](~/.claude/projects/.../memory/project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md) |
| `unused_index` (`idx_meal_plans_audit_action_created`) | **INFO intencional** | Sirve analytics del SOP P3-AUDIT-6 paso 7 (post-mortem si incidentes se repiten >3 por semana sobre el mismo field): `SELECT action, COUNT(*) WHERE created_at > NOW() - INTERVAL ? GROUP BY action`. Misma lección P2-PERF-1. | [`project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md`](~/.claude/projects/.../memory/project_p3_final_1_meal_plans_audit_advisors_2026_05_11.md) |
| `unused_index` (`idx_agent_messages_user_id`) | **INFO intencional** | Partial index `WHERE user_id IS NOT NULL` cubre FK CASCADE a `auth.users(id)` + sirve queries cross-session del chat-agent. Advisor `unused_index` NO observa FK. | Migración SSOT `p2_unused_idx_advisor_anchors_2026_05_20.sql` (P2-UNUSED-IDX · 2026-05-20). Inicial en `db_p1_chat_user_id_rls_2026_05_19.sql`. |
| `unused_index` (`idx_conversation_summaries_user_id`) | **INFO intencional** | Partial index `WHERE user_id IS NOT NULL` cubre FK CASCADE a `auth.users(id)` + sirve filtro de `search_deep_memory`. Misma lección. | Migración SSOT `p2_unused_idx_advisor_anchors_2026_05_20.sql` (P2-UNUSED-IDX · 2026-05-20). |
| `unused_index` (`idx_llm_usage_events_model_created`) | **INFO intencional** | Sirve queries analytics admin-only de cost-by-model en `/api/admin/cost-by-node` ([`routers/system.py:1010+`](backend/routers/system.py#L1010)). Endpoint esporádico (incident diagnosis) → advisor reporta 0 scans. Mantener para diagnóstico de incidentes de costo. | Migración SSOT `p2_unused_idx_advisor_anchors_2026_05_20.sql` (P2-UNUSED-IDX · 2026-05-20). |
| `unused_index` (`idx_user_depleted_items_master_ingredient_id`) | **INFO intencional** | Partial index cubre FK `ON DELETE SET NULL` desde `master_ingredients`. Misma lección P2-PERF-1. | [`project_p2_prod_harden_2026_05_23.md`](~/.claude/projects/.../memory/project_p2_prod_harden_2026_05_23.md) · migración SSOT [`p2_user_depleted_items_fk_idx_2026_05_23.sql`](migrations/p2_user_depleted_items_fk_idx_2026_05_23.sql) |

> **[P3-CLAUDEMD-MARGIN-RESTORE · 2026-08-04]** Bloque movido ÍNTEGRO desde `CLAUDE.md`
> (sección «Advisors aceptados») para restaurar el margen del cap (`test_p1_prod_final_1`
> exige ≥800 bytes bajo el cap de `test_p3_claudemd_cap`). CLAUDE.md conserva la REGLA en
> una línea + link a este doc; el INVENTARIO (qué functions, qué migración, qué GRANT) vive
> aquí. El texto de abajo es literalmente el que vivía en CLAUDE.md — cero contenido perdido.
> Nota: la línea introductoria de «Advisors aceptados» en CLAUDE.md ya prometa este
> contenido en este doc desde 2026-07-26; hasta hoy la promesa era falsa.

### Pattern: `SET search_path = ''` en functions Postgres

[P3-NEW-2 · 2026-05-10] Patrón canónico para functions nuevas: `SET search_path = ''` + `SECURITY <DEFINER|INVOKER>` explícito. La cadena vacía fuerza qualifier explícito (`public.<obj>`, `auth.<obj>`) y previene shadowing por temp tables (vs `'public'` que es vulnerable). Narrativa "por qué `''` no `'public'`" + ejemplo SQL boilerplate: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). **Functions ya bajo el pattern:**

| Function | Migración | `search_path` | EXECUTE granted to |
|---|---|---|---|
| `set_meal_plans_updated_at` | `p2_new_1_set_meal_plans_updated_at_search_path.sql` | `''` | trigger (no-direct) |
| `apply_inventory_delta` | `p0_4_apply_inventory_delta_rpc.sql` | `'public'` (acepta — refs qualified) | `service_role` |
| `increment_inventory_quantity` | runtime/historical | `auth, public, extensions` (legacy, ver P2-4 memoria) | `authenticated` + `service_role` (P2-4) |
| `handle_new_user` | [`p1_definer_functions_lockdown_2026_05_12.sql`](migrations/p1_definer_functions_lockdown_2026_05_12.sql) | `''` (P1-DEFINER-LOCKDOWN) | `service_role` (REVOKE explícito) |
| `get_monthly_plan_count` | mismo | `''` | `service_role` (REVOKE explícito; función huérfana, 0 callsites) |
| `log_unknown_ingredient_rpc` | mismo | `''` | `service_role` (REVOKE explícito; callsite [`db_plans.py`](backend/db_plans.py)) |

Si añades function nueva: aplicar el pattern, justificar excepción en COMMENT ON FUNCTION + memoria si necesitas resolver nombres sin qualifier.

**[P1-DEFINER-LOCKDOWN · 2026-05-12]** Functions `SECURITY DEFINER` que aceptan `user_id`/`p_user_id` parameter sin validar contra `auth.uid()` DEBEN incluir `REVOKE EXECUTE ... FROM PUBLIC, anon, authenticated` explícito en migración SSOT — defensa contra GRANT por error que abriría IDOR cross-user. Test: [`test_p1_definer_lockdown_migration.py`](backend/tests/test_p1_definer_lockdown_migration.py).

### Cómo verificar

Cada item está respaldado por `COMMENT ON INDEX` (índices) o `COMMENT ON FUNCTION` (definers) en migración SSOT — el linter ve el COMMENT pero sigue reportando el advisor (es informational, no auto-suprimido). El operador debe leer el comment vía `\d+ <objeto>` o `obj_description(<oid>, 'pg_class')` antes de actuar.

Si Supabase agrega supresión nativa de advisors aceptados en el dashboard, mover esta sección a la UI de Supabase y dejar este bloque como referencia.
