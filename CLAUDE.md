# MealfitRD.IA — guía operacional

Plan nutricional generado por LLM para usuario dominicano. Backend Python/FastAPI + LangGraph + Neon (Postgres + Neon Auth). Frontend React/Vite. Detalles cronológicos de los ~80 P-fixes activos: ver `~/.claude/projects/c--Users-angel-OneDrive-Escritorio-MealfitRD-IA/memory/MEMORY.md`.

---

## Lifecycle de `plan_id` (formulario → PDF)

[P3-NEW-1 · 2026-05-11] El identificador `plan_id` viaja por 6 sistemas. Conocer dónde nace, dónde se persiste y dónde puede contaminarse cierra una clase entera de bugs IDOR/drift que ya enfrentamos múltiples veces.

> **Diagrama visual** (formulario → orquestador → chunks → shopping/PDF → historial → recipe expand): ver [`runbook_plan_id_lifecycle.md`](~/.claude/projects/.../memory/runbook_plan_id_lifecycle.md). Las invariantes I1-I8 abajo son el contrato verificado por tests; el diagrama es contexto narrativo.

### Invariantes del lifecycle

| # | Invariante | Defensa |
|---|---|---|
| I1 | `plan_id` NUNCA se asigna client-side. Siempre nace del INSERT backend. | `services.py:_save_plan_and_track_background` |
| I2 | Toda mutación de `meal_plans` filtra `AND user_id = %s`. | `update_meal_plan_data` (P1-NEW-3), `update_plan_data_atomic` (P2-OPEN-1), `/restock` (P0-NEW-1), `/retry-chunk` (P0-HIST-IDOR-1), `/regenerate-simplified` (P1-NEW-4-DEFENSE), `/swap-meal/persist` (P0-NEW-A), `/grocery-start-date` (P0-NEW-B), `/restore-local` (P1-OPEN-1), `_process_pending_shopping_lists` cron recovery (P1-SHOPPING-1); tests parser-based `test_p3_next_1_i2_user_id_filter_contract.py` (routers user-facing) y `test_p1_shopping_1_cron_user_id_filter.py` (cron_tasks background) |
| I3 | Toda lectura cross-page de `plan_data` que vaya a state local valida ownership client-side. | `restorePlan` con `expectedUserId` (P1-NEW-4), `restorePlanFromHistory` pre-check |
| I4 | Invalidación de caches post-mutación. | `Recipes.jsx` (P2-NEW-3), `History.jsx` visibilitychange (P2-NEW-1), `Pantry.jsx` prefetch (P2-NEW-4) |
| I5 | El alert `plan_quality_degraded:<user_id>:<plan_id>` registra los planes entregados con `review_passed=False`. | `_emit_plan_quality_degraded_alert` invocado en las 5 ramas "end" de `should_retry` (P1-NEW-3) |
| I6 | Mutaciones a `plan_data` desde el frontend prohibidas — solo via endpoint backend con `jsonb_set` quirúrgico (NO full overwrite, salvo `restore-local` que es overwrite explícito bajo advisory lock). Las únicas escrituras directas permitidas desde el cliente son: INSERT inicial en `Plan.jsx:398` y DELETE en `user_inventory` (Pantry). | `/swap-meal/persist` (P0-NEW-A), `/grocery-start-date` (P0-NEW-B), `/recipe/expand` (P1-HIST-RECIPE-1), `/{plan_id}/name` (P1-HIST-5), `/{plan_id}/restore-local` (P1-OPEN-1); test blanket `test_p1_new_a_frontend_no_direct_meal_plans_write.py` (P1-NEW-A). **Cero whitelists activas tras P1-OPEN-1.** |
| I7 | Toda escritura de `plan_data` **full-overwrite** (`UPDATE meal_plans SET plan_data = %s::jsonb` o `= %s` con `Jsonb(...)`, NO `jsonb_set`) DEBE estar precedida por `acquire_meal_plan_advisory_lock(cursor, plan_id, purpose="general")` **O** invocarse via `update_plan_data_atomic(plan_id, callback, user_id=...)` (`SELECT … FOR UPDATE` row lock + callback fresh — cierra además la ventana lost-update read-modify-write). `jsonb_set` y jsonb merge `\|\|` exentos. | Locks: `_chunk_worker` T1/T2, `_background_shift_plan_for_user`, `api_shift_plan`, `api_restore_plan_local` (P1-OPEN-1); helper `update_meal_plan_data` (db_plans.py:957, P1-NEXT-1) sin callsites prod activos tras P1-AUDIT-1. **Patrón preferido FOR UPDATE + callback** (`update_plan_data_atomic`, db_plans.py:215, P0-2): `/recalculate-shopping-list` (P1-RECALC-LOSTUPDATE · 2026-05-14), `/recipe/expand` + `proactive_agent` JIT week-2 + `tools.execute_modify_single_meal` (P1-AUDIT-1 · 2026-05-15). Tests: `test_p1_new_b_*`, `test_p1_new_c_*`, `test_p1_open_1_*`, `test_p1_next_1_*`, `test_p1_recalc_lostupdate.py`, `test_p1_audit_1_update_meal_plan_data_lostupdate.py`. |
| I8 | **DB-level CHECK**: si `plan_data->>'generation_status' = 'complete'` entonces `jsonb_array_length(plan_data->'days') > 0`. Cierra modo de corrupción donde chunk worker T1 marcaba `complete` sin que el merge `plan_data.days = merged_days` persistiera (plan 005c5a99 vivió ~14h en prod con `status=complete + days=0`). Si esta constraint falla en runtime, el bug está aguas arriba en el chunk worker — investigar antes de relaxar. | CHECK `meal_plans_complete_requires_days` en `public.meal_plans` (migración SSOT [`migrations/p2_next_4_meal_plans_complete_requires_days.sql`](migrations/p2_next_4_meal_plans_complete_requires_days.sql), P2-NEXT-4). Test parser-based [`test_p2_next_4_meal_plans_complete_requires_days.py`](backend/tests/test_p2_next_4_meal_plans_complete_requires_days.py) ancla la regla + sanity check + idempotencia. |

### Archivos clave

- [`backend/services.py`](backend/services.py) — INSERT/UPDATE de meal_plans.
- [`backend/graph_orchestrator.py:should_retry`](backend/graph_orchestrator.py) — gate de retry + emit alert.
- [`backend/routers/plans.py`](backend/routers/plans.py) — endpoints user-facing (todos validan user_id).
- [`frontend/src/context/AssessmentContext.jsx:restorePlan,restorePlanFromHistory`](frontend/src/context/AssessmentContext.jsx) — guard ownership.
- [`frontend/src/utils/historyCaches.js`](frontend/src/utils/historyCaches.js) — TTL=30min singleton.

---

## Flujo de coherencia recetas↔lista (defensa-en-profundidad)

Tres capas que protegen la invariante "si una receta dice X, la lista de compras tiene X en cantidad ≈ X × household_multiplier". Diseñado contra cuatro modos de fallo conocidos: `cap_swallowed_modifier` (pollo en receta, ausente en lista), fantasmas en lista, drift de magnitud (qty mitad), y `_shopping_coherence_block` no consumido (bug P1-G original).

> **Diagrama visual del flujo** (assemble → review → persistencia + crons horario/diario/semanal): ver [`runbook_coherence_guard_flow.md`](~/.claude/projects/.../memory/runbook_coherence_guard_flow.md). Markers preservados: `P1-NEXT-2`, `P2-NEXT-2`, `P2-NEXT-3`, `P3-NEW-C`.

### Knobs del flujo

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_SHOPPING_COHERENCE_GUARD` | `block` (P1-NEW-1, era `warn`) | `off`/`warn`/`block` — modo del guard. Rollback: `MEALFIT_SHOPPING_COHERENCE_GUARD=warn` sin redeploy |
| `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` | `0.10` | tolerancia magnitud (0..1, exclusivo) |
| `MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION` | `reject_minor` | `degrade`/`reject_minor`/`reject_high` |
| `MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS` | `suplemento` | meals filtradas del aggregator (P2-4) |
| `MEALFIT_COHERENCE_METRICS_INTERVAL_MIN` | `60` | frecuencia cron P3-B |
| `MEALFIT_COHERENCE_METRICS_LOOKBACK_H` | `1` | ventana lookback P3-B |
| `MEALFIT_COH_ALERT_CAP_RATIO` | `0.05` | umbral alerta cron diario |
| `MEALFIT_COH_ALERT_PLAN_FRACTION` | `0.10` | umbral alerta cron diario |
| `MEALFIT_HEARTBEAT_BASELINE_EMIT` | `True` (P2-NEW-7) | emit pipeline_metrics baseline `_chunk_heartbeat_baseline` siempre (no solo anómalo); flip a False si vol problemático |
| `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` | `True` (P2-NEXT-2) | kill switch del persist de `_shopping_coherence_block_history` desde el cron diario. Flip a False si genera contención con write paths |
| `MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS` | `7` (P2-NEXT-3) | edad mínima de planes huérfanos antes de marcarlos abandoned, clamp [1, 90] |
| `MEALFIT_SWEEP_ORPHAN_PLANS_BATCH` | `100` (P2-NEXT-3) | batch size del sweep semanal, clamp [1, 1000] |
| `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` | `True` (P2-COHERENCE-1) | `_chunk_worker T2` escala warn→block selectivo cuando hay divergencias severas (cap_swallowed_modifier o magnitudes >50%). Forza retry vía `_SHOP_MAX_RETRIES`. Flip a `False` revierte al warn-only puro |

Todos los knobs `MEALFIT_*` se auto-registran en `_KNOBS_REGISTRY` vía `_env_int/_float/_bool/_str` (P3-NEW-D). `get_knobs_registry_snapshot()` expone el set actual.

### Archivos clave

- [`backend/graph_orchestrator.py`](backend/graph_orchestrator.py) — `assemble_plan_node` (productor del flag).
- [`backend/graph_orchestrator.py`](backend/graph_orchestrator.py) — `review_plan_node` (consumidor del flag).
- [`backend/shopping_calculator.py`](backend/shopping_calculator.py) — `run_shopping_coherence_guard`.
- [`backend/shopping_calculator.py`](backend/shopping_calculator.py) — `run_shopping_coherence_guard_and_append_history` (P1-NEXT-2 · 2026-05-11, helper SSOT para surfaces auxiliares).
- [`backend/shopping_calculator.py`](backend/shopping_calculator.py) — `expected_sum_from_recipes` (lado izquierdo del guard).
- [`backend/cron_tasks.py`](backend/cron_tasks.py) — `_shopping_coherence_alert_job` (cron diario).
- [`backend/cron_tasks.py`](backend/cron_tasks.py) — `_aggregate_coherence_block_history_metrics` (cron horario P3-B).

### Surfaces que escriben `aggregated_shopping_list*` y status del guard

[P3-NEXT-5 · 2026-05-11] Tabla canónica de **dónde se ejecuta el coherence guard** (cuándo se construye o se modifica `aggregated_shopping_list*` y qué garantías ofrece cada surface). Esta es la tabla "negativa" que faltaba: enumera explícitamente los surfaces que NO bloquean (solo telemetría warn) vs los que SÍ bloquean (retry forzado), para que un futuro refactor no asuma que el guard es universal.

Tabla canónica movida a [`backend/docs/coherence_surfaces_table.md`](backend/docs/coherence_surfaces_table.md). 6 surfaces, 6 valores canónicos de `action_taken`: `not_applicable` (assemble_plan_node warn/no-critical), `post_swap_revalidation` (`_recompute_aggregates_after_swap`, P2-B), `warn_only_chunk_t2` (`_chunk_worker` T2 con block-severe-only escalada), `warn_only_recalc` (`/recalculate-shopping-list`, NO bloquea — caller síncrono), `warn_only_agent_tool` (`tools.modify_single_meal`), `warn_only_cron_daily` (`_shopping_coherence_alert_job`, P2-NEXT-2 persiste history retroactivo).

Columna `Bloquea retry?` distingue las 6 surfaces: solo surface #1 (`assemble_plan_node → review_plan_node`) responde "Sí" (puede forzar retry vía `should_retry`); las 5 auxiliares responden "No" (solo emiten telemetría post-hoc en `_shopping_coherence_block_history`). Tests: [`test_p1_next_2_guard_at_persist_sites.py`](backend/tests/test_p1_next_2_guard_at_persist_sites.py), [`test_p2_next_2_cron_persists_coherence_history.py`](backend/tests/test_p2_next_2_cron_persists_coherence_history.py), [`test_p3_next_4_coherence_metrics_surface_breakdown.py`](backend/tests/test_p3_next_4_coherence_metrics_surface_breakdown.py).

### Tests de regresión

- [`backend/tests/test_p1_shopping_recipe_coherence.py`](backend/tests/test_p1_shopping_recipe_coherence.py) — guard E2E presence/absence + magnitudes + knobs (89 casos).
- [`backend/tests/test_p3_a_coherence_multiplier_e2e.py`](backend/tests/test_p3_a_coherence_multiplier_e2e.py) — escala lineal con `multiplier ∈ {1.0, 2.0, 4.0}` × estable/perecedero/pavo (24 casos).
- [`backend/tests/test_p3_b_coherence_block_metrics_cron.py`](backend/tests/test_p3_b_coherence_block_metrics_cron.py) — cron P3-B + invariantes `null_block_set`/`hydration_error` (18 casos).
- [`backend/tests/test_p2_a_shopping_coherence_block_enforcement.py`](backend/tests/test_p2_a_shopping_coherence_block_enforcement.py) — flag consumido por `review_plan_node` (cierre P1-G).
- [`backend/tests/test_p2_2_action_taken_invariant.py`](backend/tests/test_p2_2_action_taken_invariant.py) — `action_taken` jamás `None` tras review.
- [`backend/tests/test_p3_4_pavo_coherence_v3.py`](backend/tests/test_p3_4_pavo_coherence_v3.py) — `canonicalize_pavo` simétrico fresh↔procesado.

### Trade-offs y regresiones históricas

Cinco regresiones históricas que este diseño protege (P1-G mode=block no-op, `_shopping_coherence_*` en `plan_result` NO en `state`, pavo canonicalization P3-4, multiplier asimétrico P3-A, suplementos excluidos P2-4): ver [`runbook_coherence_guard_flow.md`](~/.claude/projects/.../memory/runbook_coherence_guard_flow.md) → sección "Trade-offs y regresiones históricas".

---

## RAG + Dreaming (consolidación de memoria offline)

[P1-DREAMING-1 · 2026-06-13] Sistema híbrido: el RAG de `user_facts` (Cohere embed-v4 + `match_user_facts`) + una capa de "Dreaming" — cron nocturno `_dream_consolidate_facts` que de-duplica facts globalmente (merges soft-delete reversibles), aplica salience + decay, resuelve contradicciones cross-sesión, y sintetiza un `user_memory_profile` (1 fila/usuario, `evidence_fact_ids` FK-verificada anti-confabulación) inyectado al prompt del coach + opcional al generador de planes. **Alergias/condiciones médicas EXENTAS** (floor salience 1.0, nunca auto-merge, fail-secure). Todo OFF por default (`MEALFIT_DREAMING_ENABLED`/`_RETRIEVAL_ENABLED`/`_INJECT_PLAN_ENABLED`) → 5 fases de rollout, F0 neutral (no toca `match_user_facts`). Adaptado a Neon (FK→`user_profiles`, sin RLS/`auth.uid()`, RPC estilo `match_user_facts`). Costo ~$0.0002/usuario (flash + 1 call/usuario + budget cap). Motor SSOT [`backend/dreaming.py`](backend/dreaming.py); doc canónica (knobs/ciclo/fases/alertas/seguridad) [`backend/docs/dreaming_consolidation.md`](backend/docs/dreaming_consolidation.md). Test ancla [`test_p1_dreaming_1_anchors.py`](backend/tests/test_p1_dreaming_1_anchors.py).

---

## Supermercado RD artificial

[P1-SUPERMARKET-DB · 2026-07-02] Tabla `supermarket_products` (Neon): presentaciones comprables de los +200 alimentos verificados (+variantes de marca futuras), navegable en `/supermercado` (landing, link en Footer) y editable ahí mismo con gate admin (Bearer `CRON_SECRET`, mutaciones SOLO vía [`backend/routers/supermarket.py`](backend/routers/supermarket.py) — simétrica a I6). Roadmap: conexión a lista de compras vía `master_food_name` (elegir marca de yogurt/carnes/arroz). Doc canónica (schema/endpoints/seed/roadmap): [`backend/docs/supermarket_db.md`](backend/docs/supermarket_db.md). Test ancla: [`test_p1_supermarket_db.py`](backend/tests/test_p1_supermarket_db.py).

---

## Memoria de días pasados en el chat

[P1-CHAT-PAST-DAYS · 2026-07-27] El coach recuerda los días que ya pasaron por dos vías separadas que NUNCA deben confundirse: un índice barato siempre inyectado de lo que el plan **prescribió** (nombre + slot + kcal, ~200 bytes/día) y el diario multi-día de lo que el usuario **registró** comer, con los días sin registro declarados uno a uno. El detalle caro (ingredientes con gramos + pasos de receta, ~2.5k tokens/día) va bajo demanda por la tool `consultar_dia_del_plan`. Prerequisito estructural: cada día del plan nace con `date` ISO estampada en los 3 sitios de renumeración; los planes viejos degradan a inferencia anclada por `grocery_start_date` (el campo que el shift reescribe a hoy, siguiendo a `days[0]`), con `day_name` como fallback y `cycle_start_date` como último recurso. Motor SSOT [`backend/chat_history_context.py`](backend/chat_history_context.py); doc canónica (causas, knobs, costos, lo que NO resuelve) [`backend/docs/chat_past_days_memory.md`](backend/docs/chat_past_days_memory.md). Test ancla [`test_p1_chat_past_days_memory.py`](backend/tests/test_p1_chat_past_days_memory.py).

---

## Convenciones del repo

- **Knobs operacionales**: env vars `MEALFIT_*` con defaults seguros, registrados en `_KNOBS_REGISTRY` (`graph_orchestrator.py`). Cambios de comportamiento que pueden necesitar revertirse sin redeploy van como knob, no como hardcode.
- **Logging en producción**: [P2-LOGGER-MIGRATION · 2026-05-12] archivos productivos del backend (`graph_orchestrator.py`, `fact_extractor.py`, `memory_manager.py`, `vision_agent.py`, `nutrition_calculator.py`, `db_facts.py`, `app.py`) usan `logger.<level>(...)` — NO `print(...)`. Mapeo emoji → nivel: ❌/🛑/🚨 → `error`, ⚠/🛡 → `warning`, resto → `info`. Excepciones legítimas (CLI subcommand a stdout) requieren marker `# [P2-LOGGER-EXEMPT: <razón>]` en las 3 líneas previas. Test blanket [`test_p2_logger_migration.py`](backend/tests/test_p2_logger_migration.py) escanea con AST y falla si encuentra `print()` sin marker. Whitelist `KNOWN_PRINT_EXEMPT_PATHS` para scripts CLI/scratch/refactors one-shot.
- **Readiness probe granular**: [P3-READY-REASON · 2026-05-12] `GET /ready` devuelve `{status, plan_graph, reason, message}` cuando 503. `reason` formato `build_failed:<ExcType>:<msg>:<n>` permite a orquestadores (k8s, load balancer) dispatchear por tipo de error sin abrir logs. Mensaje truncado a 240 chars para evitar leak de paths/SQL en body público del probe. Implementado vía `is_plan_graph_ready_with_reason() -> tuple[bool, str | None]` ([`graph_orchestrator.py`](backend/graph_orchestrator.py)). Test [`test_p3_ready_reason.py`](backend/tests/test_p3_ready_reason.py).
- **E2E tests (Playwright)**: [P3-E2E-PLAYWRIGHT · 2026-05-12] smoke del golden-path en [`frontend/e2e/golden_path.spec.js`](frontend/e2e/golden_path.spec.js). Regression guards: `pageerror` listener (P0-FRONTEND-ANALYTICS) + 0 requests a `fonts.gstatic.com` (P3-SELF-HOST-FONTS). NO cubre flujo autenticado (follow-up cuando exista staging Supabase). Scripts: `test:e2e` / `test:e2e:install`. Ver [`frontend/e2e/README.md`](frontend/e2e/README.md).
- **UUIDs en endpoints públicos**: [P2-HEALTH-UID-STRIP · 2026-05-12] endpoints health/observabilidad sin auth DEBEN hashear UUIDs via `_hash_uuid_for_public()` ([`routers/system.py`](backend/routers/system.py)) → `hashlib.sha256(value)[:12]` (preserva correlation visual sin enumeración). Si necesitas UUID raw, gatear con `_verify_admin_token`. Test: [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py) sección 1.
- **`datetime.utcnow()` prohibido en producción**: [P3-DEPRECATED-UTCNOW · 2026-05-12] Python 3.12+ emite `DeprecationWarning`; usar `datetime.now(timezone.utc)`. Tests legacy exentos con comment `# naive a propósito`. Test: [`test_p3_prod_audit_6.py`](backend/tests/test_p3_prod_audit_6.py) sección 2.
- **Provider LLM: DeepSeek V4 + router por tier** [P0-DEEPSEEK-MIGRATION · 2026-06-12]: Gemini eliminado por completo. SSOT [`backend/llm_provider.py`](backend/llm_provider.py): `gratis`/guests → `deepseek-v4-flash`; `basic`/`plus`/`ultra` → `deepseek-v4-pro` (**fail-cheap**: duda/fallo de lookup → flash; el reviewer médico risk-tier va a PRO para TODOS los tiers). Key `DEEPSEEK_API_KEY`. Tabla canónica surfaces/knobs/pricing: [`backend/docs/llm_tier_routing.md`](backend/docs/llm_tier_routing.md). Test ancla: [`test_p0_deepseek_migration.py`](backend/tests/test_p0_deepseek_migration.py).
- **Embeddings: Cohere Embed v4** [P1-COHERE-EMBED-V4 · 2026-06-12]: SSOT [`backend/embeddings_provider.py`](backend/embeddings_provider.py), `embed-v4.0` @1536. Claves del diseño: asimetría `input_type` (query→`search_query`, persistido→`search_document`), cache keys versionadas por model_id+purpose, gating por `COHERE_API_KEY` (sin key ⇒ degradación keyword/recency). Detalle completo: [`backend/docs/embeddings_cohere.md`](backend/docs/embeddings_cohere.md). Test ancla: [`test_p1_cohere_embed_v4.py`](backend/tests/test_p1_cohere_embed_v4.py).
- **DB + Auth: 100% Neon (Supabase eliminado por completo)** [P1-NEON-DB-MIGRATION + P1-NEON-AUTH-MIGRATION · 2026-06-12/13]: DATOS en Neon Postgres; AUTH en Neon Auth (Better Auth). Cero dependencia de Supabase (ni paquete, ni cliente, ni env vars). **Datos**: knob `MEALFIT_DB_BACKEND` (`supabase`|`neon`) en [`backend/db_core.py`](backend/db_core.py), fail-loud sin URLs Neon; PostgREST prohibido (backend `execute_sql_*`, frontend → endpoints [`backend/routers/user_data.py`](backend/routers/user_data.py)). **Auth**: el backend valida JWTs EdDSA contra el JWKS de Neon Auth ([`backend/neon_auth.py`](backend/neon_auth.py) `verify_neon_jwt`, algoritmo fijo, fail-secure, preserva P0-AUDIT-1); el frontend usa `@neondatabase/neon-js` con `SupabaseAuthAdapter` ([`frontend/src/supabase.js`](frontend/src/supabase.js), API drop-in). Env: `NEON_AUTH_BASE_URL` (backend) / `VITE_NEON_AUTH_URL` (frontend). Reemplazos app-side: trigger `handle_new_user`→`ensure_user_profile_exists`, pg_cron→APScheduler, RPCs→endpoints, Realtime→refetch/polling. **Usuarios password de Supabase NO migran** (hash distinto) — re-registro/OAuth. Storage de visual_diary pendiente de object storage (vision disabled). Docs: [`backend/docs/neon_db_migration.md`](backend/docs/neon_db_migration.md). Tests ancla: [`test_p1_neon_db_migration.py`](backend/tests/test_p1_neon_db_migration.py), [`test_p1_neon_auth_migration.py`](backend/tests/test_p1_neon_auth_migration.py).
- **Modelos LLM via knob, no hardcoded**: [P3-PREVIEW-MODEL-KNOB · 2026-05-12] callsites en crons/loops productivos leen model ID desde knob `MEALFIT_<FEATURE>_MODEL` via helper `_<feature>_model_name()`. Razón histórica: modelos preview de Google se depreciaban sin aviso (CB row stale 4.4 días, audit 2026-05-11); sigue vigente con DeepSeek (aliases legacy deprecan 2026-07-24). El override per-feature SIEMPRE gana sobre el router por tier — rollback/A-B sin redeploy.
- **DDL en runtime**: prohibido. Toda creación/alteración de tablas o índices vive en `migrations/` (P1-NEW-A índices, P2-NEW-E tablas). [P3-MIGRATION-IDEMPOTENCE-DOC · 2026-05-15] Idempotente obligatorio: `IF NOT EXISTS` en CREATE/ADD COLUMN, `DROP CONSTRAINT IF EXISTS` antes de ADD, `DO $$ RAISE EXCEPTION` sanity. Patrón: p2_next_4 + p3_multiplier_db_check.
- **SSOT de migrations** [P3-MIGRATIONS-SSOT · 2026-05-20]: TODA migration vive en DOS directorios mantenidos sincronizados — `migrations/` (workspace-root, cross-repo) Y `backend/migrations/` (backend repo). Al añadir una nueva migration, copiarla a AMBOS dirs en el mismo commit/push de cada repo. Razón: el workspace-root usa el `.gitignore` que excluye `backend/`+`frontend/` (son repos hermanos con remotes propios), así que necesitas archivos físicos en cada dir para que cada `git push` lleve el cambio. Drift histórico (audit 2026-05-20): 4 files root-only + 1 file backend-only fueron sincronizados; estado actual = 37 archivos idénticos en ambos. Verificación rápida: `diff <(ls migrations) <(ls backend/migrations)` debe retornar vacío.
- **Convención de imports DB** [P3-DB-IMPORTS-FACADE · 2026-05-20]: nuevos call sites de funciones DB deben usar la **fachada** `from db import <funcion>` (ver [`backend/db.py`](backend/db.py)) en lugar de los módulos internos (`from db_plans import ...`, `from db_inventory import ...`, etc.). Razón: `db.py` hace `from db_X import *` de los 6 sub-módulos (`db_core`, `db_profiles`, `db_chat`, `db_plans`, `db_facts`, `db_inventory`) y es el contrato público. Importar el sub-módulo directo (a) acopla el call site a la organización interna actual (si re-organizamos a `db/` paquete, hay que mover 59 imports), (b) eluda los `__all__` controlados, (c) duplica el sentinel de re-export protegido por `test_p3_new_star_imports_audit.py` (P3-NEW-STAR-IMPORTS-AUDIT). Migración: NO grep+replace masivo hoy (59 imports cross-codebase). Política "boy scout": cuando edites un archivo con `from db_<sub> import`, considera migrar ese mismo bloque a `from db import`. Los 5 sub-módulos seguirán siendo SSOT del código real — la fachada es API pública únicamente.
- **Console output frontend** [P3-CONSOLE-DEV-GUARDS · 2026-05-15]: `error/trace/assert` preservados prod (Sentry los captura) — NO DEV-guard. `log/warn/debug/info` dropeados por esbuild `pure:[...]` (P3-FRONTEND-1).
- **Crons**: registrados en `register_plan_chunk_scheduler` ([cron_tasks.py](backend/cron_tasks.py)) — SSOT. Listener `_scheduler_alert_listener` ([app.py](backend/app.py)) escala MISSED/ERROR a `system_alerts`.
- **Tests**: cuando un test parsea source-de-prod con regex, incluir tooltip-anchor en el código fuente para que un renombre falle el test antes de cambiar producción.
- **`TODO`/`TODOS` en comentarios — solo marker de deuda**: [P3-TODOS-NARRATIVE · 2026-05-13] mayúsculas (`TODO`/`FIXME`/`XXX`/`HACK`) reservadas exclusivamente para markers de trabajo pendiente real; el sustantivo español "todo/todos" va en minúscula. Razón: audit 2026-05-12 encontró 243 matches grep, prácticamente todos sustantivo español — ruido. Convención editorial; cero enforcement automático.
- **Memoria persistente**: cada cierre de P-fix se documenta en `~/.claude/projects/.../memory/` con frontmatter `name/description/type` y se referencia en `MEMORY.md`.
- **`_LAST_KNOWN_PFIX`** ([`backend/app.py`](backend/app.py)): marker textual del último P-fix mergeado en HEAD. Cada cierre de P-fix DEBE bumpearlo (formato `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`). `/health/version` lo expone para diagnóstico de deploy rezagado vs. árbol — sin bump, un operador no puede confirmar que su último fix está vivo en producción. Dos tests de regresión enforzan el contrato:
  - [`test_p3_1_last_known_pfix_freshness.py`](backend/tests/test_p3_1_last_known_pfix_freshness.py) — formato (`Pn-...· YYYY-MM-DD`) + floor de fecha (rechaza markers stale).
  - [`test_p2_hist_audit_14_marker_test_link.py`](backend/tests/test_p2_hist_audit_14_marker_test_link.py) — **cross-link**: el slug del marker (`P2-HIST-AUDIT-14` → `p2_hist_audit_14`) DEBE matchear al menos un archivo `tests/test_<slug>*.py`. Cierra el gap "bump cosmético" donde alguien actualizaba el marker sin añadir el test de regresión correspondiente.
- **Tamaño de CLAUDE.md (cap)**: [P3-CLAUDEMD-CAP · 2026-05-14] [`test_p3_claudemd_cap.py`](backend/tests/test_p3_claudemd_cap.py) falla si CLAUDE.md > 52000 chars (knob `MEALFIT_CLAUDE_MD_MAX_CHARS`, clamp [10k, 200k]). CLAUDE.md auto-carga cada turn → chars = tokens proporcionales. **Doc-first**: contenido nuevo nace en `docs/` (tabla con test parser) o `~/.claude/projects/.../memory/` (narrativa/runbook); CLAUDE.md tiene header + 1-line + link. Bump del cap visible en code review — si sube >10% en una sesión, planificar limpieza estructural (pattern 2026-05-14: -46% en 6 fases).
- **Dev local — auto-reload del backend**: setear `UVICORN_RELOAD=1` en `.env` evita el modo de fallo "fix está en HEAD pero binary no lo ve". Python no recarga módulos automáticamente; edits a `constants.py`/`routers/*.py` requieren restart manual SIN reload activo. Default `0` en prod (P2-UVICORN-RELOAD-ENV). Verificación post-restart: `curl /health/version` → comparar `last_known_pfix` vs HEAD. Detalle + SOP: [`runbook_dev_local_setup_2026_05_23.md`](~/.claude/projects/.../memory/runbook_dev_local_setup_2026_05_23.md).
- **SQL forense antes de tocar código**: cuando un bug depende de datos persistidos (`plan_data`, `user_inventory`, `master_ingredients`), ejecuta el SELECT **antes** de teorizar. La sesión 2026-05-23 cerró 3 bugs solo porque el SELECT reveló data corrupta; la del 2026-07-26 refutó 5 hipótesis igual de rápido. Hoy la DB es **Neon**: `load_dotenv()` + `psycopg.connect(os.environ['NEON_DATABASE_URL'])` desde un script (el MCP de Supabase ya no aplica). ⚠️ Fuera de FastAPI hay que **abrir el pool** (`db_core.connection_pool.open()`) o `master_ingredients` sale vacío y mides el vacío, no el sistema. Templates: [`runbook_sql_forensic_sop_2026_05_23.md`](~/.claude/projects/.../memory/runbook_sql_forensic_sop_2026_05_23.md).
- **Soft-fail pattern (HTTP 200 + body flag)** [P3-SWAP-SOFT-FAIL-200 · 2026-05-23]: para endpoints donde el "fallo" es business-as-usual (LLM no convergió, etc), retornar 200 con `operation_failed:true` + `error_code` canónico + `error_message` es preferible a 4xx — evita ruido rojo en DevTools del browser sin perder observability (logger.warning + knob de rollback). NO aplicar a validation/auth/not-found errors (esos siguen 4xx). Criterios + templates backend/frontend + endpoints actuales bajo el pattern: [`runbook_soft_fail_pattern_2026_05_23.md`](~/.claude/projects/.../memory/runbook_soft_fail_pattern_2026_05_23.md).

### Historial-quota-exemption

[P1-AUDIT-3 · 2026-05-10] Los GET endpoints de polling del Historial usan `Depends(get_verified_user_id)` **intencionalmente** (NO `verify_api_quota`):

| Endpoint | Razón |
|---|---|
| `/history-list` ([routers/plans.py](backend/routers/plans.py)) | Polling read-only del listado del Historial. Cero costo LLM. |
| `/lessons-counts` ([routers/plans.py](backend/routers/plans.py)) | Single-roundtrip de conteos por plan. Cero costo LLM. |
| `/history-status-summary` ([routers/plans.py](backend/routers/plans.py)) | Reconciliación de estados `plan_chunk_queue`. Cero costo LLM. |
| `/recalculate-shopping-list` ([routers/plans.py](backend/routers/plans.py)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Recalc derivativo. Cero costo LLM. `Depends(_RECALC_LIMITER)` (20/60s) reemplaza `get_verified_user_id` (RateLimiter retorna `verified_user_id`). |
| `/telemetry/pdf-stale-fallback` ([routers/plans.py](backend/routers/plans.py)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Sink fire-and-forget PDF. Cero costo LLM. `Depends(_PDF_TELEMETRY_LIMITER)` (30/60s). |
| `/shift-plan` ([routers/plans.py](backend/routers/plans.py)) | **[P3-SHIFT-PLAN-QUOTA-EXEMPT · 2026-06-15]** Avance de la ventana rolling de un plan YA generado (mantenimiento, no plan nuevo). Antes `verify_api_quota` + `log_api_usage("shift_plan")` → 402 + crédito extra al llegar al cap, congelando un plan ya pagado. Ahora `Depends(_SHIFT_LIMITER)` (20/60s) y NO cuenta contra el cap. Anti-hammering (P2-LIVE-7) cerrado por el RateLimiter + idempotencia. Test [`test_p3_shift_plan_quota_exempt.py`](backend/tests/test_p3_shift_plan_quota_exempt.py). |
| `/restock` ([routers/plans.py](backend/routers/plans.py)) | **[P1-NEVERA-QUOTA-EXEMPT · 2026-06-24]** "Ya compré la lista" → INSERT/UPDATE `user_inventory`. Cero costo LLM. Antes `verify_api_quota` + `log_api_usage("restock_inventory")` → al cap congelaba la Nevera Inteligente (no se podía meter la compra) Y quemaba crédito de planes (`get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint). Ahora `Depends(_RESTOCK_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_nevera_quota_exempt.py`](backend/tests/test_p1_nevera_quota_exempt.py). |
| `/inventory/consume` ([routers/plans.py](backend/routers/plans.py)) | **[P1-NEVERA-QUOTA-EXEMPT · 2026-06-24]** Vaciar consumidos (`quantity=0`), sub-paso de renovar plan (`useRegeneratePlan.js`). Cero costo LLM. Antes `verify_api_quota` → al cap el 402 abortaba la renovación entera con "Error al sincronizar despensa física" + quemaba crédito. Ahora `Depends(_CONSUME_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_nevera_quota_exempt.py`](backend/tests/test_p1_nevera_quota_exempt.py). |
| `/api/diary/upload` ([routers/diary.py](backend/routers/diary.py)) | **[P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-LUNA · 2026-07-28 → P1-VISION-NO-LOCAL · 2026-07-28]** "Escanear comida" → provider CLOUD pago (Luna) — el provider LOCAL (gemma vía Ollama) fue eliminado por completo, el laptop del owner no podía sostenerlo. `Depends(_VISION_UPLOAD_LIMITER)` (10/60s). El gasto del scan NO va a `api_usage` (`log_api_usage` salió del call site) — vive en `llm_usage_events` (libro de COSTO) vía `log_llm_usage_event(node="vision_scan")`, así un scan nunca quema crédito de plan. Tests [`test_p1_vision_luna.py`](backend/tests/test_p1_vision_luna.py), [`test_p1_vision_no_local.py`](backend/tests/test_p1_vision_no_local.py). |
| `DELETE /api/diary/consumed/{meal_id}` ([routers/diary.py](backend/routers/diary.py)) | **[P1-DIARY-EDITABLE · 2026-07-28]** "Deshacer registro" de una comida mal loggeada → `DELETE consumed_meals` filtrado por `user_id`. Cero costo LLM. Aplicarle `verify_api_quota` sería absurdo: al llegar al cap el usuario no podría CORREGIR un error suyo, y `get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint — borrar una fila quemaría crédito de planes. Ahora `Depends(_DELETE_CONSUMED_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_diary_editable.py`](backend/tests/test_p1_diary_editable.py). |

**Por qué no `verify_api_quota`:** el paywall mensual (gratis=15, basic=50, plus=200) devuelve `HTTP 402` al exceder. Aplicarlo a GETs read-only del Historial impediría al usuario ver su propio historial tras alcanzar el cap (UX inaceptable); aplicarlo a recalc/telemetry sin costo LLM bloquearía cambios legítimos de household + telemetría operacional durante incidentes. Para rate-limiting per-spam, `RateLimiter` per-bucket es la herramienta correcta (NO el paywall). Tests [`test_p1_audit_3_history_quota_exemption.py`](backend/tests/test_p1_audit_3_history_quota_exemption.py) (3 rows originales) + [`test_p3_pdf_polish_4.py`](backend/tests/test_p3_pdf_polish_4.py) (2 rows del bundle PDF) anclan ambas decisiones.

---

## Decisiones de producto (no son gaps técnicos)

Esta sección documenta decisiones de producto que un auditor técnico podría confundir con deuda. La diferencia: un gap técnico se cierra implementando; una decisión de producto se cierra con consenso explícito. Si quieres revertir una de estas decisiones, lee la memoria correspondiente para entender la razón antes de invertir esfuerzo de implementación.

### `i18n: es-DO permanente`

[P3-I18N-DEFERRED · 2026-05-13] El producto es 100% español dominicano (es-DO). UI copy, mensajes de validación, toasts, aria-labels, error handlers — todo hardcoded en literal strings es-DO. **NO hay infraestructura i18n** (cero deps `react-i18next` / `i18next` / `react-intl`) y es intencional.

**Por qué:** mercado RD únicamente, sin roadmap multilocale; `react-i18next` hoy = bundle +30KB + mantenimiento por string + abstracción no-usada ("Don't design for hypothetical future requirements"). El refactor incremental futuro cuesta lo mismo que el scaffold preventivo de hoy.

**Cuándo revisitar:** si producto decide expandir geográficamente (reabrir con `react-i18next` + `src/i18n/locales/` empezando por `components/common/`). Floor de revisión: 2027-01-01. Test [`test_p3_i18n_deferred.py`](backend/tests/test_p3_i18n_deferred.py): si alguien añade `react-i18next`/`i18next` a `package.json` sin actualizar esta sección, falla con copy explicativo.

### `chat-agent safety_settings relajados` (SUPERSEDED por DeepSeek)

[P3-CHAT-SAFETY-OFF-DECISION · 2026-05-20 · superseded P0-DEEPSEEK-MIGRATION 2026-06-12] La decisión aplicaba a los content-filters configurables de Gemini (`DANGEROUS_CONTENT: OFF` + resto `BLOCK_ONLY_HIGH`) por false-positives en charlas de déficit/ayuno. DeepSeek no expone safety_settings client-side — el bloque fue eliminado de [agent.py](backend/agent.py) y la intención (no bloquear conversación nutricional legítima) queda cubierta por el default del provider. Memoria histórica: [`project_p3_chat_safety_off_decision_2026_05_20.md`](~/.claude/projects/.../memory/project_p3_chat_safety_off_decision_2026_05_20.md).

---

## Advisors aceptados (no actuar)

[P3-CLAUDEMD-CAP · movido a docs 2026-07-26] Advisors auditados y declarados **intencionales**
(7 entradas: 3 de security, 9 de performance). Si vuelven a aparecer en un linter, **no actuar** —
la decision esta tomada y la razon esta fija en cada fila.

⚠️ Los emitia el linter de **Supabase**, que ya no corre (migracion completa a Neon, 2026-06-12).
Se conservan porque el razonamiento sigue vigente: por que un indice "sin uso" es load-bearing
(cubre una FK que el advisor no observa, o sirve un SOP de incidente) y por que una funcion es
`SECURITY DEFINER` a proposito.

Tabla canonica (advisor / estado / razon / memoria de cierre) + el pattern `SET search_path = ''`
y el lockdown de DEFINERs: [`backend/docs/advisors_aceptados.md`](backend/docs/advisors_aceptados.md).
Anclajes en migraciones: [`test_p2_whitelist_advisors_anchors_alive.py`](backend/tests/test_p2_whitelist_advisors_anchors_alive.py).

### Sentry sampling driven from env (NO hardcodear `1.0`)

[P1-SENTRY-SAMPLE-COST · 2026-05-12] Backend y frontend leen sample rate desde env var con default seguro 0.1 (10%). Hardcodear `1.0` satura cuota Sentry a escala (≥10k req/día) y throttling dropea errores genuinos. Detalle narrativa + "cuándo subir a 1.0" en [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). Test: [`test_p1_sentry_sample_cost.py`](backend/tests/test_p1_sentry_sample_cost.py).

| Capa | Env var | Default | Clamp |
|---|---|---|---|
| Backend traces | `MEALFIT_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |
| Backend profiling | `MEALFIT_SENTRY_PROFILES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |
| Frontend traces | `VITE_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |

### Security headers en nginx (defensa-en-profundidad en mealfitrd.com)

[P1-VERCEL-SECURITY-HEADERS · 2026-05-12 · migrado a nginx 2026-06-12] Los 6 headers (HSTS, X-Content-Type-Options nosniff, X-Frame-Options DENY, Referrer-Policy, Permissions-Policy, CSP-Report-Only) viven en el snippet `/etc/nginx/snippets/mealfit-security.conf` del VPS Oracle, incluido en el server block HTTPS **y** en cada `location` con `add_header` propio (nginx no hereda `add_header` a un location que define los suyos). Antes vivían en `frontend/vercel.json`, eliminado al migrar de Vercel al VPS (despliegue 2026-06-12). CSP arranca **Report-Only**. Detalle: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md).

### Admin gate en `/api/system/health` (no es público)

[P1-SYSTEM-HEALTH-ADMIN-GATE · 2026-05-12] [`backend/routers/system.py:get_system_health`](backend/routers/system.py) gateado por `_verify_admin_token` (mismo `CRON_SECRET` que admin endpoints). Pre-fix era público y exponía business-intel agregada (nudge rate, abandono, distribución emocional, quality score). Probe público de liveness: `GET /health` y `GET /ready` (solo `{status: ok}`). Detalle: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). Test: [`test_p1_system_health_admin_gate.py`](backend/tests/test_p1_system_health_admin_gate.py).

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

### Ciclo de vida del KV `llm_circuit_breaker:*`

[P3-NEW-E · 2026-05-11] Estado persistente del `LLMCircuitBreaker` ([`graph_orchestrator.py`](backend/graph_orchestrator.py)). Patterns de key en `app_kv_store`: `llm_circuit_breaker` (legacy global) + `llm_circuit_breaker:<model>` (P1-Q3 per-modelo, sufijo `f":{model_name}"` construido en `LLMCircuitBreaker.__init__`). Payload `{failures, last_failure, is_open}`; canonical zero post-reset. Tres vías de reset: `_atomic_reset_db()` (post-success UPSERT), `can_proceed()` runtime auto-expira sin tocar la fila DB → gap "stale", cron `_sweep_stale_llm_circuit_breakers` (P2-NEW-D) reescribe filas stale. Diagrama de transiciones + storage layers + SOPs detallados: [`runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md`](~/.claude/projects/.../memory/runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md). Test ancla: [`test_p3_new_e_cb_kv_lifecycle_doc.py`](backend/tests/test_p3_new_e_cb_kv_lifecycle_doc.py).

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CB_FAILURE_THRESHOLD` | 3 | N fallos consecutivos antes de abrir el breaker |
| `MEALFIT_CB_RESET_TIMEOUT_S` | 30 | Ventana tras `last_failure` tras la cual `can_proceed` retorna True |
| `MEALFIT_CB_LOCAL_HEALTH_TTL_S` | 1.0 | TTL del cache local in-process antes de re-consultar Redis/DB |
| `MEALFIT_CB_KV_STALENESS_HOURS` | 2 | Edad mínima de `last_failure` para que el sweep P2-NEW-D considere stale |
| `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` | 60 | Frecuencia del cron del sweep |

---

---

## Anti-patrones de frontend prohibidos

[P3-NEW-A · 2026-05-11] El cliente (React/Vite) NO debe escribir directo a tablas user-scoped de Supabase via `supabase.from(...).update|delete|upsert(...)`. Cada uno de esos callsites produce el mismo modo de fallo: **lost-update** contra escritores backend legítimos (`_chunk_worker`, crons, otros endpoints). RLS protege IDOR pero NO previene lost-update — ambos escritores son el mismo `user_id`.

La invariante I6 (sección "Invariantes del lifecycle") documenta el contrato. Esta tabla enumera **operaciones prohibidas vs su reemplazo backend**, con cross-link al test blanket [`test_p1_new_a_frontend_no_direct_meal_plans_write.py`](backend/tests/test_p1_new_a_frontend_no_direct_meal_plans_write.py) que enforza el contrato a CI.

### Operaciones prohibidas y sus reemplazos

| Operación cliente prohibida | Reemplazo backend | P-fix de cierre |
|---|---|---|
| `supabase.from('meal_plans').update({plan_data, ...})` | `POST /api/plans/{plan_id}/swap-meal/persist` con `jsonb_set` quirúrgico sobre `{days,<i>,meals,<j>}` | P0-NEW-A · 2026-05-11 |
| `supabase.from('meal_plans').update({plan_data})` inyectando `grocery_start_date`/`cycle_start_date` | `POST /api/plans/{plan_id}/grocery-start-date` con `jsonb_set` idempotente (`IS NULL` per-key) | P0-NEW-B · 2026-05-11 |
| `supabase.from('meal_plans').update({name})` (rename) | `PATCH /api/plans/{plan_id}/name` con `jsonb_set` sobre `name` top-level Y `plan_data.name` atómico | P1-HIST-5 · 2026-05-09 |
| `supabase.from('meal_plans').update({plan_data, name, calories, macros})` (revertir regen rechazado) | `POST /api/plans/{plan_id}/restore-local` con full-overwrite atómico bajo `acquire_meal_plan_advisory_lock(purpose='general')` + `AND user_id = %s` | P1-OPEN-1 · 2026-05-11 |
| `supabase.from('meal_plans').delete()` | `DELETE /api/plans/{plan_id}` (cancel chunks + release locks + audit) | P0-HIST-1 / pre-existente |
| Persistir `expanded_recipe` desde el cliente | `POST /api/plans/recipe/expand` con `update_meal_plan_data(target_plan_id, ..., user_id=user_id)` | P1-HIST-RECIPE-1 · 2026-05-10, persistencia restaurada P1-NEW-7 · 2026-05-11 |
| Restaurar plan desde historial via cliente | `POST /api/plans/restore` (endpoint atómico: 6 columnas + cancel chunks + lock release) | P0-HIST-1 · 2026-05-09 |

### Operaciones permitidas (whitelist documentada)

| Operación cliente permitida | Razón | P-fix de referencia |
|---|---|---|
| `supabase.from('user_inventory').delete().eq(...)` en [`Pantry.jsx`](frontend/src/pages/Pantry.jsx) | Acciones del usuario sobre SU propia despensa (eliminar item / deleteAll). Pantry recalc se invoca tras cada delete vía endpoint backend. | P3-AUDIT-8 · 2026-05-10 |
| `supabase.rpc('increment_inventory_quantity', ...)` | RPC `SECURITY DEFINER` con filtro interno `WHERE user_id = auth.uid()`. Bypasses RLS intencionalmente para soportar increment atómico bajo concurrencia. | P2-4 · 2026-05-10 (advisor WARN aceptado) |

> **[P3-DOC-1 · 2026-05-11]** Eliminada la entry de `supabase.from('meal_plans').insert(...)` desde `Plan.jsx`. La función `savePlanToHistory` que la contenía era dead code (0 callers cross-codebase). El backend ya persiste vía `services._save_plan_and_track_background` post-SSE-completion. La señal `mealfit_history_dirty_at` se movió a `AssessmentContext.jsx::saveGeneratedPlan` (callsite real post-SSE-success). **Cero excepciones whitelisted sobre `meal_plans` desde el frontend** — la invariante I6 ahora aplica sin excepciones para esta tabla.

Si un futuro callsite necesita whitelist explícita (sin migrar al patrón backend), añadir inline el marker `// [P1-NEW-A WHITELIST: <razón ≥1 char>]` en las 30 líneas previas. El test blanket lo respeta. **Tras P1-OPEN-1 (2026-05-11) NO hay whitelists activas** — `restorePlan` legacy migrado a `/api/plans/{plan_id}/restore-local`. El test `test_p1_open_1_restore_local_endpoint::test_p1_new_a_whitelist_removed_from_frontend` enforza el estado cero-whitelist; si alguien añade una nueva, debe primero documentarla aquí y en la tabla de exceptions del test P1-OPEN-1.

### Cómo verificar

```bash
# Escanear frontend en busca de violations no-whitelisted:
pytest backend/tests/test_p1_new_a_frontend_no_direct_meal_plans_write.py -v
```

Test relacionado: [`test_p1_new_a_frontend_no_direct_meal_plans_write.py`](backend/tests/test_p1_new_a_frontend_no_direct_meal_plans_write.py) — bloquea `update|delete|upsert` no-whitelisted; permite `insert`.

---

## Anti-patrones de agent tools prohibidos

[P0-AGENT-1 · 2026-05-11] El nodo LangGraph `execute_tools` ([backend/agent.py](backend/agent.py)) NO debe confiar en el `user_id` que la LLM emite dentro de `tool_args`. Antes de invocar cualquier tool con signature `(user_id: str, ...)` el nodo DEBE force-overridear `tool_args["user_id"]` al valor autenticado del state (`state["user_id"]` o `state["session_id"]` para guests).

**Razón:** la LLM recibe el `user_id` autenticado en plano dentro del system prompt vía `build_tools_instructions(user_id)` ([prompts/chat_agent.py:128, 148](backend/prompts/chat_agent.py)). Eso es **prompt-trustable, NO enforced**. Una entrada adversaria del usuario (mensaje hostil, contenido importado vía `vision_agent`, recetas externas) puede inducir a la LLM a emitir `tool_call` con `user_id` ajeno, abriendo IDOR cross-user sobre `user_inventory`, `consumed_meals`, `user_facts`, `health_profile`, `meal_plans`.

Es la simétrica de las invariantes I2/I6 (filtros server-side `AND user_id = %s` en SQL + endpoints backend que no aceptan user_id arbitrario del cliente) aplicada al chat-agent layer. Defensa-en-profundidad junto con la sanitización P1-Q8/P0-A1 del pipeline de generación.

### Las 12 tools cubiertas

[P2-CHAT-CLEANUP · 2026-05-20 · +P3-MICRO-FOOD-SUGGEST 2026-06-15 · +P1-CHAT-PAST-DAYS 2026-07-27] Tabla canónica completa de las 12 tools de `agent_tools` ([backend/tools.py](backend/tools.py)) cubiertas por el override + descripción de la mutación cross-user que cada una impediría sin el override: [`backend/docs/agent_tools_user_id_table.md`](backend/docs/agent_tools_user_id_table.md). El override es genérico al tope del loop `execute_tools` — cubre TODAS las tools que añadas a `agent_tools` automáticamente, NO requiere update por-tool del nodo.

### Cómo verificar

```bash
# Override estructural + funcional (mockea tool_call con user_id ajeno):
pytest backend/tests/test_p0_agent_1_user_id_override.py -v

# Paridad bidireccional tabla doc ↔ tools.py::agent_tools:
pytest backend/tests/test_p2_chat_cleanup.py -v
```

[`test_p0_agent_1_user_id_override.py`](backend/tests/test_p0_agent_1_user_id_override.py) escanea el cuerpo del loop `for tool_call in last_message.tool_calls:` y exige que el override `tool_args["user_id"] = _trusted_uid` aparezca **antes** de cualquier `if tool_name == "..."` branch o `t.invoke(tool_args)` callsite. [`test_p2_chat_cleanup.py`](backend/tests/test_p2_chat_cleanup.py) enforza que cada tool en `agent_tools` tenga entry en el doc (y viceversa) — falla si añades una tool sin documentarla.

Override emite `WARN [P0-AGENT-1]` con `tool=/llm_user_id=/trusted=` para identificar prompt-injection attempts. Si una tool nueva acepta otra identidad sensitiva (e.g. `session_id`), añadir override análogo + branch en el test funcional. Detalle: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).

---

## Anti-patrones de autenticación prohibidos

[P0-AUDIT-1 · 2026-05-12] `backend/auth.py::get_verified_user_id` es la **única** capa de autenticación del backend porque `SUPABASE_KEY = SERVICE_ROLE` bypassea RLS. Cualquier debilitamiento abre IDOR universal sobre `meal_plans` / `user_inventory` / `consumed_meals` / `user_facts` / `health_profile`.

- **❌ NUNCA**: `base64.urlsafe_b64decode(jwt_payload)` → `return payload["sub"]` sin verificar firma (account takeover universal: atacante construye JWT con `sub=victim_id` + firma random).
- **✓ Único path válido**: `supabase.auth.get_user(token)` (valida firma server-side). Fail-secure en exception → `None` o `HTTPException 403`, NUNCA retornar claim no verificado. Validación offline opcional via `jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")`.
- **[P2-AUTH-ASYNC-SLEEP · 2026-05-12]** `async def get_verified_user_id` + `await asyncio.sleep(0.5)` (NO `time.sleep`) + `await asyncio.to_thread(supabase.auth.get_user, token)` para no bloquear worker thread durante roundtrip Supabase (~50-200ms). FastAPI awaits async deps transparentemente.

Ejemplos de código prohibido completos + vector de ataque + contrato post-fix: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md). Tests: [`test_p0_audit_1_auth_bypass.py`](backend/tests/test_p0_audit_1_auth_bypass.py), [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py).

---

## Anti-patrones de billing y webhook prohibidos

5 invariantes que protegen los únicos surfaces cliente→tier-upgrade y webhook-trigger del sistema. Ejemplos completos + vectores en [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).

- **Billing** (`/api/subscription/verify`, [`routers/billing.py`](backend/routers/billing.py)):
  - **I-Billing-1** [P0-BILLING-1]: `tier` server-derived desde PayPal `plan_id` (env vars `PAYPAL_PLAN_{BASIC,PLUS,ULTRA}_ID`), NO `data.get("tier")` del cliente.
  - **I-Billing-2** [P0-BILLING-2]: fail-secure cuando faltan env vars PayPal en prod (`HTTPException(503)`, NO `success=True`). Knob `MEALFIT_ALLOW_PAYPAL_BYPASS` solo dev.
  - **I-Billing-3** [P1-BILLING-FAIL-LOUD]: PayPal cancel non-success → `_persist_billing_alert` + `HTTPException(409|502)`, NO `logger.warning` (evita doble cobro). Helper `_is_paypal_cancel_idempotent_success` clasifica 200/204/404/422-terminal como success.
- **Webhook**:
  - **I-Webhook-1** [P0-WEBHOOK-1]: `/api/webhooks/process-pending-facts` con `WEBHOOK_SECRET=None AND ENVIRONMENT=production → 503` fail-secure. Set → `hmac.compare_digest` constant-time.
  - **I-Webhook-2** [P2-WEBHOOK-FAIL-SECURE-ALWAYS]: PayPal webhook firma fail-secure SIEMPRE; sandbox NO salta verificación. Knob `MEALFIT_ALLOW_WEBHOOK_UNSIGNED` nunca respetado en prod.

Tests: [`test_p0_billing_1_tier_server_side.py`](backend/tests/test_p0_billing_1_tier_server_side.py), [`test_p0_billing_2_fail_secure.py`](backend/tests/test_p0_billing_2_fail_secure.py), [`test_p1_billing_fail_loud.py`](backend/tests/test_p1_billing_fail_loud.py), [`test_p0_webhook_1_fail_secure.py`](backend/tests/test_p0_webhook_1_fail_secure.py), [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py).

---

## Detección de deploy lag (operacional)

[P0-PROD-1-DEPLOY · 2026-05-12] El cron `_alert_deploy_lag_marker_stale` ([`backend/cron_tasks.py`](backend/cron_tasks.py)) corre cada `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` (default **1h** desde 2026-05-12, antes 24h) y compara `_LAST_KNOWN_PFIX` del binario corriendo vs `app_kv_store.expected_last_known_pfix`. Si divergen → alert `deploy_lag_drift_vs_expected`.

Endpoint admin [`POST /api/system/admin/deploy-lag/check`](backend/routers/system.py) (auth `Bearer <CRON_SECRET>`) invoca el detector inline + retorna `{live_marker, expected_marker, drift, message}` para validación inmediata post-deploy sin esperar al cron.

**SOP operador post-merge**: `git push` → merge → redeploy en el VPS Oracle (deploy-mealfit.ps1) → `curl POST /api/system/admin/deploy-lag/check` (auth Bearer CRON_SECRET) → espera `drift=false`. Si `drift=true` el binario rezagado sigue corriendo (deploy no aplicado, rollback). Update `expected_last_known_pfix` solo tras confirmar `drift=false`. Test: [`test_p0_prod_1_deploy_force_check.py`](backend/tests/test_p0_prod_1_deploy_force_check.py).

### Endpoint público para blackbox monitor externo

[P2-HEALTHZ-DEEP · 2026-05-12] `GET /health/version` ([`backend/app.py`](backend/app.py)) público sin auth, expone 5 keys (`expected_marker`, `drift`, `last_pipeline_metrics_tick_at`, `has_p0_prod_1_gate`, `has_p1_perf_1_cache`) para poller externo. Cierra paradoja "binary roto se vigila a sí mismo". Tabla detallada + SOP UptimeRobot (URL + assertions): [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md) → "Endpoint público `/health/version`". Test: [`test_p2_healthz_deep_extended.py`](backend/tests/test_p2_healthz_deep_extended.py).

### SOP: resolver `deploy_lag_drift_vs_expected`

[P3-CLEANUP · 2026-05-11 · restaurado P1-SCHEDULER-1 2026-05-12] Cuando el cron `_alert_deploy_lag_marker_stale` inserta esta alert: usar el endpoint admin `POST /api/system/admin/deploy-lag/check` (auth `Bearer $CRON_SECRET`) para identificar el delta `{live_marker, expected_marker, drift}`. Pasos detallados (6 fases: identificar → decidir lado → bumpear KV via script SSOT o SQL fallback → cerrar alert → verificar → post-mortem si recurrente) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md) → "SOP: resolver `deploy_lag_drift_vs_expected`".

---

## Política de `system_alerts` resolution

[P2-NEW-3 · 2026-05-10 · reconciliada P2-AUDIT-4 · 2026-05-10] Modelo: **upsert por `alert_key` + `resolved_at` mutable** (alert "vive" mientras `resolved_at IS NULL`). 4 modelos canónicos: **Auto (explicit)** UPDATE explícito, **Auto (implicit)** productor re-emite mientras condición existe, **Handler-driven** endpoint cierra, **Manual** SRE.

**Tabla canónica completa de ~32 `alert_key`** (productor / resolver / modelo) y SOP "Cómo añadir un nuevo alert_key": [`backend/docs/system_alerts_resolution_table.md`](backend/docs/system_alerts_resolution_table.md). SOPs detallados para alerts Manual (`plan_data_corrupted:*`, `deploy_lag_drift_vs_expected` + limpieza one-shot huérfanas) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md). Drift detection bidireccional via [`test_p2_audit_4_alert_keys_documented.py`](backend/tests/test_p2_audit_4_alert_keys_documented.py) (parsea `backend/docs/system_alerts_resolution_table.md` + call sites en `cron_tasks.py`/`db_inventory.py`/`memory_manager.py`/`app.py`/`graph_orchestrator.py`/`routers/billing.py`).

