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
| I6 | Mutaciones a `plan_data` desde el frontend prohibidas — solo via endpoint backend con `jsonb_set` quirúrgico (NO full overwrite, salvo `restore-local` que es overwrite explícito bajo advisory lock). La ÚNICA escritura directa permitida desde el cliente es el DELETE en `user_inventory` (Pantry) — sobre `meal_plans` no queda ninguna. [P0-TEST-DB-ISOLATION · 2026-07-29] El INSERT de `Plan.jsx:398` que esta fila documentaba ya no existe (0 call sites `from('meal_plans')` en `frontend/src`; solo comentarios históricos en tests). Una excepción documentada que no existe es peor que ninguna: invita a reañadirla citando la doc. | `/swap-meal/persist` (P0-NEW-A), `/grocery-start-date` (P0-NEW-B), `/recipe/expand` (P1-HIST-RECIPE-1), `/{plan_id}/name` (P1-HIST-5), `/{plan_id}/restore-local` (P1-OPEN-1); test blanket `test_p1_new_a_frontend_no_direct_meal_plans_write.py` (P1-NEW-A). **Cero whitelists activas tras P1-OPEN-1.** |
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
| `MEALFIT_GUARD_UNDERSUPPLY_SEVERE` | `False` | `magnitude_undersupply` (compra <50% sin deducción) escala severa; encender tras medir history (si no: dead-letter T2) |

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

[P1-DREAMING-1 · 2026-06-13] Sistema híbrido: RAG de `user_facts` (Cohere embed-v4 + `match_user_facts`) + "Dreaming" — cron nocturno que de-duplica facts globalmente (merges soft-delete reversibles), aplica salience + decay, resuelve contradicciones cross-sesión y sintetiza `user_memory_profile` (`evidence_fact_ids` FK-verificada anti-confabulación) inyectado al coach + opcional al generador de planes. **Alergias/condiciones médicas EXENTAS** (floor salience 1.0, nunca auto-merge, fail-secure). Todo OFF por default, 5 fases de rollout. Adaptado a Neon. Motor SSOT [`backend/dreaming.py`](backend/dreaming.py); doc canónica (knobs/ciclo/fases/alertas/seguridad/costo) [`backend/docs/dreaming_consolidation.md`](backend/docs/dreaming_consolidation.md). Test ancla [`test_p1_dreaming_1_anchors.py`](backend/tests/test_p1_dreaming_1_anchors.py).

---

## Benchmark del landing (matriz clínica del formulario)

[P1-LANDING-BENCH-1 · 2026-08-07] Benchmark cuyo output alimenta las cifras públicas del landing y guía la mejora del motor (generación/swap/día): matriz de 20 perfiles FIEL a los chips del wizard (7 condiciones + Embarazo/Lactancia, 14 medicamentos, 6 alergias, 3 dietas — los harnesses previos usaban texto libre que el wizard ya no emite) + scorers deterministas (backstop clínico, vit K, FS9, ≥5 tomas). 5 modos (`structural`/`live [--provider openai]`/`remote` guest cero-claves/`telemetry`/`score`). Los claims estructurales del frontend viven en `frontend/src/data/systemFacts.js` (espejo de `benchmark.js` para lo contable). Motor SSOT [`backend/landing_benchmarks.py`](backend/landing_benchmarks.py); runner `scripts/landing_benchmark.py`; doc canónica (matriz/métrica→claim/métrica→palanca/hallazgos: `renal` ya NO expresable desde el form) [`backend/docs/landing_benchmarks.md`](backend/docs/landing_benchmarks.md). Test ancla [`test_p1_landing_bench_1_anchors.py`](backend/tests/test_p1_landing_bench_1_anchors.py).

---

## Supermercado RD artificial

[P1-SUPERMARKET-DB · 2026-07-02 · token separado P2-SUPERMARKET-TOKEN-SPLIT 2026-08-14] Tabla `supermarket_products` (Neon): presentaciones comprables de los +200 alimentos verificados (+variantes de marca futuras), navegable en `/supermercado` (landing, link en Footer) y editable ahí mismo con gate admin **PROPIO** (`_verify_supermarket_token`, Bearer `SUPERMARKET_ADMIN_TOKEN`; mutaciones SOLO vía [`backend/routers/supermarket.py`](backend/routers/supermarket.py) — simétrica a I6). **Era `CRON_SECRET`**: el mismo secreto que abre `purge-data` sobre 33 tablas, tecleado en un formulario de una página PÚBLICA. No había vector (cero sinks XSS, `compare_digest`, 60/min) — lo que se cerró es radio de daño y rotación. ⚠️ **El precio, aceptado: son DOS secretos que rotar.** Sin `SUPERMARKET_ADMIN_TOKEN` configurada se acepta el maestro (compatibilidad de rollout); en cuanto existe, el maestro deja de valer aquí. Test [`test_p2_supermarket_token_split.py`](backend/tests/test_p2_supermarket_token_split.py). Roadmap: conexión a lista de compras vía `master_food_name` (elegir marca de yogurt/carnes/arroz). Doc canónica (schema/endpoints/seed/roadmap): [`backend/docs/supermarket_db.md`](backend/docs/supermarket_db.md). Test ancla: [`test_p1_supermarket_db.py`](backend/tests/test_p1_supermarket_db.py).

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
- **Provider LLM: DeepSeek + gpt-5.6 por tier** [P0-DEEPSEEK-MIGRATION · 2026-06-12 · P1-FLASH-PRIMARY · 2026-07-31]: Gemini eliminado. SSOT [`backend/llm_provider.py`](backend/llm_provider.py): day-gen por tier [P1-DAYGEN-TIER-MODEL]: plus/ultra→Luna medium, resto→Luna low; swap y `/regenerate-day` [P1-SWAP-LUNA · 2026-08-05]: Luna fijo, effort por superficie (individual `medium` ~16,5s / día `low` ~8,2s = igual que flash — el día es bucle EN SERIE de 4-5); demás nodos flash. Red post-fallo CROSS-PROVIDER: `gpt-5.6-luna` [P1-NET-LUNA] (sin OPENAI_API_KEY → pro; NO colapsarla a flash). Rollback `MEALFIT_MODEL_PAID_TIER`/`MEALFIT_PRO_MODEL`. Reviewer clínico por tier [P1-REVIEWER-TIER-MODELS + P1-REVIEWER-SOL-HARD · 2026-07-31]: free→Luna, pagados→Terra, plus/ultra difícil (bariátrico/≥2 reglas SSOT)→Sol (fail-safe sin key → flash). Detalle: [`backend/docs/llm_tier_routing.md`](backend/docs/llm_tier_routing.md). Tests: [`test_p1_flash_primary.py`](backend/tests/test_p1_flash_primary.py), [`test_p1_reviewer_tier_models.py`](backend/tests/test_p1_reviewer_tier_models.py).
- **Embeddings: Cohere Embed v4** [P1-COHERE-EMBED-V4 · 2026-06-12]: SSOT [`backend/embeddings_provider.py`](backend/embeddings_provider.py), `embed-v4.0` @1536. Claves del diseño: asimetría `input_type` (query→`search_query`, persistido→`search_document`), cache keys versionadas por model_id+purpose, gating por `COHERE_API_KEY` (sin key ⇒ degradación keyword/recency). Detalle completo: [`backend/docs/embeddings_cohere.md`](backend/docs/embeddings_cohere.md). Test ancla: [`test_p1_cohere_embed_v4.py`](backend/tests/test_p1_cohere_embed_v4.py).
- **DB + Auth: 100% Neon (Supabase eliminado por completo)** [P1-NEON-DB-MIGRATION + P1-NEON-AUTH-MIGRATION · 2026-06-12/13]: DATOS en Neon Postgres; AUTH en Neon Auth (Better Auth). Cero dependencia de Supabase (ni paquete, ni cliente, ni env vars). **Datos**: knob `MEALFIT_DB_BACKEND` (`supabase`|`neon`) en [`backend/db_core.py`](backend/db_core.py), fail-loud sin URLs Neon; PostgREST prohibido (backend `execute_sql_*`, frontend → endpoints [`backend/routers/user_data.py`](backend/routers/user_data.py)). **Auth**: el backend valida JWTs EdDSA contra el JWKS de Neon Auth ([`backend/neon_auth.py`](backend/neon_auth.py) `verify_neon_jwt`, algoritmo fijo, fail-secure, preserva P0-AUDIT-1); el frontend usa `@neondatabase/neon-js` con `SupabaseAuthAdapter` ([`frontend/src/supabase.js`](frontend/src/supabase.js), API drop-in). Env: `NEON_AUTH_BASE_URL` (backend) / `VITE_NEON_AUTH_URL` (frontend). Reemplazos app-side: trigger `handle_new_user`→`ensure_user_profile_exists`, pg_cron→APScheduler, RPCs→endpoints, Realtime→refetch/polling. **Usuarios password de Supabase NO migran** (hash distinto) — re-registro/OAuth. Storage de visual_diary pendiente de object storage (vision disabled). Docs: [`backend/docs/neon_db_migration.md`](backend/docs/neon_db_migration.md). Tests ancla: [`test_p1_neon_db_migration.py`](backend/tests/test_p1_neon_db_migration.py), [`test_p1_neon_auth_migration.py`](backend/tests/test_p1_neon_auth_migration.py).
- **Modelos LLM via knob, no hardcoded**: [P3-PREVIEW-MODEL-KNOB · 2026-05-12] callsites en crons/loops productivos leen model ID desde knob `MEALFIT_<FEATURE>_MODEL` via helper `_<feature>_model_name()`. Razón histórica: modelos preview de Google se depreciaban sin aviso (CB row stale 4.4 días, audit 2026-05-11); sigue vigente con DeepSeek (aliases legacy deprecan 2026-07-24). El override per-feature SIEMPRE gana sobre el router por tier — rollback/A-B sin redeploy.
- **DDL en runtime**: prohibido. Toda creación/alteración de tablas o índices vive en `migrations/` (P1-NEW-A índices, P2-NEW-E tablas). [P3-MIGRATION-IDEMPOTENCE-DOC · 2026-05-15] Idempotente obligatorio: `IF NOT EXISTS` en CREATE/ADD COLUMN, `DROP CONSTRAINT IF EXISTS` antes de ADD, `DO $$ RAISE EXCEPTION` sanity. Patrón: p2_next_4 + p3_multiplier_db_check.
- **SSOT de migrations** [P3-MIGRATIONS-SSOT · 2026-05-20]: TODA migration vive en DOS directorios mantenidos sincronizados — `migrations/` (workspace-root, cross-repo) Y `backend/migrations/` (backend repo). Al añadir una nueva migration, copiarla a AMBOS dirs en el mismo commit/push de cada repo. Razón: el workspace-root usa el `.gitignore` que excluye `backend/`+`frontend/` (son repos hermanos con remotes propios), así que necesitas archivos físicos en cada dir para que cada `git push` lleve el cambio. Drift histórico (audit 2026-05-20): 4 files root-only + 1 file backend-only fueron sincronizados; estado actual = 37 archivos idénticos en ambos. Verificación rápida: `diff <(ls migrations) <(ls backend/migrations)` debe retornar vacío.
- **Convención de imports DB** [P3-DB-IMPORTS-FACADE · 2026-05-20]: nuevos call sites deben usar la **fachada** `from db import <funcion>` (ver [`backend/db.py`](backend/db.py)), no los módulos internos (`from db_plans import ...`, etc.). `db.py` hace `from db_X import *` de los 6 sub-módulos y es el contrato público — importar el sub-módulo directo acopla al call site la organización interna (59 imports a mover si reorganizamos), eluda los `__all__` controlados, y duplica el sentinel protegido por `test_p3_new_star_imports_audit.py`. Sin grep+replace masivo hoy; política "boy scout": al editar un archivo con `from db_<sub> import`, migra ese bloque. Los 5 sub-módulos siguen siendo SSOT — la fachada es API pública únicamente.
- **Console output frontend** [P3-CONSOLE-DEV-GUARDS · 2026-05-15]: `error/trace/assert` preservados prod (Sentry los captura) — NO DEV-guard. `log/warn/debug/info` dropeados por esbuild `pure:[...]` (P3-FRONTEND-1).
- **Crons**: registrados en `register_plan_chunk_scheduler` ([cron_tasks.py](backend/cron_tasks.py)) — SSOT. Listener `_scheduler_alert_listener` ([app.py](backend/app.py)) escala MISSED/ERROR a `system_alerts`.
- **Tests**: cuando un test parsea source-de-prod con regex, incluir tooltip-anchor en el código fuente para que un renombre falle el test antes de cambiar producción.
- **`TODO`/`TODOS` en comentarios — solo marker de deuda**: [P3-TODOS-NARRATIVE · 2026-05-13] mayúsculas (`TODO`/`FIXME`/`XXX`/`HACK`) reservadas exclusivamente para markers de trabajo pendiente real; el sustantivo español "todo/todos" va en minúscula. Razón: audit 2026-05-12 encontró 243 matches grep, prácticamente todos sustantivo español — ruido. Convención editorial; cero enforcement automático.
- **Memoria persistente**: cada cierre de P-fix se documenta en `~/.claude/projects/.../memory/` con frontmatter `name/description/type` y se referencia en `MEMORY.md`.
- **`_LAST_KNOWN_PFIX`** ([`backend/app.py`](backend/app.py)): marker textual del último P-fix mergeado en HEAD. Cada cierre de P-fix DEBE bumpearlo (formato `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`). `/health/version` lo expone para diagnóstico de deploy rezagado vs. árbol — sin bump, un operador no puede confirmar que su último fix está vivo en producción. Dos tests de regresión enforzan el contrato:
  - [`test_p3_1_last_known_pfix_freshness.py`](backend/tests/test_p3_1_last_known_pfix_freshness.py) — formato (`Pn-...· YYYY-MM-DD`) + floor de fecha (rechaza markers stale).
  - [`test_p2_hist_audit_14_marker_test_link.py`](backend/tests/test_p2_hist_audit_14_marker_test_link.py) — **cross-link**: el slug del marker (`P2-HIST-AUDIT-14` → `p2_hist_audit_14`) DEBE matchear al menos un archivo `tests/test_<slug>*.py`. Cierra el gap "bump cosmético" donde alguien actualizaba el marker sin añadir el test de regresión correspondiente.
- **Tamaño de CLAUDE.md (cap)**: [P3-CLAUDEMD-CAP · 2026-05-14 · medido en bytes no chars 2026-07-31] [`test_p3_claudemd_cap.py`](backend/tests/test_p3_claudemd_cap.py) falla si CLAUDE.md excede el cap vigente (knob `MEALFIT_CLAUDE_MD_MAX_CHARS`, clamp [10k, 200k] bytes — ver el test para el valor exacto, cambia con cada bump). **Doc-first**: contenido nuevo nace en `docs/` (tabla + test parser) o memoria (narrativa/runbook); CLAUDE.md tiene header + 1-line + link. Bump visible en code review — si sube >10% en una sesión, limpieza estructural (pattern 2026-05-14: -46% en 6 fases).
- **Dev local — auto-reload del backend**: setear `UVICORN_RELOAD=1` en `.env` evita el modo de fallo "fix está en HEAD pero binary no lo ve". Python no recarga módulos automáticamente; edits a `constants.py`/`routers/*.py` requieren restart manual SIN reload activo. Default `0` en prod (P2-UVICORN-RELOAD-ENV). Verificación post-restart: `curl /health/version` → comparar `last_known_pfix` vs HEAD. Detalle + SOP: [`runbook_dev_local_setup_2026_05_23.md`](~/.claude/projects/.../memory/runbook_dev_local_setup_2026_05_23.md).
- **SQL forense antes de tocar código**: cuando un bug depende de datos persistidos (`plan_data`, `user_inventory`, `master_ingredients`), ejecuta el SELECT **antes** de teorizar — cerró bugs reales en 2026-05-23 y refutó 5 hipótesis en 2026-07-26 más rápido que teorizar. Hoy la DB es **Neon**: `load_dotenv()` + `psycopg.connect(os.environ['NEON_DATABASE_URL'])` desde un script (el MCP de Supabase ya no aplica). ⚠️ Fuera de FastAPI hay que **abrir el pool** (`db_core.connection_pool.open()`) o `master_ingredients` sale vacío y mides el vacío, no el sistema. Templates: [`runbook_sql_forensic_sop_2026_05_23.md`](~/.claude/projects/.../memory/runbook_sql_forensic_sop_2026_05_23.md).
- **Soft-fail pattern (HTTP 200 + body flag)** [P3-SWAP-SOFT-FAIL-200 · 2026-05-23]: para endpoints donde el "fallo" es business-as-usual (LLM no convergió, etc), retornar 200 con `operation_failed:true` + `error_code` canónico + `error_message` es preferible a 4xx — evita ruido rojo en DevTools del browser sin perder observability (logger.warning + knob de rollback). NO aplicar a validation/auth/not-found errors (esos siguen 4xx). Criterios + templates backend/frontend + endpoints actuales bajo el pattern: [`runbook_soft_fail_pattern_2026_05_23.md`](~/.claude/projects/.../memory/runbook_soft_fail_pattern_2026_05_23.md).

### Historial-quota-exemption

[P1-AUDIT-3 · 2026-05-10] Los GET endpoints de polling del Historial usan `Depends(get_verified_user_id)` **intencionalmente** (NO `verify_api_quota`):

| Endpoint | Razón |
|---|---|
| `/history-list` ([routers/plans.py](backend/routers/plans.py)) | Polling read-only del listado del Historial. Cero costo LLM. |
| `/lessons-counts` ([routers/plans.py](backend/routers/plans.py)) | Single-roundtrip de conteos por plan. Cero costo LLM. |
| `/history-status-summary` ([routers/plans.py](backend/routers/plans.py)) | Reconciliación de estados `plan_chunk_queue`. Cero costo LLM. |
| `/recalculate-shopping-list` ([routers/plans.py](backend/routers/plans.py)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Recalc derivativo. Cero costo LLM. `Depends(_RECALC_LIMITER)` (20/60s) reemplaza `get_verified_user_id`. |
| `/telemetry/pdf-stale-fallback` ([routers/plans.py](backend/routers/plans.py)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Sink fire-and-forget PDF. Cero costo LLM. `Depends(_PDF_TELEMETRY_LIMITER)` (30/60s). |
| `/shift-plan` ([routers/plans.py](backend/routers/plans.py)) | **[P3-SHIFT-PLAN-QUOTA-EXEMPT · 2026-06-15]** Avance de la ventana rolling de un plan YA generado (mantenimiento). Antes `verify_api_quota` + `log_api_usage("shift_plan")` → 402 + crédito extra al llegar al cap, congelando un plan ya pagado. Ahora `Depends(_SHIFT_LIMITER)` (20/60s) y NO cuenta contra el cap. Anti-hammering (P2-LIVE-7) vía RateLimiter+idempotencia. Test [`test_p3_shift_plan_quota_exempt.py`](backend/tests/test_p3_shift_plan_quota_exempt.py). |
| `/restock` ([routers/plans.py](backend/routers/plans.py)) | **[P1-NEVERA-QUOTA-EXEMPT · 2026-06-24]** "Ya compré la lista" → INSERT/UPDATE `user_inventory`. Cero costo LLM. Antes `verify_api_quota` + `log_api_usage("restock_inventory")` → al cap congelaba la Nevera Inteligente Y quemaba crédito de planes (`get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint). Ahora `Depends(_RESTOCK_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_nevera_quota_exempt.py`](backend/tests/test_p1_nevera_quota_exempt.py). |
| `/inventory/consume` ([routers/plans.py](backend/routers/plans.py)) | **[P1-NEVERA-QUOTA-EXEMPT · 2026-06-24]** Vaciar consumidos (`quantity=0`), sub-paso de renovar plan (`useRegeneratePlan.js`). Cero costo LLM. Antes `verify_api_quota` → al cap el 402 abortaba la renovación entera + quemaba crédito. Ahora `Depends(_CONSUME_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_nevera_quota_exempt.py`](backend/tests/test_p1_nevera_quota_exempt.py). |
| `/api/diary/upload` ([routers/diary.py](backend/routers/diary.py)) | **[P1-MEAL-SCAN-GEMMA · 2026-07-12 → P1-VISION-LUNA · 2026-07-28 → P1-VISION-NO-LOCAL · 2026-07-28]** "Escanear comida" → provider CLOUD pago (Luna) — el provider LOCAL (gemma vía Ollama) fue eliminado por completo. `Depends(_VISION_UPLOAD_LIMITER)` (10/60s). El gasto del scan NO va a `api_usage` (`log_api_usage` salió del call site) — vive en `llm_usage_events` (libro de COSTO) vía `log_llm_usage_event(node="vision_scan")`, así un scan nunca quema crédito de plan. Tests [`test_p1_vision_luna.py`](backend/tests/test_p1_vision_luna.py), [`test_p1_vision_no_local.py`](backend/tests/test_p1_vision_no_local.py). |
| `POST /api/diary/consumed-from-plan` ([routers/diary.py](backend/routers/diary.py)) | **[P1-EAT-PLAN-MEAL · 2026-08-07]** "Me lo comí" → registra el plato del plan en el diario y descuenta sus ingredientes de la Nevera. Cero costo LLM (SELECT sobre `plan_data` + INSERT + restas). Al cap el usuario no podría registrar lo que come, y cada registro quemaría crédito de PLANES. `Depends(_PLAN_MEAL_LIMITER)` (20/60s). Test [`test_p1_eat_plan_meal.py`](backend/tests/test_p1_eat_plan_meal.py). |
| `DELETE /api/diary/consumed/{meal_id}` ([routers/diary.py](backend/routers/diary.py)) | **[P1-DIARY-EDITABLE · 2026-07-28]** "Deshacer registro" de una comida mal loggeada → `DELETE consumed_meals` filtrado por `user_id`. Cero costo LLM. Aplicarle `verify_api_quota` sería absurdo: al llegar al cap el usuario no podría CORREGIR un error suyo, y `get_monthly_api_usage` cuenta toda fila de `api_usage` sin filtrar endpoint — borrar una fila quemaría crédito de planes. Ahora `Depends(_DELETE_CONSUMED_LIMITER)` (20/60s), NO cuenta contra el cap. Test [`test_p1_diary_editable.py`](backend/tests/test_p1_diary_editable.py). |
| `POST /api/diary/consumed/manual`, `GET /api/diary/foods/frequent`, `POST /api/diary/consumed/repeat` ([routers/diary.py](backend/routers/diary.py)) | **[P1-MANUAL-FOOD-LOG · 2026-08-11]** El componedor del diario: registrar comida buscando en el catálogo (206 alimentos + 60 platos criollos), «lo que más registras» y «repetir». Cero costo LLM (aritmética server-side desde referencias + GROUP BY). Al cap el usuario no podría ANOTAR lo que come y cada registro quemaría crédito de PLANES. `Depends(_MANUAL_MEAL_LIMITER/_FOOD_FREQUENT_LIMITER/_REPEAT_MEAL_LIMITER)` (20-30/60s). De paso `POST /api/diary/consumed` ganó su `_CONSUMED_WRITE_LIMITER` — era el único write del diario sin limitador. Test [`test_p1_manual_food_log.py`](backend/tests/test_p1_manual_food_log.py). |
| `GET/PUT /api/profile/plan-mode`, `GET /api/nutrition/targets` ([routers/user_data.py](backend/routers/user_data.py)) | **[P1-PLAN-MODE · 2026-08-11]** El interruptor del modo (plan↔seguimiento) y las metas del contador. Cero costo LLM (2 UPDATEs + aritmética Mifflin server-side). El PUT es además la puerta de REANUDAR: al cap, un 402 aquí dejaría al usuario ATRAPADO en pausa (no podría volver a encender lo que ya paga), y el GET de targets es el dashboard entero del modo seguimiento. `Depends(_PLAN_MODE_LIMITER)` (15/60s) / `Depends(_TARGETS_LIMITER)` (30/60s). Test [`test_p1_plan_mode.py`](backend/tests/test_p1_plan_mode.py). |

**Por qué no `verify_api_quota`:** el paywall mensual (gratis=10, basic=50, plus=200, ultra=500 — P1-CREDITS-LADDER) devuelve `HTTP 402` al exceder. Aplicarlo a GETs read-only del Historial impediría al usuario ver su propio historial tras alcanzar el cap (UX inaceptable); aplicarlo a recalc/telemetry sin costo LLM bloquearía cambios legítimos de household + telemetría operacional durante incidentes. Para rate-limiting per-spam, `RateLimiter` per-bucket es la herramienta correcta (NO el paywall). Tests [`test_p1_audit_3_history_quota_exemption.py`](backend/tests/test_p1_audit_3_history_quota_exemption.py) (3 rows originales) + [`test_p3_pdf_polish_4.py`](backend/tests/test_p3_pdf_polish_4.py) (2 rows del bundle PDF) anclan ambas decisiones.

---

## Decisiones de producto (no son gaps técnicos)

Esta sección documenta decisiones de producto que un auditor técnico podría confundir con deuda. La diferencia: un gap técnico se cierra implementando; una decisión de producto se cierra con consenso explícito. Si quieres revertir una de estas decisiones, lee la memoria correspondiente para entender la razón antes de invertir esfuerzo de implementación.

### `P1-PLAN-MODE` (modo seguimiento: la app sin generar planes)

[P1-PLAN-MODE · 2026-08-11] El usuario puede usar la app SOLO como contador de macros/diario (estilo MyFitnessPal): paso 0 del wizard (¿plan o contador?, rama corta de 10 pasos cuyos 12 campos saltados quedan AUSENTES — no se inventan) e interruptor en Configuración → Capacidades con el plan ya creado. La pausa es DOS capas: gate SQL en el pickup del chunk worker (`plan_mode='tracking'` en `user_profiles` — LA que detiene el gasto, porque el pickup no lee flags del jsonb) + cancelación de la cola (los 5 estados resucitables, INCLUIDO `pending_user_action`: el recovery cron los revive a las 12h). Orden flag-first en ambas direcciones. El plan pausado conserva `plan_data` con snapshot `_paused_prev_generation_status` (guard I8: jamás restaurar `complete` con days=[]). Motor SSOT [`backend/plan_mode.py`](backend/plan_mode.py); knob `MEALFIT_PLAN_MODE_SWITCH`; ventana de reanudación `MEALFIT_PLAN_PAUSE_MAX_RESUME_DAYS`. Nav del dashboard por modo: SSOT `frontend/src/config/dashboardNav.js`. Tests [`test_p1_plan_mode.py`](backend/tests/test_p1_plan_mode.py) (backend, 22) + `frontend/src/__tests__/PlanMode.contract.test.jsx` (15).

### `P1-IOS-NATIVE-SHELL` (la app nativa NO vende: refleja)

[P1-IOS-NATIVE-SHELL · 2026-08-21] Wrapper Capacitor (`frontend/ios/`, `com.bioboros.app`). Apple 3.1.1: pago sólo web; en nativo **no existe comercio** (precios, «Mejorar plan», PayPal, landing). **UN gate** `frontend/src/config/platform.js`. `viewPlansLabel={null}` NO bastaba (el `??` lo rellenaba). Spec [`ios-native-shell-design.md`](docs/superpowers/specs/2026-08-21-ios-native-shell-design.md); test [`test_p1_ios_native_shell.py`](backend/tests/test_p1_ios_native_shell.py).

### `P1-VIEWPORT-ZOOM-LOCK` (el pinch-zoom bloqueado, a sabiendas)

[P1-VIEWPORT-ZOOM-LOCK · 2026-07-09 · doc 2026-08-15] `user-scalable=no` + `maximum-scale=1`. **Decisión del dueño, YA revertida una vez** (`P2-A11Y-VIEWPORT-ZOOM` lo quitó por a11y y se revirtió: feel de app nativa). Trade-off WCAG 1.4.4 aceptado; la vía real es la escala de fuente del SO (`text-size-adjust: 100%`). **Lighthouse lo reporta en CADA auditoría** y es lo que deja la nota en 91: no lo "arregles". Test [`test_p1_viewport_zoom_lock.py`](backend/tests/test_p1_viewport_zoom_lock.py).

### `P3-I18N-DEFERRED` (i18n: es-DO permanente) — SUPERSEDED por `P1-I18N-DASHBOARD`

[P1-I18N-DASHBOARD · 2026-08-15] El dashboard se lee en 5 idiomas (es-DO base, en-US, pt-BR, fr-FR, it-IT); selector en Configuración → Idioma. Se traduce la **interfaz**; NO el contenido (plan/recetas/coach los escribe el LLM en español), los legales (traducir un contrato genera obligaciones) ni **los nombres de alimentos, JAMÁS**: son el SSOT del motor — `pantry_names_match`, guard de coherencia y backstop de alergias resuelven por esos nombres exactos, así que traducir «Pollo» rompe las tres, dos en silencio.

Motor propio, cero deps. **La clave ES el texto español**: es-DO no lleva catálogo (0 bytes para la base actual) y lo no traducido cae a español, no a la clave — no crees `es-DO.json`. Cambiar el copy huérfana su traducción **en silencio**; lo paga `npm run i18n:check`, y borrarlo desarma la única defensa. Doc: [`backend/docs/i18n_dashboard.md`](backend/docs/i18n_dashboard.md). Test [`test_p1_i18n_dashboard.py`](backend/tests/test_p1_i18n_dashboard.py); `test_p3_i18n_deferred.py` reconvertido: «no añadas una librería encima del motor propio».

### `chat-agent safety_settings relajados` (SUPERSEDED por DeepSeek)

[P3-CHAT-SAFETY-OFF-DECISION · 2026-05-20 · superseded P0-DEEPSEEK-MIGRATION 2026-06-12] La decisión aplicaba a los content-filters configurables de Gemini (`DANGEROUS_CONTENT: OFF` + resto `BLOCK_ONLY_HIGH`) por false-positives en charlas de déficit/ayuno. DeepSeek no expone safety_settings client-side — el bloque fue eliminado de [agent.py](backend/agent.py) y la intención (no bloquear conversación nutricional legítima) queda cubierta por el default del provider. Memoria histórica: [`project_p3_chat_safety_off_decision_2026_05_20.md`](~/.claude/projects/.../memory/project_p3_chat_safety_off_decision_2026_05_20.md).

### `P3-LANDING-DARK-ONLY` (landing oscuro fijo) — SUPERSEDED

[P2-PAPER-NO-INK · 2026-08-01] Las 6 rutas de marketing pasan de oscuro a blanco y negro estricto («papel técnico») vía un 4º `data-theme="paper"`; dashboard intacto. `P3-LANDING-DARK-ONLY` (landing sin configuración de apariencia, siempre oscuro) queda **SUPERSEDED**: el dueño la anuló a sabiendas del precedente en `feedback_marketing_design_minimalist_scientific.md` (dos reverts 2026-07-02 pedían NO monocromatizar los centerpieces). Spec + estado real (nada mergeado aún, benchmark pendiente de decisión): [`docs/superpowers/specs/2026-08-01-landing-papel-tecnico-design.md`](docs/superpowers/specs/2026-08-01-landing-papel-tecnico-design.md).

### `P1-HERO-DEDUP-ACCENT` (el acento del landing) — supersede parcial de `P2-PAPER-NO-INK`

[P1-HERO-DEDUP-ACCENT · 2026-08-09] Nace `--pa-accent: #C1200E` (5,83:1 contra el papel en AMBAS direcciones), la única tinta de color del sistema. **Marca LA CIFRA de un SSOT — condición necesaria, no suficiente**: 2 call sites (cotas de la Fig. 00, numeral de créditos), nunca en un control ni en el CTA (el botón ya es la única tinta sólida de la pantalla: teñirlo REBAJA la jerarquía). Cierra además 7 duplicaciones hero↔header↔franja, incluida la franja de 5 celdas de la que 4 reformulaban el párrafo de encima. El titular perdió su `<br />` fijo: `text-wrap: balance` es load-bearing (sin él vuelve la línea huérfana). Spec: [`docs/superpowers/specs/2026-08-09-hero-landing-llamativo-design.md`](docs/superpowers/specs/2026-08-09-hero-landing-llamativo-design.md). Test [`test_p1_hero_dedup_accent.py`](backend/tests/test_p1_hero_dedup_accent.py).

---

## Advisors aceptados (no actuar)

[P3-CLAUDEMD-CAP · movido a docs 2026-07-26] Advisors auditados y declarados **intencionales** (7: 3 de security, 9 de performance) — si reaparecen en un linter, **no actuar**, la razón está fija por fila. Los emitía el linter de **Supabase** (ya no corre, migración a Neon 2026-06-12); el razonamiento sigue vigente (un índice "sin uso" cubre una FK que el advisor no ve, o una función es `SECURITY DEFINER` a propósito). Tabla canónica + pattern `SET search_path = ''` + lockdown de DEFINERs: [`backend/docs/advisors_aceptados.md`](backend/docs/advisors_aceptados.md). Anclajes: [`test_p2_whitelist_advisors_anchors_alive.py`](backend/tests/test_p2_whitelist_advisors_anchors_alive.py).

### Tres decisiones cuyo detalle vive SOLO en el doc

[P1-CHECKOUT-CREDITS-TRUTH · 2026-08-22, pasada doc-first] `P1-SENTRY-SAMPLE-COST` (sample rate por env var, nunca `1.0` hardcodeado), `P1-VERCEL-SECURITY-HEADERS` (los 6 headers en nginx, repetidos en CADA `location` porque nginx no hereda) y `P1-SYSTEM-HEALTH-ADMIN-GATE` (`/api/system/health` gateado por `_verify_admin_token`; el liveness público es `/health` y `/ready`). Las tres estaban aquí como muñones que ya duplicaban el doc desde el 19-ago. Contenido íntegro + tests: [`backend/docs/advisors_aceptados.md`](backend/docs/advisors_aceptados.md).

### Pattern: `SET search_path = ''` en functions Postgres

[P3-NEW-2 · 2026-05-10] Functions nuevas: `SET search_path = ''` + `SECURITY <DEFINER|INVOKER>` explícito — la cadena vacía fuerza qualifier explícito (`public.<obj>`) y previene shadowing por temp tables (`'public'` NO es equivalente: es vulnerable). **[P1-DEFINER-LOCKDOWN · 2026-05-12]** toda function `SECURITY DEFINER` que acepte `user_id`/`p_user_id` sin validar contra `auth.uid()` DEBE llevar `REVOKE EXECUTE ... FROM PUBLIC, anon, authenticated` en su migración SSOT — un GRANT por error abriría IDOR cross-user. Inventario de las 6 functions ya bajo el pattern + boilerplate: [`backend/docs/advisors_aceptados.md`](backend/docs/advisors_aceptados.md). Test: [`test_p1_definer_lockdown_migration.py`](backend/tests/test_p1_definer_lockdown_migration.py).

### Ciclo de vida del KV `llm_circuit_breaker:*`

[P3-NEW-E · 2026-05-11] Estado persistente del `LLMCircuitBreaker` ([`graph_orchestrator.py`](backend/graph_orchestrator.py)). Key en `app_kv_store`: `llm_circuit_breaker` (legacy global) + `llm_circuit_breaker:<model>` (P1-Q3 per-modelo). Payload `{failures, last_failure, is_open}`; canonical zero post-reset. Tres vías de reset: `_atomic_reset_db()` (post-success UPSERT), `can_proceed()` auto-expira runtime sin tocar la fila DB → gap "stale", cron `_sweep_stale_llm_circuit_breakers` (P2-NEW-D) reescribe filas stale. Diagrama + SOPs: [`runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md`](~/.claude/projects/.../memory/runbook_llm_circuit_breaker_kv_lifecycle_2026_05_12.md). Test ancla: [`test_p3_new_e_cb_kv_lifecycle_doc.py`](backend/tests/test_p3_new_e_cb_kv_lifecycle_doc.py).

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CB_FAILURE_THRESHOLD` | 3 | N fallos consecutivos antes de abrir el breaker |
| `MEALFIT_CB_RESET_TIMEOUT_S` | 30 | Ventana tras `last_failure` tras la cual `can_proceed` retorna True |
| `MEALFIT_CB_LOCAL_HEALTH_TTL_S` | 1.0 | TTL del cache local in-process antes de re-consultar Redis/DB |
| `MEALFIT_CB_KV_STALENESS_HOURS` | 2 | Edad mínima de `last_failure` para que el sweep P2-NEW-D considere stale |
| `MEALFIT_CB_KV_STALENESS_SWEEP_INTERVAL_MIN` | 60 | Frecuencia del cron del sweep |

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

### Las 13 tools cubiertas

[P2-CHAT-CLEANUP · 2026-05-20 · +P3-MICRO-FOOD-SUGGEST 2026-06-15 · +P1-CHAT-PAST-DAYS 2026-07-27 · +P1-CHAT-DIARY-CORRECT 2026-07-29] Tabla canónica completa de las 13 tools de `agent_tools` ([backend/tools.py](backend/tools.py)) cubiertas por el override + descripción de la mutación cross-user que cada una impediría sin el override: [`backend/docs/agent_tools_user_id_table.md`](backend/docs/agent_tools_user_id_table.md). El override es genérico al tope del loop `execute_tools` — cubre TODAS las tools que añadas a `agent_tools` automáticamente, NO requiere update por-tool del nodo.

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

## El path degradado necesita su propio backstop

[P0-DEGRADED-SAFETY-SCAN · 2026-07-31] Smart Shuffle + Edge Recipes ([`cron_tasks.py`](backend/cron_tasks.py)) arman días sin LLM y **bypasean `assemble_plan_node`** → ni review, ni allergen/diet guard. **No borres `_sieve_catalog_for_safety` creyendo que `_get_fast_filtered_catalogs` ya cubre eso**: al filtro aún se le escapan plurales (Bulgur/Pistachos) y el tamiz usa `clinical_backstop_for_meal`. Knob `MEALFIT_DEGRADED_SAFETY_SCAN`. Test [`test_p0_degraded_safety_scan.py`](backend/tests/test_p0_degraded_safety_scan.py).

[P1-DIET-CANON-SSOT] `dietType` canonicaliza SOLO en **`constants.canonicalize_diet_type`**: eran 3 tablas a mano, drifearon, y la del filtro olvidó `'vegetariana'`/`'vegana'` — servía Pollo a vegetarianas. No escribas una 4ª. Test [`test_p1_diet_canon_ssot.py`](backend/tests/test_p1_diet_canon_ssot.py).

[P1-CHUNK-OFFSET-REBASE] El shift reescribe el ANCLA del plan a hoy (`grocery_start_date` → snapshot `_plan_start_date`) y `execute_after` de cada chunk es `ancla + days_offset`. **Si mueves el ancla, mueve los offsets**: dejarlos quietos atrasa el relleno por exactamente los días archivados y el usuario se queda sin plan (3/3 planes vivos lo estaban, 2026-08-07). La aritmética es SSOT en `constants.rebase_pending_chunk_offsets` / `plan_chunk_offset_moves`; el ejecutor SQL `_rebase_pending_chunk_offsets_sql` lo llaman las DOS ramas del shift (HTTP y cron) y va ANTES de sus returns tempranos — después nace inerte justo para los planes `partial`. Knob `MEALFIT_CHUNK_OFFSET_REBASE`. Test test_p1_chunk_offset_rebase.py.

[P1-CHUNK-EXECUTE-CEILING · 2026-08-16] El rebase mueve `execute_after` por el MISMO delta que el offset (relativo, para preservar hora local y adelantos de `safety_margin`) y por eso **preserva también el error previo para siempre**: nadie compara el par contra el ancla. De los 3 planes vivos con cola, los 2 reanclados alguna vez estaban en `ancla + offset + 1` — el bloque corría el día DESPUÉS de empezar el tramo que cubre, dejando al usuario sin menú ese día y otra vez al siguiente bloque; el tercero, nunca reanclado, exacto. **La diferencia era UNA columna: `updated_at`.** `constants.chunk_execute_after_ceiling` acota por arriba con la medianoche local del primer día cubierto. `LEAST` va DENTRO de `GREATEST`: el suelo de NOW() manda, invertirlo programa al pasado y todos los vencidos salen a la vez. Sin snapshot ⇒ `None` ⇒ conducta previa. Test test_p1_chunk_execute_ceiling.py.

[P1-CHUNK-REBASE-PAUSED] La cadena del rebase incluye TAMBIÉN los `pending_user_action` (el recovery los resucita SIN recalcular offsets): dejarlos fuera reparte su tramo al siguiente y dos generaciones escriben los MISMOS días (f380821a 2026-08-08). El banner «plan incompleto» del Dashboard solo acusa con cola CONFIRMADA muerta [P1-DASH-CORRUPTED-VS-PAUSED]. Test test_p1_chunk_rebase_paused.py.

[P0-ALLERGEN-VOCAB-I18N + ola de países · 2026-08-21] Auditoría con el flip ya vivo: 107 gaps. **El catálogo de F2 era INERTE para la generación** y **la costura (a) estaba mal diagnosticada** (no es léxico DO-tuned: es un espejo sin dos ramas). Plan, gaps abiertos y tests de cada cierre: [`docs/superpowers/plans/2026-08-20-paises-produccion-gaps.md`](docs/superpowers/plans/2026-08-20-paises-produccion-gaps.md).

[P1-COUNTRY-SYSTEM-F0/F1/F2 · 2026-08-16/18] Sistema de 6 países (DO nativo + ES/US/MX/PR/CO beta), **flip ejecutado 2026-08-18**: el motor deja de forzar lo criollo en beta y el catálogo llega a 347 filas. **`constants.canonicalize_country` y `country_for_form_data` son los ÚNICOS SSOT** — no escribas otra tabla (la lección de P1-DIET-CANON-SSOT). Knobs `MEALFIT_COUNTRY_SYSTEM`, `MEALFIT_COUNTRY_COLDSTART_SEGMENT` (False), `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`. Doc canónica con el RUNBOOK DEL FLIP: backend/docs/country_system_f1.md. Spec: docs/superpowers/specs/2026-08-16-sistema-paises-design.md. Tests test_p1_country_system_f0/f1/f2.py.

[P1-FIRST-PURCHASE-PAUSE · 2026-08-16] La autonomía de `initial_plan` (nevera vacía no pausa: la lista NACE del plan) cede **UNA vez por plan**: lista entregada + JAMÁS marcó compra ⇒ pausa suave `awaiting_first_purchase`; el recovery la genera solo a las 12h en flexible. Vive DENTRO de `_pantry_gate_waiver_reason` (SSOT — ninguna guarda decide sola) vía `_first_purchase_pause_applies`; **sin hechos ⇒ autonomía intacta** y **sin lista entregada JAMÁS pausa** — esa condición separa refinar el waiver de resucitar el interbloqueo del cold start. Marker `_first_purchase_pause_at` (jsonb_set + user_id). Knob `MEALFIT_FIRST_PURCHASE_PAUSE`. Test test_p1_first_purchase_pause.py.

[P1-CULINARY-CONTRACT] Coherencia culinaria determinista: metadata en master_ingredients + scan V1/V2/V3 en review/finalize/degradado (warn). Doc: backend/docs/culinary_coherence.md. Test test_p1_culinary_contract.py.

[P2-BACKEND-SUPERMARKET-CACHE] La caché del catálogo se invalida ANTES del `await` de la escritura, así que un lector puede colarse y repoblarla con filas pre-escritura. **Mover la invalidación a después NO cierra la carrera** — deja la misma ventana desplazada. Lo que la cierra es el contador de generación: `_catalog_generation()` antes de leer, `_publish_catalog_cache(..., gen)` descarta si cambió. Test test_p2_backend_supermarket_cache.py.

[P1-HYDRATE-DERIVED-FIELDS · 2026-08-16] El merge de `hydrateLatestPlan` (poll 25s / wake / focus) era una lista blanca de 4 campos: adoptaba `days` y dejaba lo DERIVADO de esos días en el valor viejo — micronutrientes congelados hasta refrescar la página (refrescar lo curaba porque `restoreSessionData` adopta el plan entero), y el estado congelado sobrevivía a la navegación porque el merge se persiste en `mealfit_plan`. SSOT `CAMPOS_DERIVADOS_DEL_SERVIDOR` (`AssessmentContext.jsx`): eran TRES merges con TRES listas, dos ya divergidas y la tercera inexistente. **Adopta si viene, NUNCA borra si falta** — al revés que sus dos hermanos, que sí borran porque van anclados a un persist tras el cual el frontend recalcula; aquí el disparo es arbitrario y `/swap-meal/persist` vacía a propósito las 4 `aggregated_shopping_list*`. **No toques `_planMicroSig` ni la caché de micros**: la caché existe para cuando el report FALTA, así que una firma de contenido es circular, y de ella cuelgan 5-7 descartes de banners. Test `AssessmentContext.p1_hydrate_derived_fields.test.js`.

[P1-DASH-GENERATING-HONESTY · 2026-08-16] «Se llenará en unos minutos» salía con `in_flight_count > 0`, que INCLUYE chunks dormidos con `execute_after` a días vista: la pantalla prometía minutos para el martes y el usuario lo leía como congelado. Es la misma mentira que el Historial cerró en mayo (`P3-HIST-CHUNK-SCHEDULED`) y que este Dashboard nunca heredó — el desglose existía SOLO en `/history-list`. Ahora `/chunk-status` expone `scheduled_count`/`running_now_count` (sin prefijo `chunk_`, para que los asserts del guard de History sigan hablando de SU endpoint). Cuatro trampas: **no** copiar el `WHERE user_id` (rompe el binding de `(plan_id,)` en cada tick), **no** son partición de `in_flight_count` (un `processing` con `execute_after` futuro cae fuera de ambos — por eso sigue en el payload como respaldo), van en el dict INCONDICIONAL (dentro de `_upcoming_payload` los gatearía un knob que no los gobierna), y el icono solo gira con trabajo real. Test test_p1_dash_generating_honesty.py.

Reglas anti-refactor del **landing y el apex** (8: service worker diferido, observabilidad por HOST, preload gateado, SSOT de precios/sitemap, dieta de `lucide`, `@sentry` fuera del entry, precache por marcador de paquete) — movidas a [`backend/docs/landing_apex_antipatterns.md`](backend/docs/landing_apex_antipatterns.md) porque vivían bajo un título sobre el motor de planes. Plan de producción del landing (25 gaps): [`docs/superpowers/specs/2026-08-14-landing-produccion-design.md`](docs/superpowers/specs/2026-08-14-landing-produccion-design.md).

[P1-CULINARY-METADATA-BETA · 2026-08-19] Las 141 filas beta del 2026-08-17 nacieron sin `prep_methods`: cobertura 100%→59%, capa 1 en fail-open **con los tests en verde** (parser-based: ninguno mira el DATO) ⇒ el ancla es un **CHECK en DB** (patrón I8). **El orden es load-bearing**: overrides ANTES de los defaults, o el `IS NULL` no casa y los curados quedan crudos. Test test_p1_culinary_metadata_beta.py.

[P1-BEDCA-DEPROXY-ES + P1-YOGURT-NATURAL · 2026-08-19] 47 de 347 filas comparten `fdc_id`: uno sustituía a SIETE embutidos (Sobrasada: 595 kcal, no 296) y otro da **HTTP 404**: nada re-valida la procedencia. BEDCA: `<type level="3f"/>` autocerrado, **energía en kJ**. Auditar ids DUPLICADOS no ve el ÚNICO mal apuntado (Lomo embuchado 110→321). Doc: backend/docs/catalog_provenance_audit.md.

[P0-SHOPPING-CYCLE-DAYS · 2026-08-22] La lista se agregaba desde `plan_data["days"]` — la ventana VIVA, que el shift ENCOGE — y cada recálculo posterior la reconstruía más corta y la **sobrescribía**: 48 alimentos → **25**, el conjunto exacto del último día superviviente; la nevera nació como su espejo (UNA proteína) y el chunk murió en el gate. **El generador NO tuvo la culpa** (48 vs mediana de flota 46). `shopping_source_days` es SSOT y lo usan LOS DOS lados: el guard tomaba su referencia del MISMO `days` encogido, así que la divergencia **se cancelaba** — mutilar la lista MEJORABA su métrica (31→6). Knob `MEALFIT_SHOPPING_SOURCE_INCLUDES_ARCHIVED`. Doc: backend/docs/shopping_list_cycle_days.md. Test test_p0_shopping_cycle_days.py.

[P1-PANTRY-NAME-RESOLUTION] La identidad de una fila de la Nevera se decide SOLO en `constants.pantry_names_match` (case/acentos/cantidad/plural, por token completo). **No la reimplementes sobre `GLOBAL_REVERSE_MAP`**: ese mapa colapsa `pechuga`→`pollo` a propósito, así que comerte una pechuga descontaría del muslo. Los 4 call sites resolvían por igualdad exacta y `"2 huevos"` contra la fila `Huevo` devolvía **éxito sin descontar, sin fila en `failed_inventory_deductions` y sin alerta**. Doc: backend/docs/pantry_name_resolution.md. Test test_p1_pantry_name_resolution.py.

[P1-BILLING-ORPHAN-RECOVERY + P1-CHECKOUT-CREDITS-TRUTH · 2026-08-22] Auditoría de PayPal: la infra está bien y **jamás se ha completado un pago** (13 subs `APPROVAL_PENDING` purgadas). `/verify` era el ÚNICO camino a `user_profiles` y lo dispara el NAVEGADOR: pestaña cerrada = PayPal cobra y nadie se entera (los webhooks filtran por `paypal_subscription_id`, que sigue NULL). El checkout estampa `custom_id` y `ACTIVATED` adopta al huérfano, **pero el TIER sale del `plan_id`** (I-Billing-1: el `custom_id` dice A QUIÉN, jamás QUÉ) y **SOLO en `ACTIVATED`** — en `PAYMENT.SALE.COMPLETED` 0 filas es lo NORMAL y alertaría en cada renovación. Doc: backend/docs/paypal_audit_2026_08_22.md (verificado / 2 falsos positivos / lo abierto). Test test_p1_billing_orphan_recovery.py.

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

[P2-HEALTHZ-DEEP · 2026-05-12] `GET /health/version` ([`backend/app.py`](backend/app.py)) público sin auth, expone 5 keys (`expected_marker`, `drift`, `last_pipeline_metrics_tick_at`, `has_p0_prod_1_gate`, `has_p1_perf_1_cache`) para poller externo. Cierra paradoja "binary roto se vigila a sí mismo". SOP UptimeRobot: [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md) → "Endpoint público `/health/version`". Test: [`test_p2_healthz_deep_extended.py`](backend/tests/test_p2_healthz_deep_extended.py).

### SOP: resolver `deploy_lag_drift_vs_expected`

[P3-CLEANUP · 2026-05-11 · restaurado P1-SCHEDULER-1 2026-05-12] Cuando el cron `_alert_deploy_lag_marker_stale` inserta esta alert: usar `POST /api/system/admin/deploy-lag/check` (auth `Bearer $CRON_SECRET`) para el delta `{live_marker, expected_marker, drift}`. 6 fases (identificar → decidir lado → bumpear KV → cerrar alert → verificar → post-mortem) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md) → "SOP: resolver `deploy_lag_drift_vs_expected`".

---

## Política de `system_alerts` resolution

[P2-NEW-3 · 2026-05-10 · reconciliada P2-AUDIT-4 · 2026-05-10] Modelo: **upsert por `alert_key` + `resolved_at` mutable** (alert "vive" mientras `resolved_at IS NULL`). 4 modelos canónicos: **Auto (explicit)** UPDATE explícito, **Auto (implicit)** productor re-emite mientras condición existe, **Handler-driven** endpoint cierra, **Manual** SRE.

**Tabla canónica de ~32 `alert_key`** (productor / resolver / modelo) y SOP "Cómo añadir un nuevo alert_key": [`backend/docs/system_alerts_resolution_table.md`](backend/docs/system_alerts_resolution_table.md). SOPs para alerts Manual (`plan_data_corrupted:*`, `deploy_lag_drift_vs_expected` + limpieza one-shot huérfanas) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md). Drift bidireccional via [`test_p2_audit_4_alert_keys_documented.py`](backend/tests/test_p2_audit_4_alert_keys_documented.py) (parsea `backend/docs/system_alerts_resolution_table.md` + call sites en `cron_tasks.py`/`db_inventory.py`/`memory_manager.py`/`app.py`/`graph_orchestrator.py`/`routers/billing.py`).

