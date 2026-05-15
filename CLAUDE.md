# MealfitRD.IA — guía operacional

Plan nutricional generado por LLM para usuario dominicano. Backend Python/FastAPI + LangGraph + Supabase. Frontend React/Vite. Detalles cronológicos de los ~80 P-fixes activos: ver `~/.claude/projects/c--Users-angel-OneDrive-Escritorio-MealfitRD-IA/memory/MEMORY.md`.

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
| I8 | **DB-level CHECK**: si `plan_data->>'generation_status' = 'complete'` entonces `jsonb_array_length(plan_data->'days') > 0`. Cierra modo de corrupción donde chunk worker T1 marcaba `complete` sin que el merge `plan_data.days = merged_days` persistiera (plan 005c5a99 vivió ~14h en prod con `status=complete + days=0`). Si esta constraint falla en runtime, el bug está aguas arriba en el chunk worker — investigar antes de relaxar. | CHECK `meal_plans_complete_requires_days` en `public.meal_plans` (migración SSOT [`supabase/migrations/p2_next_4_meal_plans_complete_requires_days.sql`](supabase/migrations/p2_next_4_meal_plans_complete_requires_days.sql), P2-NEXT-4). Test parser-based [`test_p2_next_4_meal_plans_complete_requires_days.py`](backend/tests/test_p2_next_4_meal_plans_complete_requires_days.py) ancla la regla + sanity check + idempotencia. |

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

- [`backend/graph_orchestrator.py:6185`](backend/graph_orchestrator.py#L6185) — `assemble_plan_node` (productor del flag).
- [`backend/graph_orchestrator.py:7704`](backend/graph_orchestrator.py#L7704) — `review_plan_node` (consumidor del flag).
- [`backend/shopping_calculator.py:2686`](backend/shopping_calculator.py#L2686) — `run_shopping_coherence_guard`.
- [`backend/shopping_calculator.py`](backend/shopping_calculator.py) — `run_shopping_coherence_guard_and_append_history` (P1-NEXT-2 · 2026-05-11, helper SSOT para surfaces auxiliares).
- [`backend/shopping_calculator.py:2228`](backend/shopping_calculator.py#L2228) — `expected_sum_from_recipes` (lado izquierdo del guard).
- [`backend/cron_tasks.py:676`](backend/cron_tasks.py#L676) — `_shopping_coherence_alert_job` (cron diario).
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

## Convenciones del repo

- **Knobs operacionales**: env vars `MEALFIT_*` con defaults seguros, registrados en `_KNOBS_REGISTRY` (`graph_orchestrator.py`). Cambios de comportamiento que pueden necesitar revertirse sin redeploy van como knob, no como hardcode.
- **Logging en producción**: [P2-LOGGER-MIGRATION · 2026-05-12] archivos productivos del backend (`graph_orchestrator.py`, `fact_extractor.py`, `memory_manager.py`, `vision_agent.py`, `nutrition_calculator.py`, `db_facts.py`, `app.py`) usan `logger.<level>(...)` — NO `print(...)`. Mapeo emoji → nivel: ❌/🛑/🚨 → `error`, ⚠/🛡 → `warning`, resto → `info`. Excepciones legítimas (CLI subcommand a stdout) requieren marker `# [P2-LOGGER-EXEMPT: <razón>]` en las 3 líneas previas. Test blanket [`test_p2_logger_migration.py`](backend/tests/test_p2_logger_migration.py) escanea con AST y falla si encuentra `print()` sin marker. Whitelist `KNOWN_PRINT_EXEMPT_PATHS` para scripts CLI/scratch/refactors one-shot.
- **Readiness probe granular**: [P3-READY-REASON · 2026-05-12] `GET /ready` devuelve `{status, plan_graph, reason, message}` cuando 503. `reason` formato `build_failed:<ExcType>:<msg>:<n>` permite a orquestadores (EasyPanel, k8s) dispatchear por tipo de error sin abrir logs. Mensaje truncado a 240 chars para evitar leak de paths/SQL en body público del probe. Implementado vía `is_plan_graph_ready_with_reason() -> tuple[bool, str | None]` ([`graph_orchestrator.py:10160+`](backend/graph_orchestrator.py#L10160)). Test [`test_p3_ready_reason.py`](backend/tests/test_p3_ready_reason.py).
- **E2E tests (Playwright)**: [P3-E2E-PLAYWRIGHT · 2026-05-12] smoke del golden-path en [`frontend/e2e/golden_path.spec.js`](frontend/e2e/golden_path.spec.js). Regression guards: `pageerror` listener (P0-FRONTEND-ANALYTICS) + 0 requests a `fonts.gstatic.com` (P3-SELF-HOST-FONTS). NO cubre flujo autenticado (follow-up cuando exista staging Supabase). Scripts: `test:e2e` / `test:e2e:install`. Ver [`frontend/e2e/README.md`](frontend/e2e/README.md).
- **UUIDs en endpoints públicos**: [P2-HEALTH-UID-STRIP · 2026-05-12] endpoints health/observabilidad sin auth DEBEN hashear UUIDs via `_hash_uuid_for_public()` ([`routers/system.py`](backend/routers/system.py)) → `hashlib.sha256(value)[:12]` (preserva correlation visual sin enumeración). Si necesitas UUID raw, gatear con `_verify_admin_token`. Test: [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py) sección 1.
- **`datetime.utcnow()` prohibido en producción**: [P3-DEPRECATED-UTCNOW · 2026-05-12] Python 3.12+ emite `DeprecationWarning`; usar `datetime.now(timezone.utc)`. Tests legacy exentos con comment `# naive a propósito`. Test: [`test_p3_prod_audit_6.py`](backend/tests/test_p3_prod_audit_6.py) sección 2.
- **Modelos LLM via knob, no hardcoded**: [P3-PREVIEW-MODEL-KNOB · 2026-05-12] callsites en crons/loops productivos leen model ID desde `os.environ.get("MEALFIT_<FEATURE>_MODEL", "<default>")` via helper `_<feature>_model_name()`. Razón: modelos preview de Google pueden deprecarse sin aviso (CB row stale por `gemini-3.1-pro-preview` durante 4.4 días, audit 2026-05-11). Knob permite swap sin redeploy.
- **DDL en runtime**: prohibido. Toda creación/alteración de tablas o índices vive en `supabase/migrations/` (P1-NEW-A índices, P2-NEW-E tablas).
- **Crons**: registrados en `register_plan_chunk_scheduler` ([cron_tasks.py:793](backend/cron_tasks.py#L793)) — SSOT. Listener `_scheduler_alert_listener` ([app.py:102+](backend/app.py#L102)) escala MISSED/ERROR a `system_alerts`.
- **Tests**: cuando un test parsea source-de-prod con regex, incluir tooltip-anchor en el código fuente para que un renombre falle el test antes de cambiar producción.
- **`TODO`/`TODOS` en comentarios — solo marker de deuda**: [P3-TODOS-NARRATIVE · 2026-05-13] mayúsculas (`TODO`/`FIXME`/`XXX`/`HACK`) reservadas exclusivamente para markers de trabajo pendiente real; el sustantivo español "todo/todos" va en minúscula. Razón: audit 2026-05-12 encontró 243 matches grep, prácticamente todos sustantivo español — ruido. Convención editorial; cero enforcement automático.
- **Memoria persistente**: cada cierre de P-fix se documenta en `~/.claude/projects/.../memory/` con frontmatter `name/description/type` y se referencia en `MEMORY.md`.
- **`_LAST_KNOWN_PFIX`** ([`backend/app.py:32`](backend/app.py#L32)): marker textual del último P-fix mergeado en HEAD. Cada cierre de P-fix DEBE bumpearlo (formato `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`). `/health/version` lo expone para diagnóstico de deploy rezagado vs. árbol — sin bump, un operador no puede confirmar que su último fix está vivo en producción. Dos tests de regresión enforzan el contrato:
  - [`test_p3_1_last_known_pfix_freshness.py`](backend/tests/test_p3_1_last_known_pfix_freshness.py) — formato (`Pn-...· YYYY-MM-DD`) + floor de fecha (rechaza markers stale).
  - [`test_p2_hist_audit_14_marker_test_link.py`](backend/tests/test_p2_hist_audit_14_marker_test_link.py) — **cross-link**: el slug del marker (`P2-HIST-AUDIT-14` → `p2_hist_audit_14`) DEBE matchear al menos un archivo `tests/test_<slug>*.py`. Cierra el gap "bump cosmético" donde alguien actualizaba el marker sin añadir el test de regresión correspondiente.
- **Tamaño de CLAUDE.md (cap)**: [P3-CLAUDEMD-CAP · 2026-05-14] [`test_p3_claudemd_cap.py`](backend/tests/test_p3_claudemd_cap.py) falla si CLAUDE.md > 52000 chars (knob `MEALFIT_CLAUDE_MD_MAX_CHARS`, clamp [10k, 200k]). CLAUDE.md auto-carga cada turn → chars = tokens proporcionales. **Doc-first**: contenido nuevo nace en `docs/` (tabla con test parser) o `~/.claude/projects/.../memory/` (narrativa/runbook); CLAUDE.md tiene header + 1-line + link. Bump del cap visible en code review — si sube >10% en una sesión, planificar limpieza estructural (pattern 2026-05-14: -46% en 6 fases).

### Historial-quota-exemption

[P1-AUDIT-3 · 2026-05-10] Los GET endpoints de polling del Historial usan `Depends(get_verified_user_id)` **intencionalmente** (NO `verify_api_quota`):

| Endpoint | Razón |
|---|---|
| `/history-list` ([routers/plans.py:6226](backend/routers/plans.py#L6226)) | Polling read-only del listado del Historial. Cero costo LLM. |
| `/lessons-counts` ([routers/plans.py:4776](backend/routers/plans.py#L4776)) | Single-roundtrip de conteos por plan. Cero costo LLM. |
| `/history-status-summary` ([routers/plans.py:4888](backend/routers/plans.py#L4888)) | Reconciliación de estados `plan_chunk_queue`. Cero costo LLM. |
| `/recalculate-shopping-list` ([routers/plans.py:3915](backend/routers/plans.py#L3915)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Recalc derivativo al cambiar `householdSize`/`groceryDuration`. Cero costo LLM. Mitigación spam: `_RECALC_LIMITER = RateLimiter(20/60s)`. |
| `/telemetry/pdf-stale-fallback` ([routers/plans.py:4187](backend/routers/plans.py#L4187)) | **[P3-PDF-POLISH-4-C · 2026-05-14]** Sink fire-and-forget desde el handler PDF. Cero costo LLM. Mitigación spam: `_PDF_TELEMETRY_LIMITER = RateLimiter(30/60s)`. |

**Por qué no `verify_api_quota`:** el paywall mensual (gratis=15, basic=50, plus=200) devuelve `HTTP 402` al exceder. Aplicarlo a GETs read-only del Historial impediría al usuario ver su propio historial tras alcanzar el cap (UX inaceptable); aplicarlo a recalc/telemetry sin costo LLM bloquearía cambios legítimos de household + telemetría operacional durante incidentes. Para rate-limiting per-spam, `RateLimiter` per-bucket es la herramienta correcta (NO el paywall). Tests [`test_p1_audit_3_history_quota_exemption.py`](backend/tests/test_p1_audit_3_history_quota_exemption.py) (3 rows originales) + [`test_p3_pdf_polish_4.py`](backend/tests/test_p3_pdf_polish_4.py) (2 rows del bundle PDF) anclan ambas decisiones.

---

## Decisiones de producto (no son gaps técnicos)

Esta sección documenta decisiones de producto que un auditor técnico podría confundir con deuda. La diferencia: un gap técnico se cierra implementando; una decisión de producto se cierra con consenso explícito. Si quieres revertir una de estas decisiones, lee la memoria correspondiente para entender la razón antes de invertir esfuerzo de implementación.

### `i18n: es-DO permanente`

[P3-I18N-DEFERRED · 2026-05-13] El producto es 100% español dominicano (es-DO). UI copy, mensajes de validación, toasts, aria-labels, error handlers — todo hardcoded en literal strings es-DO. **NO hay infraestructura i18n** (cero deps `react-i18next` / `i18next` / `react-intl`) y es intencional.

**Por qué (audit production-readiness 2026-05-12 + decisión 2026-05-13):**
- Mercado objetivo: República Dominicana únicamente, sin roadmap activo de expansión multilocale.
- Añadir `react-i18next` ahora viola la convención del repo ("Don't design for hypothetical future requirements"): introduce bundle overhead (~30KB), deuda de mantenimiento (cada string nuevo debe pasar por el sistema o se vuelve inconsistente), y abstracción no-usada.
- Si en el futuro se decide expandir (Puerto Rico, México, US Latino, EU/PT/IT), el refactor incremental **cuesta lo mismo que el scaffold preventivo de hoy**, pero hoy se evita pagar la maintenance hasta que la decisión sea real.

**Cuándo revisitar:**
- Si alguien del lado de producto decide expandir geográficamente: este P3 se reabre como tarea de implementación con `react-i18next` + estructura `src/i18n/locales/{es,en,...}/<namespace>.json` + migración incremental empezando por `components/common/` y `components/home/`.
- Floor de revisión sugerido: 2027-01-01 (audit anual). Si para entonces sigue siendo es-DO only, mantener decisión.

**Cierre del gap del audit 2026-05-12:** el audit P3-1 flageó "100% español hardcoded" como deuda i18n. La decisión documentada acá cierra el gap como "decisión de producto, no técnico", análoga al patrón "Advisors aceptados" más abajo. Test parser-based [`test_p3_i18n_deferred.py`](backend/tests/test_p3_i18n_deferred.py) ancla la decisión: si alguien añade `react-i18next` / `i18next` a `package.json` sin actualizar esta sección, el test falla con copy explicativo.

---

## Advisors aceptados (no actuar)

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

### Cómo verificar

Cada item está respaldado por `COMMENT ON INDEX` (índices) o `COMMENT ON FUNCTION` (definers) en migración SSOT — el linter ve el COMMENT pero sigue reportando el advisor (es informational, no auto-suprimido). El operador debe leer el comment vía `\d+ <objeto>` o `obj_description(<oid>, 'pg_class')` antes de actuar.

Si Supabase agrega supresión nativa de advisors aceptados en el dashboard, mover esta sección a la UI de Supabase y dejar este bloque como referencia.

### Sentry sampling driven from env (NO hardcodear `1.0`)

[P1-SENTRY-SAMPLE-COST · 2026-05-12] Backend y frontend leen sample rate desde env var con default seguro 0.1 (10%). Hardcodear `1.0` satura cuota Sentry a escala (≥10k req/día) y throttling dropea errores genuinos. Detalle narrativa + "cuándo subir a 1.0" en [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). Test: [`test_p1_sentry_sample_cost.py`](backend/tests/test_p1_sentry_sample_cost.py).

| Capa | Env var | Default | Clamp |
|---|---|---|---|
| Backend traces | `MEALFIT_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |
| Backend profiling | `MEALFIT_SENTRY_PROFILES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |
| Frontend traces | `VITE_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | `[0.0, 1.0]` |

### Vercel security headers (defensa-en-profundidad en mealfitrd.com)

[P1-VERCEL-SECURITY-HEADERS · 2026-05-12] [`frontend/vercel.json`](frontend/vercel.json) declara 6 headers obligatorios para todas las rutas (HSTS, X-Content-Type-Options nosniff, X-Frame-Options DENY, Referrer-Policy, Permissions-Policy, CSP-Report-Only). CSP arranca **Report-Only**; promover a enforced tras 1 semana de observación. Tabla de headers + hosts whitelisteados críticos + modos de fallo que cada uno cierra: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). Test: [`test_p1_vercel_security_headers.py`](backend/tests/test_p1_vercel_security_headers.py).

### Admin gate en `/api/system/health` (no es público)

[P1-SYSTEM-HEALTH-ADMIN-GATE · 2026-05-12] [`backend/routers/system.py:get_system_health`](backend/routers/system.py#L21) gateado por `_verify_admin_token` (mismo `CRON_SECRET` que admin endpoints). Pre-fix era público y exponía business-intel agregada (nudge rate, abandono, distribución emocional, quality score). Probe público de liveness: `GET /health` y `GET /ready` (solo `{status: ok}`). Detalle: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). Test: [`test_p1_system_health_admin_gate.py`](backend/tests/test_p1_system_health_admin_gate.py).

### Pattern: `SET search_path = ''` en functions Postgres

[P3-NEW-2 · 2026-05-10] Patrón canónico para functions nuevas: `SET search_path = ''` + `SECURITY <DEFINER|INVOKER>` explícito. La cadena vacía fuerza qualifier explícito (`public.<obj>`, `auth.<obj>`) y previene shadowing por temp tables (vs `'public'` que es vulnerable). Narrativa "por qué `''` no `'public'`" + ejemplo SQL boilerplate: [`runbook_advisors_operational_subsections.md`](~/.claude/projects/.../memory/runbook_advisors_operational_subsections.md). **Functions ya bajo el pattern:**

| Function | Migración | `search_path` | EXECUTE granted to |
|---|---|---|---|
| `set_meal_plans_updated_at` | `p2_new_1_set_meal_plans_updated_at_search_path.sql` | `''` | trigger (no-direct) |
| `apply_inventory_delta` | `p0_4_apply_inventory_delta_rpc.sql` | `'public'` (acepta — refs qualified) | `service_role` |
| `increment_inventory_quantity` | runtime/historical | `auth, public, extensions` (legacy, ver P2-4 memoria) | `authenticated` + `service_role` (P2-4) |
| `handle_new_user` | [`p1_definer_functions_lockdown_2026_05_12.sql`](supabase/migrations/p1_definer_functions_lockdown_2026_05_12.sql) | `''` (P1-DEFINER-LOCKDOWN) | `service_role` (REVOKE explícito) |
| `get_monthly_plan_count` | mismo | `''` | `service_role` (REVOKE explícito; función huérfana, 0 callsites) |
| `log_unknown_ingredient_rpc` | mismo | `''` | `service_role` (REVOKE explícito; callsite [`db_plans.py:1196`](backend/db_plans.py#L1196)) |

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

[P0-AGENT-1 · 2026-05-11] El nodo LangGraph `execute_tools` ([backend/agent.py:371](backend/agent.py#L371)) NO debe confiar en el `user_id` que la LLM emite dentro de `tool_args`. Antes de invocar cualquier tool con signature `(user_id: str, ...)` el nodo DEBE force-overridear `tool_args["user_id"]` al valor autenticado del state (`state["user_id"]` o `state["session_id"]` para guests).

**Razón:** la LLM recibe el `user_id` autenticado en plano dentro del system prompt vía `build_tools_instructions(user_id)` ([prompts/chat_agent.py:128, 148](backend/prompts/chat_agent.py#L128)). Eso es **prompt-trustable, NO enforced**. Una entrada adversaria del usuario (mensaje hostil, contenido importado vía `vision_agent`, recetas externas) puede inducir a la LLM a emitir `tool_call` con `user_id` ajeno, abriendo IDOR cross-user sobre `user_inventory`, `consumed_meals`, `user_facts`, `health_profile`, `meal_plans`.

Es la simétrica de las invariantes I2/I6 (filtros server-side `AND user_id = %s` en SQL + endpoints backend que no aceptan user_id arbitrario del cliente) aplicada al chat-agent layer. Defensa-en-profundidad junto con la sanitización P1-Q8/P0-A1 del pipeline de generación.

### Las 9 tools cubiertas

Todas las tools de `agent_tools` ([backend/tools.py:827](backend/tools.py#L827)) aceptan `user_id` como primer argumento. El override se aplica una sola vez al tope del loop, **antes** de cualquier branch específico:

| Tool | Mutación cross-user que el override impide |
|---|---|
| `update_form_field` | `update_user_health_profile_atomic` + `delete_user_facts_by_metadata` |
| `generate_new_plan_from_chat` | pipeline completo + `save_new_meal_plan_robust` |
| `log_consumed_meal` | `db_log_consumed_meal` + `deduct_consumed_meal_from_inventory` |
| `modify_single_meal` | `update_meal_plan_data` (full plan_data overwrite) |
| `search_deep_memory` | leak de summaries cross-user |
| `check_shopping_list` | leak de pantry/plan cross-user |
| `check_current_pantry` | leak de pantry cross-user |
| `modify_pantry_inventory` | `add_or_update_inventory_item` + `deduct_consumed_meal_from_inventory` |
| `mark_shopping_list_purchased` | `restock_inventory` |

### Cómo verificar

```bash
pytest backend/tests/test_p0_agent_1_user_id_override.py -v
```

Test parser-based: [`test_p0_agent_1_user_id_override.py`](backend/tests/test_p0_agent_1_user_id_override.py) escanea el cuerpo del loop `for tool_call in last_message.tool_calls:` y exige que el override `tool_args["user_id"] = _trusted_uid` aparezca **antes** de cualquier `if tool_name == "..."` branch o `t.invoke(tool_args)` callsite. Test funcional adicional: mockea un `tool_call` con `user_id` ajeno y valida que el invoker recibe el trusted, no el inyectado.

### Telemetría + si añades tool nueva

Override emite `WARN [P0-AGENT-1]` con `tool=/llm_user_id=/trusted=` para identificar prompt-injection attempts en logs (futura mejora: cron sweep + `agent_user_id_mismatch_burst`). Si añades tool nueva que acepta `user_id`, el override ya la cubre; si acepta otra identidad sensitiva (e.g. `session_id`), añadir override análogo + entry en tabla arriba + branch en test parser-based. Detalle: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).

---

## Anti-patrones de autenticación prohibidos

[P0-AUDIT-1 · 2026-05-12] `backend/auth.py::get_verified_user_id` es la **única** capa de autenticación del backend porque `SUPABASE_KEY = SERVICE_ROLE` bypassea RLS. Cualquier debilitamiento abre IDOR universal sobre `meal_plans` / `user_inventory` / `consumed_meals` / `user_facts` / `health_profile`.

- **❌ NUNCA**: `base64.urlsafe_b64decode(jwt_payload)` → `return payload["sub"]` sin verificar firma (account takeover universal: atacante construye JWT con `sub=victim_id` + firma random).
- **✓ Único path válido**: `supabase.auth.get_user(token)` (valida firma server-side). Fail-secure en exception → `None` o `HTTPException 403`, NUNCA retornar claim no verificado. Validación offline opcional via `jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")`.
- **[P2-AUTH-ASYNC-SLEEP · 2026-05-12]** `async def get_verified_user_id` + `await asyncio.sleep(0.5)` (NO `time.sleep`) + `await asyncio.to_thread(supabase.auth.get_user, token)` para no bloquear worker thread durante roundtrip Supabase (~50-200ms). FastAPI awaits async deps transparentemente.

Ejemplos de código prohibido completos + vector de ataque + contrato post-fix: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md). Tests: [`test_p0_audit_1_auth_bypass.py`](backend/tests/test_p0_audit_1_auth_bypass.py), [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py).

---

## Anti-patrones de billing/paywall prohibidos

[P0-BILLING-1 / P0-BILLING-2 · 2026-05-12] El endpoint `/api/subscription/verify` ([`backend/routers/billing.py`](backend/routers/billing.py)) es la **única** vía por la que el cliente fuerza un upgrade de tier (`gratis` → `basic` / `plus` / `ultra`). Tres invariantes; ejemplos completos de código prohibido + vectores de ataque en [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md).

### I-Billing-1: `tier` server-derived desde PayPal `plan_id` (NO del cliente)

- **❌ NUNCA**: `tier = data.get("tier")` → `UPDATE plan_tier = tier`. Vector: sub real `basic` pasa `tier="ultra"` → upgrade gratuito.
- **✓ Válido**: mapping `plan_id → tier` server-side desde env vars `PAYPAL_PLAN_{BASIC,PLUS,ULTRA}_ID`; extraer `verified_plan_id = sub_data.get("plan_id")` de PayPal. `data.get("tier")` queda solo como `client_hint_tier` para audit.

### I-Billing-2: fail-secure cuando faltan env vars PayPal en producción

- **❌ NUNCA**: `if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET: success = True; continue`. Vector: Easypanel pierde env vars → cualquier `subscription_id` arbitrario hace upgrade.
- **✓ Válido**: gate `if not env_ready and not is_sandbox and not allow_bypass: raise HTTPException(503)`. Knob `MEALFIT_ALLOW_PAYPAL_BYPASS` solo dev; en `ENVIRONMENT=production` el gate lo ignora. Mismo patrón en `/cancel`.

### I-Billing-3: fail-loud cuando PayPal rechaza una cancel

[P1-BILLING-UPGRADE-FAIL-LOUD + P1-BILLING-CANCEL-FAIL-LOUD · 2026-05-12]

- **❌ NUNCA**: cuando `cancel_resp.status_code != 204` solo `logger.warning(...)` y seguir al UPDATE de BD. Resultado: doble cobro (sub vieja + nueva ACTIVE en PayPal) o BD CANCELLED mientras PayPal sigue cobrando.
- **✓ Válido**: helper `_is_paypal_cancel_idempotent_success(status_code, body_text)` clasifica como success: `200`/`204` (ejecutado), `404` (purgada), `422` con `details[].issue ∈ {SUBSCRIPTION_STATUS_INVALID, INVALID_SUBSCRIPTION_STATUS, SUBSCRIPTION_ALREADY_CANCELLED}` (terminal). Cualquier otro status → fail-loud: `_persist_billing_alert` + `raise HTTPException(409)` (verify-upgrade) o `HTTPException(502)` (cancel directo). El UPDATE de BD NO se ejecuta. Alert `billing_old_sub_cancel_failed:<>:<>` o `billing_cancel_failed:<>:<>` con `severity=critical` para reconciliación SRE.

Tests: [`test_p0_billing_1_tier_server_side.py`](backend/tests/test_p0_billing_1_tier_server_side.py), [`test_p0_billing_2_fail_secure.py`](backend/tests/test_p0_billing_2_fail_secure.py), [`test_p1_billing_fail_loud.py`](backend/tests/test_p1_billing_fail_loud.py).

---

## Anti-patrones de webhook prohibidos

[P0-WEBHOOK-1 · 2026-05-12] El endpoint `/api/webhooks/process-pending-facts` ([`backend/app.py`](backend/app.py)) procesa la cola `pending_facts_queue` para un `user_id` dado. Debe rechazar TODA invocación sin `WEBHOOK_SECRET` válido — atacante con UUID enumerado podría forzar `process_pending_queue_sync(victim_id)`.

- **❌ NUNCA**: gate auth bajo `if webhook_secret:` sin `else`. Si env var falta, el check entero se salta → procesa la request.
- **✓ Válido**: `WEBHOOK_SECRET=None AND ENVIRONMENT=production → 503` (fail-secure). Si set, `hmac.compare_digest(token, webhook_secret)` (constant-time).

### I-Webhook-2: PayPal webhook signature check fail-secure SIEMPRE (incluso sandbox)

[P2-WEBHOOK-FAIL-SECURE-ALWAYS · 2026-05-12] El endpoint `/api/webhooks/paypal` ([`routers/billing.py`](backend/routers/billing.py)) verifica firma con `PAYPAL_WEBHOOK_ID` + `PAYPAL_CLIENT_ID` + `PAYPAL_SECRET`.

- **❌ NUNCA**: `if not is_sandbox: raise HTTPException(400)` — sandbox saltaba verificación. Vector: sandbox expuesto a tráfico real (DNS misroute, `.env` prod marcado sandbox) → atacante forja `BILLING.SUBSCRIPTION.SUSPENDED` para downgrade arbitrario.
- **✓ Válido**: `if not is_sandbox or not allow_unsigned: raise HTTPException(503)`. Knob `MEALFIT_ALLOW_WEBHOOK_UNSIGNED` (default `off`) solo se respeta cuando `ENVIRONMENT != production` AND env vars ausentes. En producción NUNCA se respeta.

Ejemplos completos + vectores: [`runbook_security_antipatterns.md`](~/.claude/projects/.../memory/runbook_security_antipatterns.md). Tests: [`test_p0_webhook_1_fail_secure.py`](backend/tests/test_p0_webhook_1_fail_secure.py), [`test_p2_prod_audit_3.py`](backend/tests/test_p2_prod_audit_3.py).

---

## Detección de deploy lag (operacional)

[P0-PROD-1-DEPLOY · 2026-05-12] El cron `_alert_deploy_lag_marker_stale` ([`backend/cron_tasks.py:1222`](backend/cron_tasks.py#L1222)) corre cada `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` (default **1h** desde 2026-05-12, antes 24h) y compara `_LAST_KNOWN_PFIX` del binario corriendo vs `app_kv_store.expected_last_known_pfix`. Si divergen → alert `deploy_lag_drift_vs_expected`.

Endpoint admin [`POST /api/system/admin/deploy-lag/check`](backend/routers/system.py) (auth `Bearer <CRON_SECRET>`) invoca el detector inline + retorna `{live_marker, expected_marker, drift, message}` para validación inmediata post-deploy sin esperar al cron.

**SOP operador post-merge**: `git push` → merge → redeploy EasyPanel → `curl POST /api/system/admin/deploy-lag/check` (auth Bearer CRON_SECRET) → espera `drift=false`. Si `drift=true` el binario rezagado sigue corriendo (Nixpacks cache hit, rollback). Update `expected_last_known_pfix` solo tras confirmar `drift=false`. Test: [`test_p0_prod_1_deploy_force_check.py`](backend/tests/test_p0_prod_1_deploy_force_check.py).

### Endpoint público para blackbox monitor externo

[P2-HEALTHZ-DEEP · 2026-05-12] `GET /health/version` ([`backend/app.py:869`](backend/app.py#L869)) es público (sin auth) y expone, además del `last_known_pfix` original, 5 keys para que un poller externo (UptimeRobot/StatusCake/cronitor) detecte deploy lag sin `CRON_SECRET`:

| Key | Tipo | Significado |
|---|---|---|
| `expected_marker` | `str \| null` | `app_kv_store.expected_last_known_pfix`. `null` si KV unreachable. |
| `drift` | `bool \| null` | `expected_marker != _LAST_KNOWN_PFIX`. `null` si no se pudo leer expected. |
| `last_pipeline_metrics_tick_at` | ISO `str \| null` | `MAX(created_at)` de `_hardfloor_autoheal_tick` (heartbeat binary). |
| `has_p0_prod_1_gate` | `bool` | binary post-P0-PROD-1 (import `_is_guest_metrics_enabled` ok). |
| `has_p1_perf_1_cache` | `bool` | binary post-P1-PERF-1 (`_SCHEDULER_JOBS_WITH_OPEN_ALERTS` existe). |

Lecturas best-effort (exception → `None`/`False` sin fallar endpoint). Cierra la paradoja "binary roto se vigila a sí mismo": el watchdog interno `_alert_pipeline_metrics_silence` solo dispara si el binary tiene la lógica del watchdog; un poller externo detecta el caso sin depender del binary. SOP UptimeRobot (URL + assertions): URL `/health/version`, GET cada 5min, alert si `drift!=false` OR `last_pipeline_metrics_tick_at < NOW()-30min` OR `has_p0_prod_1_gate=false` OR `has_p1_perf_1_cache=false`. Test: [`test_p2_healthz_deep_extended.py`](backend/tests/test_p2_healthz_deep_extended.py).

### SOP: resolver `deploy_lag_drift_vs_expected`

[P3-CLEANUP · 2026-05-11 · restaurado P1-SCHEDULER-1 2026-05-12] Cuando el cron `_alert_deploy_lag_marker_stale` inserta esta alert: usar el endpoint admin `POST /api/system/admin/deploy-lag/check` (auth `Bearer $CRON_SECRET`) para identificar el delta `{live_marker, expected_marker, drift}`. Pasos detallados (6 fases: identificar → decidir lado → bumpear KV via script SSOT o SQL fallback → cerrar alert → verificar → post-mortem si recurrente) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md) → "SOP: resolver `deploy_lag_drift_vs_expected`".

---

## Política de `system_alerts` resolution

[P2-NEW-3 · 2026-05-10 · reconciliada P2-AUDIT-4 · 2026-05-10] Modelo: **upsert por `alert_key` + `resolved_at` mutable** (alert "vive" mientras `resolved_at IS NULL`). 4 modelos canónicos: **Auto (explicit)** UPDATE explícito, **Auto (implicit)** productor re-emite mientras condición existe, **Handler-driven** endpoint cierra, **Manual** SRE.

**Tabla canónica completa de ~32 `alert_key`** (productor / resolver / modelo) y SOP "Cómo añadir un nuevo alert_key": [`backend/docs/system_alerts_resolution_table.md`](backend/docs/system_alerts_resolution_table.md). SOPs detallados para alerts Manual (`plan_data_corrupted:*`, `deploy_lag_drift_vs_expected` + limpieza one-shot huérfanas) en [`runbook_system_alerts_sops_2026_05_11.md`](~/.claude/projects/.../memory/runbook_system_alerts_sops_2026_05_11.md). Drift detection bidireccional via [`test_p2_audit_4_alert_keys_documented.py`](backend/tests/test_p2_audit_4_alert_keys_documented.py) (parsea `backend/docs/system_alerts_resolution_table.md` + call sites en `cron_tasks.py`/`db_inventory.py`/`memory_manager.py`/`app.py`/`graph_orchestrator.py`/`routers/billing.py`).

