# Hybrid RAG + Dreaming — consolidación de memoria (P1-DREAMING-1 · 2026-06-13)

Referencia operacional del sistema de "Dreaming": una capa OFFLINE tipo sueño
sobre el RAG existente (`user_facts` + Cohere embed-v4 @1536 + `match_user_facts`).
SSOT del motor: [`backend/dreaming.py`](../dreaming.py). Cron:
[`backend/cron_tasks.py`](../cron_tasks.py) (`_dream_consolidate_facts`). Migración:
[`p1_dreaming_1_salience_profile_2026_06_13.sql`](../supabase/migrations/p1_dreaming_1_salience_profile_2026_06_13.sql).

## 1. Resumen

El RAG online solo ve K vecinos por query; nunca cruza el corpus completo de un
usuario. El Dreaming es un ciclo nocturno que sí lo hace, en **dos capas**:

1. **Consolidación de `user_facts`** — de-duplica preferencias redundantes
   (merge → 1 fact canónico), aplica `salience_score` + decay, y detecta (sin
   resolver) contradicciones cross-sesión.
2. **Reflexión `user_memory_profile`** — sintetiza UNA "memoria semántica de alto
   nivel" por usuario (6-8 frases es-DO, el "modelo del usuario") con evidencia
   FK-verificada anti-confabulación, embebida para retrieval semántico.

**Todo gateado OFF por default.** Con `MEALFIT_DREAMING_ENABLED=false` (default) el
cron hace early-return neutral: cero costo LLM, comportamiento idéntico al de hoy,
rollback sin redeploy. La migración es 100% aditiva (F0-NEUTRAL): NO modifica
`match_user_facts`.

## 2. Knobs

Todos auto-registrados en `_KNOBS_REGISTRY` (P3-NEW-D) vía `_env_*`.

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DREAMING_ENABLED` | `False` | **Master kill-switch.** OFF → cron neutral, cero costo. |
| `MEALFIT_DREAMING_RETRIEVAL_ENABLED` | `False` | Gate de LECTURA: inyecta el user_model al prompt (`_get_user_model_block`) + branch vectorial en `search_deep_memory`. |
| `MEALFIT_DREAMING_INJECT_PLAN_ENABLED` | `False` | Gate de inyección del user_model al pipeline de generación de plan (fase F4). |
| `MEALFIT_DREAMING_MODEL` | `deepseek-v4-flash` | Modelo del ciclo. NUNCA pro (offline, no médico-crítico). Override per-feature gana sobre el router por tier. |
| `MEALFIT_DREAMING_CONSOLIDATION_INTERVAL_HOURS` | `24` | Intervalo del cron + cutoff de staleness para encolar. Clamp `[6, 168]`. |
| `MEALFIT_DREAMING_MAX_USERS_PER_NIGHT` | `200` | Cap de usuarios procesados por corrida. `0` → solo reporta backlog. Clamp `[0, 5000]`. |
| `MEALFIT_DREAMING_MAX_COST_USD_PER_NIGHT` | `2.0` | Budget global diario (cap de seguridad, NO billing exacto). Clamp `[0, 100]`. |
| `MEALFIT_DREAMING_MAX_FACTS_PER_CALL` | `60` | Trunca facts/usuario por LLM call. Clamp `[10, 200]`. |
| `MEALFIT_DREAMING_SALIENCE_DECAY_RATE` | `0.05` | Decay por ciclo de salience de no-clínicos no reforzados. Clamp `[0, 0.5]`. |
| `MEALFIT_DREAMING_CONTRADICTION_ALERTS` | `True` | Emite `dream_contradiction:<user_id>` cuando hay conflicto. |
| `MEALFIT_DREAMING_BATCH` | `50` | Tamaño de batch de encolado. Clamp `[1, 500]`. |
| `MEALFIT_DREAMING_LLM_TIMEOUT_S` | `60.0` | Timeout del LLM call (`max_retries=0`). Clamp `[5, 180]`. |
| `MEALFIT_DREAMING_USD_PER_1K_TOKENS` | `0.0003` | Precio blended flash para el cap de budget. Clamp `[0, 1]`. |
| `MEALFIT_DREAMING_PROMPT_MAX_CHARS` | `1200` | Cap del bloque user_model inyectado al prompt. Clamp `[0, 4000]`. |
| `MEALFIT_DREAMING_USER_MODEL_CACHE_TTL_S` | `300` | TTL del cache in-process del bloque user_model. Clamp `[0, 3600]`. |
| `MEALFIT_DREAMING_BACKLOG_ALERT` | `1000` | Umbral de backlog que dispara `dreaming_backlog_high` (en `cron_tasks.py`). Clamp `[0, 100000]`. |

## 3. El ciclo de Dreaming

**Entrada del cron**: `run_dream_cycle()` (default-registrado en
`register_plan_chunk_scheduler` con job `dream_consolidate_facts`, intervalo
`CONSOLIDATION_INTERVAL_HOURS`).

- **Inputs**: usuarios "dream-dirty" = `>=2` facts activos + nunca consolidados o
  stale por el intervalo (`_enqueue_dirty_users`, `ORDER BY` staleness).
- **Pasos por usuario** (`consolidate_user`):
  1. `acquire_fact_lock(user_id)` — serializa contra el extractor online y otros
     workers del dream. Si falla → `skipped_locked`.
  2. `_get_active_facts`. Si `< 2` → marca consolidado + `skipped_few_facts`.
  3. Gate de budget (`_get_budget_spent >= MAX_COST`) → `budget_exhausted`; gate de
     circuit-breaker del modelo (`LLMCircuitBreaker(model).can_proceed()`) → `breaker_open`.
  4. 1 LLM call con `DreamConsolidationResult` (dedup + contradicciones + síntesis).
  5. **Merges**: `_soft_delete_facts` (reversible) + `_insert_canonical_fact` (con
     embedding `search_document` + salience boost). Solo si `>=2` fuentes reales.
  6. `_apply_salience_maintenance` (floor clínico 1.0 + decay no-clínico).
  7. Contradicciones → `_persist_dream_contradiction_alert` (nunca mutan facts).
  8. Reflexión: `user_model` con `_verify_evidence_fact_ids` → `_upsert_user_memory_profile`.
  9. Marca `last_consolidated_at` + INSERT en `dream_consolidation_log`.
- **Outputs**: telemetría agregada (`enqueued/processed/merges/contradictions/
  profiles/cost_usd/backlog/budget_exhausted`) → `pipeline_metrics` tick +
  alertas; `user_facts`/`user_memory_profile`/`dream_consolidation_log` mutados.

**Idempotencia**: `dream_work_queue` tiene unique partial index
`(user_id) WHERE processed_at IS NULL` → `ON CONFLICT DO NOTHING` dedup en encolado.
`budget_exhausted` NO marca el trabajo procesado (reintento la próxima noche).

**Locking**: doble — (a) `_claim_next_dream_work` usa
`SELECT … FOR UPDATE SKIP LOCKED LIMIT 1` (leader-safe multi-worker, tx CORTA que
NO abarca el LLM call); (b) la exclusión real por-usuario contra el extractor
online la da `acquire_fact_lock(user_id)` durante todo el ciclo.

## 4. Invariantes de seguridad

- **Exención médica**: categorías `alergia` y `condicion_medica` (`CLINICAL_CATEGORIES`)
  JAMÁS se auto-mergean ni decaen. `_soft_delete_facts` y el decay filtran
  `NOT IN ('alergia','condicion_medica')` en SQL (defensa redundante al prompt);
  el floor de salience las fuerza a `1.0` (fail-secure).
- **Evidencia FK-verificada (anti-confabulación)**: el `user_model` solo se persiste
  si sus `evidence_fact_ids` referencian facts reales, activos, del MISMO `user_id`
  (`_verify_evidence_fact_ids`, enforced en runtime — NO prompt-trustable, espíritu
  P0-AGENT-1). 0 evidencia válida → user_model descartado con WARN.
- **Soft-delete reversible**: los merges NUNCA hacen hard DELETE. Marcan
  `is_active=FALSE` + `metadata.consolidated_into`; el snapshot
  (`facts_soft_deleted: [{fact_id, fact_text}]`) queda en `dream_consolidation_log`
  para revertir.
- **Scoping `AND user_id = %s`**: toda query del motor filtra por usuario (I2, anti-IDOR).
- **Budget + circuit-breaker**: cap diario en `app_kv_store`
  (key con fecha = reset natural) + `LLMCircuitBreaker(model)` reusado por-modelo.
  Best-effort: ninguna excepción del ciclo propaga al caller.

## 5. Tablas nuevas

- **`user_facts`** (ALTER aditivo): `salience_score REAL NOT NULL DEFAULT 0.5`
  (CHECK `[0,1]`), `last_consolidated_at TIMESTAMPTZ`, `consolidation_source TEXT`
  (`'dream_merge_canonical'` | `'online'` | NULL). Index parcial
  `idx_user_facts_salience (user_id, salience_score DESC) WHERE is_active`.
- **`user_memory_profile`** (1 fila/usuario, PK `user_id`): `user_model TEXT`,
  `embedding vector(1536)` (NULL → sin branch vectorial), `evidence_fact_ids UUID[]`,
  `source_model`, `facts_synthesized_from`, `is_active`, timestamps. Index HNSW
  `vector_cosine_ops WHERE is_active`.
- **`dream_work_queue`**: cola idempotente. `user_id`, `trigger_reason`
  (`master_summary` | `nightly_sweep` | `manual_admin`), `enqueued_at`,
  `processed_at`, `attempts`, `last_error`. Unique partial pending + index de pickup.
- **`dream_consolidation_log`** (append-only forensics): `facts_in`,
  `merges_applied`, `facts_soft_deleted JSONB` (para revertir),
  `contradictions_detected`, `profile_updated`, `model_id`, `tokens_estimated`,
  `cost_usd`.

## 6. Integración de retrieval

Ambos surfaces gateados por `MEALFIT_DREAMING_RETRIEVAL_ENABLED` (default OFF →
prompt idéntico a hoy). Fail-open: cualquier error → degradación al comportamiento previo.

- **Inyección al prompt de chat** (`memory_manager._get_user_model_block` +
  `build_memory_context`): antepone un bloque `--- MODELO DEL USUARIO (síntesis
  consolidada de su memoria) ---` al `summary_context` (cap `PROMPT_MAX_CHARS`).
  Cacheado in-process por `USER_MODEL_CACHE_TTL_S` para evitar un SELECT por turno.
- **Branch vectorial** (`db_chat.search_deep_memory`): embebe la query
  (`purpose="query"`, asimetría Cohere) y llama la RPC
  `match_user_memory(query_embedding, match_threshold=0.20, match_count=1, p_user_id)`
  ANTES del ILIKE legacy sobre `summary_archive`. Sin `COHERE_API_KEY` o cualquier
  error → solo el ILIKE legacy.

## 7. Fases de rollout

| Fase | Estado | Flags |
|---|---|---|
| **F0 — neutral DDL** | Migración aplicada, sistema idéntico a hoy. | todos OFF |
| **F1 — escritura canary** | Encender el ciclo con cap bajo para validar merges/costos. | `ENABLED=true`, `MAX_USERS_PER_NIGHT=10` |
| **F2 — escritura full** | Consolidación a escala. | `ENABLED=true`, `MAX_USERS_PER_NIGHT` nominal |
| **F3 — lectura A/B** | El user_model llega al chat (prompt + deep memory). | `+ RETRIEVAL_ENABLED=true` |
| **F4 — inyección al plan** | El user_model alimenta la generación de plan. | `+ INJECT_PLAN_ENABLED=true` |

Cada fase es reversible flippeando su knob (sin redeploy). Las escrituras (F1/F2)
son independientes de las lecturas (F3/F4): se puede consolidar sin exponer.

## 8. Alertas (`system_alerts`)

Emitidas vía `_emit_dreaming_alert` (upsert por `alert_key`, `resolved_at` mutable):

| `alert_key` | Disparo | Resolución |
|---|---|---|
| `dreaming_consolidation_failures_burst` | Fallos consecutivos del cron (import o `run_dream_cycle`). | Auto-reset al primer éxito (`_track_cron_consecutive_failure`). |
| `dreaming_budget_exhausted` | El ciclo alcanzó `MAX_COST_USD_PER_NIGHT`. | Re-evaluado la próxima corrida. |
| `dreaming_backlog_high` | `backlog > MEALFIT_DREAMING_BACKLOG_ALERT`. | Auto-resuelta cuando el backlog baja del umbral. |
| `dream_contradiction:<user_id>` | El LLM detecta facts en conflicto (esp. clínicos). | Manual SRE — señal de revisión; NUNCA muta el fact. |

## 9. Adaptación a Neon

La migración está adaptada post P1-NEON-DB/AUTH-MIGRATION:

- **Sin `auth.users`**: las FK apuntan a `public.user_profiles(id) ON DELETE CASCADE`
  (la tabla no existe en Neon).
- **Sin RLS / `auth.uid()`**: el scoping es 100% app-side (`AND user_id = %s` en el
  motor); las RPC NO usan `SECURITY DEFINER` ni `REVOKE` (no hay roles
  `authenticated`/`anon` en Neon).
- **RPC estilo `match_user_facts`**: `match_user_memory` es `LANGUAGE sql` +
  `SET search_path TO 'public','extensions'`, `SECURITY INVOKER` default, filtra
  `p_user_id` internamente. El backend pasa el `p_user_id` autenticado (NUNCA del
  LLM, simétrico a P0-AGENT-1). 1 fila/usuario máx → barato.
