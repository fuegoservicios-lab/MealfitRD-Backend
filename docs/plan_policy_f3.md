# Fase 3 — Horizon allocator y superficies obedientes (capa V2.3)

[P1-ARQ25-F3-HORIZON · 2026-09-02] Motor SSOT: [`backend/horizon.py`](../horizon.py). Test ancla:
[`tests/test_p1_arq25_f3_horizon.py`](../tests/test_p1_arq25_f3_horizon.py). Fase anterior:
[`plan_policy_f2.md`](plan_policy_f2.md). Roadmap: `docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md` §6.5–6.6, Fase 3.

## Qué entrega la fase

| Entregable (roadmap) | Dónde vive | Cómo se verifica |
|---|---|---|
| Blueprint 7/15/30 (§6.5) persistido con el run | `horizon.build_blueprint` → `plan_generation_runs.blueprint` (+`blueprint_hash`, `allocator_version`), escrito por `routers/plans_generation.py` justo después de preparar las entradas | 13 golden policies: anclas dentro de su banda escalada al horizonte, fronteras = `split_with_absorb` (H2), hash estable |
| Rebanada inmutable por chunk en `pipeline_snapshot` + `input_hash` | `slice_for_chunk` → `form_data["_blueprint_slice"]` del snapshot (chunk 0: `generation_inputs`; chunks 2..N: `_enqueue_remaining_chunks`; legacy sync/SSE: `_horizon_inject`); `input_hash = sha256(huella_formulario:slice_hash)` | `slice_hash` reproducible; `chunk_input_hash` distinto por chunk |
| Validadores de fidelidad que sustituyen a los gates de variedad | `fidelity_issues` (ancla ausente / franja / banda / repetición exacta / ingrediente-en-demasiados-días) + `review_fidelity_gate` en `review_plan_node`, junto a `_variety_repeat_gate_issues` | plan construido DESDE el blueprint ⇒ 0 issues; mutado ⇒ `anchor_missing_day` |
| `update_reason='variety'` → motivo neutral versionado | `renewal.v1` (`RENEWAL_REASON_VERSIONED`); `is_renewal_reason` acepta el alias legado. Superficies §6.1: `graph_orchestrator` (skip de nevera), `prompts/plan_generator.build_pantry_context`, `ai_helpers` (seeder). Swap sin motivo: `default_swap_reason` (`renewal.v1` bajo enforce). Frontend: Configuración → Renovar envía `renewal.v1` | `test_three_surfaces_of_6_1_use_the_neutral_reason` |
| Todas las superficies de §6.6 leen la política del run | ver tabla siguiente | `test_all_surfaces_read_the_policy_through_horizon` (parser sobre los call sites) |
| Ventanas de frescos/congelación en `IngredientDemand` | `stamp_demand_windows` → `plan_data["_ingredient_demand"].windows/freezer_mode/freeze_horizon_days` (postprocess + `get_shopping_list_delta`) | `test_shopping_projection_windows_and_demand_stamp` |
| Listas 7/15/30 como proyección (`plan_jobs`) | `enqueue_shopping_projection_job` → `plan_jobs.job_type='shopping_projection'`, dedup `plan+revisión+política`. Solo bajo `enforce`; la Fase 5 la consume | parser + dedup key |

## Superficies (§6.6) y qué leen

| Superficie | Hook | Qué obedece |
|---|---|---|
| Chunk 0 (cola) | `generation_inputs.build_initial_pipeline_inputs` → `inject_policy_into_pipeline_data` | rebanada + política efectiva + flag `enforce` (recalculado al ejecutar en `run_initial_chunk`) |
| Chunk 0 (legacy sync/SSE) | `routers/plans._horizon_inject` (2 call sites) | ídem |
| Chunks 2..N | `_enqueue_remaining_chunks`: `blueprint_for_plan` (run o reconstruido) → rebanada por chunk en el snapshot + `input_hash`; el worker recalcula `enforce` | ídem |
| Renovación | mismo seeder (`ai_helpers.get_deterministic_variety_prompt`): con `enforce`, proteína del día = familia programada (`apply_slice_to_seeder_pools`), anclas del día como DATO (`out_assignment["anchors_by_day"]`) y bloque 📐 en el prompt; motivo `renewal.v1` ⇒ texto neutral, no «MAYOR VARIEDAD» | banda + anclas |
| Swap individual | `api_swap_meal` → `attach_policy_to_swap_form` (lee `plan_data->'_plan_policy'`) → `agent.swap_meal` añade el bloque 📐 con el ancla de la franja | banda + ancla de la franja |
| Regen de día | `api_regenerate_day` → `attach_policy_to_swap_form(plan_data=…, day_index=…)` por comida | ídem |
| Smart shuffle / shift (degradado) | `cron_tasks`: `rank_days_by_policy` ordena los candidatos por cobertura de anclas | anclas |
| Caché semántica | `semantic_cache_check_node`: bypass (`policy_bypass`) bajo `enforce` — un plan cacheado no conoce la banda | todo |
| Self-critique | `self_critique_node`: `{policy_block}` en el prompt del evaluador (no baja `diversity_score` por repeticiones pedidas) | banda |
| Gates de variedad | `review_fidelity_gate`: rutina ⇒ retira TODOS los gates de repetición de V1; equilibrada ⇒ solo los que hablan de un ancla; explorar ⇒ ninguno. Fruta+salado y fruta repetida (coherencia) se conservan siempre | banda |
| Aprendizaje | `exclude_anchors_from_fatigue` (orquestador y worker): un ancla nunca entra como «fatigada» | anclas |
| Shopping | `stamp_demand_windows` + `enqueue_shopping_projection_job` | ciclo, top-ups, congelación |

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PLAN_POLICY_MODE` | `shadow` en prod (`off` en código) | `enforce` = el blueprint MANDA en todas las superficies. `shadow` = se construye, se persiste y se mide, sin influir |
| `MEALFIT_PLAN_POLICY_ENFORCE_USERS` | `""` | Canary «dueño → test → flip»: con el modo global en `shadow`, estos uuids (coma) corren en `enforce` |
| `MEALFIT_FIDELITY_GATE` | `warn` | `block` = los issues high/medium rechazan el intento (retry con directiva) — NUNCA en el intento final ni fuera de `enforce` |
| `MEALFIT_SHOPPING_PROJECTION_JOBS` | `True` | Kill switch del outbox `shopping_projection` |

Todos se leen en cada llamada (rollback sin redeploy).

## Reglas del allocator (deterministas)

- **Fronteras**: `chunk_boundaries` = `split_with_absorb(total, PLAN_CHUNK_SIZE)` — la MISMA aritmética que la cola. Si cambia una, cambia la otra (H2).
- **Anclas**: `_schedule_days(T, min7, max7)` programa `ceil(min7·T/7)` días (sin superar `floor(max7·T/7)`), repartidos de forma pareja. Sin franja declarada ⇒ vale en cualquier comida del día.
- **Familias de proteína**: tabla por dieta (`_FAMILIES_BY_DIET`) filtrada por alergias/dieta/exclusiones con los helpers de la Fase 2. Pool por banda: rutina 3, equilibrada 5, explorar todas. Los nombres son el SSOT del motor: no se traducen ni se renombran.
- **Límites de repetición** (por 7 días → escalados a la ventana con `ceil`): rutina 7 exactos / 7 días de ingrediente; equilibrada 2 / 5; explorar 1 / 3. `ingredient_days_exceeded` solo se evalúa en **explorar** (arroz diario es normal en rutina/equilibrada) y nunca sobre anclas ni sazón.
- **Ventanas**: principal `[0, T)`; top-ups de frescos cada `fresh_topup_days` cuando `T > topup`; congelación `none→0`, `limited→7`, `full→T` días.

## Medición

`pipeline_metrics.node = 'plan_policy_fidelity'` por revisión (metadata: `codes`, `score`, `mode`, `gate`, `rejected`, `slice_hash`, `policy_hash`). El sello `plan_data["_fidelity_report"]` guarda el detalle del chunk. Consulta base del gate de la fase:

```sql
SELECT metadata->>'mode' AS mode, metadata->>'gate' AS gate,
       COUNT(*) AS n, AVG(confidence) AS score,
       SUM((metadata->>'rejected')::bool::int) AS rejected
FROM pipeline_metrics WHERE node = 'plan_policy_fidelity' AND created_at > NOW() - INTERVAL '7 days'
GROUP BY 1, 2;
```

## Gate de la fase y canary

1. Deploy en `shadow` + `MEALFIT_FIDELITY_GATE=warn`: cada run nace con blueprint, cada chunk con rebanada; la métrica mide sin influir.
2. Canary: `MEALFIT_PLAN_POLICY_ENFORCE_USERS=<uuid del dueño>` → generar un plan y leer `_fidelity_report` + prompt del seeder (log `📐 [P1-ARQ25-F3-HORIZON] seeder obedece la rebanada`).
3. Usuarios de test → `enforce` global. `block` solo cuando la métrica en `warn` muestre `score ≥ 0.9` sostenido.

Gate (roadmap): las 13 golden policies cumplen sus bandas (test); paridad initial/chunk/renew/swap/regen por el mismo módulo (test parser + `fidelity_report` independiente de la superficie); benchmark clínico sin regresión frente a `baseline-v1` — pendiente de correr con `enforce` vivo.

## Fuera de alcance (Fase 4/5)

Formulario progresivo que pida franjas y bandas por ancla (hoy `stapleFoods` ⇒ `min 2 / max 7`, sin franja); consumidor de `plan_jobs.shopping_projection`; explicación de relajaciones en el frontend.
