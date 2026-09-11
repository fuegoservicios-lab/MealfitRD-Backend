# F5 — Guards inertes en `backend/` (auditoría de solo lectura)

**Alcance:** 116 módulos de producción (~11.8 MB de Python). Fuera del alcance: `tests/`, `scripts/`, `scratch/`, `migrations/`,
`venv*`, `test_venv`, `node_modules`, `infra/`. **Método:** AST (`ast.walk`) + tokenización (`tokenize`, no regex sobre texto crudo,
para no contar comentarios ni docstrings) + verificación cruzada con `grep -rnw` sobre backend **y** `frontend/src`. Ningún comando escribió en disco ni tocó la base de datos. No se ejecutó
`pytest`.

**Titular:** el hallazgo más caro no es código muerto: es un **gate clínico que producción cree encendido y que no existe**
(`MEALFIT_HARDEN_SAMEDAY_PROTEIN`, §2.1), seguido de un **bypass de inventario documentado en comentarios que nunca se invoca**
(`_is_inventory_live_degraded`, §1).

**Leyenda de disposiciones:** **borrar** · **cablear** (conectarlo a un consumidor real) · **documentar** (dejarlo, pero declarar su
estado) · **needs-owner** (hace falta una decisión de producto antes de tocarlo).

---

## 1. Funciones de producción sin llamadores fuera de tests

84 funciones caen en el conjunto muerto (0 referencias reales, o referenciadas **solo** desde otra función muerta). Excluidos los
handlers de framework (`@router.*`, `@app.*`, `field_validator`, callbacks de logging). Top 40:

| # | Función — call sites | Nota | Disposición |
|---|---|---|---|
| 1 | `cron_tasks._is_inventory_live_degraded` (L9274) — 0 call sites (tests: 2) | el comentario de `cron_tasks.py:26939` promete «Bypass automático» que nunca corre | **cablear** o corregir el comentario |
| 2 | `cron_tasks._refill_emergency_backup_plan` (L17707) — 0 call sites (tests: 0) | citada como viva en `db_profiles.py:457` y `graph_orchestrator.py:794/44882` | **needs-owner** |
| 3 | `auth.session_cookie_within_absolute_cap` (L227) — 0 call sites (tests: 2) | guard de seguridad: acota la vida de una cookie robada pese al refresh deslizante | ~~cablear~~ **FALSO POSITIVO** (verificado 2026-09-11, lote 5): `_decode_session_cookie` (L221-223) rechaza todo `iat` ausente, futuro o fuera del cap en CADA path de auth (Bearer→cookie→header pasan por `verify_session_cookie`); test `test_verify_rejects_past_absolute_cap`. El helper huérfano lo usan 2 tests: se deja. *Contar llamadores no basta: hay que mirar si la REGLA vive en otro sitio* |
| 4 | `medication_rules.requires_medication_review` (L471) — 0 call sites (tests: 6) | gate FS9 «revisión profesional» por interacción fármaco-alimento | **cablear / needs-owner** |
| 5 | `condition_rules.active_condition_labels` (L722) — 0 call sites (tests: 0) | etiquetas clínicas activas; sin lector ni test | **borrar** |
| 6 | `condition_rules.active_allergen_labels` (L978) — 0 call sites (tests: 0) | ídem alérgenos | **borrar** |
| 7 | `culinary_coherence.judgment_covers_delivered` (L1592) — 0 call sites (tests: 5) | resuelve si el juicio cubrió el plan entregado (tri-estado). Nadie pregunta | **cablear** |
| 8 | `recipe_library.apply_library_recipes_to_days` (L284) — 0 call sites (tests: 1) | caso de control del encargo: confirmado 0 | **borrar** |
| 9 | `recipe_library.apply_library_recipe` (L178) — 1 ref, solo desde (8) (tests: 12) | el propio comentario L214 lo admite por escrito | **borrar** (cadena) |
| 10 | `recipe_library._escalar_agua_de_biblioteca`, `_factor_implicito`, `_gramos_por_alimento` (L264/236/219) — solo desde (9) | cadena muerta de 3 niveles | **borrar** |
| 11 | `proactive_agent.check_and_trigger_jit_rolling_windows` (L823) — 0 call sites (tests: 0) | disparador JIT de ventanas rodantes | **needs-owner** |
| 12 | `proactive_agent._trigger_week2_background_generation` (L707) + `_bg_task` + `_apply_week2_append` — solo desde (11) | subárbol completo de generación semana-2 | **borrar** (cadena) |
| 13 | `graph_orchestrator._route_model_for_day_generator` (L6962) — 0 call sites (tests: 23) | 23 tests lo ejercen, producción no | **needs-owner** |
| 14 | `graph_orchestrator._is_skeleton_fidelity_rejection` (L6330) — 1 ref, solo desde (13) (tests: 8) | — | **borrar** (cadena) |
| 15 | `graph_orchestrator._select_ab_temp_pair` (L9674) — 0 call sites (tests: 0) | 4 menciones, todas en comentarios; sustituida por la variante async | **borrar** |
| 16 | `graph_orchestrator._ingredient_is_unresolved_protein` (L21853) — 0 call sites (tests: 7) | — | **needs-owner** |
| 17 | `graph_orchestrator._coh_finite_delta_rv` (L44368, anidada en `review_plan_node`) — 0 call sites (tests: 0) | — | **borrar** |
| 18 | `graph_orchestrator.get_llm_budget_stats_snapshot` (L1422) — 0 call sites (tests: 7) | snapshot de telemetría sin endpoint | **cablear** a `/admin/metrics` |
| 19 | `graph_orchestrator.get_circuit_breaker_snapshot` (L2598) — 0 call sites (tests: 0) | ídem | **borrar** |
| 20 | `graph_orchestrator.get_progress_cb_stats_snapshot` (L4540) — 0 call sites (tests: 0) | ídem | **borrar** |
| 21 | `graph_orchestrator.get_semantic_cache_stats_snapshot` (L46413) — 0 call sites (tests: 2) | ídem | **cablear** |
| 22 | `graph_orchestrator.LLMCircuitBreaker._save_db_state` / `._asave_db_state` (L2252/L2463) — 0 call sites (tests: 3/2) | el breaker **no persiste su estado**: reinicio = amnesia | **cablear** (impacto operativo) |
| 23 | `shopping_calculator.invalidate_master_cache` (L432) — 0 call sites (tests: 88) | 88 tests la usan; producción nunca invalida el cache maestro | **cablear / needs-owner** |
| 24 | `shopping_calculator.ingredient_demand_is_fresh` (L6329) — 0 call sites (tests: 3) | — | **borrar** |
| 25 | `shopping_calculator.get_aggregated_shopping_list_for_plan` (L13429) — 0 call sites (tests: 0) | — | **borrar** |
| 26 | `portion_solver.solve_portion_macros` (L625) — 0 call sites (tests: 24) | solver de porciones completo, sin caller | **needs-owner** |
| 27 | `portion_solver._coerce_line` (L600) + `nutrition_db.IngredientNutritionDB.macros_for_line` (L899) — solo desde (26) | — | **borrar** (cadena) |
| 28 | `portion_solver.effective_row_shares` (L341) — 0 call sites (tests: 15) | — | **needs-owner** |
| 29 | `plan_policy.canonical_name_for` (L129) — 0 call sites (tests: 7) | — | **borrar** |
| 30 | `plan_policy.template_id_coverage` (L202) — 0 call sites (tests: 1) | — | **borrar** |
| 31 | `plan_policy.explain_relaxations` (L670) — 0 call sites (tests: 2) | explicador de relajaciones que nadie muestra | **cablear** |
| 32 | `constants.chunk_refill_arrives_in_time` (L1389) — 0 call sites (tests: 5) | predicado de SLA de refill sin consumidor | **cablear** |
| 33 | `constants.apply_synonyms` (L2740) — 0 call sites (tests: 3) | — | **borrar** |
| 34 | `constants._reset_pantry_alias_index_cache` (L2998) — 0 call sites (tests: 6) | helper solo-tests | **documentar** (marcar test-only) |
| 35 | `db_meal_plans_audit.record_meal_plan_audit_backup` (L67) + `_serialize_jsonb` (L204) + `list_recent_audit_backups` (L213) — 0 call sites (tests: 7/0/4) | **módulo de auditoría entero sin caller**: no se escribe ni un backup | **cablear** (trazabilidad perdida) |
| 36 | `db_inventory.replace_shopping_list_only_items` (L2921) — 0 call sites (tests: 25) | — | **needs-owner** |
| 37 | `db_core.close_connection_pool` / `aclose_connection_pool` (L388/398) — 0 call sites (tests: 0) | `app.py:1663` documenta que el teardown «que lo cubría» ya no corre | **cablear** (fuga de pool en shutdown) |
| 38 | `dreaming._dreaming_batch` (L87) + `enqueue_dream_work` (L125) — 0 call sites (tests: 0) | encolado de dreaming sin productor | **needs-owner** |
| 39 | `price_engine.ingest_inflation_index` (L109) / `import_base_prices` (L231) — 0 call sites (tests: 5/4) | solo `scripts/`; el docstring L15/L24 los presenta como API viva | **documentar** (marcar offline) |
| 40 | `milp_meal_sizer.MilpMealSizer.solve` (L91) + `_fallback_solve` (L273) — 0 call sites (tests: 4) | la clase **no se instancia en producción**: módulo entero muerto | **borrar / needs-owner** |

**Cola (mismo criterio, 0 call sites):** `vision_agent.on_llm_end_async` (L597 — alias defensivo que LangChain nunca invoca; el hook
real es `on_llm_end`), `vision_agent.async_process_and_save_visual_entry` (L844), `generation_lifecycle.clear_run_progress` (L480),
`horizon._is_delicate_fresh` (L1056), `neon_auth.is_neon_auth_configured` (L75), `llm_provider.invalidate_tier_cache` (L226, tests: 8),
`catalog_capability.reset_cache` (L46), `push_i18n.push_catalog_keys` (L385), `db_profiles.get_ai_training_consented_user_ids` (L1402),
`deterministic_day.elegir_plantilla` (L449, tests: 7), `cron_tasks._provenance_weight_factor` (L852),
`canonical_recipe.IngredientLine.as_dict` (L99).

**Módulos offline (0 importadores en producción; NO son bugs, son herramientas):** `landing_benchmarks.py` (8 funciones, consumido por
`scripts/landing_benchmark.py`), `plan_gym.py` (9 funciones; `constants.py:2429` lo marca «EXENTO, nunca en el request path»),
`cultural_benchmark.py`, `milp_meal_sizer.py`. → **documentar**: moverlos a `scripts/` o poner `# OFFLINE-ONLY` en la cabecera para que
no reaparezcan en esta lista.

---

## 2. Knobs leídos y nunca consultados

### 2.1 `MEALFIT_HARDEN_SAMEDAY_PROTEIN` — el guard inerte más caro

- `graph_orchestrator.py:536` → `HARDEN_SAMEDAY_PROTEIN = _env_bool("MEALFIT_HARDEN_SAMEDAY_PROTEIN", False)`
- Única otra aparición en todo el repo: `graph_orchestrator.py:8333`, un **comentario**: «HARDEN_SAMEDAY_PROTEIN queda declarado (OFF)
  como placeholder del follow-up». **0 usos en código.**
- Pero `prod_profile.py:59` declara que producción lo tiene en `"true"` (perfil leído del `.env` del VPS el 2026-09-06). Sus hermanos de
  clase **sí** tienen rama: `HARDEN_CONDITION_CATALOG` (:8185), `HARDEN_SALTCURED_MAIN` (:8225), `HARDEN_CROSSDAY_QUOTA` (:8247),
  `HARDEN_MAIN_ARITY` (:8289). La clase 1 —proteína repetida el mismo día— no.
- Efecto: el operador cree que el binding slot→proteína está activo en producción. No lo está.
- **Disposición: needs-owner.** O se implementa la clase 1, o se quita el knob de `graph_orchestrator.py:536` **y** de
  `prod_profile.py:59` en el mismo commit.
- **Decidido por el dueño (2026-09-11) → `P1-PLAN-LOTE-7`: QUITADO** del god-file y de `prod_profile` en el mismo commit.
  `MEALFIT_HARDEN_SAMEDAY_PROTEIN=true` queda huérfano en el `.env` del VPS: inerte; limpiarlo en la próxima pasada del operador.

### 2.2 `COUNTRY_SYSTEM_ENABLED` — snapshot congelado sin lector

- `constants.py:3593`. `app.py:2040` lo desaconseja por escrito («no usar…, es un snapshot tomado al importar»); el camino vivo relee el
  knob por llamada (`app.py:2044`). 0 lecturas reales.
- Lo sostiene un test de forma: `tests/test_p1_country_system_f0.py:89` casa la línea literal.
- **Disposición: borrar** (y ajustar ese test, que hoy protege una variable muerta).

### 2.3 `SLOT_AWARE_DAY_REPAIR` — knob sin rama, decisión aplazada

- `graph_orchestrator.py:13644`. Referencias: dos comentarios (`:48980`, `:49202`) que dicen que la decisión de encenderlo quedó
  aplazada. `tests/test_p2_solver_seeder_v4_batch.py:407` casa la línea.
- **Disposición: documentar** (fecha y dueño de la decisión) o **borrar**.

### 2.4 `CHUNK_STALE_FINAL_LIVE_TIMEOUT_SECONDS` — timeout sin reloj

- `constants.py:522`, `int(os.environ.get(..., "60"))`. Aparición única en todo el repo.
- **Disposición: borrar.**

---

## 3. Flags `_*` escritos en `plan_result`/`plan_data` y nunca leídos

153 claves `_x` se escriben sin ninguna lectura en producción (`.get("_x")`, `["_x"]`, `"_x" in …`). El grupo que importa son los
**veredictos de gate degradados a advisory en el intento final**: el gate detecta la violación, decide entregar el plan igualmente y
marca la bandera… que nadie lee.

| Fichero:línea | Flag | Guard que lo escribe |
|---|---|---|
| `graph_orchestrator.py:44018` | `_dish_quality_advisory_final` | `P2-DISH-QUALITY-GATE` |
| `graph_orchestrator.py:43966` | `_slot_incoherence_advisory_final` | `P1-SLOT-INCOHERENCE-GATE` |
| `graph_orchestrator.py:43991` | `_staple_repeat_advisory_final` | `P1-STAPLE-REPEAT-GATE` |
| `graph_orchestrator.py:43854` | `_repeat_gate_advisory_final_attempt` | `P1-VARIETY-REPEAT-GRACEFUL` |
| `graph_orchestrator.py:43608` | `_reviewer_advisories` | `P1-REVIEWER-VERIFICATION-ADVISORY` |

Verificado: **0** apariciones de estos cinco en `frontend/src` y `frontend/e2e`, y 0 lecturas en backend.
`_slot_incoherence_advisory_final` y `_staple_repeat_advisory_final` tampoco tienen test. **Disposición: cablear** — basta un lector (el
endpoint `blocked_reasons` de un plan, o el panel de calidad) que exponga los advisories del intento final; si no, retirar las cinco
escrituras. **Otros sin lector, sin test y sin frontend** (retirar, salvo que alguien los reclame): `cron_tasks.py:15049/15050`
`_learning_forced` y `_learning_forced_reason` (6 y 4 escrituras); `cron_tasks.py:29774` `_degraded_fallback_level` (4);
`graph_orchestrator.py:9534` `_parallel_duration` (3); `graph_orchestrator.py:22085` `_closer_macros_unaccounted` (2);
`graph_orchestrator.py:7843` `_skeleton_reused_days`; `graph_orchestrator.py:8033` `_meal_count_decision`; `graph_orchestrator.py:40737`
`_raw_misalign_stages_post`; `portion_solver.py:1354` `_refine_raw_by_food`.

---

## 4. `try/except` que se tragan el guard sin dejar rastro

737 manejadores `except Exception|bare` cuyo cuerpo es **exactamente** `pass`, `return None`, `return []` o `continue`. De ellos, **91**
envuelven lógica de guard (nombre de la función o cuerpo del `try` con
`guard|gate|valid|check|verif|enforc|sanit|safety|coheren|allerg|clinic|sodium|fidel|contract`). Reparto: `graph_orchestrator.py` 61,
`cron_tasks.py` 8, `shopping_calculator.py` 8, `routers/plans.py` 3, `agent.py` 2, `plan_gym.py` 2, y 1 en cada uno de `ai_helpers`,
`condition_rules`, `culinary_coherence`, `db_inventory`, `deterministic_day`, `dish_registry`, `nutrition_db`.

### 4.1 El caso grave: `graph_orchestrator.finalize_plan_data_coherence`

~20 pasadas de coherencia encadenadas, cada una en su propio `try: … except Exception: pass` (L28577, 28603, 28820, 28824, 28964, 28974,
28982, 28991, 29000, 29010, 29018, 29027, 29036, 29045, 29053, 29061, 29069, 29078, 29086, 29095, 29112, 29123). El contador `total` y
la lista `parts` solo acumulan **éxitos**: si `_restore_display_from_raw_orphans` o `_polish_finalize_display` fallan, el plan sale sin
esa reparación y el log no dice nada. Contrasta con L28960, en el mismo bloque, que sí hace
`logger.warning("[P2-BOUNDARY-DISPLAY-POLISH] … no-op: …")` — el patrón bueno ya existe en el fichero. **Disposición: cablear** —
replicar ese `logger.warning` en los ~20 handlers, o hacer `parts.append(f"{nombre}=ERR")` para que el fallo viaje con el plan.

### 4.2 Mismo patrón, mismo diagnóstico (todos **cablear**: log o bandera)

`graph_orchestrator` — `_run_assembly_validations` L25843/25891/25925/26013 (ejemplo vivo: el gate `P1-BLEND-STEP-REQUIRED` en L25890;
si el autofix del batido falla, no se inserta el paso **ni** se añade nada a `recipe_coherence_errors`: la receta sale sin licuar y sin
queja) · `_apply_food_safety_fixes` L19309/19354 (nota de seguridad del huevo) · `_day_sodium_autofix` L30261/30305/30497/30502 ·
`finalize_single_meal_recipe_coherence` L29384/29399/29527/29582 · `_repair_recipe_contract` L25161 · `_generation_sanity_autofix`
L27889 · `_apply_coherence_history_cap` L42578 · `_coherence_block_history_cap` L42474 · `refresh_clinical_band_score_post_finalize`
L49687 · `semantic_cache_check_node` L46536.

`cron_tasks` — `_refresh_chunk_pantry_inner` L9852/9927 · `_persist_fresh_pantry_to_chunks` L12963 · `_recover_pantry_paused_chunks`
L14634 · `_alert_coherence_watchdog_silent` L2734 · `_shopping_coherence_alert_job` L1374 · `_clinical_band_drift_alert_job` L6035 ·
`_alert_chunk_pantry_snapshots_stale` L22912. **Otros** — `deterministic_day.verifica_comida` L863 ·
`db_inventory.find_pantry_rows_for_name` L1511 · `agent._swap_real_pantry_ledger_lines` L929 ·
`agent._emit_checkpoint_pool_split_missing_alert_best` L4006.

### 4.3 Guards que devuelven un valor **permisivo** al fallar (fail-open silencioso)

Al reventar, el guard responde lo mismo que si el plan estuviera limpio. **Alergénicos/clínicos:**
`graph_orchestrator._allergen_pool_item_banned` L16119 → `False` («no hay alérgeno») · `dish_registry.allergen_classes_for` L143 → `[]`
· `_apply_pregnancy_food_safety_annotations` L19588 → `0` · `_apply_condition_safety_annotations` L19734 → `0` ·
`condition_rules.collect_allergen_substitutions` L945 → `pass`. **Coherencia y contrato:** `culinary_coherence.culinary_contract_scan`
L1526 → `[]` · `_variety_repeat_gate_issues` L25396 → `[]` · `_recipe_step_contract_issues` L24859 → `[]` ·
`slot_coherence_backstop_for_meal` L16404 → `[]` · `_clamp_recipe_time_temp_outliers` L24600 → `False` ·
`_maybe_mark_clinical_layer_incomplete_degraded` L50500 → `False` · `ai_helpers._n_gate_fruits` L757 → `0`.

**Disposición: cablear** — `logger.warning` + bandera `_guard_no_corrio` en el plan. Es la diferencia entre «no hay alérgeno» y «no pude
mirar», y hoy el sistema entrega las dos igual.

---

## 5. Ramas muertas y knobs default-off sin plan de activación

- **Código inalcanzable tras `return`/`raise`/`continue`/`break`: 0** en los 116 módulos. Limpio. **`if False:` / `while False:`: 0.**
- **`if True:` (envoltorio vestigial), 2 casos:** `graph_orchestrator.py:50726` (`P1-BAND-METRIC-NO-SILENT-DROP`) y
  `graph_orchestrator.py:34104` (`P1-APIO-STALK-CAP`). Restos de una condición retirada: el bloque corre siempre y el `if` solo añade
  indentación. **Disposición: borrar el `if True:`** (cosmético, riesgo cero; el comentario adyacente lo explica).

**Knobs `_env_bool(..., False)` sin mención en `backend/docs`, `docs` ni `CLAUDE.md`: 21 de 45.** Ninguno figura en `.env` salvo
`MEALFIT_CARB_TARGET_TRIM`: `graph_orchestrator.py` — `MEALFIT_SLOT_AWARE_DAY_REPAIR` (:13644, *además sin rama*, §2.3),
`MEALFIT_HARDEN_MAIN_ARITY` (:565, pero `prod_profile` lo pone `true`), `MEALFIT_MICRONUTRIENT_SOFT_REJECT` (:13584),
`MEALFIT_FAT_LEAN_SWAP` (:13067), `MEALFIT_VARIETY_GATE_BASE_DISH_REPEAT` (:12784), `MEALFIT_CARB_TARGET_TRIM` (:12244),
`MEALFIT_CORRECTOR_NONE_DIAGNOSTIC` (:704), `MEALFIT_EVALUATOR_USE_PRO` (:6186), `MEALFIT_DAYGEN_LITE_FOR_EASY` (:6378) · `constants.py`
— `MEALFIT_INITIAL_CHUNK_PANTRY_GUARD` (:979), `MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED` (:997), `MEALFIT_PANTRY_COMPLETION_LIST_ENABLED`
(:1003) · `inventory_sufficiency.py:52` — `MEALFIT_PANTRY_SUFFICIENCY_MICROS_GATE` · `db_profiles.py:67` — `MEALFIT_REQUIRE_ATOMIC_POOL`
· `db_inventory.py:52` — `MEALFIT_INVENTORY_RPC_STRICT` · `error_utils.py:30` — `MEALFIT_LEAK_DB_ERRORS` · `app.py:1790` —
`MEALFIT_READY_REQUIRE_DB` · `ai_helpers.py` — `MEALFIT_LIGHT_PROTEIN_SEED` (:84), `MEALFIT_GROCERY_CYCLE_LOCK` (:2103) ·
`micronutrients.py:26` — `MEALFIT_ANEMIA_CONDITION_TARGET` · `shopping_calculator.py:298` — `MEALFIT_DISABLE_SEMANTIC_CACHE`.

Todos tienen rama real (a diferencia de §2): el problema es de gobierno, no de código. **Disposición: documentar** en
`backend/docs/knobs_reference.md` con dueño y criterio de activación. Los dos de carga clínica
(`MEALFIT_PANTRY_SUFFICIENCY_MICROS_GATE`, `MEALFIT_ANEMIA_CONDITION_TARGET`) merecen **needs-owner** antes que documentación.

---

## Verificación

Todo se ejecutó desde `backend/` con Python 3.12.11, sin escrituras en el repo, sin tocar la base de datos y sin un solo `pytest`. Los
pasos 1-4 y 7-10 son scripts Python de una pieza sobre `ast` y `tokenize`; el paso 5 son invocaciones de `grep`. Nada depende del orden
ni de estado previo.

1. **Censo** — `os.walk` dejando fuera los directorios `tests`, `scripts`, `scratch`, `migrations`, `venv`, `venv-test`, `test_venv`,
   `node_modules`, `__pycache__`, `infra`, `data`, `uploads` y `.pytest_cache`, y los ficheros `test_*.py` y `conftest.py` → **116
   ficheros, 11 844 852 bytes**.
2. **Definiciones** — `ast.parse` + visitor sobre `FunctionDef`/`AsyncFunctionDef` guardando `lineno`, `end_lineno`, clase contenedora y
   decoradores → **3096 nombres**.
3. **Referencias** — `tokenize.generate_tokens`, conservando tokens `NAME` y los `STRING` cuyo contenido es *exactamente* un
   identificador (cubre `getattr(m, "f")` y `__import__("m").f` sin contar docstrings ni comentarios). Muerta = sus únicas apariciones
   son sus líneas `def`.
4. **Cadenas muertas** — punto fijo (3 iteraciones): si todas las referencias de `f` caen dentro del rango `[lineno, end_lineno]` de
   funciones ya muertas, `f` es muerta. Se descartan las decoradas con los verbos HTTP de `router` y `app`, `field_validator` y
   `model_validator` → **84**.
5. **Segunda pasada con alias** (obligatoria, sobre las 63 funciones del reporte) — `grep` recursivo con nombre exacto (`-rnw`),
   limitado a las extensiones `py`, `js`, `jsx`, `ts` y `tsx`, lanzado dentro de `backend` y dentro de `frontend/src`; cada hit se
   clasifica en prod, tests, scripts o frontend, descartando definiciones, comentarios y docstrings. Pasada extra (`grep -rnE`) para
   los patrones de alias (`as NOMBRE`), `getattr(…, "NOMBRE")` y `__import__(…).NOMBRE`: **ningún alias ni despacho dinámico**. Resueltos a mano:
   `_select_ab_temp_pair` (4 hits, los 4 comentarios), `record_meal_plan_audit_backup` (docstring de módulo, L23-36),
   `enqueue_dream_work`, `get_ai_training_consented_user_ids`, `import_base_prices` y `list_recent_audit_backups` (único hit = su propio
   `logger`).
6. **Falsos positivos descartados:** `correlation.CorrelationIdFilter.filter` (lo invoca `logging` vía `addFilter`, `correlation.py:210`
   y `:223`) y `vision_agent._VisionUsageCapture.on_llm_end` (lo invoca LangChain desde `vision_agent.py:631`; su alias
   `on_llm_end_async` sí está muerto).
7. **Knobs (§2)** — para cada `Assign`/`AnnAssign` cuyo valor contiene `_env_bool|_env_int|_env_float|_env_str|getenv|environ` (**1186
   asignaciones**), conteo global del identificador destino; `cuenta <= 1` ⇒ solo existe la asignación. Cruce con
   `prod_profile.PROD_KNOBS` (38 knobs) para detectar los que **producción activa** y nadie consulta. Confirmación con `grep -rn` del
   nombre de la variable y del knob sobre `.py`, `.md`, `.env`, `.env.example`, `backend/docs`, `docs` y `CLAUDE.md`.
8. **Flags `_*` (§3)** — regex de escritura (`["_k"] =` y `"_k":`) y de lectura (`.get/.pop/.setdefault("_k")`, `["_k"]` no seguido de
   `=`, `"_k" in`), con comentarios descartados → **153** claves escritas sin lectura; las finalistas se recontrastaron contra
   `frontend/src` y `frontend/e2e`.
9. **`try/except` (§4)** — `ast.walk` sobre `ast.Try` cuyos `handlers` tengan cuerpo de **un solo
   statement**: `Pass`, `Continue`, o un `Return` de None / lista o dict vacíos / bool, con tipo `Exception`, `BaseException`
   o desnudo → **737**; filtro por nombre de la función contenedora o
   cuerpo del `try` con el patrón de guard → **91**.
10. **Ramas muertas (§5)** — `ast.walk` sobre `If`/`While` con `test` `Constant` en `{False, 0, None}` (0 resultados) y en `{True, 1}`
   (2 resultados); barrido de sentencias que siguen a `Return`, `Raise`, `Continue` o `Break` en el mismo `body`, `orelse` o `finalbody`
   (0 resultados).
