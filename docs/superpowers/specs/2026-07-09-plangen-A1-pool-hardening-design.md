# Diseño: A1 — endurecer la selección de alimentos a nivel de pool

**Fecha:** 2026-07-09
**Autor:** brainstorming asistido (Claude, ultracode) + owner
**Estado:** DISEÑO APROBADO por el owner ("implementa todo") — pendiente de plan de implementación (writing-plans).
**Relación:** hija de [`2026-07-09-plangen-architecture-nextlevel-design.md`](2026-07-09-plangen-architecture-nextlevel-design.md). Este spec detalla la **Fase A (A1)** + la **Fase 0 (medir)** + el **swap-repair**, y prepara (no promueve) la **Fase B**.

---

## 0. TL;DR

La pregunta del owner fue: *"¿no hay una arquitectura más completa — determinista para cantidades y alimentos, IA+solver para personalizar, agente médico?"*. Verificado contra el código vivo (2 workflows, 7 agentes): **esa arquitectura YA existe** para cantidades (el motor re-dimensiona y descarta las del LLM), solver (`_box_lsq`, macros ~87-94% all-4) y agente médico (reviewer PRO). La **inversión total** de la selección (motor compone el plato) es un **net-negativo** (muro de Stigler → platos "correctos pero robóticos" que el propio gate §19 rechaza).

El salto real y seguro: **endurecer solo la rebanada de *identidad* de la selección a nivel de pool**, dejando al LLM la composición del plato + recetas + personalización. Esto vuelve **imposibles por construcción** ~5 clases de bug que hoy se defienden con retry-loops caros y backstops de paridad (el whack-a-mole), sin tocar la inteligencia culinaria.

Decisión del owner: **A + B, máximo salto**, con **A1 (pool-level, no composición)** como forma de A. Secuencia: **Fase 0 (medir) → Fase A (A1) → Fase B (cortar capa media, gated por canario)**.

---

## 1. Estado actual (verificado en código)

### 1.1 Cuán determinista es ya la selección

La selección de alimentos es **~50% determinista, con asimetría**: como **guardarraíl** es casi total (~90%); como **motor que elige activamente** es solo ~15-25%.

- **Guardarraíl duro:** catálogo verificado (~200 alimentos, `_get_verified_catalog_instruction`), pools por día del planner, filtros duros alérgeno/dieta (`_scan_allergen_violations`/`_scan_diet_violations`, `graph_orchestrator.py:~11663`), scrub de proteínas restringidas (`plan_skeleton_node:6389-6403`), y detectores slot/variedad/coherencia que fuerzan regen.
- **Motor que elige:** substituciones por condición (post-hoc), seeder de micros, fallback de pool vacío, night-rice autofix, `_apply_protein_pool_scrub` (`:6580`).
- **Discreción pura del LLM:** qué plato/combinación concreta y qué vegetales llenan cada slot dentro del pool asignado.

### 1.2 Dos garantías de pool YA duras (los templates a copiar)

- **Legumbre (clase 4):** `ai_helpers.py:647-680` — tras el draw de proteínas únicas, si no hay legumbre y el goal no está en `_GOALS_SKIP_LEGUME_GUARANTEE` (gain_muscle) ni es bariátrico, **force-injecta** una legumbre. Garantía constructiva, no retry.
- **Densidad proteica gain_muscle/bariátrico (clase 6, mitad densidad):** `ai_helpers.py:682-717` — reemplaza mains de baja densidad por animal denso tras el draw.

Estos dos son el **patrón de referencia** ("force/replace after draw").

### 1.3 El seam de inserción (verificado)

```
planner LLM → skeleton dict (:6334)
  → scrub existente plan_skeleton_node :6344-6454
      (cap restringidos :6389, atún+embutido :6406, fallback alergia :6424, meal_types clínicos :6441)
  → [NUEVO] harden_day_pools(skeleton, form_data, conditions)   ← INSERTAR en :6454, antes del return :6464
  → return {'plan_skeleton': skeleton}
  → generate_days_parallel_node: _other_days_brief :7367, _diversify_egg_pools :7382
  → generate_single_day :6751 → build_day_assignment_context :6773
      → PRIMER read del LLM: day_generator.py:408 (protein_pool) / :567 (carb) / :568 (fruit) / :473 (breakfast_cat)
  → post-gen _apply_protein_pool_scrub :6580  ← se queda como backstop
```

**Regla de oro:** como el primer read del LLM es `day_generator.py:408`, endurecer en `:6454` garantiza que **el LLM nunca ve un pool crudo**. **MERGE** dentro del bloque de scrub existente (mismo nodo, misma mutación in-place de `skel_days`, contexto clínico ya en scope). Reusar `_key_in_text` (normaliza acentos/word-boundaries) y `_allergy_safe_fallback_protein`.

> **Caveat de ordenamiento:** `_diversify_egg_pools` (`:7382`) re-muta `protein_pool` después. Es allergy-aware/egg-scoped (bajo riesgo), pero si una garantía debe ser estrictamente terminal, invocar también en `:7383` o reordenar el egg-diversify antes. Se resuelve en implementación por-garantía.

---

## 2. Diseño de las 5 garantías (A1)

Principio: **el pool no puede suministrar al ofensor → la violación es inalcanzable como ingrediente planificado.** Los backstops post-hoc **sobreviven degradados** (el LLM escribe nombres/pasos en texto libre que el pool no vigila; catálogos estrechos fuerzan fallback gracioso).

> **Knobs (todos nacen OFF/neutral, auto-registrados en `_KNOBS_REGISTRY`):** master `MEALFIT_HARDEN_POOLS_ENABLED` (kill-switch), `MEALFIT_HARDEN_CONDITION_CATALOG` (clase 3), `MEALFIT_HARDEN_SALTCURED_MAIN` (clase 5), `MEALFIT_HARDEN_SAMEDAY_PROTEIN` (clase 1), `MEALFIT_HARDEN_CROSSDAY_QUOTA` (clase 2), canario `MEALFIT_HARDEN_POOLS_CANARY_PCT` (0).

| # | Clase | Hoy | Garantía dura pool-level | Knob (nace OFF) | Retira → advisory |
|---|---|---|---|---|---|
| 3 | **Identidad contraindicada** (DM2 toronja/arroz-blanco, HTA embutidos/bacalao) | `_get_fast_filtered_catalogs` (`constants.py:2316`) filtra alergias/dislikes/dieta pero **NUNCA condiciones médicas** → ofensor vive en el pool, solo se corrige post-hoc por `collect_substitutions` | Filtro pre-draw condition-aware dentro de `_get_fast_filtered_catalogs` (o wrapper) que reusa los token-sets `_DM2_GLYCEMIC_SUBS`/`_HTA_SODIUM_SUBS` (`condition_rules.py:85/93`) como **SSOT**, con `detect_active_rules(form_data)` para la precedencia (renal>embarazo>bariátrico>hta>dm2) | `MEALFIT_HARDEN_CONDITION_CATALOG` | demota `collect_substitutions` (`condition_rules.py:594`, aplicado `:11521`/`:12073`) de corrector-primario a caza-escapados — **NO se retira** (caza texto libre del LLM) |
| 5 | **Salado como principal** (bacalao/salami/tocino/arenque) | Solo peso suave x0.1 (`P1-SODIUM-BOMB-POOL`, `ai_helpers.py:595`) — puede ser el principal | Exclusión dura del slot **principal** vía `_SALT_CURED_NEVER_MAIN` (= `_SALT_CURED_PROTEIN_TOKENS`), universal (budget sodio es goal-independiente). Permitido solo como saborizante capado. Fallback gracioso si todo el catálogo filtrado es salado (casi imposible) | `MEALFIT_HARDEN_SALTCURED_MAIN` | hace redundante el x0.1 en el slot principal; el scrub atún+embutido (`:6405`) queda como subconjunto |
| 1 | **Proteína repetida mismo día** | Gate degradable — **puede shippear** en el intento final (`P1-VARIETY-REPEAT-GRACEFUL`, `review_plan_node:~28892`) | Allocator determinista slot→proteína: liga 1 proteína pesada distinta a cada slot principal (almuerzo/cena); ninguna pesada (ni huevo) ocupa >1 slot principal. Gobierna `_SAME_DAY_PROTEIN_GATE_LABELS`. Carve-outs: legumbre (arroz-con-habichuela) y yogurt repetibles; ≥5-6 comidas/día se salta (espeja `_relax_high_mc:17133`) | `MEALFIT_HARDEN_SAMEDAY_PROTEIN` | baja a telemetría la rama `same_day_protein_repeats` de `_variety_repeat_gate_issues` (`:17170`, knob `:9860`) + su rechazo en `review_plan_node`; `_protein_repeat_autofix` (`:21461`) queda idempotente no-op |
| 2 | **Repetición cross-día** (proteína pesada + dish-base) | Advisory salvo el cap duro de restringidos max-1-día (`:6388`); `MEALFIT_CROSS_DAY_PROTEIN_GATE` ya default-OFF | Generaliza el cap max-1-día a **todas** las pesadas: cuota semanal round-robin `ceil(num_days / proteínas_distintas)` particionada sobre los day-skeletons antes del day-gen; misma partición al `_head_dish_base_token` (`:15925`). Cuota se **ensancha** grácilmente si el catálogo es chico | `MEALFIT_HARDEN_CROSSDAY_QUOTA` | mata `cross_day_proteins` (`MEALFIT_CROSS_DAY_PROTEIN_GATE`, `:15985`) para pesadas → telemetría. **`cross_day_dishes` (`VARIETY_GATE_CROSS_DAY_DISH`, `:17161`) SOBREVIVE** — repetición de *preparación/plato* es ortogonal al pool de ingredientes. `cross_day_preps` sigue advisory (técnica en texto libre) |
| 4/6 | Legumbre / densidad | **Ya duras** (`ai_helpers.py:647` / `:682`) | Se dejan como templates. Opción: expresar legumbre como cuota "≥1 día-legumbre/semana" (no "≥1 en las mains"), preservando el carve-out gain_muscle/bariátrico (piso proteico gana) | — | — |

### 2.1 Límites honestos (parciales, no ocultarlos)

- **Texto libre del LLM:** nombres de plato y pasos de receta no son gobernables por el pool. Backstops de texto (`collect_substitutions`, `_apply_protein_pool_scrub`, sodium autofix, `_HTA_SODIUM_SUBS`) **deben sobrevivir**.
- **Seasoning HTA** ("sal al gusto"→"sal mínima ¼ cdta", `P2-HTA-SALT-NORMALIZE`): es cantidad/frase, no identidad de pool → sin lever. Se queda como está.
- **Catálogos estrechos** (vegano/muchos dislikes): la cuota y el binding **deben degradar grácilmente** (reusar, ensanchar) — el fallback gracioso es load-bearing (`ai_helpers.py:709-710`).
- **cross_day_preps**: monotonía de técnica ("7 cenas a la plancha") es texto, no pool → advisory-only.

---

## 3. Fase 0 — el medidor (canario)

Timeboxed al mínimo que gatea A y B. **NO una plataforma de analytics.**

- **Scorecard unificado (offline):** consolida `compute_clinical_band_score` (`:33338`) + replays deterministas (`scripts/macro_sizing_replay.py`) en un JSON por-plan con guard de cobertura. Un número por plan, misma métrica offline+live.
- **Medidor de determinismo/varianza:** corre K veces el mismo perfil → **dish-overlap %**, macro-delta ~0 sanity, **distribución retry/degradado**. Mide la fricción #1 (whack-a-mole) y da el baseline.
- **Cohorte de canario:** copiar el patrón `P1-SELF-CRITIQUE-CANARY` (`graph_orchestrator.py:453-468`, `_self_critique_canary_cohort`, tag emit `:34440-34458`). Knob `MEALFIT_HARDEN_POOLS_CANARY_PCT` (0), bucketing sha256 con **salt independiente** `f"harden_pools|{_id}"` (evita confundir con la cohorte self_critique), tag `harden_pools_cohort` junto a `self_critique_cohort` en `:34446`. Compara por cohorte: **all-4-macro band** (`:34449-34451`), **degraded-ship rate** (`review_passed:34453` + alert `plan_quality_degraded:29935`), **retry-rate** (`:34443`).
  - **Requisito nuevo:** la tasa de violación same-day/cross-día **NO está hoy en la metadata** → el canario DEBE añadir `same_day_protein_repeats` (`build_variety_report:15966`), `cross_day_proteins` (`:15985`) y `cross_day_dishes` (`:16003`) al bloque `:34446`. Es la señal primaria "¿la restricción dura eliminó la clase?".

> **Nota honesta (Fase 3 "sembrar RNG" de la propuesta externa):** sembrar reduce varianza pero NO la lleva a 0 — el `dish_library` ya muestrea por `day_num` (parcialmente sembrado), pero la elección de plato del LLM vía **API DeepSeek no es seed-reproducible**. Valor real pero acotado; se implementa donde el código Python controla el RNG, no se sobre-vende.

---

## 4. Swap-repair (reactivo, complementa A1)

A1 es preventivo (el pool no compone mal); el swap-repair es reactivo: si en ensamblaje una canasta resulta **inviable o clínicamente peor**, intercambia un **hermano del pool** (mismo rol/slot) + **re-narra solo esa comida** (blast-radius chico). Knob `MEALFIT_POOL_SWAP_REPAIR` (OFF). Mata el whack-a-mole de canasta-inviable sin regenerar el día completo.

---

## 5. Qué se retira vs qué NUNCA se retira

**Retirar → advisory/telemetría** (SOLO tras canario verde por clase — el resto de la familia de variedad SOBREVIVE porque A1 liga ingredientes, no fruta/plato/timing):

| Detector / gate | Loc | Veredicto |
|---|---|---|
| rama `same_day_protein_repeats` de `_variety_repeat_gate_issues` | `:17170` (knob `:9860`) | → telemetría (clase 1 la vuelve inalcanzable) |
| `MEALFIT_CROSS_DAY_PROTEIN_GATE` (heavy ≥3 días) | `:15985` | muerto para pesadas (clase 2); ya default-OFF |
| `_protein_repeat_autofix` (rewriter same-day) | `:21461` | idempotente no-op (espejo de la rama same-day) |
| peso x0.1 sodio + scrub atún+embutido | `ai_helpers.py:595` / `:6405` | subconjuntos redundantes (NO borrar hasta canario) |

**SOBREVIVEN (NO tocar — A1 no los cubre):** `cross_day_dishes`/`VARIETY_GATE_CROSS_DAY_DISH` (`:17161`), `same_day` plato-base (`:17152`), fruta repetida + `sweet_savory_clash` (`:17134`/`:17143`), egg-overuse gate (`:28868`), y **todas** las ramas "end" de `should_retry` (`critical:30262`, `high_contextual:30281`, `max_attempts:30334`, `budget_exhausted:30403`, `approved_with_residual:30232`) — son routing de severidad/budget, no lógica de variedad. A1 baja su *tasa*, no las retira.

**NUNCA retirar (defensa-en-profundidad, degradada):** reviewer médico PRO (`review_plan_node:28271`), coherence guard (`run_shopping_coherence_guard`, `shopping_calculator.py:2686`), scans duros alérgeno/dieta (`_scan_allergen_violations:11617`/`_scan_diet_violations:11718` — **A1 filtra condiciones, NO alergias**), cap renal fail-hard (`:28840`), `collect_substitutions` (`condition_rules.py:594` — caza texto libre), `_apply_protein_pool_scrub` post-gen (`:6580` — mantiene al LLM honesto al pool), y los **fallbacks graciosos** para catálogos estrechos (`:6355`/`:6416`).

> **⚠️ Riesgo de implementación (clase 3):** el filtro por condición **estrecha el catálogo aún más** encima de alergias/dieta → un usuario sobre-restringido (p.ej. vegano + DM2 + alergia mariscos) tiene más riesgo de pool vacío. El fallback gracioso se vuelve **más** load-bearing bajo A1 → **stress-test obligatorio**.

---

## 6. Rollout + secuencia

1. **Cada garantía nace OFF** tras su propio knob `MEALFIT_*` (auto-registrado en `_KNOBS_REGISTRY`). Deploy es seguro (cero cambio de comportamiento).
2. **Fase 0 primero** (el medidor) — sin él no hay canario.
3. **Fase A (A1)** — encender garantía por garantía en cohorte canario, leer `clinical_band` OFF vs ON. Gate: all-4-macro no baja >2pp **Y** baja retry/degraded. Recomendado empezar por **clase 3** (mayor valor clínico, más self-contained).
4. **Retirar ramas muertas** solo tras canario verde por clase.
5. **Fase B (gated):** A1 **subsume parcialmente** los detectores que hoy obligan a mantener `self_critique`:
   - `_count_staple_repetitions` (`:8242`) → **subsumido** por clase 2 → gate portado `MEALFIT_STAPLE_REPEAT_GATE` (`:29005`) queda casi-muerto.
   - `_detect_slot_incoherence` **mitad proteína** (lunch↔cena comparten proteína `:8405`; heavy en ≥2 slots `:8420`) → **subsumido** por clase 1.
   - **NO subsumido** (residual): lunch↔cena comparten **carbohidrato** (`:8407`) + merienda con técnica de plato-fuerte (`:8451`) → siguen necesitando `MEALFIT_SLOT_INCOHERENCE_GATE` (`:28981`) **ON**.

   **Secuencia recomendada (paso 6):** encender `MEALFIT_SLOT_INCOHERENCE_GATE` + `MEALFIT_STAPLE_REPEAT_GATE` **ON** en la misma ventana que el canario A1 → A1 lleva su fire-rate a ~0 → señal limpia "gates ON pero inertes" = el layer determinista cubre lo que hacía self_critique → recién ahí default-OFF del nodo. **Requiere tu firma del umbral + canario en vivo.** No se flipea en este spec. Además: A1-on mata en raíz las 2 clases de bug de paridad más caras (`P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY`, `P1-CRITIQUE-CROSSDAY-DISH-PARITY`).

---

## 7. Testing (convención del repo)

- **Parser-based con tooltip-anchor**: un renombre del enforcer/knob falla el test antes de tocar prod.
- **Funcionales por clase**: asserten que **el pool no puede suministrar al ofensor** (violación inalcanzable) para perfiles DM2/HTA/gain_muscle/vegano.
- **Fallback gracioso**: catálogos estrechos (vegano, muchos dislikes) — la cuota/binding degrada, no vacía el pool ni falla-duro.
- **No-regresión**: `test_p1_shopping_recipe_coherence.py` (89 casos), band-score no baja.
- **Baseline**: correr el harness determinista antes/después (los macros ya están maxeados — vigilar que A1 no encoja el pool factible y reintroduzca misses de piso proteico).

---

## 8. Ítems abiertos

- Inventario exacto de ramas `should_retry` muertas (agente de verificación en curso).
- Orden vs `_diversify_egg_pools` por garantía (clase 1 puede necesitar re-invocación en `:7383`).
- Alcance del scorecard Fase 0 (mínimo viable, no sobre-construir).
- Umbral exacto de canario para promover Fase B (owner-gated).

---

## 9. Convenciones de implementación (del repo)

- Gates nuevos **nacen OFF**; knob `MEALFIT_*` con default seguro, registrado en `_KNOBS_REGISTRY`.
- Sin DDL en runtime; migraciones en `migrations/` **y** `backend/migrations/` (SSOT dual) — aplica si el scorecard necesita tabla.
- Tests que parsean source → tooltip-anchor.
- Bumpear `_LAST_KNOWN_PFIX` (`app.py:32`) por cada cierre, con test cross-linkeado por slug.
- Imports DB por la fachada `from db import ...`.
- Commit SCOPED con pathspec (nunca `-A`), push, `deploy-mealfit.ps1`, avisar "Clear site data".
