# Diseño: llevar el pipeline de generación de planes a nivel producción

**Fecha:** 2026-07-09
**Autor:** brainstorming asistido (Claude) + owner
**Estado:** DISEÑO — pendiente de revisión del owner. NO implementar hasta OK explícito.
**Prioridad #1 declarada por el owner:** precisión de los planes.

---

## 0. TL;DR

El pipeline es una máquina **"el LLM propone, el determinista dispone, un LLM aprueba por seguridad"** a la que le crecieron **3 capas de opinión LLM en el medio** que pelean contra el núcleo determinista. La precisión de macros es **determinista** (el motor re-dimensiona todo antes de que el reviewer LLM vea el plan), así que **la cantidad de agentes NO es la palanca de precisión**. Las dos palancas reales son:

1. **El solver de porciones** (greedy proporcional → NNLS/LP) — el mayor salto de precisión posible, determinista, ~ms.
2. **Eliminar la capa LLM del medio** (self_critique / adversarial judge) — no mejora los números, pero elimina la clase de bug de "detector-paridad" que hoy hace *entregar planes degradados*.

Veredicto sobre agentes: **RESTRUCTURAR → colapsar 10 roles LLM a ~5-6**, quitando el medio y conservando ambos extremos (skeleton/day-gen y reviewer/fact-checker/corrector quirúrgico).

Roadmap en 6 fases (0→5), reordenado para precisión-primero con una Fase 0 mínima de durabilidad que hace seguro iterar en el hot-path.

---

## 1. Estado actual (verificado en código)

### 1.1 Topología del grafo (`backend/graph_orchestrator.py`)

`StateGraph(PlanState)` con 12 nodos, `build_plan_graph()` (`:31291`), corrido vía `plan_graph.astream(...)` en `arun_plan_pipeline` (`:34842`) bajo `asyncio.wait_for(GLOBAL_PIPELINE_TIMEOUT_S=720s)`.

Spine lineal:
```
preflight_optimization → reflection(LLM) → context_compression(LLM) → semantic_cache_check
  ├─ hit  → assemble_plan
  └─ miss → plan_skeleton(LLM) → generate_days_parallel(LLM/día) → adversarial_judge(LLM)
            → self_critique(LLM eval + LLM corrector) → assemble_plan(determinista)
            → review_plan(LLM fact-check + tool clínico + LLM reviewer)
            → should_retry ─┬─ retry        → retry_reflection → plan_skeleton  (regen COMPLETA)
                            ├─ marker_regen  → surgical_marker_regen → assemble_plan (quirúrgica)
                            └─ end           → END (posible degradado)
```
El grafo genera **UN chunk (~3 días)**. Los planes multi-semana los ensambla `_chunk_worker` (`backend/cron_tasks.py`) fuera del grafo, llamando `run_plan_pipeline` por chunk (T1/T2 two-phase commit).

### 1.2 Los 10 roles LLM

1. Reflector meta-learning · 2. Compressor de contexto · 3. Plan skeleton · 4. Day generator (1 llamada/día) · 5. Adversarial judge · 6. Self-critique evaluator · 7. Self-critique corrector · 8. Fact-checker clínico · 9. Medical reviewer (único gate de seguridad, PRO) · 10. Corrector quirúrgico.

Conteo de invocaciones por corrida de UN chunk de 3 días: **~5 mínimo**, **~9-12 típico** (pagado+médico), **~80-90 techo** (3 intentos × adversarial × hedges × pro-fallbacks). Por plan de 7 días (~2 chunks): **~25-40 llamadas LLM (~30 típico)**.

### 1.3 El núcleo determinista es el que define precisión

- `assemble_plan_node` (`:24681`) corre `_apply_macro_engine` (`:24420`) → `portion_solver.solve_meal_macros` → `refine_day_portions_integer` (`portion_solver.py:496`) → caps clínicos → `_apply_portion_quantization`. **Re-dimensiona cada comida antes del reviewer.** Las cantidades del LLM se descartan.
- `compute_clinical_band_score` (`:33338`, "Cero LLM") mide la precisión.
- Coherencia receta↔lista: determinista (`shopping_calculator.py`: `expected_sum_from_recipes`, `compare_expected_vs_aggregated`, `run_shopping_coherence_guard`). Es el **único** surface determinista que puede forzar retry LLM (modo `block`).
- Enforcement clínico determinista: `condition_rules.CONDITION_RULES` + `clinical_constraints.ClinicalConstraintEngine`.

### 1.4 Precisión medida (baseline, `backend/docs/macro_precision_benchmark.md`)

| | kcal | proteína | carbos | grasas |
|---|---|---|---|---|
| MAPE (11 planes reales) | **0.5%** | **16.0%** | 12.6% | 12.2% |
| dentro ±10% | 100% | 48% | 52% | 55% |
| **4/4 macros ±10%** | — | **solo 24% de los días** | — | — |
| Tasa fallback a plan matemático | **45%** (9/20) | | | |

Causa raíz documentada: el solver es **greedy proporcional, no multi-restricción** (`portion_solver.py:204-255`); con ingredientes acoplados no clava los 4 macros a la vez. El doc prescribe: *"reemplazar por NNLS/LP (`scipy.optimize.nnls/linprog`, determinista, ~ms)"*.

### 1.5 Problemas estructurales

- **P1 — Opiniones LLM redundantes.** self_critique (`:8442`) y adversarial judge duplican lo que el determinista ya hace mejor. self_critique y el reviewer usan **detectores distintos** → self_critique declara "corregido" lo que el gate rechaza → regen completa. Raíz de: `P1-CRITIQUE-SAMEDAY-PROTEIN-PARITY`, `P1-FRUIT-DEDUP-GATE-PARITY`, `P1-CRITIQUE-CROSSDAY-DISH-PARITY`.
- **P2 — Whack-a-mole que entrega degradados.** `should_retry` (`:29969`) regenera desde skeleton; sale violación nueva; se agotan `MAX_ATTEMPTS=3`; entrega degradado por 6 ramas "end" (`critical`, `high_contextual`, `max_attempts`, `invalid_pipeline_start`, `budget_exhausted`, `approved_with_residual`). Caso canónico de no-convergencia `P1-REVIEW-COHERENCE-SEVERE-ONLY`: falso positivo por hueco de datos (`package_grams`/SKU faltante en proteínas rotativas Res→Cangrejo→Chivo).
- **P3 — Semana-1 no durable.** La generación síncrona SSE vive solo en task asyncio + guard wall-clock in-process (`routers/plans.py:3618`). Deploy/OOM/crash a mitad → plan perdido + `pending_pipeline:<user>` colgado en `generating`. El chunk system (semanas 2+) sí es durable (asimetría).
- **Síntoma:** sprawl de config (≈650-850 refs `MEALFIT_*` vs ~161 documentados; ~269 flags always-ON con ambas ramas retenidas). Cada curita de whack-a-mole se envió como flag permanente.

### 1.6 Lo que está sólido (NO tocar)

Chunk subsystem production-grade (máquina 7 estados, `FOR UPDATE SKIP LOCKED` + advisory lock por usuario, heartbeat, rescate zombies, backoff, dead-letter + cron recovery). Lost-update (I2/I6/I7/I8 + CHECK en DB). Observabilidad (~32 `system_alerts`, `/health/version`, deploy-lag). Medical reviewer como único gate de seguridad clínica.

---

## 2. Veredicto sobre cantidad de agentes

**RESTRUCTURAR — colapsar 10 → ~5-6, quitando el medio, conservando ambos extremos.**

| # | Rol | Veredicto | Justificación |
|---|---|---|---|
| 3 | Plan skeleton | MANTENER | Juicio LLM genuino (estructura proteína/plato por día) |
| 4 | Day generator | MANTENER | Única fuente de elección de alimentos + recetas |
| 8 | Fact-checker clínico | MANTENER | Valor clínico real; condicional a flags médicos |
| 9 | Medical reviewer | **NUNCA TOCAR** | Único gate de seguridad LLM (PRO risk-tier todos los tiers) |
| 10 | Corrector quirúrgico | MANTENER | Ruta barata de reparación puntual |
| 6/7 | Self-critique | **ELIMINAR → detectores deterministas** | Duplica `_trim_day_protein_to_ceiling`/`_protein_repeat_autofix`/`build_variety_report`; raíz de la clase de bug de paridad |
| 5 | Adversarial judge + 2º candidato | **Solo tiers pagados; OFF en gratis/guest; ganador por band-score** | Duplica todas las llamadas de day-gen; el juez elige "mejor" con criterio que el band-score mide mejor. (Decisión owner: conservar en pagos por calidad.) |
| 1/2 | Reflector, compressor | MANTENER (marginal) | Baratos, condicionales, bajo blast-radius |

**Piso inviolable:** `skeleton → day-gen → motor determinista → (fact-check si riesgo) → reviewer → reparación quirúrgica`. Matar reviewer/fact-checker = eliminar el único gate clínico (alergias/DM2/ERC/HTA). Inaceptable.

---

## 3. Decisiones del owner (registradas 2026-07-09)

1. **Secuencia:** Fase 0 (durabilidad) primero, luego precisión.
2. **Fase 2:** adversarial self-play se mantiene solo en tiers **pagados**; se elimina en gratis/guest. self_critique se elimina para **todos** (reemplazo determinista). *Asunción a confirmar en revisión:* la eliminación de self_critique igual se valida con canario contra el benchmark de 20 perfiles, gate conservador (all-4-macro no baja >2pp Y baja la tasa de degradados), reportando antes de promover.
3. **Proteína inalcanzable (ej. 2.8 g/kg):** clamp honesto + banner (owner dio OK clínico).
4. **Coherencia:** incluir backfill `package_grams`/SKU + reparación determinista add-to-list en Fase 3.

---

## 4. Opciones evaluadas

- **A — Status quo + más knobs.** RECHAZADA: no ataca causa raíz; el sprawl actual ES su costo acumulado.
- **B — LLM propone / determinista dispone / un gate. ELEGIDA.** Quita el medio, corrige raíces deterministas, hace durable la semana-1.
- **C — Single-pass radical.** RECHAZADA: pierde estructura/variedad del skeleton, pierde paralelismo/hedging, cae en el timeout >170s que motivó thinking-off.

---

## 5. Plan por fases

Cada fase es shippable de forma independiente y tendrá su **propio plan de implementación** (writing-plans) al momento de ejecutarla. Los detalles abajo son a nivel diseño.

### Fase 0 — Barandas de durabilidad (de-risk de todo lo que sigue)

> **ESTADO (2026-07-09): implementada parcialmente tras verificación.** La verificación de los 5 gaps revisó el scope:
> - ✅ **GAP 7 `P1-SSE-QUEUE-BOUNDED`** — `progress_queue` acotada + `asyncio.QueueFull` handling + `_done` drop-oldest. Knob `MEALFIT_SSE_PROGRESS_QUEUE_MAXSIZE` (1000). Test `test_p1_sse_queue_bounded.py`.
> - ✅ **GAP 4 `P1-GUEST-INFLIGHT-GUARD`** — guard in-flight keyed en session_id (409). Test `test_p1_guest_inflight_guard.py`.
> - ✅ **GAP 2 `P1-PIPELINE-CONCURRENCY-CAP`** — cap global de pipelines (503). Knob `MEALFIT_MAX_CONCURRENT_PLAN_PIPELINES` (8). Test `test_p1_pipeline_concurrency_cap.py`. Bumpeó `_LAST_KNOWN_PFIX`.
> - ❌ **GAP 5 (CB fail-closed) DESCARTADO** — habría sido peligroso: el CB ya tiene backpressure in-process (`_local_healthy`); fail-closed en outage de storage bloquearía el 100% de la generación. NO invertir el fail-open de storage.
> - ⚠️ **GAP 3 (semana-1 durable) — deploy-safety YA resuelto** por `P0-PENDING-PIPELINE-STARTUP-SWEEP`. Falta solo el RESUME durable real (esfuerzo grande, aplazado a su propio spec) + tweak de notificación (rompe decisión anti-doble-toast → dejado al owner).
>
> Follow-up: GAP-2 solo cubre `/analyze/stream` (vector real vía create_task); `/analyze` no-stream (await síncrono) queda pendiente.

**Cambios (diseño original, para referencia):**
- **Semana-1 durable (GAP 3):** registro recuperable para la generación primaria; un cron de recovery marca `failed` + re-encola o empuja al usuario, en vez de spinner colgado por horas. (`routers/plans.py:3618` es el guard in-process actual.) Reusar el sentinel `_sse_completed_naturally` para no doble-persistir.
- **Cola SSE acotada (GAP 7):** `asyncio.Queue(maxsize=...)` en `routers/plans.py:3241`.
- **Guest 409 + dedup (GAP 4):** aplicar el guard de pipeline activo + `P1-DEDUP-RECENT-PLAN` a guests (hoy gateado en `_deep_search_user_id`, `:3116-3176`). Evita doble gasto DeepSeek en reload.
- **CB fail-CLOSED (GAP 5):** `LLMCircuitBreaker.can_proceed` con cooldown local corto cuando Redis **y** DB caen (`:2038,:2249`), para sheddear carga en outage correlacionado en vez de martillar al provider.
- **Semáforo de concurrencia (GAP 2 mitigación):** límite global de arranques de pipeline + shed por degradación de provider (Python no puede matar los threads de timeout fugados → acotar la cantidad).

**Riesgo:** bajo-medio. La recovery durable es lo más involucrado pero es aditivo.

### Fase 1 — Solver de precisión (NNLS/LP) ⭐ ~~la palanca de precisión #1~~

> **ESTADO (2026-07-09): OBSOLETA — NO implementar. La verificación mostró que ya está hecho + resuelto.**
> - `_box_lsq` (`portion_solver.py:111`, `[M2-SOLVER-NNLS · 2026-06-14]`) YA es el solver box-constrained multi-restricción (equivalente NNLS/LP, pure-python, sin scipy). `SOLVER_LSQ` default True.
> - `_rebalance_day_macros_to_target` (`[P3-MACRO-REBALANCE · 2026-06-19]`, `MACRO_REBALANCE_ENABLED` default True, wired en `_apply_macro_engine:24639`) llevó la precisión a **all-4 ~87-94%, proteína MAPE 1.3-2.8%** (medición viva 2026-06-21). `refine_day_portions_integer` (integer 5g joint) también vivo.
> - El `docs/macro_precision_benchmark.md` que motivó esta fase (24% / 16% MAPE / "reemplazar por NNLS/LP") es de **2026-06-14, PRE-fix** — stale. La precisión de macros está **en su techo físico** (cota de porción cocinable ~85-90%). scipy NO está instalado y **no hace falta**.
> - **Corrección honesta:** mi síntesis original sobre-ponderó ese doc stale. La precisión de MACROS no es una palanca abierta. Ver memoria `project_macro_benchmark_baseline`.
>
> **Reorientación:** si "precisión" significa (a) no entregar planes DEGRADADOS → es la **Fase 2** (quitar la capa LLM del medio); (b) coherencia receta↔lista + platos compuestos no-resolubles (sancocho/mangú) → **Fase 3**; (c) **micronutrientes** (panel advisory, cobertura parcial) → eje nuevo separado. Los macros ya están resueltos.

**Objetivo (diseño original, OBSOLETO):** bajar proteína de 16% MAPE; subir el 24% all-4-macro. Determinista.
**Cambios:**
- Reemplazar el solver greedy/`_box_lsq` por NNLS/LP (`scipy.optimize.nnls`/`linprog`) como optimizador multi-restricción, preservando la elección de alimentos del LLM y re-escalando cantidades.
- Re-reconcile **después** de la cuantización (hoy la cuantización corre al final y reintroduce deriva sin re-reconcile — `benchmark` causa raíz #3).
- Contabilizar ingredientes no-resueltos (mangú/sancocho/moro caen fuera de las filas del solver — causa raíz #4).
**Gate de aceptación:** correr `scripts/benchmark_macro_compliance.py` (20 perfiles); proteína MAPE baja de forma medible y all-4-macro sube, sin regresión de kcal.
**Riesgo:** bajo (determinista, con benchmark existente como gate). Añade dependencia `scipy` si no está.

### Fase 2 — Restar la capa LLM del medio (la respuesta a "¿reducir agentes?")

> **ESTADO (2026-07-09): verificada + Pasos 1-5 implementados (self_critique NO se apaga aún).**
> Verificación (4 verificadores sobre código vivo) corrigió el plan naíf "quitar self_critique":
> - **self_critique NO es removible tal cual (RISKY_KEEP):** aún posee 3 detectores sin equivalente en el gate del reviewer — `_detect_slot_incoherence` (almuerzo↔cena carbo, merienda-plato-fuerte, heavy-protein multi-slot) y `_count_staple_repetitions` (staple cross-día). Para sanos/guest el reviewer LLM se bypasea (línea 28204) → quitarlo sin portar = punto ciego. Los fixes de paridad de julio solo alinearon 2 de ~5 dimensiones.
> - **adversarial NO está ON por defecto** (solo auto-activación rara por historial); gatearlo a solo-pagados = cero pérdida funcional para free/guest.
> - **Canario:** el harness determinista es CIEGO a la capa de crítica → solo cohorte live puede medir el Paso 6.
>
> **Implementado (TDD, todo por knob, mostly-inerte):**
> - **Paso 1 `P1-ADVERSARIAL-PAID-ONLY`** (`MEALFIT_ADVERSARIAL_PAID_ONLY` def True): self-play solo tiers pagados. Test `test_p1_adversarial_paid_only.py`. *(única mejora de comportamiento live: ahorro para free/guest.)*
> - **Paso 2 `P1-SELF-CRITIQUE-MASTER-KNOB`** (`MEALFIT_SELF_CRITIQUE_ENABLED` def True): kill-switch del nodo. Test `test_p1_self_critique_master_knob.py`. *(inerte a default.)*
> - **Paso 3:** baseline de tests de critique — 14 fallos pre-existentes son WIP del owner (surgical return-key + prompt de day_generator), NO míos.
> - **Paso 4 `P1-SLOT-INCOHERENCE-GATE` / `P1-STAPLE-REPEAT-GATE`** (ambos def **OFF**): porta los 3 detectores únicos al gate del reviewer + ruta quirúrgica (slot-incoherence day-attributable; staple cross-día → retry completo). Test `test_p1_slot_incoherence_gate.py`. *(inerte hasta canario.)*
> - **Paso 5 `P1-SELF-CRITIQUE-CANARY`** (`MEALFIT_SELF_CRITIQUE_CANARY_PCT` def 0): bucketing determinista + tag `self_critique_cohort` en la métrica `clinical_band`. Test `test_p1_self_critique_canary.py`. *(inerte a 0%.)*
>
> **Paso 6 (apagar self_critique) PENDIENTE — requiere:** (a) flip los gates portados ON + observar retry-rate, (b) `SELF_CRITIQUE_CANARY_PCT`=20 y leer cohorte OFF vs ON, (c) **firma del owner del umbral** (degraded-ship per-cohorte + spot-check manual de diversidad, sin métrica automática), luego default OFF, luego borrar nodo.

**Objetivo (diseño original):** 10 → ~5-6 roles; matar la clase de bug de paridad y el costo adversarial en gratis/guest.
**Cambios:**
- Eliminar self_critique (#6/#7, `:8442`); enrutar sus checks por los MISMOS detectores deterministas que usa el reviewer (`build_variety_report`, `_detect_slot_appropriateness`, `_trim_day_protein_to_ceiling`) → un SSOT, sin discrepancia LLM-vs-LLM. Elimina en raíz la familia `P1-CRITIQUE-*-PARITY`.
- Adversarial self-play (`use_adversarial`, `:7223`): **OFF en gratis/guest, ON en pagos**. Cuando existan 2 candidatos, elegir ganador por `compute_clinical_band_score` en vez del `adversarial_judge` LLM (#5).
- **Canario** con el mecanismo `macro_rollout`; gate = all-4-macro no regresa (>2pp) Y baja la tasa de degradados. Reportar antes de promover.
- Solo tras canario OK: borrar knobs muertos + ramas duales de los pasos removidos.
**Riesgo:** medio. Riesgo de cobertura si los detectores deterministas no cubren un caso que self_critique atrapaba → mitigado por conservar el corrector quirúrgico (#10) y correr detrás de flag con canario. **Necesita firma del owner del umbral de regresión.**

### Fase 3 — Raíz de coherencia (no curita)

> **ESTADO (2026-07-09): esencialmente DONE tras verificación — NO hay trabajo de código claramente justificado.**
> - **Platos compuestos "0-silent uncounted": YA RESUELTO.** `_compound_dish_lookup` (60 platos criollos en `dominican_dishes.json`, `nutrition_db.py:479-557`) + abstención grácil del solver (`_apply_macro_solver_to_meal` conserva macros del LLM si coverage<floor) + telemetría multi-capa (`_maybe_mark_low_resolution_degraded` floor 0.7, drift cron). Coverage prod medido: **avg 0.936, min 0.878, cero planes bajo el floor**. Era un ítem stale en este spec.
> - **Whack-a-mole coherencia: mayormente cerrado** por `MEALFIT_REVIEW_COHERENCE_BLOCK_SEVERE_ONLY` (default True, deployado 2026-07-09).
> - **Backfill `package_grams`: MISCARACTERIZADO.** El guard **nunca lee `package_grams`** (`shopping_calculator.py:3594` es artefacto de display/SKU-picker). Para carnes vendidas a granel (Res/Cangrejo/Chivo/Pulpo), NO tener contenedor es la representación CORRECTA, no un defecto. **No backfillear.**
> - **Residual A (unit-phantom recipe-g vs list-lbs): ya cubierto.** El converter está ON por default (`_get_coherence_unit_converter_enabled()` → True; el docstring "canary" es stale) y la tabla de tokens (`lb/lbs/oz/kg/g/libra(s)/onz`) es completa (`shopping_calculator.py:2139-2141`). Un "Step 1" de hardening sería belt-and-suspenders sin valor real.
> - **Residual B (missing-food genuino se entrega en silencio): ÚNICO residual real, EVIDENCE-GATED.** El severe-only exige count≥2 o magnitud≥50%, así que un único alimento genuinamente ausente ahora degrada a warn + se entrega (con telemetría + cron diario P3-B). El fix correcto NO es re-escalar presence→retry (re-abre el whack-a-mole) sino un add-to-list determinista — pero **construirlo requiere que el owner primero saque los registros reales de `_shopping_coherence_block_history`/logs VPS** para confirmar que casos genuinos (no falsos-positivos) ocurren. Podría ser un tradeoff de producto aceptable (entregar el raro miss genuino + cron lo caza). **No construir especulativamente.**
>
> **Conclusión:** Fase 3 no tiene trabajo de código claramente justificado. Lo único abierto (Residual B) necesita evidencia de prod del owner para decidir si vale construirlo.

**Objetivo (diseño original):** matar el whack-a-mole de coherencia que nunca converge.
**Cambios:**
- **Backfill `package_grams`/SKU** para proteínas menos comunes (Res/Cangrejo/Chivo/Pulpo) — raíz del falso positivo de coherencia rotativa.
- **Reparación determinista add-to-list** en `run_shopping_coherence_guard` (`shopping_calculator.py`): un ítem de coherencia faltante se arregla **en la lista**, no regenerando el plano completo.
- **Proteína inalcanzable → clamp honesto + banner** (decisión owner): el motor fija proteína a lo alcanzable dentro de las kcal y muestra aviso; deja de ser sumidero de retries.
- Extender telemetría per-pass para medir precisión futura.
**Gate:** validar contra `test_p1_shopping_recipe_coherence.py` (89 casos).
**Riesgo:** bajo-medio. Backfill es esfuerzo manual de datos, riesgo técnico bajo. add-to-list cambia salida de shopping — validar con la suite de coherencia.

### Fase 4 — Consolidación de config
**Objetivo:** reducir la carga de razonar ~650+ knobs.
**Cambios:**
- Promover flags always-ON estables a código incondicional y borrar la rama OFF.
- Retirar knobs huérfanos por las remociones de Fase 2.
- Regenerar `knobs_reference.md` desde `get_knobs_registry_snapshot()` (hoy ~161 vs real ~650-850, stale ~4.5x).
- Conservar solo knobs de rollback genuinamente operacionales (modo coherence guard, overrides de modelo, toggles de thinking).
**Riesgo:** bajo (mecánico, con tests). Hacerlo **después** de Fase 2-3 para borrar knobs cuyas capas ya no existen.

### Fase 5 — Escala horizontal + aislar generación del serving
**Objetivo:** quitar el techo/SPOF de `--workers 1` (`app.py:936`).
**Cambios:**
- Mover la generación fuera del proceso de serving — `--workers N` (el `LLMCircuitBreaker` ya está hecho para multi-worker: Redis INCR + fallback DB, hoy sin usar a 1 worker) o un pool de workers dedicado consumiendo la cola durable de Fase 0.
- Externalizar el estado in-process restante (registries de cancel/task, leader-election del scheduler — ya usa advisory lock pg).
- Re-verificar timing de heartbeat/zombie-rescue (`CHUNK_LOCK_STALE_MINUTES=3`) bajo el nuevo modelo de proceso.
**Riesgo:** alto — la más profunda, va última. Depende de Fase 0 (durabilidad) y Fase 4 (menos interacciones de flags in-process). **Necesita forecast de tráfico + decisión de gasto de infra del owner.**

---

## 6. Ítems abiertos / a confirmar

- **Umbral de regresión de calidad para Fase 2:** el owner eligió conservar adversarial en pagos; falta firmar el delta exacto de all-4-macro / degraded-ship-rate aceptable en canario para promover la eliminación de self_critique. Asunción de trabajo: all-4-macro no baja >2pp Y baja la tasa de degradados.
- **Forecast de tráfico/concurrencia:** define la urgencia de Fase 5 (GAP 1). A volumen bajo puede esperar; un push de marketing lo vuelve urgente.
- **Dependencia `scipy`:** confirmar que puede añadirse al backend (Fase 1). Si no, implementar NNLS box-constrained a mano (coordinate descent ya existe en `_box_lsq`, extenderlo a multi-restricción).
- **`--workers N` / infra multi-proceso (Fase 5):** el código ya está listo (CB Redis, scheduler advisory lock), pero cambia deployment y costo. Confirmar antes de invertir.

---

## 7. Notas de convención del repo (aplican a la implementación futura)

- Cambios de comportamiento reversibles sin redeploy → knob `MEALFIT_*` con default seguro, registrado en `_KNOBS_REGISTRY` (gates nuevos nacen OFF).
- Sin DDL en runtime → toda migración en `migrations/` **y** `backend/migrations/` (SSOT dual sincronizado).
- Tests que parsean source → incluir tooltip-anchor.
- Bumpear `_LAST_KNOWN_PFIX` (`backend/app.py:32`) por cada cierre de P-fix, con test de regresión cross-linkeado por slug.
- Commit SCOPED con pathspec (nunca `-A`), push, `deploy-mealfit.ps1`, avisar "Clear site data".
