# Benchmark del landing — matriz clínica del formulario + guía de mejora

[P1-LANDING-BENCH-1 · 2026-08-07] Doc canónica del benchmark cuyo output alimenta las cifras
públicas del landing y sirve de brújula para mejorar el motor (generación, swap individual,
regeneración de día). Motor SSOT: [`landing_benchmarks.py`](../landing_benchmarks.py) · runner
[`scripts/landing_benchmark.py`](../scripts/landing_benchmark.py) · test ancla
[`tests/test_p1_landing_bench_1_anchors.py`](../tests/test_p1_landing_bench_1_anchors.py).

## Alcance de país — **el benchmark clínico queda scoped a RD**

[P3-COUNTRY-DOC-TRUTH · 2026-08-22] La spec del sistema de países declaró esta limitación como
aceptada de v1 **con la condición explícita de anotarla aquí**, y nadie la anotó: hasta hoy este
documento no mencionaba la palabra «país» ni una vez. Una limitación que nadie defendió por escrito
vuelve sin que nadie lo note — este repo ya lo pagó con `P2-VISION-COUNTRY`, aceptado bajo la misma
clase de condición incumplida.

Qué significa en concreto:

- **Los 25 perfiles clínicos se evalúan TODOS como dominicanos.** `safety`, `gym`, `latency` y
  `changes` no tienen eje de país: sus cifras describen el motor sirviendo a un usuario de RD. Una
  regresión que sólo afecte a España o a México **no movería una sola de estas cifras**.
- **Las cifras públicas del landing heredan ese alcance.** Lo que el landing afirma con estos
  números es cierto para RD y no está medido para los cinco países beta.
- **El modo `structural` SÍ tiene eje de país** desde `P2-LANDING-BENCH-COUNTRY`: el bloque
  `structural_facts_por_pais` cuenta el catálogo comprable y las reglas por cada país del selector,
  y es gratis (no consume LLM). Es lo primero que hay que mirar antes de afirmar nada por país.
- **Salir de este alcance es un proyecto, no un flag**: exige perfiles clínicos por país y una
  decisión sobre qué significa «plan correcto» en cada cocina. Mientras no exista, cualquier claim
  del landing sobre países beta necesita otra fuente.

## Por qué existe (y qué hueco cierra)

Los benchmarks previos miden el motor con perfiles que el formulario actual **no puede producir**:

| Harness | Perfiles | Hueco |
|---|---|---|
| `scripts/benchmark_macro_compliance.py` (nightly) | 20 held-out, condiciones en texto libre (`"Diabetes tipo 2"`, `"Enfermedad renal crónica"`) | 0 medicamentos, 0 alergias, 0 veg*, texto libre que el wizard ya no emite |
| `scripts/plan_gym.py` (7 ejes) | los mismos 20 | ídem; no puntúa seguridad clínica per-se |
| `pipeline_metrics` `change_swap`/`change_regen_day` (P1-CHANGE-OUTCOME-TELEMETRY · 2026-08-05) | prod real | serie recién nacida; sin benchmark held-out de cambios |

Desde P1-MEDICAL-CONDITIONS-CAP (2026-08-01) el wizard emite **solo chips cerrados**: 7 condiciones
(+ Embarazo/Lactancia si `gender=female`), 14 medicamentos, 6 alergias, 3 dietas. Este benchmark
ejercita EXACTAMENTE ese espacio — cada chip literal, con la forma de payload que envía `Plan.jsx`.

## Los 5 modos

| Modo | Necesita | Qué mide | Secciones del JSON |
|---|---|---|---|
| `structural` | nada (DB opcional) | hechos contables: reglas clínicas, micros DRI, catálogo | `structural` |
| `live [N] --conc 2 [--changes] [--save-plans] [--provider openai]` | claves LLM + Neon | genera los planes reales de la matriz (N=0 por defecto = **los 25**) y puntúa seguridad + nutrición + gym + latencia + entrega; `--changes` ejercita swap individual y bucle de día. `--provider openai` fuerza a OpenAI los 4 knobs del pipeline (`_OPENAI_FORCE_KNOBS`: GPT-6 Luna en `MEALFIT_FLASH_MODEL`, `MEALFIT_MODEL_FREE_TIER`, `MEALFIT_MODEL_PAID_TIER`, `MEALFIT_PRO_MODEL`); requiere `OPENAI_API_KEY`, fail-loud sin ella. NO reintroduce el override global eliminado (P1-SINGLE-PROVIDER-RESTORE): reviewer/day-gen/swap conservan su routing propio por tier (OpenAI por defecto, pero un knob per-feature del entorno lo cambia), así que la corrida **no** afirma que ningún nodo use GLM | `run`, `structural`, `safety`, `nutrition`, `gym`, `latency`, `reliability`, `changes` |
| `remote [N] --api-base URL [--conc 2] [--changes] [--save-plans]` | **cero claves** (solo red al deploy) | la corrida «cuenta de invitado»: genera contra el API desplegado como `user_id=guest` (N=0 por defecto = **los 25**) y puntúa LOCALMENTE (los scorers son funciones puras). El routing de modelos lo decide el SERVIDOR con los knobs de su deploy ([`llm_tier_routing.md`](llm_tier_routing.md)); el reporte no afirma un modelo: guarda el `/health/version` AL EMPEZAR y AL TERMINAR (`meta.server_version.inicio/fin`, con `git_sha`) y de ahí sale `run.engine_commit` (ver `run`). `--changes` ejercita solo swap (regenerate-day exige plan persistido con auth). Respeta el RateLimiter de `/analyze` (3/60s por IP): el workflow corre con conc 2 (default del CLI 1), backoff ante 429 | `run`, `meta`, `safety`, `nutrition`, `gym`, `latency`, `reliability`, `changes` |
| `telemetry --days 30` | Neon | series de PROD, **solo lo entregado**: éxito de cambios a la primera; banda de las filas de entrega (`pre-INSERT` + `chunk-T1 semana N`) **emparejadas con la corrida que las produjo**; latencia de esas corridas, por tipo (plan inicial / bloque en segundo plano); fallback de los planes persistidos; lo que no se empareja, aparte (`banda_excluida_sin_corrida`, `corridas_por_entrega`); PQI, costo por nodo | `run`, `telemetry` |
| `score --plans f.json` · `score --plans-glob G --forms-root R` | nada (sin LLM) | re-puntúa planes guardados: los de una corrida `live/remote --save-plans` (el denominador sale de `attempted_ids`) o un **corpus real** (un fichero por plan con `final_plan`; el formulario del propio fichero o, con la convención `<dir>__<fichero>.json` de `cola_corpus`, de `<R>/<dir>/<fichero>.json`; perfil con la MISMA unión de texto libre que producción —`corpus_profile`— y expectativas clínicas por `derive_expectations`) | `run`, `safety`, `nutrition`, `gym` |

```bash
# desde backend/, con .env cargable
python scripts/landing_benchmark.py structural
python scripts/landing_benchmark.py live 5 --conc 2 --changes --save-plans     # smoke: cohorte partial
python scripts/landing_benchmark.py live --provider openai --conc 2               # los 25
python scripts/landing_benchmark.py remote --api-base https://app.bioboros.com --conc 2 --changes --architecture v2.2
python scripts/landing_benchmark.py telemetry --days 30
python scripts/landing_benchmark.py score --plans landing_plans_1234.json
python scripts/landing_benchmark.py score --plans-glob "/tmp/cola/*.json" --forms-root /tmp
```

Costo estimado de `live`/`remote` completo (25 perfiles): la generación mide hoy mediana ~6 min
y p90 ~15 por plan, así que con `--conc 2` son ~2-3 h — el workflow remote tiene techo de 330 min
(antes 170, que no cabía). El gasto de proveedor es el del deploy o de las claves locales —
correr de madrugada RD como el nightly. `--changes` añade ~6 llamadas de swap por perfil.

## Reporte schema v2 — trazabilidad, nutrición y entrega (P1-PLAN-LOTE-749)

[P1-PLAN-LOTE-749 · 2026-09-28] El reporte pasa a `schema_version: 2`, el formato que importa el
landing (`bioboros-cinematic/benchmark_import.py` + `contract/benchmark-v22.json`). Test:
[`tests/test_p1_plan_lote_749.py`](../tests/test_p1_plan_lote_749.py).

- **`run`** (siempre): `id`, `mode`, `architecture` (se DECLARA con `--architecture v1|v2.2`; por
  defecto `unspecified`, que el importador rechaza — no se adivina), `protocol_version`
  (`LANDING_BENCHMARK_PROTOCOL_VERSION` = `2.2-prep.1`, el `protocol.version` congelado del landing),
  `started_at`/`finished_at` UTC, `source_commit` (`git rev-parse HEAD`) y `source_dirty` (`git status`
  al ARRANCAR; `None` = no verificable, el importador lo rechaza igual que sucio — las salidas del propio
  benchmark están en `.gitignore` y los workflows mandan el stdout a `$RUNNER_TEMP`, porque un
  `tee run_stdout.txt` en el checkout ensuciaba TODA corrida de Actions), `source_commit_role` (de
  quién es ese commit: `engine_and_scorers` en live, `scorers` en remote/score, `queries` en
  telemetry), `engine_commit` + `engine_commit_status` (el motor MEDIDO: en live el mismo commit,
  `in_process`; en remote el `git_sha` del servidor solo si coincide al empezar y al terminar,
  `verified` — si no, `not_exposed`, `changed_during_run` o `unreachable` y `engine_commit=None`), `country_scope`,
  `full_profile_count` (25), `profile_count` (intentados), `cohort_status` (`complete` solo si se intentó
  la matriz ENTERA; `partial`; `not_applicable` en structural/telemetry/corpus), `publication_status`
  (siempre `candidate`: publica una persona) y `parameters` (solo los que el modo usa, con la
  concurrencia efectiva).
- **`nutrition`** (live/remote/score), sobre el plan ENTREGADO con funciones puras
  (`score_plan_nutrition` / `aggregate_nutrition`): por día y macro (kcal, proteína, carbos, grasas) el
  total RECALCULADO desde las comidas contra el objetivo de la cabecera del plan (`calories`/`macros`,
  que el motor fija al objetivo). `per_macro_mape_pct` agrega por DÍA evaluado; `macro_mape_pct` es la
  media de los 4; `worst_macro_mape_pct` el mayor; `four_macros_in_band_pct` = días con las 4 celdas en
  la **banda del motor** (`engine_band_definition()`: P/C/G en `[BAND_SCORE_LOWER, BAND_SCORE_UPPER]`,
  kcal `[0.95, 1.05]` con techo `GAINMUSCLE_KCAL_BAND_UPPER` en ganancia muscular). La paridad con
  `compute_clinical_band_score` la ancla el test (y el replay del 28-sep: 414/414 planes entregados
  únicos idénticos; el corpus reciente es un subconjunto del completo). kcal se suma SOLO de `cals`,
  como el motor, y las copias de sus helpers tienen test de igualdad.
- **`reliability`** (live/remote): entrega = entregados / intentados; fallback = fallbacks entregados +
  descartados / intentados; `latency_all_s` incluye los fallos terminales (contrato B-04) y
  `latency_delivered_s` solo lo entregado. Un plan con `_is_fallback` sin `_partial_repair` es
  `discarded_fallback` (el FALLBACK-GUARD del router lo descarta con 422/503): **ya no se puntúa como
  entregado** en `live`. En `remote` se mira el `detail`: 422 con `detail` de texto (rechazo crítico) o
  503 «IA saturada / no disponible» → `discarded_fallback`; 422 con `detail.code` de validación
  (`missing_required_fields`, `budget_insufficient`, `clinical_scope_exceeded`…) →
  `rejected_request` (`n_rejected_request`, no infla el fallback); 503 «no pudimos guardarlo»,
  `server_busy_generating` o de nginx → `error`.
- **`telemetry`** cuenta solo lo entregado (ver la tabla de modos) — antes `banda_entregada` mezclaba
  269 filas `assemble-tail` (intermedias) con 81 `pre-INSERT` en 30 días. Y filtrar por superficie
  no bastaba (ronda 1): de esas 81, **67 no eran entregas** — todas del 2-7 sep, `user_id` NULL y
  sesión `post-finalize`, sin corrida `clinical_band` ni `meal_plans` detrás (lo que deja cualquier
  llamada a `_finalize_plan_data_for_insert` fuera de una generación). Ahora cada fila de entrega se
  empareja con la corrida del MISMO usuario ≤5 min antes (para `pre-INSERT`, que no guarda
  `user_id`, por la fila de `meal_plans` de ese usuario); el tipo sale de la corrida (desde el
  lifecycle 2.5 un plan inicial también se entrega por `chunk-T1`). Medido el 28-sep (30 días, solo
  SELECT): banda plan inicial **n=19, media 0,904** (antes n=81, 0,954), bloque posterior n=7, 0,977;
  67 filas sin corrida aparte (0,964); latencia plan inicial **p50 226 s, p95 456 s (n=19)**, bloque
  posterior p50 436 s (n=7) — antes 49 corridas mezcladas, p50 347 s; corridas sin entrega: 1 de 20
  iniciales, 15 de 22 bloques (reintentos), 7 de 7 de invitado (su plan no se persiste: no hay fila
  que lo pruebe, no entra en la latencia).
- **`--save-plans`** guarda también `attempted_ids` y el estado de entrega, para que `score` reproduzca
  el denominador sin pagar LLM.

### Replay gratis del 28-sep (corpus de baterías, sin LLM, en el VPS)

`score --plans-glob "/tmp/ia6d_wf_cola744/*.json" --forms-root /tmp` (y `ia6d_wf_cola744_rec/*.json`
por separado para el corpus reciente) sobre los planes ya pasados por
la cola del lote 744 (contrato de receta + pulido). Mide el DATO del plan; **no** es la matriz: son
perfiles de baterías (bloques de 3 días casi siempre, varios adversariales), así que no sustituye a la
corrida pagada — sirve para decidir si pagarla.

| corpus | planes (entregados) | violaciones | MAPE kcal · P · C · G | media · peor | 4-en-banda (días) | gym |
|---|---|---|---|---|---|---|
| reciente (`_rec`) | 67 (67) | 0 en 848 comidas | 2,56 · 4,12 · 4,34 · 5,08 | 4,03 · grasas 5,08 | 85,8 % (204) | 90,1 |
| completo | 426 (414; 12 fallbacks `medical_critical` descartados) | 4 alérgenos en 2 planes del 25-sep | 2,85 · 4,42 · 4,40 · 5,18 | 4,21 · grasas 5,18 | 83,1 % (1263) | 88,5 |

Las 4 violaciones, leídas una a una: `salsa de soya` en un plan sin gluten (rd248) y `Harina de
Negrito` ×3 a un perfil con alergia a mariscos y gluten (rd252 `mariscos_gluten`). En el corpus
reciente, cero. La sonda negativa (inyectar `camarones` o `queso cheddar` en un plan con alergia a
mariscos y lácteos) marca las dos.
Contra el landing publicado (`frontend/src/data/benchmark.js`, N=8, junio: MAPE P 1,5 · kcal 2,0 ·
G 3,1 · C 3,2; 4-en-banda 91,7 %) el replay sale peor en las cinco cifras. El eje `micros` del gym
(~50) lee el panel persistido, que la cola no recalcula: no concluir nada de él con este corpus.

**Ronda 1 (28-sep): texto libre.** El modo corpus arma ahora el perfil con la misma unión de
«Otra…» que producción (`corpus_profile` → `profile_with_free_text`). Re-corrido sobre los dos
corpus con el árbol de la rama (sin IA, sesiones Postgres de solo lectura): **las cinco cifras, las
4 violaciones y el gym no cambian** (en rd252 `mariscos_gluten` cambia solo el sinónimo citado,
`negrito` ↔ `harina de negrito`, sobre los mismos 3 ingredientes: el escáner lo elige por el orden de
un set, que varía con `PYTHONHASHSEED` — comprobado con 4 semillas). Lo nuevo es que la corrida NOMBRA
el texto que producción descarta por el centinela «Ninguna» (P0-FORM-1) en
`run.parameters.texto_libre_descartado`: 12 planes del corpus completo — 11 celíacos con
`otherConditions: "celiaquía"` junto a «Ninguna» (llevan el chip `Gluten`, así que el guard de
alérgeno sigue activo) y `rd252__frutos_del_mar_textol252`, con `otherAllergies: "frutos del mar"`
junto a «Ninguna»: **su plan sirve «250 g de camarones cocidos»** y, por la regla P0-FORM-1, no
cuenta como violación. Pendiente de revisión humana (¿puede el wizard real mandar esa combinación?).

## Matriz de perfiles (cobertura del formulario)

25 perfiles en `build_landing_profiles()`. Invariantes ancladas por test: **cada** chip de
condición, medicamento y alergia aparece ≥1 vez; las 3 dietas y los 4 objetivos aparecen;
Embarazo/Lactancia solo en perfiles `female` (regla del wizard); máx. 3 condiciones reales
(cap del wizard, embarazo exento).

| id | label | condiciones | medicamentos | alergias/dieta | qué prueba |
|---|---|---|---|---|---|
| 1-2 | baseline_m/f | — | — | — | referencia de precisión sin capa clínica |
| 3 | dm2_metformina | Diabetes T2 | Metformina | — | sustituciones glucémicas + advisory B12 |
| 4 | hta_losartan_hctz | Hipertensión | Losartán, Hidroclorotiazida | — | subs de sodio + diurético depletor |
| 5 | dislipidemia_estatina | Colesterol Alto | Atorvastatina | — | subs de grasa saturada + estatina |
| 6 | gastritis_ibp | Gastritis | Omeprazol | — | referral gastritis + IBP |
| 7 | sop | SOP (PCOS) | Metformina | — | advisory SOP |
| 8 | hipotiroidismo_levo | Hipotiroidismo | Levotiroxina | — | timing-sensitive (Ca/Fe/soya) |
| 9 | bariatrica | Cirugía Bariátrica | — | — | **≥5 tomas/día** (claim del landing) |
| 10-11 | embarazo / lactancia | Embarazo / Lactancia | — | — | guard de mercurio |
| 12 | combo_cap3 | DM2+HTA+Colesterol | Metformina, Lisinopril, Atorvastatina | — | tope de 3 condiciones del wizard |
| 13 | warfarina_vitk | Hipertensión | Warfarina | — | `vitamin_k_consistency` (estabilidad INR) |
| 14 | potasio_doble | Hipertensión | Espironolactona, Lisinopril | — | doble potasio-elevador |
| 15 | insulina_hipoglucemia | Diabetes T2 | Insulina, Glibenclamida | — | **≥5 tomas/día** (claim del landing) |
| 16 | polifarmacia_gota | Hipertensión | Amlodipina, Prednisona, Alopurinol | — | 3 reglas de medicación simultáneas |
| 17 | alergias_lacteo_gluten_huevo | — | — | Lacteos, Gluten, Huevo | scan C2 de alérgenos |
| 18 | alergias_mar_nuez_soya | — | — | Mariscos, Frutos Secos, Soya | scan C2 + goal performance |
| 19 | vegetariana | — | — | vegetarian | P1-DIET-HARD-GUARD |
| 20 | vegana_dm2 | Diabetes T2 | Metformina | vegan | cruce dieta estricta × condición |
| 21 | renal_hta | Enfermedad Renal, Hipertensión | Losartán | — | precedencias dm2+renal / hta+renal |
| 22 | anemia_ferropenica | Anemia | — | — | regla `anemia` |
| 23 | gota_alopurinol | Gota / Ácido Úrico | Alopurinol | — | regla `gout` (condición + fármaco) |
| 24 | higado_graso | Hígado Graso | — | — | regla `nafld` |
| 25 | imao_tiramina | — | Antidepresivo IMAO | — | tiramina ↔ IMAO (crisis hipertensiva) |

## Métrica → claim del landing (pipeline de publicación)

El benchmark **nunca escribe** en el landing. El flujo es: correr → revisar JSON → decisión del
dueño → editar el SSOT frontend → los guard-tests validan. Los SSOT frontend son DOS:

- [`frontend/src/data/benchmark.js`](../../frontend/src/data/benchmark.js) — cifras **medidas**
  (MAPE, en-banda, versus). Guard: `test_p1_paper_benchmark_ssot.py`.
- [`frontend/src/data/systemFacts.js`](../../frontend/src/data/systemFacts.js) — hechos
  **estructurales** (17 micros, 200+ alimentos, 3-6 comidas, ciclos 7/15/30). Guard: sección
  de-drift de `test_p1_landing_bench_1_anchors.py`. Se refrescan con el modo `structural`.

| Métrica del reporte | Claim del landing que alimenta | Estado |
|---|---|---|
| `structural.micronutrientes_dri` | «17 micronutrientes vs DRI» (`systemFacts.MICROS_TRACKED`) | derivado de `micronutrients.dri_targets` |
| `structural.alimentos_catalogo` | «200+ alimentos verificados» (`systemFacts.VERIFIED_FOODS_LABEL`) | medido 252 (2026-07-02); label público redondea abajo |
| `safety.plans_sin_violaciones_pct` | CAPS «Se ajusta a tus condiciones» — pasar de capacidad a **cifra medida** | pendiente de 1ª corrida live |
| `safety.min_meals_compliance_pct` | «5-6 tomas en hipoglucemia, insulina o cirugía bariátrica» (FeaturesPage) | pendiente de 1ª corrida live |
| `changes.swap.ok_pct` / `telemetry.changes` | futura cifra «cambios de plato que salen a la primera» | serie prod nació 2026-08-05 |
| `latency.generation_s` / `telemetry.generacion_latencia` | «Normalmente de 4 a 5 minutos» (FAQ /como-funciona) — hoy SIN fuente | verificar antes de mantenerlo |
| `nutrition.aggregate.{macro_mape_pct, worst_macro_mape_pct, four_macros_in_band_pct}` | pilar B-01 del contrato v2.2 del landing (`benchmark_import.py`) — reemplaza a `gym.banda` como fuente | desde P1-PLAN-LOTE-749 |
| `gym.aggregate.banda` + nightly MAPE | `benchmark.MACROS` / `VERSUS` (serie N=8 JUN 2026) | refrescar serie con corrida completa (N=25) |

**Regla de honestidad** (heredada de `macro_baseline._validated`): jamás publicar una cifra de una
corrida que no se pueda re-correr; N=8 oscila ±20 pt — para claims públicos usar N≥20.

## Métrica → palanca de mejora (la guía)

Cuando una métrica sale mal, esta tabla dice QUÉ tocar (sin redeploy cuando es knob):

| Métrica floja | Eje del motor | Palancas |
|---|---|---|
| `safety.violaciones_por_categoria.alergeno` | scan C2 / sieve degradado | `_scan_allergen_violations` (sinónimos DD), `MEALFIT_DEGRADED_SAFETY_SCAN`, `_sieve_catalog_for_safety` (plurales) |
| `safety...dieta` | canonicalización + hard guard | SOLO `constants.canonicalize_diet_type` (P1-DIET-CANON-SSOT — no crear 4ª tabla), `DIET_HARD_GUARD` |
| `safety.min_meals_compliance_pct` | distribución de tomas | reglas de slots por condición en el skeleton/prompts de day-gen |
| `safety.fs9_flag_presente_pct` | gate FS9 | `requires_medication_review` + merge de `requires_professional_review` en `_apply_deterministic_clinical_layer` |
| `vitamin_k.variability = high` | variedad de hoja verde | `_HIGH_VIT_K_TERMS` (medication_rules) + variedad same-day |
| `nutrition.*` (MAPE, 4-en-banda) / `gym.banda` / MAPE nightly | motor de macros | `MEALFIT_MACRO_REBALANCE`, `MEALFIT_MACRO_SOLVER_ENABLED`, `MEALFIT_PORTION_QUANTIZE` (la precisión final la fija el MOTOR, no la generación) |
| `gym.entrega` / `reliability.fallback_rate_pct` | robustez del pipeline | circuit breaker `MEALFIT_CB_*`, red cross-provider `gpt-6-luna` (P1-NET-LUNA), reintentos `should_retry` |
| `changes.swap.ok_pct` / latencia | superficie swap | `MEALFIT_CHAT_AGENT_SWAP_MODEL`, `MEALFIT_SWAP_EFFORT_INDIVIDUAL` (medium ~16,5 s) / `MEALFIT_SWAP_EFFORT_DAY` (low ~8,2 s), `MEALFIT_SWAP_TARGET_FROM_SLOT` |
| `changes.regen_day` | bucle serial de día | mismas palancas de swap; el día es 4-5 llamadas EN SERIE — la latencia total escala lineal |
| `telemetry.fallback_rate` | entrega | cron `_plan_fallback_rate_alert_job` (umbral `MEALFIT_FALLBACK_RATE_THRESHOLD`) |
| `telemetry.quality_index` | PQI (variedad/coherencia/nutrición) | pesos `MEALFIT_PQI_PESO_*`; leer defectos en `GET /api/system/admin/plan-quality` |

## Diagnóstico de convergencia clínica (2026-08-07, corrida dirigida post-P1-LANDING-BENCH-2)

El header `X-Bioboros-Review-Diag` reveló por qué 13/20 perfiles con restricciones terminaban en
fallback crítico: los pools del skeleton NO se filtraban por dieta (camarones/atún/lácteos
AUTORIZADOS en planes vegan/vegetarian), la dieta viajaba como campo JSON sin directiva propia,
un splitter determinista fabricaba «Sal al gusto» por comida en perfiles HTA, y un rechazo
crítico abortaba con CERO retries. Fix: **P1-DAYGEN-DIET-CONVERGE** (4 capas knob-gated:
`MEALFIT_SKELETON_DIET_SCRUB`, `MEALFIT_DIET_DIRECTIVE_BLOCK`, `MEALFIT_SALT_LINE_CONDITION_GATE`,
`MEALFIT_DIET_CRITICAL_REGEN`), test ancla `test_p1_daygen_diet_converge.py`. Verificación: tras
deploy, re-correr los ids `3,4,9,10,13,17,19,20` y comparar contra la línea base (2/8 entregados).

## Hallazgos de producto del análisis del formulario (2026-08-07)

1. ~~**Condiciones solo-backend**: `[anemia, gout, nafld, renal]` — el backend tiene reglas y el
   formulario no puede expresarlas.~~ **CERRADO [P1-MEDICAL-SCOPE-GATE · 2026-08-09]**: se optó
   por (a), añadir los chips. El wizard ofrece ahora `Enfermedad Renal`, `Anemia`,
   `Gota / Ácido Úrico` e `Hígado Graso`, y `Antidepresivo IMAO` en medicamentos —
   `condiciones_solo_backend` y `medicaciones_solo_backend` quedan **vacíos**, y el test ancla
   invirtió su aserción para exigir que sigan vacíos: una regla clínica sin chip es una capa que
   el usuario no puede activar y de cuya ausencia no se entera. El sub de CAPS del landing («DM2 ·
   renal · HTA · alergias») deja de ser una promesa que el formulario no podía cumplir.
   Matriz: +5 perfiles (21-25). El 21 (`renal_hta`) es el que más aporta — activa las dos ramas de
   precedencia de `build_condition_prompt` (dm2+renal, hta+renal) que hasta ahora ningún perfil del
   formulario podía alcanzar.
2. ~~**Medicamentos fuera de los 14 chips quedan sin capturar en silencio.**~~ **CERRADO
   [P1-MEDICAL-SCOPE-GATE · 2026-08-09]**: ya no es silencio. Lo no listado se declara con los chips
   `Otra condición` / `Otro medicamento`, y esa señal **bloquea la generación** (422
   `clinical_scope_exceeded`, en las dos puertas: `/analyze` y `/analyze/stream`). El gate compara
   por VALOR EXACTO, nunca por subcadena — un blocklist sobre prosa sería la 17ª de esa clase en
   este repo, y aquí un falso positivo deniega servicio y un falso negativo entrega un plan
   inseguro. Estos dos chips NO entran en `FORM_*_CHIPS`: no son clínica, son la señal del gate.
3. **«4 a 5 minutos» (FAQ) no tiene fuente** — `latency.generation_s`/`telemetry` la miden; el
   baseline del gym (2026-07-03) tenía mediana ~10 min con outliers de 20 (motor pre-P1-FLASH-PRIMARY).
   Verificar antes de sostener el claim.
4. **`householdSize` es fantasma** (fijo en 1 sin UI): la matriz lo fija en 1 — si el producto
   reactiva hogares >1, añadir perfiles con multiplier y reusar los tests de coherencia P3-A.

## Relación con los demás harnesses

- **Mide su propio MAPE desde P1-PLAN-LOTE-749** (sección `nutrition`) con la banda del motor; el
  nightly (`benchmark_macro_compliance.py` + `tests/fixtures/macro_baseline.json`) sigue siendo el
  gate de regresión con sus 20 perfiles de texto libre — no son intercambiables.
- **Compone** `plan_gym.score_plan` tal cual (mismos 7 ejes) — un cambio de pesos del gym se
  refleja aquí sin tocar nada.
- El modo `telemetry` es la vista agregada de series que ya existen (`pipeline_metrics`,
  `llm_usage_events`, `_quality_index`) — no crea tablas ni crons nuevos.
