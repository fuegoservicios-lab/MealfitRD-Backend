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

- **Los 20 perfiles clínicos se evalúan TODOS como dominicanos.** `safety`, `gym`, `latency` y
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
| `live [N] --conc 2 [--changes] [--save-plans] [--provider openai]` | claves LLM + Neon | genera N planes reales con la matriz y puntúa seguridad + gym + latencia; `--changes` ejercita swap individual y bucle de día. `--provider openai` fuerza TODA la corrida a gpt-5.6 (cero DeepSeek) vía los 4 knobs sancionados (`MEALFIT_FLASH_MODEL`, `MEALFIT_MODEL_FREE_TIER`, `MEALFIT_MODEL_PAID_TIER`, `MEALFIT_PRO_MODEL`); requiere `OPENAI_API_KEY`, fail-loud sin ella. NO reintroduce el override global eliminado (P1-DEEPSEEK-ONLY-RESTORE): reviewer/day-gen/swap conservan su routing propio, que YA es OpenAI por defecto | `structural`, `safety`, `gym`, `latency`, `changes` |
| `remote [N] --api-base URL [--conc 1] [--changes] [--save-plans]` | **cero claves** (solo red al deploy) | la corrida «cuenta de invitado»: genera contra el API desplegado como `user_id=guest` y puntúa LOCALMENTE (los scorers son funciones puras). El routing de modelos lo decide el servidor — para un guest, day-gen/swap/reviewer corren en Luna = OpenAI (P1-DAYGEN-TIER-MODEL/P1-SWAP-LUNA/P1-REVIEWER-TIER-MODELS). `--changes` ejercita solo swap (regenerate-day exige plan persistido con auth). Respeta el RateLimiter de `/analyze` (3/60s por IP): conc default 1, backoff ante 429 | `meta`, `safety`, `gym`, `latency`, `changes` |
| `telemetry --days 30` | Neon | series de PROD: éxito de cambios a la primera, banda entregada, fallback rate, PQI, costo por nodo | `telemetry` |
| `score --plans f.json` | nada | re-puntúa planes crudos de una corrida `live/remote --save-plans` (cambio de scorer sin pagar LLM) | `safety` |

```bash
# desde backend/, con .env cargable
python scripts/landing_benchmark.py structural
python scripts/landing_benchmark.py live 5 --conc 2 --changes --save-plans
python scripts/landing_benchmark.py live 20 --provider openai --conc 2
python scripts/landing_benchmark.py remote 20 --api-base https://app.bioboros.com --changes
python scripts/landing_benchmark.py telemetry --days 30
```

Costo estimado de `live` completo (20 perfiles ≈ los 20 del nightly): 30-45 min con `--conc 2`,
cuota DeepSeek compartida con prod — correr de madrugada RD como el nightly. `--changes` añade
~6 llamadas Luna por perfil ejercitado.

## Matriz de perfiles (cobertura del formulario)

20 perfiles en `build_landing_profiles()`. Invariantes ancladas por test: **cada** chip de
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
| `gym.aggregate.banda` + nightly MAPE | `benchmark.MACROS` / `VERSUS` (serie N=8 JUN 2026) | refrescar serie con corrida ≥N=20 |

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
| `gym.banda` / MAPE nightly | motor de macros | `MEALFIT_MACRO_REBALANCE`, `MEALFIT_MACRO_SOLVER_ENABLED`, `MEALFIT_PORTION_QUANTIZE` (la precisión final la fija el MOTOR, no la generación) |
| `gym.entrega` (fallbacks) | robustez del pipeline | circuit breaker `MEALFIT_CB_*`, red cross-provider `gpt-5.6-luna` (P1-NET-LUNA), reintentos `should_retry` |
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

- **No duplica** el MAPE de macros: eso es del nightly (`benchmark_macro_compliance.py` + baseline
  `tests/fixtures/macro_baseline.json`). Este reporte referencia la banda vía el eje `banda` del gym.
- **Compone** `plan_gym.score_plan` tal cual (mismos 7 ejes) — un cambio de pesos del gym se
  refleja aquí sin tocar nada.
- El modo `telemetry` es la vista agregada de series que ya existen (`pipeline_metrics`,
  `llm_usage_events`, `_quality_index`) — no crea tablas ni crons nuevos.
