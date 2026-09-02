# Router de modelos LLM por tier (P0-LLM-PROVIDER-MIGRATION · 2026-06-12 · P1-FLASH-PRIMARY · 2026-07-31)

## [P0-GLM-MIGRATION · 2026-09-02] Provider: Z.ai GLM-5.3 (sustituye al anterior)

Decisión del owner: **Z.ai GLM-5.3** es el provider OpenAI-compatible del stack; el anterior
no sobrevive ni como nombre (`test_p1_flash_first.py::test_previous_provider_fully_removed`).
Los roles se conservan: **flash** = `glm-5.3-flash` (aux/cheap, tier gratis, red del day-gen),
**pro** = `glm-5.3` (red post-fallo cuando no hay `OPENAI_API_KEY`). OpenAI gpt-5.6
(Luna/Terra/Sol) sigue en day-gen por tier, swap, vision y reviewer clínico — sin cambios.

Hechos verificados EN VIVO 2026-09-02 (clave de pago, `api.z.ai/api/paas/v4`):

| Hecho | Medición | Consecuencia en el código |
|---|---|---|
| Thinking NO se puede apagar (`thinking.type=disabled` → 400/1210) | effort=max: 6,3 s / 173 tok en un prompt trivial; effort=low: 1,1 s / 14 tok | `ChatGLM` fija `thinking.enabled` + `reasoning_effort` default **low** (`MEALFIT_GLM_REASONING_EFFORT`); traduce `thinking.effort` y `thinking.disabled` heredados |
| Vocabulario de effort: `low`/`high`/`max` | — | `_glm_reasoning_effort`: medium→high, xhigh→max, none→low |
| `json_schema` y `json_mode` IGNORAN el esquema | devuelve `{"respuesta": …}` / markdown → `OutputParserException` | `with_structured_output` fuerza `function_calling`; `json_mode` explícito se reencamina |
| `function_calling` con `tool_choice` forzado funciona CON thinking | veredicto de 3 campos en 13 s (low) | las ramas reviewer/corrector/juez que pedían `json_mode` siguen válidas |
| Concurrencia (cuenta de pago) | 8 llamadas paralelas flash-low ~900 tok: 15–29 s cada una, cero 429; glm-5.3-low ×4: 8–10 s | el fallback cross-provider (Luna) y el breaker por modelo siguen siendo la red |
| Reasoning tokens facturan como OUTPUT y cuentan en `max_tokens` | flash-high con `max_tokens=1200` → `finish=length` | no subir el effort sin subir `max_output_tokens` |
| Precio de lista (USD/1M) | flash $0.15 in / $0.03 cached / $0.50 out · glm-5.3 $1.4 / $0.26 / $4.4 (promo -50% flash hasta 2026-09-09 NO se costea) | `db_profiles._DEFAULT_LLM_PRICING_MICROS_PER_M` |
| Multimodal nativo (imagen/vídeo/archivo) en flash | no medido | candidato a sustituir a Luna en vision (fuera de alcance hoy) |

Knobs: `ZAI_API_KEY` (env), `MEALFIT_ZAI_BASE_URL` (default `https://api.z.ai/api/paas/v4`),
`MEALFIT_GLM_REASONING_EFFORT` (`low`|`high`|`max`, default `low`). Entidad legal (legales del
frontend): JINGSHENG HENGXING TECHNOLOGY PTE. LTD., Singapur; su política declara que no
almacena el contenido enviado por la API.

> Las secciones siguientes son anteriores a la migración y se renombraron en bloque:
> donde citan mediciones fechadas antes de 2026-09-02 con "GLM", léase el proveedor anterior.


Provider único: **GLM-5.3** (API OpenAI-compatible, base
`https://api.z.ai/api/paas/v4`, key env `ZAI_API_KEY`). SSOT del router:
[`backend/llm_provider.py`](../llm_provider.py). Decisión de producto
2026-06-12: salir a producción con modelos chinos por costo; Gemini eliminado
por completo (deps, embeddings, vision, safety_settings).

## [P1-FLASH-PRIMARY · 2026-07-31] Flash primario en TODAS las superficies

Decisión del owner: **`glm-5.3-flash` es actualmente MEJOR que
`glm-5.3`** (los providers actualizan los modelos bajo el mismo ID —
la premisa "pro > flash" de 2026-06-12 caducó). Consecuencias:

- TODOS los tiers (gratis Y pagados) resuelven **flash** por default.
- El reviewer médico / fact-checker **risk-tier también es flash**
  (`_REVIEWER_RISK_TIER_DEFAULT`): mantener pro habría degradado el gate
  clínico a propósito bajo la premisa nueva. El guard
  `_warn_if_clinical_model_downgraded` detecta DESVÍO del risk-tier esperado
  (cualquier modelo ≠ flash, incluido pro, alerta).
- **La RED post-fallo es CROSS-PROVIDER** [P1-NET-LUNA · 2026-07-31]:
  `_PRO_MODEL_NAME` default = **`gpt-5.6-luna`** (OpenAI) — 2º en la cadena
  del day-gen, fallback del planner con breaker abierto, escalada del
  corrector quirúrgico y por skeleton-fidelity. Razón: flash y pro son el
  MISMO proveedor; el incidente que motivó la red (breaker abierto 172×, gym
  baseline) fue GLM rate-limiteando — pro caía JUNTO con flash y la red
  no atrapaba nada. Luna = infra/key/límites propios (diversidad real).
  Fail-safe: sin `OPENAI_API_KEY` la red vuelve sola a `glm-5.3`
  (nunca sin red). Colapsar la red a flash sigue prohibido (fallbacks no-op
  contra el mismo breaker roto). Simetría: pipeline GLM→cae a OpenAI;
  reviewer OpenAI→cae a GLM. Los 8 consumidores de modelo variable
  construyen con dispatch por proveedor (`ChatOpenAIInstrumented` para gpt-*).
  Test ancla: [`test_p1_net_luna.py`](../tests/test_p1_net_luna.py).
- Rollback sin redeploy: `MEALFIT_MODEL_PAID_TIER=glm-5.3` (tiers),
  `MEALFIT_PRO_MODEL=glm-5.3` (red), `MEALFIT_REVIEWER_RISK_TIER_MODEL`
  / `MEALFIT_FACT_CHECKER_RISK_TIER_MODEL` (gate clínico),
  `MEALFIT_BARIATRIC_DAYGEN_MODEL` (day-gen bariátrico).

Test ancla: [`test_p1_flash_primary.py`](../tests/test_p1_flash_primary.py).

## [P1-REVIEWER-TIER-MODELS · 2026-07-31] Reviewer clínico por tier (Luna/Terra)

OpenAI recortó la familia gpt-5.6: **luna -80%** ($0.20 in / $1.20 out por 1M) y
**terra -20%** ($2.00 / $12.00). Decisión del owner: el reviewer médico
risk-tier se enruta por tier de suscripción:

| Tier | Reviewer risk-tier | Costo/llamada (2.371 in / 213 out reales) | Worst-case/mes (cap × 2 calls/plan, 100% clínicos) |
|---|---|---|---|
| free/guest | `gpt-5.6-luna` | ~$0.0007 | $0.015 (10 planes, P1-CREDITS-LADDER) |
| basic ($9.99, 50 planes) | `gpt-5.6-terra` (NUNCA sol: sería ~18% del revenue) | ~$0.0073 | $0.73 = **7.3%** del revenue → rentable |
| plus ($19.99, 200) / ultra ($49.99) | `gpt-5.6-terra` | ~$0.0073 | $2.92 = 14.6% (plus) — realista <1% |
| plus/ultra + perfil DIFÍCIL | **`gpt-5.6-sol`** ($5/$30) [P1-REVIEWER-SOL-HARD · 2026-07-31] | ~$0.018 (2.5× terra) | $7.30 = 36.5% (plus, cap 200); ultra $18 = 36% (cap 500, P1-CREDITS-LADDER — antes ∞ sin acotar). Solo extremos patológicos; realista centavos |

**"Difícil"** (determinista, SSOT `condition_rules.detect_active_rules` — cero
listas de keywords nuevas): regla bariátrica activa **o** ≥2 reglas clínicas
activas. Fail-safe de detección → no-difícil (terra). Knob
`MEALFIT_REVIEWER_RISK_MODEL_PAID_HARD` (default sol). Test ancla:
[`test_p1_reviewer_sol_hard.py`](../tests/test_p1_reviewer_sol_hard.py).
⚠️ La escalera de calidad luna<terra<sol es hipótesis de PRECIO, no medición —
`llm_usage_events` por modelo+nodo dará los datos para el A/B.

- Con el recorte, **luna quedó MÁS BARATO que glm-5.3** (~0.75×) — por
  eso es viable hasta en el tier gratis.
- Knobs: `MEALFIT_REVIEWER_RISK_MODEL_FREE` (default luna) /
  `MEALFIT_REVIEWER_RISK_MODEL_PAID` (default terra);
  `MEALFIT_REVIEWER_RISK_TIER_MODEL` (global) gana sobre el map;
  `MEALFIT_REVIEWER_MODEL` (hard-override) gana sobre todo.
- **Fail-safe**: modelo OpenAI sin `OPENAI_API_KEY` → fallback
  `_REVIEWER_RISK_TIER_DEFAULT` (flash) + alerta de desvío. El gate clínico
  nunca se queda sin modelo.
- Construcción con dispatch por proveedor (`ChatOpenAIInstrumented` para
  gpt-* — backpressure + costo en `llm_usage_events` intactos). El thinking
  GLM (`MEALFIT_REVIEWER_THINKING`) se salta para OpenAI (gpt-5.6 razona
  nativo); aplica solo si el reviewer cae al fallback flash.
- El **fact-checker NO cambió** de provider (tool-calling loop medido en
  GLM risk-tier flash). Candidato a Luna cuando haya datos del reviewer.

Test ancla: [`test_p1_reviewer_tier_models.py`](../tests/test_p1_reviewer_tier_models.py).

## Mapping tier → modelo

| `user_profiles.plan_tier` | Modelo | Pricing (USD/1M tok, in miss/hit · out) |
|---|---|---|
| `gratis` / guest / NULL / desconocido / fallo de lookup | `glm-5.3-flash` | $0.15 / $0.03 · $0.50 |
| `basic` · `plus` · `ultra` | `glm-5.3-flash` (P1-FLASH-PRIMARY; era `glm-5.3`) | $0.15 / $0.03 · $0.50 |

(Pricing de la red pro, usada solo post-fallo: $1.40 / $0.26 · $4.40.)

Invariante **fail-cheap**: cualquier duda (guest, DB blip, tier corrupto)
resuelve al modelo FREE — un fallo de lookup jamás encarece la llamada.
Lookup con cache TTL in-process (`MEALFIT_TIER_CACHE_TTL_S`, default 300s);
`invalidate_tier_cache(user_id)` disponible post-upgrade de billing.

## [P1-DAYGEN-TIER-MODEL · 2026-07-31] Generador de DÍAS por tier (Luna primario)

Decisión del owner tras el A/B medido con el índice de calidad (2026-07-31):
*"glm medium dura mucho, el ganador es gpt 5.6 luna medium; medium en plus
nadamás por ahora, low o sin pensamiento en gratis; deja a glm donde no
sea necesario luna"*. El day-gen es la ÚNICA superficie de generación que sale
de GLM; el resto de nodos (planner, critique, correctores, compressor,
fact-checker) sigue flash [P1-FLASH-PRIMARY].

| Tier | Day-gen primario | Effort | Cadena completa | Evidencia A/B (índice 0-100) |
|---|---|---|---|---|
| plus / ultra | `gpt-5.6-luna` | `medium` (`reasoning_effort`) | [luna, flash] | luna-medium **95,4** (coherencia 90) vs flash 82-92 (coherencia 57-83); ~$0,040/plan, 34,5 s/día |
| gratis / basic / guest / desconocido | `gpt-5.6-luna` | `low` | [luna, flash] | fail-cheap simétrico al reviewer; low no medido aún — floor barato ($1,20/M out con poco razonamiento) |
| (cualquiera) sin `OPENAI_API_KEY` | `glm-5.3-flash` | — | [flash, red P1-NET-LUNA→pro] | fail-safe: jamás un modelo incobrable delante |

Descalificados por el mismo A/B: **luna-high** (90,8 — peor que medium en TODO,
3× latencia, 1,5× costo) y **flash+thinking-medium** (76,9 — 36k tokens de
razonamiento, 266 s/día, día muerto contra el techo → plan degradado. La
"ganga" del reasoning barato de GLM era latencia, no dinero).

Reglas de diseño ancladas por [`test_p1_daygen_tier_model.py`](../tests/test_p1_daygen_tier_model.py):
el effort del tier aplica **SOLO al modelo primario** (la red flash jamás
hereda thinking — la red rescata, no profundiza); el knob global
`MEALFIT_DAYGEN_EFFORT` (experimentos A/B) gana sobre el del tier; effort
inválido cae al default del tier, nunca a uno más caro. Knobs:
`MEALFIT_DAYGEN_{MODEL,EFFORT}_{PLUS,FREE}`. Bariátrico NO cambia
(early-return propio con `MEALFIT_BARIATRIC_DAYGEN_MODEL`, sin canario/lite).

## Surfaces tier-routed vs aux-fijo

| Surface | Routing | Cómo obtiene la identidad |
|---|---|---|
| Pipeline plan-gen (`_route_model`: planner dinámico, day-gen, correctores via PRO/FLASH) | **Tier** | ContextVar `user_id_var` (seteado por `arun_plan_pipeline`, cubre chunks de fondo) |
| Chat agent (`call_model`) | **Tier** | `state.user_id` / `state.session_id` |
| Chat swap (`swap_meal`) — y por herencia `/regenerate-day`, que es un bucle de swaps | **`gpt-5.6-luna` fijo** (P1-SWAP-LUNA · 2026-08-05), con `reasoning_effort` **por superficie**: plato individual → `medium`, día completo → `low`. Fail-safe sin `OPENAI_API_KEY` → router por tier. Knobs: `MEALFIT_CHAT_AGENT_SWAP_MODEL`, `MEALFIT_SWAP_EFFORT_INDIVIDUAL`, `MEALFIT_SWAP_EFFORT_DAY` | `form_data.user_id` (validado vs JWT en `api_swap_meal`) |
| Tool `modify_single_meal` | **Tier** | `user_id` forzado por P0-AGENT-1 |
| Reviewer médico con perfil de riesgo | **Por TIER** (P1-REVIEWER-TIER-MODELS · 2026-07-31): free/guest → `gpt-5.6-luna`; basic/plus/ultra → `gpt-5.6-terra` | `_reviewer_risk_model_for_tier()`; fail-safe sin `OPENAI_API_KEY` → flash + alerta; el guard de desvío alerta ante cualquier modelo ≠ esperado del tier |
| Fact-checker clínico con perfil de riesgo | **risk-tier FLASH fijo** (P1-FLASH-PRIMARY) | `_REVIEWER_RISK_TIER_DEFAULT` — sin cambio de provider (tool-calling loop medido en GLM) |
| Aux baratos: títulos, recipe-expand, sentiment, router RAG, fact-extractor, memoria, nudges, judge, compressor, meta-learning, planner default, médico Q&A, probe CB | **FLASH fijo** | — |

Los per-feature knobs `MEALFIT_<FEATURE>_MODEL` se preservan y **siempre
ganan** sobre el tier-routing (rollback / A-B sin redeploy, convención
P3-PREVIEW-MODEL-KNOB).

## Thinking mode (razonamiento nativo V4) por superficie

[P1-REVIEWER-THINKING · 2026-07-05 · P2-THINKING-EFFORT · 2026-07-06 · P1-FACTCHECKER-THINKING · 2026-07-08]
GLM-5.3 trae razonamiento nativo ON de fábrica; el repo lo apaga globalmente
(`P1-PROVIDER-THINKING-DEFAULT` — en day-gen midió >170s → fallback matemático). Se
re-activa SELECTIVAMENTE solo en superficies de **juicio clínico** de bajo volumen.

**Regla empírica (A/B sesión 2026-07-08):** el thinking rinde en superficies de
**output chico (juicio)** y es contraproducente en **output grande (generación)**,
donde revienta el timeout. Restricción del API: thinking NO soporta el `tool_choice`
forzado de `function_calling` → structured output vía `method="json_mode"`; `bind_tools`
sin tool_choice forzado sí lo soporta nativo.

| Superficie | Output | Knob | Estado | Effort | Medición |
|---|---|---|---|---|---|
| Reviewer médico (risk-tier) | chico (veredicto) | `MEALFIT_REVIEWER_THINKING` (+`_EFFORT`, +`_TIMEOUT_S`=90) | **ON** | `medium` | Sweep OFF/low/medium/high/max (caso látex): `low` atrapó lo MISMO que `max` (4 cross-react + gradación tomate) a igual velocidad → **max = overkill** (el reviewer solo escanea+juzga contra el reporte del fact-checker, no razona desde cero). `medium` = hedge para planes reales de 7 días; `low` es el piso probado suficiente |
| Fact-checker clínico (FASE 1) | chico (reporte) | `MEALFIT_FACT_CHECKER_THINKING` (+`_EFFORT`, +`_TIMEOUT_S`=60) | **ON** | `high` | A/B warfarina+mariscos: HIGH atrapó interacción fibra↔absorción + CYP450 + cross-react sistemática que OFF omitió. `max` (72s) no superó a `high` (53s) → high = sweet spot. Usa `bind_tools` → thinking nativo (sin json_mode) |
| Corrector quirúrgico (escalada Pro) | **grande (día completo)** | `MEALFIT_SURGICAL_PRO_THINKING` (+`_EFFORT`) | **OFF** | — | A/B caso pollo-duplicado: OFF=17s `pro_success` con fix correcto; HIGH y MAX = **timeout (120s)** → `None`. Generación grande + reasoning revienta el cap Y compite con el budget del pipeline |
| Day-gen / planner | grande | `MEALFIT_DAYGEN_EFFORT` (global A/B) · `MEALFIT_DAYGEN_EFFORT_{PLUS,FREE}` (tier) | **OFF en GLM** · ON en Luna [P1-DAYGEN-TIER-MODEL] | plus `medium` / free `low` (solo el primario Luna; la red flash sin thinking) | `P1-PROVIDER-THINKING-DEFAULT` sigue vigente para GLM: flash+thinking-medium midió 36k tok de razonamiento, 266 s/día y plan degradado 76,9 (A/B 2026-07-31). El razonamiento de LUNA sí paga: medium 95,4 vs base 82-92 |

Todos los knobs de thinking **nacen OFF** (convención medir→actuar) y hacen **fail-open
al path estándar** (nunca a aprobar/omitir el gate clínico). Test ancla del reviewer/surgical:
[`test_p1_reviewer_thinking.py`](../tests/test_p1_reviewer_thinking.py); del fact-checker:
[`test_p1_factchecker_thinking.py`](../tests/test_p1_factchecker_thinking.py).

## Knobs nuevos

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_ZAI_BASE_URL` | `https://api.z.ai/api/paas/v4` | endpoint OpenAI-compatible |
| `MEALFIT_MODEL_FREE_TIER` | `glm-5.3-flash` | modelo tier gratis/aux |
| `MEALFIT_MODEL_PAID_TIER` | `glm-5.3-flash` (P1-FLASH-PRIMARY; era pro) | modelo tiers pagados |
| `MEALFIT_BARIATRIC_DAYGEN_MODEL` | `_FLASH_MODEL_NAME` (P1-FLASH-PRIMARY) | day-gen bariátrico (era PRO hardcoded) |
| `MEALFIT_TIER_CACHE_TTL_S` | `300` (clamp [10, 3600]) | TTL del cache de tier |
| `MEALFIT_LLM_PRICING_JSON` | — | override del pricing de telemetría (antes `MEALFIT_GEMINI_PRICING_JSON`) |

## Embeddings: Cohere Embed v4 (P1-COHERE-EMBED-V4 · 2026-06-12)

| Capa | Estado | Detalle |
|---|---|---|
| Embeddings ([`embeddings_provider.py`](../embeddings_provider.py)) | **`cohere` (default)** — `embed-v4.0` @1536, gating por presencia de `COHERE_API_KEY` (sin key ⇒ degradación limpia a keyword/recency). Activación = key + restart | Asimetría `input_type`: queries→`search_query`, persistido en pgvector→`search_document` (`purpose="document"` en user_facts/visual_diary). Columnas migradas a `vector(1536)` ([`p1_cohere_embed_v4_vector_dims_2026_06_12.sql`](../migrations/p1_cohere_embed_v4_vector_dims_2026_06_12.sql), aplicada 2026-06-12; vectores Gemini legacy anulados — espacios incomparables). Cache keys versionadas por `get_embeddings_model_id()` (`embed-v4.0@1536`) + purpose. Knobs: `MEALFIT_EMBEDDINGS_{PROVIDER,MODEL,DIMENSION}` (dim ∈ {256,512,1024,1536}; cambiarla exige migrar pgvector). Rollback: `MEALFIT_EMBEDDINGS_PROVIDER=openai_compatible` + base_url/model/`EMBEDDINGS_API_KEY` |
| Vision ([`vision_agent.py`](../vision_agent.py)) | `disabled` — Diario Visual / "Escanear comida" responden `analysis_failed` (soft-fail) | `MEALFIT_VISION_PROVIDER=openai_compatible` + `MEALFIT_VISION_BASE_URL` + `MEALFIT_VISION_MODEL` + env `VISION_API_KEY`. Nota: Embed v4 soporta embeddings de IMAGEN — búsqueda visual futura sin provider extra (el ANÁLISIS generativo de fotos sí requiere un VLM) |

## Particularidades del API verificadas EN VIVO (2026-06-12)

Dos 400s reales que el wrapper `ChatGLM` resuelve centralizadamente
(NO tocar los ~15 callsites de `.with_structured_output(...)`):

1. `response_format: json_schema` (default de langchain-openai ≥1.3) →
   `400 This response_format type is unavailable`. El wrapper fuerza
   `method="function_calling"` (tools API, soportado).
2. El thinking mode (default-ON en V4) no soporta `tool_choice` forzado →
   `400 Thinking mode does not support this tool_choice`. El wrapper
   desactiva thinking (`extra_body={"thinking": {"type": "disabled"}}`)
   SOLO en runnables estructurados — relleno de schema no necesita
   reasoning y se ahorran reasoning-tokens (facturan como output).

`bind_tools` sin tool_choice forzado (chat agent, fact-checker, day-gen
nutrition tool) funciona EN thinking mode — verificado en vivo. El usage
reporta `output_token_details.reasoning` y `input_token_details.cache_read`
(alimenta `llm_usage_events` sin cambios).

## Eliminado con la migración

- `langchain-google-genai` (dep), `GEMINI_API_KEY`, `google_api_key=` en
  constructores, `safety_settings` (HarmCategory — el filtro configurable era
  Gemini-only; decisión P3-CHAT-SAFETY-OFF queda satisfecha por defecto).
- Caps de thinking-budget (`MEALFIT_*_THINKING_BUDGET`): el reasoning de
  GLM es nativo y su output cuesta 10-30× menos que el de Gemini — el
  runaway de costo que motivaba los caps no existe.
- Knobs `MEALFIT_GEMINI_EMBEDDING_TEXT_MODEL` / `_MULTIMODAL_MODEL`.
- `glm-5.3-flash`/`glm-5.3` NO se usan (aliases legacy, deprecan
  2026-07-24); el pricing dict los cubre por si un knob transitorio los nombra.

Test ancla: [`tests/test_p0_glm_migration.py`](../tests/test_p0_glm_migration.py)
(blanket no-Gemini, matriz del router, fail-cheap, wrapper, no-key-hardcodeada,
knobs registrados, consistencia CB, pricing, soft-fail de providers pendientes).
