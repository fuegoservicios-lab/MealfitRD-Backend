# Router de modelos LLM por tier (P0-DEEPSEEK-MIGRATION · 2026-06-12 · P1-FLASH-PRIMARY · 2026-07-31)

Provider único: **DeepSeek V4** (API OpenAI-compatible, base
`https://api.deepseek.com`, key env `DEEPSEEK_API_KEY`). SSOT del router:
[`backend/llm_provider.py`](../llm_provider.py). Decisión de producto
2026-06-12: salir a producción con modelos chinos por costo; Gemini eliminado
por completo (deps, embeddings, vision, safety_settings).

## [P1-FLASH-PRIMARY · 2026-07-31] Flash primario en TODAS las superficies

Decisión del owner: **`deepseek-v4-flash` es actualmente MEJOR que
`deepseek-v4-pro`** (los providers actualizan los modelos bajo el mismo ID —
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
  baseline) fue DeepSeek rate-limiteando — pro caía JUNTO con flash y la red
  no atrapaba nada. Luna = infra/key/límites propios (diversidad real).
  Fail-safe: sin `OPENAI_API_KEY` la red vuelve sola a `deepseek-v4-pro`
  (nunca sin red). Colapsar la red a flash sigue prohibido (fallbacks no-op
  contra el mismo breaker roto). Simetría: pipeline DeepSeek→cae a OpenAI;
  reviewer OpenAI→cae a DeepSeek. Los 8 consumidores de modelo variable
  construyen con dispatch por proveedor (`ChatOpenAIInstrumented` para gpt-*).
  Test ancla: [`test_p1_net_luna.py`](../tests/test_p1_net_luna.py).
- Rollback sin redeploy: `MEALFIT_MODEL_PAID_TIER=deepseek-v4-pro` (tiers),
  `MEALFIT_PRO_MODEL=deepseek-v4-pro` (red), `MEALFIT_REVIEWER_RISK_TIER_MODEL`
  / `MEALFIT_FACT_CHECKER_RISK_TIER_MODEL` (gate clínico),
  `MEALFIT_BARIATRIC_DAYGEN_MODEL` (day-gen bariátrico).

Test ancla: [`test_p1_flash_primary.py`](../tests/test_p1_flash_primary.py).

## [P1-REVIEWER-TIER-MODELS · 2026-07-31] Reviewer clínico por tier (Luna/Terra)

OpenAI recortó la familia gpt-5.6: **luna -80%** ($0.20 in / $1.20 out por 1M) y
**terra -20%** ($2.00 / $12.00). Decisión del owner: el reviewer médico
risk-tier se enruta por tier de suscripción:

| Tier | Reviewer risk-tier | Costo/llamada (2.371 in / 213 out reales) | Worst-case/mes (cap × 2 calls/plan, 100% clínicos) |
|---|---|---|---|
| free/guest | `gpt-5.6-luna` | ~$0.0007 | $0.02 (15 planes) |
| basic ($9.99, 50 planes) | `gpt-5.6-terra` (NUNCA sol: sería ~18% del revenue) | ~$0.0073 | $0.73 = **7.3%** del revenue → rentable |
| plus ($19.99, 200) / ultra ($49.99) | `gpt-5.6-terra` | ~$0.0073 | $2.92 = 14.6% (plus) — realista <1% |
| plus/ultra + perfil DIFÍCIL | **`gpt-5.6-sol`** ($5/$30) [P1-REVIEWER-SOL-HARD · 2026-07-31] | ~$0.018 (2.5× terra) | $7.30 = 36.5% (plus) solo en el extremo patológico; realista centavos |

**"Difícil"** (determinista, SSOT `condition_rules.detect_active_rules` — cero
listas de keywords nuevas): regla bariátrica activa **o** ≥2 reglas clínicas
activas. Fail-safe de detección → no-difícil (terra). Knob
`MEALFIT_REVIEWER_RISK_MODEL_PAID_HARD` (default sol). Test ancla:
[`test_p1_reviewer_sol_hard.py`](../tests/test_p1_reviewer_sol_hard.py).
⚠️ La escalera de calidad luna<terra<sol es hipótesis de PRECIO, no medición —
`llm_usage_events` por modelo+nodo dará los datos para el A/B.

- Con el recorte, **luna quedó MÁS BARATO que deepseek-v4-pro** (~0.75×) — por
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
  DeepSeek (`MEALFIT_REVIEWER_THINKING`) se salta para OpenAI (gpt-5.6 razona
  nativo); aplica solo si el reviewer cae al fallback flash.
- El **fact-checker NO cambió** de provider (tool-calling loop medido en
  DeepSeek risk-tier flash). Candidato a Luna cuando haya datos del reviewer.

Test ancla: [`test_p1_reviewer_tier_models.py`](../tests/test_p1_reviewer_tier_models.py).

## Mapping tier → modelo

| `user_profiles.plan_tier` | Modelo | Pricing (USD/1M tok, in miss/hit · out) |
|---|---|---|
| `gratis` / guest / NULL / desconocido / fallo de lookup | `deepseek-v4-flash` | $0.14 / $0.0028 · $0.28 |
| `basic` · `plus` · `ultra` | `deepseek-v4-flash` (P1-FLASH-PRIMARY; era `deepseek-v4-pro`) | $0.14 / $0.0028 · $0.28 |

(Pricing de la red pro, usada solo post-fallo: $0.435 / $0.003625 · $0.87.)

Invariante **fail-cheap**: cualquier duda (guest, DB blip, tier corrupto)
resuelve al modelo FREE — un fallo de lookup jamás encarece la llamada.
Lookup con cache TTL in-process (`MEALFIT_TIER_CACHE_TTL_S`, default 300s);
`invalidate_tier_cache(user_id)` disponible post-upgrade de billing.

## Surfaces tier-routed vs aux-fijo

| Surface | Routing | Cómo obtiene la identidad |
|---|---|---|
| Pipeline plan-gen (`_route_model`: planner dinámico, day-gen, correctores via PRO/FLASH) | **Tier** | ContextVar `user_id_var` (seteado por `arun_plan_pipeline`, cubre chunks de fondo) |
| Chat agent (`call_model`) | **Tier** | `state.user_id` / `state.session_id` |
| Chat swap (`swap_meal`) | **Tier** | `form_data.user_id` (validado vs JWT en `api_swap_meal`) |
| Tool `modify_single_meal` | **Tier** | `user_id` forzado por P0-AGENT-1 |
| Reviewer médico con perfil de riesgo | **Por TIER** (P1-REVIEWER-TIER-MODELS · 2026-07-31): free/guest → `gpt-5.6-luna`; basic/plus/ultra → `gpt-5.6-terra` | `_reviewer_risk_model_for_tier()`; fail-safe sin `OPENAI_API_KEY` → flash + alerta; el guard de desvío alerta ante cualquier modelo ≠ esperado del tier |
| Fact-checker clínico con perfil de riesgo | **risk-tier FLASH fijo** (P1-FLASH-PRIMARY) | `_REVIEWER_RISK_TIER_DEFAULT` — sin cambio de provider (tool-calling loop medido en DeepSeek) |
| Aux baratos: títulos, recipe-expand, sentiment, router RAG, fact-extractor, memoria, nudges, judge, compressor, meta-learning, planner default, médico Q&A, probe CB | **FLASH fijo** | — |

Los per-feature knobs `MEALFIT_<FEATURE>_MODEL` se preservan y **siempre
ganan** sobre el tier-routing (rollback / A-B sin redeploy, convención
P3-PREVIEW-MODEL-KNOB).

## Thinking mode (razonamiento nativo V4) por superficie

[P1-REVIEWER-THINKING · 2026-07-05 · P2-THINKING-EFFORT · 2026-07-06 · P1-FACTCHECKER-THINKING · 2026-07-08]
DeepSeek-V4 trae razonamiento nativo ON de fábrica; el repo lo apaga globalmente
(`P1-DEEPSEEK-THINKING-OFF` — en day-gen midió >170s → fallback matemático). Se
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
| Day-gen / planner | grande | — | **OFF permanente** | — | `P1-DEEPSEEK-THINKING-OFF`: numérico = motor determinista |

Todos los knobs de thinking **nacen OFF** (convención medir→actuar) y hacen **fail-open
al path estándar** (nunca a aprobar/omitir el gate clínico). Test ancla del reviewer/surgical:
[`test_p1_reviewer_thinking.py`](../tests/test_p1_reviewer_thinking.py); del fact-checker:
[`test_p1_factchecker_thinking.py`](../tests/test_p1_factchecker_thinking.py).

## Knobs nuevos

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | endpoint OpenAI-compatible |
| `MEALFIT_MODEL_FREE_TIER` | `deepseek-v4-flash` | modelo tier gratis/aux |
| `MEALFIT_MODEL_PAID_TIER` | `deepseek-v4-flash` (P1-FLASH-PRIMARY; era pro) | modelo tiers pagados |
| `MEALFIT_BARIATRIC_DAYGEN_MODEL` | `_FLASH_MODEL_NAME` (P1-FLASH-PRIMARY) | day-gen bariátrico (era PRO hardcoded) |
| `MEALFIT_TIER_CACHE_TTL_S` | `300` (clamp [10, 3600]) | TTL del cache de tier |
| `MEALFIT_LLM_PRICING_JSON` | — | override del pricing de telemetría (antes `MEALFIT_GEMINI_PRICING_JSON`) |

## Embeddings: Cohere Embed v4 (P1-COHERE-EMBED-V4 · 2026-06-12)

| Capa | Estado | Detalle |
|---|---|---|
| Embeddings ([`embeddings_provider.py`](../embeddings_provider.py)) | **`cohere` (default)** — `embed-v4.0` @1536, gating por presencia de `COHERE_API_KEY` (sin key ⇒ degradación limpia a keyword/recency). Activación = key + restart | Asimetría `input_type`: queries→`search_query`, persistido en pgvector→`search_document` (`purpose="document"` en user_facts/visual_diary). Columnas migradas a `vector(1536)` ([`p1_cohere_embed_v4_vector_dims_2026_06_12.sql`](../migrations/p1_cohere_embed_v4_vector_dims_2026_06_12.sql), aplicada 2026-06-12; vectores Gemini legacy anulados — espacios incomparables). Cache keys versionadas por `get_embeddings_model_id()` (`embed-v4.0@1536`) + purpose. Knobs: `MEALFIT_EMBEDDINGS_{PROVIDER,MODEL,DIMENSION}` (dim ∈ {256,512,1024,1536}; cambiarla exige migrar pgvector). Rollback: `MEALFIT_EMBEDDINGS_PROVIDER=openai_compatible` + base_url/model/`EMBEDDINGS_API_KEY` |
| Vision ([`vision_agent.py`](../vision_agent.py)) | `disabled` — Diario Visual / "Escanear comida" responden `analysis_failed` (soft-fail) | `MEALFIT_VISION_PROVIDER=openai_compatible` + `MEALFIT_VISION_BASE_URL` + `MEALFIT_VISION_MODEL` + env `VISION_API_KEY`. Nota: Embed v4 soporta embeddings de IMAGEN — búsqueda visual futura sin provider extra (el ANÁLISIS generativo de fotos sí requiere un VLM) |

## Particularidades del API verificadas EN VIVO (2026-06-12)

Dos 400s reales que el wrapper `ChatDeepSeek` resuelve centralizadamente
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
  DeepSeek es nativo y su output cuesta 10-30× menos que el de Gemini — el
  runaway de costo que motivaba los caps no existe.
- Knobs `MEALFIT_GEMINI_EMBEDDING_TEXT_MODEL` / `_MULTIMODAL_MODEL`.
- `deepseek-chat`/`deepseek-reasoner` NO se usan (aliases legacy, deprecan
  2026-07-24); el pricing dict los cubre por si un knob transitorio los nombra.

Test ancla: [`tests/test_p0_deepseek_migration.py`](../tests/test_p0_deepseek_migration.py)
(blanket no-Gemini, matriz del router, fail-cheap, wrapper, no-key-hardcodeada,
knobs registrados, consistencia CB, pricing, soft-fail de providers pendientes).
