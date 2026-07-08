# Router de modelos LLM por tier (P0-DEEPSEEK-MIGRATION · 2026-06-12)

Provider único: **DeepSeek V4** (API OpenAI-compatible, base
`https://api.deepseek.com`, key env `DEEPSEEK_API_KEY`). SSOT del router:
[`backend/llm_provider.py`](../llm_provider.py). Decisión de producto
2026-06-12: salir a producción con modelos chinos por costo; Gemini eliminado
por completo (deps, embeddings, vision, safety_settings).

## Mapping tier → modelo

| `user_profiles.plan_tier` | Modelo | Pricing (USD/1M tok, in miss/hit · out) |
|---|---|---|
| `gratis` / guest / NULL / desconocido / fallo de lookup | `deepseek-v4-flash` | $0.14 / $0.0028 · $0.28 |
| `basic` · `plus` · `ultra` | `deepseek-v4-pro` | $0.435 / $0.003625 · $0.87 |

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
| Reviewer médico / fact-checker con perfil de riesgo | **PRO fijo** (todos los tiers) | `_REVIEWER_RISK_TIER_DEFAULT` — la seguridad clínica no se degrada por plan de pago |
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
| Reviewer médico (risk-tier) | chico (veredicto) | `MEALFIT_REVIEWER_THINKING` (+`_EFFORT`, +`_TIMEOUT_S`=90) | **ON** | `max` | mejor estratificación de riesgo (tomate moderado vs alto); sin penalización de latencia en veredicto chico |
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
| `MEALFIT_MODEL_PAID_TIER` | `deepseek-v4-pro` | modelo tiers pagados |
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
