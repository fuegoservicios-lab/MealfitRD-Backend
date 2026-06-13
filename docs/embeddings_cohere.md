# P1-COHERE-EMBED-V4 · 2026-06-12 — Embeddings con Cohere Embed v4

Stack de DOS providers: DeepSeek (LLM) + Cohere (embeddings). SSOT:
[`embeddings_provider.py`](../embeddings_provider.py).

## Contrato

- Modelo `embed-v4.0` @ **1536 dims** (= columnas pgvector tras la migración
  [`p1_cohere_embed_v4_vector_dims_2026_06_12.sql`](../supabase/migrations/p1_cohere_embed_v4_vector_dims_2026_06_12.sql),
  aplicada 2026-06-12 — los vectores Gemini legacy @768 fueron anulados, el
  contenido se re-embeddea on-write). Dims válidas: 256/512/1024/1536
  (`MEALFIT_EMBEDDINGS_DIMENSION`).
- **Asimetría `input_type`** (palanca de precisión RAG de Embed v4):
  - queries → `search_query` (default de `get_text_embedding(text)`).
  - contenido persistido en pgvector (`user_facts`, `visual_diary`) →
    `purpose="document"` → `search_document` (save sites en
    `fact_extractor.py` ~572/~721).
- **Cache keys versionadas**: las capas cacheadas (`_cached_text_embedding`,
  LRU de `constants.get_embedding`, pantry cache) llevan
  `get_embeddings_model_id()` (`"embed-v4.0@1536"`) + `purpose` como ARGS de
  la cache key. Sin esto, Redis (TTL ~100 años) serviría vectores del espacio
  vectorial anterior tras un switch de provider/modelo → similitudes basura
  silenciosas.
- **Gating**: presencia de `COHERE_API_KEY` (fallback `EMBEDDINGS_API_KEY`).
  Sin key ⇒ `is_embeddings_enabled()=False` ⇒ degradación limpia a
  keyword/recency (cero crash). Activación = setear key + restart.
- Batching: máx 96 textos por request (límite API v2); `_CohereEmbeddingsClient`
  troza automáticamente. SDK `cohere.ClientV2`, response `.embeddings.float_`.
- `get_text_embedding` JAMÁS lanza — error ⇒ `[]` + `logger.error` con
  `provider=%s, model=%s` (convención P3-3).

## Vision

Sigue `disabled` (`MEALFIT_VISION_PROVIDER`): el chat-agent/scanner de
imágenes requiere un VLM que DeepSeek no ofrece. Embed v4 SÍ soporta
embeddings de imagen — candidato para búsqueda visual futura del diary,
decisión parqueada por el owner (2026-06-12: "lo de imágenes por ahora vacío").

## Verificación

- Test ancla: [`test_p1_cohere_embed_v4.py`](../tests/test_p1_cohere_embed_v4.py)
  (defaults, gating, asimetría input_type con ClientV2 fake, batching 96,
  cache versioning, contratos de la migración SQL).
- Validación semántica viva es-DO (2026-06-12): query relevante cosine 0.38 vs
  irrelevante 0.076 — separación sana para el RAG del formulario.
