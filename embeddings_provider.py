"""[P0-LLM-PROVIDER-MIGRATION · 2026-06-12 → P1-COHERE-EMBED-V4 · 2026-06-12]
Capa pluggable de embeddings. Provider de producción: **Cohere Embed v4**.

Decisión del owner (2026-06-12): el stack queda en DOS providers — GLM
para LLM (ver `llm_provider.py`) y Cohere para embeddings. Embed v4
(`embed-v4.0`) por su precisión multilingüe (español dominicano) en el RAG
del formulario de planes y la auditoría de adherencia de los chunks de
aprendizaje.

Surfaces que delegan acá (SSOT):
  - `constants.get_embedding`              (matching semántico de ingredientes)
  - `fact_extractor.get_embedding`         (user_facts → pgvector + RAG)
  - `vision_agent.get_multimodal_embedding`(visual diary — embebe el TEXTO de
    la descripción; Embed v4 soporta imágenes si en el futuro se quiere
    búsqueda imagen-a-imagen)
  - `shopping_calculator` (semantic cache: `get_embeddings_client()`)

### Asimetría `input_type` (la palanca de precisión del RAG)

Embed v4 es un modelo ASIMÉTRICO: los textos ALMACENADOS deben embeberse con
`input_type="search_document"` y las CONSULTAS con `input_type="search_query"`.
Por eso `get_text_embedding(text, purpose=...)` acepta `purpose`:
  - `"query"` (default) — lados de búsqueda (RAG del chat, deep-search del
    pipeline, comparaciones simétricas texto-a-texto como la auditoría de
    adherencia y el semantic-plan-cache, donde AMBOS lados usan query).
  - `"document"` — SOLO los paths que persisten a pgvector para retrieval
    asimétrico: `user_facts.embedding` y `visual_diary.embedding`.
El cliente de `get_embeddings_client()` ya es asimétrico por método:
`embed_documents` → search_document, `embed_query` → search_query.

### Dimensión y compatibilidad pgvector

`MEALFIT_EMBEDDINGS_DIMENSION` (default **1536**, valores válidos de Embed v4:
256/512/1024/1536). Las columnas pgvector se migraron a vector(1536)
(migración SSOT `p1_cohere_embed_v4_vector_dims_2026_06_12.sql`); los vectores
legacy de Gemini (768) se anularon — espacios vectoriales distintos no son
comparables. Si cambias la dimensión via knob, los vectores ya persistidos NO
matchean la columna → coordina una migración antes.

### Versionado de cache

`get_embeddings_model_id()` retorna `"<model>@<dim>"` (e.g.
`"embed-v4.0@1536"`) y los wrappers cacheados (fact_extractor / vision_agent /
constants) lo incluyen en su cache key + purpose. Sin esto, el caché Redis
(TTL ~100 años) serviría vectores del provider ANTERIOR tras un switch —
mezcla de espacios vectoriales silenciosa (bug real detectado en la
migración Gemini→Cohere).

Activación (gating por presencia de key):
  - Default `MEALFIT_EMBEDDINGS_PROVIDER=cohere` + `MEALFIT_EMBEDDINGS_MODEL=
    embed-v4.0`, pero `is_embeddings_enabled()` exige `COHERE_API_KEY` (o el
    genérico `EMBEDDINGS_API_KEY`) en env. Sin key ⇒ se comporta como
    `disabled`: `get_text_embedding` → None con UN warning por proceso y los
    callers degradan (búsqueda keyword/recency, facts sin vector se saltan).
    Setear la key + restart es el ÚNICO paso de activación.
  - `openai_compatible` queda disponible para cualquier endpoint /embeddings
    estilo OpenAI (rollback / provider alternativo sin tocar código).

Tooltip-anchor: P0-LLM-PROVIDER-MIGRATION-EMBEDDINGS · P1-COHERE-EMBED-V4.
"""
from __future__ import annotations

from collections import OrderedDict
import logging
import os
import threading
import time
from typing import List, Optional

from knobs import _env_float, _env_int, _env_str

logger = logging.getLogger(__name__)

PROVIDER_DISABLED = "disabled"
PROVIDER_COHERE = "cohere"
PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"

# Valores de output_dimension soportados por embed-v4.0 (docs.cohere.com,
# verificado 2026-06-12). El validator del knob rechaza cualquier otro.
_COHERE_V4_DIMENSIONS = (256, 512, 1024, 1536)

# Cohere v2/embed acepta máximo 96 textos por request.
_COHERE_MAX_BATCH = 96

_VALID_PURPOSES = ("query", "document")

_warned_disabled = False
_client = None
_client_lock = threading.Lock()
_client_provider_fingerprint = None

# [Etapa 2 · Hydrogen Efficiency] Caché RAM corta (TTL) + singleflight para
# embeddings de búsqueda (`purpose="query"`): elimina la doble llamada HTTP a
# Cohere cuando `_rag_text_unit` y `_rag_visual_unit` embeben la misma query.
_QUERY_CACHE_LOCK = threading.Lock()
_QUERY_EMBED_CACHE: "OrderedDict[tuple, tuple[float, List[float]]]" = OrderedDict()
_QUERY_INFLIGHT: "dict[tuple, threading.Event]" = {}
_QUERY_CACHE_MAX = 256


def _query_embed_cache_ttl_s() -> float:
    try:
        return max(0.0, float(_env_float("MEALFIT_QUERY_EMBED_CACHE_TTL_S", 120.0)))
    except Exception:
        return 120.0


def clear_query_embed_cache() -> None:
    with _QUERY_CACHE_LOCK:
        _QUERY_EMBED_CACHE.clear()


def _embeddings_provider() -> str:
    return _env_str(
        "MEALFIT_EMBEDDINGS_PROVIDER",
        PROVIDER_COHERE,
        choices={PROVIDER_DISABLED, PROVIDER_COHERE, PROVIDER_OPENAI_COMPATIBLE},
    )


def _embeddings_model() -> str:
    return _env_str("MEALFIT_EMBEDDINGS_MODEL", "embed-v4.0")


def _embeddings_base_url() -> str:
    """Solo para `openai_compatible`. Cohere usa su SDK nativo (sin base_url)."""
    return _env_str("MEALFIT_EMBEDDINGS_BASE_URL", "")


def _embeddings_dimension() -> int:
    """[P1-COHERE-EMBED-V4] output_dimension de Embed v4. Default 1536 (máxima
    precisión — el objetivo declarado de la migración); las columnas pgvector
    están en vector(1536). Cambiarla exige migración de columnas coordinada."""
    return _env_int(
        "MEALFIT_EMBEDDINGS_DIMENSION",
        1536,
        validator=lambda v: v in _COHERE_V4_DIMENSIONS,
    )


def _embeddings_timeout_s() -> float:
    """[P2-LLM-TIMEOUT-SWEEP] Mismo contrato que el knob legacy de Gemini:
    deadline duro del request de embedding para no colgar threads/event loop."""
    return _env_float(
        "MEALFIT_EMBEDDING_LLM_TIMEOUT_S",
        15.0,
        validator=lambda v: 0.0 < v <= 60.0,
    )


def _embeddings_api_key() -> str:
    """Key del provider de embeddings. `COHERE_API_KEY` (estándar del SDK)
    con fallback al genérico `EMBEDDINGS_API_KEY` (contrato documentado en
    P0-LLM-PROVIDER-MIGRATION para providers openai_compatible)."""
    return (
        os.environ.get("COHERE_API_KEY") or os.environ.get("EMBEDDINGS_API_KEY") or ""
    ).strip()


def is_embeddings_enabled() -> bool:
    """True si hay provider activo Y configuración mínima para operar.

    Para `cohere` la key es parte del gate: el default de producción es
    provider=cohere, así que sin `COHERE_API_KEY` el sistema se comporta
    como disabled (degradación limpia) en vez de error-spamear cada call.
    """
    provider = _embeddings_provider()
    if provider == PROVIDER_COHERE:
        return bool(_embeddings_model()) and bool(_embeddings_api_key())
    if provider == PROVIDER_OPENAI_COMPATIBLE:
        return bool(_embeddings_model()) and bool(_embeddings_base_url())
    return False


def get_embeddings_model_id() -> str:
    """ID del espacio vectorial activo, o `"disabled"`.

    Incluye la DIMENSIÓN (`embed-v4.0@1536`): dos dimensiones del mismo
    modelo son espacios distintos (Matryoshka) — cache keys y fingerprints
    deben diferenciarlas. Consumido por los wrappers cacheados y por la
    Redis key del semantic cache de shopping_calculator.
    """
    if not is_embeddings_enabled():
        return "disabled"
    if _embeddings_provider() == PROVIDER_COHERE:
        return f"{_embeddings_model()}@{_embeddings_dimension()}"
    return _embeddings_model()


class _CohereEmbeddingsClient:
    """Cliente Cohere v2 con la interfaz LangChain-like que consumen los
    callers (`embed_query` / `embed_documents`), asimetría `input_type`
    incluida y batching ≤96 (límite del API).

    No se usa `langchain-cohere` deliberadamente: el SDK nativo da control
    exacto de `input_type` / `output_dimension` / `embedding_types` sin
    acoplarse al matrix de versiones langchain-core.
    """

    def __init__(self):
        import cohere  # lazy: el SDK solo se carga si el provider está activo

        self._co = cohere.ClientV2(
            api_key=_embeddings_api_key(),
            timeout=_embeddings_timeout_s(),
        )
        self._model = _embeddings_model()
        self._dimension = _embeddings_dimension()

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), _COHERE_MAX_BATCH):
            batch = texts[i : i + _COHERE_MAX_BATCH]
            resp = self._co.embed(
                texts=batch,
                model=self._model,
                input_type=input_type,
                output_dimension=self._dimension,
                embedding_types=["float"],
            )
            # El SDK expone la lista como `.embeddings.float_` (alias pydantic
            # de `float`); tolerar ambos nombres entre versiones.
            vectors = getattr(resp.embeddings, "float_", None)
            if vectors is None:
                vectors = getattr(resp.embeddings, "float", None)
            if vectors is None:
                raise RuntimeError(
                    "Cohere embed response sin embeddings float "
                    f"(model={self._model}, batch={len(batch)})"
                )
            out.extend([list(v) for v in vectors])
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "search_query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "search_document")


def _build_client():
    """Singleton del cliente de embeddings (thread-safe). Se reconstruye si
    el fingerprint de config cambia (útil en tests que mutan env vars)."""
    global _client, _client_provider_fingerprint
    fingerprint = (
        _embeddings_provider(),
        _embeddings_model(),
        _embeddings_base_url(),
        _embeddings_dimension(),
        bool(_embeddings_api_key()),
    )
    with _client_lock:
        if _client is not None and _client_provider_fingerprint == fingerprint:
            return _client

        if _embeddings_provider() == PROVIDER_COHERE:
            _client = _CohereEmbeddingsClient()
        else:
            from langchain_openai import OpenAIEmbeddings  # lazy

            kwargs = {
                "model": _embeddings_model(),
                "api_key": _embeddings_api_key() or "MISSING_EMBEDDINGS_API_KEY",
                "timeout": _embeddings_timeout_s(),
                # Providers no-OpenAI no exponen el tokenizer tiktoken del modelo;
                # sin esto, OpenAIEmbeddings intenta chunkear por tokens y falla.
                "check_embedding_ctx_length": False,
            }
            base_url = _embeddings_base_url()
            if base_url:
                kwargs["base_url"] = base_url
            _client = OpenAIEmbeddings(**kwargs)
        _client_provider_fingerprint = fingerprint
        return _client


def get_embeddings_client():
    """Cliente de embeddings (`embed_query`/`embed_documents`, asimétrico por
    método) o None si el provider está disabled/sin key o la construcción
    falla. Para callers que necesitan batch embedding (e.g. semantic cache
    de ingredientes en shopping_calculator)."""
    if not is_embeddings_enabled():
        return None
    try:
        return _build_client()
    except Exception as e:
        logger.error(
            "❌ [EMBEDDINGS] No se pudo construir el cliente de embeddings "
            "(provider=%s): %s: %s",
            _embeddings_provider(),
            type(e).__name__,
            str(e)[:200],
        )
        return None


def get_text_embedding(text: str, purpose: str = "query") -> Optional[List[float]]:
    """Embedding del texto, o `None` si provider disabled/sin key o fallo.

    `purpose`: `"query"` (default — lados de búsqueda y comparaciones
    simétricas) o `"document"` (textos que se PERSISTEN para retrieval
    asimétrico: user_facts, visual_diary). Ver nota de asimetría en el
    docstring del módulo — es la palanca de precisión del RAG.

    Contrato para callers (sin cambios desde la era Gemini): `None` ⇒
    degradar a su fallback no-semántico. JAMÁS lanza — un outage del
    provider de embeddings no puede tumbar extracción de facts ni
    generación de planes.
    """
    global _warned_disabled
    if not text or not isinstance(text, str) or not text.strip():
        return None
    if purpose not in _VALID_PURPOSES:
        purpose = "query"

    if not is_embeddings_enabled():
        if not _warned_disabled:
            logger.warning(
                "⚠️ [EMBEDDINGS] Provider de embeddings INACTIVO (provider=%s; "
                "para Cohere falta COHERE_API_KEY en env). Búsqueda semántica "
                "degradada a keyword/recency. Este aviso se emite una vez.",
                _embeddings_provider(),
            )
            _warned_disabled = True
        return None

    if purpose == "query":
        try:
            from turn_cache import get_cached as _tc_get
            _hit, _tc_vec = _tc_get("query_emb", f"{get_embeddings_model_id()}:{text}")
            if _hit and isinstance(_tc_vec, list) and _tc_vec:
                return list(_tc_vec)
        except Exception:
            pass

    ttl_s = _query_embed_cache_ttl_s() if purpose == "query" else 0.0

    def _compute() -> Optional[List[float]]:
        client = _build_client()
        if ttl_s > 0.0:
            ckey = (id(client), get_embeddings_model_id(), purpose, text)
            now = time.monotonic()
            with _QUERY_CACHE_LOCK:
                hit = _QUERY_EMBED_CACHE.get(ckey)
                if hit is not None:
                    ts, vec = hit
                    if now - ts <= ttl_s and vec:
                        _QUERY_EMBED_CACHE.move_to_end(ckey)
                        return list(vec)
                    _QUERY_EMBED_CACHE.pop(ckey, None)
        if purpose == "document" and hasattr(client, "embed_documents"):
            emb = client.embed_documents([text])[0]
        else:
            emb = client.embed_query(text)
        if isinstance(emb, list) and emb:
            if ttl_s > 0.0:
                ckey = (id(client), get_embeddings_model_id(), purpose, text)
                with _QUERY_CACHE_LOCK:
                    _QUERY_EMBED_CACHE[ckey] = (time.monotonic(), list(emb))
                    _QUERY_EMBED_CACHE.move_to_end(ckey)
                    while len(_QUERY_EMBED_CACHE) > _QUERY_CACHE_MAX:
                        _QUERY_EMBED_CACHE.popitem(last=False)
            return list(emb)
        return None

    try:
        if purpose == "query":
            sf_key = (_embeddings_provider(), get_embeddings_model_id(), purpose, text)
            while True:
                with _QUERY_CACHE_LOCK:
                    ev = _QUERY_INFLIGHT.get(sf_key)
                    if ev is None:
                        ev = threading.Event()
                        _QUERY_INFLIGHT[sf_key] = ev
                        leader = True
                    else:
                        leader = False
                if leader:
                    try:
                        res = _compute()
                        if res:
                            try:
                                from turn_cache import set_cached as _tc_set
                                _tc_set("query_emb", f"{get_embeddings_model_id()}:{text}", list(res))
                            except Exception:
                                pass
                        return res
                    finally:
                        with _QUERY_CACHE_LOCK:
                            _QUERY_INFLIGHT.pop(sf_key, None)
                        ev.set()
                ev.wait(timeout=15.0)
                # Tras despertar, revisa cachés ya pobladas por el líder sin repetir llamada HTTP.
                try:
                    from turn_cache import get_cached as _tc_get
                    _hit, _tc_vec = _tc_get("query_emb", f"{get_embeddings_model_id()}:{text}")
                    if _hit and isinstance(_tc_vec, list) and _tc_vec:
                        return list(_tc_vec)
                except Exception:
                    pass
                res = _compute()
                return res
        return _compute()
    except Exception as e:
        # error-level (NO warning): convención P3-VISION-FAIL-ERROR-LOG /
        # P3-3-GET-EMBEDDING-LOGGER — una caída sostenida del provider debe
        # ser visible en Sentry/agregación, no degradar en silencio.
        logger.error(
            "❌ [EMBEDDINGS] get_text_embedding falló (provider=%s, model=%s): %s: %s",
            _embeddings_provider(),
            _embeddings_model() or "<unset>",
            type(e).__name__,
            str(e)[:200],
        )
        return None
