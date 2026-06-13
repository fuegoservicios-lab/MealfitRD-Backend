"""[P1-COHERE-EMBED-V4 · 2026-06-12] Test ancla de la migración de embeddings
a Cohere Embed v4 (`embed-v4.0`).

Decisión del owner: stack de DOS providers — DeepSeek (LLM, ver
P0-DEEPSEEK-MIGRATION) + Cohere (embeddings). Motivación: precisión
multilingüe (es-DO) en el RAG del formulario de planes y la auditoría de
adherencia de los chunks de aprendizaje.

Contratos que ancla:
  A. Defaults: provider=cohere, model=embed-v4.0, dimensión=1536 (validator
     solo acepta 256/512/1024/1536 — los output_dimension reales de v4).
  B. Gating por key: sin COHERE_API_KEY el sistema se comporta como disabled
     (None + model_id 'disabled') — la activación es key + restart, sin
     spam de errores.
  C. Asimetría input_type (la palanca de precisión del RAG): query-side →
     search_query; document-side (lo persistido en pgvector) →
     search_document. `embed_documents`/`embed_query` del cliente y el
     param `purpose` de `get_text_embedding`.
  D. output_dimension=1536 viaja al API y el model_id versionado incluye la
     dimensión (`embed-v4.0@1536`) — espacios Matryoshka distintos no deben
     compartir cache keys.
  E. Batching ≤96 textos por request (límite del API v2/embed).
  F. Versionado de caches: los wrappers cacheados (fact_extractor /
     vision_agent / constants) llevan model_id (+purpose) en la key — sin
     esto, un switch de provider serviría vectores del espacio ANTERIOR
     desde Redis (TTL ~100 años).
  G. Purpose=document en los sitios que PERSISTEN a pgvector (user_facts,
     visual_diary).
  H. Migración SSOT vector(768)→vector(1536) presente en AMBOS dirs,
     byte-idéntica, idempotente, con NULL de vectores legacy y sanity.
  I. requirements.txt pinea el SDK cohere.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------------------------
# Fake del SDK cohere (se inyecta en sys.modules ANTES del lazy import)
# ---------------------------------------------------------------------------
class _FakeClientV2:
    last_instance = None

    def __init__(self, api_key=None, timeout=None):
        self.api_key = api_key
        self.timeout = timeout
        self.calls = []
        _FakeClientV2.last_instance = self

    def embed(self, *, texts, model, input_type, output_dimension, embedding_types):
        self.calls.append(
            {
                "n_texts": len(texts),
                "model": model,
                "input_type": input_type,
                "output_dimension": output_dimension,
                "embedding_types": list(embedding_types),
            }
        )
        vectors = [[0.5] * 8 for _ in texts]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))


@pytest.fixture()
def cohere_activo(monkeypatch):
    """Activa el provider cohere con key fake + SDK fake; resetea el singleton."""
    import embeddings_provider as ep

    monkeypatch.delenv("MEALFIT_EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("MEALFIT_EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_EMBEDDINGS_DIMENSION", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", "key-fake-de-test")
    monkeypatch.setitem(sys.modules, "cohere", SimpleNamespace(ClientV2=_FakeClientV2))
    ep._client = None
    ep._client_provider_fingerprint = None
    yield ep
    ep._client = None
    ep._client_provider_fingerprint = None


# ---------------------------------------------------------------------------
# A. Defaults
# ---------------------------------------------------------------------------
def test_a_defaults(monkeypatch):
    import embeddings_provider as ep

    monkeypatch.delenv("MEALFIT_EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("MEALFIT_EMBEDDINGS_MODEL", raising=False)
    monkeypatch.delenv("MEALFIT_EMBEDDINGS_DIMENSION", raising=False)
    assert ep._embeddings_provider() == "cohere"
    assert ep._embeddings_model() == "embed-v4.0"
    assert ep._embeddings_dimension() == 1536


def test_a2_dimension_validator_rejects_invalid(monkeypatch):
    import embeddings_provider as ep

    monkeypatch.setenv("MEALFIT_EMBEDDINGS_DIMENSION", "999")
    assert ep._embeddings_dimension() == 1536, (
        "999 no es un output_dimension válido de embed-v4.0 — debe caer al "
        "default 1536 (validator del knob)."
    )
    monkeypatch.setenv("MEALFIT_EMBEDDINGS_DIMENSION", "1024")
    assert ep._embeddings_dimension() == 1024


# ---------------------------------------------------------------------------
# B. Gating por presencia de key (fail-quiet a disabled)
# ---------------------------------------------------------------------------
def test_b_sin_key_se_comporta_disabled(monkeypatch):
    import embeddings_provider as ep

    monkeypatch.delenv("MEALFIT_EMBEDDINGS_PROVIDER", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
    assert ep.is_embeddings_enabled() is False
    assert ep.get_embeddings_model_id() == "disabled"
    assert ep.get_text_embedding("habichuelas con dulce") is None
    assert ep.get_embeddings_client() is None


def test_b2_key_generica_tambien_activa(monkeypatch):
    import embeddings_provider as ep

    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "key-generica")
    assert ep.is_embeddings_enabled() is True
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# C/D. Asimetría input_type + output_dimension + model_id versionado
# ---------------------------------------------------------------------------
def test_c_query_vs_document_input_types(cohere_activo):
    ep = cohere_activo

    emb_q = ep.get_text_embedding("¿qué desayuno alto en proteína?", purpose="query")
    emb_d = ep.get_text_embedding("Al usuario le encanta el mangú", purpose="document")
    assert emb_q and emb_d

    calls = _FakeClientV2.last_instance.calls
    assert calls[0]["input_type"] == "search_query"
    assert calls[-1]["input_type"] == "search_document"
    # D: output_dimension viaja en CADA request.
    assert all(c["output_dimension"] == 1536 for c in calls)
    assert all(c["model"] == "embed-v4.0" for c in calls)


def test_c2_cliente_asimetrico_por_metodo(cohere_activo):
    ep = cohere_activo
    client = ep.get_embeddings_client()
    assert client is not None

    client.embed_query("query de prueba")
    client.embed_documents(["doc uno", "doc dos"])
    calls = _FakeClientV2.last_instance.calls
    assert calls[-2]["input_type"] == "search_query"
    assert calls[-1]["input_type"] == "search_document"
    assert calls[-1]["n_texts"] == 2


def test_d_model_id_incluye_dimension(cohere_activo, monkeypatch):
    ep = cohere_activo
    assert ep.get_embeddings_model_id() == "embed-v4.0@1536"
    monkeypatch.setenv("MEALFIT_EMBEDDINGS_DIMENSION", "512")
    assert ep.get_embeddings_model_id() == "embed-v4.0@512", (
        "Dos dimensiones del mismo modelo son espacios vectoriales distintos "
        "(Matryoshka) — el model_id debe diferenciarlas para cache keys."
    )


def test_d2_purpose_invalido_cae_a_query(cohere_activo):
    ep = cohere_activo
    emb = ep.get_text_embedding("texto", purpose="lo-que-sea")
    assert emb
    assert _FakeClientV2.last_instance.calls[-1]["input_type"] == "search_query"


# ---------------------------------------------------------------------------
# E. Batching ≤96
# ---------------------------------------------------------------------------
def test_e_batching_96(cohere_activo):
    ep = cohere_activo
    client = ep.get_embeddings_client()
    vectors = client.embed_documents([f"texto {i}" for i in range(100)])
    assert len(vectors) == 100
    batch_calls = _FakeClientV2.last_instance.calls
    assert [c["n_texts"] for c in batch_calls] == [96, 4], (
        "v2/embed acepta máx 96 textos por request — 100 docs deben ir en 2 batches."
    )


# ---------------------------------------------------------------------------
# F. Versionado de caches (parser-based)
# ---------------------------------------------------------------------------
def test_f_fact_extractor_cache_versionada():
    src = (BACKEND / "fact_extractor.py").read_text(encoding="utf-8")
    assert re.search(
        r"def _cached_text_embedding\(text: str, model_id: str, purpose: str\)", src
    ), (
        "P1-COHERE-EMBED-V4: la capa cacheada de fact_extractor debe llevar "
        "model_id y purpose como ARGUMENTOS (la cache key del decorador se "
        "construye con args) — sin esto un switch de provider sirve vectores "
        "del espacio anterior desde Redis."
    )
    assert "get_embeddings_model_id()" in src


def test_f2_vision_cache_versionada():
    src = (BACKEND / "vision_agent.py").read_text(encoding="utf-8")
    assert re.search(
        r"def _cached_multimodal_embedding\(text: str, model_id: str, purpose: str\)",
        src,
    )


def test_f3_constants_lru_versionada():
    src = (BACKEND / "constants.py").read_text(encoding="utf-8")
    assert "cache_key = (get_embeddings_model_id(), text)" in src, (
        "El LRU in-process de constants.get_embedding debe keyear por "
        "(model_id, text)."
    )


def test_f4_redis_semantic_cache_key_versionada():
    src = (BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "from embeddings_provider import get_embeddings_model_id" in src
    assert "_model_hash(get_embeddings_model_id())" in src


# ---------------------------------------------------------------------------
# G. Purpose=document en los sitios que persisten a pgvector
# ---------------------------------------------------------------------------
def test_g_document_side_en_persistencia():
    fact_src = (BACKEND / "fact_extractor.py").read_text(encoding="utf-8")
    assert 'get_embedding(fact_text, purpose="document")' in fact_src
    assert 'get_embedding(mf["fact_text"], purpose="document")' in fact_src

    vision_src = (BACKEND / "vision_agent.py").read_text(encoding="utf-8")
    assert 'get_multimodal_embedding(description, purpose="document")' in vision_src

    diary_src = (BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert 'get_multimodal_embedding(description, purpose="document")' in diary_src


# ---------------------------------------------------------------------------
# H. Migración SSOT en ambos dirs, idempotente
# ---------------------------------------------------------------------------
_MIG_NAME = "p1_cohere_embed_v4_vector_dims_2026_06_12.sql"


def test_h_migracion_ssot_ambos_dirs():
    root_mig = BACKEND.parent / "supabase" / "migrations" / _MIG_NAME
    backend_mig = BACKEND / "supabase" / "migrations" / _MIG_NAME
    assert root_mig.exists(), f"Falta {root_mig} (dir SSOT workspace-root)"
    assert backend_mig.exists(), f"Falta {backend_mig} (dir SSOT backend)"
    assert root_mig.read_bytes() == backend_mig.read_bytes(), (
        "P3-MIGRATIONS-SSOT: las dos copias deben ser byte-idénticas."
    )


def test_h2_migracion_contratos():
    sql = (BACKEND / "supabase" / "migrations" / _MIG_NAME).read_text(encoding="utf-8")
    # Idempotencia: gate por dimensión actual.
    assert sql.count("IS DISTINCT FROM 1536") >= 4  # 3 gates + sanity
    # NULL de vectores legacy (espacio Gemini incomparable) ANTES del ALTER.
    assert "SET embedding = NULL" in sql
    assert "SET profile_embedding = NULL" in sql
    # Los 3 ALTER a 1536.
    assert sql.count("TYPE vector(1536)") == 3
    # Índices hnsw recreados de forma idempotente.
    assert "DROP INDEX IF EXISTS public.idx_user_facts_embedding" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_user_facts_embedding" in sql
    assert "CREATE INDEX IF NOT EXISTS meal_plans_profile_emb_idx" in sql
    # Sanity final que aborta si alguna columna vector quedó fuera de 1536.
    assert "RAISE EXCEPTION" in sql


# ---------------------------------------------------------------------------
# I. Dependencia pineada
# ---------------------------------------------------------------------------
def test_i_requirements_cohere():
    req = (BACKEND / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^cohere==", req, re.MULTILINE), (
        "requirements.txt debe pinear el SDK `cohere` (embeddings_provider "
        "lo importa lazy cuando el provider está activo)."
    )
