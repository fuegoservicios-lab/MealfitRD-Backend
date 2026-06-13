"""[P6-EMBED-CACHE-FIX] Tests para el fix del bypass de cooldown en
el Redis cache de embeddings.

Bug observable (corrida 2026-05-05 14:01):
  Logs mostraron `embed_documents 429` con 3 reintentos + backoffs.
  Pero el cache Redis (P5-EMBED-CACHE-E) DEBÍA estar poblado tras la
  corrida 13:33 anterior. ¿Por qué se llamó Gemini igual?

Causa raíz:
  `get_semantic_cache` chequeaba el cooldown (`_semantic_cache_failed_until`)
  ANTES de intentar leer Redis. Un 429 reciente activaba 300s de cooldown,
  bloqueando TODA llamada a `get_semantic_cache` por 5 minutos — incluyendo
  los reads a Redis, que NO cuestan quota Gemini.

Fix:
  Reordenar el flujo:
    1. In-process cache check (fast)
    2. Lock + re-check
    3. **Try Redis FIRST** (no quota cost)
    4. Si Redis hit → poblar in-process cache, return
    5. Si Redis miss → AHORA chequear cooldown
    6. Si cooldown OK → llamar Gemini, persistir a Redis

Cobertura:
  - Cooldown activo + Redis hit → cache servido (NO Gemini call)
  - Cooldown activo + Redis miss → return None (sin pegar a Gemini)
  - Sin cooldown + Redis hit → cache servido
  - Sin cooldown + Redis miss → Gemini call + persist
  - In-process cache hit ignora todo (fast-path preservado)
"""
from unittest.mock import patch, MagicMock

import pytest


def _sample_master_list():
    return [
        {"name": "Pollo", "category": "Proteínas", "aliases": ["pechuga"]},
        {"name": "Arroz", "category": "Granos", "aliases": ["arroz blanco"]},
        {"name": "Cebolla", "category": "Vegetales", "aliases": []},
    ]


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Resetea state in-process antes y después de cada test."""
    import shopping_calculator as sc
    sc._semantic_cache = None
    sc._semantic_cache_failed_until = 0.0
    yield
    sc._semantic_cache = None
    sc._semantic_cache_failed_until = 0.0


# ---------------------------------------------------------------------------
# 1. Cooldown activo + Redis hit → SERVIDO (el fix)
# ---------------------------------------------------------------------------
def test_cooldown_active_redis_hit_serves_cache_without_gemini():
    """[P6-EMBED-CACHE-FIX] Caso del bug: 429 reciente activó cooldown,
    pero Redis tiene vectores válidos. PRE-fix retornaba None (gemini
    bloqueado, Redis nunca se intentaba). POST-fix sirve desde Redis."""
    import shopping_calculator as sc
    import time as _time
    import json

    master = _sample_master_list()
    cached_vectors = [[0.1] * 5, [0.2] * 5, [0.3] * 5]

    # Simular cooldown activo (Gemini falló hace 30s)
    sc._semantic_cache_failed_until = _time.time() + 270  # 4.5 min restantes

    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps(cached_vectors)

    fake_embeddings = MagicMock()
    fake_embeddings_class = MagicMock(return_value=fake_embeddings)

    with patch.object(sc, "get_master_ingredients", return_value=master), \
         patch("cache_manager.redis_client", fake_redis), \
         patch("embeddings_provider.get_embeddings_client", return_value=fake_embeddings):
        cache = sc.get_semantic_cache()

    # Critical: Cache servido a pesar del cooldown
    assert cache is not None, (
        "Cooldown activo NO debe bloquear Redis read — los datos en Redis "
        "son válidos independiente del estado de Gemini"
    )
    assert cache["vectors"] == cached_vectors
    # Y Gemini NO se llamó
    fake_embeddings.embed_documents.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Cooldown activo + Redis miss → return None (no pegamos a Gemini)
# ---------------------------------------------------------------------------
def test_cooldown_active_redis_miss_returns_none_no_gemini():
    """Si Redis no tiene los datos Y cooldown sigue activo, retornamos
    None sin pegar a Gemini (preservar el contrato de fast-fail)."""
    import shopping_calculator as sc
    import time as _time

    master = _sample_master_list()
    sc._semantic_cache_failed_until = _time.time() + 270

    fake_redis = MagicMock()
    fake_redis.get.return_value = None  # Redis miss

    fake_embeddings = MagicMock()
    fake_embeddings_class = MagicMock(return_value=fake_embeddings)

    with patch.object(sc, "get_master_ingredients", return_value=master), \
         patch("cache_manager.redis_client", fake_redis), \
         patch("embeddings_provider.get_embeddings_client", return_value=fake_embeddings):
        cache = sc.get_semantic_cache()

    assert cache is None, "Redis miss + cooldown activo → fast-fail con None"
    fake_embeddings.embed_documents.assert_not_called()
    fake_redis.get.assert_called_once()  # Sí intentamos Redis


# ---------------------------------------------------------------------------
# 3. Sin cooldown + Redis hit → cache servido
# ---------------------------------------------------------------------------
def test_no_cooldown_redis_hit_serves_cache():
    """Path normal: sin cooldown, Redis hit → cache servido sin Gemini."""
    import shopping_calculator as sc
    import json

    master = _sample_master_list()
    cached_vectors = [[0.5] * 5, [0.6] * 5, [0.7] * 5]

    fake_redis = MagicMock()
    fake_redis.get.return_value = json.dumps(cached_vectors)

    fake_embeddings = MagicMock()
    fake_embeddings_class = MagicMock(return_value=fake_embeddings)

    with patch.object(sc, "get_master_ingredients", return_value=master), \
         patch("cache_manager.redis_client", fake_redis), \
         patch("embeddings_provider.get_embeddings_client", return_value=fake_embeddings):
        cache = sc.get_semantic_cache()

    assert cache is not None
    assert cache["vectors"] == cached_vectors
    fake_embeddings.embed_documents.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Sin cooldown + Redis miss → Gemini call + persist
# ---------------------------------------------------------------------------
def test_no_cooldown_redis_miss_calls_gemini_and_persists():
    """Cold start: sin cooldown, Redis vacío → Gemini → persist."""
    import shopping_calculator as sc

    master = _sample_master_list()
    fake_vectors = [[0.9] * 5, [0.8] * 5, [0.7] * 5]

    fake_redis = MagicMock()
    fake_redis.get.return_value = None

    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.return_value = fake_vectors
    fake_embeddings_class = MagicMock(return_value=fake_embeddings)

    with patch.object(sc, "get_master_ingredients", return_value=master), \
         patch("cache_manager.redis_client", fake_redis), \
         patch("embeddings_provider.get_embeddings_client", return_value=fake_embeddings):
        cache = sc.get_semantic_cache()

    assert cache is not None
    assert cache["vectors"] == fake_vectors
    fake_embeddings.embed_documents.assert_called_once()
    fake_redis.setex.assert_called_once()


# ---------------------------------------------------------------------------
# 5. In-process cache hit ignora todo (preservar fast-path)
# ---------------------------------------------------------------------------
def test_inprocess_cache_hit_skips_redis_and_gemini():
    """Si `_semantic_cache` ya tiene valor in-process, no debemos tocar
    NI Redis NI Gemini — fast return preservado."""
    import shopping_calculator as sc

    master = _sample_master_list()
    pre_cached = {"master_list": master, "vectors": [[1, 2, 3]], "embeddings_client": object()}
    sc._semantic_cache = pre_cached

    fake_redis = MagicMock()
    fake_embeddings = MagicMock()
    fake_embeddings_class = MagicMock(return_value=fake_embeddings)

    with patch.object(sc, "get_master_ingredients") as get_master, \
         patch("cache_manager.redis_client", fake_redis), \
         patch("embeddings_provider.get_embeddings_client", return_value=fake_embeddings):
        cache = sc.get_semantic_cache()

    assert cache is pre_cached
    # Ni get_master_ingredients ni Redis ni Gemini se tocaron
    get_master.assert_not_called()
    fake_redis.get.assert_not_called()
    fake_embeddings.embed_documents.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Sanity: source code refleja el reorder
# ---------------------------------------------------------------------------
def test_source_has_redis_before_cooldown_check():
    """Sanity guard: si alguien revierte el orden (cooldown ANTES de
    Redis), este test alerta."""
    import inspect
    import shopping_calculator as sc
    src = inspect.getsource(sc.get_semantic_cache)
    redis_pos = src.find("_try_load_embed_vectors_from_redis")
    cooldown_pos = src.find("_semantic_cache_failed_until")
    # El primer cooldown_pos puede estar en docstring/comments.
    # Buscar la primera APARICIÓN como check (con `<`):
    cooldown_check_pos = src.find("if _time.time() < _semantic_cache_failed_until")
    assert redis_pos > 0, "Redis read debe estar presente en get_semantic_cache"
    assert cooldown_check_pos > 0, "Cooldown check debe estar presente"
    assert redis_pos < cooldown_check_pos, (
        f"Redis read ({redis_pos}) debe venir ANTES del cooldown check "
        f"({cooldown_check_pos}) — el bug original era el orden inverso."
    )
    assert "P6-EMBED-CACHE-FIX" in src
