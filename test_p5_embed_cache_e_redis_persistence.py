"""[P5-EMBED-CACHE-E] Tests para la persistencia de embeddings de
master_ingredients en Redis.

Bug observable (corrida 2026-05-05 04:17 + corridas anteriores):
  Cada pipeline dispara `embed_documents (master_ingredients cache init)`
  contra Gemini. Pega contra el quota minute-window → 429
  RESOURCE_EXHAUSTED → 3 reintentos × backoff (1.5s + 8s) = ~14s
  desperdiciados por pipeline. El sistema cae al Regex Fast-Path
  graciosamente, pero es cost-of-doing-business recurrente.

Fix:
  Persistir vectores en Redis con key derivada del hash estable de la
  master_list (`name + category + aliases`). Si la lista no cambia,
  Redis sirve los vectores instantáneamente. TTL 7 días.

Cobertura:
  - Hash estable y dependiente del contenido
  - Hash diferente cuando cambia name / aliases / category
  - Read fallback cuando Redis down (None retornado, no exception)
  - Read fallback cuando length mismatch (sanity check)
  - Write fallback cuando Redis down (False retornado, no exception)
  - get_semantic_cache: hit Redis → no llama Gemini
  - get_semantic_cache: miss Redis → llama Gemini → persiste
  - Key versionada para invalidaciones futuras
"""
from unittest.mock import patch, MagicMock

import pytest


def _sample_master_list():
    return [
        {"name": "Pollo", "category": "Proteínas", "aliases": ["pechuga"]},
        {"name": "Arroz", "category": "Granos", "aliases": ["arroz blanco"]},
        {"name": "Cebolla", "category": "Vegetales", "aliases": []},
    ]


# ---------------------------------------------------------------------------
# 1. Hash estabilidad y sensibilidad
# ---------------------------------------------------------------------------
class TestMasterListHash:
    def test_same_input_produces_same_hash(self):
        from shopping_calculator import _master_list_hash
        lst = _sample_master_list()
        assert _master_list_hash(lst) == _master_list_hash(lst)

    def test_order_independent(self):
        """Postgres no garantiza orden estable sin ORDER BY — el hash
        debe ser independiente del orden de devolución."""
        from shopping_calculator import _master_list_hash
        lst1 = _sample_master_list()
        lst2 = list(reversed(lst1))
        assert _master_list_hash(lst1) == _master_list_hash(lst2)

    def test_name_change_invalidates(self):
        from shopping_calculator import _master_list_hash
        lst1 = _sample_master_list()
        lst2 = [dict(m) for m in lst1]
        lst2[0]["name"] = "Pollo Asado"
        assert _master_list_hash(lst1) != _master_list_hash(lst2)

    def test_alias_change_invalidates(self):
        from shopping_calculator import _master_list_hash
        lst1 = _sample_master_list()
        lst2 = [dict(m) for m in lst1]
        lst2[0]["aliases"] = ["pechuga", "muslo"]
        assert _master_list_hash(lst1) != _master_list_hash(lst2)

    def test_category_change_invalidates(self):
        from shopping_calculator import _master_list_hash
        lst1 = _sample_master_list()
        lst2 = [dict(m) for m in lst1]
        lst2[0]["category"] = "Otra"
        assert _master_list_hash(lst1) != _master_list_hash(lst2)

    def test_hash_is_short_hex(self):
        """Trunca a 16 chars hex para mantener key Redis manejable."""
        from shopping_calculator import _master_list_hash
        h = _master_list_hash(_sample_master_list())
        assert len(h) == 16
        int(h, 16)  # debe parsear como hex


# ---------------------------------------------------------------------------
# 2. Redis cache key
# ---------------------------------------------------------------------------
class TestRedisKey:
    def test_key_includes_versioned_prefix(self):
        """Versionar la key permite invalidar todo cuando cambia el modelo
        de embedding o la fórmula de texto."""
        from shopping_calculator import _redis_embed_cache_key
        key = _redis_embed_cache_key(_sample_master_list())
        assert key.startswith("embed:master_ingredients:v1:")

    def test_key_differs_when_master_changes(self):
        from shopping_calculator import _redis_embed_cache_key
        lst1 = _sample_master_list()
        lst2 = [dict(m) for m in lst1]
        lst2[0]["name"] = "Otro"
        assert _redis_embed_cache_key(lst1) != _redis_embed_cache_key(lst2)


# ---------------------------------------------------------------------------
# 3. Read fallback (Redis down / corrupt / mismatch)
# ---------------------------------------------------------------------------
class TestReadFallback:
    def test_returns_none_when_redis_unavailable(self):
        """Si redis_client es None (cache_manager no pudo conectar al
        boot), `_try_load` debe devolver None sin lanzar."""
        import shopping_calculator as sc
        with patch("cache_manager.redis_client", None):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())
        assert result is None

    def test_returns_none_when_no_entry(self):
        """Cache miss típico: Redis OK pero sin entry para este hash."""
        import shopping_calculator as sc
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        with patch("cache_manager.redis_client", fake_redis):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())
        assert result is None
        fake_redis.get.assert_called_once()

    def test_returns_none_when_json_corrupt(self):
        """Si Redis tiene basura, fallback a None — no derribamos pipeline."""
        import shopping_calculator as sc
        fake_redis = MagicMock()
        fake_redis.get.return_value = "not_json{{["
        with patch("cache_manager.redis_client", fake_redis):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())
        assert result is None

    def test_returns_none_when_length_mismatch(self):
        """Sanity: si Redis tiene 3 vectores pero master_list ahora tiene
        4 items (raza condition), invalidamos el entry y refetch."""
        import shopping_calculator as sc
        import json
        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps([[0.1] * 768])  # 1 vector
        with patch("cache_manager.redis_client", fake_redis):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())  # 3 items
        assert result is None

    def test_returns_vectors_on_hit(self):
        import shopping_calculator as sc
        import json
        vectors = [[0.1] * 5, [0.2] * 5, [0.3] * 5]  # 3 vectores
        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(vectors)
        with patch("cache_manager.redis_client", fake_redis):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())
        assert result == vectors

    def test_redis_exception_returns_none_no_raise(self):
        """Connection refused / timeout / etc. — degradar a None."""
        import shopping_calculator as sc
        fake_redis = MagicMock()
        fake_redis.get.side_effect = Exception("Connection refused")
        with patch("cache_manager.redis_client", fake_redis):
            result = sc._try_load_embed_vectors_from_redis(_sample_master_list())
        assert result is None


# ---------------------------------------------------------------------------
# 4. Write fallback
# ---------------------------------------------------------------------------
class TestWriteFallback:
    def test_returns_false_when_redis_unavailable(self):
        import shopping_calculator as sc
        with patch("cache_manager.redis_client", None):
            ok = sc._persist_embed_vectors_to_redis(_sample_master_list(), [[0.1]])
        assert ok is False

    def test_returns_true_on_success(self):
        import shopping_calculator as sc
        fake_redis = MagicMock()
        fake_redis.setex.return_value = True
        with patch("cache_manager.redis_client", fake_redis):
            ok = sc._persist_embed_vectors_to_redis(
                _sample_master_list(), [[0.1] * 5, [0.2] * 5, [0.3] * 5]
            )
        assert ok is True
        # Verificar que llamó setex con TTL >= 1 día (we use 7d)
        args, _ = fake_redis.setex.call_args
        assert args[1] >= 86400  # >= 1 día

    def test_redis_exception_returns_false_no_raise(self):
        """Si Redis explota durante write, no lo propagamos —
        el cache in-process sigue funcionando."""
        import shopping_calculator as sc
        fake_redis = MagicMock()
        fake_redis.setex.side_effect = Exception("OOM")
        with patch("cache_manager.redis_client", fake_redis):
            ok = sc._persist_embed_vectors_to_redis(
                _sample_master_list(), [[0.1] * 5, [0.2] * 5, [0.3] * 5]
            )
        assert ok is False


# ---------------------------------------------------------------------------
# 5. Integración con get_semantic_cache (Redis hit evita Gemini)
# ---------------------------------------------------------------------------
class TestGetSemanticCacheRedisIntegration:
    def setup_method(self):
        # Resetear el caché in-process antes de cada test para ejercer la
        # ruta completa.
        import shopping_calculator as sc
        sc._semantic_cache = None
        sc._semantic_cache_failed_until = 0.0

    def teardown_method(self):
        import shopping_calculator as sc
        sc._semantic_cache = None
        sc._semantic_cache_failed_until = 0.0

    def test_redis_hit_skips_gemini_embed_documents(self):
        """[P5-EMBED-CACHE-E] El path crítico del fix: si Redis tiene los
        vectores, NO debemos llamar a Gemini's embed_documents."""
        import shopping_calculator as sc
        import json

        master = _sample_master_list()
        cached_vectors = [[0.1] * 5, [0.2] * 5, [0.3] * 5]

        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(cached_vectors)

        # Mock embedding client — embed_documents NO debe llamarse.
        fake_embeddings = MagicMock()
        fake_embeddings_class = MagicMock(return_value=fake_embeddings)

        with patch("shopping_calculator.get_master_ingredients", return_value=master), \
             patch("cache_manager.redis_client", fake_redis), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", fake_embeddings_class):
            cache = sc.get_semantic_cache()

        assert cache is not None
        assert cache["vectors"] == cached_vectors
        # Crítico: embed_documents NO se llamó (ahorramos Gemini quota).
        fake_embeddings.embed_documents.assert_not_called()

    def test_redis_miss_calls_gemini_and_persists(self):
        """Camino normal de cold-start: Redis vacío → Gemini → persist."""
        import shopping_calculator as sc

        master = _sample_master_list()
        fake_vectors = [[0.5] * 5, [0.6] * 5, [0.7] * 5]

        fake_redis = MagicMock()
        fake_redis.get.return_value = None  # miss

        fake_embeddings = MagicMock()
        fake_embeddings.embed_documents.return_value = fake_vectors
        fake_embeddings_class = MagicMock(return_value=fake_embeddings)

        with patch("shopping_calculator.get_master_ingredients", return_value=master), \
             patch("cache_manager.redis_client", fake_redis), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", fake_embeddings_class):
            cache = sc.get_semantic_cache()

        assert cache is not None
        assert cache["vectors"] == fake_vectors
        # Persist debió llamarse (setex con la key versionada)
        fake_redis.setex.assert_called_once()
        called_key = fake_redis.setex.call_args[0][0]
        assert called_key.startswith("embed:master_ingredients:v1:")
