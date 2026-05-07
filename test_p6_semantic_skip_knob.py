"""[P6-SEMANTIC-SKIP] Tests para el kill-switch del semantic cache.

Bug observable (corridas 2026-05-05 múltiples):
  Cada pipeline desperdicia ~14s en 3×retries+backoff de 429s contra
  `embed_documents (master_ingredients cache init)`. Redis nunca se
  logra poblar porque persist solo corre tras Gemini exitoso, pero
  Gemini siempre falla por quota exhausto → loop sin salida.

Fix:
  Knob `MEALFIT_DISABLE_SEMANTIC_CACHE` que hace `get_semantic_cache`
  retornar None instantáneamente. Sistema cae al Regex Fast-Path (que
  cubre ~95% de casos comunes). Ahorra ~14s por pipeline.

Cobertura:
  - Default (env unset) → comportamiento previo (intenta Gemini/Redis)
  - Knob true → return None sin tocar Redis ni Gemini
  - Parsing de valores aceptados ('1', 'true', 'yes', 'on')
  - Parsing case-insensitive ('TRUE', 'True')
  - Valores no reconocidos → tratados como off ('false', '0', 'off', '')
  - Knob check sucede ANTES de cualquier otra acción (no toma lock)
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _reset_state():
    import shopping_calculator as sc
    sc._semantic_cache = None
    sc._semantic_cache_failed_until = 0.0
    # Limpiar env var por si tests previos la dejaron
    os.environ.pop("MEALFIT_DISABLE_SEMANTIC_CACHE", None)
    yield
    sc._semantic_cache = None
    sc._semantic_cache_failed_until = 0.0
    os.environ.pop("MEALFIT_DISABLE_SEMANTIC_CACHE", None)


# ---------------------------------------------------------------------------
# 1. Helper de parsing del knob
# ---------------------------------------------------------------------------
class TestKnobParsing:
    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "True", "On", "YES"])
    def test_truthy_values(self, val, monkeypatch):
        import shopping_calculator as sc
        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", val)
        assert sc._semantic_cache_disabled() is True, (
            f"Valor {val!r} debe ser truthy"
        )

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "False", "garbage", "  "])
    def test_falsy_values(self, val, monkeypatch):
        import shopping_calculator as sc
        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", val)
        assert sc._semantic_cache_disabled() is False, (
            f"Valor {val!r} debe ser falsy"
        )

    def test_unset_is_falsy(self, monkeypatch):
        """Default sin env var → no desactivar (preservar comportamiento)."""
        import shopping_calculator as sc
        monkeypatch.delenv("MEALFIT_DISABLE_SEMANTIC_CACHE", raising=False)
        assert sc._semantic_cache_disabled() is False

    def test_whitespace_stripped(self, monkeypatch):
        """Tolerar whitespace al final/inicio."""
        import shopping_calculator as sc
        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", "  true  ")
        assert sc._semantic_cache_disabled() is True


# ---------------------------------------------------------------------------
# 2. get_semantic_cache respeta el knob
# ---------------------------------------------------------------------------
class TestGetSemanticCacheRespectsKnob:
    def test_knob_on_returns_none_immediately(self, monkeypatch):
        """[P6-SEMANTIC-SKIP] Caso clave: knob on → None sin tocar nada.
        El test mockea Redis y Gemini para asegurar que NO se llaman."""
        import shopping_calculator as sc
        from unittest.mock import patch, MagicMock

        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", "true")

        fake_redis = MagicMock()
        fake_embeddings = MagicMock()
        fake_embeddings_class = MagicMock(return_value=fake_embeddings)

        with patch.object(sc, "get_master_ingredients") as get_master, \
             patch("cache_manager.redis_client", fake_redis), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", fake_embeddings_class):
            cache = sc.get_semantic_cache()

        assert cache is None
        # Confirmar que NO se tocaron Redis, Gemini, ni master_ingredients DB.
        # Fast-path completo, ~0ms.
        get_master.assert_not_called()
        fake_redis.get.assert_not_called()
        fake_embeddings.embed_documents.assert_not_called()

    def test_knob_off_default_attempts_normally(self, monkeypatch):
        """Default: knob unset → comportamiento previo. Intenta Redis,
        si miss intenta Gemini. (Verificamos que Redis SÍ se chequea.)"""
        import shopping_calculator as sc
        from unittest.mock import patch, MagicMock
        import json

        monkeypatch.delenv("MEALFIT_DISABLE_SEMANTIC_CACHE", raising=False)

        master = [
            {"name": "Pollo", "category": "Proteínas", "aliases": []},
        ]
        cached_vectors = [[0.1] * 5]
        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(cached_vectors)

        fake_embeddings = MagicMock()
        fake_embeddings_class = MagicMock(return_value=fake_embeddings)

        with patch.object(sc, "get_master_ingredients", return_value=master), \
             patch("cache_manager.redis_client", fake_redis), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", fake_embeddings_class):
            cache = sc.get_semantic_cache()

        assert cache is not None
        assert cache["vectors"] == cached_vectors
        # Confirma que SÍ se intentó Redis (vs el caso skip)
        fake_redis.get.assert_called_once()

    def test_knob_off_explicit_false_attempts_normally(self, monkeypatch):
        """`MEALFIT_DISABLE_SEMANTIC_CACHE=false` debe equivaler a unset."""
        import shopping_calculator as sc
        from unittest.mock import patch, MagicMock
        import json

        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", "false")

        master = [{"name": "Pollo", "category": "Proteínas", "aliases": []}]
        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps([[0.1] * 5])

        with patch.object(sc, "get_master_ingredients", return_value=master), \
             patch("cache_manager.redis_client", fake_redis), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", MagicMock()):
            cache = sc.get_semantic_cache()

        assert cache is not None  # cache devuelto, knob no estaba on
        fake_redis.get.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Knob bypass es estricto: ni siquiera in-process cache se devuelve
# ---------------------------------------------------------------------------
class TestKnobBypassIsStrict:
    def test_knob_on_bypasses_inprocess_cache_too(self, monkeypatch):
        """Si knob está on, NO devolvemos ni siquiera el in-process
        cache existente. Intencional: si operador desactivó, debe ser
        consistente — sin valor en cualquier path."""
        import shopping_calculator as sc

        # Pre-poblar in-process cache
        sc._semantic_cache = {
            "master_list": [{"name": "X"}],
            "vectors": [[0.5]],
            "embeddings_client": object(),
        }

        monkeypatch.setenv("MEALFIT_DISABLE_SEMANTIC_CACHE", "true")
        result = sc.get_semantic_cache()
        assert result is None, (
            "Knob on debe forzar None aunque in-process cache exista — "
            "consistencia operacional"
        )


# ---------------------------------------------------------------------------
# 4. Sanity: source code refleja el orden correcto
# ---------------------------------------------------------------------------
def test_source_checks_knob_first():
    """Sanity guard: el knob check debe ser la PRIMERA acción (antes
    de in-process cache, lock, Redis, Gemini)."""
    import inspect
    import shopping_calculator as sc

    src = inspect.getsource(sc.get_semantic_cache)
    knob_pos = src.find("_semantic_cache_disabled()")
    assert knob_pos > 0, "Knob check debe estar en get_semantic_cache"

    # Otros checks que deben venir DESPUÉS del knob
    inprocess_pos = src.find("if _semantic_cache is not None:")
    redis_pos = src.find("_try_load_embed_vectors_from_redis")
    cooldown_pos = src.find("if _time.time() < _semantic_cache_failed_until")

    assert knob_pos < inprocess_pos, (
        f"Knob ({knob_pos}) debe venir antes que in-process check ({inprocess_pos})"
    )
    assert knob_pos < redis_pos, (
        f"Knob ({knob_pos}) debe venir antes que Redis read ({redis_pos})"
    )
    assert knob_pos < cooldown_pos, (
        f"Knob ({knob_pos}) debe venir antes que cooldown check ({cooldown_pos})"
    )
    assert "P6-SEMANTIC-SKIP" in src
