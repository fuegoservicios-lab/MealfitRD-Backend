"""[P0-4] Tests para asegurar que `PersistentLLMCache.get`/`aget` emiten SQL
parametrizado correctamente para la condición de TTL.

Bug original (audit P0-4):
  Las dos lecturas del fallback DB del cache LLM emitían:
      "SELECT value FROM app_kv_store WHERE key = %s
       AND updated_at > now() - interval '%s seconds'"
  con `params=(key, self.ttl)`.

  Psycopg NO sustituye `%s` dentro de literales SQL (string entre comillas
  simples). El parser cuenta UN solo placeholder real (`key = %s`), pero
  recibimos DOS params → `ProgrammingError` SIEMPRE. Resultado: cuando
  Redis cae, el fallback DB del cache LLM tiene 100% miss garantizado y
  cada lectura paga ~5–50 ms de query + parse error silenciado por el
  `except Exception` outer.

Fix:
  Usar `make_interval(secs => %s)` que SÍ acepta el placeholder estándar
  fuera de comillas. Aplicado a ambos paths (sync `get` y async `aget`).
  Ningún cambio en los paths de escritura (no usaban interval).

Cobertura:
  - test_get_query_uses_make_interval_not_quoted_interval
  - test_aget_query_uses_make_interval_not_quoted_interval
  - test_get_query_placeholder_count_matches_params
  - test_aget_query_placeholder_count_matches_params
  - test_source_code_no_quoted_interval_with_placeholder
  - test_get_propagates_db_value_when_redis_unavailable
"""
import asyncio
import re
from unittest.mock import patch

import pytest

import graph_orchestrator
from graph_orchestrator import PersistentLLMCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_placeholders_outside_string_literals(sql: str) -> int:
    """Cuenta `%s` placeholders REALES (los que psycopg sustituirá) — los que
    están dentro de literales `'...'` no cuentan, igual que el parser real.
    """
    cleaned = re.sub(r"'[^']*'", "''", sql)
    return cleaned.count("%s")


# ---------------------------------------------------------------------------
# 1. Sync `get` — el SQL emitido tras P0-4.
# ---------------------------------------------------------------------------
def test_get_query_uses_make_interval_not_quoted_interval():
    """`get` debe emitir `make_interval(secs => %s)`, NO `interval '%s seconds'`."""
    cache = PersistentLLMCache(ttl_seconds=300)
    captured = {}

    def fake_execute(query, params=None, fetch_one=False, **kwargs):
        captured["query"] = query
        captured["params"] = params
        captured["fetch_one"] = fetch_one
        return None  # simula miss

    with patch.object(graph_orchestrator, "execute_sql_query", side_effect=fake_execute), \
         patch.object(graph_orchestrator, "redis_client", None):
        cache.get("k_test")

    assert "query" in captured, "get() no llamó a execute_sql_query"
    sql = captured["query"]
    assert "make_interval" in sql, f"esperado make_interval(...), got: {sql!r}"
    # Defensa explícita: el patrón histórico roto no debe reaparecer.
    assert "interval '%s" not in sql, f"P0-4 regression: reapareció el bug, got {sql!r}"


def test_get_query_placeholder_count_matches_params():
    """Sanity check: número de `%s` placeholders REALES = número de params.
    Esto es exactamente lo que rompía con el bug original (1 placeholder real
    vs 2 params)."""
    cache = PersistentLLMCache(ttl_seconds=300)
    captured = {}

    def fake_execute(query, params=None, fetch_one=False, **kwargs):
        captured["query"] = query
        captured["params"] = params
        return None

    with patch.object(graph_orchestrator, "execute_sql_query", side_effect=fake_execute), \
         patch.object(graph_orchestrator, "redis_client", None):
        cache.get("k_test")

    sql = captured["query"]
    placeholders = _count_placeholders_outside_string_literals(sql)
    n_params = len(captured["params"]) if captured["params"] else 0
    assert placeholders == n_params == 2, \
        f"placeholders={placeholders}, params={n_params}; esperado 2 == 2"


def test_get_passes_ttl_seconds_as_second_param():
    """El segundo parámetro debe ser el TTL en segundos (entero) — `make_interval`
    espera un número, no un string fraccional."""
    cache = PersistentLLMCache(ttl_seconds=420)
    captured = {}

    def fake_execute(query, params=None, fetch_one=False, **kwargs):
        captured["params"] = params
        return None

    with patch.object(graph_orchestrator, "execute_sql_query", side_effect=fake_execute), \
         patch.object(graph_orchestrator, "redis_client", None):
        cache.get("any_key")

    assert captured["params"][0] == "any_key"
    assert captured["params"][1] == 420
    assert isinstance(captured["params"][1], (int, float))


# ---------------------------------------------------------------------------
# 2. Async `aget` — paridad con sync.
# ---------------------------------------------------------------------------
def test_aget_query_uses_make_interval_not_quoted_interval():
    """`aget` debe emitir el mismo SQL fixed que `get`. Usamos `asyncio.run`
    en lugar de `pytest.mark.asyncio` para no exigir `pytest-asyncio`."""
    cache = PersistentLLMCache(ttl_seconds=300)
    captured = {}

    async def fake_aexecute(query, params=None, fetch_one=False, **kwargs):
        captured["query"] = query
        captured["params"] = params
        return None

    async def _run():
        with patch.object(graph_orchestrator, "aexecute_sql_query", side_effect=fake_aexecute), \
             patch.object(graph_orchestrator, "redis_async_client", None):
            await cache.aget("k_test_async")

    asyncio.run(_run())

    sql = captured["query"]
    assert "make_interval" in sql
    assert "interval '%s" not in sql


def test_aget_query_placeholder_count_matches_params():
    cache = PersistentLLMCache(ttl_seconds=300)
    captured = {}

    async def fake_aexecute(query, params=None, fetch_one=False, **kwargs):
        captured["query"] = query
        captured["params"] = params
        return None

    async def _run():
        with patch.object(graph_orchestrator, "aexecute_sql_query", side_effect=fake_aexecute), \
             patch.object(graph_orchestrator, "redis_async_client", None):
            await cache.aget("k_test_async")

    asyncio.run(_run())

    sql = captured["query"]
    placeholders = _count_placeholders_outside_string_literals(sql)
    n_params = len(captured["params"]) if captured["params"] else 0
    assert placeholders == n_params == 2


# ---------------------------------------------------------------------------
# 3. Defensa contra regresión textual en el código fuente.
# ---------------------------------------------------------------------------
def test_source_code_no_quoted_interval_with_placeholder():
    """Defensa secundaria: el patrón `interval '%s` no debe reaparecer en
    el archivo (excepto en comentarios documentando el bug)."""
    src = open(graph_orchestrator.__file__, encoding="utf-8").read()
    # Permitimos la frase EN comentarios `[P0-4]` que documentan el bug, pero
    # NO en SQL real. Filtramos las líneas que son solo comentario.
    offending = []
    for i, line in enumerate(src.splitlines(), start=1):
        if "interval '%s" not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comentario que documenta el bug
        offending.append((i, line))
    assert not offending, (
        f"P0-4 regression: `interval '%s` reapareció en código activo: {offending}"
    )


# ---------------------------------------------------------------------------
# 4. Comportamiento end-to-end (Redis caído → DB sirve hit).
# ---------------------------------------------------------------------------
def test_get_propagates_db_value_when_redis_unavailable():
    """Si Redis está caído (None) y la DB tiene una entrada válida, el cache
    debe propagar el valor — cosa que ANTES no podía hacer porque la query
    siempre lanzaba ProgrammingError."""
    cache = PersistentLLMCache(ttl_seconds=300)
    expected = {"foo": "bar"}

    def fake_execute(query, params=None, fetch_one=False, **kwargs):
        # Simulamos el row real que la DB devuelve (psycopg con dict_row).
        return {"value": expected}

    with patch.object(graph_orchestrator, "execute_sql_query", side_effect=fake_execute), \
         patch.object(graph_orchestrator, "redis_client", None):
        result = cache.get("k_hit")

    assert result == expected, f"esperado {expected}, got {result}"


def test_get_returns_default_on_db_miss():
    """DB miss (fetch_one returns None) → default propagado, sin excepción."""
    cache = PersistentLLMCache(ttl_seconds=300)

    def fake_execute(query, params=None, fetch_one=False, **kwargs):
        return None

    with patch.object(graph_orchestrator, "execute_sql_query", side_effect=fake_execute), \
         patch.object(graph_orchestrator, "redis_client", None):
        result = cache.get("k_miss", default="sentinel")

    assert result == "sentinel"
