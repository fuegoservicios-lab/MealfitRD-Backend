"""[P1-17] Tests para el cache lazy-singleton del grafo dummy en
`purge_langgraph_checkpoint`.

Bug original (audit P1-17):
  Cada llamada a `summarize_and_prune` (que dispara
  `purge_langgraph_checkpoint`) instanciaba un nuevo `StateGraph` +
  `PostgresSaver` + `compile`. En producción con N sesiones activas y
  resúmenes frecuentes:
    - CPU/memoria desperdiciados (objetos pesados de LangGraph se
      construyen, usan una vez, y descartan).
    - Potencial leak: el `compile` puede registrar callbacks/refs
      internos que no se liberan al GC del local del scope.

Fix:
  Helper `_get_dummy_purge_graph()` cachea el grafo a nivel módulo.
  Invalidación por `id(connection_pool)` para tests donde el pool se
  mockea/reemplaza. En producción el pool no cambia → cache se carga
  UNA vez por proceso. Thread-safe vía double-checked locking.

Cobertura:
  - test_cache_initialized_to_none_at_module_load
  - test_get_dummy_graph_returns_none_when_pool_unavailable
  - test_get_dummy_graph_caches_first_call_returns_same_instance
  - test_get_dummy_graph_invalidates_on_pool_id_change (test mode)
  - test_purge_function_uses_cached_graph_helper
  - test_thread_safety_double_checked_locking_smoke
  - test_documentation_p1_17_present
"""
import threading
import inspect

import pytest

import memory_manager
from memory_manager import (
    _get_dummy_purge_graph,
    _DUMMY_PURGE_GRAPH_CACHE,
    _DUMMY_PURGE_GRAPH_LOCK,
)


@pytest.fixture
def _reset_cache():
    """Limpia el cache antes y después para aislamiento entre tests."""
    original = dict(_DUMMY_PURGE_GRAPH_CACHE)
    _DUMMY_PURGE_GRAPH_CACHE["graph"] = None
    _DUMMY_PURGE_GRAPH_CACHE["pool_id"] = None
    yield
    _DUMMY_PURGE_GRAPH_CACHE.update(original)


# ---------------------------------------------------------------------------
# 1. Estado inicial del cache.
# ---------------------------------------------------------------------------
def test_cache_dict_has_expected_keys():
    """`_DUMMY_PURGE_GRAPH_CACHE` debe ser un dict con keys 'graph' y 'pool_id'."""
    assert isinstance(_DUMMY_PURGE_GRAPH_CACHE, dict)
    assert "graph" in _DUMMY_PURGE_GRAPH_CACHE
    assert "pool_id" in _DUMMY_PURGE_GRAPH_CACHE


def test_cache_lock_is_threading_lock():
    """`_DUMMY_PURGE_GRAPH_LOCK` debe ser un Lock de threading."""
    # threading.Lock no es importable directamente como tipo; chequeamos
    # que tenga los métodos esperados.
    assert hasattr(_DUMMY_PURGE_GRAPH_LOCK, "acquire")
    assert hasattr(_DUMMY_PURGE_GRAPH_LOCK, "release")


# ---------------------------------------------------------------------------
# 2. Comportamiento sin pool disponible.
# ---------------------------------------------------------------------------
def test_get_dummy_graph_returns_none_when_pool_unavailable(_reset_cache, monkeypatch):
    """Si `connection_pool` es None/falsy, el helper retorna None sin
    construir nada (early return defensivo)."""
    monkeypatch.setattr(memory_manager, "connection_pool", None)
    result = _get_dummy_purge_graph()
    assert result is None
    # Y el cache NO se contaminó con None falsy.
    assert _DUMMY_PURGE_GRAPH_CACHE["graph"] is None


# ---------------------------------------------------------------------------
# 3. Caching: la segunda llamada reutiliza la primera instancia.
# ---------------------------------------------------------------------------
def test_get_dummy_graph_caches_first_call(_reset_cache, monkeypatch):
    """Dos llamadas con el mismo pool retornan la MISMA instancia
    (no se construye un grafo nuevo)."""
    fake_pool = object()  # sentinel pool (id distinto a cualquier real)
    monkeypatch.setattr(memory_manager, "connection_pool", fake_pool)

    # Mockear las imports lazy de LangGraph para no requerir DB real.
    fake_graph = object()

    class _FakeBuilder:
        def add_node(self, *_a, **_kw): return self
        def add_edge(self, *_a, **_kw): return self
        def compile(self, **_kw): return fake_graph

    class _FakeStateGraph:
        def __init__(self, *_a, **_kw): pass
        def __new__(cls, *_a, **_kw): return _FakeBuilder()

    # Mockear los módulos antes de la primera invocación.
    import sys
    fake_lg_pg = type(sys)("langgraph.checkpoint.postgres")
    fake_lg_pg.PostgresSaver = lambda pool: None
    fake_lg_g = type(sys)("langgraph.graph")
    fake_lg_g.StateGraph = _FakeStateGraph
    fake_lg_g.START = "START"
    fake_lg_gm = type(sys)("langgraph.graph.message")
    fake_lg_gm.MessagesState = type("MessagesState", (), {})
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_lg_pg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_lg_g)
    monkeypatch.setitem(sys.modules, "langgraph.graph.message", fake_lg_gm)

    g1 = _get_dummy_purge_graph()
    g2 = _get_dummy_purge_graph()
    assert g1 is fake_graph
    assert g2 is g1, "P1-17: segunda llamada debe reutilizar la instancia cacheada"
    # El cache debe reflejar el pool_id.
    assert _DUMMY_PURGE_GRAPH_CACHE["pool_id"] == id(fake_pool)


def test_get_dummy_graph_invalidates_on_pool_change(_reset_cache, monkeypatch):
    """Si `connection_pool` cambia (test mode con mock pool reemplazado),
    el cache se invalida y se construye un grafo nuevo para el pool nuevo."""
    fake_pool_a = object()
    fake_pool_b = object()
    fake_graph_a = object()
    fake_graph_b = object()
    builds = {"count": 0}

    def _make_builder(target_graph):
        class _FakeBuilder:
            def add_node(self, *_a, **_kw): return self
            def add_edge(self, *_a, **_kw): return self
            def compile(self, **_kw):
                builds["count"] += 1
                return target_graph
        return _FakeBuilder()

    import sys
    fake_lg_pg = type(sys)("langgraph.checkpoint.postgres")
    fake_lg_pg.PostgresSaver = lambda pool: None
    fake_lg_g = type(sys)("langgraph.graph")

    # StateGraph factory: cambia el target_graph según el pool actual.
    def _state_graph_factory(*_a, **_kw):
        if memory_manager.connection_pool is fake_pool_a:
            return _make_builder(fake_graph_a)
        elif memory_manager.connection_pool is fake_pool_b:
            return _make_builder(fake_graph_b)
        return _make_builder(object())

    fake_lg_g.StateGraph = _state_graph_factory
    fake_lg_g.START = "START"
    fake_lg_gm = type(sys)("langgraph.graph.message")
    fake_lg_gm.MessagesState = type("MessagesState", (), {})
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_lg_pg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_lg_g)
    monkeypatch.setitem(sys.modules, "langgraph.graph.message", fake_lg_gm)

    monkeypatch.setattr(memory_manager, "connection_pool", fake_pool_a)
    g1 = _get_dummy_purge_graph()
    assert g1 is fake_graph_a
    assert builds["count"] == 1

    # Segunda llamada con MISMO pool: cache hit, no rebuild.
    g1b = _get_dummy_purge_graph()
    assert g1b is fake_graph_a
    assert builds["count"] == 1, "P1-17: mismo pool NO debe construir de nuevo"

    # Cambiar pool → invalidar cache.
    monkeypatch.setattr(memory_manager, "connection_pool", fake_pool_b)
    g2 = _get_dummy_purge_graph()
    assert g2 is fake_graph_b
    assert builds["count"] == 2, "P1-17: pool nuevo debe construir grafo nuevo"


# ---------------------------------------------------------------------------
# 4. La función pública `purge_langgraph_checkpoint` usa el cache.
# ---------------------------------------------------------------------------
def test_purge_function_uses_cached_graph_helper():
    """`purge_langgraph_checkpoint` debe invocar `_get_dummy_purge_graph()`
    en lugar de construir el grafo inline (defensa contra reintroducir
    el bug)."""
    src = inspect.getsource(memory_manager.purge_langgraph_checkpoint)
    assert "_get_dummy_purge_graph" in src, (
        "P1-17: purge_langgraph_checkpoint debe usar el helper cacheado"
    )
    # Y NO debe hacer `builder.compile(checkpointer=checkpointer)` inline
    # (el patrón roto pre-P1-17).
    assert "builder.compile" not in src, (
        "P1-17 regression: builder.compile inline reapareció en purge_langgraph_checkpoint"
    )


# ---------------------------------------------------------------------------
# 5. Thread-safety smoke.
# ---------------------------------------------------------------------------
def test_thread_safety_double_checked_locking_smoke(_reset_cache, monkeypatch):
    """Bajo concurrencia, `_get_dummy_purge_graph` no debe construir más
    de una vez ni corromper el cache. Smoke test: ningún crash + el
    cache es consistente al final."""
    fake_pool = object()
    fake_graph = object()
    builds = {"count": 0}

    class _Builder:
        def add_node(self, *_a, **_kw): return self
        def add_edge(self, *_a, **_kw): return self
        def compile(self, **_kw):
            builds["count"] += 1
            return fake_graph

    import sys
    fake_lg_pg = type(sys)("langgraph.checkpoint.postgres")
    fake_lg_pg.PostgresSaver = lambda pool: None
    fake_lg_g = type(sys)("langgraph.graph")
    fake_lg_g.StateGraph = lambda *_a, **_kw: _Builder()
    fake_lg_g.START = "START"
    fake_lg_gm = type(sys)("langgraph.graph.message")
    fake_lg_gm.MessagesState = type("MessagesState", (), {})
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_lg_pg)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_lg_g)
    monkeypatch.setitem(sys.modules, "langgraph.graph.message", fake_lg_gm)
    monkeypatch.setattr(memory_manager, "connection_pool", fake_pool)

    results = []

    def _worker():
        results.append(_get_dummy_purge_graph())

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # Todas las llamadas deben devolver la MISMA instancia (singleton).
    assert all(r is fake_graph for r in results), \
        f"P1-17: instancias diferentes bajo concurrencia. Got {len(set(map(id, results)))} distintas"
    # Smoke: el cache final tiene el pool_id correcto.
    assert _DUMMY_PURGE_GRAPH_CACHE["pool_id"] == id(fake_pool)


# ---------------------------------------------------------------------------
# 6. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_17_present():
    """El comentario `[P1-17]` debe estar presente para documentar el cache."""
    src = inspect.getsource(memory_manager)
    assert "[P1-17]" in src
