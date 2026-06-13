"""[P3-COMPRESSOR-CACHE · 2026-05-29] Cache content-addressed del nodo de
compresión de contexto (optimización de costo LLM).

Contexto:
  `context_compression_node` (graph_orchestrator.py) hace una llamada LLM
  (flash-lite, budget ~30s) en CADA pipeline cuando `history_context > 2000`
  chars — el caso común de usuarios recurrentes con mucho historial. El
  resultado es determinístico para un mismo input (temperature=0.0 + prompt
  "no inventes nada, solo resume"), así que cachearlo por hash SHA-256 del
  `history_context` es seguro: cualquier cambio del historial produce otra key
  → invalidación automática (cero staleness). Un cache hit ahorra una llamada
  LLM completa.

Cobertura:
  - Parser-based (siempre corre, sin importar deps pesadas): knobs presentes,
    helpers definidos, el nodo consulta el cache ANTES del LLM y lo puebla
    DESPUÉS de una compresión exitosa, marker anchor.
  - Conductual (skip si graph_orchestrator no es importable en este entorno):
    put→get, knob de kill-switch, miss por contenido distinto, expiración por
    TTL, y cota de tamaño (eviction).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_PATH = _BACKEND_ROOT / "graph_orchestrator.py"
_GO_SRC = _GO_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Parser-based (no requiere importar el módulo)
# --------------------------------------------------------------------------
class TestSourceContract:
    def test_marker_anchor_present(self):
        assert "P3-COMPRESSOR-CACHE" in _GO_SRC

    def test_three_knobs_declared(self):
        for knob in (
            "MEALFIT_COMPRESSOR_CACHE_ENABLED",
            "MEALFIT_COMPRESSOR_CACHE_TTL_S",
            "MEALFIT_COMPRESSOR_CACHE_MAX_ENTRIES",
        ):
            assert knob in _GO_SRC, f"Falta el knob {knob} en graph_orchestrator.py"

    def test_cache_helpers_defined(self):
        assert re.search(r"def\s+_compressor_cache_key\s*\(", _GO_SRC)
        assert re.search(r"def\s+_compressor_cache_get\s*\(", _GO_SRC)
        assert re.search(r"def\s+_compressor_cache_put\s*\(", _GO_SRC)

    def test_cache_is_content_addressed_sha256(self):
        # La key DEBE ser un hash del history_context (no user_id/time) para que
        # el cambio de historial invalide automáticamente.
        m = re.search(
            r"def\s+_compressor_cache_key\s*\([^)]*\)[^\n:]*:\s*\n\s*return\s+hashlib\.sha256",
            _GO_SRC,
        )
        assert m is not None, "_compressor_cache_key debe hashear con sha256 el history_context"

    def test_node_checks_cache_before_building_llm(self):
        # El cache hit debe ocurrir ANTES de instanciar ChatGoogleGenerativeAI
        # (si no, no se ahorra la llamada). Verificamos el orden en el source.
        get_idx = _GO_SRC.find("_compressor_cache_get(history_context)")
        llm_idx = _GO_SRC.find("compressor_llm = ChatDeepSeek")
        assert get_idx != -1, "El nodo no consulta _compressor_cache_get"
        assert llm_idx != -1, "No se encontró la construcción del compressor_llm"
        assert get_idx < llm_idx, (
            "El cache hit debe consultarse ANTES de construir el LLM "
            "(de lo contrario no se ahorra la llamada)."
        )

    def test_node_populates_cache_after_success(self):
        # El put debe ocurrir tras obtener compressed_text válido y antes del
        # return exitoso.
        assert "_compressor_cache_put(history_context, compressed_text)" in _GO_SRC

    # --- Persistencia (capa 2) + telemetría ---
    def test_persistent_helpers_defined(self):
        assert re.search(r"async\s+def\s+_compressor_cache_get_persistent\s*\(", _GO_SRC)
        assert re.search(r"async\s+def\s+_compressor_cache_put_persistent\s*\(", _GO_SRC)

    def test_persistent_layer_uses_app_kv_store(self):
        assert "app_kv_store" in _GO_SRC
        assert 'compressor_cache:' in _GO_SRC  # prefijo de la key persistente

    def test_node_consults_persistent_before_llm_and_persists_on_success(self):
        get_p = _GO_SRC.find("await _compressor_cache_get_persistent(history_context)")
        llm_idx = _GO_SRC.find("compressor_llm = ChatDeepSeek")
        put_p = _GO_SRC.find("await _compressor_cache_put_persistent(history_context, compressed_text)")
        assert get_p != -1 and get_p < llm_idx, "Debe consultar la capa persistente antes del LLM"
        assert put_p != -1 and put_p > llm_idx, "Debe persistir tras la compresión exitosa"

    def test_telemetry_helpers_defined(self):
        assert re.search(r"def\s+get_compressor_cache_stats\s*\(", _GO_SRC)
        assert "_compressor_cache_record_hit" in _GO_SRC
        assert "_compressor_cache_record_miss" in _GO_SRC

    def test_persist_kill_switch_knob_present(self):
        assert "MEALFIT_COMPRESSOR_CACHE_PERSIST" in _GO_SRC


class TestSweepCatalog:
    """El prefijo persistente DEBE estar en el catálogo del sweep KV
    (`_KV_SWEEP_PREFIXES` en cron_tasks.py) para que las filas stale se GC."""

    def test_compressor_cache_prefix_in_sweep_catalog(self):
        cron_src = (_BACKEND_ROOT / "cron_tasks.py").read_text(encoding="utf-8")
        assert "_KV_SWEEP_PREFIXES" in cron_src
        assert '"compressor_cache:"' in cron_src, (
            "El prefijo `compressor_cache:` debe estar en _KV_SWEEP_PREFIXES "
            "o las filas persistentes crecerían sin límite."
        )


# --------------------------------------------------------------------------
# Conductual (import-guarded: el módulo arrastra langchain/langgraph)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def go():
    try:
        import graph_orchestrator as _go  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - entorno sin deps del pipeline
        pytest.skip(f"graph_orchestrator no importable en este entorno: {exc}")
    return _go


@pytest.fixture(autouse=True)
def _clear_cache_and_env(go, monkeypatch):
    # Aislar cada test: cache vacío + knobs por default (enabled).
    if go is not None:
        with go._COMPRESSOR_CACHE_LOCK:
            go._COMPRESSOR_CACHE.clear()
    for k in (
        "MEALFIT_COMPRESSOR_CACHE_ENABLED",
        "MEALFIT_COMPRESSOR_CACHE_TTL_S",
        "MEALFIT_COMPRESSOR_CACHE_MAX_ENTRIES",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


class TestCacheBehavior:
    def test_put_then_get_returns_value(self, go):
        ctx = "h" * 3000
        go._compressor_cache_put(ctx, "RESUMEN-A")
        assert go._compressor_cache_get(ctx) == "RESUMEN-A"

    def test_different_content_misses(self, go):
        go._compressor_cache_put("a" * 3000, "RESUMEN-A")
        assert go._compressor_cache_get("b" * 3000) is None

    def test_kill_switch_disables_cache(self, go, monkeypatch):
        ctx = "h" * 3000
        go._compressor_cache_put(ctx, "RESUMEN-A")
        # Con el knob en false, get devuelve None aunque haya entrada.
        monkeypatch.setenv("MEALFIT_COMPRESSOR_CACHE_ENABLED", "false")
        assert go._compressor_cache_get(ctx) is None

    def test_disabled_put_is_noop(self, go, monkeypatch):
        ctx = "h" * 3000
        monkeypatch.setenv("MEALFIT_COMPRESSOR_CACHE_ENABLED", "false")
        go._compressor_cache_put(ctx, "RESUMEN-A")
        # Re-habilitar para leer: no debe haberse guardado nada.
        monkeypatch.delenv("MEALFIT_COMPRESSOR_CACHE_ENABLED", raising=False)
        assert go._compressor_cache_get(ctx) is None

    def test_expired_entry_is_evicted_on_get(self, go):
        import time as _t
        ctx = "h" * 3000
        key = go._compressor_cache_key(ctx)
        # Insertar manualmente una entrada ya expirada (expiry en el pasado).
        with go._COMPRESSOR_CACHE_LOCK:
            go._COMPRESSOR_CACHE[key] = ("VIEJO", _t.time() - 10)
        assert go._compressor_cache_get(ctx) is None
        # Debe haberse purgado del dict.
        with go._COMPRESSOR_CACHE_LOCK:
            assert key not in go._COMPRESSOR_CACHE

    def test_max_entries_bounds_size(self, go, monkeypatch):
        monkeypatch.setenv("MEALFIT_COMPRESSOR_CACHE_MAX_ENTRIES", "2")
        for i in range(5):
            go._compressor_cache_put(f"{'x' * 3000}-{i}", f"R{i}")
        with go._COMPRESSOR_CACHE_LOCK:
            assert len(go._COMPRESSOR_CACHE) <= 2


class TestTelemetry:
    def test_stats_snapshot_shape(self, go):
        with go._COMPRESSOR_CACHE_LOCK:
            for k in ("hits_memory", "hits_kv", "misses"):
                go._COMPRESSOR_CACHE_STATS[k] = 0
        go._compressor_cache_record_hit("memory")
        go._compressor_cache_record_hit("kv")
        go._compressor_cache_record_miss()
        stats = go.get_compressor_cache_stats()
        assert stats["hits_memory"] == 1
        assert stats["hits_kv"] == 1
        assert stats["misses"] == 1
        assert stats["total_lookups"] == 3
        # hit_rate = 2/3 ≈ 0.6667
        assert 0.66 <= stats["hit_rate"] <= 0.67
        assert "in_process_entries" in stats

    def test_hit_rate_zero_when_no_lookups(self, go):
        with go._COMPRESSOR_CACHE_LOCK:
            for k in ("hits_memory", "hits_kv", "misses"):
                go._COMPRESSOR_CACHE_STATS[k] = 0
        stats = go.get_compressor_cache_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["total_lookups"] == 0


class TestPersistenceKillSwitch:
    def test_get_persistent_returns_none_when_persist_disabled(self, go, monkeypatch):
        # Con persistencia OFF, no toca la DB y devuelve None inmediatamente.
        import asyncio
        monkeypatch.setenv("MEALFIT_COMPRESSOR_CACHE_PERSIST", "false")
        result = asyncio.run(go._compressor_cache_get_persistent("h" * 3000))
        assert result is None

    def test_kv_key_is_prefixed_hash(self, go):
        key = go._compressor_cache_kv_key("h" * 3000)
        assert key.startswith("compressor_cache:")
        # El sufijo es el mismo hash que la key in-process.
        assert key.endswith(go._compressor_cache_key("h" * 3000))
