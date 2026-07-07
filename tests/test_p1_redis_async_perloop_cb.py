"""[P1-REDIS-ASYNC-PERLOOP-CB · 2026-07-07] Completa la migración per-loop del
cliente Redis async al circuit breaker y al cache de generación.

Logs en vivo (~25 errores / 30 min):
  asyncio: Exception in callback _SelectorSocketTransport._read_ready()
  TypeError: 'NoneType' object is not callable
  graph_orchestrator: Redis async CB reset error: ... got Future ... attached to a different loop
  graph_orchestrator: Redis async cache read error: ... attached to a different loop

Causa: la migración P2-REDIS-ASYNC-PERLOOP (2026-06-17) movió los SEMÁFOROS al factory
per-loop `get_redis_async()`, pero `LLMCircuitBreaker` (arecord_failure/arecord_success/
acan_proceed) y `PersistentLLMCache` (aget/aset) seguían usando el `redis_async_client`
module-global, atado al loop de import. La generación de planes corre en un loop FRESCO
(`asyncio.run` en hilo aislado) → await al cliente global lanza "Future attached to a
different loop". Fix: los 5 sitios usan `get_redis_async()` (cliente cacheado en el loop
corriente). Fail-soft intacto (None → path DB).
tooltip-anchor: P1-REDIS-ASYNC-PERLOOP-CB
"""
from __future__ import annotations

import re
from pathlib import Path

import graph_orchestrator as g

_SRC = (Path(g.__file__).resolve().parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_marker_present():
    assert "P1-REDIS-ASYNC-PERLOOP-CB" in _SRC


def test_get_redis_async_imported():
    assert "get_redis_async" in _SRC
    assert re.search(r"from cache_manager import[^\n]*get_redis_async", _SRC), (
        "get_redis_async debe importarse de cache_manager"
    )


def test_no_global_client_awaited():
    """NINGÚN `await redis_async_client.<x>` debe quedar — todos migrados a `_rc`
    (get_redis_async per-loop). El global solo puede aparecer en el import y comentarios."""
    # await directo al global (el patrón roto)
    assert not re.search(r"await\s+redis_async_client\.", _SRC), (
        "queda un `await redis_async_client.<metodo>` — sitio no migrado al cliente per-loop"
    )
    # acceso a atributo del global fuera de comentarios/import
    code_lines = [ln for ln in _SRC.splitlines()
                  if not ln.lstrip().startswith("#") and "import" not in ln]
    for ln in code_lines:
        assert "redis_async_client." not in ln, (
            f"uso del global redis_async_client fuera de get_redis_async: {ln.strip()[:80]}"
        )


def test_cb_and_cache_methods_use_perloop():
    """Los 5 métodos async (CB + cache) deben contener `_rc = get_redis_async()`."""
    for meth in ("async def arecord_failure", "async def arecord_success",
                 "async def acan_proceed", "async def aget", "async def aset"):
        idx = _SRC.find(meth)
        assert idx > 0, f"método {meth} no encontrado"
        # cuerpo hasta el siguiente `async def`/`def ` top-level dentro de ~1800 chars
        body = _SRC[idx: idx + 1800]
        assert "_rc = get_redis_async()" in body, (
            f"{meth} no usa el cliente per-loop `get_redis_async()`"
        )


def test_module_imports_clean():
    """El módulo importa sin error (cambié un import core)."""
    assert hasattr(g, "get_redis_async")
