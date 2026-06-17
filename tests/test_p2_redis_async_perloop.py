"""[P2-REDIS-ASYNC-PERLOOP · 2026-06-17] Tests del cliente redis.asyncio per-loop.

Cierra el warning "PerUserSem Redis async error: Event loop is closed" — el cliente
module-global se ataba al loop de import y rompía en el loop de generación. El getter
devuelve uno ligado al loop ACTUAL (cacheado en el propio loop).
"""
import asyncio

import cache_manager


def test_get_redis_async_none_without_url(monkeypatch):
    """Sin REDIS_URL → None (caller usa fallback local)."""
    monkeypatch.setattr(cache_manager, "REDIS_URL", None, raising=False)

    async def _f():
        return cache_manager.get_redis_async()

    assert asyncio.run(_f()) is None


def test_get_redis_async_none_without_running_loop(monkeypatch):
    """En contexto SYNC (sin loop activo) → None, nunca lanza."""
    monkeypatch.setattr(cache_manager, "REDIS_URL", "redis://x", raising=False)
    monkeypatch.setattr(cache_manager, "redis_async", object(), raising=False)
    assert cache_manager.get_redis_async() is None


def test_get_redis_async_caches_per_loop(monkeypatch):
    """Mismo loop → mismo cliente (cacheado en el loop). Loop nuevo → cliente nuevo."""
    created = []

    class _FakeRedisAsync:
        def from_url(self, url, decode_responses=True):
            c = object()
            created.append(c)
            return c

    monkeypatch.setattr(cache_manager, "REDIS_URL", "redis://x", raising=False)
    monkeypatch.setattr(cache_manager, "redis_async", _FakeRedisAsync(), raising=False)

    async def _twice():
        return cache_manager.get_redis_async(), cache_manager.get_redis_async()

    a, b = asyncio.run(_twice())
    assert a is b                 # mismo loop → cacheado
    assert len(created) == 1

    c, _d = asyncio.run(_twice())  # loop nuevo
    assert c is not a             # cliente distinto por loop
    assert len(created) == 2
