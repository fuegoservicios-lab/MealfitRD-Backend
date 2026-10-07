import asyncio
import ast
import logging
from pathlib import Path
import sys
import types
from typing import Optional
from catalog_cache import CatalogCache


def test_concurrent_searches_share_load_and_copies_do_not_mutate_cache():
    async def scenario():
        cache = CatalogCache()
        calls = 0
        async def load():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return [{"name": "Arroz", "names": {"en-US": "Rice"}}]
        results = await asyncio.gather(*(cache.get(load) for _ in range(8)))
        assert calls == 1
        results[0][0]["names"]["en-US"] = "Changed"
        assert (await cache.get(load))[0]["names"]["en-US"] == "Rice"
    asyncio.run(scenario())


def test_stale_copy_is_instant_while_refresh_runs():
    async def scenario():
        now = [0]
        cache = CatalogCache(ttl=5, max_stale=60, clock=lambda: now[0])
        async def initial(): return [{"name": "Arroz"}]
        await cache.get(initial)
        now[0] = 6
        ready = asyncio.Event()
        async def refresh():
            await ready.wait()
            return [{"name": "Arroz nuevo"}]
        assert await cache.get(refresh) == [{"name": "Arroz"}]
        pending = cache.pending
        ready.set()
        await pending
        assert await cache.get(refresh) == [{"name": "Arroz nuevo"}]
    asyncio.run(scenario())


def test_empty_and_failed_loads_can_retry():
    async def scenario():
        cache = CatalogCache()
        async def empty(): return []
        assert await cache.get(empty) == []
        async def failure(): raise RuntimeError("offline")
        try: await cache.get(failure)
        except RuntimeError: pass
        else: raise AssertionError("failure was hidden")
        async def success(): return [{"name": "Arroz"}]
        assert await cache.get(success) == [{"name": "Arroz"}]
    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_shared_load():
    async def scenario():
        cache = CatalogCache()
        ready = asyncio.Event()
        async def load():
            await ready.wait()
            return [{"name": "Arroz"}]
        first = asyncio.create_task(cache.get(load))
        await asyncio.sleep(0)
        second = asyncio.create_task(cache.get(load))
        first.cancel()
        try: await first
        except asyncio.CancelledError: pass
        ready.set()
        assert await second == [{"name": "Arroz"}]
    asyncio.run(scenario())


def test_router_cached_rows_keep_guest_and_account_projections_separate(monkeypatch):
    """Execute the real catalog handler without importing unrelated AI providers."""
    tree = ast.parse((Path(__file__).parents[1] / 'routers/user_data.py').read_text(encoding='utf-8'))
    handler = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'api_get_catalog')
    handler.decorator_list = []
    fields = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == '_CATALOG_CAMPOS_INVITADO' for t in n.targets))
    calls = []
    source_row = {'id': 'arroz', 'name': 'Arroz', 'name_en': 'Rice', 'price_per_lb': 25,
                  'market_container': 'libra', 'density_g_per_cup': 150}
    modules = {
        'db': {'execute_sql_query': lambda *a, **kw: calls.append(1) or [dict(source_row)]},
        'food_identity': {'anotar_catalogo': lambda rows: rows},
        'graph_orchestrator': {'_protein_gate_labels_in_text': lambda name: set()},
        'food_names_i18n': {'nombres': lambda: {}},
    }
    for name, attrs in modules.items():
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    namespace = {'asyncio': asyncio, 'Optional': Optional, 'Depends': lambda fn: None,
                 'get_verified_user_id': None, '_CATALOG_LIMITER': None,
                 'logger': logging.getLogger(__name__), 'catalog_rows': CatalogCache()}
    exec(compile(ast.Module(body=[fields, handler], type_ignores=[]), '<catalog-handler>', 'exec'), namespace)
    async def scenario():
        route = namespace['api_get_catalog']
        member = await route(verified_user_id='member')
        member['items'][0]['price_per_lb'] = 999
        guest = await route(verified_user_id=None)
        assert 'price_per_lb' not in guest['items'][0]
        assert 'market_container' not in guest['items'][0]
        assert guest['items'][0]['name'] == 'Arroz'
        member_again = await route(verified_user_id='other-member')
        assert member_again['items'][0]['price_per_lb'] == 25
        assert member_again['items'][0]['portions'][1]['grams_per_qty'] == 150
        assert len(calls) == 1
    asyncio.run(scenario())
