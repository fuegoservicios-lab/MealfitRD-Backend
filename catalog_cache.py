"""Cache the global food reference rows; user projections remain in the router."""
import asyncio
from copy import deepcopy
from time import monotonic


class CatalogCache:
    def __init__(self, ttl=300, max_stale=3600, clock=monotonic):
        self.ttl = ttl
        self.max_stale = max_stale
        self.clock = clock
        self.rows = None
        self.loaded_at = 0
        self.pending = None

    async def get(self, loader):
        age = self.clock() - self.loaded_at
        if self.rows and age < self.ttl:
            return deepcopy(self.rows)
        if self.pending is None:
            self.pending = asyncio.create_task(self._load(loader))
            # Background refresh errors keep the previous copy and are consumed.
            self.pending.add_done_callback(self._finished)
        if self.rows and age < self.max_stale:
            return deepcopy(self.rows)
        return deepcopy(await asyncio.shield(self.pending))

    async def _load(self, loader):
        rows = await loader()
        if rows:
            self.rows = deepcopy(rows)
            self.loaded_at = self.clock()
        return rows

    def _finished(self, task):
        if self.pending is task:
            self.pending = None
        if not task.cancelled():
            task.exception()


catalog_rows = CatalogCache()
