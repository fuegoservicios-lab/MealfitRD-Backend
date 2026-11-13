# -*- coding: utf-8 -*-
"""Caché thread-safe acotada a un turno HTTP para evitar lecturas duplicadas pre-TTFT.

Fuera de `turn_cache_scope()`, todas las operaciones son no-op transparentes: las
funciones de negocio y las pruebas unitarias que las llaman de forma aislada
conservan exactamente su comportamiento previo.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import threading
import time
from typing import Any, Callable, Optional, Tuple

_TURN_CACHE_VAR: contextvars.ContextVar[Optional["_TurnCache"]] = contextvars.ContextVar(
    "mealfit_turn_cache", default=None
)

_HANDOFF_LOCK = threading.Lock()
_HANDOFF_STORE: dict[Tuple[str, str], Tuple[float, dict]] = {}
_HANDOFF_TTL_S = 5.0
_HANDOFF_MAX = 64


class _TurnCache:
    """Almacén thread-safe por turno con singleflight por clave."""

    def __init__(self, initial: Optional[dict] = None) -> None:
        self._lock = threading.Lock()
        self._data: dict[Tuple[str, str], Any] = dict(initial) if initial else {}
        self._inflight: dict[Tuple[str, str], threading.Event] = {}

    def get(self, ns: str, key: str) -> Tuple[bool, Any]:
        k = (ns, str(key))
        with self._lock:
            if k in self._data:
                return True, self._data[k]
        return False, None

    def set(self, ns: str, key: str, value: Any) -> None:
        k = (ns, str(key))
        with self._lock:
            self._data[k] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def get_or_load(self, ns: str, key: str, loader: Callable[[], Any]) -> Any:
        k = (ns, str(key))
        while True:
            with self._lock:
                if k in self._data:
                    return self._data[k]
                ev = self._inflight.get(k)
                if ev is None:
                    ev = threading.Event()
                    self._inflight[k] = ev
                    leader = True
                else:
                    leader = False
            if not leader:
                ev.wait(timeout=15.0)
                with self._lock:
                    if k in self._data:
                        return self._data[k]
                return loader()
            try:
                val = loader()
                with self._lock:
                    self._data[k] = val
                return val
            finally:
                with self._lock:
                    self._inflight.pop(k, None)
                ev.set()


def current_turn_cache() -> Optional[_TurnCache]:
    return _TURN_CACHE_VAR.get()


@contextmanager
def turn_cache_scope(initial: Optional[dict] = None):
    """Abre un alcance de caché para el turno actual y lo limpia al salir."""
    parent = _TURN_CACHE_VAR.get()
    if parent is not None:
        yield parent
        return
    tc = _TurnCache(initial=initial)
    tok = _TURN_CACHE_VAR.set(tc)
    try:
        yield tc
    finally:
        tc.clear()
        _TURN_CACHE_VAR.reset(tok)


def stash_handoff(user_id: Optional[str], session_id: Optional[str]) -> None:
    """Guarda por 5s lo precargado en `/api/chat*` antes de abrir el generador SSE."""
    tc = current_turn_cache()
    if tc is None or not user_id or not session_id:
        return
    snap = tc.snapshot()
    if not snap:
        return
    now = time.monotonic()
    hkey = (str(user_id), str(session_id))
    with _HANDOFF_LOCK:
        expired = [k for k, (ts, _) in _HANDOFF_STORE.items() if now - ts > _HANDOFF_TTL_S]
        for k in expired:
            _HANDOFF_STORE.pop(k, None)
        while len(_HANDOFF_STORE) >= _HANDOFF_MAX:
            _HANDOFF_STORE.pop(next(iter(_HANDOFF_STORE)))
        _HANDOFF_STORE[hkey] = (now, snap)


def consume_handoff(user_id: Optional[str], session_id: Optional[str]) -> Optional[dict]:
    """Recupera y borra (one-shot) los datos precargados por el router para este turno."""
    if not user_id or not session_id:
        return None
    hkey = (str(user_id), str(session_id))
    now = time.monotonic()
    with _HANDOFF_LOCK:
        item = _HANDOFF_STORE.pop(hkey, None)
    if not item:
        return None
    ts, snap = item
    if now - ts > _HANDOFF_TTL_S:
        return None
    return snap


def get_cached(ns: str, key: str) -> Tuple[bool, Any]:
    tc = current_turn_cache()
    if tc is None:
        return False, None
    return tc.get(ns, key)


def set_cached(ns: str, key: str, value: Any) -> None:
    tc = current_turn_cache()
    if tc is not None:
        tc.set(ns, key, value)


def get_or_load(ns: str, key: str, loader: Callable[[], Any]) -> Any:
    tc = current_turn_cache()
    if tc is None:
        return loader()
    return tc.get_or_load(ns, key, loader)


def clear_reads() -> None:
    """Descarta lecturas previas al LLM antes de ejecutar tools que mutan estado."""
    tc = current_turn_cache()
    if tc is not None:
        tc.clear()


def submit_with_context(executor, fn, *args, **kwargs):
    """Propaga el ContextVar del hilo actual al worker de ThreadPoolExecutor."""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)
