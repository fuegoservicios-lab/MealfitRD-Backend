"""[P1-PLAN-LOTE-1026 · A-13] El grafo del chat se compila UNA vez; cada turno usa una copia con su propio saver.

Medido: `chat_builder.compile(...)` cuesta ~60 ms y se hacía en cada turno. `compiled.copy(update={"checkpointer": s})`
cuesta ~0,04 ms. El saver NO se comparte: `PostgresSaver` serializa todas sus operaciones con un `threading.Lock`
propio, así que un saver común pondría en cola los checkpoints de todos los turnos simultáneos. Cada turno sigue
creando el suyo (sobre el pool), como antes; solo deja de recompilar el grafo.

Fail-open: si el compilado no tiene `copy` (los tests sustituyen `chat_builder` por uno falso) o algo falla, se
compila como siempre. Knob `MEALFIT_CHAT_GRAFO_COMPILADO_UNA_VEZ=0` para volver a compilar en cada turno.
"""
from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_cache: dict = {"builder": None, "base": None}


def activo() -> bool:
    return (os.getenv("MEALFIT_CHAT_GRAFO_COMPILADO_UNA_VEZ", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def para_turno(builder, checkpointer):
    """El grafo compilado de este turno, con `checkpointer` como saver."""
    if not activo():
        return builder.compile(checkpointer=checkpointer)
    try:
        base = _cache["base"] if _cache["builder"] is builder else None
        if base is None:
            with _lock:
                if _cache["builder"] is not builder or _cache["base"] is None:
                    compilado = builder.compile(checkpointer=checkpointer)
                    if not callable(getattr(compilado, "copy", None)):
                        return compilado                     # un builder falso de los tests: sin caché
                    _cache["builder"], _cache["base"] = builder, compilado
                base = _cache["base"]
        return base.copy(update={"checkpointer": checkpointer})
    except Exception:
        return builder.compile(checkpointer=checkpointer)
