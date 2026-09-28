# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-760 · 2026-09-28] El coach termina su respuesta aunque el usuario salga de la app.

El dueño (28-sep, 12:25 UTC): mandó «Cómo estás?», salió de la app y al volver leyó «No llegó la respuesta del coach».
En el journal: `SSE abortado por cliente kind=GeneratorExit chunk_observed=False` a los 24 s del turno. La generación
vivía DENTRO del stream HTTP: cuando el teléfono cortó la conexión, Starlette cerró el generador, `chat_with_agent_stream`
recibió el GeneratorExit y la respuesta murió a medias, sin guardarse — ni la push del 692 podía salir, porque sale en
el `done`. El dueño: «quiero que no sea obligatorio estar dentro de la app para que las respuestas carguen».

Ahora el turno corre en SU hilo (`desacoplar`) y el stream solo lee de una cola:

  · si quien lee se va (cerró la app, cambió de pestaña, se cortó la red), el hilo sigue hasta el `done`: la respuesta
    se guarda, se cobra y sale «Bioboros te respondió». Al volver, el rescate del cliente (lote 156) ve `turn_active`
    mientras el hilo trabaja y adopta la respuesta cuando aparece en el historial;
  · «Detener» deja de ser «cortar la conexión» (que ahora no para nada): el cliente llama `POST /api/chat/stop` y
    `detener(session_id)` marca el turno; el hilo corta en el siguiente evento y cierra el generador — el mismo camino
    (GeneratorExit + cobro en el `finally`) que antes seguía el corte de la conexión.

Knob `MEALFIT_CHAT_TURN_DETACHED` (true). Con false, el stream vuelve a ser el generador tal cual (conducta anterior).
tooltip-anchor: P1-PLAN-LOTE-760-TURNO-DESACOPLADO
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

_FIN = object()
_lock = threading.Lock()
_TURNOS: dict[str, threading.Event] = {}


def activo() -> bool:
    from knobs import _env_bool
    return _env_bool("MEALFIT_CHAT_TURN_DETACHED", True)


def detener(session_id) -> bool:
    """Marca el turno en curso de ese chat para que se detenga. True si había uno."""
    if not session_id:
        return False
    with _lock:
        ev = _TURNOS.get(str(session_id))
    if ev is None:
        return False
    ev.set()
    return True


def en_curso(session_id) -> bool:
    with _lock:
        return str(session_id) in _TURNOS


def desacoplar(generador: Iterable[str], session_id) -> Iterator[str]:
    """Corre `generador` en un hilo propio HASTA EL FINAL y devuelve un iterador con sus trozos. Si el iterador se
    abandona (el cliente se fue), el hilo sigue; `detener(session_id)` lo corta en el siguiente trozo."""
    clave = str(session_id)
    cola: queue.Queue = queue.Queue()
    parar = threading.Event()
    with _lock:
        _TURNOS[clave] = parar
    gen = iter(generador)

    def _hilo():
        try:
            for trozo in gen:
                cola.put(trozo)
                if parar.is_set():
                    logger.info(f"[P1-PLAN-LOTE-760] turno de {clave[:8]} detenido por el usuario")
                    break
        except Exception as e:  # noqa: BLE001 — el generador ya emite su propio evento de error
            logger.warning(f"[P1-PLAN-LOTE-760] el turno de {clave[:8]} terminó con error: {e!r}")
        finally:
            try:
                close = getattr(gen, "close", None)
                if close:
                    close()   # con «Detener»: GeneratorExit dentro del turno (cobro en su finally, marca liberada)
            except Exception:  # noqa: BLE001
                pass
            with _lock:
                if _TURNOS.get(clave) is parar:
                    _TURNOS.pop(clave, None)
            cola.put(_FIN)

    threading.Thread(target=_hilo, name=f"chat-turno-{clave[:8]}", daemon=True).start()

    def _leer():
        while True:
            trozo = cola.get()
            if trozo is _FIN:
                return
            yield trozo

    return _leer()
