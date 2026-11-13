# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-996 · 2026-11-13] Calentador de la caché semántica de ingredientes, fuera de la petición.

Una petición (sin plazo propio) nunca construye la caché desde cero: con 354 alimentos en lotes de 3 y 3 s entre
lotes la init necesita ~6 min, así que el plazo de 30 s la abortaba SIEMPRE y el usuario se comía esos 30 s (medido:
«me comí esto» en el coach, 55 s de los que 32 fueron esto). En su lugar se lanza UN hilo daemon que la construye con
el plazo del calentador y la persiste a Redis; la petición sigue por el Regex Fast-Path.

Knob `MEALFIT_EMBED_INIT_EN_PETICION` (False): True vuelve a la conducta previa.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from knobs import _env_bool

_vivo = False


def init_en_peticion() -> bool:
    return _env_bool("MEALFIT_EMBED_INIT_EN_PETICION", False)


def lanzar(construir: Callable[[], object]) -> bool:
    """Lanza `construir()` en un hilo daemon si no hay otro en marcha. Devuelve True si lo lanzó."""
    global _vivo
    if _vivo:
        return False
    _vivo = True

    def _correr():
        global _vivo
        try:
            time.sleep(0.2)   # deja que la petición que lo lanzó suelte el lock
            construir()
        except Exception:  # noqa: BLE001 — best-effort: el fast-path sigue sirviendo
            pass
        finally:
            _vivo = False

    try:
        threading.Thread(target=_correr, name="embed-cache-warmer-peticion", daemon=True).start()
        logging.info("🔥 [P1-PLAN-LOTE-996] Caché semántica sin construir: calentador lanzado en segundo plano.")
        return True
    except Exception:  # noqa: BLE001
        _vivo = False
        return False
