"""[P1-PLAN-LOTE-1085 · 2026-11-18] Los micros de una comida, calculados UNA vez.

El dueño: «hay un poco de delay al cambiar los días y el compartir tarda en salir». Medido en el VPS:
`GET /api/diary/consumed/{uid}?date=` tardaba 0,4-0,9 s por día con el proceso caliente y 3,4 s en frío, y el 98 %
era `micros_de_ingredientes` (leer las comidas: 6 ms). Cada renglón resolvía su nombre contra el catálogo y, si no
coincidía, `normalize_name` llamaba a la API de embeddings (Cohere, ~160 ms y de pago) y comparaba el vector con TODO
el catálogo en Python puro (~2,5 s en 58 renglones). Y se repetía en cada visita al día.

Los micros de una comida dependen solo de sus cadenas de ingredientes (y del catálogo): se guardan por la huella de
esas cadenas, en memoria (LRU) y en Redis (`TTL_S`, para que un arreglo del catálogo llegue en días), y se calientan
al registrar la comida (`calentar`, en segundo plano). Todo fail-open: sin Redis, la memoria; sin nada, se calcula
como siempre. Nunca se escribe un dato de salud en el log.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

VERSION = "v1"                 # súbela si cambia la forma de lo que devuelve `micros_de_ingredientes`
TTL_S = 3 * 24 * 3600
_MAX_MEMORIA = 4096
_PREFIJO = f"mealfit:micros_comida:{VERSION}:"

_memoria: "OrderedDict[str, Any]" = OrderedDict()
_cerrojo = threading.Lock()
_NADA = "__sin_micros__"      # «esta comida no tiene micros» también se guarda (no se recalcula)
ACTIVO_EN_TESTS = False


def huella(ingredients: Any) -> Optional[str]:
    """La clave de una lista de ingredientes (cadenas), o None si no hay nada que calcular."""
    if not isinstance(ingredients, list) or not ingredients:
        return None
    try:
        crudo = json.dumps([str(x) for x in ingredients], ensure_ascii=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha1(crudo.encode("utf-8")).hexdigest()


def _redis():
    try:
        from cache_manager import redis_client
        return redis_client
    except Exception:  # noqa: BLE001
        return None


def _leer(clave: str):
    with _cerrojo:
        if clave in _memoria:
            _memoria.move_to_end(clave)
            return True, _memoria[clave]
    r = _redis()
    if r is None:
        return False, None
    try:
        crudo = r.get(_PREFIJO + clave)
    except Exception:  # noqa: BLE001
        return False, None
    if not crudo:
        return False, None
    try:
        valor = None if crudo == _NADA else json.loads(crudo)
    except Exception:  # noqa: BLE001
        return False, None
    _guardar_en_memoria(clave, valor)
    return True, valor


def _guardar_en_memoria(clave: str, valor) -> None:
    with _cerrojo:
        _memoria[clave] = valor
        _memoria.move_to_end(clave)
        while len(_memoria) > _MAX_MEMORIA:
            _memoria.popitem(last=False)


def _guardar(clave: str, valor) -> None:
    _guardar_en_memoria(clave, valor)
    r = _redis()
    if r is None:
        return
    try:
        r.setex(_PREFIJO + clave, TTL_S, _NADA if valor is None else json.dumps(valor, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        pass


def micros_de_ingredientes_memo(ingredients: Any, db) -> Optional[dict]:
    """Lo MISMO que `diary_micros.micros_de_ingredientes(ingredients, db)`, recordado por la huella de los ingredientes."""
    from diary_micros import micros_de_ingredientes
    clave = huella(ingredients)
    # en los tests, sin memoria salvo los suyos (`ACTIVO_EN_TESTS`): un resultado no debe pasar de un test a otro
    if clave is None or (os.environ.get("PYTEST_CURRENT_TEST") and not ACTIVO_EN_TESTS):
        return micros_de_ingredientes(ingredients, db)
    hay, valor = _leer(clave)
    if hay:
        return copy.deepcopy(valor)      # quien lo use puede tocarlo (micros_plausibles, resumen_micros): nunca el guardado
    valor = micros_de_ingredientes(ingredients, db)
    _guardar(clave, copy.deepcopy(valor))
    return valor


def calentar(ingredients: Any) -> None:
    """Calcula y guarda los micros de una comida recién registrada (para tareas en segundo plano). Nunca lanza."""
    try:
        # en los tests no: `log_consumed_meal` se llama con BD falsa y el catálogo no debe salir a la red
        if huella(ingredients) is None or os.environ.get("PYTEST_CURRENT_TEST"):
            return
        from nutrition_db import IngredientNutritionDB
        micros_de_ingredientes_memo(ingredients, IngredientNutritionDB())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[P1-PLAN-LOTE-1085] calentar micros: {type(e).__name__}")


def _vaciar_memoria() -> None:
    """Para los tests."""
    with _cerrojo:
        _memoria.clear()
