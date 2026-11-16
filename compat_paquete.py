# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1040 · A-33 · 2026-11-16] El servidor y los paquetes viejos de la app.

El servidor cambia en cada deploy; el teléfono aplica el paquete OTA nuevo al SEGUNDO arranque (y la PWA puede seguir
con el de su caché). El 14-nov un chat corrió con el paquete anterior justo tras el deploy. Dos piezas:

  · `CONTRATO_STREAM`: lo que el cliente LEE de cada evento del stream del chat. Mientras haya un paquete vivo que lo
    lea, el servidor no puede dejar de mandarlo. `tests/test_p1_plan_lote_1040_compat_paquete.py` recorre con AST cada
    evento que emite el servidor (cada emisión lleva los campos mínimos de su tipo) y lee `AgentPage.jsx` (el cliente
    no empieza a leer un campo que el contrato no conozca). Retirar un campo = sacarlo del contrato, y eso solo cuando
    `paquetes_vivos()` ya no muestre paquetes anteriores al cambio.
  · `registrar_paquete` / `paquetes_vivos`: qué paquetes (`ota:<sello>` o `web:<sello>`, del cuerpo de `/chat/stream`)
    hablaron con el servidor en los últimos días. Estado en `app_kv_store` (`compat_paquete:vistos`), escrito como mucho
    una vez cada 10 min por proceso salvo que aparezca un paquete nuevo. Fail-open: nunca rompe el turno.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PROTOCOLO_STREAM = 1

# Campos MÍNIMOS que lleva cada emisión de cada tipo (sin ellos el cliente no puede pintar el evento).
CAMPOS_MINIMOS = {
    "progress": frozenset({"phase", "message"}),
    "chunk": frozenset({"text"}),
    "done": frozenset({"response", "corr"}),
    "error": frozenset({"code", "message"}),
}
# Campos OPCIONALES que el cliente sabe leer (llegan solo cuando aplican: un plan nuevo, la nevera tocada…).
CAMPOS_OPCIONALES = {
    "progress": frozenset(),
    "chunk": frozenset({"budget_cap", "replayed"}),
    "done": frozenset({"updated_fields", "new_plan", "coherence_warnings", "ajustes_de_app", "pantry_modified_at",
                       "pantry_depleted_items", "diary_corrected_meal_ids", "budget_cap", "replayed"}),
    "error": frozenset(),
}
CONTRATO_STREAM = {t: CAMPOS_MINIMOS[t] | CAMPOS_OPCIONALES[t] for t in CAMPOS_MINIMOS}

CLAVE_KV = "compat_paquete:vistos"
OLVIDAR_S = 30 * 86400
ESCRIBIR_CADA_S = 600
_RE_PAQUETE = re.compile(r"^(?:ota|web):[A-Za-z0-9._@:-]{1,40}$")

_LOCK = threading.Lock()
_vistos: dict = {}            # paquete -> último visto (epoch)
_ultima_escritura = {"t": 0.0}


def paquete_limpio(valor: Any) -> Optional[str]:
    """El identificador si tiene la forma esperada; si no, None (nada del cliente entra sin forma)."""
    s = str(valor or "").strip()
    return s if _RE_PAQUETE.match(s) else None


def registrar_paquete(valor: Any, ahora: Optional[float] = None,
                      execute_sql_query: Optional[Callable] = None, execute_sql_write: Optional[Callable] = None) -> bool:
    """Anota que este paquete habló con el servidor. True si se persistió en esta llamada."""
    p = paquete_limpio(valor)
    if not p:
        return False
    ahora = time.time() if ahora is None else ahora
    with _LOCK:
        nuevo = p not in _vistos
        _vistos[p] = ahora
        toca = nuevo or (ahora - _ultima_escritura["t"] >= ESCRIBIR_CADA_S)
        if toca:
            _ultima_escritura["t"] = ahora
            locales = dict(_vistos)
    if not toca:
        return False
    try:
        if execute_sql_query is None or execute_sql_write is None:
            from db_core import execute_sql_query as _q, execute_sql_write as _w
            execute_sql_query, execute_sql_write = execute_sql_query or _q, execute_sql_write or _w
        guardados = _leer(execute_sql_query)
        for k, v in locales.items():
            guardados[k] = max(float(v), float(guardados.get(k, 0)))
        vivos = {k: v for k, v in guardados.items() if ahora - v < OLVIDAR_S}
        execute_sql_write(
            "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s::jsonb, NOW()) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
            (CLAVE_KV, json.dumps(vivos)),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[A-33] paquete no persistido: %s", type(exc).__name__)
        return False


def _leer(execute_sql_query) -> dict:
    fila = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (CLAVE_KV,), fetch_one=True)
    valor = fila.get("value") if isinstance(fila, dict) else None
    if isinstance(valor, str):
        try:
            valor = json.loads(valor)
        except ValueError:
            valor = None
    if not isinstance(valor, dict):
        return {}
    return {str(k): float(v) for k, v in valor.items() if paquete_limpio(k) and isinstance(v, (int, float))}


def paquetes_vivos(dias: int = 7, ahora: Optional[float] = None, execute_sql_query: Optional[Callable] = None) -> dict:
    """{n, por_tipo, mas_viejo, mas_nuevo} de los paquetes vistos en `dias`. Los sellos son ordenables (UTC)."""
    ahora = time.time() if ahora is None else ahora
    try:
        if execute_sql_query is None:
            from db_core import execute_sql_query
        todos = _leer(execute_sql_query)
    except Exception:  # noqa: BLE001
        todos = {}
    with _LOCK:
        for k, v in _vistos.items():
            todos[k] = max(v, todos.get(k, 0))
    vivos = sorted(k for k, v in todos.items() if ahora - v < dias * 86400)
    por_tipo: dict = {}
    for k in vivos:
        por_tipo.setdefault(k.split(":", 1)[0], []).append(k)
    return {
        "protocolo_stream": PROTOCOLO_STREAM,
        "n": len(vivos),
        "por_tipo": {t: len(v) for t, v in por_tipo.items()},
        "mas_viejo": {t: min(v) for t, v in por_tipo.items()},
        "mas_nuevo": {t: max(v) for t, v in por_tipo.items()},
    }


def reset_para_tests() -> None:
    with _LOCK:
        _vistos.clear()
        _ultima_escritura["t"] = 0.0
        _CACHE_VIVOS.update(t=0.0, valor=None)


_CACHE_VIVOS = {"t": 0.0, "valor": None}


def paquetes_vivos_cacheado(ttl_s: float = 60.0) -> dict:
    """Para `/health/version` (lo consultan crons y el deploy): una lectura de la base como mucho cada `ttl_s`."""
    ahora = time.monotonic()
    with _LOCK:
        if _CACHE_VIVOS["valor"] is not None and ahora - _CACHE_VIVOS["t"] < ttl_s:
            return _CACHE_VIVOS["valor"]
    valor = paquetes_vivos()
    with _LOCK:
        _CACHE_VIVOS.update(t=ahora, valor=valor)
    return valor
