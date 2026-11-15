"""[P1-PLAN-LOTE-1026 · A-13] El prompt del turno viaja en la config de la ejecución, no en el checkpoint.

Medido en prod el 15-nov: 267 checkpoints = 67 MB, ~58 KB cada uno; casi todo era `sys_prompt` (~61 KB), que se
guardaba en CADA paso del grafo aunque se reescribe entero en cada turno y nadie lo lee de un turno a otro.

Ahora el stream lo pone en `config["configurable"]["__mf_sys_prompt"]`. LangGraph copia a la metadata del checkpoint
las cadenas de `configurable`, SALVO las claves que empiezan por «__» (`get_checkpoint_metadata`): no se persiste en
ningún sitio. Los nodos lo leen con `langgraph.config.get_config()`. En el estado queda `sys_prompt = ""`, que además
pisa el valor de 61 KB que traen los checkpoints viejos.

Fail-open: fuera de una ejecución del grafo (tests que llaman al nodo directamente) o con el knob apagado
(`MEALFIT_CHAT_CHECKPOINT_LIGERO=0`) se lee `state["sys_prompt"]` como siempre.
"""
from __future__ import annotations

import os

CLAVE = "__mf_sys_prompt"


def activo() -> bool:
    return (os.getenv("MEALFIT_CHAT_CHECKPOINT_LIGERO", "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def preparar(config: dict, system_prompt: str) -> str:
    """Mete el prompt en la config del turno y devuelve lo que se guarda en el estado ("" si está activo)."""
    if not activo():
        return system_prompt
    config.setdefault("configurable", {})[CLAVE] = system_prompt
    return ""


def prompt_del_turno(state) -> str:
    """El prompt de este turno: el de la config de la ejecución; si no hay, el del estado."""
    try:
        from langgraph.config import get_config
        p = (get_config().get("configurable") or {}).get(CLAVE)
        if p:
            return p
    except Exception:
        pass
    try:
        return str(state.get("sys_prompt") or "")
    except Exception:
        return ""
