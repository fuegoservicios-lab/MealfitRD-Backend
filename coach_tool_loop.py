# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1016 · A-07 / A-17] Tope de rondas de herramientas por turno y corte de repeticiones.

La batería del 14-nov con la caché v2 encendida dio un caso («se me acabó el pollo») en el que el modelo llamó 277 veces seguidas a
`check_hydration_today` con el MISMO resultado: 422 s, 280 llamadas y US$ 0,116 en un solo turno, hasta que el tope de tiempo del turno
(420 s) lo cortó. LangGraph no tiene aquí un límite de recursión útil (1000 pasos) y el tope diario por usuario no corta a mitad de
turno: un bucle así era posible desde siempre y solo se vio al restringir las herramientas.

Reglas (puras, sin IA):
  · más de `MEALFIT_CHAT_MAX_TOOL_ROUNDS` (6) rondas de herramientas desde el último mensaje del usuario ⇒ cortar;
  · la MISMA llamada (nombre + argumentos) tres veces seguidas ⇒ cortar.
Cortar = la siguiente llamada va SIN herramientas, con una instrucción de contestar con lo que ya hay. El corte se cuenta (fail_open
crítico: un bucle es un fallo de una protección, no un turno normal).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from knobs import _env_int


def max_rondas() -> int:
    return _env_int("MEALFIT_CHAT_MAX_TOOL_ROUNDS", 6, lambda v: 1 <= v <= 30)


def _es_humano(m: Any) -> bool:
    return getattr(m, "type", None) in ("human", "user")


def _llamadas(m: Any) -> list:
    if getattr(m, "type", None) != "ai":
        return []
    return list(getattr(m, "tool_calls", None) or [])


def _firma(tc: Any) -> str:
    nombre = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
    try:
        a = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        a = str(args)
    return f"{nombre}|{a}"


def rondas_de_este_turno(messages: list) -> list[list[str]]:
    """Firmas de las llamadas de cada ronda (un AIMessage con tool_calls) desde el último mensaje de usuario."""
    rondas: list[list[str]] = []
    for m in reversed(messages or []):
        if _es_humano(m):
            break
        ll = _llamadas(m)
        if ll:
            rondas.append([_firma(t) for t in ll])
    rondas.reverse()
    return rondas


def motivo_de_corte(messages: list, *, repeticiones: int = 3) -> Optional[str]:
    """`"max_rondas"`, `"repeticion"` o `None`."""
    rondas = rondas_de_este_turno(messages)
    if len(rondas) >= max_rondas():
        return "max_rondas"
    if len(rondas) >= repeticiones:
        ultimas = rondas[-repeticiones:]
        if all(r == ultimas[0] for r in ultimas):
            return "repeticion"
    return None


INSTRUCCION_DE_CORTE = (
    "[Instrucción del sistema] Ya llamaste a las herramientas varias veces en este turno y no están dando información nueva. "
    "No llames a más herramientas: responde ahora, en el idioma del usuario, con lo que ya tienes. Si no pudiste hacer lo que pidió, "
    "dilo con claridad y qué puede hacer él (sin inventar que quedó hecho)."
)
