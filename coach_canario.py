# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1012 · A-03] Despliegue por canario del coach.

Cada función de riesgo del plan v2 lee un knob con tres valores:

    off     apagada para todos (por defecto)
    canary  solo para los usuarios de `MEALFIT_CHAT_CANARY_USER_IDS`
    on      para todos

El brazo con el que corrió cada turno se guarda en `pipeline_metrics.metadata.arm` (lo escribe
`ChatTurnPhaseTracker.set_arm`) y `/admin/chat-perf` lo separa. Con 8 usuarios no hay potencia estadística para medir
efectos pequeños: el canario sirve para SEGURIDAD (probar en la cuenta del dueño antes que en todos), no para comparar.

Los nombres de knob son literales (no se arman con f-strings) para que queden registrados en `_KNOBS_REGISTRY`.
"""
from __future__ import annotations

from typing import Optional

from knobs import _env_str

MODOS = frozenset({"off", "canary", "on"})

# función → (descripción corta, lector del knob)
_LECTORES = {
    "cache_v2": lambda: _env_str("MEALFIT_CHAT_CANARY_CACHE_V2", "off", MODOS),          # A-01: tools fijas + manual aparte + cebador
    "contexto_creciente": lambda: _env_str("MEALFIT_CHAT_CANARY_CONTEXTO_CRECIENTE", "off", MODOS),  # A-08
    "una_llamada": lambda: _env_str("MEALFIT_CHAT_CANARY_UNA_LLAMADA", "off", MODOS),    # A-09
    "senal_temprana": lambda: _env_str("MEALFIT_CHAT_CANARY_SENAL_TEMPRANA", "on", MODOS),  # A-10 (solo emite eventos nuevos)
    "terra": lambda: _env_str("MEALFIT_CHAT_CANARY_TERRA", "canary", MODOS),             # A-07: turnos difíciles
}

FUNCIONES = tuple(_LECTORES)


def usuarios_canario() -> frozenset[str]:
    """`MEALFIT_CHAT_CANARY_USER_IDS`: ids separados por coma (espacios y mayúsculas se normalizan)."""
    raw = _env_str("MEALFIT_CHAT_CANARY_USER_IDS", "")
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def modo(funcion: str) -> str:
    lector = _LECTORES.get(funcion)
    if lector is None:
        raise KeyError(f"función de canario desconocida: {funcion!r} (válidas: {', '.join(FUNCIONES)})")
    return lector()


def brazo(user_id: Optional[str], funcion: str) -> str:
    """`on` / `canary` / `off`: lo que le toca a ESTE usuario con esa función."""
    m = modo(funcion)
    if m == "on":
        return "on"
    if m == "canary" and user_id and str(user_id).strip().lower() in usuarios_canario():
        return "canary"
    return "off"


def activa(user_id: Optional[str], funcion: str) -> bool:
    return brazo(user_id, funcion) != "off"


def registrar_brazo(tracker, user_id: Optional[str], funcion: str) -> bool:
    """Calcula el brazo, lo anota en el tracker del turno (si lo hay) y devuelve si la función está activa."""
    b = brazo(user_id, funcion)
    if tracker is not None:
        try:
            tracker.set_arm(funcion, b)
        except Exception:  # noqa: BLE001 — la medición nunca rompe el turno
            pass
    return b != "off"
