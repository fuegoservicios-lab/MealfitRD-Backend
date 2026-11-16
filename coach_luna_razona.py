# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1043 · A-20 · 2026-11-16] Luna razonando CON herramientas, por la API de Responses.

En Chat Completions `gpt-6-luna` rechaza herramientas + `reasoning_effort` (lote 600): el coach va siempre con
`reasoning_effort='none'`. La API de Responses sí combina las dos cosas (es el camino que ya usa terra). Este módulo
lo prueba para el modelo principal, detrás de un knob APAGADO por defecto, para medirlo con la batería antes de
decidir nada:

    MEALFIT_CHAT_LUNA_RESPONSES_EFFORT = "" (apagado) | "low" | "medium"

Solo cambia el cliente del turno normal de Luna; el turno escalado (terra), la voz y los demás modelos no se tocan.
"""
from __future__ import annotations

from typing import Optional

from knobs import _env_str

_ESFUERZOS = ("low", "medium")


def esfuerzo() -> Optional[str]:
    v = (_env_str("MEALFIT_CHAT_LUNA_RESPONSES_EFFORT", "") or "").strip().lower()
    return v if v in _ESFUERZOS else None


def _es_luna(model) -> bool:
    m = str(model or "").strip().lower()
    return m.startswith("gpt-6") and "luna" in m


def activo_para(model, *, escalado: bool = False, modo_voz: bool = False) -> bool:
    return bool(esfuerzo()) and _es_luna(model) and not escalado and not modo_voz


def kwargs_cliente(model, *, escalado: bool = False, modo_voz: bool = False) -> dict:
    """Kwargs extra para el `ChatGLM` del turno normal: {} si no aplica."""
    if not activo_para(model, escalado=escalado, modo_voz=modo_voz):
        return {}
    return {"use_responses_api": True, "reasoning_effort": esfuerzo()}


def tool_choice_responses(nombres: list) -> dict:
    """`allowed_tools` en la forma de Responses (sin el envoltorio `function` de Chat Completions)."""
    return {"type": "allowed_tools", "mode": "auto",
            "tools": [{"type": "function", "name": n} for n in nombres]}
