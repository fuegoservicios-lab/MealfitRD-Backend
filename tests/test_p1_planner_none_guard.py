"""[P1-PLANNER-NONE-GUARD · 2026-06-15] Guard contra None del structured output del planner.

Bug original (gap-audit 2026-06-15, G1):
    En `plan_skeleton_node` → `invoke_planner()`, el parser de structured output
    (`with_structured_output(PlanSkeletonModel)`, PydanticToolsParser first_tool_only)
    devuelve `None` cuando el modelo NO emite tool-call (texto plano, posible bajo
    carga o con thinking deshabilitado). El código hacía:
        res = await _safe_ainvoke(...)
        await _planner_cb.arecord_success()   # ← registraba ÉXITO con res=None
        return res
    y aguas abajo:
        skeleton = response.model_dump()  /  response.dict()   # ← None.dict() → AttributeError
    Ese AttributeError escapaba FUERA del scope de tenacity (que ya había retornado),
    lo capturaba el handler global y degradaba el plan a fallback matemático TOTAL —
    quemando un transient que un simple retry (con bump de temperatura por intento)
    recupera. Era la fuente confirmada del transient "NoneType.dict()".

Cierre:
    Dentro de `invoke_planner`, ANTES de `arecord_success`, si `res is None` se lanza
    `ValueError(...)`. Como ValueError no es spend-cap, el predicado de retry de tenacity
    lo cubre → reintenta hasta 3 veces; agotados los intentos propaga un error TIPADO
    (no AttributeError). El None cuenta como fallo de salud del modelo en el CB, igual
    que cualquier fallo de parseo (consistente con el comportamiento existente).

Este test (parser-based, sin DB ni LLM) enforza que el guard existe y está en el orden
correcto. Si un refactor lo elimina o lo mueve después de `arecord_success`, falla.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _invoke_planner_body() -> str:
    """Extrae el cuerpo de `async def invoke_planner` hasta su invocación."""
    start = _SRC.find("async def invoke_planner")
    assert start != -1, "No se encontró `async def invoke_planner` — ¿renombrado?"
    end = _SRC.find("response = await invoke_planner()", start)
    assert end != -1, "No se encontró la invocación `response = await invoke_planner()`"
    return _SRC[start:end]


def test_planner_none_guard_present():
    body = _invoke_planner_body()
    # El guard debe chequear None del resultado del structured output y lanzar.
    assert re.search(r"if\s+res\s+is\s+None\s*:", body), (
        "Falta el guard `if res is None:` en invoke_planner (G1 / P1-PLANNER-NONE-GUARD). "
        "Sin él, un structured-output None degrada el plan a fallback total vía None.dict()."
    )
    assert "P1-PLANNER-NONE-GUARD" in body, "Falta el tooltip-anchor P1-PLANNER-NONE-GUARD."


# La LLAMADA real (no la mención en el comentario) lleva el prefijo del CB y paréntesis.
_RECORD_SUCCESS_CALL = "_planner_cb.arecord_success()"


def test_none_guard_raises_before_record_success():
    """El guard debe lanzar ANTES de la llamada `_planner_cb.arecord_success()` — si
    registra éxito con None, el bug persiste (retorna None y crashea aguas abajo)."""
    body = _invoke_planner_body()
    guard_idx = body.find("if res is None")
    success_idx = body.find(_RECORD_SUCCESS_CALL)
    assert guard_idx != -1 and success_idx != -1
    assert guard_idx < success_idx, (
        "El guard `if res is None` debe aparecer ANTES de `_planner_cb.arecord_success()`. "
        "Registrar éxito con res=None reintroduce el bug del None.dict()."
    )
    # Entre el guard y la llamada de éxito debe haber un raise (no un return/log silencioso).
    between = body[guard_idx:success_idx]
    assert "raise" in between, (
        "El guard de None debe `raise` (para que tenacity reintente), no continuar."
    )


def test_no_unguarded_none_record_success_pattern():
    """Regresión textual: debe existir el orden raise-on-None → arecord_success() → return res."""
    body = _invoke_planner_body()
    m_return = re.search(r"return\s+res", body)
    assert m_return, "invoke_planner debe `return res`."
    assert body.find("if res is None") < body.find(_RECORD_SUCCESS_CALL) < m_return.start()
