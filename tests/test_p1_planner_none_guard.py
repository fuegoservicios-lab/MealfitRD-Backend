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

---

[REAPUNTADO · 2026-07-26] El guard **no se perdió**: el cuerpo de `invoke_planner` se extrajo
a `_do_planner_invoke(_llm, _cb, _model_label)` para que la ruta de **fallback a PRO** lo
reutilizara. `invoke_planner` quedó en una línea:

    async def invoke_planner():
        return await _do_planner_invoke(planner_llm, _planner_cb, planner_model)

El test seguía parseando `invoke_planner` y veía un cuerpo vacío ⇒ tres rojos que parecían una
regresión grave y no lo eran. El refactor de hecho MEJORÓ la cobertura: hoy el guard protege
también el camino PRO, que antes no pasaba por él.

Dos cosas cambiadas para que esto no vuelva a pasar:
  - la extracción se ancla al **marker** y a un límite estructural, no al nombre de la función
    que hoy contiene el código;
  - se añade el test que faltaba: que **todas** las invocaciones al planner pasen por el helper
    guardado. Eso es la invariante real ahora, y un futuro camino nuevo sin guard sí la rompe.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")

_GUARDED_FN = "_do_planner_invoke"


def _invoke_planner_body() -> str:
    """Cuerpo de la función que HOY contiene la invocación guardada al planner.

    ⚠️ No se ancla al nombre `invoke_planner`: ese nombre sigue existiendo pero su cuerpo es
    una delegación de una línea. Se ancla al marker (que viaja con el código si vuelve a
    moverse) y se corta en el decorador `@retry`, que es el límite estructural siguiente.
    """
    m = re.search(rf"async def {_GUARDED_FN}\b", _SRC)
    assert m, (
        f"No se encontró `async def {_GUARDED_FN}`. Si el helper se renombró, apunta este "
        "extractor a donde viva ahora el marker P1-PLANNER-NONE-GUARD."
    )
    start = m.start()
    end = _SRC.find("@retry(", start)
    assert end != -1, "No se encontró el `@retry(` que cierra el helper."
    body = _SRC[start:end]
    assert "P1-PLANNER-NONE-GUARD" in body, (
        "El marker no está en el cuerpo extraído — el guard se movió a otra función."
    )
    return body


def test_planner_none_guard_present():
    body = _invoke_planner_body()
    # El guard debe chequear None del resultado del structured output y lanzar.
    assert re.search(r"if\s+res\s+is\s+None\s*:", body), (
        "Falta el guard `if res is None:` (G1 / P1-PLANNER-NONE-GUARD). Sin él, un "
        "structured-output None degrada el plan a fallback total vía None.dict()."
    )
    assert "P1-PLANNER-NONE-GUARD" in body, "Falta el tooltip-anchor P1-PLANNER-NONE-GUARD."


# La LLAMADA real (no la mención en el comentario) lleva paréntesis. El CB llega ahora por
# parámetro (`_cb`) en vez de por closure (`_planner_cb`) — se acepta cualquiera de los dos.
_RECORD_SUCCESS_RE = re.compile(r"_(?:planner_)?cb\.arecord_success\(\)")


def test_none_guard_raises_before_record_success():
    """El guard debe lanzar ANTES de `arecord_success()` — si registra éxito con None, el bug
    persiste (retorna None y crashea aguas abajo)."""
    body = _invoke_planner_body()
    guard_idx = body.find("if res is None")
    m_success = _RECORD_SUCCESS_RE.search(body)
    assert guard_idx != -1, "no está el guard"
    assert m_success, "no está la llamada arecord_success()"
    assert guard_idx < m_success.start(), (
        "El guard `if res is None` debe aparecer ANTES de `arecord_success()`. "
        "Registrar éxito con res=None reintroduce el bug del None.dict()."
    )
    # Entre el guard y la llamada de éxito debe haber un raise (no un return/log silencioso).
    assert "raise" in body[guard_idx:m_success.start()], (
        "El guard de None debe `raise` (para que tenacity reintente), no continuar."
    )


def test_no_unguarded_none_record_success_pattern():
    """Regresión textual: debe existir el orden raise-on-None → arecord_success() → return res."""
    body = _invoke_planner_body()
    m_return = re.search(r"return\s+res\b", body)
    assert m_return, "el helper debe `return res`."
    m_success = _RECORD_SUCCESS_RE.search(body)
    assert m_success, "no está la llamada arecord_success()"
    i_guard = body.find("if res is None")
    # ⚠️ `find` devuelve -1 cuando NO está, y `-1 < x < y` es VERDADERO: sin este assert la
    # cadena de comparaciones se satisface precisamente en el caso que debe detectar. Verificado
    # por mutación (quitar el guard del fuente dejaba este test en verde).
    assert i_guard != -1, "falta el guard `if res is None` — la cadena de orden no prueba nada"
    assert i_guard < m_success.start() < m_return.start()


# ───────────── la invariante que el refactor hizo posible (y necesaria) ─────────────

def test_TODAS_las_invocaciones_al_planner_pasan_por_el_helper_guardado():
    """Antes había un solo camino al planner y bastaba con guardar ese cuerpo. Tras extraer el
    helper hay DOS (el normal y el fallback a PRO), y lo que protege al plan ya no es "el guard
    está en esta función" sino "nadie llama al LLM del planner por fuera del helper".

    Un camino nuevo que invoque el planner directo se saltaría el guard sin que ninguno de los
    tests de arriba se entere.
    """
    i0 = _SRC.index("async def plan_skeleton_node")
    bloque = _SRC[i0:_SRC.find("\nasync def ", i0 + 100)]
    directas = re.findall(r"_safe_ainvoke\(\s*(_?p?r?o?_?planner_llm|planner_llm)", bloque)
    assert len(directas) <= 1, (
        f"Hay {len(directas)} llamadas directas a `_safe_ainvoke(planner_llm...)` en "
        "plan_skeleton_node. Solo debe existir la de dentro de "
        f"`{_GUARDED_FN}`; cualquier otra se salta el guard de None."
    )


def test_el_fallback_PRO_reutiliza_el_helper():
    """La razón por la que el cuerpo se extrajo: que el camino PRO herede el guard."""
    assert _SRC.count(f"await {_GUARDED_FN}(") >= 2, (
        "Se esperaban al menos 2 llamadas al helper (planner normal + fallback PRO)."
    )
