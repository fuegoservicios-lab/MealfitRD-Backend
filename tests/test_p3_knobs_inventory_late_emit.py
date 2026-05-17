"""[P3-KNOBS-INVENTORY-LATE-EMIT · 2026-05-15] Regression guard: la llamada
`_log_active_knobs()` debe vivir AL FINAL del módulo (después de todas las
asignaciones de knobs module-level), no en línea ~489 como pre-fix.

Pre-fix (2026-05-15 18:56:45 startup): la llamada estaba en línea 489,
ANTES de unos 25-30 knobs module-level declarados downstream
(`DAY_GEN_RETRY_USE_PRO`, `PROMPT_CACHE_SYSTEM_MESSAGE`,
`PROMPT_TRIM_FORM_DATA`, etc.). Resultado: `[KNOBS/INVENTORY]` log
emitía un snapshot incompleto → operador veía aparentemente que sus
fixes no se habían deployado. Los knobs SÍ funcionaban en runtime
(el registry se completa al final del import), solo el log mentía.

Fix: mover `_log_active_knobs()` al final del módulo (post `if __name__`
del sync wrapper). El comentario stub en línea ~489 explica el cambio y
apunta al call site real al final.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GRAPH_PATH = _BACKEND_ROOT / "graph_orchestrator.py"


def _read_graph() -> str:
    return _GRAPH_PATH.read_text(encoding="utf-8")


def _strip_comments(text: str) -> str:
    """Quita líneas que empiezan con `#` para que menciones en comentarios
    explicativos no contaminen el matching."""
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    )


def test_log_active_knobs_function_defined():
    """La función helper debe seguir existiendo (la definición no se movió,
    solo su invocación)."""
    text = _read_graph()
    assert re.search(r"^def _log_active_knobs\(", text, re.MULTILINE), (
        "P3-KNOBS-INVENTORY-LATE-EMIT: helper `_log_active_knobs` debe seguir "
        "definido top-level."
    )


def test_log_active_knobs_invocation_is_after_prompt_knobs():
    """La invocación `_log_active_knobs()` debe venir DESPUÉS de las
    asignaciones de `PROMPT_CACHE_SYSTEM_MESSAGE` y `PROMPT_TRIM_FORM_DATA`.
    Si vive antes, esos knobs no aparecen en el `[KNOBS/INVENTORY]` log."""
    text = _strip_comments(_read_graph())

    invocations = [m.start() for m in re.finditer(r"^_log_active_knobs\(\)", text, re.MULTILINE)]
    assert len(invocations) >= 1, (
        "P3-KNOBS-INVENTORY-LATE-EMIT: debe existir al menos una invocación "
        "`_log_active_knobs()` a nivel módulo (col 0)."
    )

    knob_prompt_cache = re.search(
        r"^PROMPT_CACHE_SYSTEM_MESSAGE\s*=\s*_env_bool", text, re.MULTILINE
    )
    knob_prompt_trim = re.search(
        r"^PROMPT_TRIM_FORM_DATA\s*=\s*_env_bool", text, re.MULTILINE
    )
    assert knob_prompt_cache and knob_prompt_trim, (
        "Faltan las asignaciones de los knobs PROMPT_*. Si fueron renombrados, "
        "actualizar este test."
    )

    # La invocación (la última si hay varias) debe venir DESPUÉS de ambas.
    last_invocation = invocations[-1]
    assert last_invocation > knob_prompt_cache.start(), (
        f"`_log_active_knobs()` (pos {last_invocation}) debe venir DESPUÉS "
        f"de `PROMPT_CACHE_SYSTEM_MESSAGE` (pos {knob_prompt_cache.start()}). "
        f"Si vive antes, el inventory log no captura este knob."
    )
    assert last_invocation > knob_prompt_trim.start(), (
        f"`_log_active_knobs()` (pos {last_invocation}) debe venir DESPUÉS "
        f"de `PROMPT_TRIM_FORM_DATA` (pos {knob_prompt_trim.start()}). "
    )


def test_log_active_knobs_invocation_near_module_end():
    """La invocación debe estar cerca del FINAL del archivo (últimas 200
    líneas). Es la garantía estructural de que captura *cualquier* knob
    module-level declarado en el cuerpo del módulo, no solo los actuales."""
    text = _read_graph()
    code_only = _strip_comments(text)
    invocations = [m.start() for m in re.finditer(r"^_log_active_knobs\(\)", code_only, re.MULTILINE)]
    assert invocations, "No hay invocación de `_log_active_knobs()` a nivel módulo."
    last_invocation = invocations[-1]

    # Cuántos chars del módulo (sin comentarios) vienen DESPUÉS de la invocación.
    chars_after = len(code_only) - last_invocation
    # Permitimos hasta ~2000 chars después (saltos de línea + posibles utilidades
    # cortas). Si alguien añade ~50 líneas de código module-level DESPUÉS de
    # la invocación, este test falla y obliga a re-pensar el orden.
    assert chars_after < 2000, (
        f"`_log_active_knobs()` no está suficientemente al final del módulo: "
        f"{chars_after} chars de código vienen después. Mover más al final."
    )


def test_no_orphan_early_invocation():
    """Verifica que NO queda una invocación temprana (~línea 489) huérfana
    del refactor. Si quedara, el inventory log emite DOS veces — una temprana
    incompleta y otra al final completa, lo cual confunde al operador."""
    text = _strip_comments(_read_graph())
    invocations = list(re.finditer(r"^_log_active_knobs\(\)", text, re.MULTILINE))
    assert len(invocations) == 1, (
        f"P3-KNOBS-INVENTORY-LATE-EMIT: esperaba exactamente 1 invocación de "
        f"`_log_active_knobs()` a nivel módulo (col 0), encontré {len(invocations)}. "
        f"Si pre-fix tenía una temprana + la nueva al final, eliminar la temprana."
    )
