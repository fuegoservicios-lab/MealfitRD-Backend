"""[P3-3 · 2026-05-10] Regression guard: `get_embedding` usa `logger.error`
(no `print`) en su path de fallo + documenta la decisión fail-fast.

Bug raíz (audit 2026-05-10):
    `fact_extractor.get_embedding` capturaba excepciones del provider Gemini
    con `print(f"⚠️ Error al generar embedding: {e}")`. Tres problemas:
      1. `print` no llega a Sentry ni a log aggregation (vs `logger.error`).
      2. No documentaba la política de fallo (fail-fast con `[]` vs
         fallback a modelo legacy) → operador que ve el bug en producción
         no sabe si es regresión o comportamiento esperado.
      3. Sin contexto en el mensaje (modelo usado, longitud del texto) →
         operador no puede diagnosticar entre quota/rate-limit/timeout.

Fix:
    - `print(...)` → `logger.error(...)` con `{type(e).__name__}: {e}` +
      `modelo={_model_name!r}` + `text_len={len(text)}`.
    - Docstring extendido documenta la política `return []` como decisión
      deliberada (cache permanente + downstream tolera []).
    - `_model_name` bindeado ANTES del import de `constants` para que
      aunque el import falle, el log sea informativo.

Cobertura de este test (parser-based):
    1. `fact_extractor.py` importa `logging` y declara `logger`.
    2. `get_embedding` NO usa `print(...)` para errores (regresión guard).
    3. `get_embedding` usa `logger.error(...)` en el except.
    4. El log incluye `modelo=` y `text_len=` para diagnóstico.
    5. El docstring documenta la política fail-fast.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_FACT_EXTRACTOR_PATH = _BACKEND_ROOT / "fact_extractor.py"


def _read() -> str:
    return _FACT_EXTRACTOR_PATH.read_text(encoding="utf-8")


def _get_embedding_body() -> str:
    src = _read()
    match = re.search(
        r"def\s+get_embedding\s*\(.*?(?=\ndef\s+|\Z)",
        src, re.DOTALL,
    )
    assert match is not None, "`get_embedding` no encontrada en fact_extractor.py"
    return match.group(0)


def test_logging_imported_and_logger_declared():
    src = _read()
    assert re.search(r"^import\s+logging", src, re.MULTILINE), (
        "fact_extractor.py debe `import logging` para usar logger.error."
    )
    assert re.search(r"^logger\s*=\s*logging\.getLogger\(", src, re.MULTILINE), (
        "fact_extractor.py debe declarar `logger = logging.getLogger(__name__)`."
    )


def test_get_embedding_no_print_for_errors():
    """Regresión guard: el except NO debe usar `print(...)` — solo
    `logger.error`. Si alguien revierte, este test falla loud."""
    body = _get_embedding_body()
    # Buscar print() en el cuerpo (el debug del MISS sí puede ser print
    # legacy, pero idealmente también es logger.debug). El test foco es
    # que ningún `print(f"⚠️ Error` (o similar) sobreviva en el except.
    # Heurística: si hay `print(...)` con keyword "Error" cerca, fallar.
    assert not re.search(r'print\([^)]*[Ee]rror', body), (
        "`get_embedding` NO debe usar `print(...)` con error — usar "
        "`logger.error(...)`. P3-3 cierra el gap de visibilidad en Sentry."
    )


def test_get_embedding_uses_logger_error_with_context():
    """El log de error debe incluir contexto diagnóstico: modelo + longitud."""
    body = _get_embedding_body()
    assert "logger.error" in body, (
        "`get_embedding` debe llamar `logger.error(...)` en el except."
    )
    # Contexto mínimo: modelo + text_len.
    assert re.search(r"modelo\s*=\s*\{", body), (
        "El log debe incluir `modelo={_model_name!r}` para diagnóstico de "
        "quota/rate-limit (uno por modelo)."
    )
    assert re.search(r"text_len\s*=\s*\{", body), (
        "El log debe incluir `text_len={len(text)}` para detectar si el "
        "error es por texto extremadamente largo (límite del provider)."
    )


def test_docstring_documents_fail_fast_decision():
    """El docstring debe explicar POR QUÉ no hay fallback model — sin
    esto, un operador puede pensar que es regresión y añadir uno sin
    considerar el impacto (dim mismatch, cache versioning)."""
    body = _get_embedding_body()
    # Conceptos clave que el docstring debe contener.
    required_concepts = ("fail-fast", "fallback")
    for c in required_concepts:
        assert c.lower() in body.lower(), (
            f"Docstring de `get_embedding` debe mencionar `{c}` para "
            f"documentar la decisión de no añadir fallback model."
        )


def test_model_name_bound_before_import():
    """Defense-in-depth: `_model_name` debe inicializarse a un valor
    placeholder ANTES del `from constants import` — sin esto, un fallo
    en el import deja `_model_name` unbound y el except crashea."""
    body = _get_embedding_body()
    # Buscar el patrón `_model_name = "<unknown>"` antes del try.
    match = re.search(
        r"_model_name\s*=\s*[\"']<unknown>[\"'].*?try\s*:",
        body, re.DOTALL,
    )
    assert match is not None, (
        "`_model_name` debe bindearse a `\"<unknown>\"` ANTES del `try:` — "
        "si el `from constants import GEMINI_EMBEDDING_TEXT_MODEL` falla, "
        "el log del except referenciaría una variable no-bindeada (UnboundLocalError)."
    )
