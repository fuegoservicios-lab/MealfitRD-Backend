"""[P3-1 · 2026-05-10] Regression guard: docstring SSOT en `constants.py`
documenta las 3 keys de learning (`_last_chunk_learning`,
`_recent_chunk_lessons`, `_lifetime_chunk_lessons`).

Bug raíz (audit 2026-05-10):
    Las 3 keys conviven en `meal_plans.plan_data` y se inyectan al prompt
    del LLM en momentos distintos del pipeline. Antes de P3-1 no había
    un documento único explicando cuál es cuál, retención, dónde se
    inyecta. Riesgo de doble-inyección o fallback incorrecto en recovery
    paths cuando un developer toca el flow sin contexto.

Fix:
    Bloque docstring centralizado en `constants.py` con tabla de
    semántica + retención + path de inyección para las 3 keys, +
    referencias a tests existentes que validan invariantes cruzados.

Cobertura de este test (parser-based, no DB):
    1. `constants.py` contiene el marker P3-1 + 2026-05-10.
    2. Las 3 keys están enumeradas explícitamente en el docstring.
    3. El docstring menciona la diferencia entre las 3 (cap, retención,
       inyección).
    4. Cross-references a tests de regresión relevantes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_CONSTANTS_PATH = _BACKEND_ROOT / "constants.py"


def _read() -> str:
    assert _CONSTANTS_PATH.exists(), f"constants.py no encontrado en {_CONSTANTS_PATH}"
    return _CONSTANTS_PATH.read_text(encoding="utf-8")


def test_marker_present():
    src = _read()
    assert "P3-1 · 2026-05-10" in src, (
        "El bloque docstring debe estar marcado con `[P3-1 · 2026-05-10]` "
        "para trazabilidad cross-archivo (memoria, marker, este test)."
    )


def test_three_keys_enumerated():
    """Las 3 keys de learning deben aparecer EXPLÍCITAMENTE como
    backticks-quoted en el bloque docstring."""
    src = _read()
    # Buscar la sección entre el marker P3-1 y la siguiente declaración Python.
    block_match = re.search(
        r"#\s*\[P3-1\s*·\s*2026-05-10\][\s\S]+?(?=^(?:[A-Z]|def\s|class\s))",
        src,
        re.MULTILINE,
    )
    assert block_match is not None, (
        "No se encontró el bloque docstring P3-1 (debe estar antes de "
        "`CHUNK_CRITICAL_LESSONS_MAX`)."
    )
    block = block_match.group(0)
    required_keys = (
        "`_last_chunk_learning`",
        "`_recent_chunk_lessons`",
        "`_lifetime_chunk_lessons`",
    )
    for key in required_keys:
        assert key in block, (
            f"Docstring debe mencionar `{key}` (backtick-quoted) para "
            f"distinguirla visualmente de comentarios sobre otras keys."
        )


def test_block_documents_retention_or_cap():
    """El docstring debe explicar la diferencia de retención entre las 3
    keys — sin eso, alguien que lo lea sin contexto previo no podrá
    decidir cuál tocar en cada caso."""
    src = _read()
    block_match = re.search(
        r"#\s*\[P3-1\s*·\s*2026-05-10\][\s\S]+?(?=^(?:[A-Z]|def\s|class\s))",
        src,
        re.MULTILINE,
    )
    block = block_match.group(0)
    # Conceptos clave: rolling window (para _recent), cap/MAX (para _lifetime).
    assert "rolling window" in block.lower() or "rolling-window" in block.lower(), (
        "Docstring debe mencionar `rolling window` (concepto clave de "
        "`_recent_chunk_lessons`)."
    )
    assert "CHUNK_CRITICAL_LESSONS_MAX" in block, (
        "Docstring debe citar `CHUNK_CRITICAL_LESSONS_MAX` (cap de "
        "`_lifetime_chunk_lessons`) para que el lector pueda navegar al "
        "número exacto."
    )


def test_block_cross_references_tests():
    """El docstring debe apuntar a tests existentes que protegen los
    invariantes. Sin cross-link, alguien que cambia el flow no sabrá
    qué tests deben pasar."""
    src = _read()
    block_match = re.search(
        r"#\s*\[P3-1\s*·\s*2026-05-10\][\s\S]+?(?=^(?:[A-Z]|def\s|class\s))",
        src,
        re.MULTILINE,
    )
    block = block_match.group(0)
    # Al menos un test concreto debe aparecer referenciado.
    test_refs = (
        "test_p0_1_learning_atomicity",
        "test_chunked_learning_propagation",
        "test_p0_7_critical_lessons_cap",
    )
    found = [t for t in test_refs if t in block]
    assert found, (
        f"Docstring debe citar al menos uno de los tests de regresión "
        f"relacionados ({test_refs}). Sin cross-link, el cleanup no es "
        f"navegable."
    )
