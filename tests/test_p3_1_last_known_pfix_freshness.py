"""[P3-1 · 2026-05-08] Regression guard: `_LAST_KNOWN_PFIX` debe estar fresco.

Bug observado en el audit 2026-05-08 (post-P2-2):
  `_LAST_KNOWN_PFIX` en `app.py:24` quedó stale en `"P3-B · 2026-05-08"` durante
  varias rondas de P-fixes (P3-A audit late, P1-A knobs runtime, P1-B ChatWidget,
  P2-A safeJSONParse, P3-A constants docstring) sin bump. `/health/version`
  reportaba un marker desactualizado → diagnóstico de deploy menos preciso
  ("¿el último P-fix está vivo en prod?" responde mal).

Causa: marker mantenido humanamente sin enforcement. Cada P-fix DEBE bumpearlo
pero la convención no estaba documentada ni testeada.

Fix:
  1. Convención añadida a `CLAUDE.md` ("Convenciones del repo").
  2. Comentario inline en `app.py:24` apunta a la convención + este test.
  3. Este test bloquea regresiones:
     - Formato `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD` (suffix multi-segmento OK).
     - Date parses como ISO date.
     - Date >= floor (último audit cerrado, bumpeado junto con el marker).
     - Prefix válido (`P0`-`P9`).

Cuando un P-fix se mergea, el operador bumpea AMBOS:
  - `_LAST_KNOWN_PFIX` en `app.py:24`.
  - `_PFIX_DATE_FLOOR` aquí abajo.
Ambos en el mismo commit. Si solo uno cambia, el test falla en CI.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY_PATH = _BACKEND_ROOT / "app.py"

# Floor: el P-fix más reciente cerrado en HEAD. Bumpear AL MISMO TIEMPO que
# `_LAST_KNOWN_PFIX` en `app.py:24`. Marca la fecha mínima aceptable.
#
# Si has cerrado un P-fix posterior y olvidaste subir este floor, el test
# fallará intencionalmente — es la red de seguridad que cierra P3-1.
_PFIX_DATE_FLOOR = date(2026, 5, 14)  # P3-FACT-EXTRACTOR-SHADOW-AB (shadow A/B PRO→FLASH en fact_extractor — helper _invoke_with_shadow + 2 knobs + persist diff a pipeline_metrics, cero impacto UX)

# Formato de marker permitido: `P<n>(-<seg>)+ · YYYY-MM-DD`. Suffix
# multi-segmento permitido para `P2-NEW-A`, `P3-CANDIDATE-B`, etc.
_MARKER_PATTERN = re.compile(
    r"^(?P<prefix>P\d+(?:-[A-Z0-9]+)+)\s+·\s+(?P<date>\d{4}-\d{2}-\d{2})$"
)


def _read_marker_from_app_py() -> str:
    """Extrae el valor literal de `_LAST_KNOWN_PFIX` desde `app.py` sin
    importar el módulo (que dispara cron schedulers + DB init).

    Estrategia: regex sobre el source. Más rápida y aislada que `import app`.
    """
    text = _APP_PY_PATH.read_text(encoding="utf-8")
    m = re.search(
        r'^_LAST_KNOWN_PFIX\s*=\s*["\'](?P<val>[^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    assert m is not None, (
        "No se encontró asignación literal `_LAST_KNOWN_PFIX = '...'` en "
        f"{_APP_PY_PATH}. ¿Fue movido a otro módulo o computado dinámicamente? "
        "Si es intencional, actualizar este test."
    )
    return m.group("val")


def test_marker_present_and_format_valid():
    """`_LAST_KNOWN_PFIX` existe y sigue `Pn-(seg-)+ · YYYY-MM-DD`."""
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, (
        f"`_LAST_KNOWN_PFIX={marker!r}` no sigue el formato "
        f"`Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`. "
        f"Convención en CLAUDE.md → 'Convenciones del repo'."
    )


def test_marker_date_parses_as_iso():
    """La fecha en el marker debe ser ISO válida (YYYY-MM-DD)."""
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    date_str = m.group("date")
    try:
        datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        pytest.fail(
            f"Fecha en `_LAST_KNOWN_PFIX={marker!r}` no es ISO válida "
            f"({date_str!r}): {e}. Usar YYYY-MM-DD."
        )


def test_marker_date_meets_floor():
    """La fecha del marker debe ser >= `_PFIX_DATE_FLOOR`. Si subes el floor
    sin bumpear `_LAST_KNOWN_PFIX` (o viceversa), este test falla.

    Este es el corazón del enforcement: humanos olvidan bumpear; la fecha
    del marker debe siempre estar al día con el último P-fix cerrado.
    """
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    marker_date = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
    assert marker_date >= _PFIX_DATE_FLOOR, (
        f"`_LAST_KNOWN_PFIX={marker!r}` tiene fecha {marker_date} < "
        f"floor {_PFIX_DATE_FLOOR}. Bumpear el marker en `app.py:24` "
        f"con el último P-fix cerrado, o ajustar `_PFIX_DATE_FLOOR` en "
        f"este test si el floor está desfasado."
    )


def test_marker_prefix_uses_known_pfix_category():
    """El prefix `P<n>` debe estar en {P0..P9}. P10+ no son patrones existentes
    en el repo; un valor fuera de rango sugiere un typo (`PFIX-1`, `Q3-1`, etc.).
    """
    marker = _read_marker_from_app_py()
    m = _MARKER_PATTERN.match(marker)
    assert m is not None, f"Marker mal formado: {marker!r}"
    prefix = m.group("prefix")
    pfix_num_match = re.match(r"^P(\d+)", prefix)
    assert pfix_num_match is not None, f"Prefix sin número: {prefix!r}"
    pfix_num = int(pfix_num_match.group(1))
    assert 0 <= pfix_num <= 9, (
        f"Prefix `{prefix}` con número {pfix_num} fuera del rango P0-P9. "
        f"Si es intencional (creaste P10+), actualizar este test y CLAUDE.md."
    )


def test_inline_comment_references_convention():
    """El comentario sobre `_LAST_KNOWN_PFIX` en `app.py` debe referenciar
    la convención (CLAUDE.md) o este test, para que un futuro mantenedor
    sepa POR QUÉ debe bumpearse.

    Sin este anchor, el comentario podría borrarse en un refactor cosmético
    y el siguiente operador no entendería el contexto.
    """
    text = _APP_PY_PATH.read_text(encoding="utf-8")
    # Buscar bloque de comentarios inmediatamente antes de la asignación.
    block_match = re.search(
        r"((?:^#[^\n]*\n)+)_LAST_KNOWN_PFIX\s*=", text, re.MULTILINE
    )
    assert block_match is not None, (
        "Comentario sobre `_LAST_KNOWN_PFIX` desapareció. Restaurar el bloque "
        "que apunta a la convención (CLAUDE.md) y a este test."
    )
    block = block_match.group(1)
    # Debe mencionar al menos uno: CLAUDE.md, P3-1, o el nombre del test.
    anchors = ("CLAUDE.md", "P3-1", "test_p3_1_last_known_pfix_freshness")
    assert any(a in block for a in anchors), (
        f"Comentario sobre `_LAST_KNOWN_PFIX` no menciona ninguno de "
        f"{anchors}. Sin anchor, un refactor podría borrar la convención."
    )
