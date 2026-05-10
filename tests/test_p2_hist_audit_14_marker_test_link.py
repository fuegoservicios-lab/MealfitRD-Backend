"""[P2-HIST-AUDIT-14 · 2026-05-09] Cross-link entre `_LAST_KNOWN_PFIX`
en `app.py` y la existencia de un test file que cubra ese P-fix.

Bug original (audit Historial 2026-05-09 — gap residual):
    `_LAST_KNOWN_PFIX` en `app.py:32` se bumpea con cada cierre de
    P-fix por convención (CLAUDE.md). El test P3-1 (`test_p3_1_last_known_pfix_freshness`)
    valida formato + floor de fecha, pero NO valida que el marker
    actual tenga un test de regresión asociado.

    Escenarios que P3-1 NO detecta y este test SÍ:
      - Bump cosmético del marker sin código nuevo: alguien actualiza
        `_LAST_KNOWN_PFIX` para que no quede stale en `/health/version`
        sin que haya un test cubriendo el cambio. Resultado: el operador
        confía en el marker pero no hay regresión protegida.
      - P-fix sin trazabilidad de tests: si alguien implementa el fix
        (toca código), bumpea el marker, pero olvida añadir el test —
        el marker miente sobre la cobertura.

Fix:
    Cross-link: el slug del marker (`P2-HIST-AUDIT-13` →
    `p2_hist_audit_13`) debe matchear al menos un test file
    `tests/test_<slug>*.py`. Falla loud si no hay match.

Limitaciones (out of scope):
    - NO enforza calidad del test (solo existencia).
    - NO enforza que el test esté verde (los demás runs lo cubren).
    - NO valida markers HISTÓRICOS — solo el actual. Aplicar a
      historial requeriría git log; queda fuera de un test unitario
      self-contained.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_TESTS_DIR = _BACKEND_ROOT / "tests"


def _read_marker_value() -> str:
    """Extrae el literal `_LAST_KNOWN_PFIX = "..."` de app.py vía
    regex (mismo patrón que P3-1 freshness — no importa el módulo
    para evitar disparar inits)."""
    text = _APP_PY.read_text(encoding="utf-8")
    m = re.search(
        r'_LAST_KNOWN_PFIX\s*=\s*[\'"]([^\'"]+)[\'"]',
        text,
    )
    assert m is not None, "_LAST_KNOWN_PFIX no encontrado en app.py"
    return m.group(1)


def _marker_to_slug(marker: str) -> str:
    """`P2-HIST-AUDIT-13 · 2026-05-09` → `p2_hist_audit_13`. Strip
    fecha; lower; reemplazar `-` con `_`."""
    # Quitar fecha (todo lo que sigue al ` · `).
    prefix = marker.split("·", 1)[0].strip()
    # `P2-HIST-AUDIT-13` → `p2_hist_audit_13`
    return prefix.replace("-", "_").lower()


# ---------------------------------------------------------------------------
# 1. Anchor del marker
# ---------------------------------------------------------------------------
def test_marker_has_associated_test_file():
    """El marker actual de `_LAST_KNOWN_PFIX` debe tener al menos un
    archivo `tests/test_<slug>*.py` que cubra el P-fix.

    Cuando se bumpea el marker, el dev debe haber añadido (o
    confirmado preexistente) un test correspondiente. Sin esto, el
    bump es cosmético — el marker miente sobre la cobertura.
    """
    marker = _read_marker_value()
    slug = _marker_to_slug(marker)
    pattern = f"test_{slug}*.py"
    matches = list(_TESTS_DIR.glob(pattern))
    assert matches, (
        f"El marker actual `{marker}` (slug={slug}) NO tiene test "
        f"asociado en `backend/tests/`.\n"
        f"Esperaba al menos un archivo que matchee `{pattern}`.\n"
        f"Cuando bumpees `_LAST_KNOWN_PFIX`, añade el test de "
        f"regresión correspondiente — el marker no debe ser solo "
        f"cosmético."
    )


def test_marker_test_file_naming_convention():
    """Defensa contra typos: el slug derivado del marker no debe
    contener caracteres no válidos para un nombre de archivo
    Python."""
    marker = _read_marker_value()
    slug = _marker_to_slug(marker)
    # Slug solo letras/dígitos/underscore — no espacios, no
    # caracteres especiales (excluyendo los reemplazados).
    assert re.match(r"^[a-z0-9_]+$", slug), (
        f"Slug derivado del marker contiene caracteres inválidos: "
        f"`{slug}` (de marker `{marker}`)."
    )


def test_p3_1_freshness_test_still_passing():
    """Defensa-en-profundidad: P3-1 freshness ya valida formato +
    floor. Aquí solo confirmamos que el archivo del test existe
    para que un dev no lo borre por accidente y rompa la chain
    de validaciones del marker."""
    p3_1 = _TESTS_DIR / "test_p3_1_last_known_pfix_freshness.py"
    assert p3_1.exists(), (
        "test_p3_1_last_known_pfix_freshness.py es la primera línea "
        "de defensa del marker — su ausencia rompe el contrato."
    )


# ---------------------------------------------------------------------------
# 2. Sanity: el slug parser no rompe ante markers válidos conocidos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("marker,expected_slug", [
    ("P0-AUDIT-HIST-1 · 2026-05-09", "p0_audit_hist_1"),
    ("P1-AUDIT-HIST-7 · 2026-05-09", "p1_audit_hist_7"),
    ("P2-HIST-AUDIT-13 · 2026-05-09", "p2_hist_audit_13"),
    ("P3-NEW-A · 2026-05-08", "p3_new_a"),
    ("P3-CANDIDATE-B · 2026-05-08", "p3_candidate_b"),
])
def test_marker_slug_extraction(marker, expected_slug):
    """Cobertura del parser de slug — patrones canónicos del repo."""
    assert _marker_to_slug(marker) == expected_slug
