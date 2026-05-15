"""[P3-NEW-SPLASH-A11Y · 2026-05-15] Anchor + regression guard.

`frontend/index.html` tenía el splash `.splash-dots` con sólo
`aria-label="Cargando"` — screen readers no anunciaban el estado de carga
porque el div no tenía `aria-live` ni `role="status"`. Pre-fix:

    <div class="splash-dots" aria-label="Cargando">

Post-fix: añadidos `aria-live="polite"`, `aria-busy="true"`, `role="status"`
para que screen readers (NVDA/JAWS/VoiceOver) anuncien la actividad.
`role="status"` implica `aria-live="polite"` por default pero ambos
ayudan a navegadores con bugs de a11y tree.

Defensas que el test enforza:
  1. Anchor `P3-NEW-SPLASH-A11Y` presente en index.html.
  2. El elemento `.splash-dots` tiene los 4 atributos a11y obligatorios:
     `aria-label`, `aria-live="polite"`, `aria-busy="true"`, `role="status"`.
  3. Anchor presente en este archivo (cross-link guard P2-HIST-AUDIT-14).
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDEX_HTML = _REPO_ROOT / "frontend" / "index.html"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_index_html():
    src = _read(_INDEX_HTML)
    assert "P3-NEW-SPLASH-A11Y" in src, (
        "Falta anchor `P3-NEW-SPLASH-A11Y` en frontend/index.html."
    )


def test_splash_dots_has_all_a11y_attrs():
    """El `<div class="splash-dots" ...>` debe tener TODOS los atributos a11y:
    aria-label, aria-live="polite", aria-busy="true", role="status"."""
    src = _read(_INDEX_HTML)
    # Aislar la línea con `splash-dots` que NO sea CSS rule (.splash-dots {)
    m = re.search(
        r'<div\s+class="splash-dots"[^>]*>',
        src,
    )
    assert m is not None, (
        "No se encontró `<div class=\"splash-dots\" ...>` en index.html."
    )
    tag = m.group(0)
    required_attrs = [
        ('aria-label', r'aria-label\s*=\s*"[^"]+"'),
        ('aria-live="polite"', r'aria-live\s*=\s*"polite"'),
        ('aria-busy="true"', r'aria-busy\s*=\s*"true"'),
        ('role="status"', r'role\s*=\s*"status"'),
    ]
    missing = []
    for name, pat in required_attrs:
        if not re.search(pat, tag):
            missing.append(name)
    assert not missing, (
        f"Splash dots tag falta atributos a11y: {missing}. "
        f"Tag actual: {tag}"
    )


def test_anchor_present_in_test_file():
    src = _read(Path(__file__))
    assert "P3-NEW-SPLASH-A11Y" in src
