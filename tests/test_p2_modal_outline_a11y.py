"""[P2-MODAL-OUTLINE-A11Y · 2026-05-15] Anchor + regression guard.

Pre-fix `frontend/src/components/common/Modal.jsx:166` aplicaba inline
`outline: 'none'` SIEMPRE al div del modal. El modal recibe foco programático
en `open` (para que screen readers anuncien el contenido) — pero un keyboard
user que estaba navegando no veía indicador visual de ese foco durante el
momento entre open y el primer Tab interno. Violación WCAG 2.4.7 ("Focus
Visible").

Fix:
  - Modal.jsx: añade `className="mealfit-modal-content"` al motion.div y
    elimina el inline `outline: 'none'`.
  - index.css: regla `:focus-visible` que solo dispara con keyboard nav
    (Tab) — `:focus` plain queda con `outline: none` para no romper UX
    de mouse users (click no debe pintar el ring).

Defensas que este test enforza:
  1. `Modal.jsx` tiene la className canónica `mealfit-modal-content`
     en el motion.div del role="dialog".
  2. `Modal.jsx` ya NO tiene `outline: 'none'` inline en ese motion.div.
  3. `index.css` tiene la regla `.mealfit-modal-content:focus-visible`
     con outline visible.
  4. Anchor `P2-MODAL-OUTLINE-A11Y` presente en ambos archivos.
"""
from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODAL_JSX = _REPO_ROOT / "frontend" / "src" / "components" / "common" / "Modal.jsx"
_INDEX_CSS = _REPO_ROOT / "frontend" / "src" / "index.css"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_anchor_present_in_modal():
    src = _read(_MODAL_JSX)
    assert "P2-MODAL-OUTLINE-A11Y" in src, (
        "Falta anchor `P2-MODAL-OUTLINE-A11Y` en Modal.jsx."
    )


def test_anchor_present_in_index_css():
    src = _read(_INDEX_CSS)
    assert "P2-MODAL-OUTLINE-A11Y" in src, (
        "Falta anchor `P2-MODAL-OUTLINE-A11Y` en index.css."
    )


def test_modal_uses_canonical_classname():
    src = _read(_MODAL_JSX)
    assert 'className="mealfit-modal-content"' in src, (
        "El motion.div del modal (role='dialog') debe tener "
        '`className="mealfit-modal-content"` — la clase ancla la regla '
        "`:focus-visible` en index.css."
    )


def test_modal_no_legacy_inline_outline_none():
    """El inline `outline: 'none'` aplicado al motion.div del modal era
    el origen del fail-de-a11y. Si reaparece como propiedad del style
    object del modal-content, la regla CSS queda override-eada y
    keyboard users vuelven a perder el indicador.
    """
    src = _read(_MODAL_JSX)
    # Buscar dentro del bloque del motion.div del role="dialog".
    dialog_match = re.search(
        r'role="dialog".*?style=\{\{(.+?)\}\}',
        src,
        re.DOTALL,
    )
    assert dialog_match is not None, (
        "No se pudo localizar el `style={{...}}` del motion.div del dialog."
    )
    style_block = dialog_match.group(1)
    bad = re.search(r"outline\s*:\s*['\"]none['\"]", style_block)
    assert bad is None, (
        f"P2-MODAL-OUTLINE-A11Y regresión: motion.div del modal aún tiene "
        f"`outline: 'none'` inline ({bad.group(0)!r}). Esto override-ea la "
        f"regla `.mealfit-modal-content:focus-visible` y rompe a11y. "
        f"Mover el outline a CSS (index.css) si necesitas ajustarlo."
    )


def test_index_css_focus_visible_rule_present():
    """index.css debe tener una regla `.mealfit-modal-content:focus-visible`
    con `outline` visible (no `outline: none` ni similar)."""
    src = _read(_INDEX_CSS)
    # Buscar la regla específica.
    rule_match = re.search(
        r"\.mealfit-modal-content:focus-visible\s*\{(.+?)\}",
        src,
        re.DOTALL,
    )
    assert rule_match is not None, (
        "Falta regla `.mealfit-modal-content:focus-visible { ... }` en index.css."
    )
    body = rule_match.group(1)
    # Debe tener `outline:` con valor != 'none'/'0'.
    outline_match = re.search(r"outline\s*:\s*([^;]+);", body)
    assert outline_match is not None, (
        "La regla `:focus-visible` debe declarar `outline: <visible>`. "
        "Sin outline visible, la regla es no-op."
    )
    outline_value = outline_match.group(1).strip().lower()
    assert "none" not in outline_value and outline_value not in ("0", "0px"), (
        f"`outline` del :focus-visible es `{outline_value!r}` — debe ser visible. "
        f"Sin esto la fix no resuelve el a11y gap."
    )
