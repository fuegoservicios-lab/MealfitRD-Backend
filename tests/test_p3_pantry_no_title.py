"""[P3-PANTRY-NO-TITLE · 2026-05-19] Eliminación del título "Nevera" +
Snowflake icon del header de la página Nevera.

**Pedido del usuario (2026-05-19):**
    "elimina el texto que dice nevera y el icono"

**Por qué eliminar:**
    La sidebar y la BottomTabBar ya muestran "Nevera" como pestaña activa.
    Repetir "Nevera" + Snowflake en el header del propio apartado era
    redundancia visual sin payoff informativo. El brand label "FRIO MAX"
    arriba del centro del header mantiene la identidad de electrodoméstico
    sin duplicar el nombre de la sección.

**Cambios:**
    - JSX: eliminado `<div className="nevera-title-row">` que envolvía
      el icono Snowflake y `<h1 className="nevera-title">Nevera</h1>`.
    - CSS desktop: eliminado bloque `@media (min-width: 641px) {
      .nevera-title-row { display: none; } }` (pre-fix lo escondía en
      desktop porque la sidebar ya lo mostraba; ahora innecesario).
    - CSS mobile: eliminadas reglas `.nevera-title { font-size: 1.7rem }`
      y `.nevera-snowflake-icon { width: 24px; height: 24px; }`.
    - CSS extra-small: eliminada regla `.nevera-title { font-size: 1.5rem }`.
    - CSS base: eliminadas reglas de `.nevera-title-row`, `.nevera-title`,
      `.nevera-snowflake-icon` y `@keyframes nevera-frost-rotate`.

Por qué parser-based:
    Pure delete — validamos que las clases CSS muertas se fueron y el
    JSX no las renderiza. Si vuelve algún de los tres elementos, fail.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PANTRY_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "Pantry.jsx"


def _read_pantry() -> str:
    assert _PANTRY_JSX.exists(), f"Pantry.jsx no encontrado en {_PANTRY_JSX}"
    return _PANTRY_JSX.read_text(encoding="utf-8")


def test_p3_pantry_no_title_marker_present():
    """Marker textual presente — cross-link con `test_p2_hist_audit_14`."""
    src = _read_pantry()
    assert "P3-PANTRY-NO-TITLE" in src, (
        "Marker `P3-PANTRY-NO-TITLE` no encontrado en Pantry.jsx."
    )


def test_jsx_does_not_render_title_row():
    """`<div className="nevera-title-row">` NO debe renderizarse en el JSX.
    Pre-fix envolvía el Snowflake + h1 "Nevera"; el usuario pidió eliminar
    ambos.
    """
    src = _read_pantry()
    assert 'className="nevera-title-row"' not in src, (
        "`<div className='nevera-title-row'>` se encontró en el JSX. "
        "Debe estar eliminado. Si tu cambio lo trae de vuelta, también "
        "debes traer de vuelta el CSS de `.nevera-title-row`, `.nevera-title`, "
        "y `.nevera-snowflake-icon` Y bumpear el marker `_LAST_KNOWN_PFIX`."
    )


def test_jsx_does_not_render_h1_nevera_title():
    """`<h1 className="nevera-title">Nevera</h1>` NO debe estar en el JSX."""
    src = _read_pantry()
    assert 'className="nevera-title"' not in src, (
        "`className='nevera-title'` (usado por el h1 con texto 'Nevera') "
        "se encontró en el JSX. Debe estar eliminado."
    )


def test_jsx_does_not_render_snowflake_icon_in_header():
    """El componente `<Snowflake>` con `className='nevera-snowflake-icon'`
    NO debe renderizarse. (El otro Snowflake en `.nevera-empty-fridge`
    sí puede quedar — es del empty state, no del header.)
    """
    src = _read_pantry()
    assert 'className="nevera-snowflake-icon"' not in src, (
        "`<Snowflake className='nevera-snowflake-icon'>` se encontró en "
        "el JSX. Debe estar eliminado del header. Si lo necesitas en otra "
        "ubicación, usa una className diferente para no confundir el match."
    )


def test_css_dead_rules_cleaned_up():
    """Las reglas CSS que solo aplicaban al título eliminado deben estar
    fuera del archivo. Mantenerlas crearía dead code que confunde futuros
    cambios.
    """
    src = _read_pantry()

    # Las 3 reglas base NO deben existir como declaraciones activas
    # (admitimos su mención en comentarios explicativos).
    for selector in [".nevera-title-row", ".nevera-title", ".nevera-snowflake-icon"]:
        # Match `<selector> {` indica regla CSS activa
        pattern = re.escape(selector) + r"\s*\{"
        matches = re.findall(pattern, src)
        assert len(matches) == 0, (
            f"Regla CSS activa `{selector} {{ ... }}` aún existe ({len(matches)} match). "
            f"Debe estar eliminada — el JSX ya no usa esta clase. "
            f"Si necesitas mencionarla, hazlo solo en comentarios explicativos."
        )

    # La animation tampoco debe seguir declarada
    assert "@keyframes nevera-frost-rotate" not in src or "/*" in (
        # Permitimos mención en comentario, no como declaración real
        re.search(r"@keyframes nevera-frost-rotate", src).string[
            max(0, re.search(r"@keyframes nevera-frost-rotate", src).start() - 200):
            re.search(r"@keyframes nevera-frost-rotate", src).end()
        ] if re.search(r"@keyframes nevera-frost-rotate", src) else ""
    ), (
        "`@keyframes nevera-frost-rotate` aún declarado. Era la animación de "
        "rotación del Snowflake — al quitar el icono, esta animación es muerta."
    )
