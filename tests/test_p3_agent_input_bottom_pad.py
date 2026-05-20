"""[P3-AGENT-INPUT-BOTTOM-PAD · 2026-05-19] Test parser-based: el padding
del `input-wrapper` en la variante NO centrada (chat con conversación
activa) tiene 1.75rem de padding-bottom (era 1.25rem pre-fix).

Por qué este test:
    User reportó "donde se escribe está muy pegado de la parte de abajo"
    tras el fix P3-AGENT-DESKTOP-CLIP del mismo día. Esa primera iteración
    cerró el desborde de viewport pero el padding INTERNO del input-wrapper
    seguía dejando solo 20px (1.25rem) entre el input y el borde inferior
    del card — visualmente pegado. Subido a 28px (1.75rem) para breathing
    simétrico con el top de la conversación arriba.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_agent_input_bottom_pad`
matchea este archivo `test_p3_agent_input_bottom_pad.py`.

Tooltip-anchor: P3-AGENT-INPUT-BOTTOM-PAD-START | user feedback 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = (
    _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"
)


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


def test_input_wrapper_non_centered_padding(agent_page_src: str):
    """El padding del `input-wrapper` en la variante `isCentered=false` debe
    ser `'1.25rem 2rem 1.75rem 2rem'` (top/right/bottom/left). Sin breathing
    inferior suficiente, el input se ve pegado al borde del card."""
    # Match el ternary completo:
    # `padding: isCentered ? '1.5rem 1.25rem 2.5rem 1.25rem' : '<este>'`
    pattern = re.compile(
        r"padding\s*:\s*isCentered\s*\?\s*['\"]1\.5rem\s+1\.25rem\s+2\.5rem\s+1\.25rem['\"]\s*:\s*['\"]1\.25rem\s+2rem\s+1\.75rem\s+2rem['\"]"
    )
    assert pattern.search(agent_page_src), (
        "P3-AGENT-INPUT-BOTTOM-PAD regresión: padding del `input-wrapper` "
        "(variante NO centrada) no es `'1.25rem 2rem 1.75rem 2rem'`. "
        "Pre-fix `'1.25rem 2rem'` dejaba solo 20px abajo y se veía pegado. "
        "Si quieres ajustar, sube/baja el valor del 3er componente (bottom) "
        "pero mantén el 4-component explícito para que el contrato sea "
        "explícito vs el shorthand 2-component."
    )


def test_legacy_2_component_padding_removed(agent_page_src: str):
    """El shorthand legacy `'1.25rem 2rem'` (que se expandía a top/bottom
    1.25rem ambos) NO debe quedar en el style del input-wrapper. Tooltip
    contra regresión accidental al refactor."""
    # Solo strip comentarios de bloque/línea para no flagear menciones en
    # bloques narrativos del commit history embebido.
    no_comments = re.sub(r"//[^\n]*", "", agent_page_src)
    no_comments = re.sub(r"/\*[\s\S]*?\*/", "", no_comments)

    # Buscar `'1.25rem 2rem'` aislado como literal padding de 2 componentes.
    # Excluye `'1.25rem 2rem 1.75rem 2rem'` (el correcto) — el regex es
    # estricto: cierre del string inmediato tras `2rem`.
    legacy_pattern = re.compile(r"['\"]1\.25rem\s+2rem['\"]")
    matches = legacy_pattern.findall(no_comments)
    assert not matches, (
        f"P3-AGENT-INPUT-BOTTOM-PAD regresión: encontrados {len(matches)} usos "
        f"activos del shorthand legacy `'1.25rem 2rem'` (2-component) en "
        f"AgentPage.jsx. Ese valor expandía a `1.25rem` top y bottom (20px), "
        f"el bottom quedaba pegado al borde del card. Usar 4-component "
        f"`'1.25rem 2rem 1.75rem 2rem'` para que el contrato sea explícito."
    )


def test_anchor_present(agent_page_src: str):
    assert "P3-AGENT-INPUT-BOTTOM-PAD" in agent_page_src, (
        "P3-AGENT-INPUT-BOTTOM-PAD regresión: anchor textual perdido en "
        "AgentPage.jsx."
    )
