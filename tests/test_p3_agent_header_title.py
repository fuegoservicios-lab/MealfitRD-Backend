"""[P3-AGENT-HEADER-TITLE · 2026-05-19] Test parser-based: el título del
header del chat (`AgentPage.jsx`, dentro de `<span className="agent-header-title">`)
dice `Mealfit V1.0` — NO `MealfitRD` (legacy).

Por qué este test:
    Versioning visible para el usuario en el header del chat. Decisión de
    producto. Si alguien refactoriza el header o copy y revierte
    accidentalmente al string legacy "MealfitRD", este test lo flagea antes
    de mergear. El sidebar logo del DashboardLayout conserva "MealfitRD" con
    su branding gradient — son textos independientes.

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_agent_header_title`
matchea este archivo `test_p3_agent_header_title.py`.

Tooltip-anchor: P3-AGENT-HEADER-TITLE-START | user request 2026-05-19
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


def test_header_title_says_mealfit_v1_0(agent_page_src: str):
    """El `<span className="agent-header-title">` debe contener `Mealfit V1.0`.

    El `V1.0` se envuelve en un `<span>` anidado con su propia font-family
    (versioning visible), así que el texto del header es
    `Mealfit <span ...>V1.0</span>`. El regex tolera markup anidado entre
    `Mealfit` y `V1.0` — lo que importa es que ambos tokens aparezcan, en
    orden, dentro del span del título (cierre `</span>` del nested + el del
    header dejan el texto renderizado como "Mealfit V1.0").
    """
    # Localizar la apertura del span del título y acotar a su contenido.
    open_re = re.compile(r'className\s*=\s*["\']agent-header-title["\'][\s\S]*?>')
    m_open = open_re.search(agent_page_src)
    assert m_open is not None, (
        "P3-AGENT-HEADER-TITLE regresión: `<span className=\"agent-header-title\">` "
        "no encontrado en AgentPage.jsx."
    )
    after_open = agent_page_src[m_open.end():m_open.end() + 600]
    # `Mealfit` … (markup anidado opcional) … `V1.0`, en ese orden.
    pattern = re.compile(r"Mealfit\b[\s\S]*?V1\.0")
    assert pattern.search(after_open), (
        "P3-AGENT-HEADER-TITLE regresión: el `<span className=\"agent-header-title\">` "
        "NO contiene `Mealfit` seguido de `V1.0`. Si refactorizaste el header, "
        "mantén ambos tokens — versioning visible al usuario. Cambiar a otra "
        "versión es OK; sustituir Y actualizar este test en el mismo commit."
    )


def test_legacy_mealfitrd_not_in_header_title(agent_page_src: str):
    """`MealfitRD` (legacy) NO debe aparecer DENTRO del `<span className="agent-header-title">`.
    Strict regex: solo el contenido del span. NO toca menciones de "MealfitRD"
    en otros lugares (placeholder del input, etc.)."""
    # Match el bloque del span y su contenido inmediato.
    span_block_re = re.compile(
        r'className\s*=\s*["\']agent-header-title["\'][\s\S]*?>\s*([^<]+?)\s*<'
    )
    m = span_block_re.search(agent_page_src)
    assert m is not None, (
        "P3-AGENT-HEADER-TITLE regresión: `<span className=\"agent-header-title\">` "
        "no encontrado en AgentPage.jsx."
    )
    content = m.group(1).strip()
    assert content != "MealfitRD", (
        f"P3-AGENT-HEADER-TITLE regresión: el header dice {content!r} pero "
        f"debería decir 'Mealfit V1.0'. Pre-fix decía 'MealfitRD'; el cambio "
        f"se hizo por decisión de producto (versioning visible)."
    )


def test_anchor_present(agent_page_src: str):
    assert "P3-AGENT-HEADER-TITLE" in agent_page_src, (
        "P3-AGENT-HEADER-TITLE regresión: anchor textual perdido en AgentPage.jsx."
    )
