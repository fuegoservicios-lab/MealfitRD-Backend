"""[P3-AGENT-DESKTOP-CLIP · 2026-05-19] Test parser-based: el cálculo del
`--app-height` y del fallback `height` del container de AgentPage en desktop
DEBE restar `7.25rem` (= padding-top 2.5rem + margin-top 2.25rem + padding-
bottom 2.5rem) — NO `4rem` como pre-fix.

Por qué este test:
    El container del AgentPage se renderiza dentro de `DashboardLayout.
    mainContent` (padding: 2.5rem arriba y abajo en desktop) y suma su propio
    `margin-top: 2.25rem`. Pre-fix el cálculo `calc(100dvh - 4rem)` solo
    restaba 64px → el container desbordaba el viewport por ~52px → el
    `input-wrapper` (sticky bottom: 0) quedaba parcialmente fuera del área
    visible y el usuario veía "el chat cortado en la parte inferior".

Cross-link convention (P2-HIST-AUDIT-14): el slug `p3_agent_desktop_clip`
matchea este archivo `test_p3_agent_desktop_clip.py`.

Tooltip-anchor: P3-AGENT-DESKTOP-CLIP-START | user report 2026-05-19
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


def test_app_height_uses_7_25_rem(agent_page_src: str):
    """El `setProperty('--app-height', ...)` debe restar 7.25rem en desktop.
    Pre-fix usaba 4rem (no contabilizaba padding del parent + margin-top)."""
    # Patrón: `setProperty('--app-height', 'calc(100dvh - 7.25rem)')`
    pattern = re.compile(
        r"setProperty\s*\(\s*['\"]--app-height['\"]\s*,\s*['\"]calc\(\s*100dvh\s*-\s*7\.25rem\s*\)['\"]"
    )
    assert pattern.search(agent_page_src), (
        "P3-AGENT-DESKTOP-CLIP regresión: el `setProperty('--app-height', ...)` "
        "NO usa `calc(100dvh - 7.25rem)`. Pre-fix `4rem` causaba desbordamiento "
        "vertical de ~52px en desktop (padding mainContent 2.5×2 + margin-top "
        "AgentPage 2.25 = 7.25rem). Si cambiaste el padding del mainContent o "
        "el margin-top del AgentPage, actualiza el cálculo Y este test."
    )


def test_legacy_4rem_value_removed(agent_page_src: str):
    """El valor legacy `calc(100dvh - 4rem)` NO debe quedar en NINGÚN setProperty
    ni fallback del height del container del AgentPage. Tooltip-anchor para
    detectar cualquier reintroducción accidental."""
    # Strip comentarios (los explicativos del fix mencionan `4rem` como pre-fix
    # — eso es OK, no es código activo).
    no_comments = re.sub(r"//[^\n]*", "", agent_page_src)
    no_comments = re.sub(r"/\*[\s\S]*?\*/", "", no_comments)

    legacy_pattern = re.compile(r"calc\(\s*100dvh\s*-\s*4rem\s*\)")
    matches = legacy_pattern.findall(no_comments)
    assert not matches, (
        f"P3-AGENT-DESKTOP-CLIP regresión: encontrados {len(matches)} usos "
        f"activos de `calc(100dvh - 4rem)` en AgentPage.jsx. Ese valor "
        f"desbordaba el viewport desktop. Usar `calc(100dvh - 7.25rem)` "
        f"o ajustar el cálculo según los paddings + margin actuales."
    )


def test_container_height_fallback_uses_7_25_rem(agent_page_src: str):
    """El style del `agent-container` debe usar el fallback `7.25rem` también,
    para el caso en que el useEffect aún no haya seteado la CSS var (primer
    render en desktop). Match flexible: `var(--app-height, calc(100dvh - 7.25rem))`."""
    pattern = re.compile(
        r"var\s*\(\s*--app-height\s*,\s*calc\(\s*100dvh\s*-\s*7\.25rem\s*\)\s*\)"
    )
    assert pattern.search(agent_page_src), (
        "P3-AGENT-DESKTOP-CLIP regresión: el fallback de `var(--app-height, ...)` "
        "en el style del agent-container no usa `calc(100dvh - 7.25rem)`. Sin "
        "el fallback correcto, el primer paint (antes de que el useEffect setee "
        "la CSS var) puede mostrar el container desbordado."
    )


def test_anchor_present(agent_page_src: str):
    assert "P3-AGENT-DESKTOP-CLIP" in agent_page_src, (
        "P3-AGENT-DESKTOP-CLIP regresión: anchor textual perdido en AgentPage.jsx."
    )
