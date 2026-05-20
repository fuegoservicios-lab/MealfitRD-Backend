"""[P3-AGENT-INPUT-CENTER · 2026-05-19] Test parser-based: el
`input-wrapper` del Agente en DESKTOP queda `bottom: 1.25rem` (no
`bottom: 0`) con padding top/bottom simétrico — el input box ya no
sobrepasa el border-radius inferior del card y queda centralizado
dentro de su wrapper.

Por qué este test:
    Iteración visual del mismo día sobre P3-AGENT-INPUT-BOTTOM-PAD y
    P3-AGENT-DESKTOP-CLIP. El usuario reportó (screenshot): "la parte
    inferior de abajo del chat sobrepasa el borde y debería estar un
    poco más arriba centralizada". Pre-fix:
        - `bottom: 0` pegaba el wrapper al fondo exacto del scroll
          container — combinado con `borderBottomRadius: 1.5rem` del
          card en desktop, el wrapper visualmente cubría la zona del
          radius y daba la sensación "input desbordado del card".
        - Padding desktop NO centered: `'1.5rem 3rem 2.5rem 3rem'`
          asimétrico — 24px top vs 40px bottom no centra el input box
          dentro del wrapper.
    Post-fix:
        - `bottom: isMobile ? 0 : '1.25rem'` — desktop tiene 20px de
          respiración entre input y borde inferior del card.
        - Padding desktop balanceado a `'1.5rem 3rem 1.5rem 3rem'` —
          input centrado vertical dentro del wrapper.
        - Mobile intacto (sticky bottom 0 cooperativo con
          visualViewport handler para iOS keyboard lift).

Cross-link convention (P2-HIST-AUDIT-14): el slug
`p3_agent_input_center` matchea este archivo
`test_p3_agent_input_center.py`.

Tooltip-anchor: P3-AGENT-INPUT-CENTER | user screenshot feedback 2026-05-19
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PAGE_JSX = _REPO_ROOT / "frontend" / "src" / "pages" / "AgentPage.jsx"


@pytest.fixture(scope="module")
def agent_page_src() -> str:
    return _AGENT_PAGE_JSX.read_text(encoding="utf-8")


def test_anchor_present(agent_page_src: str):
    assert "P3-AGENT-INPUT-CENTER" in agent_page_src, (
        "P3-AGENT-INPUT-CENTER regresión: anchor textual perdido en "
        "AgentPage.jsx."
    )


def test_bottom_desktop_lifted_above_card_border(agent_page_src: str):
    """El `bottom` del wrapper debe ser `isCentered ? 0 : (isMobile ? 0 : '1.25rem')`.
    Sin el lift desktop, el input toca el border-radius inferior del card
    y produce el síntoma reportado por el usuario."""
    pattern = re.compile(
        r"bottom\s*:\s*isCentered\s*\?\s*0\s*:\s*\(\s*isMobile\s*\?\s*0\s*:\s*['\"]1\.25rem['\"]\s*\)",
    )
    assert pattern.search(agent_page_src), (
        "P3-AGENT-INPUT-CENTER regresión: el `bottom` del input-wrapper ya no "
        "lift a `1.25rem` en desktop. Sin el lift, el wrapper toca el "
        "border-radius del card (1.5rem) y se ve desbordado."
    )


def test_legacy_bottom_zero_unconditional_not_present(agent_page_src: str):
    """El patrón legacy `bottom: 0,` sin condicional NO debe seguir en el
    input-wrapper. Buscamos su declaración EXACTA en la región del wrapper."""
    # Localizar la región del wrapper renderInputArea (signature → fin estilo)
    m = re.search(
        r"const\s+renderInputArea\s*=[\s\S]{0,3000}?zIndex\s*:\s*10",
        agent_page_src,
    )
    assert m, "renderInputArea region no parseable."
    region = m.group(0)
    # En la región, no debe aparecer una sola línea `bottom: 0,` sin condicional
    legacy = re.search(r"^\s*bottom\s*:\s*0\s*,\s*$", region, re.MULTILINE)
    assert legacy is None, (
        "P3-AGENT-INPUT-CENTER regresión: `bottom: 0,` unconditional volvió "
        "a aparecer en renderInputArea. Esto reabre el síntoma del "
        "screenshot del usuario — el wrapper toca el card border."
    )


def test_padding_desktop_no_centered_symmetric(agent_page_src: str):
    """En desktop (isMobile false) NO centered (chat con conversación
    activa), el padding debe ser `'1.5rem 3rem 1.5rem 3rem'` — top y
    bottom simétricos en 24px. Esto centra el input box vertical dentro
    del wrapper en lugar del legacy `'1.5rem 3rem 2.5rem 3rem'`
    asimétrico."""
    pattern = re.compile(
        r"['\"]1\.5rem\s+3rem\s+1\.5rem\s+3rem['\"]",
    )
    assert pattern.search(agent_page_src), (
        "P3-AGENT-INPUT-CENTER regresión: padding desktop NO centered ya "
        "no es `'1.5rem 3rem 1.5rem 3rem'` (top/bottom simétricos). El "
        "input box dejó de estar centrado vertical en su wrapper."
    )


def test_legacy_asymmetric_padding_desktop_removed(agent_page_src: str):
    """El padding legacy `'1.5rem 3rem 2.5rem 3rem'` (asimétrico 24/40)
    NO debe seguir en el código — era la fuente del descentrado vertical."""
    legacy_pattern = re.compile(
        r"['\"]1\.5rem\s+3rem\s+2\.5rem\s+3rem['\"]",
    )
    assert legacy_pattern.search(agent_page_src) is None, (
        "P3-AGENT-INPUT-CENTER regresión: el padding asimétrico legacy "
        "`'1.5rem 3rem 2.5rem 3rem'` reapareció. Reemplazar por "
        "`'1.5rem 3rem 1.5rem 3rem'` (simétrico 24/24)."
    )


def test_mobile_sticky_bottom_zero_preserved(agent_page_src: str):
    """Mobile DEBE preservar `bottom: 0` para sticky — es crítico para
    el cooperative con el visualViewport handler que levanta el wrapper
    con el teclado virtual iOS. El expression debe contener `isMobile ? 0`."""
    # `isMobile ? 0 : '1.25rem'` — verifica que la rama mobile retorna 0
    pattern = re.compile(r"isMobile\s*\?\s*0\s*:\s*['\"]1\.25rem['\"]")
    assert pattern.search(agent_page_src), (
        "P3-AGENT-INPUT-CENTER regresión: la rama mobile del `bottom` ya "
        "no retorna 0. Romperías el cooperative con visualViewport handler "
        "(iOS keyboard lift) — el input quedaría escondido bajo el teclado."
    )


def test_isCentered_branch_preserves_bottom_zero(agent_page_src: str):
    """La variante centered (empty state, sin mensajes) debe seguir con
    `bottom: 0` — el wrapper es absolute dentro del flex container y NO
    necesita lift (el empty state se centra vertical via flex)."""
    # `isCentered ? 0 : (...)` — verifica que centered retorna 0
    pattern = re.compile(r"isCentered\s*\?\s*0\s*:\s*\(")
    assert pattern.search(agent_page_src), (
        "P3-AGENT-INPUT-CENTER regresión: la rama centered del `bottom` "
        "ya no retorna 0. El empty state usa flex para centrar — un lift "
        "extra rompería ese layout."
    )
