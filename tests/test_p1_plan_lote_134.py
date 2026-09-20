# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-134 · 2026-09-20] ENVIAR cierra el teclado — y solo enviar.

El dueño, con una captura del chat nativo y el teclado tapando media pantalla mientras el coach piensa: «esto está
perfecto pero quiero que cuando envíe un mensaje se cierre automáticamente el teclado para enfocarnos en el mensaje, eso
mejoraría la experiencia de usuario».

Historia, para que nadie lo «arregle» de vuelta: el lote 116 lo cerraba (queja del dueño: la respuesta nacía fuera de
cuadro), el lote 130 lo revirtió a petición del propio dueño («haz lo mismo con el + y enviar» = que no cierren el
teclado) y el 134 lo restaura A SABIENDAS, solo para enviar. El «+», el micrófono y la X de la foto siguen sin llevarse
el teclado (lotes 111, 127, 129, 130). El foco se suelta dentro de `handleSend`, cuando el texto ya se capturó — no en el
gesto del botón, que sigue actuando en `pointerdown` sin perder el foco (P0-CHAT-IOS-APP-BOTONES-CON-TECLADO).

Contrato fino: `frontend/src/__tests__/lote134.test.js`. Cero cambios de backend."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _agent_page() -> str:
    p = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_enviar_cierra_el_teclado_virtual_solo_si_el_foco_era_de_la_caja():
    ap = _agent_page()
    i = ap.index("const _tecladoVirtual = tecladoAbiertoRef.current || medirTecladoDeVentana(window).abierto;")
    bloque = ap[i:i + 700]
    assert "const _cierraTeclado = Boolean(_hadFocusPreSend && _tecladoVirtual);" in bloque
    assert re.search(r"if \(_cierraTeclado\) \{\s*try \{ chatInputRef\.current\?\.blur\(\); \}", bloque)
    # con teclado físico (escritorio, iPad) el foco se conserva para seguir escribiendo: la rama de siempre
    assert re.search(r"\} else if \(_hadFocusPreSend && !callModeRef\.current\) \{\s*setTimeout\(", bloque)


def test_el_foco_se_suelta_despues_de_capturar_el_texto():
    ap = _agent_page()
    i = ap.index("const _tecladoVirtual = tecladoAbiertoRef.current || medirTecladoDeVentana(window).abierto;")
    k = ap.index("const userMsg = textToSend.trim();")
    assert k < ap.index("setInput('');", k) < i


def test_con_el_teclado_cerrado_el_mensaje_se_ancla_arriba():
    ap = _agent_page()
    assert re.search(
        r"\} else if \(_tecladoVirtual && !_cierraTeclado\) \{[\s\S]{0,700}_setMode\('bottom'\);\s*\} else \{[\s\S]{0,300}_setMode\('anchored'\);",
        ap,
    )
    # el ancla recupera el alto que gana la ventana al cerrarse el teclado (lote 116)
    assert "} else if (ventanaCambio && spacerPxRef.current > 0) {" in ap


def test_solo_enviar_el_mas_sigue_sin_llevarse_el_teclado():
    ap = _agent_page()
    assert "onTouchEnd={handleComposerTouchEnd('attachment')}" in ap
    assert ap.count("onMouseDown={keepComposerFocus}") == 2
    assert "if (abierto && !isNativeApp()) chatInputRef.current?.blur();" in ap


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 134
