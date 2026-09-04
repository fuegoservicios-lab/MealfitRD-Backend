"""[P2-CHAT-SCROLL-MODES · 2026-09-04] El scroll del chat es UNA máquina de tres modos (bottom / anchored /
free) con transiciones explícitas, no cinco heurísticas apiladas. Este test ancla la forma del modelo en
`frontend/src/pages/AgentPage.jsx`: si alguien vuelve a añadir un mecanismo paralelo (timers, un segundo
observer, un `scrollToBottom` animado), que falle aquí antes que en el chat del usuario.
"""
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_CR = chr(13)


def _frontend_file(*parts: str) -> str:
    for base in (_HERE.parents[2], _HERE.parents[1].parent):
        p = base.joinpath("frontend", "src", *parts)
        if p.exists():
            return p.read_text(encoding="utf-8").replace(_CR, "")
    pytest.skip("frontend hermano no disponible")


def _agent_page_src() -> str:
    return _frontend_file("pages", "AgentPage.jsx")


def test_a_tres_modos_y_transiciones_explicitas():
    src = _agent_page_src()
    assert "const scrollModeRef = useRef('bottom');" in src
    assert "sentAnchorRef.current = { clientMessageId, placed: false };" in src
    assert "_setMode('anchored');" in src
    assert "if (distanceFromBottom > 120) _setMode('free');" in src
    assert "} else if (distanceFromBottom <= 4) {" in src
    assert "userScrolledUpRef.current = mode === 'free';" in src, "espejo para los call sites que aún leen userScrolledUpRef"


def test_b_pins_automaticos_instantaneos_y_espaciador_en_el_dom():
    src = _agent_page_src()
    assert "try { el.scrollTo({ top: el.scrollHeight, behavior: 'instant' }); } catch { el.scrollTop = el.scrollHeight; }" in src
    assert "if (spacerRef.current) spacerRef.current.style.height = `${px}px`;" in src, "el espaciador va directo al DOM: con estado había un frame de temblor"
    assert "if (anchor.placed && spacer > spacerPxRef.current) spacer = spacerPxRef.current; // solo encoge" in src
    # el ÚNICO scroll animado del sistema: llevar el mensaje enviado arriba
    assert "try { el.scrollTo({ top: rowTop, behavior: 'smooth' }); } catch { el.scrollTop = rowTop; }" in src
    for resto in ("anchorSpacerPx", "stickToBottomRef", "_justLoaded", "[0, 250, 900].forEach"):
        assert resto not in src, f"resto del modelo viejo: {resto}"


def test_c_refrescar_asienta_y_revela_una_sola_vez():
    src = _agent_page_src()
    assert "visibility: threadSettling ? 'hidden' : 'visible'" in src
    assert "settleTimerRef.current = setTimeout(_revealThread, 150);" in src
    assert "settleCapRef.current = setTimeout(_revealThread, 900);" in src
    # el historial se carga dos veces al refrescar: solo se oculta si aún no había nada en pantalla
    assert "if (!(messagesRef.current?.length > 0)) _beginSettle();" in src


def test_d_barras_gemelas_a_la_misma_altura():
    src = _agent_page_src()
    assert "marginTop: 'calc(4.5rem + max(env(safe-area-inset-top), 12px))'" in src
    sidebar = _frontend_file("components", "agent", "SidebarRecientes.jsx")
    assert "height: '2.75rem'," in sidebar
