"""[P2-CHAT-ANCHOR-SENT-TOP · 2026-09-04] Al enviar un mensaje al coach, la conversación sube y el
mensaje recién enviado queda ARRIBA del hilo; la respuesta crece debajo (ChatGPT/Claude/Gemini).
Un espaciador al final del hilo reserva el sitio y se encoge con la respuesta. Enfocar la caja de
escribir ya no mueve el hilo en PC (solo en móvil, donde el teclado tapa el cuadro).
Frontend: `Chat.anchor_sent_top.test.js`.
"""
from __future__ import annotations

from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_send_anchors_top_and_focus_does_not_scroll_on_desktop():
    src = (_FRONT / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")
    assert "sentAnchorRef.current = { clientMessageId, scrolled: false };" in src
    assert "if (layoutSentAnchor()) return;" in src
    assert 'className="anchor-spacer"' in src
    assert "onFocus={() => { if (isMobile) setTimeout(scrollToBottom, 300); }}" in src


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P2-CHAT-ANCHOR-SENT-TOP · 2026-09-04" in app
