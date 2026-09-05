"""[P2-CHAT-ERROR-MINIMAL · 2026-09-04] La burbuja de error del coach era una caja roja con borde y un
botón bordeado de 44 px («muy feo, hazlo más minimalista»). Ahora: icono pequeño + texto apagado sin
fondo ni borde, «Reintentar» como enlace, y copias cortas sin emojis. De paso, el turno del coach en
curso marca `mealfit_chat_turn_inflight` para que la auto-aplicación del service worker no recargue
a mitad de una respuesta (fue lo que la cortó a las 20:45 UTC). Frontend: `Chat.error_bubble_minimal.test.js`.
"""
from __future__ import annotations

from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_error_bubble_is_minimal_and_turn_marks_inflight():
    mb = (_FRONT / "components" / "agent" / "MessageBubble.jsx").read_text(encoding="utf-8")
    assert "isErrorBubble ? 'var(--danger-bg)'" not in mb
    assert "borderRadius: 999," in mb  # v3: píldora fantasma, sin subrayado ni triángulo
    ap = (_FRONT / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")
    assert "t('No llegó la respuesta del coach.')" in ap
    assert "safeLocalStorageSet('mealfit_chat_turn_inflight', { startedAt: Date.now() })" in ap


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P2-CHAT-ERROR-MINIMAL · 2026-09-04" in app
