"""[P2-CHAT-JUMP-TO-LATEST-DESKTOP · 2026-09-04] El botón «ir al último mensaje» del chat solo tenía
estilo en el bloque móvil (≤1024 px): en escritorio se pintaba como un <button> sin estilo estirado a
todo el ancho («una barra rara con un punto» al scrollear arriba). Ahora vive dentro del cuadro de
escribir (sticky en PC, relative en móvil) con un estilo base fuera de cualquier @media.
Frontend: `Chat.jump_to_latest_desktop.test.js`.
"""
from __future__ import annotations

from pathlib import Path

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src"


def test_jump_button_has_base_style_outside_media_and_lives_in_the_composer():
    src = (_FRONT / "pages" / "AgentPage.jsx").read_text(encoding="utf-8")
    style = src.index("<style>{`")
    rule = src.index(".jump-to-latest {", style)
    assert rule < src.index("@media", style)
    assert "html[data-kb-open] .jump-to-latest" not in src
    i = src.index("const renderInputArea = ")
    assert "{!isCentered && showJumpToLatest && messages.length > 0 && (" in src[i:src.index('className="chat-quick-chips"', i)]


def test_marker_present():
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "P2-CHAT-JUMP-TO-LATEST-DESKTOP · 2026-09-04" in app
