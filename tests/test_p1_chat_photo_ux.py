"""[P1-CHAT-PHOTO-UX · 2026-07-12] UX del envío de fotos en el chat del Agente.

Vivo (owner, primer uso del chat con foto): (1) su burbuja con la imagen
DESAPARECÍA del hilo dejando solo el saludo, (2) el loading mostraba frases
rotativas irrelevantes ("Alineando tu genética…") durante el análisis de la
foto, (3) el saludo automático seguía visible tras enviar.

Root cause de (1): `_setWelcomeIfAbsent` — su fall-through devolvía
[{welcome}] SIEMPRE que el estado no fuera exactamente [welcome-fresco], o
sea que con [welcome, msg-del-user] REEMPLAZABA la conversación entera. La
ventana quedó expuesta porque gemma tarda 30-90s (la visión cloud previa
~3s la hacía invisible) y el mensaje aún no está persistido server-side
(eso ocurre recién en /stream, después del análisis).
tooltip-anchor: P1-CHAT-PHOTO-UX
"""

import pytest
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

@pytest.fixture(scope="module", autouse=True)
def _load_frontend_sibling_sources(frontend_repo_path):
    # La fixture compartida salta el módulo antes de cualquier I/O si falta el hermano.
    _ = frontend_repo_path
    global _AP, _PROCESSING, _ATTACHMENTS, f
    with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
              encoding="utf-8") as f:
        _AP = f.read()
    with open(os.path.join(_ROOT, "frontend", "src", "utils", "chatImageProcessing.js"),
              encoding="utf-8") as f:
        _PROCESSING = f.read()
    with open(os.path.join(_ROOT, "frontend", "src", "hooks", "useChatAttachments.js"),
              encoding="utf-8") as f:
        _ATTACHMENTS = f.read()



def test_welcome_never_clobbers_real_conversation():
    """El guard del wipe: con CUALQUIER mensaje real, _setWelcomeIfAbsent no toca."""
    i = _AP.find("_setWelcomeIfAbsent = useCallback")
    assert i != -1, "el helper del welcome desapareció"
    win = _AP[i:i + 1800]
    assert "prev.some(m => !m.isWelcome)" in win, \
        "sin este guard, el fall-through borra la conversación del usuario"
    # El guard debe RETORNAR prev (no regenerar) cuando hay conversación real.
    j = win.find("prev.some(m => !m.isWelcome)")
    assert "return prev" in win[j:j + 200]


def test_send_removes_welcome_immediately_no_late_shift():
    assert ".filter(m => !m.isWelcome)" in _AP, \
        "el saludo debe retirarse AL ENVIAR (pedido del owner), no tras el stream"
    assert "newMessages.shift()" not in _AP, \
        "el shift tardío mutaba in-place el array que ya era state"


def test_image_bubble_never_ghost():
    """La burbuja toma la mejor URL disponible y descarta entradas fantasma."""
    assert "url: item.url || item.image_url || item.thumbDataUrl || item.previewUrl" in _AP
    assert ").filter((item) => item.url);" in _AP
    assert "previewUrl = URL.createObjectURL(file)" in _ATTACHMENTS


def test_bubble_migrates_to_data_url_thumb():
    """El pipeline preparado produce thumbnail durable antes de renderizar."""
    assert "const thumbDataUrl = thumbCanvas.toDataURL" in _PROCESSING
    assert "return { file: uploadFile, thumbDataUrl, width, height };" in _PROCESSING
    assert "prepareChatImage(job.sourceFile" in _ATTACHMENTS


def test_loading_shows_literal_photo_status():
    assert "setStreamingStatus(t('Analizando tus fotos… puede tardar un minuto'))" in _AP, \
        "durante el análisis múltiple se muestra el estado literal, no frases rotativas"
