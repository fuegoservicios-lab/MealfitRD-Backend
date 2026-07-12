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
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))

with open(os.path.join(_ROOT, "frontend", "src", "pages", "AgentPage.jsx"),
          encoding="utf-8") as f:
    _AP = f.read()


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
    """La burbuja con foto siempre tiene imageUrl (fallback al blob del file)."""
    assert re.search(r"bubbleBlobUrl = currentFile\s*\n?\s*\? \(currentPreview \|\| URL\.createObjectURL\(currentFile\)\)", _AP), \
        "sin fallback, previewUrl null ⇒ burbuja fantasma sin imagen"


def test_bubble_migrates_to_data_url_thumb():
    """La imagen de la burbuja migra a dataURL: inmune a revokes/reloads y
    sobrevive el cache localStorage del chat (no hay object storage aún)."""
    assert "const fileToThumbDataUrl = (file" in _AP
    assert "fileToThumbDataUrl(currentFile).then" in _AP
    assert "m.imageUrl === bubbleBlobUrl" in _AP, \
        "el swap debe matchear la burbuja por su blob original"


def test_loading_shows_literal_photo_status():
    assert "startsWith('Analizando tu foto')" in _AP, \
        "durante el análisis de foto se muestra el estado literal, no frases rotativas"
    assert "Analizando tu foto… puede tardar un minuto" in _AP
