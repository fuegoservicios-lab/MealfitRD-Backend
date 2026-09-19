# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-117 · 2026-09-19] Las fotos del chat se ven en el teléfono y el visor las enseña enteras y nítidas.

El dueño: «las imágenes aparecen que no están disponibles en móviles… quiero que se pueda abrir la imagen y se vea
nítida, en PC también ya que se ve rara y mal». Tres causas, las tres comprobadas:
(1) `build_chat_attachment_url` entrega una ruta RELATIVA protegida por la sesión. Un `<img>` no manda cabeceras: en la
app nativa la ruta resuelve contra `capacitor://localhost` y, además, la app se autentica con `Authorization` /
`X-MF-Session`, no con cookie (la PWA de iOS tampoco la conserva). 404/403 → «Imagen no disponible». Ahora la foto se
trae con `fetchWithAuth` y se pinta desde un `blob:` (directo en nativo; en la web, solo si el `<img>` falla).
(2) El visor era un grid con pistas `auto`: `max-height: 100%` no tenía contra qué resolverse y una foto alta se salía
de la pantalla. (3) Recién enviada, el visor ampliaba la miniatura local de 360 px.
Dato forense: las tres fotos que mandó medían 739×1600 con el octavo superior y el inferior en negro — eran CAPTURAS
de pantalla de la fototeca; esas franjas negras vienen en la imagen, no las pone la app.
Contrato fino: `frontend/src/__tests__/lote117.test.jsx`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_la_premisa_sigue_en_pie_el_backend_entrega_una_ruta_relativa_con_sesion():
    """Si un día la URL pasa a ser absoluta y pública, la carga autenticada sobra: revisa el lote, no este test."""
    db_chat = (_BACKEND / "db_chat.py").read_text(encoding="utf-8")
    assert 'base = f"/api/chat/attachments/{attachment_id}"' in db_chat
    chat = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")
    assert "if not signed and not owned:" in chat


def test_la_foto_se_trae_autenticada_en_nativo_y_como_reintento_en_la_web():
    hook = _front("src/hooks/useChatImageSrc.js")
    assert "const autenticada = delBackend && (isNativeApp() || falloDirecto === clave);" in hook
    assert "const r = await fetchWithAuth(url);" in hook
    assert "export const claveDeImagen = (url) => String(url || '').split('?')[0];" in hook, "la firma caduca; la foto no"
    burbuja = _front("src/components/agent/MessageBubble.jsx")
    assert burbuja.count("<ChatImage") == 2, "la miniatura de la burbuja Y el visor"
    assert "<img" not in burbuja.split("message-media-grid")[1].split("message-image-viewer-close")[0]


def test_el_visor_encaja_la_foto_entera_y_abre_la_version_completa():
    css = _front("src/components/agent/MessageBubble.css")
    i = css.index(".message-image-viewer {")
    bloque = css[i:css.index("}", i)]
    assert "grid-template-rows: minmax(0, 1fr);" in bloque and "grid-template-columns: minmax(0, 1fr);" in bloque
    assert "media[viewerIndex]?.fullUrl || media[viewerIndex]?.url" in _front("src/components/agent/MessageBubble.jsx")
    assert "fullUrl: item.url || item.image_url || item.previewUrl || item.thumbDataUrl," in _front("src/pages/AgentPage.jsx")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 117
