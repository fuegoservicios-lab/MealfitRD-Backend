# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-123 · 2026-09-19] La foto enviada no cambia de tamaño ni le quita el final al chat.

El dueño: «le mandé la foto y le dije que es mi desayuno de hoy… estuvo bien por unos segundos pero luego la imagen
cambió de tamaño y volvió a su tamaño normal, pero el scroll se había ido hacia arriba en ese corto lapso».

LA CADENA (leída en el código, no supuesta):
  1. Al terminar la subida (~7 s de análisis) `setMessages` cambia la foto local por la del servidor, y el `id` del
     adjunto pasa del local al `attachment_id` → era la `key` de React → la foto se REMONTABA vacía.
  2. El alto de la caja lo ponía la imagen (hasta 260 px; sin imagen, 96): la burbuja bajaba 164 px y volvía.
  3. Al encoger, el navegador recorta `scrollTop`. Los eventos de scroll se despachan al INICIO del fotograma
     siguiente; si para entonces la foto ya volvió a crecer, la distancia al fondo es de 164 px > 120 y el manejador
     declaraba modo LIBRE. El observador de tamaño corre después en ese mismo fotograma: en modo libre ya no re-ancla.

SE CORTA EN CUATRO SITIOS: la caja de la foto tiene tamaño propio (la imagen, en absoluto, no opina); `clientKey`
mantiene la clave entre la versión local y la del servidor; `ChatImage` releva la foto sin hueco (precarga fuera del
DOM); y `utils/chatScrollIntent.js`: lejos del fondo SIN gesto del usuario y pegado a un cambio de alto = layout →
re-ancla. Con gesto (dedo, rueda, teclado, inercia de 3 s) sigue siendo «el usuario subió».

Medido en el arnés: caja 280×228 con foto y sin foto; en el chat real (`agente.html`) un salto de 200 px sin gesto tras
un cambio de alto vuelve a 0, y con rueda se respeta. Contrato fino: `frontend/src/__tests__/lote123.test.jsx`."""
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
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_la_caja_de_la_foto_tiene_tamano_propio():
    css = _front("src/components/agent/MessageBubble.css")
    i = css.index(".message-media-button {")
    caja = css[i:css.index("}", i)]
    assert "aspect-ratio: 16 / 13;" in caja and "position: relative;" in caja
    j = css.index(".message-media-button > img,")
    img = css[j:css.index("}", j)]
    assert "position: absolute;" in img and "inset: 0;" in img, "la imagen no puede opinar sobre el alto de su caja"
    assert "max-height" not in img, "si el alto vuelve a depender de la imagen, vuelve el vaivén de 260→96→260"


def test_la_foto_no_se_remonta_al_llegar_la_del_servidor():
    chat = _front("src/pages/AgentPage.jsx")
    assert chat.count("clientKey: item.id,") == 2, "la burbuja local y la remota comparten clave"
    assert "fullUrl: item.image_url || item.url || item.thumbDataUrl," in chat, "el visor no pierde la versión completa"
    assert "const key = attachment.clientKey || attachment.id" in _front("src/components/agent/MessageBubble.jsx")
    img = _front("src/components/agent/ChatImage.jsx")
    assert "const pre = new Image();" in img and "const visible = pintado || src;" in img


def test_un_salto_de_layout_no_es_el_usuario_subiendo():
    regla = _front("src/utils/chatScrollIntent.js")
    assert "export function decidirAlAlejarseDelFondo(" in regla
    assert "return layoutReciente ? 'fijar' : 'libre';" in regla
    chat = _front("src/pages/AgentPage.jsx")
    assert "if (alejarse === 'fijar') _pinBottomInstant();" in chat
    assert "if (distanceFromBottom > 120) _setMode('free');" in chat, "la salida a modo libre sigue existiendo"
    assert chat.count("ultimoCambioDeAltoRef.current = Date.now();") >= 2, "lo sellan el observador Y el efecto de mensajes"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 123
