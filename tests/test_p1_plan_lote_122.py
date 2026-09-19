# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-122 · 2026-09-19] El visor de fotos del chat: la X en su franja, fondo negro y cerrar deslizando.

El dueño, con una foto abierta: «mejora cuando entro a la foto, ya que por ejemplo la X está fuera del rango».

  · La X se montaba sobre la esquina de la foto. El relleno superior del visor era `max(3.5rem, zona segura)` — el
    MAYOR de los dos, no la suma—: la foto empezaba a 56 px y la X (48 px desde los 12) acababa a 60. En un iPhone con
    muesca (47 px) el solape era de 39 px. Ahora es zona segura + 4rem y la X (icono, 44 px) vive entera en su franja.
  · El fondo era azul translúcido: una foto con bandas negras (las capturas de pantalla del dueño, 739×1600) se veía
    como una caja negra de esquinas redondas flotando. Sobre negro opaco las bandas desaparecen y queda la foto.
  · En el teléfono la foto va de borde a borde, sin esquinas redondeadas.
  · Deslizar hacia ABAJO cierra, con la foto siguiendo al dedo; las decisiones son puras
    (`frontend/src/utils/imageViewerGesture.js`). Con la página ampliada no hay gestos.

Medido en el arnés (`visor.html`) a 393×852: foto 64–832, X 10–54, sin solape. Contrato fino:
`frontend/src/__tests__/lote122.test.jsx`."""
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


def _regla(css: str, selector: str) -> str:
    i = css.index(selector + " {")
    return css[i:css.index("}", i)]


def test_la_x_tiene_su_franja_y_no_pisa_la_foto():
    css = _front("src/components/agent/MessageBubble.css")
    visor = _regla(css, ".message-image-viewer")
    assert "calc(env(safe-area-inset-top, 0px) + 4rem)" in visor, "la zona segura se SUMA: con max() la X pisa la foto"
    assert "max(3.5rem, env(safe-area-inset-top" not in visor
    x = _regla(css, ".message-image-viewer-close")
    assert "top: calc(env(safe-area-inset-top, 0px) + 0.6rem);" in x
    assert "height: 44px;" in x, "0.6rem + 44 px = 53,6 < 64: la X cabe entera en su franja"


def test_fondo_negro_y_cierre_deslizando():
    css = _front("src/components/agent/MessageBubble.css")
    assert "background: rgb(0 0 0 / var(--visor-opacidad, 1));" in _regla(css, ".message-image-viewer")
    assert "transform: translateY(var(--visor-baja, 0px));" in _regla(css, ".message-image-viewer > img")
    gesto = _front("src/utils/imageViewerGesture.js")
    assert "export function decidirGestoVisor(" in gesto
    assert "if (zoom > 1.01) return 'nada';" in gesto, "con la página ampliada el dedo mueve la foto, no pide nada"
    burbuja = _front("src/components/agent/MessageBubble.jsx")
    assert "onTouchMove={onViewerTouchMove}" in burbuja
    assert "if (accion === 'cerrar') setViewerIndex(null);" in burbuja


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 122
