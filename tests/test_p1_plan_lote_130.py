# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-130 · 2026-09-19] «+» y ENVIAR no se llevan el teclado.

El dueño, tras confirmar que la X de la foto adjunta ya no lo cierra: «ya funciona, ahora haz lo mismo con el + y enviar».

  · Con el teclado abierto los dos botones ya actuaban en `pointerdown` (P0-CHAT-IOS-APP-BOTONES-CON-TECLADO), pero el foco
    no se va ahí: se va cuando WebKit sintetiza mousedown/click tras el `touchend`. `handleComposerTouchEnd` cancela ese
    `touchend` SOLO si el gesto ya se atendió — sin teclado abierto la acción sigue llegando por el click, y cancelar ahí
    dejaría el botón muerto.
  · ENVIAR hacía además `blur()` a propósito (lote 116, por una queja del propio dueño: la respuesta nacía fuera de cuadro
    con la ventana de ~200 px). El dueño lo revierte; la protección se conserva enviando en modo «abajo» mientras el teclado
    siga en pantalla.

Contrato fino: `frontend/src/__tests__/lote130.test.js`. Cero cambios de backend."""
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


def test_el_touchend_se_cancela_solo_si_el_gesto_ya_se_atendio():
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const handleComposerTouchEnd = (action) => (event) => {")
    cuerpo = ap[k:ap.index("\n    };", k)]
    assert "if (!event.cancelable || pending.action !== action || Date.now() > pending.expiresAt) return;" in cuerpo
    assert cuerpo.index("return;") < cuerpo.index("event.preventDefault();"), "cancelar sin acción previa mata el botón"
    assert "onTouchEnd={handleComposerTouchEnd('attachment')}" in ap
    assert "onTouchEnd={handleComposerTouchEnd('send')}" in ap


def test_enviar_no_cierra_el_teclado_y_sigue_a_la_respuesta():
    ap = _front("src/pages/AgentPage.jsx")
    i = ap.index("const _tecladoVirtual = tecladoAbiertoRef.current || medirTecladoDeVentana(window).abierto;")
    assert ".blur()" not in ap[i:i + 700]
    assert re.search(r"\} else if \(_tecladoVirtual\) \{[\s\S]{0,700}_setMode\('bottom'\);\s*\} else \{[\s\S]{0,300}_setMode\('anchored'\);", ap)


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 130
