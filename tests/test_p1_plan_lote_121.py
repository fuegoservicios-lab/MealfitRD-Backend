# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-121 · 2026-09-19] En el teléfono la cabecera del chat es TRANSPARENTE y no dice «Bioboros 1».

El dueño: «quiero que el encabezado sea transparente y no diga Bioboros 1 para así poder tener más espacio para que el
usuario pueda ver el chat y así tenga más comodidad visual». La cabecera era una franja opaca de 4.5rem + zona segura
con su raya, y el viewport desplazable empezaba DEBAJO de ella (P1-CHAT-HEADER-CLEARANCE). Ahora:

  · No hay franja: el fondo es un degradado opaco SOLO sobre la barra de estado (la hora no se lee sobre texto que se
    mueve) y transparente a la altura de los botones. Sin raya. Sin desenfoque (P1-KB-SIN-DESENFOQUE ya pagó eso).
  · La franja no captura toques y sus hijos sí: el hueco entre los dos botones es chat.
  · Los dos botones (conversaciones y menú) flotan como fichas con fondo propio. El rótulo se oculta en el teléfono y
    SIGUE en el código: en escritorio se pinta, y `test_p3_agent_header_title.py` lo sigue exigiendo con `<Wordmark />`.
  · El scroller empieza arriba del todo y reserva como RELLENO la altura de los botones. El anclaje del mensaje
    enviado (`_layoutAnchor`) ya descontaba el relleno: la burbuja anclada aterriza debajo de los botones.

Visto en el arnés (`C:/tmp/h-contador/agente.html`, nuevo: monta el chat REAL con una conversación de mentira) a
375×812, claro y oscuro: scroller 0–651 (antes empezaba en 96), cabecera 0–81 sin fondo.
Contrato fino: `frontend/src/__tests__/lote121.test.js`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _movil() -> str:
    p = _FRONT / "src/pages/AgentPage.jsx"
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    src = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    return src[src.index("@media (max-width: 1024px) {"):src.index("@media (min-width: 1025px) {")]


def _regla(bloque: str, selector: str) -> str:
    i = bloque.index(selector + " {")
    return bloque[i:bloque.index("}", i)]


def test_la_cabecera_no_es_una_franja():
    r = _regla(_movil(), "                    .mobile-chat-header")
    assert "background: linear-gradient(to bottom," in r
    assert "var(--bg-card) max(env(safe-area-inset-top), 24px)," in r, "opaco SOLO sobre la barra de estado"
    assert "transparent 100%) !important;" in r
    assert "border-bottom: 0 !important;" in r
    assert "backdrop-filter: none !important;" in r, "sin desenfoque: recompuesto en movimiento es el glitch de iOS"
    assert "pointer-events: none;" in r, "el hueco entre los botones es chat, no cabecera"


def test_sin_rotulo_y_con_los_botones_como_fichas():
    movil = _movil()
    assert ".agent-header-title { display: none !important; }" in movil
    assert ".mobile-chat-header > * { pointer-events: auto; }" in movil
    assert "border: 1px solid var(--border) !important;" in _regla(movil, "                    .chat-header-btn")


def test_el_chat_empieza_arriba_del_todo():
    r = _regla(_movil(), "                    .messages-container")
    assert "margin-top: 0 !important;" in r
    assert "padding-top: calc(3.7rem + max(env(safe-area-inset-top), 24px)) !important;" in r


def test_el_css_vive_en_un_template_literal():
    assert "`" not in _movil(), "un acento grave en un comentario del CSS rompe el template literal (pasó en este lote)"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 121
