# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-116 · 2026-09-19] Enviar cierra el teclado virtual y la respuesta del agente queda a la vista.

El dueño: «cuando le mando una foto de un plato o cualquier mensaje, el sistema dura mucho en redirigir hacia la
respuesta del agente». Con el teclado virtual abierto la ventana de lectura del chat mide ~200 px: el mensaje enviado
—y más si lleva foto— la llena entera, y la respuesta nace debajo, fuera de cuadro, hasta que crece lo bastante para
que el chat la persiga. El envío CONSERVABA el teclado (refoco de P3-CHAT-FOCUS-TELEM). Ahora, con teclado virtual,
enviar lo cierra —como ChatGPT y Gemini—; con teclado físico el foco se conserva para seguir escribiendo. Además el
espaciador del ancla puede crecer cuando lo que cambia es la VENTANA (el mensaje enviado vuelve arriba al ganar la
pantalla), y una foto enviada deja siempre el chat en modo «abajo». Contrato fino: `lote116.test.js`."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _agent_page() -> str:
    p = _FRONT / "src" / "pages" / "AgentPage.jsx"
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_enviar_cierra_el_teclado_virtual_y_si_no_se_cierra_sigue_a_la_respuesta():
    """[P1-PLAN-LOTE-130 · 2026-09-19] El dueño REVIRTIÓ el cierre del teclado al enviar («haz lo mismo con el + y
    enviar» = que no lo cierren)… y [P1-PLAN-LOTE-134 · 2026-09-20] lo RESTAURA: «quiero que cuando envíe un mensaje se
    cierre automáticamente el teclado para enfocarnos en el mensaje». Vuelve la forma original de este lote; la rama
    «abajo» del 130 queda para el teclado que NO se cierra (el foco no era de la caja)."""
    ap = _agent_page()
    i = ap.index("const _tecladoVirtual = tecladoAbiertoRef.current || medirTecladoDeVentana(window).abierto;")
    bloque = ap[i:i + 700]
    assert re.search(r"if \(_cierraTeclado\) \{\s*try \{ chatInputRef\.current\?\.blur\(\); \}", bloque)
    assert re.search(r"if \(_hadFocusPreSend && !callModeRef\.current\) \{\s*setTimeout\(", bloque)
    assert re.search(r"\} else if \(_tecladoVirtual && !_cierraTeclado\) \{[\s\S]{0,700}_setMode\('bottom'\);", ap)


def test_el_ancla_recupera_el_alto_de_la_ventana_y_la_foto_mira_al_final():
    ap = _agent_page()
    assert "const ventanaCambio = ventanaCambioRef.current === true;" in ap
    assert "try { _layoutAnchor(); } finally { ventanaCambioRef.current = false; }" in ap
    # la regla «solo encoge» sigue literal para el contenido: lo que cambia es el caso de la ventana
    assert "if (anchor.placed && spacer > spacerPxRef.current) spacer = spacerPxRef.current; // solo encoge" in ap
    i = ap.index("attachments: bubbleAttachments,")
    assert "_setMode('bottom');" in ap[i:i + 700]


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 116
