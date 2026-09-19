# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-115 · 2026-09-19] El chat con el teclado abierto: se ve lo que dijo el agente y la página no se arrastra.

El dueño, con la sonda encendida: «todavía hay errores visuales… quiero que se vea el mensaje del agente cuando tenga
el teclado abierto… si no puede verlo, ¿qué va a responder?». Dos causas, las dos medidas o leídas en el código:
(1) tras enviar, el chat queda ANCLADO (el mensaje del usuario arriba, la respuesta debajo) y `scrollToBottom` sin
`force` no hace nada en ese modo: al abrir el teclado la ventana perdía 403 px por ABAJO, justo donde está la pregunta
del agente. Ahora al abrir se decide una vez (`decidirScrollAlAbrirTeclado`) y el final sigue pegado mientras la
ventana encoge (el ResizeObserver también mira el contenedor). (2) La sonda mostró un toque en la caja seguido de
paneo del visual viewport 19→86 px en 60 ms y vuelta a 0: con el teclado abierto el WebView deja arrastrar la página
entera, y nuestro ajuste lo perseguía con 250 ms de transición. En nativo se bloquea el arrastre que nadie puede
aprovechar y el paneo solo se descuenta en la medición de asiento. Contrato fino: `lote115.test.js`."""
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


def test_al_abrir_el_teclado_se_decide_una_vez_que_se_ve():
    pol = _front("src/utils/chatKeyboardScroll.js")
    assert "if (mode === 'anchored') return 'forzar';" in pol, "el modo anclado tapaba la pregunta del agente"
    assert "if (streaming) return 'nada';" in pol, "con la respuesta en curso manda el ancla"
    assert "return distancia <= clientHeight ? 'forzar' : 'nada';" in pol, "leyendo historial antiguo no se arrastra"
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (abierto && !tecladoAbiertoRef.current) alAbrirTecladoRef.current?.();" in ap
    assert "ro.observe(el);" in ap, "el final sigue pegado mientras la VENTANA encoge, no solo cuando crece el contenido"


def test_el_arrastre_que_panea_la_pagina_se_bloquea_solo_en_nativo_con_teclado():
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (!toque || !tecladoAbiertoRef.current || !isNativeApp()) return;" in ap
    assert "document.addEventListener('touchmove', alMoverToque, { passive: false });" in ap
    assert "resolverInsetNativo({ kb, vvOffsetTop: forzarMedicion ? vv.offsetTop : 0 })" in ap
    pol = _front("src/utils/chatKeyboardScroll.js")
    assert "if (Math.abs(dx) > Math.abs(dy)) return 'ceder';" in pol, "la fila de sugerencias se desplaza en horizontal"
    assert "if (enCampoDeTexto && msDesdeElToque > TOQUE_LENTO_MS) return 'ceder';" in pol, "mover el cursor no se bloquea"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 115
