# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-143 · 2026-09-20] El teclado del chat nativo vuelve a como estaba en el lote 139.

Los lotes 140–142 probaron a mover el chat DESDE EL BINARIO para igualar a Gemini: capturas del chat animadas por UIKit
con la curva del teclado (140–141) y, después, el WebView entero viajando con él (142). Dos builds en el iPhone del dueño:

    140  «se medio buguea»
    141  «mejor, pero sigue habiendo delay alrededor del entorno… fallas visuales alrededor»
    142  «quiero que lo dejes como estaba, ya que nada funciona»

Se REVIERTEN enteros —código web, Swift, Info.plist, textos y sus tests (140, 141 y 142 afirmaban la presencia de lo que
ya no existe)—: el árbol del frontend es byte a byte el del lote 139. No se deja apagado dentro: un modo de prueba que el
dueño descartó es peso muerto en un fichero de 6.000 líneas, y lo aprendido vive en la memoria del proyecto.

El binario que el dueño tiene instalado todavía trae `CoberturaDelTeclado`, pero sin la geometría que le mandaba la página
nunca cubre (`activo:false`): solo retransmite el aviso del teclado, como en el lote 128. Cero cambios de backend."""
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


def test_del_modo_nativo_no_queda_nada():
    ap = _front("src/pages/AgentPage.jsx")
    assert not (_FRONT / "src" / "utils" / "keyboardNative.js").exists()
    for resto in ("keyboardNative", "esperarAperturaNativa", "esperarCierreNativo", "atenderCoberturaNativa", "data-kb-vuelo", "/nativo"):
        assert resto not in ap, resto
    sw = _front("ios/App/App/SceneDelegate.swift")
    assert "CoberturaDelTeclado" not in sw and "WKScriptMessageHandler" not in sw
    assert "CADisableMinimumFrameDurationOnPhone" not in _front("ios/App/App/Info.plist")


def test_el_camino_de_siempre_sigue_en_su_sitio():
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const anticiparApertura = (inset) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "contenedor.style.setProperty('--kb-inset', `${inset}px`);" in cuerpo
    assert "root.toggleAttribute('data-kb-scroll-lock', true);" in cuerpo
    assert "anticiparApertura(recordado);" in ap and "anticiparApertura(aviso.inset);" in ap
    # y la coreografía con transform sigue siendo un modo de prueba APAGADO (lote 139)
    assert "return safeLocalStorageGet(CLAVE_COREOGRAFIA, null) === '1';" in _front("src/utils/keyboardChoreography.js")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 143
