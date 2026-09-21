# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-140 · 2026-09-20] El BINARIO mueve el chat con el teclado (modo de prueba: `/nativo`).

El dueño, tras el revés del lote 139: «procede a mover el chat desde el código nativo para igualar a Gemini».

Gemini es nativa: sus vistas se animan DENTRO de la animación del teclado, en el servidor de render. Una página no puede
—o anima `height` (layout por fotograma en su hilo) o un `transform` que arranca tarde, y con el layout aún cerrado iOS
panea el documento (medido en el 139)—. Lo que hace este lote:

  · el binario (`SceneDelegate.swift` → `CoberturaDelTeclado`), al llegar `keyboardWillShow/Hide`, CAPTURA el chat en
    tiras (conversación, caja de escribir, y las fichas de la cabecera quietas encima) y UIKit las mueve con la curva y
    la duración EXACTAS del teclado;
  · la página le manda ANTES su geometría (en `keyboardWillShow` ya no hay tiempo de preguntar), y al recibir el aviso
    con `cubierto:true` pone su layout final DE GOLPE debajo de la captura y contesta `listo`;
  · la captura se retira cuando acabó la animación Y la página está lista — o al vencer un plazo, pase lo que pase.

Apagado por defecto (lección del 139: lo no medido en el teléfono se entrega apagado) y necesita un build de Codemagic:
en binarios anteriores el manejador no existe y `/nativo` lo dice. Cero cambios de backend: aquí, las anclas entre repos.
Contrato fino y piezas puras: `frontend/src/__tests__/lote140.test.js`."""
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


def test_apagado_por_defecto_y_solo_con_un_binario_que_lo_sepa_hacer():
    kn = _front("src/utils/keyboardNative.js")
    assert "export const CLAVE_TECLADO_NATIVO = 'mf_kb_nativo';" in kn
    assert "return safeLocalStorageGet(CLAVE_TECLADO_NATIVO, null) === '1';" in kn
    assert "return tecladoNativoElegido() && binarioMueveElChat(win);" in kn
    # el nombre del manejador es el mismo en los dos lados
    assert "export const MANEJADOR_NATIVO = 'mfTeclado';" in kn
    assert 'static let nombre = "mfTeclado"' in _front("ios/App/App/SceneDelegate.swift")


def test_el_binario_captura_antes_de_avisar_y_nunca_deja_la_captura_puesta():
    sw = _front("ios/App/App/SceneDelegate.swift")
    k = sw.index("private func retransmitirTeclado(")
    cuerpo = sw[k:sw.index("\n    }", k)]
    assert cuerpo.index("cobertura.alAvisoDelTeclado(") < cuerpo.index("evaluateJavaScript(js")
    assert "UIView.AnimationOptions(rawValue: UInt(max(0, curva)) << 16)" in sw, "la MISMA curva que el teclado"
    assert "nueva.isUserInteractionEnabled = false" in sw, "los toques llegan al WebView"
    assert "DispatchQueue.main.asyncAfter(deadline: .now() + plazo)" in sw, "pase lo que pase en la página, se retira"
    assert "cobertura.retirarYa()" in sw
    # sigue en pie el contrato del lote 128: se AÑADEN observadores, no se retira ninguno de WebKit
    assert "removeObserver" not in sw


def test_el_chat_espera_al_binario_y_bajo_la_captura_no_anima_nada():
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (!e.sinNativo && esperarAperturaNativa(campo)) return;" in ap
    assert "if (!e.deNativo && esperarCierreNativo()) return;" in ap
    k = ap.index("const alTecladoNativo = (e) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "if (atenderCoberturaNativa(e.detail, aviso)) return;" in cuerpo
    a = ap.index("const abrirBajoCobertura = (contenedor, inset, id, veniaDelFoco")   # [142] ganó un parámetro detrás
    abrir = ap[a:ap.index("\n        };", a)]
    assert abrir.index("congelarAlto(contenedor);") < abrir.index("contenedor.style.setProperty('--kb-inset', `${inset}px`);")
    assert "root.toggleAttribute('data-kb-scroll-lock', true);" in abrir
    assert "confirmarAlNativo(contenedor, id, true" in abrir
    # sin aviso a tiempo, el camino de siempre
    assert "alGanarElFoco({ target: campo, sinNativo: true });" in ap
    assert "alPerderElFoco({ relatedTarget: null, deNativo: true });" in ap


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 140
