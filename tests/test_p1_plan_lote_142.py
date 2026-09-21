# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-142 · 2026-09-20] Con `/nativo`, la PÁGINA VIVA también viaja con el teclado.

El dueño, con el 141 en el iPhone (su sonda: cubierto, `natOk` a los 12–18 ms, sin paneo de iOS): «mejor, pero sigue
habiendo delay alrededor del entorno… cuando cierro y abro ocurren fallas visuales alrededor».

Era inherente a mover SOLO una captura: todo lo que la captura no tiene —la barra de pestañas al cerrar, los mensajes que
asoman por arriba, el cursor, el velo de la cabecera— aparecía al RETIRARLA, con la animación ya acabada. Ahora:

  · el binario desplaza además el WebView ENTERO con un `transform` animado en el mismo bloque que las tiras, haciendo el
    viaje complementario (D = S − destino): la página, ya con su layout final, coincide con las tiras en cada fotograma;
  · las tiras solo tapan el relevo: al `listo` (~40 ms) se funden y lo que se ve moverse es la página VIVA. Solo si la
    conversación viaja ENTERA con la caja; con la lista quieta o a medio recorrido se quedan hasta el final (como el 141);
  · la cabecera es fija en pantalla y viajaría con el WebView: la página la esconde durante el viaje (encima están,
    quietas, las fichas y un velo que pinta el binario) y al reponerla dice `fin`;
  · SEGURO: toda salida (fin, plazo, perder la escena, `apagar`, activarse) deja `webView.transform = .identity`.

Pide OTRO build. Un segundo manejador (`mfTecladoVivo`) le dice a la página con qué binario habla ANTES del primer aviso;
con el binario del 140 todo sigue como en el 141. `/nativo vivo` vuelve a solo capturas sin otro build.
Contrato fino: `frontend/src/__tests__/lote142.test.js`. Cero cambios de backend."""
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


def test_los_dos_lados_se_reconocen_antes_del_primer_aviso():
    assert "export const MANEJADOR_NATIVO_VIVO = 'mfTecladoVivo';" in _front("src/utils/keyboardNative.js")
    sw = _front("ios/App/App/SceneDelegate.swift")
    assert 'static let nombreVivo = "mfTecladoVivo"' in sw
    assert "controlador.add(self, name: CoberturaDelTeclado.nombreVivo)" in sw
    assert "vivo: binarioMueveLaPagina() && nativoVivoElegido()," in _front("src/pages/AgentPage.jsx")


def test_el_webview_viaja_con_las_tiras_y_siempre_vuelve_a_su_sitio():
    sw = _front("ios/App/App/SceneDelegate.swift")
    k = sw.index("private func animar(")
    assert "pagina?.transform = CGAffineTransform.identity" in sw[k:sw.index("\n    }", k)]
    assert "webView.transform = CGAffineTransform(translationX: 0, y: desplazada - destinoNuevo)" in sw, "D = S − destino"
    r = sw.index("private func retirar(fundido: Double) {")
    cuerpo = sw[r:sw.index("\n    }", r)]
    assert cuerpo.index("webView.transform = CGAffineTransform.identity") < cuerpo.index("guard let saliente = capa else { return }")
    assert "cobertura.asegurarEnSuSitio()" in sw
    assert "if !mantener && listaViajaEntera { relevarTiras() }" in sw


def test_la_pagina_esconde_su_cabecera_durante_el_viaje_y_dice_fin():
    ap = _front("src/pages/AgentPage.jsx")
    assert re.search(r"html\[data-kb-vuelo\] \.mobile-chat-header,\s*html\[data-kb-vuelo\] pre\[data-mf-sonda\] \{\s*visibility: hidden !important;", ap)
    assert "const vuelo = { vivo: detalle.vivo === true, ms: aviso.ms };" in ap, "solo si el BINARIO dijo `vivo`"
    f = ap.index("const acabarVueloNativo = (id) => {")
    fin = ap[f:ap.index("\n        };", f)]
    assert fin.index("root.removeAttribute('data-kb-vuelo');") < fin.index("enviarAlNativo({ tipo: 'fin', id });")
    assert "acabarVueloNativo(null);" in ap, "al salir del chat no queda la cabecera escondida"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 142
