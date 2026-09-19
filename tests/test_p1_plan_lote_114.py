# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-114 · 2026-09-19] El teclado del chat en la app nativa: el parpadeo de `innerHeight` del WebView.

Medido con la sonda (`/sonda`, lote 112) en el iPhone del dueño, abriendo el teclado: toque 0 ms · foco 39 (la apertura
anticipada pone el inset en 403) · `resize` 113 con la geometría FINAL (kb=403) · `scroll` 140 con `innerHeight`=441
—el WebView encoge el layout viewport unos fotogramas— · `innerHeight` vuelve a 844 SIN evento · el asiento de 350 ms
repone el inset · 756 ms la caja queda en su sitio. El camino común leía el fotograma de 441 como «el documento ya
encogió» (lo que hace de verdad la PWA) y dejaba el inset en 0: la caja de escribir quedaba TAPADA por el teclado
hasta el asiento — el «delay» — o para siempre si el parpadeo duraba más («ni se encaja»).
En nativo el inset es teclado − paneo, el alto base del contenedor se fija en px mientras hay teclado y se re-mide con
el `resize` de la ventana. La PWA y Safari no cambian. Contrato fino: `frontend/src/__tests__/lote114.test.js`."""
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


def test_en_nativo_el_inset_no_depende_del_parpadeo_de_innerheight():
    kv = _front("src/utils/keyboardViewport.js")
    assert "export function resolverInsetNativo({ kb = 0, vvOffsetTop = 0 } = {}) {" in kv
    ap = _front("src/pages/AgentPage.jsx")
    assert "const insetMedido = nativo ? resolverInsetNativo({ kb, vvOffsetTop: vv.offsetTop }) : layoutInset;" in ap
    assert "const encogeDeVerdad = nativo ? false : documentoEncoge;" in ap
    assert "forzar: forzarMedicion || encogeDeVerdad," in ap
    assert "layoutInset: insetMedido," in ap, "la PWA y Safari siguen con layoutInset: solo nativo cambia"


def test_alto_base_fijo_con_teclado_y_remedicion_con_la_ventana():
    ap = _front("src/pages/AgentPage.jsx")
    assert ap.count("contenedor.style.setProperty('--app-height', `${altoDeReferencia(window.innerHeight, window.innerWidth)}px`);") == 2
    assert ap.count("contenedor.style.removeProperty('--app-height');") >= 2, "se suelta al cerrar y al salir de la ruta"
    assert "window.addEventListener('resize', alEvento);" in ap
    assert "window.removeEventListener('resize', alEvento);" in ap


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 114
