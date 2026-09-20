# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-141 · 2026-09-20] Primer build con `/nativo` en el iPhone del dueño: «se medio buguea».

Su sonda (paquete 20260920-232341) dice que la maquinaria del lote 140 FUNCIONA:

    + 107 natFoco                         ← al foco el chat no se mueve: manda su geometría y espera
    + 189 nat+ / N+335·383c   cont=509    ← el binario cubrió (`c`) y la página puso su layout final en el mismo ms
    + 235 natOk                           ← 46 ms después: `listo`
      S=0 sy=0 en todas las filas         ← sin el paneo de iOS que tumbó el lote 138

y su captura enseña el estado: la flecha de «ir al final» a la vista = chat en modo LIBRE. Ahí el ResizeObserver del
contenedor (lote 115) sube la conversación lo que encoge la ventana, pero el 140 le decía al binario «la lista no se
mueve» (solo miraba si estaba PEGADA al final). La captura la dejaba quieta, la página la subía 266 px debajo, y al
fundirse la conversación entera saltaba. Tres arreglos, todos en el frontend y por OTA (sin otro build):
  1. cuánto se mueve la lista lo decide el MODO de scroll ('bottom' y 'free' se mueven; 'anchored' no, salvo que baje);
  2. bajo la captura ese ajuste va de golpe (la lista lleva scroll suave: seguía deslizándose al retirar la captura);
  3. la flecha de «ir al final» cuelga de la caja: viaja en la tira de la caja.
Contrato fino: `frontend/src/__tests__/lote141.test.js`. Cero cambios de backend."""
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


def test_la_lista_se_mueve_segun_el_modo_de_scroll():
    kn = _front("src/utils/keyboardNative.js")
    assert "if (abierto) return modo === 'anchored' ? 0 : Math.max(0, Math.round(num(scrollTop)));" in kn
    assert "return vaAlFinal || modo !== 'anchored' ? LISTA_SIN_TOPE : 0;" in kn
    ap = _front("src/pages/AgentPage.jsx")
    assert "modo: scrollModeRef.current," in ap
    assert "wrapper.querySelector('.jump-to-latest')" in ap


def test_bajo_la_captura_el_ajuste_del_modo_libre_va_de_golpe():
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("} else if (delta !== 0 && document.documentElement.hasAttribute('data-kb-sin-anim')) {")
    assert "el.scrollTo({ top: Math.max(0, el.scrollTop + delta), behavior: 'instant' });" in ap[k:k + 700]
    # y sin captura, el ajuste de siempre (lote 115)
    assert "el.scrollTop = Math.max(0, el.scrollTop + delta);" in ap


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 141
