# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-131 · 2026-09-20] Coreografía del teclado SOLO con `transform` (modo de prueba: `/fluido`).

El dueño, con los avisos nativos del teclado ya en marcha (lote 129): «sigue igual cuando selecciono una foto, no es 100 %
fluido, aparte no sé si soy yo, pero lo siento lento el teclado cuando lo abro».

Lo que el chat animaba para acompañar al teclado era `height` (contenedor) y `padding-bottom` (caja): propiedades de LAYOUT,
que recalculan la conversación entera en cada fotograma, en el hilo principal — que al volver del selector de fotos está
además decodificando y reduciendo la imagen. El teclado de iOS lo mueve el sistema y nunca pierde un fotograma.

Con el modo encendido (`/fluido` en el chat de la app nativa; APAGADO por defecto) se mueve lo mismo con `transform` y se
hace UN cambio de layout, en el extremo donde no se ve:
  · abrir  = transform → RELEVO al acabar (fuera transform, dentro layout final, mismo fotograma);
  · cerrar = layout final de golpe + transform que lo deshace a la vista → el transform baja a 0.

MEDIDO en el arnés (`agente.html?nativa=1`, teclado de 335 px → recorrido 266 px): relevo de apertura 0,2 px, cierre
0,2 px, la caja vuelve a su sitio exacto, y reabrir a mitad de cierre no deja restos. Cazados allí antes de llegar al
teléfono: (1) la transición inline perdía contra el `transition … !important` de la hoja → la caja SALTABA 266 px;
(2) `requestAnimationFrame` se para con la página oculta —lo que ocurre al abrir el selector de fotos— y dejaba la caja
con su transform puesto: ahora lleva respaldo por tiempo. Sin medir en el iPhone: por eso es un modo de prueba.

Contrato fino: `frontend/src/__tests__/lote131.test.js`. Cero cambios de backend."""
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


def test_es_un_modo_de_prueba_apagado_por_defecto():
    kc = _front("src/utils/keyboardChoreography.js")
    assert "return safeLocalStorageGet(CLAVE_COREOGRAFIA, null) === '1';" in kc
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (isNativeApp() && textToSend.trim().toLowerCase() === '/fluido') {" in ap
    assert "if (!isNativeApp() || !coreografiaEncendida() || !(msVigente > 0)) return false;" in ap


def test_el_recorrido_descuenta_la_reserva_de_la_barra_de_pestanas():
    kc = _front("src/utils/keyboardChoreography.js")
    assert "export function recorridoDelTeclado({ inset = 0, padCerrado = 0, padAbierto = 0 } = {}) {" in kc
    assert "export const KB_PAD_ABIERTO_REM = 1.1;" in kc
    ap = _front("src/pages/AgentPage.jsx")
    regla = ap[ap.index("html[data-kb-open] .input-wrapper {"):]
    assert "padding-bottom: 1.1rem !important;" in regla[:regla.index("}")], "el espejo CSS del relleno «abierto»"


def test_lo_que_el_arnes_cazo_sigue_en_su_sitio():
    ap = _front("src/pages/AgentPage.jsx")
    assert "'important');" in ap[ap.index("const moverPiezas = (y, ms) => {"):][:700], "la hoja trae transition !important"
    k = ap.index("const alSiguienteFotograma = (fn) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "requestAnimationFrame(una);" in cuerpo and "setTimeout(una, 60);" in cuerpo, "rAF se para con la página oculta"
    assert "if (coreo.fase) { coreo.medirLuego = true; return; }" in ap, "la geometría no escribe con piezas en el aire"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 131
