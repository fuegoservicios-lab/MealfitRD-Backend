# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-129 · 2026-09-19] El chat se mueve A LA VEZ que el teclado: hace caso al aviso nativo de UIKit.

El dueño, con el build 16 y la sonda encendida: «cuando voy a seleccionar una foto con el teclado abierto se ve un poquito
glicheado». Y antes: «¿no puedes hacer más fluido el cerrar y abrir el teclado?». Lo que dijo la sonda, al volver del
selector de fotos (iOS repone el teclado SIN evento de foco, así que la apertura anticipada del lote 111 no corre):

    +5383 N+335·400   ← UIKit: el teclado EMPIEZA a subir (335 px, 400 ms)
    +5560 resize      ← la web se entera 177 ms después: la caja lleva 177 ms TAPADA por el teclado
    +5812 altoFin     ← y sube en 250 ms: aparece por detrás del teclado, tarde y a otro ritmo

Dos defectos en una fila: llegar 177 ms tarde, y mover el chat en 0,25 s fijos cuando el teclado de este iOS tarda
0,38–0,40 s. El binario (lote 128) retransmite `keyboardWillShow/Hide` como `mf:teclado-nativo`; ahora el chat lo consume:
arranca en el mismo instante, con el alto exacto y la duración real (`--kb-ms`), también en la primera apertura. Un aviso
sin animación (`N-0·0`, medido en mitad de la vuelta) se ignora; binarios viejos no emiten nada y todo sigue igual.

Contrato fino: `frontend/src/__tests__/lote129.test.js`. Cero cambios de backend."""
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


def test_la_decision_del_aviso_vive_en_una_funcion_pura():
    kv = _front("src/utils/keyboardViewport.js")
    assert "export function decidirAvisoNativo({ tipo, alto = 0, ms = 0, innerHeight = 0 } = {}) {" in kv
    assert "if (duracion <= 0) return { accion: 'ignorar', motivo: 'sin_animacion' };" in kv, "el N-0·0 medido no mueve nada"
    assert "inset < KB_UMBRAL_PX" in kv, "la barra de un teclado físico no es «hay teclado»"


def test_el_chat_consume_el_aviso_por_las_mismas_puertas_que_foco_y_blur():
    ap = _front("src/pages/AgentPage.jsx")
    assert "window.addEventListener(EVENTO_TECLADO_NATIVO, alTecladoNativo);" in ap
    assert "window.removeEventListener(EVENTO_TECLADO_NATIVO, alTecladoNativo);" in ap
    k = ap.index("const alTecladoNativo = (e) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "if (!isNativeApp()) return;" in cuerpo
    assert "alPerderElFoco({ relatedTarget: null });" in cuerpo, "cerrar = la misma puerta que el blur"
    assert "anticiparApertura(aviso.inset);" in cuerpo, "abrir = la misma que el foco"


def test_las_tres_piezas_comparten_la_duracion_real_del_teclado():
    curva = "var(--kb-ms, 0.25s) cubic-bezier(0.32, 0.72, 0, 1)"
    ap = _front("src/pages/AgentPage.jsx")
    assert ap.count(curva) == 2, "alto del chat + relleno de la caja"
    assert f"transform {curva}" in _front("src/components/dashboard/BottomTabBar.module.css"), "y la barra de pestañas"
    assert "root.style.removeProperty('--kb-ms');" in ap, "se retira al acabar: plegar la barra no es cosa del teclado"


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 129


def test_quitar_la_foto_adjunta_no_se_lleva_el_teclado():
    # «cuando cierro la foto en la x se me cierra el teclado»: en iOS el foco sale de la caja cuando WebKit sintetiza
    # mousedown/click tras el touchend. El toque se atiende EN touchend y se cancela (misma técnica que el micrófono).
    hook = _front("src/hooks/useToqueSinFoco.js")
    assert "if (!e.cancelable) return;" in hook and "e.preventDefault();" in hook
    assert "onMouseDown: (e) => e.preventDefault()," in hook
    assert "if (Date.now() - ultimoToqueRef.current < CLIC_FANTASMA_MS) return;" in hook, "un clic del mismo gesto no repite"
    assert "{...sinFoco(() => removeSelectedAttachment(item.id))}" in _front("src/pages/AgentPage.jsx")
