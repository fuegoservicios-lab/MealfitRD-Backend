# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-138 · 2026-09-20] El teclado del chat nativo: la coreografía con `transform` pasa a ser el modo por defecto.

El dueño, con la sonda puesta y el modo de prueba `/fluido` APAGADO: «aún no lo siento 100 % fluido y rápido el teclado
cuando lo abro y selecciono una foto con él abierto. En la app de Gemini se siente muy pero muy fluido y rápido».

Su captura (paquete 20260920-174852) MIDE el porqué — no es una impresión:

    +4891 N+308·383   cont=844          ← vuelve del selector: UIKit anuncia un teclado DE PASO (sin barra de sugerencias)
    +4914 N-0·0                          ← se ignora (lote 129)
    +5015 N+335·400   cont=597          ← 124 ms después, el alto firme: la animación se RE-APUNTA
    +5095 resize      cont=597          ← 80 ms CLAVADO: ni un fotograma (la foto se está reduciendo y recodificando)
    +5485 altoFin     cont=509          ← el chat acaba ~70 ms DESPUÉS que el teclado
  y al abrir:  +185 resize cont=844     ← 185 ms tras el toque el chat aún no se había movido.

Tres causas, tres arreglos (todo en el frontend; aquí solo las anclas entre repos):
  1. Animar `height` necesita al hilo principal en CADA fotograma, y el WebView pinta a 60 Hz junto a un teclado que el
     iPhone del dueño (ProMotion) mueve a 120. La coreografía con `transform` (lote 131) se entrega UNA vez al
     compositor: pasa a ser el modo por defecto y `/fluido` queda como interruptor de VUELTA. Al abrir, además, el chat
     llega un poco ANTES que el teclado (tapado nunca); al cerrar usa la duración entera.
  2. Una apertura, UNA animación: `insetDeApertura` usa el alto firme recordado cuando el anuncio se le parece, no
     re-apunta una apertura en vuelo, y el alto de paso no se recuerda.
  3. La foto se VE al instante (`previewUrl`) y su preparación —lo que atascaba el hilo— espera a que el teclado acabe.

Contrato fino y pruebas de comportamiento: `frontend/src/__tests__/lote138.test.jsx`. Cero cambios de backend.
Sin medir en el iPhone: si el dueño lo ve peor, `/fluido` vuelve al modo anterior y la sonda (`/sonda`) dirá por qué."""
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


def test_la_coreografia_viene_encendida_y_fluido_es_el_interruptor_de_vuelta():
    kc = _front("src/utils/keyboardChoreography.js")
    assert "return safeLocalStorageGet(CLAVE_COREOGRAFIA, null) !== '0';" in kc
    assert "safeLocalStorageSet(CLAVE_COREOGRAFIA, '0');" in kc, "apagar GUARDA la decisión; encender vuelve al defecto"
    ap = _front("src/pages/AgentPage.jsx")
    assert "if (isNativeApp() && textToSend.trim().toLowerCase() === '/fluido') {" in ap
    # y sigue siendo SOLO de la app nativa, con una duración conocida
    assert "if (!isNativeApp() || !coreografiaEncendida() || !(msVigente > 0)) return false;" in ap


def test_al_abrir_el_chat_llega_antes_que_el_teclado_y_al_cerrar_no():
    kc = _front("src/utils/keyboardChoreography.js")
    m = re.search(r"export const COREO_ADELANTO = (0\.\d+);", kc)
    assert m and 0.6 <= float(m.group(1)) < 1.0
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const abrirConCoreografia = (contenedor, inset, vaAlFinal) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "const msApertura = duracionDeApertura(msVigente);" in cuerpo
    assert "moverPiezas(-recorrido, msApertura);" in cuerpo
    c = ap.index("const prepararCierreConCoreografia = () => {")
    assert "duracionDeApertura" not in ap[c:ap.index("\n        };", c)], "bajar antes que el teclado mete la caja detrás"


def test_una_apertura_una_animacion():
    kc = _front("src/utils/keyboardChoreography.js")
    assert "export function insetDeApertura({ anunciado = 0, recordado = 0, vigente = 0, abierto = false, enApertura = false } = {}) {" in kc
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const alTecladoNativo = (e) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "aviso.inset = insetDeApertura({" in cuerpo
    assert cuerpo.index("aviso.inset = insetDeApertura({") < cuerpo.index("anticiparApertura(aviso.inset);")
    assert "if (aviso.inset === anunciado && String(anunciado) !== safeLocalStorageGet(CLAVE_INSET_NATIVO, null))" in cuerpo, \
        "el alto de paso (308) no pisa el firme recordado (335)"


def test_la_foto_se_ve_al_instante_y_su_preparacion_espera_al_teclado():
    hook = _front("src/hooks/useChatAttachments.js")
    assert "const addFiles = useCallback((inputFiles, { prepararTrasMs = 0 } = {}) => {" in hook
    assert "programarPump(0);" in hook, "ENVIAR no espera el aplazamiento"
    ap = _front("src/pages/AgentPage.jsx")
    assert "addFiles(files, { prepararTrasMs: reopenKeyboardAfterAttachmentRef.current ? ESPERA_PREPARAR_FOTO_MS : 0 });" in ap
    assert "{item.status !== 'error' && (item.thumbDataUrl || (item.previewUrl && !previewsRotas.has(item.id))) ? (" in ap
    # el contrato del lote de fotos sigue en pie: la burbuja toma la mejor URL y la preparación produce la miniatura
    assert "url: item.url || item.image_url || item.thumbDataUrl || item.previewUrl" in ap
    assert "prepareChatImage(job.sourceFile" in hook


def test_el_relevo_fija_el_final_de_la_lista_sin_scroll_suave():
    # Cazado en el arnés AL ENCENDERLA por defecto: la lista lleva `scroll-behavior: smooth`, y ahí `scrollTop = …` no es
    # inmediato (medido: 3958 → 3958). Al soltar el transform el contenido bajaba 266 px y volvía deslizándose.
    ap = _front("src/pages/AgentPage.jsx")
    k = ap.index("const relevoDeApertura = (contenedor, lista, acompana) => {")
    cuerpo = ap[k:ap.index("\n        };", k)]
    assert "lista.scrollTo({ top: lista.scrollHeight, behavior: 'instant' });" in cuerpo


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 138
