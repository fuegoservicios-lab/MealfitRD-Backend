# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-125 · 2026-09-19] El micrófono del chat, el vacío del Historial y la Nevera sin ruido.

El dueño: «agrega un microfonito para poder hablar en vez de escribir, hazlo lo mejor posible, fluido etc. y mejora cómo
se ve el historial visualmente, y si puedes pulir el diseño de la nevera te lo agradecería también».

  · DICTADO. El motor es el reconocimiento de voz del propio navegador: escribe MIENTRAS se habla, no cuesta por uso y el
    audio no pasa por este backend (cero endpoints nuevos, cero gasto de IA). Lo delicado es DÓNDE se ofrece, porque hay
    dos sitios donde fallaría seguro y un botón que siempre falla es peor que no tenerlo:
      – la web con `Permissions-Policy: microphone=()`: medido en producción, `allowsFeature('microphone')` daba false.
        El snippet de nginx pasa a `microphone=$pp_camera` (mismo mapa por host que la cámara; ver su comentario);
      – la app nativa cuyo binario no declara `NSMicrophoneUsageDescription` / `NSSpeechRecognitionUsageDescription`.
        La web no puede leer el Info.plist y un paquete OTA también corre sobre binarios viejos: el binario que gana las
        claves añade `BioborosNative/mic` a su user agent (`capacitor.config.ts`) y sin esa marca el botón no se pinta.
    Ciclo de vida en `frontend/src/hooks/useDictado.js`: ENVIAR cancela (un resultado tardío no vuelve a la caja), teclear
    apaga el dictado, y un micrófono olvidado se apaga solo.
  · HISTORIAL. El vacío era una caja punteada pegada arriba con media pantalla muerta debajo; ahora enseña tres fichas
    «fantasma» de lo que va a haber, centrado en el alto libre.
  · NEVERA. Vacía, ya no ofrece buscar, «Borrar todos», «Todos 0» ni el aviso de «nevera baja» sobre un «está vacía»;
    y «Borrar todos» deja de pesar lo mismo que «Añadir alimento».

Contrato fino: `frontend/src/__tests__/lote125.test.jsx`."""
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


def test_el_microfono_solo_se_ofrece_donde_puede_funcionar():
    util = _front("src/utils/dictado.js")
    assert "export const DICTADO_UA_NATIVO = 'BioborosNative/mic';" in util
    assert "if (esNativa) return String(userAgent || '').includes(DICTADO_UA_NATIVO);" in util
    assert "politica.allowsFeature('microphone') !== false" in util, "en la web manda la Permissions-Policy"
    # la marca del binario y sus permisos viajan juntos
    assert "appendUserAgent: 'BioborosNative/mic'," in _front("capacitor.config.ts")
    plist = _front("ios/App/App/Info.plist")
    assert "<key>NSMicrophoneUsageDescription</key>" in plist
    assert "<key>NSSpeechRecognitionUsageDescription</key>" in plist


def test_nginx_abre_el_microfono_solo_en_los_hosts_de_la_app():
    snippet = (_BACKEND / "infra" / "nginx" / "snippets" / "mealfit-security.conf").read_text(encoding="utf-8")
    assert "microphone=$pp_camera" in snippet, "sin esto el navegador niega el dictado sin preguntar"
    assert "P1-PLAN-LOTE-125" in snippet, "y el porqué de reusar el mapa de la cámara queda escrito"
    apex = (_BACKEND / "infra" / "nginx" / "snippets" / "bioboros-v2-security.conf").read_text(encoding="utf-8")
    assert "microphone=()" in apex, "el apex es marketing: no pide micrófono jamás"


def test_enviar_cancela_el_dictado_y_teclear_lo_apaga():
    ap = _front("src/pages/AgentPage.jsx")
    assert "const dictado = useDictado({ valor: input, alCambiar: setInput, locale: getLocale(), esNativa: isNativeApp() });" in ap
    assert re.search(r"if \(isListening\) \{\n\s+dictado\.cancelar\(\);", ap), "enviar CANCELA: `detener` deja llegar un resultado tardío"
    assert "onChange={(e) => { if (isListening) dictado.cancelar(); setInput(e.target.value); }}" in ap
    assert "{dictado.disponible && !isCallModeActive && (" in ap
    hook = _front("src/hooks/useDictado.js")
    assert "if (id !== sesionRef.current) return;" in hook, "cada sesión lleva número: lo que llega de una cerrada se tira"
    assert "silencioRef.current = setTimeout(detener, ms);" in hook, "un micrófono olvidado se apaga solo"


def test_el_vacio_del_historial_y_la_nevera_sin_ruido():
    assert "${styles.emptyState} ${styles.emptyStateArt}" in _front("src/pages/History.jsx")
    assert "border: 2px dashed #CBD5E1;" not in _front("src/pages/History.module.css")
    p = _front("src/pages/Pantry.jsx")
    assert "const neveraVacia = inventory.length === 0;" in p
    assert "{pantryStatus?.is_below && !neveraVacia && (" in p
    assert ".clear { flex: 0 0 auto;" in _front("src/pages/Pantry.mobileFridge.module.css")


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 125
