# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-112 · 2026-09-19] «Calculando tus metas…» solo la primera vez, y la sonda del teclado en la app nativa.

(1) `DashboardTracking` nacía con `targets = null` y pedía `/api/nutrition/targets` en CADA montaje: salir de «Progreso»
y volver enseñaba el aviso otra vez. Ahora las metas buenas se recuerdan por usuario (`utils/targetsCache.js`), la
pantalla pinta con ellas y vuelve a pedir por detrás; un fallo de red no pisa unas metas buenas y la caché se borra
con el resto de cachés del usuario. (2) El primer arreglo del «delay al abrir el teclado» (lote 111) se decidió sin
números y el dueño respondió «sigue igual»: la sonda del teclado pasa a poder encenderse en la app nativa con `/sonda`
en el chat, con milisegundos y el paquete OTA en la cabecera; y el alto del teclado se recuerda sin condiciones de
paneo. Contrato fino: `lote112.test.jsx` y `lote112.sonda.test.js`."""
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


def test_las_metas_se_recuerdan_por_usuario_y_se_borran_con_la_sesion():
    cache = _front("src/utils/targetsCache.js")
    assert "if (!userId || targets?.ok !== true) return;" in cache, "solo se recuerda una respuesta BUENA"
    assert "guardado?.userId === userId" in cache, "nunca las metas de otro usuario"
    dt = _front("src/components/dashboard/DashboardTracking.jsx")
    assert "useState(() => readTargetsCache(uid));" in dt
    assert "const targets = targetsPedidas ?? readTargetsCache(uid);" in dt
    assert "return !nuevo.ok && vigente?.ok && !nuevo.missing_fields?.length ? vigente : nuevo;" in dt
    ctx = _front("src/context/AssessmentContext.jsx")
    i = ctx.index("const _clearUserScopedCaches = () => {")
    assert "clearTargetsCache();" in ctx[i:i + 4000]


def test_la_sonda_nativa_es_explicita_y_la_web_no_cambia():
    sonda = _front("src/utils/keyboardProbe.js")
    assert "if (!activa && isNativeApp()) activa = safeLocalStorageGet(CLAVE_SONDA_NATIVA) === '1';" in sonda
    assert "if (!isNativeApp()) return false;" in sonda
    assert ": pedida;" in sonda, "en la web de producción el único interruptor sigue siendo ?kbprobe"
    ap = _front("src/pages/AgentPage.jsx")
    i = ap.index("const handleSend = async (overrideInput = null, options = {}) => {")
    cuerpo = ap[i:i + 1400]
    j = cuerpo.index("if (isNativeApp() && textToSend.trim().toLowerCase() === '/sonda') {")
    assert "return;" in cuerpo[j:j + 400], "`/sonda` no abre turno ni llega al servidor"
    assert "if (forzarMedicion && abierto && kb >= KB_UMBRAL_PX && isNativeApp()) {" in ap


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 112
