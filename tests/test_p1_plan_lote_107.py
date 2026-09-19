# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-107 · 2026-09-18] La fototeca directa de la app nativa no falla en silencio.

El dueño vio dos veces la hoja de tres opciones de iOS tras el lote 105 («sigue igual en la app instalada»): lo
instalado era la PWA, donde Safari impone ese menú; el binario nativo lleva la web empaquetada y solo cambia con un
build MANUAL de Codemagic. Lo que sí era un defecto: en nativo, un fallo del plugin caía callado al input de la web
—la misma hoja— y era indistinguible de «no se hizo nada». Ahora se reporta, se dice con su código y después se cae
al input. Ancla cross-repo; el contrato fino vive en `lote107.test.jsx`."""
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


def test_el_fallo_del_selector_nativo_se_ve_y_se_reporta():
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    assert "if (isNativePickerCancellation(err)) return;" in sm
    assert "captureException(err, { tags: { component: 'ScanMealModal', action: 'native_gallery_picker' } })" in sm
    assert "toast.error(t('No pudimos abrir tus fotos. Revisa los permisos e inténtalo de nuevo.'), { description: `[${codigo}]` });" in sm
    # el aviso va ANTES de caer al input: el orden es la garantía de que el fallo no es silencioso
    i = sm.index("action: 'native_gallery_picker'")
    assert "galleryInputRef.current?.click();" in sm[i:i + 600]


def test_el_binario_nativo_lleva_la_web_empaquetada_y_el_build_es_manual():
    """Las dos premisas del diagnóstico («era la PWA»): si alguna cambia, la explicación al dueño deja de valer."""
    cm = _front("codemagic.yaml")
    assert "npm run build:native" in cm and "npx cap sync ios" in cm
    assert not re.search(r"^\s*triggering:", cm, flags=re.M), "con `triggering` el build ya no sería manual"
    cap = _front("capacitor.config.ts")
    assert "webDir: 'dist'" in cap
    assert not re.search(r"^\s*server\s*:", cap, flags=re.M), "con `server.url` el binario cargaría la web remota"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 107
    assert "P1-PLAN-LOTE-107" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
