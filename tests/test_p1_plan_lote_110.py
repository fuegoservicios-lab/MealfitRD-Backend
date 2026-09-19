# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-110 · 2026-09-19] «Subir una foto en su lugar» va a la fototeca directa en la app nativa.

El lote 105 llevó «Elegir de galería» al plugin nativo, pero la salida del visor de cámara seguía haciendo click al
input de la web: en el build nativo, otra vez la hoja de iOS (Fototeca / Hacer foto / Seleccionar archivo) — captura
del dueño. Los dos escáneres comparten visor y compartían el defecto. Primer cambio que viaja a la app por OTA (lote
108). Contrato fino: `frontend/src/__tests__/lote110.test.jsx`."""
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


def test_la_salida_del_visor_de_comida_pasa_por_open_gallery():
    sm = _front("src/components/dashboard/ScanMealModal.jsx")
    i = sm.index("const handleViewfinderFallback = useCallback(() => {")
    cuerpo = sm[i:i + 200]
    assert "void openGallery();" in cuerpo and "galleryInputRef" not in cuerpo


def test_la_salida_del_visor_de_la_nevera_usa_el_plugin_en_nativo():
    ps = _front("src/components/pantry/PantryScanButton.jsx")
    i = ps.index("const handleFallbackToFile = async () => {")
    cuerpo = ps[i:i + 1100]
    assert "if (!isNativeApp()) { fileInputRef.current?.click(); return; }" in cuerpo
    assert "await chooseNativeGalleryImage();" in cuerpo
    assert "if (isNativePickerCancellation(err)) return;" in cuerpo
    assert "action: 'native_gallery_picker'" in cuerpo


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 110
