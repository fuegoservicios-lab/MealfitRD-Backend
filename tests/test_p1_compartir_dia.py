"""[P1-COMPARTIR-DIA · 2026-09-23] Compartir las macros y micros del día. El contrato fino vive en
frontend/src/__tests__/{compartirDia,tarjetaDelDia,ShareDaySheet}.test.*; aquí, lo que no debe romper nadie:
la hoja se carga bajo demanda, WhatsApp siempre está y en la app nativa no se ofrece descargar (Capacitor no
gestiona `a.download`, lote 166)."""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def test_la_tarjeta_carga_la_hoja_bajo_demanda():
    tp = _front("src/components/dashboard/TrackingProgress.jsx")
    assert "lazy(() => import('./ShareDaySheet'))" in tp
    assert "aria-label={t('Compartir mi día')}" in tp


def test_whatsapp_siempre_y_descarga_solo_en_web():
    hoja = _front("src/components/dashboard/ShareDaySheet.jsx")
    assert "href={urlWhatsApp(texto)}" in hoja
    assert "const puedeDescargar = !isNativeApp() &&" in hoja
    util = _front("src/utils/compartirDia.js")
    assert "https://wa.me/?text=${encodeURIComponent(texto)}" in util
