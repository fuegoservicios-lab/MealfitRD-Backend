# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-109 · 2026-09-19] El aviso «Instala Bioboros / Agregar a inicio» se retira.

Lo pidió el dueño con la app ya en TestFlight: no se empuja más la PWA, y dentro del WebView nativo el aviso salía
igual (WKWebView no es `standalone`), pidiendo instalar lo que ya estaba instalado. Se borra el componente, su montaje
en App.jsx y sus cinco claves de traducción."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def test_el_aviso_de_instalacion_no_existe_ni_se_monta():
    if not (_FRONT / "src").exists():
        pytest.skip("sin el repo del frontend al lado")
    assert not (_FRONT / "src" / "components" / "IOSInstallPrompt.jsx").exists()
    app = (_FRONT / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "<IOSInstallPrompt" not in app and "import IOSInstallPrompt" not in app
    assert "P1-PLAN-LOTE-109" in app
    for cat in (_FRONT / "src" / "i18n" / "locales").glob("*.json"):
        txt = cat.read_text(encoding="utf-8")
        assert "«Agregar a inicio»" not in txt and '"Instala {app}"' not in txt, cat.name


def test_marcador():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 109
