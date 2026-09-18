# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-100 · 2026-09-18] Al cerrar «Registrar comida», el dashboard vuelve a donde estaba.

En el iPhone, para revelar el campo enfocado iOS desplaza el documento de fondo aunque el body lleve
`overflow: hidden`; al cerrar la hoja la página aparecía movida «un poco hacia abajo». Se recuerda `window.scrollY`
al abrir y se restaura al desmontar. Ancla cross-repo; el contrato fino vive en `LogMealModal.lote99.test.jsx`."""
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


def test_el_scroll_del_fondo_se_recuerda_y_se_restaura():
    jsx = _front("src/components/dashboard/LogMealModal.jsx")
    assert "const scrollY0 = window.scrollY;" in jsx
    assert "if (Math.abs(window.scrollY - scrollY0) > 1) window.scrollTo(0, scrollY0);" in jsx
    i = jsx.index("vv.removeEventListener('resize', alCambiar);")
    assert "restaurarScroll();" in jsx[i:i + 120], "la limpieza del efecto restaura el scroll"


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 100
    assert "P1-PLAN-LOTE-100" in (_BACKEND / "docs" / "diario_registrar_comida.md").read_text(encoding="utf-8")
