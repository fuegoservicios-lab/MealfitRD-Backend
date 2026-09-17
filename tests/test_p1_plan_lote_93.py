# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-93 · 2026-09-17] En el teléfono, el ancho del dashboard lo decidía su texto más largo.

Reporte del dueño con capturas y la pista exacta: «se ve estrecho, mira todo el espacio que sobra… ese problema sucede
cuando se le da a "Ahora no"». Reproducido montando el ARMAZÓN REAL (`DashboardLayout` + `DashboardTracking`) en el arnés:
con la tarjeta de invitación el contenido medía 360 px de 392; al descartarla, 294 px centrados con ~33 px muertos a cada
lado.

`.mainContent` lleva `max-width: 1200px; margin: 0 auto` (base) para centrar la columna en escritorio, donde es un bloque
normal. En ≤1024 su padre pasa a `display: flex; flex-direction: column`, y un ítem flex con márgenes AUTOMÁTICOS en el eje
transversal deja de estirarse: se dimensiona a su contenido. Afectaba a TODAS las páginas del dashboard."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_CSS = "src/components/dashboard/DashboardLayout.module.css"


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _sin_comentarios(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _regla(css: str, bloque: str | None, selector: str) -> str:
    i = css.index(bloque) if bloque else 0
    j = css.index(selector, i)
    return re.sub(r"\s+", " ", css[j:css.index("}", j)])


def test_en_el_telefono_el_contenido_declara_su_ancho():
    css = _sin_comentarios(_front(_CSS))
    assert "width: 100%;" in _regla(css, "@media (max-width: 1024px) {", ".mainContent {")


def test_el_padre_es_quien_crea_el_contexto_flex():
    """Si `.mainWrapper` deja de ser flex en móvil, el `width` sobra — pero el test avisa de que el razonamiento cambió."""
    css = _sin_comentarios(_front(_CSS))
    assert "flex-direction: column;" in _regla(css, "@media (max-width: 1024px) {", ".mainWrapper {")


def test_el_centrado_de_escritorio_sigue_intacto():
    """Es justamente `margin: 0 auto` lo que obliga a declarar el ancho: no se quita, se completa."""
    base = _regla(_sin_comentarios(_front(_CSS)), None, ".mainContent {")
    assert "max-width: 1200px;" in base and "margin: 0 auto;" in base


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 93
    assert "P1-PLAN-LOTE-93" in (_BACKEND / "docs" / "armazon_ancho_movil.md").read_text(encoding="utf-8")
