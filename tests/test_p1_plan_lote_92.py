# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-92 · 2026-09-17] El diseño de teléfono del contador no puede depender de un acantilado en 480px.

Reporte del dueño con captura de su iPhone: «se ve estrecho, mira todo el espacio que tiene los bordes de los lados, hay
vacíos». Comprobado en el log del servidor que su teléfono había cargado el JS y el CSS de la release vigente y que ese JS
pasa `flatOnMobile`; medido sobre la captura, el contenido ocupaba ~75 % del ancho — que es exactamente lo que se ve cuando
el bloque `@media (max-width: 480px)` NO casa (tarjeta con relleno y borde, página a 0,9rem). Basta con que el usuario baje
el zoom del sitio para que su viewport CSS pase de 480, y el diseño entero se caía por eso.

El corte del teléfono pasa a ser 768: el MISMO con el que el armazón (`DashboardLayout`) se vuelve teléfono."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"
_DASH = "src/components/dashboard/"
_MODULOS = (_DASH + "TrackingProgress.module.css", _DASH + "WaterTracker.module.css", _DASH + "DashboardTracking.module.css")


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


def _sin_comentarios(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _cabecera_del_bloque(rel: str, aguja: str) -> str:
    """La cabecera del `@media` que contiene `aguja`."""
    css = _sin_comentarios(_front(rel))
    i = css.index(aguja)
    ini = css.rindex("@media", 0, i)
    return css[ini:css.index("{", ini)].strip()


def test_las_tres_reglas_del_telefono_estan_en_768():
    for rel, aguja in zip(_MODULOS, (".card.flatMobile", ".card.flatMobile", "padding: 0.35rem 0.15rem 0.5rem")):
        assert _cabecera_del_bloque(rel, aguja) == "@media (max-width: 768px)", rel


def test_ningun_modulo_deja_el_aplanado_en_un_bloque_de_480():
    for rel in _MODULOS:
        css = _sin_comentarios(_front(rel))
        i = css.find("@media (max-width: 480px) {")
        if i < 0:
            continue
        cuerpo = css[i:css.index("\n}\n", i)]
        for prohibido in ("flatMobile", "0.35rem 0.15rem", ":has("):
            assert prohibido not in cuerpo, f"{rel}: {prohibido} volvió al bloque de 480"


def test_768_es_el_mismo_corte_del_armazon():
    """Si el armazón cambia su frontera de teléfono, este test cae antes que el diseño en el teléfono de alguien."""
    layout = _sin_comentarios(_front(_DASH + "DashboardLayout.module.css"))
    i = layout.rindex("@media (max-width: 768px) {")
    assert ".mainContent { padding: 0.65rem 0.85rem;" in re.sub(r"\s+", " ", layout[i:])


def test_el_lote_87_no_se_toca():
    """El bloque de 480 sigue existiendo para lo que SÍ es cuestión de pantalla pequeña (aire de la tarjeta en modo plan)."""
    css = _front(_DASH + "TrackingProgress.module.css")
    i = css.index("@media (max-width: 480px) {")
    bloque = css[i:css.index("\n}\n", i)]
    assert "padding: 1.15rem;" in bloque and "P1-PLAN-LOTE-87" in bloque


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 92
    assert "P1-PLAN-LOTE-92" in (_BACKEND / "docs" / "modo_seguimiento_ui.md").read_text(encoding="utf-8")
