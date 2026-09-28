# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-703 · 2026-09-28] (G85) El separador decimal de la cantidad métrica de la lista sigue al PAÍS.

`_etiqueta_metrica` escribe «1,4 kg» cuando el país pasa la lista a métrico (ES/MX/CO). El frontend repone el
separador al pintar y lo decidía sólo el IDIOMA: un español con la app en es-DO (lo que le da la autodetección) veía
«1.4 kg». Ahora `shoppingHelpers.formatRegionFor(país, idioma)` compone la etiqueta BCP-47 (es + ES → es-ES) y
`_separadorDecimal` la usa cuando hay país. El comportamiento lo prueba `frontend/src/__tests__/lote703.test.js`;
aquí se ancla la frontera backend↔frontend: el docstring no vuelve a justificar la coma con «se lee en español».

tooltip-anchor: P1-PLAN-LOTE-703
"""
from pathlib import Path

import pytest

import shopping_calculator as sc

_FRONT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "utils" / "shoppingHelpers.js"


def test_la_etiqueta_metrica_sigue_con_coma():
    assert sc._etiqueta_metrica(1406.0) == "1,4 kg"
    assert sc._etiqueta_metrica(454.0) == "454 g"


def test_el_docstring_no_confunde_espanol_con_espana():
    doc = sc._etiqueta_metrica.__doc__ or ""
    assert "se lee en español" not in doc
    assert "formatRegionFor" in doc


def test_el_frontend_toma_la_region_del_pais():
    if not _FRONT.exists():
        pytest.skip("frontend ausente (repo hermano)")
    src = _FRONT.read_text(encoding="utf-8")
    # [P1-PLAN-LOTE-708] la definición vive en `paisDelUsuario.js` (sin dependencias); aquí se importa y re-exporta
    assert "export { formatRegionFor };" in src
    assert "export const formatRegionFor" in (_FRONT.parent / "paisDelUsuario.js").read_text(encoding="utf-8")
    i = src.index("const _separadorDecimal")
    assert "formatRegionFor(pais, getLocale())" in src[i:i + 600]
