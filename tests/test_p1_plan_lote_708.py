# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-708 · 2026-09-28] (G84) En EE. UU. y Puerto Rico, las onzas al lado de los gramos y mililitros del plan.

El plan persiste «180 g de Pechuga de pollo»: la cantidad y el nombre son identificadores del motor (Nevera, lista,
alergias) y no se tocan. Como los °F del lote 650, el frontend añade «(6.3 oz)» sólo al pintar (`conOnzas` en
`nombresDelPais.js`, aplicado por `displayMeal.mealDisplay` a ingredientes y pasos), en cualquier idioma, sólo con
país US/PR. El comportamiento lo prueba `frontend/src/__tests__/lote708.test.js`; aquí, la frontera: el backend no
reescribe unidades del plan para ningún país (la conversión es de display).

tooltip-anchor: P1-PLAN-LOTE-708
"""
from pathlib import Path

import pytest

_UTILS = Path(__file__).resolve().parents[2] / "frontend" / "src" / "utils"


def _src(nombre):
    p = _UTILS / nombre
    if not p.exists():
        pytest.skip("frontend ausente (repo hermano)")
    return p.read_text(encoding="utf-8")


def test_la_conversion_es_de_display_y_va_con_el_pais():
    src = _src("nombresDelPais.js")
    assert "export function conOnzas(texto, pais = getPaisDelUsuario())" in src
    assert "PAISES_FAHRENHEIT.has(pais)" in src[src.index("export function conOnzas"):]


def test_mealdisplay_la_aplica_a_ingredientes_y_pasos():
    src = _src("displayMeal.js")
    assert "conOnzasValor(conFahrenheitValor(d.recipe))" in src
    assert "conOnzasValor(d.ingredients)" in src
