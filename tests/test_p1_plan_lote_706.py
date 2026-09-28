# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-706 · 2026-09-28] (G36, pasos 1-2) Los tests de países de vitest no fijan el VALOR del build.

Producción corre con `VITE_COUNTRY_SYSTEM=true` desde el 18-ago; el runner no la define. Con la env encendida, 7
aserciones fallaban contra código correcto porque fijaban `COUNTRY_SYSTEM_UI === false`. Ahora:
  · `countries.p1_country_system_f0` ancla la REGLA (la bandera es la env 1/true);
  · `QBudget.p1_country_system_f1` compara el default con la bandera del build, sea cual sea;
  · la rama de ROLLBACK (`QCountry…_dark`, `useBudgetFloor…`) se fuerza con `vi.mock` explícito.
Medido: los 16 ficheros de países → 121/121 con la env encendida Y apagada. El paso 3 (encenderla en el runner)
necesita la suite entera y queda para una ventana sin gates.

tooltip-anchor: P1-PLAN-LOTE-706
"""
from pathlib import Path

import pytest

_T = Path(__file__).resolve().parents[2] / "frontend" / "src" / "__tests__"
_MOCK_APAGADA = "return { ...actual, COUNTRY_SYSTEM_UI: false };"


def _leer(nombre):
    p = _T / nombre
    if not p.exists():
        pytest.skip("frontend ausente (repo hermano)")
    return p.read_text(encoding="utf-8")


def test_countries_ancla_la_regla_y_no_el_valor():
    src = _leer("countries.p1_country_system_f0.test.js")
    assert "expect(COUNTRY_SYSTEM_UI).toBe(false)" not in src
    assert "expect(COUNTRY_SYSTEM_UI).toBe(['1', 'true'].includes(env))" in src


def test_qbudget_compara_con_la_bandera_del_build():
    src = _leer("QBudget.p1_country_system_f1.test.jsx")
    assert "effectiveBudgetCurrency('ES', 'EUR', COUNTRY_SYSTEM_UI)" in src


@pytest.mark.parametrize("nombre", ["QCountry.p1_country_system_f2_dark.test.jsx",
                                    "useBudgetFloor.p1_country_system_f2.test.jsx"])
def test_la_rama_de_rollback_se_fuerza_con_mock(nombre):
    assert _MOCK_APAGADA in _leer(nombre)
