"""[P1-AJO-SINGLE-ROW · 2026-09-04] El ajo salía en DOS filas de la lista («1 paquete (4 uds.)» + «2 Cabezas
(~10g total)») y el guard de coherencia solo veía la segunda: 6 g contra los 98 g que sumaban las recetas del
dueño, `magnitude_undersupply` en cada recálculo. Una sola fila con la demanda en gramos como base.
"""
from pathlib import Path

import pytest

import shopping_calculator as sc

_BACKEND = Path(__file__).resolve().parents[1]


def test_a_gramos_y_dientes_se_funden_en_un_paquete():
    units, demand = sc._consolidate_ajo_units({"diente": 18.5, "g": 6.01})
    assert units == {"paquete (4 uds.)": 1}, units
    assert demand == pytest.approx(98.51, abs=0.01), "18,5 dientes × 5 g + 6,01 g"


def test_b_cabezas_y_redondeo_a_paquetes_de_4():
    units, demand = sc._consolidate_ajo_units({"cabeza": 5})
    assert units == {"paquete (4 uds.)": 2}, "5 cabezas → 2 paquetes de 4"
    assert demand == pytest.approx(250.0)
    units, _ = sc._consolidate_ajo_units({"diente": 40})
    assert units == {"paquete (4 uds.)": 1}, "40 dientes = 4 cabezas = 1 paquete exacto"


def test_c_claves_ajenas_intactas_y_sin_ajo_no_hay_paquete():
    units, demand = sc._consolidate_ajo_units({"cucharadita": 2})
    assert units == {"cucharadita": 2} and demand == 0
    assert sc._consolidate_ajo_units({}) == ({}, 0)


def test_d_constantes_coherentes_con_el_catalogo():
    # density_g_per_unit=5 del master es POR DIENTE; la cabeza son 10 dientes; el paquete RD, 4 cabezas
    assert sc.AJO_G_PER_DIENTE == 5.0 and sc.AJO_DIENTES_PER_CABEZA == 10.0 and sc.AJO_CABEZAS_PER_PACK == 4
    assert sc.AJO_PACK_GRAMS == 200.0


def test_e_cableado_en_el_aggregator():
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    assert "_ajo_units, _ajo_demand_g = _consolidate_ajo_units(units)" in src
    assert "_packaged_demand_g[name] = _ajo_demand_g" in src
    assert "if str(u).lower().startswith('paquete (') and float(_packaged_demand_g.get(name) or 0) > 0:" in src
    assert 'market_obj["base_unit"] = "g"' in src
    # el dict nace ANTES del primer bucle y se lee en el segundo (mismo function scope)
    assert src.index("_packaged_demand_g: dict = {}") < src.index("_packaged_demand_g[name] = _ajo_demand_g") < src.index("_packaged_demand_g.get(name)")
    # el camino viejo (segunda fila desde units['cabeza'] + units.pop('cabeza')) ya no existe
    assert "units['cabeza'] = units.get('cabeza', 0) + (u_dientes / 10.0)" not in src
