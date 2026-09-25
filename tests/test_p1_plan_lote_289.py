# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-289 · 2026-09-25] Sin congelador, lo «congelado» no dura; y «salmón» no son hamburguesas.

Batería real del formulario del dueño (30 días de una vez, SIN congelador): el D2 llevaba «250 g de salmón previamente
congelado y apto para consumo crudo». La proyección del mes lo tomó por duradero (el texto dice «congelad») y lo copió
en todos los días → 2,5 kg de salmón, que el súper resolvió a su SKU más barato: «4 fundas (Hamburguesas 680 gr · Wala)»."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compra_unica as cu  # noqa: E402
import shopping_calculator as sc  # noqa: E402

_SIN = {"need_days": 11, "allow_frozen": False, "freezer_mode": "none"}
_CON = {"need_days": 11, "allow_frozen": True, "freezer_mode": "full"}
_SALMON = "250 g de salmón previamente congelado y apto para consumo crudo"
SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none", "batch_cooking": "never"},
          "diet": {"type": "balanced", "allergies": []}}


@pytest.fixture(autouse=True)
def _limpio():
    yield
    sc.set_single_trip_notes(False)
    cu._MEMO.clear()


def test_congelado_solo_dura_con_congelador():
    assert not cu._aguanta(_SALMON, 10, _SIN)
    assert cu._aguanta(_SALMON, 10, _CON)                         # con congelador lo congela al comprarlo (pantry_durability)
    assert not cu._aguanta("300 g de camarones congelados", 10, _SIN)
    assert cu._aguanta("300 g de camarones congelados", 10, _CON)
    assert cu._aguanta("1 lata de atún en agua", 10, _SIN)        # la lata sigue siendo despensa
    assert cu._aguanta("40 g de lentejas secas", 10, _SIN)


def test_el_salmon_se_sustituye_despues_de_su_plazo():
    r = cu.sustituir_linea(_SALMON, 10, _SIN, semilla=0)
    assert r and r[0].startswith("250 g de ") and "salm" not in r[0], r


def test_la_proyeccion_no_copia_el_salmon_fresco_al_mes():
    ings = ["1 taza de arroz blanco", _SALMON]
    p = {"total_days_requested": 30, "_plan_policy": {"effective": SINGLE},
         "days": [{"day": n, "meals": [{"meal": "Almuerzo", "name": "Bowl de salmón", "ingredients": list(ings),
                                        "ingredients_raw": list(ings)}]} for n in (1, 2, 3)]}
    dias = sc.shopping_source_days(p)
    resto = [x for d in dias[3:] for m in d["meals"] for x in m["ingredients_raw"]]
    assert not any("salm" in cu._sa(x) for x in resto), [x for x in resto if "salm" in cu._sa(x)][:3]


def test_el_super_no_resuelve_a_la_forma_procesada():
    defaults = {"salmon": [{"grams": 680.0, "price": 550.0, "label": "Hamburguesas 680 gr · Wala", "unit": "funda"},
                           {"grams": 454.0, "price": 690.0, "label": "Filete 1 Lb · Genérico", "unit": "paquete"}],
                "filete de pescado blanco": [{"grams": 907.0, "price": 550.0, "label": "Empanizado 32 Oz · Panamei", "unit": "funda"}]}
    got = sc._resolve_brand_default("Salmón", defaults)
    assert got and [p["label"] for p in got] == ["Filete 1 Lb · Genérico"], got
    assert sc._resolve_brand_default("Filete de pescado blanco", defaults) is None     # solo había empanizado
