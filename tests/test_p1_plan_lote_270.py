# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-270 · 2026-09-25] Dudas de marca = aviso; nota de etiqueta; el aderezo César.

Batería final y verificación: el revisor rechazó por «el pan integral tiene alta probabilidad de contener leche», «la
tortilla no está especificada como libre de lecitina de soya» y «el wrap no especifica la envoltura; si contiene
trigo…» — dudas de MARCA que el generador no puede zanjar y que costaban un reintento cada una.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import etiqueta_alergenos as ea  # noqa: E402
import graph_orchestrator as go  # noqa: E402


@pytest.mark.parametrize("issue", [
    "Día 1, desayuno: el pan integral familiar tiene alta probabilidad de contener leche o derivados y no está "
    "identificado como libre de lácteos; no es seguro con la alergia declarada.",
    "Día 3: la tortilla de trigo integral no está especificada como libre de lecitina, aceite o proteína de soya.",
    "El almuerzo del día 3 se titula «Wrap», pero no especifica el ingrediente de la envoltura. Si contiene una tortilla "
    "de trigo u otra fuente de gluten, no es seguro para la alergia declarada.",
])
def test_duda_de_marca_es_aviso(issue):
    aprobado, reales, sev, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "high")
    assert aprobado is True and reales == [] and avisos == [issue]


def test_un_defecto_real_sigue_siendo_rechazo():
    real = "El desayuno incluye 30 g de queso cheddar, alérgeno declarado (lácteos)."
    aprobado, reales, sev, avisos = go._downgrade_reviewer_verification_demands(False, [real], "critical")
    assert aprobado is False and reales == [real]


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Desayuno", "name": "x", "ingredients": list(ings),
                                            "recipe": ["Montaje: sirve."]}]}]}


def test_nota_para_lacteos_y_soya():
    plan = _plan("2 rebanadas de pan integral", "2 lonjas de salami", "1 huevo")
    ea.anotar_plan(plan, {"allergies": ["Lácteos", "Soya"]})
    nota = plan["days"][0]["meals"][0]["recipe"][-1]
    assert nota.startswith("⚠️ Alergia declarada: elige ")
    assert "pan sin lácteos ni soya" in nota and "embutidos sin lácteos ni soya" in nota
    assert __import__("recipe_contract")._es_nota(nota)          # los pases que leen los pasos la saltan


def test_idempotente_y_solo_con_producto_y_alergia():
    plan = _plan("2 rebanadas de pan integral")
    ea.anotar_plan(plan, {"allergies": ["Lácteos"]})
    ea.anotar_plan(plan, {"allergies": ["Lácteos"]})
    assert sum(p.startswith("⚠️ Alergia declarada: elige") for p in plan["days"][0]["meals"][0]["recipe"]) == 1
    sin = _plan("2 rebanadas de pan integral")
    antes = copy.deepcopy(sin)
    ea.anotar_plan(sin, {"allergies": ["Ninguna"]})
    assert sin == antes
    sin_producto = _plan("1 huevo", "1 taza de arroz")
    antes = copy.deepcopy(sin_producto)
    ea.anotar_plan(sin_producto, {"allergies": ["Lácteos"]})
    assert sin_producto == antes


def test_gluten_avisa_el_cubito_no_el_pan():
    plan = _plan("1 cubito de pollo", "2 tortas de casabe")
    ea.anotar_plan(plan, {"allergies": ["Gluten"]})
    assert "caldo en cubito sin gluten" in plan["days"][0]["meals"][0]["recipe"][-1]


def test_lo_tecleado_a_mano_cuenta():
    plan = _plan("1 barra de chocolate oscuro")
    ea.anotar_plan(plan, {"allergies": [], "otherAllergies": "maní"})   # (con «Ninguna» el texto se descarta: P0-FORM-1)
    assert "chocolate sin maní" in plan["days"][0]["meals"][0]["recipe"][-1]


@pytest.mark.parametrize("chip", ["Pescado", "Huevo", "Lacteos"])
def test_el_cesar_lleva_anchoas_yema_y_parmesano(chip):
    plan = {"days": [{"meals": [{"name": "x", "ingredients": ["2 cdas de aderezo César"]}]}]}
    assert go._scan_allergen_violations(plan, [chip])


def test_el_cesar_no_es_vegetariano():
    assert go._scan_diet_violations({"days": [{"meals": [{"name": "x", "ingredients": ["1 taza de ensalada César"]}]}]},
                                    "vegetariana")


@pytest.mark.parametrize("chip, linea", [("Hígado", "80 g de higaditos de pollo"), ("Aguacate", "½ palta")])
def test_rechazos_formas_alternativas(chip, linea):
    import rechazos
    plan = {"days": [{"meals": [{"name": "x", "ingredients": [linea]}]}]}
    assert rechazos._scan_dislike_violations(plan, {"dislikes": [chip]})


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("etiqueta_alergenos").anotar_plan(plan, form_data)' in src
    assert "P1-PLAN-LOTE-270-ETIQUETA" in (_BACKEND / "etiqueta_alergenos.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 270
