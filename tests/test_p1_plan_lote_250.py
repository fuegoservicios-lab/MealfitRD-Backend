# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-250 · 2026-09-25] El escáner reconoce los pescados y mariscos de mercado que faltaban.

Con «Pescado» marcado, «150 g de corvina» pasaba limpio por el backstop de alérgenos, el guard de dieta de un
vegetariano y el de rechazos (los tres leen el mismo vocabulario). «Frutos de mar» / «frutos del mar» —la forma
latinoamericana de declararlo en «Otra alergia»— no se resolvía a ninguna clase.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import rechazos  # noqa: E402
import vocabulario_mar as vm  # noqa: E402


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Plato", "ingredients": list(ings)}]}]}


@pytest.mark.parametrize("termino", vm.PESCADOS_EXTRA)
def test_pescado_nuevo_lo_ven_alergia_dieta_y_rechazo(termino):
    plan = _plan(f"150 g de {termino}")
    assert go._scan_allergen_violations(plan, ["Pescado"]), termino
    assert go._scan_diet_violations(plan, "vegetariano"), termino
    assert rechazos._scan_dislike_violations(plan, {"dislikes": ["Pescado"]}), termino
    assert not go._scan_allergen_violations(plan, ["Mariscos"]), termino  # lote 210: el pescado no es marisco


@pytest.mark.parametrize("termino", vm.MARISCOS_EXTRA)
def test_marisco_nuevo_lo_ven_alergia_dieta_y_rechazo(termino):
    plan = _plan(f"150 g de {termino}")
    assert go._scan_allergen_violations(plan, ["Mariscos"]), termino
    assert go._scan_diet_violations(plan, "vegano"), termino
    assert rechazos._scan_dislike_violations(plan, {"dislikes": ["Mariscos"]}), termino
    assert not go._scan_allergen_violations(plan, ["Pescado"]), termino


@pytest.mark.parametrize("ing", ["200 g de corvinas", "2 filetes de pargo rojo", "150 g de chipirones",
                                 "100 g de berberechos en lata", "120 g de pez espada", "1 taza de ostiones"])
def test_plurales_y_formas_reales(ing):
    assert go._scan_allergen_violations(_plan(ing), ["Pescado", "Mariscos"]), ing


@pytest.mark.parametrize("decl", ["frutos de mar", "Frutos del mar", "alergia a los frutos del mar"])
def test_frutos_de_mar_declara_mariscos(decl):
    assert go._scan_allergen_violations(_plan("150 g de camarones"), [decl])
    assert not go._scan_allergen_violations(_plan("150 g de filete de tilapia"), [decl])


@pytest.mark.parametrize("ing", ["2 dientes de ajo machacado", "100 g de carpaccio de res", "1 taza de pasta caracolitos",
                                 "1 taza de conchas de pasta", "60 g de chorizo", "1 cebolla dorada picada",
                                 "1 cucharada de aceite de oliva", "½ taza de arroz blanco"])
def test_sin_falsos_positivos(ing):
    assert not go._scan_allergen_violations(_plan(ing), ["Pescado", "Mariscos"]), ing


def test_la_nota_de_mariscos_ve_la_corvina():
    import etiquetas_clinicas as ec
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Corvina al horno",
                                            "ingredients": ["150 g de corvina", "1 taza de arroz"],
                                            "recipe": ["Hornea la corvina 15 min."]}]}]}
    assert ec._nota_mariscos_no_pescado(plan, {"allergies": ["Mariscos"]}) == 1


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('__import__("vocabulario_mar")') == 4
    assert "P1-PLAN-LOTE-250-VOCABULARIO-MAR" in (_BACKEND / "vocabulario_mar.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 250
