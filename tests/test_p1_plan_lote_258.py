# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-258 · 2026-09-25] Un rechazo que nombra la CLASE excluye la clase; uno que nombra un ALIMENTO, ese alimento.

Batería rd257: «no me gusta la corvina» (texto libre) rechazó dos veces planes con tilapia y bacalao; con el lote 250
«corvina» es miembro de la clase pescado y el guard de rechazos expandía como una alergia. Las alergias no cambian.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import rechazos  # noqa: E402


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Plato", "ingredients": list(ings)}]}]}


def _cazados(fd, *ings):
    return [v[1] for v in rechazos._scan_dislike_violations(_plan(*ings), fd)]


def test_un_pescado_concreto_no_excluye_los_demas():
    fd = {"dislikes": ["Hongos"], "otherDislikes": "corvina"}
    assert _cazados(fd, "150 g de corvina", "150 g de filete de tilapia", "100 g de bacalao") == ["150 g de corvina"]


@pytest.mark.parametrize("decl", ["Pescado", "pescados", "fish", "seafood", "no me gusta el pescado"])
def test_la_clase_si_excluye_la_clase(decl):
    assert _cazados({"dislikes": [decl]}, "150 g de filete de tilapia"), decl


def test_mariscos_clase_y_miembro():
    assert _cazados({"dislikes": ["Mariscos"]}, "150 g de camarones", "100 g de pulpo") == ["150 g de camarones", "100 g de pulpo"]
    assert _cazados({"dislikes": ["camarones"]}, "150 g de camarones", "100 g de pulpo") == ["150 g de camarones"]


@pytest.mark.parametrize("decl,si,no", [
    ("atún", "120 g de atún en agua", "150 g de salmón"),
    ("queso", "30 g de queso blanco", "1 taza de leche descremada"),
    ("vino", "½ taza de vino blanco", "2 cdas de pasas"),
])
def test_miembros_literales(decl, si, no):
    assert _cazados({"dislikes": [decl]}, si, no) == [si], decl


def test_la_alergia_sigue_por_clase():
    assert go._scan_allergen_violations(_plan("150 g de filete de tilapia"), ["corvina"])


def test_la_nota_del_revisor_no_habla_de_rechazo_al_pescado():
    import etiquetas_clinicas as ec
    assert ec._pescado_sin_mariscos({"dislikes": ["Hongos"], "otherDislikes": "corvina"}) == ""
    assert ec._pescado_sin_mariscos({"dislikes": ["Pescado"]}) == "rechazo"


def test_la_ultima_palabra_y_el_dia_determinista():
    import deterministic_day as dd
    import restricciones_finales as rf
    plan = _plan("150 g de filete de tilapia", "80 g de corvina")
    plan["days"][0]["meals"][0]["name"] = "Pescado al horno"
    p = copy.deepcopy(plan)
    rf.retirar_prohibidos(p, {"allergies": [], "dislikes": ["corvina"]})
    assert p["days"][0]["meals"][0]["ingredients"] == ["150 g de filete de tilapia"]
    _al, excl = dd.restricciones_del_formulario({}, {"dislikes": ["corvina"]})
    assert "corvina" in excl and "tilapia" not in excl


def test_ancla():
    assert "P1-PLAN-LOTE-258-RECHAZO-LITERAL" in (_BACKEND / "rechazos.py").read_text(encoding="utf-8")
    assert "def _scan_allergen_violations(plan: dict, allergies, terminos=None) -> list:" in \
        (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 258
