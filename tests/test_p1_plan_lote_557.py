# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-557 · 2026-09-27] La restricción religiosa es exclusión DURA, no sólo prompt.

Auditoría del formulario: «Súper personalización → restricción religiosa» (halal, kosher, sin cerdo…) sólo llegaba al
prompt; los cerradores de proteína (catálogo con cerdo, camarones, pulpo) y la última palabra no la conocían: «Sin
mariscos» + ganar músculo ⇒ un cerrador podía añadir camarones.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import constants  # noqa: E402
import rechazos  # noqa: E402
import restricciones_finales as rf  # noqa: E402


def _fd(rel, **kw):
    d = {"allergies": [], "dislikes": [], "dietType": "balanced", "super_personalization": {"religiousRestriction": rel}}
    d.update(kw)
    return d


def test_los_cerradores_la_excluyen():
    r = constants.alergias_y_rechazos(_fd("sin_mariscos"))
    assert "mariscos" in r
    h = constants.alergias_y_rechazos(_fd("halal"))
    assert "cerdo" in h and "jamon" in h and "ron" in h
    assert constants.alergias_y_rechazos(_fd("")) == [] and constants.alergias_y_rechazos(_fd("otra")) == []


def test_la_ultima_palabra_la_ve():
    plan = {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Arroz con pollo y ensalada",
                                           "ingredients": ["150 g de pechuga de pollo", "80 g de camarones",
                                                           "80 g de arroz blanco"],
                                           "ingredients_raw": ["150 g de pechuga de pollo", "80 g de camarones",
                                                               "80 g de arroz blanco"],
                                           "recipe": ["El Toque de Fuego: cocina el pollo 6-8 min hasta 74 °C."]}]}]}
    p = copy.deepcopy(plan)
    out = rf.retirar_prohibidos(p, _fd("kosher"))
    assert [x["line"] for x in out["retiradas"]] == ["80 g de camarones"], out
    p = copy.deepcopy(plan)
    assert rf.retirar_prohibidos(p, _fd(""))["retiradas"] == []


def test_palabra_completa_res_no_toca_las_fresas():
    t = rechazos.terminos_de_rechazo(_fd("sin_res"))
    import graph_orchestrator as go
    for linea, cae in (("1 taza de fresas", False), ("100 g de queso fresco", False), ("100 g de bistec de res", True)):
        v = go._scan_allergen_violations({"days": [{"meals": [{"name": "Plato", "ingredients": [linea]}]}]}, [],
                                         terminos=t)
        assert bool(v) is cae, linea


def test_el_contexto_clinico_del_perfil_la_lleva():
    src = (_BACKEND / "db_profiles.py").read_text(encoding="utf-8")
    assert '"super_personalization": _hp.get("super_personalization"),   # [P1-PLAN-LOTE-557]' in src
