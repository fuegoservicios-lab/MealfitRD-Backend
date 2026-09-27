# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-446 · 2026-09-27] El plan de EMERGENCIA cocina lo que compra y no cocina lo que se come crudo.

Replay de 322 planes: los 16 «Huevos y Avena» no cocinaban sus 2 huevos (la rama de la avena iba primero); toda proteína
iba a «sazona la proteína… a la plancha 6-8 minutos por lado» (el pescado 12-16 minutos, el atún EN AGUA a la plancha);
y el genérico «cocinaba» 10-12 minutos el yogur, la fruta y la ensalada."""
from __future__ import annotations

import pathlib
import re

import fallback_pools as fp
import graph_orchestrator as go

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_TODAS = [(slot, name, ings) for pools in (fp._FALLBACK_MEAL_POOLS, fp._FALLBACK_MEAL_POOLS_BARIATRIC)
          for slot, lst in pools.items() for name, _t, _d, ings in lst]


def _p(slot, ings):
    return go._fallback_recipe_steps(slot, ings)


def test_huevos_y_avena_cocina_los_huevos():
    p = _p("Desayuno", ["2 huevos", "1/2 taza de avena cocida", "1 fruta de temporada"])
    assert "cuájalos" in p[1] and "yema y clara estén firmes" in p[1], p
    assert "calienta la avena cocida" in p[1] and "huevos revueltos" in p[2]


def test_cada_proteina_con_su_punto():
    pez = _p("Almuerzo", ["filete de pescado", "arroz blanco", "ensalada verde"])[1]
    assert "3-4 minutos por lado" in pez and "63 °C" in pez and "el filete de pescado" in pez, pez
    res = _p("Almuerzo", ["carne de res magra", "arroz blanco", "vegetales al gusto"])[1]
    assert "la carne de res" in res and "cocínala" in res and "71 °C" in res, res
    pollo = _p("Cena", ["pechuga de pollo", "vegetales al vapor", "batata asada"])[1]
    assert "74 °C" in pollo and "asa la batata en el horno" in pollo, pollo
    guisado = _p("Almuerzo", ["120g pollo guisado", "90g yuca cocida", "70g tayota cocida"])[1]
    assert guisado.startswith("El Toque de Fuego: guisa el pollo") and "plancha" not in guisado, guisado


def test_lo_enlatado_se_escurre_y_lo_listo_no_se_cocina():
    atun = _p("Merienda PM", ["80g atun en agua escurrido", "20g casabe"])
    assert "escurre bien el atun" in atun[1] and "plancha" not in atun[1], atun
    yog = _p("Merienda Nocturna", ["115g yogurt griego"])
    assert "no lleva cocción" in yog[1] and "10-12 minutos" not in yog[1], yog
    queso = _p("Merienda AM", ["30g queso blanco", "60g manzana"])
    assert "no lleva cocción; corta la manzana" in queso[1], queso


def test_ninguna_plantilla_deja_barras_ni_la_proteina():
    for slot, name, ings in _TODAS:
        p = _p(slot, ings)
        assert len(p) == 3 and p[0].startswith("Mise en place") and p[1].startswith("El Toque de Fuego") \
            and p[2].startswith("Montaje"), (name, p)
        assert "la proteína" not in " ".join(p) and not re.search(r"\w/\w", " ".join(p)), (name, p)


def test_contrato_de_siempre():
    """test_p2_objective_batch_2 lo fija: pollo a la plancha con minutos, licuado, avena."""
    assert any("plancha" in s for s in _p("Almuerzo", ["150g de pollo", "1 taza de arroz"]))
    assert any("licúa" in s for s in _p("Merienda", ["1 taza de leche", "1 guineo", "batido de fresa"]))


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'return __import__("receta_emergencia").pasos(meal_type, ingredients)' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-446" in (_BACKEND / "receta_emergencia.py").read_text(encoding="utf-8")
