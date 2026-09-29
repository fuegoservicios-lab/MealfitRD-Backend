# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-920 · 2026-09-29] «desmenuza los el queso blanco restante restante»: la 2.ª mención de un alimento
repartido pasa a «el X restante» sin chocar con el artículo que ya tenía ni repetir «restante».

Batería real del 29-sep (el formulario del dueño, cena D3) y 8 de 5.301 comidas del corpus y los replays.
"""
from __future__ import annotations

import pathlib
import sys

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402


def _sync(lista, pasos):
    meal = {"meal": "Cena", "name": "Plato", "ingredients": list(lista), "recipe": list(pasos)}
    go._sync_recipe_step_quantities(meal)
    return meal["recipe"]


def test_el_articulo_viejo_se_va_y_restante_no_se_repite():
    rec = _sync(["20 g de queso blanco fresco", "65 g de harina de maíz precocida"],
                ["Mise en place: mide 20 g de queso blanco fresco y 65 g de harina de maíz precocida.",
                 "Montaje: sirve las arepitas y desmenuza los 20 g de queso blanco fresco restante por encima."])
    assert rec[1] == "Montaje: sirve las arepitas y desmenuza el queso blanco fresco restante por encima.", rec[1]


def test_el_nombre_no_se_traga_durante_unos():
    rec = _sync(["120 ml de agua", "2 cdas de semillas de chía"],
                ["Mise en place: mide 120 ml de agua y 2 cdas de semillas de chía.",
                 "Montaje: hidrata la chía con las 120 ml de agua durante unos 5 minutos más."])
    assert "con el agua restante durante unos 5 minutos más" in rec[1], rec[1]
    assert "las el" not in rec[1] and "restante minutos" not in rec[1]


def test_el_plural_concuerda():
    rec = _sync(["60 g de claras de huevo", "1 cdta de aceite de oliva"],
                ["Mise en place: mide 60 g de claras de huevo.",
                 "El Toque de Fuego: calienta el aceite y cocina las 60 g de claras de huevo restantes 2-3 minutos."])
    assert "cocina las claras de huevo restantes 2-3 minutos" in rec[1], rec[1]
    assert "las el" not in rec[1] and "restante restante" not in rec[1]


def test_la_cola_limpia_lo_que_ya_salio_roto():
    import restante_sin_choque as rsc
    casos = {
        "Montaje: sirve las arepitas, desmenuza los el queso blanco restante restante por encima.":
            "Montaje: sirve las arepitas, desmenuza el queso blanco restante por encima.",
        "El Toque de Fuego: cocina los el bulgur restante en agua hirviendo.":
            "El Toque de Fuego: cocina el bulgur restante en agua hirviendo.",
        "Montaje: sirve el pastel con las el pan integral restante.": "Montaje: sirve el pastel con el pan integral restante.",
        "Montaje: hidrata la chía con las el agua durante unos restante minutos.":
            "Montaje: hidrata la chía con el agua restante durante unos minutos.",
        # corpus: el plural, el número del tiempo detrás del «restante», «a el»
        "El Toque de Fuego: pocha 3 huevos y 3 claras de huevo y las el claras restante en agua apenas hirviendo.":
            "El Toque de Fuego: pocha 3 huevos y 3 claras de huevo y las claras restantes en agua apenas hirviendo.",
        "El Toque de Fuego: a fuego medio, dora la el pan integral 1 restante minuto por lado.":
            "El Toque de Fuego: a fuego medio, dora el pan integral restante 1 minuto por lado.",
        "El Toque de Fuego: tuesta la el pan integral 1 restante-2 min por lado y resérvalas.":
            "El Toque de Fuego: tuesta el pan integral restante 1-2 min por lado y resérvalas.",
        "El Toque de Fuego: en el mismo sartén, tuesta la el pan integral de restante 1 a 2 minutos por lado.":
            "El Toque de Fuego: en el mismo sartén, tuesta el pan integral restante de 1 a 2 minutos por lado.",
        "Montaje: sirve el revoltillo junto a la el pan integral restante y la lechosa.":
            "Montaje: sirve el revoltillo junto al pan integral restante y la lechosa.",
        "Montaje: coloca la la lechuga restante y el tomate.": "Montaje: coloca la lechuga restante y el tomate.",
        "El Toque de Fuego: hierve la yautía en el agua durante 15 restante-18 minutos, hasta que esté tierna.":
            "El Toque de Fuego: hierve la yautía en el agua restante durante 15-18 minutos, hasta que esté tierna.",
    }
    for antes, despues in casos.items():
        m = {"recipe": [antes]}
        assert rsc.limpiar_pasos(m) == 1 and m["recipe"][0] == despues, m["recipe"][0]
        assert rsc.limpiar_pasos(m) == 0, "idempotente"
    for bien in ("Montaje: sirve con el aceite restante y la sal.", "Montaje: los tomates restantes, en rodajas."):
        assert rsc.limpiar(bien) == bien
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert '__import__("restante_sin_choque").limpiar_pasos(meal)  # [P1-PLAN-LOTE-920]' in src


def test_knob_apagado_conducta_previa(monkeypatch):
    monkeypatch.setenv("MEALFIT_RESTANTE_SIN_CHOQUE", "false")
    rec = _sync(["20 g de queso blanco fresco"],
                ["Mise en place: mide 20 g de queso blanco fresco.",
                 "Montaje: desmenuza los 20 g de queso blanco fresco restante por encima."])
    assert "los el queso blanco fresco restante restante" in rec[1], rec[1]


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("restante_sin_choque").resto(mm, _core, _tail)  # [P1-PLAN-LOTE-920]' in src
    assert '__import__("restante_sin_choque").patron(_STEP_QTY_MENTION_RE).sub(_sub_rest, _st)  # [P1-PLAN-LOTE-920]' in src
    assert "tooltip-anchor: P1-PLAN-LOTE-920" in (_BACKEND / "restante_sin_choque.py").read_text(encoding="utf-8")
