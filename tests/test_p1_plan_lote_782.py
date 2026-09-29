# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-782 · 2026-09-28] El cerrador de proteína ya no ve «agua» (del atún en agua) dentro de «aguacate».

Corpus de la cola 744: 86 de 1.244 desayunos con 35-300 g de atún que el modelo no escribió, 85 llamados «…, aguacate y
atún en agua». La congruencia del cerrador comparaba por subcadena: «agua» ⊂ «aguacate» ⇒ «el atún ya está en el
plato» ⇒ se añadía atún antes de mirar huevo o lácteo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import congruencia_cerrador as cc  # noqa: E402

_GEN = ("queso", "carne", "pescado", "filete", "pechuga", "yogur", "yogurt", "proteina", "fresco", "blanco")


def test_agua_no_esta_en_aguacate():
    plato = "mangu de platano verde con huevo bien cocido y aguacate 1 platano verde 2 huevos 50 g de aguacate"
    assert not cc.congruente("atun en agua", plato, _GEN)


def test_el_atun_que_si_esta_sigue_siendo_congruente():
    assert cc.congruente("atun en agua", "ensalada de atun 80 g de atun en aceite 1 tomate", _GEN)
    assert cc.congruente("sardinas en lata", "tostadas con sardina 1 sardina en salsa", _GEN)
    assert cc.congruente("camarones", "arroz con camaron 100 g de camaron", _GEN)


def test_palabras_que_describen_no_identifican():
    assert not cc.congruente("pollo", "ensalada de repollo 1 taza de repollo", _GEN)
    assert not cc.congruente("guisantes secos", "yogur con frutos secos 20 g de frutos secos", _GEN)
    assert not cc.congruente("habichuelas rojas", "tostada con cebollas rojas", _GEN)
    assert not cc.congruente("queso de hoja", "habichuelas guisadas 1 hoja de laurel", _GEN)
    assert not cc.congruente("queso de freir", "aceite para freir 2 platanos", _GEN)


def test_el_nombre_completo_si_cuenta():
    assert cc.congruente("queso de hoja", "arepa con queso de hoja 30 g de queso de hoja", _GEN)
    assert cc.congruente("queso de freir", "mangu con queso de freir 40 g de queso de freir", _GEN)
    assert cc.congruente("queso mozzarella", "pizza casera 60 g de mozzarella", _GEN)


def test_todo_generico_nunca_es_congruente_como_antes():
    assert not cc.congruente("queso blanco", "mangu con queso blanco 60 g de queso blanco", _GEN)


def test_el_cerrador_elige_por_la_franja_y_no_por_el_aguacate():
    import graph_orchestrator as g

    class _Info:
        def __init__(self, name, protein, kcal):
            self.name, self.protein, self.carbs, self.fats, self.kcal = name, protein, 1.0, 5.0, kcal

    cands = [(0.224, "Atún", _Info("Atún en agua", 19.0, 85.0)),
             (0.075, "Queso Mozzarella", _Info("Queso Mozzarella", 22.2, 297.0))]
    meal = {"meal": "Desayuno", "name": "Mangú de plátano verde con huevo bien cocido y aguacate",
            "protein": 12, "carbs": 55, "fats": 14, "cals": 420,
            "ingredients": ["1 plátano verde", "2 huevos", "50 g de aguacate", "1 cdta de aceite de oliva"],
            "recipe": ["El Toque de Fuego: hierve el plátano 15 min y cocina los huevos en la sartén hasta que cuajen.",
                       "Montaje: sirve el mangú con los huevos y el aguacate."]}
    g._close_protein_gap_for_meal(meal, 30.0, None, cands, day_used_proteins=set())
    lista = " ".join(meal["ingredients"]).lower()
    assert "atún" not in lista, meal["ingredients"]
    assert "mozzarella" in lista, meal["ingredients"]


def test_enganchado_en_el_cerrador():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("congruencia_cerrador").congruente(nlow, meal_text, _CLOSER_GENERIC_PROTEIN_WORDS)' in src
    assert "any(t in meal_text for t in _cong_toks)" not in src
