# -*- coding: utf-8 -*-
"""[P1-CLOSER-DENSE-COOKABLE · 2026-09-06] El piso cocinable de 40 g mataba justo a las proteínas BUENAS.

La aritmética va al revés de lo que parece: cuanto MÁS densa es la proteína, MENOS gramos hacen falta para
cerrar el hueco — y menos de 40 g significaba «no añadas nada».

Medido en el plan vivo a903e339 (vegetariano, ganancia muscular, cena «Ají morrón relleno de queso fresco,
maíz y quinoa»): faltaban 13,2 g de proteína y había 188 kcal libres. Con soya texturizada (51,5 g/100 g)
bastaban 26 g, y 26 < 40, así que el cerrador no hacía NADA. La sonda `P1-SLOT-FLOOR-UNCLOSED` lo dejó por
escrito —«30 g de 47 | kcal 686/874 | candidatos=12»— y la consecuencia fue un rechazo clínico por déficit de
proteína y un intento entero de generación tirado.

Ahora, en vez de rendirse, sube la cantidad al mínimo cocinable cuando las kcal lo permiten."""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

import graph_orchestrator as go  # noqa: E402


class _Info:
    def __init__(self, name, protein, kcal, carbs=0.0, fats=0.0):
        self.name = name
        self.protein = protein
        self.kcal = kcal
        self.carbs = carbs
        self.fats = fats


class _DB:
    """DB mínima: solo responde lo que el cerrador consulta del candidato elegido."""
    def get_nutrition(self, name):
        return _CANDIDATOS.get(str(name).lower())

    def __getattr__(self, _n):
        return lambda *a, **k: None


_CANDIDATOS = {
    "soya texturizada": _Info("Soya texturizada", 51.5, 327.0, 33.0, 1.0),
    "queso cottage": _Info("Queso cottage", 12.4, 69.7, 3.0, 1.0),
}


def _cena_vegetariana():
    return {"meal": "Cena", "name": "Ají morrón relleno de queso fresco, maíz y quinoa",
            "protein": 30, "cals": 686, "carbs": 90, "fats": 20,
            "ingredients": ["2 ají morrón", "80 g de queso fresco", "60 g de maíz dulce en granos",
                            "70 g de quinoa", "10 g de aceite de oliva"]}


def _cerrar(meal, densa=True, slot_cal_target=874.0):
    info = _CANDIDATOS["soya texturizada"] if densa else _CANDIDATOS["queso cottage"]
    cands = [(0.0, info.name, info)]
    return go._close_protein_gap_for_meal(
        meal, 47.0, _DB(), cands, allergies=None, fill_pct=go.PROTEIN_FLOOR_FILL_PCT,
        max_add_g=120, slot_cal_target=slot_cal_target, enforce_min_threshold=False,
        diet="vegetariana", country="DO", goal="gain_muscle")


def test_la_proteina_densa_ya_no_se_descarta_por_ser_eficiente():
    """26 g de soya cerraban el hueco; el piso cocinable los subía a 0. Ahora suben a 40."""
    meal = _cena_vegetariana()
    g = _cerrar(meal)
    assert g >= go.CLOSER_COOKABLE_MIN_G, f"añadió {g} g"
    assert any("soya" in str(i).lower() for i in meal["ingredients"]), meal["ingredients"]


def test_la_subida_respeta_el_techo_de_kcal():
    """Si el mínimo cocinable no cabe en las kcal libres, se sigue abandonando: el tope manda."""
    meal = _cena_vegetariana()
    meal["cals"] = 860          # solo 14 kcal libres; 40 g de soya son 131
    assert _cerrar(meal, slot_cal_target=874.0) == 0
    assert not any("soya" in str(i).lower() for i in meal["ingredients"])


def test_el_candidato_poco_denso_no_necesita_la_subida():
    """Con queso cottage hacen falta ~107 g, que ya superan el mínimo: el camino de siempre."""
    meal = _cena_vegetariana()
    g = _cerrar(meal, densa=False)
    assert g > go.CLOSER_COOKABLE_MIN_G, g


def test_tiene_kill_switch(monkeypatch):
    """A False vuelve el comportamiento anterior, que es lo que un rollback necesita poder hacer."""
    monkeypatch.setattr(go, "CLOSER_DENSE_COOKABLE", False)
    meal = _cena_vegetariana()
    assert _cerrar(meal) == 0


def test_el_knob_existe_y_viene_encendido():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert 'CLOSER_DENSE_COOKABLE = _env_bool("MEALFIT_CLOSER_DENSE_COOKABLE", True)' in src


def test_la_subida_va_despues_del_make_room_de_franja_ligera():
    """En franja ligera primero se hace SITIO (P1-SNACK-MAKE-ROOM) y solo después se sube la cantidad:
    invertirlo dejaría el cerrador subiendo gramos sin haber liberado kcal."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    # Índices absolutos y no una ventana: la función pasa de 30.000 caracteres y cualquier ventana que
    # elija hoy se queda corta mañana — ya me pasó con este mismo test.
    i = src.index("def _close_protein_gap_for_meal")
    make_room = src.index("_make_room_for_protein(", i)
    dense = src.index("P1-CLOSER-DENSE-COOKABLE · 2026-09-06", i) if False else src.index("[P1-CLOSER-DENSE-COOKABLE", i)
    assert i < make_room < dense, (i, make_room, dense)
