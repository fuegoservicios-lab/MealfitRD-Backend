# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-919 · 2026-09-29] El cerrador de proteína no repite en el día el lácteo dulce que otra comida ya lleva,
si tiene otro equivalente.

Corpus del VPS (92 planes recientes, 1.170 comidas): de los 135 lácteos dulces que añadió el cerrador, 100 caían en un día
que ya llevaba ese mismo lácteo en otra comida (queso cottage 67 de 84, yogurt griego entero 33 de 51); el yogurt sale dos
veces el mismo día en 76 de 283 días. La regla de «no repetir en el día» (P1-CLOSER-DAY-AWARE-PROTEIN) sólo conoce las
etiquetas del gate —carnes, pescados y huevo—, así que para el desayuno y la merienda el primero del orden gana siempre.

La primera versión (cualquier alimento del pool, cualquier sustituto) se simuló sobre esos 92 planes y se leyó cambio por
cambio: 18 de 38 eran peores («Lechosa fresca con maní tostado y queso gouda», «Vasito de yogur con manzana, mantequilla
de maní y huevo», «Casabe crujiente con mantequilla de maní, guineo y queso cottage»). De ahí las tres condiciones de
abajo: sólo entre lácteos dulces, sólo con un equivalente que ya pasó los filtros del plato, y nunca contra lo que el
plato ya lleva.
"""
from __future__ import annotations

import copy
import pathlib
import re

import graph_orchestrator as go
import rotacion_cerrador as rc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


class _Info:
    def __init__(self, name, protein, kcal, carbs=0.0, fats=0.0):
        self.name, self.protein, self.kcal, self.carbs, self.fats = name, protein, kcal, carbs, fats


_YOGURT = _Info("Yogurt griego entero", 8.8, 94.0, 4.0, 4.4)
_COTTAGE = _Info("Queso cottage", 10.4, 81.0, 3.4, 2.3)
_RICOTTA = _Info("Queso ricotta", 7.5, 151.0, 7.3, 10.2)
_GOUDA = _Info("Queso gouda", 24.9, 356.0, 2.2, 27.4)
_EDAMAME = _Info("Edamame", 11.9, 121.0, 8.9, 5.2)
_HUEVO = _Info("Huevo", 12.6, 139.0, 0.7, 9.5)
_POR_NOMBRE = {i.name.lower(): i for i in (_YOGURT, _COTTAGE, _RICOTTA, _GOUDA, _EDAMAME, _HUEVO)}


class _DB:
    def grams_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\s+de\s+", str(s))
        return float(m.group(1).replace(",", ".")) if m else None

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        return None if g is None else {"grams": g, "kcal": g * 0.9, "protein": g * 0.09, "carbs": g * 0.04, "fats": g * 0.04}

    def get_nutrition(self, name):
        return _POR_NOMBRE.get(str(name).lower())

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _merienda():
    return {"meal": "Merienda", "name": "Casabe crujiente con guineo y canela",
            "protein": 4, "cals": 180, "carbs": 40, "fats": 1,
            "ingredients": ["1 torta pequeña de casabe", "1 guineo", "1 pizca de canela"],
            "recipe": ["Mise en place: mide el casabe.", "Montaje: sirve el casabe con el guineo y la canela."]}


def _desayuno_con_yogurt():
    return {"meal": "Desayuno", "name": "Avena cremosa con guayaba y yogurt",
            "protein": 20, "cals": 380, "carbs": 50, "fats": 8,
            "ingredients": ["40 g de avena", "100 g de guayaba", "¾ taza de yogurt griego natural"],
            "recipe": ["El Toque de Fuego: cocina la avena 7-9 minutos.", "Montaje: sirve con la guayaba y el yogurt."]}


def _cena():
    return {"meal": "Cena", "name": "Yuca guisada con queso blanco fresco y aguacate",
            "protein": 12, "cals": 460, "carbs": 70, "fats": 15,
            "ingredients": ["200 g de yuca", "40 g de queso blanco fresco", "½ aguacate"],
            "recipe": ["El Toque de Fuego: hierve la yuca 20 minutos y guísala 5 minutos.",
                       "Montaje: sirve con el queso y el aguacate."]}


def _cerrar(meal, cands, usados):
    return go._close_protein_gap_for_meal(meal, 30.0, _DB(), [(0.0, i.name, i) for i in cands], allergies=None,
                                          fill_pct=0.92, max_add_g=120, enforce_min_threshold=False,
                                          day_used_proteins=usados, diet=None, country="DO", goal="lose_fat")


def _nuevas(meal, base):
    return [x.lower() for x in meal["ingredients"] if x not in base["ingredients"]]


_CON_YOGURT = frozenset({"alimento:yogurt"})


def test_el_segundo_lacteo_del_dia_no_es_el_mismo():
    usados = go._protein_gate_labels_in_meal(_desayuno_con_yogurt()) | rc.del_plato(_desayuno_con_yogurt())
    assert usados == _CON_YOGURT
    m = _merienda()
    assert _cerrar(m, [_YOGURT, _COTTAGE, _RICOTTA], usados) > 0
    assert "queso cottage" in _nuevas(m, _merienda())[0], m["ingredients"]
    m = _merienda()
    assert _cerrar(m, [_YOGURT, _COTTAGE, _RICOTTA], {"alimento:yogurt", "alimento:cottage"}) > 0
    assert "queso ricotta" in _nuevas(m, _merienda())[0], "con los dos en el día, el tercero"


def test_sin_ese_alimento_en_el_dia_gana_el_primero_como_siempre():
    m = _merienda()
    assert _cerrar(m, [_YOGURT, _COTTAGE, _RICOTTA], rc.del_plato(_cena())) > 0
    assert "yogurt griego entero" in _nuevas(m, _merienda())[0]


def test_sin_un_equivalente_libre_se_repite_antes_que_poner_otra_cosa():
    """Ni huevo ni un queso de sal sobre la fruta: el sustituto es otro lácteo dulce o ninguno."""
    for cands in ([_YOGURT], [_YOGURT, _HUEVO], [_YOGURT, _GOUDA, _HUEVO]):
        m = _merienda()
        assert _cerrar(m, cands, _CON_YOGURT) > 0
        assert "yogurt griego entero" in _nuevas(m, _merienda())[0], (cands, m["ingredients"])
    m = _merienda()
    assert _cerrar(m, [_YOGURT, _COTTAGE], {"alimento:yogurt", "alimento:cottage"}) > 0
    assert "yogurt griego entero" in _nuevas(m, _merienda())[0], "los dos en el día: el orden de siempre"


def test_el_yogurt_del_propio_plato_no_es_una_repeticion():
    vasito = {"meal": "Merienda", "name": "Vasito de yogur con manzana y canela",
              "protein": 6, "cals": 160, "carbs": 25, "fats": 3,
              "ingredients": ["½ taza de yogur natural", "1 manzana", "1 pizca de canela"],
              "recipe": ["Montaje: sirve el yogur con la manzana y la canela."]}
    m = copy.deepcopy(vasito)
    assert _cerrar(m, [_YOGURT, _COTTAGE, _HUEVO], _CON_YOGURT) > 0
    assert not any("cottage" in x or "huevo" in x for x in _nuevas(m, vasito)), m["ingredients"]
    assert rc.choca("yogurt griego entero", vasito, [(_YOGURT, "yogurt griego entero"), (_COTTAGE, "queso cottage")],
                    _CON_YOGURT) == set()


def test_la_pasta_de_untar_sigue_sin_recibir_queso():
    """P1-CLOSER-NO-SPREAD-PLUS-CHEESE saca el queso del pool: sin equivalente, el yogurt se queda."""
    untado = {"meal": "Merienda", "name": "Casabe crujiente con mantequilla de maní y guineo",
              "protein": 8, "cals": 250, "carbs": 40, "fats": 9,
              "ingredients": ["1 torta pequeña de casabe", "1 cda de mantequilla de maní", "1 guineo"],
              "recipe": ["Montaje: unta la mantequilla de maní y sirve con el guineo."]}
    m = copy.deepcopy(untado)
    assert _cerrar(m, [_YOGURT, _COTTAGE, _RICOTTA], _CON_YOGURT) > 0
    assert "yogurt griego entero" in _nuevas(m, untado)[0], m["ingredients"]


def test_fuera_del_lacteo_dulce_nada_cambia():
    almuerzo = {"meal": "Almuerzo", "name": "Arroz con vegetales salteados y edamame",
                "ingredients": ["150 g de arroz cocido", "120 g de edamame cocido", "1 zanahoria"]}
    assert rc.del_plato(almuerzo) == set()
    m = _cena()
    assert _cerrar(m, [_EDAMAME, _HUEVO], rc.del_plato(almuerzo)) > 0
    assert "edamame" in _nuevas(m, _cena())[0]


def test_las_etiquetas_por_palabra():
    assert rc.etiquetas("yogurt griego entero") == _CON_YOGURT
    assert rc.etiquetas("¾ taza de yogur natural sin azúcar") == _CON_YOGURT
    assert rc.etiquetas("2 yogures") == _CON_YOGURT
    assert rc.etiquetas("queso cottage") == {"alimento:cottage"}
    assert rc.etiquetas("60 g de requesón") == {"alimento:ricotta"} == rc.etiquetas("queso ricotta")
    assert rc.etiquetas("yogurt con queso cottage") == {"alimento:yogurt", "alimento:cottage"}
    for ajeno in ("60 g de edamame cocido", "100 g de pechuga de pollo", "2 huevos", "40 g de queso blanco", "½ aguacate",
                  "", None):
        assert rc.etiquetas(ajeno) == set(), ajeno


def test_el_plato_suma_nombre_e_ingredientes():
    assert rc.del_plato(_desayuno_con_yogurt()) == _CON_YOGURT
    assert rc.del_plato({"name": "Tostadas con requesón", "ingredients": ["1 rebanada de pan"]}) == {"alimento:ricotta"}
    assert rc.del_plato(_cena()) == set() and rc.del_plato(None) == set() and rc.del_plato({}) == set()


def test_las_familias_son_los_lacteos_dulces_del_cerrador():
    """El vocabulario no es una tabla aparte: si el cerrador gana o pierde un lácteo dulce, este test lo dice."""
    for token in go._SWEET_DAIRY_TOKENS:
        assert rc.etiquetas(token), token
    formas = {f for _fam, fs in rc._FAMILIAS for f in fs}
    assert all(any(t in f for t in go._SWEET_DAIRY_TOKENS) for f in formas), formas


def test_con_el_knob_apagado_todo_como_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_CLOSER_DAY_FOOD_ROTATION", "false")
    assert rc.etiquetas("yogurt griego entero") == set() and rc.del_plato(_desayuno_con_yogurt()) == set()
    assert rc.choca("yogurt griego entero", _merienda(), [(_COTTAGE, "queso cottage")], _CON_YOGURT) == set()
    m = _merienda()
    assert _cerrar(m, [_YOGURT, _COTTAGE], _CON_YOGURT) > 0
    assert "yogurt griego entero" in _nuevas(m, _merienda())[0]


def test_el_reparador_de_la_franja_ligera_reparte_dos_lacteos_en_el_dia():
    dia = {"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Avena cremosa con lechosa", "protein": 6, "cals": 250, "carbs": 45, "fats": 4,
         "ingredients": ["40 g de avena", "100 g de lechosa"],
         "recipe": ["El Toque de Fuego: cocina la avena 7-9 minutos.", "Montaje: sirve con la lechosa."]},
        {"meal": "Almuerzo", "name": "Pollo guisado con arroz", "protein": 50, "cals": 700, "carbs": 80, "fats": 15,
         "ingredients": ["180 g de pechuga de pollo", "150 g de arroz cocido"], "recipe": ["Montaje: sirve."]},
        _merienda(),
        {"meal": "Cena", "name": "Pescado al horno con batata", "protein": 45, "cals": 600, "carbs": 60, "fats": 15,
         "ingredients": ["200 g de tilapia", "200 g de batata"], "recipe": ["Montaje: sirve."]}]}
    dias = [copy.deepcopy(dia)]
    cands = [(0.0, i.name, i) for i in (_YOGURT, _COTTAGE, _RICOTTA)]
    nutr = {"macros": {"protein_g": 150, "carbs_g": 250, "fats_g": 70}}
    assert go._repair_light_slot_protein(dias, nutr, {"mainGoal": "gain_muscle"}, db=_DB(), cands=cands) > 0
    desayuno, merienda = dias[0]["meals"][0], dias[0]["meals"][2]
    a = rc.del_plato({"ingredients": _nuevas(desayuno, dia["meals"][0])})
    b = rc.del_plato({"ingredients": _nuevas(merienda, dia["meals"][2])})
    assert a and b and not (a & b), (desayuno["ingredients"], merienda["ingredients"])


def test_anclas():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    cerrador = src[src.index("def _close_protein_gap_for_meal("):src.index("def _ingredient_is_protein_dominant(")]
    assert ('__import__("rotacion_cerrador").choca(_nlow_c, meal, _pool, day_used_proteins)) & set(day_used_proteins))'
            '  # [P1-PLAN-LOTE-919]') in cerrador
    assert src.count('| __import__("rotacion_cerrador").del_plato(') == 6, "las tres pasadas del cerrador, al leer y al anotar"
    modulo = _BACKEND / "rotacion_cerrador.py"
    assert "tooltip-anchor: P1-PLAN-LOTE-919" in modulo.read_text(encoding="utf-8")
    assert b"\x08" not in modulo.read_bytes() and b"\r" not in modulo.read_bytes()
    assert len(src.splitlines()) <= 52_240
