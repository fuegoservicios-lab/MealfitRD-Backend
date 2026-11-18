"""[P1-PLAN-LOTE-1096 · 2026-11-18] Cualquier ingrediente con gramos se corrige en la ficha.

Caso real (18-nov): mangú con huevos, manteca, aceite, salami y pollo. Solo huevo y pollo tenían −/+; el mangú, la
manteca, el aceite y el salami no (la regla del suelo de proteína los descarta). Y la manteca resolvía a CARNE de
cerdo (17 kcal por 10 g). El catálogo falso reproduce lo que devolvió el resolutor real en el servidor para esos
renglones (mangú → Plátano verde 1,524 kcal/g, aceite 9, salami 3,23, huevo 1,389, pollo → 130,5 kcal).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import gramos_editables as ge  # noqa: E402
import macros_por_gramos as mpg  # noqa: E402

# palabra → (fila, kcal/g, proteína/g, carbos/g, grasa/g)
_FILAS = [
    ("platano", ("Plátano verde", 1.524, 0.0125, 0.36, 0.0007)),
    ("huevo", ("Huevo", 1.389, 0.126, 0.007, 0.095)),
    ("cerdo", ("Cerdo", 1.7, 0.211, 0.0, 0.095)),     # la manteca caía aquí
    ("aceite", ("Aceite de oliva", 9.0, 0.0, 0.0, 1.0)),
    ("salami", ("Salami", 3.23, 0.16, 0.0, 0.27)),
    ("pollo", ("Pechuga de pollo", 1.45, 0.304, 0.0, 0.026)),
]


class Catalogo:
    def macros_from_ingredient_string(self, s):
        g = re.search(r"(\d+(?:\.\d+)?)\s*g\b", s)
        if not g:
            return None
        gramos = float(g.group(1))
        plano = mpg._plano(s)
        for clave, (fila, k, p, c, f) in _FILAS:
            if clave in plano:
                return {"name": fila, "grams": gramos, "kcal": round(k * gramos, 1), "protein": round(p * gramos, 2),
                        "carbs": round(c * gramos, 2), "fats": round(f * gramos, 2)}
        return None


DESAYUNO = {"calories": 855, "protein": 53, "carbs": 68, "healthy_fats": 41, "ingredients": [
    "Mangú de plátano verde (≈250 g)", "Huevos enteros (≈100 g; 2 unidades)", "Manteca de cerdo para freír (≈10 g)",
    "Aceite de oliva (≈5 g)", "Salami frito (≈40 g)", "Pechuga de pollo desmenuzada (≈90 g)"]}


def test_todo_renglon_con_gramos_que_resuelve_es_editable():
    assert ge.editables(DESAYUNO["ingredients"], Catalogo()) == {0: 250, 1: 100, 2: 10, 3: 5, 4: 40, 5: 90}


def test_sin_gramos_o_sin_fila_no_es_editable():
    ings = ["1 taza de arroz", "Sazón misterioso (≈20 g)", "Aceite de oliva (≈5 g)"]
    assert ge.editables(ings, Catalogo()) == {2: 5}


def test_la_manteca_es_grasa_y_no_carne_de_cerdo():
    m = ge.macros_de_renglon("Manteca de cerdo para freír (≈10 g)", Catalogo())
    assert m["name"] == "Manteca" and m["kcal"] == 90.2 and m["fats"] == 10.0 and m["protein"] == 0.0


def test_huevos_fritos_en_manteca_siguen_siendo_huevo():
    m = ge.macros_de_renglon("Huevos fritos en manteca de cerdo (≈100 g)", Catalogo())
    assert m["name"] == "Huevo"


def test_subir_el_mangu_mueve_los_totales_lo_que_cambia_ese_renglon():
    n = ge.ajustar(DESAYUNO, 0, 300, Catalogo())
    assert n["ingredients"][0] == "Mangú de plátano verde (≈300 g)"
    assert n["calories"] == round(855 + 1.524 * 50)
    assert n["carbs"] == round(68 + 0.36 * 50)
    assert n["ingredients"][1:] == DESAYUNO["ingredients"][1:], "los demás renglones no se tocan"


def test_bajar_la_manteca_resta_su_grasa():
    n = ge.ajustar(DESAYUNO, 2, 5, Catalogo())
    assert n["ingredients"][2] == "Manteca de cerdo para freír (≈5 g)"
    assert n["calories"] == round(855 - 9.02 * 5)
    assert n["healthy_fats"] == round(41 - 5)


def test_las_piezas_simples_siguen_por_el_camino_del_1039():
    class Simple(Catalogo):
        pass
    simple = {"calories": 500, "protein": 40, "carbs": 0, "healthy_fats": 10, "ingredients": ["150 g de Pechuga de pollo"]}
    assert ge.ajustar(simple, 0, 200, Simple()) == mpg.ajustar_gramos(simple, 0, 200, Simple())


def test_fuera_de_rango_o_de_la_lista_no_cambia_nada():
    assert ge.ajustar(DESAYUNO, 9, 100, Catalogo()) is None
    assert ge.ajustar({"ingredients": ["1 taza de arroz"]}, 0, 100, Catalogo()) is None
    bajo = ge.ajustar(DESAYUNO, 0, 0, Catalogo())
    assert bajo["gramos"] == mpg.GRAMOS_MIN


def test_la_ficha_ofrece_el_menos_mas_en_todos_y_la_manteca_con_sus_kcal_de_grasa():
    from ficha_comida import desglose_de_ingredientes
    d = desglose_de_ingredientes(DESAYUNO["ingredients"], 855, Catalogo())
    assert [l.get("ajustable_g") for l in d["lineas"]] == [250, 100, 10, 5, 40, 90]
    assert d["lineas"][2]["kcal"] == 90


def test_el_endpoint_usa_el_ajuste_de_cualquier_renglon():
    src = (_BACKEND / "routers" / "diary.py").read_text(encoding="utf-8")
    assert "from gramos_editables import ajustar as ajustar_gramos" in src
