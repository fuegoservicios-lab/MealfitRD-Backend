# -*- coding: utf-8 -*-
"""[MICROS-ORIGEN · 2026-11-11] «¿Qué me subió el sodio?»: el resumen del día dice qué alimentos aportaron cada
nutriente (de mayor a menor), agrupa el resto y nombra las comidas/alimentos sin dato."""
from diary_micros import micros_de_ingredientes, resumen_micros
from food_label_nutrients import micros_from_snapshot, snapshot


class _DB:
    TABLA = {
        "80 g de Salami": {"sodium_mg": 980.0, "fiber": 0.0},
        "2 unidades de Pan de agua": {"sodium_mg": 420.0, "fiber": 2.0},
        "100 g de Arroz blanco": {"sodium_mg": 1.0, "fiber": 0.4},
        "1 unidad de Huevo": {"sodium_mg": 70.0, "fiber": None},
    }

    def micros_from_ingredient_string(self, s):
        return self.TABLA.get(s)


def _dia():
    db = _DB()
    return [
        {"meal_name": "Desayuno", "meal_type": "desayuno",
         "micros": micros_de_ingredientes(["80 g de Salami", "2 unidades de Pan de agua", "1 unidad de Huevo"], db)},
        {"meal_name": "Almuerzo", "meal_type": "almuerzo",
         "micros": micros_de_ingredientes(["100 g de Arroz blanco", "Sal al gusto"], db)},
        {"meal_name": "Merienda", "meal_type": "merienda", "micros": None},
    ]


def test_cada_renglon_guarda_su_aporte_y_los_que_no_resuelven_quedan_vacios():
    m = micros_de_ingredientes(["80 g de Salami", "Sal al gusto"], _DB())
    assert m["items"] == [{"label": "80 g de Salami", "values": {"sodium_mg": 980.0, "fiber_g": 0.0}},
                          {"label": "Sal al gusto", "values": {}}]


def test_top_ordenado_de_mayor_a_menor_con_la_comida():
    sodio = resumen_micros(_dia())["micros_coverage"]["by_nutrient"]["sodium_mg"]
    assert [a["food"] for a in sodio["top"]] == ["80 g de Salami", "2 unidades de Pan de agua", "1 unidad de Huevo", "100 g de Arroz blanco"]
    assert sodio["top"][0] == {"food": "80 g de Salami", "meal": "Desayuno", "meal_type": "desayuno", "value": 980.0}
    assert sodio["others"] == {"count": 0, "value": 0}


def test_faltantes_nombran_la_comida_entera_y_el_alimento_sin_dato():
    by = resumen_micros(_dia())["micros_coverage"]["by_nutrient"]
    assert by["sodium_mg"]["missing"] == [
        {"meal": "Almuerzo", "meal_type": "almuerzo", "foods": ["Sal al gusto"], "whole_meal": False},
        {"meal": "Merienda", "meal_type": "merienda", "foods": [], "whole_meal": True},
    ]
    # El huevo no trae fibra: aparece como alimento sin dato en el desayuno.
    assert {"meal": "Desayuno", "meal_type": "desayuno", "foods": ["1 unidad de Huevo"], "whole_meal": False} in by["fiber_g"]["missing"]
    # Un cero documentado no es «sin dato» ni aparece como aporte.
    assert "80 g de Salami" not in [a["food"] for a in by["fiber_g"]["top"]]


def test_mas_de_cinco_aportes_se_agrupan_en_otros():
    db = _DB()
    db.TABLA = {f"{i} g de A{i}": {"sodium_mg": float(i)} for i in range(1, 9)}
    meals = [{"meal_name": "Cena", "micros": micros_de_ingredientes(list(db.TABLA), db)}]
    sodio = resumen_micros(meals)["micros_coverage"]["by_nutrient"]["sodium_mg"]
    assert [a["value"] for a in sodio["top"]] == [8.0, 7.0, 6.0, 5.0, 4.0]
    assert sodio["others"] == {"count": 3, "value": 6.0}


def test_etiqueta_sin_desglose_aporta_como_comida():
    label = {"serving_text": "1 lata", "nutrients": [{"key": "sodium_mg", "amount": 500.0, "unit": "mg"}]}
    meals = [{"meal_name": "Sopa de sobre", "micros": micros_from_snapshot(snapshot(label, 1))}]
    sodio = resumen_micros(meals)["micros_coverage"]["by_nutrient"]["sodium_mg"]
    assert sodio["top"] == [{"food": "Sopa de sobre", "meal": "Sopa de sobre", "meal_type": None, "value": 500.0}]
    assert sodio["missing"] == []
