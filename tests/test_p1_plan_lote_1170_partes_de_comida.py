# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1170 · 2026-11-23] Una comida fusionada enseña su total y CADA parte (la fila real del 22-nov)."""
from partes_de_comida import nombres_de_partes, partes_de_comida

FUSIONADA = {
    "base_meal": {"meal_name": "Mass gainer GNC Bulk 1340 (5 scoops con 1 L de agua)", "calories": 1114, "protein": 42,
                  "carbs": 233, "healthy_fats": 4, "ingredients": []},
    "addons": [{"meal_name": "Bistec de res y pechuga de pollo a la plancha con puré de papa y moro de lentejas o guandules",
                "calories": 1050, "protein": 98, "carbs": 68, "healthy_fats": 42, "ingredients": ["bistec de res (≈170 g)"]}],
    "attachment_ids": ["x"],
}


def test_fusionada_da_cada_parte_con_sus_macros():
    p = partes_de_comida(FUSIONADA)
    assert [x["meal_name"][:11] for x in p] == ["Mass gainer", "Bistec de r"]
    assert p[0] == {"meal_name": FUSIONADA["base_meal"]["meal_name"], "calories": 1114, "protein": 42, "carbs": 233, "healthy_fats": 4}
    assert sum(x["calories"] for x in p) == 2164


def test_comida_sola_o_sin_dato_no_tiene_partes():
    for n in (None, {}, {"values": {}}, {"addons": []}, {"base_meal": {"meal_name": "A"}, "addons": [{"meal_name": ""}]}, "x"):
        assert partes_de_comida(n) == []
    assert nombres_de_partes(FUSIONADA)[1].startswith("Bistec")


def test_la_lista_del_dia_y_la_ficha_las_llevan():
    import inspect
    from routers import diary
    src = inspect.getsource(diary)
    assert "nombres_de_partes(nutrition)" in src
    import ficha_comida
    assert "partes_de_comida(" in inspect.getsource(ficha_comida)
