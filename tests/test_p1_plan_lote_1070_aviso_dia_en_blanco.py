"""[P1-PLAN-LOTE-1070 · 2026-11-17] El aviso de la cena con el día en blanco proponía «una arepita… y sumar proteína»
(~450 kcal, ~13 g): sin registros el bloque no daba la meta ni pedía un plato completo."""
import aviso_del_dia as ad

_PERFIL = {"gender": "male", "age": 21, "height": 168, "weight": 55.8, "weightUnit": "kg", "activityLevel": "moderate",
           "mainGoal": "gain_muscle", "medicalConditions": ["Ninguna"]}


def test_cena_con_el_dia_en_blanco_pide_plato_completo_y_da_la_meta():
    b = ad.bloque_del_dia([], _PERFIL, "Cena")
    assert "Todavía no registró nada hoy." in b
    assert "Su meta de hoy:" in b and " kcal" in b
    assert "DÍA EN BLANCO" in b and "plato COMPLETO" in b and "NO propongas algo ligero" in b
    assert "sin preguntar si se le olvidó anotar" in b


def test_almuerzo_tambien_y_desayuno_o_merienda_no():
    assert "DÍA EN BLANCO" in ad.bloque_del_dia([], _PERFIL, "Almuerzo")
    assert "DÍA EN BLANCO" not in ad.bloque_del_dia([], _PERFIL, "Desayuno")
    assert "DÍA EN BLANCO" not in ad.bloque_del_dia([], _PERFIL, "Merienda")
    assert "DÍA EN BLANCO" not in ad.bloque_del_dia([], _PERFIL)


def test_sin_perfil_no_inventa_meta_pero_si_la_pauta():
    b = ad.bloque_del_dia([], {"mainGoal": "gain_muscle"}, "Cena")
    assert "Su meta de hoy" not in b and "~30-40 g de proteína" in b


def test_con_registros_no_hay_dia_en_blanco_y_si_la_regla_de_proteina():
    b = ad.bloque_del_dia([{"meal_type": "desayuno", "meal_name": "Avena", "calories": 400, "protein": 15}], _PERFIL, "Cena")
    assert "DÍA EN BLANCO" not in b
    assert "«suma proteína»" in b


def test_el_aviso_pasa_la_comida():
    import inspect
    import proactive_agent
    assert "bloque_del_dia(consumed, health, meal_to_check)" in inspect.getsource(proactive_agent)
