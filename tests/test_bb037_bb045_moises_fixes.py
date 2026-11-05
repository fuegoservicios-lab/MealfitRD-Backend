# -*- coding: utf-8 -*-
"""
Tests automatizados para blindaje contra corrupcion de diario, restauracion
y fidelidad de ingredientes (BB-037 a BB-045).
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from tools import (
    _limpiar_nombre_comida,
    _resolver_fila_diario_flexible,
    correct_consumed_meal,
    _nota_comidas_sin_registrar,
)
from agent import (
    _detectar_intencion_parte_de_comida,
    _es_intencion_restauracion_comida_borrada,
    _sanitizar_ingredientes_contra_alucinacion_foto,
    _promover_slot_plato_fuerte_tarde,
)


# ==============================================================================
# BB-037: Blindaje de UUID explicito y Paso 7 de Resolver Flexible
# ==============================================================================

def test_bb037_uuid_inexistente_no_toca_otra_fila():
    """Un UUID explicito que ya no esta en BD devuelve ERROR y jamas sobrescribe otra comida."""
    user_id = "test-user-bb037"
    deleted_uuid = "b40c122f-58ba-49af-b83a-d476da7cf37c"
    
    with patch("tools._leer_fila_diario", return_value=None), \
         patch("tools._resolver_fila_diario_flexible") as mock_resolver:
        
        result = correct_consumed_meal.func(
            user_id=user_id,
            meal_id=deleted_uuid,
            meal_name="Steak New York",
            calories=585,
            protein=41,
        )
        assert "ERROR: la comida con ID" in result
        assert "no existe en tu diario o fue eliminada" in result
        mock_resolver.assert_not_called()


def test_bb037_resolver_flexible_no_retorna_candidato_discrepante_en_paso_7():
    """Si solo hay 1 comida en el dia (desayuno de huevos), resolver 'steak' no debe devolverla."""
    candidatos = [{
        "id": "11111111-1111-1111-1111-111111111111",
        "meal_name": "Tostadas con mantequilla de mani y 4 huevos",
        "meal_type": "desayuno",
        "calories": 650,
        "protein": 55,
    }]
    with patch("db_facts.get_consumed_meals_today", return_value=candidatos):
        resolved_id, fila = _resolver_fila_diario_flexible(
            user_id="user1",
            meal_id="ca6ecba2",  # UUID que no coincide
            meal_name="Steak New York",
            meal_type="almuerzo",
        )
        assert resolved_id is None
        assert fila is None


# ==============================================================================
# BB-038: Restauracion de Comida Borrada -> log_consumed_meal
# ==============================================================================

def test_bb038_detecta_intencion_restauracion():
    """Detecta correctamente cuando el usuario pide restaurar o volver a poner algo borrado."""
    mensajes_restauracion = [
        [HumanMessage(content="agrega el desayuno que borre sin querer")],
        [HumanMessage(content="agrega el desayuno que borre")],
        [HumanMessage(content="he borrado la comida por error, restaurala")],
        [HumanMessage(content="borre mi cena sin querer, ponla de nuevo")],
        [HumanMessage(content="se me borro el desayuno, anotalo otra vez")],
        [HumanMessage(content="recupera el almuerzo que elimine")],
    ]
    for msgs in mensajes_restauracion:
        assert _es_intencion_restauracion_comida_borrada(msgs) is True

    mensajes_normales = [
        [HumanMessage(content="borrame el desayuno")],
        [HumanMessage(content="elimina esa merienda")],
        [HumanMessage(content="cambia el pollo por res")],
    ]
    for msgs in mensajes_normales:
        assert _es_intencion_restauracion_comida_borrada(msgs) is False


# ==============================================================================
# BB-040: Sanitizacion de Ingredientes Contra Alucinacion de Foto
# ==============================================================================

def test_bb040_sanitiza_aguacate_y_esparragos_alucinados():
    """Elimina ingredientes alucinados que no estan en la vision ni en el texto del usuario."""
    plate_photos = [{
        "desc": "bistec new york a la plancha con pure de papas y zanahorias baby.",
        "kcal": 645,
    }]
    messages = [HumanMessage(content="Aqui esta mi comida")]
    tool_args = {
        "meal_name": "Bistec New York con pure de papas, zanahorias baby, esparragos y aguacate",
        "calories": 795,
        "protein": 62,
    }
    
    sanitized = _sanitizar_ingredientes_contra_alucinacion_foto(tool_args, plate_photos, messages)
    assert "aguacate" not in sanitized["meal_name"].lower()
    assert "esparragos" not in sanitized["meal_name"].lower()
    # Las calorias infladas (+23%) se ajustan a la vision
    assert sanitized["calories"] == 645


def test_bb040_rescata_proteina_detectada_en_vision():
    """Si la vision detecto pechuga de pollo y bistec, asegura que la pechuga no se pierda."""
    plate_photos = [{
        "desc": "pechuga de pollo a la plancha con bistec new york, pure de papas y zanahorias.",
        "kcal": 645,
    }]
    messages = [HumanMessage(content="Mi comida")]
    tool_args = {
        "meal_name": "Bistec New York con pure de papas y zanahorias",
        "calories": 645,
        "protein": 62,
    }
    sanitized = _sanitizar_ingredientes_contra_alucinacion_foto(tool_args, plate_photos, messages)
    assert "pechuga de pollo a la parrilla" in sanitized["meal_name"].lower()


# ==============================================================================
# BB-042: Fusion Contextual con "Y pechuga a la parrilla"
# ==============================================================================

def test_bb042_detecta_adicion_y_pechuga():
    """Mensajes como 'Y pechuga a la parrilla' se reconocen como adicion al turno previo."""
    msgs = [
        HumanMessage(content="Foto del bistec"),
        AIMessage(content="Anote tu almuerzo: Steak New York 585 kcal", tool_calls=[{
            "name": "log_consumed_meal",
            "id": "tc1",
            "args": {"meal_name": "Steak New York", "meal_type": "almuerzo"}
        }]),
        HumanMessage(content="Y pechuga a la parrilla"),
    ]
    slot = _detectar_intencion_parte_de_comida(msgs)
    assert slot == "almuerzo"


# ==============================================================================
# BB-043: Franja de Plato Fuerte en la Tarde (>= 450 kcal con Almuerzo Vacio)
# ==============================================================================

def test_bb043_promueve_plato_fuerte_a_almuerzo():
    """Plato de 585 kcal a las 5:51 PM (17.8h) con almuerzo vacio se clasifica como 'almuerzo'."""
    tool_args = {
        "meal_name": "Steak New York con pure",
        "meal_type": "merienda",
        "calories": 585,
        "protein": 41,
    }
    messages = [HumanMessage(content="Aqui esta mi comida")]
    
    with patch("db_facts.get_consumed_meals_today", return_value=[{"meal_type": "desayuno"}]):
        res = _promover_slot_plato_fuerte_tarde(tool_args, "u1", local_hour=17.85, messages=messages)
        assert res["meal_type"] == "almuerzo"


def test_bb043_merienda_ligera_no_se_promueve():
    """Un snack ligero de 215 kcal no se promueve a almuerzo."""
    tool_args = {
        "meal_name": "Tortita de arroz con mantequilla de mani",
        "meal_type": "merienda",
        "calories": 215,
        "protein": 7,
    }
    messages = [HumanMessage(content="Aqui esta")]
    with patch("db_facts.get_consumed_meals_today", return_value=[]):
        res = _promover_slot_plato_fuerte_tarde(tool_args, "u1", local_hour=16.5, messages=messages)
        assert res["meal_type"] == "merienda"


# ==============================================================================
# BB-044: Nombres Limpios, Deteccion de No-Op y Supresion de Reclamos de Ayer
# ==============================================================================

def test_bb044_limpiar_nombre_comida():
    """Elimina ruidos conversacionales de meal_name."""
    assert _limpiar_nombre_comida("Desayuno de hoy: 3 tostadas con mani") == "3 tostadas con mani"
    assert _limpiar_nombre_comida("Desayuno: Mangu con queso") == "Mangu con queso"
    assert _limpiar_nombre_comida("Steak New York de siempre") == "Steak New York"
    assert _limpiar_nombre_comida("Mi almuerzo: Pechuga a la plancha") == "Pechuga a la plancha"


def test_bb044_correct_consumed_meal_detecta_no_op():
    """Si los datos son identicos a la BD, no ejecuta escritura en BD y confirma sin cambios."""
    fila_existente = {
        "id": "22222222-2222-2222-2222-222222222222",
        "meal_name": "Steak New York",
        "meal_type": "almuerzo",
        "calories": 585,
        "protein": 41,
        "carbs": 38,
        "healthy_fats": 29,
    }
    with patch("tools._leer_fila_diario", return_value=fila_existente), \
         patch("db_core.execute_sql_write") as mock_write, \
         patch("tools._nota_total_del_dia", return_value=""):
        
        res = correct_consumed_meal.func(
            user_id="u1",
            meal_id="22222222-2222-2222-2222-222222222222",
            meal_name="Steak New York",
            calories=585,
            protein=41,
            carbs=38,
            healthy_fats=29,
            meal_type="almuerzo",
        )
        assert "Registro confirmado sin cambios" in res
        mock_write.assert_not_called()


def test_bb044_no_pregunta_por_faltantes_de_ayer_en_registro_pasado():
    """Cuando days_ago > 0, no empuja al asistente a interrogar sobre comidas faltantes de ayer."""
    with patch("tools._diary_date_str_for_user", return_value="2026-11-05"), \
         patch("tools.user_tz_offset_min", return_value=0), \
         patch("chat_history_context.comidas_sin_registrar", return_value=["almuerzo", "cena"]):
        res = _nota_comidas_sin_registrar("u1", days_ago=1)
        assert "Ayer sigue sin registrar" not in res
        assert "registrada correctamente" in res
