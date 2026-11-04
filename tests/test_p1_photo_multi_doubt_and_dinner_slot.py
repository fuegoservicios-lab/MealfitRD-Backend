"""
Tests para [BB-026 / P1-PHOTO-MULTI-DOUBT-DINNER-RESCUE · 2026-11-03]
Soberanía de aclaración en fotos multi-duda y rescate de franja horaria real (cena) en fotos nocturnas.
"""

import pytest
from unittest.mock import patch, MagicMock
from respuestas_de_la_foto import (
    aplicar_respuesta_a_descripcion,
    aplicar_respuestas_multiples,
    preparar_vision,
    instruccion,
    instruccion_solo_foto,
)
from prompts.chat_agent import build_vision_context
from tools import _rescue_dinner_slot, _rescue_dinner_from_unlabeled_desayuno


def test_multi_duda_reemplaza_alimento_correcto():
    """Caso real: 'Sorullitos de maíz · 2 huevos enteros' ante foto con dudas de bollitos de yuca y cantidad de huevos."""
    desc = (
        "Plato servido con 10 bollitos de yuca fritos dorados, dos huevos revueltos con jamón picado y una rodaja de aguacate maduro. "
        "DUDAS (pregúntale solo esto): ¿Qué tipo de alimento frito o bollito es? (opciones: 10 bollitos de yuca fritos (supuesto) · Sorullitos de maíz · Croquetas de pollo). "
        "¿Cuántos huevos enteros tiene el revuelto? (opciones: 2 huevos enteros (supuesto) · 3 huevos enteros · Solo claras). "
        "(Estimación: Calorías: 980, Proteína: 32g, Carbohidratos: 85g, Grasas Saludables: 24g)"
    )
    respuesta = "Sorullitos de maíz · 2 huevos enteros"

    # Verificar función directa (en la parte pre-dudas)
    desc_reemplazada, sup_str, reemps = aplicar_respuestas_multiples(desc, respuesta)
    pre_reemplazada = desc_reemplazada[:desc_reemplazada.find("DUDAS")]
    assert "sorullitos de maíz" in pre_reemplazada.lower()
    assert "bollitos de yuca" not in pre_reemplazada.lower()
    assert len(reemps) == 1
    assert reemps[0][0] == "10 bollitos de yuca fritos"
    assert reemps[0][1] == "Sorullitos de maíz"

    # Wrapper retrocompatible
    desc_wrap, sup_wrap = aplicar_respuesta_a_descripcion(desc, respuesta)
    assert desc_wrap == desc_reemplazada
    assert sup_wrap == "10 bollitos de yuca fritos"

    # Probar preparar_vision
    v = {
        "kind": "multi",
        "items": [{
            "kind": "plato",
            "description": desc
        }],
        "respuestas": respuesta,
        "has_text": True
    }
    v_prep = preparar_vision(v)
    assert v_prep.get("_preparado") is True
    assert v_prep.get("_reemplazos_aplicados") == reemps
    item_desc = v_prep["items"][0]["description"]
    assert "sorullitos de maíz" in item_desc.lower()
    assert "bollitos de yuca" not in item_desc.lower()
    assert "DUDAS" not in item_desc

    # Idempotencia: segunda llamada no muta ni rompe
    v_prep2 = preparar_vision(v_prep)
    assert v_prep2 is v_prep

    # Instrucciones al coach
    ctx = build_vision_context(v_prep)
    assert "ACLARACIÓN DE LA PROTEÍNA/CARNE" in ctx
    assert "IDENTIDAD DEL ALIMENTO (LA PALABRA DEL USUARIO ES LEY)" in ctx
    assert "PROHIBIDO registrar o nombrar en tu respuesta visible los supuestos descartados" in ctx
    assert "10 bollitos de yuca fritos" in ctx
    assert "REGLA ESTRICTA DE FRANJA HORARIA EN FOTO" in ctx


def test_salmon_backward_compat():
    """El caso existente de un solo reemplazo (Salmón en vez de Pechuga de pollo) se mantiene 100% intacto."""
    desc = (
        "Plato servido con una porción de arroz blanco, un filete pequeño de pechuga de pollo dorada a la plancha, rodajas de tomate y rodajas de pepino fresco. "
        "DUDAS (pregúntale solo esto): ¿Qué tipo de carne es la que está en el plato? (opciones: Pechuga de pollo (supuesto) · Salmón · Masa de cerdo) "
        "(Estimación: Calorías: 430, Proteína: 32g, Carbohidratos: 62g, Grasas Saludables: 5g)"
    )
    desc_nueva, sup = aplicar_respuesta_a_descripcion(desc, "Salmón")
    pre_nueva = desc_nueva[:desc_nueva.find("DUDAS")]
    assert "pechuga de pollo" not in pre_nueva.lower()
    assert "salmón dorado" in pre_nueva.lower()
    assert sup == "Pechuga de pollo"


def test_rescue_dinner_slot_converts_desayuno_to_cena_at_night():
    """A las 9:23 PM (21.38h), si el modelo manda 'desayuno' habiendo ya almorzado hoy y con cena vacía, se rescata a 'cena'."""
    user_id = "test-rescue-user"

    with patch("tools._hora_local_float", return_value=21.38), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db.execute_sql_query", return_value=[{"meal_type": "almuerzo", "ultima": "2026-11-03T20:51:00Z"}]):

        res = _rescue_dinner_from_unlabeled_desayuno(user_id, "desayuno", 980, 0)
        assert res == "cena", f"Esperado 'cena', obtenido {res}"


def test_rescue_dinner_slot_keeps_desayuno_in_morning():
    """A las 9:30 AM (9.5h), un desayuno se mantiene como desayuno."""
    user_id = "test-rescue-user"

    with patch("tools._hora_local_float", return_value=9.5), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db.execute_sql_query", return_value=[]):

        res = _rescue_dinner_from_unlabeled_desayuno(user_id, "desayuno", 450, 0)
        assert res == "desayuno"


def test_rescue_dinner_slot_keeps_desayuno_if_dinner_already_eaten():
    """Si el usuario ya cenó hoy, no se rescata a cena."""
    user_id = "test-rescue-user"

    with patch("tools._hora_local_float", return_value=22.0), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db.execute_sql_query", return_value=[
             {"meal_type": "almuerzo", "ultima": "2026-11-03T18:00:00Z"},
             {"meal_type": "cena", "ultima": "2026-11-03T21:00:00Z"}
         ]):

        res = _rescue_dinner_from_unlabeled_desayuno(user_id, "desayuno", 500, 0)
        assert res == "desayuno"


def test_log_consumed_meal_rescues_desayuno_when_not_explicit():
    """log_consumed_meal rescata a cena si explicit_slot=False, pero si explicit_slot=True lo preserva."""
    from tools import log_consumed_meal
    user_id = "test-rescue-user"

    with patch("tools._hora_local_float", return_value=21.5), \
         patch("tools.user_tz_offset_min", return_value=240), \
         patch("db.execute_sql_query", return_value=[{"meal_type": "almuerzo", "ultima": "2026-11-03T18:00:00Z"}]), \
         patch("tools.db_log_consumed_meal", return_value="meal-123") as mock_db_log:

        # Caso A: explicit_slot=False (el usuario solo subió la foto sin nombrar la franja)
        log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Sorullitos con huevos",
            "calories": 980,
            "protein": 32,
            "carbs": 85,
            "healthy_fats": 24,
            "meal_type": "desayuno",
            "explicit_slot": False,
        })
        assert mock_db_log.call_args[1]["meal_type"] == "cena"

        # Caso B: explicit_slot=True (el usuario dijo explícitamente "mi desayuno fue...")
        log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Sorullitos con huevos",
            "calories": 980,
            "protein": 32,
            "carbs": 85,
            "healthy_fats": 24,
            "meal_type": "desayuno",
            "explicit_slot": True,
        })
        assert mock_db_log.call_args[1]["meal_type"] == "desayuno"
