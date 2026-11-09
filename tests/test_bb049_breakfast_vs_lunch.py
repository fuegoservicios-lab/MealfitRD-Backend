# -*- coding: utf-8 -*-
"""[BB-049 · 2026-11-09] Pruebas de corrección de franja horaria: desayuno vs almuerzo a las 11:00 AM / mañana."""
from __future__ import annotations

import unittest
from unittest.mock import patch
from langchain_core.messages import HumanMessage
import coach_day_context as cdc
from agent import _promover_slot_plato_fuerte_tarde, _corregir_almuerzo_prematuro


def test_franja_por_hora_11_am_is_desayuno():
    # 11:11 AM = 11.183 h
    # Sin lista de faltan, el horario de la mañana (4:30 a 12:00) es desayuno
    assert cdc.franja_por_hora(11.183) == "desayuno"
    assert cdc.franja_por_hora(11.0) == "desayuno"
    assert cdc.franja_por_hora(11.9) == "desayuno"
    assert cdc.franja_por_hora(12.0) == "almuerzo"
    assert cdc.franja_por_hora(13.0) == "almuerzo"


def test_franja_por_hora_with_faltan():
    # A las 11:18 AM con desayuno pendiente -> desayuno
    assert cdc.franja_por_hora(11.183, faltan=["desayuno", "almuerzo", "cena"]) == "desayuno"
    # A las 11:18 AM si el desayuno ya está registrado -> almuerzo
    assert cdc.franja_por_hora(11.183, faltan=["almuerzo", "cena"]) == "almuerzo"


def test_estado_de_las_comidas_at_11_am():
    t_vacio = cdc.estado_de_las_comidas([], 11.183)
    assert "desayuno — sin anotar, es la que toca ahora" in t_vacio
    assert "almuerzo — sin anotar, más tarde" in t_vacio
    assert "toca AHORA (desayuno)" in t_vacio
    assert "SIN anotar y su hora ya pasó" not in t_vacio

    t_con_desayuno = cdc.estado_de_las_comidas([{"meal_type": "desayuno"}], 11.183)
    assert "desayuno — anotado" in t_con_desayuno
    assert "almuerzo — sin anotar, más tarde" in t_con_desayuno
    assert "a esta hora no le toca ninguna comida" in t_con_desayuno


@patch("db_facts.get_consumed_meals_today", return_value=[])
def test_promover_slot_plato_fuerte_morning_vs_afternoon(mock_meals):
    msgs = [HumanMessage(content="Tostadas con huevo")]
    
    # Mañana (11:11 AM) con 600 kcal y slot merienda -> promueve a desayuno
    args_manana = {"meal_type": "merienda", "calories": 600}
    res_manana = _promover_slot_plato_fuerte_tarde(args_manana, "u1", 11.183, msgs)
    assert res_manana["meal_type"] == "desayuno"

    # Tarde (1:30 PM) con 600 kcal y slot merienda -> promueve a almuerzo
    args_tarde = {"meal_type": "merienda", "calories": 600}
    res_tarde = _promover_slot_plato_fuerte_tarde(args_tarde, "u1", 13.5, msgs)
    assert res_tarde["meal_type"] == "almuerzo"

    # Noche (8:00 PM) con 600 kcal y slot merienda -> promueve a cena
    args_noche = {"meal_type": "merienda", "calories": 600}
    res_noche = _promover_slot_plato_fuerte_tarde(args_noche, "u1", 20.0, msgs)
    assert res_noche["meal_type"] == "cena"


@patch("db_facts.get_consumed_meals_today", return_value=[])
def test_corregir_almuerzo_prematuro(mock_meals):
    # Caso 1: 11:11 AM con almuerzo generado por LLM sin decir almuerzo -> corrige a desayuno
    msgs = [HumanMessage(content="2 huevos · Miel de abeja")]
    args = {"meal_type": "almuerzo", "meal_name": "Tostadas francesas con sirope", "calories": 600}
    res = _corregir_almuerzo_prematuro(args, "u1", 11.183, msgs)
    assert res["meal_type"] == "desayuno"

    # Caso 2: 11:11 AM pero usuario dijo explícitamente "almorcé" -> respeta almuerzo
    msgs_exp = [HumanMessage(content="almorcé temprano hoy")]
    args_exp = {"meal_type": "almuerzo", "meal_name": "Pollo guisado", "calories": 600}
    res_exp = _corregir_almuerzo_prematuro(args_exp, "u1", 11.183, msgs_exp)
    assert res_exp["meal_type"] == "almuerzo"

    # Caso 3: 12:15 PM con tostadas francesas y desayuno vacío -> brunch tardío a desayuno
    args_brunch = {"meal_type": "almuerzo", "meal_name": "Tostadas francesas con sirope", "calories": 600}
    res_brunch = _corregir_almuerzo_prematuro(args_brunch, "u1", 12.25, msgs)
    assert res_brunch["meal_type"] == "desayuno"

    # Caso 4: 1:30 PM (almuerzo normal) -> respeta almuerzo
    args_norm = {"meal_type": "almuerzo", "meal_name": "Arroz con pollo", "calories": 650}
    res_norm = _corregir_almuerzo_prematuro(args_norm, "u1", 13.5, msgs)
    assert res_norm["meal_type"] == "almuerzo"


@patch("db_facts.get_consumed_meals_today", return_value=[{"meal_type": "desayuno"}])
def test_corregir_almuerzo_prematuro_si_ya_desayuno(mock_meals):
    # Si ya desayunó hoy a las 8 AM y a las 11:15 come almuerzo temprano -> respeta almuerzo
    msgs = [HumanMessage(content="Plato de comida")]
    args = {"meal_type": "almuerzo", "meal_name": "Pollo guisado", "calories": 500}
    res = _corregir_almuerzo_prematuro(args, "u1", 11.25, msgs)
    assert res["meal_type"] == "almuerzo"
