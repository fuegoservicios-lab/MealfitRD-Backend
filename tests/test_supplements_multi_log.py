# -*- coding: utf-8 -*-
"""Regresión [P1-SUPP-MULTI-DEDUP]: registrar 2+ suplementos distintos en el mismo turno
(ej. 'Me tome la creatina y las multivitaminas') no debe ser tragado por dedup 0-kcal ni por anti-clone."""
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage

import agent
import db_facts
import tools


def test_dedup_suplementos_distintos_0kcal_no_colisionan():
    """Dos suplementos distintos (ambos 0 kcal y sin ingredientes) deben consultar por meal_name en dedup."""
    queries = []

    def fake_query(sql, params=None, fetch_one=False, fetch_all=False):
        queries.append((sql, params))
        return None  # ninguno existe todavía con ese nombre exacto

    def fake_write(sql, params=None, returning=False):
        return [{"id": "supp-id-1"}]

    with patch.object(db_facts, "execute_sql_query", side_effect=fake_query), \
         patch.object(db_facts, "execute_sql_write", side_effect=fake_write), \
         patch.object(db_facts, "connection_pool", object()):
        r1 = db_facts.log_consumed_meal(
            user_id="u1",
            meal_name="Creatine 5000 Creatine Monohydrate Capsules (Natural Sport)",
            calories=0, protein=0, carbs=0, healthy_fats=0,
            ingredients=[], meal_type="suplemento",
        )
        r2 = db_facts.log_consumed_meal(
            user_id="u1",
            meal_name="DESIGNS FOR SPORT MULTI + PHYTO",
            calories=0, protein=0, carbs=0, healthy_fats=0,
            ingredients=[], meal_type="suplemento",
        )

    assert r1 == "supp-id-1"
    assert r2 == "supp-id-1"
    assert len(queries) == 2
    assert "LOWER(TRIM(meal_name)) = LOWER(TRIM(%s))" in queries[0][0]
    assert "LOWER(TRIM(meal_name)) = LOWER(TRIM(%s))" in queries[1][0]
    assert queries[1][1][1] == "DESIGNS FOR SPORT MULTI + PHYTO"


def test_anti_clone_permite_segundo_suplemento_distinto_mismo_turno():
    """Con 'Me tome la creatina y las multivitaminas', el segundo suplemento distinto se permite
    incluso si ambos nombres de marca comparten 'Sport' y comparten franja."""
    user_text = "Me tome la creatina y las multivitaminas"
    msgs = [
        HumanMessage(content=user_text),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc-1",
                "name": "log_consumed_meal",
                "args": {
                    "meal_name": "Creatine 5000 Creatine Monohydrate Capsules (Natural Sport)",
                    "meal_type": "suplemento",
                    "calories": 0,
                    "suplemento": True,
                },
            }],
        ),
    ]
    # Segundo suplemento distinto -> permitido
    assert agent._es_segundo_plato_distinto_del_turno(
        {
            "meal_name": "DESIGNS FOR SPORT MULTI + PHYTO",
            "meal_type": "suplemento",
            "calories": 0,
            "suplemento": True,
        },
        msgs,
        user_text,
        current_tool_id="tc-2",
    ) is True

    # Mismo suplemento clonado -> sigue bloqueado
    assert agent._es_segundo_plato_distinto_del_turno(
        {
            "meal_name": "Creatine 5000 Creatine Monohydrate Capsules (Natural Sport)",
            "meal_type": "suplemento",
            "calories": 0,
            "suplemento": True,
        },
        msgs,
        user_text,
        current_tool_id="tc-3",
    ) is False


def test_correct_consumed_meal_acepta_meal_type_none():
    """Pasar meal_type=None a correct_consumed_meal no debe fallar validación de Pydantic."""
    schema = tools.correct_consumed_meal.args_schema
    parsed = schema(user_id="u1", meal_id="m1", meal_type=None, calories=100)
    assert parsed.meal_type is None
    assert parsed.calories == 100
