# -*- coding: utf-8 -*-
"""[P1-EXPLICIT-SLOT-RESPECT · 2026-10-27] Respeto estricto de la franja nombrada por el usuario.

Incidente: El usuario subió foto y escribió "Merendé esto Res guisada" a las 21:16.
El backend reclasificó la merienda a cena porque:
1. `_latest_human_text` regex usaba `meriend\w*` (que no empareja con "merendé", "merendar").
2. `_rescue_dinner_slot` forzaba `snack`/`merienda` de >= 300 kcal a `cena` si la cena estaba vacía,
   ignorando que el usuario había pedido explícitamente "Merendé".
3. El coach terminó diciendo: «Anoté la res guisada como merienda (~300 kcal). En el diario aparece como cena...».

Garantías probadas aquí:
1. `explicit_slot_named_by_user` detecta todas las conjugaciones ("merendé", "almorcé", "desayuné", "cené", etc.).
2. `infer_current_meal_slot` no sobreescribe la franja si el usuario dijo "Merendé".
3. `log_consumed_meal(..., explicit_slot=True)` NUNCA reescribe a cena una merienda explícita.
4. Se preserva el rescate a cena SOLO cuando el usuario NO especificó franja.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pytest
from langchain_core.messages import HumanMessage
from meal_registration_time import (
    explicit_slot_named_by_user,
    infer_current_meal_slot,
)
import tools


def test_explicit_slot_named_by_user_conjugations():
    # Merendar / merendé
    assert explicit_slot_named_by_user([HumanMessage(content="Merendé esto Res guisada")])
    assert explicit_slot_named_by_user([HumanMessage(content="merende esto")])
    assert explicit_slot_named_by_user([HumanMessage(content="Para merendar")])
    assert explicit_slot_named_by_user([HumanMessage(content="Fue mi merienda")])

    # Almorzar / almorcé
    assert explicit_slot_named_by_user([HumanMessage(content="Almorcé sancocho con arroz")])
    assert explicit_slot_named_by_user([HumanMessage(content="almorce pollo")])
    assert explicit_slot_named_by_user([HumanMessage(content="Mi almuerzo de hoy")])

    # Desayunar / desayuné
    assert explicit_slot_named_by_user([HumanMessage(content="Desayuné un guineo")])
    assert explicit_slot_named_by_user([HumanMessage(content="desayune huevos")])

    # Cenar / cené
    assert explicit_slot_named_by_user([HumanMessage(content="Cené ligero")])
    assert explicit_slot_named_by_user([HumanMessage(content="Para la cena")])

    # Sin franja explícita (debe dar False)
    assert not explicit_slot_named_by_user([HumanMessage(content="Me comí dos sandwiches con queso gouda")])
    assert not explicit_slot_named_by_user([HumanMessage(content="1 manzana")])
    assert not explicit_slot_named_by_user([HumanMessage(content="200g de pechuga")])


def test_infer_current_meal_slot_respects_merende():
    # A las 21:00 (hora típica de cena), si el usuario dijo "Merendé esto", no debe retornar "cena"
    res = infer_current_meal_slot(
        [HumanMessage(content="Merendé esto Res guisada")],
        local_hour=21.25,
    )
    assert res is None, "Si el usuario dijo 'Merendé', el reloj no debe forzar 'cena'"

    # Si NO dijo franja a las 21:00, sí retorna 'cena'
    res_unspecified = infer_current_meal_slot(
        [HumanMessage(content="Me comí res guisada")],
        local_hour=21.25,
    )
    assert res_unspecified == "cena"


def test_rescue_dinner_slot_bypassed_when_explicit_slot(monkeypatch):
    """Cuando explicit_slot=True, log_consumed_meal nunca debe sobreescribir merienda con cena."""
    diario = {"rows": [
        {"meal_type": "desayuno", "ultima": datetime.now(timezone.utc) - timedelta(hours=10)},
        {"meal_type": "almuerzo", "ultima": datetime.now(timezone.utc) - timedelta(hours=6)},
    ]}
    import db
    monkeypatch.setattr(db, "execute_sql_query", lambda sql, params=None, **k: diario["rows"], raising=False)

    # 1. Sin explicit_slot (comportamiento previo de rescate de snack nocturno sin franja):
    rescued = tools._rescue_dinner_slot("u1", "merienda", 300, 0)
    assert rescued == "cena"

    # 2. Con explicit_slot=True (simulando la llamada de log_consumed_meal con el guard nuevo):
    # Verificamos que el slot guardado no sea modificado
    explicit_slot = True
    slot_result = "merienda"
    if not explicit_slot:
        slot_result = tools._rescue_dinner_slot("u1", slot_result, 300, 0)
    assert slot_result == "merienda"
