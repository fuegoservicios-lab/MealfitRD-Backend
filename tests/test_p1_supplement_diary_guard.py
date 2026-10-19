# -*- coding: utf-8 -*-
"""[P1-SUPPLEMENT-SLOT-GUARD · 2026-10-19] Pruebas del guard de suplementos en el diario:
1. `meal_type='suplemento'` es reconocido como canónico en los 5 idiomas.
2. Una toma de suplemento (o no calórica como creatina en cápsulas) NUNCA ocupa desayuno, almuerzo o cena.
3. `comidas_sin_registrar` no cuenta suplementos ni tomas de 0 kcal como desayuno/almuerzo/cena completados.
4. `log_consumed_meal` advierte al asistente que no diga «cuando almuerces» si el desayuno sigue pendiente.
"""
from __future__ import annotations

import pytest
import tools
from chat_history_context import slot_canonico, comidas_sin_registrar

_UID = "11111111-1111-1111-1111-111111111111"


def test_suplemento_es_canonico_y_tiene_sinonimos():
    assert tools._resolver_meal_type("suplemento") == "suplemento"
    assert tools._resolver_meal_type("Suplemento") == "suplemento"
    assert tools._resolver_meal_type("supplement") == "suplemento"
    assert tools._resolver_meal_type("supplements") == "suplemento"
    assert tools._resolver_meal_type("integratore") == "suplemento"
    assert tools._resolver_meal_type("complement") == "suplemento"


def test_slot_canonico_suplemento_es_none():
    # 'suplemento' no mapea a ninguna de las 4 franjas del plan (desayuno/almuerzo/merienda/cena)
    assert slot_canonico("suplemento") is None
    assert slot_canonico("supplement") is None


def test_comidas_sin_registrar_ignora_suplementos_y_cero_kcal():
    # Una cápsula de creatina con meal_type='suplemento' no quita el desayuno
    rows_supp = [{"meal_type": "suplemento", "meal_name": "1 cápsula de creatina", "calories": 0}]
    assert comidas_sin_registrar(rows_supp) == ["desayuno", "almuerzo", "cena"]

    # Incluso si una fila legacy vino con meal_type='desayuno' pero es creatina/suplemento de 0 kcal
    rows_legacy = [{"meal_type": "desayuno", "meal_name": "1 cápsula de creatina", "calories": 0}]
    assert comidas_sin_registrar(rows_legacy) == ["desayuno", "almuerzo", "cena"]

    # Pero un desayuno real con calorías sí lo satisface
    rows_real = [{"meal_type": "desayuno", "meal_name": "Huevos revueltos", "calories": 320}]
    assert comidas_sin_registrar(rows_real) == ["almuerzo", "cena"]


@pytest.fixture
def mock_diario(monkeypatch):
    registros = []
    monkeypatch.setattr(tools, "db_log_consumed_meal",
                        lambda *a, **k: registros.append((a, k)) or "row-supp-1")
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_nota_comidas_sin_registrar", lambda *a, **k: "")
    return registros


def test_log_consumed_meal_reclasifica_creatina_de_desayuno_a_suplemento(mock_diario):
    out = tools.log_consumed_meal.func(
        _UID,
        meal_name="1 cápsula de creatina",
        calories=0,
        protein=0,
        meal_type="desayuno",
    )
    assert len(mock_diario) == 1
    _, kwargs = mock_diario[0]
    # Se reclasificó automáticamente a suplemento
    assert kwargs["meal_type"] == "suplemento"
    assert "como suplemento" in out
    assert "Aviso para el asistente: registraste '1 cápsula de creatina' como suplemento" in out
    assert "NUNCA le digas «cuando almuerces»" in out


def test_log_consumed_meal_respeta_suplemento_explicito(mock_diario):
    out = tools.log_consumed_meal.func(
        _UID,
        meal_name="Creatina Monohidrato",
        calories=0,
        protein=0,
        suplemento="Creatina",
        meal_type="suplemento",
    )
    assert len(mock_diario) == 1
    _, kwargs = mock_diario[0]
    assert kwargs["meal_type"] == "suplemento"
    assert "como suplemento" in out
