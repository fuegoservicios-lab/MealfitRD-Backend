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


# ===========================================================================
# [P1-SUPPLEMENT-DIARY-GUARD · 2026-10-20] Guard en correct_consumed_meal
# ===========================================================================

def test_correct_consumed_meal_rechaza_sobreescribir_suplemento_con_almuerzo(monkeypatch):
    """Caso real en producción (19-oct): usuario anotó creatina a las 9am y a las 1pm
    mandó foto de 'Res' (arroz, habichuelas y carne). El agente llamó correct_consumed_meal
    sobre la fila de la creatina, borrándola del diario. El guard DEBE rechazarlo."""
    fila_creatina = {
        "meal_name": "Creatina (1 cápsula)",
        "meal_type": "suplemento",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "consumed_at": "2026-10-19T09:26:00+00:00",
        "inventory_synced_at": None,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_creatina)
    updates = []
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda *a, **k: updates.append((a, k)) or "row-supp-1")

    out = tools.correct_consumed_meal.func(
        user_id=_UID,
        meal_id="row-supp-1",
        meal_name="Arroz con habichuelas y carne de res",
        meal_type="almuerzo",
        calories=620,
        protein=38,
        carbs=75,
        healthy_fats=18,
    )

    assert "RECHAZADO" in out
    assert "SUPLEMENTO" in out
    assert "log_consumed_meal" in out
    assert len(updates) == 0, "No debió ejecutar update en la DB"


def test_correct_consumed_meal_rechaza_sobreescribir_suplemento_con_nombre_comida_sin_meal_type(monkeypatch):
    """Incluso si el agente no pasa meal_type pero pasa un plato como 'Res' o 'Pollo',
    el guard debe detectar que la fila original era creatina/suplemento y rechazar."""
    fila_creatina = {
        "meal_name": "1 cápsula de creatina",
        "meal_type": "suplemento",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "consumed_at": "2026-10-19T09:26:00+00:00",
        "inventory_synced_at": None,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_creatina)
    updates = []
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda *a, **k: updates.append((a, k)) or "row-supp-1")

    out = tools.correct_consumed_meal.func(
        user_id=_UID,
        meal_id="row-supp-1",
        meal_name="Carne de res guisada",
        calories=350,
    )

    assert "RECHAZADO" in out
    assert "SUPLEMENTO" in out
    assert "log_consumed_meal" in out
    assert len(updates) == 0


def test_correct_consumed_meal_rechaza_sobreescribir_comida_con_suplemento(monkeypatch):
    """Guard inverso: una comida principal (almuerzo) no puede convertirse en un suplemento."""
    fila_almuerzo = {
        "meal_name": "Pollo al horno con arroz",
        "meal_type": "almuerzo",
        "calories": 550,
        "protein": 40,
        "carbs": 60,
        "healthy_fats": 12,
        "consumed_at": "2026-10-19T13:30:00+00:00",
        "inventory_synced_at": None,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_almuerzo)
    updates = []
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda *a, **k: updates.append((a, k)) or "row-almuerzo-1")

    out = tools.correct_consumed_meal.func(
        user_id=_UID,
        meal_id="row-almuerzo-1",
        meal_name="Creatina Monohidrato",
        meal_type="suplemento",
        calories=0,
    )

    assert "RECHAZADO" in out
    assert "comida principal" in out
    assert "log_consumed_meal" in out
    assert len(updates) == 0


def test_correct_consumed_meal_permite_correccion_valida_de_suplemento(monkeypatch):
    """Una corrección legítima sobre un suplemento (ej. aclarar dosis o nombre del suplemento)
    SÍ debe proceder con normalidad."""
    fila_creatina = {
        "meal_name": "Creatina (1 cápsula)",
        "meal_type": "suplemento",
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "consumed_at": "2026-10-19T09:26:00+00:00",
        "inventory_synced_at": None,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_creatina)
    updates = []
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda *a, **k: updates.append((a, k)) or "row-supp-1")
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")

    out = tools.correct_consumed_meal.func(
        user_id=_UID,
        meal_id="row-supp-1",
        meal_name="Creatina Monohidratada (5g en polvo)",
    )

    assert "Corregido" in out
    assert len(updates) == 1


def test_correct_consumed_meal_permite_correccion_valida_de_comida(monkeypatch):
    """Una corrección normal de una comida (ej. macros de almuerzo) SÍ debe proceder."""
    fila_almuerzo = {
        "meal_name": "Pollo al horno con arroz",
        "meal_type": "almuerzo",
        "calories": 550,
        "protein": 40,
        "carbs": 60,
        "healthy_fats": 12,
        "consumed_at": "2026-10-19T13:30:00+00:00",
        "inventory_synced_at": None,
    }
    monkeypatch.setattr(tools, "_leer_fila_diario", lambda uid, mid: fila_almuerzo)
    updates = []
    monkeypatch.setattr(tools, "db_update_consumed_meal", lambda *a, **k: updates.append((a, k)) or "row-almuerzo-1")
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")

    out = tools.correct_consumed_meal.func(
        user_id=_UID,
        meal_id="row-almuerzo-1",
        meal_name="Pollo al horno con ensalada",
        calories=450,
        protein=42,
    )

    assert "Corregido" in out
    assert len(updates) == 1
