# -*- coding: utf-8 -*-
"""[P1-ZERO-CALORIE-GUARD · 2026-10-27] Ningún alimento real, snack o comida puede registrarse
con 0 kcal en el diario bajo la excusa de «no sé qué alimento es».

Caso vivo: darianbobea229@gmail.com subió foto de tarro de crema de maní (con tabla nutricional)
escribiendo «Creamy». El coach preguntó si el chofán de 2 horas antes llevaba crema, y en el
turno siguiente registró «Creamy» en 0 kcal diciendo «Anoté 'Creamy' como snack, pero quedó en 0 kcal
porque no sé qué alimento o bebida es».
"""
import pytest
from tools import _es_cero_calorias_valido, log_consumed_meal
from prompts.chat_agent import (
    build_tools_instructions,
    build_tools_instructions_stream,
    build_vision_context,
)


def test_es_cero_calorias_valido_permite_agua_y_bebidas_zero():
    # Agua y bebidas zero
    assert _es_cero_calorias_valido("Agua 500 ml") is True
    assert _es_cero_calorias_valido("Vaso de agua") is True
    assert _es_cero_calorias_valido("Coca-Cola Zero") is True
    assert _es_cero_calorias_valido("Refresco light") is True
    assert _es_cero_calorias_valido("Café negro solo") is True
    assert _es_cero_calorias_valido("Té verde sin azúcar") is True
    assert _es_cero_calorias_valido("Sal marina") is True

    # Suplementos sin calorías
    assert _es_cero_calorias_valido("Creatina monohidratada", meal_type="suplemento") is True
    assert _es_cero_calorias_valido("Cápsula de Vitamina D", meal_type="suplemento") is True
    assert _es_cero_calorias_valido("Pastilla de Magnesio", suplemento="Magnesio") is True


def test_es_cero_calorias_valido_rechaza_alimentos_reales_y_snacks():
    assert _es_cero_calorias_valido("Creamy") is False
    assert _es_cero_calorias_valido("Mantequilla de maní") is False
    assert _es_cero_calorias_valido("Snack") is False
    assert _es_cero_calorias_valido("Manzana") is False
    assert _es_cero_calorias_valido("Pollo guisado") is False
    assert _es_cero_calorias_valido("Yogur") is False
    assert _es_cero_calorias_valido("Galleta") is False


def test_log_consumed_meal_rechaza_alimento_desconocido_en_cero_kcal(monkeypatch):
    import tools
    st = {"logged": False}
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: st.update({"logged": True}) or "mid-1")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, cal, d: mt)

    # Intento de registrar "Creamy" en 0 kcal
    out = tools.log_consumed_meal.func("uid-test", "Creamy", calories=0, protein=0, carbs=0, healthy_fats=0)
    assert "no puede registrarse con 0 kcal si es un alimento o comida real" in out
    assert "⚠️ NO REGISTRADO" in out
    assert st["logged"] is False


def test_log_consumed_meal_permite_agua_en_cero_kcal(monkeypatch):
    import tools
    st = {"logged": False}
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: st.update({"logged": True}) or "mid-1")
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, cal, d: mt)
    monkeypatch.setattr(tools, "_nota_total_del_dia", lambda *a, **k: "")
    monkeypatch.setattr(tools, "_nota_comidas_sin_registrar", lambda *a, **k: "")

    out = tools.log_consumed_meal.func("uid-test", "Agua natural 500 ml", calories=0, protein=0, carbs=0, healthy_fats=0)
    assert out.startswith("¡Éxito!")
    assert st["logged"] is True


def test_prompts_contienen_reglas_anti_cero_kcal_e_independencia_fotos():
    instr = build_tools_instructions("u")
    stream_instr = build_tools_instructions_stream("u")

    for prompt_text in (instr, stream_instr):
        assert "PROHIBIDO 0 KCAL EN ALIMENTO O SNACK" in prompt_text
        assert "PROHIBIDO CON FOTO DE PLATO O ETIQUETA NUEVA" in prompt_text
        assert "ni preguntes si la comida anterior llevaba ese ingrediente" in prompt_text
