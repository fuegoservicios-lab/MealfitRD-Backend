# [P1-SPECIFIC-PROTEIN · 2026-10-31]
# Tests para rectificación de proteína específica sobre genérico 'carne' o 'carne de res'.

import pytest
from langchain_core.messages import HumanMessage
from agent import _rectificar_proteina_especifica_en_args


def test_rectificar_pollo_guisado_sobre_carne_de_res():
    # Caso real de Darian: usuario escribió "Arroz blanco con carne ensalada\nPollo guisado"
    # y el análisis o tool trajo "carne guisada" / "carne de res guisada".
    user_text = "Arroz blanco con carne ensalada\nPollo guisado"
    messages = [HumanMessage(content=user_text)]
    
    tool_args = {
        "meal_name": "Arroz blanco con carne guisada y ensalada",
        "ingredients": [
            "arroz blanco cocido (≈200 g)",
            "carne de res guisada (≈100 g)",
            "lechuga (≈30 g)",
            "tomate (≈30 g)",
            "salsa de carne (≈30 g)"
        ],
        "calories": 495,
        "protein": 32,
    }
    
    rect = _rectificar_proteina_especifica_en_args(tool_args, messages)
    
    assert "pollo guisado" in rect["meal_name"].lower()
    assert "carne guisada" not in rect["meal_name"].lower()
    assert any("pollo guisado" in ing.lower() for ing in rect["ingredients"])
    assert not any("carne de res" in ing.lower() for ing in rect["ingredients"])
    assert any("salsa de pollo guisado" in ing.lower() for ing in rect["ingredients"])


def test_rectificar_cerdo_sobre_carne_de_res():
    user_text = "Moro con carne\nCerdo guisado"
    messages = [HumanMessage(content=user_text)]
    
    tool_args = {
        "meal_name": "Moro con carne guisada",
        "ingredients": [
            "moro de habichuelas (≈200 g)",
            "carne de res guisada (≈100 g)",
        ],
        "calories": 550,
        "protein": 30,
    }
    
    rect = _rectificar_proteina_especifica_en_args(tool_args, messages)
    
    assert "cerdo guisado" in rect["meal_name"].lower()
    assert any("cerdo guisado" in ing.lower() for ing in rect["ingredients"])
    assert not any("carne de res" in ing.lower() for ing in rect["ingredients"])


def test_no_modifica_si_usuario_dijo_res():
    user_text = "Arroz con carne de res guisada"
    messages = [HumanMessage(content=user_text)]
    
    tool_args = {
        "meal_name": "Arroz con carne de res guisada",
        "ingredients": [
            "arroz blanco cocido (≈200 g)",
            "carne de res guisada (≈100 g)",
        ],
    }
    
    rect = _rectificar_proteina_especifica_en_args(tool_args, messages)
    assert "carne de res guisada" in rect["meal_name"].lower()
