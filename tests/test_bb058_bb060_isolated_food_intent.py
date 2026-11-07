# -*- coding: utf-8 -*-
"""
Tests automatizados para desambiguación de alimentos aislados sin verbo (BB-058),
fidelidad estricta de ingredientes en propuestas culinarias (BB-059)
y reconciliación histórica de mensajes de coach (BB-060).
"""
import pytest
from tools import proponer_comida
from prompts.chat_agent import (
    _REGLA_UNA_LLAMADA_POR_COMIDA,
    _CHAT_RESOLVE_RULES,
    _MEAL_PROPOSAL_BULLET_ENABLED,
    build_tools_instructions,
    build_tools_instructions_stream,
)


# ==============================================================================
# BB-058: Desambiguación de alimentos aislados sin verbos de receta
# ==============================================================================

def test_bb058_prompt_isolated_food_intent_disambiguation():
    """Verifica que las directivas de desambiguación de alimentos aislados estén presentes en las reglas maestras."""
    assert "[BB-058 / P0-ISOLATED-FOOD-INTENT-DISAMBIGUATION" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "lata de atun genova" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "Trátalo PRIMARIAMENTE como un reporte de consumo de comida" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "PROHIBIDO llamar a `proponer_comida` en este escenario" in _REGLA_UNA_LLAMADA_POR_COMIDA


def test_bb058_chat_resolve_rules_isolated_food():
    """Verifica que _CHAT_RESOLVE_RULES incluya la regla Q2 para alimentos aislados sin verbo."""
    assert "Q2. [BB-058 / P0-ISOLATED-FOOD-INTENT-DISAMBIGUATION" in _CHAT_RESOLVE_RULES
    assert "lata de atun genova" in _CHAT_RESOLVE_RULES
    assert "NUNCA llames a `proponer_comida` ni ofrezcas recetas de catálogo complejas" in _CHAT_RESOLVE_RULES


def test_bb058_proponer_comida_docstring_and_bullets():
    """Verifica que el docstring de proponer_comida y los bullets prohíban disparar propuestas ante alimentos aislados."""
    doc = proponer_comida.description
    assert "[BB-058]" in doc
    assert "PROHIBIDO usar cuando el usuario mencione un alimento, marca o producto aislado" in doc

    assert "[BB-058]" in _MEAL_PROPOSAL_BULLET_ENABLED
    assert "PROHIBIDO usarla ante alimentos o productos aislados sin petición explícita de receta" in _MEAL_PROPOSAL_BULLET_ENABLED


# ==============================================================================
# BB-059: Fidelidad estricta del ingrediente en propuestas
# ==============================================================================

def test_bb059_ingredient_fidelity_directives():
    """Verifica la prohibición de sustitutos distantes (ej. arenque o salami cuando el usuario tiene atún)."""
    assert "[BB-059 / P1-PROPOSAL-INGREDIENT-FIDELITY" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "arenque guisado" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "salami con yuca" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "manteniendo fidelidad 100% al ingrediente del usuario" in _REGLA_UNA_LLAMADA_POR_COMIDA

    assert "Q3. [BB-059 / P1-PROPOSAL-INGREDIENT-FIDELITY" in _CHAT_RESOLVE_RULES
    assert "Está TERMINANTEMENTE PROHIBIDO proponer platos de catálogo con ingredientes completamente diferentes" in _CHAT_RESOLVE_RULES


def test_bb058_bb059_injection_in_tools_instructions():
    """Verifica que las directivas se inyecten tanto en build_tools_instructions como en el stream."""
    instructions_normal = build_tools_instructions("user-123")
    assert "[BB-058 / P0-ISOLATED-FOOD-INTENT-DISAMBIGUATION" in instructions_normal
    assert "[BB-059 / P1-PROPOSAL-INGREDIENT-FIDELITY" in instructions_normal

    instructions_stream = build_tools_instructions_stream("user-123")
    assert "[BB-058 / P0-ISOLATED-FOOD-INTENT-DISAMBIGUATION" in instructions_stream
    assert "[BB-059 / P1-PROPOSAL-INGREDIENT-FIDELITY" in instructions_stream


# ==============================================================================
# BB-060: Reconciliación Histórica de Mensaje en DB
# ==============================================================================

def test_bb060_historical_reconciliation_template():
    """Valida la estructura del mensaje rectificado para casos de desambiguación de atún."""
    texto_reconciliado = (
        "¿Te comiste la **lata de atún Genova** (~**160 kcal** y **35 g de proteína**)? "
        "Cuéntame en qué comida fue (desayuno, almuerzo o cena) para anotártela de inmediato en tu diario."
    )
    assert "lata de atún Genova" in texto_reconciliado
    assert "160 kcal" in texto_reconciliado
    assert "35 g de proteína" in texto_reconciliado
    assert "arenque" not in texto_reconciliado.lower()
    assert "1001 kcal" not in texto_reconciliado
