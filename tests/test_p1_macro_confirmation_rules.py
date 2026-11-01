# [P1-PLAN-BB-012 · 2026-11-01] Tests para forzar macros en confirmación inicial y claridad en ajustes de ingredientes
import pytest
from prompts.chat_agent import (
    _CHAT_VOICE_RULES,
    build_tools_instructions,
    build_tools_instructions_stream,
    _PLATO_INSTRUCCION,
)


def test_rule_a_mandates_bold_franja_and_macros():
    """Regla A debe exigir nombrar la franja en negrita e incluir siempre calorías y proteína."""
    assert "NOMBRA SIEMPRE EXPLÍCITAMENTE la franja en negrita" in _CHAT_VOICE_RULES
    assert "INCLUYE SIEMPRE los macros estimados principales" in _CHAT_VOICE_RULES
    assert "calorías y proteína obligatorias" in _CHAT_VOICE_RULES


def test_log_consumed_meal_instructions_mandate_macros():
    """Tanto la versión inline como stream deben instruir la inclusión de macros en la confirmación."""
    inline = build_tools_instructions(user_id="test-user")
    stream = build_tools_instructions_stream(user_id="test-user")

    assert "INCLUYE SIEMPRE las calorías y proteína estimadas" in inline
    assert "NUNCA dejes la franja omitida ni omitas los macros en la primera confirmación" in inline

    assert "INCLUYE SIEMPRE las calorías y proteína estimadas" in stream
    assert "nombra SIEMPRE la franja en negrita" in stream


def test_correct_consumed_meal_mandates_ingredient_transparency():
    """Las instrucciones de corrección deben guiar la transparencia cuando el usuario da datos exactos."""
    inline = build_tools_instructions(user_id="test-user")
    stream = build_tools_instructions_stream(user_id="test-user")

    assert "AJUSTE CON DATOS EXACTOS DE UN INGREDIENTE" in inline
    assert "aplicaste sus datos exactos a ese ingrediente y muestra con claridad cómo queda el nuevo total" in inline

    assert "AJUSTE CON DATOS EXACTOS DE UN INGREDIENTE" in stream
    assert "explica de forma transparente que aplicaste sus datos exactos a ese ingrediente" in stream

    assert "datos nutricionales exactos de un ingrediente" in _PLATO_INSTRUCCION
    assert "Ajusté el atún con tus datos exactos" in _PLATO_INSTRUCCION
