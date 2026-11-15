"""[P1-PLAN-LOTE-1019 · J6] Preguntar si un plato conviene no lo registra (batería del 14-nov, dos corridas)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guarda_pregunta_no_es_registro import bloqueo_si_solo_pregunta  # noqa: E402


@pytest.mark.parametrize("texto", [
    "¿esto está bien para mi cena o me paso?", "¿puedo comer esto ahora?", "¿Cuántas calorías tiene este plato?",
    "¿me conviene para el almuerzo?", "is this ok for dinner?", "¿Debería comerme esto o es mucho?",
])
def test_una_pregunta_de_valoracion_no_registra(texto):
    assert bloqueo_si_solo_pregunta("log_consumed_meal", texto)


@pytest.mark.parametrize("texto", [
    "me comí esto, ¿está bien?", "¿cuántas calorías tiene? anótalo", "cené esto, ¿me pasé?",
    "este fue el desayuno", "Lechoza - Con azúcar", "I ate this, is it ok?", "¿qué tal? ya me lo comí",
    "me paso de calorías hoy, anota el arroz", "",
])
def test_si_dice_que_lo_comio_o_que_lo_anote_pasa(texto):
    assert bloqueo_si_solo_pregunta("log_consumed_meal", texto) is None


def test_solo_toca_el_registro_de_comida():
    assert bloqueo_si_solo_pregunta("correct_consumed_meal", "¿esto está bien?") is None
    assert bloqueo_si_solo_pregunta("log_water_glass", "¿esto está bien?") is None


def test_el_bloqueo_pide_contestar_y_no_mentir():
    m = bloqueo_si_solo_pregunta("log_consumed_meal", "¿esto está bien para mi cena?")
    assert m.startswith("BLOQUEADO") and "NO lo registres" in m and "Nunca digas que lo anotaste" in m


def test_esta_enganchada_antes_de_ejecutar_el_registro():
    s = (Path(__file__).resolve().parent.parent / "agent.py").read_text(encoding="utf-8")
    i = s.index("[P1-PLAN-LOTE-1019 · J6]")
    bloque = s[i:i + 900]
    assert "bloqueo_si_solo_pregunta(tool_name, _extraer_ultimo_texto_usuario(messages))" in bloque
    assert "ToolMessage(content=_preg, tool_call_id=tool_id)" in bloque and "continue" in bloque
    assert i < s.index("from decision_triage import freno_de_bebida")
