# -*- coding: utf-8 -*-
"""
Tests automatizados para robustez de hidratación, auto-redondeo defensivo
y directivas contra fugas técnicas (BB-053 a BB-055).
"""
import pytest
from unittest.mock import patch, MagicMock

from tools import log_water_glass
from prompts.chat_agent import _REGLA_UNA_LLAMADA_POR_COMIDA, build_tools_instructions


# ==============================================================================
# BB-054: Auto-redondeo defensivo en log_water_glass
# ==============================================================================

def test_bb054_log_water_glass_auto_rounding_decimal():
    """Valida que valores arbitrarios como 0.75, 1.25 o 0.3 se redondeen sin error a múltiplos de 0.5."""
    with patch("db.connection_pool", True), \
         patch("tools._local_date_str_for_user", return_value="2026-11-06"), \
         patch("db.execute_sql_write", return_value=[{"glasses": 2.0}]), \
         patch("routers.plans._compute_water_goal", return_value={"goal": 8}), \
         patch("ajustes_de_la_app.encender_hidratacion_al_anotar_agua", return_value=""):

        # 0.75 (ej. 3/4 de vaso) debe redondearse a 1.0 sin lanzar error
        res_075 = log_water_glass.invoke({"user_id": "u1", "count_delta": 0.75})
        assert "Error: el delta debe ser multiplo de 0.5" not in res_075
        assert "Listo: se sumaron 1 vaso(s)" in res_075

        # 1.25 debe redondearse a 1.5 sin error
        res_125 = log_water_glass.invoke({"user_id": "u1", "count_delta": 1.25})
        assert "Error: el delta debe ser multiplo de 0.5" not in res_125
        assert "Listo: se sumaron 1.5 vaso(s)" in res_125

        # 0.3 debe redondearse a 0.5 (medio vaso) sin error
        res_03 = log_water_glass.invoke({"user_id": "u1", "count_delta": 0.3})
        assert "Error: el delta debe ser multiplo de 0.5" not in res_03
        assert "Listo: se sumaron 0.5 vaso(s)" in res_03


def test_bb054_log_water_glass_bottle_equivalences():
    """Valida registro exacto de botellas enteras (2 vasos) y 3/4 de botella (1.5 vasos)."""
    with patch("db.connection_pool", True), \
         patch("tools._local_date_str_for_user", return_value="2026-11-06"), \
         patch("db.execute_sql_write", return_value=[{"glasses": 5.0}]), \
         patch("routers.plans._compute_water_goal", return_value={"goal": 8}), \
         patch("ajustes_de_la_app.encender_hidratacion_al_anotar_agua", return_value=""):

        # 1 botella = 2 vasos
        res_bot = log_water_glass.invoke({"user_id": "u1", "count_delta": 2.0})
        assert "Listo: se sumaron 2 vaso(s)" in res_bot

        # 3/4 de botella = 1.5 vasos
        res_34 = log_water_glass.invoke({"user_id": "u1", "count_delta": 1.5})
        assert "Listo: se sumaron 1.5 vaso(s)" in res_34


# ==============================================================================
# BB-053 & BB-055: Directiva Prompt Anti-Fugas Técnicas y Equivalencias
# ==============================================================================

def test_bb055_prompt_anti_leak_hidratacion():
    """Verifica la directiva BB-055 que prohíbe decir 'el registro solo acepta medios vasos'."""
    assert "[BB-055 / P0-NO-HYDRATION-TECHNICAL-LEAK" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "el registro solo acepta medios vasos" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "1 botella comercial estándar (500 ml / 16.9 oz) = SIEMPRE 2 vasos" in _REGLA_UNA_LLAMADA_POR_COMIDA
    assert "3/4 de botella de agua (~375 ml) = 1.5 vasos" in _REGLA_UNA_LLAMADA_POR_COMIDA


def test_bb053_docstring_log_water_glass_equivalencias():
    """Verifica que el docstring de log_water_glass documente las equivalencias de botellas y termos."""
    doc = log_water_glass.description
    assert "1 botella comercial estándar" in doc
    assert "2 vasos" in doc
    assert "3/4 botella" in doc
    assert "1.5 vasos" in doc
