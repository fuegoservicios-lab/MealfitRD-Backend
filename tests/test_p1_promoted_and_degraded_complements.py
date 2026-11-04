"""
Tests para [BB-027 / P1-PROMOTED-COMPLEMENT-GUARD · 2026-11-04]
Promoción de complementos calóricos (> 50 kcal con evaluación de TTL de 30 min)
y degradación de meriendas ligeras (<= 50 kcal) a complementos.
"""

from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import patch, MagicMock

import tools
from tools import (
    log_consumed_meal,
    correct_consumed_meal,
    _es_bebida_ligera_nombre,
    _buscar_comida_principal_reciente_ttl,
)
from prompts.chat_agent import _CHAT_VOICE_RULES


def test_es_bebida_ligera_nombre_detector():
    assert _es_bebida_ligera_nombre("Té verde") is True
    assert _es_bebida_ligera_nombre("Taza de café negro") is True
    assert _es_bebida_ligera_nombre("Infusión de manzanilla") is True
    assert _es_bebida_ligera_nombre("Agua con limón") is True
    assert _es_bebida_ligera_nombre("Caldo claro") is True
    assert _es_bebida_ligera_nombre("Café solo") is True
    assert _es_bebida_ligera_nombre("Espresso") is True
    assert _es_bebida_ligera_nombre("Té") is True
    assert _es_bebida_ligera_nombre("Café") is True

    # No son bebidas ligeras
    assert _es_bebida_ligera_nombre("Pollo al horno con arroz") is False
    assert _es_bebida_ligera_nombre("Sandwich de jamón y queso") is False
    assert _es_bebida_ligera_nombre("Galletas de avena") is False


def test_promocion_complemento_a_merienda_sin_comida_principal_reciente():
    """Un café en complemento que sube a 130 kcal por aditivos se promueve a merienda si no hay comida principal en 30 min."""
    user_id = "test-u-promo"
    meal_id = "mid-cafe-comp"
    fila_mock = {
        "id": meal_id,
        "user_id": user_id,
        "meal_type": "complemento",
        "meal_name": "Café negro",
        "calories": 5,
        "protein": 0,
        "carbs": 1,
        "healthy_fats": 0,
        "consumed_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
    }

    with patch("tools._leer_fila_diario", return_value=fila_mock), \
         patch("tools._buscar_comida_principal_reciente_ttl", return_value=None), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.db_update_consumed_meal", return_value=meal_id) as mock_update, \
         patch("tools._nota_total_del_dia", return_value=""):

        res = correct_consumed_meal.invoke({
            "user_id": user_id,
            "meal_id": meal_id,
            "meal_name": "Café con leche evaporada y azúcar",
            "calories": 130,
            "protein": 3,
            "carbs": 18,
            "healthy_fats": 4,
        })

        assert "¡Corregido!" in res
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs.get("meal_type") in ("merienda", "snack")
        assert kwargs.get("calories") == 130


def test_promocion_complemento_fusion_en_comida_principal_dentro_ttl():
    """Un café en complemento que sube a 130 kcal dentro de los 30 min de un almuerzo se fusiona en el almuerzo."""
    user_id = "test-u-fusion"
    comp_id = "mid-cafe-10"
    main_id = "mid-almuerzo-main"

    fila_comp = {
        "id": comp_id,
        "user_id": user_id,
        "meal_type": "complemento",
        "meal_name": "Café negro",
        "calories": 5,
        "protein": 0,
        "carbs": 1,
        "healthy_fats": 0,
        "consumed_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    }

    main_candidato = {
        "id": main_id,
        "user_id": user_id,
        "meal_type": "almuerzo",
        "meal_name": "Arroz con habichuelas y pechuga",
        "calories": 650,
        "protein": 45,
        "carbs": 70,
        "healthy_fats": 12,
        "ingredients": ["arroz", "habichuelas", "pechuga"],
        "consumed_at": (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat(),
    }

    with patch("tools._leer_fila_diario", return_value=fila_comp), \
         patch("tools._buscar_comida_principal_reciente_ttl", return_value=main_candidato), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.db_update_consumed_meal", return_value=main_id) as mock_update, \
         patch("tools.db_delete_consumed_meal", return_value=True) as mock_delete, \
         patch("tools._nota_total_del_dia", return_value=""):

        res = correct_consumed_meal.invoke({
            "user_id": user_id,
            "meal_id": comp_id,
            "meal_name": "Café con leche evaporada y azúcar",
            "calories": 130,
            "protein": 3,
            "carbs": 18,
            "healthy_fats": 4,
            "ingredients": ["cafe", "leche evaporada", "azucar"],
        })

        assert "¡Corregido y fusionado!" in res
        assert "almuerzo" in res
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs.get("calories") == 780  # 650 + 130
        assert kwargs.get("protein") == 48    # 45 + 3
        assert "Café con leche evaporada y azúcar" in kwargs.get("meal_name")
        mock_delete.assert_called_once_with(user_id, comp_id)


def test_degradacion_merienda_a_complemento_al_quedar_solo_infusion():
    """Si se quitan los alimentos de una merienda y solo queda un té de 2 kcal, se degrada a 'complemento'."""
    user_id = "test-u-degrade"
    meal_id = "mid-merienda-galletas"

    fila_merienda = {
        "id": meal_id,
        "user_id": user_id,
        "meal_type": "merienda",
        "meal_name": "Té verde con galletas integrales",
        "calories": 180,
        "protein": 4,
        "carbs": 32,
        "healthy_fats": 3,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
    }

    with patch("tools._leer_fila_diario", return_value=fila_merienda), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.db_update_consumed_meal", return_value=meal_id) as mock_update, \
         patch("tools._nota_total_del_dia", return_value=""):

        res = correct_consumed_meal.invoke({
            "user_id": user_id,
            "meal_id": meal_id,
            "meal_name": "Té verde sin azúcar",
            "calories": 2,
            "protein": 0,
            "carbs": 0,
            "healthy_fats": 0,
            "ingredients": ["te verde"],
        })

        assert "¡Corregido!" in res
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs.get("meal_type") == "complemento"
        assert kwargs.get("calories") == 2


def test_log_consumed_meal_bloquea_complemento_mayor_a_50_kcal():
    """log_consumed_meal con > 50 kcal jamás queda como 'complemento' (se promueve a merienda si no hay TTL)."""
    user_id = "test-u-block-comp"

    with patch("tools.db_log_consumed_meal", return_value="meal-promo-123") as mock_db, \
         patch("tools._buscar_comida_principal_reciente_ttl", return_value=None), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools._nota_total_del_dia", return_value=""), \
         patch("tools._nota_comidas_sin_registrar", return_value=""), \
         patch("db.get_user_profile", return_value={}), \
         patch("proactive_agent.mark_nudge_logged_from_diary", return_value=None):

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Café moca con crema batida",
            "calories": 220,
            "protein": 4,
            "carbs": 28,
            "healthy_fats": 10,
            "meal_type": "complemento",
        })

        assert "Éxito" in res or "registrado" in res.lower()
        mock_db.assert_called_once()
        _, kwargs = mock_db.call_args
        assert kwargs.get("meal_type") in ("merienda", "snack")
        assert kwargs.get("meal_type") != "complemento"


def test_log_consumed_meal_fusiona_complemento_mayor_a_50_kcal_en_comida_principal():
    """log_consumed_meal con > 50 kcal y comida principal en TTL <= 30 min se fusiona directamente."""
    user_id = "test-u-log-fusion"
    main_id = "mid-lunch-active"

    main_candidato = {
        "id": main_id,
        "user_id": user_id,
        "meal_type": "almuerzo",
        "meal_name": "Moro de guandules con pollo guisado",
        "calories": 700,
        "protein": 50,
        "carbs": 80,
        "healthy_fats": 15,
        "ingredients": ["arroz", "guandules", "pollo"],
        "consumed_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
    }

    with patch("tools._buscar_comida_principal_reciente_ttl", return_value=main_candidato), \
         patch("tools.db_update_consumed_meal", return_value=main_id) as mock_update, \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools._nota_total_del_dia", return_value=""), \
         patch("tools._nota_comidas_sin_registrar", return_value=""), \
         patch("db.get_user_profile", return_value={}), \
         patch("proactive_agent.mark_nudge_logged_from_diary", return_value=None):

        res = log_consumed_meal.invoke({
            "user_id": user_id,
            "meal_name": "Café con leche y azúcar",
            "calories": 120,
            "protein": 3,
            "carbs": 16,
            "healthy_fats": 4,
            "meal_type": "complemento",
        })

        assert "REGISTRADO: Sumé" in res
        assert "almuerzo recién registrado" in res
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs.get("calories") == 820  # 700 + 120
        assert kwargs.get("protein") == 53    # 50 + 3


def test_complemento_mantiene_tipo_si_sigue_bajo_umbral():
    """Una infusión corregida que sigue <= 50 kcal se mantiene como 'complemento'."""
    user_id = "test-u-keep-comp"
    meal_id = "mid-comp-stay"

    fila_comp = {
        "id": meal_id,
        "user_id": user_id,
        "meal_type": "complemento",
        "meal_name": "Té verde",
        "calories": 2,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "consumed_at": datetime.now(timezone.utc).isoformat(),
    }

    with patch("tools._leer_fila_diario", return_value=fila_comp), \
         patch("tools.nevera_activa", return_value=False), \
         patch("tools.db_update_consumed_meal", return_value=meal_id) as mock_update, \
         patch("tools._nota_total_del_dia", return_value=""):

        res = correct_consumed_meal.invoke({
            "user_id": user_id,
            "meal_id": meal_id,
            "meal_name": "Té verde con rodaja de limón y stevia",
            "calories": 8,
            "protein": 0,
            "carbs": 2,
            "healthy_fats": 0,
        })

        assert "¡Corregido!" in res
        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        # meal_type no fue forzado a merienda ni a principal
        assert kwargs.get("meal_type") is None or kwargs.get("meal_type") == "complemento"


def test_prompt_contiene_directiva_de_promocion_y_degradacion():
    assert "P1-PROMOTED-COMPLEMENT-GUARD" in _CHAT_VOICE_RULES
    assert "PROMOCIÓN Y DEGRADACIÓN DE COMPLEMENTOS" in _CHAT_VOICE_RULES
