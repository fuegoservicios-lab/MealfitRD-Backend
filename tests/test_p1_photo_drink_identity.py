# backend/tests/test_p1_photo_drink_identity.py
# [P1-PHOTO-DRINK-IDENTITY · 2026-11-03] Tests para el manejo de bebidas ambiguas en foto
# (té, infusión, café, jugo, caldo) y umbral de calorías para cerrar slots del plan.
from __future__ import annotations

import sys
from pathlib import Path
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import constants
from bebidas_de_la_foto import (
    es_bebida,
    nombre_generico_bebida,
    tipo_de_bebida,
    inyectar_dudas_bebida,
)
from vision_agent import _coerce_meal_scan, _opcion_coincide_con_aclaracion
from respuestas_de_la_foto import _platos_claros
from chat_history_context import comidas_sin_registrar


def test_es_bebida_detector_and_false_positive_guard():
    """Detecta bebidas reales y evita falsos positivos por subcadena o platos con comida sólida."""
    # Bebidas puras
    assert es_bebida("Té herbal", "Taza de té caliente") is True
    assert es_bebida("Café negro", "Taza de café") is True
    assert es_bebida("Infusión de manzanilla", "Líquido en taza") is True
    assert es_bebida("Jugo de naranja", "Vaso de jugo natural") is True
    assert es_bebida("Caldo de verduras", "Tazón de caldo claro") is True

    # Falsos positivos por subcadena 'te' o similares
    assert es_bebida("Filete de res con tostones", "Carne a la plancha") is False
    assert es_bebida("Chocolate con leche en barra", "Barra de chocolate amargo") is False
    assert es_bebida("Pastel de pollo", "Porción de pastel horneado") is False
    assert es_bebida("Tomate relleno", "Tomate con atún") is False

    # Plato completo con comida sólida + café secundario
    items_plato_mixto = [
        {"name": "mangú", "calories": 280},
        {"name": "huevo frito", "calories": 110},
        {"name": "café", "calories": 5},
    ]
    assert es_bebida("Mangú con huevo y café", "Plato con mangú y café", items_plato_mixto) is False


def test_coerce_meal_scan_tea_injects_doubts_and_mark():
    """Un té sin dudas del modelo debe recibir dudas de identidad y aditivos y marcarse con DUDAS."""
    raw = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Té herbal",
        "description": "Una taza de porcelana blanca con infusión de líquido amarillento.",
        "calories": 2,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "items": [{"name": "té herbal", "calories": 2, "quantity": 1, "unit": "taza"}],
        "dudas": [],
    }
    result = _coerce_meal_scan(raw, aclaracion=None)

    assert len(result["dudas"]) == 2
    # Duda 1: identidad
    d_id = result["dudas"][0]
    assert "té" in d_id["pregunta"].lower() or "infusión" in d_id["pregunta"].lower()
    assert len(d_id["opciones"]) >= 3
    assert sum(1 for o in d_id["opciones"] if o.get("supuesta")) == 1

    # Duda 2: aditivos
    d_ad = result["dudas"][1]
    assert "¿le añadiste algo" in d_ad["pregunta"].lower()

    # Description incluye la marca DUDAS para el coach
    assert "DUDAS (pregúntale solo esto):" in result["description"]
    # No califica como plato claro sin dudas
    assert _platos_claros(result) is False


def test_coerce_meal_scan_with_sin_azucar_caption():
    """Con aclaración 'Sin azúcar', la duda de identidad permanece y no se selecciona azúcar."""
    raw = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Té herbal",
        "description": "Una taza con té caliente.",
        "calories": 2,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "items": [{"name": "té", "calories": 2}],
        "dudas": [],
    }
    result = _coerce_meal_scan(raw, aclaracion="Sin azúcar")

    # La duda de qué es el té NO quedó resuelta por decir "sin azúcar"
    assert len(result["dudas"]) >= 1
    assert any("¿de qué es el té" in d["pregunta"].lower() for d in result["dudas"])

    # En las opciones de aditivos, no debe aparecer 'Con azúcar'
    for d in result["dudas"]:
        textos = [o["texto"].lower() for o in d.get("opciones", [])]
        assert "con azúcar" not in textos


def test_opcion_coincide_negation_guard():
    """`_opcion_coincide_con_aclaracion` nunca debe hacer match positivo con ingredientes negados."""
    duda = {"sobre": "té", "pregunta": "¿Qué le echaste?"}
    assert _opcion_coincide_con_aclaracion("Con azúcar", "sin azucar", duda) is False
    assert _opcion_coincide_con_aclaracion("Azúcar", "sin azucar", duda) is False
    assert _opcion_coincide_con_aclaracion("Con leche", "sin leche", duda) is False
    assert _opcion_coincide_con_aclaracion("Con miel", "sin miel", duda) is False

    # Coincidencias válidas
    assert _opcion_coincide_con_aclaracion("Jengibre", "té de jengibre sin azúcar", duda) is True
    assert _opcion_coincide_con_aclaracion("Con miel", "té con miel", duda) is True


def test_coerce_meal_scan_resolved_with_specific_tea_and_honey():
    """Si el usuario especifica 'té de jengibre con miel', ambas dudas quedan resueltas y se ajustan macros."""
    raw = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Té herbal",
        "description": "Taza de infusión.",
        "calories": 2,
        "protein": 0,
        "carbs": 0,
        "healthy_fats": 0,
        "items": [{"name": "té", "calories": 2}],
        "dudas": [],
    }
    result = _coerce_meal_scan(raw, aclaracion="té de jengibre con miel")

    # Ambas dudas resueltas
    assert len(result["dudas"]) == 0
    assert "jengibre" in result["meal_name"].lower()
    # Calorías suben con el ajuste de la miel (+60 kcal)
    assert result["calories"] >= 60
    assert result["carbs"] >= 15
    assert _platos_claros(result) is True


def test_cap_max_dos_dudas():
    """Garantiza que nunca se superen 2 dudas en total aunque existan dudas previas."""
    dudas_previas = [
        {"sobre": "porción", "pregunta": "¿Cuánto tomaste?", "opciones": [{"texto": "1 taza", "supuesta": True}]}
    ]
    raw = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Té",
        "description": "Taza de té.",
        "calories": 2,
        "items": [],
        "dudas": dudas_previas,
    }
    result = _coerce_meal_scan(raw, aclaracion=None)
    assert len(result["dudas"]) <= 2
    assert "de qué es el té" in result["dudas"][0]["pregunta"].lower()


def test_prompt_anchors_and_rules():
    """Verifica que los prompts contienen las reglas críticas de bebidas y franjas."""
    from prompts import chat_agent

    # Regla en _DUDAS_INSTRUCCION
    assert "de qué es una bebida (té, infusión, café, jugo, caldo)" in chat_agent._DUDAS_INSTRUCCION
    assert "Nunca anotes una bebida como «té herbal»" in chat_agent._DUDAS_INSTRUCCION

    # Regla de no cerrar franja en stream y normal
    instrucciones = chat_agent.build_tools_instructions_stream("u-test")
    assert "BEBIDA LIGERA NO CIERRA LA FRANJA" in instrucciones
    assert "≤ 50 kcal" in instrucciones


def test_slot_min_kcal_to_count_in_comidas_sin_registrar():
    """Una bebida de 2 kcal no satisface la comida principal (desayuno/almuerzo/cena)."""
    # 1. Solo un té de 2 kcal en desayuno
    rows_con_te = [
        {"meal_type": "desayuno", "meal_name": "Té de jengibre sin azúcar", "calories": 2}
    ]
    faltan = comidas_sin_registrar(rows_con_te)
    # Desayuno sigue figurando como pendiente
    assert "desayuno" in faltan

    # 2. Desayuno con comida completa (450 kcal)
    rows_comida = [
        {"meal_type": "desayuno", "meal_name": "Huevos revueltos con tostadas", "calories": 450}
    ]
    faltan_despues = comidas_sin_registrar(rows_comida)
    assert "desayuno" not in faltan_despues

    # 3. Múltiples registros pequeños que acumulados superan 50 kcal
    rows_acumuladas = [
        {"meal_type": "desayuno", "meal_name": "Café con leche", "calories": 35},
        {"meal_type": "desayuno", "meal_name": "Galleta", "calories": 30},
    ]
    faltan_acum = comidas_sin_registrar(rows_acumuladas)
    assert "desayuno" not in faltan_acum


def test_agent_today_remaining_block_ignores_low_kcal_beverage():
    """`_build_today_remaining_context` no tacha el slot del plan si las calorías del slot son < 50."""
    from agent import _build_today_remaining_context

    plan = {
        "days": [
            {
                "day": 1,
                "day_name": "martes",
                "meals": [
                    {"meal": "Desayuno", "name": "Mangú con huevo", "cals": 450},
                    {"meal": "Almuerzo", "name": "Arroz con pollo", "cals": 650},
                    {"meal": "Cena", "name": "Ensalada con atún", "cals": 400},
                ],
            }
        ]
    }

    # Desayuno con solo 2 kcal de té
    consumed = [
        {"meal_type": "desayuno", "meal_name": "Té herbal", "calories": 2}
    ]

    out = _build_today_remaining_context(
        current_plan=plan,
        consumed_today=consumed,
        target_cal_int=2000,
        total_consumed=2.0,
        local_date_str="2026-11-03",
    )

    # Desayuno NO debe haber sido tachado de las comidas restantes del plan
    assert "Desayuno" in out
