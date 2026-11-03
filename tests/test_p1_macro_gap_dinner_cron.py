"""[P1-MACRO-GAP-DINNER-CRON · 2026-11-03] Test suite para el evaluador de brecha de macros en el cron de las 7:00 PM.
Verifica que las sugerencias de cena se adapten rigurosamente al presupuesto de calorías y proteínas restantes.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from macro_gap_evaluator import evaluar_brecha_macros_cena
import aviso_del_dia as ad


def test_evaluacion_95_porciento_activa_cierre_hidratacion():
    """Si consumió >= 95% de sus calorías, debe activar modo cierre_hidratacion y prohibir comida sólida."""
    health = {"target_calories": 2000, "target_protein": 140}
    consumed = [
        {"meal_name": "Desayuno", "calories": 800, "protein": 50},
        {"meal_name": "Almuerzo", "calories": 1120, "protein": 60},  # Total: 1920 / 2000 = 96%
    ]
    res = evaluar_brecha_macros_cena(consumed, health)
    assert res.mode == "cierre_hidratacion"
    assert res.pct_kcal_consumed >= 0.95
    assert "PROHIBIDO proponer platos calóricos" in res.instruccion_prompt
    assert "cierre" in res.instruccion_prompt.lower()
    assert "hidratación" in res.instruccion_prompt.lower() or "infusión" in res.instruccion_prompt.lower()


def test_evaluacion_menos_de_100_kcal_restantes_activa_cierre():
    """Si le quedan <= 100 kcal aunque el porcentaje sea cercano, activa cierre_hidratacion."""
    health = {"target_calories": 1500, "target_protein": 100}
    consumed = [
        {"meal_name": "Comida 1", "calories": 1420, "protein": 80},  # Quedan 80 kcal
    ]
    res = evaluar_brecha_macros_cena(consumed, health)
    assert res.mode == "cierre_hidratacion"
    assert res.remaining_kcal <= 100


def test_evaluacion_pocas_kcal_mucha_proteina_fuerza_magra():
    """Si le quedan < 350 kcal pero le faltan >= 25 g de proteína, fuerza modo proteina_magra."""
    health = {"target_calories": 1800, "target_protein": 140}
    consumed = [
        {"meal_name": "Desayuno", "calories": 600, "protein": 30},
        {"meal_name": "Almuerzo", "calories": 1000, "protein": 50},  # Total: 1600 kcal (quedan 200), 80g P (faltan 60g)
    ]
    res = evaluar_brecha_macros_cena(consumed, health)
    assert res.mode == "proteina_magra"
    assert res.remaining_kcal == 200
    assert res.remaining_protein == 60
    assert "REGLA OBLIGATORIA DE PROTEÍNA MAGRA PURA" in res.instruccion_prompt
    assert "PROHIBIDO sugerir carbohidratos densos" in res.instruccion_prompt
    assert len(res.alimentos_magros_filtrados) > 0


def test_filtra_alimentos_ya_consumidos_en_fuentes_magras():
    """No debe sugerir pollo si el usuario ya comió pechuga de pollo hoy."""
    health = {"target_calories": 2000, "target_protein": 150}
    consumed = [
        {"meal_name": "Pechuga de pollo con ensalada", "calories": 1700, "protein": 100, "ingredients": ["pechuga", "lechuga"]},
    ]
    res = evaluar_brecha_macros_cena(consumed, health)
    assert res.mode == "proteina_magra"
    # Debe haber excluido pechuga de pollo
    nombres_sugeridos = [f.lower() for f in res.alimentos_magros_filtrados]
    assert not any("pechuga" in s or "pollo" in s for s in nombres_sugeridos)
    # Debe ofrecer alternativas puras no consumidas
    assert any("atún" in s or "claras" in s or "proteína" in s for s in nombres_sugeridos)


def test_deficit_amplio_activa_cena_completa():
    """Si quedan >= 500 kcal y >= 35 g de proteína, activa cena_completa."""
    health = {"target_calories": 2400, "target_protein": 160}
    consumed = [
        {"meal_name": "Desayuno", "calories": 500, "protein": 35},
        {"meal_name": "Almuerzo", "calories": 800, "protein": 50},  # Quedan 1100 kcal y 75g P
    ]
    res = evaluar_brecha_macros_cena(consumed, health)
    assert res.mode == "cena_completa"
    assert "CENA COMPLETA DE ALTA PROTEÍNA" in res.instruccion_prompt


def test_bloque_del_dia_inyecta_regla_de_presupuesto():
    """Verifica que bloque_del_dia incluye la directiva presupuestaria cuando el consumo es alto."""
    perfil = {
        "gender": "male", "age": 30, "height": 175, "weight": 80, "weightUnit": "kg",
        "activityLevel": "moderate", "mainGoal": "gain_muscle", "medicalConditions": ["none"],
    }
    # Consumidas casi todas las calorías de la meta (3020 de 3100 kcal = 97.4%)
    comidas = [
        {"meal_type": "almuerzo", "meal_name": "Comida pesada", "calories": 3020, "protein": 150},
    ]
    bloque = ad.bloque_del_dia(comidas, perfil)
    assert "REGLA DE PRESUPUESTO" in bloque
    assert "cierre de jornada" in bloque.lower() or "hidratación" in bloque.lower()


def test_usuario_sin_metas_degrada_a_desconocido():
    """Si no hay perfil ni metas, degrada a desconocido sin lanzar excepciones."""
    res = evaluar_brecha_macros_cena([{"meal_name": "Arroz", "calories": 300}], None)
    assert res.mode == "desconocido"
    assert res.instruccion_prompt == ""
