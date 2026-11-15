# -*- coding: utf-8 -*-
"""Tests para [ESCANER-HABITOS-Y-RELLENOS · 2026-11-14]:
1. Alimentos cerrados (sándwich, wrap, empanada, arepa) siempre preguntan por rellenos ocultos.
2. Personalización desde historial/hechos del usuario (ej. queso cottage y mantequilla de maní).
3. Eliminación de preguntas tontas / visualmente obvias (ej. sirope en pancakes secos).
4. Reemplazo fiable de cantidades y rellenos en `aplicar_respuestas_multiples` (ej. 4 huevos sobre "huevos hervidos").
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import habitos_de_la_foto as hf
import respuestas_de_la_foto as rf
import vision_agent as va


def test_extraer_habitos_detecta_rellenos_y_combos_frecuentes():
    meals = [
        {
            "meal_name": "Huevo frito con tostadas de mantequilla de maní y queso derretido",
            "ingredients": ["pan integral", "mantequilla de maní", "queso", "huevo"],
        },
        {
            "meal_name": "Desayuno con 4 huevos hervidos y sándwich de queso cottage y mantequilla de maní",
            "ingredients": ["pan integral", "queso cottage", "mantequilla de maní", "4 huevos"],
        },
    ]
    habitos = hf.extraer_habitos_desde_registros(meals, [])
    ids = [r["id"] for r in habitos["rellenos_pan"]]
    assert "cottage_mani" in ids
    assert "mani_queso" in ids
    assert "4 huevos" in habitos["resumen_prompt"]
    assert "cottage" in habitos["resumen_prompt"].lower()


def test_caso_real_sandwich_cerrado_inyecta_relleno_personalizado_y_elimina_duda_tonta():
    """Caso moises02emsd@gmail.com: 2 sándwiches cerrados + huevos cortados + 2 pancakes secos."""
    habitos = hf.extraer_habitos_desde_registros(
        [
            {"meal_name": "Tostadas con queso cottage y mantequilla de maní", "ingredients": ["pan", "queso cottage", "mantequilla de maní"]},
            {"meal_name": "Tostadas de mantequilla de maní y queso derretido", "ingredients": ["pan", "mantequilla de maní", "queso"]},
        ]
    )
    payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Sándwiches de pan integral con jamón, huevos hervidos y pancakes",
        "description": "Dos sándwiches de pan integral con jamón, huevos hervidos cortados a la mitad y dos pancakes pequeños.",
        "calories": 796,
        "protein": 47,
        "carbs": 78,
        "healthy_fats": 32,
        "items": [
            {"name": "sándwich de pan integral con jamón", "quantity": 2, "unit": "unidad", "calories": 420, "protein": 20, "carbs": 48, "healthy_fats": 16},
            {"name": "huevo hervido", "quantity": 3, "unit": "unidad", "calories": 216, "protein": 19, "carbs": 2, "healthy_fats": 14},
            {"name": "pancake pequeño", "quantity": 2, "unit": "unidad", "calories": 160, "protein": 8, "carbs": 28, "healthy_fats": 2},
        ],
        "dudas": [
            {
                "sobre": "huevo hervido",
                "pregunta": "¿Cuántos huevos hervidos eran?",
                "opciones": [
                    {"texto": "3 huevos enteros", "supuesta": True, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                    {"texto": "4 huevos enteros", "supuesta": False, "ajuste": {"calories": 72, "protein": 6, "carbs": 0, "healthy_fats": 5}},
                ],
            },
            {
                "sobre": "pancake pequeño",
                "pregunta": "¿Llevaban sirope los pancakes?",
                "opciones": [
                    {"texto": "Sin sirope", "supuesta": True, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                    {"texto": "Con sirope", "supuesta": False, "ajuste": {"calories": 100, "protein": 0, "carbs": 25, "healthy_fats": 0}},
                ],
            },
        ],
    }

    r = va._coerce_meal_scan(payload, habitos_usuario=habitos)
    preguntas = [d["pregunta"] for d in r["dudas"]]

    # 1. Eliminó la pregunta tonta de "¿Llevaban sirope?" (cuya supuesta era "Sin sirope")
    assert not any("sirope" in p.lower() for p in preguntas), preguntas
    # 2. En posición #1 puso la pregunta de qué llevan los sándwiches dentro
    assert any("sándwich" in p.lower() for p in preguntas[:1]), preguntas
    # 3. Incluyó el hábito real del usuario con ajuste multiplicado por 2 sándwiches (+310 kcal, +17g P)
    ops_sandwich = {o["texto"]: o for o in r["dudas"][0]["opciones"]}
    assert "Cottage y mantequilla de maní" in ops_sandwich
    assert ops_sandwich["Cottage y mantequilla de maní"]["ajuste"]["calories"] == 310
    assert ops_sandwich["Cottage y mantequilla de maní"]["ajuste"]["protein"] == 17.0
    # 4. Conservó también la duda real de cuántos huevos eran
    assert any("huevo" in p.lower() for p in preguntas), preguntas


def test_plato_compuesto_permite_hasta_3_dudas_reales():
    payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Sándwich con huevos y pancakes",
        "description": "Un sándwich cerrado, huevos revueltos y pancakes.",
        "calories": 650,
        "protein": 30,
        "carbs": 65,
        "healthy_fats": 28,
        "items": [
            {"name": "sándwich", "quantity": 1, "unit": "unidad", "calories": 300, "protein": 12, "carbs": 35, "healthy_fats": 12},
            {"name": "huevo", "quantity": 2, "unit": "unidad", "calories": 160, "protein": 12, "carbs": 2, "healthy_fats": 11},
            {"name": "pancake", "quantity": 2, "unit": "unidad", "calories": 190, "protein": 6, "carbs": 28, "healthy_fats": 5},
        ],
        "dudas": [
            {"sobre": "sándwich", "pregunta": "¿Qué lleva el sándwich por dentro?", "opciones": [
                {"texto": "Jamón y queso", "supuesta": True, "ajuste": {}},
                {"texto": "Pavo y queso", "ajuste": {"calories": -10}},
            ]},
            {"sobre": "huevo", "pregunta": "¿Cuántos huevos eran?", "opciones": [
                {"texto": "2 huevos", "supuesta": True, "ajuste": {}},
                {"texto": "3 huevos", "ajuste": {"calories": 72}},
            ]},
            {"sobre": "pancake", "pregunta": "¿De qué es la masa de los pancakes?", "opciones": [
                {"texto": "Harina normal", "supuesta": True, "ajuste": {}},
                {"texto": "Avena y guineo", "ajuste": {"calories": -30}},
            ]},
        ],
    }
    r = va._coerce_meal_scan(payload)
    assert len(r["dudas"]) == 3


def test_aplicar_respuestas_multiples_actualiza_huevos_y_relleno_sin_match_literal():
    desc = (
        "2 sándwiches de pan integral con jamón, huevos hervidos cortados a la mitad y 2 pancakes. "
        "DUDAS (pregúntale solo esto): ¿Qué llevan los sándwiches además de jamón (o por dentro)? "
        "(opciones: Solo jamón (supuesto) · Cottage y mantequilla de maní) "
        "¿Cuántos huevos hervidos eran? (opciones: 3 huevos enteros (supuesto) · 4 huevos enteros) "
        "(Estimación: Calorías: 796, Proteína: 47g, Carbohidratos: 78g, Grasas Saludables: 32g)"
    )
    desc_out, sup_str, reemps = rf.aplicar_respuestas_multiples(
        desc,
        "Cottage y mantequilla de maní · 4 huevos enteros",
    )
    pre = desc_out[:desc_out.find("DUDAS")]
    assert "cottage y mantequilla de maní" in pre.lower()
    assert "4 huevos enteros" in pre.lower()
    assert len(reemps) == 2
    assert sup_str is not None


def test_prompt_y_scope_de_habitos():
    va._MEAL_VISION_PROMPT.encode("ascii")
    assert "ALIMENTOS CERRADOS O CON RELLENO OCULTO" in va._MEAL_VISION_PROMPT
    assert "PROHIBIDO HACER PREGUNTAS TONTAS" in va._MEAL_VISION_PROMPT
    prompt_h = va._prompt_de_escaneo(
        None,
        base="BASE",
        habitos_usuario={"resumen_prompt": "En pan/sandwich suele usar: cottage y mantequilla de mani."},
    )
    assert "HABITOS DEL USUARIO" in prompt_h
    assert "cottage y mantequilla de mani" in prompt_h
