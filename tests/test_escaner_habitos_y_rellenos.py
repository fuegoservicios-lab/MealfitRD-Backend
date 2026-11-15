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


def test_plato_complejo_resuelve_hasta_6_dudas_necesarias():
    payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Desayuno variado",
        "description": "2 sandwiches de jamon, huevos revueltos, pancakes, cafe con leche y guineo.",
        "calories": 890,
        "protein": 44,
        "carbs": 95,
        "healthy_fats": 35,
        "items": [
            {"name": "sandwich de jamon", "quantity": 2, "unit": "unidad", "calories": 380, "protein": 18, "carbs": 42, "healthy_fats": 14},
            {"name": "huevo revuelto", "quantity": 3, "unit": "unidad", "calories": 216, "protein": 19, "carbs": 2, "healthy_fats": 14},
            {"name": "pancake", "quantity": 2, "unit": "unidad", "calories": 160, "protein": 5, "carbs": 28, "healthy_fats": 3},
            {"name": "cafe con leche", "quantity": 1, "unit": "taza", "calories": 60, "protein": 2, "carbs": 5, "healthy_fats": 3},
        ],
        "dudas": [
            {"sobre": "huevo revuelto", "pregunta": "¿Cuántos huevos revueltos eran?", "opciones": [
                {"texto": "3 huevos", "supuesta": True, "ajuste": {}},
                {"texto": "4 huevos", "ajuste": {"calories": 72}},
            ]},
            {"sobre": "huevo revuelto", "pregunta": "¿Con qué grasa hiciste los huevos?", "opciones": [
                {"texto": "Poco aceite", "supuesta": True, "ajuste": {}},
                {"texto": "Con mantequilla", "ajuste": {"calories": 60}},
            ]},
            {"sobre": "pancake", "pregunta": "¿De qué es la masa de los pancakes?", "opciones": [
                {"texto": "Harina normal", "supuesta": True, "ajuste": {}},
                {"texto": "Avena y guineo", "ajuste": {"calories": -25}},
            ]},
            {"sobre": "cafe con leche", "pregunta": "¿Qué leche y azúcar lleva el café?", "opciones": [
                {"texto": "Leche entera sin azúcar", "supuesta": True, "ajuste": {}},
                {"texto": "Con azúcar", "ajuste": {"calories": 32}},
            ]},
        ],
    }
    r = va._coerce_meal_scan(payload)
    # 4 del modelo + 1 obligatoria inyectada para el sándwich cerrado = 5 dudas reales
    assert len(r["dudas"]) == 5
    assert any("sándwich" in d["pregunta"].lower() for d in r["dudas"])


def test_cuatro_fotos_por_foto_aplica_respuestas_y_ajustes_independientes():
    vision = {
        "kind": "multi",
        "items": [
            {
                "kind": "plato",
                "description": (
                    "2 sándwiches de jamón. DUDAS (pregúntale solo esto): ¿Qué llevan? "
                    "(opciones: Solo jamón (supuesto) · Cottage y mantequilla de maní) "
                    "(Estimación: Calorías: 400, Proteína: 20g, Carbohidratos: 44g, Grasas Saludables: 14g)"
                ),
            },
            {
                "kind": "plato",
                "description": (
                    "Huevos hervidos (3 huevos enteros). DUDAS (pregúntale solo esto): ¿Cuántos huevos? "
                    "(opciones: 3 huevos enteros (supuesto) · 4 huevos enteros) "
                    "(Estimación: Calorías: 216, Proteína: 19g, Carbohidratos: 2g, Grasas Saludables: 14g)"
                ),
            },
            {
                "kind": "plato",
                "description": (
                    "2 pancakes de harina normal. DUDAS (pregúntale solo esto): ¿De qué masa? "
                    "(opciones: Harina normal (supuesto) · Avena) "
                    "(Estimación: Calorías: 180, Proteína: 6g, Carbohidratos: 30g, Grasas Saludables: 4g)"
                ),
            },
            {
                "kind": "plato",
                "description": (
                    "Café negro con leche entera. DUDAS (pregúntale solo esto): ¿Qué leche? "
                    "(opciones: Leche entera (supuesto) · Leche de almendras) "
                    "(Estimación: Calorías: 80, Proteína: 4g, Carbohidratos: 6g, Grasas Saludables: 4g)"
                ),
            },
        ],
        "respuestas": "Cottage y mantequilla de maní · 4 huevos enteros · Harina normal · Leche de almendras",
        "por_foto": [
            {"index": 0, "respuestas": "Cottage y mantequilla de maní", "ajuste": {"calories": 310, "protein": 17, "carbs": 8, "healthy_fats": 23}},
            {"index": 1, "respuestas": "4 huevos enteros", "ajuste": {"calories": 72, "protein": 6, "carbs": 0, "healthy_fats": 5}},
            {"index": 2, "respuestas": "Harina normal", "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
            {"index": 3, "respuestas": "Leche de almendras", "ajuste": {"calories": -40, "protein": -3, "carbs": -4, "healthy_fats": -2}},
        ],
    }
    prep = rf.preparar_vision(vision)
    assert prep.get("_ajuste_aplicado") is True
    assert "Calorías: 710" in prep["items"][0]["description"]
    assert "cottage y mantequilla de maní" in prep["items"][0]["description"].lower()
    assert "Calorías: 288" in prep["items"][1]["description"]
    assert "4 huevos enteros" in prep["items"][1]["description"].lower()
    assert "Calorías: 180" in prep["items"][2]["description"]
    assert "Calorías: 40" in prep["items"][3]["description"]
    assert "leche de almendras" in prep["items"][3]["description"].lower()
    assert all("pregúntale solo esto" not in it["description"] for it in prep["items"])


def test_bowl_frutas_siempre_pregunta_miel_e_inyecta_habitos_y_ensalada_pregunta_aderezo():
    """Caso real: bowl de lechosa, piña y sandía debe preguntar por miel/extras aunque esté húmeda;
    plato con ensalada/vegetales frescos debe preguntar por aceite/aderezo."""
    habitos = hf.extraer_habitos_desde_registros(
        [
            {"meal_name": "Frutas con leche condensada y granola", "ingredients": ["lechosa", "leche condensada", "granola"]},
            {"meal_name": "Ensalada verde con aceite de oliva", "ingredients": ["lechuga", "tomate", "aceite de oliva"]},
        ]
    )
    assert any(x["id"] == "leche_condensada" for x in habitos["extras_fruta"])
    assert any(x["id"] == "aceite_oliva" for x in habitos["aderezos_ensalada"])

    bowl_fruta = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Bowl de frutas picadas",
        "description": "Tazón con lechosa, piña y sandía picadas en trozos.",
        "calories": 140,
        "protein": 2,
        "carbs": 34,
        "healthy_fats": 0,
        "items": [
            {"name": "lechosa", "quantity": 1, "unit": "taza", "calories": 55, "protein": 1, "carbs": 14, "healthy_fats": 0},
            {"name": "piña", "quantity": 0.5, "unit": "taza", "calories": 40, "protein": 0.5, "carbs": 10, "healthy_fats": 0},
            {"name": "sandía", "quantity": 1, "unit": "taza", "calories": 45, "protein": 0.5, "carbs": 10, "healthy_fats": 0},
        ],
        "dudas": [],
    }
    r_fruta = va._coerce_meal_scan(bowl_fruta, habitos_usuario=habitos)
    assert len(r_fruta["dudas"]) >= 1
    d_miel = r_fruta["dudas"][0]
    assert "miel" in d_miel["pregunta"].lower()
    textos_f = [o["texto"] for o in d_miel["opciones"]]
    assert "Solo fruta" in textos_f
    assert "Con miel" in textos_f
    assert "Con leche condensada" in textos_f

    almuerzo_con_ensalada = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Arroz blanco con cerdo y ensalada",
        "description": "Plato con arroz blanco, carne de cerdo guisada, tomate y remolacha.",
        "calories": 650,
        "protein": 34,
        "carbs": 75,
        "healthy_fats": 22,
        "items": [
            {"name": "arroz blanco", "quantity": 1.5, "unit": "taza", "calories": 300, "protein": 6, "carbs": 66, "healthy_fats": 0},
            {"name": "carne de cerdo", "quantity": 120, "unit": "g", "calories": 300, "protein": 26, "carbs": 0, "healthy_fats": 22},
            {"name": "ensalada de tomate y remolacha", "quantity": 1, "unit": "porción", "calories": 50, "protein": 2, "carbs": 9, "healthy_fats": 0},
        ],
        "dudas": [
            {"sobre": "carne de cerdo", "pregunta": "¿Qué tipo de carne es?", "opciones": [
                {"texto": "Cerdo", "supuesta": True, "ajuste": {}},
                {"texto": "Res", "ajuste": {"calories": -15}},
            ]},
            {"sobre": "arroz blanco", "pregunta": "¿Cuánto arroz hay en el plato?", "opciones": [
                {"texto": "1.5 tazas", "supuesta": True, "ajuste": {}},
                {"texto": "1 taza", "ajuste": {"calories": -100}},
            ]},
        ],
    }
    r_alm = va._coerce_meal_scan(almuerzo_con_ensalada, habitos_usuario=habitos)
    assert len(r_alm["dudas"]) == 3
    assert any("aderezo" in d["pregunta"].lower() or "aceite" in d["pregunta"].lower() for d in r_alm["dudas"])

    import multiphoto_completeness as mc
    rec = mc.construir_recordatorio_multifoto(["Bowl de frutas", "Arroz con cerdo", "4 sándwiches"])
    assert "franjas distintas" in rec
    assert "cantidad exacta" in rec


def test_caso_real_pan_integral_suelto_pregunta_por_untables_e_inyecta_cottage_y_mani():
    """Regresión 15-nov (moises02emsd@gmail.com 10:54 AM): 2 rebanadas de pan integral con huevos,
    jamón y aguacate devolvía dudas=[] aunque el usuario usa cottage y mantequilla de maní en pan."""
    habitos = hf.extraer_habitos_desde_registros([
        {
            "meal_name": "2 rebanadas de pan integral con queso cottage y mantequilla de maní, 3 huevos hervidos",
            "ingredients": ["pan integral (2 rebanadas)", "queso cottage", "mantequilla de maní", "huevos"],
        },
        {
            "meal_name": "Pan integral tostado con mantequilla de maní y queso cottage, jamón cocido y atún en aceite",
            "ingredients": ["4 rebanadas de pan integral tostado", "mantequilla de maní", "queso cottage", "atún"],
        },
    ])
    assert any(h["id"] == "cottage_mani" for h in habitos["rellenos_pan"])

    payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Pan integral con huevos hervidos, jamón y aguacate",
        "description": "Dos rebanadas de pan integral, tres huevos hervidos cortados en seis mitades, dos lascas grandes de jamon y un cuarto de aguacate.",
        "calories": 525,
        "protein": 39,
        "carbs": 34,
        "healthy_fats": 27,
        "items": [
            {"name": "pan integral", "quantity": 2, "unit": "rebanada", "calories": 140, "protein": 6, "carbs": 26, "healthy_fats": 2},
            {"name": "huevo hervido", "quantity": 3, "unit": "unidad", "calories": 215, "protein": 19, "carbs": 2, "healthy_fats": 14},
            {"name": "jamón", "quantity": 2, "unit": "lasca", "calories": 85, "protein": 13, "carbs": 2, "healthy_fats": 3},
            {"name": "aguacate", "quantity": 0.25, "unit": "unidad", "calories": 85, "protein": 1, "carbs": 4, "healthy_fats": 8},
        ],
        "dudas": [],
    }

    r = va._coerce_meal_scan(payload, habitos_usuario=habitos)
    assert len(r["dudas"]) >= 1
    assert "untado" in r["dudas"][0]["pregunta"].lower() or "pan" in r["dudas"][0]["pregunta"].lower()
    ops = {o["texto"]: o for o in r["dudas"][0]["opciones"]}
    assert ops["Sin untable / pan solo"]["supuesta"] is True
    assert ops["Sin untable / pan solo"]["ajuste"]["calories"] == 0
    assert "Cottage y mantequilla de maní" in ops
    # Jamón y aguacate del plato NO deben restarse del ajuste del untable del pan
    assert ops["Cottage y mantequilla de maní"]["ajuste"]["calories"] == 155
    assert ops["Cottage y mantequilla de maní"]["ajuste"]["protein"] == 8.5
    assert "DUDAS (pregúntale solo esto):" in r["description"]

    # Si el usuario ya lo aclaró en el texto de la foto, no vuelve a preguntar lo mismo
    ya_dicho = va._coerce_meal_scan(payload, aclaracion="el pan tiene queso cottage y mantequilla de mani", habitos_usuario=habitos)
    assert not any("untado" in d["pregunta"].lower() for d in ya_dicho["dudas"])

    # Un pan acompañante junto a un plato de arroz/pollo NO dispara duda falsa de untables
    arroz_con_pan = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Arroz blanco con pollo guisado y pan",
        "description": "Plato de arroz blanco, pollo guisado y una rebanada de pan.",
        "calories": 680,
        "protein": 36,
        "carbs": 80,
        "healthy_fats": 22,
        "items": [
            {"name": "arroz blanco", "quantity": 1, "unit": "taza", "calories": 240, "protein": 5, "carbs": 52, "healthy_fats": 0},
            {"name": "pollo guisado", "quantity": 150, "unit": "g", "calories": 320, "protein": 28, "carbs": 4, "healthy_fats": 20},
            {"name": "pan", "quantity": 1, "unit": "rebanada", "calories": 120, "protein": 3, "carbs": 24, "healthy_fats": 2},
        ],
        "dudas": [],
    }
    r_arroz = va._coerce_meal_scan(arroz_con_pan, habitos_usuario=habitos)
    assert r_arroz["dudas"] == []
