# -*- coding: utf-8 -*-
"""Tests para [ESCANER-HABITOS-Y-RELLENOS · 2026-11-14]:
1. Alimentos cerrados (sándwich, wrap, empanada, arepa) siempre preguntan por rellenos ocultos.
2. Personalización desde historial/hechos del usuario (ej. queso cottage y mantequilla de maní).
3. Eliminación de preguntas tontas / visualmente obvias (ej. sirope en pancakes secos).
4. Reemplazo fiable de cantidades y rellenos en `aplicar_respuestas_multiples` (ej. 4 huevos sobre "huevos hervidos").
"""
from __future__ import annotations

import re
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


def test_sopa_con_repollo_no_dispara_duda_ensalada_y_reemplazo_proteina_escrita():
    """Regresión 15-nov (sarahacevedo13@icloud.com 4:20 PM & 10:08 AM):
    1) Una sopa con repollo y zanahoria no debe disparar duda de aderezo de ensalada.
    2) Si la duda asumió cantidad de pollo ('Normal (como 3 oz)') y el usuario escribe
       'Pescado uno pequeño', se sustituye 'pollo' en la descripción (no se suman ambos).
    3) En el desayuno con 'Mantequilla · Manteca de cerdo', la mantequilla supuesta confirmada
       sigue visible en la descripción del plato."""
    sopa_payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Arroz blanco con sopa de pollo y verduras",
        "description": "Bowl de arroz blanco cocido acompañado de un bowl de sopa de pollo con caldo, trozos de pollo, un trozo de maíz, papa, tayota, zanahoria y repollo.",
        "calories": 588,
        "protein": 37,
        "carbs": 84,
        "healthy_fats": 10,
        "items": [
            {"name": "arroz blanco cocido", "quantity": 1, "unit": "taza", "calories": 200, "protein": 4, "carbs": 44, "healthy_fats": 0},
            {"name": "sopa de pollo con maíz, papa, tayota, zanahoria y repollo", "quantity": 1, "unit": "bowl", "calories": 388, "protein": 33, "carbs": 40, "healthy_fats": 10},
        ],
        "dudas": [],
    }
    r_sopa = va._coerce_meal_scan(sopa_payload)
    assert not any("ensalada" in d["pregunta"].lower() or "aderezo" in d["pregunta"].lower() for d in r_sopa["dudas"])

    # Si de verdad hay una ensalada explícita junto a un caldo/guiso, sí debe detectarla
    sopa_y_ensalada = dict(sopa_payload, items=list(sopa_payload["items"]) + [{"name": "ensalada verde de repollo y tomate", "quantity": 1, "unit": "porción"}])
    assert any("ensalada" in d["pregunta"].lower() for d in va._coerce_meal_scan(sopa_y_ensalada)["dudas"])

    # Sustitución de proteína escrita cuando supuesta era cantidad ("Normal (como 3 oz)")
    desc_sopa = (
        "Bowl de arroz blanco cocido acompañado de un bowl de sopa de pollo con caldo, trozos de pollo, "
        "un trozo de maíz, papa, tayota, zanahoria y repollo. "
        "DUDAS (pregúntale solo esto): ¿Qué tanta carne de pollo tiene la sopa? "
        "(opciones: Normal (como 3 oz) (supuesto) · Bastante (como 5 oz) · Muy poca o casi nada) "
        "¿Cuánto arroz hay en el bowl? (opciones: 1 taza (supuesto) · 1.5 tazas · 0.5 taza) "
        "(Estimación: Calorías: 588, Proteína: 37g, Carbohidratos: 84g, Grasas Saludables: 10g)"
    )
    desc_out, sup_out, reemps_out = rf.aplicar_respuestas_multiples(desc_sopa, "Pescado uno pequeño · 1 taza")
    pre_out = desc_out[:desc_out.find("DUDAS")]
    assert "pescado uno pequeño" in pre_out.lower()
    assert not re.search(r"\bpollo\b", pre_out.lower())
    assert "repollo" in pre_out.lower()
    assert sup_out == "pollo"
    assert ("pollo", "Pescado uno pequeño") in reemps_out

    v_sopa = rf.preparar_vision({
        "kind": "multi",
        "items": [{"kind": "plato", "description": desc_sopa}],
        "respuestas": "Pescado uno pequeño · 1 taza",
        "ajuste": None,
    })
    instr = rf.instruccion(v_sopa)
    assert "REEMPLAZO, NO SUMA" in instr
    assert "NO copies esas calorías y macros idénticas" in instr
    assert "«pollo»" in instr

    # Desayuno 10:08 AM: Mantequilla (supuesta) + Manteca de cerdo -> conserva ambas grasas
    desc_mangu = (
        "Mangú de plátano (aprox. 1 taza), 2 huevos fritos y 1 lasca de aguacate. "
        "DUDAS (pregúntale solo esto): ¿Con qué grasa se hizo el mangú? "
        "(opciones: Mantequilla (supuesto) · Aceite · Manteca de cerdo · Sin grasa) "
        "¿Con qué grasa se frieron los huevos? "
        "(opciones: Aceite vegetal (supuesto) · Mantequilla · Manteca de cerdo · Sin grasa / a la plancha) "
        "(Estimación: Calorías: 590, Proteína: 16g, Carbohidratos: 67g, Grasas Saludables: 29g)"
    )
    desc_m_out, _, _ = rf.aplicar_respuestas_multiples(desc_mangu, "Mantequilla · Manteca de cerdo")
    pre_m = desc_m_out[:desc_m_out.find("DUDAS")].lower()
    assert "mantequilla" in pre_m
    assert "manteca de cerdo" in pre_m


def test_todas_las_fotos_unavailable_prohibe_inventar_y_bloquea_registro_no_identificable():
    """Regresión 15-nov (6:45 PM, HTTP 429 en Vision): si las 2 fotos caen en unavailable,
    build_vision_context prohíbe llamar a log_consumed_meal y la tool bloquea nombres
    de contenido no identificable."""
    from prompts.chat_agent import build_vision_context
    from tools import log_consumed_meal

    ctx = build_vision_context({
        "kind": "multi",
        "has_text": True,
        "items": [
            {"kind": "unavailable", "reason": "down"},
            {"kind": "unavailable", "reason": "down"},
        ],
    })
    assert "PROHIBIDO" in ctx
    assert "log_consumed_meal" in ctx

    out = log_consumed_meal.invoke({
        "user_id": "test-user-guard",
        "meal_name": "Almuerzo de contenido no identificable",
        "calories": 600,
        "protein": 25,
        "meal_type": "almuerzo",
    })
    assert "NO REGISTRADO" in out


def test_habitos_grasa_coccion_y_duda_enriquecida_en_mangu_con_fritos():
    """Regresión 15-nov (10:50 PM): si el usuario suele cocinar con manteca de cerdo y Vision
    solo pregunta grasa del mangú con 3 chips sin Manteca de cerdo ni cubre queso/huevos fritos,
    el refinador inyecta Manteca de cerdo en el mangú y añade duda de fritura sin perder conteos."""
    habitos = hf.extraer_habitos_desde_registros([
        {
            "meal_name": "Tostadas con mantequilla de maní",
            "ingredients": ["pan integral", "mantequilla de maní"],
        },
        {
            "meal_name": "Mangú con 2 huevos y queso blanco fritos en manteca de cerdo",
            "ingredients": ["mangú", "huevos fritos", "queso blanco frito", "manteca de cerdo"],
        },
    ])
    ids_grasa = [g["id"] for g in habitos["grasas_coccion"]]
    assert ids_grasa == ["manteca_cerdo"], habitos  # mantequilla de maní NO cuenta como mantequilla de cocción
    assert "manteca de cerdo" in habitos["resumen_prompt"].lower()

    raw_payload = {
        "photo_kind": "plato",
        "is_food": True,
        "meal_name": "Mangú con huevo y queso frito",
        "description": "Desayuno criollo servido con una porción de mangú de plátano verde, 2 huevos fritos y 2 lascas de queso frito.",
        "calories": 720,
        "protein": 28,
        "carbs": 64,
        "healthy_fats": 38,
        "items": [
            {"name": "mangú de plátano verde", "quantity": 1, "unit": "taza", "calories": 300, "protein": 4, "carbs": 58, "healthy_fats": 6},
            {"name": "huevo frito", "quantity": 2, "unit": "unidad", "calories": 220, "protein": 12, "carbs": 2, "healthy_fats": 18},
            {"name": "queso frito", "quantity": 2, "unit": "lasca", "calories": 200, "protein": 12, "carbs": 4, "healthy_fats": 14},
        ],
        "dudas": [
            {
                "sobre": "huevo frito",
                "pregunta": "¿Cuántos huevos fritos hay servidos?",
                "opciones": [
                    {"texto": "2 huevos", "supuesta": True, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                    {"texto": "3 huevos", "supuesta": False, "ajuste": {"calories": 90, "protein": 6, "carbs": 1, "healthy_fats": 7}},
                ],
            },
            {
                "sobre": "mangú de plátano verde",
                "pregunta": "¿Con qué grasa preparaste el mangú?",
                "opciones": [
                    {"texto": "Mantequilla", "supuesta": True, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                    {"texto": "Aceite", "supuesta": False, "ajuste": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}},
                    {"texto": "Sin grasa", "supuesta": False, "ajuste": {"calories": -75, "protein": 0, "carbs": 0, "healthy_fats": -8.5}},
                ],
            },
        ],
    }

    r = va._coerce_meal_scan(raw_payload, habitos_usuario=habitos)
    preguntas = [d["pregunta"] for d in r["dudas"]]
    # 1. Conteo de huevos se mantiene primero que las grasas
    assert "huevos" in preguntas[0].lower()
    # 2. La duda del mangú ahora incluye Manteca de cerdo sin perder Mantequilla (supuesta) ni Sin grasa
    d_mangu = next(d for d in r["dudas"] if "mang" in d["pregunta"].lower())
    ops_mangu = [o["texto"] for o in d_mangu["opciones"]]
    assert "Manteca de cerdo" in ops_mangu, ops_mangu
    assert any(o.get("supuesta") and o["texto"] == "Mantequilla" for o in d_mangu["opciones"])
    assert "Sin grasa" in ops_mangu
    # 3. Se inyecta duda de fritura cubriendo huevos y queso frito, con Manteca de cerdo y Sin grasa
    d_frito = next(d for d in r["dudas"] if "frieron" in d["pregunta"].lower())
    assert "huevos" in d_frito["pregunta"].lower() and "queso" in d_frito["pregunta"].lower()
    ops_frito = [o["texto"] for o in d_frito["opciones"]]
    assert "Manteca de cerdo" in ops_frito
    assert "Aceite vegetal" in ops_frito


def test_meal_name_backed_acepta_ensalada_o_bowl_de_frutas_cuando_hay_frutas():
    from constants import meal_name_backed_by_description

    assert meal_name_backed_by_description(
        "Ensalada de frutas con lechosa, piña y uvas",
        "Bowl de frutas picadas que contiene trozos de lechosa (papaya), trozos de piña fresca y uvas rojas.",
    ) is True
    assert meal_name_backed_by_description(
        "Fruta picada: papaya, piña y uvas",
        "Papaya picada (base); trozos de piña; uvas rojas enteras.",
    ) is True
    assert meal_name_backed_by_description(
        "Ensalada de frutas con papaya y mango",
        "Arroz blanco y carne molida guisada.",
    ) is False
