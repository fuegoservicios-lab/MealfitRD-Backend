# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-322 · 2026-09-25] Las dudas de la foto se responden con UN TOQUE.

El dueño, con la captura del escáner («¿De cuántos huevos hiciste la tortilla?» / «¿La base es panecillo, arepa o
galleta?»): «quiero una forma más fácil para responder esas preguntas». Cada duda trae sus OPCIONES (2-5) y, en la
misma llamada de visión, lo que cambia cada una en las macros del plato respecto a lo estimado (la opción supuesta
va en 0). Así tocar «4 huevos» mueve las calorías al instante, sin otra llamada a la IA. El coach las recibe en la
descripción y el chat las pinta como botones de respuesta rápida.
Tooltip-anchor: P1-PLAN-LOTE-322
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _plato(**extra):
    base = {"photo_kind": "plato", "is_food": True, "meal_name": "Tortilla", "description": "Tortilla de huevo.",
            "calories": 510, "protein": 26, "carbs": 30, "healthy_fats": 30,
            "items": [{"name": "huevo", "quantity": 3, "unit": "unidad"}]}
    base.update(extra)
    return base


def _op(texto, kcal=0, supuesta=False, **extra):
    o = {"texto": texto, "supuesta": supuesta,
         "ajuste": {"calories": kcal, "protein": kcal / 12, "carbs": 0, "healthy_fats": kcal / 15}}
    o.update(extra)
    return o


def test_las_opciones_llegan_normalizadas_con_la_supuesta_en_cero():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"sobre": "huevo", "pregunta": "¿De cuántos huevos?", "opciones": [
        _op("2 huevos", -72), _op("3 huevos", 0, True), _op("4 huevos", 72), _op("5 huevos", 144)]}]))
    ops = r["dudas"][0]["opciones"]
    assert [o["texto"] for o in ops] == ["2 huevos", "3 huevos", "4 huevos", "5 huevos"]
    assert [o["supuesta"] for o in ops] == [False, True, False, False]
    assert ops[1]["ajuste"] == {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}
    assert ops[2]["ajuste"]["calories"] == 72


def test_si_la_supuesta_no_viene_en_cero_todas_se_recalculan_contra_ella():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"pregunta": "¿Panecillo, arepa o galleta?", "opciones": [
        _op("Panecillo", 20, True), _op("Arepa", 80, nombre_plato="Arepas con queso y tortilla"), _op("Galleta", -30)]}]))
    ops = r["dudas"][0]["opciones"]
    assert [o["ajuste"]["calories"] for o in ops] == [0, 60, -50]
    assert ops[1]["nombre_plato"] == "Arepas con queso y tortilla"
    assert "nombre_plato" not in ops[0]


def test_sin_supuesta_marcada_se_toma_la_que_menos_cambia():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"pregunta": "¿Frito o a la plancha?", "opciones": [
        _op("Frito", 90), _op("A la plancha", 0)]}]))
    assert [o["supuesta"] for o in r["dudas"][0]["opciones"]] == [False, True]


def test_basura_y_topes():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"pregunta": "¿Cuánto arroz?", "opciones": [
        _op("1 taza", 0, True), "no soy un dict", {"texto": ""}, _op("x" * 90, 99999),
        _op("a", 1), _op("b", 2), _op("c", 3), _op("d", 4)]}]))
    ops = r["dudas"][0]["opciones"]
    assert len(ops) <= 5 and all(len(o["texto"]) <= 40 for o in ops)
    assert max(abs(o["ajuste"]["calories"]) for o in ops) <= 2000


def test_una_sola_opcion_valida_no_es_una_eleccion():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"pregunta": "¿Cuántos?", "opciones": [_op("3", 0, True)]}]))
    assert r["dudas"][0]["opciones"] == [] and r["dudas"][0]["pregunta"] == "¿Cuántos?"


def test_el_coach_ve_las_opciones_en_la_descripcion():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[{"pregunta": "¿De cuántos huevos?", "opciones": [
        _op("2 huevos", -72), _op("3 huevos", 0, True), _op("4 huevos", 72)]}]))
    assert "DUDAS (pregúntale solo esto)" in r["description"]
    assert "2 huevos · 3 huevos (supuesto) · 4 huevos" in r["description"]


def test_esquema_modelo_y_prompt():
    import vision_agent as va
    duda = va._MEAL_VISION_SCHEMA["properties"]["dudas"]["items"]["properties"]
    assert "opciones" in duda
    op = duda["opciones"]["items"]["properties"]
    assert {"texto", "supuesta", "ajuste", "nombre_plato"} <= set(op)
    assert "opciones" in va._MealVisionDuda.model_fields
    assert "OPCIONES" in va._MEAL_VISION_PROMPT and "supuesta" in va._MEAL_VISION_PROMPT
    va._MEAL_VISION_PROMPT.encode("ascii")   # el prompt de visión es ASCII (lote 305)


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 322
