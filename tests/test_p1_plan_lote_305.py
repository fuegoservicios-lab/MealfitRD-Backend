# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-305 · 2026-09-25] La foto declara sus DUDAS y el coach pregunta solo eso.

El dueño: «si le mandamos una foto donde no se entienden las cantidades o el alimento (4 huevos revueltos fusionados),
que me pregunte de manera proactiva la duda que le falte; pero lo que es obvio debe apuntarlo de una vez».

  · Visión: el JSON gana `dudas` (máx. 2, solo lo que NO se puede saber mirando y cambia la cuenta: cantidad no
    contable, alimento ambiguo, grasa de cocción invisible). Viajan también en la `description` que lee el coach.
  · Coach (regla en `_PLATO_INSTRUCCION`, compartida por foto sola y varias): con dudas, registra con su mejor
    estimación, dice el supuesto y hace UNA pregunta; con la respuesta, `correct_consumed_meal` sobre ESE registro.
    Si la duda es qué plato es, pregunta ANTES de registrar.
  · Escáner: `/api/diary/upload` devuelve `dudas` y el modal las muestra.
Tooltip-anchor: P1-PLAN-LOTE-305
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _plato(**extra):
    base = {"photo_kind": "plato", "is_food": True, "meal_name": "Huevos revueltos",
            "description": "Huevos revueltos.", "calories": 280, "protein": 19, "carbs": 2, "healthy_fats": 21,
            "items": [{"name": "huevo", "quantity": 3, "unit": "unidad", "calories": 280, "protein": 19, "carbs": 2,
                       "healthy_fats": 21}]}
    base.update(extra)
    return base


def test_las_dudas_se_normalizan_y_viajan_en_la_descripcion():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=[
        {"sobre": "huevo", "pregunta": "¿Cuántos huevos eran?"},
        {"sobre": "aceite", "pregunta": "¿Los hiciste con aceite o mantequilla?"},
        {"sobre": "x", "pregunta": "¿Tercera duda que sobra?"},
    ]))
    assert [d["pregunta"] for d in r["dudas"]] == ["¿Cuántos huevos eran?", "¿Los hiciste con aceite o mantequilla?"]
    assert "DUDAS" in r["description"] and "¿Cuántos huevos eran?" in r["description"]


def test_sin_dudas_nada_cambia():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato())
    assert r["dudas"] == [] and "DUDAS" not in r["description"]


def test_dudas_basura_se_descartan():
    import vision_agent as va
    r = va._coerce_meal_scan(_plato(dudas=["no es un dict", {"pregunta": ""}, {"pregunta": "x" * 400}]))
    assert len(r["dudas"]) == 1 and len(r["dudas"][0]["pregunta"]) <= 160


def test_solo_un_plato_tiene_dudas():
    import vision_agent as va
    r = va._coerce_meal_scan({"photo_kind": "items", "is_food": True, "meal_name": "", "description": "Compra",
                              "calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0, "items": [],
                              "dudas": [{"pregunta": "¿Cuántos?"}]})
    assert r["dudas"] == []


def test_el_esquema_y_el_prompt_piden_las_dudas():
    import vision_agent as va
    assert "dudas" in va._MEAL_VISION_SCHEMA["properties"]
    assert "dudas" in va._MealVisionResult.model_fields
    assert "DUDAS" in va._MEAL_VISION_PROMPT and "NO pongas dudas" in va._MEAL_VISION_PROMPT


def test_el_coach_anota_lo_obvio_y_pregunta_la_duda():
    from prompts import chat_agent
    r = chat_agent._DUDAS_INSTRUCCION
    assert "DUDAS" in r and "UNA sola pregunta" in r and "correct_consumed_meal" in r
    assert "ANTES de registrar" in r


def test_la_regla_de_dudas_solo_va_cuando_hay_dudas():
    from prompts import chat_agent
    con = chat_agent.build_vision_context({"kind": "plato", "description": "Huevos. DUDAS (pregúntale solo esto): ¿Cuántos huevos eran?"})
    sin = chat_agent.build_vision_context({"kind": "plato", "description": "Dos rebanadas de pan con queso."})
    assert "UNA sola pregunta" in con and "UNA sola pregunta" not in sin
    varias = chat_agent.build_vision_context({"kind": "multi", "items": [
        {"kind": "plato", "description": "Arroz. DUDAS (pregúntale solo esto): ¿Cuánto arroz?"}]})
    assert "UNA sola pregunta" in varias


def test_el_escaner_recibe_las_dudas():
    assert '"dudas": vision_result.get("dudas") or []' in _src("routers/diary.py")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 305
