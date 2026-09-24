# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-189 · 2026-09-23] Batería rd17 (alergia a lácteos y mariscos): plan de EMERGENCIA otra vez.

Intento 1: «La nota de seguridad del desayuno del día 2 indica que se usó yogur griego» (crítico). La nota era NUESTRA:
el huevo crudo de un batido se cambia por «Yogurt griego sin azúcar» sin mirar las alergias; el 188 lo pasó a yogur de
coco, pero la nota seguía nombrando el yogur griego. Contando hacia atrás, 2 de las 3 violaciones de lácteos de
rd11–rd16 («¼ taza de yogurt griego sin azúcar») las había metido esa pasada. En cambiar plato y en el chat la guarda
de alérgenos corre ANTES que ella: el yogur quedaba guardado.

Intento 2: «Verificar que el pan integral no contenga derivados lácteos…» (crítico) — otra forma verbal de pedir que
se revise una etiqueta. En vez de otra frase: verbo de verificación + «no contenga / libre de / puede contener» es
petición de verificación (aviso), salvo pasteurizar. Medido contra las 119 issues reales de rd2–rd17: pasa a aviso
exactamente esa, ninguna más."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


class _StubDB:
    def macros_from_ingredient_string(self, s):
        t = str(s).lower()
        if "huevo" in t:
            return {"kcal": 70.0, "protein": 6.0, "carbs": 0.0, "fats": 5.0}
        if "coco" in t:
            return {"kcal": 90.0, "protein": 1.0, "carbs": 6.0, "fats": 7.0}
        if "yogur" in t:
            return {"kcal": 60.0, "protein": 10.0, "carbs": 4.0, "fats": 0.0}
        if "tofu" in t:
            return {"kcal": 145.0, "protein": 15.0, "carbs": 3.0, "fats": 8.0}
        return None


def _batido():
    ings = ["1 manzana", "2 huevos crudos", "30 g de aguacate"]
    return {"meal": "Desayuno", "name": "Batido de manzana y huevo", "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": ["Licúa la manzana, los huevos y el aguacate hasta que quede suave."],
            "protein": 14, "carbs": 30, "fats": 14, "cals": 300}


def _plan(meal):
    return {"days": [{"day": 2, "meals": [meal]}]}


@pytest.fixture()
def stubdb(monkeypatch):
    import nutrition_db
    monkeypatch.setattr(nutrition_db, "IngredientNutritionDB", _StubDB)


def test_alergico_a_lacteos_no_recibe_yogur_griego(go, stubdb):
    meal = _batido()
    go._apply_food_safety_fixes(_plan(meal), {"allergies": ["Lacteos", "Mariscos"], "dietType": "balanced"})
    texto = " ".join([meal["name"], *meal["ingredients"], *meal["recipe"]]).lower()
    assert "griego" not in texto, texto
    assert "Yogur de coco" in meal["ingredients"], meal["ingredients"]
    assert "yogur de coco" in " ".join(meal["recipe"][:-1]).lower(), meal["recipe"]
    assert go._scan_allergen_violations(_plan(meal), ["Lacteos"]) == []
    nota = go._meal_safety_notes_for_summary(meal)
    assert "yogur de coco" in nota and "griego" not in nota, "lo que lee el revisor dice lo que hay en el plato"


def test_sin_perfil_la_conducta_de_siempre(go, stubdb):
    meal = _batido()
    go._apply_food_safety_fixes(_plan(meal))
    assert meal["ingredients"][1] == go._BLEND_EGG_REPLACEMENT
    assert meal["recipe"][-1] == go._FOOD_SAFETY_NOTE_BLENDED_SUBBED


def test_sin_candidato_limpio_no_se_sustituye_ni_se_nombra_nada(go, stubdb):
    meal = _batido()
    go._apply_food_safety_fixes(_plan(meal), {"allergies": ["Lacteos", "Coco", "Soya"]})
    assert "2 huevos crudos" in meal["ingredients"], "sin sustituto limpio el huevo se queda (nota de pasteurizado)"
    assert "yogur" not in meal["recipe"][-1].lower() and "PASTEURIZADO" in meal["recipe"][-1]


def test_alergico_al_huevo_no_le_sugiere_huevo(go, stubdb):
    meal = _batido()
    go._apply_food_safety_fixes(_plan(meal), {"allergies": ["Huevo"]})
    assert meal["ingredients"][1] == go._BLEND_EGG_REPLACEMENT
    assert "Si prefieres huevo" not in meal["recipe"][-1]


def test_superficie_de_actualizacion_usa_las_alergias(go, stubdb):
    meal = _batido()
    go.food_safety_backstop_for_meal(meal, allergies=["Lacteos"])
    assert "Yogur de coco" in meal["ingredients"], meal["ingredients"]


def test_cambiar_plato_y_chat_pasan_las_alergias():
    agent = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    tools = (_BACKEND / "tools.py").read_text(encoding="utf-8")
    assert "food_safety_backstop_for_meal(_out, form_data=form_data, allergies=allergies)" in agent
    assert "food_safety_backstop_for_meal(new_meal_data, form_data=form_data, allergies=_clin_allergies)" in tools


@pytest.mark.parametrize("issue", [
    "Verificar que el pan integral no contenga derivados lácteos, como caseína, suero o mantequilla, antes de consumirlo.",
    "Asegúrese de que la tortilla de maíz esté libre de trazas de mariscos.",
    "Comprobar que el pan puede contener leche según la etiqueta.",
])
def test_verificar_que_no_contenga_es_aviso(go, issue):
    aprobado, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert aprobado and reales == [] and avisos == [issue]


@pytest.mark.parametrize("issue", [
    "Día 2, merienda: el yogur griego contiene lácteos y contradice la alergia declarada.",
    "Verificar que el queso fresco esté libre de riesgo: debe ser pasteurizado en el embarazo.",
    "El plan incluye 460 g de pescado en tres días, por encima de la recomendación semanal.",
])
def test_lo_que_no_es_verificacion_sigue_siendo_real(go, issue):
    _a, reales, _s, avisos = go._downgrade_reviewer_verification_demands(False, [issue], "critical")
    assert reales == [issue] and avisos == []


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 189 and m.group(2) >= "2026-09-23"
