"""G03: las filas RTE beta reciben una nota clínica sin mutar el plan nutricional."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import graph_orchestrator as go


BACKEND_ROOT = Path(__file__).resolve().parents[1]
RTE_SNAPSHOT = BACKEND_ROOT / "tests" / "master_ingredients_rte_clinical_2026_08_23.json"
FORM = {"country": "ES", "medicalConditions": ["Embarazo"]}


def _plan(food: str) -> dict:
    return {
        "days": [{
            "day": 1,
            "meals": [{
                "name": f"Sonda de {food}",
                "ingredients": [f"100 g de {food}"],
                "recipe": ["Sirve al momento."],
                "macros": {"protein": 20, "carbs": 10, "fats": 8},
            }],
        }],
        "shoppingList": [{"name": food, "quantity": "100 g"}],
    }


def _note(plan: dict) -> str:
    return " ".join(plan["days"][0]["meals"][0].get("recipe") or [])


@pytest.fixture
def catalogo_rte(monkeypatch):
    import shopping_calculator as sc

    rows = json.loads(RTE_SNAPSHOT.read_text(encoding="utf-8"))
    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: copy.deepcopy(rows))
    return rows


@pytest.mark.parametrize(
    "food",
    ["Chorizo español", "Sobrasada", "Lomo embuchado", "Morcilla", "Cuajada"],
)
def test_cinco_filas_beta_antes_mudas_reciben_nota(catalogo_rte, food: str) -> None:
    plan = _plan(food)
    assert go._apply_pregnancy_food_safety_annotations(plan, FORM) == 1
    note = _note(plan)
    assert go._PREGNANCY_NOTE_PREFIX in note
    assert ("74 °C" in note) if food != "Cuajada" else ("PASTEURIZADOS" in note)


def test_capa_es_note_only_macro_preservante_y_shopping_safe(catalogo_rte) -> None:
    plan = _plan("Chorizo español")
    before = copy.deepcopy(plan)

    assert go._apply_pregnancy_food_safety_annotations(plan, FORM) == 1

    meal_before = before["days"][0]["meals"][0]
    meal_after = plan["days"][0]["meals"][0]
    assert meal_after["ingredients"] == meal_before["ingredients"]
    assert meal_after["macros"] == meal_before["macros"]
    assert plan["shoppingList"] == before["shoppingList"]
    assert meal_after["recipe"][:-1] == meal_before["recipe"]
    assert len(meal_after["recipe"]) == len(meal_before["recipe"]) + 1


def test_alta_hook_cubre_cada_rte_clinico_o_exencion_documentada(catalogo_rte) -> None:
    exempt = {
        # Ya vienen cocidos y la función conserva su exención canned explícita.
        "Atún en agua": "enlatado/listo; no requiere recocción",
        "Sardinas en lata": "enlatado/listo; no requiere recocción",
        # Bebidas/yogur vegetales: no presentan el riesgo de lácteo no pasteurizado.
        "Leche de almendras": "bebida vegetal",
        "Leche de coco": "bebida vegetal",
        "Leche de soya": "bebida vegetal",
        "Yogur de coco": "producto vegetal",
    }
    seen = {row["name"] for row in catalogo_rte}
    assert set(exempt) <= seen

    missing = []
    for row in catalogo_rte:
        food = row["name"]
        plan = _plan(food)
        annotated = go._apply_pregnancy_food_safety_annotations(plan, FORM)
        if food in exempt:
            assert annotated == 0, f"la exención de {food} dejó de cumplirse"
        elif annotated != 1 or go._PREGNANCY_NOTE_PREFIX not in _note(plan):
            missing.append(food)
    assert missing == []


def test_derivacion_usa_metadata_no_lista_dietetica_como_clausula(catalogo_rte) -> None:
    deli, dairy, marine = go._pregnancy_catalog_risk_tokens(catalogo_rte)
    assert "chorizo espanol" in deli
    assert "cuajada" in dairy
    assert "anchoas" in marine
    assert "huevos rellenos" not in deli  # tiene su cláusula de huevo
    assert "leche de coco" not in dairy


def test_funcion_recibe_plan_no_comida_suelta(catalogo_rte) -> None:
    meal = _plan("Chorizo español")["days"][0]["meals"][0]
    assert go._apply_pregnancy_food_safety_annotations(meal, FORM) == 0


@pytest.mark.parametrize(
    "food",
    ["Chorizo español", "Sobrasada", "Lomo embuchado", "Morcilla", "Cuajada"],
)
def test_fallback_estatico_protege_cinco_filas_sin_catalogo(monkeypatch, food: str) -> None:
    import shopping_calculator as sc

    monkeypatch.setattr(sc, "get_master_ingredients", lambda *a, **k: [])
    plan = _plan(food)
    assert go._apply_pregnancy_food_safety_annotations(plan, FORM) == 1


def test_mutacion_quitar_chorizo_del_fallback_pone_guard_rojo(catalogo_rte) -> None:
    deli_static = next(
        toks for key, toks, _covered, _text in go._PREGNANCY_SAFETY_CLAUSES if key == "deli"
    )
    deli_dynamic, _dairy, _marine = go._pregnancy_catalog_risk_tokens(catalogo_rte)
    assert "chorizo" in deli_static
    assert "chorizo espanol" in deli_dynamic


def test_pfix_marker_cierra_g03() -> None:
    source = (BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P0-PREG-CURED-BETA · 2026-08-23" in source
