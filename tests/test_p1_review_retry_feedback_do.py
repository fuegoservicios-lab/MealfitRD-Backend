"""[P1-REVIEW-RETRY-FEEDBACK-DO · 2026-08-23]

Los cuatro rechazos deterministas deben conservar su requisito sin imponer la
cocina dominicana a países beta. La memoria histórica se neutraliza al leer,
sin reescribir producción ni renombrar alimentos del usuario.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[1]
_BETA = ("ES", "MX", "CO", "PR", "US")


def _render_four(country: str) -> tuple[str, str, str, str]:
    from graph_orchestrator import _review_country_feedback

    return (
        _review_country_feedback(
            country,
            "egg_overuse",
            egg=9,
            total=20,
            cap=5,
        ),
        _review_country_feedback(
            country,
            "raw_staples",
            count=7,
            sample="proteína plancha + carbo blanco",
        ),
        _review_country_feedback(country, "transform_minimum"),
        _review_country_feedback(country, "recipe_contract", ratio_pct=64),
    )


def test_do_conserva_los_cuatro_feedback_historicos_byte_a_byte():
    egg, raw, transformed, recipe = _render_four("DO")
    assert egg == (
        "SOBREUSO DE HUEVO (rechazo de variedad): el huevo aparece en 9 de 20 comidas "
        "(máximo 5). Reemplaza el huevo en al menos 4 comida(s) por otras proteínas "
        "dominicanas (pollo guisado, pescado, atún, sardina, res molida magra, queso de "
        "freír, yogur griego, habichuelas) — NO uses huevo como relleno por defecto. "
        "Mantén el huevo solo en desayunos/platos donde es el protagonista."
    )
    assert raw == (
        "7 plato(s) son ingredientes crudos/hervidos sin transformación culinaria "
        "(proteína plancha + carbo blanco). Convierte los platos en PREPARACIONES "
        "dominicanas reales: guisos, locrios (almuerzo), panqueques/arepitas con las "
        "harinas, bollitos de yuca, revoltillos, ensaladas compuestas — manteniendo los "
        "mismos macros."
    )
    assert transformed == (
        "El plan no incluye NINGUNA preparación transformada: incluye al menos una "
        "preparación real con los mismos macros (panqueques de avena, arepitas, "
        "bollitos de yuca, revoltillo, guiso, locrio de almuerzo) en vez de solo staples "
        "servidos."
    )
    assert recipe.endswith("en español dominicano.")


@pytest.mark.parametrize("country", _BETA)
def test_los_cuatro_feedback_beta_preservan_requisito_sin_cocina_do(country):
    rendered = _render_four(country)
    blob = "\n".join(rendered)
    assert not re.search(r"dominican\w*", blob, flags=re.IGNORECASE)
    for do_only in ("locrio", "bollitos de yuca", "arepitas", "queso de freír"):
        assert do_only not in blob.lower()
    assert "SOBREUSO DE HUEVO" in rendered[0] and "máximo 5" in rendered[0]
    assert "transformación culinaria" in rendered[1]
    assert "preparación real" in rendered[2]
    assert "3 pilares EN ORDEN" in rendered[3]


def test_pais_del_revisor_se_deriva_antes_del_primer_feedback_ast():
    source = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "review_plan_node"
    )
    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_rpn_country" for t in node.targets)
    ]
    feedback_calls = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_review_country_feedback"
    ]
    assert len(assignments) == 1
    assert len(feedback_calls) == 4
    assert assignments[0].lineno < min(call.lineno for call in feedback_calls)


def test_memoria_legacy_se_neutraliza_al_leer_sin_renombrar_alimentos():
    from ai_helpers import _country_safe_reviewer_memory

    stale = (
        "COMIDA FUERA DE HORARIO (rechazo de coherencia cultural es-DO): Día 2, "
        "no corresponde al desayuno dominicano."
    )
    neutral = _country_safe_reviewer_memory(stale, "ES")
    assert "COMIDA FUERA DE HORARIO" in neutral and "Día 2" in neutral
    assert "es-DO" not in neutral
    assert not re.search(r"dominican\w*", neutral, flags=re.IGNORECASE)
    assert _country_safe_reviewer_memory(stale, "DO") == stale

    food_identifier = "Evita repetir Longaniza dominicana en la cena"
    assert _country_safe_reviewer_memory(food_identifier, "ES") == food_identifier

    ai_source = (_BACKEND / "ai_helpers.py").read_text(encoding="utf-8")
    assert "_country_safe_reviewer_memory(r, _variety_country)" in ai_source


def test_marker_movil_del_gap():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    graph = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '_LAST_KNOWN_PFIX = "P' in app and " · 2026-" in app
    assert "P1-REVIEW-RETRY-FEEDBACK-DO" in graph
