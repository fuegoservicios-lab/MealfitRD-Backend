"""[P3-RECIPE-COHERENCE-AUTOFIX · 2026-06-13] Convierte las violaciones de coherencia
receta↔ingrediente (caso forward: receta menciona proteína no listada; completion: falta
paso final) de reject-and-retry → auto-fix in-place → el revisor aprueba al 1er intento →
retry_penalty=1.0 en el holistic.

Causa real (generación 11d17452): "Día 2 ... La receta indica 'pavo' pero no hay ningún
ingrediente equivalente listado" → rechazo → retry → holistic 0.918 en vez de ~0.99.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_orchestrator import _run_assembly_validations


def _meal(name, recipe, ingredients):
    return {"name": name, "meal": "Almuerzo", "recipe": recipe, "ingredients": ingredients,
            "cals": 500, "protein": 30, "carbs": 40, "fats": 15}


def test_forward_orphan_protein_is_scrubbed_not_flagged():
    # La receta menciona 'pavo' pero el ingrediente es pollo → auto-fix reemplaza pavo→pollo.
    meal = _meal("Guiso", ["Saltea el pavo con cebolla.", "Sirve caliente."],
                 ["150g de pollo", "1 cebolla"])
    result = {"days": [{"day": 1, "meals": [meal]}]}
    _run_assembly_validations(result, skeleton={}, affected_days_set=set())
    errs = result.get("_recipe_coherence_errors", [])
    assert not any("pavo" in e.lower() for e in errs), f"'pavo' no debió flaggearse: {errs}"
    # la receta ya no menciona pavo (fue reemplazado por la proteína real).
    recipe_txt = " ".join(meal["recipe"]).lower()
    assert "pavo" not in recipe_txt
    assert "pollo" in recipe_txt


def test_completion_missing_step_appended():
    # Receta sin paso final de servido → auto-fix añade "Montaje: Sirve...".
    meal = _meal("Bowl", ["Mezcla los ingredientes en un bowl."], ["100g de pollo", "arroz"])
    result = {"days": [{"day": 1, "meals": [meal]}]}
    _run_assembly_validations(result, skeleton={}, affected_days_set=set())
    errs = result.get("_recipe_coherence_errors", [])
    assert not any("incompleta" in e.lower() for e in errs), f"completion no debió flaggearse: {errs}"
    assert any("sirve" in str(s).lower() or "montaje" in str(s).lower() for s in meal["recipe"])


def test_orphan_with_no_real_protein_uses_generic():
    # Snack sin proteína real que menciona 'pavo' → reemplaza por 'proteína' (genérico).
    meal = _meal("Merienda", ["Acompaña el pavo con frutas.", "Sirve frío."],
                 ["1 guineo", "mantequilla de maní"])
    result = {"days": [{"day": 1, "meals": [meal]}]}
    _run_assembly_validations(result, skeleton={}, affected_days_set=set())
    recipe_txt = " ".join(meal["recipe"]).lower()
    assert "pavo" not in recipe_txt
    assert not any("pavo" in e.lower() for e in result.get("_recipe_coherence_errors", []))
