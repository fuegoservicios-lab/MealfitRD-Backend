"""[P0-6] Tests para garantizar que las validaciones críticas post-assembly
se ejecutan SIEMPRE, tanto en path LLM como en cache-hit semántico.

Bug original (audit P0-6):
  Las 3 validaciones críticas (Skeleton Fidelity, Recipe Coherence, Schema
  Validation contra `PlanModel`) vivían inline al final de `assemble_plan_node`.
  Aunque el flujo SÍ las ejecutaba para cache-hit (no había `return` temprano),
  su entremezcla con código no relacionado entre la rama if/else (cache vs LLM)
  y las validaciones (~500 líneas de macro balancing, shopping list, humanización)
  hacía que un futuro refactor pudiera olvidar aplicarlas a una rama.

  Riesgo concreto: un plan cacheado con shape vieja (campo renombrado,
  `_cache_schema_version` literal igualó pero `PlanModel` cambió, validación
  añadida después, etc.) podría llegar al frontend sin `_schema_invalid=True`,
  y `_apply_critical_review_guardrails` no triggearía → frontend recibe plan
  no-renderizable.

Fix:
  Las 3 validaciones se extraen a un helper compartido `_run_assembly_validations`
  que se invoca SIEMPRE (sin importar la rama). Hace explícito el contrato y
  permite testearlo en aislamiento.

Cobertura:
  - test_helper_sets_schema_invalid_for_missing_required_field
  - test_helper_sets_schema_invalid_for_wrong_type
  - test_helper_does_not_set_schema_invalid_for_valid_plan
  - test_helper_runs_for_cache_hit_with_empty_skeleton
  - test_helper_detects_recipe_coherence_missing_protein
  - test_helper_detects_recipe_missing_completion_step
  - test_helper_skeleton_fidelity_trivial_pass_for_empty_skeleton
  - test_helper_skeleton_fidelity_detects_missing_proteins_with_skeleton
  - test_assemble_plan_node_delegates_to_shared_helper
"""
import re
import inspect

import pytest

import graph_orchestrator
from graph_orchestrator import _run_assembly_validations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_meal(**overrides):
    base = {
        "meal": "Desayuno",
        "time": "8:00 AM",
        "name": "Avena con frutas",
        "desc": "Avena cocida con leche de almendras y frutas frescas.",
        "prep_time": "10 min",
        "difficulty": "Fácil",
        "cals": 400,
        "protein": 15,
        "carbs": 60,
        "fats": 8,
        "macros": ["Equilibrado"],
        "ingredients": ["1 taza de avena", "1 taza de leche de almendras", "1 manzana"],
        "recipe": [
            "Mise en place: cocinar avena con leche.",
            "El Toque de Fuego: hervir 5 min.",
            "Montaje: servir con manzana picada.",
        ],
    }
    base.update(overrides)
    return base


def _valid_plan(**overrides):
    base = {
        "main_goal": "Pérdida de Peso (Déficit)",
        "calories": 1800,
        "macros": {"protein": "120g", "carbs": "180g", "fats": "60g"},
        "insights": [
            "Diagnóstico: déficit calórico moderado.",
            "Estrategia: priorizar proteína magra.",
            "Tip del Chef: usa especias para reemplazar sal.",
        ],
        "days": [
            {
                "day": 1,
                "day_name": "Lunes",
                "meals": [_valid_meal()],
            },
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Schema validation: el caso central del audit P0-6.
# ---------------------------------------------------------------------------
def test_helper_sets_schema_invalid_for_missing_required_field():
    """Plan sin campo `calories` (requerido por PlanModel) → _schema_invalid=True."""
    plan = _valid_plan()
    del plan["calories"]
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    assert plan.get("_schema_invalid") is True
    assert "_schema_errors" in plan
    assert plan["_schema_errors"]


def test_helper_sets_schema_invalid_for_wrong_type():
    """Plan con `calories` como string en lugar de int → _schema_invalid=True."""
    plan = _valid_plan()
    plan["calories"] = "not-a-number"
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    assert plan.get("_schema_invalid") is True


def test_helper_does_not_set_schema_invalid_for_valid_plan():
    """Plan que cumple PlanModel → no setea _schema_invalid."""
    plan = _valid_plan()
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    assert plan.get("_schema_invalid") is not True
    assert "_schema_errors" not in plan


# ---------------------------------------------------------------------------
# 2. Cache-hit symmetry: skeleton vacío NO debe saltar las validaciones.
# ---------------------------------------------------------------------------
def test_helper_runs_for_cache_hit_with_empty_skeleton():
    """Cache-hit pasa skeleton={}. Schema validation y recipe coherence
    deben ejecutarse igual. Skeleton fidelity es trivial-pass (no hay
    skeleton para comparar)."""
    # Plan corrupto con shape vieja simulada: calories ausente.
    plan = _valid_plan()
    del plan["calories"]
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    # Schema se valida → falla → setea _schema_invalid.
    assert plan.get("_schema_invalid") is True


def test_helper_runs_recipe_coherence_for_cache_hit():
    """Cache-hit con incoherencia receta↔ingredientes → coherence errors."""
    bad_recipe_meal = _valid_meal(
        recipe=[
            "Mise en place: marinar el pollo.",
            "El Toque de Fuego: cocinar al horno.",
            "Montaje: emplatar.",
        ],
        ingredients=["1 taza de arroz", "1 brócoli"],  # NO HAY POLLO
        name="Pollo al Horno",
    )
    plan = _valid_plan(days=[{"day": 1, "day_name": "Lunes", "meals": [bad_recipe_meal]}])
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    errors = plan.get("_recipe_coherence_errors") or []
    assert any("pollo" in e.lower() for e in errors), \
        f"esperado error de pollo no listado, got: {errors}"


def test_helper_detects_recipe_missing_completion_step():
    """Receta sin 'servir/montaje/emplatar/...' → coherence error."""
    bad_meal = _valid_meal(
        recipe=[
            "Mise en place: preparar.",
            "El Toque de Fuego: cocinar.",
            # SIN paso final tipo 'Servir'/'Montaje'.
            "Listo para comer.",
        ],
    )
    plan = _valid_plan(days=[{"day": 1, "day_name": "Lunes", "meals": [bad_meal]}])
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    errors = plan.get("_recipe_coherence_errors") or []
    assert any("incompleta" in e.lower() or "paso final" in e.lower() for e in errors), \
        f"esperado error de receta incompleta, got: {errors}"


# ---------------------------------------------------------------------------
# 3. Skeleton fidelity — depende de skeleton; trivial-pass cuando vacío.
# ---------------------------------------------------------------------------
def test_helper_skeleton_fidelity_trivial_pass_for_empty_skeleton():
    """Cache-hit (skeleton vacío) → no agrega `_skeleton_fidelity_errors`
    aunque las proteínas no estén en el día."""
    plan = _valid_plan()
    _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    # No hay assigned_proteins → no hay missing → no hay errors.
    assert "_skeleton_fidelity_errors" not in plan


def test_helper_skeleton_fidelity_detects_missing_proteins_with_skeleton():
    """Path LLM con skeleton lleno y plan que omite ≥2 proteínas asignadas
    → `_skeleton_fidelity_errors` poblado."""
    plan = _valid_plan(
        days=[{
            "day": 1,
            "day_name": "Lunes",
            "meals": [_valid_meal(ingredients=["1 taza de arroz", "1 brócoli"])],
        }]
    )
    skeleton = {
        "days": [
            {"day": 1, "protein_pool": ["pechuga de pollo", "salmón fresco"]},
        ]
    }
    _run_assembly_validations(plan, skeleton=skeleton, affected_days_set=set())
    errors = plan.get("_skeleton_fidelity_errors") or []
    assert errors, f"esperado errors de skeleton fidelity, got: {errors}"
    assert any("Día 1" in e for e in errors)


def test_helper_skeleton_fidelity_skips_unaffected_days_in_surgical_mode():
    """Si `affected_days_set` está poblado y el día no está en él → skip."""
    plan = _valid_plan(
        days=[{
            "day": 1,
            "day_name": "Lunes",
            "meals": [_valid_meal(ingredients=["1 taza de arroz"])],
        }]
    )
    skeleton = {"days": [{"day": 1, "protein_pool": ["pollo", "salmón"]}]}
    # El día 1 NO está en affected_days_set → debe saltarse.
    _run_assembly_validations(plan, skeleton=skeleton, affected_days_set={2, 3})
    assert "_skeleton_fidelity_errors" not in plan


# ---------------------------------------------------------------------------
# 4. Defensa estructural: assemble_plan_node delega al helper.
# ---------------------------------------------------------------------------
def test_assemble_plan_node_delegates_to_shared_helper():
    """`assemble_plan_node` debe invocar `_run_assembly_validations` para
    garantizar simetría entre cache-hit y LLM-path. Si alguien re-introduce
    las validaciones inline (split branches), este test rompe."""
    src = inspect.getsource(graph_orchestrator.assemble_plan_node)
    assert "_run_assembly_validations" in src, \
        "assemble_plan_node debe invocar _run_assembly_validations (extracción P0-6)"


def test_helper_signature_accepts_required_args():
    """El contrato del helper acepta (result, skeleton, affected_days_set)."""
    sig = inspect.signature(_run_assembly_validations)
    params = list(sig.parameters.keys())
    assert params == ["result", "skeleton", "affected_days_set"], \
        f"firma del helper cambió: {params}"


# ---------------------------------------------------------------------------
# 5. Mutación in-place del result.
# ---------------------------------------------------------------------------
def test_helper_mutates_result_in_place():
    """El helper muta `result` (no retorna). Verificamos mutación in-place
    sobre instancia idéntica."""
    plan = _valid_plan()
    del plan["calories"]
    plan_id = id(plan)
    rv = _run_assembly_validations(plan, skeleton={}, affected_days_set=set())
    assert rv is None  # no retorna nada
    assert id(plan) == plan_id
    assert plan.get("_schema_invalid") is True
