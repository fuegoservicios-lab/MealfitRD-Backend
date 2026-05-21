"""[P1-AUTO-PATCH-FORWARD · 2026-05-21] Cierra la asimetría histórica del
sistema de auto-patch de coherencia receta↔ingrediente.

Bug productivo (chunk 713ff43a, plan bf6f1383, 2026-05-21 00:08):
  El revisor médico rechazó tres meals porque la receta mencionaba
  proteínas que no aparecían en `ingredients`:
    "Día 4, Filete de Chillo al Vapor: La receta indica 'pescado' pero
    no hay ningún ingrediente equivalente (ej. pescado, chillo, dorado)."
  El LLM había generado un dish name + recipe text refiriéndose a
  pescado, pero los ingredientes eran `["5 claras de huevo", "60g casabe", ...]`.
  Resultado: retry del plan completo → ~10 llamadas Gemini adicionales.

Asimetría histórica:
  `_auto_patch_ingredient_coherence` (P6-AUTO-PATCH-1) cubría la dirección
  REVERSE (ingrediente listado pero no en receta → eliminar de lista).
  La dirección FORWARD (receta menciona proteína pero no está en lista)
  quedó descubierta → siempre forzaba retry.

Fix:
  Nueva función `_auto_patch_recipe_forward_coherence(plan, errors)` que
  parsea los errores forward, identifica la proteína REAL en ingredients
  (qué categoría de `_FORWARD_PATCH_SYNONYMS` matchea), y reescribe
  `meal.name` + `meal.recipe` reemplazando el orphan KEY por la proteína
  real (case-preserving, longest-first para multi-word).

  Si no podemos identificar proteína real (e.g. solo veggies) preservamos
  el error en `unpatched_errors` → escala a retry como antes. Preferimos
  retry que invención.

Cobertura:
  - Estructural: la función + el callsite + el marker existen
  - Funcional: meal de repro real (Filete de Chillo + claras) → patcheado
  - Veggie-only ingredients → unpatched (retry forzado)
  - Multi-word longest-first (filete de chillo)
  - Case-preserving (Chillo → Huevo, chillo → huevo)
  - Idempotencia (segunda pasada no rompe)
  - Tooltip anchor de sync con el dict del check forward
"""
import re
from pathlib import Path

import pytest


_GRAPH_ORCH = Path(__file__).parent.parent / "graph_orchestrator.py"


# ---------------------------------------------------------------------------
# Sección 1 — Tests estructurales (parser-based)
# ---------------------------------------------------------------------------

def test_function_defined_in_graph_orchestrator():
    """`_auto_patch_recipe_forward_coherence` debe existir como función
    module-level. Si renombras o eliminas, este test cae antes de
    bypassear el comportamiento en producción."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "def _auto_patch_recipe_forward_coherence(plan: dict, errors: list)" in src, (
        "La función _auto_patch_recipe_forward_coherence debe existir en "
        "graph_orchestrator.py. Si la renombraste, actualiza este test."
    )


def test_callsite_wires_forward_marker_and_calls_function():
    """El callsite en `review_plan_node` (o equivalente) debe:
      1. Definir COHERENCE_FORWARD_PATCHABLE_MARKER
      2. Filtrar forward_patchable_errors usando el marker
      3. Llamar a _auto_patch_recipe_forward_coherence
      4. Excluir el marker forward también del set `structural_coherence_errors`
    """
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'COHERENCE_FORWARD_PATCHABLE_MARKER = "pero no hay ningún ingrediente equivalente"' in src, (
        "El callsite debe definir COHERENCE_FORWARD_PATCHABLE_MARKER con el "
        "texto literal del check forward."
    )
    assert "forward_patchable_errors = [e for e in coherence_errors if COHERENCE_FORWARD_PATCHABLE_MARKER in e]" in src
    assert "_auto_patch_recipe_forward_coherence(plan, forward_patchable_errors)" in src
    # Estructural: el structural_coherence_errors debe excluir AMBOS markers
    # (cosmético y forward), no solo el cosmético — si alguien revierte a
    # filter solo por COHERENCE_PATCHABLE_MARKER, el forward iría a retry.
    assert (
        "if COHERENCE_PATCHABLE_MARKER not in e and COHERENCE_FORWARD_PATCHABLE_MARKER not in e"
        in src
    )


def test_tooltip_anchor_for_sync_with_forward_check_dict():
    """[P1-AUTO-PATCH-FORWARD-SYNC-TOOLTIP] El dict `_FORWARD_PATCH_SYNONYMS`
    debe declarar explícitamente la convención de sync con el `protein_synonyms`
    LOCAL del check forward (línea ~6942 en `_run_assembly_and_validation`).
    Sin esto, un dev que añada un pez nuevo allí podría no replicarlo aquí
    → el patch fallaría silenciosamente para esa proteína nueva."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "P1-AUTO-PATCH-FORWARD-SYNC-TOOLTIP" in src
    assert "_FORWARD_PATCH_SYNONYMS" in src


# ---------------------------------------------------------------------------
# Sección 2 — Tests funcionales
# ---------------------------------------------------------------------------

from graph_orchestrator import _auto_patch_recipe_forward_coherence  # noqa: E402


def _make_plan(day_num: int, meal_name: str, ingredients, recipe) -> dict:
    return {
        "days": [
            {
                "day": day_num,
                "meals": [
                    {
                        "name": meal_name,
                        "ingredients": ingredients,
                        "recipe": recipe,
                    }
                ],
            }
        ]
    }


def test_repro_incident_filete_de_chillo_with_claras():
    """Repro exacto del chunk 713ff43a 2026-05-21:
       name="Filete de Chillo al Vapor..." + ingredients=["5 claras de huevo", ...]
       + recipe mencionando 'pescado'."""
    plan = _make_plan(
        day_num=4,
        meal_name="Filete de Chillo al Vapor con Revuelto de Claras y Casabe",
        ingredients=["5 claras de huevo", "60g casabe", "1 cda aceite de oliva"],
        recipe="Sazonar el pescado con sal. Cocinar el pescado al vapor. Servir caliente.",
    )
    error = (
        "Día 4, Filete de Chillo al Vapor con Revuelto de Claras y Casabe: "
        "La receta indica 'pescado' pero no hay ningún ingrediente equivalente "
        "(ej. pescado, pescados, chillo) listado."
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [error])

    assert patched_count == 1, f"Expected 1 patched, got {patched_count}"
    assert unpatched == [], f"Expected zero unpatched, got {unpatched}"

    meal = plan["days"][0]["meals"][0]
    new_recipe = meal["recipe"]
    new_name = meal["name"]

    # 'pescado' / 'chillo' no deben sobrevivir en recipe
    assert "pescado" not in new_recipe.lower(), f"recipe aún tiene 'pescado': {new_recipe}"
    assert "chillo" not in new_recipe.lower(), f"recipe aún tiene 'chillo': {new_recipe}"
    # 'huevo' (la proteína real desde claras) debe aparecer
    assert "huevo" in new_recipe.lower(), f"recipe debería mencionar 'huevo': {new_recipe}"
    # El nombre también — 'Chillo' (capitalizado) ya no debe estar
    assert "chillo" not in new_name.lower(), f"name aún tiene 'chillo': {new_name}"
    assert "Huevo" in new_name or "huevo" in new_name, f"name debería mencionar 'huevo': {new_name}"


def test_recipe_list_form_is_patched_step_by_step():
    """El campo `recipe` puede llegar como list[str] (un step por elemento).
    El patch debe iterar la lista y reescribir cada step."""
    plan = _make_plan(
        day_num=2,
        meal_name="Estofado de Pollo con Yuca",
        ingredients=["1 taza de garbanzos cocidos", "200g de yuca"],
        recipe=[
            "Mise en place: cortar la yuca en dados.",
            "El Toque de Fuego: dorar el pollo en sartén.",
            "Montaje: servir con la yuca.",
        ],
    )
    error = (
        "Día 2, Estofado de Pollo con Yuca: La receta indica 'pollo' "
        "pero no hay ningún ingrediente equivalente (ej. pollo, pechuga, muslo) listado."
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [error])
    assert patched_count == 1
    assert unpatched == []

    new_recipe = plan["days"][0]["meals"][0]["recipe"]
    assert isinstance(new_recipe, list), "recipe debe seguir siendo list"
    joined = " ".join(new_recipe).lower()
    assert "pollo" not in joined, f"recipe aún tiene 'pollo': {new_recipe}"
    assert "garbanzos" in joined, f"recipe debería mencionar 'garbanzos': {new_recipe}"


def test_veggie_only_ingredients_left_unpatched_to_force_retry():
    """Si los ingredients son solo veggies/cereales sin proteína animal NI
    vegetal identificable, no podemos inventar — dejar al retry."""
    plan = _make_plan(
        day_num=1,
        meal_name="Salteado de Pavo Imaginario",
        ingredients=["1 taza espinacas", "100g zanahoria rallada", "1 cda aceite de oliva"],
        recipe="Sofreír el pavo en aceite. Servir con vegetales.",
    )
    error = (
        "Día 1, Salteado de Pavo Imaginario: La receta indica 'pavo' "
        "pero no hay ningún ingrediente equivalente (ej. pavo, pechuga de pavo) listado."
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [error])
    assert patched_count == 0
    assert unpatched == [error], (
        "Sin proteína identificable, el error debe propagarse para que el "
        "flujo legacy escale a retry. Preferimos retry que invención."
    )


def test_orphan_outside_safe_map_left_unpatched():
    """Si el orphan no está en `_FORWARD_PATCH_SYNONYMS` (e.g. una proteína
    nueva que el dev añadió al check forward pero olvidó replicar aquí),
    NO debemos patchear — dejar al retry para evitar text corrupto."""
    plan = _make_plan(
        day_num=1,
        meal_name="Algo con Calamar",
        ingredients=["100g de calamar", "1 cda aceite"],
        recipe="Cocinar el calamar. Servir.",
    )
    error = (
        "Día 1, Algo con Calamar: La receta indica 'calamar' "
        "pero no hay ningún ingrediente equivalente (ej. calamar) listado."
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [error])
    # 'calamar' no está en _FORWARD_PATCH_SYNONYMS → unpatched
    assert patched_count == 0
    assert unpatched == [error]


def test_multi_word_synonym_replaced_before_single_word():
    """Longest-first: 'filete de chillo' debe reemplazarse antes que 'chillo'
    para que no quede 'filete de huevo' parcialmente substituido a través
    de pasadas múltiples."""
    plan = _make_plan(
        day_num=3,
        meal_name="Plato con Filete de Chillo",
        ingredients=["5 claras de huevo", "1 cda aceite"],
        recipe="Cocinar el filete de chillo al vapor. Servir el pescado caliente.",
    )
    error = (
        "Día 3, Plato con Filete de Chillo: La receta indica 'pescado' "
        "pero no hay ningún ingrediente equivalente listado."
    )
    patched_count, _ = _auto_patch_recipe_forward_coherence(plan, [error])
    assert patched_count == 1
    new_recipe = plan["days"][0]["meals"][0]["recipe"].lower()
    # Ni 'pescado' ni 'chillo' (en cualquier forma multi-word) debe sobrevivir
    assert "pescado" not in new_recipe
    assert "chillo" not in new_recipe
    # 'huevo' debe aparecer
    assert "huevo" in new_recipe


def test_case_preservation_for_capitalized_orphan():
    """El name está capitalizado 'Chillo'; el reemplazo debe usar 'Huevo'
    (capitalizado) — no 'huevo' minúscula que se vería raro en dish name."""
    plan = _make_plan(
        day_num=1,
        meal_name="Filete de Chillo Asado",
        ingredients=["3 claras de huevo"],
        recipe="Asar el Chillo al horno. Servir.",
    )
    error = (
        "Día 1, Filete de Chillo Asado: La receta indica 'pescado' "
        "pero no hay ningún ingrediente equivalente listado."
    )
    patched_count, _ = _auto_patch_recipe_forward_coherence(plan, [error])
    assert patched_count == 1
    new_name = plan["days"][0]["meals"][0]["name"]
    # El reemplazo en el name debe preservar capitalización inicial
    assert "Huevo" in new_name, f"Expected capitalized 'Huevo' in name, got: {new_name}"
    assert "Chillo" not in new_name


def test_idempotent_second_pass_is_noop():
    """Correr el patch dos veces sobre el mismo plan no debe romper nada.
    Tras el primer pase, el orphan ya no existe en el texto → segundo pase
    encuentra meal pero no hay nada que substituir → cambio mínimo (count
    puede ser 1 o 0; lo importante es que no haya excepciones ni texto
    corrupto)."""
    plan = _make_plan(
        day_num=1,
        meal_name="Filete de Chillo al Vapor",
        ingredients=["5 claras de huevo", "60g casabe"],
        recipe="Cocinar el pescado al vapor. Servir.",
    )
    error = (
        "Día 1, Filete de Chillo al Vapor: La receta indica 'pescado' "
        "pero no hay ningún ingrediente equivalente listado."
    )
    _auto_patch_recipe_forward_coherence(plan, [error])
    name_after_1 = plan["days"][0]["meals"][0]["name"]
    recipe_after_1 = plan["days"][0]["meals"][0]["recipe"]

    # Segundo pase: el texto ya no contiene 'pescado' ni 'chillo'.
    _auto_patch_recipe_forward_coherence(plan, [error])
    name_after_2 = plan["days"][0]["meals"][0]["name"]
    recipe_after_2 = plan["days"][0]["meals"][0]["recipe"]

    assert name_after_1 == name_after_2, (
        f"Idempotencia rota en name: '{name_after_1}' → '{name_after_2}'"
    )
    assert recipe_after_1 == recipe_after_2, (
        f"Idempotencia rota en recipe: '{recipe_after_1}' → '{recipe_after_2}'"
    )


def test_empty_errors_returns_zero_no_op():
    """Lista vacía de errores → return (0, []) sin tocar el plan."""
    plan = _make_plan(
        day_num=1,
        meal_name="Test",
        ingredients=["100g pollo"],
        recipe="Cook.",
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [])
    assert patched_count == 0
    assert unpatched == []
    # Plan intacto
    assert plan["days"][0]["meals"][0]["name"] == "Test"


def test_malformed_error_left_unpatched():
    """Un error que no matchea el regex se preserva en unpatched (no
    silenciamos errores que no entendemos)."""
    plan = _make_plan(day_num=1, meal_name="X", ingredients=["pollo"], recipe="x")
    bad_error = "Algún error que no es del formato forward"
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [bad_error])
    assert patched_count == 0
    assert unpatched == [bad_error]


def test_meal_not_found_left_unpatched():
    """Si el error apunta a un meal que no existe en el plan (race
    condition con un fix anterior que lo removió), no patcheamos."""
    plan = _make_plan(day_num=1, meal_name="A", ingredients=["pollo"], recipe="x")
    error = (
        "Día 9, Plato Inexistente: La receta indica 'pescado' "
        "pero no hay ningún ingrediente equivalente listado."
    )
    patched_count, unpatched = _auto_patch_recipe_forward_coherence(plan, [error])
    assert patched_count == 0
    assert unpatched == [error]
