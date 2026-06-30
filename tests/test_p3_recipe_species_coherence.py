"""[P3-RECIPE-SPECIES-COHERENCE · 2026-06-24] Los prompts de swap y modify deben instruir al LLM a NO
renombrar un ingrediente genérico ('Filete de pescado blanco') a una especie/corte específico ('dorado',
'mero', etc.) en los pasos de la receta — eso disparaba el guard de coherencia receta↔lista y, vía
retries agotados, contribuía a abrir el circuit breaker (incidente 2026-06-24, ver
P2-CB-GUARDRAIL-NOT-FAILURE). Esta es la mitigación 'en origen' (opción a): reducir las divergencias en
vez de solo detectarlas post-hoc.

Parser-based + smoke de .format() (los templates llevan placeholders {..}).
"""
import importlib.util
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mo():
    spec = importlib.util.spec_from_file_location(
        "meal_operations_under_test", os.path.join(BACKEND, "prompts", "meal_operations.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MO = _load_mo()


def test_swap_template_has_species_coherence_rule():
    t = MO.SWAP_MEAL_PROMPT_TEMPLATE
    assert "COHERENCIA RECETA" in t
    assert "dorado" in t and "mero" in t, "debe nombrar el modo de fallo (renombrar a especie específica)"
    assert "`ingredients`" in t


def test_modify_template_has_species_coherence_rule():
    t = MO.MODIFY_MEAL_PROMPT_TEMPLATE
    assert "COHERENCIA RECETA" in t
    assert "dorado" in t


def test_templates_still_format_cleanly():
    # No debe haber llaves sueltas que rompan .format() — smoke con todos los placeholders.
    MO.SWAP_MEAL_PROMPT_TEMPLATE.format(
        rejected_meal="X", meal_type="Almuerzo", target_calories=500, target_protein=30,
        target_carbs=40, target_fats=15, diet_type="balanced", context_extras="")
    MO.MODIFY_MEAL_PROMPT_TEMPLATE.format(
        name="X", desc="d", meal="Almuerzo", time="1pm", original_cals=500, original_protein=30,
        original_carbs=40, original_fats=15, ingredients_json="[]", changes="c", context_extras="")


def test_last_known_pfix_bumped():
    # [P1-OBJECTIVE-LEVERS-ON · 2026-06-29] NO hardcodear un marker específico: quedaba stale en cada bump
    # posterior (este test ya fallaba tras P2-REGEN-DAY-BIOMETRICS-PROPAGATE). El SSOT del valor del marker es
    # test_p3_1_last_known_pfix_freshness; acá solo verificamos que el marker existe y está fechado.
    with open(os.path.join(BACKEND, "app.py"), encoding="utf-8") as f:
        src = f.read()
    assert '_LAST_KNOWN_PFIX = "' in src and "· 20" in src, \
        "app.py debe tener _LAST_KNOWN_PFIX fechado (SSOT: test_p3_1_last_known_pfix_freshness)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
