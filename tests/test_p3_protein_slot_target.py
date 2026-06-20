"""[P3-PROTEIN-SLOT-TARGET · 2026-06-19] Ancla la inyección del target de proteína EN GRAMOS por comida
al prompt del día. El LLM under-delivers proteína en crudo (~85% del target); darle el número por slot
sube el arranque (soft; el motor/swap/piso son backstop). Functional — `build_day_assignment_context` es
string-building puro (sin DB/LLM)."""
import os
import re

import pytest

from prompts.day_generator import build_day_assignment_context

_SKEL = {"protein_pool": ["Pechuga de pollo", "Huevos"], "carb_pool": ["Arroz"],
         "fruit_pool": ["Lechosa"], "meal_types": ["Desayuno", "Almuerzo", "Merienda", "Cena"],
         "brief_concept": "Criollo"}


def test_block_present_with_target_and_grams_sum():
    ctx = build_day_assignment_context(_SKEL, 1, day_name="Lunes", daily_protein_g=150)
    assert "PROTEÍNA POR COMIDA" in ctx, "no inyectó el bloque de target de proteína por slot"
    # gramos por comida presentes + razonables (4 comidas: 20/35/15/30% → 30/52/22/45)
    grams = [int(g) for g in re.findall(r"~(\d+)g", ctx)]
    # el primer set de ~Ng corresponde a los slots (Desayuno/Almuerzo/Merienda/Cena)
    slot_grams = grams[:4]
    assert sum(slot_grams) == pytest.approx(150, abs=4), f"los slots ({slot_grams}) no suman ~150g"
    assert max(slot_grams) > min(slot_grams), "los slots deben estar diferenciados (almuerzo > merienda)"
    # referencias de densidad para que el LLM sepa traducir gramos→porción
    assert "31g" in ctx and "pollo" in ctx.lower(), "falta la referencia de densidad proteica"


def test_no_block_without_target_backward_compatible():
    """Sin daily_protein_g (o knob off → None) NO añade el bloque — backward-compatible con los otros callsites."""
    ctx = build_day_assignment_context(_SKEL, 1)
    assert "PROTEÍNA POR COMIDA" not in ctx
    ctx_none = build_day_assignment_context(_SKEL, 1, daily_protein_g=None)
    assert "PROTEÍNA POR COMIDA" not in ctx_none


def test_three_meal_split():
    """Para 3 comidas (30/40/30%) el almuerzo es el mayor."""
    sk3 = dict(_SKEL, meal_types=["Desayuno", "Almuerzo", "Cena"])
    ctx = build_day_assignment_context(sk3, 1, daily_protein_g=120)
    grams = [int(g) for g in re.findall(r"~(\d+)g", ctx)][:3]
    assert sum(grams) == pytest.approx(120, abs=4)
    assert grams[1] == max(grams), "el almuerzo (40%) debe ser el slot con más proteína"


def test_knob_gates_the_pass_in_orchestrator():
    """El knob PROTEIN_SLOT_TARGET_PROMPT_ENABLED gatea el pass (rollback sin redeploy)."""
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_orchestrator.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert re.search(r'PROTEIN_SLOT_TARGET_PROMPT_ENABLED\s*=\s*_env_bool\("MEALFIT_PROTEIN_SLOT_TARGET_PROMPT",\s*True\)', src), \
        "el knob del slot-target no existe o no tiene default True"
    assert "if PROTEIN_SLOT_TARGET_PROMPT_ENABLED" in src, "el pass de daily_protein_g no está gateado por el knob"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
