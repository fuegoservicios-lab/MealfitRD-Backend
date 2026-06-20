"""[P3-CARB-TO-PROTEIN-SWAP · 2026-06-19] Ancla la extracción del motor de sizing (`_apply_macro_engine`)
y el swap carbo→proteína kcal-neutral. Parser-based (sin DB/LLM) — falla si un renombre o borrado rompe el
contrato antes de tocar producción. La validación NUMÉRICA del swap (proteína +5.6pt w10, carbos +3.7pt,
kcal-neutral) se hizo con el harness determinista `scripts/macro_sizing_replay.py` sobre corpus de planes
crudos (no en CI: necesita master_ingredients + corpus grabado). Aquí anclamos la ESTRUCTURA."""
import ast
import os
import re

import pytest

_SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_orchestrator.py")
with open(_SRC_PATH, encoding="utf-8") as _f:
    SRC = _f.read()
TREE = ast.parse(SRC)
_FUNCS = {n.name: n for n in ast.walk(TREE) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _body_src(name):
    node = _FUNCS[name]
    return ast.get_source_segment(SRC, node) or ""


def test_macro_engine_extracted_as_callable():
    """El motor de sizing determinista es una función LLAMABLE (no inline en assemble) → habilita el harness
    offline + el swap. Behavior-preserving: assemble la invoca con los mismos locals."""
    assert "_apply_macro_engine" in _FUNCS, "se eliminó _apply_macro_engine (rompe harness + extracción)"
    assert "assemble_plan_node" in _FUNCS
    assemble = _body_src("assemble_plan_node")
    assert "_apply_macro_engine(result, days, skeleton, _daily_cals, _pg, _cg, _fg, form_data, nutrition)" in assemble, \
        "assemble ya no llama _apply_macro_engine con la firma esperada"
    # El motor corre la capa clínica (los 8 guards) — sentinel de que el bloque se movió COMPLETO
    assert "_apply_deterministic_clinical_layer(result, form_data, nutrition)" in _body_src("_apply_macro_engine")


def test_capture_hook_gated_and_failsafe():
    """El hook de captura de planes crudos es GATEADO por env (no-op en prod) y fail-safe."""
    assemble = _body_src("assemble_plan_node")
    assert 'os.environ.get("MEALFIT_MACRO_CAPTURE")' in assemble, "el hook MACRO_CAPTURE no está gateado por env"
    assert "MACRO-CAPTURE" in assemble


def test_swap_function_exists_and_kcal_neutral():
    """El swap carbo→proteína existe y mantiene kcal CONSTANTE (la razón por la que no lo deshace el reconcile):
    quita kcal_add/4 gramos de carbos = exactamente las kcal que aporta la proteína añadida."""
    assert "_swap_excess_carbs_to_protein_for_day" in _FUNCS, "se eliminó el swap carbo→proteína"
    body = _body_src("_swap_excess_carbs_to_protein_for_day")
    # invariante kcal-neutral: remueve kcal_add/4 de carbos
    assert "kcal_add" in body and "/ 4" in body.replace("/4", "/ 4"), "se perdió la lógica kcal-neutral del swap"
    assert "_trim_day_carbs_to_target" in body, "el swap ya no reusa el carb-trim para quitar carbos"
    # solo aplica a días deficitarios-de-proteína Y carbo-pesados (ambos)
    assert "protein_gap" in body and "carb_excess" in body


def test_swap_wired_gated_and_skips_renal():
    """El swap se invoca tras la capa clínica, gateado OFF por default, y SKIP si el cap renal aplica
    (subir proteína violaría el cap iatrogénico KDIGO)."""
    engine = _body_src("_apply_macro_engine")
    assert "CARB_TO_PROTEIN_SWAP_ENABLED" in engine, "el swap no está gateado por el knob en el motor"
    assert "_swap_excess_carbs_to_protein_for_day" in engine, "el motor no invoca el swap"
    assert "renal_protein_cap" in engine and "_renal_capped" in engine, "el swap no hace skip del cap renal"


def test_swap_knobs_safe_defaults():
    """Knobs auto-registrables con defaults SEGUROS: enabled OFF (validar por A/B antes de flip), floor 1.0."""
    assert re.search(r'CARB_TO_PROTEIN_SWAP_ENABLED\s*=\s*_env_bool\("MEALFIT_CARB_TO_PROTEIN_SWAP",\s*False\)', SRC), \
        "el swap NO está OFF por default (debe validarse por A/B antes de encender)"
    assert re.search(r'CARB_TO_PROTEIN_SWAP_FLOOR_PCT\s*=\s*_env_float\([^)]*,\s*1\.0\)', SRC), \
        "el floor del swap no es 1.0 (el A/B del harness mostró f1.0 > f0.95)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
