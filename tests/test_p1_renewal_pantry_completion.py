"""[P1-RENEWAL-PANTRY-AWARE · 2026-06-28 · Fase 2] Lista de FALTANTES para
"completar la nevera al 100%" (compute_pantry_completion_delta + endpoint).

Contrato:
  - compute_pantry_completion_delta = lo que el plan NECESITA menos lo que el
    usuario YA TIENE en la nevera (resta CUANTITATIVA real, reusa
    get_shopping_list_delta con is_new_plan=False).
  - Deduce SOLO la nevera (consumed_override=[]) — no 'consumidos' (irrelevante
    para un plan recién renovado).
  - READ-ONLY: el endpoint /recalculate-shopping-list expone la lista en la
    RESPUESTA pero NO la persiste ni toca aggregated_shopping_list_* (la lista
    canónica queda intacta). Gated OFF por default.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants  # noqa: E402
from shopping_calculator import compute_pantry_completion_delta  # noqa: E402

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PLAN = {"days": [{"day": 1, "meals": [
    {"ingredients_raw": ["1000 g de pollo", "500 g de arroz"]},
]}]}


def test_completion_knob_off_by_default():
    assert constants.PANTRY_COMPLETION_LIST_ENABLED is False


def test_guest_returns_empty():
    assert compute_pantry_completion_delta(None, _PLAN) in ([], {})
    assert compute_pantry_completion_delta("guest", _PLAN) in ([], {})


def test_empty_fridge_returns_needs():
    # Nevera vacía → la lista de faltantes = todo lo que el plan necesita (no vacía).
    out = compute_pantry_completion_delta(
        "u1", _PLAN, multiplier=1.0, inventory_override=[], categorize=False, structured=False
    )
    assert isinstance(out, list) and len(out) > 0
    joined = " ".join(str(x).lower() for x in out)
    assert "pollo" in joined


def test_full_fridge_reduces_list():
    # Nevera con MUCHO pollo y arroz → se deducen → la lista de faltantes es más
    # corta (o vacía) que con la nevera vacía. Demuestra la resta cuantitativa.
    big = [
        {"ingredient_name": "pollo", "quantity": 99999, "unit": "g"},
        {"ingredient_name": "arroz", "quantity": 99999, "unit": "g"},
    ]
    full = compute_pantry_completion_delta(
        "u1", _PLAN, multiplier=1.0, inventory_override=[], categorize=False, structured=False
    )
    delta = compute_pantry_completion_delta(
        "u1", _PLAN, multiplier=1.0, inventory_override=big, categorize=False, structured=False
    )
    assert len(delta) < len(full)


def test_function_marker_and_deducts_only_fridge():
    src = open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8").read()
    assert "def compute_pantry_completion_delta" in src
    assert "P1-RENEWAL-PANTRY-AWARE" in src
    # Reusa get_shopping_list_delta con is_new_plan=False + deduce SOLO la nevera.
    m = re.search(
        r"def compute_pantry_completion_delta.*?return get_shopping_list_delta\(.*?\)",
        src, re.S,
    )
    assert m is not None
    body = m.group(0)
    assert "is_new_plan=False" in body
    assert "consumed_override=[]" in body


def test_endpoint_gated_and_readonly():
    src = open(os.path.join(_BACKEND, "routers", "plans.py"), encoding="utf-8").read()
    # Gated por el knob:
    assert "PANTRY_COMPLETION_LIST_ENABLED" in src
    # Expone el campo en la respuesta del recalc:
    assert '"pantry_completion_list"' in src
    # READ-ONLY: la completion list NUNCA se escribe en plan_data / la canónica.
    assert 'merged_plan_data["pantry_completion_list"]' not in src
    assert 'plan_data_fresh["pantry_completion_list"]' not in src
    assert 'aggregated_shopping_list"] = pantry_completion' not in src
