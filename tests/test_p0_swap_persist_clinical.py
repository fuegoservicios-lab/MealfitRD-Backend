"""[P0-SWAP-PERSIST-CLINICAL · 2026-07-01] (audit P0-3 · frontera de confianza clínica)

`/swap-meal/persist` era la ÚNICA superficie de update con round-trip CLIENTE entre la generación
validada y la persistencia: el body `new_meal` se escribía verbatim (solo bounds/nombre/tamaño) →
un cliente buggy/stale o una llamada API directa podía persistir un plato JAMÁS validado (incl. un
alérgeno IgE declarado) en un plan que S1 sí validó. Chat-modify y regen-day persisten server-side;
esta era la celda que faltaba de la matriz de paridad.

Fix: re-validación DETERMINISTA server-side en el handler con el MISMO SSOT del resto de updates
(`clinical_backstop_for_meal`: alérgenos C2 + dieta + mercurio-embarazo) contra el PERFIL hidratado
server-side (allergies + dietType desde health_profile — nunca del body). Violación → HTTP 422 y el
plato original (ya validado) se preserva client-side. Knob de rollback sin redeploy:
`MEALFIT_SWAP_PERSIST_CLINICAL_GUARD` (default ON).

Tests: (1) parser-based del wiring/orden en el handler; (2) funcional del SSOT del backstop.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m, f"No se encontró `def {fn_name}(` en plans.py"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def persist_body() -> str:
    return _extract_function_body(_PLANS, "api_swap_meal_persist")


# ---------------------------------------------------------------------------
# 1. Parser-based: knob + backstop + orden + fuente server-side + 422
# ---------------------------------------------------------------------------
def test_knob_present(persist_body):
    assert 'os.environ.get("MEALFIT_SWAP_PERSIST_CLINICAL_GUARD", "true")' in persist_body, \
        "falta el knob de rollback MEALFIT_SWAP_PERSIST_CLINICAL_GUARD con default ON"


def test_backstop_called_before_mutator(persist_body):
    """El backstop clínico debe correr ANTES de definir/invocar `_swap_mutator` — si corre
    después del UPDATE atómico, el alérgeno ya se persistió."""
    i_backstop = persist_body.find("clinical_backstop_for_meal")
    i_mutator = persist_body.find("def _swap_mutator")
    assert i_backstop != -1, "el handler no invoca clinical_backstop_for_meal (P0-SWAP-PERSIST-CLINICAL)"
    assert i_mutator != -1, "refactor inesperado: no existe _swap_mutator"
    assert i_backstop < i_mutator, \
        "el backstop clínico debe correr ANTES del mutator/persist (fail-secure pre-write)"


def test_backstop_scans_new_meal_with_profile_data(persist_body):
    """El backstop escanea el `new_meal` del body contra allergies/diet del PERFIL
    (hidratación server-side, espejo de I2 / P0-UPDATE-CLINICAL-GUARD)."""
    m = re.search(
        r"clinical_backstop_for_meal.*?\(\s*new_meal\s*,.*?allergies\s*=\s*_persist_allergies"
        r".*?diet_type\s*=\s*_persist_diet",
        persist_body, re.DOTALL,
    ) or re.search(
        r"_cbm_persist\(\s*new_meal\s*,\s*allergies\s*=\s*_persist_allergies",
        persist_body, re.DOTALL,
    )
    assert m, "el backstop no escanea new_meal con allergies/diet del perfil server-side"
    assert '_hp_micro.get("allergies")' in persist_body, \
        "las alergias deben nacer del health_profile server-side, no del body"


def test_violation_maps_to_422(persist_body):
    """Violación clínica → HTTP 422 (el plato original validado se preserva client-side)."""
    i_viols = persist_body.find("_persist_viols")
    assert i_viols != -1
    after = persist_body[i_viols:]
    assert "status_code=422" in after, \
        "una violación del backstop debe rechazar el persist con HTTP 422"


def test_marker_anchor_present(persist_body):
    assert "P0-SWAP-PERSIST-CLINICAL" in persist_body, "falta el tooltip-anchor en el handler"


# ---------------------------------------------------------------------------
# 2. Funcional: el SSOT del backstop detecta el alérgeno del body
# ---------------------------------------------------------------------------
def test_backstop_ssot_detects_allergen():
    """Sanity del SSOT reusado: un new_meal con camarones + alergia declarada → violación
    no-vacía (mismo comportamiento que el resto de updates)."""
    import graph_orchestrator as g
    viols = g.clinical_backstop_for_meal(
        {"name": "Camarones al ajillo", "ingredients": ["200g Camarones", "1 cda Aceite de oliva"],
         "recipe": ["Saltea los camarones."]},
        allergies=["camarones"], diet_type=None,
    )
    assert viols, "el backstop debe detectar el alérgeno declarado en new_meal"
    assert any("camar" in str(v).lower() for v in viols)
