"""[2026-06-13] Fixes de reliability descubiertos en test E2E live:
- P2-MEDICAL-FACTCHECK-GATE: el gate del fact-checking médico filtra el sentinel "Ninguna".
- P2-ANTI-REPETITION-TOLERANCE: tolera N platos repetidos antes de rechazar+reintentar.
- P2-PERSIST-NAN-GUARD: `_meal_macro_num` coacciona NaN/Inf → 0.0 (no propaga al INSERT).
"""
import os

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GO = os.path.join(BACKEND, "graph_orchestrator.py")


@pytest.fixture(scope="module")
def go_src():
    with open(GO, encoding="utf-8") as fh:
        return fh.read()


# ───────────── P2-MEDICAL-FACTCHECK-GATE (parser + unit) ─────────────
def test_factcheck_gate_uses_real_flags_helper(go_src):
    # El gate del fact-checking ya NO es `if allergies or medical_conditions` crudo.
    assert "_has_real_medical_flags(allergies) or _has_real_medical_flags(medical_conditions)" in go_src
    # El bypass del reviewer también usa el helper.
    assert "not _has_real_medical_flags(allergies) and not _has_real_medical_flags(medical_conditions)" in go_src


def test_has_real_medical_flags_filters_ninguna():
    from graph_orchestrator import _has_real_medical_flags
    assert _has_real_medical_flags(["Ninguna"]) is False
    assert _has_real_medical_flags(["Ninguno"]) is False
    assert _has_real_medical_flags([]) is False
    assert _has_real_medical_flags("") is False
    assert _has_real_medical_flags(["Maní"]) is True
    assert _has_real_medical_flags(["Ninguna", "Diabetes"]) is True  # condición real presente
    assert _has_real_medical_flags("Hipertensión") is True


# ───────────── P2-ANTI-REPETITION-TOLERANCE ─────────────
def test_anti_repetition_tolerance_knob(go_src):
    assert 'ANTI_REPETITION_TOLERANCE = _env_int("MEALFIT_ANTI_REPETITION_TOLERANCE", 2)' in go_src
    # El check usa la tolerancia, NO `> 0`.
    assert "len(filtered_repeated) > ANTI_REPETITION_TOLERANCE" in go_src


def test_anti_repetition_default_is_two():
    import graph_orchestrator
    assert graph_orchestrator.ANTI_REPETITION_TOLERANCE == 2


# ───────────── P2-PERSIST-NAN-GUARD (en _meal_macro_num) ─────────────
def test_meal_macro_num_coerces_nonfinite():
    from graph_orchestrator import _meal_macro_num
    assert _meal_macro_num(float("nan")) == 0.0
    assert _meal_macro_num(float("inf")) == 0.0
    assert _meal_macro_num(float("-inf")) == 0.0
    assert _meal_macro_num("154g") == 154.0
    assert _meal_macro_num("464 kcal") == 464.0
    assert _meal_macro_num(None) == 0.0
    assert _meal_macro_num(30) == 30.0
