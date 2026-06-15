"""[gap-audit Batch C · 2026-06-15] G15 (dri age-aware) + G16 (validación sin filtro) + G13 (gate de no-regresión).

G15: `dri_targets` ahora es age-aware en hierro (mujer 18→8 mg a los 51+) y calcio (1000→1200 mg mujer 51+/
hombre 71+). G16: el export de validación imprime una 2ª línea de integridad SIN el filtro res_pct (no esconde
el 0-silencioso). G13: `benchmark_macro_compliance._assert_no_regression` compara vs baseline commiteado y
falla ante regresión de precisión (job nightly).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import micronutrients as mn

_BACKEND = Path(__file__).resolve().parent.parent
_BASELINE = _BACKEND / "tests" / "fixtures" / "macro_baseline.json"
_BENCH = _BACKEND / "scripts" / "benchmark_macro_compliance.py"
_VALIDATION_SRC = (_BACKEND / "scripts" / "clinical_validation_export.py").read_text(encoding="utf-8")


# ── G15: dri_targets age-aware ──
@pytest.mark.parametrize("sex,age,exp_iron,exp_calcium", [
    ("F", 30, 18.0, 1000.0),   # mujer joven: hierro alto, calcio base
    ("F", 60, 8.0, 1200.0),    # mujer post-menopausia: hierro baja, calcio sube
    ("M", 30, 8.0, 1000.0),    # hombre joven
    ("M", 75, 8.0, 1200.0),    # hombre 71+: calcio sube
    ("F", None, 18.0, 1000.0), # edad desconocida → conservador (hierro alto)
])
def test_dri_targets_age_aware(sex, age, exp_iron, exp_calcium):
    t = mn.dri_targets(sex, age)
    assert t["iron_mg"]["floor"] == exp_iron, f"hierro {sex}/{age}"
    assert t["calcium_mg"]["floor"] == exp_calcium, f"calcio {sex}/{age}"


def test_dri_targets_backward_compatible_sex_only():
    """Llamar sin age (firma vieja) sigue funcionando → valores de adulto joven."""
    t = mn.dri_targets("F")
    assert t["iron_mg"]["floor"] == 18.0 and t["calcium_mg"]["floor"] == 1000.0


def test_build_report_accepts_age():
    import inspect
    assert "age" in inspect.signature(mn.build_micronutrient_report).parameters, (
        "build_micronutrient_report debe aceptar `age` (G15)."
    )


# ── G16: 2ª línea de integridad sin filtro res_pct ──
def test_validation_export_has_unfiltered_integrity_line():
    assert "integ_all_cells" in _VALIDATION_SRC and "integ_all_band" in _VALIDATION_SRC, (
        "G16: falta el agregado de integridad SIN filtro res_pct."
    )
    assert "days_low_res" in _VALIDATION_SRC, "G16: falta el conteo de días de baja resolución."
    assert "SIN filtro res_pct" in _VALIDATION_SRC, "G16: falta la 2ª línea de resumen sin filtro."


# ── G13: baseline + gate de no-regresión ──
def test_baseline_json_valid_and_shaped():
    assert _BASELINE.exists(), "Falta backend/tests/fixtures/macro_baseline.json (G13)."
    base = json.loads(_BASELINE.read_text(encoding="utf-8"))
    rp = base["REAL_PLANS"]
    assert "all4_within_10pct_days_pct" in rp
    for mac in ("kcal", "protein", "carbs", "fats"):
        assert "mape_pct" in rp["per_macro"][mac]


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_mod_g13", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assert_no_regression_logic(tmp_path):
    bench = _load_bench()
    base = {"REAL_PLANS": {"all4_within_10pct_days_pct": 28.0,
                           "per_macro": {"protein": {"mape_pct": 7.0}, "carbs": {"mape_pct": 18.0}}}}
    bp = tmp_path / "baseline.json"
    bp.write_text(json.dumps(base), encoding="utf-8")

    # Sin regresión (banda sube, MAPE baja).
    ok, _ = bench._assert_no_regression(
        {"REAL_PLANS": {"all4_within_10pct_days_pct": 30.0,
                        "per_macro": {"protein": {"mape_pct": 6.0}, "carbs": {"mape_pct": 12.0}}}},
        str(bp), 5.0, 10.0)
    assert ok is True

    # Regresión por caída de all-4-en-banda (28→10 = caída 18 > 10).
    ok2, _ = bench._assert_no_regression(
        {"REAL_PLANS": {"all4_within_10pct_days_pct": 10.0,
                        "per_macro": {"protein": {"mape_pct": 6.0}}}},
        str(bp), 5.0, 10.0)
    assert ok2 is False

    # Regresión por subida de MAPE de proteína (7→15 = subió 8 > 5).
    ok3, _ = bench._assert_no_regression(
        {"REAL_PLANS": {"all4_within_10pct_days_pct": 28.0,
                        "per_macro": {"protein": {"mape_pct": 15.0}}}},
        str(bp), 5.0, 10.0)
    assert ok3 is False
