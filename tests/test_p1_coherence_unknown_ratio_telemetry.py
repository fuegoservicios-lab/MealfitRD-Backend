"""[P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY · 2026-07-08] Distribución de ratios del bucket 'unknown'.

Forense plan vivo 70f802ec (2026-07-08): el coherence guard reportó `Hipótesis: {'unknown': 32}` — 32
divergencias de magnitud que el clasificador no pudo categorizar. `_classify_divergence_hypothesis` cae a
'unknown' cuando la magnitud no encaja en yield/unit_mismatch/pantry_overdeduct, y el propio código (P3-NEW-5)
advierte: NO añadir categorías nuevas sin evidencia forense de la FORMA de esos ratios.

Fix (telemetría, NO toca el gate): `_bucket_unknown_magnitude_ratios` bucketiza act/expected de las
divergencias 'unknown' de magnitud → el operador VE la distribución (¿sub-oferta 0.5-0.9? ¿over 2-4×?) y
decide con datos qué categoría añadir. Se emite al log del guard y se persiste en block-history.
Knob `MEALFIT_COHERENCE_UNKNOWN_RATIO_TELEMETRY` (default True).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "shopping_calculator.py"), encoding="utf-8") as f:
    _SC = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-COHERENCE-UNKNOWN-RATIO-TELEMETRY" in _SC


def test_helper_defined():
    assert "def _bucket_unknown_magnitude_ratios(" in _SC


def test_knob_present():
    assert "MEALFIT_COHERENCE_UNKNOWN_RATIO_TELEMETRY" in _SC


def test_helper_wired_into_guard_log():
    """El helper se LLAMA (no solo se define) dentro del guard, junto al log de hipótesis."""
    i_log = _SC.index("Hipótesis: {dict(by_hyp)}")
    window = _SC[i_log - 2000:i_log + 200]
    assert "_bucket_unknown_magnitude_ratios(divergences)" in window, \
        "el bucketing debe alimentar el log del guard"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def sc():
    import shopping_calculator as _sc
    return _sc


def _div(hyp, exp, act, magnitude=True):
    return {"food": "x", "side": "magnitude" if magnitude else "expected_only",
            "hypothesis": hyp, "magnitude": magnitude,
            "expected_qty": exp, "actual_qty": act}


def test_buckets_undersupply_and_oversupply(sc):
    divs = [
        _div("unknown", 100, 40),    # 0.4 → <0.5
        _div("unknown", 100, 70),    # 0.7 → 0.5-0.9
        _div("unknown", 100, 120),   # 1.2 → 1.1-1.5
        _div("unknown", 100, 300),   # 3.0 → 2-4
        _div("unknown", 100, 500),   # 5.0 → >=4
    ]
    out = sc._bucket_unknown_magnitude_ratios(divs)
    assert out.get("<0.5") == 1
    assert out.get("0.5-0.9") == 1
    assert out.get("1.1-1.5") == 1
    assert out.get("2-4") == 1
    assert out.get(">=4") == 1


def test_ignores_non_unknown_and_presence(sc):
    divs = [
        _div("yield_uncovered", 100, 135),      # no unknown
        _div("cap_swallowed_modifier", 100, 0, magnitude=False),  # presence, no qty real
        {"food": "y", "hypothesis": "unknown", "magnitude": False},  # presence unknown, sin qty
    ]
    assert sc._bucket_unknown_magnitude_ratios(divs) == {}


def test_empty_buckets_are_omitted(sc):
    divs = [_div("unknown", 100, 70)]  # solo 0.5-0.9
    out = sc._bucket_unknown_magnitude_ratios(divs)
    assert out == {"0.5-0.9": 1}


def test_fail_safe_on_garbage(sc):
    assert sc._bucket_unknown_magnitude_ratios(None) == {}
    assert sc._bucket_unknown_magnitude_ratios([None, 42, "x"]) == {}
    assert sc._bucket_unknown_magnitude_ratios([_div("unknown", 0, 50)]) == {}  # exp<=0 skip
