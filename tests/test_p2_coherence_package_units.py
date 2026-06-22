"""[P2-COHERENCE-PACKAGE-UNITS + P2-COHERENCE-TESTS-PACKAGE-UNITS · 2026-06-22] (audit fresco P2-15 + P2-14)

Falso-positivo del coherence guard por unidades de ENVASE no convertibles: cuando la receta pide un
alimento en una unidad convertible (g/ml/lb) pero `apply_smart_market_units` lo presenta en la lista como
envase (pote/frasco/Ud.), `compare_expected_vs_aggregated` veía act_qty=0 para la unidad esperada (el
conversor solo unifica g/ml/lb) → delta_pct=1.0 FINITO → entraba al subset crítico B de
`run_shopping_coherence_guard` → block + retry FALSO (costo LLM / degradación).

Fix: la divergencia se tagea `unit_mismatch=True` cuando el alimento SÍ está presente bajo otra unidad;
sigue como telemetría warn pero el crítico B la excluye (no se puede comparar magnitud entre "1 pote" y
"200 g"). Estos tests ejercitan unidades de envase reales (el gap que ocultaba el riesgo).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import shopping_calculator as sc

_SRC = Path(__file__).resolve().parent.parent.joinpath("shopping_calculator.py").read_text(encoding="utf-8")


# ─────────────────────────── A. Funcional: tag unit_mismatch ───────────────────────────

def test_package_unit_present_is_tagged_unit_mismatch():
    """Receta en 'g', lista en 'pote' (no convertible) → divergencia tagged unit_mismatch=True."""
    expected = {"Mantequilla de maní": {"g": 200.0}}
    aggregated = {"Mantequilla de maní": {"pote": 1.0}}
    divs = sc.compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    # La divergencia de la unidad esperada 'g' (act=0) debe existir y estar tagged.
    g_div = [d for d in divs if d["food"] == "Mantequilla de maní" and d["unit"] == "g"]
    assert g_div, ("debe reportar la divergencia de la unidad esperada", divs)
    d = g_div[0]
    assert d["delta_pct"] != float("inf") and d["expected_qty"] > 0, (
        "pre-fix esta divergencia (finita, expected>0) entraba al crítico B", d
    )
    assert d.get("unit_mismatch") is True, ("alimento presente bajo 'pote' → unit_mismatch", d)


def test_real_magnitude_divergence_not_tagged():
    """Misma unidad, mitad de cantidad = divergencia de magnitud REAL → NO unit_mismatch (sigue crítica)."""
    expected = {"Arroz": {"g": 1000.0}}
    aggregated = {"Arroz": {"g": 500.0}}
    divs = sc.compare_expected_vs_aggregated(expected, aggregated, tolerance=0.10)
    g_div = [d for d in divs if d["food"] == "Arroz" and d["unit"] == "g"]
    assert g_div, ("debe reportar la divergencia de magnitud real", divs)
    assert g_div[0].get("unit_mismatch") is not True, (
        "una divergencia de magnitud real (misma unidad) NO debe excluirse del crítico"
    )


def test_truly_missing_food_not_tagged_unit_mismatch():
    """Alimento ausente por completo de la lista (no en NINGUNA unidad) → NO unit_mismatch (sigue crítico)."""
    expected = {"Pollo": {"g": 300.0}}
    aggregated = {}  # el alimento no está en la lista en absoluto
    divs = sc.compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    p_div = [d for d in divs if d["food"] == "Pollo"]
    assert p_div, ("alimento de receta ausente de la lista debe reportarse", divs)
    assert p_div[0].get("unit_mismatch") is not True, (
        "un alimento genuinamente ausente NO es unit_mismatch (debe poder bloquear)"
    )


# ─────────────────────────── B. Parser-anchor: crítico B excluye unit_mismatch ───────────────────────────

def test_critical_filter_excludes_unit_mismatch():
    # El subset crítico B (el que dispara block) debe excluir las divergencias unit_mismatch.
    assert "not d.get(\"unit_mismatch\")" in _SRC, (
        "el filtro crítico B debe excluir `unit_mismatch` (si no, reabre el block falso por envases)"
    )
    assert _SRC.count("P2-COHERENCE-PACKAGE-UNITS") >= 2
