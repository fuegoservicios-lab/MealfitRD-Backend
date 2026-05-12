"""[P1-NEW-10 · 2026-05-11] Conversor de unidades dentro del mismo sistema
para `compare_expected_vs_aggregated` (canary, default OFF).

Bug original (audit 2026-05-11):
    Sin conversor, `compare_expected_vs_aggregated` iteraba unidades por
    nombre literal. `{Arroz: {kg: 1.0}}` (receta) vs `{Arroz: {g: 1000.0}}`
    (lista) producía DOS divergencias falsas (fantasma kg + fantasma g)
    en lugar de cero. Hoy producción NO observa esto porque el LLM
    normaliza simétricamente, pero el guard era frágil ante prompt drift
    o cambio de modelo. Fix preventivo con knob canary.

Fix:
    1. `canonical_units.UNIT_TO_BASE_FACTOR` + `to_base_amount(qty, unit)`
       SSOT de conversión SEGURA (peso↔peso, volumen↔volumen, NO
       cross-system).
    2. `shopping_calculator._normalize_food_units_to_base(units_dict)`
       merge entries de un mismo alimento al base unit.
    3. Knob `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED` (default True
       post-P2-UNIT-CONV-1 · 2026-05-11; era False originalmente).
    4. `compare_expected_vs_aggregated` aplica la normalización a ambos
       dicts ANTES de iterar, si el knob está True.

Estrategia del test:
    A) Behavioral: `to_base_amount` produce los gramos/ml correctos.
    B) Behavioral: `_normalize_food_units_to_base` consolida correctamente.
    C) Behavioral: con knob ON, `{kg:1}` vs `{g:1000}` no produce drift.
    D) Behavioral: con knob OFF (default), comportamiento idéntico a v1.
    E) Behavioral: cross-system (taza vs g) NO se convierte (out-of-scope).
    F) Parser-based: el knob existe y default es False.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"

import sys
sys.path.insert(0, str(_BACKEND))


# ──────────────────────────────────────────────────────────────────────
# A) `to_base_amount` correctness
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("qty,unit,expected_qty,expected_unit", [
    # Peso
    (1.0, "kg", 1000.0, "g"),
    (500.0, "g", 500.0, "g"),
    (2.0, "lb", 907.184, "g"),
    (8.0, "oz", 226.796, "g"),
    # Volumen
    (1.0, "l", 1000.0, "ml"),
    (250.0, "ml", 250.0, "ml"),
    (2.0, "taza", 480.0, "ml"),
    (3.0, "cda", 45.0, "ml"),
    (1.0, "cdta", 5.0, "ml"),
    # Aliases (case + plural + tildes via canonical_unit)
    (1.0, "Kilo", 1000.0, "g"),
    (2.0, "kilos", 2000.0, "g"),
    (1.0, "libra", 453.592, "g"),
    (4.0, "cucharadas", 60.0, "ml"),
    # No convertibles: devuelven canonical unchanged.
    (3.0, "unidad", 3.0, "unidad"),
    (1.0, "diente", 1.0, "diente"),
    (0.5, "pizca", 0.5, "pizca"),
    # Desconocidos: devuelven raw.
    (1.0, "puff", 1.0, "puff"),
])
def test_to_base_amount(qty, unit, expected_qty, expected_unit):
    from canonical_units import to_base_amount
    out_qty, out_unit = to_base_amount(qty, unit)
    assert out_unit == expected_unit, (
        f"unit mismatch: {qty} {unit} → {out_unit} (expected {expected_unit})"
    )
    assert abs(out_qty - expected_qty) < 0.01, (
        f"qty mismatch: {qty} {unit} → {out_qty} (expected {expected_qty})"
    )


def test_to_base_amount_idempotent():
    """Aplicar dos veces produce el mismo resultado."""
    from canonical_units import to_base_amount
    q1, u1 = to_base_amount(1.0, "kg")
    q2, u2 = to_base_amount(q1, u1)
    assert (q1, u1) == (q2, u2), (
        f"P1-NEW-10 regresión: no idempotente. "
        f"to_base_amount(1, kg)={q1, u1}, applied again={q2, u2}"
    )


def test_to_base_amount_non_numeric_returns_raw():
    """Inputs no numéricos no crashean; devuelven raw."""
    from canonical_units import to_base_amount
    q, u = to_base_amount("bad", "kg")
    assert q == "bad" and u == "kg"


# ──────────────────────────────────────────────────────────────────────
# B) `_normalize_food_units_to_base` consolidation
# ──────────────────────────────────────────────────────────────────────


def test_normalize_merges_kg_and_g():
    from shopping_calculator import _normalize_food_units_to_base
    out = _normalize_food_units_to_base({"kg": 0.5, "g": 100})
    assert out == {"g": 600.0}, f"P1-NEW-10 regresión: {out}"


def test_normalize_merges_taza_cda_cdta():
    from shopping_calculator import _normalize_food_units_to_base
    out = _normalize_food_units_to_base({"taza": 1.0, "cda": 2.0, "cdta": 3.0})
    # 240 + 30 + 15 = 285
    assert out == {"ml": 285.0}, f"P1-NEW-10 regresión: {out}"


def test_normalize_preserves_cross_system_separation():
    """Volumen y peso NUNCA se mezclan sin densidad explícita."""
    from shopping_calculator import _normalize_food_units_to_base
    out = _normalize_food_units_to_base({"kg": 0.5, "ml": 200})
    assert out == {"g": 500.0, "ml": 200.0}, (
        f"P1-NEW-10 regresión: cross-system mezclado: {out}"
    )


def test_normalize_preserves_non_convertibles():
    from shopping_calculator import _normalize_food_units_to_base
    out = _normalize_food_units_to_base({"unidad": 3.0, "diente": 2.0})
    assert out == {"unidad": 3.0, "diente": 2.0}, f"P1-NEW-10 regresión: {out}"


def test_normalize_empty_dict():
    from shopping_calculator import _normalize_food_units_to_base
    assert _normalize_food_units_to_base({}) == {}
    assert _normalize_food_units_to_base(None) == {}


# ──────────────────────────────────────────────────────────────────────
# C+D) `compare_expected_vs_aggregated` behavior under knob
# ──────────────────────────────────────────────────────────────────────


def test_with_knob_on_kg_vs_g_no_drift(monkeypatch):
    """{kg: 1} vs {g: 1000} debe ser CERO drift bajo knob ON."""
    monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "true")
    from shopping_calculator import compare_expected_vs_aggregated
    expected = {"Arroz": {"kg": 1.0}}
    aggregated = {"Arroz": {"g": 1000.0}}
    divs = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    assert divs == [], (
        f"P1-NEW-10 regresión: con knob ON, kg vs g del mismo food debe ser "
        f"cero drift tras normalización. Got: {divs}"
    )


def test_with_knob_off_kg_vs_g_reports_drift(monkeypatch):
    """Default (knob OFF): comportamiento v1, kg y g se ven como distintos."""
    monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "false")
    from shopping_calculator import compare_expected_vs_aggregated
    expected = {"Arroz": {"kg": 1.0}}
    aggregated = {"Arroz": {"g": 1000.0}}
    divs = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    assert len(divs) >= 1, (
        "P1-NEW-10 regresión: con knob OFF (default canary) el comportamiento "
        "debe ser idéntico a v1. v1 reportaba drift en este caso."
    )


def test_with_knob_on_taza_vs_g_does_not_merge(monkeypatch):
    """Cross-system (volumen↔peso) NO se merge sin densidad. Sigue siendo drift."""
    monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "true")
    from shopping_calculator import compare_expected_vs_aggregated
    expected = {"Arroz": {"taza": 1.0}}
    aggregated = {"Arroz": {"g": 240.0}}
    divs = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    # Esperamos drift: taza→ml, g→g, son distintos sistemas.
    assert len(divs) >= 1, (
        "P1-NEW-10 regresión: cross-system NO debe convertirse sin densidad. "
        "El canary debe ser CONSERVADOR — preferir falso positivo a falso "
        "negativo silente."
    )


def test_with_knob_on_taza_vs_cda_merges(monkeypatch):
    """Same-system aliases (taza vs cda): merge a ml."""
    monkeypatch.setenv("MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED", "true")
    from shopping_calculator import compare_expected_vs_aggregated
    # Receta: 2 tazas (=480ml). Lista: 32 cdas (=480ml). Mismo total.
    expected = {"Aceite": {"taza": 2.0}}
    aggregated = {"Aceite": {"cda": 32.0}}
    divs = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    assert divs == [], (
        f"P1-NEW-10 regresión: taza y cda (mismo sistema volumen) deben "
        f"normalizarse a ml y producir cero drift. Got: {divs}"
    )


# ──────────────────────────────────────────────────────────────────────
# F) Parser-based: knob existe y default es False
# ──────────────────────────────────────────────────────────────────────


def test_knob_default_is_true_post_p2_unit_conv_1():
    """[P2-UNIT-CONV-1 · 2026-05-11] Default flipped a True.

    Originalmente (P1-NEW-10 · 2026-05-11) el knob era canary default False.
    P2-UNIT-CONV-1 lo flippeó a True tras audit MCP que confirmó:
      - 0 entries en `_shopping_coherence_block_history` en prod (3 planes
        total, todos abandoned).
      - Contrato del converter: solo unifica unidades del mismo sistema
        físico (peso↔g, volumen↔ml). NO cross-system (kg↔ml requiere
        densidad, no se hace).
      - Tests de matemática (B/C/D abajo) cubren los casos.
      - El "drift" detectado pre-fix era PURAMENTE false positive.

    Knob queda como kill switch invertido: setear
    `MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED=false` revierte sin redeploy.
    """
    src = (_BACKEND / "shopping_calculator.py").read_text(encoding="utf-8")
    import re
    match = re.search(
        r"_knob_env_bool\(\s*[\'\"]MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED[\'\"]\s*,\s*(\w+)",
        src,
    )
    assert match, (
        "P1-NEW-10/P2-UNIT-CONV-1 regresión: el knob "
        "MEALFIT_COHERENCE_UNIT_CONVERTER_ENABLED ya no se lee vía "
        "`_knob_env_bool`. Si se removió, el conversor pierde su kill switch."
    )
    assert match.group(1) == "True", (
        f"P2-UNIT-CONV-1 regresión: el default del knob es `{match.group(1)}`, "
        f"esperado `True` (flip de canary post-P2-UNIT-CONV-1). Si genuinamente "
        f"necesitas revertir el default a False, primero documentar la razón "
        f"en memoria + bumpear el marker; el knob `=false` solo es para "
        f"rollback operacional sin redeploy, no para cambiar el contrato."
    )


def test_unit_to_base_factor_table_complete():
    """La tabla UNIT_TO_BASE_FACTOR debe cubrir los aliases canónicos
    comunes (peso + volumen)."""
    from canonical_units import UNIT_TO_BASE_FACTOR
    required = {"g", "kg", "lb", "oz", "ml", "l", "taza", "cda", "cdta"}
    missing = required - set(UNIT_TO_BASE_FACTOR.keys())
    assert not missing, (
        f"P1-NEW-10 regresión: UNIT_TO_BASE_FACTOR no cubre {missing}. "
        f"Sin estos, el conversor no normalizará y el guard volverá a "
        f"reportar falsos positivos."
    )


def test_unit_to_base_factor_bases_are_canonical():
    """Las bases de la tabla deben ser 'g' o 'ml' (no se admiten otras
    bases — eso requeriría conversión cross-system)."""
    from canonical_units import UNIT_TO_BASE_FACTOR
    bases = {entry[0] for entry in UNIT_TO_BASE_FACTOR.values()}
    assert bases == {"g", "ml"}, (
        f"P1-NEW-10 regresión: bases inesperadas {bases - {'g', 'ml'}}. "
        "Si se añadió un nuevo sistema (ej. tiempo, energía), revisar "
        "que `compare_expected_vs_aggregated` no lo merge incorrectamente."
    )
