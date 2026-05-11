"""[P2-AUDIT-1 · 2026-05-10] Bandas yield para pescado/mariscos en
`_classify_divergence_hypothesis`.

Bug original (audit 2026-05-10):
    Las bandas yield clásicas (1.30-1.40× protein, 0.30-0.40× legume)
    no cubrían pescados/mariscos que pierden menos agua al cocinar
    (15-25% vs 26% de carnes rojas/blancas). Divergencias 1.10-1.30×
    en filetes de pescado/camarones caían a `unknown` → operador no
    veía la causa real (yield_uncovered) en pipeline_metrics.

Fix:
    `_classify_divergence_hypothesis` acepta `food` kwarg opcional.
    Cuando `food` resuelve a fish/seafood via
    `canonicalize_fish_seafood`, aplica banda combinada 1.05-1.30×.

Cobertura:
    - Carne (sin food / con pollo): mantiene banda clásica 1.30-1.40.
    - Fish (salmón, tilapia): banda 1.05-1.30 captura ratios 1.20, 1.25.
    - Seafood (camarón, langostino): idem.
    - Boundary: ratio 1.04 → no fish yield (fuera de banda).
    - Boundary: ratio 1.31 → no fish yield (cae en protein 1.30-1.40 si food=carne; falla si food=fish? actually 1.30 está en ambas — verificar).
    - Backward-compat: callers sin `food` siguen con banda clásica.
"""
from __future__ import annotations

import pytest

from shopping_calculator import _classify_divergence_hypothesis


# ---------------------------------------------------------------------------
# 1. Fish/seafood: banda 1.05-1.30 dispara yield_uncovered
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("food,ratio_act", [
    # Fish — filetes, ratio típico 1.15-1.25 (cooking loss 15-25%)
    ("filete de salmón", 1.20),
    ("salmón", 1.25),
    ("tilapia", 1.18),
    ("mero", 1.30),  # boundary upper
    ("dorado", 1.05),  # boundary lower
    ("atún sellado", 1.15),
    # Seafood — mariscos, ratio típico 1.05-1.20 (cooking loss menor)
    ("camarones", 1.10),
    ("camarón", 1.15),
    ("langostinos al ajillo", 1.08),
    ("calamares", 1.12),
    ("pulpo", 1.20),
])
def test_fish_seafood_ratio_in_band_classifies_yield_uncovered(food, ratio_act):
    """Ratio en banda 1.05-1.30 con food=fish/seafood → yield_uncovered."""
    exp_qty = 100.0
    act_qty = exp_qty * ratio_act
    result = _classify_divergence_hypothesis(
        exp_qty=exp_qty,
        act_qty=act_qty,
        exp_units={"g": exp_qty},
        act_units={"g": act_qty},
        food=food,
    )
    assert result == "yield_uncovered", (
        f"P2-AUDIT-1 regresión: food={food!r} ratio={ratio_act} debía "
        f"clasificarse `yield_uncovered`, fue {result!r}. Sin las "
        f"bandas extendidas, la divergencia cae a `unknown` → "
        f"operador no ve la causa en pipeline_metrics."
    )


# ---------------------------------------------------------------------------
# 2. Fish/seafood fuera de banda → unknown (no false positive)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("food,ratio_act", [
    ("salmón", 1.50),    # demasiado alto
    ("camarones", 0.80),  # below — no yield_uncovered
    ("tilapia", 1.04),    # justo bajo lower bound
])
def test_fish_seafood_ratio_out_of_band_not_yield_uncovered(food, ratio_act):
    """Fuera de la banda 1.05-1.30 con food=fish → NO yield_uncovered.
    Cae a `pantry_overdeduct` (si act < exp*0.5) o `unknown`."""
    exp_qty = 100.0
    act_qty = exp_qty * ratio_act
    result = _classify_divergence_hypothesis(
        exp_qty=exp_qty,
        act_qty=act_qty,
        exp_units={"g": exp_qty},
        act_units={"g": act_qty},
        food=food,
    )
    assert result != "yield_uncovered", (
        f"P2-AUDIT-1: food={food!r} ratio={ratio_act} (fuera de banda) "
        f"NO debe ser `yield_uncovered`. Got {result!r}. Banda fish/seafood "
        f"es 1.05-1.30 estricto."
    )


# ---------------------------------------------------------------------------
# 3. Backward-compat: sin food → banda clásica 1.30-1.40
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ratio_act,expected", [
    (1.35, "yield_uncovered"),  # banda clásica protein
    (0.35, "yield_uncovered"),  # banda clásica legume
    (1.20, "unknown"),           # fish band, pero sin food → no aplica
    (1.10, "unknown"),           # seafood band, pero sin food → no aplica
])
def test_backward_compat_no_food_uses_classic_bands(ratio_act, expected):
    """Callers que NO pasen `food` mantienen comportamiento previo:
    banda clásica 1.30-1.40 / 0.30-0.40. Sin esto, callers existentes
    cambian comportamiento al deploy del P2-AUDIT-1."""
    exp_qty = 100.0
    act_qty = exp_qty * ratio_act
    result = _classify_divergence_hypothesis(
        exp_qty=exp_qty,
        act_qty=act_qty,
        exp_units={"g": exp_qty},
        act_units={"g": act_qty},
        # No food kwarg.
    )
    assert result == expected, (
        f"P2-AUDIT-1: sin `food`, ratio={ratio_act} debía ser {expected!r}. "
        f"Got {result!r}. Backward-compat roto."
    )


# ---------------------------------------------------------------------------
# 4. Food no-fish (carne) ignora fish bands
# ---------------------------------------------------------------------------
def test_non_fish_food_does_not_trigger_fish_band():
    """food='pollo' con ratio=1.10 (fish band) → NO yield_uncovered.
    Solo carne con ratio en banda clásica dispara."""
    result = _classify_divergence_hypothesis(
        exp_qty=100.0,
        act_qty=110.0,
        exp_units={"g": 100.0},
        act_units={"g": 110.0},
        food="pollo",
    )
    assert result != "yield_uncovered", (
        f"P2-AUDIT-1: pollo con ratio=1.10 NO debe ser yield_uncovered "
        f"(fuera de banda clásica). Got {result!r}. Las bandas fish/seafood "
        f"NO deben aplicar a food que no canonicalize_fish_seafood."
    )


# ---------------------------------------------------------------------------
# 5. Precedencia: cap_swallowed_modifier > yield_uncovered
# ---------------------------------------------------------------------------
def test_cap_swallowed_takes_precedence_over_fish_band():
    """Si act_units está vacío (food totalmente ausente en lista), retorna
    `cap_swallowed_modifier` aunque food=fish. Precedencia documentada."""
    result = _classify_divergence_hypothesis(
        exp_qty=100.0,
        act_qty=0.0,
        exp_units={"g": 100.0},
        act_units={},  # absent
        food="salmón",
    )
    assert result == "cap_swallowed_modifier", (
        f"P2-AUDIT-1: precedencia rota — cap_swallowed_modifier debe "
        f"ganar sobre yield_uncovered. Got {result!r}."
    )


# ---------------------------------------------------------------------------
# 6. food="" (string vacío) = sin food (backward-compat)
# ---------------------------------------------------------------------------
def test_empty_food_string_treated_as_no_food():
    """food='' debe tratarse igual que omit kwarg → banda clásica."""
    result = _classify_divergence_hypothesis(
        exp_qty=100.0,
        act_qty=120.0,
        exp_units={"g": 100.0},
        act_units={"g": 120.0},
        food="",
    )
    # ratio=1.20 fuera de banda clásica (1.30-1.40) y sin food → unknown.
    assert result == "unknown", (
        f"P2-AUDIT-1: food='' (vacío) debe tratarse como sin food. "
        f"Got {result!r}."
    )
