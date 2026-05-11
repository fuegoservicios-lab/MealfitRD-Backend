"""[P1-NEW-1 · 2026-05-10] Lock-the-contract: default de
`MEALFIT_SHOPPING_COHERENCE_GUARD` es "block".

Bug del plan original (pre-P1-NEW-1):
  Default era "warn" → producción detectaba divergencias críticas (ej.
  cap_swallowed_modifier: pollo en receta, ausente en lista) pero solo
  las logueaba. El usuario recibía un plan incoherente sin fricción.

Fix (P1-NEW-1):
  Bumpear default a "block" en `_get_coherence_guard_mode`
  (shopping_calculator.py). `review_plan_node` ya consumía el flag
  `_shopping_coherence_block` (cierre P1-G + P2-A) — solo faltaba que
  el guard lo escribiera por default.

Rollback operacional:
  `export MEALFIT_SHOPPING_COHERENCE_GUARD=warn` revierte en caliente
  sin redeploy (knob releído cada invocación).

Este test bloquea regresión del default y verifica que un setenv
explícito sigue funcionando como override.
"""

import os
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    yield


def _fresh_import():
    """Re-importa `shopping_calculator` para que el knob registry se popule
    contra el `os.environ` actual. El registry cachea el snapshot al primer
    acceso al helper en cada proceso."""
    import importlib
    import shopping_calculator as sc
    return importlib.reload(sc)


def test_default_is_block_when_env_unset():
    """Sin env var, el guard mode debe ser "block" (cierre P1-NEW-1)."""
    sc = _fresh_import()
    assert sc._get_coherence_guard_mode() == "block"


def test_explicit_warn_overrides_default():
    """`export MEALFIT_SHOPPING_COHERENCE_GUARD=warn` revierte sin redeploy."""
    os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"] = "warn"
    try:
        sc = _fresh_import()
        assert sc._get_coherence_guard_mode() == "warn"
    finally:
        del os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"]


def test_explicit_off_overrides_default():
    os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"] = "off"
    try:
        sc = _fresh_import()
        assert sc._get_coherence_guard_mode() == "off"
    finally:
        del os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"]


def test_invalid_choice_falls_back_to_block():
    """Cualquier valor inválido cae al default (block, post-P1-NEW-1)."""
    os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"] = "bogus_value_xyz"
    try:
        sc = _fresh_import()
        assert sc._get_coherence_guard_mode() == "block"
    finally:
        del os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"]


def test_block_default_sets_flag_on_critical_divergence(monkeypatch):
    """E2E: con default "block" (env unset) + divergencia crítica de
    magnitudes, el flag `_shopping_coherence_block` queda seteado en el
    plan — lo consumirá `review_plan_node` (cierre P1-G/P2-A)."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_GUARD", raising=False)
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT", raising=False)
    from shopping_calculator import run_shopping_coherence_guard

    plan = {
        "days": [
            {"meals": [{"meal": "almuerzo", "ingredients_raw": ["1000 g arroz"]}]}
        ],
        # Lista trae 200g cuando receta pide 1000g → magnitude crítica.
        "aggregated_shopping_list": [
            {"name": "Arroz", "market_qty_numeric": 200, "market_unit": "g"}
        ],
    }
    run_shopping_coherence_guard(plan, multiplier=1.0)
    assert "_shopping_coherence_block" in plan, (
        "Default debe ser block: un plan con divergencia crítica de "
        "magnitudes debe quedar con flag _shopping_coherence_block para "
        "que review_plan_node lo procese."
    )
    assert any(d.get("magnitude") for d in plan["_shopping_coherence_block"])
