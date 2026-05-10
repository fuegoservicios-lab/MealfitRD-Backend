"""[P2-4 · 2026-05-08] Tests del SSOT de exclusión de meals en agregación.

Bug original (audit 2026-05-07):
  3 sitios duplicaban inline `if "suplemento" in meal.get("meal", "").lower():
  continue`:
    1. `expected_sum_from_recipes` (línea ~2224, alimenta capa B del guard).
    2. `get_shopping_list_delta` (línea ~5091, aggregator principal).
    3. Extractor de facts/ingredients (línea ~5199).

  La premisa específica del audit ("snacks pueden quedar fuera del bucle")
  no se materializaba (los 3 sitios eran simétricos), pero existía riesgo
  de drift idéntico al que causó el bug `caps_asymmetry` en el pasado: si
  alguien añadía "infusión" en el aggregator pero olvidaba sincronizar con
  `expected_sum_from_recipes`, capa B reportaría divergencias falsas.

Fix:
  1. Helper `_should_skip_meal_for_aggregation(meal: dict) -> bool` como SSOT.
  2. Knob `MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS` (comma-separated, default
     `"suplemento"`) auto-registrado en `_KNOBS_REGISTRY` vía lazy import.
  3. Cache invalida cuando el env-var cambia.
  4. Los 3 call sites ahora invocan el helper — drift imposible.

Cobertura:
  - Default excluye solo "suplemento".
  - Override comma-separated añade keywords.
  - Match es substring (no exact) — "Suplemento de magnesio" matchea
    "suplemento" igual que antes.
  - Match es case-insensitive.
  - Meal sin nombre / no-dict se excluye defensivamente.
  - Cache se invalida cuando el env-var cambia entre invocaciones.
  - Smoke estructural: los 3 sitios usan el helper (no duplicados inline).
  - Knob aparece en _KNOBS_REGISTRY tras el primer call.
"""
import os
import pathlib
import re
import sys

import pytest


# [P1-A · 2026-05-08] Fix path tras P3-CANDIDATE-B (mv tests al subdir
# `tests/`); subir un nivel.
_SHOP_PATH = pathlib.Path(__file__).resolve().parent.parent / "shopping_calculator.py"


@pytest.fixture(autouse=True)
def _reset_env_and_cache():
    prev = os.environ.pop("MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS", None)
    # Resetear cache del helper para cada test.
    for m in ("shopping_calculator", "graph_orchestrator"):
        if m in sys.modules:
            del sys.modules[m]
    yield
    if prev is not None:
        os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = prev


def _import_helpers():
    import shopping_calculator as sc
    return sc._should_skip_meal_for_aggregation, sc._meal_aggregation_excluded_keywords


# ---------------------------------------------------------------------------
# 1. Comportamiento default
# ---------------------------------------------------------------------------
def test_default_excludes_only_suplemento():
    skip, kw = _import_helpers()
    assert kw() == ("suplemento",)


def test_skip_when_meal_contains_suplemento():
    skip, _ = _import_helpers()
    assert skip({"meal": "Suplemento de magnesio"}) is True
    assert skip({"meal": "Suplementos de proteína"}) is True
    assert skip({"meal": "SUPLEMENTO"}) is True


def test_no_skip_for_normal_meals():
    skip, _ = _import_helpers()
    assert skip({"meal": "Desayuno"}) is False
    assert skip({"meal": "Almuerzo"}) is False
    assert skip({"meal": "Cena"}) is False
    assert skip({"meal": "Merienda"}) is False
    assert skip({"meal": "Snack saludable"}) is False


def test_no_skip_when_meal_field_missing():
    skip, _ = _import_helpers()
    assert skip({"meal": ""}) is False
    assert skip({}) is False


def test_skip_defensively_for_non_dict():
    skip, _ = _import_helpers()
    assert skip(None) is True
    assert skip("Desayuno") is True
    assert skip(["meals", "list"]) is True


# ---------------------------------------------------------------------------
# 2. Knob override
# ---------------------------------------------------------------------------
def test_knob_override_adds_extra_keyword():
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = "suplemento,infusión"
    skip, kw = _import_helpers()
    assert "suplemento" in kw()
    assert "infusión" in kw()
    assert skip({"meal": "Infusión de manzanilla"}) is True
    assert skip({"meal": "Suplemento omega"}) is True
    assert skip({"meal": "Desayuno"}) is False


def test_knob_override_replaces_default_when_explicit():
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = "infusión,té"
    skip, kw = _import_helpers()
    # default "suplemento" NO está si el override no lo incluye.
    assert "suplemento" not in kw()
    assert skip({"meal": "Suplemento de magnesio"}) is False
    assert skip({"meal": "Té verde"}) is True


def test_knob_with_whitespace_normalized():
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = "  suplemento ,  AGUA AROMATIZADA  "
    skip, kw = _import_helpers()
    assert "suplemento" in kw()
    assert "agua aromatizada" in kw()  # lowercased
    assert skip({"meal": "agua aromatizada de pepino"}) is True


def test_knob_empty_falls_back_to_default():
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = ""
    skip, kw = _import_helpers()
    assert kw() == ("suplemento",)


def test_knob_only_commas_falls_back_to_default():
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = ",,,"
    skip, kw = _import_helpers()
    assert kw() == ("suplemento",)


# ---------------------------------------------------------------------------
# 3. Cache behavior
# ---------------------------------------------------------------------------
def test_cache_invalidates_when_env_changes():
    skip, kw = _import_helpers()
    # Primer call: default
    assert kw() == ("suplemento",)
    # Mutar env y repetir
    os.environ["MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS"] = "infusión"
    assert kw() == ("infusión",)


# ---------------------------------------------------------------------------
# 4. SSOT estructural — los 3 sitios usan el helper
# ---------------------------------------------------------------------------
def test_three_call_sites_use_helper_not_inline():
    """Verifica que los 3 sitios históricos (expected_sum_from_recipes,
    get_shopping_list_delta, extractor de facts) llaman al helper en lugar
    del patrón inline duplicado."""
    src = _SHOP_PATH.read_text(encoding="utf-8")
    # El patrón EJECUTABLE inline `if "suplemento" in meal.get(...).lower(): continue`
    # debe haber sido reemplazado por `_should_skip_meal_for_aggregation(meal)`
    # en los 3 call sites. La docstring del helper menciona el patrón
    # histórico (entre backticks, no ejecutable) — esa ocurrencia es legítima
    # y no debe contar. Excluimos líneas con backticks (docstring/markdown).
    executable_inline_lines = [
        ln for ln in src.splitlines()
        if re.search(r'if\s+"suplemento"\s+in\s+meal\.get\(\s*"meal"', ln)
        and "`" not in ln  # excluye referencia histórica en docstring
    ]
    assert not executable_inline_lines, (
        f"Patrón inline ejecutable encontrado en {len(executable_inline_lines)} línea(s): "
        f"{executable_inline_lines!r}. Debe ser reemplazado por "
        f"`_should_skip_meal_for_aggregation(meal)`."
    )


def test_helper_invoked_at_least_three_times():
    """Smoke: el helper debe invocarse ≥3 veces (los 3 call sites)."""
    src = _SHOP_PATH.read_text(encoding="utf-8")
    invocations = len(re.findall(r"_should_skip_meal_for_aggregation\s*\(", src))
    # 3 call sites + posibles tests dentro del módulo + la def misma.
    assert invocations >= 4, (
        f"Helper invocado {invocations} veces; esperado ≥4 (3 call sites + def). "
        f"Si bajó, P2-4 puede haber sido revertido."
    )


# ---------------------------------------------------------------------------
# 5. Knob registry
# ---------------------------------------------------------------------------
def test_knob_registers_in_global_registry():
    """Tras invocar el helper, MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS
    debe aparecer en _KNOBS_REGISTRY (P3-NEW-D contract)."""
    skip, kw = _import_helpers()
    kw()  # disparar registro
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_COHERENCE_EXCLUDED_MEAL_KEYWORDS" in snap, (
        "Knob debe registrarse en _KNOBS_REGISTRY para visibilidad en "
        "[KNOBS/INVENTORY] y /api/system/knobs."
    )


# ---------------------------------------------------------------------------
# 6. Smoke regresivo
# ---------------------------------------------------------------------------
def test_legacy_suplemento_behavior_preserved():
    """Sin override, el comportamiento end-user es idéntico al pre-P2-4:
    'Suplemento' se excluye, 'Snack' / 'Merienda' / nombres normales no."""
    skip, _ = _import_helpers()
    excluded_meals = [
        {"meal": "Suplemento de magnesio"},
        {"meal": "Suplemento Vitamina D"},
        {"meal": "suplementos pre-entrenamiento"},
    ]
    included_meals = [
        {"meal": "Desayuno"},
        {"meal": "Snack de almendras"},
        {"meal": "Merienda"},
        {"meal": "Cena ligera"},
    ]
    for m in excluded_meals:
        assert skip(m) is True, f"{m['meal']} debería excluirse"
    for m in included_meals:
        assert skip(m) is False, f"{m['meal']} debería incluirse"
