"""[P1-5] Tests para garantizar que los 3 cálculos paralelos del delta
shopping list (weekly/biweekly/monthly) usen el mismo snapshot de inventario.

Bug original (audit P1-5):
  `get_shopping_list_delta` se llamaba 3 veces consecutivas (multipliers
  1.0/2.0/4.0) en `graph_orchestrator.py:5359`, `routers/plans.py:2830-32`,
  `cron_tasks.py:10053-55` y `:18925-27`, `tools.py:479-81`. Cada
  invocación re-consultaba `get_raw_user_inventory(user_id)` y
  `get_consumed_meals_since(...)` independientemente. Si entre las 3
  queries un Realtime channel, restock o cron mutaba `user_inventory`,
  las 3 listas escaladas quedaban basadas en SNAPSHOTS DISTINTOS — el
  frontend mostraba cantidades inconsistentes al usuario al cambiar
  `groceryDuration` (weekly muestra X, monthly muestra Y aunque sea el
  mismo plan).

Fix:
  1. Nueva función `fetch_inventory_and_consumed_for_plan(user_id,
     plan_result, is_new_plan)` que hace los 2 fetches UNA vez.
  2. `get_shopping_list_delta` acepta `inventory_override` +
     `consumed_override` (kw-only) para reusar un snapshot dado.
  3. Los 5 callers en batch (orchestrator, plans router, cron x2, tools)
     ahora hacen UN fetch antes y pasan el snapshot a las 3 invocaciones.

Cobertura:
  - test_get_shopping_list_delta_accepts_inventory_override
  - test_get_shopping_list_delta_accepts_consumed_override
  - test_overrides_are_keyword_only_to_prevent_positional_misuse
  - test_overrides_skip_db_fetches
  - test_fetch_helper_returns_empty_for_guest
  - test_fetch_helper_returns_empty_for_none_user
  - test_orchestrator_batches_with_single_snapshot
  - test_plans_router_batches_with_single_snapshot
  - test_cron_tasks_batches_with_single_snapshot
  - test_tools_batches_with_single_snapshot
"""
import inspect
from unittest.mock import patch, MagicMock

import pytest

import shopping_calculator
from shopping_calculator import (
    get_shopping_list_delta,
    fetch_inventory_and_consumed_for_plan,
)


# ---------------------------------------------------------------------------
# 1. Contrato de signature: overrides kw-only.
# ---------------------------------------------------------------------------
def test_get_shopping_list_delta_accepts_inventory_override():
    """`inventory_override` debe estar en la firma."""
    sig = inspect.signature(get_shopping_list_delta)
    assert "inventory_override" in sig.parameters


def test_get_shopping_list_delta_accepts_consumed_override():
    """`consumed_override` debe estar en la firma."""
    sig = inspect.signature(get_shopping_list_delta)
    assert "consumed_override" in sig.parameters


def test_overrides_are_keyword_only_to_prevent_positional_misuse():
    """Ambos overrides deben ser kw-only para evitar passing posicional
    accidental que mezclaría argumentos en callers que ya tenían 6 args."""
    sig = inspect.signature(get_shopping_list_delta)
    inv = sig.parameters["inventory_override"]
    cons = sig.parameters["consumed_override"]
    assert inv.kind == inspect.Parameter.KEYWORD_ONLY
    assert cons.kind == inspect.Parameter.KEYWORD_ONLY
    # Default es None (preserva comportamiento legacy de hacer fetch interno).
    assert inv.default is None
    assert cons.default is None


# ---------------------------------------------------------------------------
# 2. Comportamiento: cuando override está presente, NO se hacen DB fetches.
# ---------------------------------------------------------------------------
def test_overrides_skip_db_fetches():
    """Si `inventory_override` está presente, `get_raw_user_inventory` NO
    se llama (señal clave del fix: el snapshot externo es la SSOT)."""
    plan = {"days": [{"meals": [{"ingredients": ["1 lb pollo"]}]}]}

    # Mock todos los fetches DB. Si alguno se llama, falla el test.
    with patch.object(shopping_calculator, "get_master_ingredients", return_value=[]) as _:
        # Espiamos `fetch_inventory_and_consumed_for_plan` para asegurar que NO
        # se invoca cuando hay override (el helper interno haría los fetches).
        with patch.object(
            shopping_calculator, "fetch_inventory_and_consumed_for_plan",
            side_effect=AssertionError("NO debe llamarse cuando hay override"),
        ):
            # No debe lanzar — los overrides cortocircuitan el fetch.
            result = get_shopping_list_delta(
                "any-user-id",
                plan,
                is_new_plan=True,
                structured=True,
                multiplier=1.0,
                inventory_override=[],
                consumed_override=[],
            )
            assert isinstance(result, list)


def test_overrides_pass_through_to_aggregator():
    """Los overrides se traducen a `items_to_deduct` para el aggregator —
    el inventario REDUCE las cantidades del delta. Comparamos con la misma
    invocación SIN override (donde el fetch interno fallará bajo test
    environment y devolverá inventory vacío) para verificar que el
    override SÍ deduce.

    [test-drift 2026-06-16] P3-CANONICAL-AGG-WEEKLY (shopping_calculator.py
    :7801, 2026-05-18 — DESPUÉS de P1-5) le dio a `is_new_plan=True`
    precedencia EXPLÍCITA sobre el override: en modo canónico no se deduce
    NADA (`physical_inventory = []` antes del check del override). Por eso
    con `is_new_plan=True` el override quedaba inerte y `qty_with ==
    qty_without`. Para ejercitar la deducción del override hay que usar
    `is_new_plan=False` (modo delta), que es donde el override existe para
    reusar un snapshot. Prod es CORRECTO; el test debe pedir el modo delta."""
    plan = {"days": [{"meals": [{"ingredients": ["500 g de pollo"]}]}]}

    # Llamada CON override (500g pollo en inventario), modo delta.
    with_override = get_shopping_list_delta(
        "guest", plan, is_new_plan=False, structured=True, multiplier=1.0,
        inventory_override=[{"quantity": 500, "unit": "g", "ingredient_name": "pollo"}],
        consumed_override=[],
    )
    # Llamada SIN override (path normal: guest → fetch helper retorna [],[]).
    without_override = get_shopping_list_delta(
        "guest", plan, is_new_plan=False, structured=True, multiplier=1.0,
    )

    def _pollo_qty(result):
        for r in result:
            if isinstance(r, dict) and "pollo" in r.get("name", "").lower():
                return r.get("market_qty_numeric") or r.get("market_qty") or 0
        return 0

    qty_with = _pollo_qty(with_override)
    qty_without = _pollo_qty(without_override)
    # Con override (inventory 500g) la cantidad neta debe ser MENOR que sin
    # override (donde el delta no descuenta nada).
    assert qty_with < qty_without, (
        f"P1-5: inventory_override debe REDUCIR el delta. "
        f"with={qty_with} >= without={qty_without}"
    )


# ---------------------------------------------------------------------------
# 3. Helper de fetch: retorna vacío para guest/None.
# ---------------------------------------------------------------------------
def test_fetch_helper_returns_empty_for_guest():
    inv, cons = fetch_inventory_and_consumed_for_plan("guest", {"days": []}, False)
    assert inv == []
    assert cons == []


def test_fetch_helper_returns_empty_for_none_user():
    inv, cons = fetch_inventory_and_consumed_for_plan(None, {"days": []}, False)
    assert inv == []
    assert cons == []


def test_fetch_helper_signature_returns_tuple():
    """Contrato: retorna (inventory_list, consumed_list) tupla, ambos
    listas iterables."""
    result = fetch_inventory_and_consumed_for_plan("guest", {"days": []}, False)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# 4. Defensa estructural: los 5 callers en batch usan el snapshot único.
# ---------------------------------------------------------------------------
def _read_file(path):
    # [test-drift 2026-06-16] Los archivos de prod (graph_orchestrator.py,
    # routers/plans.py, cron_tasks.py, tools.py) viven en el backend root
    # (el PADRE de tests/), no dentro de tests/. El join previo resolvía a
    # `tests/<file>` → FileNotFoundError. Anclamos al backend root.
    import os
    backend_root = os.path.dirname(os.path.dirname(__file__))
    full = os.path.join(backend_root, path)
    with open(full, encoding="utf-8") as f:
        return f.read()


def test_orchestrator_batches_with_single_snapshot():
    """`graph_orchestrator.py` debe importar el helper Y pasarlo a las 3 calls."""
    src = _read_file("graph_orchestrator.py")
    assert "fetch_inventory_and_consumed_for_plan" in src, (
        "P1-5: orchestrator debe importar el helper"
    )
    # Cada uno de los 3 calls del batch debe pasar `inventory_override`.
    occurrences = src.count("inventory_override=inv_snapshot")
    assert occurrences >= 3, (
        f"P1-5: orchestrator debe pasar inventory_override en las 3 calls, got {occurrences}"
    )


def test_plans_router_batches_with_single_snapshot():
    src = _read_file("routers/plans.py")
    assert "fetch_inventory_and_consumed_for_plan" in src
    occurrences = src.count("inventory_override=_inv_snap")
    assert occurrences >= 3, f"P1-5: plans.py debe pasar override en 3 calls, got {occurrences}"


def test_cron_tasks_batches_with_single_snapshot():
    src = _read_file("cron_tasks.py")
    assert "fetch_inventory_and_consumed_for_plan" in src
    occurrences = src.count("inventory_override=_inv_s")
    # 2 batches en cron_tasks (10053, 18925) × 3 calls cada uno = 6.
    assert occurrences >= 6, (
        f"P1-5: cron_tasks debe pasar override en 6 calls (2 batches × 3), got {occurrences}"
    )


def test_tools_batches_with_single_snapshot():
    src = _read_file("tools.py")
    assert "fetch_inventory_and_consumed_for_plan" in src
    occurrences = src.count("inventory_override=_inv_s")
    assert occurrences >= 3, f"P1-5: tools.py debe pasar override en 3 calls, got {occurrences}"


def test_documentation_p1_5_present():
    """Comentario `[P1-5]` debe aparecer en el helper y los call sites."""
    src = inspect.getsource(shopping_calculator)
    assert "[P1-5]" in src
