"""[P2-PROTEIN-YIELD-CANONICAL · 2026-08-03] «1 lb de pollo cocido desmenuzado» compraba
1 lb CRUDA — que tras cocción rinde ~0.74 lb — en vez del equivalente crudo real (~1.35 lb).
~26% menos proteína en el plato de la que el plan calculó, e invisible al coherence guard
(ambos lados parseaban igual, en peso literal).

## Por qué el yield estaba apagado (y por qué esta tarea NO reabre la asimetría P1-2)

`_calculate_yield_multiplier` regla #2 (proteínas cocidas → 1.35× crudo) existe desde P1-2,
pero el aggregator la apagaba SIEMPRE (`apply_yield_multiplier=False,
apply_legumbres_yield_only=True`) por una razón de simetría plan↔inventario: el
`physical_inventory` que el usuario tipea en su Nevera está en peso literal sin "cocido", así
que aplicar yield solo al lado del plan sesgaba el delta hacia over-buy.

Esa razón NO aplica en la lista CANÓNICA: `get_shopping_list_delta(is_new_plan=True)` fuerza
`physical_inventory=[]` (P3-CANONICAL-AGG-WEEKLY) — no hay lado inventario que proteger. Este
P-fix reabre la regla #2 SOLO ahí, gateada por `is_new_plan=True` Y el knob nuevo (default OFF,
A/B antes de encender).

## Medición (ya corrida contra los 23 planes vivos, 2026-08-03, SELECT-only)

12/5.899 líneas de `ingredients_raw` (0,203%) matchean la regla #2, pero **5/23 planes (~22%)
tienen al menos una línea**. Ejemplos reales: «205 g de pollo cocido y desmenuzado», «160 g de
pescado cocido», «45 g de costilla de cerdo cocida y desmenuzada», «100 g de cerdo magro cocido
y desmenuzado». Cada match es ~26% de under-buy de proteína en ese alimento — no es un no-op.

## Caso borde: líneas de REUSO

Una línea real: «205 g de pollo cocido y desmenuzado **(del almuerzo o preparado extra)**» — la
proteína ya se compró para el almuerzo; aplicarle yield la sobre-compraría. El patrón de
exclusión (`_PROTEIN_REUSE_PAREN_RE`) busca un paréntesis que mencione otro slot de comida
(desayuno/almuerzo/cena/merienda) o una frase explícita de reuso (sobrante/preparado extra).

## Espejo obligatorio (composición con el coherence guard y el backstop de Task 8)

`expected_sum_from_recipes` (el lado ESPERADO del guard, y la fuente de `text_demand_g_map`
para el backstop de shortfall P1-VEG-BACKFILL-HONESTY) recibe el MISMO flag
`apply_protein_yield`. Sin este espejo, con el knob ON: el guard vería el lado comprado subir
1.35× mientras el esperado se queda en peso literal → divergencia de magnitud ~35% (revienta la
tolerancia default del 10%); Y el backstop de Task 8 vería "se compró 135% del texto" → nota de
recompra falsa sobre una compra que en realidad es correcta.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import shopping_calculator as sc
from shopping_calculator import (
    _calculate_yield_multiplier,
    _parse_quantity,
    expected_sum_from_recipes,
    run_shopping_coherence_guard,
    aggregate_and_deduct_shopping_list,
)
from knobs import get_knobs_registry_snapshot

_LB_TO_G = 453.592


# ---------------------------------------------------------------------------
# Fixture: OFFLINE — `.env` apunta a producción y el worktree no tiene `.env`,
# así que `get_master_ingredients` se stubea a `[]` en TODOS los tests de este
# archivo. Mismo patrón que test_p1_trip_windowed_perishables.py /
# test_p1_shopping_recipe_coherence.py::no_master_db.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_master_db():
    with patch.object(sc, "get_master_ingredients", return_value=[]):
        yield


def _delta(plan, **kw):
    """Espejo de `_delta` en test_p1_trip_windowed_perishables.py: llama
    `get_shopping_list_delta` con la firma posicional canónica
    (`user_id, plan_result, is_new_plan, categorize, structured, multiplier`)."""
    return sc.get_shopping_list_delta(
        None, plan, kw.pop("is_new_plan", True), False, True, kw.pop("multiplier", 1.0), **kw
    )


def _pollo_item(items):
    return next(i for i in items if "pollo" in str(i.get("name", "")).lower())


# ---------------------------------------------------------------------------
# 1. Knob se auto-registra, default OFF
# ---------------------------------------------------------------------------
def test_knob_registrado_default_false(monkeypatch):
    monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
    assert sc._protein_yield_on_canonical_enabled() is False
    reg = get_knobs_registry_snapshot()
    assert reg["MEALFIT_PROTEIN_YIELD_ON_CANONICAL"]["default"] is False


def test_knob_on_via_env(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
    assert sc._protein_yield_on_canonical_enabled() is True


# ---------------------------------------------------------------------------
# 2. `_calculate_yield_multiplier` — unidad de la regla #2 reabierta
# ---------------------------------------------------------------------------
class TestCalculateYieldMultiplier:
    def test_only_legumbres_grains_sin_flag_sigue_apagando_proteina(self):
        """Comportamiento PREVIO intacto: sin `apply_protein_yield`, la regla #2
        sigue apagada dentro de `only_legumbres_grains` (byte-idéntico)."""
        assert _calculate_yield_multiplier("pollo cocido", only_legumbres_grains=True) == 1.0

    def test_flag_reabre_regla_2(self):
        assert _calculate_yield_multiplier(
            "pollo cocido", only_legumbres_grains=True, apply_protein_yield=True
        ) == pytest.approx(1.35)

    def test_pescado_cocido(self):
        """Caso real medido: «160 g de pescado cocido»."""
        assert _calculate_yield_multiplier(
            "pescado cocido", only_legumbres_grains=True, apply_protein_yield=True
        ) == pytest.approx(1.35)

    def test_costilla_de_cerdo_cocida_y_desmenuzada(self):
        """Caso real medido: «45 g de costilla de cerdo cocida y desmenuzada»."""
        assert _calculate_yield_multiplier(
            "costilla de cerdo cocida y desmenuzada", only_legumbres_grains=True,
            apply_protein_yield=True,
        ) == pytest.approx(1.35)

    @pytest.mark.parametrize("marker", [
        "(del almuerzo o preparado extra)",
        "(del almuerzo)",
        "(de la cena)",
        "(del desayuno)",
        "(de la merienda)",
        "(sobrante)",
        "(sobrantes)",
        "(preparado extra)",
    ])
    def test_marcador_de_reuso_excluye_yield(self, marker):
        raw = f"pollo cocido y desmenuzado {marker}"
        assert _calculate_yield_multiplier(
            raw, only_legumbres_grains=True, apply_protein_yield=True
        ) == 1.0, f"marcador {marker!r} debía suprimir el yield"

    def test_parentesis_sin_marcador_de_reuso_no_suprime(self):
        """Un paréntesis que NO menciona otro slot/reuso (ej. una nota de textura)
        no debe apagar el yield — solo los marcadores de reuso lo hacen."""
        raw = "pollo cocido y desmenuzado (muy jugoso)"
        assert _calculate_yield_multiplier(
            raw, only_legumbres_grains=True, apply_protein_yield=True
        ) == pytest.approx(1.35)

    def test_legumbres_no_afectadas_por_el_flag_nuevo(self):
        """La regla #1 (legumbres → 0.35×) se evalúa ANTES y hace return — el flag
        nuevo de proteína nunca la toca, con o sin `apply_protein_yield`."""
        assert _calculate_yield_multiplier(
            "lentejas cocidas", only_legumbres_grains=True, apply_protein_yield=True
        ) == pytest.approx(0.35)

    def test_default_path_sin_only_legumbres_grains_sin_cambios(self):
        """Con `only_legumbres_grains=False` (callers históricos que ya piden
        `apply_yield_multiplier=True`), la regla #2 sigue evaluándose SIEMPRE
        (comportamiento pre-existente) — el flag nuevo no tiene efecto aquí."""
        assert _calculate_yield_multiplier("pollo cocido") == pytest.approx(1.35)
        assert _calculate_yield_multiplier("pollo cocido", apply_protein_yield=False) == pytest.approx(1.35)


# ---------------------------------------------------------------------------
# 3. `_parse_quantity` — composición qty × yield
# ---------------------------------------------------------------------------
class TestParseQuantity:
    def test_pollo_cocido_1lb_con_flag(self):
        qty, unit, name = _parse_quantity(
            "1 lb de pollo cocido desmenuzado",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
            apply_protein_yield=True,
        )
        assert qty == pytest.approx(1.35)
        assert unit == "lb"
        assert name == "Pollo"

    def test_pollo_cocido_1lb_sin_flag_byte_identico(self):
        """Default `apply_protein_yield=False` → 1.0 lb literal (comportamiento
        pre-P-fix, sin ningún cambio)."""
        qty, unit, name = _parse_quantity(
            "1 lb de pollo cocido desmenuzado",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
        )
        assert qty == pytest.approx(1.0)

    def test_linea_de_reuso_sin_yield(self):
        qty, unit, name = _parse_quantity(
            "1 lb de pollo cocido desmenuzado (del almuerzo o preparado extra)",
            apply_yield_multiplier=False, apply_legumbres_yield_only=True,
            apply_protein_yield=True,
        )
        assert qty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. `expected_sum_from_recipes` — espejo del lado ESPERADO
# ---------------------------------------------------------------------------
class TestExpectedSumFromRecipesProteinYield:
    def test_default_false_byte_identico(self):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]},
        ]}]}
        result = expected_sum_from_recipes(plan)
        assert result["Pollo"]["lb"] == pytest.approx(1.0)

    def test_apply_protein_yield_true(self):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo", "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]},
        ]}]}
        result = expected_sum_from_recipes(plan, apply_protein_yield=True)
        assert result["Pollo"]["lb"] == pytest.approx(1.35, abs=0.01)

    def test_linea_de_reuso_sin_yield(self):
        plan = {"days": [{"meals": [
            {"meal": "almuerzo",
             "ingredients_raw": ["1 lb de pollo cocido desmenuzado (del almuerzo o preparado extra)"]},
        ]}]}
        result = expected_sum_from_recipes(plan, apply_protein_yield=True)
        assert result["Pollo"]["lb"] == pytest.approx(1.0)

    def test_legumbres_intactas_con_flag_on(self):
        """El espejo del flag de proteína no reintroduce la conversión de
        legumbres — esa regla (#1) sigue siendo independiente."""
        plan = {"days": [{"meals": [
            {"meal": "cena", "ingredients_raw": ["100 g lentejas cocidas"]},
        ]}]}
        result = expected_sum_from_recipes(plan, apply_protein_yield=True)
        assert result["Lentejas"]["g"] == pytest.approx(35.0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. `aggregate_and_deduct_shopping_list` — flag directo en el aggregator
# ---------------------------------------------------------------------------
class TestAggregateAndDeductProteinYield:
    def test_plan_loop_aplica_yield(self):
        res = aggregate_and_deduct_shopping_list(
            ["1 lb de pollo cocido desmenuzado"], [], structured=True,
            apply_protein_yield=True,
        )
        pollo = _pollo_item(res)
        assert pollo["base_qty"] == pytest.approx(1.35 * _LB_TO_G, rel=0.01)

    def test_default_false_byte_identico(self):
        res = aggregate_and_deduct_shopping_list(
            ["1 lb de pollo cocido desmenuzado"], [], structured=True,
        )
        pollo = _pollo_item(res)
        assert pollo["base_qty"] == pytest.approx(1.0 * _LB_TO_G, rel=0.01)

    def test_consumed_loop_nunca_recibe_yield_aunque_el_flag_este_on(self):
        """Defensa: incluso si un caller pasara `apply_protein_yield=True` con
        `consumed_ingredients` no vacío (no debería pasar en el path canónico,
        donde ese loop siempre está vacío), el loop de CONSUMIDO no debe
        yieldear — reintroduciría la asimetría P1-2 sobre inventario/consumo real."""
        res = aggregate_and_deduct_shopping_list(
            ["3 lb de pollo cocido desmenuzado"],
            ["1 lb de pollo cocido desmenuzado"],
            structured=True, apply_protein_yield=True,
        )
        pollo = _pollo_item(res)
        # plan: 3 * 1.35 = 4.05 lb; consumed: 1.0 lb literal (SIN yield) → delta 3.05 lb.
        # Si el consumed loop yieldeara también, sería 4.05 - 1.35 = 2.70 lb (bug).
        assert pollo["base_qty"] == pytest.approx(3.05 * _LB_TO_G, rel=0.01)


# ---------------------------------------------------------------------------
# 6. `get_shopping_list_delta` E2E — el knob + is_new_plan deciden
# ---------------------------------------------------------------------------
class TestGetShoppingListDeltaE2E:
    def test_pollo_cocido_compra_crudo_con_knob_on(self, monkeypatch):
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        # num_days=1 -> base_duration_scale=7 -> 1 lb * 1.35 * 7 = 9.45 lb crudo.
        assert pollo["base_qty"] == pytest.approx(1.35 * 7 * _LB_TO_G, rel=0.01)

    def test_knob_off_default_byte_identico(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo["base_qty"] == pytest.approx(1.0 * 7 * _LB_TO_G, rel=0.01)

    def test_linea_de_reuso_sin_yield_e2e(self, monkeypatch):
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": [
                                          "1 lb de pollo cocido desmenuzado "
                                          "(del almuerzo o preparado extra)"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo["base_qty"] == pytest.approx(1.0 * 7 * _LB_TO_G, rel=0.01)

    def test_is_new_plan_false_nunca_yieldea_ni_con_knob_on(self, monkeypatch):
        """El flag SOLO se activa con `is_new_plan=True`. Un delta contra
        inventario (`is_new_plan=False`) se queda en peso literal aunque el
        knob esté ON — ahí SÍ existe el lado inventario que P1-2 protege."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=False, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo["base_qty"] == pytest.approx(1.0 * 7 * _LB_TO_G, rel=0.01)


# ---------------------------------------------------------------------------
# 7. Composición con el coherence guard — CERO divergencias con el knob ON
# ---------------------------------------------------------------------------
class TestGuardComposition:
    def _plan_con_lista_real(self, monkeypatch, knob_on: bool):
        if knob_on:
            monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        else:
            monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        real_list = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        plan["aggregated_shopping_list_weekly"] = real_list
        return plan

    def test_cero_divergencias_con_knob_on(self, monkeypatch):
        plan = self._plan_con_lista_real(monkeypatch, knob_on=True)
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        assert divs == []
        assert "_shopping_coherence_block" not in plan

    def test_cero_divergencias_con_knob_off(self, monkeypatch):
        """Regresión: el comportamiento pre-existente (knob OFF, ambos lados en
        peso literal) sigue sin divergencias — el espejo nuevo no rompe el caso
        default."""
        plan = self._plan_con_lista_real(monkeypatch, knob_on=False)
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        assert divs == []


# ---------------------------------------------------------------------------
# 8. Composición con el backstop de Task 8 (P1-VEG-BACKFILL-HONESTY) —
#    el yield NO debe fabricar una nota de recompra falsa.
# ---------------------------------------------------------------------------
class TestBackstopComposition:
    def test_sin_capped_by_sintetico_con_knob_on(self, monkeypatch):
        """Riesgo explícito de la tarea: el yield 1.35× sube TANTO el lado
        comprado como `text_demand_g_map` (mismo `expected_sum_from_recipes`
        con el mismo flag) — así que el ratio comprado/texto se queda en ~1.0
        y el backstop `qty_reconcile_v7` no dispara sobre una compra correcta."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") is None

    def test_sin_capped_by_sintetico_con_knob_off(self, monkeypatch):
        monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") is None
