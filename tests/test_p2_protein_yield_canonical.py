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

import inspect
from unittest.mock import patch

import pytest

import shopping_calculator as sc
from shopping_calculator import (
    _calculate_yield_multiplier,
    _parse_quantity,
    expected_sum_from_recipes,
    run_shopping_coherence_guard,
    aggregate_and_deduct_shopping_list,
    get_shopping_list_delta,
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
# 1. Knob se auto-registra
#
# [P3-PROTEIN-YIELD-DECISION · 2026-08-04] El default FLIPEÓ a `True` (decisión medida,
# ver shopping_calculator.py::_protein_yield_on_canonical_enabled y
# test_p3_protein_yield_decision.py para los números). Este archivo ancla el
# COMPORTAMIENTO del A/B (la regla #2 + sus flags), no el valor del default — eso vive
# en el test de la decisión. Los tests de aquí que dependían del default (vía
# `delenv`) pasan a setear el estado EXPLÍCITO que necesitan, para no drifear con
# futuros cambios de default.
# ---------------------------------------------------------------------------
def test_knob_registrado_default_true(monkeypatch):
    monkeypatch.delenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", raising=False)
    assert sc._protein_yield_on_canonical_enabled() is True
    reg = get_knobs_registry_snapshot()
    assert reg["MEALFIT_PROTEIN_YIELD_ON_CANONICAL"]["default"] is True


def test_knob_off_via_env(monkeypatch):
    """Rollback explícito sin redeploy — sigue disponible tras el flip del default."""
    monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
    assert sc._protein_yield_on_canonical_enabled() is False


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
        "(sobras)",  # [ronda 1] stem coloquial más común, distinto de "sobrante(s)"
        "(del día anterior)",  # [ronda 1]
        "(del dia anterior)",  # [ronda 1] sin tilde
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
        # [ronda 1] El SELLO queda estampado — el guard lo leerá en vez del knob vigente.
        assert pollo.get("protein_yield_applied") is True

    def test_knob_off_explicito_byte_identico_al_legado(self, monkeypatch):
        """[P3-PROTEIN-YIELD-DECISION · 2026-08-04] Antes de la decisión este test
        confiaba en `delenv` para ejercitar el default (que era OFF). El default
        FLIPEÓ a True — este test pasa a setear el rollback EXPLÍCITO
        (`MEALFIT_PROTEIN_YIELD_ON_CANONICAL=false`), que sigue siendo el mismo
        comportamiento byte-idéntico al legado pre-Task-14."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo["base_qty"] == pytest.approx(1.0 * 7 * _LB_TO_G, rel=0.01)
        # [ronda 1] Sin el flag, ningún ítem lleva el sello (byte-idéntico: la key ni existe).
        assert "protein_yield_applied" not in pollo

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
        # [P3-PROTEIN-YIELD-DECISION · 2026-08-04] Explícito en AMBAS direcciones — el
        # default ya no es OFF, así que `knob_on=False` necesita el rollback explícito
        # para de verdad ejercitar el estado apagado.
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true" if knob_on else "false")
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
# 7b. [ronda 1 · 2026-08-03] El guard sigue al SELLO `protein_yield_applied`,
#     NO al knob VIGENTE en el momento de re-evaluar. Precedente: `trip_window_days`
#     (Task 6, P1-TRIP-WINDOWED-PERISHABLES) — "el sello ES la evidencia de cómo se
#     construyó ESTA lista", nunca el knob al momento de re-leerla (cron diario
#     re-validando `plan_data` persistido, rebuild, rollback a mitad de camino).
#
#     Medido por el revisor: lista construida con knob OFF (1.435 g), re-evaluada con
#     knob ON → 25,9% de divergencia + `magnitude=True`. Ese bug es justo lo que este
#     bloque ancla que NO pase.
# ---------------------------------------------------------------------------
class TestGuardSealNotLiveKnob:
    def _plan_con_lista_real(self, build_knob_on: bool):
        """Construye la lista con el knob en el estado `build_knob_on` — el estado del
        knob AL MOMENTO DE CONSTRUIR, que puede diferir del estado al momento en que el
        guard la re-evalúa (eso es justo lo que este bloque prueba)."""
        import os
        prev = os.environ.get("MEALFIT_PROTEIN_YIELD_ON_CANONICAL")
        os.environ["MEALFIT_PROTEIN_YIELD_ON_CANONICAL"] = "true" if build_knob_on else "false"
        try:
            plan = {"days": [{"meals": [{"meal": "almuerzo",
                                          "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
            real_list = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        finally:
            if prev is None:
                os.environ.pop("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", None)
            else:
                os.environ["MEALFIT_PROTEIN_YIELD_ON_CANONICAL"] = prev
        plan["aggregated_shopping_list_weekly"] = real_list
        return plan, real_list

    def test_lista_sellada_re_evaluada_con_knob_apagado_cero_divergencias(self, monkeypatch):
        """Rollback / cron corriendo DESPUÉS de apagar el A/B: la lista nació con el
        knob ON (sellada), pero el guard la re-evalúa con el knob YA apagado. El
        espejo debe seguir al sello -> cero divergencias, igual que si el knob
        siguiera ON."""
        plan, real_list = self._plan_con_lista_real(build_knob_on=True)
        assert any(it.get("protein_yield_applied") is True for it in real_list), (
            "la lista debía nacer sellada (knob ON al construir)"
        )
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        assert divs == [], f"el guard debía seguir el SELLO, no el knob apagado: {divs}"

    def test_lista_sin_sello_re_evaluada_con_knob_encendido_cero_divergencias(self, monkeypatch):
        """A/B recién encendido: la lista nació con el knob OFF (sin sello), pero el
        guard la re-evalúa con el knob YA encendido (cron diario re-validando un plan
        viejo). El espejo debe seguir al AUSENTE sello -> cero divergencias, igual
        que si el knob siguiera OFF."""
        plan, real_list = self._plan_con_lista_real(build_knob_on=False)
        assert not any(it.get("protein_yield_applied") for it in real_list), (
            "la lista debía nacer SIN sello (knob OFF al construir)"
        )
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        assert divs == [], f"el guard debía ignorar el knob encendido y seguir el sello ausente: {divs}"

    def test_falsabilidad_si_el_guard_leyera_el_knob_en_vez_del_sello(self, monkeypatch):
        """Reproduce el bug que esta ronda cierra: si el guard leyera el knob VIGENTE
        (en vez de `_protein_yield_seal_applied`), el escenario del test anterior SÍ
        fabricaría una divergencia de magnitud severa. Ancla la regresión, no solo el
        estado feliz — mutación de `_protein_yield_seal_applied` para que ignore el
        sello y siga el knob, restaurada por monkeypatch al salir del test."""
        plan, real_list = self._plan_con_lista_real(build_knob_on=False)
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        monkeypatch.setattr(
            sc, "_protein_yield_seal_applied",
            lambda _lst: sc._protein_yield_on_canonical_enabled(),
        )
        divs = run_shopping_coherence_guard(plan, multiplier=1.0)
        mag = [d for d in divs if d.get("magnitude")]
        assert mag, (
            "sin el sello (bug pre-fix simulado), un knob ON sobre una lista construida "
            "sin yield debería fabricar una divergencia de magnitud"
        )


# ---------------------------------------------------------------------------
# 8. Composición con el backstop de Task 8 (P1-VEG-BACKFILL-HONESTY) —
#    el yield NO debe fabricar una nota de recompra falsa.
# ---------------------------------------------------------------------------
class TestBackstopComposition:
    def test_sin_capped_by_sintetico_con_knob_on(self, monkeypatch):
        """El yield 1.35× sube TANTO el lado comprado como `text_demand_g_map` (mismo
        `expected_sum_from_recipes` con el mismo flag) — así que el ratio comprado/
        texto se queda en ~1.0 y el backstop `qty_reconcile_v7` no dispara sobre una
        compra correcta."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "true")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") is None

    def test_sin_capped_by_sintetico_con_knob_off(self, monkeypatch):
        # [P3-PROTEIN-YIELD-DECISION · 2026-08-04] Explícito: el default ya no es OFF.
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") is None


# ---------------------------------------------------------------------------
# 8b. [ronda 1 · 2026-08-03] El test anterior NO discrimina: el revisor ejecutó el
#     contrafactual y, quitando el espejo, `capped_by` seguía `None` igual (falso
#     negativo del test). La asimetría que SÍ fabrica la nota falsa es la INVERSA —
#     texto CON yield fantasma, compra SIN yield (compra literal, knob apagado o
#     desincronizado del lado de texto). Este bloque ancla ESA dirección.
# ---------------------------------------------------------------------------
class TestBackstopDiscriminatesDirection:
    def test_espejo_roto_direccion_texto_inflado_fabrica_nota_falsa(self, monkeypatch):
        """Simula el bug real que el espejo previene: `text_demand_g_map` se computa
        con yield (texto inflado 1.35×) mientras la compra real queda en peso literal
        (knob apagado — la RECETA no cambia, la interpretación del backstop sí). Sin
        el espejo correcto, esto fabrica `capped_by='qty_reconcile_v7'` sobre una
        compra que en realidad es exactamente lo que el plan pide.

        [P3-PROTEIN-YIELD-DECISION · 2026-08-04] Con el default ahora en True, `delenv`
        ya NO deja la compra real en peso literal (el knob real está ON por default) —
        el rollback explícito a `false` es lo que reproduce la asimetría que este test
        necesita (compra literal vs texto forzado a yieldear)."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", "false")
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}

        _real_expected = sc.expected_sum_from_recipes

        def _fake_expected_forces_protein_yield(plan_data, **kw):
            # Simula el bug: el cómputo de `text_demand_g_map` SIEMPRE yieldea,
            # ignorando el flag real que le llega desde `get_shopping_list_delta`.
            kw["apply_protein_yield"] = True
            return _real_expected(plan_data, **kw)

        monkeypatch.setattr(sc, "expected_sum_from_recipes", _fake_expected_forces_protein_yield)

        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") == "qty_reconcile_v7", (
            f"sin el espejo correcto, el texto inflado 1.35× debía fabricar una nota "
            f"de recompra falsa sobre una compra literal correcta: {pollo}"
        )

    @pytest.mark.parametrize("knob", ["false", "true"])
    def test_espejo_intacto_no_fabrica_nota_falsa(self, monkeypatch, knob):
        """Control positivo: con el código REAL (sin monkeypatch — mismo flag en
        ambos lados), la MISMA receta no dispara la nota, ni con knob OFF ni ON.

        [P3-PROTEIN-YIELD-DECISION · 2026-08-04] Ambos estados EXPLÍCITOS (antes uno
        de los dos era el default implícito vía `delenv`) para que el control positivo
        siga cubriendo las dos direcciones tras el flip."""
        monkeypatch.setenv("MEALFIT_PROTEIN_YIELD_ON_CANONICAL", knob)
        plan = {"days": [{"meals": [{"meal": "almuerzo",
                                      "ingredients_raw": ["1 lb de pollo cocido desmenuzado"]}]}]}
        items = _delta(plan, is_new_plan=True, inventory_override=[], consumed_override=[])
        pollo = _pollo_item(items)
        assert pollo.get("capped_by") is None

    def test_source_call_real_lleva_el_flag_del_mirror(self):
        """[ronda 1] Complemento parser-based del contrafactual de arriba: el monkeypatch
        de `expected_sum_from_recipes` prueba la CONSECUENCIA de la asimetría, pero NO
        detecta que alguien borre `apply_protein_yield=_apply_protein_yield` del callsite
        REAL que construye `text_demand_g_map` — verificado ejecutando: con ese kwarg
        quitado a mano de `get_shopping_list_delta`, TODOS los tests de
        `TestBackstopComposition` siguen en verde (knob ON compra 1.35× más que el texto
        literal → ratio >1.0, nunca dispara el backstop de shortfall) — exactamente el
        punto ciego que el revisor señaló. Este ancla de source cierra ese hueco."""
        src = inspect.getsource(get_shopping_list_delta)
        i = src.index("_text_demand_g_map: dict = {}")
        # El bloque que construye _tdg_raw_units vive ANTES de esta línea.
        block = src[:i]
        j = block.rindex("_tdg_raw_units = {")
        block = block[j:]
        assert "apply_protein_yield=_apply_protein_yield" in block, (
            "el callsite de expected_sum_from_recipes que alimenta text_demand_g_map "
            "debe pasar el MISMO flag que recibe el aggregator del lado comprado"
        )
